use bytemuck::{Pod, Zeroable};
use cgmath::SquareMatrix;
use cgmath::EuclideanSpace;
use image::GenericImageView;
use imgui::*;
use rand::Rng;
use rayon::prelude::*;
use std::sync::*;
use std::time::{Duration, Instant};
use wgpu::util::DeviceExt;
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

mod gfx {
    #[cfg_attr(rustfmt, rustfmt_skip)]
    #[allow(unused)]
    pub const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::new(
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 0.5, 0.0,
        0.0, 0.0, 0.5, 1.0,
    );

    /// Handle to core WGPU stuctures such as the device and queue
    pub struct GPU {
        _instance: wgpu::Instance,
        _adapter: wgpu::Adapter,
        device: wgpu::Device,
        queue: wgpu::Queue,
    }
    impl GPU {
        //TODO:: Support configurable features/limitations
        /// Creates the GPU handle and returns the window surface used
        async fn init(window: &Window) -> (GPU, wgpu::Surface) {
            log::info!("Initializing instance...");
            let _instance = wgpu::Instance::new(wgpu::BackendBit::PRIMARY);

            log::info!("Obtaining window surface...");
            let surface = unsafe {
                _instance.create_surface(window)
            };

            log::info!("Initializing adapter...");
            let _adapter = _instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::Default,
                    compatible_surface: Some(&surface),
                })
                .await
                .unwrap();

            let optional_features = wgpu::Features::empty();

            // TODO: support for setups without unsized_binding_array
            let required_features = wgpu::Features::default()
                | wgpu::Features::PUSH_CONSTANTS
                | wgpu::Features::UNSIZED_BINDING_ARRAY
                | wgpu::Features::SAMPLED_TEXTURE_ARRAY_NON_UNIFORM_INDEXING
                | wgpu::Features::SAMPLED_TEXTURE_BINDING_ARRAY;

            let adapter_features = _adapter.features();

            let required_limits = wgpu::Limits {
                max_push_constant_size: std::mem::size_of::<PushConstant>() as u32,
                ..wgpu::Limits::default()
            };

            let trace_dir = std::env::var("WGPU_TRACE");

            log::info!("Initializing device & queue...");
            let (device, queue) = _adapter
                .request_device(
                    &wgpu::DeviceDescriptor {
                        features: (adapter_features & optional_features) | required_features,
                        limits: required_limits,
                        shader_validation: true,
                    },
                    trace_dir.ok().as_ref().map(std::path::Path::new),
                )
                .await
                .unwrap();
            log::info!("Setup complete!");

            (GPU {
                _instance,
                _adapter,
                device,
                queue,
            },
            surface)
        }
       
        /// Wraps the async init function with blocking call
        fn new(window: &Window) -> (GPU, wgpu::Surface) {
            futures::executor::block_on(GPU::init(window))
        }
       
        /// Wraps the async init function with a blocking call
        /// Wraps the GPU handle in ARC
        fn new_with_arc(window: &Window) -> (Arc<GPU>, wgpu::Surface) {
            let (gpu, surface) = futures::executor::block_on(GPU::init(window));
            (Arc::new(gpu), surface)
        }
    }

    #[repr(C)]
    #[derive(Clone, Copy, Pod, Zeroable)]
    pub struct Vertex {
        _pos: [f32; 4],
        _tex_coord: [f32; 2],
    }
    impl Vertex {
        fn new(pos: [i8; 3], tc: [i8; 2]) -> Vertex {
            Vertex {
                _pos: [pos[0] as f32, pos[1] as f32, pos[2] as f32, 1.0],
                _tex_coord: [tc[0] as f32, tc[1] as f32],
            }
        }
    }

    #[repr(C)]
    #[derive(Clone, Copy, Pod, Zeroable)]
    pub struct PushConstant {
        eye_index: u32,
        _pad: u32,
    }
    impl PushConstant {
        fn new(eye_index: u32) -> PushConstant {
            PushConstant { eye_index, _pad: 0 }
        }
    }

    #[derive(Debug)]
    pub struct BufferDimensions {
        _width: usize,
        height: usize,
        depth: usize,
        _unpadded_bytes_per_row: usize,
        padded_bytes_per_row: usize,
        bytes_per_pixel: usize,
    }

    impl BufferDimensions {
        fn new(width: usize, height: usize, depth: usize) -> Self {
            let bytes_per_pixel = std::mem::size_of::<u32>();
            let unpadded_bytes_per_row = width * bytes_per_pixel;
            let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize;
            let padded_bytes_per_row_padding = (align - unpadded_bytes_per_row % align) % align;
            let padded_bytes_per_row = unpadded_bytes_per_row + padded_bytes_per_row_padding;
            Self {
                _width: width,
                height,
                depth,
                _unpadded_bytes_per_row: unpadded_bytes_per_row,
                padded_bytes_per_row,
                bytes_per_pixel,
            }
        }
    }

    /// A render target is a multisampled texture that may be rendered to and copied to
    /// A view is created for each layer of the texture (extent.depth). This allows for rendering to
    /// each layer
    /// For display-able and copy-able targets, see ResolveTarget
    // TODO: Handle invalid sample counts 
    #[derive(Debug)]
    pub struct RenderTarget {
        extent: wgpu::Extent3d,
        format: wgpu::TextureFormat,
        samples: usize,
        texture: wgpu::Texture,
        views: Vec<wgpu::TextureView>,
    }

    impl RenderTarget {
        fn new(device: &wgpu::Device, extent: wgpu::Extent3d, format: wgpu::TextureFormat, samples: usize) -> RenderTarget{ 
            let texture: wgpu::Texture = device.create_texture(&wgpu::TextureDescriptor {
                size: extent,
                mip_level_count: 1,
                sample_count: samples as u32,
                dimension: match extent.depth {
                    1 => wgpu::TextureDimension::D2,
                    // TODO: Should multi-layer textures always be D3?
                    _ => wgpu::TextureDimension::D2,
                },
                format,
                usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT | wgpu::TextureUsage::COPY_SRC,
                label: None,
            });

            // Create views for each layer of texture
            let views: Vec<wgpu::TextureView> = 
                (0..extent.depth)
                    .into_iter()
                    .map(|i| {
                        texture.create_view(&wgpu::TextureViewDescriptor {
                            label: None,
                            format: Some(format),
                            dimension: Some(wgpu::TextureViewDimension::D2),
                            aspect: wgpu::TextureAspect::All,
                            base_mip_level: 0,
                            level_count: None,
                            base_array_layer: i,
                            array_layer_count: std::num::NonZeroU32::new(1),
                        })
                    })
                    .collect();

            RenderTarget {
                extent,
                format,
                samples,
                texture,
                views,
            }
        }

        /// Returns a new RenderTarget object resized to the provided extent
        fn resize(&mut self, device: &wgpu::Device, extent: wgpu::Extent3d) { 
           *self = RenderTarget::new(device, extent, self.format, self.samples);
        }
    }

    pub struct ResolveTarget {
        extent: wgpu::Extent3d,
        format: wgpu::TextureFormat,
        texture: wgpu::Texture,
        views: Vec<wgpu::TextureView>,
    }
    impl ResolveTarget {
        fn from_render_target(device: &wgpu::Device, target: &RenderTarget) -> ResolveTarget{
            let texture: wgpu::Texture = device.create_texture(&wgpu::TextureDescriptor {
                size: target.extent,
                mip_level_count: 1,
                sample_count: 1,
                dimension: match target.extent.depth {
                    1 => wgpu::TextureDimension::D2,
                    // TODO: Should multi-layer textures always be D3?
                    _ => wgpu::TextureDimension::D2,
                },
                format: target.format,
                usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT | wgpu::TextureUsage::COPY_SRC,
                label: None,
            });

            // Create views for each layer of texture
            let views: Vec<wgpu::TextureView> = 
                (0..target.extent.depth)
                    .into_iter()
                    .map(|i| {
                        texture.create_view(&wgpu::TextureViewDescriptor {
                            label: None,
                            format: Some(target.format),
                            dimension: Some(wgpu::TextureViewDimension::D2),
                            aspect: wgpu::TextureAspect::All,
                            base_mip_level: 0,
                            level_count: None,
                            base_array_layer: i,
                            array_layer_count: std::num::NonZeroU32::new(1),
                        })
                    })
                    .collect();

            ResolveTarget {
                extent: target.extent,
                format: target.format,
                texture,
                views,
            }
        }
    }

    /// Array of cameras with a shared normal
    /// Since the normal is fixed, this is only supports single axis rotations
    pub struct CameraArray {
        count: usize,
        vertical_fov: cgmath::Deg<f32>,
        aspect_ratio: f32,
        normal: cgmath::Vector3<f32>,
        buf: wgpu::Buffer,
        mat: Vec<[[f32; 4]; 4]>,
    }
    impl CameraArray {
        fn build_camera(
            aspect_ratio: f32,
            eye: cgmath::Point3<f32>,
            center: cgmath::Vector3<f32>,
            up: cgmath::Vector3<f32>,
            vertical_fov: cgmath::Deg<f32>,
        ) -> [[f32; 4]; 4] {
            let proj = cgmath::perspective(vertical_fov, aspect_ratio, 1.0, 1000.0);
            let view = cgmath::Matrix4::look_at_dir(eye, center, up);
            let correction = OPENGL_TO_WGPU_MATRIX;
            (correction * proj * view).into()
        }

        /// Create a new camera array, matrices are uninitialized
        fn new(
            device: &wgpu::Device,
            count: usize,
            extent: wgpu::Extent3d,
            horizontal_fov: cgmath::Deg<f32>,
            normal: cgmath::Vector3<f32>,
        ) -> CameraArray {
            let aspect_ratio = extent.width as f32 / extent.height as f32;
            let mat : Vec<[[f32; 4]; 4]> = vec![cgmath::Matrix4::identity().into(); count];
            CameraArray {
                count,
                vertical_fov: horizontal_fov / aspect_ratio,
                aspect_ratio, 
                normal,
                buf: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::cast_slice(&mat),
                    usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_DST,
                }),
                mat, 
            }
        }

        /// Update camera matrices. Eyes are the positions of each eye, and look_dirs is a vector
        /// pointing in the desired look direction. Requires a mutable camera array
        fn update(
            &mut self,
            eyes: &Vec<cgmath::Point3<f32>>,
            look_dirs: &Vec<cgmath::Vector3<f32>>,
        ) {
            let aspect_ratio = self.aspect_ratio;
            let normal = self.normal;
            let fov = self.vertical_fov;
            self.mat.par_iter_mut().for_each(|m| {
                *m = CameraArray::build_camera(
                    aspect_ratio,
                    eyes[0],
                    look_dirs[0],
                    normal,
                    fov,
                );
            });
        }

        /// Resizes camera array, maintaining horizontal fov. Note that changes will not take affect until the next update
        fn resize(&mut self, extent: wgpu::Extent3d) {
            // maintain the previous horizontal field of view 
            let horizontal_fov = self.vertical_fov * self.aspect_ratio;
            self.aspect_ratio = extent.width as f32 / extent.height as f32;
            self.vertical_fov = horizontal_fov / self.aspect_ratio;
        }

        /// Writes the current camera matrices to the GPU
        fn write(&self, queue: &wgpu::Queue) {
            queue.write_buffer(&self.buf, 0, bytemuck::cast_slice(&self.mat));
        }
    }
}
