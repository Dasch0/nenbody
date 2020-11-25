mod gfx;
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

fn rotation_of(v: &cgmath::Vector3<f32>) -> cgmath::Rad<f32> {
    cgmath::Angle::atan2(v.y, v.x)
}

fn init_geometry_data() -> (Vec<Vertex>, Vec<u16>) {
    let vertex_data = [
        Vertex::new([-1, -1, 0], [0, 0]),
        Vertex::new([1, 0, 0], [0, 1]),
        Vertex::new([-1, 1, 0], [1, 1]),
    ];

    let index_data = [0, 1, 2, 0];
    (vertex_data.to_vec(), index_data.to_vec())
}

fn init_scene_config(
    device: &wgpu::Device,
    texture_format: wgpu::TextureFormat,
    msaa_samples: usize,
) -> (
    wgpu::BindGroupLayout,
    wgpu::PipelineLayout,
    Arc<wgpu::RenderPipeline>,
) {
    log::info!("Initializing Rendering Pipelines...");
    let scene_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStage::VERTEX,
                    ty: wgpu::BindingType::StorageBuffer {
                        dynamic: false,
                        min_binding_size: None,
                        readonly: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::SampledTexture {
                        multisampled: false,
                        component_type: wgpu::TextureComponentType::Float,
                        dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::Sampler { comparison: false },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStage::VERTEX,
                    ty: wgpu::BindingType::StorageBuffer {
                        dynamic: false,
                        min_binding_size: wgpu::BufferSize::new(64),
                        readonly: false,
                    },
                    count: None,
                },
            ],
        });

    let scene_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Scene pipeline layout"),
        bind_group_layouts: &[&scene_bind_group_layout],
        push_constant_ranges: &[wgpu::PushConstantRange {
            stages: wgpu::ShaderStage::VERTEX | wgpu::ShaderStage::FRAGMENT,
            range: (0..std::mem::size_of::<PushConstant>() as u32),
        }],
    });

    let vertex_size = std::mem::size_of::<Vertex>();
    let vertex_state = wgpu::VertexStateDescriptor {
        index_format: wgpu::IndexFormat::Uint16,
        vertex_buffers: &[wgpu::VertexBufferDescriptor {
            stride: vertex_size as wgpu::BufferAddress,
            step_mode: wgpu::InputStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttributeDescriptor {
                    format: wgpu::VertexFormat::Float4,
                    offset: 0,
                    shader_location: 0,
                },
                wgpu::VertexAttributeDescriptor {
                    format: wgpu::VertexFormat::Float2,
                    offset: 4 * 4, // TODO: cleanup
                    shader_location: 1,
                },
            ],
        }],
    };

    let scene_vert_shader =
        device.create_shader_module(wgpu::include_spirv!("../shaders/scene.vert.spv"));
    let scene_frag_shader =
        device.create_shader_module(wgpu::include_spirv!("../shaders/scene.frag.spv"));

    let scene_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Scene Pipeline"),
        layout: Some(&scene_pipeline_layout),
        vertex_stage: wgpu::ProgrammableStageDescriptor {
            module: &scene_vert_shader,
            entry_point: "main",
        },
        fragment_stage: Some(wgpu::ProgrammableStageDescriptor {
            module: &scene_frag_shader,
            entry_point: "main",
        }),
        rasterization_state: Some(wgpu::RasterizationStateDescriptor {
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: wgpu::CullMode::None,
            ..Default::default()
        }),
        primitive_topology: wgpu::PrimitiveTopology::LineStrip,
        color_states: &[wgpu::ColorStateDescriptor {
            format: texture_format,
            color_blend: wgpu::BlendDescriptor::REPLACE,
            alpha_blend: wgpu::BlendDescriptor::REPLACE,
            write_mask: wgpu::ColorWrite::ALL,
        }],
        depth_stencil_state: None,
        vertex_state: vertex_state.clone(),
        sample_count: msaa_samples as u32,
        sample_mask: !0,
        alpha_to_coverage_enabled: false,
    });

    (
        scene_bind_group_layout,
        scene_pipeline_layout,
        Arc::new(scene_pipeline),
    )
}

fn init_scene_data(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    entity_count: usize,
) -> (
    Arc<wgpu::Buffer>,
    Arc<wgpu::Buffer>,
    usize,
    Vec<[[f32; 4]; 4]>,
    wgpu::Buffer,
    wgpu::Extent3d,
    wgpu::Texture,
    wgpu::TextureView,
    wgpu::Sampler,
) {
    log::info!("Initializing buffers & textures...");
    let (vertex_data, index_data) = init_geometry_data();

    let vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Vertex Buffer"),
        contents: bytemuck::cast_slice(&vertex_data),
        usage: wgpu::BufferUsage::VERTEX,
    });

    let index_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Index Buffer"),
        contents: bytemuck::cast_slice(&index_data),
        usage: wgpu::BufferUsage::INDEX,
    });

    let mut rng = rand::thread_rng();
    // TODO: allow for different instance counts
    let instance_data: Vec<[[f32; 4]; 4]> =
        (0..entity_count)
            .map(|_| {
                cgmath::Matrix4::<f32>::from_translation(
                    [rng.gen_range(-1.0, 1.0), rng.gen_range(-1.0, 1.0), 0.0].into(),
                )
                .into()
            })
            .collect();

    let instance_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Instance Buffer"),
        contents: bytemuck::cast_slice(instance_data.as_ref()),
        usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_DST,
    });

    let skin_bytes = include_bytes!("../assets/skin.png");
    let skin_image = image::load_from_memory(skin_bytes).unwrap();
    let skin_rgba = skin_image.as_rgba8().unwrap();
    let skin_dimension = skin_image.dimensions();
    let skin_extent = wgpu::Extent3d {
        width: skin_dimension.0,
        height: skin_dimension.1,
        depth: 1,
    };

    let skin_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: skin_extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsage::SAMPLED | wgpu::TextureUsage::COPY_DST,
    });

    let skin_view = skin_texture.create_view(&wgpu::TextureViewDescriptor::default());
    queue.write_texture(
        wgpu::TextureCopyView {
            texture: &skin_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
        },
        &skin_rgba,
        wgpu::TextureDataLayout {
            offset: 0,
            bytes_per_row: 4 * skin_dimension.0,
            rows_per_image: 0,
        },
        skin_extent,
    );

    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });

    (
        Arc::new(vertex_buf),
        Arc::new(index_buf),
        index_data.len(),
        instance_data,
        instance_buf,
        skin_extent,
        skin_texture,
        skin_view,
        sampler,
    )
}

pub struct UiState {
    last_frame: std::time::Instant,
    last_queue: u32,
    show_demo: bool,
    last_cursor: Option<imgui::MouseCursor>,
    extent: wgpu::Extent3d,
    camera_count: u32,
    scene_camera: u32,
    viewport_camera: u32,
    viewport_scale: f32,
    zoom: f32,
    offset: [f32; 2],
    texture_id: imgui::TextureId,
}

fn init_imgui(
    window: &winit::window::Window,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture_format: wgpu::TextureFormat,
    texture_extent: wgpu::Extent3d,
) -> (
    imgui::Context,
    imgui_winit_support::WinitPlatform,
    imgui_wgpu::Renderer,
    imgui::TextureId,
) {
    log::info!("Initializing imgui...");
    let hidpi_factor = window.scale_factor();
    let mut imgui = imgui::Context::create();
    let mut platform = imgui_winit_support::WinitPlatform::init(&mut imgui);
    platform.attach_window(
        imgui.io_mut(),
        &window,
        imgui_winit_support::HiDpiMode::Default,
    );
    imgui.set_ini_filename(None);

    let font_size = (13.0 * hidpi_factor) as f32;
    imgui.io_mut().font_global_scale = (1.0 / hidpi_factor) as f32;
    imgui
        .fonts()
        .add_font(&[imgui::FontSource::DefaultFontData {
            config: Some(imgui::FontConfig {
                oversample_h: 1,
                pixel_snap_h: true,
                size_pixels: font_size,
                ..Default::default()
            }),
        }]);

    let imgui_render_config = imgui_wgpu::RendererConfig::new().set_texture_format(texture_format);

    let mut imgui_renderer =
        imgui_wgpu::Renderer::new(&mut imgui, &device, &queue, imgui_render_config);

    // Create imgui texture for drawing rendered textures in UI
    let imgui_texture_config =
        imgui_wgpu::TextureConfig::new(texture_extent.width, texture_extent.height).set_usage(
            wgpu::TextureUsage::COPY_SRC
                | wgpu::TextureUsage::COPY_DST
                | wgpu::TextureUsage::SAMPLED
                | wgpu::TextureUsage::OUTPUT_ATTACHMENT,
        );

    let imgui_texture = imgui_wgpu::Texture::new(device, &imgui_renderer, imgui_texture_config);
    let imgui_texture_id = imgui_renderer.textures.insert(imgui_texture);

    (imgui, platform, imgui_renderer, imgui_texture_id)
}

fn imgui_build_ui(ui: &imgui::Ui, state: &mut UiState) {
    let imgui_window = imgui::Window::new(im_str!("Learning WGPU"));
    imgui_window
        .size([500.0, 500.0], imgui::Condition::FirstUseEver)
        .build(ui, || {
            ui.text(im_str!("Frametime: {:?}", state.last_frame.elapsed()));
            ui.separator();
            ui.text(im_str!("queue_count: {:?}", state.last_queue));
            ui.separator();
            let mouse_pos = ui.io().mouse_pos;
            ui.text(im_str!(
                "Mouse Position: ({:.1},{:.1})",
                mouse_pos[0],
                mouse_pos[1],
            ));
            ui.separator();
            if ui.button(im_str!("Toggle Demo"), [100., 20.]) {
                state.show_demo = !state.show_demo
            }
            ui.separator();
            imgui::Slider::new(im_str!("Set Scene Camera"))
                .range(0..=state.camera_count - 1)
                .build(&ui, &mut state.scene_camera);
            ui.separator();
            imgui::Slider::new(im_str!("Set Viewport Camera"))
                .range(0..=state.camera_count - 1)
                .build(&ui, &mut state.viewport_camera);
            ui.separator();
            imgui::Slider::new(im_str!("Set Texture Zoom"))
                .range(0.001..=1.0)
                .build(&ui, &mut state.zoom);
            ui.separator();
            imgui::Slider::new(im_str!("Set Texture X offset"))
                .range(0.00..=1.0)
                .build(&ui, &mut state.offset[0]);
            ui.separator();
            imgui::Slider::new(im_str!("Set Texture Y offset"))
                .range(0.00..=1.0)
                .build(&ui, &mut state.offset[1]);
            ui.separator();
            imgui::Slider::new(im_str!("Set Viewport scale"))
                .range(0.001..=1.0)
                .build(&ui, &mut state.viewport_scale);
            imgui::Image::new(
                state.texture_id,
                [
                    state.extent.width as f32 * state.viewport_scale,
                    state.extent.height as f32 * state.viewport_scale,
                ],
            )
            .build(&ui);
            ui.separator();
        });

    let control_window = imgui::Window::new(im_str!("Controls"));
    control_window
        .size([500.0, 200.0], imgui::Condition::FirstUseEver)
        .build(ui, || {
            ui.text(im_str!("W : move camera along positive x axis"));
            ui.text(im_str!("A : move camera along negative y axis"));
            ui.text(im_str!("S : move camera along negative x axis"));
            ui.text(im_str!("D : move camera along positive y axis"));
            ui.text(im_str!("Q : move camera along positive z axis"));
            ui.text(im_str!("E : move camera along negative z axis"));
            ui.separator();
            ui.text(im_str!("C : capture viewport texture as .png"));
            ui.text(im_str!("ESC : quit"));
        });

    if state.show_demo {
        ui.show_demo_window(&mut false);
    }
}

fn imgui_draw_ui(
    platform: &mut imgui_winit_support::WinitPlatform,
    window: &winit::window::Window,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    renderer: &mut imgui_wgpu::Renderer,
    ui: imgui::Ui,
    state: &mut UiState,
    render_view: &wgpu::TextureView,
) {
    if state.last_cursor != ui.mouse_cursor() {
        state.last_cursor = ui.mouse_cursor();
        platform.prepare_render(&ui, &window);
    }

    let mut encoder: wgpu::CommandEncoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    let mut imgui_renderpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
            attachment: &render_view,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: true,
            },
        }],
        depth_stencil_attachment: None,
    });

    renderer
        .render(ui.render(), queue, device, &mut imgui_renderpass)
        .expect("Rendering failed");

    drop(imgui_renderpass);
    queue.submit(Some(encoder.finish()));
}


fn render<T: Pod>(
    msaa_target: &wgpu::TextureView,
    resolve_target: &wgpu::TextureView,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &wgpu::RenderPipeline,
    bind_group: &wgpu::BindGroup,
    vertex_buf: &wgpu::Buffer,
    index_buf: &wgpu::Buffer,
    index_cnt: usize,
    instance_cnt: usize,
    push_constants: T,
) {
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    let mut renderpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
            attachment: &msaa_target,
            resolve_target: Some(resolve_target),
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color {
                    r: 0.1,
                    g: 0.2,
                    b: 0.3,
                    a: 1.0,
                }),
                store: true,
            },
        }],
        depth_stencil_attachment: None,
    });
    renderpass.push_debug_group("Prepare data for draw.");
    renderpass.set_pipeline(&pipeline);
    renderpass.set_bind_group(0, &bind_group, &[]);
    renderpass.set_index_buffer(index_buf.slice(..));
    renderpass.set_vertex_buffer(0, vertex_buf.slice(..));
    renderpass.set_push_constants(
        wgpu::ShaderStage::VERTEX | wgpu::ShaderStage::FRAGMENT,
        0,
        bytemuck::cast_slice(&[push_constants]),
    );
    renderpass.pop_debug_group();
    renderpass.insert_debug_marker("Drawing frame");
    renderpass.draw_indexed(0..index_cnt as u32, 0, 0..instance_cnt as u32);
    drop(renderpass);
    queue.submit(Some(encoder.finish()));
}

const GRANULARITY: usize = 100;
fn build_command_buffer_parallel(
    cmd_buf_list: &mut Vec<wgpu::CommandBuffer>,
    msaa_target_list: &Vec<wgpu::TextureView>,
    resolve_target_list: &Vec<wgpu::TextureView>,
    device: &wgpu::Device,
    _queue: &wgpu::Queue,
    pipeline: &wgpu::RenderPipeline,
    bind_group: &wgpu::BindGroup,
    vertex_buf: &wgpu::Buffer,
    index_buf: &wgpu::Buffer,
    index_cnt: usize,
    instance_cnt: usize,
) {
    let span = msaa_target_list.len() / GRANULARITY;

    cmd_buf_list
        .into_par_iter()
        .enumerate()
        .for_each(|(m, cmd_buf)| {
            let mut encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            for n in 0..span {
                let i = m * span + n;
                let mut renderpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
                        attachment: &msaa_target_list[i],
                        resolve_target: Some(&resolve_target_list[i]),
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.1,
                                g: 0.2,
                                b: 0.3,
                                a: 1.0,
                            }),
                            store: true,
                        },
                    }],
                    depth_stencil_attachment: None,
                });
                renderpass.set_pipeline(&pipeline);
                renderpass.set_bind_group(0, &bind_group, &[]);
                renderpass.set_index_buffer(index_buf.slice(..));
                renderpass.set_vertex_buffer(0, vertex_buf.slice(..));
                renderpass.set_push_constants(
                    wgpu::ShaderStage::VERTEX | wgpu::ShaderStage::FRAGMENT,
                    0,
                    bytemuck::cast_slice(&[PushConstant::new(i as u32)]),
                );
                renderpass.draw_indexed(0..index_cnt as u32, 0, 0..instance_cnt as u32);
                drop(renderpass);
            }
            *cmd_buf = encoder.finish();
        });
}

fn main() {
    // Placeholder Constants
    // TODO: Get max allowed msaa_samples and store in GPU handle
    let msaa_samples = 8;
    // TODO: Support entity counts higher than 2048
    let entity_count = 2;

    // Window
    env_logger::init();
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new().build(&event_loop).unwrap();

    // GPU and surface 
    let (gpu, surface) = GPU::new(&window);

    // Swapchain
    let mut swapchain_desc = wgpu::SwapChainDescriptor {
        usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
        format: wgpu::TextureFormat::Bgra8UnormSrgb,
        width: window.inner_size().width,
        height: window.inner_size().height,
        present_mode: wgpu::PresentMode::Immediate,
    };
    let mut swapchain = gpu.device.create_swap_chain(&surface, &swapchain_desc);
    
    let swapchain_extent = wgpu::Extent3d {
        width: swapchain_desc.width,
        height: swapchain_desc.height,
        depth: 1,
    };

    // Render Targets
    // scene (resolved to swapchain image)
    let mut display = RenderTarget::new(&gpu.device,
                                  swapchain_extent,
                                  swapchain_desc.format,
                                  msaa_samples);

    // eyes (resolved to resolved_eyes) 
    let eye_extent = wgpu::Extent3d {
        width: 1024,
        height: 1,
        depth: entity_count as u32,
    };
    let eyes = RenderTarget::new(&gpu.device,
                                 eye_extent,
                                 swapchain_desc.format, // TODO: Consider other formats?
                                 msaa_samples);
    let resolved_eyes = ResolveTarget::from_render_target(&gpu.device, &eyes);
    
    // viewport (resolved to imgui viewport)
    let viewport_extent = wgpu::Extent3d {
        width: eye_extent.width,
        height: eye_extent.height,
        depth: 1, // Viewport is one layer of eye
    };
    let viewport = RenderTarget::new(&gpu.device,
                                     viewport_extent,
                                     eyes.format, // Format should match eye
                                     msaa_samples);
    
    // Scene
    let (
        vertex_buf,
        index_buf,
        index_count,
        mut instance_data,
        instance_buf,
        _texture_extent,
        _texture,
        texture_view,
        texture_sampler,
    ) = init_scene_data(&gpu.device, &gpu.queue, entity_count);
                                     
    // UI
    let (mut imgui_context,
         mut imgui_platform,
         mut imgui_renderer,
         imgui_texture_id) = init_imgui(
        &window,
        &gpu.device,
        &gpu.queue,
        swapchain_desc.format,
        eye_extent,
    );

    // Scene Config
    let (bind_group_layout, _pipeline_layout, pipeline) =
        init_scene_config(&gpu.device, swapchain_desc.format, msaa_samples as usize);

    // Placeholder position and velocity generation
    let mut rng = rand::thread_rng();
    let mut velocities: Vec<cgmath::Vector3<f32>> =
        vec![cgmath::Vector3::<f32>::new(0.0, 0.0, 0.0); instance_data.len()];
    let mut positions: Vec<cgmath::Point3<f32>> = (0..instance_data.len())
        .map(|_| {
            cgmath::Point3::<f32>::new(rng.gen_range(-1.0, 1.0), rng.gen_range(-1.0, 1.0), 0.0)
        })
        .collect();

    // Cameras
    // scene_camera
    let mut scene_position = vec![cgmath::Point3::<f32>::new(0.0, 0.0, 100.0); 1];
    let scene_look_dir = vec![-cgmath::Vector3::unit_z(); 1];
    let mut scene_cam = CameraArray::new(&gpu.device,
                                        1,
                                        swapchain_extent,
                                        cgmath::Deg(90.0),
                                        cgmath::Vector3::unit_x());

    // eye cameras
    let mut eye_cams = CameraArray::new(&gpu.device,
                                   entity_count,
                                   eye_extent,
                                   cgmath::Deg(90.0),
                                   cgmath::Vector3::unit_z());


    log::info!("Creating bind groups...");
    let scene_bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(scene_cam.buf.slice(..)),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&texture_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(&texture_sampler),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::Buffer(instance_buf.slice(..)),
            },
        ],
        label: Some("Scene Bind Group"),
    });

    let eye_bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(eye_cams.buf.slice(..)),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&texture_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(&texture_sampler),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::Buffer(instance_buf.slice(..)),
            },
        ],
        label: Some("Scene Bind Group"),
    });

    // UI States
    let mut ui_state = UiState {
        last_frame: std::time::Instant::now(),
        last_queue: 0,
        show_demo: false,
        last_cursor: None,
        extent: swapchain_extent,
        camera_count: instance_data.len() as u32,
        scene_camera: 0,
        viewport_camera: 1,
        viewport_scale: 0.15,
        zoom: 1.0,
        offset: [0.0, 0.0],
        texture_id: imgui_texture_id,
    };

    // Command buffer list 
    let mut cmd_buf_list: Vec<wgpu::CommandBuffer> = Vec::with_capacity(GRANULARITY);

    log::info!("Prearing event loop actions");

    log::info!("Starting event loop!");

    let mut last_update_inst = std::time::Instant::now();
    event_loop.run(move |event, _, control_flow| {
        
        // Set control flow to wait min(next event, 10ms)
        *control_flow = ControlFlow::WaitUntil(Instant::now() + Duration::from_millis(1));

        // Process events
        match event {
            Event::MainEventsCleared => {
                if last_update_inst.elapsed() > Duration::from_millis(2) {
                    window.request_redraw();
                    last_update_inst = Instant::now();
                }
            },
            Event::WindowEvent {event: WindowEvent::Resized(_), ..} => {
                let size = window.inner_size();
                // store new extent to ui_state
                ui_state.extent = wgpu::Extent3d {
                    width: size.width,
                    height: size.height,
                    depth: 1,
                };
                log::info!("Resizing to {:?}", size);
                // resize swapchain 
                swapchain_desc.width = size.width;
                swapchain_desc.height = size.height;
                swapchain = gpu.device.create_swap_chain(&surface, &swapchain_desc);
                // resize scene 
                scene_cam.resize(ui_state.extent);
                display.resize(&gpu.device, ui_state.extent);
            },
            
            Event::WindowEvent { ref event, .. } => match event {
                WindowEvent::KeyboardInput {
                    input:
                        KeyboardInput {
                            virtual_keycode: Some(virtual_code),
                            state: ElementState::Pressed,
                            ..
                        },
                    ..
                } => match virtual_code {
                    VirtualKeyCode::W => {
                        scene_position[ui_state.scene_camera as usize] +=
                            cgmath::Vector3::unit_x();
                    }
                    VirtualKeyCode::S => {
                        scene_position[ui_state.scene_camera as usize] -=
                            cgmath::Vector3::unit_x();
                    }
                    VirtualKeyCode::A => {
                        scene_position[ui_state.scene_camera as usize] -=
                            cgmath::Vector3::unit_y();
                    }
                    VirtualKeyCode::D => {
                        scene_position[ui_state.scene_camera as usize] +=
                            cgmath::Vector3::unit_y();
                    }
                    VirtualKeyCode::Q => {
                        scene_position[ui_state.scene_camera as usize] +=
                            cgmath::Vector3::unit_z();
                    }
                    VirtualKeyCode::E => {
                        scene_position[ui_state.scene_camera as usize] -=
                            cgmath::Vector3::unit_z();
                    }
                    VirtualKeyCode::C => {
                    }
                    VirtualKeyCode::I => {}
                    VirtualKeyCode::Escape => {
                        *control_flow = ControlFlow::Exit;
                    }
                    _ => {}
                },
                WindowEvent::CloseRequested => {
                    *control_flow = ControlFlow::Exit;
                }
                _ => {}
            },
            Event::RedrawRequested(_) => {
                let now = Instant::now();
                imgui_context
                    .io_mut()
                    .update_delta_time(now - ui_state.last_frame);
                ui_state.last_frame = now;

                let frame = match swapchain.get_current_frame() {
                    Ok(frame) => frame,
                    Err(e) => {
                        log::warn!("dropped frame: {:?}", e);
                        let size = window.inner_size();
                        // store new extent to ui_state
                        ui_state.extent = wgpu::Extent3d {
                            width: size.width,
                            height: size.height,
                            depth: 1,
                        };
                        log::info!("Resizing to {:?}", size);
                        // resize swapchain 
                        swapchain_desc.width = size.width;
                        swapchain_desc.height = size.height;
                        swapchain = gpu.device.create_swap_chain(&surface, &swapchain_desc);
                        // resize scene 
                        scene_cam.resize(ui_state.extent);
                        display.resize(&gpu.device, ui_state.extent);
                        // try one last time to get swapchain frame 
                        swapchain.get_current_frame().expect("failed to acquire swapchain frame")
                    }
                };

                let inst = &mut *instance_data;
                inst.into_par_iter()
                    .zip(&mut velocities)
                    .zip(&mut positions[..])
                    .for_each(|((mat, vel), pos)| {
                        let mut rng = rand::thread_rng();
                        *vel += cgmath::Vector3::<f32>::new(
                            rng.gen_range(-0.0001, 0.0001),
                            rng.gen_range(-0.0001, 0.0001),
                            0.0,
                        );
                        *pos += *vel;
                        *mat = (
                            cgmath::Matrix4::<f32>::from_translation(
                                pos.to_vec(),
                            ) * cgmath::Matrix4::from_angle_z(
                                rotation_of(vel),
                            )).into();
                    });
                gpu.queue.write_buffer(
                    &instance_buf,
                    0,
                    bytemuck::cast_slice(inst.as_ref()),
                );
                
                // Update cameras
                eye_cams.update(&positions, &velocities);
                scene_cam.update(&scene_position, &scene_look_dir);

                eye_cams.write(&gpu.queue);
                scene_cam.write(&gpu.queue);

                // Render Scene
                render(
                    &display.views[0],
                    &frame.output.view,
                    &gpu.device,
                    &gpu.queue,
                    &pipeline,
                    &scene_bind_group,
                    &vertex_buf,
                    &index_buf,
                    index_count,
                    instance_data.len(),
                    PushConstant::new(0),
                );

                build_command_buffer_parallel(
                    &mut cmd_buf_list,
                    &eyes.views,
                    &resolved_eyes.views,
                    &gpu.device,
                    &gpu.queue,
                    &pipeline,
                    &eye_bind_group,
                    &vertex_buf,
                    &index_buf,
                    index_count,
                    instance_data.len(),
                );

                // Render eyes
                gpu.queue.submit(cmd_buf_list.drain(..));

                // Render viewport
                render(
                    &viewport.views[0],
                    imgui_renderer
                        .textures
                        .get(imgui_texture_id)
                        .expect("failed to find imgui texture")
                        .view(),
                    &gpu.device,
                    &gpu.queue,
                    &pipeline,
                    &eye_bind_group,
                    &vertex_buf,
                    &index_buf,
                    index_count,
                    instance_data.len(),
                    PushConstant::new(ui_state.viewport_camera),
                );

                // Render UI
                imgui_platform
                    .prepare_frame(imgui_context.io_mut(), &window)
                    .expect("Failed to prepare frame");

                let ui = imgui_context.frame();
                imgui_build_ui(&ui, &mut ui_state);
                imgui_draw_ui(
                    &mut imgui_platform,
                    &window,
                    &gpu.device,
                    &gpu.queue,
                    &mut imgui_renderer,
                    ui,
                    &mut ui_state,
                    &frame.output.view,
                );
            }
            _ => {}
        }
        imgui_platform.handle_event(imgui_context.io_mut(), &window, &event);
    });
}
