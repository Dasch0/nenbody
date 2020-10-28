use bytemuck::{Pod, Zeroable};
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

pub struct Handle {
    _instance: wgpu::Instance,
    size: winit::dpi::PhysicalSize<u32>,
    surface: wgpu::Surface,
    _adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
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

#[cfg_attr(rustfmt, rustfmt_skip)]
#[allow(unused)]
pub const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0,
    0.0, 0.0, 0.5, 1.0,
);

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Vertex {
    _pos: [f32; 4],
    _tex_coord: [f32; 2],
}

fn vertex(pos: [i8; 3], tc: [i8; 2]) -> Vertex {
    Vertex {
        _pos: [pos[0] as f32, pos[1] as f32, pos[2] as f32, 1.0],
        _tex_coord: [tc[0] as f32, tc[1] as f32],
    }
}

fn rotation_of(v: &cgmath::Vector3<f32>) -> cgmath::Rad<f32> {
    cgmath::Angle::atan2(v.y, v.x)
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PushConstant {
    eye_index: u32,
    _pad: u32,
}
impl PushConstant {
    fn new(eye_index: u32) -> PushConstant {
        PushConstant { eye_index, _pad: 0 }
    }
}

#[derive(Debug)]
struct BufferDimensions {
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

fn init_geometry_data() -> (Vec<Vertex>, Vec<u16>) {
    let vertex_data = [
        vertex([-1, -1, 0], [0, 0]),
        vertex([1, 0, 0], [0, 1]),
        vertex([-1, 1, 0], [1, 1]),
    ];

    let index_data = [0, 1, 2, 0];
    (vertex_data.to_vec(), index_data.to_vec())
}

async fn init_gpu(window: &Window) -> Arc<Handle> {
    log::info!("Initializing instance...");
    let _instance = wgpu::Instance::new(wgpu::BackendBit::PRIMARY);

    log::info!("Obtaining window surface...");
    let (size, surface) = unsafe {
        let size = window.inner_size();
        let surface = _instance.create_surface(window);
        (size, surface)
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

    Arc::new(Handle {
        _instance,
        size,
        surface,
        _adapter,
        device,
        queue,
    })
}

fn build_swapchain(
    device: &wgpu::Device,
    surface: &wgpu::Surface,
    size: winit::dpi::PhysicalSize<u32>,
) -> (wgpu::SwapChainDescriptor, wgpu::SwapChain) {
    log::info!("Initializing swapchain...");
    let swapchain_desc = wgpu::SwapChainDescriptor {
        usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
        format: wgpu::TextureFormat::Bgra8UnormSrgb,
        width: size.width,
        height: size.height,
        present_mode: wgpu::PresentMode::Immediate,
    };
    let swapchain = device.create_swap_chain(surface, &swapchain_desc);
    (swapchain_desc, swapchain)
}

fn init_texture_array(
    device: &wgpu::Device,
    extent: wgpu::Extent3d,
    format: wgpu::TextureFormat,
    msaa_samples: usize,
) -> (wgpu::Texture, Arc<Vec<wgpu::TextureView>>) {
    // Create MSAA textures
    let texture: wgpu::Texture = device.create_texture(&wgpu::TextureDescriptor {
        size: extent,
        mip_level_count: 1,
        sample_count: msaa_samples as u32,
        dimension: wgpu::TextureDimension::D2,
        format: format,
        usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT | wgpu::TextureUsage::COPY_SRC,
        label: None,
    });

    // Create views for each layer of texture
    let views: Arc<Vec<wgpu::TextureView>> = Arc::new(
        (0..extent.depth)
            .into_par_iter()
            .map(|i| {
                texture.create_view(&wgpu::TextureViewDescriptor {
                    label: None,
                    format: Some(format),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    aspect: wgpu::TextureAspect::All,
                    base_mip_level: 0,
                    level_count: None,
                    base_array_layer: 1,
                    array_layer_count: std::num::NonZeroU32::new(i),
                })
            })
            .collect(),
    );
    (texture, views)
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

fn init_capture(
    device: &wgpu::Device,
    extent: wgpu::Extent3d,
) -> (
    wgpu::Texture,
    Vec<wgpu::TextureView>,
    BufferDimensions,
    wgpu::Buffer,
) {
    // Capture Texture & Buffer
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Bgra8UnormSrgb,
        usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT | wgpu::TextureUsage::COPY_SRC,
        label: Some(&format!("Capture texture")),
    });

    // Create views for each layer of texture
    let views: Vec<wgpu::TextureView> = (0..extent.depth)
        .into_par_iter()
        .map(|i| {
            texture.create_view(&wgpu::TextureViewDescriptor {
                label: None,
                format: Some(wgpu::TextureFormat::Bgra8UnormSrgb),
                dimension: Some(wgpu::TextureViewDimension::D2),
                aspect: wgpu::TextureAspect::All,
                base_mip_level: 0,
                level_count: None,
                base_array_layer: i,
                array_layer_count: std::num::NonZeroU32::new(1),
            })
        })
        .collect();

    let dimension = BufferDimensions::new(
        extent.width as usize,
        extent.height as usize,
        extent.depth as usize,
    );

    log::info!("capture buffer dimension: {:#?}", dimension);

    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Capture Buffer"),
        size: 4* (dimension.padded_bytes_per_row * dimension.height * dimension.depth) as u64,
        usage: wgpu::BufferUsage::MAP_READ | wgpu::BufferUsage::COPY_DST,
        mapped_at_creation: false,
    });

    (texture, views, dimension, buffer)
}

fn init_scene_data(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> (
    Arc<wgpu::Buffer>,
    Arc<wgpu::Buffer>,
    usize,
    Arc<RwLock<Vec<[[f32; 4]; 4]>>>,
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
    let instance_data: Arc<RwLock<Vec<[[f32; 4]; 4]>>> = Arc::new(RwLock::new(
        (0..2000)
            .map(|_| {
                cgmath::Matrix4::<f32>::from_translation(
                    [rng.gen_range(-1.0, 1.0), rng.gen_range(-1.0, 1.0), 0.0].into(),
                )
                .into()
            })
            .collect(),
    ));

    let instance_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Instance Buffer"),
        contents: bytemuck::cast_slice(instance_data.read().unwrap().as_ref()),
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

fn imgui_resize_texture(
    device: &wgpu::Device,
    imgui_renderer: &mut imgui_wgpu::Renderer,
    extent: wgpu::Extent3d,
    imgui_texture_id: imgui::TextureId,
) {
    // Create imgui texture for drawing rendered textures in UI
    let imgui_texture_config = imgui_wgpu::TextureConfig::new(extent.width, extent.height)
        .set_usage(
            wgpu::TextureUsage::COPY_SRC
                | wgpu::TextureUsage::COPY_DST
                | wgpu::TextureUsage::SAMPLED
                | wgpu::TextureUsage::OUTPUT_ATTACHMENT,
        );

    let imgui_texture = imgui_wgpu::Texture::new(device, imgui_renderer, imgui_texture_config);
    imgui_renderer
        .textures
        .replace(imgui_texture_id, imgui_texture)
        .expect("invalid imgui_texture_id");
}
// dedicated to SAMI!
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

fn build_camera(
    aspect_ratio: f32,
    eye: cgmath::Point3<f32>,
    center: cgmath::Point3<f32>,
    up: cgmath::Vector3<f32>,
    fov: cgmath::Deg<f32>,
) -> [[f32; 4]; 4] {
    let proj = cgmath::perspective(fov, aspect_ratio, 1.0, 1000.0);
    let view = cgmath::Matrix4::look_at(eye, center, up);
    let correction = OPENGL_TO_WGPU_MATRIX;
    (correction * proj * view).into()
}

fn init_camera_list(
    device: &wgpu::Device,
    extent: wgpu::Extent3d,
    eye_list: &Vec<cgmath::Point3<f32>>,
    center_list: &Vec<cgmath::Vector3<f32>>,
    normal: cgmath::Vector3<f32>,
    fov: cgmath::Deg<f32>,
) -> (Vec<[[f32; 4]; 4]>, wgpu::Buffer) {
    log::info!("Creating Cameras...");
    let camera_list: Vec<[[f32; 4]; 4]> = (0..eye_list.len())
        .into_par_iter()
        .map(|i| {
            build_camera(
                extent.width as f32 / extent.height as f32,
                eye_list[i],
                cgmath::EuclideanSpace::from_vec(center_list[i]),
                normal,
                fov,
            )
        })
        .collect();

    let camera_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&camera_list),
        usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_DST,
    });
    (camera_list, camera_buf)
}

// TODO: support camera rotations
fn update_camera_list(
    queue: &wgpu::Queue,
    camera_list: &mut Vec<[[f32; 4]; 4]>,
    camera_buf: &wgpu::Buffer,
    extent: wgpu::Extent3d,
    eye_list: &Vec<cgmath::Point3<f32>>,
    center_list: &Vec<cgmath::Vector3<f32>>,
    normal: cgmath::Vector3<f32>,
    fov: cgmath::Deg<f32>,
) {
    *camera_list = (0..eye_list.len())
        .into_par_iter()
        .map(|i| {
            build_camera(
                extent.width as f32 / extent.height as f32,
                eye_list[i],
                cgmath::EuclideanSpace::from_vec(center_list[i]),
                normal,
                fov,
            )
        })
        .collect();
    queue.write_buffer(&camera_buf, 0, bytemuck::cast_slice(camera_list.as_ref()));
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
        .into_iter()
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

fn build_command_buffer(
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
) -> wgpu::CommandBuffer {
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    (0..msaa_target_list.len()).for_each(|i| {
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
    });
    encoder.finish()
}

fn capture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    src_texture: &wgpu::Texture,
    extent: wgpu::Extent3d,
    dst_buffer: &wgpu::Buffer,
    buffer_dimension: &BufferDimensions,
) {
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Capture Command"),
    });

    encoder.copy_texture_to_buffer(
        wgpu::TextureCopyView {
            texture: src_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
        },
        wgpu::BufferCopyView {
            buffer: &dst_buffer,
            layout: wgpu::TextureDataLayout {
                offset: 0,
                bytes_per_row: buffer_dimension.padded_bytes_per_row as u32,
                rows_per_image: extent.height,
            },
        },
        wgpu::Extent3d {
            width: extent.width,
            height: extent.height,
            depth: extent.depth - 1,
        },
    );
    let command_buffer = encoder.finish();
    queue.submit(Some(command_buffer));
}

fn main() {
    // Init Window
    env_logger::init();
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new().build(&event_loop).unwrap();

    // Constants
    let msaa_samples = 8;

    // Device, Queue, Surface
    let h = futures::executor::block_on(init_gpu(&window));

    let (
        vertex_buf,
        index_buf,
        index_count,
        instance_data,
        instance_buf,
        _texture_extent,
        _texture,
        texture_view,
        texture_sampler,
    ) = init_scene_data(&h.device, &h.queue);

    // Swapchain
    let (swapchain_desc, mut swapchain) = build_swapchain(&h.device, &h.surface, h.size);
    let swapchain_extent = wgpu::Extent3d {
        width: swapchain_desc.width,
        height: swapchain_desc.height,
        depth: 1,
    };

    let eye_extent = wgpu::Extent3d {
        width: 1024,
        height: 1,
        depth: instance_data.read().unwrap().len() as u32,
    };

    // Render Targets
    // scene msaa target
    let mut _scene_msaa_texture = h.device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: swapchain_extent,
        mip_level_count: 1,
        sample_count: msaa_samples,
        dimension: wgpu::TextureDimension::D2,
        format: swapchain_desc.format,
        usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT | wgpu::TextureUsage::COPY_SRC,
    });

    let mut scene_msaa_view =
        _scene_msaa_texture.create_view(&wgpu::TextureViewDescriptor::default());

    // eye msaa target
    let (_eye_texture, eye_views) = init_texture_array(
        &h.device,
        eye_extent,
        swapchain_desc.format,
        msaa_samples as usize,
    );

    // eye resolve targets
    let (capture_texture, capture_views, capture_dimension, capture_buffer) =
        init_capture(&h.device, eye_extent);

    // viewport msaa target
    let viewport_texture = h.device.create_texture(&wgpu::TextureDescriptor {
        size: wgpu::Extent3d {
            width: eye_extent.width,
            height: eye_extent.height,
            depth: 1,
        },
        mip_level_count: 1,
        sample_count: msaa_samples,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Bgra8UnormSrgb,
        usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT | wgpu::TextureUsage::COPY_SRC,
        label: Some(&format!("Capture texture")),
    });

    // Create views for each layer of texture
    let viewport_view = viewport_texture.create_view(&wgpu::TextureViewDescriptor::default());

    // UI
    let (mut imgui_context, mut imgui_platform, mut imgui_renderer, imgui_texture_id) = init_imgui(
        &window,
        &h.device,
        &h.queue,
        swapchain_desc.format,
        eye_extent,
    );

    // Scene Config
    let (bind_group_layout, _pipeline_layout, pipeline) =
        init_scene_config(&h.device, swapchain_desc.format, msaa_samples as usize);

    // Placeholder entity data
    let mut rng = rand::thread_rng();
    let mut velocities: Vec<cgmath::Vector3<f32>> =
        vec![cgmath::Vector3::<f32>::new(0.0, 0.0, 0.0); instance_data.read().unwrap().len()];
    let mut positions: Vec<cgmath::Point3<f32>> = (0..instance_data.read().unwrap().len())
        .map(|_| {
            cgmath::Point3::<f32>::new(rng.gen_range(-1.0, 1.0), rng.gen_range(-1.0, 1.0), 0.0)
        })
        .collect();

    // Cameras
    // scene_camera
    let mut scene_position = vec![cgmath::Point3::<f32>::new(0.0, 0.0, 100.0); 1];
    let mut scene_center = vec![cgmath::Vector3::<f32>::new(0.0, 0.0, 0.0); 1];

    let (mut scene_camera, scene_camera_buffer) = init_camera_list(
        &h.device,
        swapchain_extent,
        &scene_position,
        &scene_center,
        cgmath::Vector3::unit_x(),
        cgmath::Deg(90f32),
    );

    // eye cameras
    let (mut camera_list, camera_buf) = init_camera_list(
        &h.device,
        eye_extent,
        &positions,
        &velocities,
        cgmath::Vector3::unit_z(),
        cgmath::Deg(0.1f32),
    );

    log::info!("Creating bind groups...");
    let scene_bind_group = h.device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(scene_camera_buffer.slice(..)),
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

    let eye_bind_group = h.device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(camera_buf.slice(..)),
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
        camera_count: instance_data.read().unwrap().len() as u32,
        scene_camera: 0,
        viewport_camera: 1,
        viewport_scale: 0.15,
        zoom: 1.0,
        offset: [0.0, 0.0],
        texture_id: imgui_texture_id,
    };
    drop(swapchain_extent);

    let instance_data_clone = instance_data.clone();
    let h_clone = h.clone();

    let mut cmd_buf_list: Vec<wgpu::CommandBuffer> = Vec::with_capacity(GRANULARITY);
    log::info!("Starting event loop!");

    let mut last_update_inst = std::time::Instant::now();
    crossbeam::thread::scope(|s| {
        s.spawn(|_| {
            loop {
                // Update entities
                {
                    let inst = &mut *instance_data_clone.write().unwrap();
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
                    h_clone.queue.write_buffer(
                        &instance_buf,
                        0,
                        bytemuck::cast_slice(inst.as_ref()),
                    );
                    // Update cameras
                    update_camera_list(
                        &h_clone.queue,
                        &mut camera_list,
                        &camera_buf,
                        eye_extent,
                        &positions,
                        &velocities,
                        cgmath::Vector3::unit_z(),
                        cgmath::Deg(0.1f32),
                    );
                }
                std::thread::sleep(std::time::Duration::new(0, 1000));

            }
        });

        event_loop.run(move |event, _, control_flow| {
            // Set control flow to wait min(next event, 10ms)
            *control_flow = ControlFlow::WaitUntil(Instant::now() + Duration::from_millis(10));

            // Process events
            match event {
                Event::MainEventsCleared => {
                    if last_update_inst.elapsed() > Duration::from_millis(20) {
                        window.request_redraw();
                        last_update_inst = Instant::now();
                    }
                }
                Event::WindowEvent {
                    event: WindowEvent::Resized(_),
                    ..
                } => {
                    let size = window.inner_size();
                    ui_state.extent = wgpu::Extent3d {
                        width: size.width,
                        height: size.height,
                        depth: 1,
                    };

                    log::info!("Resizing to {:?}", size);

                    // Swapchain
                    build_swapchain(&h.device, &h.surface, size);

                    update_camera_list(
                        &h.queue,
                        &mut scene_camera,
                        &scene_camera_buffer,
                        ui_state.extent,
                        &scene_position,
                        &scene_center,
                        cgmath::Vector3::unit_x(),
                        cgmath::Deg(90f32),
                    );

                    // scene msaa target
                    _scene_msaa_texture = h.device.create_texture(&wgpu::TextureDescriptor {
                        label: None,
                        size: ui_state.extent,
                        mip_level_count: 1,
                        sample_count: msaa_samples,
                        dimension: wgpu::TextureDimension::D2,
                        format: swapchain_desc.format,
                        usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT
                            | wgpu::TextureUsage::COPY_SRC,
                    });

                    scene_msaa_view =
                        _scene_msaa_texture.create_view(&wgpu::TextureViewDescriptor::default());
                }

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
                            scene_center[ui_state.scene_camera as usize] +=
                                cgmath::Vector3::unit_x();
                        }
                        VirtualKeyCode::S => {
                            scene_position[ui_state.scene_camera as usize] -=
                                cgmath::Vector3::unit_x();
                            scene_center[ui_state.scene_camera as usize] -=
                                cgmath::Vector3::unit_x();
                        }
                        VirtualKeyCode::A => {
                            scene_position[ui_state.scene_camera as usize] -=
                                cgmath::Vector3::unit_y();
                            scene_center[ui_state.scene_camera as usize] -=
                                cgmath::Vector3::unit_y();
                        }
                        VirtualKeyCode::D => {
                            scene_position[ui_state.scene_camera as usize] +=
                                cgmath::Vector3::unit_y();
                            scene_center[ui_state.scene_camera as usize] +=
                                cgmath::Vector3::unit_y();
                        }
                        VirtualKeyCode::Q => {
                            scene_position[ui_state.scene_camera as usize] +=
                                cgmath::Vector3::unit_z();
                            scene_center[ui_state.scene_camera as usize] +=
                                cgmath::Vector3::unit_z();
                        }
                        VirtualKeyCode::E => {
                            scene_position[ui_state.scene_camera as usize] -=
                                cgmath::Vector3::unit_z();
                            scene_center[ui_state.scene_camera as usize] -=
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
                            swapchain = h.device.create_swap_chain(&h.surface, &swapchain_desc);
                            swapchain
                                .get_current_frame()
                                .expect("Failed to acquire next swap chain texture!")
                        }
                    };

                    update_camera_list(
                        &h.queue,
                        &mut scene_camera,
                        &scene_camera_buffer,
                        swapchain_extent,
                        &scene_position,
                        &scene_center,
                        cgmath::Vector3::unit_x(),
                        cgmath::Deg(90f32),
                    );

                    // Render Scene
                    render(
                        &scene_msaa_view,
                        &frame.output.view,
                        &h.device,
                        &h.queue,
                        &pipeline,
                        &scene_bind_group,
                        &vertex_buf,
                        &index_buf,
                        index_count,
                        instance_data.read().unwrap().len(),
                        PushConstant::new(0),
                    );

                    build_command_buffer_parallel(
                        &mut cmd_buf_list,
                        &eye_views,
                        &capture_views,
                        &h.device,
                        &h.queue,
                        &pipeline,
                        &eye_bind_group,
                        &vertex_buf,
                        &index_buf,
                        index_count,
                        instance_data.read().unwrap().len(),
                    );

                    // Render eyes
                    h.queue.submit(cmd_buf_list.drain(..));

                    // Render viewport
                    render(
                        &viewport_view,
                        imgui_renderer
                            .textures
                            .get(imgui_texture_id)
                            .expect("failed to find imgui texture")
                            .view(),
                        &h.device,
                        &h.queue,
                        &pipeline,
                        &eye_bind_group,
                        &vertex_buf,
                        &index_buf,
                        index_count,
                        instance_data.read().unwrap().len(),
                        PushConstant::new(ui_state.viewport_camera),
                    );

                    log::info!("extent {:#?}", eye_extent);
                    log::info!("dimension {:#?}", capture_dimension);
                    capture(
                        &h.device,
                        &h.queue,
                        &capture_texture,
                        eye_extent,
                        &capture_buffer,
                        &capture_dimension,
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
                        &h.device,
                        &h.queue,
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
    })
    .unwrap();
}
