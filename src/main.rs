mod gfx;
mod scene;
mod ui;
use cgmath::prelude::*;
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
    window::WindowBuilder,
};

pub struct NenbodyUi {
    pub last_frame: std::time::Instant,
    pub show_demo: bool,
    pub scene_extent: wgpu::Extent3d,
    pub viewport_extent: wgpu::Extent3d,
    pub camera_count: u32,
    pub viewport_camera: u32,
    pub viewport_scale: f32,
    pub texture_id: Option<imgui::TextureId>,
}

impl NenbodyUi {
    pub fn new(
        scene_extent: wgpu::Extent3d,
        viewport_extent: wgpu::Extent3d,
        camera_count: u32,
    ) -> NenbodyUi {
        NenbodyUi {
            last_frame: std::time::Instant::now(),
            show_demo: false,
            scene_extent,
            viewport_extent,
            camera_count,
            viewport_camera: 0,
            viewport_scale: 0.1,
            texture_id: None,
        }
    }
}

impl ui::Implementation for NenbodyUi {
    fn init(&mut self, device: &wgpu::Device, renderer: &mut imgui_wgpu::Renderer) {
        // Create imgui texture for drawing rendered textures in UI
        let imgui_texture_config =
            imgui_wgpu::TextureConfig::new(self.viewport_extent.width, self.viewport_extent.height)
                .set_usage(
                    wgpu::TextureUsage::COPY_SRC
                        | wgpu::TextureUsage::COPY_DST
                        | wgpu::TextureUsage::SAMPLED
                        | wgpu::TextureUsage::OUTPUT_ATTACHMENT,
                );

        let imgui_texture = imgui_wgpu::Texture::new(device, &renderer, imgui_texture_config);
        self.texture_id = Some(renderer.textures.insert(imgui_texture));
    }

    fn update(&mut self, ui: &mut imgui::Ui) {
        let imgui_window = imgui::Window::new(im_str!("Learning WGPU"));
        imgui_window
            .size([500.0, 500.0], imgui::Condition::FirstUseEver)
            .build(ui, || {
                ui.text(im_str!("Frametime: {:?}", self.last_frame.elapsed()));
                ui.separator();
                let mouse_pos = ui.io().mouse_pos;
                ui.text(im_str!(
                    "Mouse Position: ({:.1},{:.1})",
                    mouse_pos[0],
                    mouse_pos[1],
                ));
                ui.separator();
                if ui.button(im_str!("Toggle Demo"), [100., 20.]) {
                    self.show_demo = !self.show_demo
                }
                ui.separator();
                imgui::Slider::new(im_str!("Set Viewport Camera"))
                    .range(0..=self.camera_count - 1)
                    .build(&ui, &mut self.viewport_camera);
                ui.separator();
                imgui::Slider::new(im_str!("Set Viewport scale"))
                    .range(0.001..=1.0)
                    .build(&ui, &mut self.viewport_scale);
                imgui::Image::new(
                    self.texture_id.unwrap(),
                    [
                        self.scene_extent.width as f32 * self.viewport_scale,
                        self.scene_extent.height as f32 * self.viewport_scale,
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

        if self.show_demo {
            ui.show_demo_window(&mut false);
        }
    }
}

///// a 2D model with mesh and texture data for drawing
//pub struct Model {
//    vertices: Vec<gfx::Vertex>,
//    indices: Vec<u16>,
//    texture: wgpu::Texture,
//    view: wgpu::TextureView,
//    sampler: wgpu::Sampler,
//}

fn init_geometry_data() -> (Vec<gfx::Vertex>, Vec<u16>) {
    let vertex_data = [
        gfx::Vertex::new([-1, -1, 0], [0, 0]),
        gfx::Vertex::new([1, 0, 0], [0, 1]),
        gfx::Vertex::new([-1, 1, 0], [1, 1]),
    ];

    let index_data = [0, 1, 2, 0];
    (vertex_data.to_vec(), index_data.to_vec())
}

fn rotation_of(v: &cgmath::Vector3<f32>) -> cgmath::Rad<f32> {
    cgmath::Angle::atan2(v.y, v.x)
}

fn init_scene_config(
    device: &wgpu::Device,
    texture_format: wgpu::TextureFormat,
    msaa_samples: usize,
) -> (
    wgpu::BindGroupLayout,
    wgpu::PipelineLayout,
    wgpu::RenderPipeline,
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
            range: (0..std::mem::size_of::<gfx::PushConstant>() as u32),
        }],
    });

    let vertex_size = std::mem::size_of::<gfx::Vertex>();
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
        depth_stencil_state: Some(wgpu::DepthStencilStateDescriptor {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less, // 1.
            stencil: wgpu::StencilStateDescriptor::default(), // 2.
        }),
        vertex_state: vertex_state.clone(),
        sample_count: msaa_samples as u32,
        sample_mask: !0,
        alpha_to_coverage_enabled: false,
    });

    (
        scene_bind_group_layout,
        scene_pipeline_layout,
        scene_pipeline,
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
    let instance_data: Vec<[[f32; 4]; 4]> = (0..entity_count)
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

fn update_instance_random(
    instances: &mut Vec<[[f32; 4]; 4]>,
    positions: &mut Vec<cgmath::Point3<f32>>,
    velocities: &mut Vec<cgmath::Vector3<f32>>,
) {
    instances
        .into_par_iter()
        .zip(velocities)
        .zip(positions)
        .for_each(|((mat, vel), pos)| {
            let mut rng = rand::thread_rng();
            *vel += cgmath::Vector3::<f32>::new(
                rng.gen_range(-0.0001, 0.0001),
                rng.gen_range(-0.0001, 0.0001),
                0.0,
            );
            *pos += *vel;
            *mat = (cgmath::Matrix4::<f32>::from_translation(pos.to_vec())
                * cgmath::Matrix4::from_angle_z(rotation_of(vel)))
            .into();
        });
}

fn update_instance_boids(
    instances: &mut Vec<[[f32; 4]; 4]>,
    positions: &mut Vec<cgmath::Point3<f32>>,
    old_positions: &mut Vec<cgmath::Point3<f32>>,
    velocities: &mut Vec<cgmath::Vector3<f32>>,
    old_velocities: &mut Vec<cgmath::Vector3<f32>>,
) {
    // copy current values into old first, to allow for nested, mutable iteration later
    old_positions.copy_from_slice(positions.as_slice());
    old_velocities.copy_from_slice(velocities.as_slice());
    // all vectors should have same len
    let length = instances.len();

    // compute boid rules
    // http://www.kfish.org/boids/pseudocode.html
    instances
        .iter_mut()
        .zip(positions)
        .zip(velocities)
        .enumerate()
        .for_each(|(n, ((mat, boid_n_pos), boid_n_vel))| {
            let flock_center = ((old_positions.iter().enumerate().fold(
                cgmath::Vector3::new(0.0, 0.0, 0.0),
                |sum, (i, boid_i_pos)| sum + (boid_i_pos.to_vec() * ((n != i) as u32 as f32)),
            ) / (length as f32 - 1.0))
                - boid_n_pos.to_vec())
                / 100.0;

            let flock_repel = old_positions.iter().enumerate().fold(
                cgmath::Vector3::new(0.0, 0.0, 0.0),
                |sum, (i, boid_i_pos)| {
                    sum - (boid_n_pos.to_vec()
                        - (boid_i_pos.to_vec()
                            * (n != i) as u32 as f32
                            * (boid_n_pos.distance2(*boid_i_pos).abs() > 100.0) as u32 as f32))
                },
            );

            let flock_match = ((old_velocities.iter().enumerate().fold(
                cgmath::Vector3::new(0.0, 0.0, 0.0),
                |sum, (i, boid_i_vel)| sum + (boid_i_vel * ((n != i) as u32 as f32)),
            ) / ((length - 1) as f32))
                - boid_n_vel.clone())
                / 8.0;

            *boid_n_vel = (*boid_n_vel + flock_center + flock_repel + flock_match) / 100.0;
            // need new clone of update vel
            *boid_n_pos = cgmath::Point3::from_vec(boid_n_vel.clone() + boid_n_pos.to_vec());
            *mat = (cgmath::Matrix4::<f32>::from_translation(boid_n_pos.to_vec())
                * cgmath::Matrix4::from_angle_z(rotation_of(boid_n_vel)))
            .into();
        });
}

fn render<T: bytemuck::Pod>(
    render_target: &gfx::RenderTarget,
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
            attachment: &render_target.views[0],
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
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachmentDescriptor {
            attachment: &render_target.depth_views[0],
            depth_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Clear(1.0),
                store: true,
            }),
            stencil_ops: None,
        }),
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
    render_target: &gfx::RenderTarget,
    resolve_target: &gfx::ResolveTarget,
    device: &wgpu::Device,
    _queue: &wgpu::Queue,
    pipeline: &wgpu::RenderPipeline,
    bind_group: &wgpu::BindGroup,
    vertex_buf: &wgpu::Buffer,
    index_buf: &wgpu::Buffer,
    index_cnt: usize,
    instance_cnt: usize,
) {
    let span = render_target.views.len() / GRANULARITY;

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
                        attachment: &render_target.views[i],
                        resolve_target: Some(&resolve_target.views[i]),
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
                    depth_stencil_attachment: Some(
                        wgpu::RenderPassDepthStencilAttachmentDescriptor {
                            attachment: &render_target.depth_views[i],
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Clear(1.0),
                                store: true,
                            }),
                            stencil_ops: None,
                        },
                    ),
                });
                renderpass.set_pipeline(&pipeline);
                renderpass.set_bind_group(0, &bind_group, &[]);
                renderpass.set_index_buffer(index_buf.slice(..));
                renderpass.set_vertex_buffer(0, vertex_buf.slice(..));
                renderpass.set_push_constants(
                    wgpu::ShaderStage::VERTEX | wgpu::ShaderStage::FRAGMENT,
                    0,
                    bytemuck::cast_slice(&[gfx::PushConstant::new(i as u32)]),
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
    let entity_count = 10;

    // Window
    env_logger::init();
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new().build(&event_loop).unwrap();

    // GPU and surface
    let (gpu, surface) = gfx::Gpu::new(&window);

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
    let mut display = gfx::RenderTarget::new(
        &gpu.device,
        swapchain_extent,
        swapchain_desc.format,
        msaa_samples,
    );

    // eyes (resolved to resolved_eyes)
    let eye_extent = wgpu::Extent3d {
        width: 1024,
        height: 1,
        depth: entity_count as u32,
    };
    let eyes = gfx::RenderTarget::new(
        &gpu.device,
        eye_extent,
        swapchain_desc.format, // TODO: Consider other formats?
        msaa_samples,
    );
    let resolved_eyes = gfx::ResolveTarget::from_render_target(&gpu.device, &eyes);

    // viewport (resolved to gui viewport)
    let viewport_extent = wgpu::Extent3d {
        width: eye_extent.width,
        height: eye_extent.height,
        depth: 1, // Viewport is one layer of eye
    };
    let viewport = gfx::RenderTarget::new(
        &gpu.device,
        viewport_extent,
        eyes.format, // Format should match eye
        msaa_samples,
    );

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

    // Scene Config
    let (bind_group_layout, _pipeline_layout, pipeline) =
        init_scene_config(&gpu.device, swapchain_desc.format, msaa_samples as usize);

    // Placeholder position and velocity generation
    let mut rng = rand::thread_rng();
    let mut velocities: Vec<cgmath::Vector3<f32>> =
        vec![cgmath::Vector3::<f32>::new(0.0, 0.0, 0.0); entity_count];
    let mut positions: Vec<cgmath::Point3<f32>> = (0..entity_count)
        .map(|_| {
            cgmath::Point3::<f32>::new(rng.gen_range(-1.0, 1.0), rng.gen_range(-1.0, 1.0), 0.0)
        })
        .collect();
    // extra storage for update functions
    let mut old_positions = positions.clone();
    let mut old_velocities = velocities.clone();

    // Cameras
    // scene_camera
    let mut scene_position = vec![cgmath::Point3::<f32>::new(0.0, 0.0, 100.0); 1];
    let scene_look_dir = vec![-cgmath::Vector3::unit_z(); 1];
    let mut scene_cam = gfx::CameraArray::new(
        &gpu.device,
        1,
        swapchain_extent,
        cgmath::Deg(90.0),
        cgmath::Vector3::unit_x(),
    );

    // eye cameras
    let mut eye_cams = gfx::CameraArray::new(
        &gpu.device,
        entity_count,
        eye_extent,
        cgmath::Deg(90.0),
        cgmath::Vector3::unit_z(),
    );

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

    // UI
    let mut ui_state = NenbodyUi::new(swapchain_extent, viewport_extent, entity_count as u32);
    let mut gui = ui::Core::new(
        &window,
        &gpu.device,
        &gpu.queue,
        swapchain_desc.format,
        &mut ui_state,
    );

    // Command buffer list
    let mut cmd_buf_list: Vec<wgpu::CommandBuffer> = Vec::with_capacity(GRANULARITY);

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
            }
            Event::WindowEvent {
                event: WindowEvent::Resized(_),
                ..
            } => {
                let size = window.inner_size();
                // store new extent to ui_state
                ui_state.scene_extent = wgpu::Extent3d {
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
                scene_cam.resize(ui_state.scene_extent);
                display.resize(&gpu.device, ui_state.scene_extent);
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
                    VirtualKeyCode::W => scene_position[0] += cgmath::Vector3::unit_x(),
                    VirtualKeyCode::S => scene_position[0] -= cgmath::Vector3::unit_x(),
                    VirtualKeyCode::A => scene_position[0] -= cgmath::Vector3::unit_y(),
                    VirtualKeyCode::D => scene_position[0] += cgmath::Vector3::unit_y(),
                    VirtualKeyCode::Q => scene_position[0] += cgmath::Vector3::unit_z(),
                    VirtualKeyCode::E => scene_position[0] -= cgmath::Vector3::unit_z(),
                    VirtualKeyCode::Escape => *control_flow = ControlFlow::Exit,
                    _ => {}
                },
                WindowEvent::CloseRequested => {
                    *control_flow = ControlFlow::Exit;
                }
                _ => {}
            },
            Event::RedrawRequested(_) => {
                let now = Instant::now();
                gui.context
                    .io_mut()
                    .update_delta_time(now - ui_state.last_frame);
                ui_state.last_frame = now;

                let frame = match swapchain.get_current_frame() {
                    Ok(frame) => frame,
                    Err(e) => {
                        log::warn!("dropped frame: {:?}", e);
                        let size = window.inner_size();
                        // store new extent to ui_state
                        ui_state.scene_extent = wgpu::Extent3d {
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
                        scene_cam.resize(ui_state.scene_extent);
                        display.resize(&gpu.device, ui_state.scene_extent);
                        // try one last time to get swapchain frame
                        swapchain
                            .get_current_frame()
                            .expect("failed to acquire swapchain frame")
                    }
                };

                update_instance_boids(&mut instance_data, &mut positions, &mut old_positions, &mut velocities, &mut old_velocities);
                gpu.queue
                    .write_buffer(&instance_buf, 0, bytemuck::cast_slice(instance_data.as_ref()));

                // Update cameras
                eye_cams.update(&positions, &velocities);
                scene_cam.update(&scene_position, &scene_look_dir);

                eye_cams.write(&gpu.queue);
                scene_cam.write(&gpu.queue);

                // Render Scene
                render(
                    &display,
                    &frame.output.view,
                    &gpu.device,
                    &gpu.queue,
                    &pipeline,
                    &scene_bind_group,
                    &vertex_buf,
                    &index_buf,
                    index_count,
                    instance_data.len(),
                    gfx::PushConstant::new(0),
                );

                build_command_buffer_parallel(
                    &mut cmd_buf_list,
                    &eyes,
                    &resolved_eyes,
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

                let pc = gfx::PushConstant::new(ui_state.viewport_camera);
                render(
                    &viewport,
                    gui.renderer
                        .textures
                        .get(ui_state.texture_id.unwrap())
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
                    pc,
                );

                // Render UI
                gui.draw(
                    &mut ui_state,
                    &window,
                    &gpu.device,
                    &gpu.queue,
                    &frame.output.view,
                );
            }
            _ => {}
        }
        gui.win_platform
            .handle_event(gui.context.io_mut(), &window, &event);
    });
}
