pub trait Implementation: 'static + Sized {
    fn init(&mut self, device: &wgpu::Device, renderer: &mut imgui_wgpu::Renderer);
    fn update(&mut self, ui: &mut imgui::Ui);
}

pub struct Core {
    pub context: imgui::Context,
    pub win_platform: imgui_winit_support::WinitPlatform,
    pub renderer: imgui_wgpu::Renderer,
}

impl Core {
    pub fn new<S: Implementation>(
        window: &winit::window::Window,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        format: wgpu::TextureFormat,
        state: &mut S,
    ) -> Core {
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

        let imgui_render_config = imgui_wgpu::RendererConfig::new().set_texture_format(format);

        let mut imgui_renderer =
            imgui_wgpu::Renderer::new(&mut imgui, &device, &queue, imgui_render_config);

        state.init(device, &mut imgui_renderer);

        Core {
            context: imgui,
            win_platform: platform,
            renderer: imgui_renderer,
        }
    }

    // TODO: Should draw calls return a command buffer rather than submitting themselves?
    // would allow all cmd buffers to be submitted at the same time
    pub fn draw<S: Implementation>(
        &mut self,
        state: &mut S,
        window: &winit::window::Window,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        target: &wgpu::TextureView,
    ) {
        self.win_platform
            .prepare_frame(self.context.io_mut(), &window)
            .expect("Failed to prepare frame");

        let mut ui = self.context.frame();

        state.update(&mut ui);

        self.win_platform.prepare_render(&ui, &window);

        let mut encoder: wgpu::CommandEncoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let mut imgui_renderpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
                attachment: &target,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: true,
                },
            }],
            depth_stencil_attachment: None,
        });

        self.renderer
            .render(ui.render(), queue, device, &mut imgui_renderpass)
            .expect("Rendering failed");

        drop(imgui_renderpass);
        queue.submit(Some(encoder.finish()));
    }
}
