use std::error::Error;
use std::time::Instant;

use winit::application::ApplicationHandler;
use winit::error::EventLoopError;
use winit::event::{DeviceEvent, DeviceId, ElementState, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, DeviceEvents, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::raw_window_handle::HasDisplayHandle;
use winit::window::{CursorGrabMode, Window, WindowId};

use crate::platform::input::Input;
use crate::platform::window;
use crate::render::Renderer;
use crate::scene::Scene;
use crate::ui::UiFrame;

/// application shell
/// owns the winit event loop, the window, the renderer,
/// the scene, the per-frame input snapshot and egui too
/// `update` is called once per frame with the renderer, the mutable scene
/// (physics / gameplay mutate it here), the input snapshot, the egui context
/// (build ui directly with egui) and the frame delta in seconds; runs once
/// the renderer is ready; the window redraws after every update;
pub struct Diomedes<F: FnMut(&mut Renderer, &mut Scene, &Input, &egui::Context, f64)> {
    event_loop: Option<EventLoop<()>>,
    renderer: Renderer,
    window: Option<Window>,
    scene: Scene,
    input: Input,
    cursor_locked: bool,
    egui_ctx: egui::Context,
    ui_state: Option<egui_winit::State>,
    ui_frame: Option<UiFrame>,
    update: F,
    last_tick: Option<Instant>,
}

impl<F: FnMut(&mut Renderer, &mut Scene, &Input, &egui::Context, f64)> Diomedes<F> {
    /// build the shell from the initial scene state; the scene can grow and
    /// be mutated from the update callback at any time;
    pub fn new(scene: Scene, update: F) -> Result<Self, Box<dyn Error>> {
        let event_loop = EventLoop::new()?;

        // the instance is created from the event loop's display handle, so
        // all platform surface extensions required by the display server are
        // enabled before the window exists
        let renderer = Renderer::new(event_loop.display_handle()?.as_raw())?;

        Ok(Self {
            event_loop: Some(event_loop),
            renderer,
            window: None,
            scene,
            input: Input::default(),
            cursor_locked: false,
            egui_ctx: egui::Context::default(),
            ui_state: None,
            ui_frame: None,
            update,
            last_tick: None,
        })
    }

    pub fn with_cursor_lock(mut self) -> Self {
        self.cursor_locked = true;
        self
    }

    pub fn run(mut self) -> Result<(), EventLoopError> {
        let event_loop = self.event_loop.take().expect("event loop exists");
        event_loop.run_app(&mut self)
    }
}

impl<F: FnMut(&mut Renderer, &mut Scene, &Input, &egui::Context, f64)> ApplicationHandler
    for Diomedes<F>
{
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        // raw device events (mouse motion deltas) are only delivered when
        // explicitly requested
        event_loop.listen_device_events(DeviceEvents::Always);

        let window = match window::create_window(event_loop) {
            Ok(window) => {
                log::info!("created window");
                window
            }
            Err(error) => {
                log::error!("failed to create window: {error}");
                event_loop.exit();
                return;
            }
        };

        if let Err(error) = self.renderer.attach(&window) {
            log::error!("failed to attach surface to window: {error}");
            event_loop.exit();
            return;
        }
        if let Err(error) = self.renderer.prepare(&window) {
            log::error!("failed to prepare renderer: {error}");
            event_loop.exit();
            return;
        }

        // egui needs the window as its display target and scale source
        self.ui_state = Some(egui_winit::State::new(
            self.egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &window,
            None,
            None,
            None,
        ));

        if self.cursor_locked {
            lock_cursor(&window);
        }

        self.window = Some(window);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        // the game camera gets scroll input regardless of egui: the ui is a
        // passive overlay and never needs exclusive wheel access; egui still
        // receives the event for its own widgets (sliders, scroll areas)
        if let WindowEvent::MouseWheel { delta, .. } = &event {
            let lines = match delta {
                MouseScrollDelta::LineDelta(_, y) => *y as f64,
                MouseScrollDelta::PixelDelta(pos) => pos.y as f64 / 50.0,
            };
            log::debug!("scroll wheel: {lines:.2} lines");
            self.input.add_scroll(lines);
        }

        // egui sees events first; if it consumes one (click on ui, text
        // input), it does not reach the game input
        if let (Some(window), Some(ui_state)) = (&self.window, &mut self.ui_state) {
            let response = ui_state.on_window_event(window, &event);
            if response.repaint {
                window.request_redraw();
            }
            if response.consumed {
                return;
            }
        }

        match event {
            WindowEvent::CloseRequested => {
                log::info!("close requested");
                event_loop.exit();
            }
            WindowEvent::KeyboardInput { event, .. } => {
                match event.state {
                    ElementState::Pressed => self.input.key_pressed(event.logical_key.clone()),
                    ElementState::Released => self.input.key_released(event.logical_key.clone()),
                }
                if event.logical_key == Key::Named(NamedKey::Escape)
                    && event.state == ElementState::Pressed
                {
                    log::info!("escape pressed");
                    event_loop.exit();
                }
            }
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button,
                ..
            } => self.input.mouse_button_pressed(button),
            WindowEvent::CursorEntered { .. } => {
                // on wayland, cursor visibility only applies while the
                // pointer is over the window, and the enter path may reset
                // it; re-hide on every entry
                // todo: it still doesn't work...
                if self.cursor_locked {
                    if let Some(window) = &self.window {
                        window.set_cursor_visible(false);
                    }
                }
            }
            WindowEvent::Resized(size) => self.renderer.on_resized(size),
            WindowEvent::RedrawRequested => {
                if let Err(error) = self
                    .renderer
                    .render_frame(&self.scene, self.ui_frame.as_mut())
                {
                    log::error!("failed to render frame: {error}");
                }
            }
            _ => {}
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        event: DeviceEvent,
    ) {
        match event {
            DeviceEvent::MouseMotion { delta } => self.input.add_mouse_delta(delta.0, delta.1),
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if !self.renderer.ready() {
            return;
        }

        let window = self
            .window
            .as_ref()
            .expect("window exists when renderer is ready");

        // run the egui frame: gather input; update callback builds the
        // ui, collect the output for the renderer
        let raw_input = self
            .ui_state
            .as_mut()
            .expect("ui state exists when renderer is ready")
            .take_egui_input(window);
        self.egui_ctx.begin_pass(raw_input);

        let now = Instant::now();
        let delta = self
            .last_tick
            .map(|tick| now.duration_since(tick).as_secs_f64())
            .unwrap_or(0.0);
        self.last_tick = Some(now);

        (self.update)(
            &mut self.renderer,
            &mut self.scene,
            &self.input,
            &self.egui_ctx,
            delta,
        );
        self.input.clear_frame();

        let full_output = self.egui_ctx.end_pass();
        let egui::FullOutput {
            platform_output,
            shapes,
            textures_delta,
            pixels_per_point,
            viewport_output: _,
        } = full_output;
        self.ui_state
            .as_mut()
            .expect("ui state exists")
            .handle_platform_output(window, platform_output);
        self.ui_frame = Some(UiFrame::new(
            &self.egui_ctx,
            shapes,
            textures_delta,
            pixels_per_point,
        ));

        window.request_redraw();
    }

    fn suspended(&mut self, _event_loop: &ActiveEventLoop) {
        self.renderer.detach();
        self.window = None;
        self.ui_state = None;
        self.ui_frame = None;
    }
}

fn lock_cursor(window: &Window) {
    if window.set_cursor_grab(CursorGrabMode::Locked).is_err() {
        if let Err(error) = window.set_cursor_grab(CursorGrabMode::Confined) {
            log::warn!("failed to grab cursor: {error}");
        }
    }
    window.set_cursor_visible(false);
    log::info!("cursor locked");
}
