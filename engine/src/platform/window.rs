use winit::dpi::LogicalSize;
use winit::error::OsError;
use winit::event_loop::ActiveEventLoop;
use winit::window::Window;

/// create the application window on the given event loop;
///
/// must be called from `applicationhandler::resumed`, as window creation
/// requires an `activeeventloop` and the window must live for the whole
/// event loop run
pub fn create_window(event_loop: &ActiveEventLoop) -> Result<Window, OsError> {
    let attributes = Window::default_attributes()
        .with_title("diomedes")
        .with_inner_size(LogicalSize::new(1280.0, 720.0));

    event_loop.create_window(attributes)
}
