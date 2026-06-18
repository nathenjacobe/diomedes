use std::error::Error;
use std::ffi::{CStr, c_char, c_void};

use ash::ext::debug_utils;
use ash::khr::surface;
use ash::vk;
use ash::{Entry, Instance};
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawDisplayHandle};
use winit::window::Window;

/// vulkan instance with the platform surface extension loader and an optional
/// validation debug messenger; owns the entry, instance, messenger and can
/// create/destroy surfaces for windows
pub struct InstanceContext {
    surface_fn: surface::Instance,
    messenger: Option<vk::DebugUtilsMessengerEXT>,
    debug_utils: Option<debug_utils::Instance>,
    instance: Instance,
    entry: Entry,
}

impl InstanceContext {
    /// create the instance from the event loop's display handle, enabling all
    /// surface extensions the active display server requires plus the
    /// validation layer and debug utils extension when available
    pub fn new(display: RawDisplayHandle) -> Result<Self, Box<dyn Error>> {
        let entry = unsafe { Entry::load()? };

        let mut extension_names = ash_window::enumerate_required_extensions(display)?.to_vec();
        let mut layer_names: Vec<*const c_char> = Vec::new();

        let enable_debug = debug_utils_available(&entry);
        if enable_debug {
            extension_names.push(vk::EXT_DEBUG_UTILS_NAME.as_ptr());
            layer_names.push(c"VK_LAYER_KHRONOS_validation".as_ptr());
        }
        log::info!(
            "enabled instance extensions: {:?}",
            extension_names
                .iter()
                .map(|name| unsafe { CStr::from_ptr(*name) }.to_string_lossy())
                .collect::<Vec<_>>()
        );

        let application_info = vk::ApplicationInfo::default()
            .application_name(c"Diomedes")
            .application_version(vk::make_api_version(0, 1, 0, 0))
            .engine_name(c"Diomedes Engine")
            .engine_version(vk::make_api_version(0, 1, 0, 0))
            .api_version(vk::API_VERSION_1_3);

        let create_info = vk::InstanceCreateInfo::default()
            .application_info(&application_info)
            .enabled_extension_names(&extension_names)
            .enabled_layer_names(&layer_names);

        let instance = unsafe { entry.create_instance(&create_info, None)? };
        log::info!("created vulkan instance");

        log_physical_devices(&instance)?;

        let surface_fn = surface::Instance::new(&entry, &instance);

        let (debug_utils, messenger) = if enable_debug {
            let debug_utils = debug_utils::Instance::new(&entry, &instance);
            let messenger_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
                .message_severity(
                    vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
                        | vk::DebugUtilsMessageSeverityFlagsEXT::INFO
                        | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                        | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
                )
                .message_type(
                    vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                        | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                )
                .pfn_user_callback(Some(debug_callback));
            let messenger =
                unsafe { debug_utils.create_debug_utils_messenger(&messenger_info, None)? };
            log::info!("installed debug messenger");
            (Some(debug_utils), Some(messenger))
        } else {
            (None, None)
        };

        Ok(Self {
            surface_fn,
            messenger,
            debug_utils,
            instance,
            entry,
        })
    }

    pub fn instance(&self) -> &Instance {
        &self.instance
    }

    pub fn surface_fn(&self) -> &surface::Instance {
        &self.surface_fn
    }

    /// create a surface for the given window on this instance
    pub fn create_surface(&self, window: &Window) -> Result<vk::SurfaceKHR, Box<dyn Error>> {
        unsafe {
            ash_window::create_surface(
                &self.entry,
                &self.instance,
                window.display_handle()?.as_raw(),
                window.window_handle()?.as_raw(),
                None,
            )
        }
        .map_err(Into::into)
    }

    pub fn destroy_surface(&self, surface: vk::SurfaceKHR) {
        unsafe { self.surface_fn.destroy_surface(surface, None) };
    }
}

impl Drop for InstanceContext {
    fn drop(&mut self) {
        if let Some(debug_utils) = &self.debug_utils {
            if let Some(messenger) = self.messenger.take() {
                unsafe { debug_utils.destroy_debug_utils_messenger(messenger, None) };
            }
        }
        unsafe { self.instance.destroy_instance(None) };
        log::info!("destroyed vulkan instance");
    }
}

/// whether the validation layer and debug utils extension are available
fn debug_utils_available(entry: &Entry) -> bool {
    let extensions =
        unsafe { entry.enumerate_instance_extension_properties(None) }.unwrap_or_default();
    let layers = unsafe { entry.enumerate_instance_layer_properties() }.unwrap_or_default();

    let has_extension = extensions
        .iter()
        .any(|e| extension_name_eq(&e.extension_name, vk::EXT_DEBUG_UTILS_NAME));
    let has_layer = layers
        .iter()
        .any(|l| extension_name_eq(&l.layer_name, c"VK_LAYER_KHRONOS_validation"));
    has_extension && has_layer
}

fn extension_name_eq(name: &[c_char], expected: &CStr) -> bool {
    let name = unsafe { CStr::from_ptr(name.as_ptr()) };
    name == expected
}

unsafe extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _types: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user_data: *mut c_void,
) -> vk::Bool32 {
    let message = unsafe { (&*data).message_as_c_str() }
        .map(|message| message.to_string_lossy())
        .unwrap_or_else(|| "<no message>".into());
    match severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => log::error!("vulkan: {message}"),
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => log::warn!("vulkan: {message}"),
        vk::DebugUtilsMessageSeverityFlagsEXT::INFO => log::info!("vulkan: {message}"),
        _ => log::debug!("vulkan: {message}"),
    }
    vk::FALSE
}

fn log_physical_devices(instance: &Instance) -> Result<(), Box<dyn Error>> {
    let devices = unsafe { instance.enumerate_physical_devices()? };

    if devices.is_empty() {
        log::warn!("no physical devices available");
        return Ok(());
    }

    for device in devices {
        let properties = unsafe { instance.get_physical_device_properties(device) };
        let name = unsafe { CStr::from_ptr(properties.device_name.as_ptr()) };
        log::info!("found physical device: {}", name.to_string_lossy());
    }

    Ok(())
}
