use std::error::Error;
use std::ffi::CStr;

use ash::vk;
use ash::{Device, Instance};

/// owns the `ash::device`; dropping it destroys the device (not literally)
pub struct DeviceContext {
    pub device: Device,
    pub physical: vk::PhysicalDevice,
    pub queue: vk::Queue,
    pub queue_family: u32,
    pub memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub properties: vk::PhysicalDeviceProperties,
}

impl DeviceContext {
    /// pick a physical device with a queue family that supports both graphics
    /// and presenting to `surface`, then create the logical device on it
    pub fn create(
        instance: &Instance,
        surface_fn: &ash::khr::surface::Instance,
        surface: vk::SurfaceKHR,
    ) -> Result<Self, Box<dyn Error>> {
        let (physical, queue_family) = pick_physical_device(instance, surface_fn, surface)?;

        let properties = unsafe { instance.get_physical_device_properties(physical) };
        let name = unsafe { CStr::from_ptr(properties.device_name.as_ptr()) };
        log::info!("using physical device: {}", name.to_string_lossy());

        let priorities = [1.0f32];
        let queue_create_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family)
            .queue_priorities(&priorities);

        let extension_names = [vk::KHR_SWAPCHAIN_NAME.as_ptr()];
        let queue_create_infos = [queue_create_info];
        // fillmodenonsolid allows for the wireframe render style
        let mut features = vk::PhysicalDeviceFeatures::default();
        features.fill_mode_non_solid = vk::TRUE;
        // rust-gpu shaders declare the vulkan memory mode it
        // must be opted into or the shader module fails validation
        let mut vk12 = vk::PhysicalDeviceVulkan12Features::default().vulkan_memory_model(true);
        let create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_create_infos)
            .enabled_extension_names(&extension_names)
            .enabled_features(&features)
            .push_next(&mut vk12);

        let device = unsafe { instance.create_device(physical, &create_info, None)? };
        let queue = unsafe { device.get_device_queue(queue_family, 0) };
        let memory_properties = unsafe { instance.get_physical_device_memory_properties(physical) };
        let properties = unsafe { instance.get_physical_device_properties(physical) };
        log::info!("created logical device");

        Ok(Self {
            device,
            physical,
            queue,
            queue_family,
            memory_properties,
            properties,
        })
    }

    /// find the first memory type compatible with `requirements` that offers
    /// all of `properties`
    pub fn find_memory_type(
        &self,
        requirements: vk::MemoryRequirements,
        properties: vk::MemoryPropertyFlags,
    ) -> Result<u32, Box<dyn Error>> {
        for (index, memory_type) in self.memory_properties.memory_types.iter().enumerate() {
            let supported = requirements.memory_type_bits & (1 << index) != 0;
            if supported && memory_type.property_flags.contains(properties) {
                return Ok(index as u32);
            }
        }
        Err("no suitable memory type found".into())
    }
}

impl Drop for DeviceContext {
    fn drop(&mut self) {
        unsafe { self.device.destroy_device(None) };
        log::info!("destroyed logical device");
    }
}

/// pick the best physical device
fn pick_physical_device(
    instance: &Instance,
    surface_fn: &ash::khr::surface::Instance,
    surface: vk::SurfaceKHR,
) -> Result<(vk::PhysicalDevice, u32), Box<dyn Error>> {
    let devices = unsafe { instance.enumerate_physical_devices()? };

    let mut best: Option<(vk::PhysicalDevice, u32, i32)> = None;
    for physical in devices {
        let properties = unsafe { instance.get_physical_device_properties(physical) };
        let families = unsafe { instance.get_physical_device_queue_family_properties(physical) };

        for (index, family) in families.iter().enumerate() {
            if !family.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                continue;
            }
            let supports_present = unsafe {
                surface_fn.get_physical_device_surface_support(physical, index as u32, surface)?
            };
            if !supports_present {
                continue;
            }
            let formats =
                unsafe { surface_fn.get_physical_device_surface_formats(physical, surface)? };
            let present_modes =
                unsafe { surface_fn.get_physical_device_surface_present_modes(physical, surface)? };
            if formats.is_empty() || present_modes.is_empty() {
                continue;
            }

            let score = match properties.device_type {
                vk::PhysicalDeviceType::DISCRETE_GPU => 2,
                vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
                _ => 0,
            };
            if best.is_none_or(|(_, _, best_score)| score > best_score) {
                best = Some((physical, index as u32, score));
            }
            break;
        }
    }

    best.map(|(physical, queue_family, _)| (physical, queue_family))
        .ok_or_else(|| "no physical device with a graphics + present queue family".into())
}
