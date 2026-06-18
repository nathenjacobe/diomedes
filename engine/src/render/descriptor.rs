use std::error::Error;

use ash::Device;
use ash::vk;

/// descriptor set layout, pool and set for the mesh uniform buffer:
/// one uniform buffer bound at set 0, visible to the vertex stage; the
/// shader addresses instances by index, so no dynamic offsets are needed
pub struct DescriptorSet {
    pub layout: vk::DescriptorSetLayout,
    pub pool: vk::DescriptorPool,
    pub set: vk::DescriptorSet,
}

impl DescriptorSet {
    pub fn create(device: &Device) -> Result<Self, Box<dyn Error>> {
        let binding = vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX);

        let bindings = [binding];
        let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let layout = unsafe { device.create_descriptor_set_layout(&layout_info, None)? };

        let pool_size = vk::DescriptorPoolSize {
            ty: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: 1,
        };
        let pool_sizes = [pool_size];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(1);
        let pool = unsafe { device.create_descriptor_pool(&pool_info, None)? };

        let layouts = [layout];
        let allocate_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(&layouts);
        let set = unsafe { device.allocate_descriptor_sets(&allocate_info)? }[0];

        Ok(Self { layout, pool, set })
    }

    /// point the set's binding at `buffer`;
    pub fn update(&self, device: &Device, buffer: vk::Buffer) {
        let buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE)];
        let write = vk::WriteDescriptorSet::default()
            .dst_set(self.set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(&buffer_info);

        let writes = [write];
        unsafe { device.update_descriptor_sets(&writes, &[]) };
    }

    pub fn destroy(&mut self, device: &Device) {
        unsafe {
            device.destroy_descriptor_pool(self.pool, None);
            device.destroy_descriptor_set_layout(self.layout, None);
        }
        log::info!("destroyed descriptor set");
    }
}
