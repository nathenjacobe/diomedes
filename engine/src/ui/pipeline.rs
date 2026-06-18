use std::error::Error;
use std::mem::size_of;

use ash::vk;

use crate::render::device::DeviceContext;
use crate::render::shader;

/// one egui vertex: position (physical px, y down), uv, premultiplied srgba
#[repr(C)]
#[derive(Clone, Copy)]
pub struct UiVertex {
    pub pos: [f32; 2],
    pub uv: [f32; 2],
    pub color: [u8; 4],
}

/// create the egui screen-space pipeline; `render_pass` is the color-only ui
/// pass created by the frame
pub fn create_pipeline(
    device: &DeviceContext,
    descriptor_layout: vk::DescriptorSetLayout,
    render_pass: vk::RenderPass,
) -> Result<(vk::Pipeline, vk::PipelineLayout), Box<dyn Error>> {
    let vertex_module = shader::compile(
        &device.device,
        shader::UI_VERTEX_SOURCE,
        shaderc::ShaderKind::Vertex,
        "ui.vert",
    )?;
    let fragment_module = shader::compile(
        &device.device,
        shader::UI_FRAGMENT_SOURCE,
        shaderc::ShaderKind::Fragment,
        "ui.frag",
    )?;

    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vertex_module)
            .name(c"main"),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(fragment_module)
            .name(c"main"),
    ];

    let binding = vk::VertexInputBindingDescription::default()
        .binding(0)
        .stride(size_of::<UiVertex>() as u32)
        .input_rate(vk::VertexInputRate::VERTEX);
    let attributes = [
        vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(0)
            .format(vk::Format::R32G32_SFLOAT)
            .offset(0),
        vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(1)
            .format(vk::Format::R32G32_SFLOAT)
            .offset(8),
        vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(2)
            .format(vk::Format::R8G8B8A8_UNORM)
            .offset(16),
    ];
    let bindings = [binding];
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(&bindings)
        .vertex_attribute_descriptions(&attributes);

    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);

    let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .line_width(1.0)
        .cull_mode(vk::CullModeFlags::NONE);

    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);

    // premultiplied-alpha blending (egui vertex colors are premultiplied)
    let blend_attachment = vk::PipelineColorBlendAttachmentState::default()
        .blend_enable(true)
        .src_color_blend_factor(vk::BlendFactor::ONE)
        .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        .color_blend_op(vk::BlendOp::ADD)
        .src_alpha_blend_factor(vk::BlendFactor::ONE)
        .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        .alpha_blend_op(vk::BlendOp::ADD)
        .color_write_mask(
            vk::ColorComponentFlags::R
                | vk::ColorComponentFlags::G
                | vk::ColorComponentFlags::B
                | vk::ColorComponentFlags::A,
        );
    let blend_attachments = [blend_attachment];
    let color_blend =
        vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attachments);

    let dynamic_state = vk::PipelineDynamicStateCreateInfo::default()
        .dynamic_states(&[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR]);

    let descriptor_layouts = [descriptor_layout];
    let layout_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(&descriptor_layouts)
        .push_constant_ranges(&[vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::VERTEX,
            offset: 0,
            size: 8,
        }]);
    let pipeline_layout = unsafe { device.device.create_pipeline_layout(&layout_info, None)? };

    let create_info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterizer)
        .multisample_state(&multisample)
        .color_blend_state(&color_blend)
        .dynamic_state(&dynamic_state)
        .layout(pipeline_layout)
        .render_pass(render_pass)
        .subpass(0);

    let pipelines = unsafe {
        device
            .device
            .create_graphics_pipelines(vk::PipelineCache::null(), &[create_info], None)
    }
    .map_err(|(_, error)| error)?;

    unsafe {
        device.device.destroy_shader_module(vertex_module, None);
        device.device.destroy_shader_module(fragment_module, None);
    }

    Ok((pipelines[0], pipeline_layout))
}
