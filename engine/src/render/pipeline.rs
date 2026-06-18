use std::error::Error;

use ash::Device;
use ash::vk;

use super::shader;
use super::vertex::Vertex;
use crate::scene::RenderStyle;

/// graphics pipeline for drawing meshes with the `vertex` layout and a
/// dynamic uniform buffer (mvp per mesh) at set 0
/// the viewport and scissor are dynamic, so the pipeline is independent of
/// the swapchain size and survives resize without recreation
pub struct GraphicsPipeline {
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
}

impl GraphicsPipeline {
    pub fn create(
        device: &Device,
        render_pass: vk::RenderPass,
        descriptor_layout: vk::DescriptorSetLayout,
        style: RenderStyle,
    ) -> Result<Self, Box<dyn Error>> {
        // prefer the rust-gpu compiled mesh shaders; fall back to the glsl
        // sources (shaderc) if loading fails or diomedes_glsl_shaders is ser
        let (vertex_module, fragment_module, vertex_entry, fragment_entry) =
            if std::env::var_os("DIOMEDES_GLSL_SHADERS").is_none() {
                match (
                    shader::create_module(device, shader::MESH_SPIRV, "mesh (rust-gpu)"),
                    shader::create_module(device, shader::MESH_SPIRV, "mesh (rust-gpu)"),
                ) {
                    (Ok(vertex), Ok(fragment)) => (
                        vertex,
                        fragment,
                        shader::MESH_VERTEX_ENTRY,
                        shader::MESH_FRAGMENT_ENTRY,
                    ),
                    (Err(error), _) | (_, Err(error)) => {
                        log::warn!("rust-gpu mesh shader failed ({error}); using GLSL");
                        (
                            shader::compile(
                                device,
                                shader::VERTEX_SOURCE,
                                shaderc::ShaderKind::Vertex,
                                "mesh.vert",
                            )?,
                            shader::compile(
                                device,
                                shader::FRAGMENT_SOURCE,
                                shaderc::ShaderKind::Fragment,
                                "mesh.frag",
                            )?,
                            "main",
                            "main",
                        )
                    }
                }
            } else {
                log::info!("DIOMEDES_GLSL_SHADERS set; using GLSL mesh shaders");
                (
                    shader::compile(
                        device,
                        shader::VERTEX_SOURCE,
                        shaderc::ShaderKind::Vertex,
                        "mesh.vert",
                    )?,
                    shader::compile(
                        device,
                        shader::FRAGMENT_SOURCE,
                        shaderc::ShaderKind::Fragment,
                        "mesh.frag",
                    )?,
                    "main",
                    "main",
                )
            };

        let vertex_entry_name = std::ffi::CString::new(vertex_entry)?;
        let fragment_entry_name = std::ffi::CString::new(fragment_entry)?;
        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vertex_module)
                .name(vertex_entry_name.as_c_str()),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(fragment_module)
                .name(fragment_entry_name.as_c_str()),
        ];

        let binding = Vertex::binding_description();
        let attributes = Vertex::attribute_descriptions();
        let bindings = [binding];
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&bindings)
            .vertex_attribute_descriptions(&attributes);

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);

        let (polygon_mode, cull_mode) = match style {
            RenderStyle::Solid => (vk::PolygonMode::FILL, vk::CullModeFlags::BACK),
            RenderStyle::Wireframe => (vk::PolygonMode::LINE, vk::CullModeFlags::NONE),
        };
        let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(polygon_mode)
            .line_width(1.0)
            .cull_mode(cull_mode)
            // meshes are wound ccw from "outside"; front faces carry positive
            // framebuffer area with the projection's y-flip
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE);

        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::LESS);

        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(
                vk::ColorComponentFlags::R
                    | vk::ColorComponentFlags::G
                    | vk::ColorComponentFlags::B
                    | vk::ColorComponentFlags::A,
            )
            .blend_enable(false);

        let color_blend_attachments = [color_blend_attachment];
        let color_blend =
            vk::PipelineColorBlendStateCreateInfo::default().attachments(&color_blend_attachments);

        let dynamic_state = vk::PipelineDynamicStateCreateInfo::default()
            .dynamic_states(&[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR]);

        let descriptor_layouts = [descriptor_layout];
        // lighting block: camera position, light direction/color, params
        let push_ranges = [vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
            offset: 0,
            size: 64,
        }];
        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&descriptor_layouts)
            .push_constant_ranges(&push_ranges);
        let layout = unsafe { device.create_pipeline_layout(&layout_info, None)? };

        let create_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisample)
            .depth_stencil_state(&depth_stencil)
            .color_blend_state(&color_blend)
            .dynamic_state(&dynamic_state)
            .layout(layout)
            .render_pass(render_pass)
            .subpass(0);

        let pipelines = unsafe {
            device.create_graphics_pipelines(vk::PipelineCache::null(), &[create_info], None)
        }
        .map_err(|(_, error)| error)?;
        let pipeline = pipelines[0];

        unsafe {
            device.destroy_shader_module(vertex_module, None);
            device.destroy_shader_module(fragment_module, None);
        }

        log::info!("created graphics pipeline");
        Ok(Self { pipeline, layout })
    }

    pub fn destroy(&mut self, device: &Device) {
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.layout, None);
        }
        log::info!("destroyed graphics pipeline");
    }
}
