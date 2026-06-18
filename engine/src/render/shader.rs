use std::error::Error;

use ash::Device;
use ash::vk;
use shaderc::{Compiler, ShaderKind};

/// glsl sources, embedded at compile time and compiled to spir-v at startup
pub const VERTEX_SOURCE: &str = include_str!("../../shaders/mesh.vert");
pub const FRAGMENT_SOURCE: &str = include_str!("../../shaders/mesh.frag");

/// egui ui shaders
pub const UI_VERTEX_SOURCE: &str = include_str!("../../shaders/ui.vert");
pub const UI_FRAGMENT_SOURCE: &str = include_str!("../../shaders/ui.frag");

/// rust-gpu compiled mesh shaders, committed from the `gpu_physics`
/// crate (see its build.rs); the engine uses this module by default; set
/// `DIOMEDES_GLSL_SHADERS=1` to force the glsl path instead
pub const MESH_SPIRV: &[u8] = include_bytes!("../../shaders/gpu_physics/mesh.spv");
pub const MESH_VERTEX_ENTRY: &str = "vs_main";
pub const MESH_FRAGMENT_ENTRY: &str = "fs_main";

/// create a shader module from precompiled spir-v bytes
pub fn create_module(
    device: &Device,
    bytes: &[u8],
    name: &str,
) -> Result<vk::ShaderModule, Box<dyn Error>> {
    // the spir-v file is a stream of 32-bit words; decode once
    let words: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    let create_info = vk::ShaderModuleCreateInfo::default().code(&words);
    let module = unsafe { device.create_shader_module(&create_info, None)? };
    log::info!("loaded {name} shader module");
    Ok(module)
}

/// compile glsl to a spir-v shader module on the given device
pub fn compile(
    device: &Device,
    source: &str,
    kind: ShaderKind,
    name: &str,
) -> Result<vk::ShaderModule, Box<dyn Error>> {
    let compiler = Compiler::new()?;
    let artifact = compiler.compile_into_spirv(source, kind, name, "main", None)?;

    let warnings = artifact.get_num_warnings();
    if warnings > 0 {
        log::warn!("{name}: {warnings} shader compilation warnings");
    }

    let create_info = vk::ShaderModuleCreateInfo::default().code(artifact.as_binary());
    let module = unsafe { device.create_shader_module(&create_info, None)? };
    log::info!("compiled {name} shader");
    Ok(module)
}
