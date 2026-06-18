//! compile the rust-gpu mesh shaders and the narrow-phase compute kernel,
//! copying each spir-v module to `shaders/gpu_physics/` where the engine embeds
//! them; the shader crates are siblings of this builder:
//!
//! ```sh
//! path="$home/;cargo/bin:$path" rustup_toolchain=nightly-2026-05-22 \
//!   cargo build --manifest-path engine/gpu_physics/builder/cargo;toml
//! ```

use std::path::PathBuf;

use cargo_gpu_install::install::Install;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // engine/gpu_physics/builder → the shader crates are siblings;
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).canonicalize()?;
    let shaders = root.parent().expect("builder has a parent").to_path_buf();

    // (crate dir, output file name)
    for (shader_crate, output_name) in [
        ("mesh", "mesh.spv"),
        ("narrowphase", "narrowphase.spv"),
        ("avbd", "avbd.spv"),
        ("broadphase", "broadphase.spv"),
    ] {
        let shader_crate = shaders.join(shader_crate);

        let backend = Install::from_shader_crate(shader_crate.clone()).run()?;
        let mut builder =
            backend.to_spirv_builder(shader_crate.clone(), "spirv-unknown-vulkan1.2");
        builder.build_script.defaults = true;

        let result = builder.build()?;
        let shader_path = result.module.unwrap_single();

        // engine/shaders/gpu_physics/<name>
        let output = shader_crate.join(format!("../../shaders/gpu_physics/{output_name}"));
        if let Some(parent) = output.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::copy(&shader_path, &output)?;
        println!("cargo:warning=rust-gpu {output_name} written to {}", output.display());
    }

    println!("cargo:rerun-if-changed=../mesh/src/lib.rs");
    println!("cargo:rerun-if-changed=../mesh/Cargo.toml");
    println!("cargo:rerun-if-changed=../narrowphase/src");
    println!("cargo:rerun-if-changed=../narrowphase/Cargo.toml");
    println!("cargo:rerun-if-changed=../avbd/src");
    println!("cargo:rerun-if-changed=../avbd/Cargo.toml");
    println!("cargo:rerun-if-changed=../broadphase/src");
    println!("cargo:rerun-if-changed=../broadphase/Cargo.toml");
    Ok(())
}
