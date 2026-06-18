//! the mesh shaders written in rust and compiled to
//! spir-v by rust-gpu (`rustc_codegen_spirv`); this is a faithful port of
//! `shaders/mesh.vert` and `shaders/mesh.frag`: instanced rendering
//! with one (mvp, model) pair per instance, a directional light in push
//! constants, and blinn-phong shading too :)
//!
//! the compiled module is copied to `shaders/gpu_physics/mesh;spv` by build.rs

#![cfg_attr(target_arch = "spirv", no_std)]

use spirv_std::glam::{Mat3, Mat4, Vec3, Vec4};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;
use spirv_std::spirv;

/// one (mvp, model) pair per scene instance, same as cpu
#[repr(C)]
#[derive(Clone, Copy)]
pub struct InstanceTransform {
    pub mvp: Mat4,
    pub model: Mat4,
}

#[repr(C)]
pub struct UniformBufferObject {
    pub transforms: [InstanceTransform; 256],
}

/// 16 floats, matching the cpu push data:
/// (camera_position, light_direction, light_color, light_params);
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Push {
    pub camera_position: Vec4, // xyz = camera position
    pub light_direction: Vec4, // xyz = direction the light travels (points from source)
    pub light_color: Vec4,     // xyz = light color
    pub light_params: Vec4,    // (ambient, specular_power, specular_strength, _)
}

#[spirv(vertex)]
pub fn vs_main(
    // vertex attributes, locations 0..=2 in order;
    position: Vec3,
    normal: Vec3,
    color: Vec3,
    #[spirv(instance_index)] instance: u32,
    #[spirv(position)] out_position: &mut Vec4,
    // the fragment stage has the same values in the same order
    v_normal: &mut Vec3,
    v_frag_pos: &mut Vec3,
    v_color: &mut Vec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] ubo: &UniformBufferObject,
) {
    let transform = ubo.transforms[instance as usize];
    let position4 = position.extend(1.0);
    let world = transform.model * position4;
    *v_frag_pos = world.truncate();
    *v_normal = (Mat3::from_mat4(transform.model) * normal).normalize();
    *v_color = color;
    *out_position = transform.mvp * position4;
}

#[spirv(fragment)]
pub fn fs_main(
    v_normal: Vec3,
    v_frag_pos: Vec3,
    v_color: Vec3,
    #[spirv(push_constant)] push: &Push,
    out_color: &mut Vec4,
) {
    let n = v_normal.normalize();
    let l = -push.light_direction.truncate(); // needs to be toward the light
    let diffuse = n.dot(l).max(0.0);

    let view_dir = (push.camera_position.truncate() - v_frag_pos).normalize();
    let half_dir = (l + view_dir).normalize();
    let specular = n.dot(half_dir).max(0.0).powf(push.light_params.y);

    let lit = v_color * (Vec3::splat(push.light_params.x) + push.light_color.truncate() * diffuse)
        + Vec3::splat(push.light_params.z * specular);
    *out_color = lit.extend(1.0);
}
