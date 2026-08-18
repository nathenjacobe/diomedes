//! gpu narrow phase (gjk + epa) for the diomedes engine
//!
//! buffer layout (descriptor set 0):
//!   binding 0: bodies (storage), binding 1: pairs (storage),
//!   binding 2: contacts out (storage); push constants: pair count;
//! all structs are ofc `#[repr(c)]` and mirror the cpu-side definitions in
//! `engine/src/render/compute.rs`;
//!
//! note: early `return`s from the kernel body crash the intel and llvmpipe
//! drivers (rust-gpu emits an `opswitch` both choke on); all control flow is
//! written as wrapped `if`s instead. looks ugly, but i have no choice :(

#![cfg_attr(target_arch = "spirv", no_std)]

use spirv_std::glam::{UVec3, Vec3, Vec4};
use spirv_std::spirv;

mod body;
mod cuboid;
mod epa;
mod gjk;
mod support;

use body::GpuBody;

// must match `narrowphasecompute::workgroup_size` in the engine; keep at 64:
// localsize 32 mysteriously fails to execute workgroups past the 9th on both
// anv and llvmpipe (the 10th+ workgroups run no threads, regardless of the
// pair-count guard); 64 has no such cutoff up to at least 32 workgroups
pub const WORKGROUP_SIZE: u32 = 64;
pub const CONTACTS_PER_PAIR: usize = 4;

/// one candidate pair (indices into the bodies buffer);
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Pair {
    pub a: u32,
    pub b: u32,
}

/// kernel output slot per pair; `valid` is 1 when a contact was found
/// layout matches the cpu-side `gpucontactout` exactly
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ContactOut {
    pub valid: u32,
    pub a: u32,
    pub b: u32,
    pub _pad: u32,
    pub normal: Vec4,
    pub depth: f32,
    pub _pad2: [f32; 3],
    pub point_a: Vec4,
    pub point_b: Vec4,
}

/// push constants: the number of pairs to process
#[repr(C)]
pub struct Push {
    pub pair_count: u32,
}

/// fixed-size contact slots: the runtime-array form is avoided so the
/// kernel's bounds checks fold against constants (matching the working
/// mesh shader's fixed `transforms[256]`)
#[repr(C)]
pub struct ContactBuffer {
    pub slots: [ContactOut; 65536],
}

fn write_contact(
    slot: &mut ContactOut,
    pair: Pair,
    normal: Vec3,
    depth: f32,
    point_a: Vec3,
    point_b: Vec3,
) {
    slot.valid = 1;
    slot.a = pair.a;
    slot.b = pair.b;
    slot.depth = depth;
    slot.normal = normal.extend(0.0);
    slot.point_a = point_a.extend(0.0);
    slot.point_b = point_b.extend(0.0);
}

/// dispatch one thread per pair; non-intersecting pairs leave `valid = 0`
#[spirv(compute(threads(64)))]
pub fn narrowphase_main(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] bodies: &[GpuBody],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] pairs: &[Pair],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] contacts: &mut ContactBuffer,
    #[spirv(push_constant)] pc: &Push,
) {
    let pair_index = id.x;
    if pair_index < pc.pair_count {
        let pair = pairs[pair_index as usize];
        let a = &bodies[pair.a as usize];
        let b = &bodies[pair.b as usize];

        // OBB SAT avoids the degenerate EPA initialization for cube/cube
        // contacts
        if support::both_cubes(&a.shape, &b.shape) {
            match cuboid::collide_manifold(a, b) {
                Some(manifold) => {
                    let base = pair_index as usize * CONTACTS_PER_PAIR;
                    let mut contact_index = 0u32;
                    while contact_index < manifold.count {
                        let slot = &mut contacts.slots[base + contact_index as usize];
                        write_contact(
                            slot,
                            pair,
                            manifold.normal,
                            manifold.depth,
                            manifold.point_a[contact_index as usize],
                            manifold.point_b[contact_index as usize],
                        );
                        contact_index += 1;
                    }
                }
                None => {}
            }
        } else if support::both_spheres(&a.shape, &b.shape) {
            // same convention as the cpu analytic path: normal from a
            // toward b, witnesses on each surface
            let delta = b.pos() - a.pos();
            let distance = delta.length();
            let radius = a.shape.corners[0].x + b.shape.corners[0].x;
            if distance >= radius || distance <= f32::EPSILON {
                // no contact
            } else {
                let normal = delta / distance;
                let slot = &mut contacts.slots[pair_index as usize * CONTACTS_PER_PAIR];
                write_contact(
                    slot,
                    pair,
                    normal,
                    radius - distance,
                    a.pos() + normal * a.shape.corners[0].x,
                    b.pos() - normal * b.shape.corners[0].x,
                );
            }
        } else {
            let mut simplex = gjk::Simplex::new();
            if gjk::gjk(a, b, &mut simplex) {
                match epa::epa(a, b, &simplex) {
                    Some(pen) => {
                        let slot = &mut contacts.slots[pair_index as usize * CONTACTS_PER_PAIR];
                        write_contact(slot, pair, pen.normal, pen.depth, pen.point_a, pen.point_b);
                    }
                    None => {}
                }
            }
        }
    }
}
