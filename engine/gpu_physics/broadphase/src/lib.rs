//! gpu broad phase: an aabb sweep over the bodies
//!
//! three kernels, dispatched in order in one command buffer:
//!   1; `broad_aabb`: one thread per body ; compute the support-based aabb;
//!   2; `broad_sweep_dense`: one thread per (i, j) pair ; test the aabb
//!      overlap, appending overlapping pairs to the pair buffer via an
//!      atomic counter;
//!
//! the dense O(n^2) sweep replaces a morton sort + scan: on the igpu the
//! workgroup barriers of a shared-memory bitonic sort unforutnately cost ~5 ms
//! per dispatch, dwarfing the sweep's work at demo scale

#![cfg_attr(target_arch = "spirv", no_std)]

use spirv_std::glam::{Quat, UVec3, Vec3, Vec4};
use spirv_std::spirv;

pub const SPHERE: u32 = 0;
pub const CUBE: u32 = 1;
pub const TETRAHEDRON: u32 = 2;

/// c.f. `engine/src/render/compute;rs::gpushape`
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ShapeData {
    pub tag: u32,
    pub _pad: [u32; 3],
    pub corners: [Vec4; 4],
}

/// c.f. `engine/src/render/compute;rs::gpubody`
#[repr(C)]
#[derive(Clone, Copy)]
pub struct GpuBody {
    pub position: Vec4,
    pub orientation: Vec4,
    pub shape: ShapeData,
}

/// a candidate pair, c.f. narrow phase's `pair`;
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Pair {
    pub a: u32,
    pub b: u32,
}

/// pair-count buffer: `[groups, pair_count, pad, pad]`; the groups slot is
/// reserved for an indirect dispatch when the sweep is fused with the
/// narrow phase

/// shared push constants with the narrow phase; each kernel reads its own
/// fields (16 bytes)
#[repr(C)]
pub struct Push {
    pub pair_count: u32,
    pub body_count: u32,
    pub max_pairs: u32,
    pub sort_len: u32,
}

impl GpuBody {
    pub fn pos(&self) -> Vec3 {
        self.position.truncate()
    }

    pub fn quat(&self) -> Quat {
        Quat::from_xyzw(
            self.orientation.x,
            self.orientation.y,
            self.orientation.z,
            self.orientation.w,
        )
    }
}

/// farthest local-space point of the shape along `direction` (same
/// convention as the narrow phase's support);
pub fn support(shape: &ShapeData, direction: Vec3) -> Vec3 {
    match shape.tag {
        SPHERE => direction.normalize_or_zero() * shape.corners[0].x,
        CUBE => Vec3::new(
            if direction.x >= 0.0 {
                shape.corners[0].x
            } else {
                -shape.corners[0].x
            },
            if direction.y >= 0.0 {
                shape.corners[0].y
            } else {
                -shape.corners[0].y
            },
            if direction.z >= 0.0 {
                shape.corners[0].z
            } else {
                -shape.corners[0].z
            },
        ),
        TETRAHEDRON => {
            let mut best = shape.corners[0].truncate();
            let mut best_dot = best.dot(direction);
            for i in 1..4 {
                let corner = shape.corners[i as usize].truncate();
                let dot = corner.dot(direction);
                if dot > best_dot {
                    best_dot = dot;
                    best = corner;
                }
            }
            best
        }
        _ => Vec3::ZERO,
    }
}

fn world_support(body: &GpuBody, direction: Vec3) -> Vec3 {
    let local = body.quat().inverse() * direction;
    body.quat() * support(&body.shape, local) + body.pos()
}

/// support-based aabb: exact for all three shapes
/// (a convex shape's extremes along the world axes)
fn aabb(body: &GpuBody) -> (Vec3, Vec3) {
    let mut min = body.pos();
    let mut max = body.pos();
    let sx = world_support(body, Vec3::X);
    let snx = world_support(body, -Vec3::X);
    let sy = world_support(body, Vec3::Y);
    let sny = world_support(body, -Vec3::Y);
    let sz = world_support(body, Vec3::Z);
    let snz = world_support(body, -Vec3::Z);
    if sx.x > max.x {
        max.x = sx.x;
    }
    if snx.x < min.x {
        min.x = snx.x;
    }
    if sy.y > max.y {
        max.y = sy.y;
    }
    if sny.y < min.y {
        min.y = sny.y;
    }
    if sz.z > max.z {
        max.z = sz.z;
    }
    if snz.z < min.z {
        min.z = snz.z;
    }
    (min, max)
}

/// kernel number 1: per-body support-based aabb
#[spirv(compute(threads(64)))]
pub fn broad_aabb(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] bodies: &[GpuBody],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] aabbs: &mut [Vec4],
    #[spirv(push_constant)] pc: &Push,
) {
    let i = id.x;
    if i >= pc.body_count {
        return;
    }
    let body = bodies[i as usize];
    let (min, max) = aabb(&body);
    aabbs[(2 * i) as usize] = min.extend(0.0);
    aabbs[(2 * i + 1) as usize] = max.extend(0.0);
}

/// kernel number 2: dense sweep ; one thread per (i, j) body pair, aabb test,
/// atomically appending overlapping pairs (i < j);
#[spirv(compute(threads(64)))]
pub fn broad_sweep_dense(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] pairs: &mut [Pair],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] aabbs: &[Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] pair_count: &mut [u32],
    #[spirv(push_constant)] pc: &Push,
) {
    let n = pc.body_count;
    let t = id.x;
    if t >= n * n {
        return;
    }
    let i = t / n;
    let j = t % n;
    if j <= i {
        return;
    }
    let min_a = aabbs[(2 * i) as usize].truncate();
    let max_a = aabbs[(2 * i + 1) as usize].truncate();
    let min_b = aabbs[(2 * j) as usize].truncate();
    let max_b = aabbs[(2 * j + 1) as usize].truncate();
    if min_a.x <= max_b.x
        && min_b.x <= max_a.x
        && min_a.y <= max_b.y
        && min_b.y <= max_a.y
        && min_a.z <= max_b.z
        && min_b.z <= max_a.z
    {
        let slot = unsafe {
            spirv_std::arch::atomic_i_add::<
                u32,
                { spirv_std::memory::Scope::Workgroup as u32 },
                {
                    spirv_std::memory::Semantics::ACQUIRE_RELEASE.bits()
                        | spirv_std::memory::Semantics::UNIFORM_MEMORY.bits()
                },
            >(&mut pair_count[0], 1)
        };
        if (slot as usize) < pc.max_pairs as usize {
            pairs[slot as usize] = Pair { a: i, b: j };
        }
    }
}
