//! gpu avbd solver: the block-coordinate-descent sweeps of
//! `engine/src/physics/gpu.rs` as rust-gpu compute kernels;
//! the warmstart carry-over and the csr contact indexing are cpu for performance,
//! the expensive 6-dof block solves and dual multiplier updates run here;
//!
//! positions/orientations alternate between two buffers (wiht the jacobian): iteration
//! k reads the snapshot from buffer `1-k%2` and writes the new state to
//! `k%2`, so every body's block solve sees the previous iteration's state
//! for the other bodies; same as cpu arrs

#![cfg_attr(target_arch = "spirv", no_std)]

use spirv_std::glam::{Mat3, Quat, UVec3, Vec2, Vec3, Vec4};
use spirv_std::spirv;

/// per-body dynamics state (static fields; positions live in the
/// alternating pos/rot buffers); 16-byte aligned;
#[repr(C)]
#[derive(Clone, Copy)]
pub struct GpuAvbdBodyState {
    pub vel: Vec4,
    pub ang: Vec4,
    pub prev_vel: Vec4,
    pub inv_moment: Vec4,
    pub inv_mass: f32,
    pub friction: f32,
    pub _pad: [f32; 2],
}

/// one contact, matching `physics::avbdcontact`; `b == u32::max`
/// is the container; 16-byte aligned;
#[repr(C)]
#[derive(Clone, Copy)]
pub struct GpuAvbdContact {
    pub a: u32,
    pub b: u32,
    pub friction: f32,
    pub _pad: f32, // this is lowkey weird...
    pub normal: Vec4,
    pub tangent1: Vec4,
    pub tangent2: Vec4,
    pub r_a: Vec4,
    pub r_b: Vec4,
    pub c0: Vec4,
    pub penalty: Vec4,
    pub lambda: Vec4,
}

/// constraint record; kind selects spring, rod, ball-and-socket or rope
#[repr(C)]
#[derive(Clone, Copy)]
pub struct GpuAvbdConstraint {
    pub a: u32,
    pub b: u32,
    pub kind: u32,
    pub _pad: u32,
    pub anchor_a: Vec4,
    pub anchor_b: Vec4,
}

/// push constants; layout must match the engine's `avbdpush`
#[repr(C)]
pub struct Push {
    pub dt: f32,
    pub alpha: f32,
    pub beta_lin: f32,
    pub penalty_max: f32,
    pub gravity: Vec4,
    pub body_count: u32,
    pub contact_count: u32,
    pub constraint_count: u32,
    pub parity: u32,
}

pub const CONTAINER: u32 = u32::MAX;
pub const CONSTRAINT_SPRING: u32 = 0;
pub const CONSTRAINT_ROD: u32 = 1;
pub const CONSTRAINT_BALL_SOCKET: u32 = 2;
pub const CONSTRAINT_ROPE: u32 = 3;

/// scratch layout helpers: initial/inertial buffers hold `[lin; n]` then
/// `[ang; n]`, so the angular array of body `i` is at `n + i`
fn lin_index(i: u32, n: u32) -> usize {
    i as usize
}

fn ang_index(i: u32, n: u32) -> usize {
    (n + i) as usize
}

fn quat_step(q: Quat, world_delta: Vec3) -> Quat {
    let p = Quat::from_xyzw(world_delta.x, world_delta.y, world_delta.z, 0.0);
    (q + p * q * 0.5).normalize()
}

fn ang_diff(to: Quat, from: Quat) -> Vec3 {
    let rel = to * from.inverse();
    2.0 * rel.xyz()
}

fn outer(a: Vec3, b: Vec3) -> Mat3 {
    Mat3::from_cols(a * b.x, a * b.y, a * b.z)
}

/// solve the 6x6 spd block via ldl^t (port of `physics::avbd::solve6`)
/// sorry for the messy, really verbose code :(
fn solve6(lin: Mat3, ang: Mat3, cross: Mat3, b_lin: Vec3, b_ang: Vec3) -> (Vec3, Vec3) {
    let (a11, a21, a31, a22, a32, a33) = (
        lin.col(0).x,
        lin.col(0).y,
        lin.col(0).z,
        lin.col(1).y,
        lin.col(1).z,
        lin.col(2).z,
    );
    let (a41, a51, a61, a42, a52, a62, a43, a53, a63) = (
        cross.col(0).x,
        cross.col(0).y,
        cross.col(0).z,
        cross.col(1).x,
        cross.col(1).y,
        cross.col(1).z,
        cross.col(2).x,
        cross.col(2).y,
        cross.col(2).z,
    );
    let (a44, a54, a64, a55, a65, a66) = (
        ang.col(0).x,
        ang.col(0).y,
        ang.col(0).z,
        ang.col(1).y,
        ang.col(1).z,
        ang.col(2).z,
    );

    let l21 = a21 / a11;
    let l31 = a31 / a11;
    let l41 = a41 / a11;
    let l51 = a51 / a11;
    let l61 = a61 / a11;
    let d1 = a11;
    let d2 = a22 - l21 * l21 * d1;
    let l32 = (a32 - l21 * l31 * d1) / d2;
    let l42 = (a42 - l21 * l41 * d1) / d2;
    let l52 = (a52 - l21 * l51 * d1) / d2;
    let l62 = (a62 - l21 * l61 * d1) / d2;
    let d3 = a33 - (l31 * l31 * d1 + l32 * l32 * d2);
    let l43 = (a43 - l31 * l41 * d1 - l32 * l42 * d2) / d3;
    let l53 = (a53 - l31 * l51 * d1 - l32 * l52 * d2) / d3;
    let l63 = (a63 - l31 * l61 * d1 - l32 * l62 * d2) / d3;
    let d4 = a44 - (l41 * l41 * d1 + l42 * l42 * d2 + l43 * l43 * d3);
    let l54 = (a54 - l41 * l51 * d1 - l42 * l52 * d2 - l43 * l53 * d3) / d4;
    let l64 = (a64 - l41 * l61 * d1 - l42 * l62 * d2 - l43 * l63 * d3) / d4;
    let d5 = a55 - (l51 * l51 * d1 + l52 * l52 * d2 + l53 * l53 * d3 + l54 * l54 * d4);
    let l65 = (a65 - l51 * l61 * d1 - l52 * l62 * d2 - l53 * l63 * d3 - l54 * l64 * d4) / d5;
    let d6 =
        a66 - (l61 * l61 * d1 + l62 * l62 * d2 + l63 * l63 * d3 + l64 * l64 * d4 + l65 * l65 * d5);

    let y1 = b_lin.x;
    let y2 = b_lin.y - l21 * y1;
    let y3 = b_lin.z - l31 * y1 - l32 * y2;
    let y4 = b_ang.x - l41 * y1 - l42 * y2 - l43 * y3;
    let y5 = b_ang.y - l51 * y1 - l52 * y2 - l53 * y3 - l54 * y4;
    let y6 = b_ang.z - l61 * y1 - l62 * y2 - l63 * y3 - l64 * y4 - l65 * y5;

    let z1 = y1 / d1;
    let z2 = y2 / d2;
    let z3 = y3 / d3;
    let z4 = y4 / d4;
    let z5 = y5 / d5;
    let z6 = y6 / d6;

    let x6 = z6;
    let x5 = z5 - l65 * x6;
    let x4 = z4 - l54 * x5 - l64 * x6;
    let x3 = z3 - l43 * x4 - l53 * x5 - l63 * x6;
    let x2 = z2 - l32 * x3 - l42 * x4 - l52 * x5 - l62 * x6;
    let x1 = z1 - l21 * x2 - l31 * x3 - l41 * x4 - l51 * x5 - l61 * x6;
    (Vec3::new(x1, x2, x3), Vec3::new(x4, x5, x6))
}

fn add_constraint_row(
    lhs_lin: &mut Mat3,
    lhs_ang: &mut Mat3,
    lhs_cross: &mut Mat3,
    rhs_lin: &mut Vec3,
    rhs_ang: &mut Vec3,
    j_lin: Vec3,
    j_ang: Vec3,
    force: f32,
    stiffness: f32,
) {
    *lhs_lin += outer(j_lin, j_lin) * stiffness;
    *lhs_ang += outer(j_ang, j_ang) * stiffness;
    *lhs_cross += outer(j_ang, j_lin) * stiffness;
    *rhs_lin += j_lin * force;
    *rhs_ang += j_ang * force;
}

/// primal init: record the step-start state, the inertial (unconstrained!)
/// positions, and apply the adaptive gravity warmstart
#[spirv(compute(threads(64)))]
pub fn avbd_init(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] state: &[GpuAvbdBodyState],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] initial: &mut [Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] inertial: &mut [Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] pos0: &mut [Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] rot0: &mut [Vec4],
    #[spirv(push_constant)] pc: &Push,
) {
    let i = id.x;
    if i >= pc.body_count {
        return;
    }
    let n = pc.body_count;
    let st = state[i as usize];
    let pos = pos0[i as usize].truncate();
    let rot = Quat::from_xyzw(
        rot0[i as usize].x,
        rot0[i as usize].y,
        rot0[i as usize].z,
        rot0[i as usize].w,
    );

    initial[lin_index(i, n)] = pos.extend(0.0);
    initial[ang_index(i, n)] = rot0[i as usize];
    let mut inertial_pos = pos + st.vel.truncate() * pc.dt;
    if st.inv_mass > 0.0 {
        inertial_pos += pc.gravity.truncate() * (pc.dt * pc.dt);
    }
    inertial[lin_index(i, n)] = inertial_pos.extend(0.0);
    inertial[ang_index(i, n)] = quat_step(rot, st.ang.truncate() * pc.dt).to_array().into();

    // adaptive gravity warmstart (see the cpu solver);
    let (warm_pos, warm_rot) = if st.inv_mass > 0.0 {
        let accel = (st.vel.truncate() - st.prev_vel.truncate()) / pc.dt;
        let gravity = pc.gravity.truncate();
        let gravity_mag = gravity.length();
        let weight = if gravity_mag > 0.0 {
            (accel.dot(gravity / gravity_mag) / gravity_mag).clamp(0.0, 1.0)
        } else {
            0.0
        };
        (
            pos + st.vel.truncate() * pc.dt + gravity * (weight * pc.dt * pc.dt),
            quat_step(rot, st.ang.truncate() * pc.dt),
        )
    } else {
        (pos, rot)
    };
    pos0[i as usize] = warm_pos.extend(0.0);
    rot0[i as usize] = warm_rot.to_array().into();
}

/// primal sweep: one independent 6-dof block solve per body (jacobi; the
/// snapshot lives in the buffer opposite the current parity);
#[spirv(compute(threads(64)))]
pub fn avbd_primal(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] state: &[GpuAvbdBodyState],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] contacts: &[GpuAvbdContact],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 10)] constraints: &[GpuAvbdConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] offsets: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] indices: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] initial: &[Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] inertial: &[Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] pos0: &mut [Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] rot0: &mut [Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 8)] pos1: &mut [Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 9)] rot1: &mut [Vec4],
    #[spirv(push_constant)] pc: &Push,
) {
    let i = id.x;
    if i >= pc.body_count {
        return;
    }
    let n = pc.body_count;
    let st = state[i as usize];
    if st.inv_mass == 0.0 {
        return;
    }
    let parity = pc.parity;
    // snapshot: the previous iteration's state;
    let (snap_pos, snap_rot_v) = if parity == 0 {
        (pos0[i as usize].truncate(), rot0[i as usize])
    } else {
        (pos1[i as usize].truncate(), rot1[i as usize])
    };
    let snap_rot = Quat::from_xyzw(snap_rot_v.x, snap_rot_v.y, snap_rot_v.z, snap_rot_v.w);

    let dt = pc.dt;
    let mass = 1.0 / st.inv_mass;
    let moment = Vec3::ONE / st.inv_moment.truncate();
    let m_dt2 = Vec3::splat(mass) / (dt * dt);
    let i_dt2 = moment / (dt * dt);

    let mut lhs_lin = Mat3::from_diagonal(m_dt2);
    let mut lhs_ang = Mat3::from_diagonal(i_dt2);
    let mut lhs_cross = Mat3::ZERO;
    let mut rhs_lin = m_dt2 * (snap_pos - inertial[lin_index(i, n)].truncate());
    let mut rhs_ang = i_dt2
        * ang_diff(
            snap_rot,
            Quat::from_xyzw(
                inertial[ang_index(i, n)].x,
                inertial[ang_index(i, n)].y,
                inertial[ang_index(i, n)].z,
                inertial[ang_index(i, n)].w,
            ),
        );

    let initial_ang = Quat::from_xyzw(
        initial[ang_index(i, n)].x,
        initial[ang_index(i, n)].y,
        initial[ang_index(i, n)].z,
        initial[ang_index(i, n)].w,
    );
    let initial_lin = initial[lin_index(i, n)].truncate();

    let start = offsets[i as usize] as usize;
    let end = offsets[(i + 1) as usize] as usize;
    let mut j = start;
    while j < end {
        let contact = contacts[indices[j] as usize];
        let is_a = contact.a == i;
        let other = if is_a { contact.b } else { contact.a };
        let (n0, t1, t2) = (
            contact.normal.truncate(),
            contact.tangent1.truncate(),
            contact.tangent2.truncate(),
        );
        // unrolled basis rows (rustgpu sucks here for some reason???)
        let (r0, r1, r2) = if is_a { (n0, t1, t2) } else { (-n0, -t1, -t2) };
        let r_world = if is_a {
            // the body itself is a: its own snapshot orientation
            snap_rot * contact.r_a.truncate()
        } else {
            // the body itself is b: its own snapshot orientation (the cpu
            // reads `snapshot_ang[contact;b]`, i;e; this body)
            snap_rot * contact.r_b.truncate()
        };

        let (dq_self_lin, dq_self_ang) = (snap_pos - initial_lin, ang_diff(snap_rot, initial_ang));
        // ported verbatim from the cpu: the other side's displacement from
        // the snapshot, and its lever arm; the cpu always uses `contact;r_b`
        // for `r_other` (a faithful quirk)
        let (dq_other_lin, dq_other_ang, r_other) = if other == CONTAINER {
            (Vec3::ZERO, Vec3::ZERO, Vec3::ZERO)
        } else {
            let (op, or) = if parity == 0 {
                (pos0[other as usize].truncate(), rot0[other as usize])
            } else {
                (pos1[other as usize].truncate(), rot1[other as usize])
            };
            (
                op - initial_lin_of(other, n, initial),
                ang_diff(
                    Quat::from_xyzw(or.x, or.y, or.z, or.w),
                    initial_ang_of(other, n, initial),
                ),
                Quat::from_xyzw(or.x, or.y, or.z, or.w) * contact.r_b.truncate(),
            )
        };

        // linearised constraint rows: c = c0*(1-alpha) + j.dq,also unrolled;
        let j0 = r_world.cross(r0);
        let j1 = r_world.cross(r1);
        let j2 = r_world.cross(r2);
        let k = 1.0 - pc.alpha;
        let mut c0 = contact.c0.x * k;
        let mut c1 = contact.c0.y * k;
        let mut c2 = contact.c0.z * k;
        c0 += r0.dot(dq_self_lin) + j0.dot(dq_self_ang)
            - r0.dot(dq_other_lin)
            - r_other.cross(r0).dot(dq_other_ang);
        c1 += r1.dot(dq_self_lin) + j1.dot(dq_self_ang)
            - r1.dot(dq_other_lin)
            - r_other.cross(r1).dot(dq_other_ang);
        c2 += r2.dot(dq_self_lin) + j2.dot(dq_self_ang)
            - r2.dot(dq_other_lin)
            - r_other.cross(r2).dot(dq_other_ang);

        let f0 = (contact.penalty.x * c0 + contact.lambda.x).min(0.0); // repulsive only
        let f1_raw = contact.penalty.y * c1 + contact.lambda.y;
        let f2_raw = contact.penalty.z * c2 + contact.lambda.z;
        let bounds = f0.abs() * contact.friction;
        let tang_mag = Vec2::new(f1_raw, f2_raw).length();
        let (f1, f2) = if tang_mag > bounds && tang_mag > 0.0 {
            let s = bounds / tang_mag;
            (f1_raw * s, f2_raw * s)
        } else {
            (f1_raw, f2_raw)
        };

        // stamp j^t k j and j^t f into the block
        let (k0, k1, k2) = (contact.penalty.x, contact.penalty.y, contact.penalty.z);
        lhs_lin += outer(r0, r0) * k0;
        lhs_lin += outer(r1, r1) * k1;
        lhs_lin += outer(r2, r2) * k2;
        lhs_ang += outer(j0, j0) * k0;
        lhs_ang += outer(j1, j1) * k1;
        lhs_ang += outer(j2, j2) * k2;
        lhs_cross += outer(j0, r0) * k0;
        lhs_cross += outer(j1, r1) * k1;
        lhs_cross += outer(j2, r2) * k2;
        rhs_lin += r0 * f0;
        rhs_lin += r1 * f1;
        rhs_lin += r2 * f2;
        rhs_ang += j0 * f0;
        rhs_ang += j1 * f1;
        rhs_ang += j2 * f2;
        j += 1;
    }

    // assemble conservative constraints;
    // ropes skip their row while slack
    let mut constraint_index = 0;
    while constraint_index < pc.constraint_count {
        let constraint = constraints[constraint_index as usize];
        if constraint.a == i || constraint.b == i {
            let is_a = constraint.a == i;
            let anchor_a = constraint.anchor_a.truncate();
            let anchor_b = constraint.anchor_b.truncate();
            let body_a_pos = if is_a {
                snap_pos
            } else {
                let index = constraint.a as usize;
                if parity == 0 {
                    pos0[index].truncate()
                } else {
                    pos1[index].truncate()
                }
            };
            let body_b_pos = if !is_a {
                snap_pos
            } else {
                let index = constraint.b as usize;
                if parity == 0 {
                    pos0[index].truncate()
                } else {
                    pos1[index].truncate()
                }
            };
            let body_a_rot = if is_a {
                snap_rot
            } else {
                let value = if parity == 0 {
                    rot0[constraint.a as usize]
                } else {
                    rot1[constraint.a as usize]
                };
                Quat::from_xyzw(value.x, value.y, value.z, value.w)
            };
            let body_b_rot = if !is_a {
                snap_rot
            } else {
                let value = if parity == 0 {
                    rot0[constraint.b as usize]
                } else {
                    rot1[constraint.b as usize]
                };
                Quat::from_xyzw(value.x, value.y, value.z, value.w)
            };
            let p_a = body_a_pos + body_a_rot * anchor_a;
            let p_b = body_b_pos + body_b_rot * anchor_b;
            let self_anchor = if is_a { anchor_a } else { anchor_b };
            let sign = if is_a { 1.0 } else { -1.0 };
            let delta = p_a - p_b;
            let r_self = snap_rot * self_anchor;
            let stiffness = constraint.anchor_b.w;

            if constraint.kind == CONSTRAINT_BALL_SOCKET {
                let jx = Vec3::X * sign;
                let jy = Vec3::Y * sign;
                let jz = Vec3::Z * sign;
                add_constraint_row(
                    &mut lhs_lin,
                    &mut lhs_ang,
                    &mut lhs_cross,
                    &mut rhs_lin,
                    &mut rhs_ang,
                    jx,
                    r_self.cross(jx),
                    delta.x * stiffness,
                    stiffness,
                );
                add_constraint_row(
                    &mut lhs_lin,
                    &mut lhs_ang,
                    &mut lhs_cross,
                    &mut rhs_lin,
                    &mut rhs_ang,
                    jy,
                    r_self.cross(jy),
                    delta.y * stiffness,
                    stiffness,
                );
                add_constraint_row(
                    &mut lhs_lin,
                    &mut lhs_ang,
                    &mut lhs_cross,
                    &mut rhs_lin,
                    &mut rhs_ang,
                    jz,
                    r_self.cross(jz),
                    delta.z * stiffness,
                    stiffness,
                );
            } else {
                let length = delta.length();
                let active = length > 1.0e-6
                    && (constraint.kind != CONSTRAINT_ROPE || length > constraint.anchor_a.w);
                if active {
                    let normal = delta / length;
                    let j_lin = normal * sign;
                    let j_ang = r_self.cross(j_lin);
                    let target = constraint.anchor_a.w;
                    let force = stiffness * (length - target);
                    add_constraint_row(
                        &mut lhs_lin,
                        &mut lhs_ang,
                        &mut lhs_cross,
                        &mut rhs_lin,
                        &mut rhs_ang,
                        j_lin,
                        j_ang,
                        force,
                        stiffness,
                    );
                }
            }
        }
        constraint_index += 1;
    }

    let (dx_lin, dx_ang) = solve6(lhs_lin, lhs_ang, lhs_cross, -rhs_lin, -rhs_ang);
    let new_pos = snap_pos + dx_lin;
    let new_rot = quat_step(snap_rot, dx_ang);
    if parity == 0 {
        pos1[i as usize] = new_pos.extend(0.0);
        rot1[i as usize] = new_rot.to_array().into();
    } else {
        pos0[i as usize] = new_pos.extend(0.0);
        rot0[i as usize] = new_rot.to_array().into();
    }
}

fn initial_lin_of(i: u32, n: u32, initial: &[Vec4]) -> Vec3 {
    initial[lin_index(i, n)].truncate()
}

fn initial_ang_of(i: u32, n: u32, initial: &[Vec4]) -> Quat {
    let v = initial[ang_index(i, n)];
    Quat::from_xyzw(v.x, v.y, v.z, v.w)
}

/// dual sweep: one thread per contact; recompute the error at the current
/// positions, store the clamped force as the new multiplier and ramp the
/// penalty
#[spirv(compute(threads(64)))]
pub fn avbd_dual(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] contacts: &mut [GpuAvbdContact],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] initial: &[Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] pos0: &[Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] rot0: &[Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 8)] pos1: &[Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 9)] rot1: &[Vec4],
    #[spirv(push_constant)] pc: &Push,
) {
    let ci = id.x;
    if ci >= pc.contact_count {
        return;
    }
    let n = pc.body_count;
    let parity = pc.parity;
    let mut contact = contacts[ci as usize];

    let (pos_a, rot_a_v) = if parity == 0 {
        (
            pos1[contact.a as usize].truncate(),
            rot1[contact.a as usize],
        )
    } else {
        (
            pos0[contact.a as usize].truncate(),
            rot0[contact.a as usize],
        )
    };
    let rot_a = Quat::from_xyzw(rot_a_v.x, rot_a_v.y, rot_a_v.z, rot_a_v.w);
    let dq_a_lin = pos_a - initial[lin_index(contact.a, n)].truncate();
    let dq_a_ang = ang_diff(rot_a, initial_ang_of(contact.a, n, initial));

    let (dq_b_lin, dq_b_ang, r_b_world) = if contact.b == CONTAINER {
        (Vec3::ZERO, Vec3::ZERO, Vec3::ZERO)
    } else {
        let (pos_b, rot_b_v) = if parity == 0 {
            (
                pos1[contact.b as usize].truncate(),
                rot1[contact.b as usize],
            )
        } else {
            (
                pos0[contact.b as usize].truncate(),
                rot0[contact.b as usize],
            )
        };
        let rot_b = Quat::from_xyzw(rot_b_v.x, rot_b_v.y, rot_b_v.z, rot_b_v.w);
        (
            pos_b - initial[lin_index(contact.b, n)].truncate(),
            ang_diff(rot_b, initial_ang_of(contact.b, n, initial)),
            rot_b * contact.r_b.truncate(),
        )
    };

    let r_a_world = rot_a * contact.r_a.truncate();
    let (r0, r1, r2) = (
        contact.normal.truncate(),
        contact.tangent1.truncate(),
        contact.tangent2.truncate(),
    );
    let k = 1.0 - pc.alpha;
    let mut c0 = contact.c0.x * k;
    let mut c1 = contact.c0.y * k;
    let mut c2 = contact.c0.z * k;
    c0 += r0.dot(dq_a_lin) + r_a_world.cross(r0).dot(dq_a_ang)
        - r0.dot(dq_b_lin)
        - r_b_world.cross(r0).dot(dq_b_ang);
    c1 += r1.dot(dq_a_lin) + r_a_world.cross(r1).dot(dq_a_ang)
        - r1.dot(dq_b_lin)
        - r_b_world.cross(r1).dot(dq_b_ang);
    c2 += r2.dot(dq_a_lin) + r_a_world.cross(r2).dot(dq_a_ang)
        - r2.dot(dq_b_lin)
        - r_b_world.cross(r2).dot(dq_b_ang);

    let f0 = (contact.penalty.x * c0 + contact.lambda.x).min(0.0);
    let f1_raw = contact.penalty.y * c1 + contact.lambda.y;
    let f2_raw = contact.penalty.z * c2 + contact.lambda.z;
    let bounds = f0.abs() * contact.friction;
    let tang_mag = Vec2::new(f1_raw, f2_raw).length();
    let (f1, f2) = if tang_mag > bounds && tang_mag > 0.0 {
        let s = bounds / tang_mag;
        (f1_raw * s, f2_raw * s)
    } else {
        (f1_raw, f2_raw)
    };
    contact.lambda = Vec4::new(f0, f1, f2, 0.0);

    // ramp the penalty while the force is within its bounds;
    if f0 < 0.0 {
        contact.penalty.x = (contact.penalty.x + pc.beta_lin * c0.abs()).min(pc.penalty_max);
    }
    if tang_mag <= bounds {
        contact.penalty.y = (contact.penalty.y + pc.beta_lin * c1.abs()).min(pc.penalty_max);
        contact.penalty.z = (contact.penalty.z + pc.beta_lin * c2.abs()).min(pc.penalty_max);
    }
    contacts[ci as usize] = contact;
}

/// recover velocities (bdf1): the implicit-euler velocity is the total
/// displacement over the step, divided by dt;
#[spirv(compute(threads(64)))]
pub fn avbd_recover(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] state: &mut [GpuAvbdBodyState],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] initial: &[Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] pos0: &[Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] rot0: &[Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 8)] pos1: &[Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 9)] rot1: &[Vec4],
    #[spirv(push_constant)] pc: &Push,
) {
    let i = id.x;
    if i >= pc.body_count {
        return;
    }
    let n = pc.body_count;
    let parity = pc.parity;
    let st = &mut state[i as usize];
    if st.inv_mass == 0.0 {
        return;
    }
    let (pos, rot_v) = if parity == 0 {
        (pos1[i as usize].truncate(), rot1[i as usize])
    } else {
        (pos0[i as usize].truncate(), rot0[i as usize])
    };
    let rot = Quat::from_xyzw(rot_v.x, rot_v.y, rot_v.z, rot_v.w);
    let dt = pc.dt;
    st.prev_vel = st.vel;
    st.vel = ((pos - initial[lin_index(i, n)].truncate()) / dt).extend(0.0);
    let rel = rot * initial_ang_of(i, n, initial).inverse();
    st.ang = (2.0 * rel.xyz() / dt).extend(0.0);
}
