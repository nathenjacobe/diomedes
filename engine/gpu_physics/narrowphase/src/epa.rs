//! expanding polytope algorithm (epa) :)

use spirv_std::glam::Vec3;

use crate::body::GpuBody;
use crate::gjk::{Simplex, world_support_witness};

const MAX_ITERATIONS: u32 = 64;
const TOLERANCE: f32 = 1e-4;
const MAX_VERTICES: usize = 256;
const MAX_FACES: usize = 512;

/// a penetration: `normal` is the direction to move body `b` by `depth` to
/// separate it from body `a`; the points are the contact witnesses on each
/// body's surface
pub struct Penetration {
    pub normal: Vec3,
    pub depth: f32,
    pub point_a: Vec3,
    pub point_b: Vec3,
}

#[derive(Clone, Copy)]
struct Face {
    a: u32,
    b: u32,
    c: u32,
    normal: Vec3,
    /// signed distance of the face plane from the origin, along `normal`
    distance: f32,
}

struct Polytope {
    vertices: [Vec3; MAX_VERTICES],
    witnesses: [Vec3; MAX_VERTICES],
    faces: [Face; MAX_FACES],
    vertex_count: u32,
    face_count: u32,
}

impl Polytope {
    fn new() -> Self {
        Self {
            vertices: [Vec3::ZERO; MAX_VERTICES],
            witnesses: [Vec3::ZERO; MAX_VERTICES],
            faces: [Face {
                a: 0,
                b: 0,
                c: 0,
                normal: Vec3::ZERO,
                distance: 0.0,
            }; MAX_FACES],
            vertex_count: 0,
            face_count: 0,
        }
    }

    fn add_face(&mut self, a: u32, b: u32, c: u32) {
        if self.face_count as usize >= MAX_FACES {
            return;
        }
        let va = self.vertices[a as usize];
        let normal = (self.vertices[b as usize] - va).cross(self.vertices[c as usize] - va);
        if normal.length_squared() < 1e-12 {
            return; // degenerate (zero-area) face
        }
        let normal = normal.normalize();
        let mut face = Face {
            a,
            b,
            c,
            normal,
            distance: normal.dot(va),
        };
        // the origin is inside the polytope, so outward normals have
        // positive distance; flip the winding when a face points other way
        if face.distance < 0.0 {
            face.a = c;
            face.c = a;
            face.normal = -face.normal;
            face.distance = -face.distance;
        }
        self.faces[self.face_count as usize] = face;
        self.face_count += 1;
    }
}

/// run epa on an intersecting pair, using the final res of gjk simplex
pub fn epa(a: &GpuBody, b: &GpuBody, simplex: &Simplex) -> Option<Penetration> {
    let mut poly = Polytope::new();
    for i in 0..simplex.len {
        poly.vertices[i as usize] = simplex.points[i as usize];
        poly.witnesses[i as usize] = simplex.witnesses[i as usize];
    }
    poly.vertex_count = simplex.len;
    // tetrahedron w/ four faces, normals pointing away from the origin
    poly.add_face(0, 1, 2);
    poly.add_face(0, 3, 1);
    poly.add_face(0, 2, 3);
    poly.add_face(1, 3, 2);

    for _ in 0..MAX_ITERATIONS {
        // the face closest to the origin;
        let mut min_index = 0u32;
        let mut min_distance = poly.faces[0].distance;
        for i in 1..poly.face_count {
            let distance = poly.faces[i as usize].distance;
            if distance < min_distance {
                min_distance = distance;
                min_index = i;
            }
        }
        let min_face = poly.faces[min_index as usize];

        let (support, support_witness) = world_support_witness(a, b, min_face.normal);
        let support_distance = min_face.normal.dot(support);

        // the support point does not extend past this face: the closest
        // point of the minkowski difference lies on it
        if support_distance - min_face.distance < TOLERANCE {
            let face = &poly.faces[min_index as usize];
            return Some(face_contact(face, &poly));
        }

        // expand: add the support point, drop faces it sees, stitch the
        // horizon (edges shared by exactly one visible face)
        let new_index = poly.vertex_count;
        if new_index as usize >= MAX_VERTICES {
            let face = &poly.faces[min_index as usize];
            return Some(face_contact(face, &poly));
        }
        poly.vertices[new_index as usize] = support;
        poly.witnesses[new_index as usize] = support_witness;
        poly.vertex_count += 1;

        // the rust-gpu rejects memset on `[bool; n]` :(
        // use flags instead
        let mut visible = [0u32; MAX_FACES];
        for i in 0..poly.face_count {
            let face = poly.faces[i as usize];
            visible[i as usize] =
                u32::from(face.normal.dot(support - poly.vertices[face.a as usize]) > 0.0);
        }

        let mut horizon: [(u32, u32); MAX_FACES] = [(0, 0); MAX_FACES];
        let mut horizon_count = 0u32;
        for i in 0..poly.face_count {
            if visible[i as usize] == 0 {
                continue;
            }
            let face = poly.faces[i as usize];
            let edges = [(face.a, face.b), (face.b, face.c), (face.c, face.a)];
            for e in 0..3 {
                let edge = edges[e as usize];
                let mut shared = false;
                for j in 0..poly.face_count {
                    if j == i {
                        continue;
                    }
                    if visible[j as usize] != 0 && edge_in_face(edge, &poly.faces[j as usize]) {
                        shared = true;
                        break;
                    }
                }
                if !shared {
                    horizon[horizon_count as usize] = edge;
                    horizon_count += 1;
                }
            }
        }

        // compact the kept faces
        let mut write = 0u32;
        for i in 0..poly.face_count {
            if visible[i as usize] == 0 {
                poly.faces[write as usize] = poly.faces[i as usize];
                write += 1;
            }
        }
        poly.face_count = write;

        for i in 0..horizon_count {
            let (a, b) = horizon[i as usize];
            poly.add_face(new_index, a, b);
        }
    }

    None // did not converge within the iteration budget. womp womp
}

/// contact witnesses for a face: the barycentric combination of the vertex
/// witnesses at the face's closest point to the origin
fn face_contact(face: &Face, poly: &Polytope) -> Penetration {
    let va = poly.vertices[face.a as usize];
    let vb = poly.vertices[face.b as usize];
    let vc = poly.vertices[face.c as usize];
    let closest = face.normal * face.distance;

    // barycenrtic coordinates of `closest` in the face triangle
    let v0 = vb - va;
    let v1 = vc - va;
    let v2 = closest - va;
    let d00 = v0.dot(v0);
    let d01 = v0.dot(v1);
    let d11 = v1.dot(v1);
    let d20 = v2.dot(v0);
    let d21 = v2.dot(v1);
    let denom = d00 * d11 - d01 * d01;
    let (mut w1, mut w2, mut w3) = if denom.abs() > 1e-12 {
        let w2 = (d11 * d20 - d01 * d21) / denom;
        let w3 = (d00 * d21 - d01 * d20) / denom;
        (1.0 - w2 - w3, w2, w3)
    } else {
        (1.0, 0.0, 0.0)
    };
    w1 = w1.clamp(0.0, 1.0);
    w2 = w2.clamp(0.0, 1.0);
    w3 = w3.clamp(0.0, 1.0);
    let sum = w1 + w2 + w3;
    let (w1, w2, w3) = if sum > 1e-12 {
        (w1 / sum, w2 / sum, w3 / sum)
    } else {
        (1.0, 0.0, 0.0)
    };

    let wa = poly.witnesses[face.a as usize];
    let wb = poly.witnesses[face.b as usize];
    let wc = poly.witnesses[face.c as usize];
    let point_a = wa * w1 + wb * w2 + wc * w3;
    // the b witness is the a witness minus the minkowski point
    let minkowski = va * w1 + vb * w2 + vc * w3;
    let point_b = point_a - minkowski;

    Penetration {
        normal: face.normal,
        depth: face.distance,
        point_a,
        point_b,
    }
}

fn edge_in_face(edge: (u32, u32), face: &Face) -> bool {
    let (a, b) = edge;
    (face.a == a && face.b == b)
        || (face.b == a && face.c == b)
        || (face.c == a && face.a == b)
        || (face.a == b && face.b == a)
        || (face.b == b && face.c == a)
        || (face.c == b && face.a == a)
}
