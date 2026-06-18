//! gjk boolean intersection test for two convex bodies
//! TODO: need convex decomposition for concave meshes (statically if possible;
//! last i checked, these decomposition algorithms can be quite slow!)

use spirv_std::glam::Vec3;

use crate::body::GpuBody;
use crate::support::support;

const MAX_ITERATIONS: u32 = 64;

/// the gjk simplex: minkowski support points with the support witness on
/// body a for each, so epa can reconstruct contact points
pub struct Simplex {
    pub points: [Vec3; 4],
    pub witnesses: [Vec3; 4],
    pub len: u32,
}

impl Simplex {
    pub fn new() -> Self {
        Self {
            points: [Vec3::ZERO; 4],
            witnesses: [Vec3::ZERO; 4],
            len: 0,
        }
    }

    fn push_front(&mut self, point: Vec3, witness: Vec3) {
        let mut i = self.len;
        while i > 0 {
            self.points[i as usize] = self.points[i as usize - 1];
            self.witnesses[i as usize] = self.witnesses[i as usize - 1];
            i -= 1;
        }
        self.points[0] = point;
        self.witnesses[0] = witness;
        self.len = (self.len + 1).min(4);
    }

    fn remove(&mut self, index: u32) {
        let mut i = index;
        while i + 1 < self.len {
            self.points[i as usize] = self.points[i as usize + 1];
            self.witnesses[i as usize] = self.witnesses[i as usize + 1];
            i += 1;
        }
        self.len -= 1;
    }

    fn swap(&mut self, i: u32, j: u32) {
        let p = self.points[i as usize];
        self.points[i as usize] = self.points[j as usize];
        self.points[j as usize] = p;
        let w = self.witnesses[i as usize];
        self.witnesses[i as usize] = self.witnesses[j as usize];
        self.witnesses[j as usize] = w;
    }

    fn truncate(&mut self, len: u32) {
        self.len = len;
    }
}

/// support point of the minkowski difference `a - b` in world space, plus
/// the world-space support witness on `a`
pub fn world_support_witness(a: &GpuBody, b: &GpuBody, direction: Vec3) -> (Vec3, Vec3) {
    let a_local = a.quat().inverse() * direction;
    let a_point = a.quat() * support(&a.shape, a_local) + a.pos();
    let b_local = b.quat().inverse() * -direction;
    let b_point = b.quat() * support(&b.shape, b_local) + b.pos();
    (a_point - b_point, a_point)
}

/// run gjk; on intersection, fills `out` with the final simplex
/// (containing the origin) and returns bool accordingly
pub fn gjk(a: &GpuBody, b: &GpuBody, out: &mut Simplex) -> bool {
    // initial search direction: from b toward a;
    let mut direction = a.pos() - b.pos();
    if direction == Vec3::ZERO {
        direction = Vec3::X;
    }

    let mut simplex = Simplex::new();
    let (point, witness) = world_support_witness(a, b, direction);
    simplex.push_front(point, witness);
    direction = -simplex.points[0];

    for _ in 0..MAX_ITERATIONS {
        let (point, witness) = world_support_witness(a, b, direction);

        // the origin is not reachable: the minkowski difference lies entirely
        // on the far side of this support plane;
        if point.dot(direction) < 0.0 {
            return false;
        }

        simplex.push_front(point, witness);
        if contains_origin(&mut simplex, &mut direction) {
            *out = simplex;
            return true;
        }
        // degenerate direction: escape the degenerate subspace;
        if direction.length_squared() < 1e-12 {
            direction = any_perpendicular(simplex.points[0]);
        }
    }
    false
}

/// refine the simplex toward the origin; the most recent point is at index 0
fn contains_origin(simplex: &mut Simplex, direction: &mut Vec3) -> bool {
    if simplex.len == 4 {
        let a = simplex.points[0];
        let b = simplex.points[1];
        let c = simplex.points[2];
        let d = simplex.points[3];
        let ao = -a;
        let ab = b - a;
        let ac = c - a;
        let ad = d - a;
        let abc = ab.cross(ac);
        let acd = ac.cross(ad);
        let adb = ad.cross(ab);

        // reduce to the face toward the origin; if no face is, the origin is
        // inside the tetrahedron;
        if same_direction(abc, ao) {
            simplex.remove(3); // keep a, b, c
        } else if same_direction(acd, ao) {
            simplex.remove(1); // keep a, c, d
        } else if same_direction(adb, ao) {
            simplex.remove(2); // keep a, d, b
        } else {
            return true;
        }
    }

    match simplex.len {
        1 => {
            *direction = -simplex.points[0];
            false
        }
        2 => {
            let a = simplex.points[0];
            let b = simplex.points[1];
            let ab = b - a;
            let ao = -a;
            if same_direction(ab, ao) {
                let mut d = triple_product(ab, ao, ab);
                if d.length_squared() < 1e-12 {
                    d = any_perpendicular(ab);
                }
                *direction = d;
            } else {
                simplex.remove(1);
                *direction = ao;
            }
            false
        }
        3 => {
            let a = simplex.points[0];
            let b = simplex.points[1];
            let c = simplex.points[2];
            let ab = b - a;
            let ac = c - a;
            let ao = -a;
            let abc = ab.cross(ac);

            if same_direction(abc.cross(ac), ao) {
                if same_direction(ac, ao) {
                    simplex.remove(1); // keep a, c
                    *direction = triple_product(ac, ao, ac);
                } else if same_direction(ab, ao) {
                    simplex.remove(2); // keep a, b
                    *direction = triple_product(ab, ao, ab);
                } else {
                    simplex.truncate(1); // keep a
                    *direction = ao;
                }
            } else if same_direction(ab.cross(abc), ao) {
                if same_direction(ab, ao) {
                    simplex.remove(2); // keep a, b
                    *direction = triple_product(ab, ao, ab);
                } else {
                    simplex.truncate(1); // keep a
                    *direction = ao;
                }
            } else {
                if abc.dot(ao) > 0.0 {
                    *direction = abc;
                } else {
                    simplex.swap(1, 2);
                    *direction = -abc;
                }
            }
            false
        }
        _ => false,
    }
}

fn same_direction(d1: Vec3, d2: Vec3) -> bool {
    d1.dot(d2) > 0.0
}

/// `a x b x c`, the vector perpendicular to `a` and `b` in the plane of `c`
/// not commutative (nor associative)! make sure you have the right order!
fn triple_product(a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
    a.cross(b).cross(c)
}

/// any unit-ish vector perpendicular to `v`, robust for axis-aligned input
pub fn any_perpendicular(v: Vec3) -> Vec3 {
    let axis = if v.x.abs() <= v.y.abs() && v.x.abs() <= v.z.abs() {
        Vec3::X
    } else if v.y.abs() <= v.z.abs() {
        Vec3::Y
    } else {
        Vec3::Z
    };
    axis.cross(v).normalize_or_zero()
}
