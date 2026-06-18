//! gpu-side body encoding: world transform + shape
use spirv_std::glam::{Quat, Vec3, Vec4};

use crate::support::ShapeData;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct GpuBody {
    pub position: Vec4,
    pub orientation: Vec4, // quaternion xyzw
    pub shape: ShapeData,
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
