//! scene data: what to render, independent of any graphics resources;

use glam::{Quat, Vec3};

/// world transform of an instance: scale, then rotation, then translation
#[derive(Clone, Copy, Debug)]
pub struct Transform {
    pub translation: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            translation: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        }
    }
}

impl Transform {
    pub fn new(translation: Vec3, rotation: Quat, scale: Vec3) -> Self {
        Self {
            translation,
            rotation,
            scale,
        }
    }
}

/// how an instance is rendered; the renderer keeps one pipeline per style
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RenderStyle {
    /// filled triangles with back-face culling
    #[default]
    Solid,
    /// triangle edges only (polygon mode `line`); culling is disabled so all
    /// edges are visible; reuses the mesh's triangle data
    Wireframe,
}

/// the kind of geometry an instance refers to; built-in shapes are generated
/// by the renderer
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum MeshShape {
    Triangle,
    Cube,
    /// requires `crate::render::renderer::register_mesh_data` before drawing;
    Icosphere,
    /// requires `crate::render::renderer::register_mesh_data` before drawing;
    Tetrahedron,
}

/// handle to an instance, returned by `scene::add_shape`
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InstanceId(usize);

/// one renderable object in the scene
#[derive(Clone, Debug)]
pub struct Instance {
    pub shape: MeshShape,
    pub transform: Transform,
    pub style: RenderStyle,
}

/// the set of objects to render; pure data
#[derive(Default)]
pub struct Scene {
    instances: Vec<Instance>,
}

impl Scene {
    pub fn new() -> Self {
        Self::default()
    }

    /// builder-style append of a solid shape instance, convenient for
    /// building a scene at init
    pub fn with(mut self, shape: MeshShape, transform: Transform) -> Self {
        self.add_shape(shape, transform);
        self
    }

    /// builder-style append with an explicit render style
    pub fn with_styled(
        mut self,
        shape: MeshShape,
        transform: Transform,
        style: RenderStyle,
    ) -> Self {
        self.add_styled(shape, transform, style);
        self
    }

    /// add a solid shape instance; valid at any time; the renderer picks the
    /// change up on the next frame
    pub fn add_shape(&mut self, shape: MeshShape, transform: Transform) -> InstanceId {
        self.add_styled(shape, transform, RenderStyle::Solid)
    }

    /// add an instance with an explicit render style
    pub fn add_styled(
        &mut self,
        shape: MeshShape,
        transform: Transform,
        style: RenderStyle,
    ) -> InstanceId {
        let id = InstanceId(self.instances.len());
        self.instances.push(Instance {
            shape,
            transform,
            style,
        });
        id
    }

    pub fn instances(&self) -> &[Instance] {
        &self.instances
    }

    pub fn instances_mut(&mut self) -> &mut [Instance] {
        &mut self.instances
    }

    pub fn instance_mut(&mut self, id: InstanceId) -> Option<&mut Instance> {
        self.instances.get_mut(id.0)
    }

    pub fn len(&self) -> usize {
        self.instances.len()
    }

    pub fn is_empty(&self) -> bool {
        self.instances.is_empty()
    }
}
