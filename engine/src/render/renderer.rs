use std::error::Error;

use glam::Mat4;
use winit::dpi::PhysicalSize;
use winit::raw_window_handle::RawDisplayHandle;
use winit::window::Window;

use super::camera::Camera;
use super::compute::{self, NarrowPhaseCompute};
use super::context::VulkanContext;
use super::frame::ShapeDraw;
use super::gpu_physics::{self, AvbdCompute};
use super::library::MeshLibrary;
use super::light::Light;
use crate::asset;
use crate::scene::MeshShape;
use crate::scene::Scene;
use crate::ui::UiFrame;

/// high-level renderer: owns the vulkan context, the interned mesh library
/// and the camera too
/// every frame it reads the `scene` (pure data, physics-mutable), groups
/// instances by (shape, style), and draws each group once with an instanced
/// `vkcmddrawindexed` call
pub struct Renderer {
    context: VulkanContext,
    library: Option<MeshLibrary>,
    narrow_phase: Option<NarrowPhaseCompute>,
    gpu_physics: Option<AvbdCompute>,
    camera: Camera,
    light: Light,
}

impl Renderer {
    pub fn new(display: RawDisplayHandle) -> Result<Self, Box<dyn Error>> {
        let context = VulkanContext::new(display)?;
        Ok(Self {
            context,
            library: Some(MeshLibrary::new()),
            narrow_phase: None,
            gpu_physics: None,
            camera: Camera::default(),
            light: Light::default(),
        })
    }

    /// the scene camera; mutate it per frame to move the view
    pub fn camera(&self) -> &Camera {
        &self.camera
    }

    pub fn camera_mut(&mut self) -> &mut Camera {
        &mut self.camera
    }

    /// the scene's directional light; mutate it per frame to move the light
    pub fn light(&self) -> &Light {
        &self.light
    }

    pub fn light_mut(&mut self) -> &mut Light {
        &mut self.light
    }

    /// how many instances the uniform buffer can hold on this device?
    pub fn instance_capacity(&self) -> usize {
        self.context.instance_capacity()
    }

    /// submit the gpu broad phase (aabbs + bitonic sort + sweep) over the
    /// current bodies
    /// the pipeline is created lazily on first use; call at
    /// the end of each frame; the candidate pair count is then available
    /// from `self::broad_phase_count`, which feeds `self::narrow_phase_submit`
    pub fn broad_phase_submit(
        &mut self,
        bodies: &[compute::GpuBody],
    ) -> Result<(), Box<dyn Error>> {
        let device = self.context.device().ok_or("no device")?;
        if self.narrow_phase.is_none() {
            self.narrow_phase = Some(NarrowPhaseCompute::new(device, 1024, 16384)?);
        }
        self.narrow_phase
            .as_mut()
            .expect("compute pipeline created")
            .submit_broad(device, bodies)
    }

    /// wait for the broad phase and return the candidate pair count for the
    /// narrow-phase dispatch
    pub fn broad_phase_count(&mut self) -> Result<usize, Box<dyn Error>> {
        let device = self.context.device().ok_or("no device")?;
        match &mut self.narrow_phase {
            Some(narrow_phase) => narrow_phase.read_pair_count(device),
            None => Ok(0),
        }
    }

    /// dispatch the gpu narrow phase (gjk + epa) over the pairs the broad
    /// phase wrote into the pair buffer; consume the contacts with
    /// `self::narrow_phase_read`
    pub fn narrow_phase_submit(&mut self, pair_count: usize) -> Result<(), Box<dyn Error>> {
        let device = self.context.device().ok_or("no device")?;
        match &mut self.narrow_phase {
            Some(narrow_phase) => narrow_phase.submit_narrow(device, pair_count),
            None => Ok(()),
        }
    }

    /// wait for the most recent `self::narrow_phase_submit`
    /// and return its decoded contacts; empty before the
    /// first submit
    pub fn narrow_phase_read(&mut self) -> Result<Vec<compute::GpuContact>, Box<dyn Error>> {
        let device = self.context.device().ok_or("no device")?;
        match &mut self.narrow_phase {
            Some(narrow_phase) => {
                let raw = narrow_phase.read(device)?;
                Ok(compute::decode_contacts(&raw))
            }
            None => Ok(Vec::new()),
        }
    }

    /// submit one gpu avbd step (the block sweeps) over the uploaded state
    /// without waiting; read the result with `self::gpu_physics_read` after
    /// a frame has elapsed; the pipeline is created lazily on first use
    #[allow(clippy::too_many_arguments)]
    pub fn gpu_physics_submit(
        &mut self,
        body_state: &[gpu_physics::GpuAvbdBodyState],
        positions: &[glam::Vec3],
        orientations: &[glam::Quat],
        contacts: &[gpu_physics::GpuAvbdContact],
        constraints: &[gpu_physics::GpuAvbdConstraint],
        offsets: &[u32],
        indices: &[u32],
        options: &gpu_physics::AvbdRunOptions,
    ) -> Result<(), Box<dyn Error>> {
        let device = self.context.device().ok_or("no device")?;
        if self.gpu_physics.is_none() {
            self.gpu_physics = Some(gpu_physics::AvbdCompute::new(device, 1024, 16384)?);
        }
        self.gpu_physics
            .as_mut()
            .expect("avbd compute created")
            .submit(
                device,
                body_state,
                positions,
                orientations,
                contacts,
                constraints,
                offsets,
                indices,
                options,
            )
    }

    /// wait for the most recent `self::gpu_physics_submit` (a full frame
    /// has elapsed) and return the solved state
    pub fn gpu_physics_read(&mut self) -> Result<gpu_physics::AvbdGpuResult, Box<dyn Error>> {
        let device = self.context.device().ok_or("no device")?;
        match &mut self.gpu_physics {
            Some(avbd) => avbd.read(device),
            None => Ok(gpu_physics::AvbdGpuResult {
                positions: Vec::new(),
                orientations: Vec::new(),
                velocities: Vec::new(),
                angular_velocities: Vec::new(),
                prev_velocities: Vec::new(),
                lambda: Vec::new(),
                penalty: Vec::new(),
            }),
        }
    }

    /// whether the swapchain, pipeline and buffers exist; true from the first
    /// per-frame update callback onwards; false before the window exists
    pub fn ready(&self) -> bool {
        self.context.ready()
    }

    /// upload geometry for a shape that has no built-in generator (ex:
    /// `meshshape::icosphere`) from cpu-side mesh data, typically loaded
    /// with `crate::asset::load_obj`; call once before the first frame
    /// that draws the shape
    pub fn register_mesh_data(
        &mut self,
        shape: MeshShape,
        mesh: &asset::Mesh,
    ) -> Result<(), Box<dyn Error>> {
        let device = self
            .context
            .device()
            .ok_or("renderer is not ready; register meshes from the first update onwards")?;
        let library = self.library.as_mut().ok_or("no mesh library")?;
        library.intern(device, shape, mesh)
    }

    pub(crate) fn attach(&mut self, window: &Window) -> Result<(), Box<dyn Error>> {
        self.context.attach(window)
    }

    pub(crate) fn prepare(&mut self, window: &Window) -> Result<(), Box<dyn Error>> {
        self.context.prepare(window)
    }

    pub(crate) fn on_resized(&mut self, size: PhysicalSize<u32>) {
        self.context.on_resized(size);
    }

    /// rebuild draw data from the scene and present one frame
    pub(crate) fn render_frame(
        &mut self,
        scene: &Scene,
        ui: Option<&mut UiFrame>,
    ) -> Result<(), Box<dyn Error>> {
        if !self.context.ready() {
            return Ok(());
        }

        // group instances by (shape, style) (stable sort keeps insertion
        // order within a group), so each group is drawn once with
        // instancecount = group size and firstinstance pointing at its block
        // of mvp slots
        let mut order: Vec<usize> = (0..scene.len()).collect();
        order.sort_by_key(|&i| {
            (
                scene.instances()[i].shape.clone(),
                scene.instances()[i].style,
            )
        });

        let mut models: Vec<Mat4> = Vec::with_capacity(scene.len());
        let mut draws: Vec<ShapeDraw> = Vec::new();

        {
            let device = self.context.device().ok_or("no device")?;
            let library = self.library.as_mut().ok_or("no mesh library")?;

            let mut i = 0;
            while i < order.len() {
                let shape = scene.instances()[order[i]].shape.clone();
                let style = scene.instances()[order[i]].style;
                let mut count = 0;
                while i + count < order.len()
                    && scene.instances()[order[i + count]].shape == shape
                    && scene.instances()[order[i + count]].style == style
                {
                    let instance = &scene.instances()[order[i + count]];
                    models.push(Mat4::from_scale_rotation_translation(
                        instance.transform.scale,
                        instance.transform.rotation,
                        instance.transform.translation,
                    ));
                    count += 1;
                }
                let interned = match library.get_or_intern(device, shape.clone()) {
                    Ok(interned) => interned,
                    // unregistered shape (e;g; icosphere before its data is
                    // registered): skip the group this frame;
                    Err(error) => {
                        log::warn!("skipping {shape:?}: {error}");
                        i += count;
                        continue;
                    }
                };
                draws.push(ShapeDraw {
                    vertex_buffer: interned.vertex_buffer(),
                    index_buffer: interned.index_buffer(),
                    index_count: interned.index_count(),
                    first_instance: i as u32,
                    instance_count: count as u32,
                    style,
                });
                i += count;
            }
        }

        let extent = self.context.extent();
        let aspect = extent.width as f32 / extent.height.max(1) as f32;
        let view = self.camera.view();
        let projection = self.camera.projection(aspect);
        self.context.render_frame(
            &view,
            &projection,
            &models,
            &draws,
            ui,
            self.camera.position(),
            &self.light,
        )
    }

    fn destroy_library(&mut self) {
        if let Some(device) = self.context.device() {
            // let any in-flight frame finish before the buffers go away
            unsafe {
                let _ = device.device.device_wait_idle();
            }
            if let Some(mut library) = self.library.take() {
                library.destroy(device);
            }
            if let Some(mut avbd) = self.gpu_physics.take() {
                avbd.destroy(device);
            }
        }
    }

    pub(crate) fn detach(&mut self) {
        self.destroy_library();
        self.context.detach();
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        // the interned buffers must be destroyed before the device goes away;
        // the context's own drop tears down the remaining graphics resources
        self.destroy_library();
        if let Some(device) = self.context.device() {
            if let Some(mut narrow_phase) = self.narrow_phase.take() {
                narrow_phase.destroy(device);
            }
        }
    }
}
