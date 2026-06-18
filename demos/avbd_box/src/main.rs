//! avbd demo: spheres, cubes and tetrahedra bouncing in a box, solved with the engine's
//! primitive augmented vertex block descent solver; avbd minimizes the
//! variational form of implicit euler by block coordinate descent: each body
//! is a 6-dof block whose solve is independent of every other's within a
//! sweep, so the sweeps parallelize across cores (rayon here);
//! contacts come from the gpu broad and narrow phases; the narrow phase uses
//! analytic sphere contacts plus gjk and epa witness reconstruction;
//!
//! press p to toggle the parallel sweeps; the hud shows the per-frame solver
//! time for the current mode; controls: wasd + mouse look, q/e up-down,
//! scroll to zoom;

use std::time::Instant;

use diomedes::app::Diomedes;
use diomedes::asset;
use diomedes::egui;
use diomedes::glam::{Quat, Vec3};
use diomedes::physics::Shape;
use diomedes::physics::gpu::{AvbdBody, AvbdContainer, AvbdOptions, AvbdSolver};
use diomedes::render::compute::{GpuBody, GpuShape};
use diomedes::render::gpu_physics;
use diomedes::scene::{MeshShape, RenderStyle, Scene, Transform};
use rand::Rng;

/// target sphere count; the demo clamps to the device's instance capacity;
const MAX_BODIES: usize = 240;
const SPHERE_RADIUS: f32 = 0.3;
const CUBE_COUNT: usize = 16;
const TETRA_COUNT: usize = 16;
const CUBE_HALF_EXTENT: f32 = 0.4;
const TETRA_SCALE: f32 = 0.55;
const BOX_HALF_EXTENT: f32 = 2.0;

fn main() {
    env_logger::init();

    // the wireframe box is the built-in cube, rendered with line polygons;
    // its interior is the avbd container (walls, not a solid);
    let scene = Scene::new().with_styled(
        MeshShape::Cube,
        Transform::new(Vec3::ZERO, Quat::IDENTITY, Vec3::splat(4.0)),
        RenderStyle::Wireframe,
    );

    // initial camera facing the origin;
    let mut yaw = -2.3562_f32;
    let mut pitch = -0.459_f32;

    let mut time = 0.0f32;
    let mut camera_set = false;
    let mut loaded = false;
    // per-body render scale, mirroring the spawn order (spheres, cubes and
    // tetrahedra have different baked sizes);
    let mut render_scales: Vec<f32> = Vec::with_capacity(MAX_BODIES);

    let mut solver = AvbdSolver::new(Vec::new());
    solver.container = Some(AvbdContainer {
        center: Vec3::ZERO,
        rotation: Quat::IDENTITY,
        half_extent: BOX_HALF_EXTENT,
    });
    let mut options = AvbdOptions::default();
    // the reference α = 0;99 corrects only 1% of the penetration per step;
    // a dense bouncing pile creeps its lateral pressure into the walls at
    // equilibrium penetration ≈ creep / (1-α); 0;8 keeps the surfaces at
    // the walls while staying lively;
    options.alpha = 0.8;
    let run_options = gpu_physics::AvbdRunOptions {
        dt: options.dt,
        gravity: options.gravity,
        alpha: options.alpha,
        beta_lin: options.beta_lin,
        penalty_max: options.penalty_max,
        iterations: options.iterations as u32,
    };
    // hud state;
    let mut fps_accum = 0.0f32;
    let mut fps_frames = 0u32;
    let mut fps_timer = 0.0f32;
    let mut fps = 0.0f32;
    let mut solver_ms = 0.0f32;
    let mut broad_primed = false;
    let mut solve_primed = false;
    let mut perf_timer = 0.0f32;

    let narrow_ms = 0.0f32;
    let mut update_ms = 0.0f32;
    let mut sync_ms = 0.0f32;
    let mut _egui_ms = 0.0f32;
    let submit_ms = 0.0f32;
    let mut rest_ms = 0.0f32;

    let app = Diomedes::new(scene, move |renderer, scene, input, ctx, delta| {
        let update_start = Instant::now();
        let mut accounted = 0.0f32;
        let delta = delta as f32;
        time += delta;

        // once the renderer is ready: register the icosphere mesh and spawn
        // the spheres, clamped to the device's instance capacity;
        if !loaded {
            let icosphere = asset::load_obj("res/icosphere.obj", [0.75, 0.78, 0.82])
                .expect("failed to load res/icosphere.obj");
            renderer
                .register_mesh_data(MeshShape::Icosphere, &icosphere)
                .expect("failed to register icosphere mesh");
            let count = renderer.instance_capacity().saturating_sub(1).min(MAX_BODIES);
            // tetrahedron corners from the obj (deduped by position), scaled;
            let tetra_mesh = asset::load_obj("res/tetrahedron.obj", [0.9, 0.6, 0.4])
                .expect("failed to load res/tetrahedron.obj");
            renderer
                .register_mesh_data(MeshShape::Tetrahedron, &tetra_mesh)
                .expect("failed to register tetrahedron mesh");
            let mut corners: Vec<Vec3> = Vec::new();
            for vertex in &tetra_mesh.vertices {
                let point = Vec3::from_array(vertex.position);
                if !corners.iter().any(|c| (*c - point).length_squared() < 1e-6) {
                    corners.push(point);
                }
            }
            let tetra_corners: [Vec3; 4] = corners
                .try_into()
                .expect("tetrahedron OBJ has four unique vertices");
            let tetra_corners = tetra_corners.map(|c| c * TETRA_SCALE);

            let mut rng = rand::rng();
            for index in 0..count {
                let mut position = Vec3::ZERO;
                let mut placed = false;
                for _ in 0..256 {
                    let candidate = Vec3::new(
                        rng.random_range(-1.6..1.6),
                        rng.random_range(-1.6..1.6),
                        rng.random_range(-1.6..1.6),
                    );
                    let clear = solver.bodies.iter().all(|placed: &AvbdBody| {
                        (candidate - placed.position).length_squared()
                            > 0.75 * 0.75
                    });
                    if clear {
                        position = candidate;
                        placed = true;
                        break;
                    }
                }
                if !placed {
                    position = Vec3::new(
                        rng.random_range(-1.6..1.6),
                        rng.random_range(-1.6..1.6),
                        rng.random_range(-1.6..1.6),
                    );
                }
                // a few cubes and tetrahedra exercise the epa contact path;
                // spheres make up the bulk;
                let (mut body, render_scale, shape) = match index {
                    i if i < CUBE_COUNT => (
                        AvbdBody::cube(position, CUBE_HALF_EXTENT, 1.0),
                        CUBE_HALF_EXTENT * 2.0,
                        MeshShape::Cube,
                    ),
                    i if i < CUBE_COUNT + TETRA_COUNT => (
                        AvbdBody::tetrahedron(position, tetra_corners, 1.0),
                        TETRA_SCALE,
                        MeshShape::Tetrahedron,
                    ),
                    _ => (
                        AvbdBody::sphere(position, SPHERE_RADIUS, 1.0),
                        SPHERE_RADIUS,
                        MeshShape::Icosphere,
                    ),
                };
                body.velocity = Vec3::new(
                    rng.random_range(-0.9..0.9),
                    rng.random_range(-0.9..0.9),
                    rng.random_range(-0.9..0.9),
                );
                body.angular_velocity = Vec3::new(
                    rng.random_range(-1.2..1.2),
                    rng.random_range(-1.2..1.2),
                    rng.random_range(-1.2..1.2),
                );
                solver.bodies.push(body);
                scene.add_shape(
                    shape,
                    Transform::new(position, Quat::IDENTITY, Vec3::splat(render_scale)),
                );
                render_scales.push(render_scale);
            }
            loaded = true;
        }

        if !camera_set {
            renderer.camera_mut().set_position(Vec3::new(5.0, 3.5, 5.0));
            camera_set = true;
        }

        // orbit the directional light so the shading sweeps the spheres;
        let light = renderer.light_mut();
        light.direction = Vec3::new(0.5 * time.cos(), -0.7, 0.5 * time.sin()).normalize();

        // --- simulation (pipelined): the solve for this frame was submitted
        // at the end of the previous frame and ran during its render, so the
        // read here is a cheap sync; the contacts (broad + narrow) for the
        // current state were also computed last frame's end;
        let step_start = Instant::now();
        if solve_primed {
            let result = renderer.gpu_physics_read().expect("GPU AVBD solve read failed");
            solver.sync_bodies_from_gpu(
                &result.positions,
                &result.orientations,
                &result.velocities,
                &result.angular_velocities,
                &result.prev_velocities,
                &result.lambda,
                &result.penalty,
            );
        }

        let contacts = if broad_primed {
            renderer.narrow_phase_read().expect("GPU narrow phase read failed")
        } else {
            Vec::new()
        };
        let body_contacts: Vec<_> = contacts
            .iter()
            .map(|c| {
                AvbdSolver::raw_contact(
                    c.a as usize,
                    c.b as usize,
                    &solver.bodies[c.a as usize],
                    &solver.bodies[c.b as usize],
                    c.normal,
                    c.depth,
                    c.point_a,
                    c.point_b,
                    &options,
                )
            })
            .collect();
        solver.prepare_contacts(body_contacts, &options);
        let step_piece = step_start.elapsed().as_secs_f32() * 1000.0;
        solver_ms = solver_ms * 0.95 + step_piece * 0.05;
        accounted += step_piece;

        // submit the gpu broad phase over the post-step state, then read the
        // pair count and dispatch the narrow phase ; all before the render
        // queues, so the narrow's fence has a full frame to signal;
        let gpu_bodies: Vec<GpuBody> = solver
            .bodies
            .iter()
            .map(|body| {
                let shape = match body.shape {
                    Shape::Sphere(r) => GpuShape::sphere(r),
                    Shape::Cube(h) => GpuShape::cube(h),
                    Shape::Tetrahedron(corners) => GpuShape::tetrahedron(corners),
                };
                GpuBody::new(body.position, body.orientation, shape)
            })
            .collect();
        renderer
            .broad_phase_submit(&gpu_bodies)
            .and_then(|()| renderer.broad_phase_count())
            .and_then(|count| renderer.narrow_phase_submit(count))
            .expect("GPU broad/narrow phase submit failed");
        broad_primed = true;

        // submit the next frame's solve: the current state plus the
        // contacts for it (built above), dispatched after the render so the
        // gpu runs it during the next frame's setup; read back next frame;
        {
            let state = gpu_physics::state_from_bodies(&solver.bodies);
            let positions: Vec<_> = solver.bodies.iter().map(|b| b.position).collect();
            let orientations: Vec<_> = solver.bodies.iter().map(|b| b.orientation).collect();
            let gpu_contacts = gpu_physics::contacts_from_avbd(solver.contacts());
            let gpu_constraints = gpu_physics::constraints_from_avbd(&solver.constraints);
            renderer
                .gpu_physics_submit(
                &state,
                &positions,
                &orientations,
                &gpu_contacts,
                &gpu_constraints,
                solver.contact_offsets(),
                solver.contact_indices(),
                &run_options,
                )
                .expect("GPU AVBD solve submit failed");
            solve_primed = true;
        }

        // --- scene sync --------------------------------------------------
        let t2 = Instant::now();
        for (i, instance) in scene.instances_mut().iter_mut().skip(1).enumerate() {
            instance.transform.translation = solver.bodies[i].position;
            instance.transform.rotation = solver.bodies[i].orientation;
            instance.transform.scale = Vec3::splat(render_scales[i]);
        }
        let sync_piece = t2.elapsed().as_secs_f32() * 1000.0;
        sync_ms = sync_ms * 0.9 + sync_piece * 0.1;
        accounted += sync_piece;

        // --- toggle the parallel sweeps ---------------------------------
        // --- camera controls --------------------------------------------
        const SENSITIVITY: f32 = 0.0025;
        let (dx, dy) = input.mouse_delta();
        yaw -= dx as f32 * SENSITIVITY;
        pitch -= dy as f32 * SENSITIVITY;
        pitch = pitch.clamp(-1.5, 1.5);

        let forward = Vec3::new(
            pitch.cos() * yaw.sin(),
            pitch.sin(),
            pitch.cos() * yaw.cos(),
        );

        const SPEED_CAM: f32 = 5.0;
        let mut move_right = 0.0f32;
        let mut move_forward = 0.0f32;
        let mut move_up = 0.0f32;
        if input.is_pressed_char('w') {
            move_forward += 1.0;
        }
        if input.is_pressed_char('s') {
            move_forward -= 1.0;
        }
        if input.is_pressed_char('a') {
            move_right -= 1.0;
        }
        if input.is_pressed_char('d') {
            move_right += 1.0;
        }
        if input.is_pressed_char('q') {
            move_up -= 1.0;
        }
        if input.is_pressed_char('e') {
            move_up += 1.0;
        }

        let camera = renderer.camera_mut();
        camera.set_target(camera.position() + forward);
        camera.move_local(
            move_right * SPEED_CAM * delta,
            move_forward * SPEED_CAM * delta,
            move_up * SPEED_CAM * delta,
        );
        let fov = (camera.vertical_fov() - input.scroll_delta() as f32 * 0.15).clamp(0.25, 1.6);
        camera.set_vertical_fov(fov);

        // --- hud ---------------------------------------------------------
        if delta > 0.0 {
            fps_accum += 1.0 / delta;
            fps_frames += 1;
            fps_timer += delta;
        }
        if fps_timer >= 0.5 {
            fps = fps_accum / fps_frames.max(1) as f32;
            fps_accum = 0.0;
            fps_frames = 0;
            fps_timer = 0.0;
        }
        if delta > 0.0 {
            perf_timer += delta;
        }
        if perf_timer >= 2.0 {
            let total = update_start.elapsed().as_secs_f32() * 1000.0;
            update_ms = update_ms * 0.9 + total * 0.1;
            rest_ms = rest_ms * 0.9 + (total - accounted).max(0.0) * 0.1;
            eprintln!(
                "perf: {fps:.1} FPS, update {update_ms:.2} ms (read {narrow_ms:.2} solve {solver_ms:.3} sync {sync_ms:.2} submit {submit_ms:.2} rest {rest_ms:.2}), {} bodies, {} contacts, narrow={} solve={}",
                solver.bodies.len(),
                solver.contact_count(),
                "gpu",
                "gpu",
            );
            perf_timer = 0.0;
        }

        egui::Area::new(egui::Id::new("hud"))
            .anchor(egui::Align2::LEFT_BOTTOM, [8.0, -8.0])
            .show(ctx, |ui| {
                ui.monospace(format!("{fps:.0} FPS"));
                ui.monospace(format!(
                    "solver: {solver_ms:.4} ms ({} narrow, {} solve, {} iterations)",
                    "gpu",
                    "gpu",
                    options.iterations
                ));
                ui.monospace(format!("bodies: {}", solver.bodies.len()));
            });
    })
    .expect("failed to create application")
    .with_cursor_lock();

    app.run().expect("application run failed");
}
