use diomedes::app::Diomedes;
use diomedes::asset;
use diomedes::egui;
use diomedes::glam::{Quat, Vec3};
use diomedes::physics::Shape;
use diomedes::physics::gpu::{AvbdBody, AvbdContainer, AvbdOptions, AvbdSolver};
use diomedes::platform::input::MouseButton;
use diomedes::render::compute::{GpuBody, GpuShape};
use diomedes::render::gpu_physics;
use diomedes::scene::{MeshShape, Scene, Transform};

const WALL_COLUMNS: usize = 15;
const WALL_ROWS: usize = 15;
const BLOCK_COUNT: usize = WALL_COLUMNS * WALL_ROWS;
const BLOCK_HALF_EXTENT: f32 = 0.25;
const BLOCK_SPACING: f32 = BLOCK_HALF_EXTENT * 2.0;
const PROJECTILE_RADIUS: f32 = 0.38;
const PROJECTILE_SPEED: f32 = 100.0;
const PROJECTILE_MASS: f32 = 2.0;
const MAX_PROJECTILES: usize = 24;
const PLATFORM_HALF_EXTENT: f32 = 6.0;
const CONTAINER_HALF_EXTENT: f32 = 4.0;
const CONTAINER_CENTER_Y: f32 = 4.0;

fn main() {
    env_logger::init();

    let platform_scale = Vec3::new(PLATFORM_HALF_EXTENT * 2.0, 0.5, PLATFORM_HALF_EXTENT * 2.0);
    let mut scene = Scene::new().with(
        MeshShape::Cube,
        Transform::new(Vec3::new(0.0, -0.25, 0.0), Quat::IDENTITY, platform_scale),
    );
    let mut solver_bodies = Vec::with_capacity(BLOCK_COUNT + MAX_PROJECTILES);
    let mut render_scales = Vec::with_capacity(BLOCK_COUNT + MAX_PROJECTILES);

    let horizontal_offset = (WALL_COLUMNS as f32 - 1.0) * 0.5;
    for row in 0..WALL_ROWS {
        let y = BLOCK_HALF_EXTENT + row as f32 * BLOCK_SPACING;
        for column in 0..WALL_COLUMNS {
            let position = Vec3::new((column as f32 - horizontal_offset) * BLOCK_SPACING, y, 0.0);
            solver_bodies.push(AvbdBody::cube(position, BLOCK_HALF_EXTENT, 1.0));
            render_scales.push(Vec3::splat(BLOCK_HALF_EXTENT * 2.0));
            scene = scene.with(
                MeshShape::Cube,
                Transform::new(
                    position,
                    Quat::IDENTITY,
                    Vec3::splat(BLOCK_HALF_EXTENT * 2.0),
                ),
            );
        }
    }

    let mut solver = AvbdSolver::new(solver_bodies);
    solver.container = Some(AvbdContainer {
        center: Vec3::new(0.0, CONTAINER_CENTER_Y, 0.0),
        rotation: Quat::IDENTITY,
        half_extent: CONTAINER_HALF_EXTENT,
    });

    let mut options = AvbdOptions::default();
    options.iterations = 32;
    options.alpha = 0.99;
    options.beta_lin = 5_000.0;
    options.margin = 0.002;
    options.penalty_max = 1.0e6;
    options.gamma = 0.995;
    options.friction = 0.25;
    let run_options = gpu_physics::AvbdRunOptions {
        dt: options.dt,
        gravity: options.gravity,
        alpha: options.alpha,
        beta_lin: options.beta_lin,
        penalty_max: options.penalty_max,
        iterations: options.iterations as u32,
    };

    let mut yaw = -2.459_f32;
    let mut pitch = -0.261_f32;
    let mut camera_set = false;
    let mut icosphere_loaded = false;
    let mut projectiles_fired = 0usize;
    let mut broad_primed = false;
    let mut solve_primed = false;
    let physics_debug = std::env::var_os("DIOMEDES_AVBD_DEBUG").is_some();
    let mut physics_frame = 0u64;

    let mut fps_accum = 0.0f32;
    let mut fps_frames = 0u32;
    let mut fps_timer = 0.0f32;
    let mut fps = 0.0f32;

    let app = Diomedes::new(scene, move |renderer, scene, input, ctx, delta| {
        let delta = delta as f32;
        physics_frame += 1;

        if !icosphere_loaded {
            let icosphere = asset::load_obj("res/icosphere.obj", [0.92, 0.76, 0.32])
                .expect("failed to load res/icosphere.obj");
            renderer
                .register_mesh_data(MeshShape::Icosphere, &icosphere)
                .expect("failed to register icosphere mesh");
            icosphere_loaded = true;
        }

        if !camera_set {
            renderer.camera_mut().set_position(Vec3::new(13.0, 9.0, 16.0));
            camera_set = true;
        }

        const SENSITIVITY: f32 = 0.0025;
        let (dx, dy) = input.mouse_delta();
        yaw -= dx as f32 * SENSITIVITY;
        pitch -= dy as f32 * SENSITIVITY;
        pitch = pitch.clamp(-1.5, 1.5);
        let look_direction = Vec3::new(
            pitch.cos() * yaw.sin(),
            pitch.sin(),
            pitch.cos() * yaw.cos(),
        );

        const CAMERA_SPEED: f32 = 5.0;
        let mut move_right = 0.0;
        let mut move_forward = 0.0;
        let mut move_up = 0.0;
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
        camera.set_target(camera.position() + look_direction);
        camera.move_local(
            move_right * CAMERA_SPEED * delta,
            move_forward * CAMERA_SPEED * delta,
            move_up * CAMERA_SPEED * delta,
        );
        let fov = (camera.vertical_fov() - input.scroll_delta() as f32 * 0.15).clamp(0.25, 1.6);
        camera.set_vertical_fov(fov);

        if solve_primed {
            let result = renderer
                .gpu_physics_read()
                .expect("GPU AVBD solve read failed");
            solver.sync_bodies_from_gpu(
                &result.positions,
                &result.orientations,
                &result.velocities,
                &result.angular_velocities,
                &result.prev_velocities,
                &result.lambda,
                &result.penalty,
            );
            if physics_debug {
                let mut max_lambda = 0.0f32;
                let mut max_penalty = 0.0f32;
                for lambda in &result.lambda {
                    max_lambda = max_lambda.max(lambda.x.abs());
                }
                for penalty in &result.penalty {
                    max_penalty = max_penalty.max(penalty.x);
                }
                let floor_y = CONTAINER_CENTER_Y - CONTAINER_HALF_EXTENT;
                let mut max_floor_penetration = 0.0f32;
                let mut floor_body = 0usize;
                for (index, body) in solver.bodies.iter().enumerate() {
                    let support_y = match body.shape {
                        Shape::Sphere(radius) => body.position.y - radius,
                        Shape::Cube(half_extent) => {
                            let axes = [
                                body.orientation * Vec3::X,
                                body.orientation * Vec3::Y,
                                body.orientation * Vec3::Z,
                            ];
                            body.position.y
                                - half_extent
                                    * (axes[0].y.abs() + axes[1].y.abs() + axes[2].y.abs())
                        }
                        Shape::Tetrahedron(_) => body.position.y,
                    };
                    let penetration = (floor_y - support_y).max(0.0);
                    if penetration > max_floor_penetration {
                        max_floor_penetration = penetration;
                        floor_body = index;
                    }
                }
                if physics_frame % 10 == 0 || max_lambda > 1.0e3 {
                    eprintln!(
                        "avbd frame={} max_normal_lambda={max_lambda:.6e} max_normal_penalty={max_penalty:.6e} max_floor_penetration={max_floor_penetration:.6e} floor_body={floor_body}",
                        physics_frame,
                    );
                }
            }
        }

        let contacts = if broad_primed {
            renderer
                .narrow_phase_read()
                .expect("GPU narrow phase read failed")
        } else {
            Vec::new()
        };
        let body_contacts: Vec<_> = contacts
            .iter()
            .map(|contact| {
                AvbdSolver::raw_contact(
                    contact.a as usize,
                    contact.b as usize,
                    &solver.bodies[contact.a as usize],
                    &solver.bodies[contact.b as usize],
                    contact.normal,
                    contact.depth,
                    contact.point_a,
                    contact.point_b,
                    &options,
                )
            })
            .collect();
        if physics_debug && (physics_frame % 10 == 0 || !contacts.is_empty()) {
            let mut deepest = 0.0f32;
            let mut deepest_pair = (0usize, 0usize);
            let mut deepest_normal = Vec3::ZERO;
            for contact in &contacts {
                if contact.depth > deepest {
                    deepest = contact.depth;
                    deepest_pair = (contact.a, contact.b);
                    deepest_normal = contact.normal;
                }
            }
            eprintln!(
                "narrow frame={} contacts={} deepest={deepest:.6e} pair={}:{} normal={deepest_normal:?}",
                physics_frame,
                contacts.len(),
                deepest_pair.0,
                deepest_pair.1,
            );
        }

        if input.is_mouse_button_pressed(MouseButton::Left)
            && projectiles_fired < MAX_PROJECTILES
            && scene.len() < renderer.instance_capacity()
        {
            let launch_direction = renderer.camera().forward();
            let launch_position = renderer.camera().position() + launch_direction * 1.25;
            let mut projectile = AvbdBody::sphere(launch_position, PROJECTILE_RADIUS, PROJECTILE_MASS);
            projectile.velocity = launch_direction * PROJECTILE_SPEED;
            solver.bodies.push(projectile);
            render_scales.push(Vec3::splat(PROJECTILE_RADIUS));
            scene.add_shape(
                MeshShape::Icosphere,
                Transform::new(
                    launch_position,
                    Quat::IDENTITY,
                    Vec3::splat(PROJECTILE_RADIUS),
                ),
            );
            projectiles_fired += 1;
        }

        solver.prepare_contacts(body_contacts, &options);

        let gpu_bodies: Vec<GpuBody> = solver
            .bodies
            .iter()
            .map(|body| {
                let shape = match body.shape {
                    Shape::Sphere(radius) => GpuShape::sphere(radius),
                    Shape::Cube(half_extent) => GpuShape::cube(half_extent),
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

        let state = gpu_physics::state_from_bodies(&solver.bodies);
        let positions: Vec<_> = solver.bodies.iter().map(|body| body.position).collect();
        let orientations: Vec<_> = solver.bodies.iter().map(|body| body.orientation).collect();
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

        for ((instance, body), scale) in scene
            .instances_mut()
            .iter_mut()
            .skip(1)
            .zip(solver.bodies.iter())
            .zip(render_scales.iter())
        {
            instance.transform.translation = body.position;
            instance.transform.rotation = body.orientation;
            instance.transform.scale = *scale;
        }

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

        egui::Area::new(egui::Id::new("wall_hud"))
            .anchor(egui::Align2::LEFT_TOP, [8.0, 8.0])
            .show(ctx, |ui| {
                ui.monospace(format!(
                    "{fps:.0} FPS   blocks: {BLOCK_COUNT}   bodies: {}",
                    solver.bodies.len()
                ));
            });
    })
    .expect("failed to create application")
    .with_cursor_lock();

    app.run().expect("application run failed");
}
