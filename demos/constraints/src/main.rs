//! constraint showcase; springs, rods, ball-and-socket joints and ropes move
//! side by side under the gpu avbd solver
mod visualisations;

use diomedes::app::Diomedes;
use diomedes::egui;
use diomedes::glam::{Quat, Vec3};
use diomedes::physics::{
    AvbdBody, AvbdOptions, AvbdSolver, BallSocket, Constraint, Rod, Rope, Spring,
};
use diomedes::render::gpu_physics;
use diomedes::scene::{MeshShape, Scene, Transform};
use visualisations::ConstraintVisualisation;

const BODY_SCALE: f32 = 0.3;

fn kicked_body(position: Vec3, velocity: Vec3, angular_velocity: Vec3) -> AvbdBody {
    let mut body = AvbdBody::cube(position, BODY_SCALE, 1.0);
    body.velocity = velocity;
    body.angular_velocity = angular_velocity;
    body
}

fn trace_constraint_state(solver: &AvbdSolver, frame: u64, enabled: bool) {
    if !enabled {
        return;
    }
    let invalid = solver.bodies.iter().any(|body| {
        !body.position.is_finite()
            || !body.orientation.is_finite()
            || !body.velocity.is_finite()
            || !body.angular_velocity.is_finite()
    });
    if !invalid && frame >= 10 && frame % 30 != 0 {
        return;
    }
    let level = if invalid {
        log::Level::Error
    } else {
        log::Level::Info
    };
    for (index, body) in solver.bodies.iter().enumerate() {
        log::log!(
            level,
            "constraint frame {frame} body {index}: position={:?} velocity={:?} angular_velocity={:?} orientation={:?}",
            body.position,
            body.velocity,
            body.angular_velocity,
            body.orientation,
        );
    }
    for (index, constraint) in solver.constraints.iter().enumerate() {
        let endpoints = constraint.endpoints(&solver.bodies);
        log::log!(
            level,
            "constraint frame {frame} constraint {index} {}: endpoints={endpoints:?}",
            constraint.name(),
        );
    }
}

fn main() {
    env_logger::init();

    let body_positions = [
        Vec3::new(-7.0, 1.6, 0.0),
        Vec3::new(-7.0, 0.0, 0.0),
        Vec3::new(-2.5, 1.6, 0.0),
        Vec3::new(-1.7, 0.2, 0.0),
        Vec3::new(-2.2, -1.0, 0.0),
        Vec3::new(2.2, 1.7, 0.0),
        Vec3::new(2.2, 2.05, 0.0),
        Vec3::new(6.5, 1.7, 0.0),
        Vec3::new(6.5, -0.2, 0.0),
    ];
    let mut scene = Scene::new();
    let body_instances: Vec<_> = body_positions
        .iter()
        .enumerate()
        .map(|(index, &position)| {
            let scale = if matches!(index, 0 | 2 | 5 | 7) {
                Vec3::splat(0.4)
            } else {
                Vec3::splat(BODY_SCALE)
            };
            scene.add_shape(
                MeshShape::Cube,
                Transform::new(position, Quat::IDENTITY, scale),
            )
        })
        .collect();

    let visualisations = vec![
        ConstraintVisualisation::new(&mut scene, 18, 0.18),
        ConstraintVisualisation::new(&mut scene, 12, 0.0),
        ConstraintVisualisation::new(&mut scene, 12, 0.0),
        ConstraintVisualisation::new(&mut scene, 12, 0.0),
        ConstraintVisualisation::new(&mut scene, 12, 0.0),
    ];

    let bodies = vec![
        AvbdBody::cube(body_positions[0], 0.4, 0.0),
        kicked_body(
            body_positions[1],
            Vec3::new(9.2, 4.4, 5.0),
            Vec3::new(0.0, 0.8, 0.0),
        ),
        AvbdBody::cube(body_positions[2], 0.4, 0.0),
        kicked_body(
            body_positions[3],
            Vec3::new(-5.4, 4.0, 5.0),
            Vec3::new(0.0, 0.7, 0.0),
        ),
        kicked_body(
            body_positions[4],
            Vec3::new(5.7, 4.2, 4.0),
            Vec3::new(0.0, -0.5, 0.0),
        ),
        AvbdBody::cube(body_positions[5], 0.4, 0.0),
        kicked_body(
            body_positions[6],
            Vec3::new(5.0, 4.0, 4.8),
            Vec3::new(0.0, 0.0, 1.2),
        ),
        AvbdBody::cube(body_positions[7], 0.4, 0.0),
        kicked_body(
            body_positions[8],
            Vec3::new(-5.8, 5.5, 4.0),
            Vec3::new(0.0, 0.6, 0.0),
        ),
    ];
    let constraints = vec![
        Constraint::Spring(Spring::new(0, 1, Vec3::ZERO, Vec3::ZERO, 12.0, 1.6)),
        Constraint::Rod(Rod::new(2, 3, Vec3::ZERO, Vec3::ZERO, 1.6)),
        Constraint::Rod(Rod::new(3, 4, Vec3::ZERO, Vec3::ZERO, 1.4)),
        Constraint::BallSocket(BallSocket::new(
            5,
            6,
            Vec3::ZERO,
            Vec3::new(0.0, -BODY_SCALE, 0.0),
        )),
        Constraint::Rope(Rope::new(7, 8, Vec3::ZERO, Vec3::ZERO, 1.5)),
    ];
    let mut solver = AvbdSolver::new(bodies);
    solver.constraints = constraints;

    let mut options = AvbdOptions::default();
    options.gravity = Vec3::new(0.0, -3.5, 0.0);
    options.iterations = 16;
    let run_options = gpu_physics::AvbdRunOptions {
        dt: options.dt,
        gravity: options.gravity,
        alpha: options.alpha,
        beta_lin: options.beta_lin,
        penalty_max: options.penalty_max,
        iterations: options.iterations as u32,
    };

    let mut camera_set = false;
    let mut yaw = std::f32::consts::FRAC_PI_8;
    let mut pitch = -0.12f32;
    let mut time = 0.0f32;
    let mut gpu_primed = false;
    let trace_constraints = std::env::var_os("DIOMEDES_TRACE_CONSTRAINTS").is_some();
    let mut frame = 0u64;
    if trace_constraints {
        log::info!("constraint tracing enabled");
    }
    let app = Diomedes::new(scene, move |renderer, scene, input, ctx, delta| {
        let delta = delta as f32;
        time += delta;
        if !camera_set {
            renderer
                .camera_mut()
                .set_position(Vec3::new(0.0, 2.8, 19.0));
            renderer.camera_mut().set_target(Vec3::new(0.0, 0.5, 0.0));
            camera_set = true;
        }
        if gpu_primed {
            let result = renderer
                .gpu_physics_read()
                .expect("gpu constraint solve read failed");
            solver.sync_bodies_from_gpu(
                &result.positions,
                &result.orientations,
                &result.velocities,
                &result.angular_velocities,
                &result.prev_velocities,
                &result.lambda,
                &result.penalty,
            );
            trace_constraint_state(&solver, frame, trace_constraints);
        }
        frame += 1;

        solver.prepare_contacts(Vec::new(), &options);
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
            .expect("gpu constraint solve submit failed");
        gpu_primed = true;

        for (body, instance_id) in solver.bodies.iter().zip(&body_instances) {
            if let Some(instance) = scene.instance_mut(*instance_id) {
                instance.transform.translation = body.position;
                instance.transform.rotation = body.orientation;
            }
        }
        for (constraint, visualisation) in solver.constraints.iter().zip(&visualisations) {
            if let Some((start, end)) = constraint.endpoints(&solver.bodies) {
                visualisation.update(scene, start, end);
            }
        }

        const SENSITIVITY: f32 = 0.0025;
        let (dx, dy) = input.mouse_delta();
        yaw -= dx as f32 * SENSITIVITY;
        pitch = (pitch - dy as f32 * SENSITIVITY).clamp(-1.5, 1.5);
        let forward = Vec3::new(
            pitch.cos() * yaw.sin(),
            pitch.sin(),
            pitch.cos() * yaw.cos(),
        );
        let mut right = 0.0;
        let mut forward_amount = 0.0;
        let mut up = 0.0;
        if input.is_pressed_char('w') {
            forward_amount += 1.0;
        }
        if input.is_pressed_char('s') {
            forward_amount -= 1.0;
        }
        if input.is_pressed_char('a') {
            right -= 1.0;
        }
        if input.is_pressed_char('d') {
            right += 1.0;
        }
        if input.is_pressed_char('q') {
            up -= 1.0;
        }
        if input.is_pressed_char('e') {
            up += 1.0;
        }
        let camera = renderer.camera_mut();
        camera.set_target(camera.position() + forward);
        camera.move_local(
            right * 6.0 * delta,
            forward_amount * 6.0 * delta,
            up * 6.0 * delta,
        );
        camera.set_vertical_fov(
            (camera.vertical_fov() - input.scroll_delta() as f32 * 0.15).clamp(0.25, 1.6),
        );

        renderer.light_mut().direction = Vec3::new(time.cos(), -0.8, time.sin()).normalize();
        egui::Area::new(egui::Id::new("constraint_status"))
            .anchor(egui::Align2::LEFT_TOP, [12.0, 12.0])
            .show(ctx, |ui| {
                for constraint in &solver.constraints {
                    let distance = constraint
                        .endpoints(&solver.bodies)
                        .map_or(0.0, |(a, b)| (b - a).length());
                    ui.monospace(format!("{}  {distance:.2}", constraint.name()));
                }
            });
    })
    .expect("failed to create application... not good")
    .with_cursor_lock();

    app.run().expect("application run failed... also not good");
}
