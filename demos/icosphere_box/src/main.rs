//! icosphere box demo: an icosphere mesh loaded from `res/icosphere;obj`,
//! fourteen instances drifting and bouncing off each other and inside a
//! rotating local-space box; the wireframe box spins about the y axis;
//! simple sphere contacts and wall reflection keep this demo self-contained;

use diomedes::app::Diomedes;
use diomedes::asset;
use diomedes::egui;
use diomedes::glam::{Quat, Vec3};
use diomedes::scene::{MeshShape, RenderStyle, Scene, Transform};
use rand::Rng;

/// the icosphere obj has unit radius; both rendering and physics scale by
/// this factor (the scale lives in the simulation);
const SCALE: f32 = 0.6;
const ICOSPHERE_RADIUS: f32 = 1.0 * SCALE;
/// half-extent of the enclosing box (the wireframe cube is scaled 4x);
const BOX_HALF_EXTENT: f32 = 2.0;

const SPHERE_COUNT: usize = 14;

fn main() {
    env_logger::init();

    // the wireframe box is the built-in cube, rendered with line polygons;
    let mut scene = Scene::new();
    scene = scene.with_styled(
        MeshShape::Cube,
        Transform::new(Vec3::ZERO, Quat::IDENTITY, Vec3::splat(4.0)),
        RenderStyle::Wireframe,
    );

    // random starting positions, uniformly spread through the box (not all
    // at the same shell radius);
    let mut rng = rand::rng();
    let mut positions: Vec<Vec3> = (0..SPHERE_COUNT)
        .map(|_| {
            Vec3::new(
                rng.random_range(-1.4..1.4),
                rng.random_range(-1.4..1.4),
                rng.random_range(-1.4..1.4),
            )
        })
        .collect();
    const SPEED: f32 = 0.6;
    let mut velocities: Vec<Vec3> = (0..SPHERE_COUNT)
        .map(|_| {
            Vec3::new(
                rng.random_range(-1.0..1.0),
                rng.random_range(-1.0..1.0),
                rng.random_range(-1.0..1.0),
            )
            .normalize_or_zero()
                * SPEED
        })
        .collect();

    for &position in &positions {
        scene = scene.with(
            MeshShape::Icosphere,
            Transform::new(position, Quat::IDENTITY, Vec3::splat(SCALE)),
        );
    }

    // initial camera facing the origin;
    let mut yaw = -2.3562_f32;
    let mut pitch = -0.459_f32;

    let mut time = 0.0f32;
    let mut box_angle = 0.0f32;
    let mut camera_set = false;
    let mut icosphere_loaded = false;

    // fps counter state;
    let mut fps_accum = 0.0f32;
    let mut fps_frames = 0u32;
    let mut fps_timer = 0.0f32;
    let mut fps = 0.0f32;

    let app = Diomedes::new(scene, move |renderer, scene, input, ctx, delta| {
        let delta = delta as f32;
        time += delta;
        box_angle += delta * 0.4; // radians per second

        // load the icosphere once the renderer is ready and register it; the
        // scene already references the shape;
        if !icosphere_loaded {
            let icosphere = asset::load_obj("res/icosphere.obj", [0.75, 0.78, 0.82])
                .expect("failed to load res/icosphere.obj");
            renderer
                .register_mesh_data(MeshShape::Icosphere, &icosphere)
                .expect("failed to register icosphere mesh");
            icosphere_loaded = true;
        }

        if !camera_set {
            renderer.camera_mut().set_position(Vec3::new(5.0, 3.5, 5.0));
            camera_set = true;
        }

        // --- simulation -------------------------------------------------
        // move;
        for (i, position) in positions.iter_mut().enumerate() {
            *position += velocities[i] * delta;
        }

        for i in 0..positions.len() {
            for j in (i + 1)..positions.len() {
                let delta = positions[j] - positions[i];
                let distance = delta.length();
                let overlap = 2.0 * ICOSPHERE_RADIUS - distance;
                if overlap <= 0.0 || distance <= f32::EPSILON {
                    continue;
                }
                let normal = delta / distance;
                let vrel = (velocities[j] - velocities[i]).dot(normal);
                if vrel < 0.0 {
                    velocities[i] += normal * vrel;
                    velocities[j] -= normal * vrel;
                }
                let push = normal * (overlap * 0.5);
                positions[i] -= push;
                positions[j] += push;
            }
        }

        let box_rotation = Quat::from_rotation_y(box_angle);
        for i in 0..positions.len() {
            contain_sphere(
                &mut positions[i],
                &mut velocities[i],
                box_rotation,
                BOX_HALF_EXTENT,
                ICOSPHERE_RADIUS,
            );
        }

        // --- scene sync -------------------------------------------------
        let mut sphere_index = 0;
        for instance in scene.instances_mut() {
            if instance.shape == MeshShape::Cube {
                instance.transform.rotation = Quat::from_rotation_y(box_angle);
                continue;
            }
            if instance.shape != MeshShape::Icosphere {
                continue;
            }
            instance.transform.translation = positions[sphere_index];
            let phase = sphere_index as f32 * 0.7;
            instance.transform.rotation =
                Quat::from_axis_angle(Vec3::new(0.4, 1.0, 0.3).normalize(), time * 0.8 + phase);
            sphere_index += 1;
        }

        // --- camera controls -------------------------------------------
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

        // --- ui ---------------------------------------------------------
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

        egui::Area::new(egui::Id::new("hud"))
            .anchor(egui::Align2::LEFT_BOTTOM, [8.0, -8.0])
            .show(ctx, |ui| {
                ui.monospace(format!("{fps:.0} FPS"));
            });
    })
    .expect("failed to create application")
    .with_cursor_lock();

    app.run().expect("application run failed");
}

fn contain_sphere(
    center: &mut Vec3,
    velocity: &mut Vec3,
    rotation: Quat,
    half_extent: f32,
    radius: f32,
) {
    let mut local = rotation.inverse() * *center;
    let mut local_velocity = rotation.inverse() * *velocity;
    let mut clamped = false;

    for axis in 0..3 {
        if local[axis] - radius < -half_extent {
            if local_velocity[axis] < 0.0 {
                local_velocity[axis] = -local_velocity[axis];
            }
            local[axis] = -half_extent + radius;
            clamped = true;
        }
        if local[axis] + radius > half_extent {
            if local_velocity[axis] > 0.0 {
                local_velocity[axis] = -local_velocity[axis];
            }
            local[axis] = half_extent - radius;
            clamped = true;
        }
    }

    if clamped {
        *center = rotation * local;
        *velocity = rotation * local_velocity;
    }
}
