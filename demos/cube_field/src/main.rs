//! cube field demo: 25 spinning cubes built into a scene at init, drawn
//! with a single instanced draw call; camera controls: mouse moves the look
//! direction, wasd move in the camera's local plane, q/e move down/up, and
//! the scroll wheel zooms (changes fov); an egui fps label is drawn in the
//! bottom-left corner;

use diomedes::app::Diomedes;
use diomedes::egui;
use diomedes::glam::Vec3;
use diomedes::scene::{MeshShape, Scene, Transform};

fn main() {
    env_logger::init();

    // build the scene at init: a 5x5 grid of cubes;
    let mut scene = Scene::new();
    for i in 0..25 {
        let x = (i % 5) as f32 - 2.0;
        let z = (i / 5) as f32 - 2.0;
        scene = scene.with(
            MeshShape::Cube,
            Transform::new(
                Vec3::new(x * 1.6, 0.0, z * 1.6),
                diomedes::glam::Quat::IDENTITY,
                Vec3::splat(0.55),
            ),
        );
    }

    // initial camera: (8, 6, 8) looking at the origin; the yaw/pitch here
    // reproduce that look direction; the mouse takes over from the first
    // frame;
    let mut yaw = -2.3562_f32; // atan2(-8, -8)
    let mut pitch = -0.4878_f32; // asin(-6 / |(8,6,8)|)

    let mut time = 0.0f32;
    let mut camera_set = false;
    let mut triangle_spawned = false;

    // fps counter state (smoothed over half-second windows);
    let mut fps_accum = 0.0f32;
    let mut fps_frames = 0u32;
    let mut fps_timer = 0.0f32;
    let mut fps = 0.0f32;

    let app = Diomedes::new(scene, move |renderer, scene, input, ctx, delta| {
        let delta = delta as f32;
        time += delta;

        // place the camera once; the controls below move it from there;
        if !camera_set {
            renderer.camera_mut().set_position(Vec3::new(8.0, 6.0, 8.0));
            camera_set = true;
        }

        // mouse look: yaw around the world up axis, pitch around the local
        // right axis; clamp pitch so the camera never flips over the poles;
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

        // wasd / qe movement along the camera's local basis, speed in
        // units per second;
        const SPEED: f32 = 5.0;
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
            move_right * SPEED * delta,
            move_forward * SPEED * delta,
            move_up * SPEED * delta,
        );

        // scroll wheel zooms: fov between 14 and 90 degrees; scroll up (away)
        // zooms in;
        // ~8;6 degrees of fov per wheel notch;
        let fov = (camera.vertical_fov() - input.scroll_delta() as f32 * 0.15).clamp(0.25, 1.6);
        camera.set_vertical_fov(fov);

        // runtime add: scenes can grow after init; the triangle's geometry is
        // interned lazily the first time it is drawn;
        if !triangle_spawned && time > 3.0 {
            scene.add_shape(
                MeshShape::Triangle,
                Transform::new(
                    Vec3::new(0.0, 2.2, 0.0),
                    diomedes::glam::Quat::IDENTITY,
                    Vec3::splat(1.2),
                ),
            );
            triangle_spawned = true;
        }

        // spin and kick every instance, phase-shifted across the grid;
        for (i, instance) in scene.instances_mut().iter_mut().enumerate() {
            let phase = i as f32 * 0.35;
            instance.transform.translation.y = (time * 1.2 + phase).sin() * 0.3;
            instance.transform.rotation = diomedes::glam::Quat::from_axis_angle(
                Vec3::new(1.0, 1.0, 0.0).normalize(),
                time * 0.6 + phase,
            );
        }

        // fps counter, smoothed over half a second;
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

        // egui ui: a monospace fps label anchored to the bottom-left;
        egui::Area::new(egui::Id::new("fps_counter"))
            .anchor(egui::Align2::LEFT_BOTTOM, [8.0, -8.0])
            .show(ctx, |ui| {
                ui.monospace(format!("{fps:.0} FPS"));
            });
    })
    .expect("failed to create application")
    .with_cursor_lock();

    app.run().expect("application run failed");
}
