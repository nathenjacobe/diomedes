//! per-frame input snapshot: keyboard state and mouse movement accumulated
//! from winit events since the previous frame

use std::collections::HashSet;

pub use winit::keyboard::{Key, NamedKey};

/// input state for one frame; the app shell fills it from window/device
/// events and clears it after each update callback, so it always represents
/// input received since the previous frame
#[derive(Default)]
pub struct Input {
    keys_down: HashSet<Key>,
    mouse_delta: (f64, f64),
    scroll: f64,
}

impl Input {
    /// whether the given key is currently held
    pub fn is_pressed(&self, key: Key) -> bool {
        self.keys_down.contains(&key)
    }

    /// whether the given named key (letters, arrows, ...) is currently held
    pub fn is_pressed_named(&self, key: NamedKey) -> bool {
        self.is_pressed(Key::Named(key))
    }

    /// whether the given character key (e;g; `'w'`) is currently held
    pub fn is_pressed_char(&self, c: char) -> bool {
        self.is_pressed(Key::Character(c.to_string().into()))
    }

    /// mouse movement since the previous frame (dx, dy) in screen pixels
    pub fn mouse_delta(&self) -> (f64, f64) {
        self.mouse_delta
    }

    /// scroll wheel movement since the previous frame, in lines (positive
    /// when scrolling up / away from the user)
    pub fn scroll_delta(&self) -> f64 {
        self.scroll
    }

    pub(crate) fn key_pressed(&mut self, key: Key) {
        self.keys_down.insert(key);
    }

    pub(crate) fn key_released(&mut self, key: Key) {
        self.keys_down.remove(&key);
    }

    pub(crate) fn add_mouse_delta(&mut self, dx: f64, dy: f64) {
        self.mouse_delta.0 += dx;
        self.mouse_delta.1 += dy;
    }

    pub(crate) fn add_scroll(&mut self, lines: f64) {
        self.scroll += lines;
    }

    pub(crate) fn clear_frame(&mut self) {
        self.mouse_delta = (0.0, 0.0);
        self.scroll = 0.0;
    }
}
