//! pb-core — the PixelBoost engine.

pub mod analyze;
pub mod classify;
pub mod compose;
pub mod develop;
pub mod draw;
pub mod edit;
pub mod engine;
pub mod enhance;
pub mod fidelity;
pub mod filters;
pub mod fusion;
pub mod fx;
pub mod image;
pub mod mask;
pub mod meta;
pub mod metrics;
pub mod models;
pub mod ops;
pub mod pipeline;
pub mod pixelart;
pub mod presets;
pub mod resample;
pub mod shape;
pub mod stream;
pub mod strips;
pub mod telemetry;
pub mod text;

pub use engine::{backend, gpu_name, gpu_note, gpu_speed, set_backend, set_gpu_tune_cache, Backend, Engine, TileEvent, TileOptions, Tuning};
pub use image::Image;

/// Which SIMD convolution kernel is active.
pub fn kernel_name() -> &'static str {
    pb_nn::conv::kernel_name()
}
