//! pb-gpu — PixelBoost's GPU inference backend.
//!
//! The same networks pb-nn runs on the CPU, run as hand-written WGSL compute
//! kernels through wgpu: Metal on iPhone, Vulkan on Linux. Feature maps use
//! pb-nn's padded HWC layout, so dense-block "concats" are channel offsets
//! and the CPU and GPU paths stay bit-for-bit comparable in tests.
//!
//! Nothing here is required: `Gpu::get()` returns `None` when there is no
//! usable adapter (or `PB_GPU=0`), and every caller falls back to pb-nn.

pub mod check;
mod net;
pub mod shaders;
pub mod tune;

pub use net::GpuNet;

/// The CPU-fallback hook type.
pub mod net_fallback {
    pub use crate::net::CpuFallback;
}

use shaders::ConvCfg;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};


pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub info: wgpu::AdapterInfo,
    /// Largest single storage binding (bytes).
    pub max_binding: u64,
    pub(crate) param_stride: u64,
    pub(crate) layout: wgpu::BindGroupLayout,
    pub(crate) pipe_layout: wgpu::PipelineLayout,
    pipelines: Mutex<HashMap<ConvCfg, Arc<wgpu::ComputePipeline>>>,
    tail: OnceLock<Arc<wgpu::ComputePipeline>>,
}

/// Set after a GPU failure (or device loss): new work goes to the CPU
/// until `Gpu::reset()`.
static BROKEN: AtomicBool = AtomicBool::new(false);

/// While set (app in the background, where iOS forbids GPU work), networks
/// run on the CPU without being marked as failed.
static PAUSED: AtomicBool = AtomicBool::new(false);

pub fn set_paused(p: bool) {
    PAUSED.store(p, Ordering::Relaxed);
}

pub fn paused() -> bool {
    PAUSED.load(Ordering::Relaxed)
}

static GPU: Mutex<Option<Option<Arc<Gpu>>>> = Mutex::new(None);

/// f16 storage failed the self-check on this device: stay on f32.
static NO_HALF: AtomicBool = AtomicBool::new(false);

/// Whether feature maps and weights are stored as f16.
pub fn half() -> bool {
    shaders::half()
}

/// Why there is no GPU (no adapter, software only, failed self-check).
static UNAVAILABLE: Mutex<Option<String>> = Mutex::new(None);

/// The reason `Gpu::get()` returned `None`, if it did.
pub fn unavailable_reason() -> Option<String> {
    UNAVAILABLE.lock().unwrap().clone()
}

impl Gpu {
    /// The shared GPU, created on first use. `None` when disabled
    /// (`PB_GPU=0`), when only a software adapter exists, or after a failure.
    pub fn get() -> Option<Arc<Gpu>> {
        if BROKEN.load(Ordering::Relaxed) {
            return None;
        }
        let mut slot = GPU.lock().unwrap();
        slot.get_or_insert_with(|| {
            if std::env::var("PB_GPU").map(|v| v == "0" || v == "off").unwrap_or(false) {
                return None;
            }
            let mut g = match Gpu::init() {
                Ok(g) => Arc::new(g),
                Err(e) => {
                    eprintln!("pb-gpu: no GPU backend ({e}); using the CPU");
                    pb_nn::telemetry::event("gpu", "unavailable").tag("reason", &e).emit();
                    *UNAVAILABLE.lock().unwrap() = Some(e);
                    return None;
                }
            };
            pb_nn::telemetry::event("gpu", "init").tag("device", g.describe()).tag("driver", format!("{} {}", g.info.driver, g.info.driver_info)).tag("type", format!("{:?}", g.info.device_type)).emit();
            // Never trust a GPU that can't reproduce the CPU networks.
            if std::env::var("PB_GPU_NOCHECK").is_err() {
                let t0 = std::time::Instant::now();
                let mut r = check::self_check(&g);
                pb_nn::telemetry::event("gpu", "self_check").since(t0).tag("device", g.describe()).flag("ok", r.is_ok()).flag("half", shaders::half()).tag("error", r.as_ref().err().map(|s| s.as_str()).unwrap_or("")).emit();
                if r.is_err() && shaders::half() {
                    // Try again with f32 storage before giving up on the GPU.
                    eprintln!("pb-gpu: f16 storage failed the self-check ({}); retrying with f32", r.as_ref().unwrap_err());
                    NO_HALF.store(true, Ordering::Relaxed);
                    match Gpu::init() {
                        Ok(g2) => {
                            g = Arc::new(g2);
                            r = check::self_check(&g);
                            pb_nn::telemetry::event("gpu", "self_check").tag("device", g.describe()).flag("ok", r.is_ok()).flag("half", false).tag("error", r.as_ref().err().map(|s| s.as_str()).unwrap_or("")).emit();
                        }
                        Err(e) => r = Err(e),
                    }
                }
                if let Err(e) = r {
                    eprintln!("pb-gpu: {} failed the self-check ({e}); using the CPU", g.describe());
                    *UNAVAILABLE.lock().unwrap() = Some(format!("{} failed the self-check: {e}", g.describe()));
                    return None;
                }
            }
            Some(g)
        })
        .clone()
    }

    /// After a failure: drop the device so the next `get()` makes a new one.
    /// Returns true if there was a failure to recover from. Networks built
    /// on the old device keep running on the CPU; reload them.
    pub fn reset() -> bool {
        let was = BROKEN.swap(false, Ordering::Relaxed);
        if was {
            *GPU.lock().unwrap() = None;
        }
        was
    }

    pub(crate) fn is_lost(&self) -> bool {
        BROKEN.load(Ordering::Relaxed)
    }

    /// "Apple A17 Pro GPU (Metal)" etc.
    pub fn describe(&self) -> String {
        format!("{} ({:?}{})", self.info.name, self.info.backend, if shaders::half() { ", f16" } else { "" })
    }

    pub(crate) fn mark_broken(&self, why: &str) {
        eprintln!("pb-gpu: disabling the GPU backend: {why}");
        pb_nn::telemetry::event("gpu", "disabled").tag("device", self.describe()).tag("why", why).emit();
        BROKEN.store(true, Ordering::Relaxed);
    }

    fn init() -> Result<Gpu, String> {
        let inst = wgpu::Instance::new(&wgpu::InstanceDescriptor { backends: wgpu::Backends::METAL | wgpu::Backends::VULKAN, ..Default::default() });
        let adapter = pollster::block_on(inst.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .map_err(|e| e.to_string())?;
        let info = adapter.get_info();
        if info.device_type == wgpu::DeviceType::Cpu && std::env::var("PB_GPU").map(|v| v != "any").unwrap_or(true) {
            return Err(format!("{} is a software rasterizer", info.name));
        }
        let al = adapter.limits();
        let limits = wgpu::Limits {
            max_buffer_size: al.max_buffer_size,
            max_storage_buffer_binding_size: al.max_storage_buffer_binding_size,
            max_compute_workgroup_storage_size: al.max_compute_workgroup_storage_size.min(32768),
            max_compute_invocations_per_workgroup: 256,
            max_compute_workgroup_size_x: 256,
            max_storage_buffers_per_shader_stage: al.max_storage_buffers_per_shader_stage.min(8),
            ..wgpu::Limits::downlevel_defaults()
        };
        // f16 storage halves every feature map and the weight bandwidth.
        let half = adapter.features().contains(wgpu::Features::SHADER_F16) && std::env::var("PB_GPU_F16").map(|v| v != "0").unwrap_or(true) && !NO_HALF.load(Ordering::Relaxed);
        crate::shaders::set_half(half);
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("pixelboost"),
            required_features: if half { wgpu::Features::SHADER_F16 } else { wgpu::Features::empty() },
            required_limits: limits.clone(),
            memory_hints: wgpu::MemoryHints::Performance,
            ..Default::default()
        }))
        .map_err(|e| e.to_string())?;
        device.on_uncaptured_error(Arc::new(|e| eprintln!("pb-gpu: uncaptured error: {e}")));
        device.set_device_lost_callback(|reason, msg| {
            eprintln!("pb-gpu: device lost ({reason:?}): {msg}");
            pb_nn::telemetry::event("gpu", "device_lost").tag("reason", format!("{reason:?}")).tag("msg", &msg).emit();
            BROKEN.store(true, Ordering::Relaxed);
        });
        let storage = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only }, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pb-conv"),
            entries: &[
                storage(0, true),
                storage(1, false),
                storage(2, false),
                storage(3, false),
                storage(4, false),
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
            ],
        });
        let pipe_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("pb-conv"), bind_group_layouts: &[&layout], immediate_size: 0 });
        let param_stride = (limits.min_uniform_buffer_offset_alignment as u64).max(net::PARAMS_BYTES);
        Ok(Gpu {
            device,
            queue,
            info,
            max_binding: (limits.max_storage_buffer_binding_size as u64).min(limits.max_buffer_size),
            param_stride,
            layout,
            pipe_layout,
            pipelines: Mutex::new(HashMap::new()),
            tail: OnceLock::new(),
        })
    }

    fn compile(&self, label: &str, src: String) -> Arc<wgpu::ComputePipeline> {
        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some(label), source: wgpu::ShaderSource::Wgsl(src.into()) });
        Arc::new(self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(&self.pipe_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions { constants: &[], zero_initialize_workgroup_memory: false },
            cache: None,
        }))
    }

    pub(crate) fn conv_pipeline(&self, cfg: ConvCfg) -> Arc<wgpu::ComputePipeline> {
        let mut map = self.pipelines.lock().unwrap();
        map.entry(cfg).or_insert_with(|| self.compile(&format!("conv {cfg:?}"), shaders::conv_source(cfg))).clone()
    }

    pub(crate) fn tail_pipeline(&self) -> Arc<wgpu::ComputePipeline> {
        self.tail.get_or_init(|| self.compile("srvgg tail", shaders::tail_source())).clone()
    }
}
