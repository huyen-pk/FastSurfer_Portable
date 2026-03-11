use criterion::{Criterion, black_box, criterion_group, criterion_main};
use std::fs;
use std::path::PathBuf;
use std::time::{Duration, Instant};

#[path = "../src/inference/onnx_loader_candle.rs"]
mod onnx_loader;
#[path = "../src/inference/preprocess.rs"]
mod preprocess;

fn find_repo_root() -> Option<PathBuf> {
    let mut candidates = vec![PathBuf::from(env!("CARGO_MANIFEST_DIR"))];
    if let Ok(cwd) = std::env::current_dir() {
        candidates.push(cwd);
    }

    for mut current in candidates {
        loop {
            if current.join("app/backend/ipc_server.py").exists() {
                return Some(current);
            }
            if !current.pop() {
                break;
            }
        }
    }
    None
}

fn vm_rss_kb() -> Option<u64> {
    let status = fs::read_to_string("/proc/self/status").ok()?;
    for line in status.lines() {
        if let Some(value) = line.strip_prefix("VmRSS:") {
            let kb = value
                .split_whitespace()
                .next()
                .and_then(|v| v.parse::<u64>().ok())?;
            return Some(kb);
        }
    }
    None
}

fn faults() -> Option<(i64, i64)> {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
    let rc = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
    if rc != 0 {
        return None;
    }
    let usage = unsafe { usage.assume_init() };
    Some((usage.ru_minflt, usage.ru_majflt))
}

fn channel_count_from_shape(shape: &Option<Vec<usize>>) -> usize {
    shape
        .as_ref()
        .and_then(|dims| dims.get(1).copied())
        .filter(|channels| *channels > 0)
        .unwrap_or(7)
}

struct BenchState {
    sessions: onnx_loader::NativeOnnxSessions,
    volume: preprocess::InputVolume,
    coronal_slice: usize,
}

fn bench_forward_enabled() -> bool {
    std::env::var("FASTSURFER_BENCH_FORWARD")
        .map(|value| {
            let normalized = value.trim().to_ascii_lowercase();
            normalized == "1" || normalized == "true" || normalized == "yes" || normalized == "on"
        })
        .unwrap_or(false)
}

fn prepare_state() -> Result<BenchState, String> {
    let repo_root = find_repo_root().ok_or_else(|| "failed to locate repo root".to_string())?;
    let fixture_dir = repo_root.join("app/gui/desktop/src-tauri/testing/data/.tmp_e2e_output_py");
    let input_nii = fixture_dir.join("140_orig.native_input.nii.gz");
    if !input_nii.exists() {
        return Err(format!(
            "missing fixture input '{}'. generate fixtures first",
            input_nii.display()
        ));
    }

    unsafe {
        std::env::set_var(
            "FASTSURFER_REPO_ROOT",
            repo_root.to_string_lossy().to_string(),
        );
        std::env::set_var("FASTSURFER_NATIVE_TRACE_TIMING", "0");
        std::env::remove_var("FASTSURFER_NATIVE_PLANE_TIMEOUT_SECS");
        if std::env::var("FASTSURFER_NATIVE_CPU_THREADS").is_err() {
            std::env::set_var("FASTSURFER_NATIVE_CPU_THREADS", "16");
        }
    }

    let sessions = onnx_loader::NativeOnnxSessions::load_default()?;
    let volume = preprocess::load_input_volume(&input_nii.to_string_lossy())?;
    let [_, _, coronal_slices] =
        preprocess::transformed_volume_shape(volume.shape_xyz, preprocess::InferencePlane::Coronal);
    if coronal_slices == 0 {
        return Err("coronal slices is zero".to_string());
    }

    Ok(BenchState {
        sessions,
        volume,
        coronal_slice: coronal_slices / 2,
    })
}

fn bench_preprocess_and_forward(c: &mut Criterion) {
    let state = match prepare_state() {
        Ok(state) => state,
        Err(error) => {
            eprintln!("[bench] skipping candle bench setup error: {error}");
            return;
        }
    };

    let coronal_channels = channel_count_from_shape(&state.sessions.coronal.input_shape);

    let mut load_group = c.benchmark_group("native_model_load");
    load_group.measurement_time(Duration::from_secs(6));
    load_group.sample_size(10);
    load_group.bench_function("native_onnx_sessions_load_default", |b| {
        b.iter(|| {
            let sessions = onnx_loader::NativeOnnxSessions::load_default()
                .expect("native onnx session load should succeed");
            black_box(sessions.coronal.input_shape.clone());
        })
    });
    load_group.finish();

    let mut preprocess_group = c.benchmark_group("native_preprocess");
    preprocess_group.measurement_time(Duration::from_secs(12));
    preprocess_group.sample_size(10);
    preprocess_group.bench_function("coronal_center_slice", |b| {
        b.iter(|| {
            let prepared = preprocess::prepare_plane_input_for_slice(
                &state.volume,
                preprocess::InferencePlane::Coronal,
                coronal_channels,
                1.0,
                state.coronal_slice,
            )
            .expect("preprocess should succeed");
            black_box(prepared.tensor_shape);
            black_box(prepared.scale_factor);
            black_box(prepared.tensor_data.len());
        })
    });
    preprocess_group.finish();

    if bench_forward_enabled() {
        unsafe {
            std::env::set_var("FASTSURFER_NATIVE_PLANE_TIMEOUT_SECS", "10");
        }

        let rss0 = vm_rss_kb();
        let faults0 = faults();
        let warmup = preprocess::prepare_plane_input_for_slice(
            &state.volume,
            preprocess::InferencePlane::Coronal,
            coronal_channels,
            1.0,
            state.coronal_slice,
        )
        .and_then(|prepared| {
            state.sessions.coronal.run(
                prepared.tensor_shape.as_slice(),
                prepared.tensor_data.as_slice(),
                prepared.scale_factor,
                "coronal",
            )
        });
        if let Err(error) = warmup {
            eprintln!("[bench] forward warmup failed (skipping forward bench): {error}");
            return;
        }

        let rss1 = vm_rss_kb();
        let faults1 = faults();
        if let (Some(before), Some(after)) = (rss0, rss1) {
            eprintln!(
                "[bench] VmRSS delta (forward warmup): {} kB",
                (after as i64) - (before as i64)
            );
        }
        if let (Some((min0, maj0)), Some((min1, maj1))) = (faults0, faults1) {
            eprintln!(
                "[bench] fault delta (forward warmup): minor={} major={}",
                min1 - min0,
                maj1 - maj0
            );
        }

        let mut forward_group = c.benchmark_group("native_forward_candle");
        forward_group.measurement_time(Duration::from_secs(10));
        forward_group.sample_size(10);
        forward_group.bench_function("coronal_center_slice", |b| {
            b.iter_custom(|iters| {
                let effective = iters.max(1).min(1);
                let start = Instant::now();
                for _ in 0..effective {
                    let prepared = preprocess::prepare_plane_input_for_slice(
                        &state.volume,
                        preprocess::InferencePlane::Coronal,
                        coronal_channels,
                        1.0,
                        state.coronal_slice,
                    )
                    .expect("preprocess should succeed");
                    let run = state
                        .sessions
                        .coronal
                        .run(
                            prepared.tensor_shape.as_slice(),
                            prepared.tensor_data.as_slice(),
                            prepared.scale_factor,
                            "coronal",
                        )
                        .expect("candle forward should succeed");
                    black_box(run.shape_chw);
                    black_box(run.logits.len());
                }
                let elapsed = start.elapsed();
                if iters > effective {
                    let scale = (iters as f64) / (effective as f64);
                    Duration::from_secs_f64(elapsed.as_secs_f64() * scale)
                } else {
                    elapsed
                }
            })
        });
        forward_group.finish();
    } else {
        eprintln!(
            "[bench] forward benchmark disabled by default. Set FASTSURFER_BENCH_FORWARD=1 to enable experimental forward profiling."
        );
    }
}

criterion_group!(benches, bench_preprocess_and_forward);
criterion_main!(benches);
