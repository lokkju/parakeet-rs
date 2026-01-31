/*
Benchmark ASR backends (CPU vs WebGPU)

Runs the Nemotron streaming model on .wav files, comparing execution providers.

Usage:
  cargo run --release --features webgpu --example benchmark -- speech_samples/OSR_us_000_0030_8k.wav
  cargo run --release --features webgpu --example benchmark -- speech_samples/ --iterations 3
  cargo run --release --example benchmark -- speech_samples/           # CPU only

  # Auto-download model from HuggingFace Hub (requires hf-hub feature):
  cargo run --release --features webgpu,hf-hub --example benchmark -- speech_samples/ \
    --hf-model altunenes/parakeet-rs --hf-subdir nemotron-speech-streaming-en-0.6b

Options:
  --warmup N        Number of warmup runs (default: 1)
  --iterations N    Number of timed runs per file (default: 3, reports best-of)
  --model-dir DIR   Path to nemotron model directory (default: ./nemotron)
  --batch           Use non-streaming transcribe_audio() instead of per-chunk streaming
  --verbose         Enable ORT verbose logging to see EP node assignments
  --hf-model REPO   HuggingFace repo ID (requires hf-hub feature)
  --hf-subdir PATH  Subdirectory within the HF repo (e.g. nemotron-speech-streaming-en-0.6b)
  --hf-revision REV HuggingFace revision/branch (default: main)
*/

use parakeet_rs::{ExecutionConfig, ExecutionProvider, Nemotron};
use std::env;
use std::path::PathBuf;
use std::time::{Duration, Instant};

const TARGET_SAMPLE_RATE: u32 = 16000;

struct BenchResult {
    provider: String,
    file: String,
    audio_duration_s: f32,
    inference_duration: Duration,
    transcript: String,
}

impl BenchResult {
    fn rtf(&self) -> f32 {
        self.audio_duration_s / self.inference_duration.as_secs_f32()
    }
}

fn load_and_resample(path: &std::path::Path) -> Result<(Vec<f32>, f32), Box<dyn std::error::Error>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();

    let mut audio: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<Vec<_>, _>>()?,
        hound::SampleFormat::Int => reader
            .samples::<i16>()
            .map(|s| s.map(|s| s as f32 / 32768.0))
            .collect::<Result<Vec<_>, _>>()?,
    };

    // Mix to mono
    if spec.channels > 1 {
        audio = audio
            .chunks(spec.channels as usize)
            .map(|c| c.iter().sum::<f32>() / spec.channels as f32)
            .collect();
    }

    let original_duration = audio.len() as f32 / spec.sample_rate as f32;

    // Resample to 16kHz if needed
    if spec.sample_rate != TARGET_SAMPLE_RATE {
        let ratio = spec.sample_rate as f64 / TARGET_SAMPLE_RATE as f64;
        let output_len = (audio.len() as f64 / ratio).ceil() as usize;
        let mut resampled = Vec::with_capacity(output_len);
        for i in 0..output_len {
            let src_pos = i as f64 * ratio;
            let idx = src_pos as usize;
            let frac = (src_pos - idx as f64) as f32;
            let sample = if idx + 1 < audio.len() {
                audio[idx] * (1.0 - frac) + audio[idx + 1] * frac
            } else {
                audio[idx.min(audio.len() - 1)]
            };
            resampled.push(sample);
        }
        audio = resampled;
    }

    // Normalize
    let max_val = audio.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
    if max_val > 1e-6 {
        for s in &mut audio {
            *s /= max_val + 1e-5;
        }
    }

    Ok((audio, original_duration))
}

fn run_benchmark(
    provider: ExecutionProvider,
    provider_name: &str,
    model_dir: &str,
    files: &[(PathBuf, Vec<f32>, f32)],
    warmup: usize,
    iterations: usize,
    batch_mode: bool,
) -> Result<Vec<BenchResult>, Box<dyn std::error::Error>> {
    let config = ExecutionConfig::new().with_execution_provider(provider);

    eprintln!("[{}] Loading model...", provider_name);
    let load_start = Instant::now();
    let mut model = Nemotron::from_pretrained(model_dir, Some(config))?;
    let load_elapsed = load_start.elapsed();
    eprintln!(
        "[{}] Model loaded in {:.2}s",
        provider_name,
        load_elapsed.as_secs_f32()
    );

    let chunk_size = 8960; // 560ms

    // Warmup runs
    if warmup > 0 && !files.is_empty() {
        eprintln!("[{}] Warming up ({} runs)...", provider_name, warmup);
        for _ in 0..warmup {
            model.reset();
            let (_, audio, _) = &files[0];
            if batch_mode {
                let _ = model.transcribe_audio(audio);
            } else {
                for chunk in audio.chunks(chunk_size) {
                    let chunk_vec = if chunk.len() < chunk_size {
                        let mut p = chunk.to_vec();
                        p.resize(chunk_size, 0.0);
                        p
                    } else {
                        chunk.to_vec()
                    };
                    let _ = model.transcribe_chunk(&chunk_vec);
                }
                for _ in 0..3 {
                    let _ = model.transcribe_chunk(&vec![0.0; chunk_size]);
                }
            }
        }
    }

    // Benchmark runs
    let mut results = Vec::new();

    for (path, audio, duration) in files {
        let fname = path.file_name().unwrap().to_string_lossy().to_string();
        let mut best_time = Duration::MAX;
        let mut best_transcript = String::new();

        for iter in 0..iterations {
            model.reset();
            let iter_start = Instant::now();

            if batch_mode {
                let _ = model.transcribe_audio(audio)?;
            } else {
                for chunk in audio.chunks(chunk_size) {
                    let chunk_vec = if chunk.len() < chunk_size {
                        let mut p = chunk.to_vec();
                        p.resize(chunk_size, 0.0);
                        p
                    } else {
                        chunk.to_vec()
                    };
                    let _ = model.transcribe_chunk(&chunk_vec)?;
                }
                for _ in 0..3 {
                    let _ = model.transcribe_chunk(&vec![0.0; chunk_size])?;
                }
            }

            let elapsed = iter_start.elapsed();
            if iter == 0 || elapsed < best_time {
                best_time = elapsed;
                best_transcript = model.get_transcript();
            }

            eprintln!(
                "[{}] {} iter {}/{}: {:.3}s (RTF: {:.1}x)",
                provider_name,
                fname,
                iter + 1,
                iterations,
                elapsed.as_secs_f32(),
                duration / elapsed.as_secs_f32(),
            );
        }

        results.push(BenchResult {
            provider: provider_name.to_string(),
            file: fname,
            audio_duration_s: *duration,
            inference_duration: best_time,
            transcript: best_transcript,
        });
    }

    Ok(results)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!(
            "Usage: benchmark <audio_dir_or_file> [--warmup N] [--iterations N] [--model-dir DIR] [--batch] [--verbose]"
        );
        std::process::exit(1);
    }

    let audio_arg = &args[1];
    let mut warmup = 1;
    let mut iterations = 3;
    let mut model_dir = "./nemotron".to_string();
    let mut batch_mode = false;
    let mut verbose = false;
    #[allow(unused_mut)]
    let mut hf_model: Option<String> = None;
    #[allow(unused_mut)]
    let mut hf_subdir: Option<String> = None;
    #[allow(unused_mut)]
    let mut hf_revision = "main".to_string();

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--warmup" => {
                warmup = args.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(1);
                i += 2;
            }
            "--iterations" => {
                iterations = args.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(3);
                i += 2;
            }
            "--model-dir" => {
                model_dir = args.get(i + 1).cloned().unwrap_or(model_dir);
                i += 2;
            }
            "--batch" => {
                batch_mode = true;
                i += 1;
            }
            "--verbose" => {
                verbose = true;
                i += 1;
            }
            "--hf-model" => {
                hf_model = args.get(i + 1).cloned();
                i += 2;
            }
            "--hf-subdir" => {
                hf_subdir = args.get(i + 1).cloned();
                i += 2;
            }
            "--hf-revision" => {
                hf_revision = args.get(i + 1).cloned().unwrap_or(hf_revision);
                i += 2;
            }
            _ => i += 1,
        }
    }

    // Download model from HuggingFace Hub if requested
    #[cfg(feature = "hf-hub")]
    if let Some(repo_id) = hf_model {
        use hf_hub::api::sync::Api;

        eprintln!(
            "Fetching model from HuggingFace: {} (rev: {})...",
            repo_id, hf_revision
        );
        let api = Api::new().expect("Failed to create HF Hub API client");
        let repo = api.repo(hf_hub::Repo::with_revision(
            repo_id.clone(),
            hf_hub::RepoType::Model,
            hf_revision.clone(),
        ));

        let model_files = ["encoder.onnx", "encoder.onnx.data", "decoder_joint.onnx", "tokenizer.model"];
        let mut cached_dir: Option<PathBuf> = None;

        for file in &model_files {
            let hf_path = match &hf_subdir {
                Some(sub) => format!("{}/{}", sub, file),
                None => file.to_string(),
            };
            match repo.get(&hf_path) {
                Ok(path) => {
                    eprintln!("  {} -> {}", file, path.display());
                    if cached_dir.is_none() {
                        if let Some(parent) = path.parent() {
                            cached_dir = Some(parent.to_path_buf());
                        }
                    }
                }
                Err(e) => {
                    eprintln!("  {} -> FAILED: {}", file, e);
                }
            }
        }

        if let Some(dir) = cached_dir {
            // ORT resolves symlinks when validating external data paths.
            // HF Hub stores files as symlinks to a blobs/ dir, so ORT sees
            // the blob directory as the model dir and can't find sibling files.
            // Fix: create a staging dir with symlinks to the *snapshot* paths
            // (not the resolved blobs). Then override the model dir to point
            // at the staging dir. ORT will resolve each file independently
            // and find them all via their original names.
            //
            // Actually — the snapshot dir already has the right filenames as
            // symlinks. The problem is ORT canonicalizes the .onnx path first,
            // lands in blobs/, then looks for .onnx.data there.
            //
            // Real fix: stage with symlinks pointing at the canonical blob
            // paths, so ORT resolves the .onnx symlink and lands in the
            // staging dir (which is a real dir, not a symlink), then finds
            // .onnx.data as a sibling symlink and resolves that independently.
            // This works because ORT checks that the *unresolved* relative
            // path doesn't escape — and it won't, since both files are
            // siblings in the staging dir.
            let staging_dir = std::env::temp_dir().join("parakeet-rs-benchmark-model");
            std::fs::create_dir_all(&staging_dir)?;

            for file in &model_files {
                let src = dir.join(file);
                let dest = staging_dir.join(file);
                let _ = std::fs::remove_file(&dest);
                if src.exists() {
                    let real_path = std::fs::canonicalize(&src)?;
                    #[cfg(unix)]
                    std::os::unix::fs::symlink(&real_path, &dest)?;
                    #[cfg(not(unix))]
                    std::fs::copy(&real_path, &dest)?;
                }
            }

            model_dir = staging_dir.to_string_lossy().to_string();
            eprintln!("Using staged model dir: {}\n", model_dir);
        } else {
            return Err("Failed to download any model files from HuggingFace".into());
        }
    }

    #[cfg(not(feature = "hf-hub"))]
    if hf_model.is_some() {
        return Err("--hf-model requires the 'hf-hub' feature. Rebuild with: --features hf-hub".into());
    }

    // Enable ORT verbose logging to see EP node placement
    if verbose {
        std::env::set_var("ORT_LOG_LEVEL", "verbose");
        eprintln!("ORT verbose logging enabled (look for node placement info)\n");
    }

    // Collect audio files
    let audio_path = std::path::Path::new(audio_arg);
    #[allow(unused_mut)]
    let mut wav_files: Vec<PathBuf> = if audio_path.is_dir() {
        let mut files: Vec<PathBuf> = std::fs::read_dir(audio_path)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().map_or(false, |ext| ext == "wav"))
            .collect();
        files.sort();
        files
    } else {
        vec![audio_path.to_path_buf()]
    };

    if wav_files.is_empty() {
        eprintln!("No .wav files found in {}", audio_arg);
        std::process::exit(1);
    }

    if wav_files.len() > 20 {
        eprintln!("Found {} files, using first 20", wav_files.len());
        wav_files.truncate(20);
    }

    // Pre-load all audio
    eprintln!("Loading {} audio file(s)...", wav_files.len());
    let mut files: Vec<(PathBuf, Vec<f32>, f32)> = Vec::new();
    let mut total_audio_s = 0.0f32;
    for path in &wav_files {
        let (audio, duration) = load_and_resample(path)?;
        total_audio_s += duration;
        files.push((path.clone(), audio, duration));
    }
    eprintln!(
        "Total audio: {:.1}s across {} files",
        total_audio_s,
        files.len()
    );
    eprintln!(
        "Mode: {}\n",
        if batch_mode {
            "batch (non-streaming)"
        } else {
            "streaming (560ms chunks)"
        }
    );

    // Build list of providers to test
    #[allow(unused_mut)]
    let mut providers: Vec<(ExecutionProvider, &str)> = vec![(ExecutionProvider::Cpu, "CPU")];

    #[cfg(feature = "webgpu")]
    providers.push((ExecutionProvider::WebGPU, "WebGPU"));

    #[cfg(feature = "cuda")]
    providers.push((ExecutionProvider::Cuda, "CUDA"));

    #[cfg(feature = "tensorrt")]
    providers.push((ExecutionProvider::TensorRT, "TensorRT"));

    eprintln!(
        "Backends: {}",
        providers
            .iter()
            .map(|(_, name)| *name)
            .collect::<Vec<_>>()
            .join(", ")
    );
    eprintln!("Warmup: {}, Iterations: {} (best-of)\n", warmup, iterations);

    // Run benchmarks per provider
    let mut all_results: Vec<Vec<BenchResult>> = Vec::new();

    for (provider, name) in &providers {
        eprintln!("=== {} ===", name);
        match run_benchmark(
            *provider,
            name,
            &model_dir,
            &files,
            warmup,
            iterations,
            batch_mode,
        ) {
            Ok(results) => all_results.push(results),
            Err(e) => {
                eprintln!("[{}] FAILED: {}\n", name, e);
                continue;
            }
        }
        eprintln!();
    }

    // Print summary table
    println!();

    // Header with provider columns
    print!("{:<30} {:>8}", "File", "Audio(s)");
    for results in &all_results {
        if let Some(r) = results.first() {
            print!(" | {:>10} {:>8}", r.provider, "RTF");
        }
    }
    println!();
    println!("{}", "-".repeat(30 + 9 + all_results.len() * 21));

    // Per-file rows
    for file_idx in 0..files.len() {
        let (path, _, duration) = &files[file_idx];
        let fname = path.file_name().unwrap().to_string_lossy();
        let short_name: String = if fname.len() > 28 {
            format!("..{}", &fname[fname.len() - 26..])
        } else {
            fname.to_string()
        };

        print!("{:<30} {:>8.2}", short_name, duration);
        for results in &all_results {
            if let Some(r) = results.get(file_idx) {
                print!(
                    " | {:>8.3}s {:>7.1}x",
                    r.inference_duration.as_secs_f32(),
                    r.rtf()
                );
            } else {
                print!(" | {:>8} {:>8}", "-", "-");
            }
        }
        println!();
    }

    // Totals
    println!("{}", "-".repeat(30 + 9 + all_results.len() * 21));
    print!("{:<30} {:>8.2}", "TOTAL", total_audio_s);
    for results in &all_results {
        let total_inf: f32 = results
            .iter()
            .map(|r| r.inference_duration.as_secs_f32())
            .sum();
        let total_audio: f32 = results.iter().map(|r| r.audio_duration_s).sum();
        print!(
            " | {:>8.3}s {:>7.1}x",
            total_inf,
            total_audio / total_inf
        );
    }
    println!();

    // Speedup comparison
    if all_results.len() >= 2 {
        let cpu_total: f32 = all_results[0]
            .iter()
            .map(|r| r.inference_duration.as_secs_f32())
            .sum();
        for results in &all_results[1..] {
            let other_total: f32 = results
                .iter()
                .map(|r| r.inference_duration.as_secs_f32())
                .sum();
            if let Some(r) = results.first() {
                println!(
                    "\n{} speedup vs CPU: {:.2}x",
                    r.provider,
                    cpu_total / other_total
                );
            }
        }
    }

    // Print a sample transcript from the first file
    if let Some(results) = all_results.first() {
        if let Some(r) = results.first() {
            println!("\nSample transcript ({}, {}):", r.provider, r.file);
            let preview: String = r.transcript.chars().take(200).collect();
            println!(
                "  {}{}",
                preview,
                if r.transcript.len() > 200 { "..." } else { "" }
            );
        }
    }

    Ok(())
}
