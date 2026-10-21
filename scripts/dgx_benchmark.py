#!/usr/bin/env python3
"""DGX GPU-accelerated Benchmark Runner for ASR & Speaker Diarization Engines.

Supports evaluating:
1. Whisper (openai/whisper-small or whisper-large-v3)
2. MOSS-Transcribe-Diarize (0.9B)
3. Alibaba SenseVoiceSmall + CAM++

Works natively on NVIDIA DGX (CUDA), Apple Silicon (MPS), or distributed with PySpark.
"""
import argparse
import json
from pathlib import Path
import sys
import time

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from speech_pipeline.dgx.benchmark_engines import get_engine
from speech_pipeline.dgx.device_utils import (
    clear_gpu_memory,
    get_platform_info,
    resolve_device,
    resolve_dtype,
)
from speech_pipeline.dgx.evaluation import evaluate_3tier
from speech_pipeline.dgx.spark_runner import is_spark_available, run_local_dgx_batch, run_spark_batch


def print_system_banner():
    info = get_platform_info()
    print("=" * 68)
    print("       NVIDIA DGX / SPARK AUDIO BENCHMARK RUNNER       ")
    print("=" * 68)
    print(f"Host OS:          {info['os']} ({info['os_release']})")
    print(f"Compute Device:   {info['device_name']} (CUDA: {info['cuda_available']})")
    print(f"BF16 Supported:   {info['bf16_supported']}")
    print(f"PyTorch Version:  {info['torch_version']}")
    print(f"PySpark Present:  {is_spark_available()}")
    print("=" * 68, flush=True)


def main():
    parser = argparse.ArgumentParser(description="DGX & Spark Audio Benchmark Runner")
    parser.add_argument("--audio", nargs="+", type=Path, help="One or more audio files to benchmark.")
    parser.add_argument("--audio-dir", type=Path, help="Directory containing audio files.")
    parser.add_argument("--engine", choices=["whisper", "moss", "sensevoice", "all"], default="whisper", help="ASR/Diarization engine.")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output destination folder.")
    parser.add_argument("--model-path", type=str, default=None, help="Custom local model weights path or HuggingFace ID.")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto", help="Compute device.")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default=None, help="Precision override.")
    parser.add_argument("--use-spark", action="store_true", help="Distribute batch inference using PySpark if available.")
    
    args = parser.parse_args()
    print_system_banner()
    
    # Collect audio files
    audio_paths = []
    if args.audio:
        audio_paths.extend([p.resolve() for p in args.audio if p.exists()])
    if args.audio_dir and args.audio_dir.exists():
        for ext in ("*.wav", "*.mp3", "*.flac", "*.m4a"):
            audio_paths.extend(sorted(args.audio_dir.glob(ext)))
            
    if not audio_paths:
        parser.error("No valid audio files found. Specify --audio <file...> or --audio-dir <dir>.")
        
    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    
    engines_to_run = ["whisper", "moss", "sensevoice"] if args.engine == "all" else [args.engine]
    
    overall_summary = {}
    for eng in engines_to_run:
        print(f"\n[>>>] Benchmarking Engine: {eng.upper()}", flush=True)
        t_start = time.monotonic()
        
        if args.use_spark and is_spark_available():
            results = run_spark_batch(
                audio_paths,
                engine_name=eng,
                output_dir=out_dir / eng,
                model_path_or_name=args.model_path,
                device=args.device
            )
        else:
            if args.use_spark and not is_spark_available():
                print("[!] PySpark not detected. Falling back to high-throughput local DGX batching.", flush=True)
            results = run_local_dgx_batch(
                audio_paths,
                engine_name=eng,
                output_dir=out_dir / eng,
                model_path_or_name=args.model_path,
                device=args.device
            )
            
        elapsed = time.monotonic() - t_start
        overall_summary[eng] = {
            "processed_recordings": len(results),
            "total_runtime_seconds": elapsed,
            "average_seconds_per_recording": elapsed / max(1, len(results))
        }
        clear_gpu_memory()

    summary_file = out_dir / "benchmark_summary.json"
    summary_file.write_text(json.dumps(overall_summary, indent=2), encoding="utf-8")
    
    print("\n" + "=" * 68)
    print("               BENCHMARK EXECUTION COMPLETE                 ")
    print("=" * 68)
    for eng, stats in overall_summary.items():
        print(f"Engine: {eng.upper():<12} | Files: {stats['processed_recordings']:<4} | Runtime: {stats['total_runtime_seconds']:.2f}s ({stats['average_seconds_per_recording']:.2f}s/file)")
    print(f"Summary Artifact: {summary_file}")
    print("=" * 68)


if __name__ == "__main__":
    main()
