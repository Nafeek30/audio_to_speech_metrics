#!/usr/bin/env python3
"""CLI for Fine-Tuning Audio Diarization & ASR Models on NVIDIA DGX.

Supports:
- MOSS-Transcribe-Diarize (0.9B) with LoRA or selective fine-tuning.
- Whisper (small / large-v3).

Enforces strict participant-independent cohort isolation on original training sets.
"""
import argparse
import json
from pathlib import Path
import sys

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from speech_pipeline.dgx.device_utils import get_platform_info
from speech_pipeline.dgx.finetuning import run_finetuning_session


def main():
    parser = argparse.ArgumentParser(description="NVIDIA DGX Audio Model Fine-Tuning")
    parser.add_argument("--model", type=str, default="outputs/runtime/moss_audio_v1/base", help="Base model directory or Hugging Face ID.")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Destination directory for fine-tuned checkpoints.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of fine-tuning epochs.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per GPU.")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--use-lora", action="store_true", default=True, help="Use Low-Rank Adaptation (LoRA).")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto", help="Compute device.")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default=None, help="Precision override.")
    
    args = parser.parse_args()
    
    info = get_platform_info()
    print("=" * 68)
    print("          NVIDIA DGX AUDIO FINE-TUNING SESSION          ")
    print("=" * 68)
    print(f"Host Device:      {info['device_name']} (CUDA: {info['cuda_available']})")
    print(f"Base Model:       {args.model}")
    print(f"Epochs:           {args.epochs} | Batch Size: {args.batch_size} | LR: {args.lr}")
    print(f"LoRA PEFT:        {args.use_lora}")
    print("=" * 68, flush=True)
    
    session = run_finetuning_session(
        model_name_or_path=args.model,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        use_lora=args.use_lora,
        device=args.device,
        dtype=args.dtype
    )
    
    manifest_path = Path(args.output) / "finetune_config.json"
    manifest_path.write_text(json.dumps(session, indent=2), encoding="utf-8")
    print(f"[+] Fine-tuning session initialized. Config saved to: {manifest_path}")


if __name__ == "__main__":
    main()
