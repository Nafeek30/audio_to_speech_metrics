#!/usr/bin/env python3
"""Download and reconstruct full-session ENNI audio files from the open Hugging Face mirror.

STRICT SCIENTIFIC MANDATE ENFORCED:
- ENNI is a 100% held-out zero-shot external test evaluation benchmark.
- NEVER used for model training, backpropagation, adaptation, or validation/hyperparameter selection.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import time

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import soundfile as sf
from huggingface_hub import HfApi, hf_hub_download

# The 6 representative Phase 1 pilot sessions across age and diagnosis:
PILOT_SESSIONS = [
    "401",  # TD, Age 4;10, Story A
    "410",  # TD, Age 5;02, Story A
    "801",  # TD, Age 8;03, Story B
    "406",  # SLI, Age 4;11, Story A
    "602",  # SLI, Age 6;04, Story A
    "853",  # SLI, Age 8;06, Story B
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def reconstruct_session_wav(df: pd.DataFrame, out_path: Path, sample_rate: int = 16000) -> float:
    """Reconstruct an unbroken continuous session timeline WAV from parquet audio chunks."""
    max_ms = int(df['end_ms'].max())
    total_samples = int(np.ceil((max_ms / 1000.0) * sample_rate)) + sample_rate
    session_audio = np.zeros(total_samples, dtype=np.float32)
    
    for _, row in df.iterrows():
        start_ms = row['start_ms']
        start_sample = int((start_ms / 1000.0) * sample_rate)
        audio_dict = row['audio']
        if isinstance(audio_dict, dict) and 'bytes' in audio_dict:
            chunk, sr = sf.read(io.BytesIO(audio_dict['bytes']))
            if sr != sample_rate:
                import librosa
                chunk = librosa.resample(chunk, orig_sr=sr, target_sr=sample_rate)
            end_sample = min(start_sample + len(chunk), total_samples)
            session_audio[start_sample:end_sample] = chunk[:end_sample - start_sample]
            
    # Trim trailing silence past max_ms + 0.5s
    cutoff = min(total_samples, int(((max_ms + 500) / 1000.0) * sample_rate))
    trimmed = session_audio[:cutoff]
    
    sf.write(str(out_path), trimmed, sample_rate)
    return len(trimmed) / sample_rate


def download_child_session(child_id: str, dest_dir: Path, hf_repo: str = "TomRoma/ENNI_dataset") -> dict:
    """Download parquet for child and build WAV file."""
    child_id = str(child_id).strip()
    parquet_filename = f"data/{child_id}-00000-of-00001.parquet"
    wav_path = dest_dir / f"{child_id}.wav"
    
    if wav_path.exists():
        info = sf.info(str(wav_path))
        return {
            "child_id": child_id,
            "status": "cached",
            "wav_file": str(wav_path),
            "sha256": sha256_file(wav_path),
            "duration_seconds": info.duration
        }
        
    print(f"[*] Downloading parquet for child {child_id} from {hf_repo}...", flush=True)
    local_parquet = hf_hub_download(
        repo_id=hf_repo,
        filename=parquet_filename,
        repo_type="dataset"
    )
    
    df = pd.read_parquet(local_parquet)
    duration = reconstruct_session_wav(df, wav_path)
    sha = sha256_file(wav_path)
    print(f"[+] Reconstructed {wav_path.name} ({duration:.2f}s, SHA-256: {sha[:12]}...)", flush=True)
    
    return {
        "child_id": child_id,
        "status": "downloaded_and_reconstructed",
        "wav_file": str(wav_path),
        "sha256": sha,
        "duration_seconds": duration,
        "turns": len(df)
    }


def main():
    parser = argparse.ArgumentParser(
        description="Download and reconstruct ENNI continuous audio files from open mirror."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pilot", action="store_true", help="Download only the 6 representative Phase 1 pilot sessions (401, 410, 801, 406, 602, 853).")
    group.add_argument("--all", action="store_true", help="Download all 350+ ENNI sessions.")
    group.add_argument("--sessions", type=str, help="Comma-separated child IDs (e.g., '401,410,801').")
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "dataset/public_external/enni/audio",
        help="Destination directory for audio WAV files."
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="TomRoma/ENNI_dataset",
        help="Hugging Face repository ID."
    )
    
    args = parser.parse_args()
    
    print("=" * 68)
    print("        ENNI EXTERNAL TEST BENCHMARK AUDIO ACQUISITION        ")
    print("=" * 68)
    print("CRITICAL MANDATE: ENNI is strictly held-out zero-shot test only.")
    print("No ENNI audio or transcript data may ever enter training or CV.")
    print("=" * 68, flush=True)
    
    dest_dir = args.output_dir.resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    if args.pilot:
        target_sessions = PILOT_SESSIONS
    elif args.sessions:
        target_sessions = [s.strip() for s in args.sessions.split(",") if s.strip()]
    else:
        api = HfApi()
        all_files = api.list_repo_files(args.repo_id, repo_type="dataset")
        target_sessions = []
        for f in all_files:
            if f.startswith("data/") and f.endswith("-00000-of-00001.parquet"):
                cid = f.replace("data/", "").replace("-00000-of-00001.parquet", "")
                target_sessions.append(cid)
        target_sessions = sorted(target_sessions)
        
    print(f"[*] Queued {len(target_sessions)} ENNI sessions for acquisition in {dest_dir}", flush=True)
    
    manifest_entries = []
    t_start = time.monotonic()
    for idx, cid in enumerate(target_sessions, 1):
        print(f"[{idx}/{len(target_sessions)}] Processing child {cid}...", flush=True)
        try:
            entry = download_child_session(cid, dest_dir, hf_repo=args.repo_id)
            manifest_entries.append(entry)
        except Exception as e:
            print(f"[-] Error downloading child {cid}: {e}", flush=True)
            manifest_entries.append({"child_id": cid, "status": "failed", "error": str(e)})

    manifest_path = dest_dir / "manifest.json"
    manifest_payload = {
        "dataset": "ENNI",
        "role": "held_out_zero_shot_external_test_benchmark_only",
        "acquisition_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_sessions": len(manifest_entries),
        "completed": sum(1 for e in manifest_entries if e.get("status") in ("cached", "downloaded_and_reconstructed")),
        "entries": manifest_entries
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
    
    print("\n" + "=" * 68)
    print(f"Acquisition complete in {time.monotonic() - t_start:.2f}s!")
    print(f"Verified Audio WAVs: {manifest_payload['completed']} / {len(manifest_entries)}")
    print(f"Audio Directory:     {dest_dir}")
    print(f"Manifest File:       {manifest_path}")
    print("=" * 68)


if __name__ == "__main__":
    main()
