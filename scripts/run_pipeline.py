#!/usr/bin/env python3
"""Unified CLI for Audio Transcription, Clinical Utterance Segmentation, and Speech Metrics.

Supports three operational modes:
1. end-to-end: Audio -> Transcription (Whisper ASR) -> C-unit Segmentation -> Clinical LSA Metrics.
2. transcribe: Audio -> Raw Diarized/Timestamped Transcript.
3. metrics:    Transcript (Text or JSON) -> C-unit Segmentation -> Clinical LSA Metrics.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from speech_pipeline import utterances as auto
from speech_pipeline.transcript import run as run_transcript_analysis


def print_summary_card(result: dict, out_dir: Path):
    """Print an executive clinical summary card and write summary.txt."""
    chat_m = result.get('chat_metrics', {}).get('values', {})
    lex_m = result.get('lexical_metrics', {})
    segmenter_info = result.get('segmenter', {})
    
    ntu = chat_m.get('total_utterances') or len(result.get('utterances', []))
    ntw = chat_m.get('TNW') or chat_m.get('mlu_words')
    ndw = chat_m.get('NDW')
    ttr = chat_m.get('TTR')
    mluw = chat_m.get('MLUw')
    sd = chat_m.get('length_sd')
    
    mattr = lex_m.get('mattr50_aw', chat_m.get('MATTR'))
    hdd = lex_m.get('hdd42_aw')
    mtld = lex_m.get('mtld_original_aw')
    
    lines = [
        "=" * 64,
        "              SPEECH PIPELINE ANALYSIS REPORT               ",
        "=" * 64,
        f"Input Source:       {result.get('input', 'Unknown')}",
        f"Execution Time:     {result.get('run_utc', datetime.now(timezone.utc).isoformat())}",
        f"Segmenter Model:    {segmenter_info.get('name', 'rules')} ({segmenter_info.get('kind', 'rules')})",
        f"Context Mode:       {result.get('segmentation_context', 'paragraph_ml')}",
        "-" * 64,
        "--- CLINICAL LANGUAGE SAMPLE ANALYSIS (LSA) METRICS ---",
        f"  Total Utterances (NTU):      {ntu if ntu is not None else 'N/A'}",
        f"  Total Words (NTW/TNW):       {ntw if ntw is not None else 'N/A'}",
        f"  Number of Diff Words (NDW):  {ndw if ndw is not None else 'N/A'}",
        f"  Type-Token Ratio (TTR):      {f'{ttr:.3f}' if ttr is not None else 'N/A'}",
        f"  Mean Length of Utt (MLUw):   {f'{mluw:.2f} words' if mluw is not None else 'N/A'}",
        f"  Utterance Length SD:         {f'{sd:.2f}' if sd is not None else 'N/A'}",
        "-" * 64,
        "--- LEXICAL DIVERSITY & DENSITY (TAALED) ---",
        f"  Moving-Average TTR (MATTR50): {f'{mattr:.3f}' if mattr is not None else 'N/A'}",
        f"  Vocd / HD-D (HDD42):          {f'{hdd:.3f}' if hdd is not None else 'N/A'}",
        f"  Measure of Textual Div (MTLD):{f'{mtld:.2f}' if mtld is not None else 'N/A'}",
        "=" * 64,
        f"Output Artifacts Saved to: {out_dir.resolve()}",
        f"  - CHAT File (CLAN-compatible): {out_dir / 'transcript.cha'}",
        f"  - Full Structured Metrics:     {out_dir / 'results.json'}",
        f"  - Summary Report:              {out_dir / 'summary.txt'}",
        "=" * 64,
    ]
    summary_text = "\n".join(lines)
    print(summary_text, flush=True)
    
    summary_file = out_dir / 'summary.txt'
    summary_file.write_text(summary_text + "\n", encoding='utf-8')


def transcribe_audio(audio_path: Path, model_name: str = "openai/whisper-small", device: str = "cpu") -> dict:
    """Transcribe an audio file using Hugging Face Whisper pipeline."""
    from transformers import pipeline
    print(f"[*] Loading ASR engine ({model_name}) on {device}...", flush=True)
    asr = pipeline(
        "automatic-speech-recognition",
        model=model_name,
        device=device,
        generate_kwargs={"task": "transcribe", "language": "en"}
    )
    print(f"[*] Transcribing audio: {audio_path.name} ...", flush=True)
    raw_result = asr(str(audio_path), return_timestamps=True)
    return raw_result


def run_transcribe_mode(args):
    """Run transcription only."""
    audio_path = Path(args.audio).resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    
    asr_res = transcribe_audio(audio_path, model_name=args.asr_model, device=args.device)
    raw_text = asr_res.get("text", "").strip()
    
    (out_dir / "transcript.txt").write_text(raw_text + "\n", encoding='utf-8')
    (out_dir / "transcript.json").write_text(json.dumps(asr_res, indent=2) + "\n", encoding='utf-8')
    
    print(f"[+] Transcription complete! Saved to {out_dir}", flush=True)
    print(f"    Recognized Text: {raw_text[:200]}..." if len(raw_text) > 200 else f"    Recognized Text: {raw_text}")


def run_metrics_mode(args):
    """Run utterance segmentation and clinical metrics on an existing transcript."""
    transcript_path = Path(args.transcript).resolve()
    if not transcript_path.exists():
        raise FileNotFoundError(f"Transcript file not found: {transcript_path}")
    
    out_dir = Path(args.output).resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Output directory must be empty to avoid overwriting: {out_dir}")
    
    print(f"[*] Analyzing transcript: {transcript_path.name} ...", flush=True)
    res = run_transcript_analysis(
        transcript_path,
        out_dir,
        segmenter=args.segmenter,
        boundary_margin=args.boundary_margin
    )
    print_summary_card(res, out_dir)


def run_end_to_end_mode(args):
    """Run full audio -> transcription -> C-unit segmentation -> metrics."""
    audio_path = Path(args.audio).resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    out_dir = Path(args.output).resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Output directory must be empty to avoid overwriting: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Transcribe audio
    asr_res = transcribe_audio(audio_path, model_name=args.asr_model, device=args.device)
    raw_text = asr_res.get("text", "").strip()
    
    # Save raw ASR outputs
    (out_dir / "asr_transcript.txt").write_text(raw_text + "\n", encoding='utf-8')
    (out_dir / "asr_transcript.json").write_text(json.dumps(asr_res, indent=2) + "\n", encoding='utf-8')
    
    # 2. Canonical Normalization (strip punctuation, extract continuous words)
    words = auto.WORD.findall(raw_text)
    if not words:
        raise ValueError("No recognizable speech words were found in the audio transcription.")
    norm_text = " ".join(words)
    
    with tempfile.TemporaryDirectory(prefix="speech_pipeline_") as tmp_dir:
        tmp_transcript = Path(tmp_dir) / "normalized_input.txt"
        tmp_transcript.write_text(norm_text, encoding='utf-8')
        
        # 3. Downstream Clinical Utterance Segmentation & LSA Metrics
        print("[*] Segmenting continuous speech stream into C-units & computing clinical metrics...", flush=True)
        res = run_transcript_analysis(
            tmp_transcript,
            out_dir / "analysis",
            segmenter=args.segmenter,
            boundary_margin=args.boundary_margin
        )
        
        # Move primary artifacts to root of out_dir
        analysis_dir = out_dir / "analysis"
        if (analysis_dir / "transcript.cha").exists():
            (analysis_dir / "transcript.cha").replace(out_dir / "transcript.cha")
        if (analysis_dir / "results.json").exists():
            (analysis_dir / "results.json").replace(out_dir / "results.json")
        try:
            analysis_dir.rmdir()
        except OSError:
            pass
        
        res['input'] = str(audio_path)
        print_summary_card(res, out_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Speech Analysis Pipeline: Audio Transcription, Utterance Segmentation, and Clinical Metrics."
    )
    parser.add_argument(
        "--mode",
        choices=["end-to-end", "transcribe", "metrics"],
        default=None,
        help="Operation mode: 'end-to-end' (audio to metrics), 'transcribe' (audio to text), or 'metrics' (text to metrics)."
    )
    parser.add_argument(
        "--audio",
        type=Path,
        help="Path to input audio file (WAV, MP3, FLAC, M4A)."
    )
    parser.add_argument(
        "--transcript",
        type=Path,
        help="Path to input transcript file (.txt for plain text, or .json for structured conversational turns)."
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Path to destination directory for output artifacts."
    )
    parser.add_argument(
        "--segmenter",
        choices=["improved", "rules", "auto"],
        default="improved",
        help="Utterance segmentation model: 'improved' (trained MiniLM transformer), 'rules' (linguistic rule heuristics), or 'auto' (transformer with fallback to rules)."
    )
    parser.add_argument(
        "--boundary-margin",
        type=float,
        default=None,
        help="Optional decision margin override for C-unit boundary classification (e.g., 0.0 or 0.25)."
    )
    parser.add_argument(
        "--asr-model",
        type=str,
        default="openai/whisper-small",
        help="Hugging Face model ID for ASR transcription (default: 'openai/whisper-small')."
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "mps"],
        default="cpu",
        help="Computation device ('cpu', 'cuda', or 'mps')."
    )
    
    args = parser.parse_args()
    
    # Determine mode if not explicitly supplied
    mode = args.mode
    if mode is None:
        if args.audio and not args.transcript:
            mode = "end-to-end"
        elif args.transcript and not args.audio:
            mode = "metrics"
        else:
            parser.error("Please specify --mode (or provide either --audio or --transcript).")
            
    if mode in ["end-to-end", "transcribe"] and not args.audio:
        parser.error(f"Mode '{mode}' requires --audio <path_to_audio_file>.")
    if mode == "metrics" and not args.transcript:
        parser.error("Mode 'metrics' requires --transcript <path_to_transcript_file>.")
        
    if mode == "end-to-end":
        run_end_to_end_mode(args)
    elif mode == "transcribe":
        run_transcribe_mode(args)
    elif mode == "metrics":
        run_metrics_mode(args)


if __name__ == "__main__":
    main()
