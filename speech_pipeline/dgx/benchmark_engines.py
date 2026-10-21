"""Unified ASR and Diarization engine wrappers optimized for DGX GPU inference.

Supports:
1. Whisper (openai/whisper-small, large-v3)
2. MOSS-Transcribe-Diarize (0.9B end-to-end diarized ASR)
3. Alibaba SenseVoiceSmall + CAM++ (FunASR)
4. NVIDIA NeMo (FastConformer / Sortformer)
"""
import abc
import json
import math
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional

import torch

from speech_pipeline.dgx.device_utils import (
    clear_gpu_memory,
    optimize_cuda_env,
    resolve_device,
    resolve_dtype,
)


class BaseAudioEngine(abc.ABC):
    """Abstract base class for audio transcription and diarization engines."""
    
    def __init__(self, model_name_or_path: str, device: str = "auto", dtype: Optional[str] = None):
        self.model_identifier = model_name_or_path
        self.device = resolve_device(device)
        self.dtype = resolve_dtype(self.device, dtype)
        optimize_cuda_env()

    @abc.abstractmethod
    def transcribe_and_diarize(self, audio_path: Path) -> Dict[str, Any]:
        """Transcribe and diarize an audio file into standardized speaker segments."""
        pass


class WhisperEngine(BaseAudioEngine):
    """DGX-accelerated Whisper transcription engine."""
    
    def __init__(self, model_name_or_path: str = "openai/whisper-small", device: str = "auto", dtype: Optional[str] = None):
        super().__init__(model_name_or_path, device, dtype)
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
        
        torch_dtype = self.dtype
        # MPS doesn't support bfloat16 for Whisper in some versions
        if self.device.type == "mps" and torch_dtype == torch.bfloat16:
            torch_dtype = torch.float16
            
        print(f"[*] Initializing Whisper ({self.model_identifier}) on {self.device} ({torch_dtype})...", flush=True)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_identifier,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True
        ).to(self.device)
        
        processor = AutoProcessor.from_pretrained(self.model_identifier)
        
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=self.device,
            chunk_length_s=30,
            batch_size=8 if self.device.type == "cuda" else 1,
            return_timestamps=True
        )

    def transcribe_and_diarize(self, audio_path: Path) -> Dict[str, Any]:
        start_time = time.monotonic()
        audio_path = Path(audio_path).resolve()
        
        # Whisper standard outputs chunks with timestamps
        res = self.pipe(
            str(audio_path),
            generate_kwargs={"task": "transcribe", "language": "en"}
        )
        runtime = time.monotonic() - start_time
        
        # Standardize output segments (single speaker default without external diarizer)
        raw_chunks = res.get("chunks", [])
        segments = []
        for idx, chunk in enumerate(raw_chunks, 1):
            ts = chunk.get("timestamp", (0.0, 0.0))
            t_start = ts[0] if (ts and ts[0] is not None) else 0.0
            t_end = ts[1] if (ts and len(ts) > 1 and ts[1] is not None) else t_start
            text_str = chunk.get("text", "").strip()
            if text_str:
                segments.append({
                    "segment_id": idx,
                    "speaker": "SPEAKER_00",
                    "text": text_str,
                    "start": float(t_start),
                    "end": float(t_end)
                })
        
        # Fallback if chunks was empty or not populated by pipeline
        if not segments and res.get("text", "").strip():
            segments.append({
                "segment_id": 1,
                "speaker": "SPEAKER_00",
                "text": res["text"].strip(),
                "start": 0.0,
                "end": runtime
            })
            
        return {
            "engine": "whisper",
            "model": self.model_identifier,
            "device": str(self.device),
            "dtype": str(self.dtype),
            "audio_file": str(audio_path),
            "runtime_seconds": runtime,
            "text": res.get("text", "").strip(),
            "segments": segments,
        }


class MossDiarizeEngine(BaseAudioEngine):
    """DGX-accelerated MOSS-Transcribe-Diarize engine."""
    
    SEGMENT_REGEX = re.compile(r'\[(\d+(?:\.\d+)?)\]\[(S\d+)\](.*?)\[(\d+(?:\.\d+)?)\]', re.S)

    def __init__(self, model_dir: Path, device: str = "auto", dtype: Optional[str] = None):
        super().__init__(str(model_dir), device, dtype)
        self.model_dir = Path(model_dir).resolve()
        from transformers import AutoModelForCausalLM, AutoProcessor
        
        print(f"[*] Initializing MOSS ({self.model_dir.name}) on {self.device} ({self.dtype})...", flush=True)
        # Select SDPA attention for acceleration on CUDA/MPS
        attn_impl = "sdpa" if hasattr(torch.nn.functional, "scaled_dot_product_attention") else "eager"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.model_dir),
            trust_remote_code=True,
            local_files_only=True,
            dtype=self.dtype,
            attn_implementation=attn_impl
        ).to(self.device).eval()
        
        self.processor = AutoProcessor.from_pretrained(
            str(self.model_dir),
            trust_remote_code=True,
            local_files_only=True
        )

    def transcribe_and_diarize(self, audio_path: Path, max_new_tokens: int = 8192) -> Dict[str, Any]:
        from moss_transcribe_diarize.inference_utils import (
            build_transcription_messages,
            generate_transcription,
        )
        
        start_time = time.monotonic()
        audio_path = Path(audio_path).resolve()
        
        messages = build_transcription_messages(str(audio_path))
        gen_res = generate_transcription(
            self.model,
            self.processor,
            messages,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            device=self.device,
            dtype=self.dtype
        )
        runtime = time.monotonic() - start_time
        raw_text = gen_res.get("text", "")
        
        # Parse MOSS [start][speaker] text [end]
        segments = []
        for m in self.SEGMENT_REGEX.finditer(raw_text):
            t_start, spk, spoken, t_end = float(m[1]), m[2], m[3].strip(), float(m[4])
            if spoken:
                segments.append({
                    "segment_id": len(segments) + 1,
                    "speaker": spk,
                    "text": spoken,
                    "start": t_start,
                    "end": t_end
                })
                
        return {
            "engine": "moss_transcribe_diarize",
            "model": str(self.model_dir),
            "device": str(self.device),
            "dtype": str(self.dtype),
            "audio_file": str(audio_path),
            "runtime_seconds": runtime,
            "text": " ".join(s["text"] for s in segments),
            "segments": segments,
            "finish_reason": gen_res.get("finish_reason", "eos")
        }


class SenseVoiceEngine(BaseAudioEngine):
    """DGX-accelerated FunASR SenseVoiceSmall + CAM++ diarization engine."""
    
    def __init__(self, device: str = "auto", dtype: Optional[str] = None):
        super().__init__("iic/SenseVoiceSmall", device, dtype)
        from funasr import AutoModel
        print(f"[*] Initializing SenseVoice + CAM++ on {self.device}...", flush=True)
        # FunASR accepts device string like 'cuda:0', 'mps', 'cpu'
        dev_str = "cuda:0" if self.device.type == "cuda" else str(self.device)
        self.model = AutoModel(
            model="iic/SenseVoiceSmall",
            spk_model="iic/speech_campplus_sv_zh-cn_16k-common",
            device=dev_str,
            disable_update=True
        )

    def transcribe_and_diarize(self, audio_path: Path) -> Dict[str, Any]:
        start_time = time.monotonic()
        audio_path = Path(audio_path).resolve()
        
        res = self.model.generate(
            input=str(audio_path),
            batch_size_s=300,
            hotword="",
            merge_vad=True,
            merge_length_s=15
        )
        runtime = time.monotonic() - start_time
        
        segments = []
        full_text = []
        if isinstance(res, list) and res:
            item = res[0]
            text = item.get("text", "")
            full_text.append(text)
            sentence_info = item.get("sentence_info", [])
            for idx, s in enumerate(sentence_info, 1):
                segments.append({
                    "segment_id": idx,
                    "speaker": f"SPEAKER_{s.get('spk', 0):02d}",
                    "text": s.get("text", "").strip(),
                    "start": float(s.get("start", 0) / 1000.0),
                    "end": float(s.get("end", 0) / 1000.0)
                })
        
        return {
            "engine": "sensevoice_campp",
            "model": self.model_identifier,
            "device": str(self.device),
            "audio_file": str(audio_path),
            "runtime_seconds": runtime,
            "text": " ".join(full_text),
            "segments": segments
        }


class NeMoEngine(BaseAudioEngine):
    """DGX-accelerated NVIDIA NeMo FastConformer transcription and diarization engine."""
    
    def __init__(self, model_name_or_path: str = "nvidia/stt_en_fastconformer_transducer_large", device: str = "auto", dtype: Optional[str] = None):
        super().__init__(model_name_or_path, device, dtype)
        import nemo.collections.asr as nemo_asr
        print(f"[*] Initializing NVIDIA NeMo ({self.model_identifier}) on {self.device}...", flush=True)
        try:
            self.model = nemo_asr.models.ASRModel.from_pretrained(model_name=self.model_identifier)
        except Exception:
            try:
                self.model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_name=self.model_identifier)
            except Exception:
                self.model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name=self.model_identifier)
        self.model = self.model.to(self.device).eval()

    def transcribe_and_diarize(self, audio_path: Path) -> Dict[str, Any]:
        start_time = time.monotonic()
        audio_path = Path(audio_path).resolve()
        
        transcriptions = self.model.transcribe([str(audio_path)])
        runtime = time.monotonic() - start_time
        
        text = transcriptions[0] if isinstance(transcriptions, list) and transcriptions else str(transcriptions)
        if hasattr(text, 'text'):
            text = text.text
        text = str(text).strip()
        
        segments = [{
            "segment_id": 1,
            "speaker": "SPEAKER_00",
            "text": text,
            "start": 0.0,
            "end": runtime
        }]
        
        return {
            "engine": "nemo_fastconformer",
            "model": self.model_identifier,
            "device": str(self.device),
            "audio_file": str(audio_path),
            "runtime_seconds": runtime,
            "text": text,
            "segments": segments
        }


def get_engine(engine_name: str, model_path_or_name: Optional[str] = None, device: str = "auto", dtype: Optional[str] = None) -> BaseAudioEngine:
    """Factory helper to load requested engine."""
    eng = engine_name.lower().strip()
    if eng == "whisper":
        target = model_path_or_name or "openai/whisper-small"
        return WhisperEngine(target, device=device, dtype=dtype)
    elif eng in ("moss", "moss_diarize"):
        target = Path(model_path_or_name) if model_path_or_name else Path("outputs/runtime/moss_audio_v1/base")
        return MossDiarizeEngine(target, device=device, dtype=dtype)
    elif eng in ("sensevoice", "funasr"):
        return SenseVoiceEngine(device=device, dtype=dtype)
    elif eng in ("nemo", "fastconformer"):
        target = model_path_or_name or "nvidia/stt_en_fastconformer_transducer_large"
        return NeMoEngine(target, device=device, dtype=dtype)
    else:
        raise ValueError(f"Unknown engine: {engine_name}. Choose 'whisper', 'moss', 'sensevoice', or 'nemo'.")
