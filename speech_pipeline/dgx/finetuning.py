"""DGX GPU-accelerated Fine-Tuning Engine for Audio Diarization & ASR.

STRICT SCIENTIFIC INTEGRITY ENFORCEMENT:
- Training and validation MUST use original frozen cohorts (outputs/validation/cohorts_v1/manifest.json).
- NEVER fine-tune, adapt, or validate on external benchmarks (ENNI, LUCID, DementiaBank).
"""
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from speech_pipeline import ROOT
from speech_pipeline.cohorts import load_split, read_manifest, verify
from speech_pipeline.dgx.device_utils import (
    clear_gpu_memory,
    optimize_cuda_env,
    resolve_device,
    resolve_dtype,
)


def verify_training_cohort_safety(dataset_name: str, split_name: str):
    """Enforce absolute mandate: never train or validate on external benchmark datasets."""
    forbidden = ("enni", "lucid", "dementiabank")
    ds_clean = dataset_name.lower().strip()
    if any(f in ds_clean for f in forbidden):
        raise PermissionError(
            f"CRITICAL MANDATE VIOLATION: '{dataset_name}' is a strictly held-out zero-shot external test benchmark. "
            "Never use external datasets for model training, gradient updates, or hyperparameter selection."
        )
    if split_name.lower() not in ("train", "eval", "validation"):
        raise ValueError(f"Invalid split for training/validation: {split_name}")


class AudioSupervisionDataset(Dataset):
    """Dataset of audited audio windows and ground-truth CHAT speaker/text supervision."""
    
    def __init__(self, records: List[Dict[str, Any]], audio_dir: Path):
        self.records = records
        self.audio_dir = Path(audio_dir)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.records[idx]


def setup_lora_model(model: nn.Module, r: int = 16, lora_alpha: int = 32, lora_dropout: float = 0.05):
    """Configure Low-Rank Adaptation (LoRA) for parameter-efficient fine-tuning on DGX."""
    try:
        from peft import LoraConfig, TaskType, get_peft_model
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
        )
        lora_model = get_peft_model(model, peft_config)
        trainable_params, all_param = lora_model.get_nb_trainable_parameters()
        print(f"[+] LoRA initialized: {trainable_params:,} / {all_param:,} params ({100 * trainable_params / all_param:.2f}%)", flush=True)
        return lora_model
    except ImportError:
        print("[!] `peft` not installed. Falling back to selective linear head fine-tuning.", flush=True)
        # Freeze base parameters, keep top layers trainable
        for param in model.parameters():
            param.requires_grad = False
        if hasattr(model, "lm_head"):
            for param in model.lm_head.parameters():
                param.requires_grad = True
        return model


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: torch.device,
    dtype: torch.dtype,
    gradient_accumulation_steps: int = 4
) -> float:
    """Execute one training epoch with mixed precision and gradient accumulation."""
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    
    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler('cuda', enabled=(use_amp and dtype == torch.float16))
    
    for step, batch in enumerate(dataloader, 1):
        # Forward pass inside autocast for DGX Tensor Core acceleration
        with torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
            input_ids = batch.get("input_ids").to(device)
            labels = batch.get("labels", input_ids).to(device)
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss / gradient_accumulation_steps
            
        if scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()
            
        total_loss += loss.item() * gradient_accumulation_steps
        
        if step % gradient_accumulation_steps == 0:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()
                
    return total_loss / max(1, len(dataloader))


def run_finetuning_session(
    model_name_or_path: str,
    output_dir: Path,
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-5,
    use_lora: bool = True,
    device: str = "auto",
    dtype: Optional[str] = None
) -> Dict[str, Any]:
    """Orchestrate DGX fine-tuning session under strict cohort isolation."""
    dev = resolve_device(device)
    target_dtype = resolve_dtype(dev, dtype)
    optimize_cuda_env()
    
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[*] Starting fine-tuning session for {model_name_or_path} on {dev} ({target_dtype})...", flush=True)
    
    # Verify cohorts
    verify_training_cohort_safety("ocsc", "train")
    verify()
    train_records = load_split("train")
    eval_records = load_split("validation")
    print(f"[+] Loaded verified training cohort: {len(train_records)} sections across {len(set(r['participant'] for r in train_records))} children.", flush=True)
    print(f"[+] Loaded validation cohort: {len(eval_records)} sections.", flush=True)
    
    return {
        "status": "configured",
        "model": model_name_or_path,
        "device": str(dev),
        "dtype": str(target_dtype),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "use_lora": use_lora,
        "train_sections": len(train_records),
        "eval_sections": len(eval_records),
        "output_directory": str(output_dir)
    }
