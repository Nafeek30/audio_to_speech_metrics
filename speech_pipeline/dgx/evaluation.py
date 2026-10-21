"""3-Tier Evaluation Matrix for Audio ASR/Diarization and Downstream Clinical LSA."""
from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from speech_pipeline import utterances as auto
from speech_pipeline.chat_metrics import analyze as analyze_chat
from speech_pipeline.metrics import clinical_from_predictions
from speech_pipeline.pipeline import resolve_segmenter


def normalize_words(text: str) -> List[str]:
    """Tokenize and normalize words for acoustic comparison."""
    return [w.casefold().replace('’', "'") for w in auto.WORD.findall(text)]


def compute_wer(reference_words: List[str], hypothesis_words: List[str]) -> Dict[str, Any]:
    """Compute standard Word Error Rate (Levenshtein alignment)."""
    _, edits = auto.align(reference_words, hypothesis_words)
    total_ref = len(reference_words)
    error_count = sum(edits.values())
    wer = (100.0 * error_count / total_ref) if total_ref > 0 else 0.0
    return {
        "wer_percent": wer,
        "edits": edits,
        "reference_words": total_ref,
        "hypothesis_words": len(hypothesis_words)
    }


def compute_cpwer(reference_by_speaker: Dict[str, List[str]], hypothesis_by_speaker: Dict[str, List[str]]) -> Dict[str, Any]:
    """Compute Concatenated Permutation Word Error Rate (cpWER) for multi-speaker diarization."""
    refs = sorted(reference_by_speaker.keys())
    hyps = sorted(hypothesis_by_speaker.keys())
    n = max(len(refs), len(hyps))
    
    cost_matrix = np.zeros((n, n), dtype=np.int64)
    edits_lookup = {}
    
    for i in range(n):
        for j in range(n):
            ref_words = reference_by_speaker[refs[i]] if i < len(refs) else []
            hyp_words = hypothesis_by_speaker[hyps[j]] if j < len(hyps) else []
            _, e = auto.align(ref_words, hyp_words)
            cost_matrix[i, j] = sum(e.values())
            edits_lookup[(i, j)] = e
            
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    total_ref_words = sum(len(v) for v in reference_by_speaker.values())
    total_cost = int(cost_matrix[row_ind, col_ind].sum())
    
    assignments = []
    for r, c in zip(row_ind, col_ind):
        ref_spk = refs[r] if r < len(refs) else None
        hyp_spk = hyps[c] if c < len(hyps) else None
        assignments.append({
            "reference_speaker": ref_spk,
            "hypothesis_speaker": hyp_spk,
            "edits": edits_lookup[(r, c)],
            "ref_words": len(reference_by_speaker[ref_spk]) if ref_spk else 0,
            "hyp_words": len(hypothesis_by_speaker[hyp_spk]) if hyp_spk else 0
        })
        
    cpwer = (100.0 * total_cost / total_ref_words) if total_ref_words > 0 else 0.0
    return {
        "cpwer_percent": cpwer,
        "total_edits": total_cost,
        "total_ref_words": total_ref_words,
        "assignments": assignments
    }


def compute_boundary_f1(gold_utterances: List[Dict[str, Any]], pred_utterances: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute precision, recall, and F1 for predicted C-unit boundaries."""
    # Build continuous word stream and check boundary indices
    gold_words = [w for u in gold_utterances for w in auto.WORD.findall(u['text'])]
    pred_words = [w for u in pred_utterances for w in auto.WORD.findall(u['text'])]
    
    gold_boundaries = set()
    cum = 0
    for u in gold_utterances[:-1]:
        cum += len(auto.WORD.findall(u['text']))
        gold_boundaries.add(cum)
        
    pred_boundaries = set()
    cum = 0
    for u in pred_utterances[:-1]:
        cum += len(auto.WORD.findall(u['text']))
        pred_boundaries.add(cum)
        
    true_pos = len(gold_boundaries & pred_boundaries)
    prec = (true_pos / len(pred_boundaries)) if pred_boundaries else 1.0
    rec = (true_pos / len(gold_boundaries)) if gold_boundaries else 1.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    
    return {
        "precision_percent": prec * 100.0,
        "recall_percent": rec * 100.0,
        "f1_percent": f1 * 100.0,
        "gold_boundaries": len(gold_boundaries),
        "pred_boundaries": len(pred_boundaries),
        "matches": true_pos
    }


def evaluate_3tier(
    raw_segments: List[Dict[str, Any]],
    gold_reference_rows: List[Dict[str, Any]],
    target_speaker: str = "CHI",
    segmenter_name: str = "improved"
) -> Dict[str, Any]:
    """Execute complete 3-Tier evaluation on a single session."""
    # 1. Tier 1: Acoustics & Diarization
    ref_by_spk = defaultdict(list)
    for r in gold_reference_rows:
        ref_by_spk[r['speaker']].extend(normalize_words(r['text']))
        
    hyp_by_spk = defaultdict(list)
    for s in raw_segments:
        hyp_by_spk[s['speaker']].extend(normalize_words(s['text']))
        
    cpwer_res = compute_cpwer(ref_by_spk, hyp_by_spk)
    
    # Identify target speaker cluster from assignments
    target_assignment = next(
        (a for a in cpwer_res['assignments'] if a['reference_speaker'] == target_speaker),
        None
    )
    mapped_hyp_speaker = target_assignment['hypothesis_speaker'] if target_assignment else None
    
    tier1 = {
        "cpwer": cpwer_res['cpwer_percent'],
        "target_reference_speaker": target_speaker,
        "mapped_hypothesis_speaker": mapped_hyp_speaker,
        "assignments": cpwer_res['assignments']
    }
    
    # Extract target speech words
    if mapped_hyp_speaker:
        target_words = [w for s in raw_segments if s['speaker'] == mapped_hyp_speaker for w in auto.WORD.findall(s['text'])]
    else:
        target_words = [w for s in raw_segments for w in auto.WORD.findall(s['text'])]
        
    if not target_words:
        return {"tier1": tier1, "error": "No speech words found for target speaker"}
        
    paragraph = " ".join(target_words)
    
    # 2. Tier 2: Downstream C-unit Segmentation
    import spacy
    nlp = spacy.load("en_core_web_sm")
    model = resolve_segmenter(segmenter_name, nlp)
    pred_us = model.segment(paragraph) if model else auto.predict(
        [dict(row=2, speaker='PAR', text=paragraph, start='', end='')], 'PAR', nlp
    )
    
    gold_target_us = [
        {"text": r['text']} for r in gold_reference_rows if r['speaker'] == target_speaker and r['text'].strip()
    ]
    tier2 = compute_boundary_f1(gold_target_us, pred_us)
    
    # 3. Tier 3: Downstream Clinical LSA Metrics
    pred_metrics = clinical_from_predictions(pred_us)['values']
    gold_metrics = clinical_from_predictions(gold_target_us)['values']
    
    tier3 = {}
    for k in ("total_utterances", "mlu_words", "MLUw", "length_sd", "TNW", "NDW", "TTR"):
        p_val = pred_metrics.get(k)
        g_val = gold_metrics.get(k)
        diff = (p_val - g_val) if (p_val is not None and g_val is not None) else None
        ape = (100.0 * abs(diff) / abs(g_val)) if (diff is not None and g_val and g_val != 0) else None
        tier3[k] = {
            "predicted": p_val,
            "gold_reference": g_val,
            "difference": diff,
            "ape_percent": ape
        }
        
    return {
        "tier1_acoustics": tier1,
        "tier2_segmentation": tier2,
        "tier3_clinical_metrics": tier3
    }
