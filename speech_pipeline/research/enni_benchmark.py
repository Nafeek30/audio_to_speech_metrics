"""Benchmark our frozen production pipeline on the unseen public ENNI dataset.

This script performs zero-shot / out-of-domain evaluation on all 361 ENNI CHAT transcripts
(286 typically developing, 75 specific language impairment) without any fine-tuning.
It computes:
1. Utterance/C-unit segmentation Precision, Recall, and F1 (comparing directly with Hou et al., 2026 Table 5).
2. Downstream native CLAN and TAALED metrics (comparing with Hou et al., 2026 Table 9).
"""
import csv
import glob
import json
from pathlib import Path
import re
import sys
import time

import numpy as np
import scipy.stats as stats
import spacy

from speech_pipeline import ROOT, utterances as auto
from speech_pipeline.chat_metrics import analyze
from speech_pipeline.lexical_compat import clean_pauses, normalize_doc, calculate_normalized
from speech_pipeline.pipeline import resolve_segmenter, digest

OUT_DIR = ROOT / 'outputs/results/enni_benchmark'
ENNI_DIR = ROOT / 'dataset/public_external/enni/transcripts'


def clean_words_from_line(line_text):
    """Clean a CHAT line to spoken words, stripping timestamps, mazes, and non-speech."""
    clean = re.sub(r'\d+_\d+', '', line_text)
    clean = re.sub(r'\[[/?\w\s-]+\]', '', clean)
    clean = re.sub(r'&=[^\s<>\[\]]+', '', clean)
    clean = re.sub(r'\(.*?\)', '', clean)
    return auto.WORD.findall(clean)


def parse_enni_file(path):
    """Parse an ENNI CHAT file into participant metadata and clean speaker turns."""
    text = Path(path).read_text(encoding='utf-8-sig')
    
    # Extract metadata from headers
    cohort = 'TD' if '/TD/' in str(path) else ('SLI' if '/SLI/' in str(path) else 'Unknown')
    age = None
    age_m = re.search(r'@ID:\s*eng\|ENNI\|CHI\|([^|]*)\|', text)
    if age_m and age_m[1].strip():
        age = age_m[1].strip()
        
    rows = []
    for line in text.splitlines():
        if line.startswith('*'):
            parts = line.split(':', 1)
            speaker = parts[0][1:].strip()
            content = parts[1] if len(parts) > 1 else ''
            words = clean_words_from_line(content)
            if words:
                rows.append({'speaker': speaker, 'words': words, 'raw': content})
                
    return {
        'path': str(path),
        'filename': Path(path).name,
        'cohort': cohort,
        'age': age,
        'rows': rows
    }


def evaluate_segmentation_boundaries(gold_boundaries, pred_boundaries, total_words):
    """Compute token-level Precision, Recall, and F1 on word gaps."""
    # Boundary gaps are indices 1..total_words-1
    tp = len(gold_boundaries & pred_boundaries)
    fp = len(pred_boundaries - gold_boundaries)
    fn = len(gold_boundaries - pred_boundaries)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    
    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def safe_div(num, den):
    return num / den if den and den != 0 else 0.0


def run_benchmark():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    files = sorted(glob.glob(str(ENNI_DIR / '**/*.cha'), recursive=True))
    if not files:
        raise FileNotFoundError(f'No ENNI .cha files found in {ENNI_DIR}')
        
    print(f'Found {len(files)} ENNI transcript files.')
    print('Loading spaCy model and frozen production segmenter...')
    
    nlp = spacy.load('en_core_web_sm')
    workflow_path = ROOT / 'outputs/results/benchmark/transcript_workflow.json'
    workflow = json.loads(workflow_path.read_text()) if workflow_path.exists() else None
    model_opts = dict(model_directory=workflow['model_directory'], expected_weights_sha256=workflow['weights_sha256']) if workflow and workflow.get('model_directory') else {}
    margin = workflow['margin'] if workflow and 'margin' in workflow else 0.0
    print(f"Using segmenter from: {workflow.get('model_directory', 'default')} with margin {margin}")
    model = resolve_segmenter('improved', nlp, **model_opts)
    
    records = []
    t0 = time.time()
    
    for idx, fpath in enumerate(files, 1):
        parsed = parse_enni_file(fpath)
        rows = parsed['rows']
        
        # Extract child gold utterances
        gold_utts = []
        all_child_words = []
        gold_boundaries = set()
        child_turns = []
        cur_turn_words = []
        
        w_counter = 0
        for r in rows:
            if r['speaker'] == 'CHI':
                gold_utts.append({'text': ' '.join(r['words'])})
                all_child_words.extend(r['words'])
                w_counter += len(r['words'])
                gold_boundaries.add(w_counter)
                cur_turn_words.extend(r['words'])
            else:
                if cur_turn_words:
                    child_turns.append(list(cur_turn_words))
                    cur_turn_words = []
        if cur_turn_words:
            child_turns.append(list(cur_turn_words))
            
        gold_boundaries.discard(len(all_child_words))
        total_child_words = len(all_child_words)
        
        if total_child_words == 0:
            print(f'Skipping empty child speech: {parsed["filename"]}')
            continue
            
        # 1. Unpunctuated paragraph stream (continuous text, matches SATE Table 5 baseline)
        paragraph_text = ' '.join(all_child_words)
        pred_utts_paragraph = model.segment(paragraph_text, boundary_margin=margin)
        
        pred_boundaries_para = set()
        c_idx = 0
        for u in pred_utts_paragraph[:-1]:
            u_words = auto.WORD.findall(u['text'])
            c_idx += len(u_words)
            pred_boundaries_para.add(c_idx)
            
        seg_para = evaluate_segmentation_boundaries(gold_boundaries, pred_boundaries_para, total_child_words)
        
        # 2. Turn-aware stream (matches our production workflow)
        pred_utts_turn = []
        pred_boundaries_turn = set()
        c_turn_idx = 0
        for turn_w in child_turns:
            t_text = ' '.join(turn_w)
            t_preds = model.segment(t_text, boundary_margin=margin)
            pred_utts_turn.extend(t_preds)
            for u in t_preds[:-1]:
                c_turn_idx += len(auto.WORD.findall(u['text']))
                pred_boundaries_turn.add(c_turn_idx)
            c_turn_idx += len(auto.WORD.findall(t_preds[-1]['text']))
            if c_turn_idx < total_child_words:
                pred_boundaries_turn.add(c_turn_idx)
                
        seg_turn = evaluate_segmentation_boundaries(gold_boundaries, pred_boundaries_turn, total_child_words)
        
        # Compute native CLAN metrics on Gold vs Predicted (turn-aware production)
        clan_gold = analyze(gold_utts)['values']
        clan_pred = analyze(pred_utts_turn)['values']
        
        # TAALED lexical metrics
        doc_gold = nlp(' '.join(all_child_words))
        # Note: both gold and pred share the exact same child words
        
        rec = {
            'filename': parsed['filename'],
            'cohort': parsed['cohort'],
            'age': parsed['age'],
            'total_words': total_child_words,
            'gold_utterances': len(gold_utts),
            'pred_utterances_para': len(pred_utts_paragraph),
            'pred_utterances_turn': len(pred_utts_turn),
            
            # Paragraph segmentation
            'para_precision': seg_para['precision'],
            'para_recall': seg_para['recall'],
            'para_f1': seg_para['f1'],
            'para_tp': seg_para['tp'],
            'para_fp': seg_para['fp'],
            'para_fn': seg_para['fn'],
            
            # Turn segmentation
            'turn_precision': seg_turn['precision'],
            'turn_recall': seg_turn['recall'],
            'turn_f1': seg_turn['f1'],
            'turn_tp': seg_turn['tp'],
            'turn_fp': seg_turn['fp'],
            'turn_fn': seg_turn['fn'],
            
            # Downstream CLAN measures (turn-aware model)
            'gold_ntu': clan_gold['total_utterances'],
            'pred_ntu': clan_pred['total_utterances'],
            'ntu_diff': clan_pred['total_utterances'] - clan_gold['total_utterances'],
            'ntu_pct_diff': 100 * (clan_pred['total_utterances'] - clan_gold['total_utterances']) / clan_gold['total_utterances'] if clan_gold['total_utterances'] else 0.0,
            'ntu_ape': 100 * abs(clan_pred['total_utterances'] - clan_gold['total_utterances']) / clan_gold['total_utterances'] if clan_gold['total_utterances'] else 0.0,
            
            'gold_ntw': clan_gold['mlu_words'],
            'pred_ntw': clan_pred['mlu_words'],
            'ntw_diff': clan_pred['mlu_words'] - clan_gold['mlu_words'],
            'ntw_pct_diff': 100 * (clan_pred['mlu_words'] - clan_gold['mlu_words']) / clan_gold['mlu_words'] if clan_gold['mlu_words'] else 0.0,
            'ntw_ape': 100 * abs(clan_pred['mlu_words'] - clan_gold['mlu_words']) / clan_gold['mlu_words'] if clan_gold['mlu_words'] else 0.0,
            
            'gold_ndw': clan_gold['NDW'],
            'pred_ndw': clan_pred['NDW'],
            'ndw_diff': clan_pred['NDW'] - clan_gold['NDW'],
            'ndw_pct_diff': 100 * (clan_pred['NDW'] - clan_gold['NDW']) / clan_gold['NDW'] if clan_gold['NDW'] else 0.0,
            'ndw_ape': 100 * abs(clan_pred['NDW'] - clan_gold['NDW']) / clan_gold['NDW'] if clan_gold['NDW'] else 0.0,
            
            'gold_mluw': clan_gold['MLUw'],
            'pred_mluw': clan_pred['MLUw'],
            'mluw_diff': clan_pred['MLUw'] - clan_gold['MLUw'] if (clan_pred['MLUw'] and clan_gold['MLUw']) else 0.0,
            'mluw_pct_diff': 100 * (clan_pred['MLUw'] - clan_gold['MLUw']) / clan_gold['MLUw'] if (clan_pred['MLUw'] and clan_gold['MLUw']) else 0.0,
            'mluw_ape': 100 * abs(clan_pred['MLUw'] - clan_gold['MLUw']) / clan_gold['MLUw'] if (clan_pred['MLUw'] and clan_gold['MLUw']) else 0.0,
            
            'gold_length_sd': clan_gold['length_sd'],
            'pred_length_sd': clan_pred['length_sd'],
            'length_sd_ape': 100 * abs(clan_pred['length_sd'] - clan_gold['length_sd']) / clan_gold['length_sd'] if (clan_pred['length_sd'] and clan_gold['length_sd']) else 0.0,
            
            'gold_ttr': clan_gold['TTR'],
            'pred_ttr': clan_pred['TTR'],
            'ttr_ape': 100 * abs(clan_pred['TTR'] - clan_gold['TTR']) / clan_gold['TTR'] if (clan_pred['TTR'] and clan_gold['TTR']) else 0.0,
            
            'gold_mattr': clan_gold['MATTR'],
            'pred_mattr': clan_pred['MATTR'],
            'mattr_ape': 100 * abs(clan_pred['MATTR'] - clan_gold['MATTR']) / clan_gold['MATTR'] if (clan_pred['MATTR'] and clan_gold['MATTR']) else None
        }
        records.append(rec)
        
        if idx % 50 == 0 or idx == len(files):
            print(f'Processed {idx}/{len(files)} files ({time.time() - t0:.1f}s)...')
            
    # Write per-child CSV
    csv_path = OUT_DIR / 'per_child_predictions.csv'
    fieldnames = list(records[0].keys())
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
        
    print(f'Wrote per-child detailed results to {csv_path}')
    
    # Compute aggregate summaries by cohort: All, TD, SLI
    cohorts = {'All': records, 'TD': [r for r in records if r['cohort'] == 'TD'], 'SLI': [r for r in records if r['cohort'] == 'SLI']}
    summary = {}
    
    for c_name, c_recs in cohorts.items():
        n = len(c_recs)
        if n == 0:
            continue
            
        # Micro-averaged segmentation F1 (pooling TPs, FPs, FNs across all tokens)
        para_tp = sum(r['para_tp'] for r in c_recs)
        para_fp = sum(r['para_fp'] for r in c_recs)
        para_fn = sum(r['para_fn'] for r in c_recs)
        para_p = safe_div(para_tp, para_tp + para_fp)
        para_r = safe_div(para_tp, para_tp + para_fn)
        para_f1 = safe_div(2 * para_tp, 2 * para_tp + para_fp + para_fn)
        
        turn_tp = sum(r['turn_tp'] for r in c_recs)
        turn_fp = sum(r['turn_fp'] for r in c_recs)
        turn_fn = sum(r['turn_fn'] for r in c_recs)
        turn_p = safe_div(turn_tp, turn_tp + turn_fp)
        turn_r = safe_div(turn_tp, turn_tp + turn_fn)
        turn_f1 = safe_div(2 * turn_tp, 2 * turn_tp + turn_fp + turn_fn)
        
        # Macro-averaged segmentation
        macro_para_f1 = float(np.mean([r['para_f1'] for r in c_recs]))
        macro_turn_f1 = float(np.mean([r['turn_f1'] for r in c_recs]))
        
        # Pearson correlations
        r_ntu = float(stats.pearsonr([r['gold_ntu'] for r in c_recs], [r['pred_ntu'] for r in c_recs])[0])
        r_ntw = float(stats.pearsonr([r['gold_ntw'] for r in c_recs], [r['pred_ntw'] for r in c_recs])[0])
        r_ndw = float(stats.pearsonr([r['gold_ndw'] for r in c_recs], [r['pred_ndw'] for r in c_recs])[0])
        valid_mluw = [r for r in c_recs if r['gold_mluw'] is not None and r['pred_mluw'] is not None]
        r_mluw = float(stats.pearsonr([r['gold_mluw'] for r in valid_mluw], [r['pred_mluw'] for r in valid_mluw])[0])
        
        mattr_apes = [r['mattr_ape'] for r in c_recs if r['mattr_ape'] is not None]
        
        summary[c_name] = {
            'count': n,
            'segmentation_paragraph': {
                'micro_precision': float(para_p),
                'micro_recall': float(para_r),
                'micro_f1': float(para_f1),
                'macro_f1': float(macro_para_f1)
            },
            'segmentation_turn_aware': {
                'micro_precision': float(turn_p),
                'micro_recall': float(turn_r),
                'micro_f1': float(turn_f1),
                'macro_f1': float(macro_turn_f1)
            },
            'downstream_metrics': {
                'total_utterances_ntu': {
                    'pearson_r': r_ntu,
                    'mae': float(np.mean([abs(r['ntu_diff']) for r in c_recs])),
                    'signed_bias_raw': float(np.mean([r['ntu_diff'] for r in c_recs])),
                    'mean_percentage_diff': float(np.mean([r['ntu_pct_diff'] for r in c_recs])),
                    'mape': float(np.mean([r['ntu_ape'] for r in c_recs]))
                },
                'total_words_ntw': {
                    'pearson_r': r_ntw,
                    'mae': float(np.mean([abs(r['ntw_diff']) for r in c_recs])),
                    'signed_bias_raw': float(np.mean([r['ntw_diff'] for r in c_recs])),
                    'mean_percentage_diff': float(np.mean([r['ntw_pct_diff'] for r in c_recs])),
                    'mape': float(np.mean([r['ntw_ape'] for r in c_recs]))
                },
                'different_words_ndw': {
                    'pearson_r': r_ndw,
                    'mae': float(np.mean([abs(r['ndw_diff']) for r in c_recs])),
                    'signed_bias_raw': float(np.mean([r['ndw_diff'] for r in c_recs])),
                    'mean_percentage_diff': float(np.mean([r['ndw_pct_diff'] for r in c_recs])),
                    'mape': float(np.mean([r['ndw_ape'] for r in c_recs]))
                },
                'mlu_in_words_mluw': {
                    'pearson_r': r_mluw,
                    'mae': float(np.mean([abs(r['mluw_diff']) for r in valid_mluw])),
                    'signed_bias_raw': float(np.mean([r['mluw_diff'] for r in valid_mluw])),
                    'mean_percentage_diff': float(np.mean([r['mluw_pct_diff'] for r in valid_mluw])),
                    'mape': float(np.mean([r['mluw_ape'] for r in valid_mluw]))
                },
                'utterance_length_sd': {
                    'mape': float(np.mean([r['length_sd_ape'] for r in c_recs if r['length_sd_ape'] is not None]))
                },
                'ttr': {
                    'mape': float(np.mean([r['ttr_ape'] for r in c_recs]))
                },
                'mattr': {
                    'mape': float(np.mean(mattr_apes)) if mattr_apes else None
                }
            }
        }
        
    summary_path = OUT_DIR / 'metrics_summary.json'
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f'Wrote summary results to {summary_path}')
    
    # Generate direct published comparison table
    comparison_table = [
        {
            'System': 'SaT 3L (Frohmann et al., 2024)',
            'Domain_Trained_on_ENNI': 'No (Zero-shot)',
            'Seg_Precision': '74.2%',
            'Seg_Recall': '47.7%',
            'Seg_F1': '58.1%',
            'NTU_r': '-',
            'NTU_MAE': '-',
            'NTU_Bias': '-',
            'MLU_r': '-',
            'MLU_MAE': '-',
            'MLU_Bias': '-'
        },
        {
            'System': 'Batchalign 0.10.4 (Liu et al., 2023)',
            'Domain_Trained_on_ENNI': 'No (Zero-shot)',
            'Seg_Precision': '84.3%',
            'Seg_Recall': '93.0%',
            'Seg_F1': '88.4%',
            'NTU_r': '-',
            'NTU_MAE': '-',
            'NTU_Bias': '-',
            'MLU_r': '-',
            'MLU_MAE': '-',
            'MLU_Bias': '-'
        },
        {
            'System': 'SATE Pipeline (Hou et al., 2026)',
            'Domain_Trained_on_ENNI': 'Yes (70% fine-tuned)',
            'Seg_Precision': '93.9%',
            'Seg_Recall': '94.3%',
            'Seg_F1': '94.1%',
            'NTU_r': '.92',
            'NTU_MAE': '19.88 utts',
            'NTU_Bias': '+19.64%',
            'MLU_r': '.78 (MLUm)',
            'MLU_MAE': '1.89 morph',
            'MLU_Bias': '-25.69%'
        },
        {
            'System': 'Our Pipeline (Paragraph Mode, Zero-Shot)',
            'Domain_Trained_on_ENNI': 'No (Zero-shot)',
            'Seg_Precision': f"{summary['All']['segmentation_paragraph']['micro_precision']*100:.1f}%",
            'Seg_Recall': f"{summary['All']['segmentation_paragraph']['micro_recall']*100:.1f}%",
            'Seg_F1': f"{summary['All']['segmentation_paragraph']['micro_f1']*100:.1f}%",
            'NTU_r': f"{summary['All']['downstream_metrics']['total_utterances_ntu']['pearson_r']:.2f}",
            'NTU_MAE': f"{summary['All']['downstream_metrics']['total_utterances_ntu']['mae']:.2f} utts",
            'NTU_Bias': f"{summary['All']['downstream_metrics']['total_utterances_ntu']['mean_percentage_diff']:+.2f}%",
            'MLU_r': f"{summary['All']['downstream_metrics']['mlu_in_words_mluw']['pearson_r']:.2f} (MLUw)",
            'MLU_MAE': f"{summary['All']['downstream_metrics']['mlu_in_words_mluw']['mae']:.2f} words",
            'MLU_Bias': f"{summary['All']['downstream_metrics']['mlu_in_words_mluw']['mean_percentage_diff']:+.2f}%"
        },
        {
            'System': 'Our Pipeline (Turn-Aware Mode, Zero-Shot)',
            'Domain_Trained_on_ENNI': 'No (Zero-shot)',
            'Seg_Precision': f"{summary['All']['segmentation_turn_aware']['micro_precision']*100:.1f}%",
            'Seg_Recall': f"{summary['All']['segmentation_turn_aware']['micro_recall']*100:.1f}%",
            'Seg_F1': f"{summary['All']['segmentation_turn_aware']['micro_f1']*100:.1f}%",
            'NTU_r': f"{summary['All']['downstream_metrics']['total_utterances_ntu']['pearson_r']:.2f}",
            'NTU_MAE': f"{summary['All']['downstream_metrics']['total_utterances_ntu']['mae']:.2f} utts",
            'NTU_Bias': f"{summary['All']['downstream_metrics']['total_utterances_ntu']['mean_percentage_diff']:+.2f}%",
            'MLU_r': f"{summary['All']['downstream_metrics']['mlu_in_words_mluw']['pearson_r']:.2f} (MLUw)",
            'MLU_MAE': f"{summary['All']['downstream_metrics']['mlu_in_words_mluw']['mae']:.2f} words",
            'MLU_Bias': f"{summary['All']['downstream_metrics']['mlu_in_words_mluw']['mean_percentage_diff']:+.2f}%"
        }
    ]
    
    comp_path = OUT_DIR / 'comparison_with_published.csv'
    with open(comp_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(comparison_table[0].keys()))
        writer.writeheader()
        writer.writerows(comparison_table)
    print(f'Wrote comparison table to {comp_path}')
    
    return summary


if __name__ == '__main__':
    run_benchmark()
