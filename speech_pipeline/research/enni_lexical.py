"""Calculate all 15 TAALED lexical diversity and density metrics on ENNI transcripts.

Metrics:
1. Total words / tokens (basic_ntokens)
2. Different words / types (basic_ntypes)
3. Content-word count (basic_ncontent_tokens)
4. Different content words (basic_ncontent_types)
5. Function-word count (basic_nfunction_tokens)
6. Different function words (basic_nfunction_types)
7. Lexical density by tokens (lexical_density_tokens)
8. Lexical density by types (lexical_density_types)
9. TTR (simple_ttr_aw)
10. Log TTR (log_ttr_aw)
11. Maas (maas_ttr_aw)
12. MATTR50 (mattr50_aw)
13. MSTTR50 (msttr50_aw)
14. HD-D42 (hdd42_aw)
15. Original MTLD (mtld_original_aw)
"""
import csv
import glob
import json
from pathlib import Path
import re
import numpy as np
import spacy

from speech_pipeline import ROOT, utterances as auto
from speech_pipeline.lexical_compat import HistoricalTaaled

ENNI_DIR = ROOT / 'dataset/public_external/enni/transcripts'
OUT_DIR = ROOT / 'outputs/results/enni_benchmark'


def run_lexical():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(str(ENNI_DIR / '**/*.cha'), recursive=True))
    nlp = spacy.load('en_core_web_sm')
    taaled = HistoricalTaaled(ROOT / 'outputs/validation/lucid_validation/upstream_gui')
    
    rows = []
    for f in files:
        cohort = 'TD' if '/TD/' in f else ('SLI' if '/SLI/' in f else 'Unknown')
        text = Path(f).read_text(encoding='utf-8-sig')
        words = []
        for line in text.splitlines():
            if line.startswith('*CHI:'):
                clean = re.sub(r'\d+_\d+', '', line.split(':', 1)[1])
                clean = re.sub(r'\[[/?\w\s-]+\]', '', clean)
                clean = re.sub(r'&=[^\s<>\[\]]+', '', clean)
                clean = re.sub(r'\(.*?\)', '', clean)
                words.extend(auto.WORD.findall(clean))
        if not words:
            continue
        doc = nlp(' '.join(words))
        res = taaled.calculate(doc)
        res['filename'] = Path(f).name
        res['cohort'] = cohort
        rows.append(res)
        
    csv_path = OUT_DIR / 'enni_taaled_15_metrics.csv'
    fieldnames = ['filename', 'cohort'] + [k for k in rows[0].keys() if k not in ('filename', 'cohort')]
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        
    # Aggregate summary by cohort
    summary = {}
    for c_name in ('All', 'TD', 'SLI'):
        selected = rows if c_name == 'All' else [r for r in rows if r['cohort'] == c_name]
        c_summary = {'count': len(selected)}
        for metric in [k for k in fieldnames if k not in ('filename', 'cohort')]:
            vals = [r[metric] for r in selected if r[metric] is not None]
            c_summary[metric] = {
                'mean': float(np.mean(vals)) if vals else None,
                'std': float(np.std(vals)) if vals else None,
                'median': float(np.median(vals)) if vals else None
            }
        summary[c_name] = c_summary
        
    json_path = OUT_DIR / 'enni_taaled_summary.json'
    json_path.write_text(json.dumps(summary, indent=2))
    print(f'Wrote ENNI TAALED 15 metrics to {csv_path} and {json_path}')
    return summary


if __name__ == '__main__':
    run_lexical()
