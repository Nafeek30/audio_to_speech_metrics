"""Replot the preserved early three-way benchmark; never run models."""
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'speech_plot_mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
files = {name: ROOT / 'outputs/results/benchmark' / f'{name}.json'
         for name in ['talkbank_model', 'hybrid_model']}
data = {name: json.loads(path.read_text()) for name, path in files.items()}
a, b = data['talkbank_model'], data['hybrid_model']
assert a['input_sha256'] == b['input_sha256']
assert a['splits'] == b['splits']
assert a['held_out_test']['overall']['baseline'] == b['held_out_test']['overall']['baseline']
methods = [('spaCy + rules', a, 'baseline'),
           ('Model only', a, 'transformer'),
           ('Combined', b, 'transformer')]
rows = []
for label, saved, method in methods:
    overall = saved['held_out_test']['overall']
    c = overall[method]
    tp, fp, fn = (c[k] for k in ['tp', 'fp', 'fn'])
    f1 = 2 * tp / (2 * tp + fp + fn)
    assert abs(f1 - c['f1']) < 1e-12
    rows.append(dict(method=label, participants=overall['participants'],
                     records=overall['records'], tp=tp, fp=fp, fn=fn,
                     Precision=100 * tp / (tp + fp),
                     Recall=100 * tp / (tp + fn), F1=100 * f1))
assert len({r['tp'] + r['fn'] for r in rows}) == 1
metrics = ['Precision', 'Recall', 'F1']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
                     'axes.spines.top': False, 'axes.spines.right': False})
fig, ax = plt.subplots(figsize=(10.5, 6.2))
x = np.arange(3)
for i, (row, color) in enumerate(zip(rows, ['#94A3B8', '#D98C30', '#007F78'])):
    values = [row[m] for m in metrics]
    bars = ax.bar(x + (i - 1) * 0.25, values, width=0.235,
                  label=row['method'], color=color, zorder=3)
    ax.bar_label(bars, labels=[f'{v:.2f}' for v in values], padding=5, fontsize=11)
ax.set_xticks(x, metrics)
ax.set_ylim(0, 100)
ax.set_yticks(np.arange(0, 101, 20))
ax.set_ylabel('Boundary score (%) — higher is better')
ax.grid(axis='y', color='#E5E7EB', zorder=0)
for side in ['left', 'bottom']:
    ax.spines[side].set_color('#BBC1C7')
ax.tick_params(length=0, pad=8)
fig.suptitle('Earlier three-way comparison', fontsize=20, fontweight='bold', y=0.97)
fig.text(0.5, 0.895, 'Saved TalkBank test benchmark · 49 children · paragraph input', ha='center', color='#485563')
fig.legend(*ax.get_legend_handles_labels(), loc='lower center',
           bbox_to_anchor=(0.5, 0.075), ncol=3, frameon=False)
fig.text(0.5, 0.025, 'Historical two-layer transformer experiment, before MiniLM. Pooled boundary counts.',
         ha='center', fontsize=10, color='#485563')
fig.subplots_adjust(left=0.1, right=0.98, bottom=0.23, top=0.8)
for suffix in ['png', 'svg']:
    fig.savefig(OUT / f'historical_three_way_precision_recall_f1.{suffix}', dpi=220, facecolor='white')
plt.close(fig)
with (OUT / 'historical_three_way_precision_recall_f1.csv').open('w') as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
metadata = dict(scope='Historical held-out benchmark, already inspected across research rounds; not fresh confirmation.',
                architecture=a['architecture'], combined_architecture=b['architecture'],
                original_figure='outputs/results/benchmark/talkbank_results.png',
                source_files={name: dict(path=str(path.relative_to(ROOT)),
                  sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for name, path in files.items()},
                aggregation='Micro-aggregated saved TP/FP/FN; no new predictions or evaluation run.', results=rows)
(OUT / 'historical_three_way_figure_data.json').write_text(json.dumps(metadata, indent=2) + '\n')
for row in rows:
    print(row['method'], {m: round(row[m], 2) for m in metrics})
