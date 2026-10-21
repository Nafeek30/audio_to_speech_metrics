"""Plot existing saved scores only; no inference, training, or scoring run."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SOURCES = {
    'spaCy + rules': ROOT / 'outputs/results/transcript_validation/chat_validation.json',
    'Our model': ROOT / 'outputs/results/transcript_validation/chat_calibrated_results.json',
}
methods = {'spaCy + rules': 'turn_rules', 'Our model': 'turn_ml/margin=0.25'}
records = {}
for label, path in SOURCES.items():
    saved = json.loads(path.read_text())
    records[label] = [r for r in saved['records']
                      if r['split'] == 'validation' and r['method'] == methods[label]]
    assert len(records[label]) == 139

def key(row):
    return row['corpus'], row['participant'], row['recording']

lookups = {label: {key(r): r for r in rows} for label, rows in records.items()}
assert set(lookups['spaCy + rules']) == set(lookups['Our model'])
for k, rule in lookups['spaCy + rules'].items():
    model = lookups['Our model'][k]
    assert rule['reference'] == model['reference']
    assert rule['boundaries']['eligible_gaps'] == model['boundaries']['eligible_gaps']
    for row in (rule, model):
        assert row['boundaries']['status'] == 'complete_shared_stream'
    rc, mc = rule['boundaries']['counts'], model['boundaries']['counts']
    assert rc['tp'] + rc['fn'] == mc['tp'] + mc['fn']

metrics = ['Precision', 'Recall', 'F1']
rows = []
for corpus in ['ocsc', 'nyu_emerson']:
    for label, saved_rows in records.items():
        subset = [r for r in saved_rows if r['corpus'] == corpus]
        counts = {k: sum(r['boundaries']['counts'][k] for r in subset)
                  for k in ['tp', 'fp', 'fn']}
        tp, fp, fn = (counts[k] for k in ['tp', 'fp', 'fn'])
        rows.append(dict(corpus=corpus, method=label, saved_method=methods[label],
                         participants=len({r['participant'] for r in subset}),
                         sections=len(subset), **counts,
                         Precision=100 * tp / (tp + fp),
                         Recall=100 * tp / (tp + fn),
                         F1=200 * tp / (2 * tp + fp + fn)))

with (OUT / 'boundary_precision_recall_f1_validation.csv').open('w') as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
                     'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(1, 2, figsize=(12, 5.8), sharey=True)
colors = {'spaCy + rules': '#9AABC0', 'Our model': '#007F78'}
for ax, corpus, title in zip(axes, ['ocsc', 'nyu_emerson'], ['OCSC', 'NYU–Emerson']):
    x = np.arange(3)
    for index, label in enumerate(SOURCES):
        row = next(r for r in rows if r['corpus'] == corpus and r['method'] == label)
        values = [row[m] for m in metrics]
        bars = ax.bar(x + (index - 0.5) * 0.36, values, 0.34,
                      color=colors[label], label=label, zorder=3)
        ax.bar_label(bars, labels=[f'{v:.2f}' for v in values], padding=5, fontsize=11)
    ax.set_title(f"{title}\n{row['participants']} children · {row['sections']} sections", fontsize=13, pad=13)
    ax.set_xticks(x, metrics)
    ax.set_ylim(0, 105)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.grid(axis='y', color='#E5E7EB', zorder=0)
    ax.spines['left'].set_color('#BBC1C7')
    ax.spines['bottom'].set_color('#BBC1C7')
    ax.tick_params(length=0, pad=8)
axes[0].set_ylabel('Boundary score (%) — higher is better')
fig.suptitle('Our model compared with spaCy + rules', fontsize=19, fontweight='bold', y=0.98)
fig.text(0.5, 0.905, 'Saved validation results · same observed speaker-turn context', ha='center', fontsize=12, color='#485563')
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.065), ncol=2, frameon=False)
fig.text(0.5, 0.025, 'Our model: MiniLM + rule features, margin 0.25. Scores pool saved boundary counts within each corpus.',
         ha='center', fontsize=10, color='#485563')
fig.subplots_adjust(top=0.74, bottom=0.22, left=0.08, right=0.985, wspace=0.11)
for suffix in ['png', 'svg']:
    fig.savefig(OUT / f'boundary_precision_recall_f1_validation.{suffix}', dpi=220, facecolor='white')
plt.close(fig)

metadata = dict(scope='Replotted existing validation counts only; no model rerun or fresh evaluation.',
                split='validation', aggregation='Micro-aggregated saved boundary TP/FP/FN within corpus',
                source_files={label: dict(path=str(path.relative_to(ROOT)),
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                    for label, path in SOURCES.items()}, results=rows)
(OUT / 'figure_data.json').write_text(json.dumps(metadata, indent=2) + '\n')
for row in rows:
    print(row['corpus'], row['method'], {m: round(row[m], 2) for m in metrics})
