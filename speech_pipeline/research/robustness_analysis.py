"""Summarize saved pilot comparisons without changing transcripts or human labels.

Resample participants as clusters, retaining each participant's recordings.
The baseline comparison is retrospective and uses development data.
"""
from pathlib import Path
from collections import defaultdict
import csv
import hashlib
import json
import statistics
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/results/paper/robustness_analysis.json'
SOURCE = ROOT / 'outputs/results/human_comparison'
SEED = 20261007
REPLICATES = 10000

def read(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def interval(values):
    return [float(x) for x in np.quantile(values, [.025, .975])]

def main():
    pairs = read(SOURCE / 'metric_pairs.csv')
    boundary = read(SOURCE / 'boundary_recordings.csv')
    participants = sorted({r['patient_id'] for r in pairs if r['reference'] == 'Nafee'}, key=lambda x: int(x[1:]))
    assert len(participants) == 10
    rng = np.random.default_rng(SEED)
    samples = rng.integers(0, len(participants), (REPLICATES, len(participants)))
    manual, corrected, baseline, participant_scores = {}, {}, {}, []
    def metric_summary(reference, metric, method):
        rs = [r for r in pairs if (r['reference'], r['metric'], r['method']) == (reference, metric, method)]
        g = defaultdict(list)
        for r in rs:
            assert float(r['reference_value']) > 0 and r['absolute_percent_error']
            g[r['patient_id']].append(r)
        assert set(g) == set(participants)
        errors = np.array([statistics.mean(float(r['absolute_percent_error']) for r in g[p]) for p in participants])
        differences = np.array([statistics.mean(float(r['difference']) for r in g[p]) for p in participants])
        return dict(n_recordings=len(rs), n_participants=len(g), mape=float(errors.mean()), mape_ci=interval(errors[samples].mean(axis=1)),
                    bias=float(differences.mean()), bias_ci=interval(differences[samples].mean(axis=1)),
                    participant_mape_min=float(errors.min()), participant_mape_max=float(errors.max()),
                    participant_values=dict(zip(participants, map(float, errors))), _errors=errors)
    for metric in ['basic_ntokens', 'basic_ntypes', 'basic_ncontent_tokens', 'basic_ncontent_types',
                   'basic_nfunction_tokens', 'basic_nfunction_types', 'simple_ttr_aw', 'mattr50_aw',
                   'total_utterances', 'mlu_words', 'mlu_words_sd']:
        result = metric_summary('Nafee', metric, 'Revised')
        assert result['n_recordings'] == 14
        result.pop('_errors'); manual[metric] = result
        ref = 'Corrected CLAN' if metric in ['total_utterances', 'mlu_words', 'mlu_words_sd'] else 'Corrected TAALED'
        software_metric = 'mlu_nutterances' if metric == 'total_utterances' else metric
        result = metric_summary(ref, software_metric, 'Revised')
        assert result['n_recordings'] == 37
        result.pop('_errors'); corrected[software_metric] = result
    for metric in ['total_utterances', 'mlu_words', 'mlu_words_sd']:
        earlier = metric_summary('Nafee', metric, 'Previous')
        current = metric_summary('Nafee', metric, 'Revised')
        delta = current.pop('_errors') - earlier.pop('_errors')
        baseline[metric] = dict(previous=earlier, current=current, paired_change=float(delta.mean()),
                                paired_change_ci=interval(delta[samples].mean(axis=1)))
    for method in ['Previous', 'Revised']:
        rs = [r for r in boundary if r['method'] == method and r['reference'] == 'Nafee']
        assert len(rs) == 14 and all(int(r['word_edits']) == 0 and float(r['reference_gap_coverage']) == 1 for r in rs)
        g = defaultdict(lambda: np.zeros(4))
        for r in rs:
            g[r['patient_id']] += [int(r[k]) for k in ['tp','tn','fn','fp']]
        values = np.array([g[p] for p in participants]); pooled = values.sum(axis=0)
        tp, tn, fn, fp = pooled
        boot = values[samples].sum(axis=1)
        f1_boot = 100 * 2 * boot[:,0] / (2 * boot[:,0] + boot[:,2] + boot[:,3])
        baseline[method + '_boundaries'] = dict(evaluable_gaps=int(pooled.sum()),
            counts=dict(zip(['tp','tn','fn','fp'], map(int, pooled))),
            f1_percent=float(100 * 2 * tp / (2 * tp + fn + fp)), f1_ci=interval(f1_boot),
            precision_percent=float(100*tp/(tp+fp)), recall_percent=float(100*tp/(tp+fn)),
            coverage=1.0, word_edits=0)
    wers = read(ROOT / 'outputs/results/paper/transcript_wer_participants.csv')
    indexed = {r['patient_id']: float(r['mean_recording_wer_percent']) for r in wers}
    a = np.array([indexed[p] for p in participants])
    complete_pairs = read(SOURCE / 'participant_sample_pairs.csv')
    complete, complete_baseline = {}, {}
    def complete_summary(metric, method):
        indexed = {r['patient_id']: r for r in complete_pairs if r['metric'] == metric and r['method'] == method}
        assert set(indexed) == set(participants)
        errors = np.array([float(indexed[p]['absolute_percent_error']) for p in participants])
        differences = np.array([float(indexed[p]['difference']) for p in participants])
        return dict(n_participants=len(participants), mape=float(errors.mean()),
                    mape_ci=interval(errors[samples].mean(axis=1)),
                    bias=float(differences.mean()), bias_ci=interval(differences[samples].mean(axis=1)),
                    participant_mape_min=float(errors.min()), participant_mape_max=float(errors.max()),
                    participant_values=dict(zip(participants, map(float, errors))), _errors=errors)
    for metric in manual:
        current = complete_summary(metric, 'Revised')
        earlier = complete_summary(metric, 'Previous')
        delta = current.pop('_errors') - earlier.pop('_errors')
        complete[metric] = current
        complete_baseline[metric] = dict(previous=earlier, current=current,
                                        paired_change=float(delta.mean()),
                                        paired_change_ci=interval(delta[samples].mean(axis=1)))
    result = dict(bootstrap=dict(seed=SEED, replicates=REPLICATES, method='Participant-cluster percentile bootstrap', confidence_level=.95),
                  manual=manual, corrected=corrected, baseline=baseline,
                  complete_samples=complete, complete_sample_baseline=complete_baseline,
                  wer=dict(mean=float(a.mean()), ci=interval(a[samples].mean(axis=1)), minimum=float(a.min()), maximum=float(a.max())),
                  sources={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in [
                      SOURCE/'metric_pairs.csv',SOURCE/'boundary_recordings.csv',SOURCE/'participant_sample_pairs.csv',
                      ROOT/'dataset/annotations/all_nafee_utterances.xlsx',
                      ROOT/'outputs/results/paper/transcript_wer_participants.csv']})
    OUT.write_text(json.dumps(result, indent=2) + '\n')
    for metric in ['total_utterances','mlu_words','mlu_words_sd']:
        row=baseline[metric]
        print(metric,'earlier',round(row['previous']['mape'],2),'current',round(row['current']['mape'],2),
              '95% CI',[round(v,2) for v in row['current']['mape_ci']],
              'change',[round(v,2) for v in row['paired_change_ci']])
    for method in ['Previous','Revised']:
        row=baseline[method+'_boundaries'];print(method,'F1',round(row['f1_percent'],2),'CI',[round(v,2) for v in row['f1_ci']],row['counts'])
    print('WER',result['wer'])
    print('Current manual SD bias',manual['mlu_words_sd']['bias'],'participant error range',manual['mlu_words_sd']['participant_mape_min'],manual['mlu_words_sd']['participant_mape_max'])
    for m in ['simple_ttr_aw','mattr50_aw','mlu_words','mlu_words_sd']:
        print('Corrected',m,corrected[m]['mape'],corrected[m]['mape_ci'])

if __name__ == '__main__':
    main()
