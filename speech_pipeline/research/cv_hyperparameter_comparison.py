"""Report the frozen final model versus the preserved production predictions.

Reporting only: requires completed study audits; never selects or deploys a model.
"""
from collections import defaultdict
import json
import numpy as np

from speech_pipeline import ROOT, utterances as auto
from speech_pipeline.pipeline import digest
from speech_pipeline.research import cv_hyperparameter as study
from speech_pipeline.research.transcript_chat import METRICS, PRIMARY, summarize


def run():
    audit = json.loads((study.EVIDENCE / 'final_integrity_audit.json').read_text())
    if not audit['complete']:
        raise ValueError('Complete and audit the frozen study before comparison')
    choice = json.loads((study.OUT / 'cv_selection.json').read_text())
    final_path = study.OUT / choice['configuration'] / 'final/test_results.json'
    if digest(final_path) != audit['final_test_sha256']:
        raise ValueError('Audited final predictions changed')
    new = json.loads(final_path.read_text())
    old_path = ROOT / 'outputs/results/transcript_validation/chat_calibrated_results.json'
    old = [r for r in json.loads(old_path.read_text())['records'] if r['split'] == 'test']
    old_by_key = {study.key(r): r for r in old}
    if set(old_by_key) != {study.key(r) for r in new['records']}:
        raise ValueError('Comparison test sections differ')
    deltas = defaultdict(lambda: defaultdict(list))
    lexical_equal = {m: True for m in ['TNW', 'NDW', 'TTR', 'MATTR']}
    for row in new['records']:
        before = old_by_key[study.key(row)]
        if before['participant'] != row['participant'] or before['reference'] != row['reference']:
            raise ValueError('Comparison participant/reference differs')
        words = lambda r: auto.stream([dict(text=u) for u in r['utterances']])[0]
        if words(before) != words(row):
            raise ValueError('Comparison word streams differ')
        for m in lexical_equal:
            lexical_equal[m] &= before['predicted'][m] == row['predicted'][m]
        for metric in METRICS:
            g, a, b = row['reference'][metric], before['predicted'][metric], row['predicted'][metric]
            if g and a is not None and b is not None:
                delta = 100 * (abs(b-g) - abs(a-g)) / abs(g)
                deltas[row['corpus'], metric][row['participant']].append(delta)
    rng = np.random.default_rng(study.SEED)
    intervals = {}
    for (corpus, metric), children in sorted(deltas.items()):
        values = np.array([np.mean(children[p]) for p in sorted(children)])
        draws = rng.choice(values, size=(5000, len(values)), replace=True).mean(axis=1)
        low, high = np.percentile(draws, [2.5, 97.5])
        intervals[corpus, metric] = dict(corpus=corpus, metric=metric,
            common_participants=len(values), common_sections=sum(map(len, children.values())),
            mean_paired_change_pp=float(values.mean()), ci95_low=float(low), ci95_high=float(high),
            bootstrap_draws=5000)
    old_summary = {(r['corpus'], r['metric']): r for r in summarize(old)}
    table = []
    for r in new['summary']:
        prior = old_summary[r['corpus'], r['metric']]
        table.append(dict(corpus=r['corpus'], metric=r['metric'],
            previous_ape=prior['mean_participant_ape'], final_ape=r['mean_participant_ape'],
            previous_missing=prior['missing'], final_missing=r['missing'],
            **{k:v for k,v in intervals.get((r['corpus'],r['metric']),{}).items() if k not in {'corpus','metric'}}))
    result = dict(configuration=choice['configuration'], final_test_sha256=digest(final_path),
        previous_results_sha256=digest(old_path), sections=len(new['records']),
        unchanged_words_and_human_references=True, exact_lexical_parity=lexical_equal,
        metrics=table, scope='Descriptive paired comparison on the preserved historically evaluated test cohort. Reporting only; no selection or deployment feedback. Bootstrap resamples children, conditional on these fitted models; it does not include training or tuning uncertainty.')
    study.save(study.OUT / 'previous_production_comparison.json', result)
    lines = ['# Frozen final model versus previous production', '', result['scope'], '',
        '| Corpus | Measure | Previous APE % | Final APE % | Paired change pp [95% bootstrap interval] | Missing sections, previous/final |',
        '|---|---|---:|---:|---:|---:|']
    for r in table:
        if r['metric'] not in PRIMARY:
            continue
        interval = f"{r['mean_paired_change_pp']:+.2f} [{r['ci95_low']:+.2f}, {r['ci95_high']:+.2f}]"
        lines.append(f"| {r['corpus']} | {r['metric']} | {r['previous_ape']:.2f} | {r['final_ape']:.2f} | {interval} | {r['previous_missing']}/{r['final_missing']} |")
    lines += ['', 'Negative changes indicate lower error. Paired changes use sections available for both methods; individual APE columns use each method’s available sections. NYU has only four test children, so its uncertainty must be interpreted cautiously.', '',
        f"All {result['sections']} sections have identical words and human reference values. Exact lexical parity by measure: " + ', '.join(f'{m}: {v}' for m,v in lexical_equal.items()) + '.', '',
        'The CV ranking and final test have different roles and training sample sizes. A lower CV selection score alone is not evidence of better production accuracy.', '']
    (study.OUT / 'MODEL_COMPARISON.md').write_text('\n'.join(lines))
    print('Saved frozen-model comparison:', study.OUT / 'MODEL_COMPARISON.md')
    return result


if __name__ == '__main__':
    run()
