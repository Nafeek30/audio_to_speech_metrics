"""Matched public transcript benchmark; no training or private-cohort access.

Run ``register`` before ``evaluate``. Batchalign is loaded from an independently
downloaded, SHA-verified wheel; its official UtSeg backend and local pinned model
are used without modifying inference. This evaluates a segmentation component,
not Batchalign's audio pipeline or a reproduction of its published cohort.
"""
import argparse
import csv
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

from speech_pipeline import ROOT, utterances as auto
from speech_pipeline.metrics import clinical_from_predictions
from speech_pipeline.pipeline import digest, save

OUT = ROOT / 'outputs/results/common_benchmark'
EVIDENCE = ROOT / 'outputs/validation/common_benchmark'
TEST = ROOT / 'outputs/validation/benchmark/talkbank_spoken_test.jsonl'
FROZEN = ROOT / 'outputs/results/benchmark/improved_model.json'
METRICS = ('total_utterances', 'MLUw', 'length_sd', 'TNW', 'NDW', 'TTR', 'MATTR')
PRIMARY = METRICS[:3]
METHODS = ('rules', 'previous_minilm', 'latest_minilm', 'batchalign_0.10.4')
REVISION = '764ec3f762c2e24df2def8df98b5fe34940085c6'


def key(r):
    return r['corpus'], r['participant'], r['recording']


def units(words, cuts):
    cuts = list(cuts)
    if cuts != sorted(set(cuts)) or any(not 0 < c < len(words) for c in cuts):
        raise ValueError('Invalid word-gap cuts')
    ends = [0, *cuts, len(words)]
    return [dict(utterance_id=i + 1, text=' '.join(words[a:b]))
            for i, (a, b) in enumerate(zip(ends, ends[1:])) if a < b]


def values(utterances):
    surface = clinical_from_predictions(utterances)['values']
    lengths = auto.word_mlu(utterances)
    return dict(total_utterances=lengths['total_utterances'],
                length_sd=lengths['mlu_words_sd'],
                **{k: surface[k] for k in METRICS if k in surface})


def errors(reference, prediction):
    return {k: dict(reference=reference[k], predicted=prediction[k],
                    signed_error=prediction[k] - reference[k]
                    if prediction[k] is not None and reference[k] is not None else None,
                    ape=100 * abs(prediction[k] - reference[k]) / abs(reference[k])
                    if prediction[k] is not None and reference[k] else None)
            for k in METRICS}


def boundaries(record, output):
    """Require the entire shared word stream; never score aligned subsets."""
    tokens, cuts = auto.stream(output)
    if tokens != record['tokens']:
        return dict(status='unavailable_word_stream_changed', scored_gaps=0,
                    eligible_gaps=sum(v >= 0 for v in record['labels']),
                    reference_words=len(record['tokens']), predicted_words=len(tokens))
    mask = [i + 1 for i, label in enumerate(record['labels']) if label >= 0]
    gold = set(record['gold_boundaries'])
    c = dict(tp=sum(i in gold and i in cuts for i in mask),
             fp=sum(i not in gold and i in cuts for i in mask),
             fn=sum(i in gold and i not in cuts for i in mask))
    def spans(cs):
        ends = [0, *sorted(cs), len(tokens)]
        return set(zip(ends, ends[1:]))
    a, b = spans(gold), spans(cuts)
    return dict(status='complete_shared_stream', counts=c,
                eligible_gaps=len(mask), scored_gaps=len(mask),
                predicted_boundaries=sorted(cuts),
                exact=dict(reference=len(a), predicted=len(b), matched=len(a & b)))


def summarize(rows):
    result = {}
    for method in METHODS:
        result[method] = {}
        for corpus in sorted({r['corpus'] for r in rows}):
            selected = [r for r in rows if r['method'] == method and r['corpus'] == corpus]
            summary = dict(records=len(selected), participants=len({r['participant'] for r in selected}))
            scores = {}
            for metric in METRICS:
                children = defaultdict(list)
                for row in selected:
                    v = row['metrics'][metric]['ape']
                    if v is not None:
                        children[row['participant']].append(v)
                means = [float(np.mean(v)) for v in children.values()]
                scores[metric] = dict(mean_participant_mean_recording_ape=float(np.mean(means)) if means else None,
                                     median_participant_mean_recording_ape=float(np.median(means)) if means else None,
                                     available_recordings=sum(len(v) for v in children.values()),
                                     missing_recordings=len(selected)-sum(len(v) for v in children.values()),
                                     available_participants=len(means))
            summary['metrics'] = scores
            primary = [scores[k]['mean_participant_mean_recording_ape'] for k in PRIMARY]
            summary['primary_objective'] = float(np.mean(primary)) if all(v is not None for v in primary) else None
            complete = [r['boundaries'] for r in selected if r['boundaries']['status'] == 'complete_shared_stream']
            counts = {k: sum(r['counts'][k] for r in complete) for k in ('tp', 'fp', 'fn')}
            tp, fp, fn = (counts[k] for k in ('tp', 'fp', 'fn'))
            exact = {k: sum(r['exact'][k] for r in complete) for k in ('reference', 'predicted', 'matched')}
            summary['boundaries'] = dict(counts=counts, complete_records=len(complete),
                eligible_gaps=sum(r['boundaries']['eligible_gaps'] for r in selected),
                scored_gaps=sum(r['boundaries']['scored_gaps'] for r in selected),
                precision=tp/(tp+fp) if tp+fp else 0., recall=tp/(tp+fn) if tp+fn else 0.,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0., exact=exact,
                exact_f1=2*exact['matched']/(exact['reference']+exact['predicted'])
                if exact['reference']+exact['predicted'] else 0.)
            result[method][corpus] = summary
        objectives = [v['primary_objective'] for v in result[method].values()]
        result[method]['corpus_balanced_primary'] = float(np.mean(objectives)) if all(v is not None for v in objectives) else None
    return result


def paired_intervals(rows, draws=10000):
    """Stratified paired child bootstrap, preserving all sessions of a child."""
    rng = np.random.default_rng(42)
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    for row in rows:
        for metric in PRIMARY:
            v = row['metrics'][metric]['ape']
            if v is not None:
                grouped[row['corpus']][row['participant']][row['method']][metric].append(v)
    changes = {m: [] for m in METHODS if m != METHODS[-1]}
    points = {m: [] for m in changes}
    for corpus, children in sorted(grouped.items()):
        ids = sorted(children)
        if any(any(not children[p][m][k] for m in METHODS for k in PRIMARY) for p in ids):
            raise ValueError('Paired primary endpoint missing; do not silently drop children')
        samples = rng.integers(0, len(ids), size=(draws, len(ids)))
        matrices = {m: np.array([[np.mean(children[p][m][k]) for k in PRIMARY] for p in ids]) for m in METHODS}
        for m in changes:
            delta = (matrices[m]-matrices[METHODS[-1]]).mean(axis=1)
            changes[m].append(delta[samples].mean(axis=1))
            points[m].append(float(delta.mean()))
    return {m: dict(delta_ape_points_vs_batchalign=float(np.mean(points[m])),
                    percentile_95_interval=np.quantile(np.mean(changes[m], axis=0), [.025,.975]).tolist(),
                    draws=draws, interpretation='Negative favors our method; descriptive reused benchmark, not model selection')
            for m in changes}


def register(software):
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    plan = EVIDENCE / 'protocol.json'
    if plan.exists():
        raise FileExistsError('Protocol already registered; preserve it')
    package = json.loads((software / 'package.json').read_text())
    save(plan, dict(registered_utc=datetime.now(timezone.utc).isoformat(),
        purpose='Validate language metric accuracy against public human CHAT references and an established published-system segmentation component on exactly the same data',
        scope='Transcript-only, condition A: one unpunctuated target-child paragraph per existing task section. No observed turns, audio, gold punctuation, morphology or timing in predictor inputs.',
        excluded='All original 10 participants; no access or evaluation in this benchmark',
        data_sha256=digest(TEST), frozen_predictions_sha256=digest(FROZEN),
        methods=list(METHODS), batchalign_package=package,
        model_id='talkbank/CHATUtterance-en', model_revision=REVISION,
        primary='Equal mean across count, surface MLUw and population length-SD APE; recording mean within child, child mean within corpus, equal corpus mean. Zero reference denominators unavailable.',
        secondary='Boundary P/R/F1 and exact spans; TNW/NDW/TTR/MATTR50 APE and missingness; signed recording errors; paired child bootstrap of primary difference vs Batchalign, 10000 draws seed 42.',
        reference='Corrected spoken words and CHAT continuation-aware units frozen in talkbank_spoken_test.jsonl; normalized reference policy differs from original fully coded CHAT/CLAN.',
        integrity='Retain every record, raw official API outputs, complete eligible gap masks, source/input/output/software/model hashes. Word changes make boundary score unavailable for the entire record; never score convenient aligned subsets.',
        calculator_check='First 3 distinct children in each corpus in frozen input order, all 4 methods plus reference, official native CLAN on same normalized text, counts exact and ratios/SD within 0.000501.',
        inference='CPU, official CHATUtteranceBackend.call and BertUtteranceModel from wheel, supported local-model initialization; no altered chunking, punctuation decoding or cleanup. Frozen predictions reused for MiniLM; frozen paragraph rule predictions.',
        budget='20 minute bounded setup/scoring; no training, model choice or tuning from final results.',
        caveat='Both corpora and test have historical exposure. This is a descriptive common-data component benchmark, not replication of the 2023 paper scores, independent confirmation, human equivalence or audio-to-metric validation.'))
    print('Registered common-input public benchmark', flush=True)


def evaluate(software, model_path):
    from tempfile import TemporaryDirectory
    from speech_pipeline.pipeline import clan_software_check
    plan = json.loads((EVIDENCE/'protocol.json').read_text())
    if digest(TEST) != plan['data_sha256'] or digest(FROZEN) != plan['frozen_predictions_sha256']:
        raise ValueError('Frozen inputs changed')
    if (OUT/'results.json').exists():
        raise FileExistsError('Completed comparison exists; preserve results')
    records = [json.loads(s) for s in TEST.read_text().splitlines()]
    if any(r['corpus'] not in {'ocsc','nyu_emerson'} for r in records):
        raise ValueError('Only the declared public corpora are allowed')
    for r in records:
        source = ROOT/'dataset/public'/r['corpus']/r['recording'].split('#section')[0]
        if digest(source) != r['source_sha256']:
            raise ValueError('CHAT reference source changed')
    frozen = json.loads(FROZEN.read_text())['metric_round']
    saved = {m: {key(r): r['predicted_boundaries'] for r in c['public_test']['boundaries']['recordings']}
             for m,c in [('previous_minilm',frozen['incumbent']),('latest_minilm',frozen)]}
    if any(set(v) != {key(r) for r in records} for v in saved.values()):
        raise ValueError('Frozen prediction coverage differs')
    sys.path[:0] = [str(software/'package'),str(software/'dependencies')]
    from batchalign.backends.asr.chatwhisper import BertUtteranceModel
    from batchalign.backends.utseg.chatutterance import CHATUtteranceBackend
    from batchalign._core.proto import UtSegInput, AsrSegment, AsrWord
    import torch
    torch.set_num_threads(4)
    if model_path.name != REVISION:
        raise ValueError('Require exact pinned model snapshot directory')
    segmenter = BertUtteranceModel(str(model_path),device='cpu')
    backend = CHATUtteranceBackend(lang='eng',segmenter=segmenter)
    model_hashes = {p.name:digest(p) for p in model_path.iterdir() if p.is_file()}
    source_hashes = {str(p.relative_to(software/'package')):digest(p)
                     for p in (software/'package/batchalign').rglob('*.py')}
    OUT.mkdir(parents=True,exist_ok=True)
    raw_path = OUT/'batchalign_raw.jsonl'
    rows=[]; checks=[]; chosen=defaultdict(set); start=time.monotonic()
    with raw_path.open('w') as raw, TemporaryDirectory(prefix='common_clan_') as temp:
        for index,r in enumerate(records):
            words = r.get('surface_words',r['tokens'])
            inp = UtSegInput(source_id=str(index), language={'kind':'code','value':'eng'}, segments=[AsrSegment(
                start_ms=0,end_ms=0,text=' '.join(words),speaker='CHI',
                words=[AsrWord(text=w,start_ms=0,end_ms=0,confidence=1.) for w in words])])
            began=time.monotonic(); out=backend.call([inp])[0]
            elapsed=time.monotonic()-began
            raw.write(json.dumps(dict(corpus=r['corpus'],participant=r['participant'],recording=r['recording'],
                                     input=inp.model_dump(),output=out.model_dump(),runtime_seconds=elapsed),ensure_ascii=False)+'\n');raw.flush()
            predictions = {m:units(words,saved[m][key(r)]) for m in saved}
            predictions['rules'] = units(words,[i+1 for i,v in enumerate(r['baseline'][:-1]) if v])
            predictions[METHODS[-1]] = [dict(text=u.text,utterance_id=i+1) for i,u in enumerate(out.utterances)]
            reference = units(words,r['gold_boundaries']); truth = values(reference)
            do_check = r['participant'] not in chosen[r['corpus']] and len(chosen[r['corpus']])<3
            if do_check:
                chosen[r['corpus']].add(r['participant'])
                checks.append(dict(corpus=r['corpus'],participant=r['participant'],recording=r['recording'],method='reference',check=clan_software_check(reference,Path(temp))))
            for method,utterances in predictions.items():
                b=boundaries(r,utterances)
                rows.append(dict(corpus=r['corpus'],participant=r['participant'],recording=r['recording'],gem=r['gem'],method=method,
                    metrics=errors(truth,values(utterances)),boundaries=b,
                    words_preserved=auto.stream(utterances)[0]==r['tokens'],
                    input_sha256=hashlib.sha256(' '.join(words).encode()).hexdigest(),
                    utterances=[u['text'] for u in utterances],
                    eligible_gap_mask=[v>=0 for v in r['labels']]))
                if do_check:
                    checks.append(dict(corpus=r['corpus'],participant=r['participant'],recording=r['recording'],method=method,check=clan_software_check(utterances,Path(temp))))
            if index%10==0:
                print(f'Scored {index+1}/{len(records)} public paragraphs; {time.monotonic()-start:.1f}s',flush=True)
    computed = [c for c in checks if c['check']['status']=='software_reference_computed']
    passed = len(computed)==len(checks) and all(v['absolute_error'] is None or v['absolute_error'] <=
        (0 if k in ('TNW','NDW','mlu_nwords','mlu_nutterances') else .000501)
        for c in computed for group in ('metrics','requested_metrics') for k,v in c['check'][group].items())
    result=dict(protocol_sha256=digest(EVIDENCE/'protocol.json'),completed_utc=datetime.now(timezone.utc).isoformat(),
        summary=summarize(rows),paired_primary_intervals=paired_intervals(rows),recordings=rows,
        software=dict(package=plan['batchalign_package'],source_hashes=source_hashes,model_hashes=model_hashes,
                      torch=torch.__version__,device='cpu',backend=backend.name),
        runtime_seconds=time.monotonic()-start,raw_output_sha256=digest(raw_path),
        calculator_checks=dict(expected=len(checks),computed=len(computed),passed=passed))
    save(EVIDENCE/'calculator_checks.json',checks)
    save(OUT/'results.json',result)
    report(result)
    if not passed:
        raise ValueError('Native CLAN discrepancy; inspect retained evidence before claiming validation')
    print(json.dumps({m:v['corpus_balanced_primary'] for m,v in result['summary'].items()}),flush=True)


def report(result):
    table=[]
    for row in result['recordings']:
        for metric,score in row['metrics'].items():
            table.append(dict(method=row['method'],corpus=row['corpus'],participant=row['participant'],
                recording=row['recording'],gem=row['gem'],metric=metric,**score))
    with (OUT/'metric_pairs.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    lines=['# Public common-input benchmark','',
           'All four methods receive the same unpunctuated target-child paragraphs, without source-turn cuts. '
           'References are frozen human CHAT units on corrected spoken words. No original ten-participant data were accessed.','',
           'Primary: mean APE across utterance count, surface MLUw and population length SD, recording-balanced within child, '
           'child-balanced within corpus and equally weighted across corpora. Lower is better.','',
           '| Method | Corpus | Children | Paragraphs | Count APE % | MLUw APE % | SD APE % | Primary % | Boundary F1 % | Exact span F1 % |',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for method,corpora in result['summary'].items():
        for corpus,v in corpora.items():
            if not isinstance(v,dict):continue
            m=v['metrics'];b=v['boundaries']
            cells=[method,corpus,str(v['participants']),str(v['records']),
                   *[f"{m[k]['mean_participant_mean_recording_ape']:.2f}" for k in PRIMARY],
                   f"{v['primary_objective']:.2f}",f"{100*b['f1']:.2f}",f"{100*b['exact_f1']:.2f}"]
            lines.append('| '+' | '.join(cells)+' |')
    lines+=['','## Corpus-balanced primary comparison','',
            '| Method | Mean APE % | Difference vs Batchalign (95% paired child interval) |',
            '|---|---:|---|']
    for method,v in result['summary'].items():
        ci=result['paired_primary_intervals'].get(method)
        text=(f"{ci['delta_ape_points_vs_batchalign']:+.2f} [{ci['percentile_95_interval'][0]:+.2f}, {ci['percentile_95_interval'][1]:+.2f}]" if ci else 'Reference competitor')
        lines.append(f"| {method} | {v['corpus_balanced_primary']:.2f} | {text} |")
    lines+=['','## Lexical metric errors and coverage','',
            '| Method | Corpus | TNW APE % | NDW APE % | TTR APE % | MATTR50 APE % | Missing MATTR50 |',
            '|---|---|---:|---:|---:|---:|---:|']
    for method,corpora in result['summary'].items():
        for corpus,v in corpora.items():
            if not isinstance(v,dict):continue
            ms=v['metrics']; cells=[method,corpus]
            for k in METRICS[3:]:
                a=ms[k]['mean_participant_mean_recording_ape'];cells.append(f'{a:.2f}' if a is not None else 'unavailable')
            cells.append(str(ms['MATTR']['missing_recordings']));lines.append('| '+' | '.join(cells)+' |')
    intact=sum(r['words_preserved'] for r in result['recordings'])
    lines+=['',f"Word streams preserved in {intact}/{len(result['recordings'])} method-recording pairs. Entire eligible gap masks and raw official inputs/outputs are retained. Exact spans use all original word positions; boundary F1 masks final and unintelligible-adjacent gaps.",
            '',f"Native CLAN calculator checks: {result['calculator_checks']}. These check normalized arithmetic, not agreement of predicted boundaries with professionals. Raw outputs are in the validation folder.",
            '', '## Interpretation limits','',
            'Batchalign 0.10.4 official CHATUtteranceBackend and its pinned English model are evaluated as a segmentation component. '
            'This is not a rerun of the complete 2023 published audio pipeline. Existing tests have historical exposure. '
            'Our MiniLM models were trained on disjoint children from these corpus domains; the official pretrained component was not adapted here. This is not an architecture-only or equal-adaptation comparison. '
            'NYU has only four test children. These are descriptive matched-data results, with no model selection or tuning. '
            'Normalized spoken-word references differ from full original CHAT coding; retraces, morphology and professional review remain separate. '
            'Lexical errors can change through utterance-level unintelligibility exclusions even with unchanged words. '
            'MLUm, WPM, PGU, production errors and mazes remain unavailable without validated corresponding predictions. '
            'MOSS audio transcription, target-speaker identification and audio-to-metric accuracy have not been validated by this experiment.',
            '', 'Files: `results.json` contains per-recording signed/absolute percentage errors, predictions, hashes, coverage and uncertainty; '
            '`batchalign_raw.jsonl` contains official UtSeg inputs/outputs. Protocol and CLAN evidence: `outputs/validation/common_benchmark/`.','']
    (OUT/'REPORT.md').write_text('\n'.join(lines))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['register','evaluate','report'])
    p.add_argument('--software',type=Path,default=Path('/private/tmp/speech_common_benchmark'))
    p.add_argument('--model',type=Path)
    a=p.parse_args()
    if a.stage=='register':register(a.software)
    elif a.stage=='evaluate':
        if a.model is None:p.error('--model is required')
        evaluate(a.software,a.model)
    else:report(json.loads((OUT/'results.json').read_text()))


if __name__=='__main__':main()
