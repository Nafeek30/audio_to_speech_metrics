"""Compare frozen paragraph models with existing ten-participant references.

No training, selection or publication-data writes. Run after the canonical
metric-selected paragraph pipeline. Preserve the previous canonical summaries
in OUT before replacing them, if full-recording incumbent results are wanted.
"""
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import spacy
from openpyxl import load_workbook
from speech_pipeline import ROOT, utterances as auto
from speech_pipeline.metrics import clinical_from_predictions
from speech_pipeline.pipeline import digest, lexical, save

OUT = ROOT / 'outputs/results/participant_pipeline_comparison'
HUMAN = ROOT / 'outputs/results/human_comparison'


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(name, rows):
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(exist_ok=True)
    source = ROOT / 'dataset/annotations/all_nafee_utterances.xlsx'
    sources = [source, HUMAN / 'participant_sample_pairs.csv', HUMAN / 'metric_pairs.csv']
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in sources}
    final = json.loads((ROOT / 'outputs/results/benchmark/improved_model.json').read_text())['metric_round']
    folder = ROOT / 'outputs/results/pipeline/uncorrected'
    fresh = json.loads((folder / 'comparison.json').read_text())
    assert fresh['segmenter']['selected'] == final['selected']['name']
    assert fresh['segmenter']['artifacts']['model.safetensors'] == final['selected']['weights_sha256']
    fresh_index = {(r['patient_id'], r['source_file']): r for r in fresh['recordings']}
    incumbent = {(r['participant'], r['recording']): r for r in final['incumbent']['private_paragraphs']['boundaries']['recordings']}
    frozen = {(r['participant'], r['recording']): r for r in final['private_paragraphs']['boundaries']['recordings']}
    old_folder = ROOT / 'outputs/validation/method_comparison/analysis/uncorrected'
    established = json.loads((old_folder / 'comparison.json').read_text())
    old_index = {(r['patient_id'], r['source_file']): r for r in established['recordings']}
    wb = load_workbook(source, read_only=True, data_only=True)
    nlp = spacy.load('en_core_web_sm')
    pairs, boundaries, manifest = [], [], []

    def add(scope, method, pid, filename, metric, value, truth):
        available = value is not None and truth is not None
        pairs.append(dict(scope=scope, method=method, participant=pid, recording=filename,
                          metric=metric, reference=truth, predicted=value,
                          absolute_error=abs(value-truth) if available else None,
                          ape=100*abs(value-truth)/abs(truth) if available and truth else None))

    with TemporaryDirectory(prefix='participant_comparison_') as temp:
        def score(us):
            result = {f'normalized/{m}': v['custom'] for m, v in lexical(us, nlp, Path(temp)/'sample.txt').items()}
            result.update({f'surface/{m}': v for m, v in clinical_from_predictions(us)['values'].items()})
            result.update({m: v for m, v in auto.word_mlu(us).items() if m in ('total_utterances', 'mlu_words', 'mlu_words_sd')})
            return result

        for sheet in wb:
            pooled = defaultdict(list)
            files = auto.flatten_labels(sheet)
            for name, texts in files.items():
                key = sheet.title, name
                ref = [dict(text=auto.plain(t)) for t in texts]
                record = fresh_index[key]
                assert digest(ROOT/'dataset/uncorrected'/sheet.title/name) == record['raw_sha256']
                selected = json.loads((folder / record['utterance_file']).read_text())['utterances']
                assert [u['text'] for u in selected] == frozen[key]['utterance_texts']
                previous = [dict(text=t) for t in incumbent[key]['utterance_texts']]
                rules = json.loads((old_folder / old_index[key]['utterance_file']).read_text())['utterances']
                assert auto.stream(selected)[0] == auto.stream(previous)[0] == auto.stream(rules)[0]
                pooled['human'].extend(ref)
                for method, us in [('Established rules', rules), ('Previous paragraph', previous), ('New paragraph', selected)]:
                    pooled[method].extend(us)
                    counts = auto.boundary_counts(ref, us)
                    boundaries.append(dict(method=method, participant=sheet.title, recording=name,
                                           **counts, **auto.boundary_scores(counts)))
                manifest.append(dict(participant=sheet.title, recording=name, manual_utterances=len(ref),
                                     new_utterances=len(selected), source_words_preserved=True,
                                     fresh_matches_frozen=True, raw_sha256=record['raw_sha256']))
            truth = score(pooled['human'])
            for method in ['Established rules', 'Previous paragraph', 'New paragraph']:
                values = score(pooled[method])
                for metric, reference in truth.items():
                    add('Manual complete participant samples', method, sheet.title,
                        f'{len(files)} labeled recordings pooled', metric, values[metric], reference)
            print('Compared manual sample', sheet.title, flush=True)
    wb.close()
    assert len(manifest) == 14 and len({r['participant'] for r in manifest}) == 10
    # Reproduce the existing publication's complete-sample values and references.
    saved = {(r['patient_id'], r['metric']): r for r in read_csv(HUMAN/'participant_sample_pairs.csv') if r['method']=='Revised'}
    for row in pairs:
        metric = row['metric'].removeprefix('normalized/')
        if row['method']=='Established rules' and (row['participant'], metric) in saved:
            prior = saved[row['participant'], metric]
            assert math.isclose(row['reference'], float(prior['reference_value']), abs_tol=1e-9)
            assert math.isclose(row['predicted'], float(prior['pipeline_value']), abs_tol=1e-9)
    # Same preserved corrected-transcript references for all 37 ASR recordings.
    reference_rows = [r for r in read_csv(HUMAN/'metric_pairs.csv') if r['method']=='Revised' and r['reference'] in {'Corrected CLAN','Corrected TAALED'}]
    old = json.loads((OUT/'previous_paragraph_uncorrected.json').read_text())
    old = {(r['patient_id'],r['source_file']): r for r in old['recordings']}
    for r in reference_rows:
        key = r['patient_id'], r['source_file']; metric = r['metric']
        truth = float(r['reference_value'])
        add(r['reference'], 'Established rules', *key, metric, float(r['pipeline_value']), truth)
        for method, index in [('Previous paragraph', old), ('New paragraph', fresh_index)]:
            record = index[key]
            value = record['mlu'].get(metric) if r['reference']=='Corrected CLAN' else record['lexical'][metric]['custom']
            add(r['reference'], method, *key, metric, value, truth)
    groups = defaultdict(list)
    for row in pairs:
        if row['ape'] is not None:
            groups[row['scope'], row['method'], row['metric']].append(row)
    summary = []
    for (scope, method, metric), rows in groups.items():
        people = defaultdict(list)
        for row in rows:
            people[row['participant']].append(row['ape'])
        errors = [statistics.mean(values) for values in people.values()]
        summary.append(dict(scope=scope, method=method, metric=metric,
                            participants=len(people), pairs=len(rows),
                            mean_participant_error=statistics.mean(errors),
                            median_participant_error=statistics.median(errors)))
    for name, rows in [('metric_pairs.csv', pairs), ('metric_summary.csv', summary),
                       ('boundary_recordings.csv', boundaries), ('recording_manifest.csv', manifest)]:
        write_csv(name, rows)
    participant_values = []
    for pid in sorted({r['participant'] for r in manifest}, key=lambda p: int(p[1:])):
        row = dict(participant=pid)
        for metric in ('total_utterances', 'mlu_words', 'mlu_words_sd'):
            for method in ('Established rules', 'Previous paragraph', 'New paragraph'):
                item = next(r for r in pairs if r['scope']=='Manual complete participant samples' and r['participant']==pid and r['metric']==metric and r['method']==method)
                row[f'{metric}/human'] = item['reference']
                row[f'{metric}/{method}'] = item['predicted']
                row[f'{metric}/{method}/APE'] = item['ape']
        participant_values.append(row)
    write_csv('participant_values.csv', participant_values)
    for filename, expected in hashes.items():
        assert digest(ROOT/filename) == expected
    save(OUT/'comparison.json', dict(selected=final['selected'], sources=hashes,
         pipeline_report_sha256=digest(folder/'comparison.json'), summary=summary,
         runtime_verification=final.get('runtime_verification'),
         scope='Frozen models; manual references cover 14 recordings / 10 participants; corrected software references cover 37 recordings / 10 participants. No tuning.',
         aggregation='Manual: pool labeled recordings within participant without merging utterances. Corrected software: mean recording APE within participant. Summaries report mean and median across participants.',
         lexical_policy='normalized uses existing spaCy/TAALED preprocessing; surface uses documented filler/unintelligibility policy. These are distinct measures.'))
    lines = ['# New pipelines versus existing ten-participant results', '',
             'The established pipeline is substantially closer to the manual participant values than either paragraph model. The new paragraph model oversegments all ten participants, reducing MLUw and length SD. The earlier modest improvement in the corpus-balanced mean recording objective does not imply improvement in these pooled-participant median errors.', '',
             'All three methods use the same ASR source recordings. Established rules retain observed speaker turns; paragraph models withhold those cuts. Models were frozen before this comparison; no training or selection used these results.', '',
             'The manual comparison pools the same 14 labeled recordings within each of the 10 participants, matching the existing complete-sample analysis. The corrected CLAN/TAALED comparison uses all 37 recordings and the original saved reference values; it includes transcription and reference-boundary differences.', '',
             'Errors below are median participant APE, in percent; lower is better. For corrected references, first average recording APE within each participant.', '',
             '| Reference / metric | Established rules | Previous paragraph | New paragraph |',
             '| --- | ---: | ---: | ---: |']
    for scope, metrics in [('Manual complete participant samples', ['total_utterances','mlu_words','mlu_words_sd','normalized/basic_ntokens','normalized/basic_ntypes','normalized/simple_ttr_aw','normalized/mattr50_aw']), ('Corrected CLAN', ['mlu_nutterances','mlu_words','mlu_words_sd']), ('Corrected TAALED', ['basic_ntokens','basic_ntypes','simple_ttr_aw','mattr50_aw'])]:
        for metric in metrics:
            cells = [next(r for r in summary if r['scope']==scope and r['method']==m and r['metric']==metric)['median_participant_error'] for m in ['Established rules','Previous paragraph','New paragraph']]
            lines.append(f'| {scope}: {metric} | '+' | '.join(f'{v:.2f}' for v in cells)+' |')
    lines += ['', '## Individual participants: human → new pipeline (APE)', '',
              '| Participant | Utterance count | MLUw | Length SD |',
              '| --- | ---: | ---: | ---: |']
    for row in participant_values:
        cells = []
        for metric in ('total_utterances', 'mlu_words', 'mlu_words_sd'):
            a, b, error = row[f'{metric}/human'], row[f'{metric}/New paragraph'], row[f'{metric}/New paragraph/APE']
            cells.append(f'{a:.2f} → {b:.2f} ({error:.2f}%)')
        lines.append(f'| {row["participant"]} | '+' | '.join(cells)+' |')
    lines += ['', 'Each participant’s reference, predicted value, signed direction (recoverable from predicted minus reference), absolute error and APE are in `metric_pairs.csv`; all measures and both mean/median summaries are in `metric_summary.csv`. Exact per-record boundary counts are in `boundary_recordings.csv`.', '',
              'Source-word preservation and exact agreement with the previously frozen new-model predictions were checked for all 14 labeled recordings. Native CLAN agreement verifies arithmetic on predicted text, not agreement with human boundaries. Missing morphology, timing, grammaticality, error and maze annotations remain unavailable. Historical benchmark exposure prevents a fresh confirmatory claim.', '']
    if final.get('runtime_verification'):
        lines += ['', 'The full new-model run completed for 37 ASR and 37 corrected recordings, covering all 10 participants in each condition. Runtime auditing verified all 74 native CLAN checks within displayed precision, source-word preservation, 17 TAALED comparisons per recording, and exact reproduction of all 14 frozen labeled-recording predictions. TAALED preprocessing differences are retained rather than described as exact calculator agreement. Publication files were unchanged.', '']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
