"""Recompute the complete sampled speech for each participant against Nafee.

All manual boundaries come from column E of all_nafee_utterances.xlsx.
Recording-level comparisons remain separate from pooled participant samples.
Use the project's existing NLP environment for spaCy and TAALED scoring.
"""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import statistics
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]
import spacy
from openpyxl import load_workbook
import speech_pipeline.utterances as auto
from speech_pipeline.utterances import flatten_labels
from speech_pipeline.utterances import boundary_counts, boundary_scores, plain
from speech_pipeline.pipeline import lexical

SOURCE = ROOT / 'dataset/annotations/all_nafee_utterances.xlsx'
OUT = ROOT / 'outputs/results/human_comparison'
METRICS = ['basic_ntokens', 'basic_ntypes', 'basic_ncontent_tokens',
           'basic_ncontent_types', 'basic_nfunction_tokens', 'basic_nfunction_types',
           'simple_ttr_aw', 'mattr50_aw', 'total_utterances', 'mlu_words', 'mlu_words_sd']

def write_csv(name, rows):
    with (OUT / name).open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    original_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    report = json.loads((ROOT / 'outputs/validation/method_comparison/analysis/uncorrected/comparison.json').read_text())
    index = {(r['patient_id'], r['source_file']): r for r in report['recordings']}
    wb = load_workbook(SOURCE, read_only=True, data_only=True)
    nlp = spacy.load('en_core_web_sm')
    pairs, participants, verification = [], [], []
    with TemporaryDirectory() as temp:
        for sheet in wb:
            files = flatten_labels(sheet)
            manual = [{'text': plain(t)} for values in files.values() for t in values]
            automatic, previous = [], []
            for filename, texts in files.items():
                rec = index[sheet.title, filename]
                raw = ROOT / 'dataset/uncorrected' / sheet.title / filename
                assert hashlib.sha256(raw.read_bytes()).hexdigest() == rec['raw_sha256']
                predicted = auto.predict(list(auto.read_rows(raw)), rec['target_speaker'], nlp)
                saved = json.loads((ROOT / 'outputs/validation/method_comparison/analysis/uncorrected' / rec['utterance_file']).read_text())['utterances']
                assert [u['text'] for u in predicted] == [u['text'] for u in saved]
                old = json.loads((ROOT / 'outputs/validation/method_comparison/baseline_analysis' / Path(rec['utterance_file']).name).read_text())['utterances']
                reference = [{'text': plain(t)} for t in texts]
                counts = boundary_counts(reference, predicted)
                assert counts['word_edits'] == 0
                assert boundary_scores(counts)['reference_gap_coverage'] == 1
                verification.append(dict(patient_id=sheet.title, source_file=filename,
                                         fresh_predictions_equal_saved=True, raw_sha256=rec['raw_sha256'],
                                         manual_utterances=len(reference), automatic_utterances=len(predicted),
                                         **counts, **boundary_scores(counts)))
                automatic.extend(predicted)
                previous.extend(old)
            def score(utterances):
                lexical_scores = lexical(utterances, nlp, Path(temp) / 'sample.txt')
                return {**{k: lexical_scores[k]['custom'] for k in METRICS[:8]},
                        **auto.word_mlu(utterances)}
            truth = score(manual)
            current = score(automatic)
            baseline = score(previous)
            for method, values in [('Previous', baseline), ('Revised', current)]:
                for metric in METRICS:
                    x, y = values[metric], truth[metric]
                    pairs.append(dict(method=method, reference='Nafee', patient_id=sheet.title,
                                      sample_unit='All sampled utterances combined within participant',
                                      n_recordings=len(files), metric=metric, pipeline_value=x,
                                      reference_value=y, difference=x-y, absolute_error=abs(x-y),
                                      absolute_percent_error=100*abs(x-y)/abs(y) if y else None))
            participants.append(dict(patient_id=sheet.title, n_recordings=len(files),
                worksheet_auto_rows=sum(bool(row[2]) and bool(row[3]) for row in list(sheet.values)[1:]),
                manual_total=truth['total_utterances'], pipeline_total=current['total_utterances'],
                manual_eligible=truth['mlu_nutterances'], pipeline_eligible=current['mlu_nutterances'],
                manual_mlu=truth['mlu_words'], pipeline_mlu=current['mlu_words'],
                manual_sd=truth['mlu_words_sd'], pipeline_sd=current['mlu_words_sd'],
                sd_ape=100*abs(current['mlu_words_sd']-truth['mlu_words_sd'])/truth['mlu_words_sd']))
    wb.close()
    assert len(participants) == 10 and len(verification) == 14
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == original_hash
    write_csv('participant_sample_pairs.csv', pairs)
    write_csv('participant_complete_samples.csv', participants)
    write_csv('fresh_prediction_verification.csv', verification)
    groups = defaultdict(list)
    for row in pairs:
        groups[row['method'], row['metric']].append(row)
    summary = [dict(method=method, metric=metric, n_participants=len(rows),
                    pipeline_mean=statistics.mean(r['pipeline_value'] for r in rows),
                    reference_mean=statistics.mean(r['reference_value'] for r in rows),
                    mape=statistics.mean(r['absolute_percent_error'] for r in rows))
               for (method, metric), rows in groups.items()]
    write_csv('participant_sample_summary.csv', summary)
    result = dict(primary_rater='Nafee', primary_source=str(SOURCE.relative_to(ROOT)),
                  primary_source_sha256=original_hash, participants=participants,
                  n_recordings=len(verification), manual_utterances=sum(r['manual_total'] for r in participants),
                  pipeline_utterances=sum(r['pipeline_total'] for r in participants),
                  worksheet_auto_rows=sum(r['worksheet_auto_rows'] for r in participants),
                  aggregation='Combine each participant\'s sampled utterances without merging boundaries across recordings.',
                  summary=summary)
    (OUT / 'complete_sample_analysis.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
