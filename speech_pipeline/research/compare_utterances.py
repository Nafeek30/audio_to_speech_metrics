"""Compare revised utterances with rater labels and saved software references.

Read archived predictions from outputs/validation/method_comparison and write fresh
comparisons to outputs/results/human_comparison. Source workbooks stay intact.
"""
import csv
from collections import defaultdict
import json
from pathlib import Path
import statistics

from openpyxl import load_workbook
import spacy
import speech_pipeline.utterances as auto
from speech_pipeline.utterances import flatten_labels, split_label_cell, plain, boundary_counts, boundary_scores
from speech_pipeline.pipeline import ROOT, digest, get_corrected_filename, lexical, METRICS, save
from tempfile import TemporaryDirectory

OUT = ROOT / 'outputs/validation/method_comparison'
RESULTS = ROOT / 'outputs/results/human_comparison'
MLU = ['total_utterances', 'mlu_nutterances', 'mlu_nwords', 'mlu_words', 'mlu_words_sd']


def write_csv(path, rows):
    if not rows:
        raise ValueError(f'No rows for {path}')
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    new = json.loads((OUT/'analysis/uncorrected/comparison.json').read_text())
    old = json.loads((OUT/'baseline_comparison.json').read_text())
    corrected = json.loads((OUT/'analysis/corrected/comparison.json').read_text())
    index = {(r['patient_id'],r['source_file']): r for r in new['recordings']}
    corr = {(r['patient_id'],r['source_file']): r for r in corrected['recordings']}
    predictions = {}
    for method, report, directory in [('Previous',old,OUT/'baseline_analysis'),
                                       ('Revised',new,OUT/'analysis/uncorrected')]:
        for r in report['recordings']:
            predictions[method,r['patient_id'],r['source_file']] = json.loads((directory/Path(r['utterance_file']).name if method=='Previous' else directory/r['utterance_file']).read_text())['utterances']
    nlp = spacy.load('en_core_web_sm')
    pairs, boundaries, cells, sources, utterance_rows, software_checks, clan_checks = [], [], [], [], [], [], []
    def add(method, reference, pid, filename, metric, x, y):
        if x is None or y is None:
            return
        pairs.append(dict(method=method,reference=reference,patient_id=pid,source_file=filename,
                          metric=metric,pipeline_value=x,reference_value=y,difference=x-y,
                          absolute_error=abs(x-y),absolute_percent_error=100*abs(x-y)/abs(y) if y else None))
    with TemporaryDirectory() as temp:
        for name, label in [('jade','Jade'),('yamen','Yamen'),('all_nafee_utterances','Nafee')]:
            path = ROOT/f'dataset/annotations/{name}.xlsx'
            sources.append(dict(reference=label,source_file=str(path.relative_to(ROOT)),sha256=digest(path)))
            workbook = load_workbook(path,read_only=True,data_only=True)
            for sheet in workbook:
                for number,row in enumerate(list(sheet.values)[1:],2):
                    if row[2] and row[4]:
                        split = split_label_cell(row[4])
                        cells.append(dict(reference=label,patient_id=sheet.title,source_file=str(row[2]),
                                          cell=f'E{number}',utterance_count=len(split),original_text=str(row[4]),
                                          parsed_utterances='\n'.join(split)))
                for filename,texts in flatten_labels(sheet).items():
                    if (sheet.title,filename) not in index:
                        raise ValueError(f'No mapped recording for {sheet.title}/{filename}')
                    reference = [{'text':plain(t)} for t in texts]
                    mlu = auto.word_mlu(reference)
                    ref_lex = lexical(reference,nlp,Path(temp)/'reference.txt')
                    for method in ['Previous','Revised']:
                        us = predictions[method,sheet.title,filename]
                        mine = auto.word_mlu(us)
                        c = boundary_counts(reference,us)
                        boundaries.append(dict(method=method,reference=label,patient_id=sheet.title,source_file=filename,
                                               reference_utterances=len(reference),pipeline_utterances=len(us),**c,**boundary_scores(c)))
                        for metric in MLU:
                            add(method,label,sheet.title,filename,metric,mine[metric],mlu[metric])
                        scores = lexical(us,nlp,Path(temp)/'pipeline.txt') if method=='Previous' else index[sheet.title,filename]['lexical']
                        for metric in METRICS:
                            add(method,label,sheet.title,filename,metric,scores[metric]['custom'],ref_lex[metric]['custom'])
            workbook.close()
        for (pid, filename), r in index.items():
            cfile = get_corrected_filename(pid,filename)
            reference = corr[pid,cfile]
            if reference['clan']['status'] != 'software_reference_imported':
                raise ValueError(f'Fresh CLAN reference missing: {pid}/{cfile}')
            clan = {k:v['reference'] for k,v in reference['clan']['metrics'].items()}
            cpath = OUT/'analysis/corrected'/reference['chat_file']
            sources.append(dict(reference='Corrected CLAN',patient_id=pid,source_file=filename,
                                input_file=str(cpath.relative_to(ROOT)),input_sha256=digest(cpath),
                                output_sha256=reference['clan']['output_sha256']))
            for kind, record in [('Uncorrected',r),('Corrected',reference)]:
                if record['clan']['status'] != 'software_reference_imported':
                    raise ValueError(f'Fresh same-input CLAN check missing: {pid}/{filename}')
                for metric,value in record['clan']['metrics'].items():
                    clan_checks.append(dict(dataset=kind,patient_id=pid,source_file=filename,metric=metric,**value))
            for method in ['Previous','Revised']:
                us = predictions[method,pid,filename]
                mlu = auto.word_mlu(us)
                scores = lexical(us,nlp,Path(temp)/'pipeline.txt') if method=='Previous' else r['lexical']
                for metric in METRICS:
                    add(method,'Corrected TAALED',pid,filename,metric,scores[metric]['custom'],reference['lexical'][metric]['reference'])
                for metric,value in clan.items():
                    add(method,'Corrected CLAN',pid,filename,metric,mlu[metric],value)
                # Calculator agreement is assessed on exactly the same text.
                for metric,value in scores.items():
                    software_checks.append(dict(method=method,patient_id=pid,source_file=filename,metric=metric,**value))
            for u in predictions['Revised',pid,filename]:
                utterance_rows.append(dict(patient_id=pid,source_file=filename,utterance_id=u['utterance_id'],
                                          text=u['text'],source_rows=','.join(map(str,u['source_rows'])),
                                          source_start=u['source_start'],source_end=u['source_end'],
                                          boundary_reason=u['boundary_reason'],include_mlu=u['scoring']['include_mlu'],
                                          mlu_words=u['scoring']['word_count'],excluded_fillers=' '.join(u['scoring']['excluded_fillers'])))
    groups = defaultdict(list)
    for r in pairs:
        groups[r['method'],r['reference'],r['metric']].append(r)
    summary = []
    for (method,reference,metric), rs in sorted(groups.items()):
        by_patient = defaultdict(list)
        for r in rs:
            if r['absolute_percent_error'] is not None:
                by_patient[r['patient_id']].append(r['absolute_percent_error'])
        summary.append(dict(method=method,reference=reference,metric=metric,n_recordings=len(rs),
                            n_participants=len({r['patient_id'] for r in rs}),
                            pipeline_mean=statistics.mean(r['pipeline_value'] for r in rs),
                            reference_mean=statistics.mean(r['reference_value'] for r in rs),
                            mean_difference=statistics.mean(r['difference'] for r in rs),
                            mean_absolute_error=statistics.mean(r['absolute_error'] for r in rs),
                            participant_balanced_mape=statistics.mean(statistics.mean(v) for v in by_patient.values()) if by_patient else None))
    bgroups = defaultdict(list)
    for r in boundaries:
        bgroups[r['method'],r['reference']].append(r)
    pooled = []
    for (method,reference),rs in sorted(bgroups.items()):
        keys = ['tp','tn','fn','fp','reference_tokens','pipeline_tokens','reference_possible_gaps','pipeline_possible_gaps','word_edits']
        counts = {k:sum(r[k] for r in rs) for k in keys}
        pooled.append(dict(method=method,reference=reference,n_recordings=len(rs),
                           n_participants=len({r['patient_id'] for r in rs}),**counts,**boundary_scores(counts)))
    for name,rows in [('metric_pairs',pairs),('metric_summary',summary),('boundary_recordings',boundaries),
                      ('boundary_summary',pooled),('label_cell_audit',cells),('revised_utterances',utterance_rows),
                      ('taaled_same_text_checks',software_checks),('clan_same_text_checks',clan_checks)]:
        write_csv(RESULTS/(name+'.csv'),rows)
    save(RESULTS/'comparison_summary.json',dict(metrics=summary,boundaries=pooled,
        policy=auto.POLICY,notes=[
            'Previous and revised boundaries scored under the same filler exclusion policy.',
            'Filler words and filler-only utterances retained in boundary comparisons and total utterances.',
            'MLU excludes listed fillers and filler-only utterances; repetitions retained.',
            'Rater cells split at explicit terminal punctuation and line breaks, including lowercase continuations.',
            'Boundary statistics use only exactly aligned adjacent word pairs; gap coverage reported.',
            'Corrected software reference scope: '+corrected['scope'],
            'CLAN input and output hashes verified during import. Corrected references use the returned CLAN run; uncorrected same-input checks use the saved official CLAN run.',
            'Software calculator agreement on shared text is reported separately from raw versus corrected transcript agreement.',
            'Nafee manual annotations are the primary human reference in all_nafee_utterances.xlsx.',
            'Retrospective development-set comparisons; rules were refined using these recordings.'],
        sources=sources,code_sha256={name:digest(ROOT/name) for name in ['speech_pipeline/utterances.py','speech_pipeline/research/compare_utterances.py']}))
    for r in pooled:
        print(r['method'],r['reference'],'F1',round(r['boundary_f1'],4),'kappa',round(r['kappa'],4))
    for r in summary:
        if r['metric'] in ['total_utterances','mlu_words','mlu_nutterances']:
            print(r['method'],r['reference'],r['metric'],'MAPE',round(r['participant_balanced_mape'],2))


if __name__ == '__main__':
    main()
