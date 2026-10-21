"""Score the predeclared first public audio task; never tune from its results."""
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from speech_pipeline import ROOT,utterances as auto
from speech_pipeline.corpora import read_chat,run_clan
from speech_pipeline.metrics import clinical_from_predictions
from speech_pipeline.moss import parse_transcript,run,faithful_chat
from speech_pipeline.pipeline import save,digest

OUT=ROOT/'outputs/results/moss_pilot'
EVIDENCE=ROOT/'outputs/validation/moss_pilot'


def normalized(text):
    return [w.casefold().replace('’',"'") for w in auto.WORD.findall(text)]


def oracle_speaker_mapping(reference,predicted):
    """Evaluation assignment across all speakers, not patient identification."""
    refs=sorted(reference);hyps=sorted(predicted);n=max(len(refs),len(hyps))
    costs=np.zeros((n,n),dtype=np.int64);edits={}
    for i in range(n):
        for j in range(n):
            a=reference[refs[i]] if i<len(refs) else []
            b=predicted[hyps[j]] if j<len(hyps) else []
            _,e=auto.align(a,b);costs[i,j]=sum(e.values());edits[i,j]=e
    left,right=linear_sum_assignment(costs)
    assignments=[]
    for i,j in zip(left,right):
        assignments.append(dict(reference_speaker=refs[i] if i<len(refs) else None,
            predicted_speaker=hyps[j] if j<len(hyps) else None,edits=edits[i,j],
            reference_words=len(reference[refs[i]]) if i<len(refs) else 0,
            predicted_words=len(predicted[hyps[j]]) if j<len(hyps) else 0))
    total=sum(len(v) for v in reference.values())
    return dict(assignments=assignments,word_edits=int(costs[left,right].sum()),reference_words=total,
                cpwer_percent=100*int(costs[left,right].sum())/total if total else None,
                role='Oracle minimum-edit mapping for evaluation only; automatic target-role identification is unvalidated')


def metric_values(us):
    a=auto.word_mlu(us);b=clinical_from_predictions(us)['values']
    return dict(total_units=a['total_utterances'],eligible_mlu_utterances=a['mlu_nutterances'],
        MLUw=a['mlu_words'],length_sd=a['mlu_words_sd'],mlu_words=a['mlu_nwords'],
        **{f'freq/{k}':b[k] for k in ('TNW','NDW','TTR','MATTR')})


def main():
    if (OUT/'evaluation.json').exists():raise FileExistsError('Preserve completed pilot')
    plan=json.loads((EVIDENCE/'protocol.json').read_text())
    if not (EVIDENCE/'scoring_protocol.json').exists():raise ValueError('Scoring plan required')
    raw=json.loads((OUT/'moss_raw.json').read_text())
    if raw['finish_reason']!='eos':raise ValueError('Do not score a truncated generation')
    segments=parse_transcript(raw['text'])
    temporal=dict(segments=len(segments),first_start=min(r['start'] for r in segments),
        last_end=max(r['end'] for r in segments),audio_seconds=plan['audio_seconds'],
        outside_audio=[i for i,r in enumerate(segments) if r['end']>plan['audio_seconds']+.05],
        out_of_start_order=[i for i in range(1,len(segments)) if segments[i]['start']<segments[i-1]['start']],
        scope='Predicted segment envelopes, not reviewed word timings or DER reference')
    source=ROOT/'dataset/public/ocsc/transcripts/4/4002.cha'
    if digest(source)!=plan['reference_sha256']:raise ValueError('Reference changed')
    sections=[]
    for row in read_chat(source):
        if not sections or sections[-1][0]['gem']!=row['gem']:sections.append([])
        sections[-1].append(row)
    reference_rows=sections[4]
    gold=defaultdict(list);predicted=defaultdict(list)
    for row in reference_rows:gold[row['speaker']]+=normalized(row['text'])
    for row in segments:predicted[row['speaker']]+=normalized(row['text'])
    mapping=oracle_speaker_mapping(gold,predicted)
    save(EVIDENCE/'speaker_mapping.json',mapping)
    target=next(r for r in mapping['assignments'] if r['reference_speaker']=='CHI')
    if target['predicted_speaker'] is None:
        save(OUT/'evaluation.json',dict(status='missing_target_speaker',mapping=mapping));return
    identity=target['predicted_speaker']
    result=run(OUT/'moss_raw.json',OUT/'latest_minilm',identity,
        'Evaluation-only oracle mapping to CHAT CHI by minimum total speaker-permutation word edit cost. Not automatic patient identification.',
        verify_clan=True)
    native=json.loads((ROOT/'outputs/results/common_benchmark/original_chat_sensitivity.json').read_text())['references'][0]
    if native['recording']!='transcripts/4/4002.cha#section5':raise ValueError('Native reference mismatch')
    candidates=dict(latest_minilm=result['utterances'],
        moss_source_segments=[dict(text=r['text']) for r in segments if r['speaker']==identity])
    def native_values(mlu,freq):
        return dict(eligible_mlu_utterances=mlu['utterances'],MLUw=mlu['total']/mlu['utterances'] if mlu['utterances'] else None,
            length_sd=mlu['sd'],mlu_words=mlu['total'],
            **{f'freq/{k}':(freq['NDW']/freq['TNW'] if freq['TNW'] else None) if k=='TTR'
               else freq[k] if k!='MATTR' or freq['TNW']>=50 else None for k in ('TNW','NDW','TTR','MATTR')})
    reference_metrics=native_values(native['official_outputs']['mlu'],native['official_outputs']['freq'])
    from tempfile import TemporaryDirectory
    native_predictions={}
    for method,us in candidates.items():
        with TemporaryDirectory(prefix='moss_native_pred_') as temp:
            chat=Path(temp)/'full_words.cha';body=faithful_chat(us,'MOSS pilot',identity,'Evaluation-only oracle role mapping')
            chat.write_text(body);mlu=run_clan('mlu',chat);freq=run_clan('freq',chat,window=50)
            native_predictions[method+'_native_clan']=dict(values=native_values(mlu,freq),original_chat=body,official_outputs=dict(mlu=mlu,freq=freq))
    pairs=[]
    for method,us in candidates.items():
        mine=metric_values(us)
        for metric,reference in reference_metrics.items():
            value=mine[metric]
            pairs.append(dict(method=method,metric=metric,reference=reference,predicted=value,
                signed_error=value-reference if value is not None and reference is not None else None,
                ape=100*abs(value-reference)/abs(reference) if value is not None and reference else None))
    for method,item in native_predictions.items():
        for metric,reference in reference_metrics.items():
            value=item['values'][metric]
            pairs.append(dict(method=method,metric=metric,reference=reference,predicted=value,
                signed_error=value-reference if value is not None and reference is not None else None,
                ape=100*abs(value-reference)/abs(reference) if value is not None and reference else None))
    unknown=any(w in {'xxx','yyy','www'} for words in gold.values() for w in words)
    edits=target['edits'];target_wer=100*sum(edits.values())/target['reference_words'] if target['reference_words'] else None
    score=dict(status='completed_single_sample_pilot',protocol_sha256=digest(EVIDENCE/'protocol.json'),
        scoring_protocol_sha256=digest(EVIDENCE/'scoring_protocol.json'),raw_output_sha256=digest(OUT/'moss_raw.json'),
        audio_seconds=plan['audio_seconds'],mapping=mapping,target_speaker=identity,temporal_audit=temporal,
        target_wer_percent=None if unknown else target_wer,
        cpwer_percent=None if unknown else mapping['cpwer_percent'],
        unavailable_reason='Gold unintelligibility placeholders prevent ordinary WER' if unknown else None,
        metrics=pairs,native_reference=reference_metrics,native_predictions=native_predictions,calculator_check=result['clan'],
        native_precision='MLUw and TTR reconstructed from exact native counts to avoid false zero errors from rounded ratios. Native SD/MATTR retain three-decimal output precision.',
        boundary_f1=None,boundary_reason='ASR changes the word stream; no partial aligned-gap F1 used',
        caveat='One preselected exposed test child/task, oracle speaker-role mapping. No population estimate, published-score ranking or validated automatic patient identification. FREQ/clinical eligibility policy differences retained.')
    save(OUT/'evaluation.json',score)
    import csv
    with (OUT/'metric_pairs.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(pairs[0]));w.writeheader();w.writerows(pairs)
    lines=['# First MOSS public-audio pilot','',
        'The first frozen public test task was selected before generation: OCSC 4002, ExpPictures, '
        f"{plan['audio_seconds']:.3f} seconds. MOSS received audio and its official default prompt only. "
        'The existing latest MiniLM then segmented its unpunctuated target-speaker words. No model was changed.','',
        f"Oracle target speaker: **{identity}**. Target WER: **{score['target_wer_percent']:.2f}%**; speaker-permutation cpWER: **{score['cpwer_percent']:.2f}%**."
        if not unknown else 'Ordinary WER unavailable because the reference contains unintelligibility placeholders.',
        '', 'Speaker roles were assigned using the gold-reference minimum-edit permutation for evaluation. '
        'This does not validate automatic patient identification. The MOSS speaker IDs themselves were predicted from audio.','',
        '| Method | Metric | Original CHAT/native CLAN | Predicted | Absolute error % |','|---|---|---:|---:|---:|']
    for r in pairs:
        fmt=lambda v:f'{v:.3f}' if v is not None else 'unavailable'
        lines.append('| '+' | '.join([r['method'],r['metric'],fmt(r['reference']),fmt(r['predicted']),fmt(r['ape'])])+' |')
    lines+=['','Native references preserve original professional CHAT coding. Count means eligible MLU utterances. MLUw and TTR are reconstructed from exact native counts; SD and MATTR retain three-decimal native precision. Methods suffixed `_native_clan` use identical native MLU/FREQ programs on full recognized-word CHAT; other rows use the normalized clinical surface profile. '
        'FREQ vs clinical surface-word differences include coding/eligibility differences. '
        'MOSS source segments have audio context; they are a different-input baseline from the paragraph model. '
        'No word-gap boundary F1 is reported on different ASR word streams. '
        'The faithful provisional CHAT and separate normalized calculator input are in `latest_minilm/`. '
        'No precise utterance timestamps, MOR, grammar, maze or production-error labels were fabricated.','',
        '**This is one feasibility pilot, not a validated cohort result or an exact published-result replication.** '
        'Next: freeze a multi-participant audio manifest, specify reviewed target mapping and common word/CHAT eligibility, '
        'then compare identical recordings with an established audio system and published reference metrics.','']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    print('Target WER',score['target_wer_percent'],'cpWER',score['cpwer_percent'],flush=True)


if __name__=='__main__':main()
