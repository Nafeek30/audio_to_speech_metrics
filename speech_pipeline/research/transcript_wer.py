"""Recompute transcript WER tables from the paired source transcripts."""
import csv
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[2]
from speech_pipeline.utterances import align
from speech_pipeline.utterances import read_rows, WORD
from speech_pipeline.pipeline import get_corrected_filename

OUT=ROOT/'outputs/results/paper'

def csv_read(p):
    return list(csv.DictReader(p.open(encoding='utf-8-sig')))

def csv_write(p,rows):
    with p.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def normalized(text):
    return [w.casefold().replace('’',"'") for w in WORD.findall(text)]

recordings=[]
mapping=csv_read(ROOT/'dataset/target_speakers.csv')
for m in mapping:
    pid,name,speaker=(m[k] for k in ['patient_id','source_file','target_speaker'])
    raw=ROOT/'dataset/uncorrected'/pid/name
    corrected=ROOT/'dataset/corrected'/pid/get_corrected_filename(pid,name)
    # WER retains fillers and repetitions. It is independent of utterance segmentation.
    raw_text=' '.join(r['text'] for r in read_rows(raw) if r['speaker'].casefold()==speaker.casefold())
    corrected_text=corrected.read_text()
    reference,hypothesis=normalized(corrected_text),normalized(raw_text)
    assert reference and hypothesis
    _,edits=align(reference,hypothesis)
    edit_count=sum(edits.values())
    recordings.append(dict(patient_id=pid,source_file=name,corrected_file=corrected.name,
                           target_speaker=speaker,reference_words=len(reference),uncorrected_words=len(hypothesis),
                           **edits,word_edits=edit_count,wer_percent=100*edit_count/len(reference),
                           raw_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),
                           corrected_sha256=hashlib.sha256(corrected.read_bytes()).hexdigest()))
groups=defaultdict(list)
for r in recordings:groups[r['patient_id']].append(r)
participants=[]
for pid,rows in sorted(groups.items(),key=lambda x:int(x[0][1:])):
    participants.append(dict(patient_id=pid,n_recordings=len(rows),
                             mean_recording_wer_percent=statistics.mean(r['wer_percent'] for r in rows),
                             pooled_wer_percent=100*sum(r['word_edits'] for r in rows)/sum(r['reference_words'] for r in rows),
                             **{k:sum(r[k] for r in rows) for k in ['reference_words','uncorrected_words','substitutions','deletions','insertions','word_edits']}))
wer=dict(n_recordings=len(recordings),n_participants=len(participants),
         participant_balanced_wer_percent=statistics.mean(r['mean_recording_wer_percent'] for r in participants),
         pooled_wer_percent=100*sum(r['word_edits'] for r in recordings)/sum(r['reference_words'] for r in recordings),
         minimum_recording_wer_percent=min(r['wer_percent'] for r in recordings),
         maximum_recording_wer_percent=max(r['wer_percent'] for r in recordings),
         **{k:sum(r[k] for r in recordings) for k in ['reference_words','uncorrected_words','substitutions','deletions','insertions','word_edits']})
assert wer['word_edits']==wer['substitutions']+wer['deletions']+wer['insertions']

OUT.mkdir(parents=True, exist_ok=True)
csv_write(OUT/'transcript_wer_recordings.csv', recordings)
csv_write(OUT/'transcript_wer_participants.csv', participants)
print(json.dumps(wer, indent=2))
