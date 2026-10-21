"""Import canonical MOSS output into target-speaker CHAT and language metrics.

MOSS speaker IDs are anonymous. Require an explicit target mapping; never infer
the patient from talk time. Segment timestamps remain source envelopes, since
MOSS does not supply reviewed word timings for newly split utterances.
"""
import argparse
import json
import math
from pathlib import Path
import re

from speech_pipeline import utterances as auto
from speech_pipeline.metrics import clinical_from_predictions
from speech_pipeline.pipeline import digest, save

SEGMENT = re.compile(r'\[(\d+(?:\.\d+)?)\]\[(S\d+)\](.*?)\[(\d+(?:\.\d+)?)\]',re.S)


def parse_transcript(text):
    """Reject incomplete generations and unparsed text instead of dropping it."""
    if not isinstance(text,str) or not text.strip():raise ValueError('Empty MOSS transcript')
    rows=[];end=0
    for m in SEGMENT.finditer(text):
        if text[end:m.start()].strip():raise ValueError('Unparsed MOSS content; review complete raw output')
        start,stop=float(m[1]),float(m[4]);spoken=m[3].strip()
        if not math.isfinite(start) or not math.isfinite(stop) or stop<=start:
            raise ValueError('MOSS timestamps must have positive finite duration')
        if not spoken:raise ValueError('Empty MOSS speech segment requires review')
        rows.append(dict(row=len(rows)+2,speaker=m[2],text=spoken,start=start,end=stop))
        end=m.end()
    if not rows or text[end:].strip():raise ValueError('Incomplete or unsupported MOSS transcript')
    return rows


def target_rows(rows,speaker):
    if not re.fullmatch(r'S\d+',speaker):raise ValueError('Use an explicit MOSS speaker ID such as S01')
    target=[r for r in rows if r['speaker']==speaker]
    if not target:raise ValueError(f'Target {speaker} absent; available IDs: {sorted({r["speaker"] for r in rows})}')
    if any(re.search(r'[\[\]<>@&|\\\x15]',r['text']) for r in target):
        raise ValueError('Target contains acoustic/annotation tags; reviewed spoken-word extraction is required')
    return target


def faithful_chat(utterances,source,speaker,mapping_note):
    """Preserve every recognized word, including fillers and repetitions."""
    def line(s):return str(s).replace('\n',' ').replace('\r',' ')
    header=['@UTF8','@Begin','@Languages:\teng','@Participants:\tPAR Target Participant',
            '@ID:\teng|speechpipeline|PAR|||||Participant|||',
            f'@Comment:\tMOSS source: {line(source)}; selected anonymous speaker: {speaker}.',
            f'@Comment:\tTarget mapping evidence: {line(mapping_note)}',
            '@Comment:\tAutomatic unreviewed transcript and boundaries; no inferred diagnosis or MOR.',
            '@Comment:\tFillers and repetitions retained. No invented retrace, maze, or grammar coding.',
            '@Comment:\tNo utterance bullets: source segment envelopes are not word alignments.']
    for u in utterances:
        words=auto.WORD.findall(u['text'])
        if words:header.append('*PAR:\t'+' '.join(words)+' .')
    return '\n'.join(header+['@End',''])


def run(source,output,speaker,mapping_note,segmenter='improved',verify_clan=False):
    from tempfile import TemporaryDirectory
    import spacy
    from speech_pipeline.pipeline import resolve_segmenter,lexical,clan_software_check,segmenter_evidence
    if not mapping_note.strip():raise ValueError('Document how the patient/target speaker ID was identified')
    source=Path(source);output=Path(output)
    if output.exists() and any(output.iterdir()):raise FileExistsError('Use an empty output directory; preserve previous run evidence')
    raw=source.read_text(encoding='utf-8-sig')
    if source.suffix.lower()=='.json':
        response=json.loads(raw)
        if response.get('finish_reason') in {'length','max_tokens'}:raise ValueError('MOSS generation was truncated')
        text=response['text']
    else:text=raw
    rows=parse_transcript(text);target=target_rows(rows,speaker)
    # Strip model punctuation before any boundary model; preserve recognized
    # source words. Source MOSS segment endpoints are never utterance labels.
    words=[w for r in target for w in auto.WORD.findall(r['text'])]
    if not words:raise ValueError('No usable target speech words')
    paragraph=' '.join(words)
    nlp=spacy.load('en_core_web_sm');model=resolve_segmenter(segmenter,nlp)
    predicted=(model.segment(paragraph) if model else auto.predict(
        [dict(row=2,speaker='PAR',text=paragraph,start='',end='')],'PAR',nlp))
    expected=[w.casefold().replace('’',"'") for w in words]
    if auto.stream(predicted)[0]!=expected:raise ValueError('Segmentation changed recognized words')
    output.mkdir(parents=True,exist_ok=True)
    faithful=faithful_chat(predicted,source.name,speaker,mapping_note)
    with TemporaryDirectory(prefix='moss_metrics_') as temp:
        temp=Path(temp)
        result=dict(source=str(source.resolve()),source_sha256=digest(source),target_speaker=speaker,
            target_mapping_note=mapping_note,source_segments=len(rows),target_segments=len(target),
            words_preserved=True,input_condition='Unpunctuated target-speaker paragraph; MOSS segment cuts withheld',
            segmenter=segmenter_evidence(model),utterances=predicted,mlu=auto.word_mlu(predicted),
            clinical=clinical_from_predictions(predicted),lexical=lexical(predicted,nlp,temp/'lexical.txt'),
            clan=clan_software_check(predicted,temp) if verify_clan else dict(status='not_run'),
            source_timing_policy='Segment envelopes retained in target_segments.json; no fabricated timestamps for split utterances',
            validation_status='MOSS-output import implemented; no audio accuracy or patient-identification validation is implied')
        if verify_clan:
            from speech_pipeline.corpora import run_clan
            chat=temp/'faithful.cha';chat.write_text(faithful)
            native_mlu=run_clan('mlu',chat)
            native_freq=run_clan('freq',chat,window=50)
            result['native_faithful_clan']=dict(mlu=native_mlu,freq=native_freq,
                scope='Native programs on the full recognized-word CHAT, fillers/repetitions retained without invented codes. Distinct from normalized clinical surface profile.')
    save(output/'target_segments.json',dict(all_segments=rows,target_segments=target,mapping_note=mapping_note))
    (output/'target_paragraph.txt').write_text(paragraph+'\n')
    (output/'transcript.cha').write_text(faithful)
    (output/'clan_metric_input.cha').write_text(auto.chat_text(predicted,source.name))
    save(output/'results.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('moss_output',type=Path,help='Canonical [start][Sxx]text[end] output, or JSON with text field')
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--target-speaker',required=True)
    p.add_argument('--mapping-note',required=True)
    p.add_argument('--segmenter',choices=['rules','improved'],default='improved')
    p.add_argument('--verify-clan',action='store_true')
    a=p.parse_args()
    result=run(a.moss_output,a.output,a.target_speaker,a.mapping_note,a.segmenter,a.verify_clan)
    print(json.dumps(dict(target_speaker=a.target_speaker,utterances=len(result['utterances']),clinical=result['clinical']['values']),indent=2))


if __name__=='__main__':main()
