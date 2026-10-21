"""Transcript-only entry point using the benchmarked lexical/CHAT policies."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from speech_pipeline import ROOT, utterances as auto
from speech_pipeline.chat_metrics import analyze
from speech_pipeline.lexical_compat import HistoricalTaaled, clean_pauses
from speech_pipeline.pipeline import digest, save, resolve_segmenter, segmenter_evidence


def target_contexts(rows,target):
    """Use observed speaker changes, merging adjacent target transcript rows."""
    turns=[];current=[]
    for row in rows:
        if not isinstance(row,dict) or not isinstance(row.get('speaker'),str) or not isinstance(row.get('text'),str):
            raise ValueError('Each transcript row needs speaker and text strings')
        if not row['text'].strip():continue
        if row['speaker']==target:current.append(row['text'])
        elif current:turns.append(' '.join(current));current=[]
    if current:turns.append(' '.join(current))
    return turns


def run(path, output, segmenter='improved', nlp=None,boundary_margin=None):
    """Analyze a prepared target-speaker transcript; no reference values enter."""
    import spacy
    path, output = Path(path), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use an empty output directory; preserve results')
    source_text = path.read_text(encoding='utf-8-sig')
    rows=None;target='PAR'
    if path.suffix.lower()=='.json':
        data=json.loads(source_text);rows=data.get('rows');target=data.get('target_speaker')
        if not isinstance(rows,list) or not isinstance(target,str) or not target:
            raise ValueError('JSON transcript needs rows and an explicit target_speaker')
        turns=target_contexts(rows,target)
        text=' '.join(turns)
    else:text=source_text;turns=[text]
    if any(c in text for c in '[]<>@&+*=|\\\x15'):
        raise ValueError('Prepare target plain text first; do not strip CHAT scopes implicitly')
    nlp = nlp if nlp is not None else spacy.load('en_core_web_sm')
    workflow_path=ROOT/'outputs/results/benchmark/transcript_workflow.json'
    workflow=json.loads(workflow_path.read_text()) if workflow_path.exists() else None
    context='turn_ml' if rows is not None and workflow and workflow['context']=='turn_ml' else 'paragraph_ml'
    model_options=dict(model_directory=workflow['model_directory'],expected_weights_sha256=workflow['weights_sha256']) if workflow and workflow.get('model_directory') and workflow['context']==context else {}
    model = resolve_segmenter(segmenter, nlp,**model_options)
    if boundary_margin is None and workflow:
        boundary_margin=workflow['margin'] if workflow['context']==context else workflow.get('paragraph_fallback_margin')
    # Row ends and punctuation are not supplied as gold utterance boundaries.
    paragraph = ' '.join(auto.WORD.findall(text))
    if not paragraph:
        raise ValueError('No speech words')
    contexts=[' '.join(auto.WORD.findall(t)) for t in turns] if context=='turn_ml' else [paragraph]
    us = [u for t in contexts for u in model.segment(t,boundary_margin=boundary_margin)] if model else auto.predict(
        [dict(row=2, speaker='PAR', text=paragraph, start='', end='')], 'PAR', nlp)
    for i,u in enumerate(us,1):u['utterance_id']=i
    if auto.stream(us)[0] != auto.stream([dict(text=paragraph)])[0]:
        raise ValueError('Segmentation changed source words')
    clinical = analyze(us, path.name)
    cleaned, removals = clean_pauses(text)
    lexicons = ROOT/'outputs/validation/lucid_validation/upstream_gui'
    lexical = HistoricalTaaled(lexicons).calculate(nlp(cleaned))
    output.mkdir(parents=True, exist_ok=True)
    chat = output/'transcript.cha'; chat.write_text(clinical.pop('chat'))
    result = dict(run_utc=datetime.now(timezone.utc).isoformat(),
        input=str(path.resolve()), input_sha256=digest(path),
        segmenter=segmenter_evidence(model), utterances=us,
        segmentation_context=context,boundary_margin=boundary_margin,
        workflow_selection=workflow,
        chat_file=str(chat.resolve()), chat_sha256=digest(chat),
        chat_metrics=clinical, lexical_metrics=lexical,
        lexical_profile='Historical TAALED 1.4.1 dictionary/category rules with current spaCy model; fixed pause cleaning. Vocabulary measures use the whole source text, independent of predicted utterance cuts.',
        lexical_pause_removals=removals,
        lexical_assets={p.name:digest(p) for p in (lexicons/'real_words.txt',lexicons/'adj_lem_list.txt')},
        scope='Target-speaker transcript only. Automatic boundaries remain unreviewed; no audio, clinical diagnosis or invented morphology/grammar/maze labels.')
    save(output/'results.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('transcript', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--segmenter', choices=['improved','rules'], default='improved')
    p.add_argument('--boundary-margin',type=float,help='Explicit override; default uses the saved development-selected workflow when available')
    args = p.parse_args()
    r = run(args.transcript, args.output, args.segmenter,boundary_margin=args.boundary_margin)
    print('Saved', args.output, 'units:', len(r['utterances']), flush=True)


if __name__ == '__main__':main()
