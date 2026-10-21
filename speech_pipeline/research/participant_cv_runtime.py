"""Verify the frozen final model's runtime, then activate its saved workflow."""
import argparse
import json
from pathlib import Path

from speech_pipeline import ROOT
from speech_pipeline.cohorts import load_split
from speech_pipeline.pipeline import digest, save
from speech_pipeline.research import participant_cv as cv
from speech_pipeline.research.transcript_chat import section_rows


def run(activate=False):
    import spacy
    from speech_pipeline.segmentation import Segmenter
    from speech_pipeline.transcript import run as transcript_run
    audit=json.loads((cv.EVIDENCE/'final_integrity_audit.json').read_text())
    if not audit['complete'] or audit['completed_jobs']!=6:raise ValueError('Complete all six model audits first')
    folder=cv.OUT/'final';selection=json.loads((folder/'selection.json').read_text());model_path=folder/'model'
    if digest(model_path/'model.safetensors')!=selection['weights_sha256']:raise ValueError('Frozen final model changed')
    nlp=spacy.load('en_core_web_sm');model=Segmenter(model_path,nlp)
    rows=load_split('validation');prepared=cv.prepare_rows(rows)
    selected=json.loads((folder/'selected_inner_calibrated.json').read_text())
    expected={cv.key(r):r for r in selected['records']}
    texts=[];owners=[]
    for r in rows:
        for context in prepared[cv.key(r)]['contexts']:
            texts.append(' '.join(context['words']));owners.append(cv.key(r))
    outputs={cv.key(r):[] for r in rows}
    for owner,utterances in zip(owners,model.segment_many(texts,boundary_margin=selection['margin'])):
        outputs[owner].extend(u['text'] for u in utterances)
    comparisons=[dict(key=key,matches=utterances==expected[key]['utterances']) for key,utterances in outputs.items()]
    if not all(r['matches'] for r in comparisons):raise ValueError('Runtime differs from frozen inner predictions; preserve model and investigate implementation')
    evidence=dict(model_sha256=selection['weights_sha256'],margin=selection['margin'],context='turn_ml',
        validation_sections_checked=len(comparisons),all_utterances_matched=True,comparisons=comparisons,activated=False)
    save(cv.EVIDENCE/'runtime_parity.json',evidence)
    if not activate:
        print('Runtime parity passed:',len(comparisons),'validation sections. Activation not requested.',flush=True)
        return evidence
    workflow_path=ROOT/'outputs/results/benchmark/transcript_workflow.json'
    archived=cv.OUT/'previous_production/transcript_workflow.json'
    workflow=dict(context='turn_ml',margin=selection['margin'],method=f"participant_cv_final/turn_ml/margin={selection['margin']:g}",
        validation_primary_ape=selection['inner_objective'],model_directory=str(model_path.relative_to(ROOT)),
        weights_sha256=selection['weights_sha256'],selection_source=str((folder/'selection.json').relative_to(ROOT)),
        cross_validation_source=str((cv.OUT/'cross_validation_results.json').relative_to(ROOT)),
        scope='Fresh final model: original train only for gradients; original eval only for epoch/margin selection; participant-independent test after freeze')
    current=json.loads(workflow_path.read_text())
    if current!=workflow:
        if digest(workflow_path)!=digest(archived):raise ValueError('Production workflow changed outside this study; preserve it')
        temp=workflow_path.with_suffix('.tmp');save(temp,workflow);temp.replace(workflow_path)
    # Actual transcript command path, one deterministic eval example per corpus.
    sourcecache={};checks=[]
    for corpus in ['nyu_emerson','ocsc']:
        row=next(r for r in rows if r['corpus']==corpus and 100<=len(r['tokens'])<=300)
        source,_,_=section_rows(row,sourcecache)
        inp=cv.EVIDENCE/f'production_smoke_{corpus}_input.json'
        save(inp,dict(target_speaker='CHI',rows=[dict(speaker=x['speaker'],text=x['text']) for x in source]))
        output=cv.OUT/f'production_smoke_{corpus}'
        if (output/'results.json').exists():actual=json.loads((output/'results.json').read_text())
        else:actual=transcript_run(inp,output,nlp=nlp)
        target=expected[cv.key(row)]
        if [u['text'] for u in actual['utterances']]!=target['utterances'] or actual['chat_metrics']['values']!=target['predicted']:
            raise ValueError('Transcript command does not reproduce frozen predictions/native metrics')
        if actual['segmenter']['artifacts']['model.safetensors']!=selection['weights_sha256'] or actual['boundary_margin']!=selection['margin']:
            raise ValueError('Transcript command used the wrong model or margin')
        checks.append(dict(corpus=corpus,key=cv.key(row),utterances=len(target['utterances']),exact_utterances_and_native_metrics=True,
            model_sha256=actual['segmenter']['artifacts']['model.safetensors']))
    evidence.update(activated=True,production_workflow_sha256=digest(workflow_path),transcript_command_checks=checks,
        deployment_basis='Predeclared final-model procedure and inner eval selection; no best-fold or test-score comparison')
    save(cv.EVIDENCE/'runtime_parity.json',evidence)
    print('Activated frozen fresh final model; runtime matched all',len(comparisons),'eval sections and both command checks.',flush=True)
    return evidence


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--activate',action='store_true')
    args=parser.parse_args();run(args.activate)


if __name__=='__main__':main()
