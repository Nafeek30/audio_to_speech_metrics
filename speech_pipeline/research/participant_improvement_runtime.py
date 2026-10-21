"""Verify and activate the development-selected ensemble in the transcript CLI."""
import argparse
import gc
import json

from speech_pipeline import ROOT
from speech_pipeline.cohorts import load_split
from speech_pipeline.ensemble import EnsembleSegmenter
from speech_pipeline.pipeline import digest, save
from speech_pipeline.research import participant_cv as cv, participant_improvement as study
from speech_pipeline.research.transcript_chat import section_rows


def run(activate=False):
    import spacy
    from speech_pipeline.transcript import run as transcript_run
    study.plan()
    audit=json.loads((study.EVIDENCE/'final_integrity_audit.json').read_text())
    if not audit['complete']:raise ValueError('Finish study integrity audit first')
    selected=json.loads((study.OUT/'selection.json').read_text());model_path=study.OUT/'model'
    if audit['selection_sha256']!=digest(study.OUT/'selection.json'):raise ValueError('Audited selection changed')
    if audit['test_results_sha256']!=digest(study.OUT/'test_results.json'):raise ValueError('Audited results changed')
    nlp=spacy.load('en_core_web_sm');model=EnsembleSegmenter(model_path,nlp)
    rows=load_split('validation');prepared=cv.prepare_rows(rows)
    expected={cv.key(r):r for r in json.loads((study.OUT/'selected_validation_results.json').read_text())['records']}
    texts=[];owners=[]
    for r in rows:
        for context in prepared[cv.key(r)]['contexts']:
            texts.append(' '.join(context['words']));owners.append(cv.key(r))
    outputs={cv.key(r):[] for r in rows}
    for owner,utterances in zip(owners,model.segment_many(texts,boundary_margin=selected['margin'])):
        outputs[owner].extend(u['text'] for u in utterances)
    comparisons=[dict(key=key,matches=utterances==expected[key]['utterances']) for key,utterances in outputs.items()]
    if not all(r['matches'] for r in comparisons):raise ValueError('Runtime differs from frozen eval predictions; investigate before activation')
    evidence=dict(candidate=selected['candidate'],members=selected['members'],margin=selected['margin'],context='turn_ml',
        validation_sections_checked=len(comparisons),all_utterances_matched=True,comparisons=comparisons,activated=False,
        ensemble_manifest_sha256=digest(model_path/'ensemble.json'))
    save(study.EVIDENCE/'runtime_parity.json',evidence)
    del model;gc.collect()
    import torch
    if torch.backends.mps.is_available():torch.mps.empty_cache()
    if not activate:
        print('Runtime parity passed for',len(comparisons),'eval sections.',flush=True);return evidence
    workflow_path=ROOT/'outputs/results/benchmark/transcript_workflow.json'
    archived=cv.OUT/'previous_production/transcript_workflow.json'
    policy=json.loads((study.EVIDENCE/'deployment_policy.json').read_text())
    if digest(archived)!=policy['baseline_workflow_sha256']:raise ValueError('Archived deployment baseline changed')
    if selected['objective']>=policy['baseline_eval_objective']:
        if digest(workflow_path)!=policy['baseline_workflow_sha256']:raise ValueError('Preserve the better development-selected baseline workflow')
        evidence.update(activated=False,retained_previous_workflow=True,baseline_eval_objective=policy['baseline_eval_objective'],
            selected_eval_objective=selected['objective'],deployment_basis=policy['policy'])
        save(study.EVIDENCE/'runtime_parity.json',evidence)
        print('Retained previous production: new candidate did not improve the registered eval objective.',flush=True)
        return evidence
    workflow=dict(context='turn_ml',margin=selected['margin'],method=f"participant_improvement_v2/{selected['candidate']}/margin={selected['margin']:g}",
        validation_primary_ape=selected['objective'],model_directory=str(model_path.relative_to(ROOT)),
        weights_sha256=digest(model_path/'ensemble.json'),selection_source=str((study.OUT/'selection.json').relative_to(ROOT)),
        cross_validation_source=str((cv.OUT/'cross_validation_results.json').relative_to(ROOT)),
        scope='Original train only for gradients; original eval only for epoch, ensemble and margin selection; participant-independent test after freeze')
    current=json.loads(workflow_path.read_text())
    if current!=workflow:
        if digest(workflow_path)!=digest(archived):raise ValueError('Production workflow changed outside study; preserve it')
        temp=workflow_path.with_suffix('.tmp');save(temp,workflow);temp.replace(workflow_path)
    sourcecache={};checks=[]
    for corpus in ['nyu_emerson','ocsc']:
        row=next(r for r in rows if r['corpus']==corpus and 100<=len(r['tokens'])<=300)
        source,_,_=section_rows(row,sourcecache);inp=study.EVIDENCE/f'production_smoke_{corpus}_input.json'
        save(inp,dict(target_speaker='CHI',rows=[dict(speaker=x['speaker'],text=x['text']) for x in source]))
        output=study.OUT/f'production_smoke_{corpus}'
        actual=json.loads((output/'results.json').read_text()) if (output/'results.json').exists() else transcript_run(inp,output,nlp=nlp)
        target=expected[cv.key(row)]
        if [u['text'] for u in actual['utterances']]!=target['utterances'] or actual['chat_metrics']['values']!=target['predicted']:
            raise ValueError('Transcript CLI differs from frozen utterances/native metrics')
        if actual['segmenter']['artifacts']['ensemble.json']!=workflow['weights_sha256'] or actual['boundary_margin']!=selected['margin']:
            raise ValueError('Transcript CLI loaded wrong model/margin')
        checks.append(dict(corpus=corpus,key=cv.key(row),utterances=len(target['utterances']),exact_utterances_and_native_metrics=True))
    evidence.update(activated=True,production_workflow_sha256=digest(workflow_path),transcript_command_checks=checks,
        deployment_basis='Minimum registered development objective; test scores and best outer fold did not select deployment')
    save(study.EVIDENCE/'runtime_parity.json',evidence)
    print('ACTIVATED',selected['candidate'],'runtime matched all',len(comparisons),'eval sections and both CLI checks.',flush=True)
    return evidence


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--activate',action='store_true')
    args=parser.parse_args();run(args.activate)


if __name__=='__main__':main()
