"""Verify and deploy the CV-selected fresh final transcript model."""
import argparse,gc,json
import torch
from speech_pipeline import ROOT
from speech_pipeline.cohorts import load_split
from speech_pipeline.pipeline import digest,save
from speech_pipeline.research import cv_hyperparameter as study
from speech_pipeline.research.transcript_chat import section_rows


def run(activate=False):
 import spacy
 from speech_pipeline.segmentation import Segmenter
 from speech_pipeline.transcript import run as transcript_run
 study.plan();audit=json.loads((study.EVIDENCE/'final_integrity_audit.json').read_text())
 if not audit['complete']:raise ValueError('Audit complete CV search and final test first')
 choice=json.loads((study.OUT/'cv_selection.json').read_text());config=choice['configuration'];folder=study.OUT/config/'final'
 selected=json.loads((folder/'selection.json').read_text());model_path=ROOT/selected['model_directory']
 if digest(study.OUT/'cv_selection.json')!=audit['cv_selection_sha256'] or digest(folder/'selection.json')!=audit['final_selection_sha256']:raise ValueError('Audited selections changed')
 if digest(model_path/'model.safetensors')!=selected['weights_sha256']:raise ValueError('Selected weights changed')
 nlp=spacy.load('en_core_web_sm');model=Segmenter(model_path,nlp);rows=load_split('validation');prepared=study.prepare_rows(rows)
 expected={study.key(r):r for r in json.loads((folder/'selected_inner_calibrated.json').read_text())['records']}
 texts=[];owners=[]
 for r in rows:
  for c in prepared[study.key(r)]['contexts']:texts.append(' '.join(c['words']));owners.append(study.key(r))
 actual={study.key(r):[] for r in rows}
 for owner,utterances in zip(owners,model.segment_many(texts,boundary_margin=selected['margin'])):actual[owner].extend(u['text'] for u in utterances)
 if any(utterances!=expected[key]['utterances'] for key,utterances in actual.items()):raise ValueError('Runtime differs from frozen eval predictions')
 evidence=dict(configuration=config,recipe=choice['recipe'],margin=selected['margin'],eval_sections_checked=len(rows),all_utterances_matched=True,activated=False,
  model_sha256=selected['weights_sha256'],cv_selection_sha256=digest(study.OUT/'cv_selection.json'))
 study.save(study.EVIDENCE/'runtime_parity.json',evidence);del model;gc.collect()
 if torch.backends.mps.is_available():torch.mps.empty_cache()
 if not activate:return evidence
 path=ROOT/'outputs/results/benchmark/transcript_workflow.json';before=path.read_bytes();previous=study.previous.OUT/'previous_production/transcript_workflow.json'
 workflow=dict(context='turn_ml',margin=selected['margin'],method=f'cv_hyperparameter_v4/{config}/turn_ml',model_directory=selected['model_directory'],
  weights_sha256=selected['weights_sha256'],validation_primary_ape=selected['inner_objective'],cv_selection_source=str((study.OUT/'cv_selection.json').relative_to(ROOT)),
  selection_source=str((folder/'selection.json').relative_to(ROOT)),scope='Recipe chosen by five-fold participant-grouped training-only CV; original eval for final checkpoint/margin; original test after freeze')
 if json.loads(before)!=workflow and digest(path)!=digest(previous):raise ValueError('Preserve unexpected external production workflow change')
 checks=[]
 try:
  study.save(path,workflow);sourcecache={}
  for corpus in ['nyu_emerson','ocsc']:
   row=next(r for r in rows if r['corpus']==corpus and 100<=len(r['tokens'])<=300);source,_,_=section_rows(row,sourcecache)
   inp=study.EVIDENCE/f'production_smoke_{corpus}.json';study.save(inp,dict(target_speaker='CHI',rows=[dict(speaker=x['speaker'],text=x['text']) for x in source]))
   output=study.OUT/f'production_smoke_{corpus}'
   result=json.loads((output/'results.json').read_text()) if (output/'results.json').exists() else transcript_run(inp,output,nlp=nlp)
   target=expected[study.key(row)]
   if [u['text'] for u in result['utterances']]!=target['utterances'] or result['chat_metrics']['values']!=target['predicted']:raise ValueError('CLI utterances/native metrics differ')
   if result['segmenter']['artifacts']['model.safetensors']!=selected['weights_sha256']:raise ValueError('CLI loaded wrong weights')
   checks.append(dict(corpus=corpus,key=study.key(row),exact_utterances_and_native_metrics=True))
 except BaseException:
  temp=path.with_suffix('.restore');temp.write_bytes(before);temp.replace(path);raise
 evidence.update(activated=True,actual_cli_checks=checks,workflow_sha256=digest(path),deployment_basis='Registered five-fold CV recipe ranking; final eval checkpoint/margin; no test-based deployment choice')
 study.save(study.EVIDENCE/'runtime_parity.json',evidence);print('ACTIVATED CV-selected model',config,'parity passed all139 eval sections and both corpus CLI checks',flush=True)
 return evidence


def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--activate',action='store_true');args=parser.parse_args();run(args.activate)


if __name__=='__main__':main()
