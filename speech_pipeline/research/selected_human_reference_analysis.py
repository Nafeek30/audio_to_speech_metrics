"""Historical selected-reference experiment, not used in the current manuscript.

Use the secondary raters' manual columns for their four annotated files.

Keep all source workbooks immutable. The original primary rater remains the
reference for the other files. Rescore current and archived baseline predictions
with the same measurement policy; bootstrap participants, not recordings.
"""
from pathlib import Path
from collections import defaultdict
import csv,hashlib,json,statistics,math
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
import spacy
from openpyxl import load_workbook
import speech_pipeline.utterances as auto
import speech_pipeline.metrics as custom
from speech_pipeline.utterances import flatten_labels
from speech_pipeline.utterances import plain,boundary_counts,boundary_scores
OUT=ROOT/'outputs/validation/reference_experiment'
METRICS=['basic_ntokens','basic_ntypes','basic_ncontent_tokens','basic_ncontent_types','basic_nfunction_tokens','basic_nfunction_types','simple_ttr_aw','mattr50_aw','total_utterances','mlu_words','mlu_words_sd']
SEED=20261007;REPLICATES=10000

def read(path):
 with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(name,rs):
 with (OUT/name).open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
def main():
 OUT.mkdir(exist_ok=True)
 sources={k:ROOT/'dataset/annotations'/v for k,v in [('Rater 1','all_nafee_utterances.xlsx'),('Rater 2','jade.xlsx'),('Rater 3','yamen.xlsx')]}
 hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources.values()}
 selected={};overrides=[]
 for rater,path in sources.items():
  wb=load_workbook(path,read_only=True,data_only=True)
  for sheet in wb:
   assert list(next(iter(sheet.values)))[:5]==['Start time','End time','Filename','Auto utterance','Manual utterance']
   for filename,texts in flatten_labels(sheet).items():
    key=(sheet.title,filename)
    if rater!='Rater 1':assert key in selected;overrides.append(key)
    selected[key]=dict(rater=rater,texts=texts)
  wb.close()
 assert len(selected)==14 and len(overrides)==4
 nlp=spacy.load('en_core_web_sm')
 report=json.loads((ROOT/'outputs/validation/method_comparison/analysis/uncorrected/comparison.json').read_text());index={(r['patient_id'],r['source_file']):r for r in report['recordings']}
 # Validate all fresh automatic predictions against the saved outputs.
 pairs=[];recording_pairs=[];boundaries=[];manifest=[];group=defaultdict(lambda:dict(human=[],Revised=[],Previous=[]))
 def score(us):
  text='\n'.join(c for u in us if (c:=auto.analysis_text(u['text'])))
  tokens,tagged,meta=custom.preprocess_text(text,nlp)
  lex=custom.calculate_speech_metrics(text,tokens,tagged,meta)
  return {**{m:lex[m] for m in METRICS[:8]},**{m:auto.word_mlu(us)[m] for m in METRICS[8:]}}
 def add(rows,method,pid,filename,values,truth,nfiles):
  for m in METRICS:
   x,y=values[m],truth[m];assert y>0
   rows.append(dict(method=method,reference='Selected human labels',patient_id=pid,source_file=filename,n_recordings=nfiles,metric=m,pipeline_value=x,reference_value=y,difference=x-y,absolute_error=abs(x-y),absolute_percent_error=100*abs(x-y)/abs(y)))
 for (pid,filename),s in selected.items():
  ref=[{'text':plain(t)} for t in s['texts']];rec=index[pid,filename];raw=ROOT/'dataset/uncorrected'/pid/filename
  assert hashlib.sha256(raw.read_bytes()).hexdigest()==rec['raw_sha256']
  current=auto.predict(list(auto.read_rows(raw)),rec['target_speaker'],nlp)
  saved=json.loads((ROOT/'outputs/validation/method_comparison/analysis/uncorrected'/rec['utterance_file']).read_text())['utterances']
  assert [u['text'] for u in current]==[u['text'] for u in saved]
  prev=json.loads((ROOT/'outputs/validation/method_comparison/baseline_analysis'/Path(rec['utterance_file']).name).read_text())['utterances']
  truth=score(ref);group[pid]['human'].extend(ref)
  manifest.append(dict(patient=pid,source_file=filename,reference_rater=s['rater'],manual_column='E',human_utterances=len(ref),pipeline_utterances=len(current),normalized_word_edits=boundary_counts(ref,current)['word_edits']))
  for method,us in [('Previous',prev),('Revised',current)]:
   group[pid][method].extend(us);add(recording_pairs,method,pid,filename,score(us),truth,1)
   c=boundary_counts(ref,us);boundaries.append(dict(method=method,reference_rater=s['rater'],patient_id=pid,source_file=filename,**c,**boundary_scores(c)))
  print(pid,s['rater'],'reference utterances',len(ref),'word edits',manifest[-1]['normalized_word_edits'],flush=True)
 for pid,us in group.items():
  truth=score(us['human']);nfiles=sum(1 for p,f in selected if p==pid)
  for method in ['Previous','Revised']:add(pairs,method,pid,'Complete participant sample',score(us[method]),truth,nfiles)
 ids=sorted(group,key=lambda p:int(p[1:]));assert len(ids)==10
 samples=np.random.default_rng(SEED).integers(0,10,(REPLICATES,10))
 def interval(x):return list(map(float,np.quantile(x,[.025,.975])))
 def summary(rs):
  g=defaultdict(list)
  for row in rs:g[row['patient_id']].append(row)
  assert set(g)==set(ids)
  arrays={k:np.array([statistics.mean(float(r[field]) for r in g[p]) for p in ids]) for k,field in [('e','absolute_percent_error'),('p','pipeline_value'),('r','reference_value')]}
  e=arrays['e'];boot=np.median(e[samples],axis=1)
  return dict(mdape=float(np.median(e)),mdape_ci=interval(boot),mean_ape=float(e.mean()),pipeline_median=float(np.median(arrays['p'])),reference_median=float(np.median(arrays['r'])),minimum=float(e.min()),maximum=float(e.max()),participant_errors=dict(zip(ids,map(float,e)))),boot
 result=dict(reference_policy='Use Rater 2 manual labels for P2/P9 and Rater 3 for P4/P7; retain Rater 1 for other recordings.',bootstrap=dict(seed=SEED,replicates=REPLICATES,confidence_level=.95),manifest=manifest,complete_samples={},recording_samples={},baseline={},boundaries={},sources=hashes)
 for m in METRICS:
  cur,cb=summary([r for r in pairs if r['metric']==m and r['method']=='Revised']);old,ob=summary([r for r in pairs if r['metric']==m and r['method']=='Previous'])
  seg,_=summary([r for r in recording_pairs if r['metric']==m and r['method']=='Revised'])
  result['complete_samples'][m]=cur;result['recording_samples'][m]=seg
  result['baseline'][m]=dict(previous=old,current=cur,reduction=old['mdape']-cur['mdape'],reduction_ci=interval(ob-cb))
 for method in ['Previous','Revised']:
  rs=[r for r in boundaries if r['method']==method];ps=defaultdict(lambda:np.zeros(4))
  for r in rs:ps[r['patient_id']]+=[r[k] for k in ['tp','tn','fn','fp']]
  vals=np.array([ps[p] for p in ids]);tot=vals.sum(axis=0);boots=vals[samples].sum(axis=1);tp,tn,fn,fp=tot
  c=dict(zip(['tp','tn','fn','fp'],map(int,tot)));reference_possible=sum(r['reference_possible_gaps'] for r in rs);pipeline_possible=sum(r['pipeline_possible_gaps'] for r in rs)
  scores=boundary_scores({**c,'reference_tokens':sum(r['reference_tokens'] for r in rs),'word_edits':sum(r['word_edits'] for r in rs),'reference_possible_gaps':reference_possible,'pipeline_possible_gaps':pipeline_possible})
  result['boundaries'][method]=dict(counts=c,evaluable_gaps=int(tot.sum()),f1_percent=100*scores['boundary_f1'],f1_ci=interval(100*2*boots[:,0]/(2*boots[:,0]+boots[:,2]+boots[:,3])),agreement_percent=100*scores['gap_agreement'],precision_percent=100*scores['precision'],recall_percent=100*scores['recall'],kappa=scores['kappa'],reference_coverage_percent=100*scores['reference_gap_coverage'],pipeline_coverage_percent=100*scores['pipeline_gap_coverage'],word_edits=sum(r['word_edits'] for r in rs),recording_reference_coverage_range=[100*min(r['reference_gap_coverage'] for r in rs),100*max(r['reference_gap_coverage'] for r in rs)])
 result['human_utterances']=sum(len(v['human']) for v in group.values());result['pipeline_utterances']=sum(len(v['Revised']) for v in group.values());result['human_count_range']=[min(len(v['human']) for v in group.values()),max(len(v['human']) for v in group.values())]
 participant_rows=[]
 for pid in ids:
  ix={r['metric']:r for r in pairs if r['patient_id']==pid and r['method']=='Revised'}
  participant_rows.append(dict(participant=pid,reference_raters='/'.join(sorted({s['rater'] for (p,f),s in selected.items() if p==pid})),n_recordings=ix['basic_ntokens']['n_recordings'],human_total=ix['total_utterances']['reference_value'],pipeline_total=ix['total_utterances']['pipeline_value'],human_mlu=ix['mlu_words']['reference_value'],pipeline_mlu=ix['mlu_words']['pipeline_value'],human_sd=ix['mlu_words_sd']['reference_value'],pipeline_sd=ix['mlu_words_sd']['pipeline_value'],sd_ape=ix['mlu_words_sd']['absolute_percent_error'],all_word_ape=ix['basic_ntokens']['absolute_percent_error']))
 result['participants']=participant_rows
 # References on other recordings must reproduce their original scores.
 orig=read(ROOT/'outputs/results/human_comparison/participant_sample_pairs.csv');original={(r['method'],r['patient_id'],r['metric']):r for r in orig}
 for r in pairs:
  if r['patient_id'] not in ['P2','P4','P7','P9']:
   o=original[r['method'],r['patient_id'],r['metric']]
   for k in ['pipeline_value','reference_value','absolute_percent_error']:assert math.isclose(float(r[k]),float(o[k]),abs_tol=1e-10)
 for r in pairs:
  assert math.isclose(float(r['pipeline_value']),float(original[r['method'],r['patient_id'],r['metric']]['pipeline_value']),abs_tol=1e-10)
 for p in sources.values():assert hashlib.sha256(p.read_bytes()).hexdigest()==hashes[str(p.relative_to(ROOT))]
 write('participant_sample_pairs.csv',pairs);write('recording_pairs.csv',recording_pairs);write('boundary_recordings.csv',boundaries);write('reference_manifest.csv',manifest);write('participant_results.csv',participant_rows)
 (OUT/'selected_reference_analysis.json').write_text(json.dumps(result,indent=2)+'\n')
 # Historical audit only: do not overwrite current primary-label publication data.
 print('Human utterances',result['human_utterances'],'range',result['human_count_range'])
 for m in METRICS:print(m,result['complete_samples'][m]['mdape'],result['complete_samples'][m]['mdape_ci'])
 for method in ['Previous','Revised']:print('Boundaries',method,result['boundaries'][method])
 for m in METRICS[8:]:print('Reduction',m,result['baseline'][m]['reduction'],result['baseline'][m]['reduction_ci'])
if __name__=='__main__':main()
