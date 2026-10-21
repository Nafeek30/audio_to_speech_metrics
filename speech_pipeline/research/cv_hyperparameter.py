"""Participant-grouped five-fold hyperparameter search with external eval/test."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import gc,json,random,shutil,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from torch import nn
from speech_pipeline import ROOT,utterances as auto
from speech_pipeline.cohorts import BUNDLE,load_split,read_manifest,validate_training_views
from speech_pipeline.boundary_model import paragraph_rules
from speech_pipeline.pipeline import digest,save as write_json
from speech_pipeline.segmentation import PretrainedGapModel,collate,load_bundle,save_bundle,word_windows
from speech_pipeline.research.metric_training import count_loss
from speech_pipeline.research.transcript_chat import PRIMARY,summarize
from speech_pipeline.research import participant_cv as previous

EVIDENCE=ROOT/'outputs/validation/cv_hyperparameter_v4'
OUT=ROOT/'outputs/results/cv_hyperparameter_v4'
PLAN=EVIDENCE/'protocol.json'
SEED=previous.SEED
MARGINS=(0.,-.125,.125,-.25,.25,-.5,.5,-1.,.75,1.,1.5,2.)
RECIPES={
 'lr2_count1':dict(encoder_lr=2e-5,head_lr=1e-4,count_weight=1.),
 'lr2_count025':dict(encoder_lr=2e-5,head_lr=1e-4,count_weight=.25),
 'lr3_count1':dict(encoder_lr=3e-5,head_lr=1.5e-4,count_weight=1.),
 'lr3_count025':dict(encoder_lr=3e-5,head_lr=1.5e-4,count_weight=.25)}
now=previous.now
key=previous.key
prepare_rows=previous.prepare_rows
infer=previous.infer
objective=previous.objective
score=previous.score
tensor_hash=previous.tensor_hash
cpu_tree=previous.cpu_tree


def save(path,value):
 path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
 write_json(temp,value);temp.replace(path)


def register():
 if PLAN.exists():raise FileExistsError('Preserve registered CV search')
 parent=previous.plan();train=load_split('train')
 EVIDENCE.mkdir(parents=True);OUT.mkdir(parents=True)
 save(PLAN,dict(registered_utc=now(),parent_protocol_sha256=digest(previous.PLAN),cohort_manifest_sha256=digest(BUNDLE/'manifest.json'),
  folds=parent['folds'],pretrained=parent['pretrained'],recipes=RECIPES,margins=list(MARGINS),seed=SEED,
  design='Four fixed recipes x five participant-grouped validation folds inside original train only. Same fit/inner/CV participants and seeds per recipe for paired comparison.',
  role_policy='Fit participants only for gradients/crops; inner participant holdout only for epoch and margin; CV heldout participants only for recipe-ranking score. Original eval/test outside all CV jobs.',
  epoch_margin_policy='Choose minimum inner native objective at margin zero for epoch; patience two/max six. Then minimum inner objective across twelve fixed margins; exact ties favor listed order starting at zero and small changes.',
  selection='Minimum pooled out-of-fold corpus-balanced mean participant APE across eligible-count, MLUw and length SD. Every recipe must complete all five folds. Exact ties retain listed recipe order. No best-fold selection.',
  interpretation='CV scores tune hyperparameters, so the selected minimum is a model-selection score, not an unbiased final performance estimate. Original test evaluates the frozen final procedure. This is five-fold search with within-fold early stopping/calibration, not a 5x5 nested-CV claim.',
  final_fit='After recipe selection, fresh original pretrained encoder/task head fitted on original train; original eval selects checkpoint/margin under same rule. Freeze before scoring original test. No ensemble deployment.',
  fixed_settings=parent['fixed_recipe'],baseline_reuse='Only lr2_count1 reuses the five completed original-pretrained baseline fold models/histories with verified hashes. Recalibrate its inner groups with the same twelve margins as all other recipes. No reuse of outer scores for epoch/margin selection.',
  initialization='Same original pretrained encoder, independent fresh initialization per job. Same seed/head initialization across recipes for the same fold is intentional paired control, not transferred weights.',
  previous_branch='v2/v3 one-eval ensemble exploration preserved and superseded for selection. v3 test preparation/emission caching began after its frozen choice, but stopped before any fresh test metric result; none used for this search.',
  prior_partial_test_cache_files=[str(x.relative_to(ROOT)) for x in sorted((ROOT/'outputs/results/participant_improvement_v3/logits/test').glob('*.json'))],
  test_scores_used_for_recipe_design=False,
  sources=['https://scikit-learn.org/stable/modules/grid_search.html','https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data'],
  implementation_sha256={str(Path(__file__).relative_to(ROOT)):digest(Path(__file__)),
   'speech_pipeline/segmentation.py':digest(ROOT/'speech_pipeline/segmentation.py'),'speech_pipeline/chat_metrics.py':digest(ROOT/'speech_pipeline/chat_metrics.py')}))
 print('REGISTERED 4 training recipes x 5 participant folds; original eval/test outside recipe search',flush=True)


def plan():
 p=json.loads(PLAN.read_text());previous.plan()
 if p['parent_protocol_sha256']!=digest(previous.PLAN) or p['cohort_manifest_sha256']!=digest(BUNDLE/'manifest.json'):raise ValueError('Parent protocol/cohorts changed')
 for filename,sha in p['implementation_sha256'].items():
  if digest(ROOT/filename)!=sha:raise ValueError('CV search implementation changed after registration')
 if p['recipes']!=RECIPES:raise ValueError('Registered recipes changed')
 return p


def job_rows(config,name,role):
 plan()
 if name!='final':return previous.job_rows(name,role)
 choice=json.loads((OUT/'cv_selection.json').read_text())
 if choice['configuration']!=config:raise ValueError('Only CV winner may fit final model')
 if role=='fit':return load_split('train')
 if role=='inner':return load_split('validation')
 if role=='outer':
  if not(OUT/config/'final/selection.json').exists():raise ValueError('Freeze final weights and margin before opening test')
  return load_split('test')
 raise ValueError('Unknown role')



def save_training_checkpoint(path, model, optimizer, scheduler, **state):
    """Atomic epoch checkpoint; interrupted writes preserve the prior epoch."""
    payload=dict(state,model=cpu_tree(model.state_dict()),optimizer=cpu_tree(optimizer.state_dict()),
        scheduler=scheduler.state_dict(),python_rng=random.getstate(),numpy_rng=np.random.get_state(),
        torch_rng=torch.get_rng_state(),mps_rng=torch.mps.get_rng_state() if next(model.parameters()).device.type=='mps' else None,
        saved_utc=now())
    temp=path.with_suffix('.tmp');torch.save(payload,temp);temp.replace(path)


def restore_training_checkpoint(path, name, model, optimizer, scheduler):
    # This pickle was produced locally by this runner, never downloaded.
    state=torch.load(path,map_location='cpu',weights_only=False)
    if state['protocol_sha256']!=digest(PLAN) or state['name']!=name:raise ValueError('Checkpoint protocol/job mismatch')
    model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
    random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng']);torch.set_rng_state(state['torch_rng'])
    if state['mps_rng'] is not None:
        if next(model.parameters()).device.type!='mps':raise ValueError('Resume this checkpoint on its original MPS device')
        torch.mps.set_rng_state(state['mps_rng'])
    return state


def train_job(config,name):
    from transformers import AutoTokenizer
    p=plan();recipe=p['recipes'][config];folder=OUT/config/name;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'selection.json').exists():raise FileExistsError('Completed selection is frozen')
    checkpoint=folder/'training_state.pt'
    if (folder/'history.json').exists() and not checkpoint.exists():
        raise FileExistsError('Interrupted training needs its recorded checkpoint, not an unlogged restart')
    fit=job_rows(config,name,'fit');inner=job_rows(config,name,'inner');outer_ids=set()
    if name!='final':outer_ids=set(p['folds'][int(name.split('_')[1])-1]['outer'])
    fit_ids={r['participant'] for r in fit};inner_ids={r['participant'] for r in inner}
    if fit_ids&inner_ids or (fit_ids|inner_ids)&outer_ids:raise ValueError('Training/selection participant leakage')
    pretrained=Path(p['pretrained']['path'])
    if not all((pretrained/n).is_file() for n in p['pretrained']['files']):
        pretrained=previous.EVIDENCE/'pretrained_original'
    if any(digest(pretrained/n)!=sha for n,sha in p['pretrained']['files'].items()):raise ValueError('Original pretrained files changed')
    seed=SEED+(100 if name=='final' else int(name.split('_')[1]))
    torch.set_num_threads(4);device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    torch.use_deterministic_algorithms(True,warn_only=device.type=='mps');torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    tokenizer=AutoTokenizer.from_pretrained(str(pretrained),use_fast=True,local_files_only=True)
    model=PretrainedGapModel(pretrained_path=str(pretrained))
    initial=dict(configuration=config,recipe=recipe,encoder_sha256=tensor_hash(model.encoder.named_parameters()),head_sha256=tensor_hash(model.head.named_parameters()),
        pretrained_files=p['pretrained']['files'],fresh_task_head=True,old_finetuned_weights_loaded=False,seed=seed,device=str(device),
        fit_participants=sorted(fit_ids),inner_participants=sorted(inner_ids),outer_participants=sorted(outer_ids),started_utc=now())
    if (folder/'initialization.json').exists():
        previous_initial=json.loads((folder/'initialization.json').read_text())
        if any(initial[k]!=previous_initial[k] for k in ['configuration','recipe','encoder_sha256','head_sha256','seed','fit_participants','inner_participants']):
            raise ValueError('Fresh initialization or participant roles changed on resume')
    else:save(folder/'initialization.json',initial)
    import spacy
    nlp=spacy.load('en_core_web_sm');augmented=list(fit);rng=random.Random(seed)
    for r in fit:
        n=min(len(r['tokens']),rng.choice([8,16,32,64,128]));start=rng.randrange(len(r['tokens'])-n+1)
        if n<2:continue
        tokens=r['tokens'][start:start+n];labels=r['labels'][start:start+n].copy();labels[-1]=-100
        augmented.append(dict(corpus=r['corpus'],participant=r['participant'],recording=r['recording']+'#crop0',
            tokens=tokens,labels=labels,baseline=paragraph_rules(tokens,nlp)))
    validate_training_views(augmented,fit)
    augment_path=EVIDENCE/config/name/'augmentation.jsonl';augment_path.parent.mkdir(parents=True,exist_ok=True)
    augment_path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in augmented))
    del nlp
    windows=[]
    for ri,r in enumerate(augmented):windows.extend(dict(w,record=ri) for w in word_windows(r,tokenizer,256,128))
    windows=[w for w in windows if any(o and augmented[w['record']]['labels'][g]>=0 for g,o in zip(w['gaps'],w['owned']))]
    counts=Counter(augmented[w['record']]['corpus'] for w in windows)
    weights=[1/counts[augmented[w['record']]['corpus']] for w in windows]
    prepared=prepare_rows(inner);model.to(device)
    optimizer=torch.optim.AdamW([dict(params=model.encoder.parameters(),lr=recipe['encoder_lr']),
        dict(params=[*model.rule_embedding.parameters(),*model.head.parameters()],lr=recipe['head_lr'])],weight_decay=.01)
    total=6*((len(windows)+15)//16);warm=max(1,int(total*.1))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda s:min((s+1)/warm,max(0.,(total-s)/max(1,total-warm))))
    ce=nn.CrossEntropyLoss(ignore_index=-100);history=[];best=float('inf');stale=0;started=time.monotonic();first_epoch=1;previous_elapsed=0.
    metadata=dict(name='cv_search_'+config+'_'+name,configuration=config,recipe=recipe,decoder='independent',decoding='argmax',threshold=None,rule_feature_kind='clause',
        input_sha256=digest(BUNDLE/'manifest.json'),protocol_sha256=digest(PLAN),max_length=256,stride=128,inference_batch_size=8,
        pretrained=p['pretrained'],selection_scope='Fit participants only for gradients; inner participants only for epoch/margin selection',seed=seed)
    if checkpoint.exists():
        restored=restore_training_checkpoint(checkpoint,config+'/'+name,model,optimizer,scheduler)
        history=restored['history'];best=restored['best'];stale=restored['stale'];first_epoch=restored['epoch']+1
        previous_elapsed=history[-1]['elapsed_seconds'] if history else 0.
        print('RESUMED',name,'after epoch',restored['epoch'],'optimizer/scheduler/RNG restored',flush=True)
    else:
        save_training_checkpoint(checkpoint,model,optimizer,scheduler,name=config+'/'+name,epoch=0,history=[],best=best,stale=stale,
            protocol_sha256=digest(PLAN),fit_participants=sorted(fit_ids),inner_participants=sorted(inner_ids))
    print('TRAIN',config,name,'fit children',len(fit_ids),'inner children',len(inner_ids),'windows',len(windows),'device',device,flush=True)
    for epoch in range(first_epoch,7):
        if stale>=2:break
        model.train();optimizer.zero_grad(set_to_none=True);losses=[]
        order=random.choices(windows,weights=weights,k=len(windows))
        for at in range(0,len(order),8):
            inputs,labels=collate(order[at:at+8],tokenizer.pad_token_id,augmented)
            inputs={k:v.to(device) for k,v in inputs.items()};labels=labels.to(device)
            ids=inputs['input_ids'];allowed=(ids!=tokenizer.cls_token_id)&(ids!=tokenizer.sep_token_id)&(ids!=tokenizer.pad_token_id)
            ids[(torch.rand(ids.shape,device=device)<.05)&allowed]=tokenizer.unk_token_id
            logits=model(**inputs);loss=ce(logits.reshape(-1,2),labels.reshape(-1))+recipe['count_weight']*count_loss(logits,labels)
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            (loss/2).backward();losses.append(float(loss.detach()))
            if (at//8+1)%2==0 or at+8>=len(order):
                nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True)
            if (at//8+1)%50==0:print('BATCH',config,name,'epoch',epoch,at//8+1,'/',(len(order)+7)//8,flush=True)
        if device.type=='mps':torch.mps.empty_cache()
        inner_logits=infer(model,tokenizer,inner,prepared)
        result=score(inner,prepared,inner_logits,0.,'inner_zero',name)
        entry=dict(epoch=epoch,mean_training_loss=float(np.mean(losses)),inner_objective=result['objective'],
            elapsed_seconds=previous_elapsed+time.monotonic()-started);history.append(entry)
        save(folder/'history.json',history)
        if result['objective']<best:
            best=result['objective'];stale=0
            save_bundle(model,tokenizer,folder/'model',dict(metadata,selected_epoch=epoch,history=history))
            save(folder/'selected_inner_zero.json',result)
        else:stale+=1
        save_training_checkpoint(checkpoint,model,optimizer,scheduler,name=config+'/'+name,epoch=epoch,history=history,best=best,stale=stale,
            protocol_sha256=digest(PLAN),fit_participants=sorted(fit_ids),inner_participants=sorted(inner_ids))
        save(folder/'checkpoint_status.json',dict(epoch=epoch,saved_utc=now(),next_epoch=epoch+1,
            full_optimizer_scheduler_rng_saved=True,path=str(checkpoint.relative_to(ROOT))))
        print('EPOCH',config,name,json.dumps(entry),flush=True)
        if stale>=2:break
    del optimizer,scheduler,model;gc.collect()
    if device.type=='mps':torch.mps.empty_cache()
    model,tokenizer,meta=load_bundle(folder/'model');model.to(device)
    inner_logits=infer(model,tokenizer,inner,prepared);candidates=[];selected_results=None
    for margin in MARGINS:
        result=score(inner,prepared,inner_logits,margin,'inner_calibration',name)
        if margin==0.:
            before=json.loads((folder/'selected_inner_zero.json').read_text())
            if [r['utterances'] for r in result['records']]!=[r['utterances'] for r in before['records']]:
                raise ValueError('Reloaded selected checkpoint changed inner predictions')
        candidate=dict(margin=margin,objective=result['objective']);candidates.append(candidate)
        if selected_results is None or result['objective']<selected_results['objective']:selected_results=result;chosen=margin
    save(folder/'selected_inner_calibrated.json',selected_results)
    selection=dict(configuration=config,recipe=recipe,model_directory=str((folder/'model').relative_to(ROOT)),name=name,frozen_utc=now(),selected_epoch=meta['selected_epoch'],margin=chosen,context='turn_ml',
        weights_sha256=digest(folder/'model/model.safetensors'),initialization_sha256=digest(folder/'initialization.json'),
        inner_objective=selected_results['objective'],candidates=candidates,fit_participants=sorted(fit_ids),inner_participants=sorted(inner_ids),
        outer_participants=sorted(outer_ids),outer_scores_accessed_for_selection=False,protocol_sha256=digest(PLAN))
    save(folder/'selection.json',selection)
    del model;gc.collect()
    if device.type=='mps':torch.mps.empty_cache()
    print('FROZEN',config,name,json.dumps({k:selection[k] for k in ['selected_epoch','margin','inner_objective']}),flush=True)


def reuse_baseline(name):
 """Reuse a fitted baseline, not its old outer-score selection/calibration."""
 p=plan();config='lr2_count1';source=previous.OUT/name;folder=OUT/config/name;folder.mkdir(parents=True,exist_ok=True)
 old=json.loads((source/'selection.json').read_text())
 if digest(source/'model/model.safetensors')!=old['weights_sha256']:raise ValueError('Baseline weights changed')
 fit=job_rows(config,name,'fit');inner=job_rows(config,name,'inner');prepared=prepare_rows(inner)
 model,tokenizer,meta=load_bundle(source/'model');torch.set_num_threads(4)
 device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu');model.to(device)
 logits=infer(model,tokenizer,inner,prepared);candidates=[];best=None;chosen=None
 for margin in MARGINS:
  result=score(inner,prepared,logits,margin,'inner_calibration',name)
  if margin==0:
   expected=json.loads((source/'selected_inner_zero.json').read_text())
   if [r['utterances'] for r in expected['records']]!=[r['utterances'] for r in result['records']]:raise ValueError('Reused model changed inner predictions')
  candidates.append(dict(margin=margin,objective=result['objective']))
  if best is None or result['objective']<best['objective']:best=result;chosen=margin
 for filename in ['initialization.json','history.json','selected_inner_zero.json']:shutil.copy2(source/filename,folder/filename)
 save(folder/'selected_inner_calibrated.json',best)
 selection=dict(configuration=config,recipe=RECIPES[config],name=name,frozen_utc=now(),selected_epoch=old['selected_epoch'],margin=chosen,context='turn_ml',
  model_directory=str((source/'model').relative_to(ROOT)),weights_sha256=old['weights_sha256'],initialization_sha256=digest(folder/'initialization.json'),
  inner_objective=best['objective'],candidates=candidates,fit_participants=sorted({r['participant'] for r in fit}),inner_participants=sorted({r['participant'] for r in inner}),
  outer_participants=old['outer_participants'],outer_scores_accessed_for_selection=False,protocol_sha256=digest(PLAN),
  reused_baseline_model=True,parent_selection_sha256=digest(source/'selection.json'),parent_protocol_sha256=digest(previous.PLAN))
 save(folder/'selection.json',selection);del model;gc.collect()
 if device.type=='mps':torch.mps.empty_cache()
 print('REUSED BASELINE',name,'epoch',old['selected_epoch'],'inner margin',chosen,flush=True)


def evaluate_job(config,name):
 folder=OUT/config/name;destination=folder/('test_results.json' if name=='final' else 'cv_results.json')
 if destination.exists():return
 selection=json.loads((folder/'selection.json').read_text());model_path=ROOT/selection['model_directory']
 if selection['protocol_sha256']!=digest(PLAN) or digest(model_path/'model.safetensors')!=selection['weights_sha256']:raise ValueError('Frozen choice changed')
 rows=job_rows(config,name,'outer');forbidden=set(selection['fit_participants']+selection['inner_participants'])
 if any(r['participant'] in forbidden for r in rows):raise ValueError('Heldout participant overlap')
 save(folder/'evaluation_started.json',dict(started_utc=now(),selection_sha256=digest(folder/'selection.json'),role='original_test' if name=='final' else 'CV_validation_original_train'))
 prepared=prepare_rows(rows);model,tokenizer,_=load_bundle(model_path);torch.set_num_threads(4)
 device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu');model.to(device)
 logits=infer(model,tokenizer,rows,prepared)
 result=score(rows,prepared,logits,selection['margin'],'cv_inner_selected' if name!='final' else config,name,retain=True)
 result.update(configuration=config,selection=selection,evaluated_utc=now(),prediction_role='original_test' if name=='final' else 'CV_validation_original_train')
 save(destination,result);del model;gc.collect()
 if device.type=='mps':torch.mps.empty_cache()
 print('COMPLETE',config,name,'sections',len(rows),'objective',result['objective'],flush=True)


def candidate_summary(config):
 p=plan();records=[];folds=[];expected={key(r):r for r in load_split('train')}
 for f in p['folds']:
  name=f"fold_{f['fold']}";result=json.loads((OUT/config/name/'cv_results.json').read_text())
  if {r['participant'] for r in result['records']}!=set(f['outer']):raise ValueError('Wrong CV heldout identities')
  if result['configuration']!=config:raise ValueError('Mixed recipe results')
  records.extend(result['records']);folds.append(dict(fold=f['fold'],objective=result['objective'],summary=result['summary']))
 if Counter(key(r) for r in records)!=Counter(expected.keys()):raise ValueError('CV requires exactly one heldout prediction per training section')
 summary=summarize(records)
 return dict(configuration=config,recipe=p['recipes'][config],objective=objective(summary),fold_mean=float(np.mean([f['objective'] for f in folds])),
  fold_sd=float(np.std([f['objective'] for f in folds],ddof=1)),folds=folds,records=records,summary=summary)


def select_recipe():
 p=plan()
 if (OUT/'cv_selection.json').exists():return
 candidates=[candidate_summary(config) for config in p['recipes']]
 winner=min(candidates,key=lambda c:c['objective'])
 save(OUT/'cv_results.json',dict(candidates=candidates,protocol_sha256=digest(PLAN),scope='Hyperparameter selection on original training participants only'))
 selection=dict(frozen_utc=now(),configuration=winner['configuration'],recipe=winner['recipe'],cv_objective=winner['objective'],
  candidates=[{k:c[k] for k in ['configuration','recipe','objective','fold_mean','fold_sd']} for c in candidates],
  cv_results_sha256=digest(OUT/'cv_results.json'),protocol_sha256=digest(PLAN),original_eval_or_test_used_for_recipe_selection=False,
  ranking_rule=p['selection'])
 save(OUT/'cv_selection.json',selection)
 print('CV RECIPE SELECTED',json.dumps(selection),flush=True)


def audit():
 from speech_pipeline.research import participant_cv_audit as checks
 from speech_pipeline.cohorts import verify
 verify();p=plan();choice=json.loads((OUT/'cv_selection.json').read_text());combined=json.loads((OUT/'cv_results.json').read_text())
 checks.require(digest(OUT/'cv_results.json')==choice['cv_results_sha256'],'CV selection results changed')
 checks.require(choice['configuration']==min(combined['candidates'],key=lambda c:c['objective'])['configuration'],'Recipe was not chosen by full CV objective')
 refs={r['key']:r['result']['values'] for r in previous.read_jsonl(ROOT/'outputs/validation/transcript_validation/chat_native_references.jsonl')}
 encoder_hashes=set();head_by_fold={};checked=[]
 for config in p['recipes']:
  for f in p['folds']:
   name=f"fold_{f['fold']}";folder=OUT/config/name;selection=json.loads((folder/'selection.json').read_text());initial=json.loads((folder/'initialization.json').read_text());history=json.loads((folder/'history.json').read_text());result=json.loads((folder/'cv_results.json').read_text())
   roles={r:job_rows(config,name,r) for r in ['fit','inner','outer']};ids={r:{x['participant'] for x in rows} for r,rows in roles.items()}
   checks.require(not(ids['fit']&ids['inner'] or ids['fit']&ids['outer'] or ids['inner']&ids['outer']),'CV participant overlap')
   checks.require(set(initial['fit_participants'])==ids['fit'] and set(initial['inner_participants'])==ids['inner'],'Actual model participant roles changed')
   checks.require(set(selection['fit_participants'])==ids['fit'] and set(selection['inner_participants'])==ids['inner'],'Selected participant roles changed')
   checks.require(not initial['old_finetuned_weights_loaded'] and initial['fresh_task_head'],'Fine-tuned weights reused')
   encoder_hashes.add(initial['encoder_sha256'])
   head_by_fold.setdefault(name,set()).add(initial['head_sha256'])
   if not selection.get('reused_baseline_model'):
    checks.require(initial['configuration']==config and initial['recipe']==p['recipes'][config],'Actual initialization recipe differs')
   checks.require(selection['recipe']==p['recipes'][config],'Selected recipe differs')
   checks.require(digest(ROOT/selection['model_directory']/'model.safetensors')==selection['weights_sha256'],'Model weights changed')
   checks.require(selection['selected_epoch']==min(history,key=lambda h:h['inner_objective'])['epoch'],'Epoch selection used wrong score')
   checks.require(selection['margin']==min(selection['candidates'],key=lambda c:c['objective'])['margin'],'Calibration used wrong score')
   checks.require(result['selection']==selection,'Frozen selection changed')
   checks.require(datetime.fromisoformat(selection['frozen_utc'])<datetime.fromisoformat(result['evaluated_utc']),'CV evaluated before freeze')
   checks.check_predictions(result['records'],{key(r):r for r in roles['outer']},refs)
   aug=previous.EVIDENCE/name/'augmentation.jsonl' if selection.get('reused_baseline_model') else EVIDENCE/config/name/'augmentation.jsonl'
   validate_training_views(previous.read_jsonl(aug),roles['fit'])
   checked.append(dict(configuration=config,fold=name,sections=len(result['records']),fit=len(ids['fit']),inner=len(ids['inner']),heldout=len(ids['outer']),overlap=0))
 checks.require(len(encoder_hashes)==1,'Original encoder initialization differs')
 checks.require(all(len(hashes)==1 for hashes in head_by_fold.values()),'Paired within-fold initialization differs across recipes')
 for candidate in combined['candidates']:
  recomputed=candidate_summary(candidate['configuration'])
  checks.require(recomputed==candidate,'Pooled CV scores do not reproduce')
 config=choice['configuration'];folder=OUT/config/'final';selected=json.loads((folder/'selection.json').read_text());result=json.loads((folder/'test_results.json').read_text())
 fit=job_rows(config,'final','fit');inner=job_rows(config,'final','inner');test=job_rows(config,'final','outer');initial=json.loads((folder/'initialization.json').read_text());history=json.loads((folder/'history.json').read_text())
 checks.require(set(initial['fit_participants'])=={r['participant'] for r in fit} and set(initial['inner_participants'])=={r['participant'] for r in inner},'Final fit/eval roles differ')
 checks.require(not initial['old_finetuned_weights_loaded'] and initial['fresh_task_head'],'Final model was not fresh')
 checks.require(initial['encoder_sha256'] in encoder_hashes,'Final original encoder differs')
 validate_training_views(previous.read_jsonl(EVIDENCE/config/'final/augmentation.jsonl'),fit)
 checks.require(selected['recipe']==choice['recipe'],'Final model did not use CV-selected recipe')
 checks.require(selected['selected_epoch']==min(history,key=lambda h:h['inner_objective'])['epoch'],'Final epoch not selected by eval')
 checks.require(selected['margin']==min(selected['candidates'],key=lambda c:c['objective'])['margin'],'Final margin not selected by eval')
 checks.require(datetime.fromisoformat(choice['frozen_utc'])<datetime.fromisoformat(initial['started_utc']),'Final fit began before recipe selection')
 checks.require(datetime.fromisoformat(selected['frozen_utc'])<datetime.fromisoformat(result['evaluated_utc']),'Test before final freeze')
 checks.require(result['selection']==selected,'Final frozen choice changed')
 checks.check_predictions(result['records'],{key(r):r for r in test},refs)
 checks.require(result['summary']==summarize(result['records']),'Final native summary mismatch')
 checks.write_csv(OUT/'final_test_metric_pairs.csv',checks.pairs(result['records']))
 save(OUT/'participant_bootstrap_intervals.json',dict(final_test=checks.intervals(result['records'])))
 save(EVIDENCE/'final_integrity_audit.json',dict(complete=True,audited_utc=now(),cohort_conclusion='Train, eval and test cohorts are participant-independent.',
  configurations=len(p['recipes']),folds_per_configuration=5,cv_jobs=checked,final_training_recipe=choice['recipe'],selected_configuration=config,
  all_cv_gradient_inner_and_heldout_roles_disjoint=True,original_eval_test_outside_hyperparameter_search=True,all_original_words_native_references_outputs_verified=True,
  cv_results_sha256=digest(OUT/'cv_results.json'),cv_selection_sha256=digest(OUT/'cv_selection.json'),final_selection_sha256=digest(folder/'selection.json'),final_test_sha256=digest(folder/'test_results.json')))
 print('CV SEARCH AND FINAL TEST AUDIT PASSED',flush=True)


def report():
 choice=json.loads((OUT/'cv_selection.json').read_text());config=choice['configuration'];test=json.loads((OUT/config/'final/test_results.json').read_text())
 lines=['# Five-fold participant-grouped hyperparameter selection','',
  'Train, eval and test cohorts remain participant-independent. Original train supplies all five-fold hyperparameter comparisons; original eval/test stay outside the search. Each recipe has five separately trained models, with independent fitting/inner-validation/CV-heldout participants within a fold.','',
  '| Recipe | Encoder / head LR | Count-loss weight | Pooled CV objective % | Fold mean ± SD % |','|---|---|---:|---:|---:|']
 for c in choice['candidates']:
  r=c['recipe'];lines.append(f"| {c['configuration']} | {r['encoder_lr']:g} / {r['head_lr']:g} | {r['count_weight']:g} | {c['objective']:.2f} | {c['fold_mean']:.2f} ± {c['fold_sd']:.2f} |")
 lines+=['',f"Selected **{config}** by the complete five-fold ranking, then trained a separate fresh final model. Original eval selected its epoch and boundary margin before original test scoring.",'',
  'The CV minimum is a hyperparameter-selection score. Final performance is assessed separately below; selecting on CV does not make the selected CV minimum an unbiased generalization estimate. Fold SD is descriptive, not an independent-fold confidence interval.','',
  '| Corpus | Final-test eligible-count APE % | MLUw APE % | Length-SD APE % |','|---|---:|---:|---:|']
 for corpus in ['nyu_emerson','ocsc']:
  lookup={r['metric']:r['mean_participant_ape'] for r in test['summary'] if r['corpus']==corpus}
  lines.append('| '+corpus+' | '+' | '.join(f'{lookup[m]:.2f}' for m in PRIMARY)+' |')
 lines+=['','All original cohort assignments, source words, native human references and preceding results are preserved. LUCID lexical results are unchanged and remain separate from boundary-model training. Full native outputs, missingness, predictions, pairwise CSV and participant-bootstrap intervals accompany this report.','']
 (OUT/'REPORT.md').write_text('\n'.join(lines));print('REPORT',OUT/'REPORT.md',flush=True)


def run():
 p=plan()
 for config in p['recipes']:
  for f in p['folds']:
   name=f"fold_{f['fold']}";folder=OUT/config/name
   if not(folder/'selection.json').exists():
    if config=='lr2_count1':reuse_baseline(name)
    else:train_job(config,name)
   evaluate_job(config,name)
 select_recipe();choice=json.loads((OUT/'cv_selection.json').read_text());config=choice['configuration']
 if not(OUT/config/'final/selection.json').exists():train_job(config,'final')
 evaluate_job(config,'final');audit();report()


def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['register','run','audit','report'])
 args=parser.parse_args();globals()[args.stage]()


if __name__=='__main__':main()
