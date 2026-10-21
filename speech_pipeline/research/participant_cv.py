"""Five participant-grouped folds inside the original training cohort only.

Each fold starts from the pinned original pretrained encoder. Inner validation
selects epochs/margins; outer labels never select anything. Original eval/test
assignments remain untouched. A separate final model uses train/eval normally.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
from torch import nn

from speech_pipeline import ROOT, utterances as auto
from speech_pipeline.boundary_model import paragraph_rules
from speech_pipeline.chat_metrics import analyze
from speech_pipeline.cohorts import BUNDLE, load_split, read_manifest, validate_training_views
from speech_pipeline.pipeline import digest, save
from speech_pipeline.research.common_benchmark import boundaries, units
from speech_pipeline.research.metric_training import count_loss
from speech_pipeline.research.transcript_chat import PRIMARY, observed_turns, reference, section_rows, summarize
from speech_pipeline.segmentation import PretrainedGapModel, collate, load_bundle, pretrained_logits, save_bundle, word_windows

EVIDENCE = ROOT/'outputs/validation/participant_cv_v1'
OUT = ROOT/'outputs/results/participant_cv_v1'
PLAN = EVIDENCE/'protocol.json'
SEED = 20261015
MARGINS = (0., .25, .5, .75, 1., 1.5, 2.)


def now():return datetime.now(timezone.utc).isoformat()
def key(row):return row['corpus']+'/'+row['recording']
def read_jsonl(path):return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def make_folds(rows, n_splits=5, seed=SEED):
    participants = sorted({r['participant'] for r in rows})
    by_corpus = {c:[p for p in participants if p.startswith(c+'/')] for c in sorted({r['corpus'] for r in rows})}
    if any(len(ps)<n_splits for ps in by_corpus.values()):raise ValueError('Each corpus needs at least k participant groups')
    rng = random.Random(seed)
    for ps in by_corpus.values():rng.shuffle(ps)
    folds = []
    for index in range(n_splits):
        outer = sorted(p for ps in by_corpus.values() for p in ps[index::n_splits])
        inner = []; fit = []; local = random.Random(seed+1000+index)
        for corpus, ps in by_corpus.items():
            remaining = sorted(set(ps)-set(outer));local.shuffle(remaining)
            n = max(1, round(.2*len(remaining)))
            if n>=len(remaining):raise ValueError('Insufficient inner training participants')
            inner.extend(remaining[:n]);fit.extend(remaining[n:])
        fold = dict(fold=index+1, fit=sorted(fit), inner=sorted(inner), outer=outer)
        audit_fold(fold, participants);folds.append(fold)
    counts = Counter(p for fold in folds for p in fold['outer'])
    if set(counts)!=set(participants) or any(n!=1 for n in counts.values()):raise ValueError('Outer coverage must be exactly once per child')
    return folds


def audit_fold(fold, expected):
    roles = [set(fold[name]) for name in ['fit','inner','outer']]
    if any(roles[a]&roles[b] for a,b in [(0,1),(0,2),(1,2)]):raise ValueError('Participant overlap inside fold')
    if set.union(*roles)!=set(expected):raise ValueError('Incomplete fold participant coverage')


def register():
    if PLAN.exists():raise FileExistsError('Preserve registered five-fold study')
    manifest = read_manifest();train = load_split('train')
    folds = make_folds(train)
    forbidden = set(manifest['assignments']['validation']+manifest['assignments']['test'])
    if any(forbidden & set(f[name]) for f in folds for name in ['fit','inner','outer']):raise ValueError('Original eval/test entered CV')
    pretrained = Path(Path('/private/tmp/speech_minilm_model_path.txt').read_text().strip())
    expected_revision = '44acabbec0ef496f6dbc93adadea57f376b7c0ec'
    if pretrained.name!=expected_revision:raise ValueError('Wrong original pretrained revision')
    originals = {p.name:digest(p) for p in pretrained.iterdir() if p.is_file()}
    if 'pytorch_model.bin' not in originals:raise ValueError('Original pretrained weights unavailable')
    EVIDENCE.mkdir(parents=True);OUT.mkdir(parents=True)
    archive = OUT/'previous_production';shutil.copytree(ROOT/'outputs/results/benchmark/metric_segmenter',archive)
    shutil.copy2(ROOT/'outputs/results/benchmark/transcript_workflow.json',archive/'transcript_workflow.json')
    saved = json.loads((ROOT/'outputs/validation/benchmark/metric_training.json').read_text())
    plan = dict(registered_utc=now(), k=5, seed=SEED, scope='Original training cohort only; original eval/test stay separate and outside CV',
        original_cohort_manifest_sha256=digest(BUNDLE/'manifest.json'),
        cohort_input_hashes={s:{c:v['sha256'] for c,v in manifest['files'][s].items()} for s in ['train','validation','test']},
        pool_participants=sorted({r['participant'] for r in train}), pool_sections=len(train), folds=folds,
        pretrained=dict(id='microsoft/MiniLM-L12-H384-uncased', revision=expected_revision,path=str(pretrained),files=originals),
        fixed_recipe=dict(architecture='PretrainedGapModel: original MiniLM encoder + learned 16d rule embedding + 128d classifier',
            context='turn_ml', rule_feature_kind='clause', max_epochs=6,patience=2,batch_size=8,accumulation=2,
            encoder_lr=2e-5,head_lr=1e-4,weight_decay=.01,warmup_fraction=.1,max_length=256,stride=128,
            loss='Cross entropy plus relative boundary-count loss, fixed coefficient 1',
            augmentation='One seeded position crop per fit paragraph; masked final gap; fit participants only',
            sampling='Equal expected sampling mass per corpus at training-window level',
            corruption='5% non-special training subtokens replaced by UNK',averaging=False),
        selection=dict(epoch='Minimum inner native CLAN count/MLUw/SD mean participant APE, corpus balanced, at margin zero; ties keep earlier epoch',
            margins=list(MARGINS),margin='After epoch selection, minimum same inner objective; ties use listed margin order',
            outer='Exactly one frozen prediction per original training section; no candidate, fold, margin or deployment choice from outer results'),
        final_model='After all five folds, independently initialize original pretrained model, fit original train, select epoch/margin on original eval, freeze before original test. Never choose the best outer fold as production.',
        prior_evidence=dict(cross_validation_previously_run=False, participant_overlap=0,
            prior_selected_utc=saved['selection']['recorded_utc'],prior_final_evaluation_started_utc=saved['final_evaluation_started_utc'],
            prior_selection_preceded_test=True, prior_weights_sha256=saved['selection']['weights_sha256']),
        interpretation='Participant-independent out-of-fold internal validation of a fixed recipe. Earlier benchmark/recipe history remains documented; CV does not create an external new corpus.',
        metric_policy='Unchanged original coded CHAT/native CLAN references; no invented annotations; all words preserved; zero denominators unavailable; LUCID lexical evaluation stays separate',
        sources=['https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data',
            'https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html'],
        code_sha256={str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__),ROOT/'speech_pipeline/segmentation.py',ROOT/'speech_pipeline/chat_metrics.py']})
    save(PLAN,plan)
    for f in folds:
        folder=EVIDENCE/f"fold_{f['fold']}";folder.mkdir()
        for role in ['fit','inner','outer']:
            rows=[r for r in train if r['participant'] in f[role]]
            path=folder/f'{role}.jsonl';path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
            save(folder/f'{role}_manifest.json',dict(sha256=digest(path),participants=f[role],sections=len(rows),original_role='train'))
    print('REGISTERED',len(plan['pool_participants']),'training participants',len(train),'sections; original eval/test excluded',flush=True)


def plan():
    p=json.loads(PLAN.read_text())
    if digest(BUNDLE/'manifest.json')!=p['original_cohort_manifest_sha256']:raise ValueError('Original cohorts changed')
    accepted=dict(p['code_sha256']);amendments=EVIDENCE/'implementation_amendments.json'
    if amendments.exists():
        for amendment in json.loads(amendments.read_text()):
            if amendment['protocol_sha256']!=digest(PLAN):raise ValueError('Amendment belongs to another protocol')
            accepted.update(amendment['code_sha256'])
    for filename,sha in accepted.items():
        if digest(ROOT/filename)!=sha:raise ValueError('Registered CV implementation changed; record an amendment before continuing')
    return p


def job_rows(name, role):
    p=plan()
    if name=='final':
        if role=='fit':return load_split('train')
        if role=='inner':return load_split('validation')
        if role=='outer':
            if not (OUT/'final/selection.json').exists():raise ValueError('Freeze final model before opening test')
            return load_split('test')
    folder=EVIDENCE/name;info=json.loads((folder/f'{role}_manifest.json').read_text());path=folder/f'{role}.jsonl'
    if digest(path)!=info['sha256']:raise ValueError('Fold records changed')
    rows=read_jsonl(path)
    f=p['folds'][int(name.split('_')[1])-1]
    if any(r['participant'] not in f[role] for r in rows):raise ValueError('Wrong participant role in fold')
    return rows


def prepare_rows(rows):
    """Reference-free contexts/features; native references stay evaluator-only."""
    import spacy
    folder=EVIDENCE/'prepared';folder.mkdir(exist_ok=True)
    checked={}
    for r in rows:
        source=ROOT/'dataset/public'/r['corpus']/r['recording'].split('#section')[0]
        if source not in checked:checked[source]=digest(source)
        if checked[source]!=r['source_sha256']:raise ValueError('Original CHAT source changed')
    missing=[r for r in rows if not (folder/(hashlib.sha256(key(r).encode()).hexdigest()+'.json')).exists()]
    if missing:
        nlp=spacy.load('en_core_web_sm');sourcecache={};memo={};jobs=[]
        for r in missing:
            source,path,section=section_rows(r,sourcecache);contexts=[]
            for text in observed_turns(source):
                words=auto.WORD.findall(text)
                if not words:continue
                tokens=[w.casefold().replace('’',"'") for w in words];signature=tuple(tokens)
                if signature not in memo:memo[signature]=paragraph_rules(tokens,nlp) if len(tokens)>1 else [0]
                contexts.append(dict(words=words,tokens=tokens,baseline=memo[signature]))
            if [w for c in contexts for w in c['tokens']]!=r['tokens']:raise ValueError('Preparation changed words')
            jobs.append((r,source,path,section,contexts))
        def compute(job):
            r,source,path,section,contexts=job
            return r,dict(key=key(r),participant=r['participant'],source_sha256=r['source_sha256'],contexts=contexts,
                reference=reference(r,source,path,section)['values'])
        with ThreadPoolExecutor(max_workers=4) as pool:
            for index,(r,value) in enumerate(pool.map(compute,jobs)):
                save(folder/(hashlib.sha256(key(r).encode()).hexdigest()+'.json'),value)
                if index%100==0:print('PREPARED',index+1,'/',len(jobs),flush=True)
    result={}
    for r in rows:
        value=json.loads((folder/(hashlib.sha256(key(r).encode()).hexdigest()+'.json')).read_text())
        if value['participant']!=r['participant'] or value['source_sha256']!=r['source_sha256']:raise ValueError('Prepared source changed')
        result[key(r)]=value
    return result


def infer(model, tokenizer, rows, prepared):
    contexts=[];owners=[]
    for r in rows:
        for context in prepared[key(r)]['contexts']:
            contexts.append(context);owners.append(key(r))
    logits=pretrained_logits(model,tokenizer,contexts,256,128,8)
    outputs=defaultdict(list)
    for owner,context,values in zip(owners,contexts,logits):outputs[owner].append((context['words'],values[:,1]-values[:,0]))
    return outputs


def objective(summary):
    values=[r['mean_participant_ape'] for r in summary if r['metric'] in PRIMARY]
    if len(values)!=6 or any(v is None for v in values):raise ValueError('Incomplete corpus-balanced objective')
    return float(np.mean(values))


def score(rows, prepared, logits, margin, name, fold, retain=False):
    def one(r):
        us=[]
        for words,values in logits[key(r)]:
            cuts=[i+1 for i,v in enumerate(values[:-1]) if v>margin];us.extend(units(words,cuts))
        if auto.stream(us)[0]!=r['tokens']:raise ValueError('Predicted words differ from original')
        native=analyze(us,key(r))
        result=dict(corpus=r['corpus'],participant=r['participant'],recording=r['recording'],fold=fold,method=name,
            reference=prepared[key(r)]['reference'],predicted=native['values'],utterances=[u['text'] for u in us])
        if retain:result.update(boundaries=boundaries(r,us),chat=native['chat'],official_outputs=native['official_outputs'])
        return result
    with ThreadPoolExecutor(max_workers=4) as pool:records=list(pool.map(one,rows))
    summary=summarize(records)
    return dict(records=records,summary=summary,objective=objective(summary))


def tensor_hash(parameters):
    value=hashlib.sha256()
    for name,param in parameters:
        value.update(name.encode());value.update(param.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def cpu_tree(value):
    if torch.is_tensor(value):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_tree(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_tree(v) for v in value)
    return value


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


def train_job(name):
    from transformers import AutoTokenizer
    p=plan();folder=OUT/name;folder.mkdir(exist_ok=True)
    if (folder/'selection.json').exists():raise FileExistsError('Completed selection is frozen')
    checkpoint=folder/'training_state.pt'
    if (folder/'history.json').exists() and not checkpoint.exists():
        raise FileExistsError('Interrupted training needs its recorded checkpoint, not an unlogged restart')
    fit=job_rows(name,'fit');inner=job_rows(name,'inner');outer_ids=set()
    if name!='final':outer_ids=set(p['folds'][int(name.split('_')[1])-1]['outer'])
    fit_ids={r['participant'] for r in fit};inner_ids={r['participant'] for r in inner}
    if fit_ids&inner_ids or (fit_ids|inner_ids)&outer_ids:raise ValueError('Training/selection participant leakage')
    pretrained=Path(p['pretrained']['path'])
    if not all((pretrained/n).is_file() for n in p['pretrained']['files']):
        pretrained=EVIDENCE/'pretrained_original'
    if any(digest(pretrained/n)!=sha for n,sha in p['pretrained']['files'].items()):raise ValueError('Original pretrained files changed')
    seed=SEED+(100 if name=='final' else int(name.split('_')[1]))
    torch.set_num_threads(4);device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    torch.use_deterministic_algorithms(True,warn_only=device.type=='mps');torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    tokenizer=AutoTokenizer.from_pretrained(str(pretrained),use_fast=True,local_files_only=True)
    model=PretrainedGapModel(pretrained_path=str(pretrained))
    initial=dict(encoder_sha256=tensor_hash(model.encoder.named_parameters()),head_sha256=tensor_hash(model.head.named_parameters()),
        pretrained_files=p['pretrained']['files'],fresh_task_head=True,old_finetuned_weights_loaded=False,seed=seed,device=str(device),
        fit_participants=sorted(fit_ids),inner_participants=sorted(inner_ids),outer_participants=sorted(outer_ids),started_utc=now())
    if (folder/'initialization.json').exists():
        previous=json.loads((folder/'initialization.json').read_text())
        if any(initial[k]!=previous[k] for k in ['encoder_sha256','head_sha256','seed','fit_participants','inner_participants']):
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
    augment_path=EVIDENCE/name/'augmentation.jsonl';augment_path.parent.mkdir(exist_ok=True)
    augment_path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in augmented))
    del nlp
    windows=[]
    for ri,r in enumerate(augmented):windows.extend(dict(w,record=ri) for w in word_windows(r,tokenizer,256,128))
    windows=[w for w in windows if any(o and augmented[w['record']]['labels'][g]>=0 for g,o in zip(w['gaps'],w['owned']))]
    counts=Counter(augmented[w['record']]['corpus'] for w in windows)
    weights=[1/counts[augmented[w['record']]['corpus']] for w in windows]
    prepared=prepare_rows(inner);model.to(device)
    optimizer=torch.optim.AdamW([dict(params=model.encoder.parameters(),lr=2e-5),
        dict(params=[*model.rule_embedding.parameters(),*model.head.parameters()],lr=1e-4)],weight_decay=.01)
    total=6*((len(windows)+15)//16);warm=max(1,int(total*.1))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda s:min((s+1)/warm,max(0.,(total-s)/max(1,total-warm))))
    ce=nn.CrossEntropyLoss(ignore_index=-100);history=[];best=float('inf');stale=0;started=time.monotonic();first_epoch=1;previous_elapsed=0.
    metadata=dict(name='participant_cv_'+name,decoder='independent',decoding='argmax',threshold=None,rule_feature_kind='clause',
        input_sha256=digest(BUNDLE/'manifest.json'),protocol_sha256=digest(PLAN),max_length=256,stride=128,inference_batch_size=8,
        pretrained=p['pretrained'],selection_scope='Fit participants only for gradients; inner participants only for epoch/margin selection',seed=seed)
    if checkpoint.exists():
        restored=restore_training_checkpoint(checkpoint,name,model,optimizer,scheduler)
        history=restored['history'];best=restored['best'];stale=restored['stale'];first_epoch=restored['epoch']+1
        previous_elapsed=history[-1]['elapsed_seconds'] if history else 0.
        print('RESUMED',name,'after epoch',restored['epoch'],'optimizer/scheduler/RNG restored',flush=True)
    else:
        save_training_checkpoint(checkpoint,model,optimizer,scheduler,name=name,epoch=0,history=[],best=best,stale=stale,
            protocol_sha256=digest(PLAN),fit_participants=sorted(fit_ids),inner_participants=sorted(inner_ids))
    print('TRAIN',name,'fit children',len(fit_ids),'inner children',len(inner_ids),'windows',len(windows),'device',device,flush=True)
    for epoch in range(first_epoch,7):
        if stale>=2:break
        model.train();optimizer.zero_grad(set_to_none=True);losses=[]
        order=random.choices(windows,weights=weights,k=len(windows))
        for at in range(0,len(order),8):
            inputs,labels=collate(order[at:at+8],tokenizer.pad_token_id,augmented)
            inputs={k:v.to(device) for k,v in inputs.items()};labels=labels.to(device)
            ids=inputs['input_ids'];allowed=(ids!=tokenizer.cls_token_id)&(ids!=tokenizer.sep_token_id)&(ids!=tokenizer.pad_token_id)
            ids[(torch.rand(ids.shape,device=device)<.05)&allowed]=tokenizer.unk_token_id
            logits=model(**inputs);loss=ce(logits.reshape(-1,2),labels.reshape(-1))+count_loss(logits,labels)
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            (loss/2).backward();losses.append(float(loss.detach()))
            if (at//8+1)%2==0 or at+8>=len(order):
                nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True)
            if (at//8+1)%50==0:print('BATCH',name,'epoch',epoch,at//8+1,'/',(len(order)+7)//8,flush=True)
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
        save_training_checkpoint(checkpoint,model,optimizer,scheduler,name=name,epoch=epoch,history=history,best=best,stale=stale,
            protocol_sha256=digest(PLAN),fit_participants=sorted(fit_ids),inner_participants=sorted(inner_ids))
        save(folder/'checkpoint_status.json',dict(epoch=epoch,saved_utc=now(),next_epoch=epoch+1,
            full_optimizer_scheduler_rng_saved=True,path=str(checkpoint.relative_to(ROOT))))
        print('EPOCH',name,json.dumps(entry),flush=True)
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
    selection=dict(name=name,frozen_utc=now(),selected_epoch=meta['selected_epoch'],margin=chosen,context='turn_ml',
        weights_sha256=digest(folder/'model/model.safetensors'),initialization_sha256=digest(folder/'initialization.json'),
        inner_objective=selected_results['objective'],candidates=candidates,fit_participants=sorted(fit_ids),inner_participants=sorted(inner_ids),
        outer_participants=sorted(outer_ids),outer_scores_accessed_for_selection=False,protocol_sha256=digest(PLAN))
    save(folder/'selection.json',selection)
    del model;gc.collect()
    if device.type=='mps':torch.mps.empty_cache()
    print('FROZEN',name,json.dumps({k:selection[k] for k in ['selected_epoch','margin','inner_objective']}),flush=True)


def evaluate_job(name):
    if name=='final' and (EVIDENCE/'evaluation_hold.json').exists():
        raise ValueError('Final single-model test held: complete registered development ensemble selection first')
    folder=OUT/name
    if (folder/'outer_results.json').exists():raise FileExistsError('Preserve completed held-out fold evaluation')
    selection=json.loads((folder/'selection.json').read_text())
    if selection['protocol_sha256']!=digest(PLAN) or selection['weights_sha256']!=digest(folder/'model/model.safetensors'):
        raise ValueError('Frozen selection/model changed')
    rows=job_rows(name,'outer')
    forbidden=set(selection['fit_participants']+selection['inner_participants'])
    if any(r['participant'] in forbidden for r in rows):raise ValueError('Held-out participant was used in fitting/selection')
    prepared=prepare_rows(rows);model,tokenizer,_=load_bundle(folder/'model')
    torch.set_num_threads(4);device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu');model.to(device)
    logits=infer(model,tokenizer,rows,prepared)
    result=score(rows,prepared,logits,selection['margin'],'inner_selected_margin',name,retain=True)
    result.update(selection=selection,evaluated_utc=now(),prediction_role='original_test' if name=='final' else 'out_of_fold_original_train')
    save(folder/'outer_results.json',result)
    print('COMPLETE',name,'participants',len({r['participant'] for r in rows}),'sections',len(rows),
        'summary',json.dumps([{k:r[k] for k in ['corpus','metric','mean_participant_ape']} for r in result['summary'] if r['metric'] in PRIMARY]),flush=True)
    del model;gc.collect()
    if device.type=='mps':torch.mps.empty_cache()


def report():
    p=plan();all_rows=[];folds=[];initial_encoders=set();expected={key(r):r for r in load_split('train')}
    for f in p['folds']:
        name=f"fold_{f['fold']}";folder=OUT/name
        result=json.loads((folder/'outer_results.json').read_text());initial=json.loads((folder/'initialization.json').read_text())
        initial_encoders.add(initial['encoder_sha256'])
        if initial['old_finetuned_weights_loaded']:raise ValueError('Fold reused old task weights')
        for r in result['records']:
            if r['participant'] not in f['outer']:raise ValueError('Wrong outer prediction ownership')
            if r['participant'] in set(initial['fit_participants']+initial['inner_participants']):raise ValueError('Out-of-fold leakage')
        all_rows.extend(result['records']);folds.append(dict(fold=f['fold'],summary=result['summary'],selection=result['selection']))
    if len(initial_encoders)!=1:raise ValueError('Folds did not share the same original pretrained initialization')
    if Counter(key(r) for r in all_rows)!=Counter(expected.keys()):raise ValueError('Every training section must have exactly one out-of-fold prediction')
    summary=summarize(all_rows);variation=[]
    for r in summary:
        values=[next(x['mean_participant_ape'] for x in f['summary'] if x['corpus']==r['corpus'] and x['metric']==r['metric']) for f in folds]
        eligible=[v for v in values if v is not None]
        variation.append(dict(corpus=r['corpus'],metric=r['metric'],fold_values=values,
            fold_mean=float(np.mean(eligible)) if eligible else None,fold_sd=float(np.std(eligible,ddof=1)) if len(eligible)>1 else None))
    result=dict(k=5,scope=p['scope'],records=all_rows,summary=summary,folds=folds,fold_variation=variation,
        audit=dict(participants=len({r['participant'] for r in all_rows}),sections=len(all_rows),outer_prediction_exactly_once=True,
            within_fold_participant_overlap=0,original_eval_or_test_in_cv=False,all_folds_fresh_original_pretrained_initialization=True,
            fold_choice_from_outer_results=False),protocol_sha256=digest(PLAN))
    boundary_summary=[]
    for corpus in sorted({r['corpus'] for r in all_rows}):
        rr=[r for r in all_rows if r['corpus']==corpus]
        t,fp,fn=[sum(r['boundaries']['counts'][k] for r in rr) for k in ['tp','fp','fn']]
        ex={k:sum(r['boundaries']['exact'][k] for r in rr) for k in ['matched','predicted','reference']}
        boundary_summary.append(dict(corpus=corpus,precision=100*t/(t+fp),recall=100*t/(t+fn),f1=200*t/(2*t+fp+fn),
            exact_span_f1=200*ex['matched']/(ex['predicted']+ex['reference'])))
    result['boundary_summary']=boundary_summary
    save(OUT/'cross_validation_results.json',result)
    lines=['# Participant-grouped five-fold cross-validation','',
        'Five outer folds use only the original training cohort. Each model starts from the original pinned pretrained MiniLM encoder with a newly initialized task head. '
        'A separate inner participant holdout selects epochs and the fixed seven-margin calibration; the outer participants never enter either fitting or selection. '
        'The original eval and test cohorts retain their assignments and are outside this cross-validation study.','',
        f"Coverage: {result['audit']['participants']} participants, {len(all_rows)} sections, one held-out prediction per section. "
        'All words are preserved. Results are internal participant-independent validation of the fixed recipe, not evaluation on a newly collected external corpus.','',
        '| Corpus | Metric | Pooled participant APE % | Fold mean ± SD % |','|---|---|---:|---:|']
    lookup={(r['corpus'],r['metric']):r for r in variation}
    for r in summary:
        if r['metric'] not in ['eligible_mlu_utterances','MLUw','length_sd','TNW','NDW','TTR','MATTR']:continue
        v=lookup[r['corpus'],r['metric']]
        value='unavailable' if r['mean_participant_ape'] is None else f"{r['mean_participant_ape']:.2f}"
        fold_mean='unavailable' if v['fold_mean'] is None else f"{v['fold_mean']:.2f}"
        fold_sd='unavailable' if v['fold_sd'] is None else f"{v['fold_sd']:.2f}"
        lines.append(f"| {r['corpus']} | {r['metric']} | {value} | {fold_mean} ± {fold_sd} |")
    lines+=['','Fold SD describes variation across folds; it is not a confidence interval from independent fold estimates. '
        'Zero-reference denominators and unavailable SD/MATTR remain missing in detailed summaries. Native CLAN reference and predicted outputs are retained.','',
        '| Corpus | Boundary precision % | Recall % | F1 % | Exact span F1 % |','|---|---:|---:|---:|---:|']
    for r in boundary_summary:
        lines.append('| '+r['corpus']+' | '+' | '.join(f'{r[k]:.2f}' for k in ['precision','recall','f1','exact_span_f1'])+' |')
    lines+=['',
        '## Cohort and history checks','',
        'No original train/eval/test assignment was altered. Original model/results are archived under previous_production. '
        'The preceding run froze its model at '+p['prior_evidence']['prior_selected_utc']+' before its recorded final evaluation at '+p['prior_evidence']['prior_final_evaluation_started_utc']+'. '
        'Those cohorts were participant-disjoint; previous protocols also disclose earlier benchmark inspection. The new CV adds held-out predictions from fresh task training without erasing that history.','']
    final=OUT/'final/outer_results.json'
    if final.exists():
        final_result=json.loads(final.read_text());lines+=['## Separate fresh final model','',
            'This model was freshly trained on the full original training cohort. The original eval cohort selected its epoch and margin; its choice was frozen before opening the original test cohort. '
            'It is not the best outer fold, and test scores do not choose it.','',
            '| Corpus | Metric | Original-test participant APE % |','|---|---|---:|']
        for r in final_result['summary']:
            if r['metric'] in PRIMARY or r['metric'] in ['TNW','NDW','TTR','MATTR']:
                value='unavailable' if r['mean_participant_ape'] is None else f"{r['mean_participant_ape']:.2f}"
                lines.append(f"| {r['corpus']} | {r['metric']} | {value} |")
        lines+=['','This is a participant-independent test of the frozen fresh model on the retained benchmark; prior benchmark exposure is documented separately.','']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    print('REPORT',OUT/'REPORT.md',flush=True)


def run():
    p=plan()
    for f in p['folds']:
        name=f"fold_{f['fold']}"
        if not (OUT/name/'selection.json').exists():train_job(name)
        if not (OUT/name/'outer_results.json').exists():evaluate_job(name)
    report()
    if not (OUT/'final/selection.json').exists():train_job('final')
    if (EVIDENCE/'evaluation_hold.json').exists():
        save(EVIDENCE/'development_complete.json',dict(completed_utc=now(),final_test_opened=False,
            reason='Registered ensemble comparison on original eval must freeze before final test'))
        print('DEVELOPMENT COMPLETE; final single-model test held for ensemble selection',flush=True)
        return
    if not (OUT/'final/outer_results.json').exists():evaluate_job('final')
    report()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['register','run','report'])
    args=parser.parse_args();{'register':register,'run':run,'report':report}[args.stage]()


if __name__=='__main__':main()
