"""Metric-first development on public train/validation; final cohorts stay sealed.

The prior reassessment remains locked. This separate round never uses final
scores for epoch, objective, architecture, or deployment selection.
"""
import argparse
from collections import deque,defaultdict
from datetime import datetime,timezone
import gc,json,random,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from speech_pipeline import ROOT,utterances as auto
from speech_pipeline.corpora import EVIDENCE,OUT,read_chat,chat_reference_units
from speech_pipeline.boundary_model import paragraph_rules
from speech_pipeline.pipeline import digest,save
from speech_pipeline.research.segmentation import load_partitions
from speech_pipeline.segmentation import PretrainedGapModel,word_windows,collate,pretrained_predictions,metrics,save_bundle,load_bundle,sentence_clause_features

WORK=Path('/private/tmp/speech_metric_research')
PLAN=EVIDENCE/'metric_training.json'
MEASURES=('total_utterances','mlu_words','mlu_words_sd')
PRETRAINED={
    'deberta':dict(id='microsoft/deberta-v3-base',revision='8ccc9b6f36199bec6961081d44eb72fb3f7353f3',files=['config.json','pytorch_model.bin','spm.model','tokenizer_config.json']),
    'roberta':dict(id='FacebookAI/roberta-base',revision='e2da8e2f811d1448a5b465c236feacd80ffbac7b',files=['config.json','model.safetensors','tokenizer.json','tokenizer_config.json','vocab.json','merges.txt']),
    'minilm':dict(id='microsoft/MiniLM-L12-H384-uncased',revision='44acabbec0ef496f6dbc93adadea57f376b7c0ec',files=['config.json','pytorch_model.bin','vocab.txt','tokenizer_config.json','special_tokens_map.json']),
    'chat':dict(id='talkbank/CHATUtterance-en',revision='764ec3f762c2e24df2def8df98b5fe34940085c6',files=['config.json','pytorch_model.bin','tokenizer.json','vocab.txt','tokenizer_config.json','special_tokens_map.json'])}


def fetch_models():
    from huggingface_hub import snapshot_download
    for name,spec in PRETRAINED.items():
        snapshot_download(spec['id'],revision=spec['revision'],allow_patterns=spec['files'],local_dir=WORK/'pretrained'/name)
        print('Fetched pinned official model',name,spec['revision'],flush=True)

def read():return json.loads(PLAN.read_text())

def public_development():
    from speech_pipeline.cohorts import BUNDLE, load_split, validate_records
    p=read()
    if 'selection' in p:raise ValueError('This round is selected; development is closed')
    if BUNDLE.exists():
        if 'development_data_sha256' not in p:raise ValueError('Prepare the corrected spoken-word development data first')
        if digest(EVIDENCE/'talkbank_spoken_development.jsonl')!=p['development_data_sha256']:
            raise ValueError('Spoken development data changed')
        return {g:load_split(g,expected_assignments=p['partitions']) for g in ('train','validation')},p
    splits,old=load_partitions()
    if old['input_sha256']!=p['input_sha256'] or old['partitions']!=p['partitions']:raise ValueError('Frozen public groups changed')
    for group in splits.values():
        if any(r['corpus'] not in {'ocsc','nyu_emerson'} for r in group):raise ValueError('Non-public development input')
    if 'development_data_sha256' not in p:raise ValueError('Prepare the corrected spoken-word development data first')
    path=EVIDENCE/'talkbank_spoken_development.jsonl'
    if digest(path)!=p['development_data_sha256']:raise ValueError('Spoken development data changed')
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    validate_records(rows,p['partitions'],('train','validation'))
    return {g:[r for r in rows if r['participant'] in p['partitions'][g]] for g in ('train','validation')},p


def spoken_records(records,allowed,nlp):
    """Rebuild exact sections using spoken words; never expose event descriptions."""
    output=[];cache={};audit=dict(records_before=len(records),words_before=sum(len(r['tokens']) for r in records),changed_records=0,removed_unscorable_records=0,reference_continuation_joins=0)
    for r in records:
        if r['corpus'] not in {'ocsc','nyu_emerson'} or r['participant'] not in allowed:raise ValueError('Non-development child in preparation')
        filename,section=r['recording'].rsplit('#section',1);path=ROOT/'dataset/public'/r['corpus']/filename
        if digest(path)!=r['source_sha256']:raise ValueError('Raw CHAT source changed')
        if path not in cache:
            sections=[]
            for row in read_chat(path):
                if not sections or sections[-1][0]['gem']!=row['gem']:sections.append([])
                sections[-1].append(row)
            cache[path]=sections
        section_rows=cache[path][int(section)-1]
        if section_rows[0]['gem']!=r['gem']:raise ValueError('Task section alignment changed')
        gold=chat_reference_units(section_rows,'CHI')
        audit['reference_continuation_joins']+=sum(len(u['source_rows'])-1 for u in gold)
        tokens,boundaries=auto.stream(gold)
        labels=[int(i+1 in boundaries) if i+1<len(tokens) and not({tokens[i],tokens[i+1]}&{'xxx','yyy','www'}) else -100 for i in range(len(tokens))]
        if not any(v>=0 for v in labels):audit['removed_unscorable_records']+=1;continue
        surface=[w for u in gold for w in auto.WORD.findall(u['text'])]
        if len(surface)!=len(tokens):raise ValueError('Spoken source alignment changed')
        if tokens!=r['tokens']:audit['changed_records']+=1
        output.append(dict(r,tokens=tokens,surface_words=surface,gold_boundaries=sorted(boundaries),labels=labels,baseline=paragraph_rules(tokens,nlp),input_policy='Spoken words only: &= non-speech events excluded before word extraction; genuine repetitions/fillers/fragments retained. Gold punctuation, row breaks, turns, morphology and timestamps withheld.'))
    audit.update(records_after=len(output),words_after=sum(len(r['tokens']) for r in output),participants=len({r['participant'] for r in output}))
    return output,audit


def prepare():
    import spacy
    p=read()
    if 'selection' in p:raise ValueError('No preparation changes after selection')
    if 'development_data_sha256' in p:raise ValueError('Corrected development preparation already frozen')
    original,old=load_partitions();nlp=spacy.load('en_core_web_sm')
    if old['input_sha256']!=p['input_sha256']:raise ValueError('Original historical data changed')
    allowed=set(p['partitions']['train']+p['partitions']['validation'])
    rows,audit=spoken_records(original['train']+original['validation'],allowed,nlp)
    path=EVIDENCE/'talkbank_spoken_development.jsonl';path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    p['development_data_sha256']=digest(path);p['spoken_preparation_audit']=audit;p['status']='registered';save(PLAN,p)
    print('Prepared spoken-word train/validation only:',json.dumps(audit),flush=True)


def augmented(train):
    """One context crop per corrected public training paragraph; never gold-driven."""
    path=WORK/'spoken_augmented_training.jsonl'
    if path.exists():
        from speech_pipeline.cohorts import validate_training_views
        rows=[json.loads(s) for s in path.read_text().splitlines()]
        validate_training_views(rows,train)
        return rows
    import spacy
    nlp=spacy.load('en_core_web_sm');rng=random.Random(42);rows=list(train)
    for r in train:
        n=min(len(r['tokens']),rng.choice([8,16,32,64,128]))
        if n<2:continue
        start=rng.randrange(len(r['tokens'])-n+1);tokens=r['tokens'][start:start+n];labels=r['labels'][start:start+n].copy();labels[-1]=-100
        rows.append(dict(corpus=r['corpus'],participant=r['participant'],recording=r['recording']+'#crop0',tokens=tokens,labels=labels,baseline=paragraph_rules(tokens,nlp)))
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows));return rows

def register():
    if PLAN.exists():raise ValueError('Preserve the existing metric-first round')
    splits,old=load_partitions()
    p=dict(registered_utc=datetime.now(timezone.utc).isoformat(),status='registered',
        task='Audio-to-target-speaker CLAN/CHAT analysis; this training stage improves transcript utterances, not ASR or diarization',
        user_priority='Minimize final language-measure error; 90% boundary F1 is diagnostic, not the objective',
        reference_unit='Human CLAN/CHAT utterances, explicitly confirmed by user',
        input_sha256=old['input_sha256'],partitions=old['partitions'],seed=42,
        sealed_data='Public test and all 10 original participants are evaluation only. No examples, error review, training, adaptation or choice from them in this round. Historical benchmark use is disclosed.',
        primary_selection='Lowest corpus-balanced mean participant mean recording APE, equally averaged across utterance count, surface MLUw and population length SD; zero reference denominators unavailable, never zero-filled. Equal corpus weighting prevents OCSC size hiding NYU error. Lexical errors and boundary F1 reported separately.',
        tie_break='Exact objective ties: higher validation boundary F1, then fewer parameters, then lexical name',
        candidates=['incumbent','deberta_context','deberta_metric','roberta_metric','minilm_metric','deberta_sentence','chat_metric'],
        candidates_scope='Bounded before development; no final score can expand or revise this round',
        training=dict(max_epochs=6,patience=2,batch_size=8,accumulation=2,encoder_lr=2e-5,head_lr=1e-4,weight_decay=.01,warmup_fraction=.1,
            max_length=256,stride=128,views='Original training paragraphs and one seeded position crop per paragraph; no gold-cut geometry',
            metric_loss='CE plus mean squared relative boundary-count deviation per owned valid window; denominator gold positive count + 1. Weight 1 for metric candidates, zero for context control. Training labels only.',
            corruption='Metric candidates randomly replace 5% non-special input subtokens with UNK during training only. This is synthetic robustness, not claimed to reproduce real ASR errors.',
            balancing='Metric candidates sample training windows with equal expected mass for each public corpus; context control uses natural corpus mix',
            initialization='deberta_metric continues validation-selected deberta_context for at most 3 epochs; other candidates start official pretrained encoders',
            averaging='Also validate equal parameter average of final three epochs; fixed class argmax for all candidates'),
        pretrained=PRETRAINED,
        deployment='Preserve earlier production model and workbook routing until selection; deploy only validation choice. Never change deployment from private final results.',
        metric_policy='Existing source-word preservation, filler/unintelligibility eligibility and metric formulas unchanged. No posthoc count forcing or gold lengths in inference.',
        sources=['https://arxiv.org/abs/2111.09543','https://arxiv.org/abs/1907.11692','https://aclanthology.org/2020.nlpmc-1.8/','https://aclanthology.org/2020.iwslt-1.33/'],experiments={})
    save(PLAN,p);print('Registered metric-first public-only development before any new validation scores',flush=True)

def units(record,cuts):
    ends=[0,*cuts,len(record['tokens'])];words=record.get('surface_words',record['tokens'])
    return [dict(text=' '.join(words[a:b])) for a,b in zip(ends,ends[1:]) if a<b]

def metric_result(records,predictions):
    """Metrics on complete paragraphs, not window-local count proxies."""
    rows=[];by_corpus=defaultdict(lambda:defaultdict(list))
    for r,pred in zip(records,predictions):
        cuts=[i+1 for i,v in enumerate(pred[:-1]) if v]
        gold=r['metric_reference'] if 'metric_reference' in r else auto.word_mlu(units(r,r['gold_boundaries']));mine=auto.word_mlu(units(r,cuts))
        errors={k:100*abs(mine[k]-gold[k])/abs(gold[k]) if mine[k] is not None and gold[k] else None for k in MEASURES}
        for k,v in errors.items():
            if v is not None:by_corpus[r['corpus']][(r['participant'],k)].append(v)
        rows.append(dict(corpus=r['corpus'],participant=r['participant'],recording=r['recording'],reference={k:gold[k] for k in MEASURES},predicted={k:mine[k] for k in MEASURES},ape=errors))
    summary={};corpus_values=[]
    for corpus,children in by_corpus.items():
        values={k:[float(np.mean(v)) for (child,m),v in children.items() if m==k] for k in MEASURES}
        means={k:float(np.mean(v)) if v else None for k,v in values.items()}
        score=float(np.mean([v for v in means.values() if v is not None]));corpus_values.append(score)
        summary[corpus]=dict(mean_participant_mean_recording_ape=means,median_participant_mean_recording_ape={k:float(np.median(v)) if v else None for k,v in values.items()},objective=score)
    return dict(objective=float(np.mean(corpus_values)),objective_units='mean APE percent, lower better',by_corpus=summary,recordings=rows)

def count_loss(logits,labels):
    valid=labels>=0;expected=(logits.softmax(-1)[...,1]*valid).sum(-1);gold=(labels==1).sum(-1)
    active=valid.any(-1)
    return (((expected-gold)/(gold+1))**2)[active].mean() if active.any() else logits.sum()*0


def expected_span_square(logits):
    """Exact E[sum(length**2)] for independent gap probabilities, in log space."""
    if not len(logits):return logits.sum()+1
    prefix=logits.log_softmax(-1)[:,0].cumsum(0)
    previous=torch.cat([prefix.new_zeros(1),-prefix[:-1]])
    pairs=(prefix+torch.logcumsumexp(previous,0)).exp().sum()
    return len(logits)+1+2*pairs


def length_moment_loss(logits,labels):
    """Train-only span dispersion proxy on contiguous known-label regions.

    The small auxiliary calculation runs on CPU for portable MPS support;
    differentiable copies retain gradients. No duration/length cap is imposed.
    """
    losses=[]
    for scores,target in zip(logits,labels):
        known=(target.detach().cpu().numpy()>=0)
        edges=np.diff(np.r_[False,known,False].astype(int));starts=np.flatnonzero(edges==1);ends=np.flatnonzero(edges==-1)
        for start,end in zip(starts,ends):
            local=scores[start:end].to('cpu');gold=target[start:end].detach().cpu()
            cuts=torch.cat([torch.tensor([0]),torch.nonzero(gold==1).flatten()+1,torch.tensor([len(gold)+1])])
            square=cuts.diff().square().sum().to(local.dtype)
            losses.append(((expected_span_square(local)-square)/(square+1)).square())
    return torch.stack(losses).mean().to(logits.device) if losses else logits.sum()*0

def evaluate_validation(model,t,records):
    pred=pretrained_predictions(model,t,records,batch_size=8)
    return dict(language_metrics=metric_result(records,pred),boundaries=metrics(records,pred))

def record(name,value):
    p=read()
    if 'selection' in p:raise ValueError('No development after selection')
    p['experiments'][name]=value;save(PLAN,p)

def incumbent():
    splits,p=public_development();torch.set_num_threads(4)
    model,t,meta=load_bundle(OUT/'context_segmenter')
    model.to('mps' if torch.backends.mps.is_available() else 'cpu')
    score=evaluate_validation(model,t,splits['validation'])
    record('incumbent',dict(status='complete',artifact=str(OUT/'context_segmenter'),parameters=sum(x.numel() for x in model.parameters()),validation=score))
    print('incumbent validation objective',score['language_metrics']['objective'],'F1',score['boundaries']['overall']['f1'],flush=True)

def train(name):
    from transformers import AutoTokenizer
    splits,p=public_development();torch.set_num_threads(4)
    device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    torch.use_deterministic_algorithms(True,warn_only=device.type=='mps');torch.manual_seed(42);np.random.seed(42);random.seed(42)
    if name not in p['candidates'] or name=='incumbent':raise ValueError('Unregistered candidate')
    if p['experiments'].get(name,{}).get('status')=='complete':raise ValueError('Completed candidate is frozen; do not retrain')
    sentence=name=='deberta_sentence';moments=name=='deberta_moments';metric=name.endswith('metric') or sentence or moments;continuation=name in {'deberta_metric','deberta_sentence','deberta_moments'};key=name.split('_')[0]
    if continuation:
        entry=p['experiments']['deberta_context']
        if entry.get('status')!='complete':raise ValueError('Complete context selection before continuation training')
        model,t,old=load_bundle(entry['artifact'])
        if sentence:
            original=model.rule_embedding.weight.detach().clone();model.rule_embedding=nn.Embedding(4,16)
            with torch.no_grad():model.rule_embedding.weight.copy_(original[[0,1,0,1]])
    else:
        pointer={'minilm':'speech_minilm_model_path.txt','chat':'speech_batchalign_model_path.txt'}.get(key)
        folder=str(WORK/'pretrained'/key)
        if pointer and not any((Path(folder)/f).exists() for f in ('pytorch_model.bin','model.safetensors')):folder=(Path('/private/tmp')/pointer).read_text().strip()
        t=AutoTokenizer.from_pretrained(folder,use_fast=True,local_files_only=True,**({'add_prefix_space':True} if key=='roberta' else {}))
        model=PretrainedGapModel(pretrained_path=folder)
    # Reuse audited public-training-only crops; one view per paragraph here.
    expanded=augmented(splits['train']);train=splits['train']+[r for r in expanded[len(splits['train']):] if r['recording'].endswith('#crop0')]
    if any(r['participant'] not in p['partitions']['train'] for r in train):raise ValueError('Training identity leak')
    validation=splits['validation']
    if sentence:
        import spacy
        nlp=spacy.load('en_core_web_sm');views=[]
        cache=WORK/'spoken_sentence_clause_development.jsonl'
        original=train+validation
        if cache.exists():
            views=[json.loads(s) for s in cache.read_text().splitlines()]
            if len(views)!=len(original) or any(a['tokens']!=b['tokens'] or a['participant']!=b['participant'] for a,b in zip(views,original)):raise ValueError('Automatic cue cache changed')
        else:
            for i,r in enumerate(original):
                views.append(dict(r,baseline=sentence_clause_features(r['tokens'],nlp)))
                if i%100==0:print('Public-only sentence cues',i,'/',len(original),flush=True)
            cache.write_text(''.join(json.dumps(r)+'\n' for r in views))
        train,validation=views[:len(train)],views[len(train):]
    windows=[]
    for ri,r in enumerate(train):windows.extend(dict(w,record=ri) for w in word_windows(r,t))
    windows=[w for w in windows if any(o and train[w['record']]['labels'][g]>=0 for g,o in zip(w['gaps'],w['owned']))]
    weights=None
    if metric:
        sizes=defaultdict(int)
        for w in windows:sizes[train[w['record']]['corpus']]+=1
        weights=[1/sizes[train[w['record']]['corpus']] for w in windows]
    model.to(device);optimizer=torch.optim.AdamW([dict(params=model.encoder.parameters(),lr=1e-5 if continuation else 2e-5),dict(params=[*model.rule_embedding.parameters(),*model.head.parameters()],lr=1e-4)],weight_decay=.01)
    epochs=3 if continuation else 6;steps=epochs*((len(windows)+15)//16);warm=max(1,int(steps*.1))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda s:min((s+1)/warm,max(0.,(steps-s)/max(1,steps-warm))))
    ce=nn.CrossEntropyLoss(ignore_index=-100);history=[];best=float('inf');stale=0;tail=deque(maxlen=3);start=time.monotonic()
    meta=dict(name=name,decoder='independent',decoding='argmax',threshold=None,input_sha256=p['development_data_sha256'],source_input_sha256=p['input_sha256'],protocol_sha256=digest(PLAN),max_length=256,stride=128,inference_batch_size=8,seed=42,device=str(device),pretrained=p['pretrained'][key],selection_scope='Public validation downstream metric error; no final inputs used',metric_loss=metric,rule_feature_kind='sentence_clause' if sentence else 'clause',code_sha256={str(f.relative_to(ROOT)):digest(f) for f in (Path(__file__),ROOT/'speech_pipeline/segmentation.py',ROOT/'speech_pipeline/corpora.py')})
    destination=WORK/name
    print(name,len(windows),'windows',sum(x.numel() for x in model.parameters()),'parameters',device,flush=True)
    for epoch in range(epochs):
        order=random.choices(windows,weights=weights,k=len(windows)) if metric else random.sample(windows,len(windows))
        model.train();losses=[];optimizer.zero_grad(set_to_none=True)
        for at in range(0,len(order),8):
            batch=order[at:at+8];inputs,labels=collate(batch,t.pad_token_id,train);inputs={k:v.to(device) for k,v in inputs.items()};labels=labels.to(device)
            if metric:
                ids=inputs['input_ids'];allowed=(ids!=t.cls_token_id)&(ids!=t.sep_token_id)&(ids!=t.pad_token_id)
                ids[(torch.rand(ids.shape,device=device)<.05)&allowed]=t.unk_token_id
            emissions=model(**inputs);loss=ce(emissions.reshape(-1,2),labels.reshape(-1))
            if metric:loss=loss+count_loss(emissions,labels)
            if moments:loss=loss+length_moment_loss(emissions,labels)
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            (loss/2).backward();losses.append(float(loss.detach()))
            if (at//8+1)%2==0 or at+8>=len(order):
                nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True)
            if (at//8+1)%25==0:print(name,'epoch',epoch+1,'batch',at//8+1,'/',(len(order)+7)//8,flush=True)
        if device.type=='mps':torch.mps.empty_cache()
        score=evaluate_validation(model,t,validation);objective=score['language_metrics']['objective']
        history.append(dict(epoch=epoch+1,train_loss=float(np.mean(losses)),validation_objective=objective,validation_f1=score['boundaries']['overall']['f1']))
        print(name,'epoch',epoch+1,'metric objective',objective,'F1',history[-1]['validation_f1'],flush=True)
        tail.append({k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
        if objective<best:
            best=objective;stale=0;save_bundle(model,t,destination,dict(meta,selected_epoch=epoch+1,history=history,parameter_average=False))
        else:stale+=1
        record(name,dict(status='training',history=history,artifact=str(destination),parameters=sum(x.numel() for x in model.parameters()),elapsed_seconds=time.monotonic()-start))
        if device.type=='mps':torch.mps.empty_cache()
        if stale>=2:break
    if len(tail)==3:
        average={k:torch.stack([v[k] for v in tail]).mean(0) for k in tail[0]};model.load_state_dict(average)
        score=evaluate_validation(model,t,validation);objective=score['language_metrics']['objective']
        history.append(dict(parameter_average=True,validation_objective=objective,validation_f1=score['boundaries']['overall']['f1']))
        if objective<best:save_bundle(model,t,destination,dict(meta,selected_epoch=None,history=history,parameter_average=True))
    del model,optimizer,tail;gc.collect()
    if device.type=='mps':torch.mps.empty_cache()
    model,t,saved=load_bundle(destination);model.to(device);score=evaluate_validation(model,t,validation)
    expected=next(row for row in reversed(history) if row.get('parameter_average',False)==saved['parameter_average'] and
        (saved['parameter_average'] or row.get('epoch')==saved['selected_epoch']))
    if not (np.isclose(score['language_metrics']['objective'],expected['validation_objective'],rtol=0,atol=1e-10) and
            np.isclose(score['boundaries']['overall']['f1'],expected['validation_f1'],rtol=0,atol=1e-10)):
        raise ValueError('Saved bundle changed validation predictions after reload; preserve evidence and keep final cohorts sealed')
    record(name,dict(status='complete',validation=score,history=history,artifact=str(destination),parameters=sum(x.numel() for x in model.parameters()),selected_epoch=saved['selected_epoch'],parameter_average=saved['parameter_average'],device=str(device),elapsed_seconds=time.monotonic()-start))
    print(name,'complete objective',score['language_metrics']['objective'],flush=True)

def select():
    """Freeze public-validation choice and deployment before final access."""
    import shutil
    p=read()
    if 'selection' in p:raise ValueError('Selection already frozen; do not choose from final outcomes')
    if any(p['experiments'].get(n,{}).get('status')!='complete' for n in p['candidates']):raise ValueError('Complete every registered candidate before selection')
    entries=p['experiments']
    winner=min(p['candidates'],key=lambda n:(entries[n]['validation']['language_metrics']['objective'],-entries[n]['validation']['boundaries']['overall']['f1'],entries[n]['parameters'],n))
    source=Path(entries[winner]['artifact']);choice=dict(name=winner,validation_objective=entries[winner]['validation']['language_metrics']['objective'],validation_f1=entries[winner]['validation']['boundaries']['overall']['f1'],weights_sha256=digest(source/'model.safetensors'),recorded_utc=datetime.now(timezone.utc).isoformat(),before_final_evaluation=True)
    p['selection']=choice;p['status']='selected';save(PLAN,p)
    target=OUT/'metric_segmenter';target.mkdir(exist_ok=True)
    for filename in ('model.safetensors','tokenizer.json','segmenter.json'):shutil.copy2(source/filename,target/filename)
    metadata=json.loads((target/'segmenter.json').read_text());metadata.update(name=winner,decoding='argmax',threshold=None,inference_batch_size=8,selection_scope=p['primary_selection'],metric_round=True)
    save(target/'segmenter.json',metadata)
    if digest(target/'model.safetensors')!=choice['weights_sha256']:raise ValueError('Deployment copy changed weights')
    print('Frozen metric-first selection',json.dumps(choice),flush=True)


def evaluate():
    """Final stage only: never revise the model or routing after seeing scores."""
    from openpyxl import load_workbook
    from speech_pipeline import taaled
    from speech_pipeline.segmentation import Segmenter
    p=read()
    if 'selection' not in p:raise ValueError('Freeze selection before opening final records')
    path=OUT/'improved_model.json';previous=json.loads(path.read_text())
    if 'metric_round' in previous:
        print('Final evaluation already saved; not rerunning or reselecting',flush=True);return previous['metric_round']
    if digest(OUT/'metric_segmenter/model.safetensors')!=p['selection']['weights_sha256']:raise ValueError('Frozen weights changed')
    p['final_evaluation_started_utc']=datetime.now(timezone.utc).isoformat();save(PLAN,p)
    splits,_=load_partitions(include_test=True)
    import spacy
    test,test_audit=spoken_records(splits['test'],set(p['partitions']['test']),spacy.load('en_core_web_sm'))
    test_path=EVIDENCE/'talkbank_spoken_test.jsonl';test_path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in test))
    p=read();p['test_data_sha256']=digest(test_path);p['spoken_test_preparation_audit']=test_audit;save(PLAN,p)
    for r in test:
        if digest(ROOT/'dataset/public'/r['corpus']/r['recording'].split('#')[0])!=r['source_sha256']:raise ValueError('Public test source changed')
    source_meta=json.loads((EVIDENCE/'boundary_dataset_sources.json').read_text())
    for filename,h in source_meta['sources'].items():
        if digest(ROOT/filename)!=h:raise ValueError('Private source/reference changed')
    if digest(EVIDENCE/'boundary_dataset.jsonl')!=source_meta['dataset_sha256']:raise ValueError('Private frozen dataset changed')
    private=[json.loads(line) for line in (EVIDENCE/'boundary_dataset.jsonl').read_text().splitlines()]
    workbook=load_workbook(ROOT/'dataset/annotations/all_nafee_utterances.xlsx',read_only=True,data_only=True)
    human={(s.title,name):[dict(text=auto.plain(text)) for text in texts] for s in workbook for name,texts in auto.flatten_labels(s).items()};workbook.close()
    segmenter=Segmenter(OUT/'metric_segmenter');model,t=segmenter.model,segmenter.tokenizer
    selected_input=[dict(r,baseline=sentence_clause_features(r['tokens'],segmenter.nlp)) for r in test] if segmenter.metadata.get('rule_feature_kind')=='sentence_clause' else test
    public_pred=pretrained_predictions(model,t,selected_input,batch_size=8)
    baseline=Segmenter(OUT/'context_segmenter',segmenter.nlp)
    baseline_public=pretrained_predictions(baseline.model,baseline.tokenizer,test,batch_size=8)
    mapping=taaled.load_target_speakers();private_pred=[];private_units=[];baseline_private=[];baseline_units=[];target=[]
    for r in private:
        raw=ROOT/'dataset/uncorrected'/r['participant']/r['recording']
        us=segmenter.target_paragraph(list(auto.read_rows(raw)),mapping[r['participant'],r['recording']]);words,cuts=auto.stream(us)
        if words!=r['tokens']:raise ValueError('Private production word stream changed')
        reference=human[r['participant'],r['recording']]
        # Boundary labels retain the frozen alignment; metric references use human words.
        target.append(dict(r,surface_words=[w for u in us for w in auto.WORD.findall(u['text'])],metric_reference=auto.word_mlu(reference)))
        private_pred.append(np.array([i+1 in cuts for i in range(len(words))],dtype=bool));private_units.append(us)
        old_us=baseline.target_paragraph(list(auto.read_rows(raw)),mapping[r['participant'],r['recording']]);old_words,old_cuts=auto.stream(old_us)
        if old_words!=words:raise ValueError('Matched incumbent changed words')
        baseline_private.append(np.array([i+1 in old_cuts for i in range(len(words))],dtype=bool));baseline_units.append(old_us)
    result=dict(selected=p['selection'],validation=p['experiments'][p['selection']['name']]['validation'],
        primary_task='Target-speaker unpunctuated transcript paragraph to human CLAN/CHAT utterances and supported language measures; audio deferred',
        public_test=dict(boundaries=metrics(test,public_pred),language_metrics=metric_result(test,public_pred)),
        private_paragraphs=dict(boundaries=metrics(target,private_pred),language_metrics=metric_result(target,private_pred)),
        input_sha256=p['development_data_sha256'],private_data_sha256=digest(EVIDENCE/'boundary_dataset.jsonl'),inference_device=str(segmenter.device),
        evaluation_status='Reused research benchmarks. All latest choices frozen on public train/validation; no post-test tuning, including no tuning on the 10 participants.')
    for row,us in zip(result['private_paragraphs']['boundaries']['recordings'],private_units):row['utterance_texts']=[u['text'] for u in us]
    result['incumbent']=dict(name='Previous frozen MiniLM, evaluated on identical corrected inputs',weights_sha256=digest(OUT/'context_segmenter/model.safetensors'),
        public_test=dict(boundaries=metrics(test,baseline_public),language_metrics=metric_result(test,baseline_public)),
        private_paragraphs=dict(boundaries=metrics(target,baseline_private),language_metrics=metric_result(target,baseline_private)))
    for row,us in zip(result['incumbent']['private_paragraphs']['boundaries']['recordings'],baseline_units):row['utterance_texts']=[u['text'] for u in us]
    previous['metric_round']=result;save(path,previous);p=read();p['status']='evaluated';save(PLAN,p)
    print('Final predictions saved. No more development in this round.',flush=True)
    print('Public final language objective',result['public_test']['language_metrics']['objective'],'boundary F1',result['public_test']['boundaries']['overall']['f1'],flush=True)
    print('Private final language objective',result['private_paragraphs']['language_metrics']['objective'],'boundary F1',result['private_paragraphs']['boundaries']['overall']['f1'],flush=True)
    return result


def downstream():
    """Same frozen predictions through established metric and CLAN calculators."""
    import csv,statistics,spacy
    from tempfile import TemporaryDirectory
    from speech_pipeline.metrics import clinical_from_predictions
    from speech_pipeline.pipeline import lexical,clan_software_check
    p=read()
    if p.get('status')!='evaluated':raise ValueError('Complete frozen final inference first')
    path=OUT/'improved_model.json';saved=json.loads(path.read_text());result=saved['metric_round']
    if digest(OUT/'metric_segmenter/model.safetensors')!=result['selected']['weights_sha256']:raise ValueError('Do not score changed weights')
    test_path=EVIDENCE/'talkbank_spoken_test.jsonl'
    if digest(test_path)!=p['test_data_sha256']:raise ValueError('Corrected frozen public test changed')
    test=[json.loads(line) for line in test_path.read_text().splitlines()]
    private=[json.loads(line) for line in (EVIDENCE/'boundary_dataset.jsonl').read_text().splitlines()]
    records=test+private;index={(r['corpus'],r['participant'],r['recording']):r for r in records}
    source_meta=json.loads((EVIDENCE/'boundary_dataset_sources.json').read_text())
    for filename,h in source_meta['sources'].items():
        if digest(ROOT/filename)!=h:raise ValueError('Frozen source changed')
    table_path=OUT/'improved_metric_errors.csv'
    with table_path.open(encoding='utf-8-sig',newline='') as f:old=list(csv.DictReader(f))
    gold={}
    for row in old:
        if row['method']=='improved' and row['corpus']=='existing_development_cohort':gold.setdefault((row['corpus'],row['participant'],row['recording']),{})[row['metric']]=float(row['reference']) if row['reference'] else None
    nlp=spacy.load('en_core_web_sm');table=[];checks=[];chosen=defaultdict(set);coverage=[]
    def score(us,temp,key):
        values={f'surface/{k}':v for k,v in clinical_from_predictions(us)['values'].items()}
        values.update({f'utterances/{k}':v for k,v in auto.word_mlu(us).items() if k in ('total_utterances','mlu_nutterances','mlu_words_sd')})
        try:
            lex=lexical(us,nlp,temp/'lexical.txt')
            values.update({f'custom_normalized/{k}':v['custom'] for k,v in lex.items()});values.update({f'taaled/{k}':v['reference'] for k,v in lex.items()})
        except (ValueError,ZeroDivisionError) as error:coverage.append(dict(corpus=key[0],participant=key[1],recording=key[2],detail=str(error)))
        return values
    with TemporaryDirectory(prefix='metric_selected_checks_') as temp:
        temp=Path(temp)
        # Recompute public gold metrics on corrected spoken words, never reuse
        # the historical event-contaminated reference values.
        for r in test:
            key=r['corpus'],r['participant'],r['recording'];gold[key]=score(units(r,r['gold_boundaries']),temp,key)
        for method,container in [('metric_baseline',result['incumbent']),('metric_selected',result)]:
            predictions=container['public_test']['boundaries']['recordings']+container['private_paragraphs']['boundaries']['recordings']
            for i,pred in enumerate(predictions):
                key=pred['corpus'],pred['participant'],pred['recording'];r=index[key]
                us=[dict(text=s) for s in pred['utterance_texts']] if 'utterance_texts' in pred else units(r,pred['predicted_boundaries'])
                mine=score(us,temp,key)
                for metric,truth in gold[key].items():
                    value=mine.get(metric)
                    table.append(dict(method=method,corpus=r['corpus'],participant=r['participant'],recording=r['recording'],gem=r.get('gem'),metric=metric,reference=truth,predicted=value,
                        absolute_error=abs(value-truth) if value is not None and truth is not None else None,
                        absolute_percent_error=100*abs(value-truth)/abs(truth) if value is not None and truth else None,status='available' if value is not None and truth is not None else 'unavailable_required_annotation_or_sample_size'))
                if method=='metric_selected' and (r['corpus']=='existing_development_cohort' or len(chosen[r['corpus']])<3 and r['participant'] not in chosen[r['corpus']]):
                    chosen[r['corpus']].add(r['participant']);checks.append(dict(corpus=r['corpus'],participant=r['participant'],recording=r['recording'],clan=clan_software_check(us,temp)))
                if i%20==0:print(method,'frozen metric checks',i+1,'/',len(predictions),flush=True)
    retained=[r for r in old if r['method'] not in {'metric_baseline','metric_selected'}]
    with table_path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(retained+table)
    audit_path=EVIDENCE/'improved_calculator_checks.json';audit=json.loads(audit_path.read_text());audit['metric_round']=checks;audit['metric_round_coverage']=coverage;save(audit_path,audit)
    for method,container in [('metric_baseline',result['incumbent']),('metric_selected',result)]:
        container['downstream']={}
        for corpus in ('ocsc','nyu_emerson','existing_development_cohort','all_public'):
            summary={}
            for metric in ('surface/MLUw','surface/TNW','surface/NDW','surface/TTR','surface/MATTR','utterances/total_utterances','utterances/mlu_words_sd'):
                children=defaultdict(list)
                for r in table:
                    if r['method']==method and r['metric']==metric and (r['corpus'] in {'ocsc','nyu_emerson'} if corpus=='all_public' else r['corpus']==corpus) and r['absolute_percent_error'] is not None:children[r['participant']].append(r['absolute_percent_error'])
                values=[statistics.mean(v) for v in children.values()]
                summary[metric]=dict(available=sum(len(v) for v in children.values()),median_child_mean_ape=statistics.median(values) if values else None,mean_child_mean_ape=statistics.mean(values) if values else None)
            container['downstream'][corpus]=summary
    computed=[c for c in checks if c['clan']['status']=='software_reference_computed']
    passed=all(v['absolute_error'] is None or v['absolute_error']<=(0 if k in ('TNW','NDW','mlu_nwords','mlu_nutterances') else .000501) for c in computed for group in ('metrics','requested_metrics') for k,v in c['clan'][group].items())
    result['calculator_checks']=dict(records=len(checks),computed_records=len(computed),all_within_display_precision=passed)
    result['uncertainty']=metric_intervals(table)
    result['objective_uncertainty']={scope:objective_intervals(result['incumbent'][key]['language_metrics'],result[key]['language_metrics'])
        for scope,key in [('public','public_test'),('private','private_paragraphs')]}
    saved['metric_round']=result;save(path,saved)
    if not passed:raise ValueError('Official calculator discrepancy; preserve frozen model and investigate arithmetic only')
    print('Saved final downstream metric errors; model unchanged',flush=True)


def objective_intervals(before,after):
    """Paired participant bootstrap of the declared corpus-balanced objective."""
    rng=np.random.default_rng(42);drawn={};estimates=[]
    for result in (before,after):
        grouped=defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
        for row in result['recordings']:
            for measure,value in row['ape'].items():
                grouped[row['corpus']][row['participant']][measure]
                if value is not None:grouped[row['corpus']][row['participant']][measure].append(value)
        corpus_draws=[]
        for corpus,children in sorted(grouped.items()):
            ids=sorted(children)
            matrix=np.array([[np.mean(children[child][m]) if children[child][m] else np.nan for m in MEASURES] for child in ids])
            if corpus not in drawn:
                if estimates:raise ValueError('Paired objective intervals require the same corpora')
                drawn[corpus]=(ids,rng.integers(0,len(ids),size=(10000,len(ids))))
            expected,indices=drawn[corpus]
            if ids!=expected:raise ValueError('Paired objective intervals require the same participant groups')
            # Match the objective's missing-denominator policy within each draw.
            values=matrix[indices];counts=np.isfinite(values).sum(1)
            means=np.divide(np.nansum(values,axis=1),counts,out=np.full(counts.shape,np.nan),where=counts>0)
            active=np.isfinite(means).sum(1)
            corpus_draws.append(np.divide(np.nansum(means,axis=1),active,out=np.full(len(indices),np.nan),where=active>0))
        if set(grouped)!=set(drawn):raise ValueError('Paired objective intervals require the same corpora')
        estimates.append(np.mean(corpus_draws,axis=0))
    valid=np.isfinite(estimates[0])&np.isfinite(estimates[1])
    if not valid.any():raise ValueError('No evaluable composite bootstrap draws')
    return dict(method='10,000 paired participant-cluster bootstrap resamples stratified by corpus; same recording/participant/measure/corpus averaging and missingness policy as primary objective; descriptive reused-benchmark uncertainty',
        valid_draws=int(valid.sum()),participants_by_corpus={c:len(ids) for c,(ids,_) in drawn.items()},
        selected_error_ci=np.quantile(estimates[1][valid],[.025,.975]).tolist(),
        difference_ci=np.quantile((estimates[1]-estimates[0])[valid],[.025,.975]).tolist())


def metric_intervals(rows):
    """Paired child-cluster intervals; repeated benchmark use is disclosed."""
    result={};rng=np.random.default_rng(42)
    for scope in ('public','private'):
        result[scope]={}
        for metric in ('surface/MLUw','utterances/total_utterances','utterances/mlu_words_sd'):
            children={'metric_baseline':defaultdict(list),'metric_selected':defaultdict(list)}
            for r in rows:
                relevant=r['corpus'] in {'ocsc','nyu_emerson'} if scope=='public' else r['corpus']=='existing_development_cohort'
                value=r.get('absolute_percent_error')
                if relevant and r['metric']==metric and r['method'] in children and value not in (None,''):
                    children[r['method']][r['participant']].append(float(value))
            ids=sorted(set(children['metric_baseline'])&set(children['metric_selected']))
            if not ids:continue
            before=np.array([np.mean(children['metric_baseline'][i]) for i in ids]);after=np.array([np.mean(children['metric_selected'][i]) for i in ids])
            draws=rng.integers(0,len(ids),size=(10000,len(ids)))
            new=np.median(after[draws],axis=1);change=new-np.median(before[draws],axis=1)
            result[scope][metric]=dict(participants=len(ids),method='10,000 paired participant-cluster bootstrap resamples; descriptive on reused benchmarks; median child mean recording APE',
                median_error_ci=np.quantile(new,[.025,.975]).tolist(),difference_ci=np.quantile(change,[.025,.975]).tolist())
    return result


def runtime():
    """Audit a completed canonical run without any model/routing changes."""
    from importlib.metadata import version
    from speech_pipeline.pipeline import METRICS
    p=read()
    if p.get('status')!='evaluated':raise ValueError('Final access requires completed frozen evaluation')
    path=OUT/'improved_model.json';saved=json.loads(path.read_text());result=saved['metric_round'];choice=p['selection']
    frozen={(r['participant'],r['recording']):r for r in result['private_paragraphs']['boundaries']['recordings']}
    checked=0;matched=0;reports=[];maxima={m:0. for m in METRICS}
    for condition in ('uncorrected','corrected'):
        folder=ROOT/'outputs/results/pipeline'/condition;rpt=json.loads((folder/'comparison.json').read_text())
        if rpt.get('boundary_context')!='paragraph' or rpt['segmenter']['selected']!=choice['name'] or rpt['segmenter']['artifacts']['model.safetensors']!=choice['weights_sha256']:raise ValueError('Canonical outputs do not use frozen selected paragraph model')
        if len(rpt['recordings'])!=37 or len(rpt['patients'])!=10:raise ValueError('Unexpected original-cohort run coverage')
        if rpt['segmenter']['threshold'] is not None or rpt['segmenter']['decoding']!='argmax':raise ValueError('Unexpected inference decision rule')
        for record in rpt['recordings']:
            us=json.loads((folder/record['utterance_file']).read_text())['utterances'];raw=ROOT/'dataset'/condition/record['patient_id']/record['source_file']
            if digest(raw)!=record['raw_sha256'] or digest(folder/record['chat_file'])!=record['chat_sha256']:raise ValueError('Run source/CHAT hash mismatch')
            text=' '.join(row['text'] for row in auto.read_rows(raw) if row['speaker'].casefold()==record['target_speaker'].casefold()) if condition=='uncorrected' else raw.read_text()
            words,cuts=auto.stream(us)
            if words!=[w.casefold().replace('’',"'") for w in auto.WORD.findall(text)]:raise ValueError('Run changed source words')
            if any('turn_id' in u or u['context']!='one_target_speaker_paragraph' for u in us):raise ValueError('Source turns leaked into paragraph mode')
            key=record['patient_id'],record['source_file']
            if condition=='uncorrected' and key in frozen:
                if sorted(cuts)!=frozen[key]['predicted_boundaries'] or [u['text'] for u in us]!=frozen[key]['utterance_texts']:raise ValueError('Canonical and frozen private predictions differ')
                matched+=1
            if set(record['lexical'])!=set(METRICS):raise ValueError('Missing TAALED comparisons')
            for m,v in record['lexical'].items():
                if v['absolute_error'] is not None:maxima[m]=max(maxima[m],v['absolute_error'])
            clan=record['clan']
            if clan['status']!='software_reference_computed':raise ValueError('Missing actual CLAN check')
            for group in ('metrics','requested_metrics'):
                for m,v in clan[group].items():
                    if v['absolute_error'] is not None and v['absolute_error']>(0 if m in ('TNW','NDW','mlu_nutterances','mlu_nwords') else .000501):raise ValueError('CLAN calculator discrepancy')
            if any(record['requested_metrics']['values'][m] is not None for m in ('MLUm','WPM','PGU','omission_count','substitution_count','maze_percentage')):raise ValueError('Annotation-dependent metric invented')
            checked+=1
        expected={Path(r['chat_file']).name for r in rpt['recordings']}
        if {f.name for f in (folder/'clan_software_input_files').glob('*_auto_*.cha')}!=expected:raise ValueError('Stale current CHAT inputs')
        reports.append(dict(dataset=condition,comparison_sha256=digest(folder/'comparison.json'),device=rpt['segmenter']['device']))
    if matched!=14:raise ValueError('Not all labeled recording predictions audited')
    public=[json.loads(line) for line in (EVIDENCE/'talkbank_training.jsonl').read_text().splitlines()]
    for r in public:
        if digest(ROOT/'dataset/public'/r['corpus']/r['recording'].split('#')[0])!=r['source_sha256']:raise ValueError('Frozen public source changed')
    publications=saved['reassessment']['runtime_verification']['publication_sha256']
    for filename,h in publications.items():
        if digest(ROOT/filename)!=h:raise ValueError('Final publication changed')
    if digest(OUT/'metric_segmenter/model.safetensors')!=choice['weights_sha256']:raise ValueError('Final weights changed')
    result['runtime_verification']=dict(checked_recordings=checked,exact_frozen_private_predictions=matched,conditions=reports,source_words_preserved=True,all_clan_within_display_precision=True,length_sd_verified=True,taaled_measures_per_recording=17,maximum_pipeline_taaled_absolute_errors=maxima,taaled_scope='Separate preprocessing retained; differences not described as exact calculator agreement',public_sources_unchanged=len(public),publication_sha256=publications,selected_weights_unchanged=True)
    saved['metric_round']=result;save(path,saved)
    evidence_path=EVIDENCE/'software.json';evidence=json.loads(evidence_path.read_text())
    evidence['metric_round']=dict(selected=choice,protocol_sha256=digest(PLAN),packages={n:version(n) for n in ('torch','transformers','safetensors','sentencepiece','protobuf','numpy','spacy','en-core-web-sm','taaled','pylats')},
        production_bundle={f.name:digest(f) for f in (OUT/'metric_segmenter').iterdir() if f.is_file()},pretrained=p['pretrained'],verification=result['runtime_verification'],
        final_code_sha256={str(f.relative_to(ROOT)):digest(f) for f in [*sorted((ROOT/'speech_pipeline').glob('*.py')),Path(__file__),ROOT/'speech_pipeline/research/validation.py']},
        scope='Transcript paragraphs only; audio deferred. Final groups opened only after public-validation selection. No post-final tuning.')
    save(evidence_path,evidence);print('Verified 74 recording-condition outputs, all 14 frozen private predictions, CLAN, source words and publications',flush=True)


def report_section():
    """Only saved validation evidence during development; final results after lock."""
    if not PLAN.exists():return []
    p=read();lines=['## Metric-first paragraph development','',
        'The user confirmed CLAN/CHAT utterance units and clarified that low final language-measure error is the goal; a 90% boundary score is not the acceptance criterion. Audio work is deferred. This round processes unpunctuated target-speaker transcript paragraphs, with human CHAT formatting withheld. The original 233/49/49 public child partitions are retained; test and all 10 original participants are final evaluation only.',
        '', 'Candidate selection minimizes the equally averaged count, surface MLUw and population length-SD absolute percentage errors, first averaging recordings within each child, then children within each corpus, then equally weighting the two public corpora. Zero-reference denominators are unavailable rather than zero-filled. This is a declared development objective, not a claim that every metric improves. Boundary precision/recall/F1 and exact spans remain separate diagnostics of agreement with human units.',
        '', 'Published automated and human comparisons are documented in the [research review](../../../speech_pipeline/research/SEGMENTATION_RESEARCH.md#human-comparisons-and-metric-first-benchmarks-verified-october-15-2026), with sourced numerical records in [literature_benchmarks.json](../../validation/benchmark/literature_benchmarks.json). Different populations, input stages, annotation units and aggregation prevent a direct ranking. Human percent agreement, consistency ICC, transcription WER, boundary F1 and downstream metric errors are distinct quantities.',
        '', 'Larger DeBERTa and RoBERTa encoders, metric-oriented count supervision, training-only token corruption, corpus-balanced sampling and independent automatic spaCy sentence cues are compared. Every model still learns human boundary labels. Neither inference nor postprocessing receives gold utterance lengths/counts. No predicted cut is forced to satisfy a reference metric. The MOSS implementation is not present and is not claimed to have run. Transformer training and inference use local MPS when available, with CPU fallback.',
        '', '| Public-validation candidate | Status | Composite metric error (lower better) | Boundary F1 diagnostic |','| --- | --- | ---: | ---: |']
    for name in p['candidates']:
        e=p['experiments'].get(name,{})
        if e.get('status')=='complete':lines.append(f'| {name} | complete | {e["validation"]["language_metrics"]["objective"]:.2f}% | {100*e["validation"]["boundaries"]["overall"]["f1"]:.2f}% |')
        else:lines.append(f'| {name} | {e.get("status","pending")} | — | — |')
    if p.get('deadline_amendment'):
        lines+=['','The user required a five-minute finish. The remaining DeBERTa moments training was interrupted and excluded before final evaluation; seven completed candidates remained eligible. The original registration and amendment are preserved in the protocol.']
    if p.get('completion_limits'):
        lines+=['',p['completion_limits']]
    if p.get('status')!='evaluated':return lines+['','This round is still in development or awaiting frozen final evaluation. No new public-test or original-participant result is claimed. Previous rounds below remain historical.','']
    r=json.loads((OUT/'improved_model.json').read_text())['metric_round']
    lines+=['',f'The frozen public-validation selection was **{r["selected"]["name"]}**. Selection and production bundle were saved before opening final records. Final transformer inference device: `{r["inference_device"]}`.',
        '', 'The following is the declared composite metric-error objective, in mean APE percentage points. Public children are averaged within corpus before equally weighting corpora; the original cohort has one corpus. Lower is better.',
        '', '| Final scope | Previous MiniLM | Selected model | Change |','| --- | ---: | ---: | ---: |']
    for label,key in [('Public test','public_test'),('Original participants','private_paragraphs')]:
        a,b=r['incumbent'][key]['language_metrics']['objective'],r[key]['language_metrics']['objective']
        lines.append(f'| {label} | {a:.2f} | {b:.2f} | {b-a:+.2f} |')
    lines+=['','Available recording-level APE pairs are shown explicitly. Zero reference denominators and unavailable predictions are excluded from the corresponding error average; they are not filled with zero error.',
        '', '| Scope / model | Paragraphs | Count APE pairs | MLUw APE pairs | SD APE pairs |','| --- | ---: | ---: | ---: | ---: |']
    for scope,key in [('Public','public_test'),('Original','private_paragraphs')]:
        for label,container in [('previous',r['incumbent']),('selected',r)]:
            rows=container[key]['language_metrics']['recordings']
            available=[str(sum(row['ape'][m] is not None for row in rows)) for m in MEASURES]
            lines.append(f'| {scope} / {label} | {len(rows)} | '+' | '.join(available)+' |')
    lines+=[
        '', '| Final paragraph boundary diagnostic | Paragraphs | Participants | Precision | Recall | F1 | Exact span F1 |','| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for label,e in [('OCSC',r['public_test']['boundaries']['by_corpus']['ocsc']),('NYU–Emerson',r['public_test']['boundaries']['by_corpus']['nyu_emerson']),('All public test',r['public_test']['boundaries']['overall']),('Original 10 participants',r['private_paragraphs']['boundaries']['overall'])]:
        exact=e['exact'];lines.append(f'| {label} | {e["records"]} | {e["participants"]} | {100*e["precision"]:.2f}% | {100*e["recall"]:.2f}% | {100*e["f1"]:.2f}% | '+(f'{100*exact["f1"]:.2f}%' if exact else 'unavailable')+' |')
    lines+=['','The private evaluation uses 14 labeled recordings across all 10 participants, one target-speaker paragraph per recording. The 37 source recordings are not 37 independent participants and do not all have these gold labels. Both final benchmarks have historical inspection; these are reused research evaluations, not pristine confirmatory tests.']
    lines+=['','Individual participant results use mean recording APE within each participant. Boundary F1 pools the participant\'s eligible gaps; it is not the percentage of complete utterances recovered.',
        '', '| Participant | Labeled recordings | Boundary F1 | Count error | MLUw error | Length-SD error |','| --- | ---: | ---: | ---: | ---: | ---: |']
    private=r['private_paragraphs']
    for participant,boundary in sorted(private['boundaries']['by_child'].items(),key=lambda item:(len(item[0]),item[0])):
        rows=[row for row in private['language_metrics']['recordings'] if row['participant']==participant]
        errors=[]
        for measure in MEASURES:
            values=[row['ape'][measure] for row in rows if row['ape'][measure] is not None]
            errors.append(f'{np.mean(values):.2f}%' if values else 'unavailable')
        lines.append(f'| {participant} | {boundary["records"]} | {100*boundary["f1"]:.2f}% | '+' | '.join(errors)+' |')
    if 'downstream' in r:
        old=r['incumbent']['downstream']
        lines+=['','The following uses **median child mean recording APE**, matching the prior paragraph comparison. It is different from the corpus-balanced mean objective used for validation selection. Lower is better. Entries are previous MiniLM → new selected model.',
            '', '| Metric error | OCSC | NYU–Emerson | Original participants |','| --- | ---: | ---: | ---: |']
        for label,key in [('Utterance count','utterances/total_utterances'),('MLUw','surface/MLUw'),('Length SD','utterances/mlu_words_sd'),('TNW','surface/TNW'),('NDW','surface/NDW'),('TTR','surface/TTR'),('MATTR-50','surface/MATTR')]:
            cells=[]
            for corpus in ('ocsc','nyu_emerson','existing_development_cohort'):
                a,b=old[corpus][key]['median_child_mean_ape'],r['downstream'][corpus][key]['median_child_mean_ape']
                cells.append(' → '.join(f'{v:.2f}%' if v is not None else 'unavailable' for v in (a,b)))
            lines.append(f'| {label} | '+' | '.join(cells)+' |')
        c=r['calculator_checks'];lines+=['',f'Official CLAN calculator checks: {c["computed_records"]}/{c["records"]} selected predictions computed; agreement within displayed precision: **{c["all_within_display_precision"]}**. Counts must match exactly; ratio and SD tolerance .000501 reflects CLAN\'s three-decimal output, not a segmentation threshold. All 17 TAALED comparisons use the same selected segments. Per-record errors append `method=metric_selected` to the existing CSV; prior rows remain.']
        lines+=['','Zero lexical error when source words are fixed does not establish ASR accuracy or correct utterances. Changed grouping of unintelligible material can change eligible words. Surface measures, TAALED normalization and fully coded original CHAT/CLAN analyses are different scopes. MLUm, WPM, PGU, production-error and maze measures remain unavailable without the required supported annotations; they are not invented from boundary predictions.']
    if 'runtime_verification' in r:
        v=r['runtime_verification'];lines+=['',f'The selected model was run through the full paragraph pipeline for {v["checked_recordings"]} recording-condition outputs. Source-word preservation, 17 TAALED comparisons per recording, frozen private predictions and actual CLAN checks were verified. Canonical outputs are in `outputs/results/pipeline/`; established speaker-turn workbook routing and final publications remain unchanged.']
    if 'uncertainty' in r:
        lines+=['','Descriptive 95% paired participant-cluster bootstrap intervals (10,000 resamples). Error difference is new minus previous; negative favors the new model. Historical benchmark exposure prevents a fresh confirmatory claim.',
            '', '| Scope / measure | New median error interval | Error difference interval |','| --- | ---: | ---: |']
        for scope,items in r['uncertainty'].items():
            for metric,v in items.items():
                a,b=v['median_error_ci'];c,d=v['difference_ci'];lines.append(f'| {scope}: {metric} | {a:.2f}–{b:.2f}% | {c:.2f}–{d:.2f} percentage points |')
    if 'objective_uncertainty' in r:
        lines+=['','The primary composite also has paired participant bootstrap intervals, stratified by corpus to retain equal corpus weighting:',
            '', '| Composite scope | Selected error interval | Change interval |','| --- | ---: | ---: | ---: |']
        for scope,v in r['objective_uncertainty'].items():
            a,b=v['selected_error_ci'];c,d=v['difference_ci'];lines.append(f'| {scope} | {a:.2f}–{b:.2f} | {c:.2f}–{d:.2f} percentage points |')
    lines+=['','The primary-source literature review includes [DeBERTaV3](https://arxiv.org/abs/2111.09543), [RoBERTa](https://arxiv.org/abs/1907.11692), [robust medical ASR punctuation](https://aclanthology.org/2020.nlpmc-1.8/) and [overlapping-window inference](https://aclanthology.org/2020.iwslt-1.33/). Their published scores are not transferred to these data. Synthetic token corruption here is a limited robustness experiment, not reproduction of real n-best ASR augmentation. Candidate details, protocol amendments and selection timing are in `outputs/validation/benchmark/metric_training.json`. No final outcome is used to revise this completed round.','']
    return lines


def report():
    from speech_pipeline.research.validation import write_report
    write_report(json.loads((OUT/'existing_cohort.json').read_text()))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['fetch-models','register','prepare','incumbent','deberta_context','deberta_metric','deberta_sentence','deberta_moments','roberta_metric','minilm_metric','chat_metric','select','evaluate','downstream','runtime','report'])
    name=parser.parse_args().stage
    {'fetch-models':fetch_models,'register':register,'prepare':prepare,'incumbent':incumbent,'select':select,'evaluate':evaluate,'downstream':downstream,'runtime':runtime,'report':report}.get(name,lambda:train(name))()
