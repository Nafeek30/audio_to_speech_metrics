"""A fixed-decoder reassessment; private cohort is evaluation-only.

One evidence file records the plan, validation candidates and frozen selection.
Candidate weights and augmented training examples remain in temporary storage.
"""
import argparse
from collections import deque
import copy
from datetime import datetime,timezone
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn

from speech_pipeline import ROOT
from speech_pipeline.corpora import EVIDENCE,OUT
from speech_pipeline.pipeline import digest,save
from speech_pipeline.boundary_model import paragraph_rules
from speech_pipeline.segmentation import (BoundaryCRF,PretrainedGapModel,collate,load_bundle,
    metrics,pretrained_predictions,save_bundle,word_windows)
from speech_pipeline.research.segmentation import load_partitions

WORK=Path('/private/tmp/speech_reassessment')
EVIDENCE_FILE=EVIDENCE/'reassessment.json'


def read():return json.loads(EVIDENCE_FILE.read_text())


def fetch_models():
    """Fetch exact pretrained/reference revisions; normal inference is offline."""
    from huggingface_hub import snapshot_download
    specs=[('bert','google/bert_uncased_L-4_H-256_A-4','387825ce42dbb39b87911cdf8e383ee3b25184f8'),
           ('minilm','microsoft/MiniLM-L12-H384-uncased','44acabbec0ef496f6dbc93adadea57f376b7c0ec'),
           ('batchalign','talkbank/CHATUtterance-en','764ec3f762c2e24df2def8df98b5fe34940085c6')]
    for name,repo,revision in specs:
        folder=snapshot_download(repo,revision=revision,cache_dir=WORK/'pretrained',
            allow_patterns=['config.json','vocab.txt','tokenizer.json','tokenizer_config.json','special_tokens_map.json','pytorch_model.bin','model.safetensors'])
        Path('/private/tmp',f'speech_{name}_model_path.txt').write_text(folder)
        print('Pinned',repo,revision,flush=True)


def register():
    if EVIDENCE_FILE.exists():raise ValueError('A registered round already exists; preserve its history')
    _,previous=load_partitions()
    plan=dict(task='Recover CHAT utterance boundaries from target-child transcript paragraphs; preserve all words',
        input_sha256=previous['input_sha256'],partitions=previous['partitions'],seed=42,
        private_cohort_role='Final evaluation only. No training, adaptation, threshold/epoch/rule/model/routing selection on the 10 participants.',
        historical_caveat='The private cohort and public test benchmark were examined in prior rounds; they cannot be represented as pristine independent tests.',
        decoding='Independent class argmax or learned CRF maximum-score sequence; no tuned probability cutoff, length cap or pause threshold',
        primary_selection='Highest full-paragraph TalkBank validation micro boundary F1; exact ties prefer fewer parameters, then lexical candidate name',
        candidates=['incumbent_argmax','bert_context','minilm_context','minilm_sequence'],
        training=dict(full_paragraphs=True,extra_crops_per_paragraph=2,crop_word_lengths=[8,16,32,64,128],
            crop_selection='Seeded word positions independent of gold labels; mask crop final gap; recompute rules from cropped input',
            max_epochs=8,patience=2,batch_size=16,encoder_lr=3e-5,head_lr=2e-4,weight_decay=.01,
            sequence_adaptation='Initialize from training/validation-selected MiniLM; max 3 epochs; encoder 1e-5, head 3e-5, CRF 3e-3',
            averaging='Additionally score equal parameter average of the final 3 trained epochs, if available; validation only'),
        pretrained=dict(bert=dict(id='google/bert_uncased_L-4_H-256_A-4',revision='387825ce42dbb39b87911cdf8e383ee3b25184f8'),
            minilm=dict(id='microsoft/MiniLM-L12-H384-uncased',revision='44acabbec0ef496f6dbc93adadea57f376b7c0ec')),
        final_evaluation='Freeze one candidate and all deployment choices before opening public test/private cohort. Evaluate once in paragraph-per-recording mode; speaker-turn transfer is separately reported, never used for changes.',
        metric_policy='Existing surface eligibility/filler policy and TAALED remain unchanged; annotation-dependent metrics remain missing without evidence',
        experiments={},status='registered')
    save(EVIDENCE_FILE,plan);print('Registered fixed-decoder reassessment before validation experiments',flush=True)


def partitions():
    plan=read()
    if 'selection' in plan:raise ValueError('Selection is frozen: further training is prohibited in this round')
    splits,previous=load_partitions()
    if previous['input_sha256']!=plan['input_sha256']:raise ValueError('Registered data changed')
    for rows in splits.values():
        if any(r['corpus'] not in {'ocsc','nyu_emerson'} for r in rows):raise ValueError('Only public TalkBank data may enter development')
    return splits,plan


def augmented(train):
    """No private/test data or gold boundaries affect crop input geometry."""
    path=WORK/'augmented_training.jsonl'
    if path.exists():
        rows=[json.loads(s) for s in path.read_text().splitlines()]
        if rows[:len(train)]!=train or any(r['participant'] not in {t['participant'] for t in train} for r in rows):
            raise ValueError('Cached augmentation differs from the frozen training partition')
        return rows
    import spacy
    nlp=spacy.load('en_core_web_sm');rng=random.Random(42);rows=list(train)
    for r in train:
        for ci in range(2):
            n=min(len(r['tokens']),rng.choice([8,16,32,64,128]))
            if n<2:continue
            start=rng.randrange(len(r['tokens'])-n+1);end=start+n
            tokens=r['tokens'][start:end];labels=r['labels'][start:end].copy();labels[-1]=-100
            rows.append(dict(corpus=r['corpus'],participant=r['participant'],recording=r['recording']+f'#crop{ci}',
                tokens=tokens,labels=labels,baseline=paragraph_rules(tokens,nlp)))
    WORK.mkdir(parents=True,exist_ok=True)
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    print('Training-only context augmentation:',len(train),'paragraphs ->',len(rows),'views',flush=True)
    return rows


def record(name,entry):
    plan=read()
    if 'selection' in plan:raise ValueError('No new candidates after selection')
    plan['experiments'][name]=entry;save(EVIDENCE_FILE,plan)


def incumbent():
    splits,plan=partitions();torch.set_num_threads(4)
    model,t,meta=load_bundle(OUT/'segmenter')
    pred=pretrained_predictions(model,t,splits['validation'])
    result=metrics(splits['validation'],pred)
    record('incumbent_argmax',dict(status='complete',decoder='argmax',validation=result,
        parameters=sum(p.numel() for p in model.parameters()),artifact=str(OUT/'segmenter'),
        note='Existing checkpoint, evaluated without its old validation-calibrated cutoff'))
    print('Incumbent fixed-argmax validation F1',result['overall']['f1'],flush=True)


def train(name):
    from transformers import AutoTokenizer
    splits,plan=partitions();WORK.mkdir(parents=True,exist_ok=True)
    if name not in {'bert_context','minilm_context','minilm_sequence'}:raise ValueError('Unregistered candidate')
    device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True,warn_only=device.type=='mps')
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    start=time.monotonic();train=augmented(splits['train']);validation=splits['validation']
    sequence=name.endswith('sequence')
    if sequence:
        source=plan['experiments'].get('minilm_context')
        if not source or source['status']!='complete':raise ValueError('Complete the registered MiniLM candidate before sequence adaptation')
        model,tokenizer,_=load_bundle(source['artifact']);model.crf=BoundaryCRF()
    else:
        path=Path('/private/tmp/speech_bert_model_path.txt' if name.startswith('bert') else '/private/tmp/speech_minilm_model_path.txt').read_text().strip()
        tokenizer=AutoTokenizer.from_pretrained(path,use_fast=True,local_files_only=True)
        model=PretrainedGapModel(pretrained_path=path)
    windows=[]
    for ri,r in enumerate(train):windows.extend(dict(w,record=ri) for w in word_windows(r,tokenizer))
    windows=[w for w in windows if any(owned and train[w['record']]['labels'][g]>=0 for g,owned in zip(w['gaps'],w['owned']))]
    model.to(device)
    groups=[dict(params=model.encoder.parameters(),lr=1e-5 if sequence else 3e-5),
        dict(params=[*model.head.parameters(),*model.rule_embedding.parameters()],lr=3e-5 if sequence else 2e-4)]
    if sequence:groups.append(dict(params=model.crf.parameters(),lr=3e-3))
    optimizer=torch.optim.AdamW(groups,weight_decay=.01);epochs=3 if sequence else 8
    steps=epochs*((len(windows)+15)//16);warm=max(1,int(steps*.1))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:min((step+1)/warm,max(0.,(steps-step)/max(1,steps-warm))))
    ce=nn.CrossEntropyLoss(ignore_index=-100);history=[];best=-1.;stale=0;tail=deque(maxlen=3)
    meta=dict(name=name,decoder='crf' if sequence else 'independent',decoding='viterbi' if sequence else 'argmax',
        threshold=None,input_sha256=plan['input_sha256'],protocol_sha256=digest(EVIDENCE_FILE),
        max_length=256,stride=128,seed=42,device=str(device),pretrained=plan['pretrained']['minilm' if name.startswith('minilm') else 'bert'],
        rule_features='Joint learned rule embedding; no rule boundary is forced',
        code_sha256={str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__),ROOT/'speech_pipeline/segmentation.py']})
    folder=WORK/name
    print(name,len(windows),'training windows;',sum(p.numel() for p in model.parameters()),'parameters',flush=True)
    for epoch in range(epochs):
        model.train();random.shuffle(windows);losses=[]
        for at in range(0,len(windows),16):
            batch=windows[at:at+16];inputs,labels=collate(batch,tokenizer.pad_token_id,train)
            inputs={k:v.to(device) for k,v in inputs.items()};labels=labels.to(device)
            optimizer.zero_grad(set_to_none=True);emissions=model(**inputs)
            loss=model.crf.loss(emissions,labels,torch.tensor([len(w['gaps']) for w in batch],device=device)) if sequence else ce(emissions.reshape(-1,2),labels.reshape(-1))
            if not torch.isfinite(loss):raise ValueError('Non-finite training objective')
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();scheduler.step()
            losses.append(float(loss.detach()))
            if (at//16+1)%100==0:print(name,'epoch',epoch+1,'batch',at//16+1,'/',(len(windows)+15)//16,flush=True)
        score=metrics(validation,pretrained_predictions(model,tokenizer,validation))
        entry=dict(epoch=epoch+1,train_loss=float(np.mean(losses)),validation_f1=score['overall']['f1']);history.append(entry)
        tail.append({k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
        print(name,'epoch',epoch+1,'validation F1',entry['validation_f1'],flush=True)
        if entry['validation_f1']>best:
            best=entry['validation_f1'];stale=0
            save_bundle(model,tokenizer,folder,dict(meta,selected_epoch=epoch+1,history=history,parameter_average=False))
        else:stale+=1
        if stale>=2:break
    if len(tail)==3:
        averaged={key:torch.stack([s[key] for s in tail]).mean(0) for key in tail[0]}
        model.load_state_dict(averaged)
        score=metrics(validation,pretrained_predictions(model,tokenizer,validation))
        history.append(dict(parameter_average=True,epochs=list(range(len(history)-2,len(history)+1)),validation_f1=score['overall']['f1']))
        print(name,'parameter-average validation F1',score['overall']['f1'],flush=True)
        if score['overall']['f1']>best:save_bundle(model,tokenizer,folder,dict(meta,selected_epoch=None,history=history,parameter_average=True))
    model,tokenizer,saved=load_bundle(folder)
    result=metrics(validation,pretrained_predictions(model,tokenizer,validation))
    record(name,dict(status='complete',decoder=saved['decoding'],validation=result,history=history,
        selected_epoch=saved['selected_epoch'],parameter_average=saved['parameter_average'],
        artifact=str(folder),parameters=sum(p.numel() for p in model.parameters()),device=str(device),elapsed_seconds=time.monotonic()-start))
    print(name,'completed validation F1',result['overall']['f1'],flush=True)


def select():
    plan=read()
    if 'selection' in plan:raise ValueError('The choice is frozen; do not reselect after viewing final data')
    if any(n not in plan['experiments'] or plan['experiments'][n]['status']!='complete' for n in plan['candidates']):
        raise ValueError('Complete all registered candidates, or explicitly document a failed candidate before selection')
    entries=plan['experiments'];winner=min(entries,key=lambda n:(-entries[n]['validation']['overall']['f1'],entries[n]['parameters'],n))
    plan['selection']=dict(name=winner,validation_f1=entries[winner]['validation']['overall']['f1'],
        decoder=entries[winner]['decoder'],weights_sha256=digest(Path(entries[winner]['artifact'])/'model.safetensors'),
        before_final_evaluation=True,recorded_utc=datetime.now(timezone.utc).isoformat());plan['status']='selected';save(EVIDENCE_FILE,plan)
    print('Frozen selection:',json.dumps(plan['selection']),flush=True)


def batchalign_predictions(records):
    """Pinned six-class checkpoint adapted to the shared word-gap task.

    The checkpoint's first-subtoken label convention is retained. This adapter
    omits the original text-rewriting/Punkt/hanging-word postprocessing, so it
    is a model comparison and not a replication of the full Batchalign suite.
    """
    from transformers import AutoModelForTokenClassification,AutoTokenizer
    torch.set_num_threads(4)
    folder=Path('/private/tmp/speech_batchalign_model_path.txt').read_text().strip()
    tokenizer=AutoTokenizer.from_pretrained(folder,use_fast=True,local_files_only=True)
    device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model=AutoModelForTokenClassification.from_pretrained(folder,local_files_only=True).to(device);model.eval()
    windows=[];totals=[np.zeros((len(r['tokens']),6)) for r in records]
    weights=[np.zeros(len(r['tokens'])) for r in records]
    for ri,r in enumerate(records):
        encoded=tokenizer(r['tokens'],is_split_into_words=True,add_special_tokens=False,return_token_type_ids=False)
        ids=encoded['input_ids'];mapping=encoded.word_ids()
        first=np.full(len(r['tokens']),-1,dtype=int);last=first.copy()
        for i,w in enumerate(mapping):
            if first[w]<0:first[w]=i
            last[w]=i
        for start in range(0,len(ids),256):
            end=min(start+510,len(ids));valid=np.where((first>=start)&(last<end))[0]
            if len(valid):windows.append(dict(record=ri,words=valid,positions=first[valid]-start+1,
                weights=np.minimum(first[valid]-start+1,end-last[valid]).clip(1,128),
                ids=[tokenizer.cls_token_id,*ids[start:end],tokenizer.sep_token_id]))
            if end==len(ids):break
    with torch.inference_mode():
        for at in range(0,len(windows),8):
            batch=windows[at:at+8];width=max(len(w['ids']) for w in batch)
            ids=torch.full((len(batch),width),tokenizer.pad_token_id,dtype=torch.long);mask=torch.zeros_like(ids)
            for i,w in enumerate(batch):ids[i,:len(w['ids'])]=torch.tensor(w['ids']);mask[i,:len(w['ids'])]=1
            logits=model(input_ids=ids.to(device),attention_mask=mask.to(device)).logits.cpu().numpy()
            for i,w in enumerate(batch):
                totals[w['record']][w['words']]+=logits[i,w['positions']]*w['weights'][:,None]
                weights[w['record']][w['words']]+=w['weights']
            if at%80==0:print('Frozen CHATUtterance reference windows',min(at+8,len(windows)),'/',len(windows),flush=True)
    output=[]
    for total,weight in zip(totals,weights):
        if np.any(weight<=0):raise ValueError('Reference tokenizer did not cover every source word')
        labels=(total/weight[:,None]).argmax(axis=1);pred=np.isin(labels,[2,3,4]);pred[-1]=False;output.append(pred)
    return output


def evaluate():
    """Freeze/export first, then perform final evaluation with no reselection."""
    from speech_pipeline import utterances as auto,taaled
    from speech_pipeline.segmentation import Segmenter
    plan=read()
    if 'selection' not in plan:raise ValueError('Freeze the validation-only selection before opening final data')
    prior=json.loads((OUT/'improved_model.json').read_text())
    if 'reassessment' in prior:
        print('Final inference is already complete; returning saved results without rerunning or changing the model',flush=True)
        return prior['reassessment']
    choice=plan['selection'];entry=plan['experiments'][choice['name']];source=Path(entry['artifact'])
    if digest(source/'model.safetensors')!=choice['weights_sha256']:raise ValueError('Frozen selected weights changed')
    model,tokenizer,meta=load_bundle(source)
    destination=OUT/'context_segmenter';destination.mkdir(exist_ok=True)
    meta.update(name=choice['name'],decoding='viterbi' if model.crf is not None else 'argmax',threshold=None,
        input_sha256=plan['input_sha256'],selection_scope='TalkBank training/validation only; private cohort evaluation only')
    save_bundle(model,tokenizer,destination,meta)
    plan['final_evaluation_started_utc']=datetime.now(timezone.utc).isoformat()
    plan['deployment']=dict(paragraph='Public-validation-selected fixed-decoder model',
        workbook_default='Established rules, unchanged regardless of private results',private_tuning='prohibited')
    save(EVIDENCE_FILE,plan)
    # The first access to final records in this round occurs only below this line.
    splits,_=load_partitions(include_test=True);test=splits['test']
    target_path=EVIDENCE/'boundary_dataset.jsonl';target=[json.loads(s) for s in target_path.read_text().splitlines()]
    source_meta=json.loads((EVIDENCE/'boundary_dataset_sources.json').read_text())
    if digest(target_path)!=source_meta['dataset_sha256']:raise ValueError('Final paragraph dataset changed')
    for name,sha in source_meta['sources'].items():
        if digest(ROOT/name)!=sha:raise ValueError('Final input or annotation changed: '+name)
    segmenter=Segmenter(destination);mapping=taaled.load_target_speakers()
    model,tokenizer=segmenter.model,segmenter.tokenizer
    cache=WORK/'final_primary_predictions.json'
    if cache.exists():
        results=json.loads(cache.read_text())
        if results['selected']!=choice or results['private_data_sha256']!=digest(target_path):raise ValueError('Cached final inference differs from the frozen selection or data')
    else:
        paragraphs=[];texts=[]
        for r in target:
            raw=ROOT/'dataset/uncorrected'/r['participant']/r['recording']
            units=segmenter.target_paragraph(list(auto.read_rows(raw)),mapping[r['participant'],r['recording']])
            words,boundaries=auto.stream(units)
            if words!=r['tokens']:raise ValueError('Paragraph pipeline changed frozen source words')
            pred=np.array([i+1 in boundaries for i in range(len(words))]);pred[-1]=False
            paragraphs.append(pred);texts.append([u['text'] for u in units])
        results=dict(selected=choice,validation=entry['validation'],input_sha256=plan['input_sha256'],
            public_test=metrics(test,pretrained_predictions(model,tokenizer,test)),
            private_paragraphs=metrics(target,paragraphs),
            private_data_sha256=digest(target_path),evaluation_status=plan['historical_caveat'])
        for row,parts in zip(results['private_paragraphs']['recordings'],texts):row['utterance_texts']=parts
        # Save this inference before the slower independent reference model runs.
        save(cache,results)
    print('Final primary predictions saved; no further training or selection is allowed',flush=True)
    context=[];context_text=[]
    for r in target:
        path=ROOT/'dataset/uncorrected'/r['participant']/r['recording']
        units=segmenter.target_rows(list(auto.read_rows(path)),mapping[r['participant'],r['recording']])
        tokens,boundaries=auto.stream(units)
        if tokens!=r['tokens']:raise ValueError('Turn-aware model changed the frozen source words')
        context.append(np.array([i+1 in boundaries for i in range(len(tokens))]));context[-1][-1]=False
        context_text.append([u['text'] for u in units])
    results['private_speaker_turns']=metrics(target,context)
    for row,texts in zip(results['private_speaker_turns']['recordings'],context_text):row['utterance_texts']=texts
    results['reference_checkpoint']=dict(specification=plan['reference_baseline'],
        public_test=metrics(test,batchalign_predictions(test)),
        private_paragraphs=metrics(target,batchalign_predictions(target)))
    prior['reassessment']=results;save(OUT/'improved_model.json',prior)
    plan=read();plan['status']='evaluated';plan['final_results_file']='outputs/results/benchmark/improved_model.json';save(EVIDENCE_FILE,plan)
    print('Final TalkBank F1',results['public_test']['overall']['f1'],flush=True)
    print('Final 10-participant paragraph F1',results['private_paragraphs']['overall']['f1'],flush=True)
    print('Secondary speaker-turn F1',results['private_speaker_turns']['overall']['f1'],flush=True)
    return results


def downstream():
    """Score the frozen new segments; never feed these results back to training."""
    import csv,statistics,spacy
    from openpyxl import load_workbook
    from tempfile import TemporaryDirectory
    from speech_pipeline import utterances as auto
    from speech_pipeline.metrics import clinical_from_predictions
    from speech_pipeline.pipeline import lexical,clan_software_check
    from speech_pipeline.research.segmentation import bootstrap_comparison
    prior=json.loads((OUT/'improved_model.json').read_text())
    if 'reassessment' not in prior:raise ValueError('Complete frozen final inference first')
    result=prior['reassessment'];splits,_=load_partitions(include_test=True)
    source_meta=json.loads((EVIDENCE/'boundary_dataset_sources.json').read_text())
    for name,sha in source_meta['sources'].items():
        if digest(ROOT/name)!=sha:raise ValueError('Final input or human reference changed: '+name)
    target=[json.loads(s) for s in (EVIDENCE/'boundary_dataset.jsonl').read_text().splitlines()]
    records=splits['test']+target;index={(r['corpus'],r['participant'],r['recording']):r for r in records}
    workbook=load_workbook(ROOT/'dataset/annotations/all_nafee_utterances.xlsx',read_only=True,data_only=True)
    human={(s.title,name):[dict(text=auto.plain(t)) for t in texts] for s in workbook for name,texts in auto.flatten_labels(s).items()};workbook.close()
    nlp=spacy.load('en_core_web_sm');table=[];checks=[];coverage=[];selected_children={c:set() for c in ('ocsc','nyu_emerson')}
    def units(r,cuts):
        boundaries=[0,*cuts,len(r['surface_words'])]
        return [dict(text=' '.join(r['surface_words'][a:b]),utterance_id=i+1,method='reassessed_'+result['selected']['name']) for i,(a,b) in enumerate(zip(boundaries,boundaries[1:])) if a<b]
    def scores(utterances,temp):
        value={f'surface/{k}':v for k,v in clinical_from_predictions(utterances)['values'].items()}
        value.update({f'utterances/{k}':v for k,v in auto.word_mlu(utterances).items() if k in ('total_utterances','mlu_nutterances','mlu_words_sd')})
        if not any(auto.analysis_text(u['text']) for u in utterances):return value
        try:comparison=lexical(utterances,nlp,temp/'lexical.txt')
        except (ValueError,ZeroDivisionError) as error:
            coverage.append(dict(corpus=r['corpus'],participant=r['participant'],recording=r['recording'],detail=str(error)))
            return value
        value.update({f'custom_normalized/{k}':v['custom'] for k,v in comparison.items()})
        value.update({f'taaled/{k}':v['reference'] for k,v in comparison.items()})
        return value
    predictions=result['public_test']['recordings']+result['private_paragraphs']['recordings']
    old_table=OUT/'improved_metric_errors.csv'
    with old_table.open(encoding='utf-8-sig',newline='') as f:old_rows=list(csv.DictReader(f))
    # The earlier comparison already preserves the exact gold metric values.
    reference={}
    for row in old_rows:
        if row['method']=='improved':
            key=row['corpus'],row['participant'],row['recording']
            reference.setdefault(key,{})[row['metric']]=float(row['reference']) if row['reference'] else None
    with TemporaryDirectory(prefix='reassessed_metrics_') as temp:
        temp=Path(temp)
        for at,p in enumerate(predictions):
            key=p['corpus'],p['participant'],p['recording'];r=index[key];predicted=units(r,p['predicted_boundaries'])
            truth=reference[key];mine=scores(predicted,temp)
            for metric,gold in truth.items():
                value=mine.get(metric)
                table.append(dict(method='reassessed',corpus=r['corpus'],participant=r['participant'],recording=r['recording'],gem=r.get('gem'),
                    metric=metric,reference=gold,predicted=value,absolute_error=abs(value-gold) if value is not None and gold is not None else None,
                    absolute_percent_error=100*abs(value-gold)/abs(gold) if gold and value is not None else None,
                    status='available' if value is not None and gold is not None else 'unavailable_required_annotation_or_sample_size'))
            corpus=r['corpus'];verify=corpus=='existing_development_cohort' or corpus in selected_children and len(selected_children[corpus])<3 and r['participant'] not in selected_children[corpus]
            if verify:
                if corpus in selected_children:selected_children[corpus].add(r['participant'])
                checks.append(dict(corpus=corpus,participant=r['participant'],recording=r['recording'],clan=clan_software_check(predicted,temp)))
            if at%20==0:print('Frozen-model downstream metrics',at+1,'/',len(predictions),flush=True)
        context_mlu=[]
        for row in result['private_speaker_turns']['recordings']:
            predicted=[dict(text=t) for t in row['utterance_texts']]
            context_mlu.append(dict(participant=row['participant'],recording=row['recording'],
                predicted=auto.word_mlu(predicted),reference=auto.word_mlu(human[row['participant'],row['recording']])))
    keep=[r for r in old_rows if r['method']!='reassessed']
    with old_table.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(keep+table)
    audit_path=EVIDENCE/'improved_calculator_checks.json';audit=json.loads(audit_path.read_text());audit['reassessment']=checks;audit['reassessment_coverage']=coverage;save(audit_path,audit)
    result['downstream']={}
    for corpus in ('ocsc','nyu_emerson','existing_development_cohort','all_public'):
        result['downstream'][corpus]={}
        for metric in ('surface/MLUw','surface/TNW','surface/NDW','surface/TTR','surface/MATTR','utterances/total_utterances','utterances/mlu_words_sd'):
            chosen=[r for r in table if r['metric']==metric and (r['corpus'] in {'ocsc','nyu_emerson'} if corpus=='all_public' else r['corpus']==corpus)]
            child={}
            for row in chosen:
                if row['absolute_percent_error'] is not None:child.setdefault(row['participant'],[]).append(row['absolute_percent_error'])
            result['downstream'][corpus][metric]=dict(available=sum(len(v) for v in child.values()),
                median_child_mean_ape=statistics.median(statistics.mean(v) for v in child.values()) if child else None)
    def interval(rows,new,old):
        value=bootstrap_comparison(rows,new,[dict(p,transformer=p['counts']) for p in old])
        value['difference_ci']=value.pop('difference_vs_previous_joint_ci');return value
    result['uncertainty']=dict(public_vs_previous=interval(splits['test'],result['public_test']['recordings'],prior['test']['recordings']),
        private_vs_previous=interval(target,result['private_paragraphs']['recordings'],prior['original_asr']['recordings']),
        private_turns_vs_published_rules=interval(target,result['private_speaker_turns']['recordings'],
            [dict(p,counts=p['previous_counts']) for p in prior['runtime_context']['recordings']]),
        public_vs_reference=interval(splits['test'],result['public_test']['recordings'],result['reference_checkpoint']['public_test']['recordings']))
    result['private_context_mlu']=context_mlu
    result['publication_comparison']=publication_comparison(context_mlu)
    result['calculator_checks']=dict(records=len(checks),computed_records=sum(c['clan']['status']=='software_reference_computed' for c in checks),all_within_display_precision=all(
        v['absolute_error'] is None or v['absolute_error']<=(0 if m in ('TNW','NDW') else .000501)
        for c in checks for m,v in c['clan']['requested_metrics'].items()))
    result['calculator_checks']['length_sd_within_display_precision']=all(
        c['clan']['metrics'].get('mlu_words_sd',{}).get('absolute_error') is not None and
        c['clan']['metrics']['mlu_words_sd']['absolute_error']<=.000501 for c in checks if c['clan']['status']=='software_reference_computed')
    if not result['calculator_checks']['all_within_display_precision']:raise ValueError('Official calculator mismatch; retain evidence and investigate without model tuning')
    if not result['calculator_checks']['length_sd_within_display_precision']:raise ValueError('CLAN length-SD mismatch; preserve the model and investigate calculator evidence')
    prior['reassessment']=result;save(OUT/'improved_model.json',prior)
    print('Final metric comparisons saved; no weights, rules or decisions changed',flush=True)
    return result


def publication_comparison(rows):
    """Use each publication's aggregation, retaining zero-reference missingness."""
    import math,statistics
    people={}
    for row in rows:people.setdefault(row['participant'],[]).append(row)
    complete={k:[] for k in ('total_utterances','MLUw','SD')};recording={k:[] for k in complete}
    for child in people.values():
        pooled={}
        for side in ('predicted','reference'):
            n=sum(r[side]['mlu_nutterances'] for r in child);words=sum(r[side]['mlu_nwords'] for r in child)
            mean=words/n if n else None
            variance=sum(r[side]['mlu_nutterances']*((r[side]['mlu_words_sd'] or 0)**2+(r[side]['mlu_words'] or 0)**2) for r in child)/n-mean**2 if n else None
            pooled[side]=dict(total_utterances=sum(r[side]['total_utterances'] for r in child),MLUw=mean,SD=math.sqrt(max(0,variance)) if variance is not None else None)
        for metric,key in [('total_utterances','total_utterances'),('MLUw','mlu_words'),('SD','mlu_words_sd')]:
            a,b=pooled['predicted'][metric],pooled['reference'][metric]
            if a is not None and b:complete[metric].append(100*abs(a-b)/abs(b))
            errors=[100*abs(r['predicted'][key]-r['reference'][key])/abs(r['reference'][key]) for r in child if r['predicted'][key] is not None and r['reference'][key]]
            if errors:recording[metric].append(statistics.mean(errors))
    return dict(paper_mdape={k:statistics.median(v) if v else None for k,v in complete.items()},
        poster_mape={k:statistics.mean(v) if v else None for k,v in recording.items()})


def report_section():
    """Produce current results from saved evidence, without test inference."""
    if not EVIDENCE_FILE.exists():return []
    plan=read();path=OUT/'improved_model.json';prior=json.loads(path.read_text())
    result=prior.get('reassessment')
    lines=['## Fresh reassessment: fixed class or sequence decisions','',
        'This round re-examines the pipeline as word-preserving supervised boundary recovery. It compares stronger pretrained context, training on different context lengths, and a learned sequence decoder. Independent decisions use class argmax; CRF decisions use maximum-score Viterbi decoding. There is no tuned probability cutoff, hard utterance-length limit, or pause threshold. Both words and the existing rules contribute as model inputs.',
        '', 'The frozen child partitions remain 233 training / 49 validation / 49 public-test children across OCSC and NYU–Emerson. Training includes two additional seeded word-position crops per training paragraph; crop locations and lengths do not depend on gold cuts. Every candidate, epoch and parameter average is selected using public validation. The original 10-participant cohort is final evaluation only for this round. Its historical use prevents a claim that it is a pristine independent test.',
        '', '| Validation candidate | Fixed decoder | Boundary F1 |','| --- | --- | ---: |']
    for name,e in plan['experiments'].items():lines.append(f'| {name} | {e["decoder"]} | {100*e["validation"]["overall"]["f1"]:.2f}% |')
    if not result:return lines+['', 'Training/final evaluation is incomplete; no new final-test score is claimed.','']
    choice=result['selected'];lines += ['',f'The validation-only choice was **{choice["name"]}** with {choice["decoder"]} decoding. This choice and deployment behavior were frozen before this round opened the final records. Candidate training and source fingerprints are in `outputs/validation/benchmark/reassessment.json`.',
        '', '| Matched paragraph evaluation | Previous selected model | Reassessed model | Frozen CHATUtterance checkpoint adapter |','| --- | ---: | ---: | ---: |']
    reference=result['reference_checkpoint']
    for label,new,old,base in [('TalkBank public test',result['public_test']['overall'],prior['test']['overall'],reference['public_test']['overall']),
        ('Original 10 participants',result['private_paragraphs']['overall'],prior['original_asr']['overall'],reference['private_paragraphs']['overall'])]:
        lines.append(f'| {label} | {100*old["f1"]:.2f}% | {100*new["f1"]:.2f}% | {100*base["f1"]:.2f}% |')
    lines += ['', '| Public test corpus | Children | Boundary precision | Recall | F1 | Exact utterance-span F1 |','| --- | ---: | ---: | ---: | ---: | ---: |']
    for corpus,e in result['public_test']['by_corpus'].items():
        lines.append(f'| {corpus} | {e["participants"]} | {100*e["precision"]:.2f}% | {100*e["recall"]:.2f}% | {100*e["f1"]:.2f}% | {100*e["exact"]["f1"]:.2f}% |')
    p=result['private_paragraphs']['overall'];context=result['private_speaker_turns']['overall']
    lines += ['',f'The private paragraph evaluation uses 14 human-labeled recordings across all 10 participants, one target-speaker paragraph per recording. It has precision {100*p["precision"]:.2f}%, recall {100*p["recall"]:.2f}%, {p["counts"]["fp"]} extra boundaries and {p["counts"]["fn"]} missed boundaries. This does not create human references for all 37 recordings.',
        '', f'The separately declared speaker-turn transfer test scores **{100*context["f1"]:.2f}% F1**, compared with the published rules\' **88.64%** on that same context. This secondary result is not used to select or revise any model, rule, cutoff or deployment mode. Workbook defaults remain the established rules; plain paragraphs use the public-validation-selected model.',
        '', '![Matched paragraph evaluation](reassessment_results.png)',
        '', '| Private participant | Labeled recordings | Paragraph boundary F1 |','| --- | ---: | ---: |']
    for child,e in result['private_paragraphs']['by_child'].items():lines.append(f'| {child} | {e["records"]} | {100*e["f1"]:.2f}% |')
    if 'publication_comparison' in result:
        c=result['publication_comparison'];lines += ['', 'For the speaker-turn comparison with the publications, errors are computed with each artifact\'s original aggregation. The paper combines each participant\'s sampled speech and takes median APE; the poster averages within-participant recording APE and then averages participants. Lower is better.',
            '', '| Measure | Paper: published → reassessed | Poster: published → reassessed |','| --- | ---: | ---: |']
        for label,key,paper,poster in [('Utterance count','total_utterances',6.311047889995258,7.959851305756537),('MLUw','MLUw',6.398058252427191,9.117114336859553),('Length SD','SD',16.764646909791562,25.09295342257285)]:
            a,b=c['paper_mdape'][key],c['poster_mape'][key]
            lines.append(f'| {label} | {paper:.2f}% → '+(f'{a:.2f}%' if a is not None else 'unavailable')+f' | {poster:.2f}% → '+(f'{b:.2f}%' if b is not None else 'unavailable')+' |')
    if 'downstream' in result:
        import csv,statistics
        with (OUT/'improved_metric_errors.csv').open(encoding='utf-8-sig',newline='') as f:metric_rows=list(csv.DictReader(f))
        lines += ['', 'Matched paragraph downstream errors use the median of each child\'s mean recording APE. This aggregation and continuous-paragraph context differ from the publication comparison above. Lower is better.',
            '', '| Paragraph measure | Public: previous → reassessed | Private: previous → reassessed |','| --- | ---: | ---: |']
        for label,metric in [('Utterance count','utterances/total_utterances'),('MLUw','surface/MLUw'),('Length SD','utterances/mlu_words_sd')]:
            cells=[]
            for private in (False,True):
                values=[]
                for method in ('improved','reassessed'):
                    children={}
                    for r in metric_rows:
                        if r['method']==method and r['metric']==metric and (r['corpus']=='existing_development_cohort' if private else r['corpus'] in {'ocsc','nyu_emerson'}) and r['absolute_percent_error']:
                            children.setdefault(r['participant'],[]).append(float(r['absolute_percent_error']))
                    values.append(statistics.median(statistics.mean(v) for v in children.values()) if children else None)
                cells.append(' → '.join(f'{v:.2f}%' if v is not None else 'unavailable' for v in values))
            lines.append(f'| {label} | '+' | '.join(cells)+' |')
        lines += ['', '| Public-test metric | Reassessed median child mean APE |','| --- | ---: |']
        for metric,e in result['downstream']['all_public'].items():
            value=e['median_child_mean_ape'];lines.append(f'| {metric} | '+(f'{value:.2f}%' if value is not None else 'unavailable')+' |')
        check=result['calculator_checks'];lines += ['', f'Actual official CLAN MLU/FREQ checks were performed on {check["records"]} frozen new predictions (all 14 private paragraphs plus six distinct-child public paragraphs). Count agreement and three-decimal ratio precision are verified. The same segments feed all 17 existing TAALED comparisons; individual metric errors are appended as `method=reassessed` in `improved_metric_errors.csv`. MLUm, WPM, PGU, omission/substitution and maze measures remain conditional on the required reviewed evidence.']
    if 'uncertainty' in result:
        lines += ['', 'Paired participant-cluster bootstrap intervals (10,000 resamples; descriptive because of historical benchmark use):','',
            '| Comparison | New F1 interval | F1 difference interval |','| --- | ---: | ---: |']
        for label,key in [('Public vs previous selected','public_vs_previous'),('Private paragraphs vs previous selected','private_vs_previous'),('Private speaker turns vs published rules','private_turns_vs_published_rules'),('Public vs frozen reference adapter','public_vs_reference')]:
            u=result['uncertainty'][key];a,b=u['boundary_f1_ci'];c,d=u['difference_ci']
            lines.append(f'| {label} | {100*a:.2f}–{100*b:.2f}% | {100*c:.2f}–{100*d:.2f} percentage points |')
    if 'runtime_verification' in result:
        v=result['runtime_verification'];lines += ['', f'The complete paragraph-mode pipeline was also run on all 37 recordings in each transcript condition ({v["checked_recordings"]} recording-condition outputs). All preserve source words; every recording has 17 custom/TAALED comparisons and official CLAN checks within displayed precision, including length SD. All 14 human-labeled ASR outputs exactly reproduce the frozen final-evaluation cuts. Current canonical outputs are in `outputs/results/pipeline/`; this explicit paragraph run does not change the speaker-turn application default.',
            '', 'The TAALED pipeline comparisons retain differences caused by the two preprocessing implementations; they are not claimed to agree exactly. Across the 74 recordings, the largest normalized all-word token/type-count difference is nine tokens/types and the largest MATTR-50 absolute difference is 0.008254. The earlier same-normalized-token calculator checks are a separate analysis. Lexical agreement does not measure boundary accuracy.']
    lines += ['', '### Comparisons with other research','',
        'The [expanded primary-source review](../../../speech_pipeline/research/SEGMENTATION_RESEARCH.md#reassessment-from-first-principles) includes Batchalign (2023), joint conversational segmentation (2017), clinical-narrative segmentation (2017), IDEAS (2024), school-age DLD transcription/LSA (2021), SATE (2026), and Swiss child-language NLP (2026), with verified results and evaluation differences.',
        '', 'Batchalign\'s paper reports 86.9% segmentation F1 for adults with MCI, on a different corpus. Its pinned public model card says MICASE training. The local reference above is the six-class checkpoint adapted to common word gaps with fixed overlapping windows; the original punctuation rewriting, Punkt and hanging-word filter are omitted to preserve our words. It is **not** a replication of the full published system. Our models are adapted to the public child corpora, while this checkpoint is not. These comparisons do not isolate architecture effects or establish overall superiority over Batchalign or other papers.',
        '', 'All source words are retained. No private result changes the frozen model. Both benchmarks have historical inspection; external pretraining provenance is documented but not independently exhaustively audited. These are bounded single-seed experiments, not proof of global optimality, clinical interchangeability, acoustic segmentation accuracy, or time savings. Final paper, poster and abstract remain unchanged.', '']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4.5),dpi=180);x=np.arange(2)
    for i,(name,values,color) in enumerate([
        ('Previous selected',[prior['test']['overall']['f1'],prior['original_asr']['overall']['f1']],'#487da3'),
        ('Reassessed',[result['public_test']['overall']['f1'],p['f1']],'#237a59'),
        ('CHATUtterance adapter',[reference['public_test']['overall']['f1'],reference['private_paragraphs']['overall']['f1']],'#b37a36')]):
        ax.bar(x+(i-1)*.24,np.array(values)*100,.24,label=name,color=color)
    ax.set(xticks=x,xticklabels=['TalkBank test: 49 children','Private final: 10 participants'],ylabel='Word-gap boundary F1 (%)',ylim=(0,100),title='Frozen models on the matched paragraph task')
    ax.legend(loc='lower left');fig.tight_layout();fig.savefig(OUT/'reassessment_results.png');plt.close(fig)
    return lines


def report():
    from speech_pipeline.research.validation import write_report
    write_report(json.loads((OUT/'existing_cohort.json').read_text()))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['fetch-models','register','incumbent','bert_context','minilm_context','minilm_sequence','select','evaluate','downstream','report'])
    stage=parser.parse_args().stage
    {'fetch-models':fetch_models,'register':register,'incumbent':incumbent,'select':select,'evaluate':evaluate,'downstream':downstream,'report':report}.get(stage,lambda:train(stage))()
