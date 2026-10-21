"""Pretrained word-gap segmentation; labels never define input windows.

One model combines bidirectional pretrained word context and paragraph-rule
features. Word and character coordinates are retained for common metric export.
Research orchestration, model selection and benchmark reporting live separately
in research.segmentation so inference has no access to evaluation labels.
"""
import copy
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn

from speech_pipeline import utterances as auto
from speech_pipeline.boundary_model import counts, f1, total_counts, paragraph_rules


def sentence_clause_features(tokens,nlp):
    """Two independent automatic cues from unpunctuated words; no gold rows."""
    clause=paragraph_rules(tokens,nlp);text=' '.join(tokens)
    matches=list(auto.WORD.finditer(text));starts={m.start():i for i,m in enumerate(matches)}
    sentence={starts[s.start_char] for s in nlp(text).sents if s.start_char in starts}
    return [int(v)+2*int(i+1 in sentence) for i,v in enumerate(clause)]


def choose_threshold(records, probabilities):
    """Validation-only threshold search; deterministic tie toward 0.5."""
    labels=np.concatenate([np.asarray(r['labels']) for r in records])
    probs=np.concatenate(probabilities);valid=labels>=0
    labels=labels[valid];probs=probs[valid]
    candidates=[]
    for threshold in np.linspace(.05,.95,91):
        c=counts(labels,probs>=threshold)
        candidates.append((f1(c),-abs(threshold-.5),float(threshold),c))
    value,_,threshold,c=max(candidates)
    return dict(f1=value,threshold=threshold,counts=c)


def word_windows(record, tokenizer, max_length=256, stride=128):
    """Map full left/right words to overlapping subword context windows.

    Window geometry uses words only. Each gap owns one training loss at its
    most central view; inference averages all views with center weights.
    """
    words=record['tokens']
    if len(words)<2:return []
    encoded=tokenizer(words,is_split_into_words=True,add_special_tokens=False,
                      return_attention_mask=False,return_token_type_ids=False)
    ids=encoded['input_ids'];mapping=encoded.word_ids()
    first=np.full(len(words),-1,dtype=int);last=first.copy()
    for i,word in enumerate(mapping):
        if word is None:raise ValueError('Unexpected special token in word encoding')
        if first[word]<0:first[word]=i
        last[word]=i
    if np.any(first<0):raise ValueError('Tokenizer omitted a source word')
    width=max_length-2
    if not 0<stride<width:raise ValueError('Stride must be positive and smaller than context')
    windows=[];best=np.full(len(words)-1,-1,dtype=int);quality=np.zeros(len(words)-1)
    for start in range(0,len(ids),stride):
        end=min(start+width,len(ids))
        gaps=np.where((first[:-1]>=start)&(last[1:]<end))[0]
        if len(gaps):
            weights=np.minimum(last[gaps]-start+1,end-first[gaps+1]).clip(1,64).astype(float)
            window=dict(ids=[tokenizer.cls_token_id,*ids[start:end],tokenizer.sep_token_id],
                gaps=gaps.tolist(),left=(last[gaps]-start+1).tolist(),
                right=(first[gaps+1]-start+1).tolist(),weights=weights.tolist(),
                rules=[int(record.get('baseline',[0]*len(words))[i]) for i in gaps])
            wi=len(windows);windows.append(window)
            for gap,weight in zip(gaps,weights):
                if weight>quality[gap]:best[gap]=wi;quality[gap]=weight
        if end==len(ids):break
    if np.any(best<0):raise ValueError('A word gap lacks complete overlapping context')
    for wi,window in enumerate(windows):window['owned']=[best[g]==wi for g in window['gaps']]
    return windows


def collate(windows, pad_id, records=None):
    length=max(len(w['ids']) for w in windows);gaps=max(len(w['gaps']) for w in windows)
    ids=torch.full((len(windows),length),pad_id,dtype=torch.long)
    attention=torch.zeros_like(ids)
    left=torch.zeros((len(windows),gaps),dtype=torch.long);right=left.clone();rules=left.clone()
    labels=torch.full_like(left,-100)
    for i,w in enumerate(windows):
        n=len(w['ids']);m=len(w['gaps']);ids[i,:n]=torch.tensor(w['ids']);attention[i,:n]=1
        left[i,:m]=torch.tensor(w['left']);right[i,:m]=torch.tensor(w['right']);rules[i,:m]=torch.tensor(w['rules'])
        if records is not None:
            labels[i,:m]=torch.tensor([records[w['record']]['labels'][g] if owned else -100
                                      for g,owned in zip(w['gaps'],w['owned'])])
    return dict(input_ids=ids,attention_mask=attention,left=left,right=right,rules=rules),labels


class BoundaryCRF(nn.Module):
    """Two-state sequence likelihood, including partially observed labels.

    Transitions are learned from training labels. Viterbi inference uses no
    hand-set probability cutoff or minimum/maximum utterance length.
    """
    def __init__(self):
        super().__init__()
        self.transitions=nn.Parameter(torch.zeros(2,2))
        self.start=nn.Parameter(torch.zeros(2));self.end=nn.Parameter(torch.zeros(2))

    def partition(self,emissions,lengths,labels=None):
        states=torch.arange(2,device=emissions.device)
        def allowed(t):
            if labels is None:return torch.zeros_like(emissions[:,t])
            valid=(labels[:,t,None]<0)|(labels[:,t,None]==states)
            return torch.where(valid,0.,-1e4)
        score=self.start+emissions[:,0]+allowed(0)
        for t in range(1,emissions.shape[1]):
            update=torch.logsumexp(score[:,:,None]+self.transitions[None],dim=1)+emissions[:,t]+allowed(t)
            score=torch.where((t<lengths)[:,None],update,score)
        return torch.logsumexp(score+self.end,dim=1)

    def loss(self,emissions,labels,lengths):
        return (self.partition(emissions,lengths)-self.partition(emissions,lengths,labels)).sum()/(labels>=0).sum().clamp_min(1)

    def decode(self,emissions):
        if not len(emissions):return np.array([],dtype=bool)
        scores=self.start.detach().cpu().numpy()+emissions[0]
        transition=self.transitions.detach().cpu().numpy();back=[]
        for e in emissions[1:]:
            candidates=scores[:,None]+transition
            back.append(candidates.argmax(axis=0));scores=candidates.max(axis=0)+e
        state=int((scores+self.end.detach().cpu().numpy()).argmax());path=[state]
        for previous in reversed(back):state=int(previous[state]);path.append(state)
        return np.array(path[::-1],dtype=bool)


class PretrainedGapModel(nn.Module):
    def __init__(self, config=None, pretrained_path=None,decoder='independent',rule_states=2):
        super().__init__()
        from transformers import AutoModel
        self.encoder=(AutoModel.from_pretrained(pretrained_path,local_files_only=True)
                      if pretrained_path else AutoModel.from_config(config))
        width=self.encoder.config.hidden_size
        self.rule_embedding=nn.Embedding(rule_states,16)
        self.head=nn.Sequential(nn.Linear(width*2+16,128),nn.GELU(),nn.Dropout(.15),nn.Linear(128,2))
        self.crf=BoundaryCRF() if decoder=='crf' else None

    def forward(self,input_ids,attention_mask,left,right,rules):
        hidden=self.encoder(input_ids=input_ids,attention_mask=attention_mask).last_hidden_state
        width=hidden.shape[-1]
        a=hidden.gather(1,left.unsqueeze(-1).expand(-1,-1,width))
        b=hidden.gather(1,right.unsqueeze(-1).expand(-1,-1,width))
        return self.head(torch.cat([a,b,self.rule_embedding(rules)],dim=-1))


def pretrained_logits(model, tokenizer, records, max_length=256, stride=128, batch_size=16):
    device=next(model.parameters()).device
    windows=[]
    for ri,r in enumerate(records):
        windows.extend(dict(w,record=ri) for w in word_windows(r,tokenizer,max_length,stride))
    sums=[np.zeros(len(r['tokens'])) for r in records];weights=[np.zeros(len(r['tokens'])) for r in records]
    model.eval()
    with torch.inference_mode():
        for at in range(0,len(windows),batch_size):
            group=windows[at:at+batch_size];inputs,_=collate(group,tokenizer.pad_token_id)
            inputs={k:v.to(device) for k,v in inputs.items()}
            logits=model(**inputs).cpu().numpy()
            for i,w in enumerate(group):
                gap=np.asarray(w['gaps']);weight=np.asarray(w['weights']);ri=w['record']
                sums[ri][gap]+=(logits[i,:len(gap),1]-logits[i,:len(gap),0])*weight
                weights[ri][gap]+=weight
    output=[]
    for r,total,weight in zip(records,sums,weights):
        if len(r['tokens'])>1 and np.any(weight[:-1]<=0):raise ValueError('Incomplete inference coverage')
        logits=np.divide(total,weight,out=np.zeros_like(total),where=weight>0)
        output.append(np.column_stack([np.zeros_like(logits),logits]))
    return output


def pretrained_probabilities(model,tokenizer,records,max_length=256,stride=128,batch_size=16):
    values=pretrained_logits(model,tokenizer,records,max_length,stride,batch_size)
    output=[]
    for logits in values:
        probs=1/(1+np.exp(-np.clip(logits[:,1],-50,50)));probs[-1]=0;output.append(probs)
    return output


def pretrained_predictions(model,tokenizer,records,max_length=256,stride=128,batch_size=16):
    output=[]
    for logits in pretrained_logits(model,tokenizer,records,max_length,stride,batch_size):
        pred=model.crf.decode(logits[:-1]) if model.crf is not None else logits[:-1].argmax(axis=1).astype(bool)
        output.append(np.r_[pred,False])
    return output


def save_bundle(model,tokenizer,folder,metadata):
    from safetensors.torch import save_file
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},folder/'model.safetensors')
    tokenizer.backend_tokenizer.save(str(folder/'tokenizer.json'))
    metadata=dict(metadata,format='pretrained_gap_v1',config=model.encoder.config.to_dict(),
                  rule_states=model.rule_embedding.num_embeddings,
                  special_tokens=tokenizer.special_tokens_map)
    (folder/'segmenter.json').write_text(json.dumps(metadata,indent=2)+'\n')


def load_bundle(folder):
    from transformers import AutoConfig,PreTrainedTokenizerFast
    from safetensors.torch import load_file
    folder=Path(folder);meta=json.loads((folder/'segmenter.json').read_text())
    if meta['format']!='pretrained_gap_v1':raise ValueError('Unsupported segmenter format')
    config=dict(meta['config']);kind=config.pop('model_type')
    config=AutoConfig.for_model(kind,**config)
    model=PretrainedGapModel(config=config,decoder=meta.get('decoder','independent'),rule_states=meta.get('rule_states',2));model.load_state_dict(load_file(folder/'model.safetensors'))
    # Transformers' generic fast-tokenizer constructor defaults this flag to
    # False and rewrites a saved ByteLevel pre-tokenizer unless it is explicit.
    # Preserve the serialized training behavior, including older bundles.
    state=json.loads((folder/'tokenizer.json').read_text()).get('pre_tokenizer') or {}
    options={key:state[key] for key in ('add_prefix_space',) if key in state}
    tokenizer=PreTrainedTokenizerFast(tokenizer_file=str(folder/'tokenizer.json'),**options,**meta['special_tokens'])
    return model,tokenizer,meta


def fit_pretrained(train,validation,pretrained_path,output,metadata,epochs=8,mask_probability=0.,seed=42):
    from transformers import AutoTokenizer
    if epochs<1:raise ValueError('Positive epoch count required')
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed);random.seed(seed);np.random.seed(seed)
    tokenizer=AutoTokenizer.from_pretrained(pretrained_path,use_fast=True,local_files_only=True)
    model=PretrainedGapModel(pretrained_path=pretrained_path)
    windows=[]
    for ri,r in enumerate(train):
        windows.extend(dict(w,record=ri) for w in word_windows(r,tokenizer))
    windows=[w for w in windows if any(owned and train[w['record']]['labels'][g]>=0 for g,owned in zip(w['gaps'],w['owned']))]
    optimizer=torch.optim.AdamW([
        dict(params=model.encoder.parameters(),lr=3e-5),
        dict(params=[*model.rule_embedding.parameters(),*model.head.parameters()],lr=2e-4)],weight_decay=.01)
    total=epochs*math.ceil(len(windows)/16);warm=max(1,int(total*.1))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:min((step+1)/warm,max(0.,(total-step)/max(1,total-warm))))
    loss_fn=nn.CrossEntropyLoss(ignore_index=-100)
    best=-1.;best_state=None;stale=0;history=[];selected=None
    print(f'Pretrained model: {len(train)} train paragraphs, {len(windows)} windows, {sum(p.numel() for p in model.parameters()):,} parameters',flush=True)
    for epoch in range(epochs):
        random.shuffle(windows);model.train();losses=[]
        for at in range(0,len(windows),16):
            inputs,labels=collate(windows[at:at+16],tokenizer.pad_token_id,train)
            if mask_probability:
                ids=inputs['input_ids'];eligible=(ids!=tokenizer.cls_token_id)&(ids!=tokenizer.sep_token_id)&(ids!=tokenizer.pad_token_id)
                ids[(torch.rand(ids.shape)<mask_probability)&eligible]=tokenizer.unk_token_id
            optimizer.zero_grad(set_to_none=True)
            logits=model(**inputs);loss=loss_fn(logits.reshape(-1,2),labels.reshape(-1))
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();scheduler.step()
            losses.append(float(loss.detach()))
            if (at//16+1)%100==0:print(f'Epoch {epoch+1}: batch {at//16+1}/{math.ceil(len(windows)/16)}',flush=True)
        probs=pretrained_probabilities(model,tokenizer,validation)
        selection=choose_threshold(validation,probs)
        entry=dict(epoch=epoch+1,train_loss=float(np.mean(losses)),**selection);history.append(entry)
        print(f'Epoch {epoch+1}: validation F1={selection["f1"]:.4f}, threshold={selection["threshold"]:.2f}',flush=True)
        if selection['f1']>best+1e-6:
            best=selection['f1'];best_state=copy.deepcopy(model.state_dict());stale=0
            selected=dict(entry,mask_probability=mask_probability)
            save_bundle(model,tokenizer,output,dict(metadata,selected=selected,history=history,
                threshold=selection['threshold'],max_length=256,stride=128,seed=seed))
        else:stale+=1
        if stale>=2:break
    model.load_state_dict(best_state)
    metadata=json.loads((Path(output)/'segmenter.json').read_text());metadata['history']=history
    (Path(output)/'segmenter.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return model,tokenizer,metadata


def metrics(records,probabilities,threshold=None):
    """Boundary and exact-utterance results, retaining every recording."""
    rows=[]
    for r,p in zip(records,probabilities):
        values=np.asarray(p)
        if threshold is None:
            if values.dtype!=np.bool_:raise ValueError('Discrete boolean predictions required when no cutoff is supplied')
            pred=values.copy()
        else:pred=values>=threshold
        pred[-1]=False
        c=counts(r['labels'],pred);known=sum(v>=0 for v in r['labels'])
        cuts=[i+1 for i,v in enumerate(pred[:-1]) if v]
        exact=None
        if 'gold_boundaries' in r:
            a=[0,*r['gold_boundaries'],len(pred)];b=[0,*cuts,len(pred)]
            gold=set(zip(a,a[1:]));mine=set(zip(b,b[1:]))
            exact=dict(reference=len(gold),predicted=len(mine),matched=len(gold&mine))
        rows.append(dict(corpus=r['corpus'],participant=r['participant'],recording=r['recording'],gem=r.get('gem'),
            counts=c,labeled_gaps=known,f1=f1(c),predicted_boundaries=cuts,exact=exact))
    def summarize(items):
        c=total_counts([r['counts'] for r in items]);tp,fp,fn=(c[k] for k in ('tp','fp','fn'))
        full=[r['exact'] for r in items if r['exact'] is not None]
        exact={k:sum(r[k] for r in full) for k in ('reference','predicted','matched')} if full else None
        if exact:
            exact.update(precision=exact['matched']/exact['predicted'] if exact['predicted'] else 0.,
                recall=exact['matched']/exact['reference'] if exact['reference'] else 0.,
                f1=2*exact['matched']/(exact['predicted']+exact['reference']) if exact['predicted']+exact['reference'] else 0.)
        return dict(records=len(items),participants=len({r['participant'] for r in items}),counts=c,
            precision=tp/(tp+fp) if tp+fp else 0.,recall=tp/(tp+fn) if tp+fn else 0.,f1=f1(c),exact=exact)
    by_child={p:summarize([r for r in rows if r['participant']==p]) for p in sorted({r['participant'] for r in rows})}
    result=dict(overall=summarize(rows),by_corpus={c:summarize([r for r in rows if r['corpus']==c]) for c in sorted({r['corpus'] for r in rows})},
        by_task={f'{c}/{g}':summarize([r for r in rows if r['corpus']==c and (r['gem'] or 'Unspecified')==g]) for c,g in sorted({(r['corpus'],r['gem'] or 'Unspecified') for r in rows})},
        by_child=by_child,recordings=rows)
    result['participant_macro_f1']=float(np.mean([r['f1'] for r in by_child.values()])) if by_child else None
    return result


class Segmenter:
    """Load once, then segment new paragraphs without any reference labels."""
    def __init__(self,folder,nlp=None,device='auto'):
        torch.set_num_threads(min(4,torch.get_num_threads()))
        self.folder=Path(folder)
        self.model,self.tokenizer,self.metadata=load_bundle(folder)
        self.device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu') if device=='auto' else torch.device(device)
        self.model.to(self.device)
        if nlp is None:
            import spacy
            nlp=spacy.load('en_core_web_sm')
        self.nlp=nlp

    def probabilities(self,tokens,rules=None):
        """Local emission probabilities; these are not CRF posterior marginals."""
        if not tokens:return np.array([])
        rules=self.features(tokens) if rules is None else rules
        return pretrained_probabilities(self.model,self.tokenizer,[dict(tokens=tokens,baseline=rules)],
            self.metadata.get('max_length',256),self.metadata.get('stride',128),self.metadata.get('inference_batch_size',16))[0]

    def segment(self,text,boundary_margin=None):
        if boundary_margin is not None:
            return self.segment_many([text],boundary_margin=boundary_margin)[0]
        matches=list(auto.WORD.finditer(text))
        if not matches:return []
        tokens=[m.group().casefold().replace('’',"'") for m in matches]
        if self.metadata.get('decoding') in {'argmax','viterbi'}:
            rules=self.features(tokens)
            pred=pretrained_predictions(self.model,self.tokenizer,[dict(tokens=tokens,baseline=rules)],
                self.metadata.get('max_length',256),self.metadata.get('stride',128),self.metadata.get('inference_batch_size',16))[0]
            boundaries=[i+1 for i,p in enumerate(pred[:-1]) if p]
        else:
            probs=self.probabilities(tokens);threshold=self.metadata['threshold']
            boundaries=[i+1 for i,p in enumerate(probs[:-1]) if p>=threshold]
        cuts=[0,*[matches[b].start() for b in boundaries],len(text)]
        output=[]
        for a,b in zip(cuts,cuts[1:]):
            while a<b and text[a].isspace():a+=1
            while b>a and text[b-1].isspace():b-=1
            output.append(dict(utterance_id=len(output)+1,text=text[a:b],source_start=a,source_end=b,
                reviewed=False,method='pretrained_joint_gap_segmentation',scoring=auto.score_utterance(text[a:b])))
        return output

    def segment_many(self,texts,boundary_margin=None):
        """Batch independent texts with the same decoding and retained words."""
        texts=list(texts);matches=[list(auto.WORD.finditer(t)) for t in texts]
        active=[i for i,m in enumerate(matches) if m]
        records=[]
        for i in active:
            tokens=[m.group().casefold().replace('’',"'") for m in matches[i]]
            records.append(dict(tokens=tokens,baseline=self.features(tokens)))
        if not active:return [[] for _ in texts]
        options=(self.metadata.get('max_length',256),self.metadata.get('stride',128),self.metadata.get('inference_batch_size',16))
        if boundary_margin is not None:
            if self.model.crf is not None or not np.isfinite(boundary_margin):
                raise ValueError('A finite boundary margin requires independent decoding')
            values=pretrained_logits(self.model,self.tokenizer,records,*options)
            flags=[(v[:,1]-v[:,0])>boundary_margin for v in values]
        elif self.metadata.get('decoding') in {'argmax','viterbi'}:
            flags=pretrained_predictions(self.model,self.tokenizer,records,*options)
        else:
            flags=[p>=self.metadata['threshold'] for p in pretrained_probabilities(self.model,self.tokenizer,records,*options)]
        output=[[] for _ in texts]
        for i,pred in zip(active,flags):
            cuts=[0,*[matches[i][j+1].start() for j,p in enumerate(pred[:-1]) if p],len(texts[i])]
            for a,b in zip(cuts,cuts[1:]):
                while a<b and texts[i][a].isspace():a+=1
                while b>a and texts[i][b-1].isspace():b-=1
                body=texts[i][a:b]
                output[i].append(dict(utterance_id=len(output[i])+1,text=body,source_start=a,source_end=b,
                    reviewed=False,method='pretrained_joint_gap_segmentation',scoring=auto.score_utterance(body)))
        return output

    def features(self,tokens):
        return sentence_clause_features(tokens,self.nlp) if self.metadata.get('rule_feature_kind')=='sentence_clause' else paragraph_rules(tokens,self.nlp)

    def target_rows(self,rows,speaker):
        """Use observed speaker turns, never individual transcript row cuts.

        This production mode uses the same speaker context the original pipeline
        already had. The paragraph-only research benchmark uses segment(text)
        and has no speaker-turn constraints. Timing remains an envelope only.
        """
        output=[]
        for turn_id,turn in enumerate(auto.target_turns(rows,speaker),1):
            parts=[];spans=[];offset=0
            for row in turn:
                if parts:offset+=1
                parts.append(row['text']);spans.append((offset,offset+len(row['text']),row));offset+=len(row['text'])
            for u in self.segment(' '.join(parts)):
                a,b=u.pop('source_start'),u.pop('source_end')
                sources=[r for left,right,r in spans if left<b and right>a]
                u.update(utterance_id=len(output)+1,turn_id=turn_id,turn_char_start=a,turn_char_end=b,
                    source_rows=[r['row'] for r in sources],source_start=sources[0].get('start',''),
                    source_end=sources[-1].get('end',''),timing_status='source_segment_envelope_not_utterance_alignment')
                output.append(u)
        return output

    def target_paragraph(self,rows,speaker):
        """Concatenate only the target speaker, without retaining turn cuts."""
        target=[dict(r,text=' '.join(auto.WORD.findall(r['text']))) for r in rows if r['speaker'].casefold()==speaker.casefold()]
        units=self.target_rows(target,speaker)
        for u in units:
            u.pop('turn_id')
            u['paragraph_char_start']=u.pop('turn_char_start');u['paragraph_char_end']=u.pop('turn_char_end')
            u['context']='one_target_speaker_paragraph'
            u['normalization']='source_punctuation_withheld_word_spelling_retained'
        return units
