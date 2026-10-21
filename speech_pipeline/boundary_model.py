"""Learn to divide continuous paragraphs into utterances from TalkBank labels.

Run with a Python environment containing torch (the transformer extra). Public
training JSONL is produced by speech_pipeline.corpora prepare-training.
Human boundaries are labels only. The model receives case-folded words, with
optional learned embeddings for the separate rule prediction at each word gap.
Training never replaces production segmentation rules automatically.
"""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import random
import re

import numpy as np
import torch
from torch import nn


class BoundaryTransformer(nn.Module):
    def __init__(self, vocabulary_size, width=64, length=256, use_rules=False):
        super().__init__()
        self.words=nn.Embedding(vocabulary_size,width,padding_idx=0)
        self.positions=nn.Embedding(length,width)
        self.rules=nn.Embedding(2,width) if use_rules else None
        layer=nn.TransformerEncoderLayer(width,4,width*2,dropout=.15,batch_first=True)
        self.encoder=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
        self.output=nn.Linear(width,2)

    def forward(self,words,padding,rules=None):
        positions=torch.arange(words.shape[1],device=words.device).unsqueeze(0)
        x=self.words(words)+self.positions(positions)
        if self.rules is not None:
            if rules is None:raise ValueError('Hybrid model requires aligned rule features')
            x=x+self.rules(rules)
        return self.output(self.encoder(x,src_key_padding_mask=padding))


def counts(labels,predicted):
    valid=np.asarray(labels)>=0
    gold=np.asarray(labels)[valid].astype(bool);pred=np.asarray(predicted)[valid].astype(bool)
    return dict(tp=int((gold&pred).sum()),fp=int((~gold&pred).sum()),fn=int((gold&~pred).sum()))


def f1(c):
    denom=2*c['tp']+c['fp']+c['fn']
    return 2*c['tp']/denom if denom else 0.


def total_counts(rows):
    return {k:sum(r[k] for r in rows) for k in ('tp','fp','fn')}


def chunks(records,vocabulary,length=256):
    result=[]
    for ri,r in enumerate(records):
        for start in range(0,len(r['tokens']),length):
            stop=min(len(r['tokens']),start+length)
            result.append(dict(record=ri,start=start,
                ids=[vocabulary.get(w,1) for w in r['tokens'][start:stop]],
                labels=r['labels'][start:stop],
                rules=r.get('baseline',[0]*len(r['tokens']))[start:stop]))
    return result


def batch(items):
    length=max(len(x['ids']) for x in items)
    words=torch.zeros((len(items),length),dtype=torch.long)
    labels=torch.full_like(words,-100)
    rules=torch.zeros_like(words)
    for i,x in enumerate(items):
        n=len(x['ids']);words[i,:n]=torch.tensor(x['ids']);labels[i,:n]=torch.tensor(x['labels'])
        rules[i,:n]=torch.tensor(x['rules'])
    return words,labels,words==0,rules


def probabilities(model,records,vocabulary):
    parts=chunks(records,vocabulary);output=[np.zeros(len(r['tokens'])) for r in records]
    model.eval()
    with torch.no_grad():
        for at in range(0,len(parts),8):
            group=parts[at:at+8];words,_,padding,rules=batch(group)
            values=torch.softmax(model(words,padding,rules),dim=-1)[:,:,1].numpy()
            for i,x in enumerate(group):output[x['record']][x['start']:x['start']+len(x['ids'])]=values[i,:len(x['ids'])]
    return output


def score(records,probs,threshold):
    return f1(total_counts([counts(r['labels'],p>=threshold) for r,p in zip(records,probs)]))


def fit(train_rows, validation, epochs, seed, use_rules=False):
    # Vocabulary uses training participants only, including frequency pruning.
    frequencies=Counter(w for r in train_rows for w in r['tokens'])
    vocabulary={w:i+2 for i,(w,n) in enumerate(sorted(frequencies.items())) if n>=2}
    # Gaps from rare words may have left unused IDs; make IDs contiguous.
    vocabulary={w:i+2 for i,w in enumerate(vocabulary)}
    torch.manual_seed(seed);random.seed(seed);np.random.seed(seed)
    model=BoundaryTransformer(len(vocabulary)+2,use_rules=use_rules)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    labels=[v for r in train_rows for v in r['labels'] if v>=0]
    positives=sum(labels);negatives=len(labels)-positives
    if not positives or not negatives:raise ValueError('Training split needs positive and negative boundary labels')
    loss_fn=nn.CrossEntropyLoss(weight=torch.tensor([1.,negatives/positives]),ignore_index=-100)
    parts=[x for x in chunks(train_rows,vocabulary) if any(y>=0 for y in x['labels'])]
    best=-1.;best_state=None;stale=0;best_epoch=0
    for epoch in range(epochs):
        random.shuffle(parts);model.train()
        for at in range(0,len(parts),8):
            words,target,padding,rules=batch(parts[at:at+8])
            optimizer.zero_grad(set_to_none=True)
            logits=model(words,padding,rules)
            loss=loss_fn(logits.reshape(-1,2),target.reshape(-1))
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        val=probabilities(model,validation,vocabulary)
        value=score(validation,val,.5)
        print(f'Epoch {epoch+1}: validation boundary F1={value:.4f}',flush=True)
        if value>best+1e-8:
            best=value;best_state=copy.deepcopy(model.state_dict());best_epoch=epoch+1;stale=0
        else:stale+=1
        if stale>=7:break
    model.load_state_dict(best_state)
    val=probabilities(model,validation,vocabulary)
    threshold=max(np.linspace(.05,.95,19),key=lambda t:(score(validation,val,t),-abs(t-.5)))
    return model, vocabulary, float(threshold), best_epoch


def grouped_split(records, seed):
    """70/15/15 by child within corpus; exact duplicate sessions stay together."""
    people=sorted({r['participant'] for r in records})
    parent={p:p for p in people}
    def root(p):
        while parent[p]!=p:
            parent[p]=parent[parent[p]];p=parent[p]
        return p
    seen={}
    for row in records:
        if len(row['tokens'])<20:continue  # short generic responses are not sessions
        fingerprint=hashlib.sha256(json.dumps(row['tokens']).encode()).hexdigest()
        person=row['participant']
        if fingerprint in seen:parent[root(person)]=root(seen[fingerprint])
        else:seen[fingerprint]=person
    groups={}
    for person in people:groups.setdefault(root(person),[]).append(person)
    corpora={p:{r['corpus'] for r in records if r['participant']==p} for p in people}
    strata={}
    for members in groups.values():
        stratum=tuple(sorted({c for p in members for c in corpora[p]}))
        strata.setdefault(stratum,[]).append(members)
    result={k:[] for k in ('train','validation','test')}
    rng=random.Random(seed)
    for stratum,items in sorted(strata.items()):
        if len(items)<5:
            raise ValueError(f'Need at least five independent child groups in {stratum}')
        rng.shuffle(items);n=max(1,round(.15*len(items)))
        for split,subset in [('test',items[:n]),('validation',items[n:2*n]),('train',items[2*n:])]:
            result[split].extend(p for group in subset for p in group)
    return result


def evaluate(model, vocabulary, threshold, records):
    rows=[]
    for r,prob in zip(records,probabilities(model,records,vocabulary)):
        predicted=(prob>=threshold).astype(int);predicted[-1]=0
        rows.append(dict(corpus=r['corpus'],participant=r['participant'],recording=r['recording'],
            gem=r.get('gem'),labeled_gaps=sum(v>=0 for v in r['labels']),
            baseline=counts(r['labels'],r['baseline']),transformer=counts(r['labels'],predicted),
            predicted_boundaries=[i+1 for i,v in enumerate(predicted[:-1]) if v]))
    def summary(subset):
        return dict(records=len(subset),participants=len({r['participant'] for r in subset}),
            **{method:dict(**total_counts([r[method] for r in subset]),
                          f1=f1(total_counts([r[method] for r in subset]))) for method in ('baseline','transformer')})
    tasks=sorted({(r['corpus'],r.get('gem') or 'Unspecified') for r in rows})
    return dict(overall=summary(rows),by_corpus={c:summary([r for r in rows if r['corpus']==c])
        for c in sorted({r['corpus'] for r in rows})},
        by_task={f'{c}/{g}':summary([r for r in rows if r['corpus']==c and (r.get('gem') or 'Unspecified')==g])
                 for c,g in tasks},recordings=rows)


def train_talkbank(data, output, epochs=25, seed=42, evaluation_data=None, use_rules=False):
    """Fit on TalkBank only; choose model/threshold on validation; test once."""
    if epochs<1:raise ValueError('epochs must be positive')
    torch.set_num_threads(min(4,torch.get_num_threads()))
    torch.use_deterministic_algorithms(True)
    records=[json.loads(line) for line in data.read_text().splitlines() if line.strip()]
    if not records:raise ValueError('No training records')
    for r in records:
        if r.get('corpus') not in {'ocsc','nyu_emerson'}:
            raise ValueError('TalkBank training accepts the requested public corpora only')
        if r.get('participant_verified') is not True:
            raise ValueError('Verify participant identities before splitting')
        if r.get('task')!='paragraph_utterance_boundaries' or r.get('forced_boundaries'):
            raise ValueError('Prepare paragraph boundary targets without speaker-turn constraints')
        n=len(r['tokens'])
        if not n or not(n==len(r['baseline'])==len(r['labels'])) or r['labels'][-1]!=-100:
            raise ValueError('Misaligned arrays or unmasked recording end')
        if any(v not in (0,1) for v in r['baseline']):raise ValueError('Rule features must be binary word-gap predictions')
    splits=grouped_split(records,seed)
    subsets={s:[r for r in records if r['participant'] in ids] for s,ids in splits.items()}
    if not all(any(v>=0 for r in subset for v in r['labels']) for subset in subsets.values()):
        raise ValueError('All splits must contain labeled gaps')
    split_counts={s:dict(records=len(rows),words=sum(len(r['tokens']) for r in rows),
        labeled_gaps=sum(v>=0 for r in rows for v in r['labels'])) for s,rows in subsets.items()}
    split_manifest=dict(task='Paragraph-to-utterance boundaries',seed=seed,
        input_sha256=hashlib.sha256(data.read_bytes()).hexdigest(),participants=splits,counts=split_counts,
        records={s:[dict(corpus=r['corpus'],participant=r['participant'],paragraph=r['recording']) for r in rows]
                 for s,rows in subsets.items()},
        policy='Split children before fitting. Test labels are evaluated only after vocabulary, weights, epoch and threshold are frozen.')
    (data.parent/'talkbank_split.json').write_text(json.dumps(split_manifest,indent=2)+'\n')
    print('TalkBank children: '+', '.join(f'{s}={len(ids)}' for s,ids in splits.items()),flush=True)
    model,vocabulary,threshold,epoch=fit(subsets['train'],subsets['validation'],epochs,seed,use_rules)
    result=dict(scope='Paragraph-to-utterance segmentation trained on TalkBank; no speaker-turn cuts; no audio ASR claim',
        use_rules=use_rules,
        architecture='Two transformer encoder layers, width 64, four heads; word and position embeddings'+
            (' plus learned rule-gap embeddings' if use_rules else '')+'; binary word-gap boundary prediction',
        design='Child-disjoint 70/15/15 per corpus; exact duplicate sessions grouped; validation-only epoch and threshold selection',
        seed=seed,torch_version=torch.__version__,input_sha256=hashlib.sha256(data.read_bytes()).hexdigest(),
        splits=splits,split_counts=split_counts,
        parameters=sum(p.numel() for p in model.parameters()),selected_epoch=epoch,threshold=threshold,
        validation=evaluate(model,vocabulary,threshold,subsets['validation']),
        held_out_test=evaluate(model,vocabulary,threshold,subsets['test']))
    if evaluation_data is not None:
        external=[json.loads(line) for line in evaluation_data.read_text().splitlines() if line.strip()]
        if any(r['participant'] in {p for ids in splits.values() for p in ids} for r in external):
            raise ValueError('Additional evaluation participants overlap the TalkBank data')
        result['additional_evaluation']=dict(input_sha256=hashlib.sha256(evaluation_data.read_bytes()).hexdigest(),
            scope='Frozen model; additional labels never used for training or model selection',
            **evaluate(model,vocabulary,threshold,external))
    output.mkdir(parents=True,exist_ok=True)
    stem='hybrid_model' if use_rules else 'talkbank_model'
    torch.save(dict(architecture='ParagraphBoundaryTransformer',width=64,length=256,use_rules=use_rules,state_dict=model.state_dict(),
        vocabulary=vocabulary,threshold=threshold,input_sha256=result['input_sha256'],splits=splits),output/f'{stem}.pt')
    (output/f'{stem}.json').write_text(json.dumps(result,indent=2)+'\n')
    test=result['held_out_test']['overall']
    print(f'{stem} held-out F1: rules={test["baseline"]["f1"]:.4f}; transformer={test["transformer"]["f1"]:.4f}',flush=True)
    return result


def paragraph_rules(tokens, nlp=None):
    """Run the existing rules on the same normalized, unpunctuated paragraph."""
    import spacy
    from speech_pipeline import utterances as auto
    nlp=nlp if nlp is not None else spacy.load('en_core_web_sm')
    predicted=auto.predict([dict(row=2,speaker='CHI',text=' '.join(tokens),start='',end='')],'CHI',nlp)
    words,boundaries=auto.stream(predicted)
    if words!=tokens:raise ValueError('Rule generation changed the input word stream')
    return [int(i+1 in boundaries) for i in range(len(tokens))]


def segment_paragraph(text, checkpoint, nlp=None):
    """Return utterances with original spelling/text and source character spans."""
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if saved.get('architecture')!='ParagraphBoundaryTransformer':
        raise ValueError('Expected a trained paragraph segmentation checkpoint')
    use_rules=saved.get('use_rules',False)
    model=BoundaryTransformer(len(saved['vocabulary'])+2,width=saved['width'],length=saved['length'],use_rules=use_rules)
    model.load_state_dict(saved['state_dict'])
    matches=list(re.finditer(r"[^\W_]+(?:['’\-][^\W_]+)*",text))
    if not matches:return []
    tokens=[m.group().casefold().replace('’',"'") for m in matches]
    record=dict(tokens=tokens,labels=[-100]*len(tokens))
    if use_rules:record['baseline']=paragraph_rules(tokens,nlp)
    probs=probabilities(model,[record],saved['vocabulary'])[0]
    boundaries=[i+1 for i,p in enumerate(probs[:-1]) if p>=saved['threshold']]
    cuts=[0,*[matches[b].start() for b in boundaries],len(text)]
    result=[]
    for i,(a,b) in enumerate(zip(cuts,cuts[1:])):
        while a<b and text[a].isspace():a+=1
        while b>a and text[b-1].isspace():b-=1
        result.append(dict(utterance_id=i+1,text=text[a:b],source_start=a,source_end=b))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    training=commands.add_parser('train')
    training.add_argument('--data',type=Path,required=True);training.add_argument('--output',type=Path,required=True)
    training.add_argument('--epochs',type=int,default=25);training.add_argument('--seed',type=int,default=42)
    training.add_argument('--use-rules',action='store_true',help='Learn from paragraph words and existing rule-gap predictions together')
    training.add_argument('--evaluate-data',type=Path,help='Untouched target cohort; evaluated after the TalkBank model is frozen')
    segment=commands.add_parser('segment')
    segment.add_argument('--model',type=Path,required=True)
    inputs=segment.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--text');inputs.add_argument('--input',type=Path)
    segment.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.command=='train':train_talkbank(args.data,args.output,args.epochs,args.seed,args.evaluate_data,args.use_rules)
    else:
        text=args.input.read_text() if args.input else args.text
        result=json.dumps(segment_paragraph(text,args.model),ensure_ascii=False,indent=2)+'\n'
        if args.output:args.output.write_text(result)
        else:print(result,end='')
