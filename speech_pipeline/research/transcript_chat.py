"""Transcript-only CHAT workflow selection on development, then all-data audit."""
import argparse,csv,json
from collections import defaultdict,Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from speech_pipeline import ROOT,utterances as auto
from speech_pipeline.chat_metrics import analyze,native_values
from speech_pipeline.corpora import read_chat,run_clan,chat_reference_units
from speech_pipeline.pipeline import save,digest
from speech_pipeline.research.original_chat_audit import section_chat
from speech_pipeline.research.common_benchmark import boundaries

OUT=ROOT/'outputs/results/transcript_validation'
EVIDENCE=ROOT/'outputs/validation/transcript_validation'
BASE=ROOT/'outputs/validation/benchmark'
METHODS=('paragraph_ml','turn_ml','turn_rules')
METRICS=('total_utterances','eligible_mlu_utterances','MLUw','length_sd','mlu_words','TNW','NDW','TTR','MATTR')
PRIMARY=('eligible_mlu_utterances','MLUw','length_sd')


def read_records(split):
    """Explicit cohort access; selection callers never open test record files."""
    from speech_pipeline.cohorts import BUNDLE, load_split, partition_lookup, read_manifest, validate_records
    if split not in {'train','validation','test','all'}:raise ValueError('Request an explicit cohort or all-data evaluation')
    plan=json.loads((BASE/'metric_training.json').read_text())
    lookup=partition_lookup(plan['partitions'])
    requested=('train','validation','test') if split=='all' else (split,)
    if BUNDLE.exists():
        rows=[r for group in requested for r in load_split(group,expected_assignments=plan['partitions'])]
        order={tuple(key):i for i,key in enumerate(read_manifest()['record_order'])}
        rows.sort(key=lambda r:order[r['corpus'],r['recording']])
        return [dict(r,split=lookup[r['participant']]) for r in rows]
    rows=[]
    for name in ['talkbank_spoken_development.jsonl','talkbank_spoken_test.jsonl']:
        if name=='talkbank_spoken_test.jsonl' and 'test' not in requested:continue
        if name=='talkbank_spoken_development.jsonl' and requested==('test',):continue
        allowed=('test',) if name=='talkbank_spoken_test.jsonl' else ('train','validation')
        source=[]
        for line in (BASE/name).read_text().splitlines():
            r=json.loads(line);source.append(r)
        validate_records(source,plan['partitions'],allowed)
        for r in source:
            r['split']=lookup[r['participant']]
            if r['split'] in requested:rows.append(r)
    return rows


def register():
    p=EVIDENCE/'chat_protocol.json'
    if p.exists():raise FileExistsError('Preserve registered CHAT experiment')
    save(p,dict(registered_utc=datetime.now(timezone.utc).isoformat(),
        purpose='Fix native CHAT/FREQ eligibility policy and compare three transcript workflows on public validation only. No audio or private participants.',
        candidates=list(METHODS),
        inputs='Same target spoken words. Paragraph ML withholds all turn cuts. Turn ML/rules use only observed speaker transitions, merging consecutive target-speaker CHAT rows; no gold target-row punctuation/utterance ends supplied. This compares workflows with explicitly different available context, not architecture-only effects.',
        model='Existing frozen latest MiniLM bundle, no retraining/threshold changes.',
        model_sha256=digest(ROOT/'outputs/results/benchmark/metric_segmenter/model.safetensors'),
        native_policy='Faithful predicted words; only uh/um explicitly pause-coded. Native MLU -t%mor and FREQ +k +b50 on both predicted and original coded CHAT. Ratios from exact counts; native SD/MATTR precision retained. Original retrace/unintelligibility conventions remain reference information, not fabricated predictions.',
        selection='Validation participants only: mean recording APE within participant, mean participant within corpus, mean across eligible-count/MLUw/population-SD and equally across corpora. Select minimum; fixed tie order paragraph_ml,turn_ml,turn_rules. Test results not used for selection. All-data report separates train/validation/test.',
        reference='All prepared sections in both original corpora; original coded human CHAT, same target CHI. Verbal reference units join explicit continuations; eligible MLU count distinct. Zero denominators/short MATTR unavailable.',
        targets='Project engineering goals, not clinical standards: mean participant APE <=5% for counts/MLUw/TNW/NDW/TTR/MATTR, <=10% SD. Do not conceal failing metrics behind composite.',
        previous_test_exposure='Existing final tests already inspected; corrected reruns are descriptive, not independent confirmation.',
        input_hashes={n:digest(BASE/n) for n in ['talkbank_spoken_development.jsonl','talkbank_spoken_test.jsonl','metric_training.json']},
        code={str(x.relative_to(ROOT)):digest(x) for x in [ROOT/'speech_pipeline/chat_metrics.py',Path(__file__).resolve()]}))
    print('CHAT protocol registered',flush=True)


def section_rows(record,cache):
    filename,number=record['recording'].rsplit('#section',1);source=ROOT/'dataset/public'/record['corpus']/filename
    if source not in cache:
        if digest(source)!=record['source_sha256']:raise ValueError('Source changed')
        sections=[]
        for row in read_chat(source):
            if not sections or sections[-1][0]['gem']!=row['gem']:sections.append([])
            sections[-1].append(row)
        cache[source]=sections
    rows=cache[source][int(number)-1]
    return rows,source,int(number)


def reference(record,rows,source,section):
    from tempfile import TemporaryDirectory
    text,gem=section_chat(source,section)
    if gem!=record['gem']:raise ValueError('Task mismatch')
    with TemporaryDirectory(prefix='human_chat_') as temp:
        path=Path(temp)/'human.cha';path.write_text(text)
        mlu=run_clan('mlu',path,speaker='CHI');freq=run_clan('freq',path,speaker='CHI',window=50)
    return dict(values=native_values(mlu,freq,len(chat_reference_units(rows,'CHI'))),
                official_outputs=dict(mlu=mlu,freq=freq),chat_sha256=__import__('hashlib').sha256(text.encode()).hexdigest())


def predict(method,rows,model,nlp):
    if method not in METHODS:raise ValueError('Unknown transcript workflow')
    if method=='paragraph_ml':return model.segment(' '.join(r['text'] for r in rows if r['speaker']=='CHI' and r['text']))
    if method=='turn_rules':return auto.predict(rows,'CHI',nlp)
    return [u for turn in observed_turns(rows) for u in model.segment(turn)]


def observed_turns(rows):
    turns=[];current=[]
    for r in rows:
        if not r['text']:continue
        if r['speaker']=='CHI':current.append(r['text'])
        elif current:turns.append(' '.join(current));current=[]
    if current:turns.append(' '.join(current))
    return turns


def summarize(rows):
    groups=defaultdict(lambda:defaultdict(list));counts=Counter();missing=Counter()
    for r in rows:
        for metric in METRICS:
            p=r['predicted'][metric];g=r['reference'][metric];k=r['method'],r['corpus'],metric
            if p is not None and g:
                groups[k][r['participant']].append(100*abs(p-g)/abs(g));counts[k]+=1
            else:missing[k]+=1
    out=[]
    for method,corpus in sorted({(r['method'],r['corpus']) for r in rows}):
        for metric in METRICS:
            k=method,corpus,metric;v=[float(np.mean(x)) for x in groups[k].values()]
            error=float(np.mean(v)) if v else None;target=10 if metric=='length_sd' else 5
            out.append(dict(method=method,corpus=corpus,metric=metric,mean_participant_ape=error,available=counts[k],missing=missing[k],participants=len(v),target_ape=target,meets_engineering_target=error<=target if error is not None else None))
    return out


def run(stage):
    import spacy,torch
    from speech_pipeline.segmentation import Segmenter
    protocol=json.loads((EVIDENCE/'chat_protocol.json').read_text())
    for name,sha in protocol['input_hashes'].items():assert digest(BASE/name)==sha
    records=read_records('validation' if stage=='select' else 'all')
    final=OUT/('chat_validation.json' if stage=='select' else 'chat_results.json')
    if final.exists():raise FileExistsError('Preserve completed CHAT comparison')
    if stage=='select':methods=METHODS
    else:methods=[json.loads((OUT/'chat_validation.json').read_text())['selection']['method']]
    nlp=spacy.load('en_core_web_sm');model=Segmenter(ROOT/'outputs/results/benchmark/metric_segmenter',nlp)
    cache={};results=[];refs={};stream=EVIDENCE/f'chat_{stage}_records.jsonl'
    if stream.exists():
        results=[json.loads(s) for s in stream.read_text().splitlines()];print('Resume retained records',len(results),flush=True)
    done={(r['corpus'],r['recording'],r['method']) for r in results}
    batched={}
    if stage=='evaluate' and methods[0] in {'paragraph_ml','turn_ml'}:
        texts=[];owners=[]
        for r in records:
            if (r['corpus'],r['recording'],methods[0]) in done:continue
            rows,_,_=section_rows(r,cache)
            turns=observed_turns(rows) if methods[0]=='turn_ml' else [' '.join(x['text'] for x in rows if x['speaker']=='CHI' and x['text'])]
            for turn in turns:texts.append(turn);owners.append((r['corpus'],r['recording']))
        print('Batching independent transcript contexts:',len(texts),flush=True)
        for owner,us in zip(owners,model.segment_many(texts)):
            batched.setdefault(owner,[]).extend(us)
        validation=json.loads((OUT/'chat_validation.json').read_text())
        parity=[]
        for r in validation['records']:
            if r['method']!=methods[0]:continue
            owner=r['corpus'],r['recording']
            if owner in batched:
                same=[u['text'] for u in batched[owner]]==r['utterances']
                parity.append(dict(corpus=owner[0],recording=owner[1],matches=same))
                if not same:raise ValueError('Batched inference changed frozen validation predictions; stop before final scoring')
        save(EVIDENCE/'batch_inference_parity.json',dict(checked=len(parity),all_matched=all(r['matches'] for r in parity),records=parity))
        print('Batched validation parity passed:',len(parity),flush=True)
    # Prepare references in a bounded worker pool; each CLAN call uses its own PTY/files.
    jobs=[]
    for r in records:
        if all((r['corpus'],r['recording'],m) in done for m in methods):continue
        rows,source,section=section_rows(r,cache);jobs.append((r,rows,source,section))
    ref_path=EVIDENCE/'chat_native_references.jsonl'
    if ref_path.exists():
        for line in ref_path.read_text().splitlines():
            x=json.loads(line);refs[x['key']]=x['result']
    missing=[job for job in jobs if job[0]['corpus']+'/'+job[0]['recording'] not in refs]
    def ref_job(job):
        r,rows,source,section=job
        return dict(key=r['corpus']+'/'+r['recording'],result=reference(r,rows,source,section))
    with ThreadPoolExecutor(max_workers=4) as pool, ref_path.open('a') as f:
        for i,x in enumerate(pool.map(ref_job,missing)):
            refs[x['key']]=x['result'];f.write(json.dumps(x)+'\n');f.flush()
            if i%100==0:print('Native human references',i+1,'/',len(missing),flush=True)
    with stream.open('a') as f:
        for i,r in enumerate(records):
            rows,_,_=section_rows(r,cache);ref=refs.get(r['corpus']+'/'+r['recording'])
            for method in methods:
                if (r['corpus'],r['recording'],method) in done:continue
                us=batched.get((r['corpus'],r['recording']))
                if us is None:us=predict(method,rows,model,nlp)
                words,_=auto.stream(us)
                if words!=r['tokens']:raise ValueError('Source words changed')
                scored=analyze(us,r['corpus']+'/'+r['recording'])
                item=dict(corpus=r['corpus'],participant=r['participant'],recording=r['recording'],split=r['split'],method=method,
                    reference=ref['values'],predicted=scored['values'],utterances=[u['text'] for u in us],
                    boundaries=boundaries(r,us),chat=scored['chat'],official_outputs=scored['official_outputs'])
                results.append(item);f.write(json.dumps(item)+'\n');f.flush()
            if i%50==0:print('CHAT workflows',stage,i+1,'/',len(records),flush=True)
    summary=summarize(results);result=dict(protocol_sha256=digest(EVIDENCE/'chat_protocol.json'),records=results,summary=summary)
    if stage=='select':
        objectives={}
        for method in METHODS:
            values=[x['mean_participant_ape'] for x in summary if x['method']==method and x['metric'] in PRIMARY]
            if len(values)!=6 or any(x is None for x in values):raise ValueError('Incomplete validation selection')
            objectives[method]=float(np.mean(values))
        winner=min(METHODS,key=lambda m:objectives[m]);result['selection']=dict(method=winner,validation_primary_ape=objectives,scope='Validation-only workflow selection; weights and thresholds unchanged')
        print('Selected',result['selection'],flush=True)
    else:
        result['split_summaries']={s:summarize([r for r in results if r['split']==s]) for s in ['train','validation','test']}
        result['coverage']=dict(records=len(results),participants=len({r['participant'] for r in results}),by_split=dict(Counter(r['split'] for r in results)),files_by_corpus={c:len({r['recording'].rsplit('#section',1)[0] for r in results if r['corpus']==c}) for c in ['ocsc','nyu_emerson']})
        pairs=[]
        for r in results:
            for metric in METRICS:
                p=r['predicted'][metric];g=r['reference'][metric]
                pairs.append(dict(corpus=r['corpus'],participant=r['participant'],recording=r['recording'],split=r['split'],method=r['method'],metric=metric,reference=g,predicted=p,signed_error=p-g if p is not None and g is not None else None,ape=100*abs(p-g)/abs(g) if p is not None and g else None))
        with (OUT/'chat_metric_pairs.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(pairs[0]));w.writeheader();w.writerows(pairs)
        print('Coverage',result['coverage'],flush=True)
    save(final,result)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['register','select','evaluate']);a=p.parse_args()
    register() if a.stage=='register' else run(a.stage)

if __name__=='__main__':main()
