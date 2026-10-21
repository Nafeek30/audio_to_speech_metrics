"""One bounded development-only global boundary calibration; no forced counts."""
import argparse,json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from speech_pipeline import ROOT,utterances as auto
from speech_pipeline.chat_metrics import analyze
from speech_pipeline.pipeline import save,digest
from speech_pipeline.research.common_benchmark import boundaries,units
from speech_pipeline.research.transcript_chat import OUT,EVIDENCE,read_records,section_rows,observed_turns,summarize,PRIMARY

MARGINS=(0.,.25,.5,.75,1.,1.5,2.)
CONTEXTS=('paragraph_ml','turn_ml')
PLAN=EVIDENCE/'calibration_protocol.json'


def register():
    if PLAN.exists():raise FileExistsError('Preserve bounded calibration protocol')
    save(PLAN,dict(registered_utc=datetime.now(timezone.utc).isoformat(),
        reason='Native CHAT count/MLUw errors remain high. Test one global decision-margin calibration instead of forcing per-recording counts or training more encoders.',
        contexts=list(CONTEXTS),margins=list(MARGINS),
        selection='Exactly 14 fixed candidates. Use only existing validation participants and native reference metrics; same count/MLUw/population-SD objective, participant then corpus balanced. Lowest objective, tie by listed context then margin. No participant/corpus-specific choice. No further grid expansion from these results.',
        inference='Frozen latest MiniLM weights, features, word windows and logits; split when boundary logit difference > global margin. Observed-turn candidate keeps available speaker transitions. Paragraph candidate withholds them. No reference counts enter prediction.',
        speed='Cache reference-free independent-context logits; spaCy.pipe docs supplied to the unchanged rule-feature functions. Mandatory margin-zero parity against every saved validation prediction in both contexts before selection/final scoring.',
        source_hashes={str(p.relative_to(ROOT)):digest(p) for p in [ROOT/'speech_pipeline/segmentation.py',ROOT/'speech_pipeline/chat_metrics.py',EVIDENCE/'chat_protocol.json',OUT/'chat_validation.json']},
        weights_sha256=digest(ROOT/'outputs/results/benchmark/metric_segmenter/model.safetensors'),
        limitations='Existing corpora/test already exposed. Reruns are descriptive. No audio/private data. Preserve prior results; report regressions. Engineering targets unchanged; no claim of clinical acceptability.'))
    print('Registered 14 validation-only candidates',flush=True)


def cache(selected_only=False):
    import spacy
    from speech_pipeline.segmentation import Segmenter,pretrained_logits,sentence_clause_features
    from speech_pipeline.boundary_model import paragraph_rules
    prefix='calibration_selected' if selected_only else 'calibration'
    target=EVIDENCE/f'{prefix}_logits.npz'
    if target.exists():raise FileExistsError('Preserve cached logits')
    p=json.loads(PLAN.read_text());assert digest(ROOT/'outputs/results/benchmark/metric_segmenter/model.safetensors')==p['weights_sha256']
    rows=read_records('all' if selected_only else 'validation');sourcecache={};contexts=[]
    choices=[json.loads((OUT/'chat_calibration_validation.json').read_text())['selection']['context']] if selected_only else CONTEXTS
    for r in rows:
        source,_,_=section_rows(r,sourcecache)
        for context in choices:
            texts=[' '.join(x['text'] for x in source if x['speaker']=='CHI' and x['text'])] if context=='paragraph_ml' else observed_turns(source)
            for text in texts:
                words=auto.WORD.findall(text)
                if words:contexts.append(dict(corpus=r['corpus'],recording=r['recording'],participant=r['participant'],split=r['split'],context=context,words=words))
    nlp=spacy.load('en_core_web_sm');model=Segmenter(ROOT/'outputs/results/benchmark/metric_segmenter',nlp)
    if model.model.crf is not None:raise ValueError('This registered calibration requires independent argmax decoding')
    records=[dict(tokens=[w.casefold().replace('’',"'") for w in c['words']]) for c in contexts]
    active=[i for i,r in enumerate(records) if len(r['tokens'])>1]
    for r in records:
        if len(r['tokens'])==1:r['baseline']=[0] # no scored gaps or encoder windows
    print('Cached-feature contexts',len(contexts),'multiword',len(active),flush=True)
    memo={}
    for at,i in enumerate(active):
        key=tuple(records[i]['tokens'])
        if key not in memo:memo[key]=model.features(records[i]['tokens'])
        records[i]['baseline']=memo[key]
        if at%5000==0:print('Rule features',at+1,'/',len(active),flush=True)
    print('Computing frozen independent-context logits',flush=True)
    logits=pretrained_logits(model.model,model.tokenizer,records,model.metadata.get('max_length',256),model.metadata.get('stride',128),model.metadata.get('inference_batch_size',8))
    offsets=np.r_[0,np.cumsum([len(x) for x in logits])]
    np.savez_compressed(target,differences=np.concatenate([x[:,1]-x[:,0] for x in logits]),offsets=offsets)
    save(EVIDENCE/f'{prefix}_contexts.json',contexts)
    save(EVIDENCE/f'{prefix}_cache_provenance.json',dict(protocol_sha256=digest(PLAN),logits_sha256=digest(target),contexts_sha256=digest(EVIDENCE/f'{prefix}_contexts.json'),contexts=len(contexts),weights_sha256=p['weights_sha256'],feature_policy='Original single-document feature function; memoize identical token sequences only. Validation-only cache before selection; all-data chosen-context cache afterward.'))
    print('Cached logits',len(contexts),flush=True)


def decoded(context,margin,selected_only=False):
    prefix='calibration_selected' if selected_only else 'calibration'
    contexts=json.loads((EVIDENCE/f'{prefix}_contexts.json').read_text());npz=np.load(EVIDENCE/f'{prefix}_logits.npz');offsets=npz['offsets'];values=npz['differences']
    result=defaultdict(list)
    for i,c in enumerate(contexts):
        if c['context']!=context:continue
        cuts=[j+1 for j,v in enumerate(values[offsets[i]:offsets[i+1]-1]) if v>margin]
        result[c['corpus'],c['recording']].extend(units(c['words'],cuts))
    return dict(result)


def reference_map(split):
    if split=='validation':
        # Reuse only the prior validation reference values during selection;
        # never open the combined all-data native-reference file here.
        from speech_pipeline.cohorts import partition_lookup
        groups=json.loads((ROOT/'outputs/validation/benchmark/metric_training.json').read_text())['partitions']
        lookup=partition_lookup(groups);references={}
        for r in json.loads((OUT/'chat_validation.json').read_text())['records']:
            if lookup.get(r['participant'])!='validation' or r['split']!='validation':
                raise ValueError('Non-validation reference in selection input')
            key=r['corpus']+'/'+r['recording'];value=dict(values=r['reference'])
            if key in references and references[key]!=value:raise ValueError('Inconsistent validation references')
            references[key]=value
        return references
    if split!='all':raise ValueError('Request validation references or explicit all-data evaluation')
    return {r['key']:r['result'] for r in (json.loads(s) for s in (EVIDENCE/'chat_native_references.jsonl').read_text().splitlines())}


def select():
    path=OUT/'chat_calibration_validation.json'
    if path.exists():raise FileExistsError('Preserve calibration selection')
    plan=json.loads(PLAN.read_text());baseline=json.loads((OUT/'chat_validation.json').read_text())['records'];parity=[]
    provenance=json.loads((EVIDENCE/'calibration_cache_provenance.json').read_text())
    assert provenance['protocol_sha256']==digest(PLAN)
    assert provenance['logits_sha256']==digest(EVIDENCE/'calibration_logits.npz')
    assert provenance['contexts_sha256']==digest(EVIDENCE/'calibration_contexts.json')
    save(EVIDENCE/'calibration_implementation.json',dict(code_sha256=digest(Path(__file__)),protocol_sha256=digest(PLAN),phase='Before calibration selection; integrity guard only'))
    for context in CONTEXTS:
        zero=decoded(context,0.)
        for r in baseline:
            if r['method']!=context:continue
            same=[u['text'] for u in zero[r['corpus'],r['recording']]]==r['utterances']
            parity.append(dict(corpus=r['corpus'],recording=r['recording'],context=context,matches=same))
            if not same:raise ValueError('Margin-zero parity failed; stop rather than change the registered experiment')
    save(EVIDENCE/'calibration_zero_parity.json',dict(checked=len(parity),all_matched=all(r['matches'] for r in parity),records=parity))
    records=read_records('validation');refs=reference_map('validation');results=[];scores=[]
    checkpoint=EVIDENCE/'calibration_validation_records.jsonl'
    if checkpoint.exists():results=[json.loads(s) for s in checkpoint.read_text().splitlines()]
    done={(r['method'],r['corpus'],r['recording']) for r in results}
    for context in CONTEXTS:
        for margin in MARGINS:
            method=f'{context}/margin={margin:g}';predictions=decoded(context,margin)
            jobs=[r for r in records if (method,r['corpus'],r['recording']) not in done]
            def score(r):
                us=predictions[r['corpus'],r['recording']]
                if auto.stream(us)[0]!=r['tokens']:raise ValueError('Words changed')
                native=analyze(us)
                return dict(corpus=r['corpus'],participant=r['participant'],recording=r['recording'],split=r['split'],method=method,
                    reference=refs[r['corpus']+'/'+r['recording']]['values'],predicted=native['values'],utterances=[u['text'] for u in us],
                    official_outputs=native['official_outputs'])
            with ThreadPoolExecutor(max_workers=4) as pool,checkpoint.open('a') as f:
                for r in pool.map(score,jobs):results.append(r);f.write(json.dumps(r)+'\n');f.flush()
            summary=summarize([r for r in results if r['method']==method]);v=[r['mean_participant_ape'] for r in summary if r['metric'] in PRIMARY]
            if len(v)!=6 or any(x is None for x in v):raise ValueError('Incomplete calibration objective')
            scores.append(dict(context=context,margin=margin,method=method,validation_primary_ape=float(np.mean(v))))
            print(scores[-1],flush=True)
    selected=min(scores,key=lambda r:r['validation_primary_ape'])
    save(path,dict(protocol_sha256=digest(PLAN),selection=selected,candidates=scores,summary=summarize(results),records=results))
    print('Frozen calibration selection',selected,flush=True)


def evaluate():
    path=OUT/'chat_calibrated_results.json'
    if path.exists():raise FileExistsError('Preserve calibrated all-data evaluation')
    validation=json.loads((OUT/'chat_calibration_validation.json').read_text());selection=validation['selection']
    provenance=json.loads((EVIDENCE/'calibration_selected_cache_provenance.json').read_text())
    if validation['protocol_sha256']!=digest(PLAN) or provenance['protocol_sha256']!=digest(PLAN):
        raise ValueError('Calibration protocol changed')
    for name in ['logits','contexts']:
        suffix='npz' if name=='logits' else 'json'
        if provenance[f'{name}_sha256']!=digest(EVIDENCE/f'calibration_selected_{name}.{suffix}'):
            raise ValueError('Selected cache changed')
    if provenance['weights_sha256']!=digest(ROOT/'outputs/results/benchmark/metric_segmenter/model.safetensors'):
        raise ValueError('Model weights changed')
    protocol=json.loads((EVIDENCE/'chat_protocol.json').read_text())
    for name,sha in protocol['input_hashes'].items():
        if digest(ROOT/'outputs/validation/benchmark'/name)!=sha:raise ValueError('Prepared inputs changed')
    predictions=decoded(selection['context'],selection['margin'],True);refs=reference_map('all');records=read_records('all');results=[]
    parity=[]
    for r in validation['records']:
        if r['method']!=selection['method']:continue
        same=[u['text'] for u in predictions[r['corpus'],r['recording']]]==r['utterances']
        parity.append(dict(corpus=r['corpus'],recording=r['recording'],matches=same))
        if not same:raise ValueError('Selected-cache validation parity failed before final scoring')
    save(EVIDENCE/'calibrated_final_parity.json',dict(checked=len(parity),all_matched=all(r['matches'] for r in parity),records=parity))
    checkpoint=EVIDENCE/'calibrated_evaluate_records.jsonl'
    if checkpoint.exists():results=[json.loads(s) for s in checkpoint.read_text().splitlines()]
    done={(r['corpus'],r['recording']) for r in results}
    expected={(r['corpus'],r['recording']) for r in records}
    if len(done)!=len(results) or not done<=expected:raise ValueError('Unexpected or duplicate checkpoint records')
    if any(r['method']!=selection['method'] for r in results):raise ValueError('Checkpoint selection changed')
    sourcecache={}
    for r in records:section_rows(r,sourcecache) # Verify original source hashes, including resumed sections.
    def score(r):
        us=predictions[r['corpus'],r['recording']]
        if auto.stream(us)[0]!=r['tokens']:raise ValueError('Words changed')
        native=analyze(us,r['corpus']+'/'+r['recording'])
        return dict(corpus=r['corpus'],participant=r['participant'],recording=r['recording'],split=r['split'],method=selection['method'],
            reference=refs[r['corpus']+'/'+r['recording']]['values'],predicted=native['values'],utterances=[u['text'] for u in us],
            boundaries=boundaries(r,us),chat=native['chat'],official_outputs=native['official_outputs'])
    for corpus in ['nyu_emerson','ocsc']:
        jobs=[r for r in records if r['corpus']==corpus and (r['corpus'],r['recording']) not in done]
        with ThreadPoolExecutor(max_workers=4) as pool,checkpoint.open('a') as f:
            for i,r in enumerate(pool.map(score,jobs)):
                results.append(r);f.write(json.dumps(r)+'\n');f.flush()
                if i%100==0:print('Calibrated sections',corpus,i+1,'/',len(jobs),flush=True)
        corpus_rows=[r for r in results if r['corpus']==corpus]
        corpus_result=dict(selection=selection,records=corpus_rows,summary=summarize(corpus_rows),
            split_summaries={s:summarize([r for r in corpus_rows if r['split']==s]) for s in ['train','validation','test']})
        corpus_path=OUT/f'chat_calibrated_{corpus}.json'
        if not corpus_path.exists():save(corpus_path,corpus_result)
        print('DATASET COMPLETE',corpus,'sections',len(corpus_rows),'test summary',json.dumps(corpus_result['split_summaries']['test']),flush=True)
    if {(r['corpus'],r['recording']) for r in results}!=expected:raise ValueError('Incomplete final coverage')
    save(path,dict(protocol_sha256=digest(PLAN),selection=selection,records=results,summary=summarize(results),split_summaries={s:summarize([r for r in results if r['split']==s]) for s in ['train','validation','test']}))
    save(ROOT/'outputs/results/benchmark/transcript_workflow.json',dict(**selection,model_directory='outputs/results/benchmark/metric_segmenter',weights_sha256=json.loads(PLAN.read_text())['weights_sha256'],selection_source='outputs/results/transcript_validation/chat_calibration_validation.json',scope='Chosen only on development participants; original model weights and previous bundles unchanged'))
    print('Completed',path,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['register','cache','select','cache-selected','evaluate']);a=p.parse_args()
    {'register':register,'cache':cache,'select':select,'cache-selected':lambda:cache(True),'evaluate':evaluate}[a.stage]()

if __name__=='__main__':main()
