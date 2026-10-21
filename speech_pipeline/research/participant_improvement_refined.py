"""Fixed eval-only ensemble improvement after training-only five-fold CV."""
import argparse
from collections import Counter
from datetime import datetime
import gc
import json

import numpy as np
import torch

from speech_pipeline import ROOT
from speech_pipeline.cohorts import load_split, read_manifest, validate_training_views
from speech_pipeline.ensemble import mean_probability_logits
from speech_pipeline.pipeline import digest, save
from speech_pipeline.research import participant_cv as cv
from speech_pipeline.research import participant_cv_audit as checks
from speech_pipeline.research.transcript_chat import PRIMARY, summarize
from speech_pipeline.segmentation import load_bundle

EVIDENCE=ROOT/'outputs/validation/participant_improvement_v3'
OUT=ROOT/'outputs/results/participant_improvement_v3'
PLAN=EVIDENCE/'protocol.json'
REFINED_MARGINS=(-1.,-.5,-.25,-.125,0.,.125,.25,.5,.75,1.,1.5,2.)
CANDIDATES=[('full_train',['final']),('five_fold_ensemble',[f'fold_{i}' for i in range(1,6)]),
    ('six_model_ensemble',[f'fold_{i}' for i in range(1,6)]+['final'])]


def register():
    if PLAN.exists():raise FileExistsError('Preserve registered improvement protocol')
    if (cv.OUT/'final/outer_results.json').exists():raise ValueError('Register improvement before fresh final test')
    if (ROOT/'outputs/results/participant_improvement_v2/test_results.json').exists():raise ValueError('Refinement must precede fresh test scoring')
    EVIDENCE.mkdir(parents=True);OUT.mkdir(parents=True)
    prior=ROOT/'outputs/results/participant_improvement_v2/development_candidates_in_progress.json'
    prior_candidates=json.loads(prior.read_text())
    save(EVIDENCE/'previous_development_candidates.json',prior_candidates)
    import shutil
    source=ROOT/'outputs/results/participant_improvement_v2/logits/validation'
    shutil.copytree(source,OUT/'logits/validation')
    shutil.copy2(ROOT/'outputs/validation/participant_improvement_v2/deployment_policy.json',EVIDENCE/'deployment_policy.json')
    save(PLAN,dict(registered_utc=cv.now(),reason='User requests best possible results. Fresh full-train model eval optimum is at lower grid endpoint zero; prospectively test a fixed range in both directions before fresh test. No weight/participant changes.',
        parent_protocol_sha256=digest(cv.PLAN),cohort_manifest_sha256=digest(cv.BUNDLE/'manifest.json'),
        candidates=[dict(name=n,members=m) for n,m in CANDIDATES],margins=list(REFINED_MARGINS),
        method='Equal mean of boundary probabilities, converted to log odds before the common boundary margin. Singleton uses exact original logits.',
        selection='Minimum original-eval corpus-balanced mean participant APE across eligible utterance count, MLUw and length SD. Ties keep listed candidate then margin order.',
        gradients='Original training participants only; five-fold training recipe unchanged; final full-training model selected epoch on original eval.',
        cv_use='Fold models provide a fixed variance-reduction ensemble candidate. Outer-fold metrics never choose candidates, members, weights or margins.',
        original_test='Open only after candidate, margin and all component weight hashes are frozen; score chosen candidate once. Never select from final test.',
        oof_policy='Five-fold OOF scores describe the dedicated held-out fold model only. Ensemble is evaluated on original eval/test, which every member excludes from gradients.',
        new_final_test_scored_at_registration=False,
        extension_of_protocol_sha256=digest(ROOT/'outputs/validation/participant_improvement_v2/protocol.json'),
        previous_development_candidates_sha256=digest(EVIDENCE/'previous_development_candidates.json'),
        preserved_prior_comparisons=len(prior_candidates),
        refinement_evidence='Full-train original-eval objective: margin 0 = 11.8071%, .25 = 19.8952%; test never opened. Broaden threshold calibration to both sides for newly trained scores.',
        implementation_sha256={str(p.relative_to(ROOT)):digest(p) for p in [__file_path(),ROOT/'speech_pipeline/ensemble.py']}))
    print('REGISTERED 3 fixed model candidates x 12 margins; original eval only',flush=True)


def __file_path():
    from pathlib import Path
    return Path(__file__)


def plan():
    p=json.loads(PLAN.read_text());cv.plan()
    if p['parent_protocol_sha256']!=digest(cv.PLAN) or p['cohort_manifest_sha256']!=digest(cv.BUNDLE/'manifest.json'):
        raise ValueError('Parent protocol/cohorts changed')
    for name,sha in p['implementation_sha256'].items():
        if digest(ROOT/name)!=sha:raise ValueError('Improvement implementation changed after registration')
    return p


def members(names):
    result=[]
    for name in names:
        folder=cv.OUT/name;selection=json.loads((folder/'selection.json').read_text())
        if digest(folder/'model/model.safetensors')!=selection['weights_sha256']:raise ValueError('Frozen component changed')
        result.append(dict(name=name,model_directory=str((folder/'model').relative_to(ROOT)),weights_sha256=selection['weights_sha256'],
            selection_sha256=digest(folder/'selection.json')))
    return result


def cached_infer(item,rows,prepared,role):
    folder=OUT/'logits'/role;folder.mkdir(parents=True,exist_ok=True);path=folder/(item['name']+'.json')
    signatures={cv.key(r):dict(participant=r['participant'],source_sha256=r['source_sha256']) for r in rows}
    if path.exists():
        result=json.loads(path.read_text())
        if result['member']!=item or result['sources']!=signatures:raise ValueError('Emission cache provenance changed')
    else:
        model,tokenizer,_=load_bundle(ROOT/item['model_directory']);torch.set_num_threads(4)
        device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu');model.to(device)
        logits=cv.infer(model,tokenizer,rows,prepared)
        result=dict(member=item,sources=signatures,created_utc=cv.now(),values={k:[[words,v.tolist()] for words,v in contexts] for k,contexts in logits.items()})
        save(path,result);del model;gc.collect()
        if device.type=='mps':torch.mps.empty_cache()
        print('EMISSIONS',role,item['name'],'sections',len(rows),flush=True)
    return {k:[(words,np.array(v)) for words,v in contexts] for k,contexts in result['values'].items()}


def combine(names,emissions):
    first=emissions[names[0]];result={}
    if any(set(emissions[n])!=set(first) for n in names):raise ValueError('Ensemble section coverage differs')
    for key,contexts in first.items():
        result[key]=[]
        if any(len(emissions[n][key])!=len(contexts) for n in names):raise ValueError('Ensemble contexts differ')
        for i,(words,_) in enumerate(contexts):
            if any(emissions[n][key][i][0]!=words for n in names):raise ValueError('Ensemble words differ')
            result[key].append((words,mean_probability_logits([emissions[n][key][i][1] for n in names])))
    return result


def select():
    p=plan()
    if (OUT/'selection.json').exists():return
    if not (cv.EVIDENCE/'development_complete.json').exists():raise ValueError('Finish component training first')
    if (OUT/'test_results.json').exists():raise ValueError('Test was opened before selection')
    rows=load_split('validation');prepared=cv.prepare_rows(rows);components=members([f'fold_{i}' for i in range(1,6)]+['final'])
    emissions={m['name']:cached_infer(m,rows,prepared,'validation') for m in components}
    candidates=[]
    previous=json.loads((EVIDENCE/'previous_development_candidates.json').read_text())
    if digest(EVIDENCE/'previous_development_candidates.json')!=p['previous_development_candidates_sha256']:
        raise ValueError('Preserved development comparisons changed')
    reuse={(c['name'],c['margin']):c for c in previous}
    progress=OUT/'development_candidates_in_progress.json'
    if progress.exists():
        for c in json.loads(progress.read_text()):reuse[c['name'],c['margin']]=c
    for candidate in p['candidates']:
        logits=combine(candidate['members'],emissions)
        for margin in p['margins']:
            entry=reuse.get((candidate['name'],margin))
            if entry is not None:
                if entry['members']!=candidate['members']:raise ValueError('Cached candidate membership changed')
                print('REUSED DEVELOPMENT',candidate['name'],'margin',margin,'objective',entry['objective'],flush=True)
            else:
                result=cv.score(rows,prepared,logits,margin,candidate['name'],'original_eval')
                entry=dict(name=candidate['name'],members=candidate['members'],margin=margin,objective=result['objective'],summary=result['summary'])
                print('DEVELOPMENT',candidate['name'],'margin',margin,'objective',result['objective'],flush=True)
            candidates.append(entry);save(progress,candidates)
    winner=min(candidates,key=lambda c:c['objective'])
    logits=combine(winner['members'],emissions)
    best=cv.score(rows,prepared,logits,winner['margin'],winner['name'],'original_eval',retain=True)
    if best['objective']!=winner['objective'] or best['summary']!=winner['summary']:
        raise ValueError('Selected eval result differs from cached candidate comparison')
    save(OUT/'selected_validation_results.json',best)
    selected_members=[m for m in components if m['name'] in winner['members']]
    selection=dict(frozen_utc=cv.now(),protocol_sha256=digest(PLAN),candidate=winner['name'],margin=winner['margin'],
        objective=winner['objective'],members=selected_members,candidates=candidates,validation_participants=sorted({r['participant'] for r in rows}),
        validation_sections=len(rows),validation_results_sha256=digest(OUT/'selected_validation_results.json'),test_used_for_selection=False)
    save(OUT/'selection.json',selection)
    (OUT/'model').mkdir(exist_ok=True)
    save(OUT/'model/ensemble.json',dict(format='gap_probability_ensemble_v1',name=selection['candidate'],margin=selection['margin'],
        members=selected_members,rule_feature_kind='clause',threshold=None,decoding='ensemble_probability_margin',
        input_sha256=digest(cv.BUNDLE/'manifest.json'),protocol_sha256=digest(PLAN)))
    print('IMPROVEMENT FROZEN',winner['name'],'margin',winner['margin'],'eval objective',winner['objective'],flush=True)


def evaluate():
    plan()
    if (OUT/'test_results.json').exists():return
    selection=json.loads((OUT/'selection.json').read_text())
    if selection['protocol_sha256']!=digest(PLAN) or selection['test_used_for_selection']:raise ValueError('Invalid frozen selection')
    if members([m['name'] for m in selection['members']])!=selection['members']:raise ValueError('Frozen component mismatch')
    # Test records are deliberately loaded only here, after frozen choice checks.
    rows=load_split('test');prepared=cv.prepare_rows(rows)
    forbidden=set(selection['validation_participants'])|{r['participant'] for r in load_split('train')}
    if any(r['participant'] in forbidden for r in rows):raise ValueError('Test participant overlap')
    emissions={m['name']:cached_infer(m,rows,prepared,'test') for m in selection['members']}
    logits=combine([m['name'] for m in selection['members']],emissions)
    result=cv.score(rows,prepared,logits,selection['margin'],selection['candidate'],'original_test',retain=True)
    result.update(selection=selection,evaluated_utc=cv.now(),prediction_role='original_test',selection_sha256=digest(OUT/'selection.json'))
    save(OUT/'test_results.json',result)
    print('FINAL TEST COMPLETE',json.dumps([{k:r[k] for k in ['corpus','metric','mean_participant_ape']} for r in result['summary'] if r['metric'] in PRIMARY]),flush=True)


def audit():
    plan();base=checks.audit(development_only=True)
    manifest=read_manifest();train=load_split('train');validation=load_split('validation');test=load_split('test')
    folder=cv.OUT/'final';initial=json.loads((folder/'initialization.json').read_text());selected=json.loads((folder/'selection.json').read_text());history=json.loads((folder/'history.json').read_text())
    fit={r['participant'] for r in train};inner={r['participant'] for r in validation};outer={r['participant'] for r in test}
    checks.require(not(fit&inner or fit&outer or inner&outer),'Original participant cohorts overlap')
    checks.require(set(initial['fit_participants'])==fit and set(initial['inner_participants'])==inner,'Final model training roles changed')
    checks.require(not initial['old_finetuned_weights_loaded'] and initial['fresh_task_head'],'Final model reused old task weights')
    checks.require(initial['pretrained_files']==cv.plan()['pretrained']['files'],'Final encoder provenance changed')
    checks.require(initial['encoder_sha256']==json.loads((cv.OUT/'fold_1/initialization.json').read_text())['encoder_sha256'],'Final encoder initialization differs')
    checks.require(selected['initialization_sha256']==digest(folder/'initialization.json'),'Final initialization record changed')
    checks.require(selected['selected_epoch']==min(history,key=lambda x:x['inner_objective'])['epoch'],'Final epoch not chosen by eval')
    checks.require(digest(folder/'model/model.safetensors')==selected['weights_sha256'],'Final weights changed')
    validate_training_views(cv.read_jsonl(cv.EVIDENCE/'final/augmentation.jsonl'),train)
    selection=json.loads((OUT/'selection.json').read_text());result=json.loads((OUT/'test_results.json').read_text());dev=json.loads((OUT/'selected_validation_results.json').read_text())
    winner=min(selection['candidates'],key=lambda x:x['objective'])
    checks.require((selection['candidate'],selection['margin'],selection['objective'])==(winner['name'],winner['margin'],winner['objective']),'Ensemble selection differs from development minimum')
    checks.require(selection['objective']==cv.objective(summarize(dev['records'])),'Selected eval objective does not reproduce')
    checks.require(selection['validation_results_sha256']==digest(OUT/'selected_validation_results.json'),'Selected eval results changed')
    checks.require(result['selection']==selection and result['selection_sha256']==digest(OUT/'selection.json'),'Frozen choice changed')
    checks.require(datetime.fromisoformat(selection['frozen_utc'])<datetime.fromisoformat(result['evaluated_utc']),'Final test preceded freeze')
    refs={r['key']:r['result']['values'] for r in cv.read_jsonl(ROOT/'outputs/validation/transcript_validation/chat_native_references.jsonl')}
    checks.check_predictions(dev['records'],{cv.key(r):r for r in validation},refs)
    checks.check_predictions(result['records'],{cv.key(r):r for r in test},refs)
    checks.require(summarize(result['records'])==result['summary'],'Final test summary differs')
    checks.write_csv(OUT/'final_test_metric_pairs.csv',checks.pairs(result['records']))
    checks.write_csv(OUT/'selected_validation_metric_pairs.csv',checks.pairs(dev['records']))
    save(OUT/'participant_bootstrap_intervals.json',dict(final_test=checks.intervals(result['records'])))
    evidence=dict(complete=True,audited_utc=cv.now(),cohort_conclusion='Train, eval and test cohorts are participant-independent.',
        training_participants=len(fit),validation_participants=len(inner),test_participants=len(outer),test_sections=len(test),
        original_assignments_preserved=True,all_augmentation_training_only=True,five_fold_audit_sha256=digest(cv.EVIDENCE/'cv_integrity_audit.json'),
        final_model_fresh_original_pretrained_initialization=True,development_candidates=36,selected_candidate=selection['candidate'],
        candidate_and_margin_chosen_by_original_eval_only=True,test_scored_after_freeze=True,
        all_words_native_references_outputs_and_boundary_scores_verified=True,selection_sha256=digest(OUT/'selection.json'),test_results_sha256=digest(OUT/'test_results.json'))
    save(EVIDENCE/'final_integrity_audit.json',evidence);print('IMPROVEMENT AUDIT PASSED',json.dumps(evidence),flush=True)


def report():
    selection=json.loads((OUT/'selection.json').read_text());result=json.loads((OUT/'test_results.json').read_text())
    lines=['# Evaluation-selected model improvement','',
        'Train, eval and test cohorts are participant-independent. All original assignments are preserved. Five grouped folds use training participants only; every ensemble member excludes original eval/test participants from gradients.','',
        f"Selected on original eval: **{selection['candidate']}**, margin **{selection['margin']:g}**. Eval count/MLUw/SD objective: **{selection['objective']:.2f}%**. Candidate and margin were frozen before final test scoring.",'',
        '| Candidate | Best eval margin | Eval objective APE % |','|---|---:|---:|']
    for candidate in CANDIDATES:
        best=min([c for c in selection['candidates'] if c['name']==candidate[0]],key=lambda c:c['objective'])
        lines.append(f"| {candidate[0]} | {best['margin']:g} | {best['objective']:.2f} |")
    lines+=['','| Corpus | Metric | Final-test participant APE % |','|---|---|---:|']
    for r in result['summary']:
        if r['metric'] in PRIMARY or r['metric'] in ['TNW','NDW','TTR','MATTR']:
            v='unavailable' if r['mean_participant_ape'] is None else f"{r['mean_participant_ape']:.2f}"
            lines.append(f"| {r['corpus']} | {r['metric']} | {v} |")
    lines+=['','The five-fold out-of-fold report remains separate: it evaluates each dedicated held-out fold model, not the ensemble on participants seen by other members. '
        'LUCID lexical results are unchanged because this comparison changes only speech boundaries. Earlier benchmark exposure remains documented in the parent methods; this work preserves participant separation and does not claim a new external corpus.','']
    (OUT/'REPORT.md').write_text('\n'.join(lines));print('IMPROVEMENT REPORT',OUT/'REPORT.md',flush=True)


def run():
    plan();cv.run();select();evaluate();audit();report()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['register','run','select','evaluate','audit','report'])
    args=parser.parse_args();globals()[args.stage]()


if __name__=='__main__':main()
