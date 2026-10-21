"""Read-only lineage/metric audit of the fresh participant-grouped CV study."""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json

import numpy as np

from speech_pipeline import ROOT, utterances as auto
from speech_pipeline.chat_metrics import native_values, plain_chat
from speech_pipeline.cohorts import load_split, read_manifest, validate_training_views, verify
from speech_pipeline.pipeline import digest, save
from speech_pipeline.research import participant_cv as cv
from speech_pipeline.research.common_benchmark import boundaries
from speech_pipeline.research.transcript_chat import METRICS, summarize


def require(condition, message):
    if not condition:raise ValueError(message)


def check_predictions(records, expected, native_references):
    require(Counter(cv.key(r) for r in records)==Counter(expected.keys()),'Held-out coverage is incomplete or repeated')
    for row in records:
        source=expected[cv.key(row)]
        require(row['participant']==source['participant'],'Prediction assigned to the wrong child')
        utterances=[dict(text=text) for text in row['utterances']]
        require(auto.stream(utterances)[0]==source['tokens'],'Predicted word stream changed')
        require(row['reference']==native_references[cv.key(row)],'Native human reference values changed')
        official=row['official_outputs']
        require(native_values(official['mlu'],official['freq'],len(utterances))==row['predicted'],'Predicted metric/native count mismatch')
        for output in official.values():
            require(hashlib.sha256(output['output'].encode()).hexdigest()==output['output_sha256'],'Native output hash mismatch')
        require(row['chat']==plain_chat(utterances,cv.key(row)),'Predicted CHAT export changed')
        require(row['boundaries']==boundaries(source,utterances),'Boundary/span scores do not reproduce')


def pairs(rows):
    result=[]
    for row in rows:
        for metric in METRICS:
            g,p=row['reference'][metric],row['predicted'][metric]
            result.append(dict(corpus=row['corpus'],participant=row['participant'],recording=row['recording'],
                fold=row['fold'],metric=metric,reference=g,predicted=p,
                signed_error=p-g if p is not None and g is not None else None,
                ape=100*abs(p-g)/abs(g) if p is not None and g else None))
    return result


def intervals(rows, draws=5000):
    grouped=defaultdict(lambda:defaultdict(list))
    for row in pairs(rows):
        if row['ape'] is not None:grouped[row['corpus'],row['metric']][row['participant']].append(row['ape'])
    output=[];rng=np.random.default_rng(cv.SEED)
    for (corpus,metric),children in sorted(grouped.items()):
        means=np.array([np.mean(children[p]) for p in sorted(children)])
        bootstrap=rng.choice(means,size=(draws,len(means)),replace=True).mean(axis=1)
        lo,hi=np.percentile(bootstrap,[2.5,97.5])
        output.append(dict(corpus=corpus,metric=metric,participants=len(means),mean_participant_ape=float(means.mean()),
            ci95_low=float(lo),ci95_high=float(hi),bootstrap_draws=draws,
            scope='Participant-cluster bootstrap conditional on the saved fitted models; all sections from a child resampled together'))
    return output


def write_csv(path, rows):
    with path.open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def audit(partial=False, development_only=False):
    protocol=cv.plan();manifest=read_manifest();verify()
    original_train={cv.key(r):r for r in load_split('train')}
    originals={s:set(manifest['assignments'][s]) for s in ['train','validation','test']}
    require(not(originals['train']&originals['validation'] or originals['train']&originals['test'] or originals['validation']&originals['test']),
        'Original participant cohorts overlap')
    native_path=ROOT/'outputs/validation/transcript_validation/chat_native_references.jsonl'
    native_references={r['key']:r['result']['values'] for r in cv.read_jsonl(native_path)}
    checked=[];oof=[];encoder_hashes=set();head_hashes=set();fold_summaries=[]
    names=[f'fold_{i}' for i in range(1,6)]+([] if development_only else ['final'])
    for name in names:
        folder=cv.OUT/name
        if not (folder/'outer_results.json').exists():
            if partial:continue
            raise FileNotFoundError('Study not complete: '+name)
        selection=json.loads((folder/'selection.json').read_text())
        initial=json.loads((folder/'initialization.json').read_text())
        history=json.loads((folder/'history.json').read_text())
        result=json.loads((folder/'outer_results.json').read_text())
        require(selection['protocol_sha256']==digest(cv.PLAN),'Selection belongs to another protocol')
        require(digest(folder/'model/model.safetensors')==selection['weights_sha256'],'Selected model weights changed')
        require(digest(folder/'initialization.json')==selection['initialization_sha256'],'Initialization evidence changed')
        require(not initial['old_finetuned_weights_loaded'] and initial['fresh_task_head'],'Model initialization was not fresh')
        require(initial['pretrained_files']==protocol['pretrained']['files'],'Original pretrained file provenance changed')
        encoder_hashes.add(initial['encoder_sha256']);head_hashes.add(initial['head_sha256'])
        roles={role:cv.job_rows(name,role) for role in ['fit','inner','outer']}
        identities={role:{r['participant'] for r in rows} for role,rows in roles.items()}
        require(not(identities['fit']&identities['inner'] or identities['fit']&identities['outer'] or identities['inner']&identities['outer']),
            'Within-job participant overlap')
        require(set(initial['fit_participants'])==identities['fit'] and set(initial['inner_participants'])==identities['inner'],
            'Actual fit/selection identities differ from initialization')
        require(set(selection['fit_participants'])==identities['fit'] and set(selection['inner_participants'])==identities['inner'],
            'Selected fit/selection identities differ')
        if name!='final':
            require(set.union(*identities.values())<=originals['train'],'Original eval/test entered CV')
            require(set(selection['outer_participants'])==identities['outer'],'Wrong outer participant assignment')
        else:
            require(identities['fit']<=originals['train'] and identities['inner']<=originals['validation'] and identities['outer']<=originals['test'],
                'Final model mixed the original cohort roles')
        validate_training_views(cv.read_jsonl(cv.EVIDENCE/name/'augmentation.jsonl'),roles['fit'])
        require(selection['selected_epoch']==min(history,key=lambda h:h['inner_objective'])['epoch'],'Epoch was not selected by the registered inner objective')
        best_margin=min(selection['candidates'],key=lambda c:c['objective'])
        require(selection['margin']==best_margin['margin'],'Margin was not selected by the registered inner objective')
        inner=json.loads((folder/'selected_inner_calibrated.json').read_text())
        require({r['participant'] for r in inner['records']}==identities['inner'],'Calibration included the wrong participants')
        require(Counter(cv.key(r) for r in inner['records'])==Counter(cv.key(r) for r in roles['inner']),'Incomplete inner calibration coverage')
        require(abs(cv.objective(summarize(inner['records']))-selection['inner_objective'])<1e-10,'Inner objective does not reproduce')
        require(not selection['outer_scores_accessed_for_selection'],'Selection used outer scores')
        require(result['selection']==selection,'Selection changed after evaluation')
        frozen=datetime.fromisoformat(selection['frozen_utc']);evaluated=datetime.fromisoformat(result['evaluated_utc'])
        require(frozen<evaluated,'Evaluation preceded the frozen choice')
        check_predictions(result['records'],{cv.key(r):r for r in roles['outer']},native_references)
        require(summarize(result['records'])==result['summary'],'Held-out summary does not reproduce')
        checked.append(dict(job=name,fit_participants=len(identities['fit']),inner_participants=len(identities['inner']),
            outer_participants=len(identities['outer']),outer_sections=len(result['records']),participant_overlap=0,
            fresh_original_pretrained_initialization=True,epoch=selection['selected_epoch'],margin=selection['margin'],
            selection_frozen_utc=selection['frozen_utc'],evaluated_utc=result['evaluated_utc'],weights_sha256=selection['weights_sha256'],
            words_references_native_outputs_and_boundary_scores_verified=True))
        if name!='final':
            oof.extend(result['records'])
            fold_summaries.extend(dict(fold=name,**r) for r in result['summary'])
    require(len(encoder_hashes)<=1,'Different pretrained encoder initialization across jobs')
    require(len(head_hashes)==len(checked),'Task head initialization was reused across jobs')
    require(digest(cv.OUT/'previous_production/model.safetensors')==protocol['prior_evidence']['prior_weights_sha256'],'Previous weights archive changed')
    if not partial:
        require(Counter(cv.key(r) for r in oof)==Counter(original_train.keys()),'Out-of-fold coverage is not exactly once')
        combined=json.loads((cv.OUT/'cross_validation_results.json').read_text())
        require(combined['records']==oof and combined['summary']==summarize(oof),'Combined CV results do not reproduce')
        write_csv(cv.OUT/'cross_validation_metric_pairs.csv',pairs(oof))
        write_csv(cv.OUT/'fold_metric_summary.csv',fold_summaries)
        bootstrap=dict(cross_validation=intervals(oof))
        if not development_only:
            final=json.loads((cv.OUT/'final/outer_results.json').read_text())
            write_csv(cv.OUT/'final_test_metric_pairs.csv',pairs(final['records']))
            bootstrap['final_test']=intervals(final['records'])
        save(cv.OUT/'participant_bootstrap_intervals.json',bootstrap)
    result=dict(audited_utc=datetime.now(timezone.utc).isoformat(),complete=not partial,completed_jobs=len(checked),jobs=checked,
        cohort_conclusion='Train, eval and test cohorts are participant-independent; no within-fold participant overlap.',
        original_cohort_assignments_preserved=True,original_eval_test_excluded_from_cv=True,all_crops_fit_only=True,
        all_jobs_fresh_original_pretrained_initialization=True,all_outer_choices_frozen_before_evaluation=True,
        outer_prediction_sections_checked=len(oof),all_words_and_native_reference_values_unchanged=True,
        native_outputs_hashes_and_metrics_reproduced=True,protocol_sha256=digest(cv.PLAN))
    save(cv.EVIDENCE/('partial_integrity_audit.json' if partial else 'cv_integrity_audit.json' if development_only else 'final_integrity_audit.json'),result)
    print(json.dumps(result,indent=2))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--partial',action='store_true')
    args=parser.parse_args();audit(args.partial)


if __name__=='__main__':main()
