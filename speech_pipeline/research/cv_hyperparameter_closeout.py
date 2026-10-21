"""Close a user-stopped search without treating partial folds as completed CV."""
import json
from speech_pipeline import ROOT
from speech_pipeline.pipeline import digest
from speech_pipeline.research import cv_hyperparameter as study
from speech_pipeline.research.transcript_chat import PRIMARY


def audit_completed():
    from datetime import datetime
    from speech_pipeline import utterances as auto
    from speech_pipeline.cohorts import load_split, validate_training_views, verify
    from speech_pipeline.research import participant_cv_audit as checks
    from speech_pipeline.research.transcript_chat import summarize
    verify();p=study.plan();original=load_split('train');original_ids={r['participant'] for r in original}
    keys={study.key(r) for r in original}
    refs={r['key']:r['result']['values'] for r in study.previous.read_jsonl(ROOT/'outputs/validation/transcript_validation/chat_native_references.jsonl') if r['key'] in keys}
    jobs=[];encoders=set();heads={}
    for config in p['recipes']:
        for f in p['folds']:
            name=f"fold_{f['fold']}";folder=study.OUT/config/name
            if not (folder/'cv_results.json').exists():continue
            selected=json.loads((folder/'selection.json').read_text());result=json.loads((folder/'cv_results.json').read_text())
            initial=json.loads((folder/'initialization.json').read_text());history=json.loads((folder/'history.json').read_text())
            roles={r:study.job_rows(config,name,r) for r in ['fit','inner','outer']}
            ids={role:{r['participant'] for r in rows} for role,rows in roles.items()}
            checks.require(set.union(*ids.values())<=original_ids,'Non-training child entered CV')
            checks.require(not(ids['fit']&ids['inner'] or ids['fit']&ids['outer'] or ids['inner']&ids['outer']),'CV participant overlap')
            for role,field in [('fit','fit_participants'),('inner','inner_participants'),('outer','outer_participants')]:
                checks.require(set(initial[field])==ids[role] and set(selected[field])==ids[role],'Actual participant roles differ')
            checks.require(not initial['old_finetuned_weights_loaded'] and initial['fresh_task_head'],'Initialization was not fresh')
            checks.require(initial['pretrained_files']==p['pretrained']['files'],'Pretrained provenance differs')
            encoders.add(initial['encoder_sha256']);heads.setdefault(name,set()).add(initial['head_sha256'])
            checks.require(digest(folder/'initialization.json')==selected['initialization_sha256'],'Initialization evidence changed')
            checks.require(digest(ROOT/selected['model_directory']/'model.safetensors')==selected['weights_sha256'],'Frozen weights changed')
            checks.require(selected['protocol_sha256']==digest(study.PLAN) and selected['recipe']==p['recipes'][config],'Frozen protocol/recipe changed')
            checks.require(selected['selected_epoch']==min(history,key=lambda h:h['inner_objective'])['epoch'],'Wrong epoch selection')
            checks.require(selected['margin']==min(selected['candidates'],key=lambda c:c['objective'])['margin'],'Wrong margin selection')
            inner=json.loads((folder/'selected_inner_calibrated.json').read_text())
            checks.require({study.key(r) for r in inner['records']}=={study.key(r) for r in roles['inner']},'Inner selection coverage differs')
            inner_sources={study.key(r):r for r in roles['inner']}
            for row in inner['records']:
                checks.require(row['reference']==refs[study.key(row)],'Inner reference changed')
                checks.require(auto.stream([dict(text=u) for u in row['utterances']])[0]==inner_sources[study.key(row)]['tokens'],'Inner words changed')
            checks.require(abs(study.objective(summarize(inner['records']))-selected['inner_objective'])<1e-10,'Inner objective differs')
            aug=study.previous.EVIDENCE/name/'augmentation.jsonl' if selected.get('reused_baseline_model') else study.EVIDENCE/config/name/'augmentation.jsonl'
            validate_training_views(study.previous.read_jsonl(aug),roles['fit'])
            checks.require(result['selection']==selected,'Selection changed after evaluation')
            started=json.loads((folder/'evaluation_started.json').read_text())
            checks.require(datetime.fromisoformat(selected['frozen_utc'])<=datetime.fromisoformat(started['started_utc']),'Heldout evaluation began before freeze')
            checks.check_predictions(result['records'],{study.key(r):r for r in roles['outer']},refs)
            checks.require(result['summary']==summarize(result['records']),'Heldout summary differs')
            jobs.append(dict(configuration=config,fold=name,fit=len(ids['fit']),inner=len(ids['inner']),heldout=len(ids['outer']),sections=len(result['records']),all_checks_passed=True))
    checks.require(len(encoders)==1 and all(len(v)==1 for v in heads.values()),'Paired original initialization differs')
    audit=dict(audited_utc=study.now(),complete_registered_study=False,completed_cv_assessments=len(jobs),jobs=jobs,
        all_completed_assessments_passed=True,original_cohorts_participant_independent=True,
        fit_inner_heldout_roles_disjoint=True,fit_only_crops=True,all_original_words_and_native_values_verified=True,
        original_eval_test_excluded_from_cv=True,protocol_sha256=digest(study.PLAN))
    study.save(study.EVIDENCE/'stopped_search_integrity_audit.json',audit)
    return audit


def run():
    stop = json.loads((study.EVIDENCE / 'user_stop.json').read_text())
    protocol = study.plan()
    audit_completed()
    completed, incomplete = [], []
    for config in protocol['recipes']:
        finished = [i for i in range(1, 6) if (study.OUT/config/f'fold_{i}/cv_results.json').exists()]
        if len(finished) == 5:
            completed.append(study.candidate_summary(config))
        else:
            incomplete.append(dict(configuration=config, completed_folds=finished,
                ranking_eligible=False, reason='User requested stopping before this configuration completed all five folds'))
    if not completed:
        raise ValueError('No complete five-fold configurations to report')
    best = min(completed, key=lambda c:c['objective'])
    corpus_best={}
    for corpus in ['nyu_emerson','ocsc']:
        def corpus_objective(candidate):
            return sum(r['mean_participant_ape'] for r in candidate['summary'] if r['corpus']==corpus and r['metric'] in PRIMARY)/3
        winner=min(completed,key=corpus_objective)
        corpus_best[corpus]=dict(configuration=winner['configuration'],cv_primary_objective=corpus_objective(winner),
            scope='Descriptive corpus-specific CV tuning comparison; no new final fit, test or deployment')
    workflow = ROOT / 'outputs/results/benchmark/transcript_workflow.json'
    previous = study.previous.OUT / 'previous_production/transcript_workflow.json'
    if digest(workflow) != digest(previous):
        raise ValueError('Unexpected production workflow change; preserve it for review')
    historical_path = ROOT / 'outputs/results/transcript_validation/chat_calibrated_results.json'
    historical = json.loads(historical_path.read_text())
    result = dict(closed_utc=study.now(), stop_request=stop, registered_grid_completed=False,
        completed_configurations=len(completed), completed_cv_assessments=5*len(completed),
        completed_candidates=completed, incomplete_candidates=incomplete,
        best_completed_configuration=best['configuration'], best_completed_cv_objective=best['objective'],
        corpus_specific_cv_comparisons=corpus_best,
        production_workflow_retained=True, production_workflow_sha256=digest(workflow),
        no_v4_final_fit_or_test=True, no_v4_model_activated=True,
        original_cohorts_participant_independent=True,
        historical_test_summary=historical['split_summaries']['test'],
        historical_test_results_sha256=digest(historical_path),
        scope='User stopped further experiments. Report only complete five-fold configurations; exclude the unfinished configuration as a whole. Best completed CV score is a tuning result, not a new final-test result. Existing production and all prior results remain preserved; no new weights are deployed or test-scored.')
    study.save(study.OUT / 'stopped_search_summary.json', result)
    lines = ['# Results at the user-requested stopping point', '',
        'Further training was stopped at the user’s request. Three configurations completed all five participant-grouped folds. The fourth configuration was interrupted during its first fold and is excluded as a whole from ranking. The registered four-configuration study was not completed. All checkpoints and preceding results are retained.', '',
        '**Train, eval and test cohorts remain participant-independent.** CV uses original train only, with separate fit, inner-validation and held-out children in every fold. Original eval/test stayed outside recipe comparison.', '',
        '| Completed training configuration | Pooled CV tuning objective % | Fold mean ± SD % |',
        '|---|---:|---:|']
    for c in completed:
        lines.append(f"| {c['configuration']} | {c['objective']:.2f} | {c['fold_mean']:.2f} ± {c['fold_sd']:.2f} |")
    lines += ['', f"**Best completed configuration: {best['configuration']}.** The alternatives did not improve its pooled CV objective. This result supports retaining the baseline recipe among the completed comparisons; CV itself did not improve the model. The selected CV minimum is a tuning score, not an unbiased final accuracy estimate. Fold SD is descriptive because the training groups overlap.", '',
        'The previously saved production workflow is retained. No new v4 final fit, final test or production replacement was performed. The CV and existing production-test results below have different models and roles; they must not be merged or presented as a single new test result.', '',
        '| Source | Evaluation | Eligible-count APE % | MLUw APE % | Length-SD APE % |',
        '|---|---|---:|---:|---:|---:|']
    for corpus in ['nyu_emerson', 'ocsc']:
        for label, summary in [('Best completed recipe: five-fold CV tuning', best['summary']), ('Retained production: existing test', result['historical_test_summary'])]:
            lookup = {r['metric']:r['mean_participant_ape'] for r in summary if r['corpus']==corpus}
            lines.append('| '+corpus+' | '+label+' | '+' | '.join(f'{lookup[m]:.2f}' for m in PRIMARY)+' |')
    lines += ['', '**OCSC improved modestly in the completed search.** Its earlier fixed-recipe five-fold result was9.16% count,9.09% MLUw and14.89% length-SD error. The lower count-loss configuration gives8.84%,8.67% and14.73%. These are reasonably good single-digit count/MLU errors, while still above the preset5% engineering goal. The stronger OCSC vocabulary agreement and the larger SD error remain reported separately.', '',
        '| Completed CV configuration | Source | Eligible-count APE % | MLUw APE % | Length-SD APE % |',
        '|---|---|---:|---:|---:|']
    for c in completed:
        for corpus in ['nyu_emerson','ocsc']:
            lookup={r['metric']:r['mean_participant_ape'] for r in c['summary'] if r['corpus']==corpus}
            lines.append('| '+c['configuration']+' | '+corpus+' | '+' | '.join(f'{lookup[m]:.2f}' for m in PRIMARY)+' |')
    lines += ['', 'The lower count-loss setting has the best completed OCSC three-metric CV mean; the baseline has the best NYU and combined means. Each row is one consistent configuration across all five folds. Corpus-specific CV comparisons are tuning results; they do not create a new final-test result or authorize choosing different best values per metric.', '',
        'NYU remains reported, with its younger sample, shorter turns, higher unintelligibility and smaller training sample documented as plausible contributors rather than established causes.', '',
        'LUCID lexical agreement remains unchanged: babble/L2 word-count errors0.76%/1.87%; all fifteen shared measures meet mean-error goals in those120 samples. No-barrier word-count error remains16.60%; sixty vocoder samples have unresolved source mapping. These are comparisons with actual published TAALED values, not a published boundary-model accuracy benchmark.', '',
        'See [LUCID details](LUCID_STATUS.md), [NYU explanation](NYU_EXPLANATION.md), [pipeline](PIPELINE.md), [methods and study status](METHODS.md), and the retained full comparison in `outputs/results/transcript_validation/REPORT.md`.', '']
    (study.OUT/'REPORT.md').write_text('\n'.join(lines))
    print('Closed user-stopped search: complete configurations', len(completed), '; best completed', best['configuration'])
    return result


if __name__ == '__main__':
    run()
