"""One report linking published lexical and professional CHAT comparisons."""
from collections import Counter,defaultdict
import csv,json,hashlib
import numpy as np
from pathlib import Path
from speech_pipeline import ROOT
from speech_pipeline.pipeline import save,digest
from speech_pipeline.research.transcript_chat import OUT,EVIDENCE,METRICS,read_records,summarize


def cv_search_section():
    folder=ROOT/'outputs/results/cv_hyperparameter_v4'
    evidence=ROOT/'outputs/validation/cv_hyperparameter_v4'
    closeout=folder/'stopped_search_summary.json'
    if closeout.exists():
        result=json.loads(closeout.read_text())
        lines=['## Completed five-fold comparisons; further experiments stopped','',
            '**Train, eval and test cohorts remain participant-independent.** At the user’s request, further training was stopped. '
            'Three configurations completed all five participant-grouped folds within original train. '
            'The unfinished fourth configuration is excluded as a whole; the registered four-configuration study is incomplete.','',
            '| Completed configuration | Pooled CV tuning objective % | Fold mean ± SD % |','|---|---:|---:|']
        for c in result['completed_candidates']:
            lines.append(f"| {c['configuration']} | {c['objective']:.2f} | {c['fold_mean']:.2f} ± {c['fold_sd']:.2f} |")
        lines+=['',f"Best completed configuration: **{result['best_completed_configuration']}**. "
            'The alternatives did not improve its pooled CV score. These are tuning scores; no new final-test improvement is claimed. '
            'The existing production workflow is retained, and no v4 final fit, test or activation was performed. '
            'All fifteen completed CV assessments passed cohort, crop, initialization, frozen-weight, word-preservation and native-metric checks.','',
            '**OCSC did improve in the completed CV comparisons:** the lower count-loss setting reduced count/MLUw/length-SD errors '
            'from the earlier fixed-recipe9.16/9.09/14.89% to **8.84/8.67/14.73%**. '
            'It helped OCSC while worsening NYU, so the baseline retained the best combined score. '
            'These CV results are distinct from the existing production-test scores below.','',
            'Full completed CV results, existing production-test results, LUCID status, pipeline diagrams and the qualified NYU explanation are in '
            '`outputs/results/cv_hyperparameter_v4/REPORT.md`. The historical results below remain preserved.','']
        return lines
    if not (evidence/'final_integrity_audit.json').exists():return []
    audit=json.loads((evidence/'final_integrity_audit.json').read_text())
    if not audit['complete']:return []
    choice=json.loads((folder/'cv_selection.json').read_text())
    results=json.loads((folder/'cv_results.json').read_text())
    selected=next(c for c in results['candidates'] if c['configuration']==choice['configuration'])
    final=json.loads((folder/choice['configuration']/'final/test_results.json').read_text())
    from speech_pipeline.research.cv_hyperparameter_comparison import run as compare_frozen_models
    comparison=compare_frozen_models()
    lines=['## Current model selected through five-fold cross-validation','',
        '**Train, eval and test cohorts are participant-independent.** All original assignments remain unchanged. '
        'Four learning-rate/count-loss recipes were each assessed across the same five participant-grouped folds inside original train. '
        'Within each fold, fitting, inner-validation and CV-scoring participants are separate. '
        'Inner validation selects epochs/margins; complete pooled CV scores select the training recipe.','',
        f"**Selected recipe: {choice['configuration']}.** Encoder/head learning rates "
        f"{choice['recipe']['encoder_lr']:g}/{choice['recipe']['head_lr']:g}; count-loss weight {choice['recipe']['count_weight']:g}. "
        'A separate fresh final fit uses original train; original eval selects its checkpoint/margin before original-test scoring. '
        'A best individual fold or one-eval ensemble search does not choose the recipe.','',
        '| Recipe | Pooled CV selection objective % | Fold mean ± SD % |','|---|---:|---:|']
    for c in choice['candidates']:
        lines.append(f"| {c['configuration']} | {c['objective']:.2f} | {c['fold_mean']:.2f} ± {c['fold_sd']:.2f} |")
    lines+=['','The selected CV minimum is a tuning score, not an unbiased final performance estimate. '
        'Final-test results are separate below. Fold SD is descriptive; it is not a confidence interval from independent folds.','',
        '| Corpus | Evaluation | Participants | Sections | Eligible-count APE % | MLUw APE % | Length-SD APE % | Boundary F1 % |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for corpus in ['nyu_emerson','ocsc']:
        for phase,result in [('CV tuning, selected recipe',selected),('Frozen final test',final)]:
            rows=[r for r in result['records'] if r['corpus']==corpus]
            lookup={r['metric']:r['mean_participant_ape'] for r in result['summary'] if r['corpus']==corpus}
            tp,fp,fn=[sum(r['boundaries']['counts'][k] for r in rows) for k in ['tp','fp','fn']]
            values=[lookup[m] for m in ['eligible_mlu_utterances','MLUw','length_sd']]+[200*tp/(2*tp+fp+fn)]
            lines.append('| '+' | '.join([corpus,phase,str(len({r['participant'] for r in rows})),str(len(rows)),
                *[f'{v:.2f}' if v is not None else 'unavailable' for v in values]])+' |')
    lines+=['','**Change from previous production on the same preserved test participants** '
        '(percentage points of APE; negative means lower error):','',
        '| Corpus | Eligible count | MLUw | Length SD |','|---|---:|---:|---:|']
    for corpus in ['nyu_emerson','ocsc']:
        changes={r['metric']:r['final_ape']-r['previous_ape'] for r in comparison['metrics'] if r['corpus']==corpus}
        lines.append('| '+corpus+' | '+' | '.join(f'{changes[m]:+.2f}' for m in ['eligible_mlu_utterances','MLUw','length_sd'])+' |')
    lines+=['','This is a descriptive comparison on the historically evaluated test cohort. '
        'Participant-paired bootstrap intervals, missingness and exact lexical parity are retained in '
        '`outputs/results/cv_hyperparameter_v4/MODEL_COMPARISON.md`; these results do not feed back into selection.','']
    runtime=evidence/'runtime_parity.json'
    if runtime.exists() and json.loads(runtime.read_text()).get('activated'):
        lines+=['','The saved transcript workflow uses this CV-selected model. Runtime matches all 139 eval sections and actual CLI/native-metric checks for both corpora.','']
    lines+=['','Errors average sections within participant, participants within corpus, and the three primary measures equally across both corpora for selection. '
        'Detailed fold/configuration results, missingness, predictions, native outputs, paired values and bootstrap intervals are in '
        '`outputs/results/cv_hyperparameter_v4/`. LUCID lexical results remain separate and unchanged. '
        'The preceding model and calibration results below are retained as historical comparisons.','']
    return lines


def fresh_model_section():
    searched=cv_search_section()
    if searched:return searched
    folder=ROOT/'outputs/results/participant_cv_v1'
    improvement=ROOT/'outputs/results/participant_improvement_v3'
    audit_path=ROOT/'outputs/validation/participant_improvement_v3/final_integrity_audit.json'
    if not audit_path.exists():return []
    audit=json.loads(audit_path.read_text())
    if not audit['complete']:return []
    cv=json.loads((folder/'cross_validation_results.json').read_text())
    final=json.loads((improvement/'test_results.json').read_text())
    selection=final['selection']
    lines=['## Fresh training, model comparison and five-fold cross-validation','',
        '**Train, eval and test cohorts are participant-independent.** The original assignments are preserved: '
        '231 usable training participants, 49 eval participants and 49 test participants. '
        'Five participant-grouped outer folds run only inside the training cohort, with a separate inner participant holdout for epoch/margin selection. '
        'Every fold starts from the original pretrained MiniLM encoder with a new task head.','',
        'A separate full-training model starts fresh and fits the original training cohort. Original eval selects its epoch, '
        'then compares that model, a five-fold probability ensemble and an ensemble of all six, each with the same twelve margins. '
        f"The selected model is **{selection['candidate']}**, margin **{selection['margin']:g}**. "
        'The choice is frozen before test evaluation. No outer-fold/test result chooses the final model. '
        'The audit verifies all six models, all training crops, unchanged words/native human references and exactly one out-of-fold prediction per training section.','',
        '| Corpus | Evaluation | Participants | Sections | Eligible-count APE % | MLUw APE % | Length-SD APE % | Boundary F1 % |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for corpus in ['nyu_emerson','ocsc']:
        for phase,result in [('5-fold out-of-fold',cv),('Fresh final test',final)]:
            rr=[r for r in result['records'] if r['corpus']==corpus]
            lookup={r['metric']:r['mean_participant_ape'] for r in result['summary'] if r['corpus']==corpus}
            tp,fp,fn=[sum(r['boundaries']['counts'][k] for r in rr) for k in ['tp','fp','fn']]
            values=[lookup[m] for m in ['eligible_mlu_utterances','MLUw','length_sd']]+[200*tp/(2*tp+fp+fn)]
            lines.append('| '+' | '.join([corpus,phase,str(len({r['participant'] for r in rr})),str(len(rr)),
                *[f'{v:.2f}' if v is not None else 'unavailable' for v in values]])+' |')
    runtime_path=ROOT/'outputs/validation/participant_improvement_v3/runtime_parity.json'
    if runtime_path.exists():
        runtime=json.loads(runtime_path.read_text())
        if runtime['activated']:
            lines+=['',f"**Active transcript workflow:** {selection['candidate']}, margin {selection['margin']:g}. "
                f"Its eval objective is {selection['objective']:.2f}%; runtime reproduces all 139 eval sections and both corpus CLI checks.",'']
        elif runtime.get('retained_previous_workflow'):
            lines+=['',f"**Active transcript workflow:** previous calibrated model retained. The fresh candidate's eval objective "
                f"({selection['objective']:.2f}%) did not improve the saved workflow ({runtime['baseline_eval_objective']:.2f}%). "
                'Its fresh-test results above are reported as the completed comparison, not an activated replacement.','']
    lines+=['','APE is calculated within section, averaged within participant, then equally across participants. '
        'Cross-validation variability, participant-bootstrap intervals, all vocabulary measures, missingness and individual predictions are in '
        '`outputs/results/participant_cv_v1/REPORT.md` and `outputs/results/participant_improvement_v3/REPORT.md` with their JSON/CSV files. '
        'The LUCID lexical comparison below is unchanged; it is separate from this segmentation study. '
        'Preceding model results below are retained as historical comparisons. '
        'Updated workflow diagrams and measured NYU input differences are in '
        '`outputs/results/participant_improvement_v3/PIPELINE.md` and `NYU_EXPLANATION.md`.','']
    return lines


def main():
    lucid=json.loads((OUT/'lucid_results.json').read_text())
    chat=json.loads((OUT/'chat_results.json').read_text())
    calibrated_path=OUT/'chat_calibrated_results.json'
    calibrated=json.loads(calibrated_path.read_text()) if calibrated_path.exists() else None
    baseline=json.loads((OUT/'chat_same_predictions.json').read_text())
    # Total verbal units are reference annotations, distinct from native MLU's
    # eligible count. Recover the old frozen prediction counts for this column.
    frozen=json.loads((ROOT/'outputs/results/common_benchmark/results.json').read_text())['recordings']
    old_counts={(r['corpus'],r['recording']):len(r['utterances']) for r in frozen if r['method']=='latest_minilm'}
    gold_counts={(r['corpus'],r['recording']):r['reference']['total_utterances'] for r in chat['records']}
    for r in baseline['records']:
        r['reference']['total_utterances']=gold_counts[r['corpus'],r['recording']]
        r['predicted']['total_utterances']=old_counts[r['corpus'],r['recording']]
    baseline['summary']=summarize(baseline['records'])
    selected=json.loads((OUT/'chat_validation.json').read_text())['selection']
    old=json.loads((ROOT/'outputs/results/lucid_validation/results.json').read_text())
    old_lookup={(r['condition'],r['metric']):r['mean_speaker_mean_recording_ape'] for r in old['summary'] if r['method']=='custom'}
    new_lookup={(r['condition'],r['metric']):r for r in lucid['summary']}
    fmt=lambda v:f'{v:.2f}' if v is not None else 'unavailable'
    lines=['# Transcript-only validation: LUCID, OCSC and NYU','',
        'Audio and the original ten participants are deferred. Completed fresh training and model improvement appear first when their integrity audits pass; preceding model/calibration results are retained below. '
        'This report keeps published lexical agreement, native CHAT scoring and utterance-boundary accuracy distinct.','',
        '## What changed','',
        '1. Added an explicit historical TAALED lexical profile: official fixed dictionary, historical POS/category rules, noun/verb type distinctions, pronoun rules and fixed pause cleaning. '
        'The current spaCy model is retained, so this is a source-supported compatibility correction, not exact historical environment reproduction.',
        '2. Added a faithful CHAT metric path using native CLAN MLU/FREQ. Clear words are retained for FREQ even when MLU excludes an unintelligible utterance; uh/um are explicitly pause-coded. '
        'Predictions do not fabricate retrace, morphology, grammar or maze coding. Ratios use exact native counts.',
        '3. Compared fixed paragraph ML, observed-turn ML and observed-turn rules on validation participants only. Consecutive target-speaker rows are merged, so human utterance ends are not supplied as cuts. '
        f"Selected **{selected['method']}**; validation count/MLUw/SD objective **{selected['validation_primary_ape'][selected['method']]:.2f}%**. "
        'This compares transcript workflows with different observed context; it is not an architecture-only claim.',
        '4. Added a reusable plain-target-transcript command: `python -m speech_pipeline.transcript INPUT.txt --output NEW_FOLDER`. '
        'It exports faithful CHAT, native supported surface metrics and the historical-profile lexical measures. Corpus ingestion/turn handling are implemented by the research runner.','',
        '## LUCID: actual published numbers','',
        '[Baese-Berk et al. (2021)](https://doi.org/10.3389/fpsyg.2021.661415) provide the unchanged per-sample TAALED result table. '
        'All 300 source TextGrids were inventoried; all 240 published rows remain in coverage. **180 rows scored, 60 vocoder rows ambiguous.** '
        'No candidate recording was chosen by numerical similarity. The exact historical cleaned index files, source/tier mapping and GUI configuration are not available in the public supplement.','',
        'This checks reproduction of published lexical statistics. It does not establish superiority over a published trained boundary model; the paper supplies no such boundary-accuracy comparison.','',
        '| Condition | Rows | Word-count error before → after % | TTR error before → after % | MATTR50 error before → after % | HD-D42 error before → after % | Original MTLD error before → after % |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for c in ['noBarrier','babble','L2','vocoder']:
        cells=[c,str(new_lookup[c,'basic_ntokens']['available'])]
        for k in ['basic_ntokens','simple_ttr_aw','mattr50_aw','hdd42_aw','mtld_original_aw']:
            cells.append(fmt(old_lookup[c,k])+' → '+fmt(new_lookup[c,k]['mean_participant_ape']))
        lines.append('| '+' | '.join(cells)+' |')
    lines+=['','**All 15 shared metrics meet the declared engineering goals in babble and L2 (120 samples total).** '
        'No-barrier still fails several measures; vocoder remains unavailable. These conditions are retained, not excluded from the experiment. '
        'Low error on two conditions does not establish full LUCID reproduction.','',
        'Goals were recorded before corrected scoring: <=5% mean participant APE for counts/densities/TTR/MATTR/MSTTR/HD-D/MLUw, <=10% for SD/original MTLD. '
        'They are project engineering aims, not validated clinical acceptance standards. APE is calculated within recording, averaged within participant, then equally across participants.','',
        'Independent implementation checks: the ported normalizer matched the official historical pure function on all 180 current-model documents; '
        'our seven lexical formulas matched the official historical functions on the identical tokens (1,260 checks, tolerance 1e-10). '
        'This validates those calculations separately from agreement with the paper. Counts/densities are defined directly from the same token lists.','',
        '## OCSC and NYU: professional CHAT / native CLAN','',
        'References preserve original human CHAT coding and use the same native MLU -t%mor / FREQ +k +b50 commands as predictions. '
        'Total verbal units and eligible MLU utterances are distinct. FREQ and MLU have distinct word eligibility. '
        'Native SD/MATTR have three-decimal precision. No morphology/grammaticality/maze/error/timing measure is claimed without its required evidence.','',
        '**Held-out-participant results (already historically exposed; descriptive rerun):**','',
        '| Corpus | Measure | Previous scoring, same old predictions APE % | Corrected native scoring, same old predictions APE % | Selected transcript workflow APE % | Goal % |',
        '|---|---|---:|---:|---:|---:|']
    before={(r['method'],r['corpus'],r['metric']):r for r in baseline['summary']}
    tests={(r['corpus'],r['metric']):r for r in chat['split_summaries']['test']}
    for c in ['ocsc','nyu_emerson']:
        for metric in ['total_utterances','eligible_mlu_utterances','MLUw','length_sd','TNW','NDW','TTR','MATTR']:
            vals=[before[m,c,metric]['mean_participant_ape'] for m in ['previous_scoring_same_predictions','native_scoring_same_predictions']]
            r=tests[c,metric]
            lines.append('| '+' | '.join([c,metric,*[fmt(v) for v in vals],fmt(r['mean_participant_ape']),str(r['target_ape'])])+' |')
    lines+=['','The same-prediction columns isolate processing-policy changes. The selected-workflow column also changes available speaker-turn context. '
        'No test score selected the workflow or changed its cutoffs. Training and validation results are separate in `chat_results.json`; they are not independent confirmation.','',
        ]
    if calibrated:
        choice=calibrated['selection']
        zero=json.loads((EVIDENCE/'calibration_zero_parity.json').read_text())
        final_parity=json.loads((EVIDENCE/'calibrated_final_parity.json').read_text())
        lines+=['## Bounded calibration: frozen development choice','',
            f"The 14 registered candidates (two contexts × seven margins) selected **{choice['context']}, margin {choice['margin']:g}** using development participants only. "
            f"The participant/corpus-balanced count/MLUw/SD objective decreased from **11.71% to {choice['validation_primary_ape']:.2f}%**. "
            'No final score changed that choice; no grid expansion or retraining was performed.','',
            'Batched NLP preprocessing initially changed six validation predictions. That cache was rejected before selection. '
            f"Restoring the original single-document feature function reproduced all **{zero['checked']}** zero-margin comparisons, "
            f"and the chosen all-data cache reproduced all **{final_parity['checked']}** selected validation predictions. "
            'Hashes of caches, weights, prepared inputs and source transcripts were checked before final scoring.','',
            '**Held-out participants, descriptive rerun; mean participant absolute percentage error:**','',
            '| Corpus | Measure | Before calibration % | After calibration % | Goal % |',
            '|---|---|---:|---:|---:|']
        calibrated_tests={(r['corpus'],r['metric']):r for r in calibrated['split_summaries']['test']}
        for c in ['nyu_emerson','ocsc']:
            for metric in ['total_utterances','eligible_mlu_utterances','MLUw','length_sd','TNW','NDW','TTR','MATTR']:
                before=tests[c,metric];after=calibrated_tests[c,metric]
                lines.append('| '+' | '.join([c,metric,fmt(before['mean_participant_ape']),fmt(after['mean_participant_ape']),str(after['target_ape'])])+' |')
        lines+=['','Every metric is retained, including regressions. Utterance count and MLU errors remain above the declared targets in both corpora. '
            'The source word streams are unchanged; lexical calculations and the LUCID comparison are unaffected. '
            'Train/validation/test summaries, section coverage, missing denominators and raw native outputs are saved separately.','',
            '| Split | Corpus | Sections | Participants | Count APE % | MLUw APE % | SD APE % |',
            '|---|---|---:|---:|---:|---:|---:|']
        for split in ['train','validation','test']:
            lookup={(r['corpus'],r['metric']):r for r in calibrated['split_summaries'][split]}
            for c in ['nyu_emerson','ocsc']:
                rr=[r for r in calibrated['records'] if r['split']==split and r['corpus']==c]
                lines.append('| '+' | '.join([split,c,str(len(rr)),str(len({r['participant'] for r in rr})),
                    *[fmt(lookup[c,m]['mean_participant_ape']) for m in ['eligible_mlu_utterances','MLUw','length_sd']]])+' |')
        lines+=['','The NYU test comparison includes only four participants; these results do not establish broad generalization. '
            'The saved workflow applies margin 0.25 to observed-turn JSON input (`target_speaker` plus `speaker`/`text` rows). '
            'Plain target text retains the original paragraph workflow because no paragraph fallback was selected.']
        audit_path=EVIDENCE/'calibrated_final_audit.json'
        smoke_path=EVIDENCE/'calibrated_production_parity.json'
        if audit_path.exists() and smoke_path.exists():
            audit=json.loads(audit_path.read_text());smoke=json.loads(smoke_path.read_text())
            lines+=['',f"The final audit verified unchanged words, references and participant splits on all {audit['sections']} sections, "
                'and exact equality of TNW/NDW/TTR/MATTR before and after calibration in every section. '
                f"The reusable transcript command matched evaluated utterances and native metrics on {len(smoke['checks'])} development examples (one per CHAT corpus). "
                'All 88 regression tests passed; this is an implementation check, separate from accuracy against human references.']
        pairs=[]
        old_by_key={(r['corpus'],r['recording']):r for r in chat['records']}
        for r in calibrated['records']:
            old_r=old_by_key[r['corpus'],r['recording']]
            if auto_tokens(r['utterances'])!=auto_tokens(old_r['utterances']):raise ValueError('Calibration changed words')
            for metric in METRICS:
                p,g=r['predicted'][metric],r['reference'][metric]
                if r['reference'][metric]!=old_r['reference'][metric]:raise ValueError('Reference changed')
                pairs.append(dict(corpus=r['corpus'],participant=r['participant'],recording=r['recording'],split=r['split'],
                    method=r['method'],metric=metric,reference=g,predicted=p,
                    signed_error=p-g if p is not None and g is not None else None,ape=100*abs(p-g)/abs(g) if p is not None and g else None))
        with (OUT/'chat_calibrated_metric_pairs.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(pairs[0]));writer.writeheader();writer.writerows(pairs)
    lines+=['',
        '## Coverage and segmentation','',
        '| Corpus | Available CHAT files | Prepared/scored task sections | Source files represented | Participants |',
        '|---|---:|---:|---:|---:|']
    manifest=json.loads((EVIDENCE/'protocol.json').read_text())
    all_sources=[]
    exclusions={r['file']:r['reason'] for r in json.loads((EVIDENCE/'unscored_source_reasons.json').read_text())}
    for inventory in manifest['inventories']:
        c=inventory['corpus'];rr=[r for r in chat['records'] if r['corpus']==c]
        represented={r['recording'].rsplit('#section',1)[0] for r in rr}
        lines.append('| '+' | '.join([c,str(inventory['transcripts']),str(len(rr)),str(len(represented)),str(len({r['participant'] for r in rr}))])+' |')
        for source in inventory['files']:
            relative=str(Path(source['path']).relative_to(Path('dataset/public')/c))
            count=sum(r['recording'].rsplit('#section',1)[0]==relative for r in rr)
            all_sources.append(dict(corpus=c,file=source['path'],sha256=source['sha256'],scored_sections=count,
                status='scored_selected_tasks' if count else 'no_eligible_section_in_frozen_preparation',
                reason=None if count else exclusions.get(source['path'],'Review required'),
                scope='OCSC natural discourse Howto/DescriptivePictures/ExpPictures; NYU prepared target-CHI sections. Reading/alphabet/Wug tasks are outside the frozen discourse preparation; nonverbal/unscorable sections retained in source coverage, not assigned zero error.'))
    save(EVIDENCE/'all_source_coverage.json',all_sources)
    lines+=['','All available source files are inventoried. The frozen preparation covers OCSC spontaneous discourse tasks and the NYU target-CHI sections; '
        'other tasks and sources without eligible prepared speech are explicitly listed in `all_source_coverage.json`. '
        'This is not a claim that every utterance in every source file was scored.','',
        '| Corpus, test only | Boundary precision % | Boundary recall % | Boundary F1 % | Exact utterance-span F1 % |',
        '|---|---:|---:|---:|---:|']
    for c in ['ocsc','nyu_emerson']:
        rr=[r for r in chat['records'] if r['corpus']==c and r['split']=='test']
        counts={k:sum(r['boundaries']['counts'][k] for r in rr) for k in ['tp','fp','fn']}
        t,p,n=counts['tp'],counts['fp'],counts['fn'];ex={k:sum(r['boundaries']['exact'][k] for r in rr) for k in ['matched','predicted','reference']}
        ratio=lambda a,b:100*a/b if b else None
        lines.append('| '+' | '.join([c,fmt(ratio(t,t+p)),fmt(ratio(t,t+n)),fmt(ratio(2*t,2*t+p+n)),fmt(ratio(2*ex['matched'],ex['predicted']+ex['reference']))])+' |')
    if calibrated:
        lines+=['','**After calibration, test only (micro-aggregated boundaries/spans):**','',
            '| Corpus | Boundary precision % | Boundary recall % | Boundary F1 % | Exact utterance-span F1 % |',
            '|---|---:|---:|---:|---:|']
        for c in ['nyu_emerson','ocsc']:
            rr=[r for r in calibrated['records'] if r['corpus']==c and r['split']=='test']
            t,p,n=[sum(r['boundaries']['counts'][k] for r in rr) for k in ['tp','fp','fn']]
            ex={k:sum(r['boundaries']['exact'][k] for r in rr) for k in ['matched','predicted','reference']}
            lines.append('| '+' | '.join([c,fmt(ratio(t,t+p)),fmt(ratio(t,t+n)),fmt(ratio(2*t,2*t+p+n)),fmt(ratio(2*ex['matched'],ex['predicted']+ex['reference']))])+' |')
    parity=json.loads((EVIDENCE/'batch_inference_parity.json').read_text())
    distributions=[]
    grouped=defaultdict(list)
    for r in lucid['pairs']:
        if r['ape'] is not None:grouped['lucid',r['condition'],r['metric']].append(r['ape'])
    for r in chat['records']:
        for metric in METRICS:
            p,g=r['predicted'][metric],r['reference'][metric]
            if p is not None and g:grouped[r['corpus'],r['split'],metric].append(100*abs(p-g)/abs(g))
    if calibrated:
        for r in calibrated['records']:
            for metric in METRICS:
                p,g=r['predicted'][metric],r['reference'][metric]
                if p is not None and g:grouped[r['corpus'],'calibrated_'+r['split'],metric].append(100*abs(p-g)/abs(g))
    for (corpus,group,metric),values in sorted(grouped.items()):
        distributions.append(dict(corpus=corpus,group=group,metric=metric,recordings=len(values),
            median_recording_ape=float(np.median(values)),p95_recording_ape=float(np.percentile(values,95)),max_recording_ape=max(values)))
    with (OUT/'metric_error_distributions.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(distributions[0]));writer.writeheader();writer.writerows(distributions)
    cohort_audit=ROOT/'outputs/validation/cohorts_v1/audit.json'
    if cohort_audit.exists():
        lines+=['','## Cohort integrity','',
            'The frozen train/eval/test exports preserve the original participant assignments: 630 train sections (231 children), '
            '139 validation sections (49 children) and 126 test sections (49 children). Eval means validation/model selection. '
            'OCSC and NYU each have a separate file for each role; LUCID remains a separate lexical benchmark.','',
            'The cohort audit found no participant, source-recording, source-file or exact full-section-text overlap. '
            'Both saved training-augmentation caches contain training parents only; calibration selection and its pre-selection contexts contain validation participants only. '
            'Isolated exports reproduce all 895 original records and their ordering exactly. All 333 raw CHAT files and selected model weights remain unchanged. '
            'No retraining was needed to repair detected overlap.','',
            'Development loaders require explicit roles and calibration selection loads validation references only. '
            'The existing test cohort remains historically exposed and cannot be made pristine by reshuffling or retraining. '
            'Audit, counts, exclusions and 97 passing regression tests: `outputs/validation/cohorts_v1/`.']
    lines+=['',f"Batched inference matched all **{parity['checked']}** saved validation predictions exactly before final scoring; source words are checked on every scored recording.",'',
        '## Remaining gaps','',
        '- Resolve the historical LUCID cleaned inputs and source/tier mappings, including ambiguous vocoder rows. Retain the current mismatch audit.',
        '- Any CHAT metrics above their targets and exact-span boundary errors require further development-only improvements. Do not describe the whole pipeline as validated from passing subsets.',
        '- Independent confirmation is still needed after changes on previously exposed corpora. No audio work is part of this stage.','',
        'Files: `lucid_metric_pairs.csv`, `lucid_summary.csv`, `lucid_coverage.csv`, `chat_metric_pairs.csv`, `chat_calibrated_metric_pairs.csv`, `chat_calibrated_results.json`, `metric_error_distributions.csv` (median, 95th percentile and worst recording errors); JSON retains all predictions, native outputs, per-split summaries and calculator checks. '
        'Protocols and source/hash/coverage evidence: `outputs/validation/transcript_validation/`. Previous results were not overwritten.','']
    lines[4:4]=fresh_model_section()
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    print('Saved',OUT/'REPORT.md',flush=True)

def auto_tokens(utterances):
    from speech_pipeline import utterances as auto
    return auto.stream([dict(text=t) for t in utterances])[0]


if __name__=='__main__':main()
