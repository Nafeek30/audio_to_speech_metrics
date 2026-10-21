"""Prepare publication data from the current saved recording comparisons."""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
from speech_pipeline.research.human_interrater import repeated_ba, write_csv, METRIC_LABELS

SOURCE = ROOT / 'outputs/results/human_comparison'
OUT = ROOT / 'outputs/results/paper'
rows = list(csv.DictReader((SOURCE/'metric_pairs.csv').open(encoding='utf-8-sig')))
rows = [r for r in rows if r['method']=='Revised']
complete_rows = list(csv.DictReader((SOURCE/'participant_sample_pairs.csv').open(encoding='utf-8-sig')))
complete_rows = [r for r in complete_rows if r['method']=='Revised']
metrics = ['basic_ntokens','basic_ntypes','basic_ncontent_tokens','basic_ncontent_types',
           'basic_nfunction_tokens','basic_nfunction_types','simple_ttr_aw','mattr50_aw',
           'total_utterances','mlu_words','mlu_words_sd']
labels = ['All words','Different words','Content words','Different content words',
          'Function words','Different function words','TTR','MATTR-50',
          'Utterance count','Mean utterance length (MLUw)','Utterance length standard deviation']
def summary(rs):
    groups = defaultdict(list)
    for r in rs: groups[r['patient_id']].append(r)
    result = {}
    for key, field in [('pipeline_mean','pipeline_value'),('reference_mean','reference_value'),
                       ('mape','absolute_percent_error')]:
        result[key] = statistics.mean(statistics.mean(float(r[field]) for r in v if r[field]) for v in groups.values())
    # Preserve mean summaries for the audit; publication errors use medians.
    for key, field in [('pipeline_median','pipeline_value'),('reference_median','reference_value'),
                       ('mdape','absolute_percent_error')]:
        result[key] = statistics.median(statistics.mean(float(r[field]) for r in v if r[field]) for v in groups.values())
    return result

tables = []
pairs = []
for metric,label in zip(metrics,labels):
    software_metric = 'mlu_nutterances' if metric=='total_utterances' else metric
    software_reference = 'Corrected CLAN' if metric in metrics[8:] else 'Corrected TAALED'
    sr = [r for r in rows if r['reference']==software_reference and r['metric']==software_metric]
    mr = [r for r in rows if r['reference']=='Nafee' and r['metric']==metric]
    cr = [r for r in complete_rows if r['metric']==metric]
    assert len(sr)==37 and len(mr)==14
    assert len(cr)==10
    tables.append(dict(metric=metric,label=label,software_metric=software_metric,
                       software_reference=software_reference,software=summary(sr),manual=summary(cr),
                       manual_recording_segments=summary(mr)))
    for comp,selected,score_metric in [('uncorrected_custom_vs_corrected_'+software_reference.split()[-1],sr,software_metric),
                                       ('uncorrected_custom_vs_human_labels',mr,metric)]:
        for r in selected:
            x,y=float(r['pipeline_value']),float(r['reference_value'])
            pairs.append(dict(comparison=comp,patient_id=r['patient_id'],source_file=r['source_file'],
                              metric=score_metric,custom_value=x,reference_value=y,difference=x-y))

br = list(csv.DictReader((SOURCE/'boundary_recordings.csv').open(encoding='utf-8-sig')))
br = [r for r in br if r['method']=='Revised' and r['reference'] in ['Jade','Yamen']]
assert len(br)==4
boundary = []
def bscore(tp,tn,fn,fp):
    n=tp+tn+fn+fp; po=(tp+tn)/n
    pe=((tp+fn)*(tp+fp)+(tn+fp)*(tn+fn))/n**2
    return dict(evaluable_gaps=n,boundary_percent_agreement=100*po,cohens_kappa=(po-pe)/(1-pe),
                boundary_f1=2*tp/(2*tp+fn+fp),both_boundary=tp,both_no_boundary=tn,
                primary_only_boundary=fp,secondary_only_boundary=fn)
for r in br:
    boundary.append(dict(patient_id=r['patient_id'],source_file=r['source_file'],secondary_rater=r['reference'],
                         **bscore(*(int(r[k]) for k in ['tp','tn','fn','fp'])),
                         reference_gap_coverage=float(r['reference_gap_coverage']),
                         primary_gap_coverage=float(r['pipeline_gap_coverage']),
                         filler_policy='Retained in utterance text and boundary comparisons'))
boundary.append(dict(patient_id='',source_file='',secondary_rater='Jade and Yamen',
                     **bscore(*(sum(int(r[k]) for r in br) for k in ['tp','tn','fn','fp'])),
                     reference_gap_coverage=sum(x['evaluable_gaps'] for x in boundary)/sum(int(r['reference_possible_gaps']) for r in br),
                     primary_gap_coverage=sum(x['evaluable_gaps'] for x in boundary)/sum(int(r['pipeline_possible_gaps']) for r in br),
                     filler_policy='Retained in utterance text and boundary comparisons'))

data=dict(tables=tables,boundaries=boundary,
          sources={name:hashlib.sha256((SOURCE/name).read_bytes()).hexdigest() for name in ['metric_pairs.csv','boundary_recordings.csv']})
(OUT/'publication_data.json').write_text(json.dumps(data,indent=2))
write_csv(OUT/'publication_tables.csv',[dict(metric=r['software_metric'],label=r['label'],comparison='Corrected software',**r['software']) for r in tables]+
          [dict(metric=r['metric'],label=r['label'],comparison='Nafee complete participant sample',**r['manual']) for r in tables]+
          [dict(metric=r['metric'],label=r['label'],comparison='Nafee recording segments',**r['manual_recording_segments']) for r in tables])
write_csv(OUT/'speech_recording_pairs.csv',pairs)
write_csv(OUT/'interrater_results.csv',boundary)

groups=defaultdict(list)
for r in pairs:groups[r['comparison'],r['metric'],r['patient_id']].append(r)
participant_pairs=[]
for (comp,metric,pid),rs in sorted(groups.items()):
    x=statistics.mean(r['custom_value'] for r in rs);y=statistics.mean(r['reference_value'] for r in rs)
    participant_pairs.append(dict(comparison=comp,metric=metric,patient_id=pid,n_recordings=len(rs),custom_value=x,reference_value=y,difference=x-y,
                                  within_patient_mean_recording_APE=statistics.mean(100*abs(r['difference'])/abs(r['reference_value']) for r in rs if r['reference_value'])))
write_csv(OUT/'speech_participant_pairs.csv',participant_pairs)
groups=defaultdict(list)
for r in pairs:groups[r['comparison'],r['metric']].append(r)
results=[]
for (comp,metric),rs in sorted(groups.items()):
    raw=repeated_ba(rs); relative=repeated_ba(rs,log_scale=True)
    pr=[r for r in participant_pairs if (r['comparison'],r['metric'])==(comp,metric)]
    result=dict(comparison=comp,metric=metric,metric_label=METRIC_LABELS[metric],n_recordings=len(rs),n_participants=len(pr),
                custom_mean=statistics.mean(r['custom_value'] for r in rs),reference_mean=statistics.mean(r['reference_value'] for r in rs),
                **raw,participant_balanced_recording_MAPE=statistics.mean(r['within_patient_mean_recording_APE'] for r in pr),
                participant_balanced_mean_difference=statistics.mean(r['difference'] for r in pr),
                analysis_unit='Recording pair, with repeated recordings clustered within participant',
                loa_method='Repeated-measures Bland-Altman, random-intercept REML',ci_method='Participant-cluster percentile bootstrap',
                bootstrap_replicates=10000,bootstrap_seed=20260924)
    result.update({'log_'+k:v for k,v in relative.items()})
    import math
    for k in ['bias','lower_loa','upper_loa']:
        for suffix in ['', '_ci_lower','_ci_upper']:
            value=relative[k+suffix];result['ratio_'+k+suffix]=None if value is None else math.exp(value)
    results.append(result)
    print(comp,metric,round(result['participant_balanced_recording_MAPE'],4),flush=True)
write_csv(OUT/'speech_agreement_results.csv',results)
print('Pooled boundary results:',boundary[-1],flush=True)
