"""Report-only comparison after the model and final test results are frozen."""
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/speech_pipeline_matplotlib')
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from speech_pipeline import ROOT
from speech_pipeline.pipeline import digest, save
from speech_pipeline.research import participant_improvement as study
from speech_pipeline.research.transcript_chat import summarize


def run():
    audit=json.loads((study.EVIDENCE/'final_integrity_audit.json').read_text())
    if not audit['complete']:raise ValueError('Audit final results before drawing comparisons')
    current=json.loads((study.OUT/'test_results.json').read_text())
    if digest(study.OUT/'test_results.json')!=audit['test_results_sha256']:raise ValueError('Audited test results changed')
    previous=json.loads((ROOT/'outputs/results/transcript_validation/chat_calibrated_results.json').read_text())
    old_rows=[r for r in previous['records'] if r['split']=='test']
    old_keys={(r['corpus'],r['recording']) for r in old_rows};new_keys={(r['corpus'],r['recording']) for r in current['records']}
    if old_keys!=new_keys:raise ValueError('Before/after test cohorts differ')
    previous_by_key={(r['corpus'],r['recording']):r for r in old_rows}
    for row in current['records']:
        before=previous_by_key[row['corpus'],row['recording']]
        if row['reference']!=before['reference']:raise ValueError('Before/after human reference values differ')
        if any(row['predicted'][metric]!=before['predicted'][metric] for metric in ['TNW','NDW','TTR','MATTR']):
            raise ValueError('Boundary-only model change unexpectedly altered vocabulary values')
    save(study.EVIDENCE/'vocabulary_parity.json',dict(test_sections=len(current['records']),same_test_sections=True,
        all_reference_values_unchanged=True,all_four_vocabulary_values_unchanged=True,
        previous_results_sha256=digest(ROOT/'outputs/results/transcript_validation/chat_calibrated_results.json'),
        new_results_sha256=digest(study.OUT/'test_results.json')))
    old=summarize(old_rows);new=current['summary']
    lookup=lambda summary,c,m:next(r['mean_participant_ape'] for r in summary if r['corpus']==c and r['metric']==m)
    metrics=['eligible_mlu_utterances','MLUw','length_sd'];labels=['Eligible\nutterance count','MLU in words','Utterance-length\nSD']
    maximum=max(lookup(summary,c,m) for summary in [old,new] for c in ['nyu_emerson','ocsc'] for m in metrics)
    comparisons=[]
    fig,axes=plt.subplots(1,2,figsize=(10,4.8),sharey=True,layout='constrained')
    for ax,corpus,title in zip(axes,['nyu_emerson','ocsc'],['NYU–Emerson · 4 test participants','OCSC · 45 test participants']):
        a=np.array([lookup(old,corpus,m) for m in metrics]);b=np.array([lookup(new,corpus,m) for m in metrics]);x=np.arange(3)
        bars1=ax.bar(x-.19,a,.35,label='Previous calibrated model',color='#94a3b8')
        bars2=ax.bar(x+.19,b,.35,label='New eval-selected candidate',color='#117a86')
        ax.bar_label(bars1,fmt='%.1f',padding=3,fontsize=9);ax.bar_label(bars2,fmt='%.1f',padding=3,fontsize=9)
        ax.set_xticks(x,labels);ax.set_title(title,fontsize=12,pad=12);ax.set_ylim(0,max(40,float(maximum*1.25)))
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        for m,before,after in zip(metrics,a,b):comparisons.append(dict(corpus=corpus,metric=m,previous_test_ape=float(before),new_test_ape=float(after),change_percentage_points=float(after-before)))
    axes[0].set_ylabel('Mean participant absolute percentage error (%)')
    handles,legendlabels=axes[0].get_legend_handles_labels();fig.legend(handles,legendlabels,loc='outside lower center',ncol=2,frameon=False)
    fig.suptitle('Same participant-independent test cohorts · lower error is better',fontsize=13)
    fig.savefig(study.OUT/'test_comparison.png',dpi=200);fig.savefig(study.OUT/'test_comparison.svg');plt.close(fig)
    save(study.OUT/'test_comparison.json',dict(comparisons=comparisons,selected_candidate=current['selection']['candidate'],
        scope='Report-only comparison on retained identical test sections. Candidate and deployment chosen on eval only. No test-based tuning.'))
    print(json.dumps(comparisons,indent=2))


if __name__=='__main__':run()
