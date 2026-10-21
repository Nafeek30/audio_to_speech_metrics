"""Slide-ready figures from saved results; no model fitting or scoring."""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR'] = str(OUT / '.matplotlib')
import json
import hashlib
import statistics
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PATHS = {
    'test': ROOT / 'outputs/results/transcript_validation/chat_calibrated_results.json',
    'lucid': ROOT / 'outputs/results/transcript_validation/lucid_results.json',
    'cv': ROOT / 'outputs/results/cv_hyperparameter_v4/stopped_search_summary.json',
}
DATA = {k: json.loads(p.read_text()) for k,p in PATHS.items()}
plt.rcParams.update({'font.family':'Arial', 'font.size':18, 'axes.titlesize':25,
                     'axes.labelsize':18, 'xtick.labelsize':16, 'ytick.labelsize':19,
                     'savefig.facecolor':'white', 'figure.facecolor':'white'})
BLUE, ORANGE = '#156082', '#E97132'
NAMES = ['Ohio Child Speech Corpus', 'New York University and Emerson College corpus']
CORPORA = ['ocsc', 'nyu_emerson']
records = {}

def style(ax, xmax):
    ax.set_axisbelow(True)
    ax.grid(axis='x', color='#E2E2E2', linewidth=.8)
    for side in ['top','right','left']: ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color('#B0B0B0')
    ax.tick_params(axis='y', length=0, pad=13)
    ax.set_xlim(0, xmax)
    ax.set_xlabel('Mean participant absolute percentage error (%) · Lower is better', labelpad=15)

def heading(fig,title,subtitle):
    fig.text(.055,.895,subtitle,ha='left',va='top',fontsize=17,color='#444444')

def save(fig,name,metadata):
    fig.savefig(OUT/name,dpi=200)
    plt.close(fig)
    records[name]=metadata

def child_plot(name,title,metrics):
    fig,ax=plt.subplots(figsize=(16,9))
    fig.subplots_adjust(left=.28,right=.94,bottom=.25,top=.79)
    values=[]; y=np.arange(len(metrics)); height=.30
    for j,(corpus,label,color) in enumerate(zip(CORPORA,NAMES,[BLUE,ORANGE])):
        v=[next(r['mean_participant_ape'] for r in DATA['test']['split_summaries']['test']
                if r['corpus']==corpus and r['metric']==m) for _,m in metrics]
        bars=ax.barh(y+(j-.5)*.34,v,height=height,color=color,label=label)
        ax.bar_label(bars,labels=[f'{x:.2f}%' for x in v],padding=7,fontsize=18)
        values.append(dict(corpus=corpus,values=dict(zip([m for _,m in metrics],v))))
    ax.set_yticks(y,[label for label,_ in metrics]);ax.invert_yaxis()
    style(ax,max(max(v['values'].values()) for v in values)*1.2)
    if name == '02_child_utterance_errors.png':
        heading(fig,title,'Ohio Child Speech Corpus: 45 participants\nNew York University and Emerson College corpus: 4 participants')
    else:
        heading(fig,title,'Retained pipeline · Test cohorts: 45 children / 111 sections and 4 children / 15 sections')
    fig.legend(*ax.get_legend_handles_labels(),loc='lower left',bbox_to_anchor=(.275,.035),
               frameon=False,fontsize=17,ncol=1)
    save(fig,name,dict(evaluation='Existing retained-model test',values=values))

child_plot('01_child_vocabulary_errors.png','Vocabulary measures: test results',[
    ('Total words','TNW'),('Different words','NDW'),('TTR','TTR'),('MATTR-50','MATTR')])
child_plot('02_child_utterance_errors.png','Utterance measures: test results',[
    ('Total utterance count','total_utterances'),('MLUw','MLUw'),
    ('Utterance-length SD','length_sd')])

metrics=[('Total words','basic_ntokens'),('Different words','basic_ntypes'),
         ('TTR','simple_ttr_aw'),('MATTR-50','mattr50_aw')]
values={};counts={}
for label,metric in metrics:
    people=defaultdict(list)
    for row in DATA['lucid']['pairs']:
        if row['metric']==metric and row.get('ape') is not None and row.get('status')=='scored':
            people[row['speaker']].append(row['ape'])
    values[metric]=statistics.mean(statistics.mean(v) for v in people.values())
    counts[metric]={'speakers':len(people),'samples':sum(map(len,people.values()))}
    assert counts[metric]=={'speakers':40,'samples':180}
fig,ax=plt.subplots(figsize=(16,9));fig.subplots_adjust(left=.24,right=.93,bottom=.19,top=.75)
v=list(values.values());bars=ax.barh(np.arange(4),v,height=.5,color=BLUE)
ax.bar_label(bars,labels=[f'{x:.2f}%' for x in v],padding=9,fontsize=21)
ax.set_yticks(np.arange(4),[x[0] for x in metrics]);ax.invert_yaxis();style(ax,max(v)*1.25)
heading(fig,'LUCID vocabulary measures: combined results',
        'London UCL Clear Speech in Interaction · 180 scored samples / 40 speakers')
save(fig,'03_lucid_vocabulary_errors.png',dict(evaluation='Provisional lexical-reference comparison',
    aggregation='Mean APE within speaker over all available scored samples, then equal mean over 40 speakers; all three scored conditions included',
    values=values,counts=counts))

configs=['lr2_count1','lr2_count025'];labels=['Baseline training recipe','Lower count-loss weight']
metric_keys=['eligible_mlu_utterances','MLUw','length_sd']
metric_names=['Eligible utterance count','Average length in words','Length standard deviation']
fig,axes=plt.subplots(2,1,figsize=(16,9));fig.subplots_adjust(left=.055,right=.98,bottom=.23,top=.78,hspace=.65)
cv_values=[]
for ax,corpus,title in zip(axes,CORPORA,NAMES):
    x=np.arange(3);row={}
    for j,(cfg,label,color) in enumerate(zip(configs,labels,[BLUE,ORANGE])):
        candidate=next(c for c in DATA['cv']['completed_candidates'] if c['configuration']==cfg)
        v=[next(r['mean_participant_ape'] for r in candidate['summary']
                if r['corpus']==corpus and r['metric']==metric) for metric in metric_keys]
        bars=ax.bar(x+(j-.5)*.34,v,width=.3,color=color,label=label)
        ax.bar_label(bars,labels=[f'{n:.2f}%' for n in v],padding=4,fontsize=16)
        row[cfg]=dict(zip(metric_keys,v))
    ax.set_xticks(x,metric_names,fontsize=17);ax.set_ylim(0,25)
    ax.set_title(title,loc='left',fontsize=19,pad=10)
    ax.set_ylabel('Error (%)',fontsize=16);ax.set_axisbelow(True);ax.grid(axis='y',color='#E2E2E2')
    for side in ['top','right']:ax.spines[side].set_visible(False)
    cv_values.append({'corpus':corpus,'values':row})
heading(fig,'Cross-validation: baseline and lower count-loss weight',
        'Five-fold tuning on original training participants · Separate from test results · Lower is better')
fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,.045),
           frameon=False,fontsize=18,ncol=2)
save(fig,'04_cross_validation_comparison.png',dict(evaluation='Five-fold CV tuning, not a new final test',values=cv_values))

manifest={'figures':records,'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in PATHS.values()},
          'format':'PNG, white background, 3200 x 1800 pixels, 200 DPI',
          'notes':'No HD-D42 plotted. No source results, cohorts or model weights modified. LUCID aggregation is a new descriptive summary of saved per-sample errors, not a new model run.'}
(OUT/'plot_data.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'files':list(records),'lucid_combined_errors':values},indent=2))
