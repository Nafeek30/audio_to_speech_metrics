"""Individual-result analysis and leakage-controlled training-data preparation."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import statistics
from tempfile import TemporaryDirectory

import numpy as np
from openpyxl import load_workbook

from speech_pipeline import ROOT, utterances as auto
from speech_pipeline.corpora import CORPORA, calculator_checks, EVIDENCE, OUT
from speech_pipeline.pipeline import save, digest

SOURCE=ROOT/'outputs/results/human_comparison'
METRICS=[('basic_ntokens','Normalized words'),('basic_ntypes','Normalized word types'),
 ('basic_ncontent_tokens','Content words'),('basic_ncontent_types','Content types'),
 ('basic_nfunction_tokens','Function words'),('basic_nfunction_types','Function types'),
 ('simple_ttr_aw','TTR (lemma/POS)'),('mattr50_aw','MATTR-50 (lemma/POS)'),
 ('total_utterances','All utterances'),('mlu_words','MLUw'),('mlu_words_sd','Length SD')]


def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def talkbank_analysis():
    """Compare rules, words-only and joint paragraph/rule models on fixed splits."""
    from speech_pipeline.corpora import DATA
    from speech_pipeline.metrics import clinical_from_predictions
    data_path=EVIDENCE/'talkbank_training.jsonl'
    records=[json.loads(s) for s in data_path.read_text().splitlines()]
    models={name:json.loads((OUT/f'{stem}.json').read_text()) for name,stem in
            [('words','talkbank_model'),('joint','hybrid_model')] if (OUT/f'{stem}.json').exists()}
    index={(r['corpus'],r['participant'],r['recording']):r for r in records}
    local_path=EVIDENCE/'boundary_dataset.jsonl'
    if any(m.get('additional_evaluation') for m in models.values()):
        index.update({(r['corpus'],r['participant'],r['recording']):r for r in
                      map(json.loads,local_path.read_text().splitlines())})
    def segmented(source, boundaries):
        words=source['surface_words'];ends=[0,*boundaries,len(words)]
        return [dict(text=' '.join(words[a:b])) for a,b in zip(ends,ends[1:]) if b>a]
    results=[]
    for name,model in models.items():
        if digest(data_path)!=model['input_sha256']:raise ValueError('Training data changed')
        external=model.get('additional_evaluation')
        if external and digest(local_path)!=external['input_sha256']:raise ValueError('Target data changed')
        for evaluation in [model['held_out_test']]+([external] if external else []):
            for prediction in evaluation['recordings']:
                source=index[prediction['corpus'],prediction['participant'],prediction['recording']]
                base=segmented(source,[i+1 for i,v in enumerate(source['baseline'][:-1]) if v])
                learned=segmented(source,prediction['predicted_boundaries'])
                if source['corpus'] in CORPORA:
                    relative=source['recording'].rsplit('#section',1)[0]
                    path=DATA/source['corpus']/relative
                    if digest(path)!=source['source_sha256']:raise ValueError(f'Source changed: {path}')
                    gold=segmented(source,source['gold_boundaries'])
                    truth=clinical_from_predictions(gold)['values']
                    scores={method:clinical_from_predictions(utterances)['values'] for method,utterances in [('rules',base),('transformer',learned)]}
                else:
                    truth={k:source['reference_mlu'][k] for k in ('total_utterances','mlu_words','mlu_words_sd')}
                    scores={method:auto.word_mlu(utterances) for method,utterances in [('rules',base),('transformer',learned)]}
                for metric,reference in truth.items():
                    if reference is None:continue
                    a=scores['rules'][metric];b=scores['transformer'][metric]
                    results.append(dict(model=name,corpus=source['corpus'],participant=source['participant'],recording=source['recording'],
                        gem=source.get('gem'),metric=metric,reference=reference,rules=a,transformer=b,
                        rules_absolute_error=abs(a-reference) if a is not None else None,
                        transformer_absolute_error=abs(b-reference) if b is not None else None,
                        rules_ape=100*abs(a-reference)/abs(reference) if reference and a is not None else None,
                        transformer_ape=100*abs(b-reference)/abs(reference) if reference and b is not None else None))
    write_csv(OUT/'talkbank_metric_errors.csv',results)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,5),dpi=180)
    groups={name:{**m['held_out_test']['by_corpus'],**m.get('additional_evaluation',{}).get('by_corpus',{})} for name,m in models.items()}
    labels=list(groups['words']);x=np.arange(len(labels))
    methods=[('rules','#256da8'),('words','#d87526')]+([('joint','#348652')] if 'joint' in models else [])
    for i,(method,color) in enumerate(methods):
        values=[100*groups['words' if method=='rules' else method][c]['baseline' if method=='rules' else 'transformer']['f1'] for c in labels]
        axes[0].bar(x+(i-(len(methods)-1)/2)*.24,values,width=.24,label=method,color=color)
        points=[r for r in results if r['metric'] in ('MLUw','mlu_words') and r['model']==('words' if method=='rules' else method)]
        field='rules' if method=='rules' else 'transformer'
        points=[r for r in points if r[field] is not None]
        axes[1].scatter([r['reference'] for r in points],[r[field] for r in points],s=15,alpha=.5,label=method,color=color)
    axes[0].set_xticks(x,[c.replace('existing_development_cohort','original ASR cohort').replace('_',' ') for c in labels],rotation=15,ha='right')
    axes[0].set_ylabel('Boundary F1 (%)');axes[0].set_ylim(0,100);axes[0].legend()
    points=[r for r in results if r['metric'] in ('MLUw','mlu_words')]
    limit=max([v for r in points for v in (r['reference'],r['rules'],r['transformer']) if v is not None],default=1)*1.03
    axes[1].plot([0,limit],[0,limit],color='gray',linestyle='--');axes[1].set(xlim=(0,limit),ylim=(0,limit),xlabel='Reference MLUw',ylabel='Predicted MLUw')
    axes[1].legend();fig.suptitle('Paragraph segmentation: rules, words, and joint words + rules')
    fig.tight_layout();fig.savefig(OUT/'talkbank_results.png');plt.close(fig)
    print(f'Wrote {len(results)} individual metric comparisons for {len(models)} trained models')
    return results


def validation_agreement(model):
    """Agreement between the two input sources, not accuracy against labels."""
    records=[json.loads(s) for s in (EVIDENCE/'talkbank_training.jsonl').read_text().splitlines()]
    index={(r['corpus'],r['participant'],r['recording']):r for r in records}
    counts=dict(both=0,rule_only=0,neural_only=0,neither=0)
    for prediction in model['validation']['recordings']:
        r=index[prediction['corpus'],prediction['participant'],prediction['recording']]
        neural=set(prediction['predicted_boundaries'])
        for i,label in enumerate(r['labels']):
            if label<0:continue
            a=bool(r['baseline'][i]);b=i+1 in neural
            counts['both' if a and b else 'rule_only' if a else 'neural_only' if b else 'neither']+=1
    return dict(counts=counts,gap_agreement=(counts['both']+counts['neither'])/sum(counts.values()),
                boundary_dice=2*counts['both']/(2*counts['both']+counts['rule_only']+counts['neural_only']))


def prepare_target():
    import spacy
    nlp=spacy.load('en_core_web_sm')
    source=ROOT/'dataset/annotations/all_nafee_utterances.xlsx'
    workbook=load_workbook(source,read_only=True,data_only=True)
    base=ROOT/'outputs/validation/method_comparison/analysis/uncorrected'
    report=json.loads((base/'comparison.json').read_text())
    index={(r['patient_id'],r['source_file']):r for r in report['recordings']}
    records=[];sources={str(source.relative_to(ROOT)):digest(source)}
    for sheet in workbook:
        for filename,texts in auto.flatten_labels(sheet).items():
            rec=index[sheet.title,filename]
            raw=ROOT/'dataset/uncorrected'/sheet.title/filename
            assert digest(raw)==rec['raw_sha256']
            sources[str(raw.relative_to(ROOT))]=digest(raw)
            predicted=json.loads((base/rec['utterance_file']).read_text())['utterances']
            gold=[dict(text=auto.plain(t)) for t in texts]
            a,ab=auto.stream(gold);b,bb=auto.stream(predicted)
            mapping,_=auto.align(a,b)
            labels=[-100]*len(b)
            for i in range(len(a)-1):
                j=mapping.get(i)
                if j is not None and mapping.get(i+1)==j+1 and not({a[i],a[i+1]}&{'xxx','yyy','www'}):
                    labels[j]=int(i+1 in ab)
            labels[-1]=-100
            surface=[w for u in predicted for w in auto.WORD.findall(auto.plain(u['text']))]
            assert len(surface)==len(b)
            record=dict(participant=sheet.title,participant_verified=True,recording=filename,
                corpus='existing_development_cohort',tokens=b,surface_words=surface,
                baseline=[int(i+1 in bb) for i in range(len(b))],labels=labels,
                reference_mlu=auto.word_mlu(gold),baseline_mlu=auto.word_mlu(predicted),
                input_policy='Raw ASR transcript words and pre-existing rule boundaries. No manual punctuation or manual boundaries in features.',
                label_policy='Human column E boundaries on exactly aligned adjacent word gaps; all other gaps masked -100')
            expected=auto.boundary_counts(gold,predicted)
            known=[i for i,x in enumerate(labels) if x>=0]
            actual=dict(tp=sum(labels[i]==1 and record['baseline'][i]==1 for i in known),
                        fp=sum(labels[i]==0 and record['baseline'][i]==1 for i in known),
                        fn=sum(labels[i]==1 and record['baseline'][i]==0 for i in known),
                        tn=sum(labels[i]==0 and record['baseline'][i]==0 for i in known))
            assert all(actual[k]==expected[k] for k in actual)
            paragraph=' '.join(b)
            paragraph_predictions=auto.predict([dict(row=2,speaker='CHI',text=paragraph,start='',end='')],'CHI',nlp)
            paragraph_tokens,paragraph_boundaries=auto.stream(paragraph_predictions)
            assert paragraph_tokens==b
            record.update(task='paragraph_utterance_boundaries',
                baseline=[int(i+1 in paragraph_boundaries) for i in range(len(b))],
                baseline_mlu=auto.word_mlu(paragraph_predictions),
                input_policy='One case-folded ASR paragraph; words only; no source row, punctuation or speaker-turn constraints')
            records.append(record)
    workbook.close()
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    path=EVIDENCE/'boundary_dataset.jsonl'
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
    save(EVIDENCE/'boundary_dataset_sources.json',dict(sources=sources,records=len(records),
        participants=len({r['participant'] for r in records}),dataset_sha256=digest(path),
        restriction='Original ASR cohort: additional frozen-model evaluation only; never used to train or tune the TalkBank model'))
    print(f'Prepared {len(records)} recordings from {len({r["participant"] for r in records})} participants: {path}')
    return records


def requested_metric_table():
    from speech_pipeline.metrics import clinical_from_predictions
    rows=[]
    for kind in ('uncorrected','corrected'):
        base=ROOT/'outputs/validation/method_comparison/analysis'/kind
        report=json.loads((base/'comparison.json').read_text())
        for r in report['recordings']:
            us=json.loads((base/r['utterance_file']).read_text())['utterances']
            profile=clinical_from_predictions(us)
            for metric,value in profile['values'].items():
                rows.append(dict(dataset=kind,participant=r['patient_id'],recording=r['source_file'],
                    metric=metric,value=value,status=profile['status'][metric],
                    source_sha256=r['raw_sha256'],policy=profile['policy']))
    write_csv(OUT/'requested_metrics.csv',rows)
    return rows


def analyze():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from speech_pipeline.research.human_interrater import repeated_ba
    pairs=[r for r in read_csv(SOURCE/'metric_pairs.csv') if r['method']=='Revised' and r['reference']=='Nafee']
    boundaries=[r for r in read_csv(SOURCE/'boundary_recordings.csv') if r['method']=='Revised' and r['reference']=='Nafee']
    records=sorted({(r['patient_id'],r['source_file']) for r in pairs},key=lambda x:(int(x[0][1:]),x[1]))
    index={(r['patient_id'],r['source_file'],r['metric']):r for r in pairs}
    aliases={m:l for m,l in METRICS};output=[]
    for pid,name in records:
        for metric,label in METRICS:
            r=index[pid,name,metric];x,y=float(r['pipeline_value']),float(r['reference_value'])
            output.append(dict(participant=pid,recording=name,metric=metric,label=label,pipeline=x,reference=y,
                signed_error=x-y,absolute_error=abs(x-y),absolute_percent_error=100*abs(x-y)/abs(y) if y else None,
                comparison='Current rules vs primary human sample; per recording',
                normalization='TAALED-style lemma/POS for lexical measures; existing surface-word policy for utterance measures'))
    write_csv(OUT/'individual_results.csv',output)
    matrix=np.array([[float(index[pid,name,m]['absolute_percent_error']) for m,_ in METRICS] for pid,name in records])
    labels=[];recording_numbers=defaultdict(int)
    for pid,name in records:recording_numbers[pid]+=1;labels.append(f'{pid} / sample {recording_numbers[pid]}')
    plt.rcParams.update({'font.size':10,'figure.dpi':150})
    fig,ax=plt.subplots(figsize=(13,8));im=ax.imshow(matrix,cmap='YlOrRd',vmin=0,vmax=30,aspect='auto')
    ax.set_xticks(range(len(METRICS)),[x[1] for x in METRICS],rotation=45,ha='right')
    ax.set_yticks(range(len(records)),labels)
    for i in range(len(records)):
        for j in range(len(METRICS)):ax.text(j,i,f'{matrix[i,j]:.1f}',ha='center',va='center',fontsize=8,color='white' if matrix[i,j]>19 else '#202020')
    ax.set_title('Individual recording error against primary human annotations\nAbsolute percentage error; color capped at 30%, numbers show full values')
    fig.colorbar(im,ax=ax,label='Absolute percentage error (%)');fig.tight_layout();fig.savefig(OUT/'individual_errors.png');plt.close(fig)
    mlu=[r for r in output if r['metric']=='mlu_words']
    ba=repeated_ba([dict(patient_id=r['participant'],custom_value=r['pipeline'],reference_value=r['reference']) for r in mlu])
    x=np.array([r['reference'] for r in mlu]);y=np.array([r['pipeline'] for r in mlu]);fig,axes=plt.subplots(1,2,figsize=(12,5))
    axes[0].scatter(x,y);lo=min(x.min(),y.min())*.9;hi=max(x.max(),y.max())*1.05
    axes[0].plot([lo,hi],[lo,hi],'k--',lw=1);axes[0].set(xlabel='Human MLUw',ylabel='Pipeline MLUw',title='Recording-level agreement')
    axes[1].scatter((x+y)/2,y-x)
    for name,color in [('bias','#222222'),('lower_loa','#b02020'),('upper_loa','#b02020')]:axes[1].axhline(ba[name],color=color,linestyle='--',label=f'{name}: {ba[name]:.2f}')
    axes[1].set(xlabel='Mean of human and pipeline MLUw',ylabel='Pipeline − human MLUw',title='Repeated-measures Bland–Altman')
    axes[1].legend(fontsize=8)
    for i,label in enumerate(labels):
        axes[0].annotate(label,(x[i],y[i]),xytext=(3,3),textcoords='offset points',fontsize=7)
        axes[1].annotate(label,((x[i]+y[i])/2,y[i]-x[i]),xytext=(3,3),textcoords='offset points',fontsize=7)
    fig.suptitle('14 recordings from 10 participants; limits account for repeated recordings')
    fig.tight_layout();fig.savefig(OUT/'mlu_agreement.png');plt.close(fig)
    by_patient=defaultdict(lambda:dict(tp=0,tn=0,fn=0,fp=0))
    for r in boundaries:
        for k in ('tp','tn','fn','fp'):by_patient[r['patient_id']][k]+=int(r[k])
    def bf1(counts):
        c={k:sum(r[k] for r in counts) for k in ('tp','tn','fp','fn')}
        return 2*c['tp']/(2*c['tp']+c['fp']+c['fn'])
    patients=list(by_patient);rng=np.random.default_rng(42)
    boots=[bf1([by_patient[patients[i]] for i in sample]) for sample in rng.integers(0,len(patients),(10000,len(patients)))]
    summary={}
    for metric,_ in METRICS:
        groups=defaultdict(list)
        selected=[r for r in output if r['metric']==metric]
        for r in selected:
            if r['absolute_percent_error'] is not None:groups[r['participant']].append(r['absolute_percent_error'])
        summary[metric]=dict(median_participant_mean_recording_ape=statistics.median(statistics.mean(v) for v in groups.values()),
            min_recording_ape=min(r['absolute_percent_error'] for r in selected if r['absolute_percent_error'] is not None),
            max_recording_ape=max(r['absolute_percent_error'] for r in selected if r['absolute_percent_error'] is not None))
    result=dict(recordings=len(records),participants=len(patients),metric_errors=summary,
        boundary_f1=bf1(list(by_patient.values())),boundary_f1_cluster_bootstrap_ci=list(map(float,np.quantile(boots,[.025,.975]))),
        boundary_alignment_coverage_range=[min(float(r['reference_gap_coverage']) for r in boundaries),max(float(r['reference_gap_coverage']) for r in boundaries)],
        mlu_repeated_measures_agreement=ba,
        source_hashes={str(p.relative_to(ROOT)):digest(p) for p in [SOURCE/'metric_pairs.csv',SOURCE/'boundary_recordings.csv']},
        assessment='Development-cohort descriptive results. No universal clinical acceptable-error threshold; prespecify intended-use tolerances and seek external validation.')
    save(OUT/'existing_cohort.json',result)
    requested_metric_table()
    write_report(result)
    print(f'Analyzed {len(records)} individual recordings; boundary F1={result["boundary_f1"]:.4f}. Report: {OUT/"REPORT.md"}')
    return result


def write_report(summary):
    access_path=EVIDENCE/'access.json';access=json.loads(access_path.read_text()) if access_path.exists() else {}
    lines=['# Speech validation and paragraph segmentation','',
        '## Dataset access','',
        '| Corpus | Verified access status | Intended role |','| --- | --- | --- |']
    for name,spec in CORPORA.items():
        status=access.get(name,{}).get('status','not_downloaded')
        lines.append(f'| [{name}]({spec["page"]}) | {status} | '+('Autism parent–child conversation' if name=='nyu_emerson' else 'Child speech; analyze task types separately')+' |')
    lines += ['', 'TalkBank requires registration/sign-in for transcripts and media. No external corpus results or audio ASR results are claimed without actual downloaded data. NYU–Emerson is the closer domain match; OCSC offers a larger task-diverse test. Its alphabet, number, Wug, reading, and discourse tasks should not be pooled. Preserve corpus citations and access conditions.',
      '', 'The current repository starts from transcript workbooks; it does not contain the upstream MOSS audio transcription/diarization implementation. Audio-to-metric evaluation needs that stage or a specified local ASR baseline. Oracle transcript segmentation is a separate experiment from ASR errors and speaker-assignment errors.',
      '', '## Requested transformer training: TalkBank corpora','',
      'The models learn to divide continuous child-speech paragraphs into utterances using the authorized OCSC and NYU–Emerson CHAT transcripts. CHAT boundaries supply target labels. The words-only comparison receives case-folded words; the joint model combines those words with rule-gap embeddings. All source punctuation, row breaks, speaker-turn cuts, morphology and timestamps are withheld. Rules are also evaluated separately.',
      '', 'The corpus preparation command derives boundary targets from CHAT utterances while withholding published punctuation, sentence casing, row breaks, morphology and timestamps from predictor features. A reviewed child identity table groups repeated sessions. The TalkBank trainer uses child-disjoint training/validation/test partitions per corpus, groups exact duplicate sessions, selects epochs and threshold on validation only, and reports test children separately by corpus, task and recording. An optional frozen-model evaluation on the existing ASR cohort uses no target-cohort labels for training or tuning. Commands and identity-table format are in the root README.',
      '', '## Existing cohort: individual results','',
      f'Analyzed {summary["recordings"]} sampled recordings from {summary["participants"]} participants, against primary human labels. These are development data, not independent external validation.',
      f'Boundary F1: **{100*summary["boundary_f1"]:.2f}%**, descriptive participant-bootstrap 95% interval **{100*summary["boundary_f1_cluster_bootstrap_ci"][0]:.2f}–{100*summary["boundary_f1_cluster_bootstrap_ci"][1]:.2f}%**. Evaluate boundary precision/recall and alignment coverage as well as F1; gap accuracy can be misleading because most word gaps are not boundaries.',
      '', '| Metric | Median participant mean recording APE | Worst individual recording APE |','| --- | ---: | ---: |']
    for metric,label in METRICS:
        value=summary['metric_errors'][metric];lines.append(f'| {label} | {value["median_participant_mean_recording_ape"]:.2f}% | {value["max_recording_ape"]:.2f}% |')
    ba=summary['mlu_repeated_measures_agreement']
    lines += ['',f'MLUw repeated-measures bias: {ba["bias"]:.3f} words/utterance; limits of agreement: {ba["lower_loa"]:.3f} to {ba["upper_loa"]:.3f}. Correlation alone would not establish agreement.',
      '', 'The heatmap reports each recording separately. The Bland–Altman and identity plots identify systematic bias and outliers while accounting for repeated recordings in the agreement estimates. `individual_results.csv` preserves both absolute and percentage errors, including the recording identifier. Zero-reference percentage errors remain missing, rather than being fabricated as zero.',
      '', '![Individual recording errors](individual_errors.png)','', '![MLUw agreement](mlu_agreement.png)',
      '', '### Is the error good enough?','',
      'These results support a descriptive research comparison; they do not establish clinical interchangeability. Small median errors can hide large recording-level boundary and length-SD errors. Predefine acceptable errors in words/utterance or percentage points for the intended use, compare those tolerances with agreement limits and individual outliers, and evaluate a frozen method on untouched participants/corpora. Treat manual-review effort and failure/coverage rates as outcomes, not just averages.',
      '', '## Requested clinical measures','',
      '| Measure | What can be produced responsibly |','| --- | --- |',
      '| TNW, NDW, TTR, MATTR | Available under an explicit word-form/lemma, filler, retrace, eligibility, and MATTR-window policy. The existing TAALED normalization is not SALT surface-word counting. |',
      '| MLUw | Available from predicted boundaries; compare with human boundaries and official CLAN on identical inputs separately. |',
      '| MLUm | Requires complete supported reviewed morphology, or official CLAN with an appropriate `%mor` tier. spaCy lemmas and UD feature counts are not morpheme counts. |',
      '| WPM | Requires reliable timing and a declared elapsed-time versus speaking-time denominator. |',
      '| PGU | Requires explicit complete grammaticality labels or a separately validated grammaticality model. An unmarked utterance is not automatically grammatical. |',
      '| Omission/substitution counts | Require complete reviewed language-error annotations; distinguish these from ASR insertions/deletions/substitutions. |',
      '| Maze percentage | Requires explicitly coded maze words and a matched denominator. Filler deletion and CHAT retrace-marker counts alone are not total SALT maze percentage. |',
      '', '`requested_metrics.csv` contains the requested profile for all 37 existing recordings in both transcript conditions. The new `--metric-profile clinical` supports the same profile on new runs: five surface/boundary measures, with annotation-dependent values left missing and explained. The reviewed-annotation API supports the remaining measures when their evidence is supplied. Maze scope must be explicit; a whole-sample denominator includes excluded utterances. Existing published results remain unchanged. No SALT software comparison or automatic clinical equivalence is claimed.',
      '', 'Primary scoring references: [CLAN manual](https://talkbank.org/0info/manuals/CLAN.html), [CHAT manual](https://talkbank.org/0info/manuals/CHAT.html), [SALT transcription requirements](https://www.saltsoftware.com/blog/transcription-shortcuts), [SALT maze definitions](https://www.saltsoftware.com/blog/mazes-abandoned-utterances).',
      '', '## Independent calculator checks','']
    audit_path=EVIDENCE/'calculator_checks.json'
    if audit_path.exists():
        audited=json.loads(audit_path.read_text())
        lines += [f'Fresh official CLAN MLU/FREQ runs completed for {len(audited)} existing recordings. Case folding is explicit (`freq +k`) and the MATTR window is 50.',
            '', '| Same-input CLAN measure | Maximum absolute difference |','| --- | ---: |']
        for m in ['TNW','NDW','TTR','MATTR','MLUw']:
            value=max((r['checks']['surface_calculator_checks'][m]['absolute_error'] or 0) for r in audited)
            lines.append(f'| {m} | {value:.6f} |')
        lines += ['', 'Word and type counts match exactly. Ratio/MATTR/MLUw differences are within CLAN’s three-decimal output precision. All four shared lexical measures match TAALED exactly when both calculators receive the identical normalized token sequence. Full-text preprocessing comparisons remain separate: the two normalizers can differ in tokenization and lexical policy, which is not an arithmetic error. Raw official outputs and version/build hashes are in `outputs/validation/benchmark/`.']
    public_checks=EVIDENCE/'talkbank_calculators.json'
    if public_checks.exists():
        checks=json.loads(public_checks.read_text())['results']
        lines += [f'An additional {len(checks)} preselected validation paragraphs from both public corpora were checked with official CLAN and TAALED. TNW/NDW match exactly, shared-token TAALED measures match exactly, and CLAN ratios/MLUw agree within its three-decimal display precision. These are calculator checks on identical normalized input, separate from segmentation accuracy; raw outputs are in `talkbank_calculators.json`.']
    lines += ['', '## Paragraph segmentation: held-out results','']
    public_model=OUT/'talkbank_model.json'
    if public_model.exists():
        trained=json.loads(public_model.read_text());test=trained['held_out_test']
        sources=json.loads((EVIDENCE/'talkbank_training_sources.json').read_text())
        splits=trained['splits']
        lines += ['The requested TalkBank training run is complete.',
            f'Training corpus: {sources["participants"]} children, {sources["records"]} paragraphs, '+
            f'{sum(v["words"] for v in sources["by_corpus"].values()):,} words. Child partitions: '+
            ', '.join(f'{name} {len(ids)}' for name,ids in splits.items())+'. '+
            f'Model selection used validation only; selected epoch {trained["selected_epoch"]}, threshold {trained["threshold"]:.2f}.',
            '', '| Historical evaluation group | Children | Rules boundary F1 | Transformer boundary F1 |',
            '| --- | ---: | ---: | ---: |']
        groups={**test['by_corpus'],**trained.get('additional_evaluation',{}).get('by_corpus',{})}
        for corpus,scores in groups.items():
            lines.append(f'| {corpus} | {scores["participants"]} | {100*scores["baseline"]["f1"]:.2f}% | {100*scores["transformer"]["f1"]:.2f}% |')
        lines += ['', 'The original ASR cohort is an additional frozen-model test: its labels are not used for weights, epochs or thresholds. Both methods receive the same continuous paragraph, and the rules were originally developed on this cohort, so treat that comparison as descriptive. Public test children are absent from training/validation, but the benchmark has now been inspected across successive research rounds and both public corpus domains are represented in training; this is not an unseen-corpus or fresh confirmatory test. The test scores do not establish clinical interchangeability or audio ASR performance.',
            '', '`talkbank_split.json` in the validation folder records the child/paragraph partitions before training; `talkbank_model.pt` is the trained checkpoint; `talkbank_model.json` preserves data hashes, child partitions, per-task/per-recording boundary results and model selection. `talkbank_metric_errors.csv` records individual downstream word-metric errors. Eligibility and filler policies remain explicit; annotation-dependent clinical measures have not been invented.',
            '', '![TalkBank-trained model held-out results](talkbank_results.png)']
        if trained.get('additional_evaluation'):
            score=trained['additional_evaluation']['overall']
            lines += ['', 'The words-only model improves the held-out TalkBank paragraph task, but its original-ASR paragraph score '+
                ('is lower than' if score['transformer']['f1']<score['baseline']['f1'] else 'exceeds')+
                ' the rules. Keep model use experimental; clean-transcript results alone do not establish improvement on ASR paragraphs. The earlier cohort plots describe the original pipeline with its available transcript context; this table compares continuous-paragraph inputs.']
    else:
        lines += ['TalkBank training has not yet produced a completed checkpoint. No TalkBank performance result is claimed.']
    hybrid_path=OUT/'hybrid_model.json'
    if hybrid_path.exists() and public_model.exists():
        joint=json.loads(hybrid_path.read_text())
        if joint['input_sha256']!=trained['input_sha256'] or joint['splits']!=trained['splits']:
            raise ValueError('Model comparisons require identical source data and child partitions')
        agreement=validation_agreement(trained)
        lines += ['', '## Joint paragraph and rule model','',
            'Word/position representations and learned rule-gap representations enter the same transformer. Both inputs contribute to one boundary prediction, trained against CHAT labels. The model does not use gold utterance cuts as a feature. New-paragraph inference automatically computes the rules on the same normalized paragraph.',
            f'Joint model: epoch {joint["selected_epoch"]}, validation-selected threshold {joint["threshold"]:.2f}, {joint["parameters"]:,} parameters. Child partitions and source-data hash match the words-only run.',
            '', '| Evaluation | Rules F1 | Words-only F1 | Joint F1 |','| --- | ---: | ---: | ---: |']
        for label,key in [('Validation','validation'),('Held-out TalkBank benchmark','held_out_test'),('Original ASR paragraphs','additional_evaluation')]:
            a=trained[key]['overall'];b=joint[key]['overall']
            lines.append(f'| {label} | {100*a["baseline"]["f1"]:.2f}% | {100*a["transformer"]["f1"]:.2f}% | {100*b["transformer"]["f1"]:.2f}% |')
        lines += ['', f'The original rules and words-only model agree on {100*agreement["gap_agreement"]:.2f}% of eligible validation gaps, but boundary-specific Dice agreement is {100*agreement["boundary_dice"]:.2f}%. Most gaps are non-boundaries, so overall gap agreement alone can hide differences in actual utterance cuts. These agreement measures compare the input sources; they are not accuracy against CHAT labels.',
            '', 'The joint model was requested after the words-only benchmark had been examined. Training and model selection still use train/validation only, with no child overlap; this is an additional research comparison on the same benchmark. A new independent corpus/cohort is needed for a confirmatory final evaluation. Each checkpoint is retained to make the comparison reproducible; metrics and the three-way plot share the existing output files.',
            '', 'Run `boundary_model segment` with `outputs/results/benchmark/hybrid_model.pt` for joint inference. Performance remains experimental, particularly on ASR text; no claim of clinical interchangeability is made.']
    lines += ['', '## Validation protocol','',
      '1. Download and inspect authorized CHAT transcripts and matching media; audit speaker, timing, morphology, error, and disfluency coverage.',
      '2. Freeze rules and the metric protocol. Evaluate each task and recording separately; cluster uncertainty by participant, not word or utterance.',
      '3. Withhold gold CHAT punctuation, utterance row breaks, morphology, and timestamps from transcript segmentation inputs. Use gold information only as reference labels. For the learned paragraph task, omit all speaker-turn cuts and let the model predict every internal boundary.',
      '4. Compare custom versus TAALED on the same normalized lexical input, and custom versus official CLAN on the same scored CHAT input. Report policy differences separately from implementation disagreements.',
      '5. Evaluate audio ASR and diarization separately, then evaluate downstream metric error. Do not use gold child utterance timestamps to claim an end-to-end diarization result.',
      '6. If an external result motivates new rules or model tuning, designate that corpus a development set and reserve a new corpus/participant partition for final testing.',
      '', 'Current limitations: audio is separate from these transcript experiments; morphology, grammar, and maze predictions require additional annotations/models. Corpus acquisition status is listed above; absent results are never replaced with fabricated values.']
    if (OUT/'improved_model.json').exists():
        from speech_pipeline.research.segmentation import improvement_report
        lines[2:2]=improvement_report()
    if (EVIDENCE/'reassessment.json').exists():
        from speech_pipeline.research.reassessment import report_section
        lines[2:2]=report_section()
    if (EVIDENCE/'metric_training.json').exists():
        from speech_pipeline.research.metric_training import report_section
        lines[2:2]=report_section()
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')


def calculator_audit():
    import spacy
    nlp=spacy.load('en_core_web_sm');base=ROOT/'outputs/validation/method_comparison/analysis/uncorrected'
    report=json.loads((base/'comparison.json').read_text());results=[]
    # All 37 recordings: an independent current official CLAN/TAALED calculator run.
    for r in report['recordings']:
        utterances=json.loads((base/r['utterance_file']).read_text())['utterances']
        with TemporaryDirectory(prefix='same_input_check_') as temp:
            checks=calculator_checks(utterances,nlp,Path(temp),'recording')
        results.append(dict(participant=r['patient_id'],recording=r['source_file'],checks=checks))
        print('Calculator check',len(results),'/ 37',flush=True)
    save(EVIDENCE/'calculator_checks.json',results)
    return results


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task',choices=['analyze','prepare-target','calculator-checks','talkbank-analysis'])
    args=parser.parse_args()
    {'analyze':analyze,'prepare-target':prepare_target,'calculator-checks':calculator_audit,'talkbank-analysis':talkbank_analysis}[args.task]()
