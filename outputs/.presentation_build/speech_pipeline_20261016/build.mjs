import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile} from '@oai/artifact-tool';

const ROOT='/Users/nafeekazi/Desktop/PhD/speech_pipeline_code';
const BUILD=path.join(ROOT,'outputs/.presentation_build/speech_pipeline_20261016');
const SKILL='/Users/nafeekazi/.codex/plugins/cache/openai-primary-runtime/presentations/26.1013.11914/skills/presentations';
const RUNTIME='/Users/nafeekazi/.cache/codex-runtimes/codex-primary-runtime/dependencies';
process.env.RUNTIME_NODE_MODULES=path.join(RUNTIME,'node/node_modules');
process.env.RUNTIME_NODE=path.join(RUNTIME,'node/bin/node');
process.env.RUNTIME_PYTHON=path.join(RUNTIME,'python/bin/python3');
const FINAL=path.join(ROOT,'outputs/presentations/speech_pipeline_results_2026-10-16_final.pptx');
const {finalizePresentation,applyPresentationChartFont}=await import(pathToFileURL(path.join(SKILL,'container_tools/artifact_tool_utils.mjs')).href);
const D=JSON.parse(await fs.readFile(path.join(BUILD,'data.json'),'utf8'));
const C={bg:'#F8F7F3',ink:'#182E40',teal:'#007F78',muted:'#586B78',pale:'#E3EFED',white:'#FFFFFF',line:'#CBD5D9',amber:'#9B5A15',old:'#AAB7C1'};
const FONT='Arial';
const P=Presentation.create({slideSize:{width:1280,height:720}});
const nativeTables=[];const nativeCharts=[];
const f=n=>Number(n).toFixed(2);
const lookup=(rows,corpus,metric)=>rows.find(r=>r.corpus===corpus&&r.metric===metric).mean_participant_ape;
const primary=['eligible_mlu_utterances','MLUw','length_sd'];
const vals=(rows,corpus)=>primary.map(m=>f(lookup(rows,corpus,m))+'%');
const base=D.cv.find(x=>x.configuration==='lr2_count1');
const low=D.cv.find(x=>x.configuration==='lr2_count025');
const higher=D.cv.find(x=>x.configuration==='lr3_count1');
const U={ocsc:'https://talkbank.org/childes/access/Eng-NA/OCSC.html',nyu:'https://talkbank.org/asd/access/English/NYU-Emerson.html',lucid:'https://doi.org/10.3389/fpsyg.2021.661415',osf:'https://osf.io/dfhpu/',ocscPaper:'https://doi.org/10.1016/j.specom.2025.103206',cv:'https://scikit-learn.org/stable/modules/grid_search.html',groups:'https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data',clan:'https://talkbank.org/manuals/CLAN.pdf',mini:'https://huggingface.co/microsoft/MiniLM-L12-H384-uncased'};
function txt(s,text,x,y,w,h,size=26,color=C.ink,bold=false,align='left'){
 const a=s.shapes.add({geometry:'textbox',name:`text-${s.shapes.items.length}`,position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 a.text=text;a.text.style={typeface:FONT,fontSize:size,color,bold,alignment:align,verticalAlignment:'top',autoFit:'none',insets:0,wrap:'square'};return a;
}
function slide(title,subtitle=''){
 const s=P.slides.add();s.background.fill=C.bg;
 txt(s,title,64,46,1150,68,44,C.ink,true);
 if(subtitle)txt(s,subtitle,66,120,1148,52,24,C.muted);
 txt(s,String(P.slides.items.length).padStart(2,'0'),1170,676,46,23,17,C.muted,false,'right');
 return s;
}
function note(s,text,sources=[]){s.speakerNotes.text=text+'\n\nSources\n'+sources.join('\n');}
function table(s,values,y,widths,height,font=25,x=64){
 const t=s.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width:widths.reduce((a,b)=>a+b,0),height,columnWidths:widths,values});
 t.styleOptions={headerRow:true,bandedRows:false};
 t.borders.assign({fill:C.line,width:0.7,style:'solid'});
 for(let r=0;r<values.length;r++){
  t.rows[r].height=height/values.length;
  for(let c=0;c<values[r].length;c++){
   const cell=t.getCell(r,c);cell.fill=r===0?C.ink:(r%2===0?C.white:C.bg);
   cell.text.style={typeface:FONT,fontSize:r===0?font-1:font,color:r===0?C.white:C.ink,bold:r===0,verticalAlignment:'middle',alignment:'left',insets:12};
  }
 }
 nativeTables.push(P.slides.items.length);return t;
}
function box(s,text,x,y,w,h,fill=C.pale,size=25){
 const a=s.shapes.add({geometry:'rect',name:`diagram-${s.shapes.items.length}`,position:{left:x,top:y,width:w,height:h},fill,line:{fill:C.teal,width:1.3}});
 a.text=text;a.text.style={typeface:FONT,fontSize:size,color:C.ink,bold:false,alignment:'center',verticalAlignment:'middle',insets:16,autoFit:'none'};return a;
}
function edge(s,a,b,from='right',to='left'){
 return s.shapes.connect(a,b,{kind:'elbow',fromSide:from,toSide:to,line:{fill:C.teal,width:2},tail:{type:'triangle',width:'med',length:'med'}});
}
function row(s,label,body,y,labelW=250){txt(s,label,66,y,labelW,70,29,C.teal,true);txt(s,body,66+labelW+30,y,1118-labelW-30,92,26);}

// 1
{
 const s=P.slides.add();s.background.fill=C.ink;
 txt(s,'Speech pipeline validation',72,132,1125,100,64,C.white,true);
 txt(s,'Datasets, model development and results',74,250,1080,75,34,C.white);
 txt(s,'LUCID, OCSC and NYU–Emerson',74,390,1080,60,30,'#9FD7D0');
 txt(s,'Research update\n16 October 2026',76,566,1080,68,23,'#D2DDE3');
 note(s,'This presentation covers the current public transcript-validation work. Three complete five-fold comparisons are available. Further training stopped at the user’s request. The existing production model remains active. No v4 fresh final fit or test occurred.',[D.sources.closed]);
}
// 2
{
 const s=slide('Research objective','Measure language accurately while keeping each error source visible');
 txt(s,'11 project measures',66,202,1090,48,33,C.teal,true);
 txt(s,'Six vocabulary counts, TTR, MATTR-50, utterance count,\nmean utterance length in words and utterance-length SD',66,263,1090,98,29);
 const a=box(s,'Current phase\nHuman transcripts',66,414,324,118);
 const b=box(s,'Our boundary model\nand scoring pipeline',478,414,324,118);
 const c=box(s,'Reference comparisons\nand error analysis',890,414,324,118);
 edge(s,a,b);edge(s,b,c);
 txt(s,'Next phase adds audio transcription and speaker selection on the original study.',66,581,1130,55,24,C.muted);
 note(s,'The original pilot uses 37 recordings from 10 autistic participants aged 8–17 years. That work and audio processing remain separate from the public transcript evaluation summarized here. Public LUCID checks also cover additional shared lexical indices beyond the project’s 11 primary outputs.',[path.join(ROOT,'conference_paper/conference_paper.docx'),path.join(ROOT,'thompson_center_poster/Thompson_Center_Abstract.docx'),path.join(ROOT,'SESSION_HANDOFF.md')]);
}
// 3
{
 const s=slide('Dataset sources and selection','Each source tests a different part of the system');
 table(s,[['Dataset and source','What we use','Why we chose it'],['LUCID\nPublic TextGrids and OSF','Adult dialogue\nPublished lexical values','Numeric reference for\nlexical calculations'],['Ohio Child Speech Corpus\nTalkBank CHILDES (OCSC)','Child speech tasks\nHuman CHAT transcripts','Larger child sample\nHuman utterance coding'],['NYU–Emerson\nTalkBank ASDBank','Parent–child CHAT\nYoung autistic children','Autism context\nTest of transfer']],194,[350,390,412],350,25);
 txt(s,'Corpus references: Baese-Berk et al. (2021), Wagner et al. (2025), Steen et al. (2026)',66,586,1145,58,21,C.muted);
 note(s,'LUCID source includes the authors’ TAALED_results.csv at OSF and publicly distributed DiapixUK speaker-labeled TextGrids. The paper describes 40 adult native Southern British English speakers. LUCID supplies lexical statistics, not gold utterances or MLU. OCSC’s published corpus contains 303 children and elicited tasks collected in a science-museum lab. NYU–Emerson’s current release contains 30 autistic children in parent–child interactions, largely recorded at home over Zoom. The reasons in the final column describe our evaluation rationale.',[U.lucid,U.osf,'https://nuwildcat.sharepoint.com/:f:/s/WNB-OSCAAR/Et1cBwxG3eFMgo7l6UWDNA0B98o1Z37glbdyz5f0ZzdzvQ?e=kuhfkY',U.ocsc,U.ocscPaper,U.nyu,'https://doi.org/10.21415/60TV-D669','https://doi.org/10.21415/VT7P-YF83']);
}
// 4
{
 const s=slide('Participant-independent cohorts','All sections and training crops from a child keep that child’s role');
 table(s,[['Usable source','Train\nchildren / sections','Eval\nchildren / sections','Test\nchildren / sections'],['OCSC','211 / 567','45 / 125','45 / 111'],['NYU–Emerson','20 / 63','4 / 14','4 / 15'],['Total','231 / 630','49 / 139','49 / 126']],190,[290,284,284,294],280,26);
 txt(s,'Zero participant, recording or exact full-section overlap across roles',66,513,1120,48,29,C.teal,true);
 txt(s,'LUCID stays outside boundary-model training. Four source files lack eligible discourse samples.',66,581,1120,62,24,C.muted);
 note(s,'All 333 CHAT source files were inventoried and 895 usable task sections retained for 329 participants. Two additional assigned NYU training children lack eligible target speech. Two OCSC files contain only excluded alphabet, number or Wug tasks. Exclusions remain explicit and are not zero-error results. The audit checks participants, source files and exact full-section text. Participant disjointness and the history of earlier benchmark exposure are both preserved.',[D.sources.cohorts,path.join(ROOT,'outputs/validation/cohorts_v1/AUDIT.md')]);
}
// 5
{
 const s=slide('Current transcript pipeline','Observed speaker changes supply context. The model predicts utterance boundaries');
 const a=box(s,'Target-speaker\ntranscript',66,201,240,105);
 const b=box(s,'Observed turns\nand word preparation',368,201,310,105);
 const c=box(s,'Fixed spaCy\ngrammatical cues',740,201,300,105);
 edge(s,a,b);edge(s,b,c);
 const d=box(s,'Custom MiniLM\nboundary model',740,391,300,105);
 edge(s,c,d,'bottom','top');
 const e=box(s,'Predicted utterances\nmargin 0.25',368,391,310,105);
 edge(s,d,e,'left','right');
 const g=box(s,'Faithful CHAT\nNative CLAN MLU / FREQ',66,391,240,105,C.pale,23);
 edge(s,e,g,'left','right');
 txt(s,'Lexical branch',66,555,250,42,28,C.teal,true);
 txt(s,'The same retained words feed historical TAALED compatibility rules\nfor lexical counts, density and diversity.',368,555,820,81,25);
 note(s,'The active production workflow uses the previous calibrated observed-turn model and margin 0.25. Consecutive target-speaker source rows merge into observed turns, removing human utterance endings as inference cuts. Reference coding and gold target boundaries remain evaluator information. Words remain unchanged. Native CLAN calculates supported CHAT measures, while the historical lexical profile supports the LUCID comparison. Plain text without speaker rows uses the existing paragraph workflow.',[path.join(ROOT,'speech_pipeline/transcript.py'),path.join(ROOT,'speech_pipeline/research/transcript_chat.py'),path.join(ROOT,'speech_pipeline/chat_metrics.py'),path.join(ROOT,'speech_pipeline/lexical_compat.py'),path.join(ROOT,'outputs/results/benchmark/transcript_workflow.json')]);
}
// 6
{
 const s=slide('Our custom boundary model','Pretrained language representations support a learned split-or-continue decision');
 const a=box(s,'MiniLM-L12-H384 encoder\nLeft and right word context',66,206,360,115);
 const b=box(s,'Custom rule embedding\n16 dimensions',66,389,360,115);
 const c=box(s,'Custom classifier\n128-unit GELU layer\nDropout 0.15',509,284,337,145);
 const d=box(s,'Continue\nor split',929,284,285,145);
 edge(s,a,c);edge(s,b,c);edge(s,c,d);
 txt(s,'Task training updates the encoder, rule embedding and classifier together.',66,565,1140,48,28,C.teal,true);
 txt(s,'SpaCy preprocessing and official CLAN scoring remain fixed external components.',66,620,1140,35,23,C.muted);
 note(s,'The Microsoft MiniLM encoder is an off-the-shelf pretrained component. Our rule embedding and gap classifier are custom trainable layers. The classifier concatenates two 384-dimensional context vectors and a 16-dimensional rule embedding, then uses a 128-unit GELU layer, dropout 0.15 and two output logits. The registered CV models use independent gap decoding, cross entropy plus a relative boundary-count loss, 256-token windows and stride 128. No CRF or ensemble is active in this study. All new fits begin from the original pinned pretrained encoder, not another task model’s fine-tuned weights.',[D.sources.model,U.mini,path.join(ROOT,'speech_pipeline/research/metric_training.py'),path.join(ROOT,'outputs/validation/cv_hyperparameter_v4/protocol.json')]);
}
// 7
{
 const s=slide('References and error calculation','Lexical agreement, native CHAT scoring and boundary accuracy have separate references');
 table(s,[['Comparison','Reference','Reported evidence'],['LUCID lexical results','Authors’ actual TAALED table','Mean participant error\nand coverage by condition'],['OCSC / NYU measures','Human CHAT scored by\nthe same native CLAN commands','Eligible count, MLUw, length SD\nand vocabulary errors'],['Utterance boundaries','Human utterance labels\nwithheld from model inputs','Boundary F1 and exact spans']],190,[285,424,443],311,25);
 txt(s,'Absolute percentage error = 100 × |prediction − reference| / |reference|',66,543,1140,46,29,C.teal,true);
 txt(s,'Average sections within a participant, then give each participant equal weight.',66,604,1140,40,24,C.muted);
 txt(s,'Project goals: 5% for most measures, 10% for length SD and original MTLD',66,649,1120,24,19,C.muted);
 note(s,'MLUw is native eligible MLU words divided by native eligible MLU utterances. FREQ and MLU have different eligibility rules. Zero reference denominators and unavailable values remain missing. The cross-corpus tuning objective averages the three primary participant errors equally across OCSC and NYU. Engineering goals are 5% for most measures and 10% for length SD and original MTLD. These are project goals, not validated clinical acceptance standards. APE is a metric agreement measure, not classification accuracy.',[path.join(ROOT,'speech_pipeline/research/transcript_chat.py'),path.join(ROOT,'speech_pipeline/chat_metrics.py'),U.clan,D.sources.lucid]);
}
// 8
{
 const s=slide('What we tried before the CV search','Corrections targeted known scoring and segmentation differences');
 table(s,[['Experiment','Change','Result'],['Historical lexical compatibility','Dictionary, POS/category rules\nand fixed pause handling','Strong LUCID agreement\nin babble and L2'],['Native CHAT scoring','Faithful export with official\nCLAN MLU and FREQ','Vocabulary errors mostly\nbelow 3% on existing tests'],['Input-mode comparison','Paragraph ML, observed-turn ML\nand observed-turn rules','Eval selected observed-turn ML'],['Bounded margin calibration','Seven fixed margins in\ntwo input modes','Eval selected margin 0.25\nCount / MLU gains were modest']],190,[310,426,416],376,25);
 txt(s,'An eval-only ensemble comparison did not replace the saved production model.',66,602,1130,48,23,C.muted);
 note(s,'The compatibility correction preserves the current spaCy model and does not claim exact reproduction of the authors’ historical environment. Boundary calibration leaves words, lexical calculations and trained weights unchanged. The original zero-margin batched implementation failed parity for six recordings, so the original preprocessing path replaced it before accepting calibration. A later full-model/five-model/six-model ensemble exploration used original eval and was superseded when the user clarified CV’s selection role. Its frozen-choice test preparation began but stopped before a fresh metric result. No ensemble entered production.',[path.join(ROOT,'outputs/results/transcript_validation/REPORT.md'),path.join(ROOT,'outputs/validation/transcript_validation/calibration_zero_parity.json'),path.join(ROOT,'outputs/validation/cv_hyperparameter_v4/protocol.json')]);
}
// 9
{
 const s=slide('Five-fold cross-validation design','Original training participants supply every configuration comparison');
 const t=box(s,'Original train: 231 children',440,184,400,74);
 const a=box(s,'Fit\n147–148 children',66,332,310,96);
 const b=box(s,'Inner validation\n37 different children',484,332,310,96);
 const h=box(s,'CV heldout\n46–47 different children',902,332,310,96);
 edge(s,t,a,'bottom','top');edge(s,t,b,'bottom','top');edge(s,t,h,'bottom','top');
 const c=box(s,'Fresh fit and inner selection\nFreeze checkpoint and margin',66,507,728,96);
 const score=box(s,'Score heldout\nparticipants',902,507,310,96);
 edge(s,a,c,'bottom','top');edge(s,b,c,'bottom','top');edge(s,c,score);edge(s,h,score,'bottom','top');
 txt(s,'Repeat on the same five folds per setting. Pool all five heldout scores.',66,638,1140,35,27,C.teal,true);
 note(s,'Within each fold, fitting, inner and CV-heldout participants are disjoint. Each child contributes all sections to a single role. The inner split selects epoch by native objective at margin 0, then calibrates over 12 fixed margins. CV-heldout scores rank training recipes, and original eval/test remain outside that comparison. Five folds contain 4 NYU children each and 42–43 OCSC children. Shared initial seeds within a fold across recipes deliberately pair initialization and augmentation. This is five-fold hyperparameter search with a within-fold holdout, not a 5-by-5 nested-CV claim. CV minimum is a tuning score. Fold SD is descriptive because training groups overlap.',[path.join(ROOT,'outputs/validation/cv_hyperparameter_v4/protocol.json'),D.sources.cohorts,U.cv,U.groups]);
}
// 10
{
 const s=slide('Completed training configurations','Three settings completed all five folds. The fourth remains incomplete');
 table(s,[['Configuration','Encoder / head LR','Count-loss\nweight','Pooled CV\nerror'],['Baseline','2e−5 / 1e−4','1.00',f(base.objective)+'%'],['Lower count loss','2e−5 / 1e−4','0.25',f(low.objective)+'%'],['Higher learning rate','3e−5 / 1.5e−4','1.00',f(higher.objective)+'%'],['Higher LR + lower loss','3e−5 / 1.5e−4','0.25','Incomplete']],190,[350,340,225,237],333,25);
 txt(s,'Baseline had the best completed combined CV score.',66,565,1140,45,31,C.teal,true);
 txt(s,'OCSC favored lower count loss. That setting increased NYU’s error.',66,622,1140,33,24,C.muted);
 note(s,'Pooled CV objective equally averages eligible-count, MLUw and length-SD participant APE across both corpora. Completed fold means and sample SDs are baseline 13.16 ± 2.48, lower count loss 13.36 ± 2.68, and higher learning rate 13.53 ± 3.10. The five verified baseline fits were reused with frozen weights and roles checked, then recalibrated on their inner groups with the same 12 margins. Ten new nonbaseline fold fits completed. The fourth configuration stopped during fold 1 after its complete epoch-4 checkpoint, before a frozen fold selection or heldout score. Exclude that setting as a whole. The registered four-setting study is incomplete. No new v4 final fit, test or activation occurred.',[D.sources.closed,path.join(ROOT,'outputs/validation/cv_hyperparameter_v4/stopped_search_integrity_audit.json'),path.join(ROOT,'outputs/validation/cv_hyperparameter_v4/user_stop.json')]);
}
// 11
{
 const s=slide('LUCID results against published values','Lexical comparison with the authors’ actual per-sample TAALED table');
 const lm=(c,m)=>D.lucid.find(r=>r.condition===c&&r.metric===m).mean_participant_ape;
 table(s,[['Condition','Scored / published','Word-count\nerror','MATTR-50\nerror'],...['babble','L2','noBarrier'].map((c,i)=>[['Babble','L2','No barrier'][i],'60 / 60',f(lm(c,'basic_ntokens'))+'%',f(lm(c,'mattr50_aw'))+'%']),['Vocoder','0 / 60','Unresolved','Unresolved']],190,[285,310,280,277],310,26);
 txt(s,'All 15 shared mean errors meet targets in babble and L2.',66,543,1140,48,31,C.teal,true);
 txt(s,'No barrier still has larger errors. Vocoder input mapping remains ambiguous.',66,606,1140,40,24,C.muted);
 note(s,'Coverage retains all 300 source TextGrids and all 240 published rows. We scored 180 rows, with 60 vocoder rows unavailable because two source candidates map to each target. No recording was selected by numeric similarity to the published values. Exact cleaned index files, historical GUI configuration and source/tier mapping remain unresolved. Babble and L2 contain 120 samples total and all 15 shared measures meet the declared mean-error targets. A source-only comment/tier audit found no contradiction among available Speaker A/B comments and numeric tier labels. These results check lexical statistic reproduction and provide no boundary-model superiority claim.',[U.lucid,U.osf,D.sources.lucid,path.join(ROOT,'outputs/validation/cv_hyperparameter_v4/lucid_comment_tier_summary.json')]);
}
// 12
{
 const s=slide('OCSC improved in cross-validation','Same participant folds, with each result pooled over all five');
 const ch=s.charts.add('bar',{position:{left:70,top:185,width:1120,height:388},categories:['Eligible count','MLU in words','Length SD'],series:[{name:'Earlier baseline',values:primary.map(m=>Number(f(lookup(D.old_cv,'ocsc',m)))),fill:C.old},{name:'Lower count loss',values:primary.map(m=>Number(f(lookup(low.summary,'ocsc',m)))),fill:C.teal}],barOptions:{direction:'column',grouping:'clustered',gapWidth:100},hasLegend:true,legend:{position:'bottom',overlay:false,textStyle:{fontSize:23,fill:C.ink}},xAxis:{textStyle:{fontSize:24,fill:C.ink},line:{fill:C.line,width:1}},yAxis:{title:{text:'Mean participant APE (%)',textStyle:{fontSize:21,fill:C.muted}},min:0,max:20,majorUnit:5,numberFormatCode:'0',textStyle:{fontSize:21,fill:C.muted},majorGridlines:{fill:C.line,width:0.8}},dataLabels:{showValue:true,position:'outEnd',numberFormatCode:'0.00',textStyle:{fontSize:23,fill:C.ink}},chartFill:'none',plotAreaFill:'none'});
 applyPresentationChartFont(ch,{fontFamily:FONT});nativeCharts.push(P.slides.items.length);
 txt(s,'Count 8.84%     MLUw 8.67%     Length SD 14.73%',66,591,1140,46,31,C.teal,true);
 txt(s,'CV tuning on 211 OCSC children. Length SD available in 558 of 567 sections.',66,647,1120,24,19,C.muted);
 note(s,'Earlier fixed-recipe five-fold OCSC errors are 9.16497845%, 9.09250479% and 14.88540339%. Lower count-loss setting errors are 8.83619768%, 8.66736307% and 14.73139231%. The lower count-loss setting has the best completed OCSC average across these three measures. Its comparison also includes the common expanded 12-margin inner calibration policy. The higher-learning-rate setting has lower SD error alone, but a worse OCSC three-metric mean. We do not combine per-metric minima from different models. These are CV tuning results, not new final-test results. Count/MLU agreement is reasonably good; the 5% project goals remain unmet.',[path.join(ROOT,'outputs/results/participant_cv_v1/cross_validation_results.json'),D.sources.closed]);
}
// 13
{
 const s=slide('NYU results across the completed CV settings','No completed alternative improved the combined selection objective');
 table(s,[['Configuration','Count error','MLUw error','Length-SD error'],['Baseline',...vals(base.summary,'nyu_emerson')],['Lower count loss',...vals(low.summary,'nyu_emerson')],['Higher learning rate',...vals(higher.summary,'nyu_emerson')]],205,[395,250,250,257],280,27);
 txt(s,'Baseline remains the best completed NYU three-metric mean.',66,533,1140,48,30,C.teal,true);
 txt(s,'NYU has 20 usable training children. Each heldout fold contains four.',66,598,1140,45,25,C.muted);
 note(s,'All NYU numbers on this slide are cross-validation tuning results on original-training participants. The baseline gives count 16.27527326%, MLUw 10.27484130%, SD 19.47100597%. The lower count-loss setting gives 17.48621240%, 10.99835804%, 19.42027262%. Higher learning rate gives 16.1239%, 12.1032%, 20.3088%. Each row corresponds to one consistent setting assessed on all five folds. Baseline has the best NYU and combined primary-metric mean. We retain individual gains, regressions and missing values rather than selecting the best metric from each row.',[D.sources.closed]);
}
// 14
{
 const s=slide('Existing production-test results','The retained model’s test scores remain separate from CV tuning results');
 table(s,[['Test source','Children / sections','Count error','MLUw error','Length-SD error'],['OCSC','45 / 111',...vals(D.test,'ocsc')],['NYU–Emerson','4 / 15',...vals(D.test,'nyu_emerson')]],185,[260,263,210,210,209],180,25);
 txt(s,'Vocabulary agreement is stronger',66,400,1140,40,30,C.teal,true);
 table(s,[['Test source','Total words','Different words','TTR','MATTR-50'],...['ocsc','nyu_emerson'].map(c=>[c==='ocsc'?'OCSC':'NYU–Emerson',...['TNW','NDW','TTR','MATTR'].map(m=>f(lookup(D.test,c,m))+'%')])],458,[260,223,249,210,210],174,24);
 txt(s,'All values are mean participant APE. OCSC has 2 unavailable SD and 7 unavailable MATTR sections.',66,645,1120,25,18,C.muted);
 note(s,'These are the already evaluated, preserved production-test results for the prior calibrated observed-turn model, margin 0.25. No new v4 final test or model activation occurred. All entries are mean participant APE. OCSC length SD has 2 missing sections and MATTR has 7, retained as unavailable. NYU has no missing sections for these metrics but only 4 test participants. This historically evaluated test cohort supports reproducible descriptive comparisons, not a claim of newly unexposed external confirmation. The table above is not the 8.84% OCSC CV result from a different fitted procedure.',[D.sources.test,path.join(ROOT,'outputs/results/benchmark/transcript_workflow.json')]);
}
// 15
{
 const s=slide('Why NYU count and MLU errors are larger','Descriptive differences support an explanation, not a single proven cause');
 table(s,[['Input characteristic','NYU–Emerson','OCSC'],['Median age, years','4.58','7.22'],['Median words per observed turn','2','5'],['Turns with at most 3 words','70.07%','40.56%'],['Unintelligibility tokens / 1,000 words','79.49','5.73'],['Training words','5,336','229,378']],182,[640,256,256],354,25);
 txt(s,'These input differences plausibly contribute to NYU’s larger count and MLU errors.',66,571,1140,65,26,C.teal,true);
 txt(s,'Original train + eval statistics. Word totals use train only. Ages come from CHAT headers.',66,645,1120,24,19,C.muted);
 note(s,'All descriptive comparisons use original train and eval, except training-word totals, which use train only. Ages come from local target-child CHAT @ID headers and describe our usable cohorts. Published corpus descriptions differ in age precision and coverage; header demographics should be reconciled before making causal age claims. NYU’s official source describes home Zoom recordings without external microphones and notes unintelligible material. Local native FREQ/MLU word-total gaps are 7.22% NYU versus 1.79% OCSC. Training masks 11.80% of NYU nonfinal gaps near unintelligibility versus 1.08% OCSC. Sample size, task, participant ages and transcription conventions vary together. We cannot identify their separate causal contributions. Earlier fixed-recipe NYU boundary F1 (88.81%) exceeded OCSC (87.22%), so the difficulty is metric-specific.',[path.join(ROOT,'outputs/validation/cv_hyperparameter_v4/development_input_profile.json'),path.join(ROOT,'outputs/validation/cv_hyperparameter_v4/development_age_profile.json'),path.join(ROOT,'outputs/results/cv_hyperparameter_v4/NYU_EXPLANATION.md'),U.nyu]);
}
// 16
{
 const s=slide('Final state of this phase');
 row(s,'LUCID','Strong published lexical agreement in babble and L2.\nNo-barrier word-count error remains 16.60%. Vocoder is unresolved.',187,230);
 row(s,'OCSC','Modest CV improvement to 8.84% count and 8.67% MLUw error.\nLength-SD error remains 14.73%.',321,230);
 row(s,'NYU','Count and MLU errors remain larger than OCSC’s.\nThe younger, smaller and less intelligible sample is a plausible contributor.',455,230);
 txt(s,'Further experiments stopped. Existing production and all cohorts remain preserved.',66,615,1140,35,24,C.muted);
 note(s,'The completed evidence supports good lexical agreement and stronger OCSC count/MLU performance than NYU. It does not establish clinical validity or universal accuracy across corpora. Baseline had the best completed combined CV objective, and lower count loss had the best OCSC three-metric CV objective. The fourth configuration remains incomplete and excluded. All fifteen completed assessments passed integrity checks. No new final model was deployed.',[D.sources.closed,path.join(ROOT,'outputs/validation/cv_hyperparameter_v4/stopped_search_integrity_audit.json')]);
}
// 17
{
 const s=slide('Next steps in the existing plan','Public transcript validation now provides a fixed reference for the next phase');
 const steps=[['Freeze this research version','Archive results and document the CV/test distinction in the paper and poster.'],['Return to the original ten participants','Evaluate corrected transcripts against human utterance labels with this fixed pipeline.'],['Reintroduce MOSS and speaker selection','Measure transcription and speaker errors separately from boundaries and metric calculations.'],['Extend validation to new recordings','Assess generalization and human review time with predefined evaluation rules.']];
 for(let i=0;i<steps.length;i++){
  const y=191+i*117;txt(s,String(i+1),66,y,70,60,40,C.teal,true);
  txt(s,steps[i][0],158,y,1040,42,29,C.ink,true);
  txt(s,steps[i][1],158,y+48,1040,61,24,C.muted);
 }
 note(s,'The staged plan validates transcripts and metric calculations first, then returns to the original 10 participants before evaluating the audio path. The original pilot includes 37 recordings from autistic participants aged 8–17 years, with manual focused samples from 14 recordings. MOSS-Transcribe-Diarize 0.9B supplies the planned audio transcript/speaker path with brief human target-speaker identification. Re-evaluate corrected wording and human boundaries separately. Preserve participant roles for any future training or tuning. New recordings are needed for genuinely unexposed generalization evidence and measurement of human review time. Updating the paper/poster is a proposed next step, not a claim those files were edited in this task.',[path.join(ROOT,'conference_paper/conference_paper.docx'),path.join(ROOT,'thompson_center_poster/Thompson_Center_Abstract.docx'),path.join(ROOT,'SESSION_HANDOFF.md')]);
}
// 18
{
 const s=slide('Sources and reproducibility');
 const refs=[['LUCID','Baese-Berk et al. (2021), Frontiers in Psychology\nPublished TAALED values at OSF',U.lucid],['OCSC','Wagner et al. (2025), Speech Communication 170, 103206\nTalkBank CHILDES, DOI 10.21415/60TV-D669',U.ocscPaper],['NYU–Emerson','Steen et al. (2026), Meeting on Language in Autism\nTalkBank ASDBank, DOI 10.21415/VT7P-YF83',U.nyu],['Tools and methods','Microsoft MiniLM, spaCy, official CLAN and grouped CV\nSaved cohort manifests, protocols, predictions and audits',U.cv]];
 for(let i=0;i<refs.length;i++){
  const y=176+i*116;txt(s,refs[i][0],66,y,240,58,28,C.teal,true);
  const lines=refs[i][1].split('\n');
  const secondLinks=[U.osf,U.ocsc,'https://doi.org/10.21415/VT7P-YF83',U.groups];
  for(let j=0;j<2;j++){
   const text=txt(s,lines[j],340,y+j*32,870,40,25);
   text.text.getRange(0,lines[j].length).link={uri:j===0?refs[i][2]:secondLinks[i],isExternal:true};
  }
 }
 note(s,'Corpus references\nBaese-Berk et al. (2021). Lexical Diversity, Lexical Sophistication, and Predictability for Speech in Multiple Listening Conditions. Frontiers in Psychology, 12, 661415.\nWagner et al. (2025). The Ohio Child Speech Corpus. Speech Communication, 170, 103206.\nSteen, K., Buonocore, T., Luyster, R., Sancimino, C., & Arunachalam, S. (2026, March). Introducing the NYU-Emerson Corpus: Transcripts of parent-child interaction with young autistic children. Poster presentation, Meeting on Language in Autism, Atlanta, GA.\n\nExternal sources\n'+Object.entries(U).map(([k,v])=>k+': '+v).join('\n')+'\n\nLocal result lineage\n'+Object.entries(D.sources).map(([k,v])=>k+': '+v).join('\n')+'\n\nThe full retained report is outputs/results/transcript_validation/REPORT.md. Completed CV closeout, source profiles, model architecture and stopped-run status are versioned under cv_hyperparameter_v4.');
}

await fs.mkdir(path.join(BUILD,'renders'),{recursive:true});
for(let i=0;i<P.slides.items.length;i++){
 const s=P.slides.items[i];const png=await P.export({slide:s,format:'png',scale:1});
 await fs.writeFile(path.join(BUILD,'renders',`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await png.arrayBuffer()));
 const layout=await s.export({format:'layout'});await fs.writeFile(path.join(BUILD,'renders',`slide-${String(i+1).padStart(2,'0')}.json`),await layout.text());
 console.log('Rendered',i+1);
}
const candidate=path.join(BUILD,'candidate.pptx');await (await PresentationFile.exportPptx(P)).save(candidate);
await fs.writeFile(path.join(BUILD,'deck.proto.json'),JSON.stringify(P.toProto()));
const result=await finalizePresentation({workspaceDir:ROOT,candidatePath:candidate,finalPath:FINAL,
 pythonExecutable:path.join(RUNTIME,'python/bin/python3'),
 integrityValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',...[...new Set(nativeTables)].flatMap(n=>['--require-native-table-slide',String(n)])],
 requiredNativeTableOwnerSlides:[...new Set(nativeTables)],requiredNativeChartOwnerSlides:nativeCharts,
 fontPolicy:{basis:'design',families:[FONT]},verifyArtifactToolImport:true,
 materializeLiteralChartWorkbooks:true,
 receiptPath:path.join(BUILD,'validation_final.json')});
console.log('FINAL',JSON.stringify(result));
