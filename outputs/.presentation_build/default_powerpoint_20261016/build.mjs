import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import {spawnSync} from 'node:child_process';
import {pathToFileURL} from 'node:url';
import {Presentation,PresentationFile} from '@oai/artifact-tool';

const ROOT='/Users/nafeekazi/Desktop/PhD/speech_pipeline_code';
const BUILD=path.join(ROOT,'outputs/.presentation_build/default_powerpoint_20261016');
const SKILL='/Users/nafeekazi/.codex/plugins/cache/openai-primary-runtime/presentations/26.1013.11914/skills/presentations';
const RT='/Users/nafeekazi/.cache/codex-runtimes/codex-primary-runtime/dependencies';
process.env.RUNTIME_NODE=path.join(RT,'node/bin/node');
process.env.RUNTIME_NODE_MODULES=path.join(RT,'node/node_modules');
process.env.RUNTIME_PYTHON=path.join(RT,'python/bin/python3');
const {finalizePresentation,makeNativeBulletParagraphs}=await import(pathToFileURL(path.join(SKILL,'container_tools/artifact_tool_utils.mjs')).href);
const local=p=>path.join(ROOT,p);
const read=async p=>JSON.parse(await fs.readFile(local(p),'utf8'));
const CHAT=await read('outputs/results/transcript_validation/chat_calibrated_results.json');
const LUCID=await read('outputs/results/transcript_validation/lucid_results.json');
const CV=await read('outputs/results/cv_hyperparameter_v4/stopped_search_summary.json');
const PILOT=await read('outputs/results/moss_pilot/evaluation.json');
const MANIFEST=await read('outputs/validation/cohorts_v1/manifest.json');
const P=Presentation.create({slideSize:{width:1280,height:720}});
P.theme.defaultFont='Aptos';
const BODY='Aptos',HEADING='Aptos Display';
const NAME={ohio:'Ohio Child Speech Corpus',parent:'New York University and Emerson College',lucid:'London UCL Clear Speech in Interaction'};
const U={ohio:'https://talkbank.org/childes/access/Eng-NA/OCSC.html',parent:'https://talkbank.org/asd/access/English/NYU-Emerson.html',lucid:'https://valeriehazan.com/index.php/lucid-corpus-london-ucl-clear-speech-in-interaction/',paper:'https://doi.org/10.3389/fpsyg.2021.661415',osf:'https://osf.io/dfhpu/?view_only=49d95d90424941da82217a239ab7450c',clan:'https://talkbank.org/manuals/CLAN.pdf',taaled:'https://www.linguisticanalysistools.org/taaled.html',model:'https://huggingface.co/microsoft/MiniLM-L12-H384-uncased'};
const R={chat:local('outputs/results/transcript_validation/chat_calibrated_results.json'),lucid:local('outputs/results/transcript_validation/lucid_results.json'),cv:local('outputs/results/cv_hyperparameter_v4/stopped_search_summary.json'),cohorts:local('outputs/validation/cohorts_v1/manifest.json'),provenance:local('outputs/results/transcript_validation/LUCID_METRIC_PROVENANCE.md'),explain:local('outputs/results/cv_hyperparameter_v4/NYU_EXPLANATION.md'),methods:local('outputs/results/cv_hyperparameter_v4/METHODS.md'),pilot:local('outputs/results/moss_pilot/evaluation.json')};
const f=n=>Number(n).toFixed(2)+'%';
const test=(c,m)=>CHAT.split_summaries.test.find(r=>r.corpus===c&&r.metric===m).mean_participant_ape;
const cv=(cfg,c,m)=>CV.completed_candidates.find(r=>r.configuration===cfg).summary.find(r=>r.corpus===c&&r.metric===m).mean_participant_ape;
const lex=(c,m)=>LUCID.summary.find(r=>r.condition===c&&r.metric===m).mean_participant_ape;
const texts=[],tableOwners=[],coverage={};
const master=P.masters.add('Office Theme');master.background.fill='#FFFFFF';
const layout=P.layouts.add('Title and Content');layout.setParentLayoutId(master.id);
layout.placeholders.add({type:'title',index:0,geometry:'textbox',position:{left:64,top:40,width:1152,height:75},fill:'none',line:{fill:'none',width:0}});
function put(s,value,x,y,w,h,size=30,bold=false,font=BODY){
 const a=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 a.text=value;a.text.style={typeface:font,fontSize:size,color:'#000000',bold,alignment:'left',verticalAlignment:'top',wrap:'square',autoFit:'none',insets:0};
 texts[P.slides.items.indexOf(s)].push(value);return a;
}
function slide(title,sub='',key=''){
 const s=P.slides.add();s.setLayout(layout);texts.push([]);
 const a=s.placeholders.getItem('title');a.text=title;a.text.style={typeface:HEADING,fontSize:44,color:'#000000',bold:false,insets:0,autoFit:'none'};
 texts.at(-1).push(title);if(sub)put(s,sub,66,123,1145,65,27);
 if(key)coverage[key]=P.slides.items.length;return s;
}
function notes(s,value,refs=[]){s.speakerNotes.text=value+'\n\nSources\n'+refs.join('\n');}
function bullets(s,items,{y=205,h=420,size=31,space=15,x=76,w=1124}={}){
 const a=put(s,'',x,y,w,h,size);a.text=makeNativeBulletParagraphs(items,{marginLeftPoints:22,hangingPoints:11,spaceAfterPoints:space});
 a.text.style={typeface:BODY,fontSize:size,color:'#000000',autoFit:'none',insets:0};texts[P.slides.items.indexOf(s)].push(...items);return a;
}
function table(s,values,{y=205,widths=[540,612],height=350,font=27}={}){
 const t=s.tables.add({rows:values.length,columns:values[0].length,left:64,top:y,width:1152,height,columnWidths:widths,values});
 t.styleOptions={headerRow:true,bandedRows:false};t.borders.assign({fill:'#BFBFBF',width:0.7,style:'solid'});
 for(let r=0;r<values.length;r++){t.rows[r].height=height/values.length;for(let c=0;c<values[r].length;c++){
  const cell=t.getCell(r,c);cell.fill=r===0?'accent1':'#FFFFFF';cell.text.style={typeface:BODY,fontSize:font,bold:r===0,color:r===0?'#FFFFFF':'#000000',verticalAlignment:'middle',insets:12};
 }}tableOwners.push(P.slides.items.length);texts[P.slides.items.indexOf(s)].push(...values.flat());return t;
}
function flow(s,labels){
 const boxes=labels.map((value,i)=>{
  const a=s.shapes.add({geometry:'rect',position:{left:64+i*237,top:235,width:204,height:170},fill:'#FFFFFF',line:{fill:'#000000',width:1.3}});
  a.text=value;a.text.style={typeface:BODY,fontSize:26,color:'#000000',alignment:'center',verticalAlignment:'middle',insets:12,autoFit:'none'};texts[P.slides.items.indexOf(s)].push(value);return a;
 });
 for(let i=0;i<4;i++)s.shapes.connect(boxes[i],boxes[i+1],{kind:'straight',fromSide:'right',toSide:'left',line:{fill:'#000000',width:1.7},tail:{type:'triangle',width:'med',length:'med'}});
}
// 1
{
 const s=slide('Speech pipeline','Datasets, methods, results and next steps','cover');
 s.placeholders.getItem('title').text.style={typeface:HEADING,fontSize:58,color:'#000000',insets:0};
 put(s,'London UCL Clear Speech in Interaction\nOhio Child Speech Corpus\nNew York University and Emerson College corpus',66,250,1140,210,36);
 put(s,'Research update, 16 October 2026',66,565,1140,55,28);
 notes(s,'Current transcript comparisons, earlier single-recording audio feasibility, and planned multi-participant audio evaluation. Existing model and research results are retained.',[R.chat,R.lucid,R.cv,R.pilot]);
}
// 2
{
 const s=slide('The three datasets','Source participants and the parts we used','dataset_overview');
 table(s,[['Dataset','Source participants','Our transcript comparison'],[NAME.lucid,'40 adults','300 human transcripts\n180 of 240 reference samples scored'],[NAME.ohio,'303 children','301 children\n803 selected task sections'],[NAME.parent+' corpus','30 autistic children','28 children\n92 usable task sections']],{widths:[480,230,442],height:390,font:26});
 notes(s,'Source participants differ from usable evaluation participants. LUCID TextGrids are time-aligned human transcripts. The retained local lexical table has40 speaker IDs and240 rows. Its author-source provenance still needs independent reconfirmation. Counts and exclusions follow the immutable cohort manifest.',[U.lucid,U.ohio,U.parent,R.cohorts,R.provenance]);
}
// 3
{
 const s=slide('London UCL Clear Speech in Interaction','LUCID: adult conversations collected by Rachel Baker and Valerie Hazan at UCL','lucid_dataset');
 bullets(s,['40 adults aged 18–29 spoke southern British English.','Pairs described pictures to find differences without seeing each other’s picture.','Listening conditions included normal conversation, background voices, vocoded sound and a partner learning English.','The corpus contains audio and human transcripts with words, timestamps and speaker labels.','We chose it to check our vocabulary calculations against existing reference values.'],{size:29,h:425,space:14});
 notes(s,'Corpus materials come from SpeechBox at Northwestern University. Full corpus also includes sentence reading and picture naming, which we did not use for this dialogue comparison. Vocoded listening simulates aspects of cochlear-implant listening. The downloaded TextGrid README explains stereo Speaker A/B tiers and the sole target-native-speaker tier for L2.',[U.lucid,local('dataset/public/lucid_published/source_docs/LUCID - TextGrid README.txt')]);
}
// 4
{
 const s=slide('LUCID: the files and reference values','','lucid_files_and_provenance');
 bullets(s,['Downloaded human transcripts: 300 Praat TextGrid files from SpeechBox.','Retained reference file: TAALED_results.csv, recorded as the paper’s OSF export. It has 240 samples and 40 speaker IDs.','Each reference condition contains 20 speakers and three scenes per speaker.','We scored 180 samples. The other 60 have ambiguous vocoder recording matches.','The local file matches saved acquisition metadata. Independent confirmation against the author-hosted file is still pending.'],{y:165,size:29,h:465,space:14});
 notes(s,'Separate corpus creators from later lexical analysts: Melissa Baese-Berk and colleagues reused LUCID in2021 and shared TAALED/TAALES exports and R analysis code. Footnote2 and the Data availability statement link OSF. Current download attempts failed; local metadata alone cannot independently authenticate the historical transfer. Preserve all files and numerical results as provisional for publication. There are60 babble,60 L2,60 noBarrier and120 vocoder TextGrids. No sample is chosen by closeness to a reference score.',[U.paper,U.osf,R.provenance,local('dataset/public/lucid_published/source.json')]);
}
// 5
{
 const s=slide(NAME.ohio,'Wagner and colleagues, 2025. Available through TalkBank CHILDES.','ohio_dataset');
 bullets(s,['The published corpus contains 303 children, ages 4–9.','Researchers recorded speech in a science-museum language lab in Ohio.','We use the children’s human CHAT transcripts.','We selected picture descriptions and “how to” explanations: 301 children and 803 task sections.','This provides a larger child-speech reference for testing utterance measures.'],{size:30,h:415});
 notes(s,'Selected gems: ExpPictures, Howto, DescriptivePictures. Reading, alphabet, number and Wug tasks remain outside the selected discourse preparation. Two source children have no eligible discourse section. Source media include lapel-microphone audio. Published age range is4–9; local CHAT age-header minimum differs, so development medians are explicitly header-derived.',[U.ohio,R.cohorts,R.explain,'https://doi.org/10.1016/j.specom.2025.103206']);
}
// 6
{
 const s=slide(NAME.parent,'Parent–child interaction corpus, available through TalkBank ASDBank','parent_dataset');
 bullets(s,['The released corpus contains 30 autistic children. Recruitment targeted ages 3 to under 6.','Families completed play and conversation, mainly over Zoom at home.','We use the child’s human CHAT transcript from free play, pretend play and sensory play.','28 children contribute 92 usable task sections. Two have too little eligible speech for boundary scoring.','We chose it to test transfer to young autistic children. The source also provides video.'],{size:29,h:440,space:14});
 notes(s,'Official corpus title is NYU–Emerson Corpus. Expanded institution names are used on slides at the user’s request. Citation: Steen, Buonocore, Luyster, Sancimino and Arunachalam2026. The current release contains30 autistic children, not the larger parent-study sample. Two excluded sources remain assigned to original train but supply no prepared records.',[U.parent,R.cohorts]);
}
// 7
{
 const s=slide('Train, evaluation and test participants','All sections from a child stay in that child’s original cohort.','participant_splits');
 table(s,[['Dataset','Train','Evaluation','Test'],[NAME.ohio,'211','45','45'],[NAME.parent,'20','4','4'],['Total usable children','231','49','49']],{widths:[570,194,194,194],height:310,font:27});
 bullets(s,['Train fits model weights. Evaluation selects settings. Test measures a frozen choice.','Both child datasets contribute to the training cohort. LUCID stays outside boundary-model training.'],{y:555,h:120,size:27,space:11});
 notes(s,'Cohorts are participant-independent and immutable. Original NYU train assigns22 children, of whom20 have eligible sections. Historical test exposure remains documented; these existing test values are descriptive evaluation, not a claim of newly unexposed confirmation. No participant reshuffling occurred.',[R.cohorts,local('outputs/validation/cohorts_v1/AUDIT.md')]);
}
// 8
{
 const s=slide('Task sections in each cohort','A child can contribute several sections from the same recording.','section_splits_and_used_tasks');
 table(s,[['Dataset','Train','Evaluation','Test'],[NAME.ohio,'567','125','111'],[NAME.parent,'63','14','15'],['Total sections','630','139','126']],{widths:[570,194,194,194],height:310,font:27});
 bullets(s,['Picture descriptions and explanations supply the selected Ohio child sections. Reading, alphabet, number and Wug tasks stay outside this comparison.','Parent–child play supplies the other sections. All four excluded source files remain documented.'],{y:555,h:130,size:25,space:10});
 notes(s,'803 Ohio sections plus92 parent–child sections equals895. Exclusions: two Ohio children lack selected discourse tasks; two parent–child sources lack eligible boundary gaps. They are unavailable for this comparison, never zero-error results. All333 raw CHAT files remain preserved.',[R.cohorts]);
}
// 9
{
 const s=slide('Our current transcript pipeline','','current_pipeline');
 flow(s,['Human\ntranscript','Keep the\nparticipant’s words','Our model predicts\nutterance endings','Calculate\nlanguage measures','Save\nresults']);
 bullets(s,['The model receives words and observed speaker turns.','Human utterance endings stay in the reference used for comparison.'],{y:470,h:135,size:30,space:18});
 notes(s,'Same five-step abstraction as the simplified Excalidraw pipeline. Detailed implementation: lexical calculations use the historical TAALED compatibility branch; predicted utterances export faithful CHAT and native CLAN computes surface measures. The retained workflow is turn_ml, margin0.25. Human utterance cuts are withheld at inference. Consecutive same-speaker human rows merge into observed turns; that context remains available.',[local('outputs/flowcharts/all_pipelines_simple.excalidraw'),local('outputs/results/cv_hyperparameter_v4/PIPELINE.md')]);
}
// 10
{
 const s=slide('Our model and the software it uses','','model_and_tools');
 table(s,[['Component','Role'],['Microsoft MiniLM','Pretrained language encoder that represents words in context'],['Our custom model','A learned rule embedding and classifier predict utterance endings'],['Task training','We fine-tune MiniLM and our custom components together'],['spaCy','Provides grammatical cues'],['CLAN','Calculates measures from CHAT transcripts'],['TAALED conventions','Guide vocabulary processing and calculation rules']],{y:165,widths:[360,792],height:450,font:27});
 notes(s,'The model is a pretrained encoder with task-specific learned components, fine-tuned jointly. Architecture: MiniLM-L12-H384-uncased,16-dimensional rule embedding and128-unit GELU/dropout gap classifier. Historical TAALED conventions are a fixed software profile, not a model trained on LUCID. Current spaCy software remains in use; historical environment was not reconstructed exactly.',[U.model,U.clan,U.taaled,local('speech_pipeline/segmentation.py'),local('speech_pipeline/lexical_compat.py')]);
}
// 11
{
 const s=slide('The language measures we compare','Percentage error describes the difference from the reference. Lower is better.','metric_definitions');
 table(s,[['Measure','Meaning'],['Word counts','Total words and different words'],['Type–token ratio (TTR)','Different words divided by total words'],['Moving-average TTR (MATTR-50)','Vocabulary variety in moving 50-word windows'],['Eligible utterance count','Utterances included in CLAN’s length calculation'],['Mean length in words (MLUw)','Average words per eligible utterance'],['Length standard deviation','How much utterance length varies']],{widths:[500,652],height:380,font:24});
 bullets(s,['We calculate absolute percentage error per sample, then average within participants and across participants.'],{y:620,h:65,size:25,space:0});
 notes(s,'APE=100*abs(predicted-reference)/abs(reference). Undefined/zero denominators remain missing. Native FREQ and MLU have different word eligibility. Eligible count differs from all verbal units. LUCID uses per-recording values, then speaker averaging. Our percentage errors and engineering targets are our evaluation outputs, not paper-reported model accuracy.',[R.chat,R.lucid,U.clan]);
}
// 12
{
 const s=slide('What we tried','','experiments');
 bullets(s,['Aligned vocabulary processing with historical TAALED dictionary, word-category and pause rules.','Used native CLAN calculations to respect CHAT word and utterance eligibility.','Compared prediction from whole paragraphs, prediction within observed speaker turns, and rules.','Tested fixed settings for deciding where utterances end on development participants.','Compared learning rates and count-loss weights using five-fold participant cross-validation.'],{y:170,h:455,size:30,space:16});
 notes(s,'Earlier production selection favored observed-turn ML, margin0.25. An initial batching inconsistency was caught and calibration reran through original preprocessing with exact parity checks. A single-evaluation ensemble branch was superseded by the participant-CV search. The fourth training configuration stopped in fold1. The lexical and native-scoring changes did not rewrite source words. No new final fit or activation occurred after the user stopped further experiments.',[local('outputs/results/transcript_validation/REPORT.md'),R.methods,R.cv]);
}
// 13
{
 const s=slide('Five-fold cross-validation','Model-setting comparisons use only the original 231 usable training children.','cross_validation');
 bullets(s,['Each configuration has five fits, with different children held out each time.','Separate inner children select the checkpoint and boundary setting within each fold.','We pool held-out predictions across all five folds to compare configurations.'],{y:195,h:200,size:28,space:12});
 table(s,[['Completed configuration','Combined error'],['Baseline learning rate and count loss',f(CV.completed_candidates[0].objective)],['Lower count-loss weight',f(CV.completed_candidates[1].objective)],['Higher learning rate',f(CV.completed_candidates[2].objective)]],{y:415,widths:[850,302],height:190,font:26});
 bullets(s,['Three configurations completed all five folds. A fourth stopped in fold 1 and stays outside the ranking.'],{y:630,h:60,size:25,space:0});
 notes(s,'Each fold has147–148 fit children,37 inner children and46–47 CV-heldout children. Original evaluation/test participants stay outside recipe comparison. Same folds across configurations. Objective equally averages count, MLUw and length-SD participant APE over the two child corpora. This is a tuning objective, not an unbiased final accuracy estimate. Baseline is best completed combined score13.1602309. Fold means/SD:13.16256±2.47768,13.36125±2.68190,13.52578±3.09875. The baseline folds were reusable only after initialization/role/recipe checks. Three configurations correspond to15 completed assessments. No new final fit, test or model activation followed.',[R.cv,R.methods]);
}
// 14
{
 const s=slide('LUCID comparison (provisional)','','lucid_comparison_flow');
 flow(s,['Human\nTextGrids','Keep the target\nspeaker’s words','Calculate\nvocabulary measures','Compare with saved\nreference values','Report the\ndifferences']);
 bullets(s,['We match each sample by speaker, listening condition and scene.','This comparison checks vocabulary calculations from human transcripts.'],{y:470,h:140,size:30,space:18});
 notes(s,'Reference provenance remains pending as described in the source-data slide. Exact historical cleaned indexN.txt inputs and full software configuration remain unavailable. LUCID has no gold utterance/MLU reference and supplies no boundary training. Do not choose a source recording by numeric agreement. Current coverage180 scored/60 vocoder ambiguous.',[R.provenance,R.lucid]);
}
// 15
{
 const s=slide('LUCID: paper metrics and exported data','Baese-Berk et al. (2021): TAALED for diversity, TAALES for sophistication.','lucid_reference_metrics');
 table(s,[['Main article','Field in the local reference CSV'],['Table 1: mean word count','basic_ntokens'],['Figure 1: type–token ratio','simple_ttr_aw'],['Figure 2: moving-average TTR','mattr50_aw'],['Figure 3: HD-D42 diversity measure','hdd42_aw']],{widths:[650,502],height:285,font:26});
 bullets(s,['Our 15-field comparison also includes additional counts, densities and diversity measures in the export.','Our original-MTLD field differs from Figure 4’s moving-average version.'],{y:535,h:120,size:27,space:12});
 notes(s,'TAALED=Tool for the Automatic Analysis of Lexical Diversity, developed by Kristopher Kyle with Scott Crossley and Scott Jarvis. Baese-Berk etal2021 used TAALED diversity and TAALES sophistication, and shared an OSF export and R analysis script. The local reference identity requires independent reconfirmation before publication. We did not reproduce TAALES outputs, figure4 moving-average MTLD, mixed-model coefficients or p-values. Compared15 columns: basic_ntokens,basic_ntypes,basic_ncontent_tokens,basic_ncontent_types,basic_nfunction_tokens,basic_nfunction_types,lexical_density_tokens,lexical_density_types,simple_ttr_aw,log_ttr_aw,maas_ttr_aw,mattr50_aw,msttr50_aw,hdd42_aw,mtld_original_aw.',[U.paper,U.osf,U.taaled,R.provenance]);
}
// 16
{
 const s=slide('LUCID results (provisional)','Mean participant absolute percentage error against the retained reference values','lucid_results');
 table(s,[['Condition','Scored','Word count','TTR','MATTR-50','HD-D42'],...['babble','L2','noBarrier'].map((c,i)=>[['Babble','English-learner partner','No barrier'][i],'60 / 60',...['basic_ntokens','simple_ttr_aw','mattr50_aw','hdd42_aw'].map(m=>f(lex(c,m)))]),['Vocoder','0 / 60','Unavailable','Unavailable','Unavailable','Unavailable']],{widths:[250,150,190,170,201,191],height:325,font:25});
 bullets(s,['Babble and the English-learner condition show close agreement. No-barrier word counts remain less consistent.','Vocoder recording matches are unresolved. Independent author-source confirmation is also pending.'],{y:555,h:130,size:26,space:12});
 notes(s,'The English-learner partner condition is coded L2 in the reference file. Each scored condition includes60 samples from20 speakers. All240 local reference rows remain visible in coverage. All15 shared means meet project engineering targets in babble/L2, but those thresholds are internal goals, not clinical standards. No-barrier discrepancies may include source matching and preparation differences. These are lexical transcript comparisons, not ASR or boundary-model accuracy.',[R.lucid,R.provenance]);
}
// 17
{
 const s=slide(NAME.ohio,'How we compared predicted utterances with human transcripts','ohio_comparison_flow');
 flow(s,['Human\ntranscript','Child’s words\nwithout utterance\nbreaks','Our model predicts\nutterance endings','Calculate\nlanguage measures','Compare with the\nhuman reference']);
 bullets(s,['The human reference keeps the original utterance coding.','Native CLAN calculates the same measures for both versions.'],{y:470,h:140,size:30,space:18});
 notes(s,'301 usable children and803 selected discourse sections. Retained test contains45 children and111 sections. Gold utterance ends remain outside inference. Observed speaker turns remain available. We preserve source words; plain predicted CHAT does not fabricate retrace, error, morphology, grammar or maze annotations.',[R.chat,R.cohorts,U.clan]);
}
// 18
{
 const s=slide('Ohio Child Speech Corpus: test results','Retained model. 45 children and 111 task sections.','ohio_test_results');
 const ms=[['Total words','TNW'],['Different words','NDW'],['Type–token ratio','TTR'],['Moving-average TTR','MATTR'],['Eligible utterance count','eligible_mlu_utterances'],['Average utterance length in words','MLUw'],['Utterance-length standard deviation','length_sd']];
 table(s,[['Measure','Mean participant error'],...ms.map(([label,key])=>[label,f(test('ocsc',key))])],{widths:[800,352],height:390,font:27});
 bullets(s,['Vocabulary agreement is strong. Utterance count and average length remain less precise.'],{y:625,h:60,size:26,space:0});
 notes(s,'Existing retained workflowturn_ml/margin0.25, not the later CV fold models. Available sections111 for word counts,TTR,eligible count,MLUw,104 forMATTR,109 forSD. All45 participants contribute. Missing values stay missing. Engineering goals5% for count/MLU and10% forSD remain unmet. Historical test exposure remains documented.',[R.chat,R.cohorts]);
}
// 19
{
 const s=slide(NAME.parent,'How we compared predicted utterances with human transcripts','parent_comparison_flow');
 flow(s,['Human\ntranscript','Child’s words\nwithout utterance\nbreaks','Our model predicts\nutterance endings','Calculate\nlanguage measures','Compare with the\nhuman reference']);
 bullets(s,['We use the same comparison procedure as for the Ohio Child Speech Corpus.','The existing test cohort contains four children and 15 sections.'],{y:470,h:140,size:30,space:18});
 notes(s,'Official nameNYU–Emerson Corpus.28 usable children/92 selected sections. Two assigned training children lack eligible boundary gaps. Human coding and CLAN eligibility remain in the reference; predictions do not invent missing annotation information. The same frozen retained model supplies the existing test result.',[R.chat,R.cohorts,U.clan]);
}
// 20
{
 const s=slide('Parent–child corpus: test results',NAME.parent+'. Retained model, four children and 15 sections.','parent_test_results');
 const ms=[['Total words','TNW'],['Different words','NDW'],['Type–token ratio','TTR'],['Moving-average TTR','MATTR'],['Eligible utterance count','eligible_mlu_utterances'],['Average utterance length in words','MLUw'],['Utterance-length standard deviation','length_sd']];
 table(s,[['Measure','Mean participant error'],...ms.map(([label,key])=>[label,f(test('nyu_emerson',key))])],{widths:[800,352],height:390,font:27});
 bullets(s,['Vocabulary agreement remains close. Count and average-length errors are higher, with a small test sample.'],{y:625,h:65,size:26,space:0});
 notes(s,'All15 sections and4 children contribute to the displayed measures. NYU length-SD error13.70% is slightly lower than Ohio14.24%; the deficit is not universal across metrics. Existing tests and later CV tuning represent different fitted models and evaluation roles. No new final fit/test followed the CV search.',[R.chat,R.cohorts]);
}
// 21
{
 const s=slide('Ohio Child Speech Corpus: model tuning','Five-fold cross-validation results, separate from the test results','ohio_cv_changes');
 const vals=cfg=>['eligible_mlu_utterances','MLUw','length_sd'].map(m=>f(cv(cfg,'ocsc',m)));
 table(s,[['Training configuration','Count error','Length error','Length SD error'],['Earlier fixed recipe','9.16%','9.09%','14.89%'],['Current CV baseline',...vals('lr2_count1')],['Lower count-loss weight',...vals('lr2_count025')]],{widths:[540,204,204,204],height:280,font:27});
 bullets(s,['Lower count-loss weight modestly improved the count and average-length errors.','The later cross-validation search did not produce a new final model or test result.'],{y:530,h:135,size:29,space:15});
 notes(s,'Count refers to eligible MLU utterances. Length error refers toMLUw. Earlier fixed-recipe v1 values are historical and use that protocol. The paired within-v4 baseline versus lower-count comparison uses the same folds and selection policy. Lower-count configuration has the best completed Ohio three-metric mean. Higher learning rate gives9.08/9.06/14.44 but a worse three-metric mean; no minimum-cell combination is claimed. Production remains retained.',[R.cv,local('outputs/results/cv_hyperparameter_v4/REPORT.md')]);
}
// 22
{
 const s=slide('Parent–child corpus: model tuning',NAME.parent+'. Five-fold cross-validation tuning results.','parent_cv_changes');
 const vals=cfg=>['eligible_mlu_utterances','MLUw','length_sd'].map(m=>f(cv(cfg,'nyu_emerson',m)));
 table(s,[['Training configuration','Count error','Length error','Length SD error'],['Current CV baseline',...vals('lr2_count1')],['Lower count-loss weight',...vals('lr2_count025')]],{widths:[540,204,204,204],height:230,font:27});
 bullets(s,['The lower count-loss setting worsened count and average-length errors here.','The baseline had the best combined score across both child corpora.','We retained the existing production workflow when further experiments stopped.'],{y:490,h:185,size:29,space:15});
 notes(s,'Each row pools five different heldout models under one consistent recipe. These are tuning values, not heldout final-test improvements. Baseline is best combined and best NYU three-metric mean among completed settings. Fourth configuration is excluded entirely. No new final fit, test or model activation followed.',[R.cv,R.methods]);
}
// 23
{
 const s=slide('Why the child-corpus results differ','Measured differences in training and evaluation inputs','explanations');
 table(s,[['Input characteristic','New York University\nand Emerson College','Ohio Child\nSpeech Corpus'],['Median child age, years','4.58','7.22'],['Median words per speaker turn','2','5'],['Turns with at most three words','70%','41%'],['Unintelligibility markers per 1,000 words','79.49','5.73']],{widths:[600,276,276],height:350,font:25});
 bullets(s,['The parent–child corpus provides shorter contexts, more missing speech and much less training text.','Age, task and coding vary together. These observations do not establish age as the cause.'],{y:590,h:100,size:25,space:10});
 notes(s,'Profiles use original train+eval only. Training words5336 vs229378; native reference FREQ/MLU word-total gaps7.22% vs1.79%. NYU has more masked supervision near unintelligibility. Corpus-balanced sampling does not create additional independent children. BoundaryF1 in preceding CV wasNYU88.81% vsOhio87.22%, so the difficulty is metric-specific. LUCID checks a different lexical component; no-barrier discrepancies may involve source/preparation differences and vocoder matching remains ambiguous.',[R.explain,local('outputs/validation/cv_hyperparameter_v4/development_input_profile.json'),local('outputs/validation/cv_hyperparameter_v4/development_age_profile.json')]);
}
// 24
{
 const s=slide('Earlier audio feasibility pilot','One Ohio Child Speech Corpus picture-description section, 300 seconds','existing_audio_pilot');
 const m=k=>PILOT.metrics.find(r=>r.method==='latest_minilm_native_clan'&&r.metric===k).ape;
 table(s,[['Measure','Error on this recording'],['Child word error rate',f(PILOT.target_wer_percent)],['Eligible utterance count',f(m('eligible_mlu_utterances'))],['Average utterance length in words',f(m('MLUw'))],['Utterance-length standard deviation',f(m('length_sd'))]],{widths:[800,352],height:290,font:27});
 bullets(s,['The pilot used MOSS transcription, a frozen MiniLM model and native CLAN.','Speaker-role matching used the human reference. Automatic child identification and cohort accuracy still need evaluation.'],{y:545,h:140,size:27,space:12});
 notes(s,'Existing saved pilot, not a new run. OCSC4002 ExpPictures,300.021s. ChildWER18.8172% uses oracle minimum-edit role mapping after ASR, not reference text supplied to generation. Native metric rowlatest_minilm_native_clan. Reference72utterances/179words versus predicted70/174 gives near-identical MLU ratios; small MLU error alone does not establish general accuracy. Reported SD/MATTR retain native three-decimal precision. Saved raw output and audio hashes match the manifest/evaluation. Keep this single-recording evidence separate from participant test summaries.',[R.pilot,local('outputs/results/moss_pilot/REPORT.md'),local('outputs/validation/moss_pilot/precision_audit.json')]);
}
// 25
{
 const s=slide('Planned audio comparison across participants','','future_audio_pipeline');
 flow(s,['Audio\nrecording','MOSS transcribes\nand labels speakers','Keep the\nchild’s speech','Run our\nlanguage pipeline','Compare with\nhuman references']);
 bullets(s,['Use the same child, recording and task section in both paths.','Measure word recognition errors separately from vocabulary and utterance-measure errors.'],{y:470,h:150,size:29,space:17});
 notes(s,'ASR=automatic speech recognition. Future multi-participant evaluation extends the earlier feasibility pilot. Review target-speaker/channel mapping, time intervals, privacy edits and word eligibility before scoring. Preserve cohort assignments. Use human transcripts to measure word error rate, then downstream language-measure differences. This slide describes planned work, not completed cohort audio results.',[local('scripts/README_AUDIO_DOWNLOAD.md'),local('outputs/flowcharts/all_pipelines_simple.excalidraw')]);
}
// 26
{
 const s=slide('Next steps','','next_steps');
 bullets(s,['Independently reconfirm the LUCID author-reference file when OSF access returns.','Download matching recordings into separate source and cohort folders.','Check audio quality, target speaker and task timestamps before transcription.','Extend the audio pilot to the original study: 10 participants and 37 recordings, then the public cohorts.','Report recognition errors and downstream language measures separately.'],{y:170,h:450,size:30,space:16});
 notes(s,'Original study plan describes10 autistic participants and37 recordings aged8–17. Bulk acquisition and multi-participant audio evaluation remain pending. Download script inventories301 Ohio,28 parent–child and300 LUCID recordings, preserving roles. LUCID WAV storage alone29.09GiB; TalkBank requires authorized session cookies. Existing audio includes one LUCID WAV and the earlier Ohio pilot excerpt. Do not restart the user-stopped training search or contact authors without a new instruction.',[local('SESSION_HANDOFF.md'),local('scripts/README_AUDIO_DOWNLOAD.md')]);
}
// 27
{
 const s=slide('Dataset sources and references','','sources');
 table(s,[['Source','Reference and access'],['London UCL Clear Speech\nin Interaction','Baker and Hazan. Corpus materials through SpeechBox.'],['LUCID lexical analysis','Baese-Berk et al. (2021), Frontiers in Psychology.\nReference export recorded at OSF, confirmation pending.'],['Ohio Child Speech Corpus','Wagner et al. (2025), Speech Communication.\nTalkBank CHILDES, DOI 10.21415/60TV-D669.'],['New York University\nand Emerson College corpus','Steen et al. (2026), Meeting on Language in Autism.\nTalkBank ASDBank, DOI 10.21415/VT7P-YF83.']],{y:165,widths:[450,702],height:440,font:25});
 notes(s,'Primary sources\n'+Object.entries(U).map(([k,v])=>k+': '+v).join('\n')+'\n\nLocal result records\n'+Object.entries(R).map(([k,v])=>k+': '+v).join('\n')+'\nOfficial title for the parent–child corpus is NYU–Emerson Corpus. Slide labels expand the institutional abbreviations as requested.');
}

// Keep a traceable coverage map for all user-requested sections and previous content.
await fs.writeFile(path.join(BUILD,'coverage.json'),JSON.stringify(coverage,null,2)+'\n');
await fs.writeFile(path.join(BUILD,'slide_text.json'),JSON.stringify(texts,null,2)+'\n');
await fs.writeFile(path.join(BUILD,'deck.proto.json'),JSON.stringify(P.toProto()));
const sourceHashes={};for(const p of Object.values(R))sourceHashes[p]=crypto.createHash('sha256').update(await fs.readFile(p)).digest('hex');
await fs.writeFile(path.join(BUILD,'source_hashes.json'),JSON.stringify(sourceHashes,null,2)+'\n');
const draft=path.join(BUILD,'artifact_draft.pptx');await(await PresentationFile.exportPptx(P)).save(draft);
// Apply the exact Office theme shipped with the installed PowerPoint application.
// Content, layouts, notes and native diagrams remain the Artifact Tool output.
const candidate=path.join(BUILD,'candidate.pptx');
const patch=spawnSync(process.env.RUNTIME_PYTHON,['-c',`import sys,zipfile,re
src,theme,dst=sys.argv[1:]
data=open(theme,'rb').read()
with zipfile.ZipFile(src) as source,zipfile.ZipFile(dst,'w',zipfile.ZIP_DEFLATED) as target:
 for info in source.infolist():
  target.writestr(info,data if re.fullmatch(r'ppt/theme/theme\\d+\\.xml',info.filename) else source.read(info.filename))
`,draft,path.join(BUILD,'office_theme.xml'),candidate],{encoding:'utf8'});
if(patch.status!==0)throw new Error(patch.stderr);
const final=local('outputs/presentations/speech_pipeline_office_theme_2026-10-16_final.pptx');
const result=await finalizePresentation({workspaceDir:ROOT,candidatePath:candidate,finalPath:final,pythonExecutable:process.env.RUNTIME_PYTHON,integrityValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_layout_geometry.py'),layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',...tableOwners.flatMap(n=>['--require-native-table-slide',String(n)])],requiredNativeTableOwnerSlides:tableOwners,requiredNativeChartOwnerSlides:[],fontPolicy:{basis:'design',families:[BODY,HEADING]},verifyArtifactToolImport:true,receiptPath:path.join(BUILD,'validation_final.json')});
console.log(JSON.stringify({file:result.finalPath,slides:P.slides.items.length,sha256:result.finalSha256,layout:result.presentationLayout},null,2));
