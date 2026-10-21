import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile} from '@oai/artifact-tool';

const ROOT='/Users/nafeekazi/Desktop/PhD/speech_pipeline_code';
const BUILD=path.join(ROOT,'outputs/.presentation_build/simple_pipeline_20261016');
const SKILL='/Users/nafeekazi/.codex/plugins/cache/openai-primary-runtime/presentations/26.1013.11914/skills/presentations';
const RUNTIME='/Users/nafeekazi/.cache/codex-runtimes/codex-primary-runtime/dependencies';
process.env.RUNTIME_NODE_MODULES=path.join(RUNTIME,'node/node_modules');
process.env.RUNTIME_NODE=path.join(RUNTIME,'node/bin/node');
process.env.RUNTIME_PYTHON=path.join(RUNTIME,'python/bin/python3');
const FINAL=path.join(ROOT,'outputs/presentations/speech_pipeline_simple_2026-10-16_final_v2.pptx');
const DRAW=path.join(ROOT,'outputs/flowcharts/speech_pipeline_and_comparisons.excalidraw');
const {finalizePresentation,makeNativeBulletParagraphs}=await import(pathToFileURL(path.join(SKILL,'container_tools/artifact_tool_utils.mjs')).href);
const D=JSON.parse(await fs.readFile(path.join(ROOT,'outputs/.presentation_build/speech_pipeline_20261016/data.json'),'utf8'));
const PILOT=JSON.parse(await fs.readFile(path.join(ROOT,'outputs/results/moss_pilot/evaluation.json'),'utf8'));
const M=JSON.parse(await fs.readFile(path.join(ROOT,'outputs/validation/cohorts_v1/manifest.json'),'utf8'));
const C={ink:'#202020',blue:'#2F5496',gray:'#666666',line:'#707070',header:'#E7E6E6',white:'#FFFFFF',pending:'#8A4B08'};
const P=Presentation.create({slideSize:{width:1280,height:720}});
const FONT='Arial';const tableOwners=[];const slideTexts=[];
const f=n=>Number(n).toFixed(2);
const source={ocsc:'https://talkbank.org/childes/access/Eng-NA/OCSC.html',nyu:'https://talkbank.org/asd/access/English/NYU-Emerson.html',lucid:'https://valeriehazan.com/index.php/lucid-corpus-london-ucl-clear-speech-in-interaction/',paper:'https://doi.org/10.3389/fpsyg.2021.661415',osf:'https://osf.io/dfhpu/?view_only=49d95d90424941da82217a239ab7450c',taaled:'https://www.linguisticanalysistools.org/taaled.html',clan:'https://talkbank.org/manuals/CLAN.pdf',model:'https://huggingface.co/microsoft/MiniLM-L12-H384-uncased'};
const local=r=>path.join(ROOT,r);
const resultSource=local('outputs/results/transcript_validation/chat_calibrated_results.json');
const cvSource=local('outputs/results/cv_hyperparameter_v4/stopped_search_summary.json');
const lucidSource=local('outputs/results/transcript_validation/lucid_results.json');
const provenance=local('outputs/results/transcript_validation/LUCID_METRIC_PROVENANCE.md');
const test=(corpus,metric)=>D.test.find(r=>r.corpus===corpus&&r.metric===metric).mean_participant_ape;
const cv=(config,corpus,metric)=>D.cv.find(r=>r.configuration===config).summary.find(r=>r.corpus===corpus&&r.metric===metric).mean_participant_ape;
const lucid=(condition,metric)=>D.lucid.find(r=>r.condition===condition&&r.metric===metric).mean_participant_ape;
function txt(s,text,x,y,w,h,size=30,color=C.ink,bold=false,align='left'){
 const a=s.shapes.add({geometry:'textbox',name:`text-${s.shapes.items.length}`,position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 a.text=text;a.text.style={typeface:FONT,fontSize:size,color,bold,alignment:align,verticalAlignment:'top',autoFit:'none',insets:0,wrap:'square'};
 slideTexts[P.slides.items.indexOf(s)].push(text);return a;
}
function slide(title,sub=''){
 const s=P.slides.add();slideTexts.push([]);s.background.fill=C.white;
 txt(s,title,64,42,1152,65,44,C.ink,true);
 if(sub)txt(s,sub,66,118,1146,52,25,C.gray);
 txt(s,String(P.slides.items.length),1178,674,38,24,18,C.gray,false,'right');return s;
}
function notes(s,text,urls=[]){s.speakerNotes.text=text+'\n\nSources\n'+urls.join('\n');}
function bullets(s,items,{x=76,y=200,w=1128,h=385,size=31,space=17}={}){
 const a=txt(s,'',x,y,w,h,size);a.text=makeNativeBulletParagraphs(items,{marginLeftPoints:22,hangingPoints:11,spaceAfterPoints:space});
 a.text.style={typeface:FONT,fontSize:size,color:C.ink,autoFit:'none',insets:0,verticalAlignment:'top'};
 slideTexts[P.slides.items.indexOf(s)].push(...items);return a;
}
function table(s,values,{y=190,widths=[400,350,402],height=330,font=27,x=64}={}){
 const t=s.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width:widths.reduce((a,b)=>a+b,0),height,columnWidths:widths,values});
 t.styleOptions={headerRow:true,bandedRows:false};t.borders.assign({fill:'#B7B7B7',width:0.7,style:'solid'});
 for(let r=0;r<values.length;r++){t.rows[r].height=height/values.length;for(let c=0;c<values[r].length;c++){
  const a=t.getCell(r,c);a.fill=r===0?C.header:C.white;a.text.style={typeface:FONT,fontSize:font,color:C.ink,bold:r===0,verticalAlignment:'middle',alignment:'left',insets:12};
 }}tableOwners.push(P.slides.items.length);slideTexts[P.slides.items.indexOf(s)].push(...values.flat());return t;
}
function box(s,n){
 const a=s.shapes.add({geometry:'rect',name:n.id,position:{left:n.x,top:n.y,width:n.w,height:n.h},fill:C.white,line:{fill:n.planned?C.gray:C.line,width:1.5,style:n.planned?'dash':'solid'}});
 a.text=n.text;a.text.style={typeface:FONT,fontSize:n.size??24,color:C.ink,alignment:'center',verticalAlignment:'middle',insets:12,autoFit:'none'};
 slideTexts[P.slides.items.indexOf(s)].push(n.text);return a;
}
function graph(s,g){
 const by={};for(const n of g.nodes)by[n.id]=box(s,n);
 for(const e of g.edges)s.shapes.connect(by[e.a],by[e.b],{kind:'elbow',fromSide:e.from??'right',toSide:e.to??'left',line:{fill:e.planned?C.gray:C.blue,width:2,style:e.planned?'dash':'solid'},tail:{type:'triangle',width:'med',length:'med'}});
 txt(s,g.note,66,605,1148,55,23,g.pending?C.pending:C.gray);
}
const node=(id,text,x,y,w,h=90,extra={})=>({id,text,x,y,w,h,...extra});
const edge=(a,b,from='right',to='left',planned=false)=>({a,b,from,to,planned});
const graphs=[
 {id:'whole',title:'Whole pipeline',sub:'Current transcript workflow. A small audio pilot exists. Broader audio testing is next.',note:'Vocabulary processing and utterance processing use separate branches.',nodes:[
  node('input','Target-speaker text\nHuman now / ASR later',64,200,240),node('prepare','Prepare words\nand observed turns',350,200,270),node('model','Predict utterances\nCustom MiniLM model',690,200,270),
  node('chat','Export predicted CHAT',690,335,270,72),node('clan','Native CLAN\nMLU and FREQ',1010,335,205,90),node('surface','Utterance count\nMLUw and length SD',1010,465,205,104),
  node('lex','Historical TAALED\nprocessing rules',350,465,270),node('vocab','Vocabulary counts\nand diversity measures',690,465,270),
  node('audio','Planned audio / ASR\nSpeaker selection',64,465,240,90,{planned:true,size:23})
 ],edges:[edge('input','prepare'),edge('prepare','model'),edge('model','chat','bottom','top'),edge('chat','clan'),edge('clan','surface','bottom','top'),edge('prepare','lex','bottom','top'),edge('lex','vocab'),edge('audio','input','top','bottom',true)]},
 {id:'lucid',title:'LUCID comparison',sub:'Vocabulary measurements from human transcripts',note:'180 samples scored. 60 vocoder mappings remain ambiguous. Author-source reconfirmation is pending.',pending:true,nodes:[
  node('tg','300 downloaded\nLUCID TextGrids',64,205,275),node('speaker','Extract target speaker\nApply lexical preparation',420,205,300),node('ours','Our lexical calculations\n15 shared CSV fields',860,205,350),
  node('ref','Local reference CSV\n240 sample rows\nSource check pending',64,425,275,110,{size:23}),node('match','Match speaker,\ncondition and scene',420,425,300,110),node('compare','Compare corresponding values\nReport percentage differences',860,425,350,110)
 ],edges:[edge('tg','speaker'),edge('speaker','ours'),edge('ref','match'),edge('speaker','match','bottom','top'),edge('match','compare'),edge('ours','compare','bottom','top')]},
 ...[['ocsc','OCSC','303','301','803','45','111'],['nyu','NYU–Emerson','30','28','92','4','15']].map(([id,name,all,used,sections,testN,testSections])=>({id,title:`${name} comparison`,sub:'Human CHAT supplies the reference words and utterance coding.',note:'Inference receives words and observed speaker turns. Human utterance endings stay outside model inputs.',nodes:[
  node('raw',`${id==='nyu'?'NYU–Emerson':'OCSC'} CHAT\n${all} source children`,64,205,280),node('select',`Select child speech\n${used} children / ${sections} sections`,425,205,300),node('predict','Remove human utterance cuts\nRun retained boundary model',855,205,355,95,{size:23}),
  node('reference','Human CHAT + native CLAN\nReference metrics',425,450,300,105,{size:23}),node('predicted','Predicted CHAT + native CLAN\nComputed metrics',855,345,355,85,{size:23}),node('compare',`Compare values on existing test\n${testN} children / ${testSections} sections`,855,475,355,95,{size:23})
 ],edges:[edge('raw','select'),edge('select','predict'),edge('select','reference','bottom','top'),edge('predict','predicted','bottom','top'),edge('predicted','compare','bottom','top'),edge('reference','compare')]}))
];

// Excalidraw v2: four editable frames, using the same diagram content as the slides.
let serial=0;const elements=[];const emap=new Map();
function basic(type,id,x,y,width,height,frameId=null){return {id,type,x,y,width,height,angle:0,strokeColor:C.ink,backgroundColor:'transparent',fillStyle:'solid',strokeWidth:1.5,strokeStyle:'solid',roughness:0,opacity:100,groupIds:[],frameId,roundness:null,seed:1000+(++serial),version:1,versionNonce:2000+serial,isDeleted:false,boundElements:[],updated:1792123200000,link:null,locked:false};}
function add(e){elements.push(e);emap.set(e.id,e);return e;}
function textElement(id,text,x,y,w,h,frameId,containerId=null,fontSize=24,align='left'){
 return {...basic('text',id,x,y,w,h,frameId),text,originalText:text,fontSize,fontFamily:2,textAlign:align,verticalAlign:containerId?'middle':'top',containerId,autoResize:true,lineHeight:1.25};
}
function port(n,side){return side==='right'?[n.x+n.w,n.y+n.h/2]:side==='left'?[n.x,n.y+n.h/2]:side==='bottom'?[n.x+n.w/2,n.y+n.h]:[n.x+n.w/2,n.y];}
for(let gi=0;gi<graphs.length;gi++){
 const g=graphs[gi],ox=(gi%2)*1380,oy=Math.floor(gi/2)*820,fid=`frame-${g.id}`;
 add({...basic('frame',fid,ox,oy,1280,720),name:g.title,strokeColor:'#A0A0A0'});
 add(textElement(`${g.id}-title`,g.title,ox+64,oy+42,1150,55,fid,null,36));
 add(textElement(`${g.id}-sub`,g.sub,ox+66,oy+118,1140,64,fid,null,24));
 const by={};
 for(const n of g.nodes){const id=`${g.id}-${n.id}`,tid=`${id}-text`;by[n.id]=n;
  add({...basic('rectangle',id,ox+n.x,oy+n.y,n.w,n.h,fid),strokeColor:n.planned?C.gray:C.line,strokeStyle:n.planned?'dashed':'solid',backgroundColor:C.white,boundElements:[{id:tid,type:'text'}]});
  const h=n.text.split('\n').length*(n.size??24)*1.25;
  add(textElement(tid,n.text,ox+n.x+10,oy+n.y+(n.h-h)/2,n.w-20,h,fid,id,n.size??24,'center'));
 }
 for(let ei=0;ei<g.edges.length;ei++){
  const e=g.edges[ei],a=by[e.a],b=by[e.b],p=port(a,e.from),q=port(b,e.to);
  const step=6; if(e.from==='right')p[0]+=step;if(e.from==='left')p[0]-=step;if(e.from==='bottom')p[1]+=step;if(e.from==='top')p[1]-=step;
  if(e.to==='left')q[0]-=step;if(e.to==='right')q[0]+=step;if(e.to==='top')q[1]-=step;if(e.to==='bottom')q[1]+=step;
  let pts;
  if(p[0]===q[0]||p[1]===q[1])pts=[p,q];else if(e.from==='right'||e.from==='left'){const m=(p[0]+q[0])/2;pts=[p,[m,p[1]],[m,q[1]],q];}else{const m=(p[1]+q[1])/2;pts=[p,[p[0],m],[q[0],m],q];}
  const id=`${g.id}-edge-${ei}`,aid=`${g.id}-${e.a}`,bid=`${g.id}-${e.b}`;
  add({...basic('arrow',id,ox+p[0],oy+p[1],Math.abs(q[0]-p[0]),Math.abs(q[1]-p[1]),fid),strokeColor:e.planned?C.gray:C.blue,strokeStyle:e.planned?'dashed':'solid',points:pts.map(v=>[v[0]-p[0],v[1]-p[1]]),startBinding:{elementId:aid,focus:0,gap:step},endBinding:{elementId:bid,focus:0,gap:step},startArrowhead:null,endArrowhead:'arrow',lastCommittedPoint:null,elbowed:false});
  emap.get(aid).boundElements.push({id,type:'arrow'});emap.get(bid).boundElements.push({id,type:'arrow'});
 }
 const foot=textElement(`${g.id}-note`,g.note,ox+66,oy+605,1140,70,fid,null,22);foot.strokeColor=g.pending?C.pending:C.gray;add(foot);
}
const excalidraw={type:'excalidraw',version:2,source:'https://excalidraw.com',elements,appState:{viewBackgroundColor:'#FFFFFF',gridSize:null,theme:'light',scrollX:40,scrollY:40,zoom:{value:0.45},frameRendering:{enabled:true,clip:true,name:true,outline:true}},files:{}};
await fs.writeFile(DRAW,JSON.stringify(excalidraw,null,2)+'\n');
await fs.writeFile(path.join(BUILD,'graphs.json'),JSON.stringify(graphs,null,2)+'\n');

// 1
{
 const s=slide('Speech pipeline: data, methods and results');
 txt(s,'LUCID, OCSC and NYU–Emerson',66,220,1120,65,40);
 txt(s,'Current transcript comparisons and the planned audio evaluation',66,330,1110,105,33);
 txt(s,'16 October 2026',66,535,1100,45,26,C.gray);
 notes(s,'Research status at the requested stopping point. No further model training or evaluation runs occurred for this presentation. LUCID author-source reconfirmation remains pending.',[cvSource,provenance]);
}
// 2
{
 const s=slide('The three datasets','Source participants and the subset used in our transcript comparisons');
 table(s,[['Dataset','Source participants','Our usable subset'],['LUCID','40 adults','300 TextGrids\n180 of 240 reference rows scored'],['OCSC','303 children','301 children\n803 task sections'],['NYU–Emerson','30 autistic children','28 children\n92 task sections']],{widths:[280,350,522],height:350,font:27});
 txt(s,'A participant can contribute several recordings or task sections.',66,590,1130,48,26,C.gray);
 notes(s,'LUCID counts derive from the corpus documentation and local source inventory. The 240-row lexical reference is retained locally with provenance reconfirmation pending. OCSC and NYU counts of usable children and sections derive from the frozen cohort manifest, not the source website totals. LUCID is a separate lexical reference.',[source.lucid,source.ocsc,source.nyu,local('outputs/validation/cohorts_v1/manifest.json'),provenance]);
}
// 3
{
 const s=slide('LUCID: adult conversations','Rachel Baker and Valerie Hazan, University College London');
 bullets(s,['40 adults aged 18–29, speaking southern British English.','Pairs described pictures to find differences.','Conditions cover normal listening, babble, electronically altered sound, or an English learner as partner.','We use human TextGrids to check our lexical calculations.','The local reference has 20 speakers and three scenes per condition.'],{y:190,h:365,size:29,space:13});
 txt(s,'Corpus files: SpeechBox at Northwestern. Lexical reference: the paper-linked OSF export.',66,574,1130,68,25,C.gray);
 notes(s,'The original corpus also contains sentence reading and picture naming. Our selected packages concern dialogue. TextGrids contain word text, times and speaker tiers. The downloaded collection contains 60 babble,60 L2,60 no-barrier and120 vocoder TextGrids. Multiple vocoder candidates explain why300 transcripts do not equal240 reference samples. Original source files remain unchanged.',[source.lucid,local('dataset/public/lucid_published/source_docs/LUCID - TextGrid README.txt'),provenance]);
}
// 4
{
 const s=slide('OCSC: children describing and explaining','Ohio Child Speech Corpus, Wagner and colleagues, 2025');
 bullets(s,['303 children, ages 4–9 in the published corpus description.','Researchers collected speech in a science-museum lab in Ohio.','We downloaded human CHAT transcripts from TalkBank CHILDES.','We use picture descriptions and “how to” explanations: 301 children and 803 sections.','This gives us a larger child-speech reference for utterance measures.'],{y:190,h:355,size:29,space:13});
 txt(s,'We leave reading, alphabet, number and Wug tasks outside this discourse comparison.',66,567,1120,72,25,C.gray);
 notes(s,'Selected gems are ExpPictures, Howto and DescriptivePictures. Two of303 child transcripts contain no eligible discourse section and remain documented exclusions. The official media release provides lapel-microphone audio. Age4–9 is the published description; local CHAT age headers are not fully harmonized and include a younger minimum. Our explanatory age medians use those headers explicitly.',[source.ocsc,local('outputs/validation/cohorts_v1/manifest.json'),local('outputs/results/cv_hyperparameter_v4/NYU_EXPLANATION.md')]);
}
// 5
{
 const s=slide('NYU–Emerson: parent–child play','NYU and Emerson College, shared through TalkBank ASDBank');
 bullets(s,['The released set contains 30 autistic children. Recruitment targeted ages 3 to under 6.','Families took part in play and conversation, mainly over Zoom at home.','We use the child’s speech from human CHAT transcripts.','28 children contribute 92 usable sections. Two have too little eligible speech for boundary scoring.','This tests transfer to younger autistic children in parent–child interaction.'],{y:185,h:395,size:29,space:13});
 txt(s,'The source includes video. Future audio testing will extract its audio track.',66,604,1130,42,25,C.gray);
 notes(s,'Corpus citation: Steen, Buonocore, Luyster, Sancimino and Arunachalam (2026), Introducing the NYU–Emerson Corpus, Meeting on Language in Autism. Sections include free play/conversation, independent pretend play, pretend play with parent, and sensory/cause-and-effect play. Current published data include autistic children, although the larger parent study also recruited non-autistic children. User-requested comparison uses the released set only.',[source.nyu,local('outputs/validation/cohorts_v1/manifest.json')]);
}
// 6
{
 const s=slide('Participant splits','Every section and training crop from a child stays in the same cohort');
 table(s,[['Usable children','Train','Eval','Test'],['OCSC','211','45','45'],['NYU–Emerson','20','4','4'],['Total','231','49','49']],{widths:[420,244,244,244],height:285,font:29});
 bullets(s,['Train fits weights. Eval selects settings. Test measures a frozen choice.','OCSC and NYU training data contribute to one model. Eval and test stay separate.','LUCID supplies no boundary-model training examples.'],{y:505,h:145,size:25,space:8});
 notes(s,'Task-section counts: OCSC train567/eval125/test111, NYU63/14/15, total630/139/126. NYU originally assigns22 to train, of whom2 have no eligible records. OCSC2 exclusions have no selected discourse tasks. Train/eval/test participants are disjoint. Existing test results have prior benchmark exposure and are descriptive; preserving disjointness does not create newly unexposed confirmation.',[local('outputs/validation/cohorts_v1/AUDIT.md'),local('outputs/results/cv_hyperparameter_v4/METHODS.md')]);
}
// 7
{
 const s=slide('What the reported errors mean','Lower percentage error means closer agreement with the reference');
 table(s,[['Measure','Plain-language meaning'],['Total / different words','Amount of speech and vocabulary'],['TTR / MATTR-50','Vocabulary variety overall or in 50-word windows'],['Eligible utterance count','Utterances included in CLAN’s MLU calculation'],['MLUw','Average words per eligible utterance'],['Length SD','How much utterance length varies']],{widths:[410,742],height:355,font:26});
 txt(s,'Percentage error is averaged within participants, then across participants.',66,586,1125,68,24,C.gray);
 notes(s,'Absolute percentage error is100*abs(predicted-reference)/abs(reference). Zero or undefined denominators remain unavailable. LUCID uses per-recording values before participant averaging. Eligible count differs from total verbal units. MLU means length in words, not morphemes. Native CHAT FREQ and MLU have different word eligibility. Results include exclusions/missingness, never replace unavailable values with zero.',[resultSource,lucidSource,source.clan]);
}
// 8
{const g=graphs[0],s=slide(g.title,g.sub);graph(s,g);notes(s,'The retained workflow is turn_ml with margin0.25. Observed turns merge consecutive same-speaker transcript rows. Human utterance endings are withheld at inference. Lexical calculations and native CHAT surface scoring are separate branches. The audio/ASR node is planned, not an input to these reported results.',[local('outputs/results/cv_hyperparameter_v4/PIPELINE.md'),local('speech_pipeline/lexical_compat.py'),local('speech_pipeline/segmentation.py')]);}
// 9
{
 const s=slide('Our model and the external tools','The boundary model includes custom training on our task');
 table(s,[['Component','What we use'],['Pretrained encoder','Microsoft MiniLM represents words in context'],['Custom components','Learned rule embedding and word-boundary classifier'],['Training','We fine-tune MiniLM and the custom components together'],['spaCy and CLAN','Grammar cues and native CHAT measures, respectively'],['TAALED conventions','Historical rules guide our vocabulary calculations']],{widths:[390,762],height:360,font:26});
 txt(s,'Lexical measures use fixed calculation rules. LUCID does not train the boundary model.',66,580,1120,66,26,C.gray);
 notes(s,'Architecture: MiniLM-L12-H384-uncased, learned16-dimensional rule embedding and128-unit GELU/dropout gap head. The current model is not merely an unchanged off-the-shelf model. Native CLAN MLU/FREQ and historical TAALED compatibility are external/software-rule components. Source-supported historical processing uses the installed spaCy model, not a reconstructed2021 environment.',[source.model,local('speech_pipeline/segmentation.py'),local('speech_pipeline/lexical_compat.py')]);
}
// 10
{
 const s=slide('What we tried','Changes addressed vocabulary processing, utterance boundaries and model selection');
 bullets(s,['Aligned lexical processing with historical TAALED dictionary, word-category and pause rules.','Used native CLAN scoring to respect CHAT word and utterance eligibility.','Compared whole-paragraph prediction, observed-turn prediction and rules.','Tested fixed boundary settings on development participants.','Compared learning rates and count-loss weights using five-fold participant CV.'],{y:190,h:430,size:28,space:15});
 notes(s,'The chosen earlier production workflow is turn_ml, margin0.25. A batching inconsistency in the initial boundary calibration was caught before accepting selection, then original preprocessing and parity checks were used. A single-eval ensemble branch was superseded by the CV search. Three CV configurations completed all five folds; a fourth stopped during fold1. We preserve that history and do not claim it produced a new final model.',[local('outputs/results/transcript_validation/REPORT.md'),local('outputs/results/cv_hyperparameter_v4/METHODS.md'),cvSource]);
}
// 11
{
 const s=slide('Five-fold cross-validation','Recipe comparison uses the original 231 usable training children only');
 bullets(s,['Each configuration uses five fits, with a different group of children held out each time.','Within each fold, separate inner children select the checkpoint and boundary setting.','We pool held-out predictions across all five folds to compare configurations.'],{y:190,h:225,size:28,space:13});
 table(s,[['Completed configuration','Combined CV error'],['Baseline learning rate and count loss',f(D.cv[0].objective)+'%'],['Lower count-loss weight',f(D.cv[1].objective)+'%'],['Higher learning rate',f(D.cv[2].objective)+'%']],{y:425,widths:[850,302],height:180,font:25});
 txt(s,'Three configurations completed five folds. A fourth stopped in fold 1 and is excluded.',66,630,1140,35,23,C.gray);
 notes(s,'CV objective equally averages count, MLUw and length-SD participant APE across the two corpora. Configurations: lr2_count1,lr2_count025,lr3_count1. Same grouped folds per recipe. Fit147–148,inner37,heldout46–47 per fold. Three configurations complete15 assessments. Fourth lr3_count025 stopped in first fold and is excluded entirely. Original eval/test do not enter recipe comparison. The selected CV minimum is a tuning score, not an unbiased final accuracy estimate.',[cvSource,local('outputs/results/cv_hyperparameter_v4/METHODS.md')]);
}
// 12
{const g=graphs[1],s=slide(g.title,g.sub);graph(s,g);notes(s,'The local reference is recorded as the authors’ OSF TAALED export. It matches saved OSF metadata, but current access failed and independent author-source reconfirmation is pending. Comparison matches speaker, condition and scene without selecting recordings by score agreement. Exact historical cleaned inputs remain unavailable.',[provenance,lucidSource,source.paper,source.osf]);}
// 13
{
 const s=slide('LUCID: the reference and the metrics','The main article presents a subset of the fields in the released export');
 table(s,[['Main article','Corresponding CSV field'],['Table 1: mean word count','basic_ntokens'],['Figure 1: TTR','simple_ttr_aw'],['Figure 2: MATTR-50','mattr50_aw'],['Figure 3: HD-D42','hdd42_aw']],{widths:[610,542],height:285,font:28});
 bullets(s,['Our 15-field comparison includes additional counts, densities and diversity fields from the local export.','Our original-MTLD column differs from Figure 4’s moving-average variant.'],{y:510,h:110,size:25,space:12});
 txt(s,'Publication status: independently reconfirm the author-hosted CSV before citing this comparison.',66,636,1140,34,23,C.pending);
 notes(s,'Baese-Berk and colleagues2021 used TAALED for lexical diversity and TAALES for sophistication. Footnote2 links exported data/code/preregistration to OSF. Our comparison does not reproduce their TAALES measures, mixed-model coefficients or p-values. All15 exact columns and source chain appear in the local provenance note. Figure4 explicitly uses moving-average forward/backward MTLD, while our compared field is mtld_original_aw. CSV/hash consistency alone is not independent proof of historical transfer.',[source.paper,source.osf,source.taaled,provenance]);
}
// 14
{
 const s=slide('LUCID results: provisional reference comparison','Mean participant absolute percentage error. Lower is better.');
 table(s,[['Condition','Rows scored','Words','TTR','MATTR-50','HD-D42'],...['babble','L2','noBarrier'].map((c,i)=>[['Babble','L2','No barrier'][i],'60 / 60',...['basic_ntokens','simple_ttr_aw','mattr50_aw','hdd42_aw'].map(m=>f(lucid(c,m))+'%')]),['Vocoder','0 / 60','Unavailable','Unavailable','Unavailable','Unavailable']],{widths:[205,195,175,175,201,201],height:300,font:25});
 bullets(s,['Babble and L2 show close lexical agreement. No-barrier word counts remain less consistent.','Vocoder has ambiguous recording mappings, so it contributes no scored result.'],{y:520,h:105,size:25,space:12});
 txt(s,'Author-source reconfirmation is pending. These results assess vocabulary calculations from transcripts.',66,638,1140,32,23,C.pending);
 notes(s,'Values come from retained transcript_validation/lucid_results.json, not printed paper accuracy percentages. Each available condition has60 samples from20 speakers. All240 local reference rows remain in coverage,180 scored. All15 shared means meet project engineering targets in babble/L2, but those are internal goals, not clinical standards or paper thresholds. Exact historical cleaned inputs/software are unavailable. Possible source/preparation differences remain, particularly no-barrier. Independent remote source verification is pending.',[lucidSource,provenance]);
}
// 15
{const g=graphs[2],s=slide(g.title,g.sub);graph(s,g);notes(s,'OCSC raw source303 children,301 usable with803 selected sections. Retained model input uses words and observed turns from the corresponding selected sections. Original human CHAT boundaries and coding remain in the reference path. Existing test45 children111 sections. The same native CLAN commands score predictions and references.',[resultSource,local('outputs/validation/cohorts_v1/manifest.json'),source.clan]);}
// 16
{
 const s=slide('OCSC: retained model on the existing test cohort','45 children and 111 sections. Mean participant percentage error.');
 const metrics=[['Total words','TNW'],['Different words','NDW'],['TTR','TTR'],['MATTR-50','MATTR'],['Eligible utterance count','eligible_mlu_utterances'],['MLUw','MLUw'],['Utterance-length SD','length_sd']];
 table(s,[['Measure','Error'],...metrics.map(([label,key])=>[label,f(test('ocsc',key))+'%'])],{widths:[850,302],height:370,font:27});
 txt(s,'Vocabulary agreement is strong. Count and MLU are less precise. Length SD remains the hardest measure.',66,589,1120,68,25,C.gray);
 notes(s,'Methodturn_ml/margin0.25. Available sections: all111 for TNW/NDW/TTR/count/MLUw,104 for MATTR,109 for length SD. All45 participants contribute to these summaries. Missing values stay missing. Eligible count is the CLAN MLU-eligible count, not all verbal units. Project5% count/MLU and10% SD engineering goals remain unmet. Historical test exposure is documented. These values are not scores from the later CV fold models.',[resultSource,local('outputs/validation/cohorts_v1/AUDIT.md')]);
}
// 17
{const g=graphs[3],s=slide(g.title,g.sub);graph(s,g);notes(s,'NYU30 source children,28 usable with92 selected sections, test4 children15 sections. Two assigned training children lack eligible boundary gaps and remain explicit exclusions. Human CHAT references include coding/eligibility information beyond what plain predictions can recreate. We do not invent human annotations.',[resultSource,local('outputs/validation/cohorts_v1/manifest.json'),source.clan]);}
// 18
{
 const s=slide('NYU: retained model on the existing test cohort','4 children and 15 sections. Mean participant percentage error.');
 const metrics=[['Total words','TNW'],['Different words','NDW'],['TTR','TTR'],['MATTR-50','MATTR'],['Eligible utterance count','eligible_mlu_utterances'],['MLUw','MLUw'],['Utterance-length SD','length_sd']];
 table(s,[['Measure','Error'],...metrics.map(([label,key])=>[label,f(test('nyu_emerson',key))+'%'])],{widths:[850,302],height:370,font:27});
 txt(s,'Vocabulary agreement remains close. Utterance count and MLU are weaker, with a small test sample.',66,589,1120,68,25,C.gray);
 notes(s,'All15 test sections contribute to these metrics and all4 test participants contribute. Methodturn_ml/margin0.25. Existing test results are retained descriptive evaluations. NYU length-SD error13.70% is slightly lower than OCSC14.24%, so the conclusion is metric-specific. Count/MLU are worse; not every measure is worse. No new final fit/test followed the CV search.',[resultSource,local('outputs/validation/cohorts_v1/AUDIT.md')]);
}
// 19
{
 const s=slide('What model tuning improved','Five-fold CV tuning results, separate from the existing test results');
 const vals=(cfg,c)=>['eligible_mlu_utterances','MLUw','length_sd'].map(m=>f(cv(cfg,c,m))+'%');
 table(s,[['Dataset / configuration','Count','MLUw','Length SD'],['OCSC: earlier fixed recipe','9.16%','9.09%','14.89%'],['OCSC: current CV baseline',...vals('lr2_count1','ocsc')],['OCSC: lower count-loss weight',...vals('lr2_count025','ocsc')],['NYU: current CV baseline',...vals('lr2_count1','nyu_emerson')],['NYU: lower count-loss weight',...vals('lr2_count025','nyu_emerson')]],{widths:[570,194,194,194],height:325,font:25});
 bullets(s,['Lower count loss modestly improved OCSC but worsened NYU count and MLU.','The baseline kept the best combined CV score. We retained the existing production model.'],{y:553,h:105,size:25,space:11});
 notes(s,'Each CV row pools five heldout models under one consistent recipe. Do not combine minimum cells across rows or relabel these tuning values as external test performance. Earlier fixed-recipe v1 compared with current v4 protocol remains a historical comparison; within-v4 baseline versus lower-count is the paired recipe comparison. Fourth configuration excluded entirely. No fresh final model was trained after the user stopped the search.',[cvSource,local('outputs/results/cv_hyperparameter_v4/REPORT.md')]);
}
// 20
{
 const s=slide('Why the results differ','Measured development-data differences support hypotheses, not a proven cause');
 table(s,[['Train + eval input characteristic','NYU','OCSC'],['Median child age, years','4.58','7.22'],['Median words per observed turn','2','5'],['Turns with at most 3 words','70%','41%'],['Unintelligibility markers per 1,000 words','79.49','5.73']],{widths:[730,211,211],height:285,font:26});
 bullets(s,['NYU has shorter turns, more missing speech and much less training text.','Age, task and coding vary together. These data do not isolate an age effect.','LUCID tests vocabulary. No-barrier errors may reflect source matching or text preparation.'],{y:510,h:145,size:25,space:9});
 notes(s,'Development profiling uses original train+eval only, not test-based tuning. Training source words: NYU5336,OCSC229378. Native FREQ/MLU word-total gaps7.22% vs1.79%. Different eligibility/coding and small reference denominators may amplify percentage errors. NYU was not worse on every metric: preceding CV micro boundaryF1 NYU88.81% versusOCSC87.22%; retained test length SD NYU13.70% vsOCSC14.24%. LUCID no-barrier discrepancies also suggest unresolved source/preparation differences. No causal attribution has been established.',[local('outputs/results/cv_hyperparameter_v4/NYU_EXPLANATION.md'),local('outputs/validation/cv_hyperparameter_v4/development_input_profile.json'),local('outputs/validation/cv_hyperparameter_v4/development_age_profile.json'),provenance]);
}
// 21: earlier audio work, separate from the cohort transcript comparisons
{
 const s=slide('Earlier audio feasibility pilot','One OCSC picture-description section, 300 seconds');
 const m=k=>PILOT.metrics.find(r=>r.method==='latest_minilm_native_clan'&&r.metric===k).ape;
 table(s,[['Measure','Error on this recording'],['Child word error rate',f(PILOT.target_wer_percent)+'%'],['Eligible utterance count',f(m('eligible_mlu_utterances'))+'%'],['MLUw',f(m('MLUw'))+'%'],['Utterance-length SD',f(m('length_sd'))+'%']],{widths:[800,352],height:285,font:28});
 bullets(s,['The pilot used MOSS words, a frozen MiniLM segmenter and native CLAN.','This shows feasibility on one recording. A participant-cohort comparison is still needed.'],{y:510,h:105,size:25,space:11});
 txt(s,'Speaker-role mapping uses the human reference. Automatic child identification remains unvalidated.',66,633,1140,38,23,C.gray);
 notes(s,'Existing saved pilot, not a new run for this deck. OCSC4002/ExpPictures,300.021s. The pinned MOSS model used official default prompt and no reference text for generation. Role mapping afterward used the minimum-edit gold speaker permutation, so target WER uses oracle role mapping. Native metric method latest_minilm_native_clan, reconstructed exact-ratio MLU rather than rounded CLAN display. References72 utterances/179 words vs predicted70/174 explain the near-equal MLU ratios. A small MLU ratio difference does not establish ASR or cohort accuracy. Raw output and excerpt hashes match the saved evaluation and manifest. Do not merge this single-example pilot with later participant test summaries.',[local('outputs/results/moss_pilot/evaluation.json'),local('outputs/results/moss_pilot/REPORT.md'),local('outputs/validation/moss_pilot/audio_manifest.json'),local('outputs/validation/moss_pilot/precision_audit.json')]);
}
// 22
{
 const s=slide('Planned audio evaluation across participants','Automatic speech recognition (ASR) adds recognition errors to the current transcript workflow.');
 const ns=[node('audio','Original recording\nSame child and task section',64,205,300),node('asr','MOSS audio transcription\nand speaker selection',485,205,310),node('pipeline','Current text pipeline\nLexical and CHAT measures',900,205,310),node('human','Human target-speaker transcript\nand reference measures',64,450,350,105,{size:24}),node('compare','Compare words and final measures\nSeparate ASR errors from scoring errors',800,450,410,105,{size:24})];
 const by={};for(const n of ns)by[n.id]=box(s,n);
 for(const e of [edge('audio','asr'),edge('asr','pipeline'),edge('pipeline','compare','bottom','top'),edge('human','compare')])s.shapes.connect(by[e.a],by[e.b],{kind:'elbow',fromSide:e.from,toSide:e.to,line:{fill:C.blue,width:2},tail:{type:'triangle',width:'med',length:'med'}});
 txt(s,'Planned: word error rate plus vocabulary and utterance-measure differences on matching speech.',66,608,1140,55,24,C.gray);
 notes(s,'This slide describes the proposed multi-participant extension. The preceding slide documents the existing single-recording feasibility pilot. Use matching task intervals and target-speaker channels so adult speech, privacy edits and excluded tasks do not contaminate comparison. Compare ASR transcript to human words, then compute downstream metric differences with the frozen text pipeline. Target identity/channel handling must be verified before scoring. LUCID lexical references remain pending provenance confirmation and vocoder mapping.',[local('scripts/README_AUDIO_DOWNLOAD.md'),local('SESSION_HANDOFF.md'),local('outputs/results/moss_pilot/REPORT.md')]);
}
// 23
{
 const s=slide('Next steps','Keep the completed transcript work reproducible while preparing audio testing');
 bullets(s,['Independently reconfirm the LUCID author-reference file when OSF access returns.','Use the download scripts to obtain matching recordings in separate source and cohort folders.','Check audio quality, target speaker and task timestamps before ASR.','Extend the audio pilot to the original study and then the public cohorts.','Report recognition errors and downstream language measures separately.'],{y:185,h:405,size:29,space:16});
 txt(s,'Bulk acquisition and a multi-participant audio evaluation remain pending.',66,610,1125,55,24,C.gray);
 notes(s,'Original study plan:10 autistic participants and37 recordings, aged8–17 in project documents. Revisit those recordings after the public transcript work, without moving public-test participants into train/eval. Acquisition plan has301 OCSC,28 NYU and300 LUCID recordings. LUCID WAVs alone need29.09GiB; TalkBank requires authorized cookies. Downloads preserve existing roles and raw media. This presentation does not restart stopped training or authorize contacting authors.',[local('scripts/README_AUDIO_DOWNLOAD.md'),local('SESSION_HANDOFF.md'),local('outputs/validation/audio_acquisition_v1/access_check.json')]);
}
// 24
{
 const s=slide('Sources and result records');
 table(s,[['Source','Reference'],['LUCID corpus','Baker and Hazan, UCL. Corpus materials via SpeechBox.'],['LUCID lexical analysis','Baese-Berk et al. (2021), Frontiers in Psychology.\nAuthor-reference reconfirmation pending.'],['OCSC','Wagner et al. (2025), Speech Communication.\nTalkBank CHILDES, DOI 10.21415/60TV-D669.'],['NYU–Emerson','Steen et al. (2026), Meeting on Language in Autism.\nTalkBank ASDBank, DOI 10.21415/VT7P-YF83.']],{widths:[350,802],height:335,font:25});
 txt(s,'Speaker notes link the dataset pages, source files, metric records and CV reports used on each slide.',66,583,1125,64,25,C.gray);
 notes(s,'Primary external references\n'+Object.entries(source).map(([k,v])=>k+': '+v).join('\n')+'\nOCSC paper: https://doi.org/10.1016/j.specom.2025.103206\n\nLocal provenance and results\n'+[provenance,resultSource,lucidSource,cvSource,local('outputs/validation/cohorts_v1/manifest.json')].join('\n')+'\n\nAll LUCID comparison values in this presentation remain provisional for publication pending independent source reconfirmation. Existing CHAT test values and later CV tuning values have different evaluation roles. No additional research scoring occurred.');
}

await fs.writeFile(path.join(BUILD,'slide_text.json'),JSON.stringify(slideTexts,null,2)+'\n');
const sources=[resultSource,lucidSource,cvSource,local('outputs/validation/cohorts_v1/manifest.json'),provenance,local('outputs/results/moss_pilot/evaluation.json'),local('outputs/validation/moss_pilot/audio_manifest.json')];
const hashes={};for(const p of sources)hashes[p]=crypto.createHash('sha256').update(await fs.readFile(p)).digest('hex');
await fs.writeFile(path.join(BUILD,'source_hashes.json'),JSON.stringify(hashes,null,2)+'\n');
await fs.writeFile(path.join(BUILD,'deck.proto.json'),JSON.stringify(P.toProto()));
const candidate=path.join(BUILD,'candidate.pptx');await(await PresentationFile.exportPptx(P)).save(candidate);
const result=await finalizePresentation({workspaceDir:ROOT,candidatePath:candidate,finalPath:FINAL,pythonExecutable:path.join(RUNTIME,'python/bin/python3'),integrityValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_layout_geometry.py'),layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',...tableOwners.flatMap(n=>['--require-native-table-slide',String(n)])],requiredNativeTableOwnerSlides:tableOwners,requiredNativeChartOwnerSlides:[],fontPolicy:{basis:'design',families:[FONT]},verifyArtifactToolImport:true,receiptPath:path.join(BUILD,'validation_final_v2.json')});
console.log('FINAL',JSON.stringify(result));
console.log('EXCALIDRAW',DRAW,'frames',graphs.length,'elements',elements.length);
