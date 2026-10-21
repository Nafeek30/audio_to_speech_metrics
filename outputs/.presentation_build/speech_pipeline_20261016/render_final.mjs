import fs from 'node:fs/promises';
import path from 'node:path';
import {PresentationFile,FileBlob} from '@oai/artifact-tool';
const root='/Users/nafeekazi/Desktop/PhD/speech_pipeline_code';
const base=path.join(root,'outputs/.presentation_build/speech_pipeline_20261016/final_renders');
await fs.mkdir(base,{recursive:true});
const file=path.join(root,'outputs/presentations/speech_pipeline_results_2026-10-16_final.pptx');
const p=await PresentationFile.importPptx(await FileBlob.load(file));
for(let i=0;i<p.slides.items.length;i++){
 const b=await p.export({slide:p.slides.items[i],format:'png',scale:1});
 await fs.writeFile(path.join(base,`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await b.arrayBuffer()));
 console.log('Final slide rendered',i+1);
}
