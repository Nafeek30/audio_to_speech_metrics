import fs from 'node:fs/promises';
import path from 'node:path';
import {PresentationFile,FileBlob} from '@oai/artifact-tool';
const root='/Users/nafeekazi/Desktop/PhD/speech_pipeline_code';
const file=process.argv[2]??path.join(root,'outputs/presentations/speech_pipeline_simple_2026-10-16.pptx');
const base=path.join(root,'outputs/.presentation_build/simple_pipeline_20261016',process.argv[3]??'renders');
await fs.mkdir(base,{recursive:true});
const p=await PresentationFile.importPptx(await FileBlob.load(file));
for(let i=0;i<p.slides.items.length;i++){
 const png=await p.export({slide:p.slides.items[i],format:'png',scale:1});
 await fs.writeFile(path.join(base,`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await png.arrayBuffer()));
 console.log('Rendered',i+1);
}
