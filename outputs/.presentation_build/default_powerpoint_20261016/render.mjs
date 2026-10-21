import fs from 'node:fs/promises';
import path from 'node:path';
import {PresentationFile,FileBlob} from '@oai/artifact-tool';
const root='/Users/nafeekazi/Desktop/PhD/speech_pipeline_code';
const file=process.argv[2]??path.join(root,'outputs/presentations/speech_pipeline_office_theme_2026-10-16.pptx');
const out=path.join(root,'outputs/.presentation_build/default_powerpoint_20261016',process.argv[3]??'renders');
await fs.mkdir(out,{recursive:true});
const p=await PresentationFile.importPptx(await FileBlob.load(file));
for(let i=0;i<p.slides.items.length;i++){
 const png=await p.export({slide:p.slides.items[i],format:'png',scale:1});
 await fs.writeFile(path.join(out,`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await png.arrayBuffer()));
 console.log('Rendered',i+1);
}
