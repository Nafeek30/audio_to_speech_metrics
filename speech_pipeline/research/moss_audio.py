"""Pinned local MOSS audio inference, intended for its separate runtime.

No gold transcripts, hotwords, speaker identities or task annotations enter the
model. Run downstream scoring with the project's original environment instead.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import time


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('audio',type=Path)
    p.add_argument('--model-directory',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--device',choices=['cpu','mps','cuda'],default='cpu')
    p.add_argument('--max-new-tokens',type=int,default=8192)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve existing inference result')
    import torch
    from transformers import AutoModelForCausalLM,AutoProcessor
    from moss_transcribe_diarize.inference_utils import build_transcription_messages,generate_transcription
    if a.device=='mps' and not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable; choose CPU explicitly')
    torch.set_num_threads(4)
    began=time.monotonic();device=torch.device(a.device);dtype=torch.float32
    print('Loading pinned local MOSS model',flush=True)
    model=AutoModelForCausalLM.from_pretrained(str(a.model_directory),trust_remote_code=True,
        local_files_only=True,dtype=dtype,attn_implementation='sdpa').to(device).eval()
    processor=AutoProcessor.from_pretrained(str(a.model_directory),trust_remote_code=True,local_files_only=True)
    print('Model loaded; starting deterministic audio inference',flush=True)
    def progress(count):
        if count%100==0:print(f'Generated {count} tokens; elapsed {time.monotonic()-began:.1f}s',flush=True)
    result=generate_transcription(model,processor,build_transcription_messages(str(a.audio)),
        max_new_tokens=a.max_new_tokens,do_sample=False,device=device,dtype=dtype,
        input_callback=lambda n:print(f'Audio/text prompt tokens: {n}',flush=True),token_callback=progress)
    result.update(finish_reason='length' if result['generated_tokens']>=a.max_new_tokens else 'eos',
        audio_sha256=sha(a.audio),audio_file=str(a.audio.resolve()),model_directory=str(a.model_directory),
        model_files_sha256={x.name:sha(x) for x in a.model_directory.iterdir() if x.is_file()},
        versions={n:metadata.version(n) for n in ['torch','transformers','librosa','soundfile','av']},
        device=a.device,dtype=str(dtype),runtime_seconds=time.monotonic()-began,
        completed_utc=datetime.now(timezone.utc).isoformat(),
        prediction_inputs='Audio and official default transcription/diarization prompt only; no reference text or speaker mapping')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print('Saved raw MOSS result',a.output,'finish:',result['finish_reason'],flush=True)


if __name__=='__main__':main()
