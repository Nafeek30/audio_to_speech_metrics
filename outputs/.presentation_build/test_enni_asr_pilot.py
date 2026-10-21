import urllib.request, ssl, certifi, pyarrow.parquet as pq, io, soundfile as sf, numpy as np
from pathlib import Path
import time
import torch

OUT_DIR = Path('outputs/runtime/enni_audio_pilot')
OUT_DIR.mkdir(parents=True, exist_ok=True)
audio_path = OUT_DIR / '401_session.wav'

# 1. Download & build continuous audio if not exists
if not audio_path.exists():
    ctx = ssl.create_default_context(cafile=certifi.where())
    url = 'https://huggingface.co/datasets/TomRoma/ENNI_dataset/resolve/main/data/401-00000-of-00001.parquet'
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    print('Downloading 401.parquet...', flush=True)
    with urllib.request.urlopen(req, context=ctx) as resp:
        data = resp.read()
    df = pq.read_table(io.BytesIO(data)).to_pandas()
    
    sr = 16000
    max_ms = df['end_ms'].max()
    total_samples = int((max_ms / 1000.0) * sr) + sr
    continuous = np.zeros(total_samples, dtype=np.float32)
    
    for _, row in df.iterrows():
        a_bytes = row['audio']['bytes']
        sig, sample_rate = sf.read(io.BytesIO(a_bytes))
        start_samp = int((row['start_ms'] / 1000.0) * sr)
        end_samp = start_samp + len(sig)
        if end_samp <= len(continuous):
            continuous[start_samp:end_samp] += sig
            
    sf.write(str(audio_path), continuous, sr)
    print(f'Saved continuous audio to {audio_path} (Duration: {len(continuous)/sr/60:.2f} min)', flush=True)
else:
    print(f'Using existing {audio_path}', flush=True)

# 2. Test SenseVoiceSmall
print('--- Running FunASR SenseVoiceSmall ---', flush=True)
from funasr import AutoModel
model_sv = AutoModel(model='FunAudioLLM/SenseVoiceSmall', hub='hf', disable_update=True)
res_sv = model_sv.generate(input=str(audio_path), language='en', use_itn=True)
sv_text = res_sv[0]['text']
print(f'SenseVoice raw output (first 250 chars):\n{sv_text[:250]}...\n', flush=True)

# 3. Test Whisper-small
print('--- Running Whisper-small ---', flush=True)
from transformers import pipeline
pipe_whisper = pipeline('automatic-speech-recognition', model='openai/whisper-small', device='cpu')
res_wh = pipe_whisper(str(audio_path), chunk_length_s=30, stride_length_s=5)
wh_text = res_wh['text']
print(f'Whisper raw output (first 250 chars):\n{wh_text[:250]}...\n', flush=True)

