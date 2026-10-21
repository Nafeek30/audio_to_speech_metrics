from pathlib import Path
import json,hashlib,zipfile,wave
from datetime import datetime,timezone
import requests
from speech_pipeline import ROOT
from speech_pipeline.research.lucid_download import RangeReader,SHARE
base=ROOT/'dataset/public/lucid_published'
prepared=json.loads((ROOT/'outputs/validation/lucid_validation/prepared.json').read_text())
first=prepared[0]
assert first['reference_id']=='index1.txt' and len(first['candidates'])==1
spec=json.loads((base/'textgrids/babbleCondition/source.json').read_text())
entry=next(e for e in spec['entries'] if e['name']=='babbleCondition/LCD_F14BEF3B4cln2.wav')
folder=base/'audio';folder.mkdir(exist_ok=True)
protocol=ROOT/'outputs/validation/lucid_validation/audio_download_protocol.json'
record=dict(registered_utc=datetime.now(timezone.utc).isoformat(),selection='First published CSV row, index1.txt; unique documented speaker/condition/scene candidate. Source selected before any LUCID MOSS inference. This is acquisition only, not an ASR benchmark or proof of exact historical cleaned-input mapping.',reference_id=first['reference_id'],speaker=first['speaker'],condition=first['condition'],scene=first['scene'],source_textgrid=first['candidates'][0],archive=spec['archive'],url=spec['url'],etag=spec['etag'],entry=entry)
if protocol.exists():assert json.loads(protocol.read_text())['entry']==entry
else:protocol.write_text(json.dumps(record,indent=2)+'\n')
target=folder/Path(entry['name']).name
if target.exists():raise FileExistsError('Preserve existing audio')
session=requests.Session();session.get(SHARE,timeout=(10,30)).raise_for_status()
reader=RangeReader(session,spec,block_size=1024*1024)
partial=target.with_suffix('.partial')
sha=hashlib.sha256();size=0
with zipfile.ZipFile(reader) as archive:
 info=archive.getinfo(entry['name'])
 assert info.CRC==entry['crc32'] and info.file_size==entry['bytes']
 with archive.open(info) as src,partial.open('xb') as dst:
  while chunk:=src.read(1024*1024):
   dst.write(chunk);sha.update(chunk);size+=len(chunk)
  # ZipExtFile validates ZIP CRC on the completed read.
assert size==entry['bytes']
with wave.open(str(partial)) as wav:
 audio=dict(channels=wav.getnchannels(),sample_rate=wav.getframerate(),sample_width=wav.getsampwidth(),frames=wav.getnframes(),duration_seconds=wav.getnframes()/wav.getframerate())
partial.rename(target)
result=dict(**record,path=str(target.relative_to(ROOT)),bytes=size,sha256=sha.hexdigest(),audio=audio,downloaded_bytes=reader.bytes_downloaded,validation='Full extracted WAV length and ZIP CRC verified; local SHA-256 computed; no full archive hash claimed. LUCID audio not yet run through MOSS.')
target.with_suffix('.source.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ['path','bytes','audio','downloaded_bytes']},indent=2),flush=True)
