"""Selectively acquire authorized LUCID TextGrids from public corpus ZIPs.

The user completed SpeechBox's data-use form and authorized its sharing link.
No browser cookies or credentials are read. Standard ZIP byte ranges avoid
downloading 17 GB of audio merely to obtain text. ZIP entry CRCs are verified.
"""
from collections import OrderedDict
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import zipfile

import requests

from speech_pipeline import ROOT

SHARE='https://nuwildcat.sharepoint.com/:f:/s/WNB-OSCAAR/Et1cBwxG3eFMgo7l6UWDNA0B98o1Z37glbdyz5f0ZzdzvQ?e=kuhfkY'
INVENTORY=ROOT/'outputs/validation/common_benchmark/lucid_archive_inventory.json'
DEST=ROOT/'dataset/public/lucid_published/textgrids'


class RangeReader(io.RawIOBase):
    def __init__(self,session,spec,block_size=65536):
        self.session=session;self.spec=spec;self.url=spec['url'];self.size=int(spec['content_range'].split('/')[-1])
        self.pos=0;self.block_size=block_size;self.cache=OrderedDict();self.bytes_downloaded=0

    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.pos
    def seek(self,offset,whence=0):
        new=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset
        if new<0:raise ValueError('Negative ZIP seek')
        self.pos=new;return new

    def block(self,index):
        if index in self.cache:return self.cache[index]
        start=index*self.block_size;end=min(self.size,start+self.block_size)-1
        r=self.session.get(self.url,headers={'Range':f'bytes={start}-{end}'},stream=True,timeout=(10,30))
        try:
            if r.status_code!=206 or r.headers.get('Content-Range')!=f'bytes {start}-{end}/{self.size}':
                raise ValueError('Server did not honor exact ZIP range; refusing a whole-audio download')
            if r.headers.get('etag') and r.headers['etag']!=self.spec['etag']:raise ValueError('Remote archive changed')
            data=r.content
            if len(data)!=end-start+1:raise ValueError('Truncated ZIP byte range')
        finally:r.close()
        self.bytes_downloaded+=len(data);self.cache[index]=data
        while len(self.cache)>128:self.cache.popitem(last=False)
        return data

    def read(self,n=-1):
        n=self.size-self.pos if n<0 else min(n,self.size-self.pos)
        if n>16*1024*1024:raise ValueError('Unexpected large ZIP text read')
        chunks=[]
        while n>0:
            index,offset=divmod(self.pos,self.block_size);block=self.block(index)
            length=min(n,len(block)-offset)
            if length<=0:raise ValueError('Unexpected ZIP range end')
            chunks.append(block[offset:offset+length]);self.pos+=length;n-=length
        return b''.join(chunks)


def acquire(spec):
    session=requests.Session();session.get(SHARE,timeout=(10,30)).raise_for_status()
    reader=RangeReader(session,spec);folder=DEST/Path(spec['archive']).stem
    folder.mkdir(parents=True,exist_ok=True);entries=[];saved=[]
    with zipfile.ZipFile(reader) as archive:
        for item in archive.infolist():
            entries.append(dict(name=item.filename,bytes=item.file_size,compressed_bytes=item.compress_size,crc32=item.CRC))
            name=Path(item.filename)
            if item.is_dir() or '__MACOSX' in name.parts or name.suffix.casefold()!='.textgrid':continue
            target=(folder/name).resolve()
            if not target.is_relative_to(folder.resolve()):raise ValueError('Unsafe ZIP entry path')
            if item.file_size>10*1024*1024:raise ValueError('Unexpected large TextGrid')
            data=archive.read(item) # zipfile checks CRC after decompression.
            if target.exists() and target.read_bytes()!=data:raise FileExistsError('Preserve differing downloaded source')
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
            saved.append(dict(name=item.filename,path=str(target.relative_to(ROOT)),bytes=len(data),
                              crc32=item.CRC,sha256=hashlib.sha256(data).hexdigest()))
        print(spec['archive'],'TextGrids',len(saved),'ZIP bytes fetched',reader.bytes_downloaded,flush=True)
    result=dict(**spec,entries=entries,textgrids=saved,bytes_downloaded=reader.bytes_downloaded,
                provenance='Archive ETag, exact content ranges, entry CRC and extracted SHA-256 retained. Full archive was NOT downloaded or SHA-hashed.')
    (folder/'source.json').write_text(json.dumps(result,indent=2)+'\n');return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory',type=Path,default=INVENTORY)
    args=parser.parse_args()
    specs=json.loads(args.inventory.read_text())
    if not specs or any(s['status']!=206 for s in specs):raise ValueError('Require UI-observed archives with exact byte-range support')
    DEST.mkdir(parents=True,exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(acquire,specs))
    summary=dict(archives=len(results),textgrids=sum(len(r['textgrids']) for r in results),
                 archive_bytes=sum(int(r['content_range'].split('/')[-1]) for r in results),
                 downloaded_bytes=sum(r['bytes_downloaded'] for r in results),sources=results)
    (DEST/'manifest.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('Acquired',summary['textgrids'],'TextGrids; fetched',summary['downloaded_bytes'],'bytes',flush=True)


if __name__=='__main__':main()
