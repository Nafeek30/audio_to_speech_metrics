#!/usr/bin/env python3
"""Verify explicitly listed browser downloads and import into frozen cohorts."""
from pathlib import Path
import argparse
from collections import Counter
import json
import shutil
import struct
import sys
from urllib.parse import unquote, urlsplit
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from speech_pipeline.audio_download import (AUDIO, VIDEO, MEDIA_ROOTS, check_media,
    commit, existing_original, extract_audio, now, prepare_plan, save, verified)


def import_one(receipt, jobs):
    path = Path(receipt['path']).expanduser().resolve()
    url = urlsplit(receipt['url'])
    if url.scheme != 'https' or url.hostname != 'media.talkbank.org' or url.username or url.password:
        raise ValueError('Require an observed official TalkBank media URL')
    remote_path = unquote(url.path)
    extension = Path(remote_path).suffix.lower()
    if extension not in AUDIO + VIDEO:
        raise ValueError('Unsupported original media format')
    matching = [j for j in jobs if remote_path ==
        urlsplit(MEDIA_ROOTS[j['source']]).path +
        ('' if j['directory'] == '.' else j['directory']+'/') + j['media_id']+extension]
    if len(matching) != 1:
        raise ValueError('Download does not match exactly one frozen source recording')
    job = matching[0]
    prior = existing_original(job)
    if prior:
        return prior, extract_audio(prior, job)
    with path.open('rb') as handle:
        header = handle.read(8192)
    check_media(header, extension)
    validation = 'Browser download completed; local signature and SHA256 verified. No remote hash claimed.'
    if extension == '.wav':
        if header[:4] == b'RIFF' and struct.unpack('<I', header[4:8])[0]+8 != path.stat().st_size:
            raise ValueError('Truncated or inconsistent WAV container length')
        with wave.open(str(path), 'rb') as audio:
            frames = audio.getnframes()
            if not frames:
                raise ValueError('Empty WAV')
            audio.setpos(max(0, frames-1024))
            expected = min(frames, 1024)*audio.getnchannels()*audio.getsampwidth()
            if len(audio.readframes(1024)) != expected:
                raise ValueError('Truncated WAV audio frames')
        validation += ' WAV RIFF length and final declared audio frames checked.'
    target = Path((job['media_base'] if extension in VIDEO else job['audio_base'])+extension)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(str(target)+'.browser-import.part')
    if partial.exists():
        raise FileExistsError('Preserve existing import partial for review')
    shutil.copyfile(path, partial)
    result = commit(partial, target, job, kind='original_media', url=receipt['url'],
        browser_download_path=str(path), verification=validation)
    return result, extract_audio(result, job)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('receipts', type=Path, help='JSON list of {url, path} returned by browser download')
    parser.add_argument('--state-dir', type=Path, required=True)
    args = parser.parse_args()
    jobs = [j for j in prepare_plan()['jobs'] if j['source'] in MEDIA_ROOTS]
    args.state_dir.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (args.state_dir/'download.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        results = []
        for receipt in json.loads(args.receipts.read_text()):
            original, audio = import_one(receipt, jobs)
            results.append(dict(job_id=original['job_id'], source=original['source'],
                participant=original['participant'], cohort=original['cohort'], status='verified',
                original=original['path'], audio=(audio or original)['path'], sha256=original['sha256']))
        save(args.state_dir/'browser_import_status.json', dict(updated_utc=now(),
            records=results, counts=dict(Counter(r['source'] for r in results))))
        print('Verified browser imports:', len(results))


if __name__ == '__main__':
    main()
