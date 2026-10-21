#!/usr/bin/env python3
"""Verify full audio coverage, hashes and original participant cohort placement."""
from pathlib import Path
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
import struct
import sys
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from speech_pipeline.audio_download import AUDIO, VIDEO, existing_original, now, prepare_plan, save
from speech_pipeline.cohorts import verify as verify_cohorts


def verify_job(job):
    result = dict(job_id=job['id'], source=job['source'], participant=job['participant'],
        cohort=job['cohort'], transcript=job['transcript'])
    try:
        original = existing_original(job)  # Includes full SHA256 and byte count.
        if not original:
            return dict(result, status='missing')
        for key, expected in [('source', job['source']), ('cohort', job['cohort']),
                ('participant', job['participant']), ('transcript', job['transcript']),
                ('transcript_sha256', job['transcript_sha256'])]:
            if original.get(key) != expected:
                raise ValueError('Audio metadata differs from frozen source/cohort: '+key)
        path = Path(original['path'])
        extension = path.suffix.lower()
        expected_path = Path((job['media_base'] if extension in VIDEO else job['audio_base'])+extension)
        if path.resolve() != expected_path.resolve() or path.is_symlink():
            raise ValueError('Audio path differs from its frozen cohort location')
        facts = {}
        if path.suffix.lower() == '.wav':
            with path.open('rb') as handle:
                header = handle.read(12)
            if header[:4] == b'RIFF' and struct.unpack('<I', header[4:8])[0]+8 != path.stat().st_size:
                raise ValueError('WAV container size differs from complete local file')
            with wave.open(str(path), 'rb') as audio:
                frames = audio.getnframes()
                audio.setpos(max(0, frames-1024))
                expected = min(frames, 1024)*audio.getnchannels()*audio.getsampwidth()
                if not frames or len(audio.readframes(1024)) != expected:
                    raise ValueError('Empty or truncated audio frames')
                facts = dict(sample_rate=audio.getframerate(), channels=audio.getnchannels(),
                    frames=frames, seconds=frames/audio.getframerate())
        return dict(result, status='verified', path=str(path), bytes=original['bytes'],
            sha256=original['sha256'], eligible_sections=job['eligible_sections'], **facts)
    except (OSError, ValueError, RuntimeError, wave.Error) as error:
        return dict(result, status='error', detail=str(error))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', nargs='+', choices=['ocsc', 'nyu_emerson'], required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    cohorts = verify_cohorts()
    plan = prepare_plan()
    jobs = [j for j in plan['jobs'] if j['source'] in args.sources]
    with ThreadPoolExecutor(max_workers=4) as workers:
        records = list(workers.map(verify_job, jobs))
    hashes = defaultdict(list)
    for record in records:
        if record['status'] == 'verified':
            hashes[record['sha256']].append(record)
    duplicates = [[{k: r[k] for k in ('source', 'participant', 'cohort', 'path')} for r in group]
                  for group in hashes.values() if len({r['participant'] for r in group}) > 1]
    unexpected = []
    allowed = {str(Path(r['path']).resolve()) for r in records if r.get('path')}
    for source in args.sources:
        folder = Path(plan['output_root'])/source/'audio'
        unexpected.extend(str(p) for p in folder.rglob('*') if p.is_file()
            and p.suffix.lower() in AUDIO+VIDEO and str(p.resolve()) not in allowed)
    counts = dict(Counter(r['status'] for r in records))
    complete = counts.get('verified', 0) == len(jobs) and not duplicates and not unexpected
    result = dict(checked_utc=now(), complete=complete, sources=args.sources,
        expected_recordings=len(jobs), counts=counts, frozen_cohorts=cohorts,
        verified_cohorts=dict(Counter(r['cohort'] for r in records if r['status']=='verified')),
        verified_bytes=sum(r.get('bytes', 0) for r in records),
        duplicate_audio_across_participants=duplicates, unexpected_audio_paths=unexpected,
        verification='Full local SHA256, byte count, source/transcript identity, original cohort and WAV final-frame/container checks. No remote SHA256 claim.',
        records=records)
    save(args.report, result)
    print(json.dumps({k: result[k] for k in ('complete', 'expected_recordings', 'counts', 'verified_cohorts', 'verified_bytes')}, indent=2))
    return 0 if complete else 2


if __name__ == '__main__':
    raise SystemExit(main())
