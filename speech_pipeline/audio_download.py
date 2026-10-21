"""Acquire the public recordings that match this project's frozen transcripts.

Standard-library HTTP client with an optional installed certifi CA bundle.
This command does not train, score, or assign cohorts.
"""
from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import closing
from datetime import datetime, timezone
import hashlib
from http.cookiejar import CookieJar, MozillaCookieJar, LoadError
from http.client import HTTPException
from html.parser import HTMLParser
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import ssl
import subprocess
import threading
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, quote, unquote, urljoin, urlsplit, urlunsplit
from urllib.request import HTTPCookieProcessor, HTTPSHandler, Request, build_opener
import zipfile
import zlib

from speech_pipeline import ROOT

SOURCES = ('ocsc', 'nyu_emerson', 'lucid')
MEDIA_ROOTS = {
    'ocsc': 'https://media.talkbank.org/childes/Eng-NA/OCSC/',
    'nyu_emerson': 'https://media.talkbank.org/asd/English/NYU-Emerson/',
}
PAGES = {
    'ocsc': 'https://talkbank.org/childes/access/Eng-NA/OCSC.html',
    'nyu_emerson': 'https://talkbank.org/asd/access/English/NYU-Emerson.html',
    'lucid': 'https://doi.org/10.3389/fpsyg.2021.661415',
}
LUCID_SHARE = 'https://nuwildcat.sharepoint.com/:f:/s/WNB-OSCAAR/Et1cBwxG3eFMgo7l6UWDNA0B98o1Z37glbdyz5f0ZzdzvQ?e=kuhfkY'
AUDIO = ('.wav', '.flac', '.m4a', '.mp3', '.ogg')
VIDEO = ('.mp4', '.mov', '.webm', '.m4v')
CHUNK = 1024 * 1024


class AuthenticationRequired(RuntimeError):
    pass


def now():
    return datetime.now(timezone.utc).isoformat()


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(CHUNK), b''):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)


def safe_relative(value):
    value = str(value)
    p = PurePosixPath(value)
    if not value or '\\' in value or p.is_absolute() or '..' in p.parts or ':' in value:
        raise ValueError('Unsafe relative media path')
    return p


def inside(base, relative):
    base = Path(base).resolve()
    target = base.joinpath(*safe_relative(relative).parts).resolve()
    if not target.is_relative_to(base):
        raise ValueError('Media path escapes destination')
    return target


def sidecar(path):
    return Path(str(path) + '.source.json')


def status_code(response):
    return response.status if hasattr(response, 'status') else response.getcode()


def looks_html(data):
    sample = data[:8192].lstrip().lower()
    return sample.startswith((b'<!doctype html', b'<html', b'<body', b'<script')) or b'authmodals' in sample


def check_media(data, extension, content_type=''):
    if looks_html(data) or 'text/html' in content_type.lower():
        raise AuthenticationRequired('Server returned an HTML page, not media. Sign in and supply --cookies.')
    ok = {
        '.wav': len(data) >= 12 and data[:4] in (b'RIFF', b'RF64', b'RIFX') and data[8:12] == b'WAVE',
        '.flac': data.startswith(b'fLaC'),
        '.mp3': data.startswith(b'ID3') or len(data) > 1 and data[0] == 255 and data[1] & 224 == 224,
        '.ogg': data.startswith(b'OggS'),
        '.webm': data.startswith(b'\x1aE\xdf\xa3'),
    }
    if extension in ('.m4a', '.mp4', '.mov', '.m4v'):
        valid = len(data) >= 12 and data[4:8] in (b'ftyp', b'moov', b'wide', b'free', b'mdat')
    else:
        valid = ok.get(extension, False)
    if not valid:
        raise ValueError(f'Response has no valid {extension} media signature')


class Client:
    def __init__(self, cookies=None, timeout=60, ca_bundle=None):
        jar = CookieJar()
        if cookies:
            jar = MozillaCookieJar(str(Path(cookies).expanduser()))
            try:
                jar.load(ignore_discard=True, ignore_expires=False)
            except (OSError, LoadError):
                raise ValueError('Cannot load the provided cookie file. Require valid Netscape format and a current session.') from None
        certificate_file = ca_bundle or os.environ.get('SSL_CERT_FILE')
        if not certificate_file:
            try:
                import certifi
                certificate_file = certifi.where()
            except ImportError:
                pass  # A correctly configured Python uses its default CA store.
        context = ssl.create_default_context(cafile=str(Path(certificate_file).expanduser()) if certificate_file else None)
        self.opener = build_opener(HTTPCookieProcessor(jar), HTTPSHandler(context=context))
        self.timeout = timeout
        self.listings = {}

    def open(self, url, headers=None):
        parsed = urlsplit(url)
        if parsed.scheme not in ('https', 'http') or parsed.username or parsed.password:
            raise ValueError('Expected a media HTTP(S) URL without embedded credentials')
        # Source catalogs retain readable SharePoint paths containing spaces.
        # Preserve existing percent escapes, especially signed/encoded paths.
        encoded_url = urlunsplit((parsed.scheme, parsed.netloc,
            quote(parsed.path, safe="/%:@!$&'()*+,;=-._~"),
            quote(parsed.query, safe="=&%:@/+,;?-._~"), ''))
        request = Request(encoded_url, headers={'User-Agent': 'SpeechPipelineAudio/1.0', 'Accept-Encoding': 'identity', **(headers or {})})
        try:
            return self.opener.open(request, timeout=self.timeout)
        except HTTPError as e:
            if e.code in (401, 403):
                raise AuthenticationRequired('Media access requires an authorized signed-in session (--cookies).') from e
            raise

    def authorize_lucid(self):
        with closing(self.open(LUCID_SHARE)) as r:
            # Cookies from this previously authorized sharing link stay in memory.
            if status_code(r) != 200:
                raise AuthenticationRequired('LUCID sharing session is unavailable')

    def probe(self, url, extension):
        with closing(self.open(url, {'Range': 'bytes=0-63'})) as r:
            data = r.read(8192)
            check_media(data, extension, r.headers.get('Content-Type', ''))
            status = status_code(r)
            if status == 206:
                match = re.fullmatch(r'bytes 0-\d+/(\d+)', r.headers.get('Content-Range', ''))
                if not match:
                    raise ValueError('Invalid media probe Content-Range')
                size = int(match[1])
            elif status == 200:
                size = int(r.headers['Content-Length']) if r.headers.get('Content-Length') else None
            else:
                raise ValueError(f'Unexpected media status {status}')
            return dict(url=url, extension=extension, bytes=size, etag=r.headers.get('ETag'), last_modified=r.headers.get('Last-Modified'))

    def directory(self, url):
        if url not in self.listings:
            with closing(self.open(url)) as r:
                body = r.read(2 * CHUNK + 1)
                if len(body) > 2 * CHUNK:
                    raise ValueError('Unexpectedly large directory listing')
                parser = Links()
                parser.feed(body.decode('utf-8', errors='replace'))
                links = [urljoin(url, h) for h in parser.links]
                requested = urlsplit(url)
                prefix = requested.path.rstrip('/')+'/'
                # TalkBank includes authModals.js on authenticated listings too.
                # Require actual child links before treating that script as an
                # access failure. Parent/login links alone do not establish access.
                children = [h for h in links if urlsplit(h).scheme in ('https', 'http')
                    and urlsplit(h).hostname == requested.hostname
                    and urlsplit(h).path.startswith(prefix)
                    and len(urlsplit(h).path) > len(prefix)]
                if b'not authorized' in body.lower() and not children:
                    raise AuthenticationRequired('The signed-in TalkBank account is not authorized for this corpus. Account access is required.')
                if b'authmodals' in body.lower() and not children:
                    raise AuthenticationRequired('TalkBank returned its sign-in page. Supply a Netscape cookie file.')
                self.listings[url] = links
        return self.listings[url]


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() == 'a':
            self.links.extend(v for k, v in attrs if k.lower() == 'href' and v)


def prepare_plan(project=ROOT, output_root=None, include_excluded=False):
    project = Path(project).resolve()
    output_root = Path(output_root or project/'dataset/public').expanduser().resolve()
    bundle = project/'outputs/validation/cohorts_v1'
    manifest_path = bundle/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    inputs = {str(manifest_path.relative_to(project)): sha_file(manifest_path)}
    lookup = {}
    for role, participants in manifest['assignments'].items():
        for participant in participants:
            if participant in lookup:
                raise ValueError('Frozen cohort assignments overlap')
            lookup[participant] = role
    sections = {}
    for role, corpora in manifest['files'].items():
        for corpus, spec in corpora.items():
            p = inside(bundle, spec['path'])
            if sha_file(p) != spec['sha256']:
                raise ValueError('Frozen cohort file changed')
            inputs[str(p.relative_to(project))] = spec['sha256']
            for line in p.read_text().splitlines():
                row = json.loads(line)
                if lookup[row['participant']] != role:
                    raise ValueError('Prepared participant cohort differs')
                recording = row['recording'].split('#section')[0]
                source_key = corpus, recording
                entry = sections.setdefault(source_key, dict(participant=row['participant'], cohort=role, source_sha256=row['source_sha256'], sections=[]))
                if (entry['participant'], entry['cohort'], entry['source_sha256']) != (row['participant'], role, row['source_sha256']):
                    raise ValueError('One source has conflicting participant roles')
                entry['sections'].append(dict(recording=row['recording'], gem=row['gem']))
    exclusions = {str(Path(r['file'])): r['reason'] for r in manifest['exclusions']}
    jobs = []
    for corpus in ('ocsc', 'nyu_emerson'):
        source_root = project/'dataset/public'/corpus
        for source in sorted((source_root/'transcripts').rglob('*.cha')):
            relative = source.relative_to(source_root).as_posix()
            prepared = sections.get((corpus, relative))
            excluded = exclusions.get(source.relative_to(project).as_posix())
            if not prepared and not excluded:
                raise ValueError('Uninventoried transcript; do not infer a new participant split')
            if not prepared and not include_excluded:
                continue
            text = source.read_text(encoding='utf-8-sig')
            match = re.search(r'^@Media:\s*([^,\r\n]+),\s*([^\r\n]+)', text, re.M)
            if not match:
                raise ValueError(f'Transcript has no @Media header: {relative}')
            media_id, kind = match[1].strip(), match[2].strip().lower()
            safe_relative(media_id)
            # These excluded files are only mapped when the frozen manifest already assigns them.
            participant = prepared['participant'] if prepared else corpus+'/'+source.stem
            role = prepared['cohort'] if prepared else lookup.get(participant)
            actual_sha = sha_file(source)
            if prepared and actual_sha != prepared['source_sha256']:
                raise ValueError('Original CHAT source changed')
            physical_role = {'validation': 'eval'}.get(role, role or 'unassigned')
            if not prepared:
                physical_role = 'excluded/'+physical_role
            stem = Path(relative).relative_to('transcripts').with_suffix('').as_posix()
            job = dict(source=corpus, transcript=relative, transcript_sha256=actual_sha,
                participant=participant, cohort=role, eligible=bool(prepared), exclusion_reason=excluded,
                eligible_sections=prepared['sections'] if prepared else [], media_id=media_id, media_kind=kind,
                directory=Path(relative).relative_to('transcripts').parent.as_posix(),
                audio_base=str(output_root/corpus/'audio'/physical_role/stem),
                media_base=str(output_root/corpus/'media'/physical_role/stem), source_page=PAGES[corpus])
            job['id'] = fingerprint(job)
            jobs.append(job)
    catalog_root = project/'dataset/public/lucid_published/textgrids'
    catalogs = sorted(catalog_root.glob('*/source.json'))
    if len(catalogs) != 8:
        raise ValueError('Require the eight verified LUCID packages (split L2/vocoder, babble and noBarrier)')
    for catalog in catalogs:
        spec = json.loads(catalog.read_text())
        inputs[str(catalog.relative_to(project))] = sha_file(catalog)
        archive = spec['archive']
        condition = next((v for k, v in [('babbleCondition', 'babble'), ('noBarrierCondition', 'noBarrier'), ('L2Condition', 'L2'), ('vocoderCondition', 'vocoder')] if archive.startswith(k)), None)
        if not condition:
            raise ValueError('Unknown LUCID condition')
        archive_spec = {k: spec[k] for k in ('archive', 'url', 'etag', 'content_range')}
        for entry in spec['entries']:
            member = safe_relative(entry['name'])
            if member.suffix.lower() != '.wav' or '__MACOSX' in member.parts or member.name.startswith('._'):
                continue
            textgrid = catalog.parent/Path(str(member.with_suffix('.TextGrid')))
            if not textgrid.is_file():
                raise ValueError('LUCID WAV lacks its matching original TextGrid')
            target = output_root/'lucid_published/audio'/condition/member.name
            job = dict(source='lucid', cohort='external_reference', eligible=True, condition=condition,
                transcript=str(textgrid.relative_to(project)), transcript_sha256=sha_file(textgrid),
                archive=archive_spec, member=dict(entry), target=str(target), source_page=PAGES['lucid'],
                published_mapping='ambiguous_vocoder' if condition == 'vocoder' else 'unique_metadata_mapping')
            job['id'] = fingerprint(job)
            jobs.append(job)
    ids = [j['id'] for j in jobs]
    targets = [j.get('target', j.get('audio_base')) for j in jobs]
    if len(ids) != len(set(ids)) or len(targets) != len(set(targets)):
        raise ValueError('Duplicate planned recording or target path')
    return dict(schema_version=1, purpose='Recording acquisition only; frozen cohorts retained, no training or scoring',
        output_root=str(output_root), input_hashes=inputs, include_excluded=include_excluded, jobs=jobs)


def verified(path, job_id):
    path = Path(path)
    if not path.exists():
        return None
    meta_path = sidecar(path)
    if not meta_path.exists():
        raise FileExistsError(f'Preserve existing media without this downloader’s metadata: {path}')
    meta = json.loads(meta_path.read_text())
    if meta['job_id'] != job_id or meta['bytes'] != path.stat().st_size or meta['sha256'] != sha_file(path):
        raise ValueError(f'Existing media or its provenance differs: {path}')
    return meta


def metadata(path, job, **extra):
    data = dict(job_id=job['id'], source=job['source'], cohort=job['cohort'], participant=job.get('participant'),
        eligible=job.get('eligible'), published_mapping=job.get('published_mapping'), transcript=job['transcript'],
        transcript_sha256=job['transcript_sha256'], path=str(path), bytes=Path(path).stat().st_size,
        sha256=sha_file(path), acquired_utc=now(), **extra)
    save(sidecar(path), data)
    return data


def commit(partial, path, job, **extra):
    """Write provenance first, then publish without replacing an existing file."""
    if path.exists():
        raise FileExistsError(f'Preserve existing media: {path}')
    data = dict(job_id=job['id'], source=job['source'], cohort=job['cohort'], participant=job.get('participant'),
        eligible=job.get('eligible'), published_mapping=job.get('published_mapping'), transcript=job['transcript'],
        transcript_sha256=job['transcript_sha256'], path=str(path), bytes=partial.stat().st_size,
        sha256=sha_file(partial), acquired_utc=now(), **extra)
    save(sidecar(path), data)
    # Same directory/filesystem, atomic no-clobber publication on macOS/Linux.
    os.link(partial, path)
    partial.unlink()
    return data


def http_download(client, remote, path, job):
    path = Path(path)
    existing = verified(path, job['id'])
    if existing:
        return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    partial, partial_meta = Path(str(path)+'.part'), Path(str(path)+'.part.json')
    identity = dict(job_id=job['id'], url=remote['url'], etag=remote.get('etag'), bytes=remote.get('bytes'))
    if remote.get('last_modified'):
        identity['last_modified'] = remote['last_modified']
    if partial.exists():
        if not partial_meta.exists() or json.loads(partial_meta.read_text()) != identity:
            raise ValueError('Partial download belongs to different source/version; preserve and review it')
    offset = partial.stat().st_size if partial.exists() else 0
    if offset and not remote.get('etag') and not remote.get('last_modified'):
        offset = 0  # Without a version validator, never append potentially changed source bytes.
    total = remote.get('bytes')
    if total is not None and offset > total:
        raise ValueError('Partial download is larger than the declared source')
    if not (total is not None and offset == total):
        headers = {}
        if offset:
            headers['Range'] = f'bytes={offset}-'
        if remote.get('etag') and not remote['etag'].startswith('W/'):
            headers['If-Match'] = remote['etag']
        elif remote.get('last_modified'):
            headers['If-Unmodified-Since'] = remote['last_modified']
        with closing(client.open(remote['url'], headers)) as response:
            status = status_code(response)
            first = response.read(CHUNK)
            if looks_html(first) or 'text/html' in response.headers.get('Content-Type', '').lower():
                raise AuthenticationRequired('Refusing to save an HTML login page as a recording')
            etag = response.headers.get('ETag')
            if remote.get('etag') and etag and remote['etag'] != etag:
                raise ValueError('Remote media version changed')
            if status == 206:
                match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range', ''))
                if not match or int(match[1]) != offset or total is not None and int(match[3]) != total:
                    raise ValueError('Incorrect resumed Content-Range')
                total = int(match[3])
            elif status == 200:
                # A server may ignore Range. Restart only this matching temporary download.
                offset = 0
                if total is not None and response.headers.get('Content-Length') and int(response.headers['Content-Length']) != total:
                    raise ValueError('Remote media length changed from the source probe')
                total = int(response.headers['Content-Length']) if response.headers.get('Content-Length') else total
            else:
                raise ValueError(f'Unexpected download status {status}')
            if not offset:
                check_media(first, remote['extension'], response.headers.get('Content-Type', ''))
            save(partial_meta, identity)
            with partial.open('ab' if offset else 'wb') as output:
                output.write(first)
                for chunk in iter(lambda: response.read(CHUNK), b''):
                    output.write(chunk)
    if total is not None and partial.stat().st_size != total:
        raise ValueError('Truncated recording; matching partial retained for resume')
    with partial.open('rb') as f:
        check_media(f.read(8192), remote['extension'])
    result = commit(partial, path, job, kind='original_media', url=remote['url'], etag=remote.get('etag'), expected_bytes=total)
    partial_meta.unlink(missing_ok=True)
    return result


class RangeZipReader(io.RawIOBase):
    """Bounded HTTP ranges. Never accept a server's whole-archive fallback."""
    def __init__(self, client, spec, block_size=8*CHUNK):
        self.client, self.spec = client, spec
        self.block_size = block_size
        self.size = int(spec['content_range'].split('/')[-1])
        self.pos = 0
        self.cache = OrderedDict()

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        if self.pos < 0:
            raise ValueError('Negative ZIP offset')
        return self.pos

    def read(self, count=-1):
        count = min(count if count >= 0 else self.size-self.pos, self.size-self.pos)
        if count > 16*CHUNK:
            raise ValueError('Unexpectedly large ZIP range read')
        chunks = []
        while count > 0:
            block, within = divmod(self.pos, self.block_size)
            if block not in self.cache:
                start, end = block*self.block_size, min((block+1)*self.block_size, self.size)-1
                with closing(self.client.open(self.spec['url'], {'Range': f'bytes={start}-{end}', 'If-Match': self.spec['etag']})) as r:
                    if status_code(r) != 206 or r.headers.get('Content-Range') != f'bytes {start}-{end}/{self.size}':
                        raise ValueError('LUCID server did not honor exact byte range; whole ZIP refused')
                    if r.headers.get('ETag') and r.headers['ETag'] != self.spec['etag']:
                        raise ValueError('LUCID archive changed from the verified source version')
                    data = r.read(end-start+2)
                if len(data) != end-start+1:
                    raise ValueError('Truncated LUCID archive range')
                self.cache[block] = data
                while len(self.cache) > 4:
                    self.cache.popitem(last=False)
            data = self.cache[block]
            take = min(count, len(data)-within)
            if take <= 0:
                raise ValueError('Unexpected ZIP end')
            chunks.append(data[within:within+take])
            self.pos += take
            count -= take
        return b''.join(chunks)


def zip_download(archive, job):
    path = Path(job['target'])
    existing = verified(path, job['id'])
    if existing:
        return existing
    expected = job['member']
    info = archive.getinfo(expected['name'])
    if info.file_size != expected['bytes'] or info.CRC != expected['crc32'] or (info.external_attr >> 16) & 0o170000 == 0o120000:
        raise ValueError('LUCID entry differs from the frozen catalog or is a symlink')
    path.parent.mkdir(parents=True, exist_ok=True)
    partial, partial_meta = Path(str(path)+'.part'), Path(str(path)+'.part.json')
    identity = dict(job_id=job['id'], archive=job['archive'], member=expected)
    if partial.exists() and (not partial_meta.exists() or json.loads(partial_meta.read_text()) != identity):
        raise ValueError('LUCID partial belongs to another source/version')
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > info.file_size:
        raise ValueError('LUCID partial is oversized')
    save(partial_meta, identity)
    with archive.open(info) as source, partial.open('ab' if offset else 'wb') as output:
        # Deflate must reread the prefix to restore decompression/CRC state.
        if offset:
            source.seek(offset)
        for chunk in iter(lambda: source.read(CHUNK), b''):
            output.write(chunk)
    crc = 0
    with partial.open('rb') as f:
        first = f.read(8192)
        check_media(first, '.wav')
        crc = zlib.crc32(first)
        for chunk in iter(lambda: f.read(CHUNK), b''):
            crc = zlib.crc32(chunk, crc)
    if partial.stat().st_size != info.file_size or crc & 0xffffffff != info.CRC:
        raise ValueError('LUCID WAV length or CRC mismatch; partial preserved')
    result = commit(partial, path, job, kind='original_media', archive=job['archive'], member=expected,
        verification='Extracted size, ZIP CRC32, WAV signature and local SHA256. No whole-archive SHA256 claimed.')
    partial_meta.unlink(missing_ok=True)
    return result


def resolve_media(client, job, url_map):
    key = job['source']+'/'+job['transcript']
    if key in url_map:
        url = url_map[key]
        extension = PurePosixPath(unquote(urlsplit(url).path)).suffix.lower()
        return client.probe(url, extension)
    root = MEDIA_ROOTS[job['source']]
    directory = '' if job['directory'] == '.' else job['directory'].strip('/')+'/'
    relative = str(safe_relative(job['media_id']))
    # Prefer original audio; NYU's source @Media explicitly identifies video.
    extensions = VIDEO+AUDIO if job['media_kind'].startswith('video') else AUDIO+VIDEO
    candidates = []
    for folder in [urljoin(root, quote(directory, safe='/')), root]:
        try:
            links = client.directory(folder)
        except HTTPError as e:
            if e.code != 404: raise
            links = []
        for extension in extensions:
            for url in links:
                parsed = urlsplit(url)
                name = PurePosixPath(unquote(parsed.path))
                root_url = urlsplit(root)
                if (parsed.hostname == root_url.hostname and parsed.path.startswith(root_url.path)
                        and name.stem == PurePosixPath(relative).name and name.suffix.lower() == extension):
                    candidates.append((url, extension))
    for extension in extensions:
        for prefix in dict.fromkeys([directory, '']):
            candidates.append((urljoin(root, quote(prefix+relative+extension, safe='/')), extension))
    # Use the site's explicit download control. Its ordinary media/player URL
    # can return an eleven-byte default range even for a GET without Range.
    ordered = sorted(dict.fromkeys(candidates),
        key=lambda pair: parse_qs(urlsplit(pair[0]).query).get('f') != ['save'])
    for url, extension in ordered:
        try:
            return client.probe(url, extension)
        except HTTPError as e:
            if e.code not in (404, 410): raise
    raise FileNotFoundError('No matching authorized media. Supply an exact URL with --url-map if the host layout differs.')


def extract_audio(media, job, ffmpeg='ffmpeg'):
    source = Path(media['path'])
    if source.suffix.lower() not in VIDEO:
        return None
    target = Path(job['audio_base']+'.wav')
    if verified(target, job['id']):
        meta = json.loads(sidecar(target).read_text())
        if meta.get('original_media_sha256') != media['sha256']:
            raise ValueError('Derived WAV references different original media')
        return meta
    binary = shutil.which(ffmpeg)
    if not binary:
        raise FileNotFoundError('Video saved. Install ffmpeg, then rerun to extract audio without downloading again.')
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(str(target)+'.extracting.wav')
    result = subprocess.run([binary, '-hide_banner', '-loglevel', 'error', '-nostdin', '-y', '-i', str(source),
        '-map', '0:a:0', '-vn', '-c:a', 'pcm_s16le', str(partial)], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError('ffmpeg could not extract the audio stream; original video retained. '+result.stderr[-600:])
    with partial.open('rb') as f: check_media(f.read(8192), '.wav')
    return commit(partial, target, job, kind='derived_audio', original_media=str(source), original_media_sha256=media['sha256'],
        conversion='First audio stream, PCM16 WAV; original channel count and sample rate retained; no downmix or resampling')


def existing_original(job):
    if job['source'] == 'lucid':
        path = Path(job['target'])
        return verified(path, job['id'])
    for extension in AUDIO+VIDEO:
        base = job['media_base'] if extension in VIDEO else job['audio_base']
        path = Path(base+extension)
        if path.exists():
            meta = verified(path, job['id'])
            if meta.get('kind') == 'original_media':
                return meta
    return None


def download_jobs(jobs, client, report_path, url_map=None, extract=True, ffmpeg='ffmpeg'):
    results = []
    blocked = {}
    archive_name = None
    archive = None
    lucid_ready = False
    try:
        for index, job in enumerate(jobs, 1):
            source = job['source']
            try:
                media = existing_original(job)
                if not media:
                    if source in blocked:
                        results.append(dict(job_id=job['id'], source=source, cohort=job['cohort'], status='authentication_required', detail=blocked[source]))
                        continue
                    if source == 'lucid':
                        if not lucid_ready:
                            client.authorize_lucid()
                            lucid_ready = True
                        if archive_name != job['archive']['archive']:
                            if archive: archive.close()
                            archive = zipfile.ZipFile(RangeZipReader(client, job['archive']))
                            archive_name = job['archive']['archive']
                        media = zip_download(archive, job)
                    else:
                        remote = resolve_media(client, job, url_map or {})
                        base = job['media_base'] if remote['extension'] in VIDEO else job['audio_base']
                        media = http_download(client, remote, Path(base+remote['extension']), job)
                derived = extract_audio(media, job, ffmpeg) if extract and source != 'lucid' else None
                audio_path = derived['path'] if derived else None if Path(media['path']).suffix.lower() in VIDEO else media['path']
                results.append(dict(job_id=job['id'], source=source, cohort=job['cohort'], participant=job.get('participant'), status='verified',
                    original=media['path'], audio=audio_path, sha256=media['sha256']))
                print(f'[{index}/{len(jobs)}] {source}: verified {Path(media["path"]).name}', flush=True)
            except (OSError, ValueError, RuntimeError, URLError, HTTPException, zipfile.BadZipFile) as error:
                authentication = isinstance(error, AuthenticationRequired)
                if authentication:
                    blocked[source] = str(error)
                results.append(dict(job_id=job['id'], source=source, cohort=job['cohort'],
                    status='authentication_required' if authentication else 'error', detail=str(error)))
                print(f'[{index}/{len(jobs)}] {source}: {results[-1]["status"]}: {error}', flush=True)
            finally:
                save(report_path, dict(updated_utc=now(), requested=len(jobs), counts=dict(Counter(r['status'] for r in results)), records=results))
    finally:
        if archive: archive.close()
    return results


def download_talkbank_parallel(jobs, client_factory, report_path, url_map=None,
                              extract=True, ffmpeg='ffmpeg', workers=4):
    """Independent media transfers; one client per thread, one report writer."""
    if not 1 <= workers <= 8 or any(j['source'] not in MEDIA_ROOTS for j in jobs):
        raise ValueError('Parallel transfers require 1–8 workers and TalkBank jobs only')
    identities = [j['id'] for j in jobs]
    if len(identities) != len(set(identities)):
        raise ValueError('Duplicate parallel media job')
    clients = threading.local()

    def acquire(job):
        try:
            media = existing_original(job)
            if not media:
                if not hasattr(clients, 'client'):
                    clients.client = client_factory()
                remote = resolve_media(clients.client, job, url_map or {})
                base = job['media_base'] if remote['extension'] in VIDEO else job['audio_base']
                media = http_download(clients.client, remote, Path(base+remote['extension']), job)
            derived = extract_audio(media, job, ffmpeg) if extract else None
            audio = derived['path'] if derived else None if Path(media['path']).suffix.lower() in VIDEO else media['path']
            return dict(job_id=job['id'], source=job['source'], cohort=job['cohort'],
                participant=job['participant'], status='verified', original=media['path'],
                audio=audio, sha256=media['sha256'])
        except (OSError, ValueError, RuntimeError, URLError, HTTPException) as error:
            return dict(job_id=job['id'], source=job['source'], cohort=job['cohort'],
                status='authentication_required' if isinstance(error, AuthenticationRequired) else 'error',
                detail=str(error))

    finished = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(acquire, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            record = future.result()
            finished[job['id']] = record
            results = [finished[key] for key in identities if key in finished]
            save(report_path, dict(updated_utc=now(), requested=len(jobs), workers=workers,
                counts=dict(Counter(r['status'] for r in results)), records=results))
            print(f'[{len(results)}/{len(jobs)}] {job["source"]}: {record["status"]} {job["media_id"]}', flush=True)
    return [finished[key] for key in identities]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['plan', 'check-access', 'download'])
    parser.add_argument('--sources', nargs='+', choices=['all', *SOURCES], default=['all'])
    parser.add_argument('--split', choices=['all', 'train', 'eval', 'test'], default='all', help='CHAT roles only. LUCID stays external_reference.')
    parser.add_argument('--output-root', type=Path, default=ROOT/'dataset/public')
    parser.add_argument('--state-dir', type=Path, default=ROOT/'outputs/validation/audio_acquisition_v1')
    parser.add_argument('--cookies', type=Path, help='User-provided Netscape cookie file. Never reads browser credential stores.')
    parser.add_argument('--ca-bundle', type=Path, help='Optional trusted CA PEM file. TLS verification always remains enabled.')
    parser.add_argument('--url-map', type=Path, help='Optional JSON object mapping corpus/transcripts/path.cha to an exact authorized URL')
    parser.add_argument('--include-excluded', action='store_true', help='Also acquire the four documented excluded sources in separate excluded folders')
    parser.add_argument('--no-extract', action='store_true', help='Save original video only; default extracts WAV while retaining the video')
    parser.add_argument('--ffmpeg', default='ffmpeg')
    parser.add_argument('--limit', type=int, help='Limit selected recordings for an intentional smoke download')
    parser.add_argument('--workers', type=int, choices=range(1, 9), default=1,
                        help='1–8 independent TalkBank transfers; LUCID requires 1')
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    state = args.state_dir.expanduser().resolve()
    state.mkdir(parents=True, exist_ok=True)
    # macOS/Linux: kernel releases this lock even after interruption.
    import fcntl
    with (state/'download.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error('Another acquisition command holds this state directory’s lock')
        plan = prepare_plan(output_root=args.output_root, include_excluded=args.include_excluded)
        plan_path = state/('plan_with_excluded.json' if args.include_excluded else 'plan.json')
        if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
            raise ValueError('Saved plan differs. Preserve it and choose a new --state-dir.')
        if not plan_path.exists(): save(plan_path, plan)
        chosen = set(SOURCES if 'all' in args.sources else args.sources)
        role = 'validation' if args.split == 'eval' else args.split
        jobs = [j for j in plan['jobs'] if j['source'] in chosen and (role == 'all' or j['source'] == 'lucid' or j['cohort'] == role)]
        if args.limit: jobs = jobs[:args.limit]
        print('Selected recordings:', dict(Counter(j['source'] for j in jobs)))
        print('CHAT roles:', dict(Counter(j['cohort'] for j in jobs if j['source'] != 'lucid')))
        lucid_bytes = sum(j['member']['bytes'] for j in jobs if j['source'] == 'lucid')
        if lucid_bytes: print(f'LUCID uncompressed WAVs: {lucid_bytes/1024**3:.2f} GiB (plus TalkBank media; video WAV extraction needs extra space)')
        print('Plan:', plan_path)
        if args.command == 'plan': return 0
        client = Client(args.cookies, ca_bundle=args.ca_bundle)
        url_map = json.loads(args.url_map.read_text()) if args.url_map else {}
        if args.command == 'check-access':
            checks = []
            for source in SOURCES:
                job = next((j for j in jobs if j['source'] == source), None)
                if not job: continue
                try:
                    if source == 'lucid':
                        client.authorize_lucid()
                        with zipfile.ZipFile(RangeZipReader(client, job['archive'], block_size=65536)) as z:
                            info = z.getinfo(job['member']['name'])
                            if info.CRC != job['member']['crc32'] or info.file_size != job['member']['bytes']:
                                raise ValueError('Archive entry metadata changed')
                    else: resolve_media(client, job, url_map)
                    checks.append(dict(source=source, status='accessible'))
                except (OSError, ValueError, RuntimeError, URLError, HTTPException, zipfile.BadZipFile) as error:
                    checks.append(dict(source=source, status='authentication_required' if isinstance(error, AuthenticationRequired) else 'error', detail=str(error)))
                print(source+': '+checks[-1]['status'])
            save(state/'access_check.json', dict(checked_utc=now(), recordings_downloaded=0, checks=checks))
            return 0 if all(c['status'] == 'accessible' for c in checks) else 2
        if args.workers > 1:
            results = download_talkbank_parallel(jobs,
                lambda: Client(args.cookies, ca_bundle=args.ca_bundle),
                state/'download_status.json', url_map, not args.no_extract, args.ffmpeg, args.workers)
        else:
            results = download_jobs(jobs, client, state/'download_status.json', url_map, not args.no_extract, args.ffmpeg)
        counts = Counter(r['status'] for r in results)
        print('Finished:', dict(counts))
        return 0 if counts['verified'] == len(jobs) else 2


if __name__ == '__main__':
    raise SystemExit(main())
