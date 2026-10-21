# Download the matching public recordings

The script downloads the recordings that match our existing transcript evaluation.
It uses Python 3.11 or later and the standard-library HTTP client. Run these
commands from the project folder with the existing project environment. It never starts MOSS, trains a model, scores a recording, or changes
the frozen train/eval/test assignments.

## Preview the inventory

```sh
.venv/bin/python scripts/download_public_audio.py plan
```

The default inventory contains **629 recordings**:

| Source | Recordings | Original source |
|---|---:|---|
| OCSC | 301 | [TalkBank CHILDES media](https://media.talkbank.org/childes/Eng-NA/OCSC/) |
| NYU–Emerson | 28 | [TalkBank ASDBank media](https://media.talkbank.org/asd/English/NYU-Emerson/) |
| LUCID | 300 WAVs | Previously authorized SpeechBox distribution, using eight verified ZIP packages |

OCSC/NYU cover all 329 usable source recordings, with 231 training, 49 eval and
49 test children. Complete recordings include additional tasks beyond our selected
895 discourse sections. Future audio evaluation must use the existing section and
participant manifest rather than score every task in the file.

The four documented excluded CHAT files remain outside the default selection.
`--include-excluded` adds them in separate `excluded/` folders and preserves any
existing assignment in metadata. It creates no new cohort assignments. In total,
that option selects all 333 locally inventoried CHAT files, plus LUCID.

LUCID's 300 WAVs occupy **29.09 GiB uncompressed**, plus TalkBank media and any
derived NYU WAVs. The downloader does not retain duplicate full LUCID ZIP archives.
It uses the working split L2/vocoder packages, not the problematic original large
packages. Both candidate vocoder recordings remain available. Their ambiguous
mapping to the 60 published lexical rows stays explicit.

## TalkBank sign-in

The official media folders currently return TalkBank's sign-in page for anonymous
requests. Sign in through TalkBank with your authorized account and complete any
required access agreement. Supply a **Netscape-format cookie file** exported from
that authorized session using your normal trusted method:

```sh
.venv/bin/python scripts/download_public_audio.py check-access \
  --cookies /absolute/path/talkbank-cookies.txt
```

The script reads only the cookie file you specify. It never reads browser credential
stores, prints cookies, or treats an HTML sign-in page as audio. Treat the cookie
file as a credential and keep it outside the project. Expired sessions require a
fresh export. Cookie delivery follows the cookie jar's domain rules.

TLS verification remains enabled. The script uses an installed `certifi` bundle
when available, otherwise Python’s default CA store. `--ca-bundle /path/trusted-ca.pem`
selects a trusted CA bundle if your Python installation needs one.

LUCID uses the previously authorized sharing link and a server-issued anonymous
session, matching our TextGrid acquisition. The check command reads only bounded
archive ranges and media probes. It saves no complete recordings.

## Download all three sources

```sh
.venv/bin/python scripts/download_public_audio.py download \
  --cookies /absolute/path/talkbank-cookies.txt
```

On macOS, keep the computer awake during a long download:

```sh
caffeinate -i .venv/bin/python scripts/download_public_audio.py download \
  --cookies /absolute/path/talkbank-cookies.txt
```

Download sources separately:

```sh
.venv/bin/python scripts/download_public_audio.py download --sources ocsc \
  --cookies /absolute/path/talkbank-cookies.txt
.venv/bin/python scripts/download_public_audio.py download --sources nyu_emerson \
  --cookies /absolute/path/talkbank-cookies.txt
.venv/bin/python scripts/download_public_audio.py download --sources lucid
```

For faster TalkBank acquisition, add `--workers 4` (one independent client and
partial file per recording). A single coordinator writes the progress report;
the original source/cohort assignment remains attached to every job. LUCID ZIP
acquisition still uses one worker. Authenticated TalkBank listings may include
the site's login script; access is checked against actual child/file links and
media signatures. The downloader prefers the site's explicit `?f=save` download
controls over player URLs that may return only a tiny default range.

When signing in through a browser, `scripts/import_browser_audio.py` can import
an explicit JSON list of `{url, path}` download receipts into the same frozen
layout. It checks the original media signature and WAV container/frame lengths,
copies the source file without modifying it, and records local SHA256 provenance.
`scripts/talkbank_request_to_cookies.py REQUEST.txt /private/tmp/COOKIES.txt`
converts a user-exported Chrome cURL request into a private cookie file. The
request is parsed as text and never executed; credentials are never printed.

After acquisition, verify complete coverage and cohort placement:

```sh
.venv/bin/python scripts/verify_public_audio.py --sources ocsc \
  --report outputs/validation/moss_audio_v1/acquisition/ocsc_verification.json
```

The report also checks full local hashes, WAV structure and duplicate recordings
across participant identities. A missing or changed recording prevents a complete
status. These checks do not run inference, training or scoring.

For just the existing CHAT test participants:

```sh
.venv/bin/python scripts/download_public_audio.py download --sources ocsc nyu_emerson \
  --split test --cookies /absolute/path/talkbank-cookies.txt
```

Use `--split eval` for eval participants, or `--split train` for training participants.
LUCID has no CHAT cohort assignment and remains `external_reference`, even when a
CHAT split filter is present. `--limit 1` selects an intentional single-recording
smoke download. A failed source does not prevent the other sources from proceeding.
The process exits with status 2 if anything requires authentication or fails.

## Folder layout

```text
dataset/public/
  ocsc/
    audio/train/4/4001.wav
    audio/eval/...
    audio/test/...
  nyu_emerson/
    media/train/2001.mp4       # original video, when video is the available source
    media/eval/...
    media/test/...
    audio/train/2001.wav       # extracted audio
    audio/eval/...
    audio/test/...
  lucid_published/
    audio/babble/*.wav
    audio/L2/*.wav
    audio/noBarrier/*.wav
    audio/vocoder/*.wav
```

Examples illustrate the structure. Each recording follows its actual original
cohort, not the illustrative participant numbers above. Original audio formats stay
unchanged. If a TalkBank recording is available as audio rather than video, its
original audio file goes directly into that source's `audio/` folder.

NYU's official release supplies video. **ffmpeg** must be on PATH to extract its
first audio stream as PCM16 WAV. Extraction preserves the source sample rate and
channel count, and keeps the original video. This does not resample to MOSS input
format. If ffmpeg is missing, the video stays saved; rerun after installing it to
extract without downloading again. `--ffmpeg /absolute/path/ffmpeg` specifies a
binary. `--no-extract` intentionally saves original video only.

## Resume, verification and provenance

- Rerun the same command after interruption. Complete files with matching SHA-256
  metadata are verified and skipped. Ordinary media transfers resume with HTTP Range
  when supported and the source provides a version validator. If the server ignores
  Range or supplies no ETag/Last-Modified validator, only the matching temporary
  transfer restarts. Existing complete files with unknown/different provenance are preserved.
- LUCID verifies extracted length, ZIP CRC32, WAV signature and local SHA-256.
  Completed files are skipped. An interrupted compressed member must reread its
  prefix to restore decompression state. The script makes no full-archive SHA claim.
- Downloads use `.part` files. A recording appears at its final path only after
  checks pass. A neighboring `.source.json` records its cohort, original transcript
  hash, source URL or archive member, byte count and SHA-256.
- No overwrite mode exists. Foreign, changed or corrupted files trigger an error
  for review. One command at a time holds a kernel lock in the state directory.
  Final publication uses a hard link, so the destination filesystem must support
  hard links (the current project filesystem does).
- Plans and progress live in `outputs/validation/audio_acquisition_v1/`:
  `plan.json`, `access_check.json` and `download_status.json`. Join download status to
  the plan by `job_id` to recover each recording's participant and section metadata.

`--output-root /path/to/storage` changes the parent containing the source folders.
Use a separate `--state-dir` for a new destination or changed source selection policy
so the earlier acquisition plan remains preserved.

If authenticated directory layout differs from the known transcript-relative paths,
provide a JSON URL map of exact authorized download links:

```json
{
  "ocsc/transcripts/4/4001.cha": "https://media.talkbank.org/childes/Eng-NA/OCSC/4/4001.wav",
  "nyu_emerson/transcripts/2001.cha": "https://media.talkbank.org/asd/English/NYU-Emerson/2001.mp4"
}
```

Pass it with `--url-map /absolute/path/media_urls.json`. These are path examples;
the downloader verifies the returned media signature and transfer length. It also
reads authenticated directory links when available. It never circumvents an access
gate or silently claims inaccessible files were downloaded.

## Dataset attribution

Use the sources' required citations with subsequent analyses:

- Wagner et al. (2025), *The Ohio Child Speech Corpus*, Speech Communication 170,
  103206. [Paper](https://doi.org/10.1016/j.specom.2025.103206),
  [corpus](https://doi.org/10.21415/60TV-D669).
- Steen, Buonocore, Luyster, Sancimino and Arunachalam (2026), *Introducing the
  NYU–Emerson Corpus: Transcripts of parent–child interaction with young autistic
  children*, Meeting on Language in Autism. [Corpus](https://doi.org/10.21415/VT7P-YF83).
- Baese-Berk et al. (2021), *Lexical Diversity, Lexical Sophistication, and
  Predictability for Speech in Multiple Listening Conditions*.
  [Paper](https://doi.org/10.3389/fpsyg.2021.661415). The exact LUCID distribution
  links and archive ETags are retained in the saved plan and existing source catalogs.
