import json
from pathlib import Path
import stat
from tempfile import TemporaryDirectory
import unittest
import wave

from scripts.import_browser_audio import import_one
from scripts.talkbank_request_to_cookies import convert, request_cookies
from scripts.verify_public_audio import verify_job


class BrowserAudioTests(unittest.TestCase):
    def test_request_is_parsed_as_text_and_cookie_file_is_private(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source = root/'request.txt'
            source.write_text("curl 'https://media.talkbank.org/childes/Eng-NA/OCSC/' "
                "-b 'session=example-only' -H 'User-Agent: $(do-not-execute)' ")
            target = root/'cookies.txt'
            convert(source, target)
            self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o600)
            self.assertIn('media.talkbank.org', target.read_text())
            with self.assertRaises(FileExistsError):
                convert(source, target)

    def test_only_signed_in_official_media_requests_are_accepted(self):
        for value in ["curl https://example.com -b 'session=example-only'",
                      "curl http://media.talkbank.org -b 'session=example-only'",
                      "curl https://media.talkbank.org/",
                      "sh -c 'echo unsafe'"]:
            with self.assertRaises(ValueError):
                request_cookies(value)

    def make_recording(self, root):
        path = root/'download.wav'
        with wave.open(str(path), 'wb') as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(16000)
            audio.writeframes(b'\0\0'*16000)
        job = dict(source='ocsc', directory='4', media_id='4001', id='test-job',
            audio_base=str(root/'audio/train/4/4001'), media_base=str(root/'media/train/4/4001'),
            participant='ocsc/4001', cohort='train', eligible=True,
            transcript='transcripts/4/4001.cha', transcript_sha256='test-source-sha', eligible_sections=[])
        receipt = dict(url='https://media.talkbank.org:443/childes/Eng-NA/OCSC/4/4001.wav?f=save', path=str(path))
        return path, job, receipt

    def test_import_retains_frozen_cohort_and_does_not_change_source(self):
        with TemporaryDirectory() as temp:
            source, job, receipt = self.make_recording(Path(temp))
            original = source.read_bytes()
            meta, derived = import_one(receipt, [job])
            self.assertEqual(meta['participant'], 'ocsc/4001')
            self.assertEqual(meta['cohort'], 'train')
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(Path(meta['path']).read_bytes(), original)
            self.assertIsNone(derived)
            self.assertEqual(import_one(receipt, [job])[0], meta)

    def test_truncated_wav_is_rejected_before_publication(self):
        with TemporaryDirectory() as temp:
            source, job, receipt = self.make_recording(Path(temp))
            source.write_bytes(source.read_bytes()[:-100])
            with self.assertRaisesRegex(ValueError, 'Truncated'):
                import_one(receipt, [job])
            self.assertFalse(Path(job['audio_base']+'.wav').exists())

    def test_unregistered_or_foreign_recording_cannot_be_imported(self):
        with TemporaryDirectory() as temp:
            source, job, receipt = self.make_recording(Path(temp))
            for url in [receipt['url'].replace('4001.wav','4002.wav'),
                        receipt['url'].replace('media.talkbank.org','example.com')]:
                with self.assertRaises(ValueError):
                    import_one(dict(receipt, url=url), [job])

    def test_verifier_checks_recording_and_preserves_cohort(self):
        with TemporaryDirectory() as temp:
            source, job, receipt = self.make_recording(Path(temp))
            import_one(receipt, [job])
            record=verify_job(job)
            self.assertEqual(record['status'],'verified')
            self.assertEqual(record['cohort'],'train')
            self.assertEqual(record['seconds'],1)

    def test_verifier_rejects_relabelled_provenance(self):
        with TemporaryDirectory() as temp:
            source, job, receipt = self.make_recording(Path(temp))
            meta,_=import_one(receipt,[job])
            sidecar=Path(meta['path']+'.source.json')
            altered=json.loads(sidecar.read_text());altered['cohort']='test'
            sidecar.write_text(json.dumps(altered))
            self.assertEqual(verify_job(job)['status'],'error')

    def test_verifier_does_not_accept_missing_recordings(self):
        with TemporaryDirectory() as temp:
            source, job, receipt = self.make_recording(Path(temp))
            self.assertEqual(verify_job(job)['status'],'missing')

    def test_verifier_rejects_provenance_pointing_to_another_location(self):
        with TemporaryDirectory() as temp:
            source, job, receipt = self.make_recording(Path(temp))
            meta,_=import_one(receipt,[job])
            sidecar=Path(meta['path']+'.source.json')
            altered=json.loads(sidecar.read_text());altered['path']=str(source)
            sidecar.write_text(json.dumps(altered))
            self.assertEqual(verify_job(job)['status'],'error')


if __name__ == '__main__':
    unittest.main()
