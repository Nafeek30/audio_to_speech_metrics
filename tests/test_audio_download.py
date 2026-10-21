import io
import json
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch
import wave
import zipfile

from speech_pipeline import audio_download as a


def wav_bytes():
    out = io.BytesIO()
    with wave.open(out, 'wb') as w:
        w.setnchannels(2);w.setsampwidth(2);w.setframerate(16000)
        w.writeframes(b'\0' * 3200)
    return out.getvalue()


class Response(io.BytesIO):
    def __init__(self, body, status=200, **headers):
        super().__init__(body)
        self.status = status
        self.headers = headers


class AudioAcquisitionTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.body = wav_bytes()
        self.job = dict(id='frozen-job',source='ocsc',cohort='test',participant='ocsc/9001',eligible=True,transcript='transcripts/9/9001.cha',transcript_sha256='sourcehash',
            audio_base=str(self.root/'audio/test/9/9001'), media_base=str(self.root/'media/test/9/9001'))
        self.remote = dict(url='https://media.example/9001.wav',extension='.wav',bytes=len(self.body),etag='"version-1"')
        self.target = Path(self.job['audio_base']+'.wav')

    def client(self, response):
        client = Mock();client.open.return_value = response
        return client

    def partial(self, body):
        self.target.parent.mkdir(parents=True)
        Path(str(self.target)+'.part').write_bytes(body)
        a.save(Path(str(self.target)+'.part.json'),dict(job_id=self.job['id'],url=self.remote['url'],etag=self.remote['etag'],bytes=self.remote['bytes']))

    def test_html_login_is_never_saved_as_recording(self):
        c = self.client(Response(b'<html>Sign in</html>',**{'Content-Type':'text/html'}))
        with self.assertRaises(a.AuthenticationRequired):a.http_download(c,self.remote,self.target,self.job)
        self.assertFalse(self.target.exists())
        self.assertFalse(Path(str(self.target)+'.part').exists())

    def test_sharepoint_spaces_are_encoded_without_changing_existing_escapes(self):
        client=a.Client();client.opener=Mock()
        client.open('https://example.test/Shared Documents/already%20escaped%2Fname.zip')
        request=client.opener.open.call_args.args[0]
        self.assertEqual(request.full_url,'https://example.test/Shared%20Documents/already%20escaped%2Fname.zip')

    def test_authenticated_listing_can_include_login_script(self):
        client=a.Client();client.open=Mock(return_value=Response(
            b'<script src="/authModals.js"></script><a href="https://media.talkbank.org:443/childes/Eng-NA/OCSC//4">4/</a>'))
        links=client.directory('https://media.talkbank.org/childes/Eng-NA/OCSC/')
        self.assertEqual(len(links),1)

    def test_login_script_without_authenticated_child_links_is_rejected(self):
        client=a.Client();client.open=Mock(return_value=Response(
            b'<script src="/authModals.js"></script><a href="/login">Login</a>'))
        with self.assertRaises(a.AuthenticationRequired):
            client.directory('https://media.talkbank.org/childes/Eng-NA/OCSC/')

    def test_account_permission_denial_is_explicit(self):
        client=a.Client();client.open=Mock(return_value=Response(
            b'Not authorized. <a href="https://talkbank.org/0share/access.html">member access</a>'))
        with self.assertRaisesRegex(a.AuthenticationRequired,'not authorized'):
            client.directory('https://media.talkbank.org/asd/English/NYU-Emerson/')

    def test_observed_download_link_with_explicit_https_port_precedes_player(self):
        client=Mock()
        player='https://media.talkbank.org:443/childes/Eng-NA/OCSC/4/4001.wav'
        client.directory.return_value=[player,player+'?f=save']
        job=dict(source='ocsc',directory='4',media_id='4001',media_kind='audio',
            transcript='transcripts/4/4001.cha')
        a.resolve_media(client,job,{})
        client.probe.assert_called_once_with(player+'?f=save','.wav')

    def test_parallel_results_keep_all_frozen_cohort_roles_and_stable_order(self):
        jobs=[dict(self.job,id=str(i),media_id=str(i),participant='ocsc/'+str(i),cohort=role)
              for i,role in enumerate(['train','validation','test'])]
        def existing(job):
            return dict(path=str(self.root/(job['id']+'.wav')),sha256='sha-'+job['id'])
        report=self.root/'parallel.json'
        factory=Mock()
        with patch.object(a,'existing_original',side_effect=existing):
            result=a.download_talkbank_parallel(jobs,factory,report,extract=False,workers=3)
        self.assertEqual([r['job_id'] for r in result],['0','1','2'])
        self.assertEqual([r['cohort'] for r in result],['train','validation','test'])
        self.assertEqual(json.loads(report.read_text())['counts'],{'verified':3})
        factory.assert_not_called()

    def test_parallel_rejects_duplicate_jobs_or_non_talkbank_sources(self):
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            a.download_talkbank_parallel([self.job,self.job],Mock(),self.root/'never.json')
        with self.assertRaisesRegex(ValueError,'TalkBank'):
            a.download_talkbank_parallel([dict(self.job,source='lucid')],Mock(),self.root/'never.json')

    def test_completed_download_is_hashed_and_verified_without_network_on_rerun(self):
        c = self.client(Response(self.body,**{'Content-Length':str(len(self.body)),'ETag':'"version-1"'}))
        result=a.http_download(c,self.remote,self.target,self.job)
        self.assertEqual(self.target.read_bytes(),self.body)
        self.assertEqual(result['sha256'],a.sha_file(self.target))
        self.assertEqual((result['participant'],result['cohort']),('ocsc/9001','test'))
        c.open.reset_mock();a.http_download(c,self.remote,self.target,self.job);c.open.assert_not_called()

    def test_resume_requests_exact_offset_and_appends_without_duplication(self):
        offset=80;self.partial(self.body[:offset])
        c=self.client(Response(self.body[offset:],206,**{'Content-Range':f'bytes {offset}-{len(self.body)-1}/{len(self.body)}','ETag':'"version-1"'}))
        a.http_download(c,self.remote,self.target,self.job)
        self.assertEqual(c.open.call_args.args[1]['Range'],f'bytes={offset}-')
        self.assertEqual(self.target.read_bytes(),self.body)

    def test_server_ignoring_range_restarts_matching_partial_cleanly(self):
        self.partial(self.body[:80]);c=self.client(Response(self.body,**{'Content-Length':str(len(self.body))}))
        a.http_download(c,self.remote,self.target,self.job)
        self.assertEqual(self.target.read_bytes(),self.body)

    def test_resume_without_source_version_validator_restarts_instead_of_mixing_versions(self):
        self.remote['etag']=None;self.partial(self.body[:80])
        c=self.client(Response(self.body,**{'Content-Length':str(len(self.body))}))
        a.http_download(c,self.remote,self.target,self.job)
        self.assertNotIn('Range',c.open.call_args.args[1])
        self.assertEqual(self.target.read_bytes(),self.body)

    def test_wrong_resume_range_does_not_mutate_partial(self):
        before=self.body[:80];self.partial(before)
        c=self.client(Response(self.body,206,**{'Content-Range':f'bytes 0-{len(self.body)-1}/{len(self.body)}'}))
        with self.assertRaisesRegex(ValueError,'Content-Range'):a.http_download(c,self.remote,self.target,self.job)
        self.assertEqual(Path(str(self.target)+'.part').read_bytes(),before)
        self.assertFalse(self.target.exists())

    def test_changed_version_and_foreign_partial_are_rejected(self):
        self.partial(self.body[:80]);c=self.client(Response(self.body[80:],206,**{'ETag':'"version-2"'}))
        with self.assertRaisesRegex(ValueError,'version changed'):a.http_download(c,self.remote,self.target,self.job)
        meta=Path(str(self.target)+'.part.json');v=json.loads(meta.read_text());v['job_id']='another-child';a.save(meta,v)
        with self.assertRaisesRegex(ValueError,'different source'):a.http_download(c,self.remote,self.target,self.job)

    def test_truncated_body_stays_partial_and_is_not_marked_complete(self):
        c=self.client(Response(self.body[:-10],**{'Content-Length':str(len(self.body))}))
        with self.assertRaisesRegex(ValueError,'Truncated'):a.http_download(c,self.remote,self.target,self.job)
        self.assertFalse(self.target.exists());self.assertFalse(a.sidecar(self.target).exists())
        self.assertTrue(Path(str(self.target)+'.part').exists())

    def test_existing_unknown_or_changed_media_is_preserved(self):
        self.target.parent.mkdir(parents=True);self.target.write_bytes(b'keep me')
        with self.assertRaises(FileExistsError):a.http_download(Mock(),self.remote,self.target,self.job)
        self.assertEqual(self.target.read_bytes(),b'keep me')

    def test_complete_temporary_body_can_commit_after_interruption_without_network(self):
        self.partial(self.body);client=Mock()
        a.http_download(client,self.remote,self.target,self.job);client.open.assert_not_called()
        self.assertEqual(self.target.read_bytes(),self.body)

    def test_path_traversal_and_symlink_escape_are_rejected(self):
        for name in ['../outside.wav','/outside.wav','C:/outside.wav','sub\\outside.wav']:
            with self.assertRaises(ValueError):a.safe_relative(name)
        (self.root/'escape').symlink_to(self.root.parent,target_is_directory=True)
        with self.assertRaises(ValueError):a.inside(self.root,'escape/out.wav')

    def zip_job(self, z):
        info=z.getinfo('condition/sample.wav')
        return dict(self.job,source='lucid',cohort='external_reference',target=str(self.target),archive={'archive':'sample.zip','etag':'version','url':'https://archive.example/sample.zip','content_range':'bytes 0-0/1'},
            member=dict(name=info.filename,bytes=info.file_size,crc32=info.CRC,compressed_bytes=info.compress_size))

    def test_zip_download_validates_size_crc_and_preserves_wave(self):
        data=io.BytesIO()
        with zipfile.ZipFile(data,'w',zipfile.ZIP_DEFLATED) as z:z.writestr('condition/sample.wav',self.body)
        with zipfile.ZipFile(data) as z:
            job=self.zip_job(z);result=a.zip_download(z,job)
            self.assertEqual(self.target.read_bytes(),self.body)
            self.assertEqual(result['member']['crc32'],job['member']['crc32'])

    def test_corrupt_complete_zip_partial_fails_crc_instead_of_becoming_audio(self):
        data=io.BytesIO()
        with zipfile.ZipFile(data,'w',zipfile.ZIP_DEFLATED) as z:z.writestr('condition/sample.wav',self.body)
        with zipfile.ZipFile(data) as z:
            job=self.zip_job(z);self.target.parent.mkdir(parents=True)
            Path(str(self.target)+'.part').write_bytes(self.body[:-1]+b'X')
            a.save(Path(str(self.target)+'.part.json'),dict(job_id=job['id'],archive=job['archive'],member=job['member']))
            with self.assertRaisesRegex(ValueError,'CRC'):a.zip_download(z,job)
        self.assertFalse(self.target.exists())

    def test_range_reader_refuses_whole_archive_or_wrong_range(self):
        spec=dict(url='https://example/archive.zip',etag='v1',content_range='bytes 0-0/10')
        for response in [Response(b'0123456789'),Response(b'0123456789',206,**{'Content-Range':'bytes 1-9/10'})]:
            with self.assertRaisesRegex(ValueError,'exact byte range'):a.RangeZipReader(self.client(response),spec).read(2)

    def test_real_plan_keeps_original_roles_and_excludes_noneligible_sources(self):
        before=a.sha_file(a.ROOT/'outputs/validation/cohorts_v1/manifest.json')
        plan=a.prepare_plan(output_root=self.root/'downloads');jobs=plan['jobs']
        self.assertEqual(a.Counter(j['source'] for j in jobs),{'ocsc':301,'nyu_emerson':28,'lucid':300})
        self.assertEqual(a.Counter(j['cohort'] for j in jobs if j['source']!='lucid'),{'train':231,'validation':49,'test':49})
        self.assertEqual(before,a.sha_file(a.ROOT/'outputs/validation/cohorts_v1/manifest.json'))
        self.assertFalse((self.root/'downloads').exists())

    def test_optional_excluded_files_never_receive_new_cohort_assignments(self):
        plan=a.prepare_plan(output_root=self.root/'downloads',include_excluded=True)
        excluded=[j for j in plan['jobs'] if not j['eligible']]
        self.assertEqual(len(excluded),4)
        self.assertTrue(all('/excluded/' in j['audio_base'] for j in excluded))
        self.assertTrue(all(j['cohort'] is None for j in excluded if j['source']=='ocsc'))
        self.assertTrue(all(j['cohort']=='train' for j in excluded if j['source']=='nyu_emerson'))

    @unittest.skipUnless(shutil.which('ffmpeg'),'ffmpeg integration requires the optional external tool')
    def test_video_audio_extraction_keeps_original_and_uses_no_resampling(self):
        wav=self.root/'input.wav';wav.write_bytes(self.body);video=Path(self.job['media_base']+'.mp4');video.parent.mkdir(parents=True)
        subprocess.run([shutil.which('ffmpeg'),'-hide_banner','-loglevel','error','-i',str(wav),'-c:a','aac',str(video)],check=True)
        meta=a.metadata(video,self.job,kind='original_media')
        derived=a.extract_audio(meta,self.job)
        self.assertTrue(video.exists())
        with wave.open(derived['path']) as w:self.assertEqual((w.getnchannels(),w.getframerate()),(2,16000))
        with unittest.mock.patch.object(a.shutil,'which',return_value=None):self.assertEqual(a.extract_audio(meta,self.job)['sha256'],derived['sha256'])
        self.assertEqual(a.existing_original(self.job)['path'],str(video))
        self.assertEqual(a.sha_file(video),meta['sha256'])


if __name__=='__main__':unittest.main()
