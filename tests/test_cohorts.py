import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from speech_pipeline import cohorts
from speech_pipeline.pipeline import digest


GROUPS = dict(train=['ocsc/1'], validation=['ocsc/2'], test=['nyu_emerson/3'])


def record(participant='ocsc/1', recording='transcripts/1.cha#section1', sha='source-one'):
    return dict(corpus=participant.split('/')[0], participant=participant,
        recording=recording, source_sha256=sha, tokens=['cats','play','outside'], labels=[0,1,-100])


class CohortTests(unittest.TestCase):
    def test_participant_cannot_belong_to_multiple_roles_or_repeat_within_role(self):
        for groups in [dict(GROUPS, test=['ocsc/1']), dict(GROUPS, train=['ocsc/1','ocsc/1'])]:
            with self.assertRaisesRegex(ValueError,'participant'):
                cohorts.partition_lookup(groups)

    def test_eval_or_test_records_are_rejected_from_training(self):
        for participant in ['ocsc/2','nyu_emerson/3','ocsc/unknown']:
            with self.assertRaisesRegex(ValueError,'forbidden'):
                cohorts.validate_records([record(participant)], GROUPS, ('train',))

    def test_recordings_and_source_files_cannot_be_relabelled_as_other_children(self):
        first=record()
        second=record('ocsc/2',recording='transcripts/1.cha#section2',sha='different')
        with self.assertRaisesRegex(ValueError,'source recording'):
            cohorts.validate_records([first,second],GROUPS)
        second=record('ocsc/2',recording='transcripts/2.cha#section1')
        with self.assertRaisesRegex(ValueError,'Identical source'):
            cohorts.validate_records([first,second],GROUPS)

    def test_foreign_crops_and_changed_training_labels_are_rejected(self):
        parent=record();crop=dict(parent,recording=parent['recording']+'#crop0',
            tokens=['play','outside'],labels=[1,-100])
        cohorts.validate_training_views([parent,crop],[parent])
        for bad in [dict(crop,participant='ocsc/2'),dict(crop,tokens=['dogs','outside']),
                    dict(crop,labels=[0,-100]),dict(crop,labels=[1,1])]:
            with self.assertRaises(ValueError):
                cohorts.validate_training_views([parent,bad],[parent])

    def make_bundle(self,root):
        path=root/'train/ocsc.jsonl';path.parent.mkdir()
        path.write_text(json.dumps(record())+'\n')
        manifest=dict(assignments=GROUPS,files={
            'train':{'ocsc':dict(path='train/ocsc.jsonl',sha256=digest(path),records=1)},
            'validation':{},'test':{'nyu_emerson':dict(path='test/never-open.jsonl',sha256='unused',records=1)}},
            record_order=[['ocsc',record()['recording']]])
        self.seal(root,manifest)
        return manifest

    def seal(self,root,manifest):
        (root/'manifest.json').write_text(json.dumps(manifest))
        (root/'manifest.sha256').write_text(digest(root/'manifest.json')+'\n')

    def test_train_load_does_not_open_or_require_test_files(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);self.make_bundle(root)
            self.assertEqual(cohorts.load_split('train',root),[record()])
            with self.assertRaises(FileNotFoundError):cohorts.load_split('test',root)
            with self.assertRaises(ValueError):cohorts.load_split('all',root)

    def test_modified_files_manifests_or_roles_are_rejected(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);manifest=self.make_bundle(root)
            (root/'train/ocsc.jsonl').write_text(json.dumps(record('ocsc/2'))+'\n')
            with self.assertRaisesRegex(ValueError,'file changed'):cohorts.load_split('train',root)
            manifest['files']['train']['ocsc']['sha256']=digest(root/'train/ocsc.jsonl')
            self.seal(root,manifest)
            with self.assertRaisesRegex(ValueError,'forbidden'):cohorts.load_split('train',root)
            (root/'manifest.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'manifest changed'):cohorts.load_split('train',root)

    def test_loader_rejects_a_file_pointing_into_another_role_directory(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);manifest=self.make_bundle(root)
            manifest['files']['train']['ocsc']['path']='test/ocsc.jsonl';self.seal(root,manifest)
            with self.assertRaisesRegex(ValueError,'outside'):cohorts.load_split('train',root)

    def test_transcript_selection_requests_only_validation(self):
        from speech_pipeline.research import transcript_chat
        with TemporaryDirectory() as temp:
            root=Path(temp);(root/'metric_training.json').write_text(json.dumps(dict(partitions=GROUPS)))
            with patch.object(cohorts,'BUNDLE',root),patch.object(transcript_chat,'BASE',root),\
                 patch.object(cohorts,'read_manifest',return_value={'record_order':[]}),\
                 patch.object(cohorts,'load_split',return_value=[]) as loader:
                self.assertEqual(transcript_chat.read_records('validation'),[])
                self.assertEqual([call.args[0] for call in loader.call_args_list],['validation'])
        with self.assertRaises(TypeError):transcript_chat.read_records()

    def test_calibration_selection_uses_only_validation_reference_file(self):
        from speech_pipeline.research import transcript_calibration as calibration
        with TemporaryDirectory() as temp:
            root=Path(temp);base=root/'outputs/validation/benchmark';base.mkdir(parents=True)
            (base/'metric_training.json').write_text(json.dumps(dict(partitions=GROUPS)))
            output=root/'results';output.mkdir()
            row=dict(corpus='ocsc',participant='ocsc/2',recording='transcripts/2.cha#section1',
                split='validation',reference=dict(MLUw=2))
            (output/'chat_validation.json').write_text(json.dumps(dict(records=[row])))
            with patch.object(calibration,'ROOT',root),patch.object(calibration,'OUT',output),\
                 patch.object(calibration,'EVIDENCE',root/'missing_test_references'):
                self.assertEqual(calibration.reference_map('validation'),
                    {'ocsc/transcripts/2.cha#section1':dict(values=dict(MLUw=2))})
                row['participant']='ocsc/1'
                (output/'chat_validation.json').write_text(json.dumps(dict(records=[row])))
                with self.assertRaisesRegex(ValueError,'Non-validation'):
                    calibration.reference_map('validation')


if __name__ == '__main__':unittest.main()
