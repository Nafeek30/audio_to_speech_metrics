import unittest

from speech_pipeline.moss import faithful_chat,parse_transcript,target_rows
from speech_pipeline import utterances as auto


class MossImportTests(unittest.TestCase):
    def test_canonical_overlap_and_explicit_target(self):
        rows=parse_transcript('[0.0][S01]um I I went[2.0][1.8][S02]where[2.3][2.4][S01]to school[3.5]')
        self.assertEqual(len(rows),3)
        self.assertEqual([r['text'] for r in target_rows(rows,'S01')],['um I I went','to school'])
        self.assertEqual(rows[1]['start'],1.8) # Overlap is valid, not discarded.
        with self.assertRaises(ValueError):target_rows(rows,'S03')

    def test_incomplete_or_partially_parsed_output_is_rejected(self):
        for text in ('','prefix [0][S01]hi[1]','[0][S01]hi[1] trailing',
                     '[0][S01]hi[1][2][S02]incomplete','[2][S01]hi[1]', '[1][S01]hi[1]'):
            with self.subTest(text=text),self.assertRaises(ValueError):parse_transcript(text)

    def test_acoustic_labels_require_review(self):
        with self.assertRaises(ValueError):target_rows(parse_transcript('[0][S01]hello [laughter][1]'),'S01')

    def test_faithful_chat_preserves_fillers_and_repetitions_without_fake_timing(self):
        us=[dict(text='um I I went'),dict(text='xxx school')]
        text=faithful_chat(us,'sample','S01','Reviewed target voice')
        self.assertIn('*PAR:\tum I I went .',text)
        self.assertIn('*PAR:\txxx school .',text)
        self.assertNotIn('\x15',text)
        self.assertEqual(auto.stream(us)[0],['um','i','i','went','xxx','school'])


class SpeakerEvaluationTests(unittest.TestCase):
    def test_anonymous_speaker_permutation_is_evaluation_only(self):
        from speech_pipeline.research.moss_pilot import oracle_speaker_mapping
        r=oracle_speaker_mapping({'CHI':['red','ball'],'EXP':['what','color']},
                                 {'S01':['what','color'],'S02':['red','ball']})
        mapping={a['reference_speaker']:a['predicted_speaker'] for a in r['assignments']}
        self.assertEqual(mapping,{'CHI':'S02','EXP':'S01'})
        self.assertEqual(r['cpwer_percent'],0)

    def test_missing_and_extra_speakers_are_not_dropped(self):
        from speech_pipeline.research.moss_pilot import oracle_speaker_mapping
        r=oracle_speaker_mapping({'CHI':['red','ball'],'EXP':['what','color']},{'S01':['what','color']})
        self.assertEqual(r['word_edits'],2)
        self.assertEqual(r['reference_words'],4)
        self.assertEqual(r['cpwer_percent'],50)
        r=oracle_speaker_mapping({'CHI':['red']},{'S01':['red'],'S02':['hello']})
        self.assertEqual(r['word_edits'],1)
        self.assertTrue(any(a['reference_speaker'] is None for a in r['assignments']))


if __name__=='__main__':unittest.main()
