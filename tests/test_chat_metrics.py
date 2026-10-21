import unittest
from speech_pipeline.chat_metrics import analyze,plain_chat


class NativeTranscriptTests(unittest.TestCase):
    def test_native_policy_retains_communicators_and_clear_words_near_unknown(self):
        us=[dict(text='uh oh cats play'),dict(text='xxx dogs run'),dict(text='um')]
        r=analyze(us)['values']
        self.assertEqual(r['total_utterances'],3)
        self.assertEqual(r['eligible_mlu_utterances'],1)
        self.assertEqual(r['MLUw'],3)
        self.assertEqual(r['TNW'],5)
        self.assertEqual(r['NDW'],5)
        self.assertIsNone(r['MATTR'])

    def test_export_keeps_repetition_and_cannot_silently_strip_chat_scope(self):
        body=plain_chat([dict(text='um I I like it')])
        self.assertIn('&-um I I like it .',body)
        self.assertNotIn('%mor',body)
        with self.assertRaises(ValueError):plain_chat([dict(text='<I I> [/] like it')])


if __name__=='__main__':unittest.main()
