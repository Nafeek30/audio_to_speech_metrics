import unittest
from speech_pipeline.research.lucid_validation import parse_chronological,cleaned_labels,file_metadata


class LucidSourceTests(unittest.TestCase):
    def test_empty_unused_tier_is_valid_and_words_stay_on_correct_tier(self):
        text='"Praat chronological TextGrid text file"\n0 2 ! Time\n2 ! Tiers\n"IntervalTier" "Speaker A" 0 2\n"IntervalTier" "" 0 2\n\n! A\n1 0 1\n"hello"\n2 0 1\n"other"\n1 1 2\n"a ""quote"""\n'
        tiers,rows=parse_chronological(text)
        self.assertEqual(tiers,['Speaker A',''])
        self.assertEqual([r['label'] for r in rows if r['tier']=='Speaker A'],['hello','a "quote"'])

    def test_noise_repair_preserves_spoken_payload_and_fragments(self):
        text,removed,payloads=cleaned_labels(['!SIL','<BR>hello','<LG_yes>','then>','s-','<GA<','<L>S'])
        self.assertEqual(text,'hello yes then s-')
        self.assertEqual(payloads['LG'],1)
        self.assertEqual(removed['BR'],1)
        with self.assertRaises(ValueError):cleaned_labels(['[unresolved]'])

    def test_unparsed_source_and_invalid_timing_are_rejected(self):
        base='"Praat chronological TextGrid text file"\n0 2 ! Time\n1 ! Tiers\n"IntervalTier" "Speaker A" 0 2\n'
        with self.assertRaises(ValueError):parse_chronological(base+'1 1 0\n"word"\n')
        with self.assertRaises(ValueError):parse_chronological(base+'unparsed text\n1 0 1\n"word"\n')

    def test_documented_filename_maps_identity_and_condition(self):
        m=file_metadata('LCD_F13F14S2clv2.TextGrid')
        self.assertEqual((m['speaker'],m['partner'],m['condition'],m['scene']),('F13','F14','vocoder','shop'))
        m=file_metadata('LCD_M14BEM3S1cln1.TextGrid')
        self.assertEqual((m['speaker'],m['partner'],m['condition']),('M14','BEM3','babble'))


if __name__=='__main__':unittest.main()
