import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import numpy as np

from speech_pipeline.research import transcript_calibration as calibration


class CalibrationTests(unittest.TestCase):
    def test_global_margin_changes_cuts_without_reference_counts_or_word_edits(self):
        with TemporaryDirectory() as temp:
            root=Path(temp)
            (root/'calibration_contexts.json').write_text(json.dumps([
                dict(corpus='a',recording='x',context='paragraph_ml',words=['cats','play','outside'])]))
            np.savez_compressed(root/'calibration_logits.npz',offsets=np.array([0,3]),differences=np.array([-.1,.8,0.]))
            with patch.object(calibration,'EVIDENCE',root):
                low=calibration.decoded('paragraph_ml',0.)['a','x']
                high=calibration.decoded('paragraph_ml',1.)['a','x']
            self.assertEqual([x['text'] for x in low],['cats play','outside'])
            self.assertEqual([x['text'] for x in high],['cats play outside'])

    def test_observed_turns_remain_separate_and_contexts_do_not_mix(self):
        with TemporaryDirectory() as temp:
            root=Path(temp)
            (root/'calibration_contexts.json').write_text(json.dumps([
                dict(corpus='a',recording='x',context='turn_ml',words=['yes']),
                dict(corpus='a',recording='x',context='paragraph_ml',words=['irrelevant']),
                dict(corpus='a',recording='x',context='turn_ml',words=['cats','play'])]))
            np.savez_compressed(root/'calibration_logits.npz',offsets=np.array([0,1,2,4]),differences=np.zeros(4))
            with patch.object(calibration,'EVIDENCE',root):
                r=calibration.decoded('turn_ml',2.)['a','x']
            self.assertEqual([x['text'] for x in r],['yes','cats play'])


if __name__=='__main__':unittest.main()
