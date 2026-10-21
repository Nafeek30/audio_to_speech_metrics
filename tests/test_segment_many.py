import unittest
from unittest.mock import patch
import numpy as np
from speech_pipeline.segmentation import Segmenter


class SegmentManyTests(unittest.TestCase):
    def test_batch_preserves_context_ownership_empty_inputs_and_word_offsets(self):
        model=object.__new__(Segmenter)
        model.metadata={'decoding':'argmax'}
        model.model=None;model.tokenizer=None
        model.features=lambda words:[0]*len(words)
        def predict(_,__,records,*args):
            return [np.array([i==0 and len(r['tokens'])>1 for i in range(len(r['tokens']))]) for r in records]
        texts=['  Cats play outside  ','','dogs','we like fish']
        with patch('speech_pipeline.segmentation.pretrained_predictions',predict):
            expected=[model.segment(t) for t in texts]
            batched=model.segment_many(texts)
        self.assertEqual(batched,expected)
        self.assertEqual([[u['text'] for u in us] for us in batched],
                         [['Cats','play outside'],[],['dogs'],['we','like fish']])


if __name__=='__main__':unittest.main()
