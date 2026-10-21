import unittest
from unittest.mock import patch
import numpy as np
from speech_pipeline.ensemble import mean_probability_logits, EnsembleSegmenter


class ProbabilityEnsembleTests(unittest.TestCase):
    def test_equal_mean_probability_not_mean_logit(self):
        a=np.array([4.,-3.]);b=np.array([0.,1.])
        logits=mean_probability_logits([a,b])
        actual=1/(1+np.exp(-logits))
        expected=((1/(1+np.exp(-a)))+(1/(1+np.exp(-b))))/2
        np.testing.assert_allclose(actual,expected,rtol=1e-14)
        self.assertFalse(np.allclose(logits,(a+b)/2))

    def test_singleton_preserves_exact_decisions_and_extremes_finite(self):
        a=np.array([-1000.,0.,.25,1000.],dtype=np.float32)
        np.testing.assert_array_equal(mean_probability_logits([a]),a)
        self.assertTrue(np.isfinite(mean_probability_logits([a,a])).all())
        with self.assertRaises(ValueError):mean_probability_logits([])
        with self.assertRaises(ValueError):mean_probability_logits([a,np.zeros(2)])
        with self.assertRaises(ValueError):mean_probability_logits([np.array([np.nan])])

    def test_runtime_margin_preserves_words_and_source_spans(self):
        obj=EnsembleSegmenter.__new__(EnsembleSegmenter);obj.metadata={'margin':.25}
        obj.features=lambda tokens:[0]*len(tokens)
        obj._logits=lambda records:[np.array([.25,.26,999.]) for _ in records]
        texts=['One two three','',"One   two three"]
        actual=obj.segment_many(texts)
        self.assertEqual([[u['text'] for u in rows] for rows in actual],[['One two','three'],[],['One   two','three']])
        for text,rows in zip(texts,actual):
            for u in rows:self.assertEqual(text[u['source_start']:u['source_end']],u['text'])

    def test_mismatched_component_words_refused(self):
        from speech_pipeline.research.participant_improvement import combine
        values={'a':{'k':[(['word'],np.array([0.]))]},'b':{'k':[(['other'],np.array([0.]))]}}
        with self.assertRaises(ValueError):combine(['a','b'],values)
