"""Guard against optimistic scoring and mismatched clinical denominators."""
import unittest

from speech_pipeline.research.common_benchmark import (
    METHODS, METRICS, boundaries, errors, paired_intervals, summarize, units, values,
)


class CommonBenchmarkTests(unittest.TestCase):
    def test_no_partial_alignment_score_when_a_word_changes(self):
        r=dict(tokens=['we','like','cats'],labels=[0,1,-100],gold_boundaries=[2])
        score=boundaries(r,[dict(text='we like dogs')])
        self.assertEqual(score['status'],'unavailable_word_stream_changed')
        self.assertEqual(score['eligible_gaps'],2)
        self.assertEqual(score['scored_gaps'],0)
        self.assertNotIn('counts',score)

    def test_masked_gap_does_not_change_boundary_counts(self):
        r=dict(tokens=['we','xxx','cats','play'],labels=[-100,-100,1,-100],gold_boundaries=[2,3])
        score=boundaries(r,units(r['tokens'],[1,3]))
        self.assertEqual(score['counts'],dict(tp=1,fp=0,fn=0))
        self.assertEqual(score['scored_gaps'],1)
        self.assertEqual(score['exact']['matched'],1)

    def test_zero_reference_is_unavailable_and_short_mattr_missing(self):
        v=values([dict(text='cats play')])
        self.assertEqual(v['length_sd'],0)
        self.assertIsNone(v['MATTR'])
        e=errors(v,v)
        self.assertIsNone(e['length_sd']['ape'])
        self.assertIsNone(e['MATTR']['ape'])
        self.assertEqual(e['MLUw']['ape'],0)

    def test_corpus_and_child_balance_and_paired_sign(self):
        rows=[]
        for corpus,children in [('a',[('a1',1),('a2',3)]),('b',[('b1',5)])]:
            for participant,n in children:
                for recording in range(n):
                    for method in METHODS:
                        ape=(10 if participant=='a1' else 30 if corpus=='a' else 100)
                        ape += 2 if method==METHODS[-1] else 0
                        rows.append(dict(corpus=corpus,participant=participant,recording=str(recording),method=method,
                            metrics={k:dict(ape=ape) for k in METRICS},
                            boundaries=dict(status='complete_shared_stream',counts=dict(tp=1,fp=0,fn=0),
                                eligible_gaps=1,scored_gaps=1,exact=dict(reference=2,predicted=2,matched=2))))
        summary=summarize(rows)
        self.assertEqual(summary['rules']['a']['primary_objective'],20)
        self.assertEqual(summary['rules']['corpus_balanced_primary'],60)
        interval=paired_intervals(rows,draws=100)['rules']
        self.assertEqual(interval['delta_ape_points_vs_batchalign'],-2)
        self.assertEqual(interval['percentile_95_interval'],[-2,-2])

    def test_invalid_boundary_units_rejected(self):
        for cuts in ([0],[3],[2,1],[1,1]):
            with self.assertRaises(ValueError):units(['a','b','c'],cuts)


if __name__=='__main__':unittest.main()
