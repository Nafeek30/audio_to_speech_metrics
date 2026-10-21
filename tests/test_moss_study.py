from copy import deepcopy
import unittest

from speech_pipeline.research.moss_study import check_folds, label_flags


class MossStudyTests(unittest.TestCase):
    def setUp(self):
        self.pool = ['ocsc/'+str(i) for i in range(5)]
        self.folds = [dict(fold=i+1, outer=[p], inner=[self.pool[(i+1)%5]],
            fit=[q for q in self.pool if q not in (p, self.pool[(i+1)%5])])
            for i, p in enumerate(self.pool)]

    def test_existing_grouped_folds_cover_train_once(self):
        check_folds(self.folds, self.pool)

    def test_validation_or_test_participant_cannot_enter_cv(self):
        folds = deepcopy(self.folds)
        folds[0]['fit'][0] = 'ocsc/validation-child'
        with self.assertRaisesRegex(ValueError, 'original training'):
            check_folds(folds, self.pool)

    def test_child_cannot_enter_multiple_roles(self):
        folds = deepcopy(self.folds)
        folds[0]['fit'].append(folds[0]['outer'][0])
        with self.assertRaisesRegex(ValueError, 'leakage'):
            check_folds(folds, self.pool)

    def test_outer_must_hold_each_child_once(self):
        folds = deepcopy(self.folds)
        folds[-1] = deepcopy(folds[0])
        folds[-1]['fold'] = 5
        with self.assertRaisesRegex(ValueError, 'held out once'):
            check_folds(folds, self.pool)

    def test_unknown_speech_and_names_are_explicit(self):
        row = dict(text='Childname says xxx', time_marks=[1, 2])
        self.assertEqual(label_flags(row), ['unintelligible_or_untranscribed_speech', 'anonymized_spoken_name'])

    def test_missing_negative_zero_or_reversed_timing_is_flagged(self):
        for times in [None, [], [1], [-1, 2], [2, 2], [3, 2]]:
            self.assertIn('missing_or_invalid_timing', label_flags(dict(text='hello', time_marks=times)))


if __name__ == '__main__':
    unittest.main()
