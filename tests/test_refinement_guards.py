from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from speech_pipeline.research import participant_improvement_refined as study


class RefinementGuardTests(unittest.TestCase):
    def test_refinement_cannot_register_after_prior_fresh_test(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);test=root/'outputs/results/participant_improvement_v2/test_results.json';test.parent.mkdir(parents=True);test.write_text('{}')
            with patch.object(study,'ROOT',root),patch.object(study,'PLAN',root/'protocol.json'),\
                 patch.object(study.cv,'OUT',root/'parent'),patch.object(study,'load_split') as loader:
                with self.assertRaisesRegex(ValueError,'precede fresh test'):study.register()
                loader.assert_not_called()

    def test_refined_test_cannot_load_records_without_frozen_choice(self):
        with TemporaryDirectory() as temp:
            with patch.object(study,'plan',return_value={}),patch.object(study,'OUT',Path(temp)),patch.object(study,'load_split') as loader:
                with self.assertRaises(FileNotFoundError):study.evaluate()
                loader.assert_not_called()
