import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from speech_pipeline.research import cv_hyperparameter as search


class HyperparameterCVTests(unittest.TestCase):
 def test_final_data_not_loaded_until_cross_validated_recipe_is_frozen(self):
  with TemporaryDirectory() as temp:
   with patch.object(search,'OUT',Path(temp)),patch.object(search,'plan',return_value={}),patch.object(search,'load_split') as loader:
    with self.assertRaises(FileNotFoundError):search.job_rows('lr2_count1','final','fit')
    loader.assert_not_called()

 def test_final_must_use_cv_winner_and_test_needs_final_freeze(self):
  with TemporaryDirectory() as temp:
   root=Path(temp);(root/'cv_selection.json').write_text(json.dumps(dict(configuration='lr3_count025')))
   with patch.object(search,'OUT',root),patch.object(search,'plan',return_value={}),patch.object(search,'load_split',return_value=[]) as loader:
    with self.assertRaisesRegex(ValueError,'CV winner'):search.job_rows('lr2_count1','final','fit')
    loader.assert_not_called()
    search.job_rows('lr3_count025','final','fit');search.job_rows('lr3_count025','final','inner')
    self.assertEqual([x.args[0] for x in loader.call_args_list],['train','validation']);loader.reset_mock()
    with self.assertRaisesRegex(ValueError,'Freeze'):search.job_rows('lr3_count025','final','outer')
    loader.assert_not_called()

 def test_recipe_ranking_cannot_skip_incomplete_folds_or_open_eval_test(self):
  with TemporaryDirectory() as temp:
   root=Path(temp)
   with patch.object(search,'OUT',root),patch.object(search,'plan',return_value=dict(recipes=search.RECIPES)),\
        patch.object(search,'candidate_summary',side_effect=FileNotFoundError('missing fifth fold')) as summarize,\
        patch.object(search,'load_split') as loader:
    with self.assertRaises(FileNotFoundError):search.select_recipe()
    self.assertFalse((root/'cv_selection.json').exists());loader.assert_not_called()

 def test_every_recipe_is_compared_and_minimum_cv_score_selected(self):
  with TemporaryDirectory() as temp:
   root=Path(temp);protocol=root/'protocol.json';protocol.write_text('{}')
   def result(config):
    v={'lr2_count1':10.,'lr2_count025':9.,'lr3_count1':11.,'lr3_count025':8.}[config]
    return dict(configuration=config,recipe=search.RECIPES[config],objective=v,fold_mean=v,fold_sd=1.,records=[],summary=[])
   with patch.object(search,'OUT',root),patch.object(search,'PLAN',protocol),patch.object(search,'plan',return_value=dict(recipes=search.RECIPES,selection='minimum CV')),\
        patch.object(search,'candidate_summary',side_effect=result) as summarize,patch.object(search,'load_split') as loader:
    search.select_recipe();selected=json.loads((root/'cv_selection.json').read_text())
    self.assertEqual(selected['configuration'],'lr3_count025');self.assertEqual(summarize.call_count,4);loader.assert_not_called()
