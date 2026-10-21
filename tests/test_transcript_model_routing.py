from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from speech_pipeline import pipeline


class TranscriptModelRoutingTests(unittest.TestCase):
    def test_explicit_frozen_model_is_loaded_with_verified_weight_hash(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);folder=root/'fresh/model';folder.mkdir(parents=True)
            (folder/'segmenter.json').write_text('{}');(folder/'model.safetensors').write_bytes(b'fresh-weights')
            expected=pipeline.digest(folder/'model.safetensors');nlp=object()
            with patch.object(pipeline,'ROOT',root),patch('speech_pipeline.segmentation.Segmenter') as loader:
                pipeline.resolve_segmenter('improved',nlp,model_directory='fresh/model',expected_weights_sha256=expected)
                loader.assert_called_once_with(folder.resolve(),nlp)

    def test_missing_or_altered_selected_model_cannot_fall_back_to_old_weights(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);folder=root/'fresh';folder.mkdir();(folder/'segmenter.json').write_text('{}')
            (folder/'model.safetensors').write_bytes(b'changed')
            with patch.object(pipeline,'ROOT',root),patch('speech_pipeline.segmentation.Segmenter') as loader:
                with self.assertRaisesRegex(ValueError,'mismatch'):
                    pipeline.resolve_segmenter('improved',None,'fresh','old-hash')
                with self.assertRaises(FileNotFoundError):
                    pipeline.resolve_segmenter('improved',None,'missing','old-hash')
                with self.assertRaisesRegex(ValueError,'inside the project'):
                    pipeline.resolve_segmenter('improved',None,'../outside','old-hash')
                loader.assert_not_called()

    def test_rules_do_not_load_a_model_or_require_a_weight_hash(self):
        self.assertIsNone(pipeline.resolve_segmenter('rules',None,'missing',None))

    def test_ensemble_requires_manifest_hash_and_does_not_fall_back(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);folder=root/'ensemble';folder.mkdir();(folder/'ensemble.json').write_text('{}')
            expected=pipeline.digest(folder/'ensemble.json')
            with patch.object(pipeline,'ROOT',root),patch('speech_pipeline.ensemble.EnsembleSegmenter') as loader:
                pipeline.resolve_segmenter('improved',None,'ensemble',expected)
                loader.assert_called_once_with(folder.resolve(),None)
                loader.reset_mock()
                with self.assertRaisesRegex(ValueError,'manifest mismatch'):
                    pipeline.resolve_segmenter('improved',None,'ensemble','wrong')
                loader.assert_not_called()


if __name__=='__main__':unittest.main()
