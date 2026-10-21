import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch,MagicMock
from speech_pipeline import transcript


class WorkflowContextTests(unittest.TestCase):
    def test_turn_selected_model_is_used_for_turns_and_original_paragraph_path_retained(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);workflow=root/'outputs/results/benchmark/transcript_workflow.json';workflow.parent.mkdir(parents=True)
            workflow.write_text(json.dumps(dict(context='turn_ml',margin=.5,model_directory='fresh/ensemble',weights_sha256='manifest-hash')))
            assets=root/'outputs/validation/lucid_validation/upstream_gui';assets.mkdir(parents=True)
            for name in ['real_words.txt','adj_lem_list.txt']:(assets/name).write_text('asset')
            text=root/'paragraph.txt';text.write_text('one two three')
            turns=root/'turns.json';turns.write_text(json.dumps(dict(target_speaker='CHI',rows=[dict(speaker='CHI',text='one two three')])))
            model=MagicMock();model.segment.side_effect=lambda t,boundary_margin=None:[dict(text=t)]
            with patch.object(transcript,'ROOT',root),patch.object(transcript,'resolve_segmenter',return_value=model) as loader,\
                 patch.object(transcript,'segmenter_evidence',return_value={}),patch.object(transcript,'HistoricalTaaled') as lexical,\
                 patch.object(transcript,'analyze',side_effect=lambda *args:dict(chat='CHAT',values={})):
                lexical.return_value.calculate.return_value={}
                a=transcript.run(text,root/'out_plain',nlp=lambda x:x)
                self.assertEqual(loader.call_args.kwargs,{})
                self.assertEqual(a['segmentation_context'],'paragraph_ml')
                self.assertIsNone(a['boundary_margin'])
                b=transcript.run(turns,root/'out_turns',nlp=lambda x:x)
                self.assertEqual(loader.call_args.kwargs,dict(model_directory='fresh/ensemble',expected_weights_sha256='manifest-hash'))
                self.assertEqual(b['segmentation_context'],'turn_ml');self.assertEqual(b['boundary_margin'],.5)
