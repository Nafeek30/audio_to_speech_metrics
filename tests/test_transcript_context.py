import unittest
from speech_pipeline.research.transcript_chat import predict


class FakeModel:
    def __init__(self):self.inputs=[]
    def segment(self,text):
        self.inputs.append(text)
        return [dict(text=text)] if text else []


class TranscriptContextTests(unittest.TestCase):
    def test_turn_input_merges_target_rows_and_uses_only_speaker_changes(self):
        model=FakeModel()
        rows=[dict(speaker='CHI',text='I like'),dict(speaker='CHI',text='cats'),
              dict(speaker='EXP',text='which cats'),dict(speaker='CHI',text='those')]
        result=predict('turn_ml',rows,model,None)
        self.assertEqual(model.inputs,['I like cats','those'])
        self.assertEqual([u['text'] for u in result],model.inputs)

    def test_paragraph_input_discards_source_rows_and_adult_words(self):
        model=FakeModel()
        rows=[dict(speaker='CHI',text='cats'),dict(speaker='EXP',text='dogs'),
              dict(speaker='CHI',text='play')]
        predict('paragraph_ml',rows,model,None)
        self.assertEqual(model.inputs,['cats play'])
        with self.assertRaises(ValueError):predict('unknown',rows,model,None)


if __name__=='__main__':unittest.main()
