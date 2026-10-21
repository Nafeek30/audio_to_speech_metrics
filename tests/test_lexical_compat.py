from types import SimpleNamespace as Token
import unittest
from speech_pipeline.lexical_compat import normalize_doc, calculate_normalized, clean_pauses


class LegacyLexicalTests(unittest.TestCase):
    def test_dictionary_filter_and_historical_word_classes(self):
        def t(text, lemma, tag, dep=''):return Token(text=text, lemma_=lemma, tag_=tag, dep_=dep)
        doc=[t('Her','her','PRP$'),t('dogs','dog','NNS'),t('are','be','VBP'),
             t('running','run','VBG'),t('quickly','quickly','RB'),t('erm','erm','UH'),
             t('unknownword','unknownword','NN'),t('.','.','.')]
        result=normalize_doc(doc,{'her','dogs','are','running','quickly'}, {'quick'})
        self.assertEqual(result['lemma'],['she_fw','dog_cw_nn','be_fw','run_cw_vb','quickly_cw'])
        values=calculate_normalized(result)
        self.assertEqual(values['basic_ntokens'],5)
        self.assertEqual(values['basic_ncontent_tokens'],3)
        self.assertIsNone(values['mattr50_aw'])

    def test_auxiliary_and_noun_verb_homographs_are_distinct(self):
        doc=[Token(text='can',lemma_='can',tag_='MD',dep_='aux'),
             Token(text='fish',lemma_='fish',tag_='NN',dep_=''),
             Token(text='fish',lemma_='fish',tag_='VB',dep_='')]
        result=normalize_doc(doc,{'can','fish'},set())
        self.assertEqual(result['lemma'],['can_fw','fish_cw_nn','fish_cw_vb'])
        self.assertEqual(calculate_normalized(result)['basic_ntypes'],3)

    def test_pause_removal_does_not_erase_partial_words(self):
        text,removed=clean_pauses('Er, the eraser is here; erm yeah.')
        self.assertIn('eraser',text)
        self.assertEqual(removed,{'er':1,'erm':1})


if __name__=='__main__':unittest.main()
