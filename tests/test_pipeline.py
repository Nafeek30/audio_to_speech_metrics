import unittest
import spacy
import speech_pipeline.utterances as auto
from speech_pipeline.pipeline import difference


class SegmentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nlp = spacy.load('en_core_web_sm')

    def split(self, text):
        return [s[2] for s in auto.split_turn(text, self.nlp)]

    def test_clause_distinctions(self):
        self.assertEqual(self.split('I went home and I made dinner.'),
                         ['I went home', 'and I made dinner.'])
        self.assertEqual(len(self.split('I went home because I was tired.')), 1)
        self.assertEqual(len(self.split('I went home and made dinner.')), 1)
        self.assertEqual(len(self.split('I went home. I made dinner.')), 2)

    def test_labeling_examples_and_repetitions(self):
        self.assertEqual(self.split('I I I missed that class.'), ['I I I missed that class.'])
        self.assertEqual(self.split('Yeah, I went to class.'), ['Yeah, I went to class.'])
        self.assertEqual(self.split('Um, for a year but sometimes that cheer involved so I have to do cheer.'),
                         ['Um, for a year', 'but sometimes that cheer involved so I have to do cheer.'])
        self.assertEqual(len(self.split('I know that she stayed and he left.')), 1)
        self.assertEqual(len(self.split('I know that she stayed but he left.')), 1)
        self.assertEqual(len(self.split('I like apples and oranges.')), 1)

    def test_fillers_retained_in_total_removed_from_mlu(self):
        result = auto.word_mlu([{'text': 'Hmm.'}, {'text': 'Huh ummm.'}, {'text': 'Yeah, I I went.'}])
        self.assertEqual(result['total_utterances'], 3)
        self.assertEqual(result['mlu_nutterances'], 1)
        self.assertEqual(result['mlu_nwords'], 4)
        self.assertEqual(auto.score_utterance("M and M's.")['counted_words'], ['M', 'and', "M's"])

    def test_explicit_cell_delimiters(self):
        from speech_pipeline.utterances import split_label_cell
        self.assertEqual(split_label_cell('It shrunk. Kind of. Yeah.'), ['It shrunk.', 'Kind of.', 'Yeah.'])
        self.assertEqual(split_label_cell('Dr. Smith stayed. he left.\nHmm.'), ['Dr. Smith stayed.', 'he left.', 'Hmm.'])
        self.assertEqual(split_label_cell('I... I know. What for me?. I am sorry.'),
                         ['I... I know.', 'What for me?.', 'I am sorry.'])

    def test_boundary_comparison_retains_filler_only_responses(self):
        from speech_pipeline.utterances import boundary_counts, boundary_scores
        reference = [{'text':'Hmm.'}, {'text':'I went.'}]
        correct = boundary_scores(boundary_counts(reference, reference))
        merged = boundary_scores(boundary_counts(reference, [{'text':'Hmm, I went.'}]))
        self.assertEqual(correct['boundary_f1'], 1)
        self.assertEqual(merged['boundary_f1'], 0)
        self.assertIsNone(boundary_scores(boundary_counts([], []))['boundary_f1'])

    def test_cross_row_continuation_and_speaker_barrier(self):
        rows = [{'row': i, 'speaker': speaker, 'text': text, 'start': str(i), 'end': str(i+1)}
                for i, (speaker, text) in enumerate([
                    ('S1', 'I went to'), ('S1', 'the store.'),
                    ('S2', 'Where?'), ('S1', 'At home.')], 2)]
        result = auto.predict(rows, 'S1', self.nlp)
        self.assertEqual([u['text'] for u in result], ['I went to the store.', 'At home.'])
        self.assertEqual(result[0]['source_rows'], [2, 3])
        self.assertEqual(result[0]['source_start'], '2')
        self.assertEqual(result[0]['source_end'], '4')
        self.assertNotEqual(result[0]['turn_id'], result[1]['turn_id'])

    def test_text_conservation(self):
        text = 'Dr. Smith went home and he made dinner. Because he was hungry, he ate.'
        parts = list(auto.split_turn(text, self.nlp))
        self.assertEqual(''.join(p[2].replace(' ', '') for p in parts), text.replace(' ', ''))
        for start, end, body, _ in parts:
            self.assertEqual(text[start:end], body)

    def test_mlu_actual_eligible_denominator(self):
        utts = [{'text': t} for t in ['Uh um.', 'I went home.', 'xxx here.', "I'm here."]]
        result = auto.word_mlu(utts)
        self.assertEqual(result['total_utterances'], 4)
        self.assertEqual(result['mlu_nutterances'], 2)
        self.assertEqual(result['mlu_words'], 2.5)
        self.assertEqual(result['mlu_words_sd'], .5)
        self.assertIsNone(result['mlu_morphemes'])

    def test_chat_does_not_invent_morphology_or_child_role(self):
        chat = auto.chat_text([{'text': "I'm here.", 'utterance_id': 1}], 'test')
        self.assertIn("*PAR:\tI'm here .", chat)
        self.assertNotIn('Child', chat)
        self.assertNotIn('%mor:', chat)
        with self.assertRaises(ValueError):
            auto.chat_text([{'text': '[unintelligible]', 'utterance_id': 1}], 'test')

    def test_zero_reference_error(self):
        self.assertIsNone(difference(0, 0)['absolute_percent_error'])
        self.assertEqual(difference(6, 5)['absolute_percent_error'], 20)

    def test_documented_fillers_and_communicators(self):
        s = auto.score_utterance('Uh um mm oh no yeah hi.')
        self.assertEqual(s['counted_words'], ['no', 'yeah', 'hi'])
        self.assertFalse(auto.score_utterance('Oh mm.')['include_mlu'])
        chat = auto.chat_text([{'text': 'Oh, yeah.','utterance_id': 1}], 'test')
        self.assertIn('*PAR:\tyeah .', chat)
        self.assertNotIn('&-oh', chat)

    def test_repetitions_flagged_not_silently_classified(self):
        s = auto.score_utterance('I, I have it.')
        self.assertIn('repetition_or_emphasis_requires_audio', s['review_flags'])
        self.assertEqual(s['word_count'], 4)
        s = auto.score_utterance('I want I want it.')
        self.assertIn('repetition_or_emphasis_requires_audio', s['review_flags'])
        result = auto.word_mlu([{'text':'I know.'}, {'text':'I know.'}])
        self.assertEqual(result['mlu_nutterances'], 2)
        self.assertIsNone(result['clinical_mlu_words'])

    def test_clinical_word_grouping_not_claimed(self):
        s = auto.score_utterance('Iron Man has a choo-choo.')
        self.assertIn('possible_multiword_name_requires_word_grouping_review', s['review_flags'])
        self.assertIn('hyphen_or_fragment_word_grouping_requires_review', s['review_flags'])

    def test_uppercase_unintelligibility_excluded_and_exported(self):
        self.assertFalse(auto.score_utterance('XXX here.')['include_mlu'])
        self.assertNotIn('*PAR:', auto.chat_text([{'text':'XXX here.', 'utterance_id':1}], 'test'))

    def test_cleaned_analysis_is_used_everywhere(self):
        self.assertEqual(auto.analysis_text('Uh, I know.'), 'I know.')
        self.assertEqual(auto.analysis_text('UM.'), '')
        self.assertEqual(auto.analysis_text('Oh, yeah?'), 'yeah?')
        self.assertEqual(auto.analysis_text('Uh-huh.'), 'Uh-huh.')
        import speech_pipeline.metrics as custom
        tokens, _, _ = custom.preprocess_text('Uh, I know.\nOh um.', self.nlp)
        self.assertFalse(any(t.split('_')[0] in auto.FILLERS for t in tokens))

    def test_backchannel_does_not_force_incomplete_speech_apart(self):
        rows = [{'row':i, 'speaker':speaker, 'text':text, 'start':str(i), 'end':str(i+1)}
                for i,(speaker,text) in enumerate([('S1','I went to'),('S2','mhm'),('S1','the store.')],2)]
        result = auto.predict(rows, 'S1', self.nlp)
        self.assertEqual([u['text'] for u in result], ['I went to the store.'])
        self.assertEqual(result[0]['source_rows'], [2,4])
        self.assertTrue(result[0]['crosses_intervening_rows'])
        rows[1]['text'] = 'Where did you go?'
        self.assertEqual(len(list(auto.target_turns(rows,'S1'))),2)



from speech_pipeline.clan import corrected_turns


class CorrectedTurnTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nlp = spacy.load('en_core_web_sm')

    def rows(self, pairs):
        return [dict(row=i,speaker=s,text=t,start='',end='')
                for i,(s,t) in enumerate(pairs,2)]

    def test_lines_are_not_forced_utterances(self):
        rows = self.rows([('PAR','I stayed home because I was sick.')])
        _, turns, _, _ = corrected_turns(rows,'PAR','I stayed home\nbecause I was sick.')
        utterances = [p[2] for t in turns for p in auto.split_turn(t['text'],self.nlp)]
        self.assertEqual(utterances,['I stayed home because I was sick.'])

    def test_uses_original_turns_and_same_clause_splitter(self):
        rows = self.rows([('PAR','I went home and I made dinner.'),
                          ('OTHER','What next?'),('PAR','Yeah, I watched TV.')])
        corrected = 'I went home and I made dinner. Yeah, I watched TV.'
        _, turns, audit, _ = corrected_turns(rows,'PAR',corrected)
        self.assertEqual(len(turns),2)
        self.assertFalse(audit[0]['review_required'])
        expected = [u['text'] for u in auto.predict(rows,'PAR',self.nlp)]
        actual = [p[2] for t in turns for p in auto.split_turn(t['text'],self.nlp)]
        self.assertEqual(actual,expected)

    def test_insertions_and_deletions_are_preserved_and_flagged(self):
        rows = self.rows([('PAR','I went.'),('OTHER','Why?'),('PAR','I slept well.')])
        text, turns, audit, _ = corrected_turns(rows,'PAR','I went home. I rested well.')
        self.assertEqual(' '.join(t['text'] for t in turns),text)
        self.assertTrue(audit[0]['review_required'])

    def test_fillers_do_not_disappear_in_turn_mapping(self):
        rows = self.rows([('PAR','Hmm.'),('OTHER','Ready?'),('PAR','Yeah, I I am.')])
        _,turns,_,_=corrected_turns(rows,'PAR','Hmm.\nYeah, I I am.')
        self.assertEqual([t['text'] for t in turns],['Hmm.','Yeah, I I am.'])



if __name__ == "__main__":
    unittest.main()
