"""Scoring-evidence, CHAT leakage, archive-safety, and official reference checks."""
import os
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import zipfile

from speech_pipeline import ROOT
from speech_pipeline.metrics import clinical_measures, clinical_from_predictions
from speech_pipeline.corpora import install_transcripts, read_chat, run_clan, prepare_training


class ClinicalEvidenceTests(unittest.TestCase):
    def test_workbook_auto_keeps_established_rules(self):
        from speech_pipeline.pipeline import run_pipeline
        nlp=object()
        with TemporaryDirectory() as temp,patch('speech_pipeline.pipeline.official.load_target_speakers',return_value={}):
            for setting,expected in [('auto','rules'),('improved','improved')]:
                with patch('speech_pipeline.pipeline.resolve_segmenter',return_value=None) as resolve:
                    run_pipeline(nlp=nlp,output_root=Path(temp)/setting,segmenter=setting)
                    resolve.assert_called_once_with(expected,nlp)

    def test_reviewed_morphology_and_annotations(self):
        profile=clinical_measures(dict(reviewed=True,morphology_scheme='legacy_mor',maze_scope='eligible_utterances',timing_reviewed=True,duration_seconds=30,utterances=[
            dict(words=['The','dogs','played'],mor=['det|the','n|dog-PL','v|play-PAST'],grammatical=True,
                 omission_count=0,substitution_count=0,maze_words=2),
            dict(words=['the','cat','slept'],mor=['det|the','n|cat','v|sleep&PAST'],grammatical=False,
                 omission_count=1,substitution_count=1,maze_words=0)]),window=3)
        values=profile['values']
        self.assertEqual(values['MLUw'],3);self.assertEqual(values['MLUm'],4)
        self.assertEqual(values['TNW'],6);self.assertEqual(values['NDW'],5)
        self.assertAlmostEqual(values['TTR'],5/6);self.assertEqual(values['MATTR'],1)
        self.assertEqual(values['WPM'],12);self.assertEqual(values['PGU'],50)
        self.assertEqual(values['omission_count'],1);self.assertEqual(values['substitution_count'],1)
        self.assertEqual(values['maze_percentage'],25)

    def test_unreviewed_labels_do_not_become_clinical_evidence(self):
        profile=clinical_measures(dict(reviewed=False,timing_reviewed=True,duration_seconds=30,
            morphology_scheme='legacy_mor',utterances=[dict(words=['dogs'],mor=['n|dog-PL'],grammatical=True,
                omission_count=0,substitution_count=0,maze_words=0)]))
        self.assertEqual(profile['values']['TNW'],1)
        for metric in ('MLUm','WPM','PGU','omission_count','substitution_count','maze_percentage'):
            self.assertIsNone(profile['values'][metric],metric)

    def test_missing_annotation_is_not_a_zero(self):
        values=clinical_measures(dict(reviewed=True,utterances=[dict(words=['hello'])]))['values']
        for k in ('MLUm','WPM','PGU','omission_count','substitution_count','maze_percentage'):
            self.assertIsNone(values[k])
        self.assertIsNone(values['MATTR'])

    @unittest.skipUnless(os.name=='posix' and (ROOT/'.venv/bin/clan-mlu').exists(),'official CLAN CLI not installed')
    def test_nonzero_population_length_sd_against_official_clan(self):
        from speech_pipeline.pipeline import clan_software_check
        units=[dict(utterance_id=1,text='we like the dogs'),dict(utterance_id=2,text='we like the dogs and cats')]
        with TemporaryDirectory() as temp:check=clan_software_check(units,Path(temp))
        value=check['metrics']['mlu_words_sd']
        self.assertEqual(value['custom'],1)
        self.assertEqual(value['reference'],1)
        self.assertEqual(value['absolute_error'],0)

    def test_ud_features_are_not_counted_as_morphemes(self):
        result=clinical_measures(dict(reviewed=True,morphology_scheme='ud',utterances=[
            dict(words=['dogs'],mor=['noun|dog-Plur-Nom'])]))
        self.assertIsNone(result['values']['MLUm'])

    def test_exclusion_denominator_and_zero_reference(self):
        result=clinical_measures(dict(reviewed=True,utterances=[
            dict(words=['xxx'],grammatical=False),dict(words=['hello'],grammatical=True),
            dict(words=['bad'],exclude=True,reason='explicit exclusion',grammatical=False)]))
        self.assertEqual(result['values']['PGU'],100)
        self.assertEqual(result['values']['MLUw'],1)
        self.assertIsNone(clinical_measures(dict(reviewed=True,utterances=[]))['values']['TTR'])

    def test_duration_and_word_annotation_validation(self):
        with self.assertRaises(ValueError):clinical_measures(dict(reviewed=True,duration_seconds=0,utterances=[]))
        with self.assertRaises(ValueError):clinical_measures(dict(reviewed=True,utterances=[dict(words=['hello world'])]))
        with self.assertRaises(ValueError):clinical_measures(dict(reviewed=True,utterances=[dict(words=['word'],exclude=True)]))

    def test_requested_profile_preserves_old_filler_policy(self):
        profile=clinical_from_predictions([dict(text='Hmm.'),dict(text='I I went.'),dict(text='xxx.')])
        self.assertEqual(profile['values']['TNW'],3)
        self.assertEqual(profile['values']['MLUw'],3)
        self.assertIsNone(profile['values']['PGU'])

    def test_whole_sample_maze_denominator_includes_excluded_utterances(self):
        profile=clinical_measures(dict(reviewed=True,maze_scope='whole_sample',utterances=[
            dict(words=['hello'],maze_words=1,produced_words=2),
            dict(words=[],exclude=True,reason='unintelligible',maze_words=2,produced_words=3)]))
        self.assertEqual(profile['values']['maze_percentage'],60)
        self.assertEqual(profile['values']['TNW'],1)
        ambiguous=clinical_measures(dict(reviewed=True,utterances=[dict(words=['hello'],maze_words=1)]))
        self.assertIsNone(ambiguous['values']['maze_percentage'])

    @unittest.skipUnless(os.name=='posix' and (ROOT/'.venv/bin/clan-mlu').exists(),'official CLAN CLI not installed')
    def test_legacy_morphology_against_official_clan(self):
        with TemporaryDirectory() as temp:
            path=Path(temp)/'sample.cha';path.write_text('@UTF8\n@Begin\n@Languages:\teng\n'
                '@Participants:\tCHI Target_Child\n@ID:\teng|test|CHI|||||Target_Child|||\n'
                '*CHI:\tthe dogs played .\n%mor:\tdet|the n|dog-PL v|play-PAST .\n'
                '*CHI:\tthe cat slept .\n%mor:\tdet|the n|cat v|sleep&PAST .\n@End\n')
            value=run_clan('mlu',path,'CHI',morphology=True)
            self.assertEqual(value['mlu'],4)
            self.assertEqual(value['total'],8)


class CorpusSafetyTests(unittest.TestCase):
    def test_rule_segmentation_preserves_hyphenated_words_in_a_paragraph(self):
        import spacy
        from speech_pipeline import utterances as auto
        nlp=spacy.load('en_core_web_sm')
        text="we go uh-huh i don't this one uh-huh yes not"
        predicted=auto.predict([dict(row=2,speaker='CHI',text=text,start='',end='')],'CHI',nlp)
        self.assertEqual(auto.stream(predicted)[0],auto.stream([dict(text=text)])[0])

    def test_talkbank_training_uses_labels_without_gold_input_boundaries(self):
        import spacy
        nlp=spacy.blank('en');nlp.add_pipe('sentencizer')
        with TemporaryDirectory() as temp:
            base=Path(temp);source=base/'public/ocsc/transcripts'
            source.mkdir(parents=True)
            (source/'child.cha').write_text('@UTF8\n@Begin\n@Languages:\teng\n'
                '@Participants:\tCHI Target_Child, MOT Mother\n@G:\tHowto\n'
                '*CHI:\tHello there .\n*CHI:\tNice day .\n*MOT:\tyes .\n'
                '*CHI:\tI like this .\n*CHI:\tMore words .\n@End\n')
            identity=base/'identities.csv'
            identity.write_text('corpus,recording,speaker,participant,verified\n'
                                'ocsc,transcripts/child.cha,CHI,child001,true\n')
            with patch('speech_pipeline.corpora.DATA',base/'public'),patch('speech_pipeline.corpora.EVIDENCE',base/'evidence'):
                result=prepare_training(['ocsc'],identity,nlp=nlp)
                import json
                row=json.loads(Path(result['path']).read_text())
            self.assertEqual(row['participant'],'ocsc/child001')
            self.assertEqual(row['tokens'],['hello','there','nice','day','i','like','this','more','words'])
            self.assertEqual(row['labels'],[0,1,0,1,0,0,1,0,-100])
            self.assertEqual(row['gold_boundaries'],[2,4,7])
            self.assertNotIn('forced_boundaries',row)
            # Rule input has neither published punctuation nor sentence casing.
            self.assertEqual(row['baseline'],[0]*9)

    def test_talkbank_training_rejects_unverified_identities(self):
        with TemporaryDirectory() as temp:
            path=Path(temp)/'identities.csv'
            path.write_text('corpus,recording,speaker,participant,verified\n'
                            'ocsc,transcripts/child.cha,CHI,child001,false\n')
            with self.assertRaisesRegex(ValueError,'must be verified'):
                prepare_training(['ocsc'],path,nlp=object())

    def test_login_html_is_not_imported(self):
        with TemporaryDirectory() as temp:
            p=Path(temp)/'fake.zip';p.write_text('<html>Login</html>')
            with self.assertRaises(ValueError):install_transcripts('ocsc',p)

    def test_zip_path_traversal_is_rejected(self):
        with TemporaryDirectory() as temp:
            p=Path(temp)/'bad.zip'
            with zipfile.ZipFile(p,'w') as z:z.writestr('../../escape.cha','*CHI: hello .')
            with patch('speech_pipeline.corpora.DATA',Path(temp)/'public'):
                with self.assertRaises(ValueError):install_transcripts('ocsc',p)
            self.assertFalse((Path(temp)/'escape.cha').exists())

    def test_import_preserves_existing_different_transcript(self):
        with TemporaryDirectory() as temp:
            p=Path(temp)/'corpus.zip'
            with zipfile.ZipFile(p,'w') as z:z.writestr('child.cha','*CHI: hello .')
            with patch('speech_pipeline.corpora.DATA',Path(temp)/'public'):
                folder=install_transcripts('ocsc',p)
                (folder/'child.cha').write_text('*CHI: changed .')
                with self.assertRaises(FileExistsError):install_transcripts('ocsc',p)
                self.assertEqual((folder/'child.cha').read_text(),'*CHI: changed .')

    def test_gold_punctuation_is_withheld_and_retracing_is_audible(self):
        with TemporaryDirectory() as temp:
            path=Path(temp)/'sample.cha'
            path.write_text('@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n'
                '@G:\tHowto\n*CHI:\t<I want> [/] I want cookies . \x150_1200\x15\n'
                '%mor:\tpro|I v|want n|cookie-PL .\n@End\n')
            row=read_chat(path)[0]
            self.assertEqual(row['text'],'I want I want cookies')
            self.assertEqual(row['gem'],'Howto');self.assertEqual(row['time_marks'],[0,1200])
            self.assertEqual(row['retrace_marks'],1)
            self.assertEqual(row['parsed_main_body'],['I','want','cookies'])
            self.assertNotIn('.',row['text'])

    def test_non_speech_chat_events_are_not_invented_spoken_words(self):
        with TemporaryDirectory() as temp:
            path=Path(temp)/'events.cha'
            path.write_text('@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n'
                '*CHI:\ttruck &=laughs .\n*CHI:\t<&=laughs> [<] .\n'
                '*CHI:\t&-um I want [/] want cookies .\n*CHI:\the laughs .\n@End\n')
            rows=read_chat(path)
            self.assertEqual(rows[0]['text'],'truck')
            self.assertEqual(rows[1]['text'],'')
            self.assertEqual(rows[2]['text'],'um I want want cookies')
            self.assertEqual(rows[3]['text'],'he laughs')
            self.assertIn('&=laughs',rows[0]['raw_chat'])

    @unittest.skipUnless((ROOT/'.venv/bin/clan-mlu').exists(),'official CLAN CLI not installed')
    def test_chat_continuation_reference_units_agree_with_official_clan(self):
        from speech_pipeline.corpora import chat_reference_units
        from speech_pipeline import utterances as auto
        cases=[('we want +/.','+, some cookies .',1),('we want +//.','some cookies .',2),
               ('we want +.','some cookies .',1),('we want +...','+, some cookies .',1)]
        with TemporaryDirectory() as temp:
            for i,(first,last,n) in enumerate(cases):
                path=Path(temp)/f'continuation{i}.cha'
                path.write_text('@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child, MOT Mother\n'
                    f'*CHI:\t{first}\n*MOT:\tyes .\n*CHI:\t{last}\n@End\n')
                reference=chat_reference_units(read_chat(path));mine=auto.word_mlu(reference);official=run_clan('mlu',path,speaker='CHI')
                self.assertEqual(len(reference),n)
                self.assertEqual(official['utterances'],n)
                self.assertEqual(mine['mlu_nwords'],official['total'])
                self.assertAlmostEqual(mine['mlu_words'],official['mlu'],places=3)

    @unittest.skipUnless(os.name=='posix' and (ROOT/'.venv/bin/clan-mlu').exists(),'official CLAN CLI not installed')
    def test_official_clan_word_counts_and_mattr(self):
        with TemporaryDirectory() as temp:
            path=Path(temp)/'sample.cha';path.write_text('@UTF8\n@Begin\n@Languages:\teng\n'
                '@Participants:\tPAR Target Participant\n@ID:\teng|test|PAR|||||Participant|||\n'
                '*PAR:\tthe dogs played .\n*PAR:\tThe cat slept .\n@End\n')
            mlu=run_clan('mlu',path);freq=run_clan('freq',path,window=3)
            self.assertEqual(mlu['utterances'],2);self.assertEqual(mlu['total'],6);self.assertEqual(mlu['mlu'],3)
            self.assertEqual(freq['TNW'],6);self.assertEqual(freq['NDW'],5)
            self.assertEqual(freq['MATTR'],1)
            self.assertEqual(mlu['sd'],0)

    @unittest.skipUnless(os.name=='posix' and (ROOT/'.venv/bin/clan-freq').exists(),'official CLAN CLI not installed')
    def test_chat_apostrophes_match_surface_word_normalization(self):
        from speech_pipeline import utterances as auto
        units=[dict(utterance_id=1,text="It's here and it’s there.")]
        profile=clinical_from_predictions(units)['values']
        with TemporaryDirectory() as temp:
            path=Path(temp)/'sample.cha';path.write_text(auto.chat_text(units,'apostrophe regression'))
            actual=run_clan('freq',path)
        self.assertEqual(actual['NDW'],profile['NDW'])
        self.assertEqual(actual['TNW'],profile['TNW'])
        self.assertEqual(units[0]['text'],"It's here and it’s there.")


@unittest.skipUnless(importlib.util.find_spec('torch'),'optional transformer dependency not installed')
class ParagraphModelTests(unittest.TestCase):
    @staticmethod
    def training_records():
        records=[]
        for corpus in ('ocsc','nyu_emerson'):
            for i in range(10):
                words=['we','like','the','games']*8+[f'child{i}',corpus]
                labels=[int((j+1)%4==0) for j in range(len(words))];labels[-1]=-100
                records.append(dict(corpus=corpus,participant=f'{corpus}/{i}',participant_verified=True,
                    task='paragraph_utterance_boundaries',recording=f'{i}.cha',tokens=words,
                    labels=labels,baseline=[0]*len(words)))
        return records

    def test_holdout_labels_and_rules_cannot_change_trained_weights(self):
        import torch
        from speech_pipeline.boundary_model import grouped_split,train_talkbank,segment_paragraph
        records=self.training_records()
        splits=grouped_split(records,42)
        self.assertFalse(set(splits['train'])&set(splits['validation']))
        self.assertFalse(set(splits['train'])&set(splits['test']))
        self.assertFalse(set(splits['validation'])&set(splits['test']))
        duplicates=records+[dict(records[0],participant='ocsc/duplicate')]
        duplicate_splits=grouped_split(duplicates,42)
        self.assertEqual(next(k for k,v in duplicate_splits.items() if 'ocsc/0' in v),
                         next(k for k,v in duplicate_splits.items() if 'ocsc/duplicate' in v))
        with TemporaryDirectory() as temp:
            base=Path(temp);data=base/'data.jsonl'
            data.write_text(''.join(json.dumps(r)+'\n' for r in records))
            first=train_talkbank(data,base/'first',epochs=1)
            for r in records:
                r['baseline']=[1]*len(r['tokens'])
                if r['participant'] in splits['test']:
                    r['labels']=[1-v if v>=0 else v for v in r['labels']]
            data.write_text(''.join(json.dumps(r)+'\n' for r in records))
            second=train_talkbank(data,base/'second',epochs=1)
            a=torch.load(base/'first/talkbank_model.pt',weights_only=True)
            b=torch.load(base/'second/talkbank_model.pt',weights_only=True)
            self.assertEqual(first['threshold'],second['threshold'])
            self.assertEqual(a['vocabulary'],b['vocabulary'])
            self.assertTrue(all(torch.equal(a['state_dict'][k],b['state_dict'][k]) for k in a['state_dict']))
            text='We like the games.  We like the games!'
            segmented=segment_paragraph(text,base/'first/talkbank_model.pt')
            self.assertEqual(''.join(r['text'].replace(' ','') for r in segmented),text.replace(' ',''))
            self.assertTrue(all(r['text']==text[r['source_start']:r['source_end']] for r in segmented))

    def test_hybrid_checkpoint_uses_rules_without_learning_from_test_labels(self):
        import torch
        from speech_pipeline.boundary_model import grouped_split,train_talkbank,segment_paragraph,paragraph_rules
        records=self.training_records();splits=grouped_split(records,42)
        for r in records:r['baseline']=[int(i%3==0) for i in range(len(r['tokens']))]
        with TemporaryDirectory() as temp:
            base=Path(temp);data=base/'data.jsonl'
            data.write_text(''.join(json.dumps(r)+'\n' for r in records))
            first=train_talkbank(data,base/'first',epochs=1,use_rules=True)
            for r in records:
                if r['participant'] in splits['test']:r['labels']=[1-v if v>=0 else v for v in r['labels']]
            data.write_text(''.join(json.dumps(r)+'\n' for r in records))
            second=train_talkbank(data,base/'second',epochs=1,use_rules=True)
            checkpoint=base/'first/hybrid_model.pt'
            a=torch.load(checkpoint,weights_only=True);b=torch.load(base/'second/hybrid_model.pt',weights_only=True)
            self.assertTrue(a['use_rules'])
            self.assertEqual(first['threshold'],second['threshold'])
            self.assertTrue(all(torch.equal(a['state_dict'][k],b['state_dict'][k]) for k in a['state_dict']))
            text='We like the games. We like the games!'
            with patch('speech_pipeline.boundary_model.paragraph_rules',wraps=paragraph_rules) as rules:
                result=segment_paragraph(text,checkpoint)
                rules.assert_called_once()
                self.assertEqual(rules.call_args.args[0],['we','like','the','games']*2)
            self.assertEqual(''.join(r['text'].replace(' ','') for r in result),text.replace(' ',''))


@unittest.skipUnless(importlib.util.find_spec('transformers') and importlib.util.find_spec('safetensors'),
                     'optional pretrained dependencies not installed')
class PretrainedSegmentationTests(unittest.TestCase):
    def test_publication_aggregation_does_not_mix_samples_and_recording_errors(self):
        from speech_pipeline import utterances as auto
        from speech_pipeline.research.reassessment import publication_comparison
        score=lambda text:auto.word_mlu([dict(text=t) for t in text])
        rows=[dict(participant='A',predicted=score(['a','b','c d']),reference=score(['a b c d'])),
              dict(participant='A',predicted=score(['e f','g h','i j']),reference=score(['e f','g h','i j'])),
              dict(participant='B',predicted=score(['x','y z w']),reference=score(['x','y z w']))]
        result=publication_comparison(rows)
        self.assertEqual(result['paper_mdape']['total_utterances'],25)
        self.assertEqual(result['poster_mape']['total_utterances'],50)

    def test_reassessment_locks_selection_and_excludes_private_development(self):
        from speech_pipeline.research.reassessment import partitions,evaluate
        with patch('speech_pipeline.research.reassessment.read',return_value={'selection':{}}),\
             patch('speech_pipeline.research.reassessment.load_partitions') as load:
            with self.assertRaisesRegex(ValueError,'frozen'):partitions()
            load.assert_not_called()
        with patch('speech_pipeline.research.reassessment.read',return_value={'input_sha256':'fixture'}),\
             patch('speech_pipeline.research.reassessment.load_partitions',return_value=(
                 {'train':[dict(corpus='existing_development_cohort')]},{'input_sha256':'fixture'})):
            with self.assertRaisesRegex(ValueError,'Only public'):partitions()
        with patch('speech_pipeline.research.reassessment.read',return_value={}):
            with self.assertRaisesRegex(ValueError,'Freeze'):evaluate()

    def test_metric_first_round_keeps_final_groups_out_of_development(self):
        from speech_pipeline.research.metric_training import public_development
        with patch('speech_pipeline.research.metric_training.read',return_value={'selection':{}}),\
             patch('speech_pipeline.research.metric_training.load_partitions') as loader:
            with self.assertRaises(ValueError):public_development()
            loader.assert_not_called()
        protocol={'input_sha256':'fixture','partitions':{'train':['one'],'validation':['two'],'test':['three']}}
        with patch('speech_pipeline.research.metric_training.read',return_value=protocol),\
             patch('speech_pipeline.research.metric_training.load_partitions',return_value=(
                 {'train':[{'corpus':'private'}],'validation':[]},protocol)):
            with self.assertRaises(ValueError):public_development()

    def test_metric_selection_prioritizes_language_error_and_guards_final_access(self):
        from speech_pipeline.research import metric_training as experiment
        with patch.object(experiment,'read',return_value={}),patch.object(experiment,'load_partitions') as loader:
            with self.assertRaises(ValueError):experiment.evaluate()
            loader.assert_not_called()
        with TemporaryDirectory() as temp:
            base=Path(temp);entries={}
            for name,error,f1 in [('language_best',7.,.80),('boundary_best',10.,.95)]:
                source=base/name;source.mkdir();(source/'model.safetensors').write_bytes(name.encode())
                (source/'tokenizer.json').write_text('{}');(source/'segmenter.json').write_text('{}')
                entries[name]=dict(status='complete',artifact=str(source),parameters=100,
                    validation=dict(language_metrics=dict(objective=error),boundaries=dict(overall=dict(f1=f1))))
            plan=dict(candidates=list(entries),experiments=entries,primary_selection='fixture language error')
            with patch.object(experiment,'read',return_value=plan),patch.object(experiment,'OUT',base),patch.object(experiment,'PLAN',base/'plan.json'):
                experiment.select()
            selected=json.loads((base/'plan.json').read_text())['selection']
            self.assertEqual(selected['name'],'language_best')
            self.assertEqual((base/'metric_segmenter/model.safetensors').read_bytes(),b'language_best')
            self.assertTrue(selected['before_final_evaluation'])

    def test_metric_training_preserves_completed_candidates_and_context_dependency(self):
        from speech_pipeline.research import metric_training as experiment
        plan=dict(candidates=['deberta_context','deberta_metric'],experiments={
            'deberta_context':dict(status='complete')})
        with patch.object(experiment,'public_development',return_value=({},plan)),\
             patch.object(experiment,'load_bundle') as loader:
            with self.assertRaisesRegex(ValueError,'Completed candidate is frozen'):
                experiment.train('deberta_context')
            loader.assert_not_called()
        plan['experiments']['deberta_context']['status']='training'
        with patch.object(experiment,'public_development',return_value=({},plan)),\
             patch.object(experiment,'load_bundle') as loader:
            with self.assertRaisesRegex(ValueError,'Complete context selection'):
                experiment.train('deberta_metric')
            loader.assert_not_called()

    def test_spoken_development_preparation_removes_event_rows_without_gold_input_cues(self):
        from speech_pipeline.research import metric_training as experiment
        from speech_pipeline.pipeline import digest
        with TemporaryDirectory() as temp:
            base=Path(temp);path=base/'dataset/public/ocsc/transcripts/child.cha';path.parent.mkdir(parents=True)
            path.write_text('@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Child\n@G:\tHowto\n'
                '*CHI:\tI want cookies .\n*CHI:\t&=laughs .\n*CHI:\tmore cookies .\n@End\n')
            old=dict(corpus='ocsc',participant='ocsc/child',recording='transcripts/child.cha#section1',gem='Howto',
                tokens=['i','want','cookies','laughs','more','cookies'],source_sha256=digest(path))
            with patch.object(experiment,'ROOT',base),patch.object(experiment,'paragraph_rules',side_effect=lambda words,nlp:[0]*len(words)) as rules:
                rows,audit=experiment.spoken_records([old],{'ocsc/child'},object())
            self.assertEqual(rows[0]['tokens'],['i','want','cookies','more','cookies'])
            self.assertEqual(rows[0]['gold_boundaries'],[3])
            self.assertEqual(rows[0]['labels'],[0,0,1,0,-100])
            self.assertEqual(rules.call_args.args[0],rows[0]['tokens'])
            self.assertEqual(audit['words_after'],5)
            with patch.object(experiment,'read_chat') as reader:
                with self.assertRaises(ValueError):experiment.spoken_records([old],set(),object())
                reader.assert_not_called()

    def test_metric_uncertainty_clusters_recordings_by_participant_and_keeps_zero_error(self):
        from speech_pipeline.research.metric_training import metric_intervals
        rows=[]
        for child,before,after in [('a',10.,0.),('b',20.,10.)]:
            for recording in ('one','two'):
                for method,value in [('metric_baseline',before),('metric_selected',after)]:
                    rows.append(dict(corpus='ocsc',participant=child,recording=recording,method=method,metric='surface/MLUw',absolute_percent_error=value))
        result=metric_intervals(rows)['public']['surface/MLUw']
        self.assertEqual(result['participants'],2)
        self.assertEqual(result['median_error_ci'],[0.,10.])
        self.assertEqual(result['difference_ci'],[-10.,-10.])

    def test_metric_objective_uses_length_variation_and_corpus_balance(self):
        import numpy as np
        from speech_pipeline.research.metric_training import metric_result
        row=dict(corpus='ocsc',participant='a',recording='one',tokens=['one','two','three','four','five','six'],gold_boundaries=[2])
        perfect=np.array([False,True,False,False,False,False])
        equal_lengths=np.array([False,False,True,False,False,False])
        self.assertEqual(metric_result([row],[perfect])['objective'],0)
        bad=metric_result([row],[equal_lengths]);self.assertEqual(bad['recordings'][0]['ape']['total_utterances'],0)
        self.assertEqual(bad['recordings'][0]['ape']['mlu_words'],0)
        self.assertEqual(bad['recordings'][0]['ape']['mlu_words_sd'],100)
        other=dict(row,corpus='nyu_emerson',participant='b')
        balanced=metric_result([row,row,other],[perfect,perfect,equal_lengths])
        self.assertAlmostEqual(balanced['objective'],100/6)
        zero_sd=dict(row,gold_boundaries=[3])
        self.assertIsNone(metric_result([zero_sd],[equal_lengths])['recordings'][0]['ape']['mlu_words_sd'])

    def test_metric_reference_uses_human_words_with_aligned_boundary_labels(self):
        import numpy as np
        from speech_pipeline.research.metric_training import metric_result
        row=dict(corpus='private',participant='a',recording='one',tokens=['one','two'],
                 metric_reference={'total_utterances':1,'mlu_words':4,'mlu_words_sd':0})
        result=metric_result([row],[np.array([False,False])])['recordings'][0]
        self.assertEqual(result['reference']['mlu_words'],4)
        self.assertEqual(result['ape']['mlu_words'],50)
        self.assertIsNone(result['ape']['mlu_words_sd'])

    def test_composite_intervals_preserve_equal_corpus_weight_and_paired_children(self):
        from speech_pipeline.research.metric_training import MEASURES,objective_intervals
        before=[];after=[]
        for corpus,children,error in [('ocsc',5,20.),('nyu_emerson',1,40.)]:
            for child in range(children):
                row=dict(corpus=corpus,participant=f'{corpus}/{child}',recording='one')
                before.append(dict(row,ape={m:error for m in MEASURES}))
                after.append(dict(row,ape={m:error-10 for m in MEASURES}))
        result=objective_intervals(dict(recordings=before),dict(recordings=after))
        self.assertEqual(result['selected_error_ci'],[20.,20.])
        self.assertEqual(result['difference_ci'],[-10.,-10.])
        self.assertEqual(result['participants_by_corpus'],{'ocsc':5,'nyu_emerson':1})
        self.assertEqual(result['valid_draws'],10000)
        with self.assertRaisesRegex(ValueError,'same participant'):
            objective_intervals(dict(recordings=before),dict(recordings=after[1:]))

    def test_metric_count_loss_ignores_masked_gaps_and_has_finite_gradients(self):
        import torch
        from speech_pipeline.research.metric_training import count_loss
        logits=torch.tensor([[[0.,0.],[99.,-99.],[0.,0.]]],requires_grad=True)
        labels=torch.tensor([[1,-100,0]])
        loss=count_loss(logits,labels)
        self.assertEqual(float(loss.detach()),0)
        loss.backward();self.assertTrue(torch.isfinite(logits.grad).all())
        torch.testing.assert_close(logits.grad[:,1],torch.zeros_like(logits.grad[:,1]))
        wrong=torch.zeros((1,3,2),requires_grad=True);value=count_loss(wrong,torch.tensor([[1,-100,1]]))
        self.assertGreater(float(value.detach()),0)
        value.backward();self.assertGreater(float(wrong.grad[:,0].abs().sum()),0)
        torch.testing.assert_close(wrong.grad[:,1],torch.zeros_like(wrong.grad[:,1]))

    def test_automatic_sentence_and_clause_cues_are_separate_word_gap_inputs(self):
        from types import SimpleNamespace
        from speech_pipeline.segmentation import sentence_clause_features
        nlp=lambda text:SimpleNamespace(sents=[SimpleNamespace(start_char=i) for i in (0,8,13)])
        with patch('speech_pipeline.segmentation.paragraph_rules',return_value=[0,1,0,0,0]) as clause:
            cues=sentence_clause_features(['we','like','dogs','and','cats'],nlp)
        self.assertEqual(cues,[0,3,2,0,0])
        self.assertEqual(clause.call_args.args[0],['we','like','dogs','and','cats'])

    def test_expected_utterance_length_moment_matches_all_possible_segmentations(self):
        import itertools,torch
        from speech_pipeline.research.metric_training import expected_span_square,length_moment_loss
        logits=torch.tensor([[.5,-.2],[.1,.6],[-.8,.9]],dtype=torch.float64,requires_grad=True)
        probabilities=logits.softmax(-1);exact=logits.sum()*0
        for path in itertools.product((0,1),repeat=3):
            probability=torch.stack([probabilities[i,label] for i,label in enumerate(path)]).prod()
            cuts=[0]+[i+1 for i,label in enumerate(path) if label]+[4]
            exact=exact+probability*sum((b-a)**2 for a,b in zip(cuts,cuts[1:]))
        torch.testing.assert_close(expected_span_square(logits),exact)
        value=length_moment_loss(logits[None],torch.tensor([[0,1,0]]));value.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad.abs().sum()),0)

    def test_four_state_sentence_cue_model_bundle_roundtrip(self):
        import numpy as np,torch
        from transformers import BertConfig
        from speech_pipeline.segmentation import PretrainedGapModel,pretrained_probabilities,save_bundle,load_bundle
        tokenizer=self.tokenizer();torch.set_num_threads(2)
        config=BertConfig(vocab_size=len(tokenizer),hidden_size=16,num_hidden_layers=1,num_attention_heads=2,intermediate_size=32)
        model=PretrainedGapModel(config=config,rule_states=4)
        row=dict(tokens=['we','like','dogs','and','cats']*4,baseline=[0,1,2,3,0]*4)
        original=pretrained_probabilities(model,tokenizer,[row])[0]
        with TemporaryDirectory() as temp:
            save_bundle(model,tokenizer,temp,dict(name='sentence_cue_fixture',rule_feature_kind='sentence_clause',decoding='argmax',threshold=None))
            restored,t,meta=load_bundle(temp)
            self.assertEqual(meta['rule_states'],4)
            self.assertEqual(restored.rule_embedding.num_embeddings,4)
            np.testing.assert_allclose(original,pretrained_probabilities(restored,t,[row])[0],atol=1e-7)

    def test_byte_level_bundle_preserves_training_word_ids_after_reload(self):
        from tokenizers import Tokenizer,models,pre_tokenizers,trainers
        from transformers import PreTrainedTokenizerFast,RobertaConfig
        from speech_pipeline.segmentation import PretrainedGapModel,save_bundle,load_bundle
        backend=Tokenizer(models.BPE(unk_token='<unk>'))
        backend.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=True)
        backend.train_from_iterator(['we like dogs and cats']*3,trainers.BpeTrainer(
            vocab_size=50,special_tokens=['<s>','</s>','<unk>','<pad>','<mask>']))
        tokenizer=PreTrainedTokenizerFast(tokenizer_object=backend,add_prefix_space=True,
            cls_token='<s>',sep_token='</s>',unk_token='<unk>',pad_token='<pad>',mask_token='<mask>')
        words=['we','like','dogs','and','cats']
        expected=tokenizer(words,is_split_into_words=True,add_special_tokens=False)['input_ids']
        config=RobertaConfig(vocab_size=len(tokenizer),hidden_size=16,num_hidden_layers=1,
            num_attention_heads=2,intermediate_size=32,pad_token_id=tokenizer.pad_token_id)
        with TemporaryDirectory() as temp:
            save_bundle(PretrainedGapModel(config=config),tokenizer,temp,dict(name='byte_level_fixture'))
            _,restored,_=load_bundle(temp)
            self.assertEqual(restored(words,is_split_into_words=True,add_special_tokens=False)['input_ids'],expected)
            self.assertTrue(json.loads(restored.backend_tokenizer.pre_tokenizer.__getstate__())['add_prefix_space'])

    def test_sequence_decoder_matches_exhaustive_paths_and_partial_labels(self):
        import itertools,numpy as np,torch
        from speech_pipeline.segmentation import BoundaryCRF
        crf=BoundaryCRF()
        with torch.no_grad():
            crf.transitions.copy_(torch.tensor([[.2,-.1],[.3,-.2]]))
            crf.start.copy_(torch.tensor([.1,-.2]));crf.end.copy_(torch.tensor([-.1,.2]))
        emissions=torch.tensor([[[.3,-.4],[-.1,.7],[.2,.45]]],requires_grad=True)
        paths=list(itertools.product(range(2),repeat=3))
        scores=torch.stack([crf.start[p[0]]+crf.end[p[-1]]+
            sum(emissions[0,i,p[i]] for i in range(3))+
            sum(crf.transitions[p[i-1],p[i]] for i in range(1,3)) for p in paths])
        np.testing.assert_array_equal(crf.decode(emissions.detach().numpy()[0]),paths[int(scores.argmax())])
        self.assertAlmostEqual(float(crf.partition(emissions,torch.tensor([3]))[0].detach()),float(scores.logsumexp(0).detach()),places=6)
        labels=torch.tensor([[-100,1,0]])
        constrained=torch.logsumexp(torch.stack([s for s,p in zip(scores,paths) if p[1]==1 and p[2]==0]),0)
        expected=(scores.logsumexp(0)-constrained)/2
        loss=crf.loss(emissions,labels,torch.tensor([3]))
        self.assertAlmostEqual(float(loss.detach()),float(expected.detach()),places=6)
        loss.backward();self.assertTrue(torch.isfinite(emissions.grad).all())
        padded=torch.cat([emissions.detach(),torch.tensor([[[99.,-99.]]])],dim=1)
        np.testing.assert_allclose(crf.partition(padded,torch.tensor([3])).detach(),crf.partition(emissions,torch.tensor([3])).detach(),rtol=1e-6)

    @staticmethod
    def tokenizer():
        from tokenizers import Tokenizer
        from tokenizers.models import WordPiece
        from tokenizers.pre_tokenizers import BertPreTokenizer
        from transformers import PreTrainedTokenizerFast
        vocab={w:i for i,w in enumerate(['[PAD]','[UNK]','[CLS]','[SEP]','[MASK]',
            'we','like','dogs','and','cats','un','##bel','##ievable'])}
        backend=Tokenizer(WordPiece(vocab,unk_token='[UNK]'));backend.pre_tokenizer=BertPreTokenizer()
        return PreTrainedTokenizerFast(tokenizer_object=backend,pad_token='[PAD]',unk_token='[UNK]',
            cls_token='[CLS]',sep_token='[SEP]',mask_token='[MASK]')

    def test_long_subword_windows_preserve_every_gap_without_gold_features(self):
        import copy
        from speech_pipeline.segmentation import word_windows
        words=['we','like','unbelievable','dogs','and','cats']*60
        r=dict(tokens=words,baseline=[0]*len(words),labels=[0]*(len(words)-1)+[-100],gold_boundaries=[7,30])
        changed=copy.deepcopy(r);changed['labels']=[1]*len(words);changed['gold_boundaries']=[1,2,4,100]
        a=word_windows(r,self.tokenizer());b=word_windows(changed,self.tokenizer())
        self.assertEqual(a,b)
        owned=[g for w in a for g,own in zip(w['gaps'],w['owned']) if own]
        self.assertEqual(sorted(owned),list(range(len(words)-1)))
        self.assertTrue(all(len(w['ids'])<=256 for w in a))

    def test_exact_utterances_are_distinct_from_matching_utterance_counts(self):
        import numpy as np
        from speech_pipeline.segmentation import metrics
        r=dict(corpus='fixture',participant='child',recording='one',tokens=['we','like','dogs','cats'],
            labels=[0,1,0,-100],gold_boundaries=[2])
        result=metrics([r],[np.array([.9,.1,.1,0])],.5)['overall']
        self.assertEqual(result['exact']['reference'],result['exact']['predicted'])
        self.assertEqual(result['exact']['matched'],0)
        self.assertEqual(result['f1'],0)

    def test_pretrained_bundle_roundtrip_and_shared_metric_pipeline(self):
        import numpy as np
        import torch,spacy
        from transformers import BertConfig
        from speech_pipeline.segmentation import PretrainedGapModel,save_bundle,load_bundle,pretrained_probabilities,Segmenter
        from speech_pipeline.pipeline import run_paragraph
        from speech_pipeline import utterances as auto
        torch.manual_seed(42);torch.set_num_threads(2)
        tokenizer=self.tokenizer()
        config=BertConfig(vocab_size=len(tokenizer),hidden_size=16,num_hidden_layers=1,
            num_attention_heads=2,intermediate_size=32,pad_token_id=tokenizer.pad_token_id)
        model=PretrainedGapModel(config=config)
        words=['we','like','unbelievable','dogs','and','cats']*20
        row=dict(tokens=words,baseline=[0]*len(words))
        a=pretrained_probabilities(model,tokenizer,[row])
        nlp=spacy.load('en_core_web_sm')
        with TemporaryDirectory() as temp:
            base=Path(temp);folder=base/'model'
            save_bundle(model,tokenizer,folder,dict(name='offline_test_fixture',threshold=.99,input_sha256='fixture',max_length=256,stride=128))
            loaded,t,_=load_bundle(folder)
            b=pretrained_probabilities(loaded,t,[row]);np.testing.assert_allclose(a[0],b[0],atol=1e-7)
            segmenter=Segmenter(folder,nlp)
            text='  '+('We like unbelievable dogs and cats. '*20)+' '
            segments=segmenter.segment(text)
            self.assertEqual(auto.stream(segments)[0],[w.casefold() for w in auto.WORD.findall(text)])
            self.assertTrue(all(s['text']==text[s['source_start']:s['source_end']] for s in segments))
            rows=[dict(row=2,speaker='CHI',text='we like',start='0',end='1'),
                  dict(row=3,speaker='CHI',text='dogs',start='1',end='2'),
                  dict(row=4,speaker='MOT',text='where',start='2',end='3'),
                  dict(row=5,speaker='CHI',text='and cats',start='3',end='4')]
            contextual=segmenter.target_rows(rows,'CHI')
            self.assertEqual([u['source_rows'] for u in contextual],[[2,3],[5]])
            self.assertEqual(contextual[0]['source_start'],'0')
            self.assertEqual(contextual[0]['timing_status'],'source_segment_envelope_not_utterance_alignment')
            paragraph_units=segmenter.target_paragraph(rows,'CHI')
            self.assertEqual([u['source_rows'] for u in paragraph_units],[[2,3,5]])
            self.assertNotIn('turn_id',paragraph_units[0])
            self.assertEqual(auto.stream(paragraph_units)[0],['we','like','dogs','and','cats'])
            paragraph=base/'input.txt';paragraph.write_text(text)
            with patch('speech_pipeline.pipeline.resolve_segmenter',return_value=segmenter),\
                 patch('speech_pipeline.pipeline.segmenter_evidence',return_value=dict(name='offline_test_fixture')):
                report=run_paragraph(paragraph,nlp,base/'results','clinical',verify_clan=(ROOT/'.venv/bin/clan-mlu').exists())
            record=report['recordings'][0]
            self.assertEqual(record['requested_metrics']['values']['TNW'],120)
            self.assertEqual(len(record['lexical']),17)
            self.assertIsNone(record['requested_metrics']['values']['MLUm'])
            self.assertTrue((base/'results/paragraph/clinical_metrics.csv').is_file())
            self.assertTrue((base/'results/paragraph'/record['chat_file']).is_file())
            if (ROOT/'.venv/bin/clan-mlu').exists():
                self.assertEqual(record['clan']['requested_metrics']['TNW']['absolute_error'],0)
                self.assertLessEqual(record['clan']['requested_metrics']['MLUw']['absolute_error'],.000501)
            from speech_pipeline.segmentation import BoundaryCRF,pretrained_predictions
            model.crf=BoundaryCRF()
            sequence=base/'sequence_model'
            save_bundle(model,tokenizer,sequence,dict(name='sequence_fixture',threshold=None,
                decoder='crf',decoding='viterbi',input_sha256='fixture',max_length=256,stride=128))
            restored,t,_=load_bundle(sequence)
            np.testing.assert_array_equal(pretrained_predictions(model,tokenizer,[row])[0],
                                          pretrained_predictions(restored,t,[row])[0])
            units=Segmenter(sequence,nlp).segment(text)
            self.assertEqual(auto.stream(units)[0],[w.casefold() for w in auto.WORD.findall(text)])


if __name__=='__main__':unittest.main()
