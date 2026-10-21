from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import random
import numpy as np
import torch

from speech_pipeline.research import participant_cv as cv


def rows():
    return [dict(corpus=c,participant=f'{c}/{p}',recording=f'{p}#section{s}')
        for c,n in [('ocsc',30),('nyu_emerson',10)] for p in range(n) for s in range(1,4)]


class ParticipantCVTests(unittest.TestCase):
    def test_each_child_is_held_out_once_with_all_sections_together(self):
        data=rows();folds=cv.make_folds(data)
        self.assertEqual(folds,cv.make_folds(data))
        self.assertEqual(len(folds),5)
        all_ids={r['participant'] for r in data};seen=Counter()
        for f in folds:
            cv.audit_fold(f,all_ids)
            seen.update(f['outer'])
            self.assertEqual(Counter(p.split('/')[0] for p in f['outer']),dict(ocsc=6,nyu_emerson=2))
            for participant in all_ids:
                roles=[role for role in ['fit','inner','outer'] if participant in f[role]]
                self.assertEqual(len(roles),1)
                self.assertEqual(sum(r['participant']==participant for r in data),3)
        self.assertEqual(set(seen),all_ids)
        self.assertTrue(all(n==1 for n in seen.values()))

    def test_inner_or_outer_overlap_and_missing_children_are_rejected(self):
        f=cv.make_folds(rows())[0];expected={r['participant'] for r in rows()}
        with self.assertRaisesRegex(ValueError,'overlap'):
            cv.audit_fold(dict(f,inner=f['inner']+[f['outer'][0]]),expected)
        with self.assertRaisesRegex(ValueError,'Incomplete'):
            cv.audit_fold(dict(f,outer=f['outer'][1:]),expected)

    def test_too_few_participants_is_not_silently_split_by_section(self):
        data=[dict(corpus='nyu_emerson',participant=f'nyu_emerson/{i}',recording=str(j)) for i in range(4) for j in range(50)]
        with self.assertRaisesRegex(ValueError,'participant groups'):cv.make_folds(data)

    def test_original_test_cannot_be_loaded_before_final_freeze(self):
        with TemporaryDirectory() as temp:
            with patch.object(cv,'plan',return_value={}),patch.object(cv,'OUT',Path(temp)),\
                 patch.object(cv,'load_split') as loader:
                with self.assertRaisesRegex(ValueError,'Freeze'):cv.job_rows('final','outer')
                loader.assert_not_called()

    def test_final_fit_and_inner_request_only_original_train_and_validation(self):
        with patch.object(cv,'plan',return_value={}),patch.object(cv,'load_split',return_value=[]) as loader:
            cv.job_rows('final','fit');cv.job_rows('final','inner')
            self.assertEqual([call.args[0] for call in loader.call_args_list],['train','validation'])

    def test_objective_requires_both_corpora_and_all_primary_metrics(self):
        complete=[dict(metric=m,mean_participant_ape=v) for v in [2.,4.] for m in cv.PRIMARY]
        self.assertEqual(cv.objective(complete),3.)
        with self.assertRaisesRegex(ValueError,'Incomplete'):cv.objective(complete[:-1])
        complete[0]['mean_participant_ape']=None
        with self.assertRaisesRegex(ValueError,'Incomplete'):cv.objective(complete)

    def test_epoch_checkpoint_restores_optimizer_scheduler_and_all_cpu_rngs(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);protocol=root/'protocol.json';protocol.write_text('{}')
            model=torch.nn.Linear(2,1);optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
            scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:1/(step+1))
            def update():
                optimizer.zero_grad();loss=model(torch.rand(3,2)).square().sum();loss.backward();optimizer.step();scheduler.step()
                return [p.detach().clone() for p in model.parameters()],random.random(),float(np.random.rand())
            update()
            with patch.object(cv,'PLAN',protocol):
                cv.save_training_checkpoint(root/'state.pt',model,optimizer,scheduler,
                    name='fold_1',epoch=1,protocol_sha256=cv.digest(protocol),history=[],best=1.,stale=0)
                expected=update()
                cv.restore_training_checkpoint(root/'state.pt','fold_1',model,optimizer,scheduler)
                actual=update()
                self.assertTrue(all(torch.equal(a,b) for a,b in zip(expected[0],actual[0])))
                self.assertEqual(expected[1:],actual[1:])
                with self.assertRaisesRegex(ValueError,'mismatch'):
                    cv.restore_training_checkpoint(root/'state.pt','fold_2',model,optimizer,scheduler)


if __name__=='__main__':unittest.main()
