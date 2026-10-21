from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.omr.demo_manifest import derive, artifacts, publish, verify, masks, quotas
from experiments.omr.task_manifest import digest, split_catalog
from experiments.omr.active_task_planner import ActiveTaskPlanner, metrics
from agent_system.milestone_reuse.sampler import RestartSampler
from agent_system.milestone_reuse.controller import ReuseController
from agent_system.milestone_reuse.types import StateEntry, TaskSpec


def parent(bench='scienceworld', n=400):
    train=[dict(id=f'train-{i:04}',target=[0,1],family=str(i%4)) for i in range(n)]
    test=[dict(id=f'test-{i:04}',target=[0,i+2],family=str(i%4)) for i in range(40)]
    return split_catalog(train,test,{'benchmark':bench})


class SubsetTests(unittest.TestCase):
    def test_exact_stratified_holdouts_and_deterministic_bytes(self):
        for bench in ('alfworld','scienceworld','webshop'):
            p=parent(bench); a,m=derive(p,bench)
            self.assertEqual(len(a['splits']['train']),(len(p['splits']['train'])+9)//10)
            self.assertEqual(a['splits']['validation'],p['splits']['validation'])
            self.assertEqual(a['splits']['test'],p['splits']['test'])
            self.assertEqual(set(r['family'] for r in a['splits']['train']),{'0','1','2','3'})
            self.assertEqual(sum(r['selected'] for r in m),36)
            self.assertEqual(artifacts(p,bench,'a'*40),artifacts(p,bench,'a'*40))
            perm=deepcopy(p); random.Random(53).shuffle(perm['splits']['train'])
            perm['sha256']=digest({k:v for k,v in perm.items() if k!='sha256'})
            self.assertEqual(a['splits']['train'],derive(perm,bench)[0]['splits']['train'])

    def test_sparse_webshop_quotas_and_small_pool(self):
        p=parent('webshop')
        for row in p['splits']['train']: row['family']=None
        p['sha256']=digest({k:v for k,v in p.items() if k!='sha256'})
        d,_=derive(p,'webshop')
        self.assertEqual(d['demo']['category_policy'],'single-bucket-hash')
        self.assertEqual(d['demo']['missing_category_fraction'],1)
        for i,row in enumerate(p['splits']['train']): row['family']=str(i)
        p['sha256']=digest({k:v for k,v in p.items() if k!='sha256'})
        d,_=derive(p,'webshop')
        self.assertEqual(d['demo']['minimum_per_stratum'],0)
        self.assertEqual(sum(d['demo']['quotas'].values()),36)
        with self.assertRaisesRegex(ValueError,'fewer than 16'): derive(parent(n=100),'scienceworld')
        self.assertEqual(quotas({'large':99,'tiny':1},10,1),{'large':9,'tiny':1})

    def test_freeze_tampering_and_parent_drift(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); source=root/'parent.json'; source.write_text(json.dumps(parent()))
            target=root/'frozen'
            a=publish(source,target,'scienceworld','a'*40)
            self.assertEqual(a,publish(source,target,'scienceworld','a'*40))
            with self.assertRaisesRegex(ValueError,'Native catalog'): verify(target,parent(n=500))
            with self.assertRaisesRegex(ValueError,'Frozen subset differs'): publish(source,target,'scienceworld','b'*40)
            (target/'branch_masks.json').write_text('[]')
            with self.assertRaisesRegex(ValueError,'modified'): verify(target)


def entries(tasks):
    return {str(i):StateEntry(str(i),TaskSpec('alfworld','train','f',t['id']),
                             'cat','condition','payload','history') for i,t in enumerate(tasks)}


class PlannerTests(unittest.TestCase):
    def setUp(self):
        self.tasks=[dict(id=str(i),family='f',target=i) for i in range(16)]
        self.planner=ActiveTaskPlanner(self.tasks,masks('alfworld'),'bank')

    def test_empty_one_and_many_tasks_exact_budget(self):
        rows=self.planner.plan(self.tasks,{},0)
        self.assertEqual(sum(r['actual_request_branch']=='root' for r in rows),16)
        self.assertEqual(metrics(rows,[None]*16)['demo/empty_fallback_groups'],8)
        self.assertNotIn('demo/restore_success_rate',metrics(rows,[None]*16))
        for n,update in [(1,1),(3,2),(16,3)]:
            rows=self.planner.plan(self.tasks,entries(self.tasks[:n]),update)
            selected=[r for r in rows if r['actual_request_branch']=='archive']
            self.assertEqual(len(selected),8)
            counts=Counter(r['task']['id'] for r in selected)
            self.assertEqual(len(counts),min(n,8))
            self.assertLessEqual(max(counts.values())-min(counts.values()),1)
            self.assertTrue(all(r['task_conditional_probability']==1/len(r['conditional_task_set']) for r in selected))
        self.assertGreater(len({tuple(m) for m in masks('alfworld')}),1)

    def test_resume_next_plan_and_foreign_tasks(self):
        self.planner.plan(self.tasks,entries(self.tasks[:3]),0)
        state=json.loads(json.dumps(self.planner.state_dict()))
        replica=ActiveTaskPlanner(self.tasks,masks('alfworld'),'bank'); replica.load_state_dict(state)
        self.assertEqual(replica.plan(self.tasks,entries(self.tasks[:5]),1),
                         self.planner.plan(self.tasks,entries(self.tasks[:5]),1))
        with self.assertRaisesRegex(ValueError,'outside demo'):
            self.planner.plan(self.tasks,entries([dict(id='foreign')]),2)

    def test_forced_restart_ignores_alpha_without_changing_inner_distribution(self):
        controller=ReuseController(sampler=RestartSampler(alpha=0))
        e=next(iter(entries(self.tasks[:1]).values())); controller.archive.admit(e)
        controller.begin(0)
        expected=controller.sampler.distributions(e.task)
        before=controller.sampler.rng.getstate()
        root=controller.plan(e.task,'root',force='root')
        self.assertIsNone(root.checkpoint_id)
        self.assertEqual(before,controller.sampler.rng.getstate())
        p=controller.plan(e.task,'restart',force='restart')
        self.assertEqual(p.checkpoint_id,e.checkpoint_id)
        self.assertEqual(p.p_root_or_restart,1)
        self.assertEqual(p.p_state_given_channel,expected[p.channel][e.checkpoint_id])
        with self.assertRaisesRegex(ValueError,'eligible'):
            controller.plan(TaskSpec('alfworld','train','f','missing'),'bad',force='restart')


class EvaluationIsolationTests(unittest.TestCase):
    def test_evaluation_preserves_rng_and_detects_evidence_mutation(self):
        import numpy as np
        import torch
        from omegaconf import OmegaConf
        from experiments.omr.evaluation_guard import isolated_evaluation
        state={'count':0}
        obj=SimpleNamespace(config=OmegaConf.create({'omr':{'demo':{'profile':'test'}}}),
                            train_observer=SimpleNamespace(state_dict=lambda:state.copy()))
        r=random.getstate(); n=np.random.get_state(); t=torch.get_rng_state()
        @isolated_evaluation
        def good(self):
            random.random(); np.random.rand(); torch.rand(1)
            return 7
        self.assertEqual(good(obj),7)
        self.assertEqual(random.getstate(),r)
        self.assertTrue(np.array_equal(np.random.get_state()[1],n[1]))
        self.assertTrue(torch.equal(torch.get_rng_state(),t))
        @isolated_evaluation
        def bad(self): state['count']+=1
        with self.assertRaisesRegex(ValueError,'mutated'): bad(obj)


class SubmissionTests(unittest.TestCase):
    def test_commands_are_six_separate_fresh_runs(self):
        from experiments.omr.submit_demo10 import job_command
        from experiments.omr.demo_runtime import BENCHMARKS,METHODS,run_id
        cmds=[job_command(Path('/scratch/project'), 'a'*40,b,m) for b in BENCHMARKS for m in METHODS]
        self.assertEqual(len({next(x for x in cmd if x.startswith('--job-name=')) for cmd in cmds}),6)
        for cmd in cmds:
            self.assertIn('--time=18:00:00',cmd); self.assertIn('--partition=testqueue',cmd)
            self.assertIn('--gres=gpu:h200:2',cmd); self.assertEqual(cmd[-1],'fresh')
            self.assertFalse(any(x.startswith(('--array','--dependency')) for x in cmd))
        with self.assertRaises(ValueError): run_id('webshop','../old')


if __name__=='__main__': unittest.main()
