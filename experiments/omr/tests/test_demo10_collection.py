"""Run the production collector/controller with 128 deterministic native-interface workers.

These are CPU contract tests, not claims about remote simulator/GPU acceptance.
"""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch

from experiments.omr.tests.test_demo10 import parent
from experiments.omr.tests.test_full_protocol import config_for, native_manager, NativeScience, Actor
from experiments.omr.demo_manifest import publish
from experiments.omr.full_protocol import FullCollectorObserver


class ActiveCollectionTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.root=Path(temp.name)
        source=self.root/'parent.json'; source.write_text(json.dumps(parent()))
        self.manifest,self.schedule,_,_=publish(source,self.root/'bundle','scienceworld','a'*40)
        self.config=config_for(self.root)
        self.config.env.rollout.n=8
        self.config.omr.demo={'directory':str(self.root/'bundle')}
        self.config.omr.task_planner_profile_id='fixed-eight-active-tasks-v1'
        self.config.omr.restart_fraction=0.0  # Forced archive branches must bypass this old coin.
        self.workers=[NativeScience() for _ in range(128)]
        self.manager=native_manager(self.config,[Actor(w) for w in self.workers])
        class Tokenizer:
            def apply_chat_template(self, chat, **kwargs): return chat[0]['content']
            def __call__(self, text, **kwargs): return {'input_ids':list(text.encode())}
        self.observer=FullCollectorObserver(self.config,self.manager,self.manifest,self.root/'rollouts',Tokenizer())
        ray=ModuleType('ray'); ray.get=lambda refs:refs
        patched=patch.dict(sys.modules,{'ray':ray}); patched.start(); self.addCleanup(patched.stop)

    def collect(self, update):
        o=self.observer
        o.plan([t for t in self.schedule[update] for _ in range(8)],update=update,split='train')
        obs,infos=o.initialize(); o.reset(obs,infos)
        next_obs,rewards,dones,infos=self.manager.step(['progress']*128)
        groupids=[f'update-{update}-group-{i//8}' for i in range(128)]
        batch=SimpleNamespace(batch={'responses':torch.ones(128,2),'attention_mask':torch.ones(128,5)},
            non_tensor_batch={'uid':np.array(groupids),'traj_uid':np.array([f'{update}-traj-{i}' for i in range(128)]),
                'analysis_step_id':np.array([f'{update}-step-{i}' for i in range(128)]),
                'omr_synced_policy_version':np.array([update]*128),
                'omr_weight_sync_sequence':np.array([1]*128),'raw_response':np.array(['progress']*128)})
        o.record(batch,obs,next_obs,infos,rewards,dones,np.ones(128,dtype=bool),0)
        o.finish()
        return o.commit(update+1),batch

    def test_empty_then_eight_archive_groups_resume_and_group_identity(self):
        metrics,_=self.collect(0)
        self.assertEqual(metrics['demo/actual_archive_groups'],0)
        self.assertEqual(metrics['demo/empty_fallback_groups'],8)
        # Captures did not become starts in update0; they are eligible in update1.
        self.assertGreater(len(self.observer.controller.archive.active()),8)
        state=json.loads(json.dumps(self.observer.state_dict()))
        self.observer.checkpoint_committed()
        metrics,batch=self.collect(1)
        self.assertEqual(metrics['demo/actual_archive_groups'],8)
        self.assertEqual(metrics['demo/actual_root_groups'],8)
        self.assertEqual(len(set(batch.non_tensor_batch['uid'])),16)
        self.assertEqual(len(self.observer.steps),128)  # Excludes replayed prefix actions.
        self.assertEqual(self.observer.compute['replayed_action_steps'],64)
        plan=deepcopy(self.observer.active_rows)
        keys=[p.checkpoint_id for p in self.observer.plans]
        self.observer.load_state_dict(state)
        self.observer.plan([t for t in self.schedule[1] for _ in range(8)],update=1,split='train')
        self.assertEqual(plan,self.observer.active_rows)
        self.assertEqual(keys,[p.checkpoint_id for p in self.observer.plans])

    def test_failed_restore_charges_actual_root_only(self):
        self.collect(0)
        for worker in self.workers: worker.diverge=True
        metrics,_=self.collect(1)
        self.assertEqual(metrics['demo/actual_archive_groups'],0)
        self.assertEqual(metrics['demo/restore_failed_groups'],8)
        self.assertEqual(metrics['demo/actual_root_groups'],16)
        self.assertTrue(all(k.startswith('root:') for k in self.observer.controller.statistics.events))
        self.assertEqual(len(self.observer.plans),16)

    def test_same_task_repeats_keep_sixteen_groups(self):
        self.collect(0)
        archive=self.observer.controller.archive
        first=next(iter(archive.entries.values())).task.instance_id
        archive.entries={k:e for k,e in archive.entries.items() if e.task.instance_id==first}
        metrics,batch=self.collect(1)
        self.assertEqual(metrics['demo/actual_archive_groups'],8)
        self.assertEqual(metrics['demo/distinct_archive_tasks'],1)
        self.assertEqual(metrics['demo/repeated_archive_task_groups'],7)
        self.assertEqual(len(set(batch.non_tensor_batch['uid'])),16)
        self.assertEqual(len(set(batch.non_tensor_batch['traj_uid'])),128)
        self.assertEqual(len({p.group_id for p in self.observer.plans}),16)

    def test_production_checkpoint_saves_demo_planner_and_rejects_wrong_bundle(self):
        import ast
        import random
        import time
        from omegaconf import OmegaConf
        from experiments.omr.latest_checkpoint import LatestCheckpoint, atomic_json, file_sha256
        from experiments.omr.root_protocol import accumulated_cost
        from experiments.omr.task_manifest import digest
        self.collect(0)
        tree=ast.parse((Path(__file__).resolve().parents[3]/'experiments/omr/training.py').read_text())
        node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ProtocolTrainer')
        namespace=dict(globals(),random=random,time=time,RayPPOTrainer=object,LatestCheckpoint=LatestCheckpoint,
                       atomic_json=atomic_json,file_sha256=file_sha256,accumulated_cost=accumulated_cost,
                       digest=digest,__package__='experiments.omr')
        exec(compile(ast.Module([node],[]),'training.py','exec'),namespace)
        cls=namespace['ProtocolTrainer']; trainer=cls.__new__(cls)
        class Actors:
            world_size=2
            def save_checkpoint(self,path,remote,update,max_ckpt_to_keep):
                Path(path).mkdir()
                for rank in range(2):
                    for kind in ('model','optim','extra_state'):
                        torch.save({'update':update},Path(path)/f'{kind}_world_size_2_rank_{rank}.pt')
            def load_checkpoint(self,path,del_local_after_load): pass
        class Loader:
            def state_dict(self): return {'cursor':16}
            def load_state_dict(self,state): self.loaded=state
        trainer.run_dir=self.root; trainer.attempt=self.root/'attempt'; trainer.attempt.mkdir()
        trainer.config=self.config; trainer.signature='demo-fixture'; trainer.current_epoch=0
        trainer.global_steps=1; trainer.last_saved=-1; trainer.final_test_done=False
        trainer.manifest=self.manifest; trainer.schedule=self.schedule; trainer.cost={}
        trainer.actor_rollout_wg=Actors(); trainer.train_dataloader=Loader()
        trainer.train_observer=self.observer; trainer.checkpoints=LatestCheckpoint(self.root/'checkpoints','demo-fixture')
        trainer.logger=SimpleNamespace(logger={'swanlab':SimpleNamespace(status={
            'mode':'online','run_id':'demo-fixture','last_logged_update':1,'last_log_sequence':1})})
        expected=self.observer.state_dict()
        with patch('experiments.omr.latest_checkpoint.shutil.disk_usage',return_value=SimpleNamespace(free=10**12)):
            trainer._save_checkpoint()
        path,_=trainer.checkpoints.latest()
        saved=torch.load(path/'protocol.pt',weights_only=False)
        self.assertEqual(saved['demo_binding'],dict(self.config.omr.demo))
        self.assertEqual(saved['omr']['active_planner']['cursor'],1)
        self.collect(1)
        OmegaConf.set_struct(self.config,True)
        trainer._load_checkpoint()
        self.assertEqual(self.observer.state_dict(),expected)
        self.assertEqual(trainer.global_steps,1)
        self.config.omr.demo.directory='different'
        with self.assertRaisesRegex(ValueError,'demo/parent/bank/mask'):
            trainer._load_checkpoint()


if __name__=='__main__': unittest.main()
