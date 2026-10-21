"""Exercise the CPU preparation -> six job submission path with local file fixtures."""
from contextlib import nullcontext
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.omr.tests.test_demo10 import parent
from experiments.omr.prepare_demo10 import prepare,PARENTS
from experiments.omr.submit_demo10 import submit
from experiments.omr.demo_runtime import TAG, BENCHMARKS, METHODS, run_id
from experiments.omr.latest_checkpoint import file_sha256


class PreparationSubmissionTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.root=Path(temp.name); self.revision='a'*40
        for bench, tag in PARENTS.items():
            path=self.root/'analysis/beacon-omr/manifests'/tag/(bench+'.json')
            path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(parent(bench)))
        model=self.root/'data/model'; model.mkdir(parents=True)
        (model/'model.safetensors').write_bytes(b'fixture weights, not a usable model')
        (model/'tokenizer.json').write_text('{}')
        evidence=self.root/'evidence/beacon-omr/webshop-prepare-165873'; evidence.mkdir(parents=True)
        (evidence/'report.json').write_text(json.dumps(dict(status='prepared_and_tested',source_commit='b'*40)))
        (evidence/'site.json').write_text(json.dumps(dict(source_commit='b'*40,runtime=str(self.root/'envs/webshop'),
            model={'path':str(model),'revision':'c'*40},webshop={'paths':{}})))
        self.original={str(p):file_sha256(p) for p in self.root.rglob('*') if p.is_file()}
        self.calls=[]; self.next_job=123

    def prepare(self):
        with patch('experiments.omr.prepare_demo10.clean_revision'), patch(
            'experiments.omr.prepare_demo10.subprocess.check_output',return_value='fixture common diff'):
            prepare(self.root,self.revision)

    def invoke(self, argv):
        self.calls.append(argv)
        if argv[0]=='squeue':
            return '168028|omr-alfworld-full-resume\n169824|omr-scienceworld-full\n999|omr-demo10-prepare'
        if argv[0]=='sbatch':
            self.next_job+=1
            return str(self.next_job)
        raise AssertionError('Unexpected mutation/query: '+repr(argv))

    def test_publish_configs_submit_once_and_preserve_old_files(self):
        self.prepare(); self.prepare()
        directory=self.root/'analysis/beacon-omr/manifests'/TAG
        for bench in BENCHMARKS:
            for method in METHODS:
                c=json.loads((directory/(bench+'-'+method+'-resolved-config.json')).read_text())
                self.assertEqual(c['ray_init']['num_cpus'],32)
                self.assertEqual(c['omr']['tracking']['step_mode'],'policy_update')
                self.assertEqual(c['omr']['enabled'],method=='active_full')
                self.assertEqual(c['env']['max_steps'],15 if bench=='webshop' else 30)
                self.assertEqual(c['trainer']['project_name'],'beacon-omr-demo10')
        with patch('experiments.omr.submit_demo10.locked',lambda p:nullcontext()), patch(
            'experiments.omr.submit_demo10.clean_revision'), patch(
            'experiments.omr.submit_demo10.shutil.disk_usage',return_value=SimpleNamespace(free=10**12)):
            # The real lock creates this directory; the Windows fixture uses a no-op lock.
            (self.root/'analysis/beacon-omr/training-submissions'/TAG).mkdir(parents=True)
            preview=submit(self.root,self.revision,invoke=self.invoke)
            self.assertEqual(len(preview['commands']),6)
            state=submit(self.root,self.revision,apply=True,invoke=self.invoke)
            again=submit(self.root,self.revision,apply=True,invoke=self.invoke)
        self.assertEqual(state,again)
        self.assertEqual(len([c for c in self.calls if c[0]=='sbatch']),6)
        self.assertEqual(self.original,{p:file_sha256(p) for p in self.original})
        self.assertTrue(all(c[0] in ('squeue','sbatch') for c in self.calls))

    def test_quota_partial_retry_submits_only_missing_jobs(self):
        self.prepare()
        quota=True
        def invoke(argv):
            nonlocal quota
            if argv[0]=='sbatch' and self.next_job==125 and quota:
                quota=False
                raise subprocess.CalledProcessError(1,argv,output='',stderr=
                    'QOSMaxSubmitJobPerUserLimit\nBatch job submission failed: Job violates accounting/QOS policy')
            return self.invoke(argv)
        with patch('experiments.omr.submit_demo10.locked',lambda p:nullcontext()), patch(
            'experiments.omr.submit_demo10.clean_revision'), patch(
            'experiments.omr.submit_demo10.shutil.disk_usage',return_value=SimpleNamespace(free=10**12)):
            (self.root/'analysis/beacon-omr/training-submissions'/TAG).mkdir(parents=True)
            with self.assertRaisesRegex(RuntimeError,'quota rejected'):
                submit(self.root,self.revision,apply=True,invoke=invoke)
            state=submit(self.root,self.revision,apply=True,invoke=invoke)
            self.assertEqual(len(state['jobs']),6)
            self.assertEqual(len([c for c in self.calls if c[0]=='sbatch']),6)


if __name__=='__main__': unittest.main()
