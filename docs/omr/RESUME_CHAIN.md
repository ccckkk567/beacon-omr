# One queued continuation per training condition

The user reported v2 array **166362**, elements 0–5, pending on testqueue, and
requested one additional 18-hour job per condition. The earlier v1 baseline
jobs were paused by the user. Do not resume or cancel those older jobs.

The user returned successful submission output on 2026-10-19:

| Condition | First job | Continuation job |
|---|---|---|
| ALFWorld baseline | 166362_0 | 166364_0 |
| WebShop baseline | 166362_1 | 166365_1 |
| ScienceWorld baseline | 166362_2 | 166366_2 |
| ALFWorld full | 166362_3 | 166367_3 |
| WebShop full | 166362_4 | 166368_4 |
| ScienceWorld full | 166362_5 | 166369_5 |

Orchestration revision: `6993666a2a8305072dbb3024fcec76100fa943ec`.
Keep that existing orchestration checkout if repeating the idempotent command;
do not submit an additional chain from a different controller revision.
These receipts establish submission, not running status or completed GPU updates.

Later the six-condition smoke submission was rejected by the submit-job quota.
The user explicitly requested withdrawing these six continuations to free slots
for smoke tests. The returned script output confirms all six IDs left the queue
before smoke array **166370** was submitted successfully. These continuations
are cancelled. Their old `jobs.json` entries are historical submissions,
not proof of active continuations;
do not use the original idempotent submit command to claim they were resubmitted.

The six continuations retain the frozen training revision
`c193e617acf03d94e18cad023934e65e64ff6923`, identical logical runs, manifests,
model, optimizer and OMR checkpoint state. The newer checkout provides only
submission and checkpoint gating. It does not migrate training code or alter
the protocol signature of an existing checkpoint.

Run from the published orchestration checkout on the login node:

```bash
python3 -I -S -B "$CONTROL/experiments/omr/resume_chain.py" submit \
  --root "$ROOT" --parent-array 166362 \
  --training-revision c193e617acf03d94e18cad023934e65e64ff6923
```

Each singleton child array retains its original index. Child 0 depends on
`afterany:166362_0`, child 1 on `afterany:166362_1`, and so on. Each requests
two H200, 32 CPUs, 256 GiB and 18 hours on testqueue. A child can run after its
own parent terminates, including TIMEOUT/FAILED, without waiting for the other
five. There is exactly one continuation per condition, not an unlimited retry
chain. The parent's actual start time and queue time do not need to be known.
See [Slurm array dependencies](https://slurm.schedmd.com/job_array.html) and
[afterany](https://slurm.schedmd.com/sbatch.html#OPT_dependency).

The submitter verifies the parent array command, working directory and six
indices, and records child IDs under
`analysis/beacon-omr/resume-chains/166362-<training-revision>/jobs.json`.
Repeating the same successful command reprints the recorded IDs rather than
submitting six more jobs. An ambiguous sbatch outcome blocks a blind retry.
Submission receipts do not establish that any training ran.

At execution, the child checks the authoritative complete latest checkpoint on
its GPU allocation. Missing/incompatible/incomplete checkpoints stop the child;
it does not start another experiment from update zero. The frozen trainer then
performs its full checkpoint hash and payload checks before loading the state.
If 150 updates and the final evaluation already completed, the child exits
without loading the model. If only training completed, final evaluation still
runs. New receipt files are in each logical run's `resume-guards/`; logs are
`logs/beacon-omr/resume-<child-array>_<index>.out`.

## SwanLab and the learning curve

The frozen trainer already restores monitoring before creating the logger:

- If the last logged update does not exceed the checkpoint, it uses the saved
  cloud run ID with `resume=must` and continues its monotonic upload sequence.
- If execution logged beyond the saved checkpoint, those later updates have
  been rolled back. A new cloud run is created (`resume=never`, no old ID), with
  `resume_parent_id`, `resume_from_update`, checkpoint and previous attempt saved
  in `attempts/<job>/resume-lineage.json`.
- Both branches retain the actual training update in `axis/policy_update` and
  the durable local metrics. A new run's cloud upload sequence starts again at
  zero, but the policy update does not. When joining attempts for analysis,
  discard the parent's updates beyond the child's checkpoint boundary; do not
  count the discarded suffix twice or rename its steps.

This uses the SDK's documented [resume behavior](https://docs.swanlab.cn/guide_cloud/experiment_track/resume-experiment.html).
It does not treat cloud initialization as verified server readback. An online
initialization failure is reported rather than pretending it uploaded; its
returned error determines whether a separately identified new cloud run is
needed. No logging API keys are stored in submission metadata.

Local fixtures cover all six individual dependencies, repeat-submission
idempotence, uncertain submission handling, wrong parent/source rejection,
missing/damaged checkpoints, completed-run skipping, outstanding final
evaluation, and SwanLab sequence/rollback behavior. Real Slurm IDs and resumed
GPU updates remain evidence to return from the HPC terminal. The Slurm IDs above
have now been returned; resumed execution has not.
