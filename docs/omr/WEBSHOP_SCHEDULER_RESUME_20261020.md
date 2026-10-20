# WebShop resume configuration correction — 2026-10-20

## Observed failure

User-returned logs for baseline168005 and Full168006 (training2ebc0a7) fail in
`webshop_migration.authorize -> check_invariants`, before loading saved actor
weights/optimizer state or advancing any new policy update:

```text
ValueError: Unauthorized scientific configuration difference: actor_rollout_ref, critic
```

Both logs show model initialization and CUDA graph capture completed before the
check. These are not checkpoint restore acceptance. The final accounting returned
by the user is FAILED/1:0 for both. Latest pointers and COMPLETE hashes still match:

| Arm | Latest update | COMPLETE SHA-256 |
|---|---:|---|
| Baseline | 35 | 5ece432fb20374212e32b06b63b2da6ce3d155e21feacbfff2a6dc7e0046c303 |
| Full | 10 | e131f52b920a7133a0188087d2599ee252088f9abc9dfb1f6580b3fb5b9e53a6 |

Baseline additionally reports `OSError: [Errno 77] File descriptor in bad state`
while unwinding the outer run-lock file context after `SystemExit(1)`. It is a
secondary cleanup error; its filesystem-level cause is not established or fixed
by this patch. Retrying must acquire the existing exclusive run lock normally.

## Reproduced cause and bounded fix

The source composer returns `-1` for both
`actor_rollout_ref.actor.optim.total_training_steps` and
`critic.optim.total_training_steps`. The existing native trainer's
`_create_dataloader` sets both to the configured total150, including the unused
critic field. Migration compared freshly composed old config with initialized
new config, rather than comparing their effective schedules.

The regression executes the actual initialization method extracted from the
current and each frozen source revision. Only dataset IO is replaced by a small
CPU fixture. Before the fix it produces the exact same two-section error.

Comparison now materializes these two derived values on copies, only if each is
exactly the sentinel integer-1 or the explicit positive configured training
total. Other values are rejected; the trainer total itself stays in the compared
configuration. Learning rate, optimizer parameters, B/G/H, model, task schedule,
rewards, archive and engine settings remain protected. Unexpected differences
report full leaf paths instead of only the top-level actor/critic sections.
Migration receipts record the two derived fields and effective training total.
The learner configuration and checkpoint contents are not rewritten.

Thirteen focused tests pass: actual frozen-source composition/initialization,
production migration authorization, rejected scientific changes, original and
retry submission guards, and partial quota rejection without duplicate jobs.
This is local CPU evidence, not HPC resume/native-input/GPU acceptance.

## Execution

`retry_webshop_config.py` is a separate, journaled retry controller. It requires
the original recovery receipt to bind exact failed jobs168005/168006 to2ebc0a7,
accounting FAILED/1:0, no live old writer, the exact failed attempts, failed
allocation receipts, and unchanged original checkpoint metadata/file sizes.
An attempt with a migration permit or restored-state receipt is preserved for
inspection instead of automatically replaced.

It creates two ordinary named formal jobs on testqueue,18h,2H200,32CPU,256GiB,
without cancellation, arrays, smoke jobs or automatic continuation children.
Original recovery requests/receipts remain under `webshop-input-20261020`; retry
receipts and requests use `webshop-scheduler-20261020`. Accepted submissions are
retained on retries, including when their training subsequently advances.

The new training revision still uses the original narrow migration from source
98d7a69/update35 and ddae412/update10. Full hashes, native failed-input checks,
optimizer/cursor/RNG/archive restoration and SwanLab lineage are performed by
the existing formal GPU path. The next policy updates must be36 and11.
No actual new submission or GPU acceptance has yet been returned.
