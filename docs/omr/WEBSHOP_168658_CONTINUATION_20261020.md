# WebShop 168658: one requested continuation

The user requested one h200p/48-hour child of currently running WebShop job
168658. Keep pending other-arm job 168659, every ALFWorld/ScienceWorld job,
and all frozen training checkouts unchanged.

`submit_webshop_168658_resume.py submit --root ROOT --controller-revision SHA
--apply` reads the existing fresh-start submission receipt. It resolves the
parent's baseline/full identity, requires the original training source
df98029d703d64a1931dcebf0f626fb1e468af9b and input-v2-fresh-20261020 tag, and
submits exactly one normal job with afterany:168658, h200p, 48 hours, two H200,
32 CPUs and 256 GiB. The receipt avoids duplicate submissions on a retry.

The child reads the latest complete checkpoint at execution time. Missing,
incomplete or wrong-source state blocks training rather than starting at zero.
Completed update 150 plus final evaluation makes the child a successful no-op.
The frozen trainer restores optimizer, scheduler, extra state, dataloader,
protocol state and RNG, and owns the USR1 stop/checkpoint behavior.

SwanLab uses that frozen trainer's existing policy: continue the persisted run
ID and log sequence when continuous; if cloud logging advanced beyond the
checkpoint, create a new run with local resume-parent/from-update lineage.
This release does not patch the SDK, frozen training source, credentials or
previous monitoring status. General network/SDK initialization failures are
not automatically masked by this controller and need the actual error log.

Local validation: six isolated controller tests, five monitoring tests, six
existing resume-chain tests; 17 passed. Slurm and SwanLab transports are test
doubles. Actual submission, checkpoint load and cloud resumption require HPC
receipts/logs and are not established by local tests.
