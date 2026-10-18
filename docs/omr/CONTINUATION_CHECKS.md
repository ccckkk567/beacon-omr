# WebShop continuation and SwanLab online acceptance

Preparation job 165873 at `b019cfc` passed native interaction, Ray provenance,
existing model weight loading and offline monitoring. Reuse its private
`report.json` and adjacent `site.json`. No additional installation or download is
part of this check. Submit `submit_frozen_continuation.sh ROOT REPORT FULL_COMMIT`
from the login node; actual work runs on one allocated H200, eight CPUs and 48 GiB
with a 40-minute Slurm limit. The original dirty checkout is preserved.

## Native replay contract

`webshop/restart.py` binds explicit tasks to canonical goal contents and a profile
covering loaded products/prices, index files/provenance, native code/templates and observation settings.
The manifest preserves BEACON's actual test [0,500) / train [500,N) split and rejects
duplicate content identities; it does not filter or reshuffle the task pool.

Capture stores executed commands, transition signatures, the root's simulator RNG,
final session/browser/detector state and complete manager history as JSON. It does
not copy a server, Lucene handle or actor. Replay resets the exact goal, executes
each original command through the native worker and checks every receipt. It
checks HTML/URL, query/page/product/options, native history, detector phase and
matched options. Python randomness used by native random search is isolated from
other sampling streams and replayed. A mismatch raises `RestoreMismatch`.

Manager restoration checks task text, current observation, history profile and
the exact rendered next policy input. The normal baseline renderer is reused.
Restored milestone flags are cleared, while detector progress is retained. Replay
does not create policy samples, new-action rewards or q/D outcomes; its action
count and duration are returned separately. No simulator clock is reset to extend
a rollout. WebShop has no additional native step limit in this profile.

The GPU fixture uses two independent native servers, captures search and detail
states (including a product option when available), restores the same inputs,
compares a new continuation, restores a recursive prefix and executes 15 further
nonterminal actions. It also rejects corrupted payloads and terminal purchases.
Its dataset-query actions are test-only and never enter an archive or policy loss.

This is a **state restoration API**, not completed OMR training. Candidate progress
eligibility, context-token admission, parent group planning, whole-group fallback,
Ray assignment wiring, collector/trainer commits, ALFWorld/ScienceWorld adapters,
full checkpoint/resume and two-GPU optimizer/weight synchronization remain to be
implemented or accepted. Capturing an arbitrary continuing fixture state does not
prove it qualifies as an archive candidate.

## Online monitoring contract

Only the online probe child receives the literal `SWANLAB_API_KEY`; the credentials
file is never executed. The driver redacts that value from captured child output.
The probe creates two small `test_only` runs in `beacon-omr-readiness`: one normal
finish and one intentional failure. It uploads two scalar values and update-axis
points, then uses SwanLab's public `Api.run(...).metrics(...)` to read them back.
The expected remote states are FINISHED and CRASHED. Empty data, incorrect steps,
wrong values or an unconfirmed final state do not pass. Readback retries are bounded,
and the child process has an eight-minute timeout. The implementation follows the
installed SwanLab 0.10.2 public SDK source; actual remote acceptance is pending.

Simulator and online probes run independently, so one failure does not erase the
other's evidence. The summary includes both results, test run URLs and a private
report path. `training_ready` remains false after this bounded check.
