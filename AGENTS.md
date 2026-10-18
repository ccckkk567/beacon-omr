# BEACON online milestone reuse

Work only in this independent checkout. Preserve the upstream source identity
`c88af45736b0a178106bc14ea89603ff02d8b222` and the Apache-2.0 notices.

The user approved a public origin repository and local commit/push for GitHub
synchronization. Remote HPC commands are executed by the user through VS Code
Tunnel; the assistant cannot connect directly to Orion.

Current user overrides to the execution specification:
- Use SwanLab, with SWANLAB_API_KEY supplied only at runtime, instead of W&B.
- Use Hugging Face for model/dataset downloads, with HF_TOKEN supplied at runtime.
- Check and configure environments on GPU compute nodes. Pure downloads may use
  CPU jobs. While the CPU queue is unavailable, the user approved lightweight Git
  clone/check on login, followed by direct GPU inventory submission. Keep large
  asset downloads and package installation on compute nodes.
- Prefer existing compatible environments/data/models. Do not mutate environments
  used by old experiments. Use the main assigned storage first; overflow storage
  is for large data/models when needed, never an unconditional duplicate.
- Check only ALFWorld, WebShop and ScienceWorld. Do not inspect or probe AppWorld
  environments, including as a candidate dependency base. The user explicitly
  excluded those installations from further checks.
- Keep local site paths and all credentials outside Git. Do not print credentials
  or a complete environment-variable dump.
- Stop separate readiness/probe rounds. Reuse accepted evidence and advance real
  training; essential protocol checks execute inside the training job. Begin the
  three baselines while completing OMR integration. Never label baseline as full.
- Submit independent ready benchmark runs concurrently, with two H200 per run
  (six H200 for all three). Let Slurm queue for available resources; do not impose
  a one-at-a-time throttle or benchmark completion dependencies.

Frozen experiment: ALFWorld, ScienceWorld, WebShop; baseline/full; seed 0;
Qwen2.5-1.5B-Instruct; one Slurm node and two H200 per training job. Preserve the
upstream per-environment code profiles and baseline/full common fixes. A milestone
does not stop the current rollout. Only later rollouts may restart, each with H
new actions; replay is separately costed. No teacher, SFT, extra reward or ablations.

Real environment restore, two-GPU updates, monitoring and resume need evidence
from actual execution. Unit fixtures are not evidence of completed HPC training.
Checkpoint replacement must commit the new complete generation before removing
the previous one. Preserve full scientific records and active restore payloads.
