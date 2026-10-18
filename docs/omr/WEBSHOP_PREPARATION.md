# Targeted WebShop preparation

This follows returned preflight job 165837. It does not launch any formal run or
recheck AppWorld. `submit_frozen_webshop.sh` takes the assigned root, overflow root,
the returned preflight report, and a full published Git commit. It fetches a
separate detached worktree, preserving local modifications in the original HPC
checkout, and submits one H200 / 8 CPU / 48 GiB / two-hour `testqueue` job.

## Reuse and dependencies

`prepare_webshop.py` selects the passing ScienceWorld runtime from the report:
Torch 2.6.0, vLLM 0.8.2, Transformers 4.51.1, Python 3.12. It creates an owned
`envs/beacon-omr-web-v1` venv. Its `.pth` file adds the existing package directory
as a plain path, without executing that directory's editable-install hooks. New
packages install only into the new venv. All currently installed distributions
are constrained to their existing versions; a dry-run install plan must contain
only new packages. Attempts to replace existing packages or download training/CUDA
libraries stop preparation. `pip check` and the resolved install manifest are saved.

The selected source is supplied per job through `OMR_SOURCE_ROOT`. Ordinary Python
children and a private Ray worker must report the dedicated venv and the selected
BEACON module paths. There is no `pip install -e` into an old environment. Do not
remove the reused ScienceWorld installation while this runtime depends on it.

Native Gym 0.24.0, Pyserini 0.17.0 and Flask/Werkzeug 2.1 are preserved. spaCy 3.7.5
is compatible with the pinned English 3.7.1 model; Selenium 4.27.1 supplies the
vendored browser-module imports without the old urllib3 1.x restriction. This
avoids replacing existing helpers merely to satisfy spaCy 3.7.2's older Typer
ceiling or Selenium 4.2.0's urllib3 ceiling. No browser is launched. The resulting
runtime is shared by this environment's baseline/full pair after acceptance.

## Assets and provenance

`asset_lock.json` pins public HF revisions and content identities:

- WebShop mirror: [HongbangYuan/webshop](https://huggingface.co/datasets/HongbangYuan/webshop/tree/0129d4a81dbdb827e76afd20a1e2c38b61098613).
  Only `items_shuffle_1000.json`, `items_ins_v2_1000.json`, and `items_human_ins.json`
  are downloaded (about 9.8 MB total). This matches the frozen `use_small=true`
  profile; the full 5.5 GB catalog is not required. This is a pinned mirror, not
  a claim that we possess an independently verified publisher checksum.
- English NLP model: [spacy/en_core_web_sm](https://huggingface.co/spacy/en_core_web_sm/tree/8fb6f8155286360bf5486fe399e2738d1df7ded2).
  Download the 3.7.1 wheel only (about 12.8 MB), verify its SHA256, and give its
  unchanged bytes a valid versioned wheel filename for pip.
- Existing policy model: [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/tree/989aa7980e4cf806f80c7fef2b1adb7bc71aa306).
  Verify the local weights, config and tokenizer against this revision; no policy
  model download or duplicate copy is requested.

HF's Git blob SHA1 is checked for ordinary Git files and LFS SHA256 for large
files. Existing matching assets are reused; a mismatch stops instead of silently
replacing them. The installed Hub 0.36.2 `hf download` entry point uses its actual
`--repo-type` option. Only this child receives HF_TOKEN, parsed as a literal from
the configured credential file. Other keys are not read, emitted or persisted.

Data goes under the main root first, then the supplied overflow root only if
asset space is insufficient. Package installs require at least 5 GiB filesystem
free space in the main root. These checks do not establish user/project quota.

Only the selected catalog is converted, with the same native `load_products`
and document text fields as BEACON's converter. Pyserini builds one index in an
owned pending directory. Its data/converter identity is committed inside that
directory before rename. Every catalog ID and the Lucene document count are checked.
No old index is overwritten. Training configuration can now pass explicit shared
`file_path`, `attr_path`, `human_attr_path` and `index_path` overrides.

## Native semantics and acceptance

The text environment now preserves the purchased product/options, score and
verbose receipt until explicit reset. Previously, automatic reset replaced the
session before the worker's milestone detector could read it. Explicit repeated
task resets discard the earlier mutable session. Reward conversion is unchanged:
native terminal score 1 gives the wrapper reward 10 and strict `won=true`.
The generic Gym constructor checker is disabled because it can execute unrecorded
reset/step calls on a simulator with text actions. Actual simulator tests remain
mandatory.

Two independently constructed native workers use the same goal-manifest seed.
Each executes reset, dataset-query search, product click, purchase, then repeated
explicit reset. The check requires matching goals/manifests/transitions, intact
terminal evidence and clean reset. Test-query actions never enter training data
or archives. This checks terminal semantics, not successful learned shopping.

Further bounded checks in the same job:

- Source provenance in the driver and its private Ray worker.
- Existing Qwen weights loaded with BF16/FlashAttention on the allocated H200 and
  at most eight generated tokens. No optimizer update.
- Real SwanLab offline init/log/explicit success and failure shutdown. Online
  upload/readback remains unverified. SDK argument names follow its installed
  public signature instead of relying on deprecated aliases.

Raw logs and full `report.json` stay in the private evidence directory. A compact
summary is printed to the Slurm log; `site.json` is emitted only when all preparation
checks pass. `training_ready` stays false until integrated OMR, two-GPU updates,
full resume and online monitoring are accepted. The assistant cannot run this
job remotely: the user submits it through their VS Code Tunnel session.
