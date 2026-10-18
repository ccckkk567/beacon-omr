"""Read-only site inventory. Run this on an allocated GPU node, before setup.

No package installation, environment activation, model download, or credential
values are performed/read into the report. Reports belong outside the repository.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def command(argv, timeout=45):
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return {"exit_code": result.returncode, "stdout": result.stdout[-20000:],
                "stderr": result.stderr[-4000:]}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"error": str(error)}


def bounded_files(root, max_depth=6, limit=30000):
    root = Path(root)
    if not root.is_dir():
        return
    seen = 0
    for base, dirs, files in os.walk(root, followlinks=False):
        rel = Path(base).relative_to(root)
        if len(rel.parts) >= max_depth:
            dirs[:] = []
        dirs[:] = sorted(d for d in dirs if d not in {
            ".git", "__pycache__", "site-packages", "node_modules", "blobs"})
        for name in sorted(files):
            seen += 1
            if seen > limit:
                return
            yield Path(base) / name


PROBE = r'''
import importlib.util, importlib.metadata, json, sys
report = {"executable":sys.executable,"python":sys.version,"packages":{},"imports":{}}
for package in ("torch","vllm","ray","transformers","flash-attn","tensordict",
                "torchdata","swanlab","huggingface-hub","scienceworld","alfworld",
                "textworld","gym","gymnasium","pyserini","py4j","numpy"):
    try: report["packages"][package]=importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError: report["packages"][package]=None
for name in ("verl","agent_system","scienceworld","alfworld","swanlab"):
    try:
        spec=importlib.util.find_spec(name)
        report["imports"][name]=None if spec is None else spec.origin
    except Exception as error: report["imports"][name]={"error":type(error).__name__}
try:
    import torch
    report["cuda"]={"available":torch.cuda.is_available(),"torch_cuda":torch.version.cuda,
      "count":torch.cuda.device_count(),"devices":[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}
except Exception as error: report["cuda"]={"error":str(error)}
print(json.dumps(report))
'''


def inspect_site(work_root, extra_storage):
    roots = [Path(work_root), Path(extra_storage)]
    report = {"schema": 1, "timestamp": time.time(), "job_id": os.getenv("SLURM_JOB_ID"),
              "node": os.uname().nodename, "source_commit": command(["git", "rev-parse", "HEAD"]),
              "storage": [], "environments": [], "assets": [], "quota_verified": False,
              "credentials_file_readable": os.access(roots[0] / "dep-prep-credentials.env", os.R_OK)}
    for root in roots:
        entry = {"path": str(root), "exists": root.exists(), "writable": os.access(root, os.W_OK)}
        if root.exists():
            total, used, free = shutil.disk_usage(root)
            entry.update(filesystem_total_bytes=total, filesystem_free_bytes=free,
                         note="Filesystem free space is not the user/project quota.")
        report["storage"].append(entry)
    report["gpu"] = command(["nvidia-smi", "--query-gpu=name,uuid,memory.total,driver_version", "--format=csv"])
    report["allocation"] = command(["scontrol", "show", "job", os.environ["SLURM_JOB_ID"]])
    report["java"] = command(["java", "-version"])
    report["partitions"] = command(["sinfo", "-o", "%P %l %G %c %m"])
    report["existing_jobs"] = command(["squeue", "-u", os.environ.get("USER", ""), "-o", "%.18i %.9P %.30j %.8T %.10M %.6D %R"])
    candidates = set()
    for base in (roots[0] / "envs", roots[0] / "runtimes"):
        for path in bounded_files(base, max_depth=5):
            if path.name == "python" and path.parent.name == "bin":
                candidates.add(path)
    for path in sorted(candidates):
        result = command([str(path), "-c", PROBE], timeout=90)
        report["environments"].append({"prefix": str(path.parent.parent), "probe": result,
                                       "pip_check": command([str(path), "-m", "pip", "check"], timeout=60)})
    for root in roots:
        bases = [root / name for name in ("data", "downloads", "models", ".cache/huggingface")]
        if root == roots[1]:
            bases.append(root)
        found = set()
        for base in bases:
            for path in bounded_files(base):
                if path in found:
                    continue
                found.add(path)
                if path.name in {"config.json", "tokenizer_config.json", "config.yaml", "pyvenv.cfg"} or path.suffix == ".jar" or path.name.startswith("items_"):
                    item = {"path": str(path), "bytes": path.stat().st_size}
                    if path.name == "config.json":
                        try:
                            config = json.loads(path.read_text())
                            item["model"] = {key: config.get(key) for key in ("model_type", "architectures", "hidden_size", "num_hidden_layers", "_name_or_path")}
                            item["weight_files"] = [{"name": p.name, "bytes": p.stat().st_size}
                                for p in path.parent.glob("*.safetensors")]
                        except (OSError, ValueError):
                            pass
                    report["assets"].append(item)
    report["top_level"] = {str(root): sorted(p.name for p in root.iterdir() if p.is_dir())
                           for root in roots if root.is_dir()}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--extra-storage", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not shutil.which("nvidia-smi"):
        parser.error("Inventory must run in a Slurm GPU allocation.")
    report = inspect_site(args.work_root, args.extra_storage)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("REPORT_PATH=" + str(target))


if __name__ == "__main__":
    main()
