"""Commit a complete generation before removing the previous model checkpoint.

The writer must wait for every FSDP rank and pass max_ckpt_to_keep=None to the
native save operation. This module owns rotation, never the rank-level writer.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import uuid


def fsync_directory(path):
    if os.name == "nt":
        return  # Windows cannot open directories with the POSIX API.
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class LatestCheckpoint:
    def __init__(self, root, logical_run_id):
        supplied = Path(root)
        if supplied.is_symlink():
            raise ValueError("Checkpoint root cannot be a symlink")
        self.root = supplied.resolve()
        self.run_id = logical_run_id
        self.root.mkdir(parents=True, exist_ok=True)
        with self.lock():
            owner = self.root / "OWNER.json"
            if owner.exists():
                if json.loads(owner.read_text())["logical_run_id"] != self.run_id:
                    raise ValueError("Checkpoint root belongs to another logical run")
            elif any(p.name != ".save.lock" for p in self.root.iterdir()):
                raise ValueError("Cannot adopt a populated checkpoint directory")
            else:
                atomic_json(owner, {"logical_run_id": self.run_id, "schema": 1})

    @contextmanager
    def lock(self):
        path = self.root / ".save.lock"
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            os.write(fd, str(os.getpid()).encode())
            os.fsync(fd)
            yield
        finally:
            os.close(fd)
            path.unlink()

    def _owned_child(self, name):
        if Path(name).name != name or name in {".", ".."}:
            raise ValueError("Invalid checkpoint generation name")
        child = self.root / name
        if child.is_symlink() or child.resolve().parent != self.root:
            raise ValueError("Checkpoint path escapes this run")
        return child

    def verify(self, generation, verify_hashes=True):
        path = self._owned_child(generation)
        manifest = json.loads((path / "COMPLETE.json").read_text())
        if manifest["logical_run_id"] != self.run_id:
            raise ValueError("Checkpoint ownership mismatch")
        for name, info in manifest["files"].items():
            item = path / name
            if item.is_symlink() or not item.resolve().is_relative_to(path):
                raise ValueError("Checkpoint file escapes generation")
            if item.stat().st_size != info["bytes"] or (verify_hashes and file_sha256(item) != info["sha256"]):
                raise ValueError("Incomplete or corrupt checkpoint file: " + name)
        return path, manifest

    def latest(self, verify_hashes=True):
        pointer = json.loads((self.root / "latest.json").read_text())
        path, manifest = self.verify(pointer["generation"], verify_hashes)
        if pointer["update"] != manifest["update"] or pointer["complete_sha256"] != file_sha256(path / "COMPLETE.json"):
            raise ValueError("Authoritative pointer disagrees with generation manifest")
        return path, manifest

    def save(self, update, writer, validator, metadata, required_free_bytes=0):
        if update < 0:
            raise ValueError("Negative policy update")
        with self.lock():
            if (self.root / "latest.json").exists():
                _, old = self.latest()
                if update <= old["update"]:
                    raise ValueError("Refusing to overwrite/reverse a committed generation")
            if shutil.disk_usage(self.root).free < required_free_bytes:
                raise OSError("Insufficient space for new staging; previous latest preserved")
            staging = self._owned_child(".staging_" + str(update) + "_" + uuid.uuid4().hex)
            staging.mkdir()
            atomic_json(staging / "OWNER.json", {"logical_run_id": self.run_id})
            writer(staging)  # An exception leaves old latest untouched; staging is recoverable.
            validator(staging)
            files = {}
            for file in sorted(staging.rglob("*")):
                if file.is_symlink():
                    raise ValueError("Checkpoint generation may not contain symlinks")
                if file.is_file():
                    with file.open("r+b") as stream:
                        os.fsync(stream.fileno())
                    files[file.relative_to(staging).as_posix()] = {"bytes": file.stat().st_size, "sha256": file_sha256(file)}
            manifest = {"schema": 1, "logical_run_id": self.run_id, "update": update,
                        "files": files, "metadata": metadata}
            atomic_json(staging / "COMPLETE.json", manifest)
            for directory in sorted((p for p in staging.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
                fsync_directory(directory)
            fsync_directory(staging)
            generation = self._owned_child(f"global_step_{update:06d}")
            if generation.exists():
                raise FileExistsError(generation)
            os.replace(staging, generation)
            fsync_directory(self.root)
            # Only latest.json is authoritative, including after a crash between pointers.
            iteration = self.root / "latest_checkpointed_iteration.txt"
            temp = iteration.with_suffix(".tmp")
            with temp.open("w") as stream:
                stream.write(str(update))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp, iteration)
            atomic_json(self.root / "latest.json", {"generation": generation.name, "update": update,
                        "complete_sha256": file_sha256(generation / "COMPLETE.json")})
            self._cleanup(generation.name)
            return generation

    def _cleanup(self, keep):
        # Delete only owned direct-child generation directories under this run.
        for candidate in self.root.iterdir():
            if candidate.name == keep or not re.fullmatch(r"global_step_\d+|\.staging_\d+_[0-9a-f]+", candidate.name):
                continue
            candidate = self._owned_child(candidate.name)
            owner = candidate / "OWNER.json"
            if candidate.is_dir() and owner.is_file() and json.loads(owner.read_text())["logical_run_id"] == self.run_id:
                shutil.rmtree(candidate)
        fsync_directory(self.root)

    def recover(self):
        with self.lock():
            path, manifest = self.latest()
            self._cleanup(path.name)
            return path, manifest
