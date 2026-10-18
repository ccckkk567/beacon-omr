"""Content-addressed immutable restore records; no world objects are pickled."""
from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
import uuid
from .types import content_id


class PayloadStore:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, ref):
        if len(ref) != 64 or any(c not in "0123456789abcdef" for c in ref):
            raise ValueError("Invalid content-addressed reference")
        path = self.root / (ref + ".json.gz")
        if path.is_symlink():
            raise ValueError("Payload symlinks are not supported")
        return path

    def put(self, payload, dependencies=()):
        for ref in dependencies:
            self.get(ref)
        record = {"schema": 1, "payload": payload, "dependencies": sorted(set(dependencies))}
        ref = content_id(record)
        path = self._path(ref)
        if not path.exists():
            temporary = path.with_suffix(".tmp-" + uuid.uuid4().hex)
            try:
                with temporary.open("xb") as stream:
                    stream.write(gzip.compress(json.dumps(record, sort_keys=True, ensure_ascii=False, allow_nan=False).encode(), mtime=0))
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, path)
                if os.name != "nt":
                    fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
                    try: os.fsync(fd)
                    finally: os.close(fd)
            finally:
                temporary.unlink(missing_ok=True)
        return ref

    def get(self, ref):
        record = json.loads(gzip.decompress(self._path(ref).read_bytes()))
        if content_id(record) != ref:
            raise ValueError("Restore payload checksum mismatch")
        return record

    def dependency_closure(self, roots):
        protected, pending = set(), list(roots)
        while pending:
            ref = pending.pop()
            if ref not in protected:
                protected.add(ref)
                pending.extend(self.get(ref)["dependencies"])
        return protected

    def garbage_collect(self, active_refs, latest_refs, staging_refs=()):
        # Call only at the sole-writer update boundary; all three roots are mandatory.
        protected = self.dependency_closure(set(active_refs) | set(latest_refs) | set(staging_refs))
        deleted = []
        for path in self.root.glob("*.json.gz"):
            ref = path.name.removesuffix(".json.gz")
            if ref not in protected:
                self._path(ref).unlink()
                deleted.append(ref)
        return sorted(deleted)
