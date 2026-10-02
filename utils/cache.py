"""On-disk response cache keyed by content."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


class Cache:
    """Content-addressed cache so reruns don't spend quota and duplicate texts are classified once."""

    def __init__(self, path: Path | None):
        self.path = path
        if path:
            path.mkdir(parents=True, exist_ok=True)

    def key(self, *parts) -> str:
        return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:32]

    def get(self, k):
        if not self.path:
            return None
        f = self.path / f"{k}.json"
        try:
            return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None
        except (OSError, json.JSONDecodeError):
            return None

    def put(self, k, value):
        if self.path:
            try:
                (self.path / f"{k}.json").write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
            except OSError:
                pass
