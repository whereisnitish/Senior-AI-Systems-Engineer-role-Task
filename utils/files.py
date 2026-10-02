"""Reading .env and input files, writing results."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


class InputError(Exception):
    """The input file can't be read or parsed at all (individual bad records are not errors)."""


def load_input(path: Path) -> list:
    try:
        raw = path.read_bytes()
    except OSError as e:
        raise InputError(f"cannot read {path}: {e}") from e
    text = raw.decode("utf-8-sig", errors="replace")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # tolerate JSON Lines
        data, bad = [], 0
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1
                data.append({"text": None, "_parse_error": line[:200]})
        if not data:
            raise InputError(f"{path} is neither JSON nor JSON Lines")
        print(f"warning: input parsed as JSON Lines ({bad} unparseable lines kept as empty records)", file=sys.stderr)
    if isinstance(data, dict):
        data = data.get("messages") if isinstance(data.get("messages"), list) else [data]
    if not isinstance(data, list):
        data = [data]
    return data


def load_env(path: Path):
    """Minimal .env reader (tolerates spaces around '=', quotes, comments, BOM)."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip().removeprefix("export ").strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def write_results(path: Path, results: list):
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=list), encoding="utf-8")
