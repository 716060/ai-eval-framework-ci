"""运行指纹：没有它，两次评测结果不可比较，也无法审计。"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import uuid

from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version

def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True,
                              timeout=5, check=False).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None

def _pkg(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None

def dataset_sha256(path: str) -> str:
    from pathlib import Path
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def build_manifest(*, dataset_path: str, dataset_sha: str, gates_path: str | None,
                   gates_sha: str, target, judge, deepeval_enabled: bool,
                   seed: int | None, repeats: int, workers: int,
                   temperature: float, started_at: float, finished_at: float) -> dict:
    prompt_fingerprint = hashlib.sha256(
        json.dumps({"schema": "v1", "temperature": temperature},
                   sort_keys=True).encode()).hexdigest()[:16]

    return {
        "schema_version": "2.0",
        "run_id": str(uuid.uuid4()),
        "started_at": datetime.fromtimestamp(started_at, timezone.utc).isoformat(),
        "finished_at": datetime.fromtimestamp(finished_at, timezone.utc).isoformat(),
        "duration_s": round(finished_at - started_at, 2),
        "git_sha": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "dataset": {"path": str(dataset_path), "sha256": dataset_sha},
        "gates": {"path": str(gates_path) if gates_path else "builtin", "sha256": gates_sha},
        "target": {"provider": getattr(target, "provider", None),
                   "model": getattr(target, "model_name", None)},
        "judge": ({"provider": getattr(judge, "provider", None),
                   "model": getattr(judge, "model_name", None)} if judge else None),
        "deepeval_enabled": deepeval_enabled,
        "decoding": {"temperature": temperature, "seed": seed, "repeats": repeats},
        "execution": {"workers": workers},
        "prompt_fingerprint": prompt_fingerprint,
        "env": {"python": sys.version.split()[0], "platform": platform.platform(),
                "deepeval": _pkg("deepeval"), "openai": _pkg("openai")},
    }


