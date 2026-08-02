#!/usr/bin/env python3
"""Build a deterministic manifest for the pedal research evidence bundle."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


SOURCE_FILES = (
    "4_eval_pedal.py",
    "6_train_mobile_pedal_regression.py",
    "collect_pedal_evidence.py",
    "diagnose_pedal_cache.py",
    "export_mobile_pedal_onnx.py",
    "ov_piano/models/mobile_pedal.py",
    "ov_piano/pedal.py",
    "PEDAL_RESEARCH_MASTER.md",
)

RESULT_ROOTS = (
    "edge-10ms",
    "edge-10ms-allocation-128x288",
    "offline-oracle",
    "offline-oracle-flatten-24ms",
    "offline-oracle-pooled-10ms",
)

PACKAGE_NAMES = ("torch", "numpy", "scipy", "h5py", "librosa", "onnx", "onnxruntime")


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def path_label(path: Path, repo: Path, workspace: Path) -> str:
    for prefix, label in ((repo, "repo"), (workspace, "workspace")):
        try:
            return f"{label}/{path.resolve().relative_to(prefix.resolve()).as_posix()}"
        except ValueError:
            pass
    return str(path.resolve())


def file_record(path: Path, repo: Path, workspace: Path) -> dict[str, object]:
    return {
        "path": path_label(path, repo, workspace),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def existing_files(paths: Iterable[Path]) -> list[Path]:
    return sorted({path.resolve() for path in paths if path.is_file()})


def git_record(repo: Path) -> dict[str, object]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=repo, check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        return result.stdout.strip()

    try:
        return {
            "commit": run("rev-parse", "HEAD"),
            "status_short": run("status", "--short").splitlines(),
        }
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"unavailable": f"{type(exc).__name__}: {exc}"}


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in PACKAGE_NAMES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def dataset_record(mel: Path | None, roll: Path | None) -> dict[str, object]:
    if mel is None or roll is None or not mel.is_file() or not roll.is_file():
        return {"available": False}
    try:
        import h5py
        import numpy as np

        with h5py.File(mel, "r") as mel_h5, h5py.File(roll, "r") as roll_h5:
            mel_idxs = mel_h5["data_idxs"][:]
            roll_idxs = roll_h5["data_idxs"][:]
            mel_metadata = mel_h5["metadata"][:]
            roll_metadata = roll_h5["metadata"][:]
            return {
                "available": True,
                "mel_path": str(mel.resolve()),
                "roll_path": str(roll.resolve()),
                "mel_shape": list(mel_h5["data"].shape),
                "roll_shape": list(roll_h5["data"].shape),
                "data_idxs_shape": list(mel_idxs.shape),
                "data_idxs_equal": bool(np.array_equal(mel_idxs, roll_idxs)),
                "metadata_equal": bool(np.array_equal(mel_metadata, roll_metadata)),
            }
    except (ImportError, OSError, KeyError, ValueError) as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_repo = Path(__file__).resolve().parent
    parser.add_argument("--repo", type=Path, default=default_repo)
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"))
    parser.add_argument(
        "--output", type=Path,
        default=default_repo / "results_eval" / "pedal-evidence-manifest.json",
    )
    parser.add_argument("--mel", type=Path)
    parser.add_argument("--roll", type=Path)
    parser.add_argument(
        "--extra", type=Path, action="append", default=[],
        help="Additional evidence file to hash; may be repeated.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = args.repo.resolve()
    workspace = args.workspace.resolve()
    output = args.output.resolve()

    result_files: list[Path] = []
    for name in RESULT_ROOTS:
        root = repo / "results_eval" / name
        if root.is_dir():
            result_files.extend(root.rglob("*.json"))

    artifact_root = workspace / "artifacts"
    artifact_files = list(artifact_root.glob("pedal*.onnx"))
    artifact_files.extend(artifact_root.glob("pedal*.onnx.json"))

    source_files = existing_files(repo / name for name in SOURCE_FILES)
    result_files = existing_files(result_files)
    artifact_files = existing_files([*artifact_files, *args.extra])

    payload = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "packages": package_versions(),
        },
        "git": git_record(repo),
        "dataset": dataset_record(args.mel, args.roll),
        "sources": [file_record(path, repo, workspace) for path in source_files],
        "results": [file_record(path, repo, workspace) for path in result_files],
        "artifacts": [file_record(path, repo, workspace) for path in artifact_files],
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(output)


if __name__ == "__main__":
    main()
