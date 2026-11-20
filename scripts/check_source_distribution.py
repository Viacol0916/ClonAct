"""Audit the explicit future source allowlist; never copy, publish, or initialize Git."""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs/maintenance/public-source-files.json"
FORBIDDEN_PARTS = {".git", "archive", "experiments", "paper", "results", "node_modules", "workspaces", "__pycache__", "gold", "swe_patches"}


def check(*, strict_git: bool = False):
    payload = json.loads(MANIFEST.read_text())
    names = payload["files"]
    if names != sorted(set(names)):
        raise ValueError("Source manifest must contain unique, sorted paths")
    records = []
    for name in names:
        rel = PurePosixPath(name)
        if rel.is_absolute() or ".." in rel.parts or FORBIDDEN_PARTS.intersection(rel.parts):
            raise ValueError(f"Forbidden source path: {name}")
        if rel.suffix in {".jsonl", ".patch", ".pyc", ".bundle", ".db", ".sqlite"} or any(p.startswith('.venv') for p in rel.parts):
            raise ValueError(f"Forbidden distribution artifact: {name}")
        if rel.name.startswith('.env') and rel.name != '.env.example':
            raise ValueError(f"Credential file: {name}")
        path = ROOT / name
        if any(p.is_symlink() for p in [path, *path.parents] if p != ROOT.parent):
            raise ValueError(f"Symlink in source path: {name}")
        if not path.is_file() or not path.resolve().is_relative_to(ROOT.resolve()):
            raise ValueError(f"Missing or external source: {name}")
        records.append((name, hashlib.sha256(path.read_bytes()).hexdigest()))
    required = {"LICENSE", "THIRD_PARTY_NOTICES.md", "README.md", "README_zh.md",
                "dsh_runner/synaptic_worker_plugin/index.mjs",
                "synaptic_bench_150/data/dataset_lock.json",
                "appworld_micro_30/data/micro30_recipe_manifest.json",
                "synaptic_stream_60/data/manifest_60.json"}
    if not required.issubset(names):
        raise ValueError("Missing license, README, Worker bridge, or reconstruction manifest")
    # In a standalone source-only checkout (no private `paper/` or `experiments/` dir), or when
    # `--strict-git` is passed, enforce exact equality between `git ls-files` and the allowlist.
    is_source_only_repo = (ROOT / ".git").exists() and not (ROOT / "paper").exists() and not (ROOT / "archive").exists()
    if (strict_git or is_source_only_repo) and (ROOT / ".git").exists():
        tracked = sorted(
            subprocess.check_output(["git", "-C", str(ROOT), "ls-files"], text=True).splitlines()
        )
        extra = sorted(set(tracked) - set(names))
        missing = sorted(set(names) - set(tracked))
        if extra or missing:
            raise ValueError(f"Git index diverges from allowlist: extra={extra}, missing={missing}")
    digest = hashlib.sha256(json.dumps(records, separators=(",", ":")).encode()).hexdigest()
    print(f"PASS: {len(records)} explicitly listed source files; manifest content SHA-256 {digest}")
    print("No files copied, Git history imported, repository created, or publication performed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict-git", action="store_true", help="Require git ls-files to exactly match allowlist")
    check(strict_git=parser.parse_args().strict_git)
