"""Seatbelt policy shared by the Worker and official comparison baseline."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess


def _escape(path: Path) -> str:
    return str(path.resolve()).replace('\\', '\\\\').replace('"', '\\"')


def repository_git_paths(project_root: Path) -> list[Path]:
    """Resolve only the protected source repository's metadata, including linked worktrees.

    Task repositories are intentionally not enumerated or denied. Ignore inherited
    Git overrides so an episode's GIT_DIR cannot redirect this security check.
    """
    root = project_root.resolve()
    marker = root / '.git'
    env = {k: v for k, v in os.environ.items() if not k.startswith('GIT_')}
    result = subprocess.run(
        ['git', '-C', str(root), 'rev-parse', '--path-format=absolute', '--git-dir', '--git-common-dir'],
        env=env, capture_output=True, text=True, timeout=15,
    )
    if result.returncode:
        if marker.exists():
            raise RuntimeError('Cannot resolve protected source repository Git metadata')
        # An unpacked source archive has no Git object database to protect.
        return []
    paths = result.stdout.splitlines()
    if len(paths) != 2 or not all(Path(p).is_absolute() and Path(p).is_dir() for p in paths):
        raise RuntimeError('Invalid protected source repository Git metadata paths')
    protected = {Path(p).resolve() for p in paths}
    if marker.exists():
        protected.add(marker.resolve())
    # Shared clones may store their blobs outside the common directory.
    pending = [Path(p).resolve() / 'objects' for p in paths]
    seen = set()
    while pending:
        objects = pending.pop().resolve()
        if objects in seen:
            continue
        seen.add(objects)
        protected.add(objects)
        alternates = objects / 'info/alternates'
        if alternates.is_file():
            for line in alternates.read_text().splitlines():
                if not line:
                    continue
                if line.startswith('"'):
                    raise RuntimeError('Quoted Git alternates require explicit isolation configuration')
                alternate = Path(line)
                pending.append(alternate if alternate.is_absolute() else objects / alternate)
    return sorted(protected, key=str)


def seatbelt_ancestor_metadata(paths: list[Path]) -> str:
    """Allow traversal to writable subtrees without granting ancestor contents."""
    ancestors = sorted({parent for path in paths for parent in path.resolve().parents}, key=str)
    return '(allow file-read-metadata\n' + '\n'.join(
        f'  (literal "{_escape(p)}")' for p in ancestors
    ) + '\n)'


def build_seatbelt_profile(project_root: Path, writable_roots: list[Path]) -> str:
    root = project_root.resolve()
    deny_dirs = [root / name for name in (
        'synaptic_bench_150', 'appworld_micro_30', 'synaptic_stream_60', 'experiments', 'clonact',
        'tests', 'workspaces', 'data', '01_problem_discovery',
        '02_architecture_design', '04_experiments', 'paper', 'archive',
    )]
    third_party = root / 'third_party'
    if third_party.is_dir():
        deny_dirs += [p for p in sorted(third_party.iterdir())
                      if p.is_dir() and p.name != 'dsh-superagent-prototype']

    def clause(operation: str, paths: list[Path]) -> str:
        return f'({operation}\n' + '\n'.join(f'  (subpath "{_escape(p)}")' for p in paths) + '\n)'

    lines = ['(version 1)', '(allow default)', clause('deny file-read* file-write*', deny_dirs),
             seatbelt_ancestor_metadata(writable_roots)]
    caches = [root / 'data/upstream_pillars' / name for name in ('venv_cache', 'repo_cache')]
    caches = [p for p in caches if p.exists()]
    if caches:
        lines.append(clause('allow file-read*', caches))
    lines.append(clause('allow file-read* file-write*', writable_roots))
    # Keep this after workspace exceptions: even a broadly allowed ancestor must
    # not expose evaluator blobs or research history via git show/cat-file.
    git_paths = repository_git_paths(root)
    if git_paths:
        lines.append(clause('deny file-read* file-write*', git_paths))
    return '\n'.join(lines) + '\n'
