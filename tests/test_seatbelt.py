"""Real kernel probes: protect source Git objects while preserving task Git workflows."""
import os
from pathlib import Path
import subprocess
import sys

import pytest

from dsh_runner.seatbelt import build_seatbelt_profile, repository_git_paths


def git(repo, *args):
    return subprocess.run(['git', '-C', str(repo), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def seed_repository(path):
    path.mkdir()
    git(path, 'init', '-b', 'main')
    git(path, 'config', 'user.email', 'test@local')
    git(path, 'config', 'user.name', 'Test')
    (path / 'protected.txt').write_text('synthetic-evaluator\n')
    git(path, 'add', 'protected.txt')
    git(path, 'commit', '-m', 'synthetic seed')


@pytest.mark.parametrize('layout', ['regular', 'linked', 'shared'])
def test_git_metadata_discovery_ignores_episode_overrides(tmp_path, monkeypatch, layout):
    origin = tmp_path / 'origin'
    seed_repository(origin)
    source = origin
    if layout == 'linked':
        source = tmp_path / 'source'
        git(origin, 'worktree', 'add', '-b', 'source', str(source))
    elif layout == 'shared':
        source = tmp_path / 'source'
        git(tmp_path, 'clone', '--shared', str(origin), str(source))
    monkeypatch.setenv('GIT_DIR', str(tmp_path / 'wrong'))
    monkeypatch.setenv('GIT_COMMON_DIR', str(tmp_path / 'wrong-common'))
    protected = repository_git_paths(source)
    assert (source / '.git').resolve() in protected
    assert (origin / '.git/objects').resolve() in protected


@pytest.mark.skipif(sys.platform != 'darwin' or not Path('/usr/bin/sandbox-exec').exists(),
                    reason='Requires macOS Seatbelt')
@pytest.mark.parametrize('layout', ['regular', 'linked', 'shared'])
def test_real_seatbelt_denies_git_objects_but_allows_task_worktrees(tmp_path, layout):
    origin = tmp_path / 'origin'
    seed_repository(origin)
    source = origin
    if layout == 'linked':
        source = tmp_path / 'source'
        git(origin, 'worktree', 'add', '-b', 'source', str(source))
    elif layout == 'shared':
        source = tmp_path / 'source'
        git(tmp_path, 'clone', '--shared', str(origin), str(source))
    sha = git(source, 'rev-parse', 'HEAD:protected.txt')
    # Exercise both loose and packed object access, including explicit --git-dir.
    git(origin, 'gc', '--prune=now')
    work = source / 'workspaces' / 'episode'
    work.mkdir(parents=True)
    task = work / 'task'
    seed_repository(task)
    task_worktree = work / 'cell'
    profile = tmp_path / 'profile.sb'
    # Deliberately broad write allowance verifies Git denial wins over ancestors.
    profile.write_text(build_seatbelt_profile(source, [tmp_path]))
    env = {k: v for k, v in os.environ.items() if not k.startswith('GIT_')}
    env.update(GIT_CONFIG_GLOBAL='/dev/null', GIT_CONFIG_NOSYSTEM='1')

    def run(*command, ok=True):
        result = subprocess.run(['/usr/bin/sandbox-exec', '-f', str(profile), *command],
                                env=env, capture_output=True, text=True)
        assert (result.returncode == 0) == ok, (command, result.returncode, result.stderr)
        return result

    run('/usr/bin/git', '-C', str(source), 'show', 'HEAD:protected.txt', ok=False)
    run('/usr/bin/git', '--git-dir='+str(origin / '.git'), 'cat-file', '-p', sha, ok=False)
    # A separately initialized Git repository must remain fully usable, even
    # inside the source checkout's workspaces/ directory.
    run('/usr/bin/git', '-C', str(task), 'status', '--porcelain')
    run('/usr/bin/git', '-C', str(task), 'worktree', 'add', '-b', 'cell', str(task_worktree))
    run('/bin/sh', '-c', 'printf "task-change\\n" > "$1/protected.txt"', 'probe', str(task_worktree))
    run('/usr/bin/git', '-C', str(task_worktree), 'add', 'protected.txt')
    run('/usr/bin/git', '-C', str(task_worktree), 'commit', '-m', 'task change')
    run('/usr/bin/git', '-C', str(task), 'merge', '--ff-only', 'cell')
    assert run('/usr/bin/git', '-C', str(task), 'show', 'HEAD:protected.txt').stdout == 'task-change\n'
    assert run('/usr/bin/git', '-C', str(task), 'status', '--porcelain').stdout == ''
