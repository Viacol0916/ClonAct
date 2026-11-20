"""DSH transport contracts with real Git state; no model inference."""
import asyncio
import subprocess
from pathlib import Path

import pytest
from fake_dsh import FakeDSHHost
from test_clonact import DeterministicStubJudge, StubLLM
from clonact.context_mmu import ContextMMU
from clonact.context_cell import ContextCell
from clonact.cell_backends import DSHAgentCellBackend
from clonact.synaptic_bus import HybridSynapticBus
from clonact.trunk import ElasticSharedTrunk


def git(repo, *args):
    return subprocess.run(['git', '-C', str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()


def make_cells(tmp_path):
    repo = tmp_path / 'repo'
    repo.mkdir()
    git(repo, 'init', '-b', 'main')
    git(repo, 'config', 'user.email', 'test@local')
    git(repo, 'config', 'user.name', 'Test')
    (repo / 'calc.py').write_text('def calc():\n    return 1\n')
    git(repo, 'add', '.')
    git(repo, 'commit', '-m', 'seed')
    host = FakeDSHHost()
    trunk = ElasticSharedTrunk()
    bus = HybridSynapticBus(DeterministicStubJudge())
    lock = asyncio.Lock()
    cells = [ContextCell(cell_id=f'Cell_{i}', topic=f'change {i}', source_kind='user_chat',
                         sandbox_paths={'repo': repo, 'world': tmp_path, 'worktrees': tmp_path / 'worktrees'},
                         trunk=trunk, bus=bus, llm=StubLLM(), dsh_host=host, trunk_merge_lock=lock)
             for i in (1, 2)]
    return repo, host, cells


def test_missing_host_and_retired_cli_names_fail_closed(tmp_path):
    from synaptic_bench_150.runner import _parse_arms_arg
    from appworld_micro_30.runner import SUPPORTED_MICRO_ARMS
    with pytest.raises(ValueError, match='DSH Worker Host'):
        ContextMMU(sandbox_paths={'repo': tmp_path})
    for name in ('synaptic_pa', 'synaptic_pa_dsh'):
        with pytest.raises(ValueError, match='Unknown arm'):
            _parse_arms_arg(name)
        assert name not in SUPPORTED_MICRO_ARMS
    assert _parse_arms_arg('clonact') == [('clonact', None)]


def test_dsh_alignment_continuity_observer_and_overturn(tmp_path):
    async def exercise():
        repo, host, (cell, observer) = make_cells(tmp_path)
        assert isinstance(cell._backend, DSHAgentCellBackend)
        await cell.align_turn('Keep the billing API unchanged.')
        assert cell.worktree_path is None
        assert not (tmp_path / 'worktrees').exists()
        assert host.sessions['ro-Cell_1']['read_only'] is True
        cell.append_followup_user_turn('Now implement the change.')
        result = await cell.run_detached_execution()
        assert result['completed_cleanly'] is True
        assert cell.worktree_path is not None
        prompt = host.messages[-1]['text']
        assert 'Keep the billing API unchanged.' in prompt
        assert 'Now implement the change.' in prompt
        assert host.sessions['worker-Cell_1']['read_only'] is False
        observer.observer_only = True
        observer.observed_peers = ['Cell_1']
        await observer.run_detached_execution()
        assert observer.worktree_path is None
        assert host.sessions['ro-Cell_2']['read_only'] is True
        cell.tool_scratchpad.append({'role': 'assistant', 'content': 'stale'})
        cell.reset_for_overturn('Cancel that change')
        assert cell.tool_scratchpad == []
        assert 'Cancel that change' in cell.intent_spine[-1]['content']
        assert cell.commit_sha is None
    asyncio.run(exercise())


@pytest.mark.parametrize('resolve', [True, False])
def test_dsh_conflict_reconciliation_preserves_trunk(tmp_path, resolve):
    async def exercise():
        repo, host, (first, second) = make_cells(tmp_path)
        wt1, wt2 = first._ensure_worktree(), second._ensure_worktree()
        (wt1 / 'calc.py').write_text('def calc():\n    return 2\n')
        (wt2 / 'calc.py').write_text('def calc():\n    return 3\n')
        await first._finalize_and_promote_async()
        assert first.merged_to_trunk
        trunk_sha = git(repo, 'rev-parse', 'HEAD')
        async def reconcile(request):
            assert request['session_id'] == 'worker-Cell_2'
            assert 'CONFLICT' in request['text']
            assert git(repo, 'rev-parse', 'HEAD') == trunk_sha
            assert git(repo, 'status', '--porcelain') == ''
            if resolve:
                (wt2 / 'calc.py').write_text('def calc():\n    return 5\n')
                git(wt2, 'add', 'calc.py')
                second.last_pytest_status = 'passed'
            return {'finalText': 'resolved' if resolve else 'unresolved'}
        host.handler = reconcile
        await second._finalize_and_promote_async()
        assert second.reconciliation_attempts == 1
        assert second.merged_to_trunk is resolve
        assert git(repo, 'status', '--porcelain') == ''
        assert (repo / 'calc.py').read_text().endswith('return 5\n' if resolve else 'return 2\n')
        if not resolve:
            assert git(repo, 'rev-parse', 'HEAD') == trunk_sha
    asyncio.run(exercise())


def test_frozen_outputs_rejected_before_bootstrap(tmp_path, monkeypatch):
    import argparse
    from synaptic_bench_150 import output_guard
    from synaptic_bench_150.output_guard import ensure_output_is_not_frozen
    project_root = tmp_path / "source"
    monkeypatch.setattr(output_guard, "PROJECT_ROOT", project_root)
    from synaptic_bench_150.runner import run_benchmark_suite
    from appworld_micro_30.runner import _run_benchmark_async
    frozen = project_root / "experiments/pa150_v201_three_arm_calibrated"
    frozen.mkdir(parents=True)
    (frozen / "results.jsonl").write_text('{"episode_id":"synthetic-protected-record"}\n')
    before = (frozen / "results.jsonl").read_bytes()
    for output in (frozen, frozen / "nested", frozen.parent, project_root / "archive"):
        with pytest.raises(ValueError, match="Frozen research evidence"):
            ensure_output_is_not_frozen(output)
    for runner in (run_benchmark_suite, _run_benchmark_async):
        with pytest.raises(ValueError, match="Frozen research evidence"):
            asyncio.run(runner(argparse.Namespace(output_dir=str(frozen))))
    assert (frozen / "results.jsonl").read_bytes() == before
    ensure_output_is_not_frozen(tmp_path / "fresh-run")


def test_micro_resume_model_mismatch_preserves_bytes(tmp_path, monkeypatch):
    import argparse, json
    from appworld_micro_30 import runner
    from appworld_micro_30.build_dataset import DEFAULT_EPISODES_PATH
    output = tmp_path / "output"
    output.mkdir()
    result = output / "results.jsonl"
    raw = json.dumps({"model": "different-model", "episode_id": "old", "arm_name": "clonact"}) + "\n"
    result.write_text(raw)
    args = argparse.Namespace(output_dir=str(output), work_root=str(tmp_path / "work"),
                              model="deepseek-flash", episode_ids="", arms="clonact",
                              episodes_file=str(DEFAULT_EPISODES_PATH), overwrite=False)
    monkeypatch.setenv("PA_LLM_MODEL", "deepseek-flash")
    with pytest.raises(ValueError, match="different model"):
        asyncio.run(runner._run_benchmark_async(args))
    assert result.read_text() == raw
