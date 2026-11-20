"""Public release compatibility and resume guards, without model or dataset access."""
import argparse
import asyncio
import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
from clonact.release import BenchmarkRelease

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('package,internal', [('synaptic_bench_150','2.0.1'), ('appworld_micro_30','1.0.1'), ('synaptic_stream_60','3.0.0')])
def test_public_identity_and_legacy_provenance(package, internal):
    release = BenchmarkRelease(ROOT / package / 'release.json')
    original = {'evaluator_version': internal, 'episode_passed': True}
    record = release.stamp(original)
    assert original == {'evaluator_version': internal, 'episode_passed': True}
    assert record['benchmark_version'] == record['evaluator_version'] == '1.0.0'
    assert record['source_versions']['evaluator'] == internal
    assert release.matches(record)
    assert release.matches(original)
    assert not release.matches(original, allow_legacy=False)
    assert not release.matches({'evaluator_version':'1.0.0'})
    for field in ('benchmark_name', 'benchmark_version', 'evaluator_version', 'dataset_sha256', 'source_versions'):
        changed = {**record, field: 'changed'}
        assert not release.matches(changed)
    incomplete = {**original, 'benchmark_version': '1.0.0'}
    assert not release.matches(incomplete)
    record['source_versions']['evaluator'] = 'mutated'
    assert release.internal_evaluator == release.metadata['source_versions']['evaluator'] == internal


@pytest.mark.parametrize(
    'package,name,version',
    [
        ('synaptic_bench_150', 'Synaptic-Bench-150', '1.0.0'),
        ('appworld_micro_30', 'AppWorld-Micro-30', '1.0.0'),
        ('synaptic_stream_60', 'Synaptic-Stream-60', '1.0.0'),
    ],
)
def test_installed_entrypoint_outside_checkout(package, name, version, tmp_path):
    process = subprocess.run([sys.executable, '-m', package+'.runner', '--version'], cwd=tmp_path,
                             capture_output=True, text=True, check=True)
    assert process.stdout.strip() == f'{name} {version}'


@pytest.mark.parametrize('mutation', [None, 'version', 'fingerprint', 'legacy'])
def test_micro_resume_release_identity_preserves_existing_records(tmp_path, monkeypatch, mutation):
    from appworld_micro_30 import runner
    from clonact import llm_client
    ep = SimpleNamespace(episode_id='test_episode')
    monkeypatch.setattr(runner, 'load_micro30_episodes', lambda *a, **k: [ep])
    monkeypatch.setattr(runner, 'episode_fingerprint', lambda value: 'expected-episode-hash')
    monkeypatch.setattr(llm_client, 'query_deepseek_balance_cny', lambda: None)
    monkeypatch.setattr(runner, 'aggregate_micro_results', lambda records: {'total_records': len(records)})
    row = runner.RELEASE.stamp({'episode_id':ep.episode_id, 'episode_sha256':'expected-episode-hash',
                               'model':'deepseek-flash', 'arm_name':'clonact', 'episode_passed':True})
    if mutation == 'version': row['benchmark_version'] = '2.0.0'
    if mutation == 'fingerprint': row['episode_sha256'] = 'changed'
    if mutation == 'legacy': row = {'episode_id':ep.episode_id,'arm_name':'clonact','model':'deepseek-flash'}
    output = tmp_path/'output';output.mkdir()
    path = output/'results.jsonl';raw=json.dumps(row)+'\n';path.write_text(raw)
    args = argparse.Namespace(output_dir=str(output),work_root=str(tmp_path/'work'),episodes_file='unused',
                              episode_ids='',arms='clonact',model='deepseek-flash',overwrite=False,concurrency=1)
    if mutation:
        with pytest.raises(ValueError, match='benchmark release|episode fingerprint'):
            asyncio.run(runner._run_benchmark_async(args))
    else:
        summary = asyncio.run(runner._run_benchmark_async(args))
        assert summary['benchmark_version'] == summary['evaluator_version'] == '1.0.0'
        assert summary['total_records'] == 1
    assert path.read_text() == raw


def test_root_layout_and_git_provenance():
    from dsh_runner.dsh_controller import PROJECT_ROOT, SUPERAGENT_REPO
    from synaptic_bench_150.runner import _current_git_head
    assert PROJECT_ROOT == ROOT
    assert SUPERAGENT_REPO == ROOT/'third_party/dsh-superagent-prototype'
    if (ROOT / '.git').exists():
        expected = subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
        assert _current_git_head() == expected
    else:
        assert isinstance(_current_git_head(), str)
