"""End-to-end verification tests for `Synaptic-Bench-150` unified harness & `Jev` cell fingerprinting."""
from __future__ import annotations

from fake_dsh import FakeDSHHost

import json
import subprocess
import sys
from pathlib import Path

import pytest

from synaptic_bench_150.runner import aggregate_benchmark_results, load_pa150_episodes
from synaptic_bench_150.schema import EpisodeSpec, TimelineEventSpec, TriggerSpec, VerifierSpec, WorkspaceMountSpec
from synaptic_bench_150.verifiers import RuntimeArtifacts, evaluate_episode_verifiers
from synaptic_bench_150.workspace_manager import (
    Gaia2ServerProcess,
    PROJECT_ROOT,
    MaterializedEpisodeWorkspace,
    build_gaia_cli_script,
    hash_directory_tree,
    hash_git_repo_state,
)
from dsh_runner.sandbox_clis import init_dsh_sandbox
from clonact.context_cell import ContextCell
from clonact.context_mmu import ContextMMU, extract_discriminative_topic
from clonact.llm_client import HomogeneousLLMClient
from clonact.system1_judge import System1Judge
from clonact.trunk import ElasticSharedTrunk


def test_jev_discriminative_cell_fingerprinting_on_same_repo_cooperbench(tmp_path: Path) -> None:
    """Two concurrent features on the exact same repo (`PA150_C2_01`) must get distinct topics & URI footprints."""
    c2_ep = load_pa150_episodes(episode_ids=["PA150_C2_01"])[0]
    ev_a = next(e for e in c2_ep.timeline if e.event_id == "ev_feat_a")
    ev_b = next(e for e in c2_ep.timeline if e.event_id == "ev_feat_b")

    topic_a = extract_discriminative_topic(ev_a.text, channel=ev_a.channel)
    topic_b = extract_discriminative_topic(ev_b.text, channel=ev_b.channel)
    assert topic_a != topic_b
    assert "-A]" in topic_a and "-B]" in topic_b

    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "t@local"], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Tester"], cwd=repo_dir, check=True, capture_output=True)
    (repo_dir / "README.md").write_text("# test\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo_dir, check=True, capture_output=True)

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    tool_events: list[tuple[str, str, int]] = []

    mmu = ContextMMU(
        sandbox_paths={
            "world": tmp_path,
            "repo": repo_dir,
            "worktrees": tmp_path / "worktrees",
            "docs": docs_dir,
        },
        judge=System1Judge(typesafe_api_key="", deepseek_api_key=""),
        llm=HomogeneousLLMClient(api_key=""),
        trunk=ElasticSharedTrunk(),
        on_tool_executed=lambda cid, tname, cnt: tool_events.append((cid, tname, cnt)),
        dsh_host=FakeDSHHost(),
    )
    c1 = ContextCell(
        cell_id="Cell_E_1",
        topic=topic_a,
        source_kind="external_event",
        sandbox_paths=mmu.sandbox_paths,
        trunk=mmu.trunk,
        llm=mmu.llm,
        bus=mmu.bus,
        initial_status="detached_executing",
        on_tool_executed=mmu.on_tool_executed,
        dsh_host=FakeDSHHost(),
    )
    c2 = ContextCell(
        cell_id="Cell_E_2",
        topic=topic_b,
        source_kind="external_event",
        sandbox_paths=mmu.sandbox_paths,
        trunk=mmu.trunk,
        llm=mmu.llm,
        bus=mmu.bus,
        initial_status="detached_executing",
        on_tool_executed=mmu.on_tool_executed,
        dsh_host=FakeDSHHost(),
    )
    mmu.cells["Cell_E_1"] = c1
    mmu.cells["Cell_E_2"] = c2
    c1.inbound_channels.append(ev_a.channel)
    c2.inbound_channels.append(ev_b.channel)
    mmu.bus.record_uri_touch(
        c1.cell_id, {"code://src/click/core.py"}, is_write=True, intent_one_liner="edit core.py"
    )
    mmu.bus.record_uri_touch(
        c2.cell_id, {"code://src/click/decorators.py"}, is_write=True, intent_one_liner="edit decorators.py"
    )
    mmu.bus.footprints[c1.cell_id].latest_milestone = "Added flag validation in core.py"

    tag_1 = mmu._format_cell_criteria_tag(c1.cell_id, c1)
    tag_2 = mmu._format_cell_criteria_tag(c2.cell_id, c2)
    assert "code://src/click/core.py" in tag_1
    assert "code://src/click/decorators.py" in tag_2
    assert "Added flag validation in core.py" in tag_1
    assert tag_1 != tag_2

    # Also verify `on_tool_executed` deterministic step callback fires on tool execution
    import asyncio

    asyncio.run(c1._backend.handle_dsh_session_event({
        "kind": "tool_result", "session_id": "ro-Cell_E_1", "tool_name": "read_file",
        "tool_args": {"path": "README.md"}, "output_excerpt": "# test", "is_error": False,
    }))
    assert tool_events == [("Cell_E_1", "read_file", 1)]


@pytest.mark.integration
def test_gaia2_env_server_oracle_and_readonly_verifiers(tmp_path: Path) -> None:
    """Real Meta ARE (`Gaia2`) server must fail empty agent run, preserve read-only hash, and pass oracle replay."""
    samples = json.loads((PROJECT_ROOT / "data" / "upstream_pillars" / "samples" / "gaia2_samples.json").read_text())
    sample_row = samples["execution"][0]
    row_id = str(sample_row["id"])
    scenario_id = str(sample_row["scenario_id"])

    diag_secret = "unit_test_pa150_diag_secret_2026"
    srv = Gaia2ServerProcess(port=19031, log_path=tmp_path / "gaia2_test.log", diag_token=diag_secret)
    try:
        info = srv.start_scenario(
            upstream_id=f"gaia2::execution::{row_id}",
            config="execution",
            scenario_id=scenario_id,
            judge_mode="deterministic",
        )
        assert info["ok"] is True
        assert info["tool_count"] > 20

        cli_fp = tmp_path / "gaia_cli"
        cli_fp.write_text(build_gaia_cli_script(srv.base_url), encoding="utf-8")
        cli_fp.chmod(0o755)

        # Read-only inspection via `./gaia_cli apps` AND `Emails__get_email_by_index` (`NEW-9`: passive `is_read` flip)
        # MUST preserve `unmodified_since_start == True` and `mutating_call_count == 0`!
        apps_proc = subprocess.run([str(cli_fp), "apps"], check=True, capture_output=True, text=True)
        assert "EmailClientV2" in apps_proc.stdout or "Emails" in apps_proc.stdout or "Calendar" in apps_proc.stdout
        read_mail_proc = subprocess.run(
            [str(cli_fp), "call", "Emails__get_email_by_index", '{"idx": 0}'],
            capture_output=True,
            text=True,
        )
        if read_mail_proc.returncode == 0:
            mail_out = json.loads(read_mail_proc.stdout)
            assert mail_out.get("mutated") is False
        mid_snap = srv.snapshot()
        assert mid_snap["unmodified_since_start"] is True
        assert mid_snap["mutating_call_count"] == 0

        ep = EpisodeSpec(
            episode_id="TEST_GAIA2_01",
            track_id="O-1",
            domain="office",
            scenario_type="aba_interleaved",
            title="Test Gaia2 Verifier",
            upstream_pillars=["gaia2"],
            upstream_task_ids=[f"gaia2::execution::{row_id}"],
            workspace_mounts=[
                WorkspaceMountSpec(
                    mount_id="mnt_gaia",
                    pillar="gaia2",
                    mount_relpath="gaia2_universe",
                    upstream_id=f"gaia2::execution::{row_id}",
                    repo_or_category=scenario_id,
                    setup_meta={"config": "execution"},
                )
            ],
            timeline=[],
            verifiers=[
                VerifierSpec(verifier_id="v_ro", kind="readonly_zero_mutation_check", mount_id="mnt_gaia", check_params={"after_event_id": "turn_1b"}),
                VerifierSpec(verifier_id="v_oracle", kind="gaia2_oracle_check", mount_id="mnt_gaia", check_params={"scenario_id": scenario_id}),
            ],
        )
        ws = MaterializedEpisodeWorkspace(
            episode=ep,
            arm_name="synaptic_pa",
            root_dir=tmp_path,
            repo_dir=tmp_path,
            gaia_server=srv,
            gaia_mount_id="mnt_gaia",
            initial_snapshots={"mnt_gaia": {"apps_state_hash": info["initial_state_hash"], "mutating_call_count": 0}},
            mid_turn_snapshots={("turn_1b", "mnt_gaia"): mid_snap},
        )

        # 1. Before agent actions: read-only check passes, oracle check fails
        res_empty = evaluate_episode_verifiers(ep, ws, RuntimeArtifacts(arm_name="synaptic_pa"))
        assert res_empty["verifiers"][0]["passed"] is True
        assert res_empty["verifiers"][1]["passed"] is False
        assert res_empty["episode_passed"] is False

        # 2. P0-6: Verify `/oracle_actions` blocks both unauthenticated requests AND legacy static headers (`403`),
        # and only accepts HMAC-matched `x-pa150-diag-token` when `PA150_GAIA_DIAG_SECRET` is configured.
        import urllib.error
        import urllib.request

        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(f"{srv.base_url}/oracle_actions", timeout=10.0)
        assert exc_info.value.code == 403

        legacy_req = urllib.request.Request(
            f"{srv.base_url}/oracle_actions",
            headers={"x-pa150-internal-diag": "1"},
        )
        with pytest.raises(urllib.error.HTTPError) as exc_legacy:
            urllib.request.urlopen(legacy_req, timeout=10.0)
        assert exc_legacy.value.code == 403

        diag_req = urllib.request.Request(
            f"{srv.base_url}/oracle_actions",
            headers={"x-pa150-diag-token": diag_secret},
        )
        with urllib.request.urlopen(diag_req, timeout=10.0) as resp:
            oracle_acts = json.loads(resp.read().decode("utf-8"))["oracle_actions"]
        assert len(oracle_acts) > 0
        for act in oracle_acts:
            subprocess.run(
                [str(cli_fp), "call", act["tool_name"], json.dumps(act["args"])],
                check=True,
                capture_output=True,
                text=True,
            )

        res_gold = evaluate_episode_verifiers(ep, ws, RuntimeArtifacts(arm_name="synaptic_pa"))
        assert res_gold["verifiers"][0]["passed"] is True
        assert res_gold["verifiers"][1]["passed"] is True
        assert res_gold["episode_passed"] is True
    finally:
        srv.stop()


@pytest.mark.integration
def test_mcpmark_verify_script_and_readonly_check(tmp_path: Path) -> None:
    """`mcpmark_verify_script` + `readonly_zero_mutation_check` accurately score filesystem mutations."""
    fs_dir = tmp_path / "office_fs"
    fs_dir.mkdir()
    (fs_dir / "notes.txt").write_text("alpha\nbeta\n", encoding="utf-8")

    verify_py = tmp_path / "verify.py"
    verify_py.write_text(
        "import os, pathlib, sys\n"
        "root = pathlib.Path(os.environ['FILESYSTEM_TEST_DIR'])\n"
        "out = root / 'summary.txt'\n"
        "sys.exit(0 if out.exists() and out.read_text().strip() == '2 lines' else 1)\n",
        encoding="utf-8",
    )

    ep = EpisodeSpec(
        episode_id="TEST_MCP_01",
        track_id="O-1",
        domain="office",
        scenario_type="aba_interleaved",
        title="Test MCPMark FS",
        upstream_pillars=["mcpmark_filesystem"],
        upstream_task_ids=["mcpmark::filesystem::test"],
        workspace_mounts=[
            WorkspaceMountSpec(
                mount_id="mnt_fs",
                pillar="mcpmark_filesystem",
                mount_relpath="office_fs",
                upstream_id="mcpmark::filesystem::test",
                repo_or_category="filesystem/desktop",
            )
        ],
        timeline=[],
        verifiers=[
            VerifierSpec(verifier_id="v_ro", kind="readonly_zero_mutation_check", mount_id="mnt_fs", check_params={"after_event_id": "turn_1b"}),
            VerifierSpec(verifier_id="v_mcp", kind="mcpmark_verify_script", mount_id="mnt_fs", check_params={"verify_py": str(verify_py)}),
        ],
    )
    init_snap = hash_directory_tree(fs_dir)
    ws = MaterializedEpisodeWorkspace(
        episode=ep,
        arm_name="synaptic_pa",
        root_dir=tmp_path,
        repo_dir=tmp_path,
        mount_paths={"mnt_fs": fs_dir},
        initial_snapshots={"mnt_fs": init_snap},
        mid_turn_snapshots={("turn_1b", "mnt_fs"): hash_directory_tree(fs_dir)},
    )

    # Before summary.txt is written: readonly True, verify.py False
    r1 = evaluate_episode_verifiers(ep, ws, RuntimeArtifacts(arm_name="synaptic_pa"))
    assert r1["verifiers"][0]["passed"] is True
    assert r1["verifiers"][1]["passed"] is False

    # Write expected output after turn_1b: BOTH pass!
    (fs_dir / "summary.txt").write_text("2 lines\n", encoding="utf-8")
    r2 = evaluate_episode_verifiers(ep, ws, RuntimeArtifacts(arm_name="synaptic_pa"))
    assert r2["episode_passed"] is True


def test_cooperbench_single_and_merged_pytest_verifiers(tmp_path: Path) -> None:
    """`cooperbench_pytest` and `cooperbench_merged_pytest` verify both isolated worktrees and clean 2-way merges."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "t@local"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Tester"], cwd=repo, check=True, capture_output=True)
    (repo / "calc.py").write_text(
        "def add(a: int, b: int) -> int:\n    return a + b\n\n\ndef mul(a: int, b: int) -> int:\n    return a * b\n",
        encoding="utf-8",
    )
    (repo / "tests").mkdir()
    (repo / "tests" / "test_calc.py").write_text(
        "from calc import add\n\ndef test_base():\n    assert add(1, 2) == 3\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=repo, check=True, capture_output=True)
    base_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()

    # Build gold tests_patch_a and tests_patch_b in temp worktrees
    patches_dir = tmp_path / "patches"
    patches_dir.mkdir()
    patch_a = patches_dir / "tests_a.patch"
    patch_b = patches_dir / "tests_b.patch"

    (repo / "tests" / "test_feat_a.py").write_text(
        "from calc import sub\n\ndef test_sub():\n    assert sub(5, 2) == 3\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "tests/test_feat_a.py"], cwd=repo, check=True, capture_output=True)
    patch_a.write_text(
        subprocess.run(["git", "diff", "--cached"], cwd=repo, capture_output=True, text=True).stdout,
        encoding="utf-8",
    )
    subprocess.run(["git", "reset", "--hard", base_commit], cwd=repo, check=True, capture_output=True)

    (repo / "tests" / "test_feat_b.py").write_text(
        "from calc import div_floor\n\ndef test_div_floor():\n    assert div_floor(7, 2) == 3\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "tests/test_feat_b.py"], cwd=repo, check=True, capture_output=True)
    patch_b.write_text(
        subprocess.run(["git", "diff", "--cached"], cwd=repo, capture_output=True, text=True).stdout,
        encoding="utf-8",
    )
    subprocess.run(["git", "reset", "--hard", base_commit], cwd=repo, check=True, capture_output=True)

    # Create two isolated agent worktrees (`wt_a` and `wt_b`)
    wt_a = tmp_path / "wt_a"
    wt_b = tmp_path / "wt_b"
    subprocess.run(["git", "worktree", "add", "-b", "cell_a", str(wt_a), base_commit], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "worktree", "add", "-b", "cell_b", str(wt_b), base_commit], cwd=repo, check=True, capture_output=True)

    # Edit non-conflicting parts of calc.py in wt_a (top) and wt_b (bottom)
    (wt_a / "calc.py").write_text(
        "def sub(a: int, b: int) -> int:\n    return a - b\n\n"
        + (wt_a / "calc.py").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (wt_b / "calc.py").write_text(
        (wt_b / "calc.py").read_text(encoding="utf-8")
        + "\ndef div_floor(a: int, b: int) -> int:\n    return a // b\n",
        encoding="utf-8",
    )

    ep = EpisodeSpec(
        episode_id="TEST_CB_01",
        track_id="C-2",
        domain="coding",
        scenario_type="high_concurrency_burst",
        title="Test CooperBench Dual Worktree",
        upstream_pillars=["cooperbench"],
        upstream_task_ids=["cb::a", "cb::b"],
        workspace_mounts=[
            WorkspaceMountSpec(
                mount_id="mnt_shared_repo",
                pillar="cooperbench",
                mount_relpath="repo",
                upstream_id="cb_pair",
                repo_or_category="local/calc",
                base_commit=base_commit,
            )
        ],
        timeline=[],
        verifiers=[
            VerifierSpec(verifier_id="v_a", kind="cooperbench_pytest", mount_id="mnt_shared_repo", check_params={"target_task_tag": "feat_a", "tests_patch": str(patch_a)}),
            VerifierSpec(verifier_id="v_b", kind="cooperbench_pytest", mount_id="mnt_shared_repo", check_params={"target_task_tag": "feat_b", "tests_patch": str(patch_b)}),
            VerifierSpec(verifier_id="v_m", kind="cooperbench_merged_pytest", mount_id="mnt_shared_repo", check_params={"tests_patch_a": str(patch_a), "tests_patch_b": str(patch_b)}),
            VerifierSpec(verifier_id="v_h", kind="noise_and_dedup_telemetry_check", mount_id="mnt_shared_repo", check_params={"expected_noise_dropped": 3, "expected_duplicates_merged": 1, "expected_worktrees": 2}),
        ],
    )
    ws = MaterializedEpisodeWorkspace(
        episode=ep,
        arm_name="synaptic_pa",
        root_dir=tmp_path,
        repo_dir=repo,
        mount_paths={"mnt_shared_repo": repo},
        shared_symlinks={".venv": PROJECT_ROOT / ".venv"},
        initial_snapshots={"mnt_shared_repo": hash_git_repo_state(repo)},
    )
    artifacts = RuntimeArtifacts(
        arm_name="synaptic_pa",
        task_worktrees={"feat_a": wt_a, "feat_b": wt_b},
        all_worktrees=[wt_a, wt_b],
        telemetry={"noise_dropped": 3, "noise_spawned_workers": 0, "duplicates_merged": 1, "duplicate_spawned_workers": 0, "worktrees_created": 2},
    )
    report = evaluate_episode_verifiers(ep, ws, artifacts)
    assert report["episode_passed"] is True, report

    # Introduce a conflicting line edit on line 1 in `wt_b` -> merged verifier must catch git merge conflict!
    (wt_b / "calc.py").write_text(
        "def conflicting_header() -> int:\n    return 999\n"
        + (wt_b / "calc.py").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    report_conflict = evaluate_episode_verifiers(ep, ws, artifacts)
    merged_v = next(v for v in report_conflict["verifiers"] if v["verifier_id"] == "v_m")
    assert merged_v["tier"] == "task_solve"
    assert merged_v["passed"] is False
    assert merged_v["merge_clean"] is False
    # `v2.0.0` (`P0-3`): `cooperbench_merged_pytest` is a gating `task_solve` verifier!
    assert report_conflict["episode_passed"] is False

    summary = aggregate_benchmark_results([report])
    assert summary["arms"]["synaptic_pa"]["episode_pass_rate"] == 1.0
    assert summary["arms"]["synaptic_pa"]["zero_shot_conflict_free_merge_rate"] == 1.0
    summary_conflict = aggregate_benchmark_results([report_conflict])
    assert summary_conflict["arms"]["synaptic_pa"]["episode_pass_rate"] == 0.0
    assert summary_conflict["arms"]["synaptic_pa"]["zero_shot_conflict_free_merge_rate"] == 0.0


def test_dsh_sandbox_accepts_custom_seed_repo(tmp_path: Path) -> None:
    """`init_dsh_sandbox(seed_repo_dir=...)` initializes `dsh-superagent` origin/clone from any upstream repo."""
    seed = tmp_path / "custom_upstream"
    seed.mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=seed, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "t@local"], cwd=seed, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Tester"], cwd=seed, check=True, capture_output=True)
    (seed / "custom_mod.py").write_text("VALUE = 42\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=seed, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "upstream commit"], cwd=seed, check=True, capture_output=True)

    paths = init_dsh_sandbox(
        tmp_path / "dsh_world",
        agents_md_override="# Custom PA150 Repo\n",
        extra_repo_files={"gaia_cli": ("#!/bin/sh\necho ok\n", 0o755)},
        seed_repo_dir=seed,
    )
    assert (paths["clone_repo"] / "custom_mod.py").exists()
    assert (paths["clone_repo"] / "gaia_cli").exists()
    assert "Custom PA150 Repo" in (paths["clone_repo"] / "AGENTS.md").read_text(encoding="utf-8")


def test_untracked_diff_two_pass_merge_similar_routing_and_l3_tombstone(tmp_path: Path) -> None:
    """Covers `P0-7` untracked files + two-pass conflicting test patches, `similar_cell_routing_check`, and `L3` tombstones."""
    from synaptic_bench_150.verifiers import _extract_agent_diff

    repo = tmp_path / "src_repo"
    (repo / "src" / "pkg").mkdir(parents=True)
    (repo / "tests").mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "t@local"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Tester"], cwd=repo, check=True, capture_output=True)
    (repo / "src" / "pkg" / "__init__.py").write_text("from .core import base_val\n", encoding="utf-8")
    (repo / "src" / "pkg" / "core.py").write_text("def base_val() -> int:\n    return 10\n", encoding="utf-8")
    (repo / "tests" / "test_shared.py").write_text(
        "from pkg.core import base_val\n\ndef test_base():\n    assert base_val() == 10\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True, capture_output=True)
    base_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()

    # Build two test patches that modify the SAME line of `tests/test_shared.py` (so combined `git apply` conflicts!)
    patches_dir = tmp_path / "patches"
    patches_dir.mkdir()
    patch_a = patches_dir / "ta.patch"
    patch_b = patches_dir / "tb.patch"

    (repo / "tests" / "test_shared.py").write_text(
        "from pkg.extra_a import feat_a_val\n\ndef test_a():\n    assert feat_a_val() == 11\n",
        encoding="utf-8",
    )
    patch_a.write_text(
        subprocess.run(["git", "diff"], cwd=repo, capture_output=True, text=True).stdout,
        encoding="utf-8",
    )
    subprocess.run(["git", "checkout", "--", "tests/test_shared.py"], cwd=repo, check=True, capture_output=True)

    (repo / "tests" / "test_shared.py").write_text(
        "from pkg.extra_b import feat_b_val\n\ndef test_b():\n    assert feat_b_val() == 22\n",
        encoding="utf-8",
    )
    patch_b.write_text(
        subprocess.run(["git", "diff"], cwd=repo, capture_output=True, text=True).stdout,
        encoding="utf-8",
    )
    subprocess.run(["git", "checkout", "--", "tests/test_shared.py"], cwd=repo, check=True, capture_output=True)

    wt_a = tmp_path / "wt_a"
    wt_b = tmp_path / "wt_b"
    subprocess.run(["git", "worktree", "add", "-b", "fa", str(wt_a), base_commit], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "worktree", "add", "-b", "fb", str(wt_b), base_commit], cwd=repo, check=True, capture_output=True)

    # Create UNTRACKED brand-new modules in `src/pkg/` without `git add`!
    (wt_a / "src" / "pkg" / "extra_a.py").write_text("def feat_a_val() -> int:\n    return 11\n", encoding="utf-8")
    (wt_b / "src" / "pkg" / "extra_b.py").write_text("def feat_b_val() -> int:\n    return 22\n", encoding="utf-8")

    diff_a = _extract_agent_diff(wt_a, base_commit)
    assert "src/pkg/extra_a.py" in diff_a
    # Real git index in `wt_a` must remain untouched (`?? src/pkg/extra_a.py`)
    st_a = subprocess.run(["git", "status", "--porcelain"], cwd=wt_a, capture_output=True, text=True).stdout
    assert "?? src/pkg/extra_a.py" in st_a

    ep = EpisodeSpec(
        episode_id="TEST_P07_SIM",
        track_id="C-2",
        domain="coding",
        scenario_type="high_concurrency_burst",
        title="Untracked src-layout + two-pass merge + similar routing",
        upstream_pillars=["cooperbench"],
        upstream_task_ids=["cb::a", "cb::b"],
        workspace_mounts=[
            WorkspaceMountSpec(
                mount_id="mnt_repo",
                pillar="cooperbench",
                mount_relpath="repo",
                upstream_id="cb_pair",
                repo_or_category="local/pkg",
                base_commit=base_commit,
            )
        ],
        timeline=[],
        verifiers=[
            VerifierSpec(verifier_id="v_m", kind="cooperbench_merged_pytest", mount_id="mnt_repo", check_params={"tests_patch_a": str(patch_a), "tests_patch_b": str(patch_b)}),
            VerifierSpec(
                verifier_id="v_sim",
                kind="similar_cell_routing_check",
                mount_id="mnt_repo",
                check_params={"event_id": "ev_similar_cell_followup", "expected_target_tag": "feat_a", "distractor_tag": "feat_b"},
            ),
        ],
    )
    ws = MaterializedEpisodeWorkspace(
        episode=ep,
        arm_name="synaptic_pa",
        root_dir=tmp_path,
        repo_dir=repo,
        mount_paths={"mnt_repo": repo},
        shared_symlinks={".venv": PROJECT_ROOT / ".venv"},
    )
    artifacts = RuntimeArtifacts(
        arm_name="synaptic_pa",
        task_worktrees={"feat_a": wt_a, "feat_b": wt_b},
        all_worktrees=[wt_a, wt_b],
        routing_events={
            "ev_similar_cell_followup": {
                "routed_tags": ["feat_a"],
                "spawned_new_worker": False,
                "distractor_overturned": False,
            }
        },
    )
    rep = evaluate_episode_verifiers(ep, ws, artifacts)
    assert rep["episode_passed"] is True, rep
    assert rep["verifiers"][0]["test_eval_mode"] == "two_pass_on_merged_code"
    assert rep["task_solve_passed"] == 1 and rep["diagnostic_passed"] == 0 and rep["orchestration_passed"] == 1

    agg = aggregate_benchmark_results([rep])
    t1 = agg["arms"]["synaptic_pa"]["tier1_orchestration_safety"]
    t2 = agg["arms"]["synaptic_pa"]["tier2_task_solve"]
    assert t1["similar_cell_routing_accuracy"] == 1.0
    assert t2["verifier_pass_rate"] == 1.0
    assert agg["arms"]["synaptic_pa"]["zero_shot_conflict_free_merge_rate"] == 1.0

    # Also test `ElasticSharedTrunk.invalidate_stale_tombstones` across BOTH `l2_entries` and `l3_cold_store`
    from clonact.trunk import TrunkEntry, compute_blob_hash

    pol = repo / "WORKSPACE_POLICY.md"
    pol.write_text("Revision 1\n", encoding="utf-8")
    rev1_hash = compute_blob_hash(pol)
    trunk = ElasticSharedTrunk(l2_max_entries=1)
    trunk.promote(
        TrunkEntry(
            entry_id="ts_1",
            category="tombstone",
            summary="Rev1 tombstone A",
            anchor_type="blob_hash",
            anchor_value=rev1_hash,
            target_path="WORKSPACE_POLICY.md",
        )
    )
    trunk.promote(
        TrunkEntry(
            entry_id="ts_2",
            category="tombstone",
            summary="Rev1 tombstone B",
            anchor_type="blob_hash",
            anchor_value=rev1_hash,
            target_path="WORKSPACE_POLICY.md",
        )
    )
    assert len(trunk.l3_cold_store) >= 1 and len(trunk.l2_pointers) >= 1
    pol.write_text("Revision 2 — updated!\n", encoding="utf-8")
    evicted = trunk.invalidate_stale_tombstones(repo)
    assert set(evicted) == {"ts_1", "ts_2"}
    assert len(trunk.l2_entries) == 0 and len(trunk.l3_cold_store) == 0 and len(trunk.l2_pointers) == 0


def test_discriminative_topic_and_dsh_open_id_normalization() -> None:
    """`extract_discriminative_topic` normalizes raw whitespace without regex heuristics & `_dsh_feishu_open_id` fits DSH Zod regex."""
    import re
    from synaptic_bench_150.runner import _dsh_feishu_open_id
    from clonact.context_mmu import extract_discriminative_topic

    for raw_sender in ("chief_of_staff", "ops_ticket", "release-bot.1", ""):
        oid = _dsh_feishu_open_id(raw_sender)
        assert re.match(r"^ou_[a-zA-Z0-9]+$", oid), oid

    mcp_legal = (
        "请立即在个人助理环境中处理以下事务：\n\n"
        "## **Task Overview**\n"
        'The folder "legal_files/" contains all versions (Preferred_Stock_Purchase_Agreement_v0.txt -- '
        "Preferred_Stock_Purchase_Agreement_v10.txt).\n"
        "Generate `dispute_review.txt`."
    )
    topic = extract_discriminative_topic(mcp_legal, channel="feishu://p2p/ops_ticket")
    assert topic.startswith("[p2p/ops_ticket] ")
    assert "legal_files/" in topic or "Preferred_Stock_Purchase_Agreement_v0.txt" in topic


def test_known1_h4_policy_propagation_and_3wave_verifier(tmp_path: Path) -> None:
    """`Known-1`: `WORKSPACE_POLICY.md` Rev 2 commits to `ws_repo_dir` + pushes to DSH `origin:main`, and 3-wave verifier checks all 3 waves + policy."""
    from synaptic_bench_150.runner import _apply_workspace_file_updates

    seed = tmp_path / "seed_repo"
    seed.mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=seed, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "t@local"], cwd=seed, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Tester"], cwd=seed, check=True, capture_output=True)
    (seed / "mod.py").write_text("def base() -> int:\n    return 1\n", encoding="utf-8")
    (seed / "tests").mkdir()
    (seed / "tests" / "test_mod.py").write_text("from mod import base\n\ndef test_base():\n    assert base() == 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=seed, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=seed, check=True, capture_output=True)
    base_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=seed, capture_output=True, text=True).stdout.strip()

    # Build 3 independent wave test patches
    patches: list[Path] = []
    for idx, fn_name in enumerate(("w1", "w2", "w3"), start=1):
        tf = seed / "tests" / f"test_{fn_name}.py"
        tf.write_text(f"from mod import {fn_name}\n\ndef test_{fn_name}():\n    assert {fn_name}() == {idx * 10}\n", encoding="utf-8")
        subprocess.run(["git", "add", str(tf.relative_to(seed))], cwd=seed, check=True, capture_output=True)
        pf = tmp_path / f"{fn_name}.patch"
        pf.write_text(subprocess.run(["git", "diff", "--cached"], cwd=seed, capture_output=True, text=True).stdout, encoding="utf-8")
        patches.append(pf)
        subprocess.run(["git", "reset", "--hard", base_commit], cwd=seed, check=True, capture_output=True)

    # Set up Synaptic repo + DSH bare/clone sandbox
    ws_repo = tmp_path / "syn_repo"
    subprocess.run(["git", "clone", str(seed), str(ws_repo)], check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "t@local"], cwd=ws_repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Tester"], cwd=ws_repo, check=True, capture_output=True)
    (ws_repo / "WORKSPACE_POLICY.md").write_text("# Policy Revision 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "WORKSPACE_POLICY.md"], cwd=ws_repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "rev1"], cwd=ws_repo, check=True, capture_output=True)

    dsh_paths = init_dsh_sandbox(
        tmp_path / "dsh_world",
        extra_repo_files={"WORKSPACE_POLICY.md": ("# Policy Revision 1\n", 0o644)},
        seed_repo_dir=seed,
    )
    wt_w1 = tmp_path / "wt_w1"
    wt_w2 = tmp_path / "wt_w2"
    subprocess.run(["git", "worktree", "add", "-b", "b_w1", str(wt_w1), "main"], cwd=ws_repo, check=True, capture_output=True)
    subprocess.run(["git", "worktree", "add", "-b", "b_w2", str(wt_w2), "main"], cwd=ws_repo, check=True, capture_output=True)
    (wt_w1 / "mod.py").write_text("def w1() -> int:\n    return 10\n\n" + (wt_w1 / "mod.py").read_text(encoding="utf-8"), encoding="utf-8")
    (wt_w2 / "mod.py").write_text((wt_w2 / "mod.py").read_text(encoding="utf-8") + "\ndef w2() -> int:\n    return 20\n", encoding="utf-8")

    # Apply Wave-3 mid-episode policy update (`WORKSPACE_POLICY.md` -> POLICY-REV-2)
    ev_w3 = TimelineEventSpec(
        event_id="ev_wave_3",
        seq=3,
        kind="user_chat",
        channel="feishu://p2p/user",
        sender="user",
        text="Implement wave 3 under POLICY-REV-2",
        trigger=TriggerSpec(mode="after_turn_settled"),
        metadata={"workspace_file_updates": {"WORKSPACE_POLICY.md": "# Policy Revision 2 (`POLICY-REV-2`)\n"}},
    )
    updated = _apply_workspace_file_updates(
        ev_w3,
        ws_repo,
        dsh_clone_repo=dsh_paths["clone_repo"],
        sync_worktrees=[wt_w1, wt_w2],
    )
    assert updated == ["WORKSPACE_POLICY.md"]

    # Newly created Synaptic-PA worktree AND freshly cloned DSH origin must both inherit POLICY-REV-2!
    wt_w3 = tmp_path / "wt_w3"
    subprocess.run(["git", "worktree", "add", "-b", "b_w3", str(wt_w3), "main"], cwd=ws_repo, check=True, capture_output=True)
    assert "POLICY-REV-2" in (wt_w3 / "WORKSPACE_POLICY.md").read_text(encoding="utf-8")
    assert "POLICY-REV-2" in (wt_w1 / "WORKSPACE_POLICY.md").read_text(encoding="utf-8")

    fresh_from_dsh_origin = tmp_path / "fresh_dsh_wt"
    subprocess.run(["git", "clone", str(dsh_paths["bare_repo"]), str(fresh_from_dsh_origin)], check=True, capture_output=True)
    assert "POLICY-REV-2" in (fresh_from_dsh_origin / "WORKSPACE_POLICY.md").read_text(encoding="utf-8")

    # Add w3 implementation in wt_w3 (creating a brand-new helper module imported by mod.py or non-overlapping edit)
    (wt_w3 / "w3_impl.py").write_text("def w3_val() -> int:\n    return 30\n", encoding="utf-8")
    (wt_w2 / "mod.py").write_text(
        (wt_w2 / "mod.py").read_text(encoding="utf-8") + "\ndef w3() -> int:\n    from w3_impl import w3_val\n    return w3_val()\n",
        encoding="utf-8",
    )

    ep = EpisodeSpec(
        episode_id="TEST_H4_01",
        track_id="H-4",
        domain="hybrid",
        scenario_type="long_horizon_evolution",
        title="Test H-4 3-wave + policy",
        upstream_pillars=["cooperbench"],
        upstream_task_ids=["cb::1", "cb::2", "cb::3"],
        workspace_mounts=[
            WorkspaceMountSpec(
                mount_id="mnt_repo",
                pillar="cooperbench",
                mount_relpath="repo",
                upstream_id="cb_3wave",
                repo_or_category="local/mod",
                base_commit=base_commit,
            )
        ],
        timeline=[ev_w3],
        verifiers=[
            VerifierSpec(
                verifier_id="v_3wave",
                kind="cooperbench_3wave_merged_and_policy_check",
                mount_id="mnt_repo",
                check_params={
                    "wave_tags": ["wave_1", "wave_2", "wave_3"],
                    "tests_patches": [str(p) for p in patches],
                    "required_policy_tag": "POLICY-REV-2",
                    "forbidden_stale_phrases": ["仍是 Revision 1"],
                },
            )
        ],
    )
    ws = MaterializedEpisodeWorkspace(
        episode=ep,
        arm_name="synaptic_pa",
        root_dir=tmp_path,
        repo_dir=ws_repo,
        mount_paths={"mnt_repo": ws_repo},
        shared_symlinks={".venv": PROJECT_ROOT / ".venv"},
    )

    # Stale Revision-1 claim must FAIL even when all 3 waves merge & pass pytest!
    bad_art = RuntimeArtifacts(
        arm_name="synaptic_pa",
        task_worktrees={"wave_1": wt_w1, "wave_2": wt_w2, "wave_3": wt_w3},
        all_worktrees=[wt_w1, wt_w2, wt_w3],
        agent_replies=["Wave 3 完成，当前文件仍是 Revision 1 (POLICY-REV-2 未生效)。"],
    )
    rep_bad = evaluate_episode_verifiers(ep, ws, bad_art)
    assert rep_bad["episode_passed"] is False
    assert rep_bad["verifiers"][0]["all_waves_pytest_ok"] is True
    assert rep_bad["verifiers"][0]["policy_rev2_ok"] is False

    good_art = RuntimeArtifacts(
        arm_name="synaptic_pa",
        task_worktrees={"wave_1": wt_w1, "wave_2": wt_w2, "wave_3": wt_w3},
        all_worktrees=[wt_w1, wt_w2, wt_w3],
        agent_replies=["Wave 3 已按照最新规范 POLICY-REV-2 完成验证并通过测试。"],
    )
    rep_good = evaluate_episode_verifiers(ep, ws, good_art)
    assert rep_good["episode_passed"] is True, rep_good


def test_known2_3_7_14_dsh_sqlite_quiescence_and_zstd_tool_telemetry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`Known-2, 3, 7, 14`: DSH trigger SHA-256 parity, SQLite filtered-trigger fallback, passive `notice` quiescence, and `.zstd` tool counts."""
    import sqlite3
    from synaptic_bench_150.runner import _match_dsh_trigger
    from dsh_runner.dsh_controller import GenuineDSHSuperagent, dsh_key, dsh_trigger_id_for_injected

    inj_fs = {"channel": "feishu", "message_id": "om_pa150_0001"}
    inj_gh = {"channel": "github_issue", "issue_number": 7}
    tid_fs = dsh_trigger_id_for_injected(inj_fs)
    tid_gh = dsh_trigger_id_for_injected(inj_gh)
    assert tid_fs == dsh_key("feishu", "om_pa150_0001") and len(tid_fs) == 32
    assert tid_gh == dsh_key("github-issue", "7") and len(tid_gh) == 32

    dsh = GenuineDSHSuperagent(instance_dir=tmp_path / "dsh_inst", port=19999)
    dsh.state_dir.mkdir(parents=True, exist_ok=True)
    db_fp = dsh.state_dir / "orchestrator.sqlite"
    conn = sqlite3.connect(str(db_fp))
    conn.execute("CREATE TABLE trigger (id TEXT PRIMARY KEY, value TEXT)")
    conn.execute("CREATE TABLE disposition (id TEXT PRIMARY KEY, value TEXT)")
    conn.execute("CREATE TABLE decision (id TEXT PRIMARY KEY, value TEXT)")
    conn.execute("CREATE TABLE ticket (id TEXT PRIMARY KEY, active INTEGER, value TEXT)")
    conn.execute("CREATE TABLE thread (id TEXT PRIMARY KEY, active INTEGER, value TEXT)")
    conn.execute(
        "INSERT INTO trigger VALUES (?, ?)",
        (tid_fs, json.dumps({"id": tid_fs, "source": "feishu", "at": "2026-11-09T10:00:00Z", "text": "lunch?"})),
    )
    conn.execute(
        "INSERT INTO disposition VALUES (?, ?)",
        (tid_fs, json.dumps({"filtered": "no_bot_mention", "fanout": [], "handbacks": []})),
    )
    conn.execute(
        "INSERT INTO trigger VALUES (?, ?)",
        (tid_gh, json.dumps({"id": tid_gh, "source": "github", "at": "2026-11-09T10:00:01Z", "text": "Fix bug"})),
    )
    conn.execute(
        "INSERT INTO ticket VALUES (?, 0, ?)",
        ("T-1", json.dumps({"id": "T-1", "origin": {"triggerId": tid_gh}, "inputs": [{"kind": "trigger", "triggerId": tid_gh}]})),
    )
    conn.commit()
    conn.close()

    def _fake_api(method: str, path: str, body: object = None, timeout: float = 5.0) -> dict[str, object]:
        if path.startswith("threads"):
            return {"items": []}  # Filtered triggers have NO Thread row!
        if path.startswith("tickets?"):
            return {"items": []}
        if path == "state":
            # Passive `notice` effect stays non-empty while idle (`effects.ts:20`) — must NOT cause false timeout!
            return {
                "health": {
                    "runtime": {"state": "ready"},
                    "notice": {"detail": "2 ticket notices waiting for next wake"},
                    "assessment": {"detail": "Nothing waiting"},
                    "launch": {"detail": "Nothing waiting"},
                    "delivery": {"detail": "Nothing waiting"},
                    "relay": {"detail": "Nothing waiting"},
                    "reply": {"detail": "Nothing waiting"},
                }
            }
        if path == "overview":
            return {"flow": {"received": 2, "evaluating": 0}, "executor": {"running": 0, "queued": 0}}
        return {}

    monkeypatch.setattr(dsh, "_api", _fake_api)

    triggers = dsh.get_triggers()
    claimed: set[str] = set()
    ev_noise = TimelineEventSpec(
        event_id="ev_noise",
        seq=1,
        kind="external_noise",
        channel="feishu://group/team",
        sender="bob",
        text="lunch?",
        trigger=TriggerSpec(mode="immediate"),
    )
    matched = _match_dsh_trigger(ev_noise, inj_fs, triggers, claimed)
    assert matched is not None and matched["id"] == tid_fs
    assert matched["disposition"]["filtered"] == "no_bot_mention"

    tks = dsh.get_all_tickets_detailed()
    assert len(tks) == 1 and tks[0]["inputs"][0]["triggerId"] == tid_gh

    q_ok = dsh.wait_until_quiescent(
        expected_received=2,
        expected_trigger_ids={tid_fs, tid_gh},
        min_wait_s=0.01,
        idle_streak_s=0.05,
        timeout_s=2.5,
    )
    assert q_ok["timed_out"] is False
    assert q_ok["busy_effects"] == []

    # Compress a synthetic worker session `.zstd` log to verify `count_worker_tool_calls` & `inspect_worker_office_reads`
    sess_dir = dsh.dsh_home / "sessions" / "sess_w1"
    sess_dir.mkdir(parents=True, exist_ok=True)
    raw_jsonl = sess_dir / "session.v4.jsonl"
    raw_jsonl.write_text(
        json.dumps({"type": "tool/call", "data": {"name": "bash", "args": {"cmd": "cat ../office_fs/release_spec_addendum.md"}}}) + "\n"
        + json.dumps({"type": "tool/result", "data": {"ok": True, "output": "PA150-SPEC-01"}}) + "\n"
        + json.dumps({"type": "tool/result", "data": {"ok": True, "output": "pytest passed"}}) + "\n",
        encoding="utf-8",
    )
    subprocess.run(["zstd", "-q", "-f", str(raw_jsonl), "-o", str(sess_dir / "session.v4.jsonl.zstd")], check=True)
    raw_jsonl.unlink()

    assert dsh.count_worker_tool_calls(session_ids={"sess_w1"}) == 2
    assert dsh.count_worker_tool_calls(session_ids={"nonexistent"}) == 0
    assert dsh.inspect_worker_office_reads() is True


def test_known6_11_12_13_15_and_lpt_scheduler(tmp_path: Path) -> None:
    """`Known-6, 11, 12, 13, 15` + LPT queue ordering: worktree dirty hash, decoupled RO metrics, `0/0 -> None`, cross-modal rigor, forensic bundle."""
    from synaptic_bench_150.runner import (
        _detect_synaptic_office_inspection,
        _write_forensic_bundle,
        aggregate_benchmark_results,
        sort_jobs_lpt,
    )

    # 1. `Known-6`: `hash_git_repo_state` detects dirty source mutations inside a spawned git worktree
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "t@local"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Tester"], cwd=repo, check=True, capture_output=True)
    (repo / "app.py").write_text("X = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True, capture_output=True)

    snap_clean = hash_git_repo_state(repo)
    wt = tmp_path / "spawned_wt"
    subprocess.run(["git", "worktree", "add", "-b", "wt_branch", str(wt), "main"], cwd=repo, check=True, capture_output=True)
    (wt / "app.py").write_text("X = 999\n", encoding="utf-8")
    snap_dirty_wt = hash_git_repo_state(repo)
    assert snap_dirty_wt["repo_sha256"] != snap_clean["repo_sha256"]
    assert len(snap_dirty_wt["dirty_worktrees"]) == 1

    # 2. `Known-11` + `Known-12` + `Known-13`: RO worktree decoupling, `0/0 -> None`, and cross-modal consistency
    ep = EpisodeSpec(
        episode_id="TEST_CM_RO",
        track_id="H-3",
        domain="hybrid",
        scenario_type="mid_execution_amendment",
        title="Test RO + Cross-Modal",
        upstream_pillars=["cooperbench", "mcpmark_filesystem"],
        upstream_task_ids=["cb::1"],
        workspace_mounts=[
            WorkspaceMountSpec(
                mount_id="mnt_repo",
                pillar="cooperbench",
                mount_relpath="repo",
                upstream_id="cb::1",
                repo_or_category="local/app",
            )
        ],
        timeline=[],
        verifiers=[
            VerifierSpec(verifier_id="v_ro", kind="readonly_zero_mutation_check", mount_id="mnt_repo", check_params={"after_event_id": "turn_1b", "max_worktrees_before_commit": 0}),
            VerifierSpec(verifier_id="v_cm_o2c", kind="cross_modal_consistency_check", mount_id="mnt_repo", check_params={"direction": "office_to_code", "spec_id": "PA150-SPEC-01"}),
            VerifierSpec(verifier_id="v_cm_c2o", kind="cross_modal_consistency_check", mount_id="mnt_repo", check_params={"direction": "code_to_office_brief"}),
        ],
    )
    ws = MaterializedEpisodeWorkspace(
        episode=ep,
        arm_name="dsh_superagent",
        root_dir=tmp_path,
        repo_dir=repo,
        mount_paths={"mnt_repo": repo},
        initial_snapshots={"mnt_repo": snap_clean},
        mid_turn_snapshots={("turn_1b", "mnt_repo"): {**snap_clean, "spawned_worktrees": 1}},  # unmodified repo, 1 premature clean worktree
    )
    art_fail_cm = RuntimeArtifacts(
        arm_name="dsh_superagent",
        all_worktrees=[wt],
        agent_replies=["Generic brief with no file path or test status"],
        telemetry={"worktrees_created": 1, "office_mount_inspected": False},
    )
    rep1 = evaluate_episode_verifiers(ep, ws, art_fail_cm)
    # `readonly_zero_mutation_check` passes (`unmodified=True`) while recording `worktree_budget_ok=False`!
    assert rep1["verifiers"][0]["passed"] is True
    assert rep1["verifiers"][0]["worktree_budget_ok"] is False
    # Both cross-modal checks fail without office inspection / modified-file + verification evidence
    assert rep1["verifiers"][1]["passed"] is False
    assert rep1["verifiers"][2]["passed"] is False
    # `task_solve_total == 0` -> `task_solve_score is None` (NOT `1.0`)
    assert rep1["task_solve_score"] is None

    agg = aggregate_benchmark_results([{**rep1, "domain": "hybrid", "telemetry": art_fail_cm.telemetry}])
    arm_s = agg["arms"]["dsh_superagent"]
    assert arm_s["tier1_orchestration_safety"]["readonly_zero_mutation_rate"] == 1.0
    assert arm_s["tier1_orchestration_safety"]["zero_premature_worktree_rate"] == 0.0
    assert arm_s["tier2_task_solve"]["verifier_pass_rate"] is None
    assert arm_s["by_domain"]["hybrid"]["task_solve_pass_rate"] is None

    # 3. `Known-14` (`_detect_synaptic_office_inspection` via `mmu.bus.footprints`) & `Known-15` (`_write_forensic_bundle`)
    mmu = ContextMMU(
        sandbox_paths={"world": tmp_path, "repo": repo, "worktrees": tmp_path / "wts", "docs": tmp_path},
        judge=System1Judge(typesafe_api_key="", deepseek_api_key=""),
        llm=HomogeneousLLMClient(api_key=""),
        trunk=ElasticSharedTrunk(),
        dsh_host=FakeDSHHost(),
    )
    from clonact.synaptic_bus import CellFootprint

    assert _detect_synaptic_office_inspection(mmu) is False
    mmu.bus.register_or_update_cell(CellFootprint(cell_id="Cell_1", topic="test", status="detached_executing"))
    mmu.bus.record_uri_touch(
        "Cell_1",
        {"doc://office_fs/release_spec_addendum.md"},
        is_write=False,
        intent_one_liner="read addendum",
    )
    assert _detect_synaptic_office_inspection(mmu) is True

    bundle_fp = _write_forensic_bundle(tmp_path / "workspaces", ep, "synaptic_pa", {"episode_id": ep.episode_id, "ok": True})
    assert bundle_fp == tmp_path / "forensic" / "TEST_CM_RO__synaptic_pa.json"
    assert json.loads(bundle_fp.read_text(encoding="utf-8"))["ok"] is True

    # 4. LPT job sorting: `dsh_superagent` before `synaptic_pa`, and `H-4` before `O-3`
    ep_o3 = EpisodeSpec(
        episode_id="PA150_O3_01", track_id="O-3", domain="office", scenario_type="mid", title="o3",
        upstream_pillars=["mcpmark_filesystem"], upstream_task_ids=["m1"], workspace_mounts=[], timeline=[], verifiers=[],
    )
    ep_h4 = EpisodeSpec(
        episode_id="PA150_H4_01", track_id="H-4", domain="hybrid", scenario_type="long", title="h4",
        upstream_pillars=["cooperbench"], upstream_task_ids=["c1"], workspace_mounts=[], timeline=[], verifiers=[],
    )
    ordered = sort_jobs_lpt([(ep_o3, "synaptic_pa"), (ep_o3, "dsh_superagent"), (ep_h4, "synaptic_pa"), (ep_h4, "dsh_superagent")])
    assert [(e.episode_id, a) for e, a in ordered] == [
        ("PA150_H4_01", "dsh_superagent"),
        ("PA150_O3_01", "dsh_superagent"),
        ("PA150_H4_01", "synaptic_pa"),
        ("PA150_O3_01", "synaptic_pa"),
    ]


def test_new2_10_11_12_13_14_15_evaluator_and_runner_regressions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Comprehensive regression suite covering `NEW-2`, `NEW-10`, `NEW-11`, `NEW-12`, `NEW-13`, `NEW-14`, and `NEW-15`."""
    import argparse
    import asyncio
    import time

    from synaptic_bench_150 import runner as runner_mod
    from synaptic_bench_150.catalog_loader import normalize_swe_pytest_target
    from synaptic_bench_150.workspace_manager import (
        PG_DEFAULT_PORT,
        create_isolated_postgres_db,
        drop_postgres_db,
        episode_touches_pg_cluster_roles,
        hash_postgres_database,
        materialize_episode_workspace,
    )

    # 1. NEW-2: `normalize_swe_pytest_target` cleans `PASSED `, ` <- `, dotted modules, missing `tests/`, and count lines
    assert normalize_swe_pytest_target("PASSED tests/test_plugins.py::test_ok") == ("tests/test_plugins.py::test_ok", None)
    assert normalize_swe_pytest_target("test_graph.py::test_deg <- networkx/classes/tests/test_graph.py") == (
        "networkx/classes/tests/test_graph.py::test_deg",
        None,
    )
    assert normalize_swe_pytest_target("tests.plugins.test_twitch::test_multivariant") == (
        "tests/plugins/test_twitch.py::test_multivariant",
        None,
    )
    assert normalize_swe_pytest_target(
        "test.autotag.test_distance.TestTrackDistance::test_dist[id]"
    ) == ("test/autotag/test_distance.py::TestTrackDistance::test_dist[id]", None)
    assert normalize_swe_pytest_target(
        "test_column_types.py::test_col", patched_files=["tests/test_column_types.py"]
    ) == ("tests/test_column_types.py::test_col", None)
    assert normalize_swe_pytest_target("====== 2202") == (None, None)
    assert normalize_swe_pytest_target("7032") == (None, None)

    # 2. NEW-13: `cooperbench_merged_pytest` edge cases (unmerged single tree vs agent-resolved conflict in trunk)
    repo = tmp_path / "cb_edge_repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "t@local"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Tester"], cwd=repo, check=True, capture_output=True)
    (repo / "mod.py").write_text("HEADER = 'v0'\n", encoding="utf-8")
    (repo / "tests").mkdir()
    (repo / "tests" / "test_mod.py").write_text("import mod\ndef test_v0(): assert mod.HEADER\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=repo, check=True, capture_output=True)
    base_sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()

    pa = tmp_path / "pa.patch"
    pb = tmp_path / "pb.patch"
    (repo / "tests" / "test_a.py").write_text("from mod import fn_a\ndef test_a(): assert fn_a() == 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "tests/test_a.py"], cwd=repo, check=True, capture_output=True)
    pa.write_text(subprocess.run(["git", "diff", "--cached"], cwd=repo, capture_output=True, text=True).stdout, encoding="utf-8")
    subprocess.run(["git", "reset", "--hard", base_sha], cwd=repo, check=True, capture_output=True)

    (repo / "tests" / "test_b.py").write_text("from mod import fn_b\ndef test_b(): assert fn_b() == 2\n", encoding="utf-8")
    subprocess.run(["git", "add", "tests/test_b.py"], cwd=repo, check=True, capture_output=True)
    pb.write_text(subprocess.run(["git", "diff", "--cached"], cwd=repo, capture_output=True, text=True).stdout, encoding="utf-8")
    subprocess.run(["git", "reset", "--hard", base_sha], cwd=repo, check=True, capture_output=True)

    ep_cb = EpisodeSpec(
        episode_id="TEST_NEW13",
        track_id="C-2",
        domain="coding",
        scenario_type="high_concurrency_burst",
        title="Edge CB",
        upstream_pillars=["cooperbench"],
        upstream_task_ids=["cb::a", "cb::b"],
        workspace_mounts=[
            WorkspaceMountSpec(
                mount_id="mnt_repo",
                pillar="cooperbench",
                mount_relpath="repo",
                upstream_id="cb_pair",
                repo_or_category="local/mod",
                base_commit=base_sha,
            )
        ],
        timeline=[],
        verifiers=[
            VerifierSpec(
                verifier_id="v_m",
                kind="cooperbench_merged_pytest",
                mount_id="mnt_repo",
                check_params={"tests_patch_a": str(pa), "tests_patch_b": str(pb)},
            )
        ],
    )
    ws_cb = MaterializedEpisodeWorkspace(
        episode=ep_cb,
        arm_name="dsh_superagent",
        root_dir=tmp_path,
        repo_dir=repo,
        mount_paths={"mnt_repo": repo},
        shared_symlinks={".venv": PROJECT_ROOT / ".venv"},
        initial_snapshots={"mnt_repo": hash_git_repo_state(repo)},
    )

    # Edge 2a (`v2.0.0` Architecture-Neutral C-2):
    # - Partial 1-feature single-tree edit (`fn_a` without `fn_b`) -> MUST fail `merge_clean`!
    # - Complete 2-feature single-tree edit (`fn_a` AND `fn_b`) -> MUST pass (`single_tree_integrated`)!
    (repo / "mod.py").write_text("HEADER = 'v1'\ndef fn_a(): return 1\n", encoding="utf-8")
    rep_partial_tree = evaluate_episode_verifiers(ep_cb, ws_cb, RuntimeArtifacts(arm_name="dsh_superagent", all_worktrees=[]))
    assert rep_partial_tree["verifiers"][0]["functional_tests_passed"] is False
    assert rep_partial_tree["verifiers"][0]["merge_clean"] is False
    assert rep_partial_tree["episode_passed"] is False

    (repo / "mod.py").write_text("HEADER = 'v1'\ndef fn_a(): return 1\ndef fn_b(): return 2\n", encoding="utf-8")
    rep_integrated_tree = evaluate_episode_verifiers(ep_cb, ws_cb, RuntimeArtifacts(arm_name="dsh_superagent", all_worktrees=[]))
    assert rep_integrated_tree["verifiers"][0]["functional_tests_passed"] is True
    assert rep_integrated_tree["verifiers"][0]["merge_clean"] is True
    assert rep_integrated_tree["verifiers"][0]["integration_source"] == "single_tree_integrated"
    assert rep_integrated_tree["episode_passed"] is True

    # Edge 2b: Conflicting raw `wt_a` and `wt_b` on line 1, resolved cleanly in `repo` (`trunk_diff`) -> MUST pass!
    wt_a = tmp_path / "wt_a"
    wt_b = tmp_path / "wt_b"
    subprocess.run(["git", "worktree", "add", "-b", "br_a", str(wt_a), base_sha], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "worktree", "add", "-b", "br_b", str(wt_b), base_sha], cwd=repo, check=True, capture_output=True)
    (wt_a / "mod.py").write_text("HEADER = 'from_a'\ndef fn_a(): return 1\n", encoding="utf-8")
    (wt_b / "mod.py").write_text("HEADER = 'from_b'\ndef fn_b(): return 2\n", encoding="utf-8")
    rep_resolved = evaluate_episode_verifiers(
        ep_cb,
        ws_cb,
        RuntimeArtifacts(arm_name="synaptic_pa", task_worktrees={"feat_a": wt_a, "feat_b": wt_b}, all_worktrees=[wt_a, wt_b]),
    )
    assert rep_resolved["verifiers"][0]["agent_resolved_in_trunk"] is True
    assert rep_resolved["verifiers"][0]["merge_clean"] is True
    assert rep_resolved["episode_passed"] is True

    # 3. NEW-10: Crash persistence (no survivorship bias) + strict filtered/versioned `--resume`
    orig_run_synaptic = runner_mod.run_episode_synaptic_pa
    ep_crash = load_pa150_episodes(episode_ids=["PA150_O1_09"])[0]
    ep_other = load_pa150_episodes(episode_ids=["PA150_O1_10"])[0]
    out_run = tmp_path / "suite_out"
    out_run.mkdir()
    # Seed `results.jsonl` with: (a) an out-of-scope episode `PA150_O1_10`, and (b) a stale-hash `PA150_O1_09`
    stale_rows = [
        {"episode_id": ep_other.episode_id, "arm_name": "synaptic_pa", "episode_sha256": runner_mod.episode_fingerprint(ep_other), "episode_passed": True},
        {"episode_id": ep_crash.episode_id, "arm_name": "synaptic_pa", "episode_sha256": "deadbeef00000000", "episode_passed": True},
    ]
    (out_run / "results.jsonl").write_text("\n".join(json.dumps(r) for r in stale_rows) + "\n", encoding="utf-8")

    async def _boom_synaptic(*_a, **_kw):
        raise RuntimeError("simulated workspace boot failure")

    monkeypatch.setattr(runner_mod, "run_episode_synaptic_pa", _boom_synaptic)
    monkeypatch.setattr(runner_mod, "query_deepseek_balance_cny", lambda: None)

    ns = argparse.Namespace(
        episodes="PA150_O1_09",
        tracks="",
        domains="",
        limit=None,
        arms="clonact",
        concurrency=1,
        output_dir=str(out_run),
        max_spend_cny=10.0,
        gaia_judge_mode="deterministic",
        dsh_port=13099,
        resume=True,
        allow_stale_resume=False,
        keep_workspaces=False,
    )
    suite_summary = asyncio.run(runner_mod.run_benchmark_suite(ns))
    # Stale `PA150_O1_09` & out-of-scope `PA150_O1_10` were ignored; crashed `PA150_O1_09` was recorded in denominator!
    assert suite_summary["total_records"] == 1
    assert suite_summary["arms"]["clonact"]["episodes_evaluated"] == 1
    assert suite_summary["arms"]["clonact"]["runner_errors"] == 1
    assert suite_summary["arms"]["clonact"]["episodes_passed"] == 0

    # 4. NEW-11 & NEW-12: PG cluster-role detection + full DDL/function-body/index digest in `hash_postgres_database`
    all_eps = load_pa150_episodes()
    role_eps = [e.episode_id for e in all_eps if episode_touches_pg_cluster_roles(e)]
    o1_09 = load_pa150_episodes(episode_ids=["PA150_O1_09"])[0]
    assert len(role_eps) >= 1
    assert episode_touches_pg_cluster_roles(o1_09) is False

    test_db = "pa_test_new12_hash"
    try:
        pg_env = create_isolated_postgres_db(test_db, "custom_empty", task_dir=None, port=PG_DEFAULT_PORT)
        import psycopg2

        conn = psycopg2.connect(
            host=pg_env["POSTGRES_HOST"],
            port=int(pg_env["POSTGRES_PORT"]),
            database=pg_env["POSTGRES_DATABASE"],
            user=pg_env["POSTGRES_USERNAME"],
            password=pg_env["POSTGRES_PASSWORD"],
        )
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("CREATE TABLE items (id INT PRIMARY KEY, val TEXT)")
            cur.execute("CREATE INDEX idx_items_val ON items(val)")
            cur.execute("CREATE OR REPLACE FUNCTION calc_bonus(x INT) RETURNS INT LANGUAGE sql AS $$ SELECT x + 1 $$")
        h0 = hash_postgres_database(pg_env)["db_sha256"]
        with conn.cursor() as cur:
            # Mutate ONLY function body keeping identical function name & return type!
            cur.execute("CREATE OR REPLACE FUNCTION calc_bonus(x INT) RETURNS INT LANGUAGE sql AS $$ SELECT x + 999 $$")
        h1 = hash_postgres_database(pg_env)["db_sha256"]
        assert h1 != h0
        with conn.cursor() as cur:
            cur.execute("DROP INDEX idx_items_val")
        h2 = hash_postgres_database(pg_env)["db_sha256"]
        assert h2 != h1
        conn.close()
    finally:
        drop_postgres_db(test_db, PG_DEFAULT_PORT)

    # 5. NEW-14: Decoupled wall-clock burst delivery (`t=0.04s` slow turn MUST NOT delay `t=0.10s` peek!)
    arrival_log: dict[str, float] = {}
    t_burst0 = 0.0

    def _fake_materialize(ep, arm, root, **_kw):
        d = root / ep.episode_id / arm
        r = d / "repo"
        r.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "init", "-b", "main"], cwd=r, check=True, capture_output=True)
        return MaterializedEpisodeWorkspace(episode=ep, arm_name=arm, root_dir=d, repo_dir=r)

    async def _slow_handle_user(self, text: str, auto_run_detached: bool = True):
        arrival_log[text] = time.perf_counter() - t_burst0
        if "SLOW_FOLLOWUP" in text:
            await asyncio.sleep(0.35)
            return {"transition": "followup", "routed_to": "Cell_1", "reply": "done slow"}
        return {"transition": "instant_live_slice_peek", "routed_to": "Cell_1", "reply": "instant peek"}

    monkeypatch.setattr(runner_mod, "SynapticDSHWorkerHost", FakeDSHHost)
    monkeypatch.setattr(runner_mod, "materialize_episode_workspace", _fake_materialize)
    monkeypatch.setattr(ContextMMU, "handle_user_message", _slow_handle_user)

    ep_burst = EpisodeSpec(
        episode_id="TEST_BURST_CLOCK",
        track_id="O-2",
        domain="office",
        scenario_type="high_concurrency_burst",
        title="Burst clock test",
        upstream_pillars=["mcpmark_filesystem"],
        upstream_task_ids=["m1"],
        workspace_mounts=[],
        timeline=[
            TimelineEventSpec(
                event_id="ev_slow",
                seq=1,
                kind="user_chat",
                channel="ui://chat/main",
                sender="user",
                text="SLOW_FOLLOWUP",
                trigger=TriggerSpec(mode="at_offset_sec", offset_sec=0.04),
            ),
            TimelineEventSpec(
                event_id="ev_peek",
                seq=2,
                kind="user_chat",
                channel="ui://chat/main",
                sender="user",
                text="FAST_PEEK",
                trigger=TriggerSpec(mode="at_offset_sec", offset_sec=0.10),
                expected_mmu_behavior="instant_status_peek_0ms",
            ),
        ],
        verifiers=[],
    )
    t_burst0 = time.perf_counter()
    res_burst = asyncio.run(orig_run_synaptic(ep_burst, tmp_path / "burst_ws"))
    assert arrival_log["FAST_PEEK"] < 0.25, f"Burst peek delayed behind slow turn: {arrival_log}"
    assert res_burst["event_trace"][1]["actual_offset_sec"] < 0.25
    assert len(res_burst["telemetry"]["peek_latencies_ms"]) == 1

    # 6. NEW-5: Verify `PA150_H3_03` (`dba_vector_analysis` + `numpy` + `pgvector`) materializes cleanly offline
    h3_03 = load_pa150_episodes(episode_ids=["PA150_H3_03"])[0]
    ws_h303 = materialize_episode_workspace(h3_03, "synaptic_pa", tmp_path / "h303_ws")
    try:
        assert len(ws_h303.mount_pg_envs) == 1
        pg_mid = next(iter(ws_h303.mount_pg_envs))
        assert ws_h303.initial_snapshots[pg_mid].get("db_sha256")
    finally:
        ws_h303.cleanup(remove_files=True)


def test_p0_p1_v130_regressions(tmp_path: Path) -> None:
    """Regression suite for v1.3.0 audit findings `P0-1 .. P0-6`, `P1-1`, `P1-4`, and `P1-6`."""
    import asyncio
    import os
    from concurrent.futures import ThreadPoolExecutor
    from synaptic_bench_150.catalog_loader import normalize_swe_pytest_target
    from synaptic_bench_150.runner import load_pa150_episodes
    from synaptic_bench_150.workspace_manager import (
        PG_DEFAULT_PORT,
        cleanup_postgres_cluster_roles,
        create_isolated_postgres_db,
        drop_postgres_db,
        hash_git_repo_state,
        materialize_episode_workspace,
    )
    from clonact.system1_judge import JudgmentResult, System1Judge

    # 1. P1-4: `normalize_swe_pytest_target` keeps LHS collector module on `<-` nodes
    node_id, k_expr = normalize_swe_pytest_target(
        "networkx/classes/tests/test_special.py::TestThinDiGraph::test_has_node <- networkx/classes/tests/test_graph.py"
    )
    assert node_id == "networkx/classes/tests/test_special.py::TestThinDiGraph::test_has_node"
    assert k_expr is None

    # 2. P0-1: `init_dsh_sandbox(..., seed_repo_dir=...)` skips no-op commit and `extra_worktrees_base_head` prevents false dirty worktrees
    seed = tmp_path / "seed_p01"
    seed.mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=seed, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "t@local"], cwd=seed, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Tester"], cwd=seed, check=True, capture_output=True)
    (seed / "AGENTS.md").write_text("# Same Policy\n", encoding="utf-8")
    (seed / "gaia_cli").write_text("#!/bin/sh\necho ok\n", encoding="utf-8")
    (seed / "gaia_cli").chmod(0o755)
    subprocess.run(["git", "add", "."], cwd=seed, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=seed, check=True, capture_output=True)
    base_sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=seed, capture_output=True, text=True).stdout.strip()

    paths_same = init_dsh_sandbox(
        tmp_path / "dsh_same",
        agents_md_override="# Same Policy\n",
        extra_repo_files={"gaia_cli": ("#!/bin/sh\necho ok\n", 0o755)},
        seed_repo_dir=seed,
    )
    clone_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=paths_same["clone_repo"], capture_output=True, text=True
    ).stdout.strip()
    assert clone_sha == base_sha

    paths_diff = init_dsh_sandbox(
        tmp_path / "dsh_diff",
        agents_md_override="# Different DSH Policy\n",
        extra_repo_files={"gaia_cli": ("#!/bin/sh\necho ok\n", 0o755)},
        seed_repo_dir=seed,
    )
    clone_diff_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=paths_diff["clone_repo"], capture_output=True, text=True
    ).stdout.strip()
    assert clone_diff_sha != base_sha
    wt_root = paths_diff["world"] / "worktrees"
    wt_root.mkdir(parents=True, exist_ok=True)
    dsh_wt = wt_root / "wt_peek"
    subprocess.run(
        ["git", "worktree", "add", "-b", "peek", str(dsh_wt), clone_diff_sha],
        cwd=paths_diff["clone_repo"],
        check=True,
        capture_output=True,
    )
    snap_git = hash_git_repo_state(
        seed,
        extra_worktrees_root=wt_root,
        extra_worktrees_base_head=clone_diff_sha,
    )
    assert snap_git["dirty_worktrees"] == []

    # 3. P0-6 & P1-6: All-skipped pytest MUST fail (`passed == 0`), and `C-2` with 2 empty worktrees + 1 trunk edit MUST report `merge_clean is False`
    skip_patch = tmp_path / "all_skipped.patch"
    (seed / "test_skip_only.py").write_text("import pytest\n\ndef test_skip_me():\n    pytest.skip('all skipped')\n", encoding="utf-8")
    subprocess.run(["git", "add", "test_skip_only.py"], cwd=seed, check=True, capture_output=True)
    skip_patch.write_text(
        subprocess.run(["git", "diff", "--cached"], cwd=seed, capture_output=True, text=True).stdout,
        encoding="utf-8",
    )
    subprocess.run(["git", "reset", "--hard", base_sha], cwd=seed, check=True, capture_output=True)

    wt1 = tmp_path / "wt_empty_1"
    wt2 = tmp_path / "wt_empty_2"
    subprocess.run(["git", "worktree", "add", "-b", "w1", str(wt1), base_sha], cwd=seed, check=True, capture_output=True)
    subprocess.run(["git", "worktree", "add", "-b", "w2", str(wt2), base_sha], cwd=seed, check=True, capture_output=True)
    # Edit trunk ONLY (`seed`), leaving `wt1` and `wt2` untouched!
    (seed / "trunk_only.py").write_text("X = 1\n", encoding="utf-8")

    ep_p06 = EpisodeSpec(
        episode_id="TEST_P06_P16",
        track_id="C-2",
        domain="coding",
        scenario_type="high_concurrency_burst",
        title="Skip + false merge guard",
        upstream_pillars=["cooperbench"],
        upstream_task_ids=["a", "b"],
        workspace_mounts=[
            WorkspaceMountSpec(
                mount_id="mnt_repo",
                pillar="cooperbench",
                mount_relpath="repo",
                upstream_id="cb",
                repo_or_category="local/pkg",
                base_commit=base_sha,
            )
        ],
        timeline=[],
        verifiers=[
            VerifierSpec(
                verifier_id="v_skip",
                kind="cooperbench_pytest",
                mount_id="mnt_repo",
                check_params={"tests_patch": str(skip_patch)},
            ),
            VerifierSpec(
                verifier_id="v_merge",
                kind="cooperbench_merged_pytest",
                mount_id="mnt_repo",
                check_params={"tests_patch_a": str(skip_patch), "tests_patch_b": str(skip_patch)},
            ),
        ],
    )
    ws_p06 = MaterializedEpisodeWorkspace(
        episode=ep_p06,
        arm_name="synaptic_pa",
        root_dir=tmp_path,
        repo_dir=seed,
        mount_paths={"mnt_repo": seed},
        shared_symlinks={".venv": PROJECT_ROOT / ".venv"},
    )
    rep_p06 = evaluate_episode_verifiers(
        ep_p06,
        ws_p06,
        RuntimeArtifacts(arm_name="synaptic_pa", task_worktrees={"a": wt1, "b": wt2}, all_worktrees=[wt1, wt2]),
    )
    assert rep_p06["verifiers"][0]["passed"] is False
    assert rep_p06["verifiers"][1]["passed"] is False
    assert rep_p06["verifiers"][1]["merge_clean"] is False

    # 4. P0-3, P0-4 & P1-1: 4-thread concurrent `CREATE DATABASE ... WITH TEMPLATE`, `.pgpass` auth when `POSTGRES_PASSWORD` is scrubbed, and role cleanup
    conc_dbs = [f"pa_conc_p03_{i}" for i in range(4)]
    conc_dirs = [tmp_path / f"pg_task_{i}" for i in range(4)]
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            envs = list(
                pool.map(
                    lambda pair: create_isolated_postgres_db(pair[0], "employees", task_dir=pair[1], port=PG_DEFAULT_PORT),
                    zip(conc_dbs, conc_dirs),
                )
            )
        assert len(envs) == 4
        # Verify `.pgpass` (`PGPASSFILE`) authenticates `psql` even when `POSTGRES_PASSWORD` & `PGPASSWORD` are scrubbed!
        scrubbed_env = dict(os.environ)
        scrubbed_env.update(envs[0])
        scrubbed_env.pop("POSTGRES_PASSWORD", None)
        scrubbed_env.pop("PGPASSWORD", None)
        assert Path(scrubbed_env["PGPASSFILE"]).exists()
        import psycopg2

        conn = psycopg2.connect(
            host=scrubbed_env["POSTGRES_HOST"],
            port=int(scrubbed_env["POSTGRES_PORT"]),
            database=scrubbed_env["POSTGRES_DATABASE"],
            user=scrubbed_env["POSTGRES_USERNAME"],
            passfile=scrubbed_env["PGPASSFILE"],
        )
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema NOT IN ('pg_catalog', 'information_schema')")
            assert cur.fetchone()[0] > 0
            cur.execute("DROP ROLE IF EXISTS pa150_temp_leaked_role")
            cur.execute("CREATE ROLE pa150_temp_leaked_role LOGIN PASSWORD 'secret'")
            cur.execute("GRANT USAGE ON SCHEMA public TO pa150_temp_leaked_role")
        dropped_roles = cleanup_postgres_cluster_roles(PG_DEFAULT_PORT)
        assert "pa150_temp_leaked_role" in dropped_roles
        conn.close()
    finally:
        for dbn in conc_dbs:
            drop_postgres_db(dbn, PG_DEFAULT_PORT)

    # 5. P0-5: Concurrent `ContextMMU` arrivals serialize admission so `live_cells` stays strictly causal!
    class _SlowCausalJudge(System1Judge):
        def __init__(self) -> None:
            self.call_history = []

        async def decide(self, state, questions, tag: str = "slow") -> JudgmentResult:
            await asyncio.sleep(0.08)
            active_cells = state.get("active_cells") or state.get("cells") or {}
            existing_cid = next(iter(active_cells.keys()), None)
            answers = {}
            for qid, spec in questions.items():
                if qid == "is_actionable":
                    answers[qid] = {"type": "noul", "noul": 0.92}
                elif qid in ("bind_target", "primary_route"):
                    choice = existing_cid if existing_cid else "NEW_CELL"
                    answers[qid] = {"type": "choice", "choice": choice, "confidence": 0.94, "probabilities": {choice: 0.94}}
                elif qid == "turn_granularity":
                    answers[qid] = {"type": "choice", "choice": "single_intent", "confidence": 0.95, "probabilities": {"single_intent": 0.95}}
                elif hasattr(spec, "criteria") and isinstance(spec.criteria, dict):
                    first_k = next(iter(spec.criteria.keys()))
                    answers[qid] = {"type": "choice", "choice": first_k, "confidence": 0.85, "probabilities": {first_k: 0.85}}
                else:
                    answers[qid] = {"type": "noul", "noul": 0.05}
            return JudgmentResult(answers=answers, latency_ms=80.0, backend_used="slow_stub")

    class _FastDummyLLM:
        async def chat(self, messages, tools=None, caller: str = ""):
            return {"role": "assistant", "content": "Ack"}

        async def chat_fast_utility(self, messages, caller: str, max_tokens: int = 512):
            return {"role": "assistant", "content": "{}"}

    async def _run_concurrent_mmu() -> tuple[dict, dict, dict, ContextMMU]:
        mmu = ContextMMU(
            sandbox_paths={"repo": seed, "root": tmp_path},
            judge=_SlowCausalJudge(),
            llm=_FastDummyLLM(),
            trunk=ElasticSharedTrunk(),
            dsh_host=FakeDSHHost(),
        )
        ev1 = {"event_id": "e1", "channel": "slack", "sender": "ops", "text": "URGENT: payment gateway timeout 504"}
        ev2 = {"event_id": "e2", "channel": "email", "sender": "ops", "text": "URGENT: duplicate payment gateway timeout 504"}
        t1 = asyncio.create_task(mmu.ingest_external_event(ev1, auto_run_background=False))
        await asyncio.sleep(0.01)
        t2 = asyncio.create_task(mmu.ingest_external_event(ev2, auto_run_background=False))
        await asyncio.sleep(0.01)
        t3 = asyncio.create_task(mmu.handle_user_message("FOLLOWUP: check logs on that timeout", auto_run_detached=False))
        r1, r2, r3 = await asyncio.gather(t1, t2, t3)
        return r1, r2, r3, mmu

    r1, r2, r3, mmu_conc = asyncio.run(_run_concurrent_mmu())
    assert r1["action"] == "spawned_proactive_cell"
    assert r2["action"] == "merged_into_existing" and r2["cell_id"] == r1["cell_id"]
    assert r3["routed_to"] == r1["cell_id"]
    assert r3["spawned_worker_cells"] == []
    assert len(mmu_conc.cells) == 1

    # 6. P0-2: `PA150_C1_12` (`pytest-dev/pytest`) materializes cached `.venv` without plugin conflict!
    c1_12 = load_pa150_episodes(episode_ids=["PA150_C1_12"])[0]
    ws_c112 = materialize_episode_workspace(c1_12, "synaptic_pa", tmp_path / "c112_ws")
    try:
        py_bin = ws_c112.repo_dir / ".venv" / "bin" / "python"
        ver = subprocess.run([str(py_bin), "-m", "pytest", "--version"], cwd=ws_c112.repo_dir, capture_output=True, text=True)
        assert ver.returncode == 0, ver.stderr
    finally:
        ws_c112.cleanup(remove_files=True)


def test_v140_a1_to_a6_and_paradigm_regressions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Comprehensive v1.4.0 regression suite covering `A1..A6` and Paradigm-Alignment Updates 1..3."""
    import asyncio
    import time
    import urllib.request

    from synaptic_bench_150.runner import (
        EVALUATOR_SCHEMA_VERSION,
        _TrackedTaskDict,
        _cell_executed_gaia_cli,
        _collect_synaptic_office_read_evidence,
        _select_synaptic_gaia_replies,
        aggregate_benchmark_results,
    )
    from synaptic_bench_150.verifiers import (
        _candidate_worktrees_for_tag,
        _pytest_env,
        _select_prioritized_p2p_targets,
    )
    from clonact.context_mmu import UnifiedStreamItem

    assert EVALUATOR_SCHEMA_VERSION == "2.0.1"

    # ------------------------------------------------------------------
    # 1. `A1`: Gaia2 paused clock + Turn-0 premature AUI buffering + VoteNet attribution guard
    # ------------------------------------------------------------------
    samples = json.loads((PROJECT_ROOT / "data" / "upstream_pillars" / "samples" / "gaia2_samples.json").read_text())
    sample_row = samples["execution"][0]
    row_id = str(sample_row["id"])
    scenario_id = str(sample_row["scenario_id"])

    diag_secret = "unit_test_v140_a1_secret"
    srv = Gaia2ServerProcess(port=19037, log_path=tmp_path / "gaia2_v140.log", diag_token=diag_secret)
    try:
        info = srv.start_scenario(
            upstream_id=f"gaia2::execution::{row_id}",
            config="execution",
            scenario_id=scenario_id,
            judge_mode="deterministic",
        )
        assert info["ok"] is True
        cli_fp = tmp_path / "gaia_cli"
        cli_fp.write_text(build_gaia_cli_script(srv.base_url), encoding="utf-8")
        cli_fp.chmod(0o755)

        # Sleep 1.2s while server is idle (simulating LLM thinking wall-clock time): simulated clock MUST stay paused!
        time.sleep(1.2)
        idle_snap = srv.snapshot()
        assert idle_snap["simulated_time_passed_sec"] < 0.05, f"Simulated clock drifted while idle: {idle_snap}"

        # Premature Turn-0 `AgentUserInterface__send_message_to_user` BEFORE real oracle tool mutations!
        early_aui = subprocess.run(
            [
                str(cli_fp),
                "call",
                "AgentUserInterface__send_message_to_user",
                json.dumps({"content": "Working on your request right now."}),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        assert early_aui.returncode == 0
        mid_aui_snap = srv.snapshot()
        assert mid_aui_snap["buffered_aui_count"] == 1
        assert mid_aui_snap["mutating_call_count"] == 0
        assert mid_aui_snap["unmodified_since_start"] is True

        # Replay all oracle tool calls AFTER the premature AUI call + another idle sleep
        diag_req = urllib.request.Request(
            f"{srv.base_url}/oracle_actions",
            headers={"x-pa150-diag-token": diag_secret},
        )
        with urllib.request.urlopen(diag_req, timeout=10.0) as resp:
            oracle_acts = json.loads(resp.read().decode("utf-8"))["oracle_actions"]
        time.sleep(0.5)
        for act in oracle_acts:
            subprocess.run(
                [str(cli_fp), "call", act["tool_name"], json.dumps(act["args"])],
                check=True,
                capture_output=True,
                text=True,
            )

        ep_gaia = EpisodeSpec(
            episode_id="TEST_V140_GAIA_A1",
            track_id="O-2",
            domain="office",
            scenario_type="high_concurrency_burst",
            title="Test A1 Turn-0 AUI + Paused Clock",
            upstream_pillars=["gaia2"],
            upstream_task_ids=[f"gaia2::execution::{row_id}"],
            workspace_mounts=[
                WorkspaceMountSpec(
                    mount_id="mnt_gaia",
                    pillar="gaia2",
                    mount_relpath="gaia2_universe",
                    upstream_id=f"gaia2::execution::{row_id}",
                    repo_or_category=scenario_id,
                    setup_meta={"config": "execution"},
                )
            ],
            timeline=[],
            verifiers=[
                VerifierSpec(
                    verifier_id="v_oracle",
                    kind="gaia2_oracle_check",
                    mount_id="mnt_gaia",
                    check_params={"scenario_id": scenario_id},
                )
            ],
        )
        ws_gaia = MaterializedEpisodeWorkspace(
            episode=ep_gaia,
            arm_name="synaptic_pa",
            root_dir=tmp_path,
            repo_dir=tmp_path,
            gaia_server=srv,
            gaia_mount_id="mnt_gaia",
        )
        gaia_art = RuntimeArtifacts(arm_name="synaptic_pa", gaia_task_replies=["Completed all calendar and email updates."])
        rep_gaia = evaluate_episode_verifiers(
            ep_gaia,
            ws_gaia,
            gaia_art,
        )
        assert rep_gaia["episode_passed"] is True, rep_gaia
        assert rep_gaia["verifiers"][0]["judge_mode"] == "deterministic"
        assert any(c.get("combined_turn0_aui") for c in gaia_art.telemetry.get("gaia_call_log", []))
    finally:
        srv.stop()

    # VoteNet (`PA150_O2_14`) `ls -la` output mentioning `gaia_cli` MUST NOT hijack `gaia_task_replies`!
    dummy_repo = tmp_path / "dummy_repo"
    dummy_repo.mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=dummy_repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "t@local"], cwd=dummy_repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Tester"], cwd=dummy_repo, check=True, capture_output=True)
    (dummy_repo / "app.py").write_text("VAL = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=dummy_repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=dummy_repo, check=True, capture_output=True)

    mmu_attr = ContextMMU(
        sandbox_paths={"world": tmp_path, "repo": dummy_repo, "worktrees": tmp_path / "wts", "docs": tmp_path},
        judge=System1Judge(typesafe_api_key="", deepseek_api_key=""),
        llm=HomogeneousLLMClient(api_key=""),
        trunk=ElasticSharedTrunk(),
        dsh_host=FakeDSHHost(),
    )
    c_gaia = ContextCell("Cell_E_1", "gaia task", "external_event", mmu_attr.sandbox_paths, mmu_attr.trunk, mmu_attr.llm, mmu_attr.bus, dsh_host=FakeDSHHost())
    c_fs = ContextCell("Cell_E_2", "votenet fs", "external_event", mmu_attr.sandbox_paths, mmu_attr.trunk, mmu_attr.llm, mmu_attr.bus, dsh_host=FakeDSHHost())
    c_noise = ContextCell("Cell_E_3", "noise", "external_event", mmu_attr.sandbox_paths, mmu_attr.trunk, mmu_attr.llm, mmu_attr.bus, dsh_host=FakeDSHHost())
    c_gaia._step_audit_trail.append({"tool": "bash", "cmd_tail": "./gaia_cli call Calendar__get_events '{}'", "apis": ["cli:gaia_cli"], "status": "ok"})
    c_fs._step_audit_trail.append({"tool": "bash", "cmd_tail": "ls -la", "out_excerpt": "lrwxr-xr-x gaia_cli -> ../gaia_cli\ndrwxr-xr-x office_fs", "status": "ok"})
    mmu_attr.cells = {"Cell_E_1": c_gaia, "Cell_E_2": c_fs, "Cell_E_3": c_noise}
    mmu_attr.unified_stream = [
        UnifiedStreamItem("agent_reply", "Cell_E_1", "Gaia2 task reply ONLY", {}),
        UnifiedStreamItem("agent_reply", "Cell_E_2", "VoteNet tally reply from office_fs", {}),
        UnifiedStreamItem("agent_reply", "Cell_E_3", "Noise proactive reply", {}),
    ]
    assert _cell_executed_gaia_cli(c_gaia) is True
    assert _cell_executed_gaia_cli(c_fs) is False
    selected_gaia = _select_synaptic_gaia_replies(
        mmu_attr,
        {"task_gaia": ["Cell_E_1"], "task_fs": ["Cell_E_2"]},
        noise_cids={"Cell_E_3"},
    )
    assert selected_gaia == ["Gaia2 task reply ONLY"]

    # ------------------------------------------------------------------
    # 2. `A2`: `H-4` multi-cell worktree ranking + 3-wave `+2` harness commit conflict check + `POLICY-REV-<num>` guard
    # ------------------------------------------------------------------
    h4_seed = tmp_path / "h4_seed"
    h4_seed.mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=h4_seed, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "t@local"], cwd=h4_seed, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Tester"], cwd=h4_seed, check=True, capture_output=True)
    (h4_seed / "shared.py").write_text("LINE = 'base'\n", encoding="utf-8")
    (h4_seed / "tests").mkdir()
    (h4_seed / "tests" / "test_base.py").write_text("import shared\ndef test_b(): assert shared.LINE\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=h4_seed, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=h4_seed, check=True, capture_output=True)
    h4_base = subprocess.run(["git", "rev-parse", "HEAD"], cwd=h4_seed, capture_output=True, text=True).stdout.strip()

    h4_patches: list[Path] = []
    for idx, wfn in enumerate(("wave1_fn", "wave2_fn", "wave3_fn"), start=1):
        tf = h4_seed / "tests" / f"test_w{idx}.py"
        tf.write_text(f"from shared import {wfn}\ndef test_w{idx}(): assert {wfn}() == {idx}\n", encoding="utf-8")
        subprocess.run(["git", "add", str(tf.relative_to(h4_seed))], cwd=h4_seed, check=True, capture_output=True)
        pf = tmp_path / f"h4_w{idx}.patch"
        pf.write_text(subprocess.run(["git", "diff", "--cached"], cwd=h4_seed, capture_output=True, text=True).stdout, encoding="utf-8")
        h4_patches.append(pf)
        subprocess.run(["git", "reset", "--hard", h4_base], cwd=h4_seed, check=True, capture_output=True)

    # Simulate `+2` harness commits (`AGENTS.md` + `WORKSPACE_POLICY.md`) on `main` above `h4_base`!
    h4_repo = tmp_path / "h4_repo"
    subprocess.run(["git", "clone", str(h4_seed), str(h4_repo)], check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "t@local"], cwd=h4_repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Tester"], cwd=h4_repo, check=True, capture_output=True)
    (h4_repo / "AGENTS.md").write_text("# Harness instructions\n", encoding="utf-8")
    subprocess.run(["git", "add", "AGENTS.md"], cwd=h4_repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "setup agents_md"], cwd=h4_repo, check=True, capture_output=True)
    (h4_repo / "WORKSPACE_POLICY.md").write_text("# POLICY-REV-2\n", encoding="utf-8")
    subprocess.run(["git", "add", "WORKSPACE_POLICY.md"], cwd=h4_repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "update policy rev2"], cwd=h4_repo, check=True, capture_output=True)

    wt_w1 = tmp_path / "h4_wt_w1"
    wt_w2_office_empty = tmp_path / "h4_wt_w2_office_empty"
    wt_w2_code = tmp_path / "h4_wt_w2_code"
    wt_w3 = tmp_path / "h4_wt_w3"
    for bname, wtp in (("bw1", wt_w1), ("bw2_off", wt_w2_office_empty), ("bw2_code", wt_w2_code), ("bw3", wt_w3)):
        subprocess.run(["git", "worktree", "add", "-b", bname, str(wtp), "main"], cwd=h4_repo, check=True, capture_output=True)

    # Conflicting line-1 edits across waves, and `wt_w3` ONLY implements `wave3_fn` (+1 source commit, +3 total commits above `h4_base`!)
    (wt_w1 / "shared.py").write_text("LINE = 'w1'\ndef wave1_fn(): return 1\n", encoding="utf-8")
    subprocess.run(["git", "commit", "-am", "w1"], cwd=wt_w1, check=True, capture_output=True)
    (wt_w2_code / "shared.py").write_text("LINE = 'w2'\ndef wave2_fn(): return 2\n", encoding="utf-8")
    subprocess.run(["git", "commit", "-am", "w2"], cwd=wt_w2_code, check=True, capture_output=True)
    (wt_w3 / "shared.py").write_text("LINE = 'w3_only'\ndef wave3_fn(): return 3\n", encoding="utf-8")
    subprocess.run(["git", "commit", "-am", "w3"], cwd=wt_w3, check=True, capture_output=True)

    ep_h4 = EpisodeSpec(
        episode_id="TEST_V140_H4_A2",
        track_id="H-4",
        domain="hybrid",
        scenario_type="long_horizon_evolution",
        title="H-4 A2 regression",
        upstream_pillars=["cooperbench"],
        upstream_task_ids=["1", "2", "3"],
        workspace_mounts=[
            WorkspaceMountSpec(
                mount_id="mnt_repo",
                pillar="cooperbench",
                mount_relpath="repo",
                upstream_id="cb3",
                repo_or_category="local/shared",
                base_commit=h4_base,
            )
        ],
        timeline=[],
        verifiers=[
            VerifierSpec(
                verifier_id="v_3wave",
                kind="cooperbench_3wave_merged_and_policy_check",
                mount_id="mnt_repo",
                check_params={
                    "wave_tags": ["wave_1", "wave_2", "wave_3"],
                    "tests_patches": [str(p) for p in h4_patches],
                    "required_policy_tag": "POLICY-REV-2",
                    "forbidden_stale_phrases": ["仍是 Revision 1"],
                },
            )
        ],
    )
    ws_h4 = MaterializedEpisodeWorkspace(
        episode=ep_h4,
        arm_name="synaptic_pa",
        root_dir=tmp_path,
        repo_dir=h4_repo,
        mount_paths={"mnt_repo": h4_repo},
        shared_symlinks={".venv": PROJECT_ROOT / ".venv"},
    )
    art_h4_conflict = RuntimeArtifacts(
        arm_name="synaptic_pa",
        task_worktrees={"wave_1": wt_w1, "wave_2": wt_w2_office_empty, "wave_3": wt_w3},
        all_worktrees=[wt_w1, wt_w2_office_empty, wt_w2_code, wt_w3],
        agent_replies=["Wave 3 complete under POLICY-REV-<num>."],
        telemetry={
            "tag_worktrees": {
                "wave_1": [str(wt_w1)],
                "wave_2": [str(wt_w2_office_empty), str(wt_w2_code)],
                "wave_3": [str(wt_w3)],
            }
        },
    )
    # Multi-cell tag ranking MUST place `wt_w2_code` (non-empty diff) ahead of `wt_w2_office_empty`!
    w2_cands = _candidate_worktrees_for_tag(ws_h4, art_h4_conflict, "wave_2")
    assert w2_cands[0] == wt_w2_code
    rep_h4_bad = evaluate_episode_verifiers(ep_h4, ws_h4, art_h4_conflict)
    v3_bad = rep_h4_bad["verifiers"][0]
    assert v3_bad["passed"] is False
    assert v3_bad["merge_clean"] is False
    assert v3_bad["policy_rev2_ok"] is False
    assert v3_bad["has_masked_policy_num"] is True

    # Now stack genuine cumulative Wave 1+2+3 implementation in `wt_w3` + valid final post-Rev-2 reply -> MUST pass!
    (wt_w3 / "shared.py").write_text(
        "LINE = 'integrated_123'\ndef wave1_fn(): return 1\ndef wave2_fn(): return 2\ndef wave3_fn(): return 3\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "commit", "-am", "integrate all 3 waves"], cwd=wt_w3, check=True, capture_output=True)
    art_h4_good = RuntimeArtifacts(
        arm_name="synaptic_pa",
        task_worktrees={"wave_1": wt_w1, "wave_2": wt_w2_code, "wave_3": wt_w3},
        all_worktrees=[wt_w1, wt_w2_office_empty, wt_w2_code, wt_w3],
        agent_replies=[
            "r1", "r2", "r3", "r4", "r5", "Wave 3 integrated and verified under POLICY-REV-2."
        ],
        telemetry={"tag_worktrees": {"wave_1": [str(wt_w1)], "wave_2": [str(wt_w2_code)], "wave_3": [str(wt_w3)]}},
    )
    rep_h4_good = evaluate_episode_verifiers(ep_h4, ws_h4, art_h4_good)
    assert rep_h4_good["episode_passed"] is True, rep_h4_good
    assert rep_h4_good["verifiers"][0]["merge_clean"] is True
    assert rep_h4_good["verifiers"][0]["policy_rev2_ok"] is True

    # ------------------------------------------------------------------
    # 3. `A3`: `_pytest_env` symlink-safe `.venv/bin` PATH prepend + proxy scrubbing
    # ------------------------------------------------------------------
    for pk in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "all_proxy", "no_proxy"):
        monkeypatch.setenv(pk, "http://127.0.0.1:9")
    fake_venv_py = PROJECT_ROOT / ".venv" / "bin" / "python"
    penv = _pytest_env(dummy_repo, fake_venv_py)
    assert all(pk not in penv for pk in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "all_proxy", "no_proxy"))
    assert penv["PATH"].split(":")[0] == str((PROJECT_ROOT / ".venv" / "bin").resolve())

    # ------------------------------------------------------------------
    # 4. `A5`: Cross-modal prompt-echo guard + hallucinated `"tests passed"` rejection + P2P priority selection
    # ------------------------------------------------------------------
    c_echo = ContextCell("Cell_CM", "h3 code", "user_chat", mmu_attr.sandbox_paths, mmu_attr.trunk, mmu_attr.llm, mmu_attr.bus, dsh_host=FakeDSHHost())
    # Prompt-echo only (system/user + bus notice suffix on tool output) MUST NOT satisfy `office_mount_inspected` or `spec_doc_read`!
    c_echo.tool_scratchpad = [
        {"role": "system", "content": "Please inspect ../office_fs/release_spec_addendum.md"},
        {"role": "tool", "tool_call_id": "tc1", "content": "def foo(): pass\n\n[LIVE USER CONSTRAINT]: read release_spec_addendum.md"},
    ]
    mmu_cm = ContextMMU(
        sandbox_paths=mmu_attr.sandbox_paths,
        judge=System1Judge(typesafe_api_key="", deepseek_api_key=""),
        llm=HomogeneousLLMClient(api_key=""),
        trunk=ElasticSharedTrunk(),
        dsh_host=FakeDSHHost(),
    )
    mmu_cm.cells = {"Cell_CM": c_echo}
    ev_echo_only = _collect_synaptic_office_read_evidence(mmu_cm)
    assert ev_echo_only["office_mount_inspected"] is False
    assert ev_echo_only["office_spec_doc_read"] is False
    assert ev_echo_only["tool_read_spec_ref_codes"] == []

    # Real `role == "tool"` read of `release_spec_addendum.md` returning `PA150-SPEC-07` MUST succeed!
    c_echo.tool_scratchpad.append(
        {"role": "tool", "tool_call_id": "tc2", "content": "# Addendum\nRequirement Reference: PA150-SPEC-07\n"}
    )
    ev_real_read = _collect_synaptic_office_read_evidence(mmu_cm)
    assert ev_real_read["office_mount_inspected"] is True
    assert ev_real_read["office_spec_doc_read"] is True
    assert ev_real_read["tool_read_spec_ref_codes"] == ["PA150-SPEC-07"]

    # `code_to_office_brief`: reject hallucinated `"tests passed"` when `cooperbench_pytest` failed!
    wt_cm = tmp_path / "wt_cm"
    subprocess.run(["git", "worktree", "add", "-b", "bcm", str(wt_cm), "main"], cwd=dummy_repo, check=True, capture_output=True)
    (wt_cm / "app.py").write_text("VAL = 42\n", encoding="utf-8")
    fail_patch = tmp_path / "fail_cm.patch"
    (dummy_repo / "test_fail_cm.py").write_text("import app\ndef test_must_be_99(): assert app.VAL == 99\n", encoding="utf-8")
    subprocess.run(["git", "add", "test_fail_cm.py"], cwd=dummy_repo, check=True, capture_output=True)
    fail_patch.write_text(subprocess.run(["git", "diff", "--cached"], cwd=dummy_repo, capture_output=True, text=True).stdout, encoding="utf-8")
    subprocess.run(["git", "reset", "--hard", "HEAD"], cwd=dummy_repo, check=True, capture_output=True)

    ep_c2o = EpisodeSpec(
        episode_id="TEST_V140_C2O",
        track_id="H-3",
        domain="hybrid",
        scenario_type="mid_execution_amendment",
        title="C2O hallucination guard",
        upstream_pillars=["cooperbench", "mcpmark_filesystem"],
        upstream_task_ids=["cb1"],
        workspace_mounts=[
            WorkspaceMountSpec(
                mount_id="mnt_repo",
                pillar="cooperbench",
                mount_relpath="repo",
                upstream_id="cb1",
                repo_or_category="local/app",
                base_commit=subprocess.run(["git", "rev-parse", "HEAD"], cwd=dummy_repo, capture_output=True, text=True).stdout.strip(),
            )
        ],
        timeline=[],
        verifiers=[
            VerifierSpec(verifier_id="v_code", kind="cooperbench_pytest", mount_id="mnt_repo", check_params={"tests_patch": str(fail_patch)}),
            VerifierSpec(verifier_id="v_c2o", kind="cross_modal_consistency_check", mount_id="mnt_repo", check_params={"direction": "code_to_office_brief"}),
        ],
    )
    ws_c2o = MaterializedEpisodeWorkspace(
        episode=ep_c2o,
        arm_name="synaptic_pa",
        root_dir=tmp_path,
        repo_dir=dummy_repo,
        mount_paths={"mnt_repo": dummy_repo},
        shared_symlinks={".venv": PROJECT_ROOT / ".venv"},
    )
    rep_hallucinated = evaluate_episode_verifiers(
        ep_c2o,
        ws_c2o,
        RuntimeArtifacts(
            arm_name="synaptic_pa",
            all_worktrees=[wt_cm],
            agent_replies=["Updated app.py implementation and all pytest unit tests passed cleanly."],
            telemetry={"worktrees_created": 1},
        ),
    )
    assert rep_hallucinated["verifiers"][0]["passed"] is False
    assert rep_hallucinated["verifiers"][1]["passed"] is False
    assert rep_hallucinated["verifiers"][1]["verification_claim_consistent"] is False

    rep_honest = evaluate_episode_verifiers(
        ep_c2o,
        ws_c2o,
        RuntimeArtifacts(
            arm_name="synaptic_pa",
            all_worktrees=[wt_cm],
            agent_replies=["Updated app.py; pytest verification failed with 1 remaining assertion error on VAL."],
            telemetry={"worktrees_created": 1},
        ),
    )
    assert rep_honest["verifiers"][1]["passed"] is True
    assert rep_honest["verifiers"][1]["verification_claim_consistent"] is True

    # P2P priority selection (`A5`): 100% of `patched_test_files` nodes MUST precede unrelated files before 80-node cap!
    p2p_unrel = [f"tests/test_other_{i}.py::test_u_{i}" for i in range(90)]
    p2p_prio = [f"tests/test_patched.py::test_p_{j}" for j in range(5)]
    sel_p2p, prio_only, _ = _select_prioritized_p2p_targets(
        p2p_raw=p2p_unrel + p2p_prio,
        all_test_patch_files=["tests/test_patched.py"],
        patched_test_files=["tests/test_patched.py"],
        f2p_nodes=[],
        bare_f2p_names=[],
        cap=80,
    )
    assert len(sel_p2p) == 80
    assert prio_only == p2p_prio
    assert sel_p2p[:5] == p2p_prio

    # ------------------------------------------------------------------
    # 5. `A6` + Paradigm-Alignment 2 & 3: `_TrackedTaskDict`, `2/3` noise hygiene + `noise_drop_rate`, cache rate & peek naming
    # ------------------------------------------------------------------
    captured_bg_errs: list[dict] = []

    async def _exercise_tracked_bg() -> None:
        tdict = _TrackedTaskDict(captured_bg_errs)

        async def _crash() -> None:
            raise ValueError("synthetic background worker failure")

        tdict["Cell_BG_1"] = asyncio.create_task(_crash())
        await asyncio.gather(*tdict.values(), return_exceptions=True)

    asyncio.run(_exercise_tracked_bg())
    assert len(captured_bg_errs) == 1
    assert captured_bg_errs[0]["cell_id"] == "Cell_BG_1"
    assert captured_bg_errs[0]["error_type"] == "ValueError"
    assert "synthetic background worker failure" in captured_bg_errs[0]["traceback"]

    ep_hyg = EpisodeSpec(
        episode_id="TEST_V140_HYG",
        track_id="O-2",
        domain="office",
        scenario_type="high_concurrency_burst",
        title="Non-cliff noise hygiene",
        upstream_pillars=["mcpmark_filesystem"],
        upstream_task_ids=["m1"],
        workspace_mounts=[
            WorkspaceMountSpec(
                mount_id="mnt_fs",
                pillar="mcpmark_filesystem",
                mount_relpath="office_fs",
                upstream_id="m1",
                repo_or_category="filesystem/desktop",
            )
        ],
        timeline=[],
        verifiers=[
            VerifierSpec(
                verifier_id="v_hyg",
                kind="noise_and_dedup_telemetry_check",
                mount_id="mnt_fs",
                check_params={"expected_noise_dropped": 3, "expected_duplicates_merged": 1, "expected_worktrees": 1},
            )
        ],
    )
    ws_hyg = MaterializedEpisodeWorkspace(episode=ep_hyg, arm_name="synaptic_pa", root_dir=tmp_path, repo_dir=dummy_repo)
    # `v2.0.0`: Strict 3/3 noise hygiene + zero spawned noise workers passes; empty `{}` telemetry or spawned noise worker fails!
    tel_3of3 = {
        "noise_events_total": 3,
        "noise_dropped": 3,
        "noise_spawned_workers": 0,
        "noise_mutated_workers": 0,
        "duplicates_merged": 1,
        "duplicate_spawned_workers": 0,
        "real_tasks_dropped": 0,
        "worktrees_created": 1,
        "peek_latencies_ms": [12.0],
        "system2_prompt_tokens": 1000,
        "system1_prompt_tokens": 500,
        "total_prompt_tokens": 1500,
        "cache_hit_tokens": 800,
        "output_tokens": 200,
        "prefix_cache_hit_rate": 0.8,
        "overall_prompt_cache_hit_rate": 0.5333,
    }
    rep_3of3 = evaluate_episode_verifiers(
        ep_hyg,
        ws_hyg,
        RuntimeArtifacts(arm_name="synaptic_pa", telemetry=tel_3of3),
    )
    assert rep_3of3["episode_passed"] is True, rep_3of3

    # Empty `{}` telemetry, mutating noise worker, OR dropping < 3/3 noise MUST fail!
    rep_empty_tel = evaluate_episode_verifiers(
        ep_hyg,
        ws_hyg,
        RuntimeArtifacts(arm_name="synaptic_pa", telemetry={}),
    )
    assert rep_empty_tel["episode_passed"] is False
    rep_mut_noise = evaluate_episode_verifiers(
        ep_hyg,
        ws_hyg,
        RuntimeArtifacts(arm_name="synaptic_pa", telemetry={**tel_3of3, "noise_mutated_workers": 1}),
    )
    assert rep_mut_noise["episode_passed"] is False
    rep_2of3 = evaluate_episode_verifiers(
        ep_hyg,
        ws_hyg,
        RuntimeArtifacts(arm_name="synaptic_pa", telemetry={**tel_3of3, "noise_dropped": 2, "noise_spawned_workers": 1}),
    )
    assert rep_2of3["episode_passed"] is False

    agg_v140 = aggregate_benchmark_results([{**rep_3of3, "domain": "office", "telemetry": tel_3of3}])
    arm_v140 = agg_v140["arms"]["synaptic_pa"]
    assert agg_v140["evaluator_version"] == "2.0.1"
    assert arm_v140["tier1_orchestration_safety"]["burst_hygiene_pass_rate"] == 1.0
    assert arm_v140["tier1_orchestration_safety"]["noise_drop_rate"] == 1.0
    assert arm_v140["tier3_efficiency"]["mean_nonblocking_peek_sec"] == 0.012
    assert arm_v140["tier3_efficiency"]["mean_peek_latency_sec"] == 0.012
    assert arm_v140["tier3_efficiency"]["prefix_cache_hit_rate"] == 0.8
    assert arm_v140["tier3_efficiency"]["overall_prompt_cache_hit_rate"] == 0.5333
    assert arm_v140["prefix_cache_hit_rate"] == 0.8


def test_v150_eleven_audit_fixes_regression(tmp_path: Path) -> None:
    """`v1.5.0`: Regression tests for all 11 audit fixes (sandbox anti-contamination, Gaia2 FS + open read-only,
    office-only zero-worktree guard, CooperBench `_filter_test_files` + DAG merge, negated failure brief parsing,
    and Turn-0 single-task decomposition guard)."""
    import asyncio
    from synaptic_bench_150.verifiers import (
        RuntimeArtifacts,
        _brief_claims_tests_passed,
        _filter_test_files,
        evaluate_episode_verifiers,
    )
    # 1. Item 2: Gaia2 demo filesystem (`gaia2_demo_fs`) is populated with >= 290 upstream files
    from clonact.context_cell import ContextCell
    from clonact.context_mmu import ContextMMU
    from clonact.llm_client import HomogeneousLLMClient
    from clonact.synaptic_bus import HybridSynapticBus
    from clonact.system1_judge import System1Judge
    from clonact.trunk import ElasticSharedTrunk

    demo_fs_dir = PROJECT_ROOT / "data" / "upstream_pillars" / "gaia2_demo_fs"
    demo_files = [p for p in demo_fs_dir.rglob("*") if p.is_file() and p.name != "README.md"]
    assert len(demo_files) >= 290

    # 2. Item 1 & Item 9b: `ContextCell` + `make_benchmark_tool_policy_guard` blocks held-out benchmark paths & GitHub PR URLs
    # while `synaptic_pa/` remains 100% free of benchmark-specific identifiers.
    from synaptic_bench_150.workspace_manager import make_benchmark_tool_policy_guard

    for syn_fp in (PROJECT_ROOT / "clonact").glob("*.py"):
        syn_src = syn_fp.read_text(encoding="utf-8")
        for forbidden_tok in ("office_fs", "office_pg", "gaia_cli", "pg_exec", "gaia2_universe", "WORKSPACE_POLICY.md", "upstream_pillars"):
            assert forbidden_tok not in syn_src, f"Benchmark token {forbidden_tok!r} leaked into {syn_fp.name}"

    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "t@local"], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Tester"], cwd=repo_dir, check=True, capture_output=True)
    (repo_dir / "app.py").write_text("X = 1\n", encoding="utf-8")
    (repo_dir / "WORKSPACE_POLICY.md").write_text("POLICY-REV-2\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo_dir, check=True, capture_output=True)

    wt_root = tmp_path / "worktrees"
    wt_root.mkdir()
    shared_links = tmp_path / "shared_links"
    shared_links.mkdir()
    (shared_links / "WORKSPACE_POLICY.md").symlink_to((repo_dir / "WORKSPACE_POLICY.md").resolve())
    office_fs = tmp_path / "office_fs"
    office_fs.mkdir()
    (office_fs / "note.txt").write_text("hello office\n", encoding="utf-8")

    # Office-only cell (`worktrees=None`) must NOT materialize a git worktree on bash commands!
    cell_office = ContextCell(
        cell_id="cell_office_v160",
        topic="office only test",
        source_kind="user_chat",
        sandbox_paths={"world": tmp_path, "repo": repo_dir, "worktrees": None, "docs": tmp_path, "shared_links": shared_links},
        trunk=ElasticSharedTrunk(),
        llm=HomogeneousLLMClient(api_key=""),
        bus=HybridSynapticBus(System1Judge(typesafe_api_key="", deepseek_api_key="")),
        extra_env={"FILESYSTEM_TEST_DIR": str(office_fs)},
        tool_policy_guard=make_benchmark_tool_policy_guard(tmp_path),
        dsh_host=FakeDSHHost(),
    )
    assert cell_office._ensure_worktree() == repo_dir
    assert cell_office.worktree_path is None

    cell = ContextCell(
        cell_id="cell_test_v160",
        topic="office + code test",
        source_kind="user_chat",
        sandbox_paths={"world": tmp_path, "repo": repo_dir, "worktrees": wt_root, "docs": tmp_path, "shared_links": shared_links},
        trunk=ElasticSharedTrunk(),
        llm=HomogeneousLLMClient(api_key=""),
        bus=HybridSynapticBus(System1Judge(typesafe_api_key="", deepseek_api_key="")),
        extra_env={"FILESYSTEM_TEST_DIR": str(office_fs)},
        tool_policy_guard=make_benchmark_tool_policy_guard(tmp_path),
        dsh_host=FakeDSHHost(),
    )
    # 2a. Blocked held-out contamination paths & outbound GitHub PR URLs, while allowing workspace `verify.py`
    guard = make_benchmark_tool_policy_guard(tmp_path)
    assert "held-out benchmark" in guard("bash", {"command": "cat /tmp/upstream_pillars/solution.patch"}, repo_dir)
    assert "outbound network access" in guard("bash", {"command": "curl -s https://github.com/org/repo/pull/42.diff"}, repo_dir)
    assert "held-out benchmark" in guard("read_repo_file", {"rel_path": "../mcpmark/tasks/verify.py"}, repo_dir)
    assert guard("edit_repo_file", {"rel_path": "verify.py", "new_text": "print('ok')"}, repo_dir) is None

    # Native DSH supplies file mutations; shared promotion must still exclude policy files.
    worktree = cell._ensure_worktree()
    (worktree / "app.py").write_text("X = 2\n")
    (worktree / "WORKSPACE_POLICY.md").write_text("POLICY-REV-2\nDIRTY\n")
    assert cell.worktree_path is not None
    asyncio.run(cell._finalize_and_promote_async())
    assert cell.commit_sha is not None
    committed_files = subprocess.run(
        ["git", "-C", str(cell.worktree_path), "show", "--name-only", "--pretty=format:", cell.commit_sha],
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert "app.py" in committed_files
    assert "WORKSPACE_POLICY.md" not in committed_files

    # 3. Item 8: `_filter_test_files` strips agent-modified test files while preserving source files
    mixed_patch = (
        "diff --git a/src/core.py b/src/core.py\n"
        "--- a/src/core.py\n"
        "+++ b/src/core.py\n"
        "@@ -1 +1,2 @@\n"
        " X = 1\n"
        "+Y = 2\n"
        "diff --git a/tests/test_core.py b/tests/test_core.py\n"
        "--- a/tests/test_core.py\n"
        "+++ b/tests/test_core.py\n"
        "@@ -1 +1,2 @@\n"
        " def test_x(): pass\n"
        "+def test_y(): pass\n"
    )
    filtered = _filter_test_files(mixed_patch)
    assert "src/core.py" in filtered
    assert "tests/test_core.py" not in filtered

    # 4. Item 10: `_brief_claims_tests_passed` handles negated failure phrases (`无报错`, `0 failed`), partial pass counts (`128 passed, 14 failed`),
    # and rejects self-contradictory `"All tests passed; 5 tests failed"` on a single line!
    cp_clean, rf_clean = _brief_claims_tests_passed("已完成代码修改，单元测试全部通过（24 passed, 0 failed），运行无报错、无异常。")
    assert cp_clean is True and rf_clean is False
    cp_partial, rf_partial = _brief_claims_tests_passed("实测 pytest 结果：128 passed, 14 failed，14 条失败均为既有环境问题。")
    assert cp_partial is False and rf_partial is True
    cp_contra, rf_contra = _brief_claims_tests_passed("All tests passed; 5 tests failed in pytest.")
    assert cp_contra is True and rf_contra is True

    # 5. Paradigm-pure Turn-0 granularity check in `ContextMMU.handle_user_message`: `check_granularity` is always True
    seen_granularity: list[bool] = []
    from test_clonact import DeterministicStubJudge
    judge = DeterministicStubJudge()
    mmu = ContextMMU(
        sandbox_paths={"world": tmp_path, "repo": repo_dir, "worktrees": wt_root, "docs": tmp_path},
        judge=judge,
        llm=HomogeneousLLMClient(api_key=""),
        trunk=ElasticSharedTrunk(),
        dsh_host=FakeDSHHost(),
    )
    orig_route = mmu._route_single_clause

    async def _spy_route(*args, **kwargs):
        seen_granularity.append(bool(kwargs.get("check_granularity", False)))
        return await orig_route(*args, **kwargs)

    mmu._route_single_clause = _spy_route  # type: ignore[method-assign]
    asyncio.run(mmu.handle_user_message("Check my calendar, email Sarah, and update the contact list.", auto_run_detached=False))
    assert seen_granularity[0] is True


def test_v170_paradigm_fairness_option_a_and_dsh_parity(tmp_path: Path) -> None:
    """Verify v1.7.0 Apple-to-Apple paradigm fairness: bot allowlists, symmetric contamination scan, and `--no-ff` trunk promotion."""
    from synaptic_bench_150.runner import (
        _collect_episode_dsh_allowlists,
        _scan_executed_text_contamination,
    )
    from synaptic_bench_150.schema import EpisodeSpec, TimelineEventSpec, TriggerSpec
    from clonact.context_cell import promote_worktree_to_main

    # 1. Bot senders are populated into `fs_bots` as well as `fs_senders`
    ep = EpisodeSpec(
        episode_id="EP_BOT_TEST",
        track_id="O-1",
        domain="office",
        scenario_type="high_concurrency_burst",
        title="Bot allowlist test",
        upstream_pillars=[],
        upstream_task_ids=[],
        workspace_mounts=[],
        timeline=[
            TimelineEventSpec(
                event_id="e1",
                seq=1,
                kind="external_task_webhook",
                channel="feishu://group/ops",
                sender="bot_sentry_alert",
                text="Alert: DB spike",
                trigger=TriggerSpec(mode="immediate"),
            ),
            TimelineEventSpec(
                event_id="e2",
                seq=2,
                kind="external_noise",
                channel="feishu://group/ops",
                sender="alice_pm",
                text="Coffee anyone?",
                trigger=TriggerSpec(mode="immediate"),
            ),
        ],
        verifiers=[],
    )
    gh_users, fs_senders, fs_bots = _collect_episode_dsh_allowlists(ep)
    assert gh_users == []
    assert len(fs_bots) == 1 and fs_bots[0]["name"] == "bot_sentry_alert"
    assert len(fs_senders) == 2

    # 2. Symmetric contamination scan catches held-out paths & GitHub PR URLs
    assert _scan_executed_text_contamination(["pytest -q tests/test_core.py"]) is False
    assert _scan_executed_text_contamination(["cat data/upstream_pillars/solution.patch"]) is True
    assert _scan_executed_text_contamination(["curl https://github.com/pallets/jinja/pull/1465"]) is True

    # 3. `promote_worktree_to_main` merges `--no-ff` into `main` + pushes `origin`, and aborts cleanly on conflict
    bare = tmp_path / "origin.git"
    seed = tmp_path / "seed"
    seed.mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=seed, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "t@local"], cwd=seed, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Tester"], cwd=seed, check=True, capture_output=True)
    (seed / "shared.py").write_text("VAL = 0\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=seed, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=seed, check=True, capture_output=True)
    subprocess.run(["git", "clone", "--bare", str(seed), str(bare)], check=True, capture_output=True)

    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", f"file://{bare}", str(clone)], check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "t@local"], cwd=clone, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Tester"], cwd=clone, check=True, capture_output=True)
    base_sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=clone, capture_output=True, text=True).stdout.strip()

    wt_a = tmp_path / "wt_a"
    wt_b = tmp_path / "wt_b"
    subprocess.run(["git", "worktree", "add", "-b", "br_a", str(wt_a), base_sha], cwd=clone, check=True, capture_output=True)
    subprocess.run(["git", "worktree", "add", "-b", "br_b", str(wt_b), base_sha], cwd=clone, check=True, capture_output=True)

    (wt_a / "shared.py").write_text("VAL = 10\n", encoding="utf-8")
    sha_a, merged_a = promote_worktree_to_main(wt_a, clone, "task_a", base_ref=base_sha, push_origin=True)
    assert sha_a is not None and merged_a is True
    assert (clone / "shared.py").read_text(encoding="utf-8") == "VAL = 10\n"

    # Conflicting edit in `wt_b` commits on `br_b` (`sha_b`), aborts merge into `clone`, leaving `clone` clean!
    (wt_b / "shared.py").write_text("VAL = 99\n", encoding="utf-8")
    sha_b, merged_b = promote_worktree_to_main(wt_b, clone, "task_b", base_ref=base_sha, push_origin=True)
    assert sha_b is not None and merged_b is False
    assert (clone / "shared.py").read_text(encoding="utf-8") == "VAL = 10\n"


def test_v200_opensource_grade_upgrades_end_to_end(tmp_path: Path) -> None:
    """Comprehensive v2.0.0 regression suite covering:
    1. `_pytest_env` preserving `ws.extra_env['PATH']` (`PA150_H2_11` GNU `sed` shim fix)
    2. Gaia2 `AgentUserInterface__send_message_to_user` preserving `unmodified_since_start` (`PA150_H1_07`/`H1_13` fix)
    3. Local `.py` comment with GitHub issue URL allowed in `make_benchmark_tool_policy_guard`, while `curl` / dirty `evil.py` is blocked
    4. Architecture-neutral single-tree `_verify_cooperbench_merged` + anti-gaming 1-feature rejection
    5. `_verify_overturn_rollback` (`overturn_rollback_check`) on files & PostgreSQL + `seed_pg_spec_rows` (`public.release_spec_addenda`)
    6. `ControlledUserSimulator` zero-question gate vs grounded clarification answer
    7. Black-box `SubprocessCLIHarnessAdapter` + `run_episode_external_adapter` + cluster-robust SE
    """
    import asyncio
    import psycopg2
    from synaptic_bench_150.harness_adapter import (
        ControlledUserSimulator,
        SubprocessCLIHarnessAdapter,
        parse_cli_arm_spec,
    )
    from synaptic_bench_150.runner import (
        _compute_cluster_robust_se,
        _parse_arms_arg,
        aggregate_benchmark_results,
        run_episode_external_adapter,
    )
    from synaptic_bench_150.verifiers import _pytest_env
    from synaptic_bench_150.workspace_manager import (
        PG_DEFAULT_PORT,
        create_isolated_postgres_db,
        drop_postgres_db,
        hash_postgres_database,
        make_benchmark_tool_policy_guard,
        materialize_episode_workspace,
    )
    from clonact.context_cell import ContextCell
    from clonact.synaptic_bus import HybridSynapticBus

    # 1. `_pytest_env` preserves workspace shim bin directory on `PATH`
    fake_shim = tmp_path / "shim_bin"
    fake_shim.mkdir()
    penv = _pytest_env(tmp_path, PROJECT_ROOT / ".venv" / "bin" / "python", {"PATH": f"{fake_shim}:/usr/bin"})
    path_parts = penv["PATH"].split(":")
    assert str(fake_shim) in path_parts
    assert path_parts.index(str(fake_shim)) < path_parts.index("/usr/bin")

    # 2. Gaia2 `AgentUserInterface__send_message_to_user` does NOT increment `mutating_call_count` or flip `unmodified_since_start`
    samples = json.loads((PROJECT_ROOT / "data" / "upstream_pillars" / "samples" / "gaia2_samples.json").read_text())
    sample_row = samples["execution"][0]
    srv = Gaia2ServerProcess(port=19039, log_path=tmp_path / "gaia2_v200_aui.log")
    try:
        srv.start_scenario(
            upstream_id=f"gaia2::execution::{sample_row['id']}",
            config="execution",
            scenario_id=str(sample_row["scenario_id"]),
            judge_mode="deterministic",
            readonly_side_mount=True,
        )
        cli_fp = tmp_path / "gaia_cli"
        cli_fp.write_text(build_gaia_cli_script(srv.base_url), encoding="utf-8")
        cli_fp.chmod(0o755)
        aui_proc = subprocess.run(
            [str(cli_fp), "call", "AgentUserInterface__send_message_to_user", '{"content": "Read-only calendar summary."}'],
            check=True,
            capture_output=True,
            text=True,
        )
        aui_json = json.loads(aui_proc.stdout)
        assert aui_json.get("mutated") is False
        snap_aui = srv.snapshot()
        assert snap_aui["unmodified_since_start"] is True
        assert snap_aui["mutating_call_count"] == 0
    finally:
        srv.stop()

    # 3. Refined `make_benchmark_tool_policy_guard`: local `.py` with GitHub issue URL comment is allowed; `curl` or dirty `evil.py` is blocked
    repo_g = tmp_path / "repo_guard"
    repo_g.mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=repo_g, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "t@local"], cwd=repo_g, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Tester"], cwd=repo_g, check=True, capture_output=True)
    (repo_g / "upstream_mod.py").write_text(
        "# Regression note: see https://github.com/org/repo/issues/123 for background\nimport urllib.request\nX = 7\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "upstream_mod.py"], cwd=repo_g, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "tracked upstream file"], cwd=repo_g, check=True, capture_output=True)

    guard = make_benchmark_tool_policy_guard(tmp_path)
    cell_g = ContextCell(
        cell_id="cell_g_v200",
        topic="guard test",
        source_kind="user_chat",
        sandbox_paths={"world": tmp_path, "repo": repo_g, "worktrees": None, "docs": tmp_path},
        trunk=ElasticSharedTrunk(),
        llm=HomogeneousLLMClient(api_key=""),
        bus=HybridSynapticBus(System1Judge(typesafe_api_key="", deepseek_api_key="")),
        tool_policy_guard=guard,
        dsh_host=FakeDSHHost(),
    )
    assert guard("bash", {"command": f"{sys.executable} -c 'import upstream_mod; print(upstream_mod.X)'"}, repo_g) is None
    assert guard("bash", {"command": "grep -n 'https://github.com/org/repo/issues/123' upstream_mod.py"}, repo_g) is None

    # Untracked `evil.py` attempting outbound GitHub PR fetch MUST be blocked!
    (repo_g / "evil.py").write_text(
        "import urllib.request\nurllib.request.urlopen('https://github.com/org/repo/pull/99.patch')\n",
        encoding="utf-8",
    )
    assert "outbound network access" in guard("bash", {"command": f"{sys.executable} evil.py"}, repo_g)
    (repo_g / "evil.py").unlink()

    # 4. Architecture-neutral single-tree `_verify_cooperbench_merged` + `_verify_overturn_rollback`
    pa = tmp_path / "pa_v200.patch"
    pb = tmp_path / "pb_v200.patch"
    base_g = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_g, capture_output=True, text=True).stdout.strip()
    (repo_g / "test_a.py").write_text("import upstream_mod\ndef test_fa(): assert upstream_mod.fa() == 11\n", encoding="utf-8")
    subprocess.run(["git", "add", "test_a.py"], cwd=repo_g, check=True, capture_output=True)
    pa.write_text(subprocess.run(["git", "diff", "--cached"], cwd=repo_g, capture_output=True, text=True).stdout, encoding="utf-8")
    subprocess.run(["git", "reset", "--hard", base_g], cwd=repo_g, check=True, capture_output=True)

    (repo_g / "test_b.py").write_text("import upstream_mod\ndef test_fb(): assert upstream_mod.fb() == 22\n", encoding="utf-8")
    subprocess.run(["git", "add", "test_b.py"], cwd=repo_g, check=True, capture_output=True)
    pb.write_text(subprocess.run(["git", "diff", "--cached"], cwd=repo_g, capture_output=True, text=True).stdout, encoding="utf-8")
    subprocess.run(["git", "reset", "--hard", base_g], cwd=repo_g, check=True, capture_output=True)

    # Single-tree agent implements ONLY `fa` (without `fb`) -> `merge_clean` MUST be False!
    (repo_g / "upstream_mod.py").write_text("X = 7\ndef fa(): return 11\n", encoding="utf-8")
    ep_single_tree = EpisodeSpec(
        episode_id="TEST_V200_SINGLE_TREE",
        track_id="C-2",
        domain="coding",
        scenario_type="high_concurrency_burst",
        title="Single tree dual feature",
        upstream_pillars=["cooperbench"],
        upstream_task_ids=["cb1", "cb2"],
        workspace_mounts=[
            WorkspaceMountSpec(
                mount_id="mnt_repo",
                pillar="cooperbench",
                mount_relpath="repo",
                upstream_id="cb_pair",
                repo_or_category="local/guard",
                base_commit=base_g,
            )
        ],
        timeline=[],
        verifiers=[
            VerifierSpec(
                verifier_id="v_merged",
                kind="cooperbench_merged_pytest",
                mount_id="mnt_repo",
                check_params={"tests_patch_a": str(pa), "tests_patch_b": str(pb)},
            ),
            VerifierSpec(
                verifier_id="v_ov",
                kind="overturn_rollback_check",
                mount_id="mnt_repo",
                check_params={"forbidden_rel_paths": ["_legacy_workaround_shim.py"]},
            ),
        ],
    )
    ws_st = MaterializedEpisodeWorkspace(
        episode=ep_single_tree,
        arm_name="ext_cli",
        root_dir=tmp_path,
        repo_dir=repo_g,
        mount_paths={"mnt_repo": repo_g},
        shared_symlinks={".venv": PROJECT_ROOT / ".venv"},
    )
    rep_partial = evaluate_episode_verifiers(ep_single_tree, ws_st, RuntimeArtifacts(arm_name="ext_cli"))
    assert rep_partial["verifiers"][0]["passed"] is False
    assert rep_partial["verifiers"][0]["merge_clean"] is False

    # Single-tree agent implements BOTH `fa` and `fb`, but leaves `_legacy_workaround_shim.py` -> `v_merged` passes, `v_ov` fails!
    (repo_g / "upstream_mod.py").write_text("X = 7\ndef fa(): return 11\ndef fb(): return 22\n", encoding="utf-8")
    (repo_g / "_legacy_workaround_shim.py").write_text("SHIM = True\n", encoding="utf-8")
    rep_leftover = evaluate_episode_verifiers(ep_single_tree, ws_st, RuntimeArtifacts(arm_name="ext_cli"))
    assert rep_leftover["verifiers"][0]["passed"] is True
    assert rep_leftover["verifiers"][0]["merge_clean"] is True
    assert rep_leftover["verifiers"][0]["integration_source"] == "single_tree_integrated"
    assert rep_leftover["verifiers"][1]["passed"] is False
    assert rep_leftover["episode_passed"] is False

    # Remove `_legacy_workaround_shim.py` -> entire episode PASSES cleanly on single-tree agent!
    (repo_g / "_legacy_workaround_shim.py").unlink()
    rep_clean = evaluate_episode_verifiers(ep_single_tree, ws_st, RuntimeArtifacts(arm_name="ext_cli"))
    assert rep_clean["episode_passed"] is True

    # 5. PostgreSQL `seed_pg_spec_rows` (`public.release_spec_addenda`), ACL hashing, & PG `overturn_rollback_check`
    pg_db_name = "pa_v200_spec_and_ov_test"
    drop_postgres_db(pg_db_name, PG_DEFAULT_PORT)
    try:
        pg_env = create_isolated_postgres_db(pg_db_name, "custom_empty", task_dir=None, port=PG_DEFAULT_PORT)
        h_before_acl = hash_postgres_database(pg_env)["db_sha256"]
        conn = psycopg2.connect(
            host=pg_env["POSTGRES_HOST"],
            port=int(pg_env["POSTGRES_PORT"]),
            database=pg_env["POSTGRES_DATABASE"],
            user=pg_env["POSTGRES_USERNAME"],
            password=pg_env["POSTGRES_PASSWORD"],
        )
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute("REVOKE ALL ON SCHEMA public FROM PUBLIC")
                    cur.execute("CREATE TABLE public.legacy_draft_staging (note TEXT)")
        finally:
            conn.close()
        h_after_acl = hash_postgres_database(pg_env)["db_sha256"]
        assert h_after_acl != h_before_acl

        ep_pg_ov = EpisodeSpec(
            episode_id="TEST_V200_PG_OV",
            track_id="O-3",
            domain="office",
            scenario_type="inflight_overturn",
            title="PG overturn rollback",
            upstream_pillars=["mcpmark_postgres"],
            upstream_task_ids=["pg1"],
            workspace_mounts=[
                WorkspaceMountSpec(
                    mount_id="mnt_pg",
                    pillar="mcpmark_postgres",
                    mount_relpath="office_pg",
                    upstream_id="pg1",
                    repo_or_category="postgres/employees",
                )
            ],
            timeline=[],
            verifiers=[
                VerifierSpec(
                    verifier_id="v_pg_ov",
                    kind="overturn_rollback_check",
                    mount_id="mnt_pg",
                    check_params={"forbidden_pg_tables": ["legacy_draft_staging"]},
                )
            ],
        )
        ws_pg_ov = MaterializedEpisodeWorkspace(
            episode=ep_pg_ov,
            arm_name="ext_cli",
            root_dir=tmp_path,
            repo_dir=repo_g,
            mount_pg_envs={"mnt_pg": pg_env},
        )
        assert evaluate_episode_verifiers(ep_pg_ov, ws_pg_ov, RuntimeArtifacts(arm_name="ext_cli"))["episode_passed"] is False

        conn2 = psycopg2.connect(
            host=pg_env["POSTGRES_HOST"],
            port=int(pg_env["POSTGRES_PORT"]),
            database=pg_env["POSTGRES_DATABASE"],
            user=pg_env["POSTGRES_USERNAME"],
            password=pg_env["POSTGRES_PASSWORD"],
        )
        try:
            with conn2:
                with conn2.cursor() as cur:
                    cur.execute("DROP TABLE public.legacy_draft_staging")
        finally:
            conn2.close()
        assert evaluate_episode_verifiers(ep_pg_ov, ws_pg_ov, RuntimeArtifacts(arm_name="ext_cli"))["episode_passed"] is True
    finally:
        drop_postgres_db(pg_db_name, PG_DEFAULT_PORT)

    # Also test `materialize_episode_workspace` seeding `public.release_spec_addenda` on `PA150_H3_01`
    h3_01 = load_pa150_episodes(episode_ids=["PA150_H3_01"])[0]
    ws_h3 = materialize_episode_workspace(h3_01, "v200_test", tmp_path / "ws_h3")
    try:
        pg_h3 = ws_h3.mount_pg_envs["mnt_office"]
        c_h3 = psycopg2.connect(
            host=pg_h3["POSTGRES_HOST"],
            port=int(pg_h3["POSTGRES_PORT"]),
            database=pg_h3["POSTGRES_DATABASE"],
            user=pg_h3["POSTGRES_USERNAME"],
            password=pg_h3["POSTGRES_PASSWORD"],
        )
        try:
            with c_h3.cursor() as cur:
                cur.execute("SELECT spec_code, length(requirements_md) FROM public.release_spec_addenda")
                rows_h3 = cur.fetchall()
                assert len(rows_h3) == 1
                assert rows_h3[0][0].startswith("PA150-SPEC-01-") and rows_h3[0][1] >= 40
        finally:
            c_h3.close()
    finally:
        ws_h3.cleanup(remove_files=True)

    # 6. `ControlledUserSimulator` zero-question gate vs clarification answer on `PA150_C1_09` (`ambig_swe`)
    c1_09 = load_pa150_episodes(episode_ids=["PA150_C1_09"])[0]
    ev_1c = c1_09.timeline[2]
    res_no_q, meta_no_q = ControlledUserSimulator.resolve_turn_1c_event(
        ev_1c,
        ["I inspected the repository structure and located the relevant module."],
        simulator_mode="deterministic",
    )
    assert meta_no_q["asked_clarification"] is False
    assert meta_no_q["revealed_chars"] == 0
    assert "补充说明如下" not in res_no_q.text

    res_with_q, meta_with_q = ControlledUserSimulator.resolve_turn_1c_event(
        ev_1c,
        ["Before modifying code, could you clarify what exception or return value should occur on invalid input?"],
        simulator_mode="deterministic",
    )
    assert meta_with_q["asked_clarification"] is True
    assert meta_with_q["revealed_chars"] > 20
    assert "补充说明如下" in res_with_q.text

    # 7. Black-box `SubprocessCLIHarnessAdapter` + `parse_cli_arm_spec` + `_parse_arms_arg` + cluster-robust SE
    assert parse_cli_arm_spec("cli:my_agent:python agent.py") == ("my_agent", "python agent.py")
    assert _parse_arms_arg("clonact,cli:codex_box:python -c 'print(1, 2)'") == [
        ("clonact", None),
        ("codex_box", "python -c 'print(1, 2)'"),
    ]

    mock_agent_py = tmp_path / "mock_cli_agent.py"
    mock_agent_py.write_text(
        "import json, os, pathlib\n"
        "ev = json.loads(os.environ['PA150_EVENT_JSON'])\n"
        "fs_dir = pathlib.Path(os.environ['FILESYSTEM_TEST_DIR'])\n"
        "if ev['event_id'] == 'ev_launch':\n"
        "    (fs_dir / '_legacy_draft_manifest.json').write_text('{}', encoding='utf-8')\n"
        "    print(json.dumps({'disposition': 'executed', 'reply': 'Created draft manifest.', 'tool_calls': 1}))\n"
        "else:\n"
        "    (fs_dir / '_legacy_draft_manifest.json').unlink(missing_ok=True)\n"
        "    (fs_dir / 'done.txt').write_text('ok\\n', encoding='utf-8')\n"
        "    print(json.dumps({'disposition': 'executed', 'reply': 'Rolled back draft and completed spec.', 'tool_calls': 2}))\n",
        encoding="utf-8",
    )
    verify_done_py = tmp_path / "verify_done.py"
    verify_done_py.write_text(
        "import os, pathlib, sys\n"
        "p = pathlib.Path(os.environ['FILESYSTEM_TEST_DIR']) / 'done.txt'\n"
        "sys.exit(0 if p.is_file() else 1)\n",
        encoding="utf-8",
    )
    ep_cli = EpisodeSpec(
        episode_id="TEST_V200_CLI_ADAPTER",
        track_id="O-3",
        domain="office",
        scenario_type="inflight_overturn",
        title="CLI Adapter Overturn E2E",
        upstream_pillars=["mcpmark_filesystem"],
        upstream_task_ids=["fs_cli_1"],
        workspace_mounts=[
            WorkspaceMountSpec(
                mount_id="mnt_office",
                pillar="mcpmark_filesystem",
                mount_relpath="office_fs",
                upstream_id="fs_cli_1",
                repo_or_category="filesystem/desktop",
            )
        ],
        timeline=[
            TimelineEventSpec(
                event_id="ev_launch",
                seq=1,
                kind="user_chat",
                channel="ui://chat/main",
                sender="user",
                text="Write _legacy_draft_manifest.json first.",
                trigger=TriggerSpec(mode="immediate"),
                target_task_tag="office_inflight",
                expected_mmu_behavior="spawn_background_worker",
            ),
            TimelineEventSpec(
                event_id="ev_mid_patch",
                seq=2,
                kind="user_chat",
                channel="ui://chat/main",
                sender="user",
                text="Overturn! Delete _legacy_draft_manifest.json and create done.txt.",
                trigger=TriggerSpec(mode="after_first_mutation", fallback_timeout_sec=5.0),
                target_task_tag="office_inflight",
                expected_mmu_behavior="overturn_and_relaunch",
            ),
        ],
        verifiers=[
            VerifierSpec(
                verifier_id="v_mcp",
                kind="mcpmark_verify_script",
                mount_id="mnt_office",
                check_params={"verify_py": str(verify_done_py)},
            ),
            VerifierSpec(
                verifier_id="v_ov",
                kind="overturn_rollback_check",
                mount_id="mnt_office",
                check_params={"forbidden_rel_paths": ["_legacy_draft_manifest.json"]},
            ),
        ],
    )
    cli_adapter = SubprocessCLIHarnessAdapter("mock_cli", f"{sys.executable} {mock_agent_py}")
    rec_cli = asyncio.run(run_episode_external_adapter(ep_cli, tmp_path / "cli_runs", cli_adapter))
    assert rec_cli["episode_passed"] is True, rec_cli
    assert rec_cli["telemetry"]["mutated_before_overturn"] is True
    assert rec_cli["event_trace"][1]["trigger_fired_via"] == "first_mutation_observed"

    se_val, g_cnt = _compute_cluster_robust_se(
        [
            {"episode_id": "E1", "cluster_key": "repoA", "episode_passed": True},
            {"episode_id": "E2", "cluster_key": "repoA", "episode_passed": True},
            {"episode_id": "E3", "cluster_key": "repoB", "episode_passed": False},
            {"episode_id": "E4", "cluster_key": "repoC", "episode_passed": False},
        ]
    )
    assert g_cnt == 3 and se_val is not None and se_val > 0.0


def test_dsh_symlink_agents_md_and_symmetric_contamination_scanner(tmp_path: Path) -> None:
    """Covers `AGENTS.md -> CLAUDE.md` symlink safety (`PA150_C1_02`), local `sandbox-gh` shims, and `_scan_dsh_session_contamination`."""
    from synaptic_bench_150.runner import _scan_dsh_session_contamination

    # 1. Upstream repo where `AGENTS.md` is a tracked symlink to `CLAUDE.md` (`PA150_C1_02`)
    seed = tmp_path / "symlink_repo"
    seed.mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=seed, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "t@local"], cwd=seed, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Tester"], cwd=seed, check=True, capture_output=True)
    (seed / "CLAUDE.md").write_text("# Original Claude Notes\n", encoding="utf-8")
    (seed / "AGENTS.md").symlink_to("CLAUDE.md")
    subprocess.run(["git", "add", "."], cwd=seed, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init with symlink"], cwd=seed, check=True, capture_output=True)

    paths = init_dsh_sandbox(
        tmp_path / "dsh_world_sym",
        agents_md_override="# Sandbox Override\n",
        seed_repo_dir=seed,
    )
    # `CLAUDE.md` must stay untouched and `AGENTS.md` must be a clean regular file
    assert (paths["clone_repo"] / "CLAUDE.md").read_text(encoding="utf-8") == "# Original Claude Notes\n"
    assert not (paths["clone_repo"] / "AGENTS.md").is_symlink()
    assert "# Sandbox Override" in (paths["clone_repo"] / "AGENTS.md").read_text(encoding="utf-8")
    assert (paths["world"] / "bin" / "gh").is_symlink()
    assert (paths["world"] / "bin" / "curl").is_file()

    # 2. Local `gh pr create` + assistant prose mentioning PR #1 + HTTP must NOT trigger false contamination
    clean_bundle = [
        (
            "worker-1",
            [
                {
                    "type": "tool/call",
                    "data": {
                        "callId": "c1",
                        "name": "bash",
                        "arguments": json.dumps({"command": "gh pr create --title 'Fix CORS' --body 'See https://github.com/Kludex/starlette/pull/1'"}),
                    },
                },
                {
                    "type": "tool/result",
                    "data": {"message": {"toolCallId": "c1", "content": [{"type": "text", "text": "https://github.com/Kludex/starlette/pull/1"}]}},
                },
                {
                    "type": "assistant/message",
                    "data": {
                        "message": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Opened [PR #1](https://github.com/Kludex/starlette/pull/1) via gh and tested HTTP requests with urllib.",
                                }
                            ]
                        }
                    },
                },
            ],
        )
    ]
    assert _scan_dsh_session_contamination(clean_bundle) == (False, 0)

    # 3. Seatbelt-blocked `upstream_pillars` access increments `blocked_count` without marking `contaminated=True`
    blocked_bundle = [
        (
            "worker-2",
            [
                {
                    "type": "tool/call",
                    "data": {
                        "callId": "c2",
                        "name": "bash",
                        "arguments": json.dumps({"command": "cat /path/to/data/upstream_pillars/mcpmark/tasks/foo/verify.py"}),
                    },
                },
                {
                    "type": "tool/result",
                    "data": {
                        "message": {
                            "toolCallId": "c2",
                            "content": [{"type": "text", "text": "cat: .../verify.py: Operation not permitted"}],
                        }
                    },
                },
            ],
        )
    ]
    assert _scan_dsh_session_contamination(blocked_bundle) == (False, 1)

    # 4. Unblocked oracle read MUST mark `contaminated=True`
    leaked_bundle = [
        (
            "worker-3",
            [
                {
                    "type": "tool/call",
                    "data": {
                        "callId": "c3",
                        "name": "bash",
                        "arguments": json.dumps({"command": "cat /path/to/CooperBench/dataset/task1/tests.patch"}),
                    },
                },
                {
                    "type": "tool/result",
                    "data": {
                        "message": {
                            "toolCallId": "c3",
                            "content": [{"type": "text", "text": "diff --git a/tests/test_x.py b/tests/test_x.py"}],
                        }
                    },
                },
            ],
        )
    ]
    assert _scan_dsh_session_contamination(leaked_bundle) == (True, 0)


def test_v201_p1_evaluator_calibration_regressions(tmp_path: Path) -> None:
    """Verify v2.0.1 calibration fixes `P1-1`, `P1-2`, and `P1-3` without weakening real failure gates."""
    from synaptic_bench_150.runner import attribute_dsh_timeline_routing
    from synaptic_bench_150.schema import (
        EpisodeSpec,
        TimelineEventSpec,
        TriggerSpec,
        VerifierSpec,
        WorkspaceMountSpec,
    )
    from synaptic_bench_150.verifiers import (
        _filter_untracked_doc_drafts,
        _verify_readonly_zero_mutation,
    )
    from synaptic_bench_150.workspace_manager import MaterializedEpisodeWorkspace

    # P1-1: Untracked standalone `.md` brief draft stripped, while tracked `.rst` / `.py` edits remain
    mixed_diff = (
        "diff --git a/reports/release_brief.md b/reports/release_brief.md\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/reports/release_brief.md\n"
        "@@ -0,0 +1,2 @@\n"
        "+# Release Brief\n"
        "+All tests passed.\n"
        "diff --git a/CHANGES.rst b/CHANGES.rst\n"
        "--- a/CHANGES.rst\n"
        "+++ b/CHANGES.rst\n"
        "@@ -1,1 +1,2 @@\n"
        "+- Fixed filter\n"
    )
    only_md_draft = (
        "diff --git a/credit_card_release_brief.md b/credit_card_release_brief.md\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/credit_card_release_brief.md\n"
        "@@ -0,0 +1,1 @@\n"
        "+Summary\n"
    )
    assert _filter_untracked_doc_drafts(only_md_draft).strip() == ""
    assert "CHANGES.rst" in _filter_untracked_doc_drafts(mixed_diff)
    assert "release_brief.md" not in _filter_untracked_doc_drafts(mixed_diff)

    # P1-2: Channel-order-invariant DSH attribution when Feishu P2P `user_chat` arrives before GitHub `external_task_webhook`
    ep_c2 = EpisodeSpec(
        episode_id="TEST_V201_C2",
        track_id="C-2",
        domain="coding",
        scenario_type="concurrent_dual_feature",
        title="C2 channel inversion",
        upstream_pillars=["cooperbench"],
        upstream_task_ids=["1"],
        workspace_mounts=[],
        timeline=[
            TimelineEventSpec(
                event_id="ev_feat_a",
                seq=1,
                kind="external_task_webhook",
                channel="github_issue",
                sender="github_user",
                text="Implement bypass()",
                trigger=TriggerSpec(mode="immediate"),
                target_task_tag="feat_a",
            ),
            TimelineEventSpec(
                event_id="ev_dup_feat_a",
                seq=2,
                kind="external_duplicate_webhook",
                channel="feishu_group",
                sender="linear_sync_bot",
                text="Sync: Implement bypass()",
                trigger=TriggerSpec(mode="immediate"),
                target_task_tag="feat_a",
            ),
            TimelineEventSpec(
                event_id="ev_similar_cell_followup",
                seq=3,
                kind="user_chat",
                channel="feishu_p2p",
                sender="user",
                text="For bypass(), keep signature clean.",
                trigger=TriggerSpec(mode="immediate"),
                target_task_tag="feat_a",
                expected_mmu_behavior="similar_cell_disambiguation_followup",
            ),
        ],
        verifiers=[],
    )
    event_trace = [
        {"event_id": "ev_feat_a", "matched_trigger_id": "tr_gh_a"},
        {"event_id": "ev_dup_feat_a", "matched_trigger_id": "tr_dup_a"},
        {"event_id": "ev_similar_cell_followup", "matched_trigger_id": "tr_chat_a"},
    ]
    triggers = [
        {"id": "tr_chat_a", "disposition": {"decision": {"outcome": "accept", "ticketId": "tk_a"}}},
        {"id": "tr_gh_a", "disposition": {"decision": {"outcome": "ignore"}}},
        {"id": "tr_dup_a", "disposition": {"decision": {"outcome": "ignore"}}},
    ]
    tickets = {"tk_a": {"id": "tk_a", "origin": {"triggerId": "tr_chat_a"}}}
    attr = attribute_dsh_timeline_routing(
        episode=ep_c2,
        event_trace=event_trace,
        triggers=triggers,
        ticket_by_id=tickets,
        use_existing_matched_ids=True,
    )
    assert attr["real_tasks_dropped"] == 0
    assert attr["dup_merged_exact"] == 1
    assert attr["routing_events"]["ev_similar_cell_followup"]["routed_tags"] == ["feat_a"]
    assert attr["routing_events"]["ev_similar_cell_followup"]["spawned_new_worker"] is False

    # P1-3: Gaia2 `Cabs__get_quotation` (`write_operation=False`) passes readonly check, while real write fails
    ep_o1 = EpisodeSpec(
        episode_id="TEST_V201_O1",
        track_id="O-1",
        domain="office",
        scenario_type="readonly_alignment",
        title="O1 cabs quote readonly",
        upstream_pillars=["gaia2"],
        upstream_task_ids=["g1"],
        workspace_mounts=[
            WorkspaceMountSpec(
                mount_id="mnt_office_main",
                pillar="gaia2",
                mount_relpath="gaia2",
                upstream_id="g1",
                repo_or_category="execution",
            )
        ],
        timeline=[],
        verifiers=[],
    )
    ws_o1 = MaterializedEpisodeWorkspace(
        episode=ep_o1,
        arm_name="synaptic_pa_dsh",
        root_dir=tmp_path,
        repo_dir=tmp_path,
        initial_snapshots={"mnt_office_main": {"apps_state_hash": "h0", "call_log": []}},
        mid_turn_snapshots={
            ("ev_align", "mnt_office_main"): {
                "apps_state_hash": "h1_quote_cache_drifted",
                "mutating_call_count": 1,
                "call_log": [
                    {"tool_name": "Cabs__get_quotation", "write_operation": False, "mutated": True},
                    {"tool_name": "Cabs__get_current_ride_status", "ok": False, "mutated": False},
                ],
            },
            ("ev_write_leak", "mnt_office_main"): {
                "apps_state_hash": "h2",
                "mutating_call_count": 1,
                "call_log": [
                    {"tool_name": "Cabs__order_ride", "write_operation": True, "mutated": True},
                ],
            },
        },
    )
    res_quote = _verify_readonly_zero_mutation(
        VerifierSpec(
            verifier_id="v_ro_ok",
            kind="readonly_zero_mutation_check",
            mount_id="mnt_office_main",
            check_params={"after_event_id": "ev_align"},
        ),
        ws_o1,
    )
    assert res_quote["passed"] is True

    res_write = _verify_readonly_zero_mutation(
        VerifierSpec(
            verifier_id="v_ro_fail",
            kind="readonly_zero_mutation_check",
            mount_id="mnt_office_main",
            check_params={"after_event_id": "ev_write_leak"},
        ),
        ws_o1,
    )
    assert res_write["passed"] is False
