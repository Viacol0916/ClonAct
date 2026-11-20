"""Unit tests for Plan A1 Paper Experiments v1 (`PA-60` subset, 3 upper-layer paradigm baselines, and 4 `ClonAct` ablations)."""
from __future__ import annotations

from fake_dsh import FakeDSHHost

import asyncio
import json
from pathlib import Path

from appworld_micro_30.runner import MICRO_CLONACT_ABLATION_CONFIGS, SUPPORTED_MICRO_ARMS
from synaptic_bench_150.build_pa60_subset import PA60_MANIFEST_JSON, build_pa60_manifest
from synaptic_bench_150.runner import (
    BUILTIN_ARMS,
    CLONACT_ABLATION_CONFIGS,
    _parse_arms_arg,
    aggregate_benchmark_results,
    load_pa150_episodes,
)
from clonact.context_mmu import ContextMMU
from clonact.llm_client import HomogeneousLLMClient
from clonact.paradigm_baselines import (
    DSHParadigmDriver,
    PARADIGM_BASELINE_ARMS,
    build_paradigm_prompts,
    format_raw_inbound_event,
)
from clonact.system1_judge import ChoiceSpec, GenerativeLLMJudge, NoulSpec, System1Judge
from clonact.trunk import ElasticSharedTrunk, TrunkEntry


def test_pa60_manifest_invariants_and_ipw_sum() -> None:
    manifest = build_pa60_manifest()
    assert PA60_MANIFEST_JSON.is_file()
    saved = json.loads(PA60_MANIFEST_JSON.read_text(encoding="utf-8"))
    assert saved["episode_ids"] == manifest["episode_ids"]
    assert len(manifest["episode_ids"]) == 60
    assert manifest["domain_counts"] == {"coding": 22, "hybrid": 20, "office": 18}

    weights = manifest["ipw_weights"]
    assert len(weights) == 60
    assert round(sum(weights.values()), 6) == 150.0

    # All 10 C-2 true-conflict episodes must be selected with Horvitz-Thompson weight 1.0
    c2_conf_ids = manifest["strata"]["C-2_conflict"]["selected_episode_ids"]
    assert len(c2_conf_ids) == 10
    assert all(weights[eid] == 1.0 for eid in c2_conf_ids)

    # Subset loader returns exact 60 episodes
    subset_eps = load_pa150_episodes(subset="pa60")
    assert [e.episode_id for e in subset_eps] == manifest["episode_ids"]


def test_pa60_ipw_aggregation_in_summary() -> None:
    manifest = json.loads(PA60_MANIFEST_JSON.read_text(encoding="utf-8"))
    mock_records = []
    for idx, eid in enumerate(manifest["episode_ids"]):
        passed = idx % 2 == 0
        mock_records.append(
            {
                "episode_id": eid,
                "track_id": "C-1",
                "domain": "coding",
                "arm_name": "clonact_no_synapse",
                "cluster_key": eid,
                "episode_passed": passed,
                "verifiers_passed": 2 if passed else 1,
                "verifiers_total": 2,
                "task_solve_passed": 1 if passed else 0,
                "task_solve_total": 1,
                "orchestration_passed": 1,
                "orchestration_total": 1,
                "verifiers": [],
                "telemetry": {"wall_clock_sec": 10.0, "estimated_cost_cny": 0.05},
            }
        )
    summary = aggregate_benchmark_results(mock_records)
    arm_sum = summary["arms"]["clonact_no_synapse"]
    assert "pa60_ipw_estimates" in arm_sum
    ipw = arm_sum["pa60_ipw_estimates"]
    assert ipw["weighted_population_size"] == 150.0
    assert 0.0 < ipw["ipw_episode_pass_rate"] < 1.0
    assert 0.0 < ipw["ipw_task_solve_pass_rate"] < 1.0


def test_runner_arm_parsers_accept_all_plan_a1_arms() -> None:
    expected_arms = {
        "clonact",
        "dsh_superagent",
        "dsh_mono",
        "dsh_subagent_fork",
        "dsh_agent_team",
        "clonact_no_synapse",
        "clonact_no_same_brain",
        "clonact_llm_mmu",
        "clonact_no_prefix",
    }
    assert expected_arms.issubset(set(BUILTIN_ARMS))
    assert expected_arms.issubset(set(SUPPORTED_MICRO_ARMS))
    assert set(CLONACT_ABLATION_CONFIGS.keys()) == set(MICRO_CLONACT_ABLATION_CONFIGS.keys())
    parsed = _parse_arms_arg(",".join(sorted(expected_arms)))
    assert {name for name, cmd in parsed if cmd is None} == expected_arms


def test_generative_llm_judge_json_extraction_and_coercion() -> None:
    raw = """```json
    {
      "is_noise": {"noul": 0.91},
      "route": {"choice": "existing_cell", "confidence": 0.88}
    }
    ```"""
    extracted = GenerativeLLMJudge._extract_json_object(raw)
    specs = {
        "is_noise": NoulSpec("Is this non-actionable noise?"),
        "route": ChoiceSpec(
            "Select routing target.",
            {"new_cell": "Spawn new cell", "existing_cell": "Route to existing cell"},
        ),
    }
    coerced = GenerativeLLMJudge._coerce_answers(extracted, specs)
    assert coerced["is_noise"] == {"type": "noul", "noul": 0.91}
    assert coerced["route"]["type"] == "choice"
    assert coerced["route"]["choice"] == "existing_cell"
    assert coerced["route"]["confidence"] == 0.88


def test_clonact_ablation_flags_single_variable_isolation(tmp_path: Path) -> None:
    ws_root = tmp_path / "ws"
    repo_dir = ws_root / "repo"
    docs_dir = ws_root / "docs"
    repo_dir.mkdir(parents=True)
    docs_dir.mkdir(parents=True)
    (repo_dir / "AGENTS.md").write_text("# Workspace Secret Guide\nUse custom helper.", encoding="utf-8")

    # 1. `clonact_no_prefix`: strips workspace_profile, AGENTS.md auto-injection, and L2/L3 promotions
    trunk_np = ElasticSharedTrunk(
        l1_pinned_rules=["User Identity: @user"],
        disable_elastic_l2_l3=True,
    )
    sample_entry = TrunkEntry(
        entry_id="k1",
        category="verified_delta",
        summary="fact 1",
        anchor_type="commit_sha",
        anchor_value="abc1234",
    )
    assert trunk_np.promote(sample_entry) is False
    assert trunk_np.epoch == 1
    assert len(trunk_np.l2_entries) == 0

    mmu_np = ContextMMU(
        sandbox_paths={"world": ws_root, "repo": repo_dir, "clone_repo": repo_dir, "worktrees": None, "docs": docs_dir},
        judge=System1Judge(deepseek_api_key="test"),
        llm=HomogeneousLLMClient(api_key="test"),
        trunk=trunk_np,
        disable_prefix_and_grounding=True,
        dsh_host=FakeDSHHost(),
    )
    assert mmu_np._build_jev_context_prefix()["workspace_profile"] == {}
    assert "Workspace Secret Guide" not in trunk_np.render_prefix_messages()[0]["content"]

    # 2. `clonact_no_synapse` + `clonact_no_same_brain` + `clonact_llm_mmu` (`bus_judge` decoupled!)
    mmu_judge = GenerativeLLMJudge(api_key="test")
    bus_judge = System1Judge(deepseek_api_key="test")
    mmu_ab = ContextMMU(
        sandbox_paths={"world": ws_root, "repo": repo_dir, "clone_repo": repo_dir, "worktrees": None, "docs": docs_dir},
        judge=mmu_judge,
        bus_judge=bus_judge,
        llm=HomogeneousLLMClient(api_key="test"),
        trunk=ElasticSharedTrunk(l1_pinned_rules=["User Identity: @user"]),
        enable_synaptic_bus=False,
        enable_peek_peer_cell=False,
        enable_synaptic_reconcile=False,
        disable_same_brain_continuity=True,
        dsh_host=FakeDSHHost(),
    )
    assert mmu_ab.judge is mmu_judge
    assert mmu_ab.bus.judge is bus_judge
    assert mmu_ab.bus.enabled is False

    c1 = mmu_ab._spawn_cell(
        cell_id="Cell_1",
        topic="Fix parser bug",
        source_kind="user_chat",
        initial_status="in_focus_aligning",
    )
    assert c1.enable_peek_peer_cell is False
    assert c1.enable_synaptic_reconcile is False
    assert c1.disable_same_brain_continuity is True


def test_paradigm_baselines_formatting_board_and_hygiene(tmp_path: Path) -> None:
    # Raw inbound formatting must never leak benchmark `kind` labels
    ext_txt = format_raw_inbound_event(
        kind="external_noise",
        channel="feishu_group",
        sender="alice",
        event_id="ev_noise_1",
        text="大家中午吃什么？",
    )
    assert "external_noise" not in ext_txt
    assert "[INBOUND EXTERNAL EVENT | channel=feishu_group | sender=alice | event_id=ev_noise_1]" in ext_txt
    assert format_raw_inbound_event(
        kind="user_chat",
        channel="feishu_dm",
        sender="user",
        event_id="ev_u1",
        text="帮我查一下账单",
    ) == "帮我查一下账单"

    for arm in PARADIGM_BASELINE_ARMS:
        t_prompt, m_prompt = build_paradigm_prompts(arm, agents_md="# Guide", has_code_repo=True)
        assert "<workspace_agents_md>" in t_prompt
        assert arm in m_prompt

    # Verify `team_board` file-claim overlap detection and 1-to-1 subagent worker attribution
    repo = tmp_path / "repo"
    repo.mkdir()
    drv_team = DSHParadigmDriver(
        "dsh_agent_team",
        instance_dir=tmp_path / "inst_team",
        allowed_workspace_root=tmp_path,
        repo_dir=repo,
        worktrees_dir=None,
        agents_md="# Guide",
        has_code_repo=False,
    )

    async def _exercise_board() -> None:
        r1 = await drv_team._on_custom_tool(
            "feat_a",
            "tm_feat_a",
            "team_board",
            {"action": "claim_files", "files": ["./src/core.py"]},
        )
        assert "no current overlaps" in r1
        r2 = await drv_team._on_custom_tool(
            "feat_b",
            "tm_feat_b",
            "team_board",
            {"action": "claim_files", "files": ["src/core.py"]},
        )
        assert "WARNING — file overlap" in r2

    asyncio.run(_exercise_board())

    drv_fork = DSHParadigmDriver(
        "dsh_subagent_fork",
        instance_dir=tmp_path / "inst_fork",
        allowed_workspace_root=tmp_path,
        repo_dir=repo,
        worktrees_dir=None,
        agents_md="# Guide",
        has_code_repo=False,
    )

    async def _exercise_subagent_hygiene() -> None:
        drv_fork._inbound_history.append(
            {
                "event_id": "ev_task_1",
                "kind": "external_task_webhook",
                "channel": "github_issue",
                "sender": "bob",
                "text": "Fix splitwise expense rounding bug in billing module",
                "tokens": {"fix", "splitwise", "expense", "rounding", "bug", "billing", "module"},
                "target_task_tag": "subtask_1",
                "workers_before": 0,
                "primary_mutations_before": 0,
                "total_mutations_before": 0,
                "was_busy_at_arrival": False,
            }
        )
        drv_fork._last_inbound_event_id = "ev_task_1"
        drv_fork._active_task_tag = "subtask_1"
        await drv_fork._on_session_event(
            {
                "kind": "tool_call",
                "session_id": "delegator_main",
                "is_subagent": False,
                "tool_call_id": "tc_1",
                "tool_name": "subagent",
                "tool_args": {"prompt": "Fix splitwise expense rounding bug in billing module"},
            }
        )
        # Child subagent starts after a duplicate webhook arrives — must bind 1-to-1 to `subcall_tc_1`!
        drv_fork._inbound_history.append(
            {
                "event_id": "ev_dup_1",
                "kind": "external_duplicate_webhook",
                "channel": "feishu_group",
                "sender": "ops_bot",
                "text": "Reminder: Fix splitwise expense rounding bug in billing module",
                "tokens": {"reminder", "fix", "splitwise", "expense", "rounding", "bug", "billing", "module"},
                "target_task_tag": "subtask_1",
                "workers_before": 1,
                "primary_mutations_before": 0,
                "total_mutations_before": 0,
                "was_busy_at_arrival": True,
            }
        )
        drv_fork._last_inbound_event_id = "ev_dup_1"
        await drv_fork._on_session_event(
            {
                "kind": "tool_result",
                "session_id": "child_uuid_1",
                "is_subagent": True,
                "tool_name": "bash",
                "tool_args": {"command": "./aw exec 'print(1)'"},
                "is_error": False,
            }
        )
        drv_fork._inbound_history.append(
            {
                "event_id": "ev_noise_1",
                "kind": "external_noise",
                "channel": "feishu_group",
                "sender": "carol",
                "text": "Anyone ordering coffee from Luckin this afternoon?",
                "tokens": {"anyone", "ordering", "coffee", "luckin", "afternoon"},
                "target_task_tag": None,
                "workers_before": 1,
                "primary_mutations_before": 0,
                "total_mutations_before": 1,
                "was_busy_at_arrival": False,
            }
        )
        assert len(drv_fork.spawned_workers) == 1
        assert drv_fork.spawned_workers[0]["bound_sid"] == "child_uuid_1"
        assert drv_fork.spawned_workers[0]["mutated"] is True
        hyg = drv_fork.compute_hygiene_attribution()
        assert hyg == {
            "noise_total": 1,
            "noise_dropped": 1,
            "noise_spawned_workers": 0,
            "noise_mutated_workers": 0,
            "duplicates_merged": 1,
            "duplicate_spawned_workers": 0,
            "real_tasks_dropped": 0,
        }

    asyncio.run(_exercise_subagent_hygiene())


def test_cross_model_routing_and_serialization() -> None:
    from dsh_runner.dsh_controller import build_dsh_llm_patch_yaml, build_dsh_model_candidates
    from clonact.llm_client import resolve_backbone_model_spec

    # 1. Provider & protocol resolution
    s_ds = resolve_backbone_model_spec("deepseek-flash")
    assert s_ds["protocol"] == "deepseek_anthropic"
    assert s_ds["dsh_provider"] == "deepseek-official"
    assert build_dsh_llm_patch_yaml("deepseek-flash") == ""

    s_oai = resolve_backbone_model_spec("gpt-6-luna")
    assert s_oai["protocol"] == "openai_responses"
    assert s_oai["dsh_provider"] == "openai"
    assert s_oai["endpoint_url"].endswith("/responses")
    oai_yaml = build_dsh_llm_patch_yaml("gpt-6-luna")
    assert "provider: openai" in oai_yaml
    assert "id: gpt-6-luna" in oai_yaml
    oai_cands = build_dsh_model_candidates("gpt-6-luna")
    assert all(c["provider"] == "openai" and c["model"] == "gpt-6-luna" for c in oai_cands)

    s_qw = resolve_backbone_model_spec("qwen3.8-flash")
    assert s_qw["protocol"] == "qwen_completions"
    assert s_qw["dsh_provider"] == "qwen-token-plan-cn"
    assert s_qw["endpoint_url"].endswith("/chat/completions")
    qw_yaml = build_dsh_llm_patch_yaml("qwen3.8-flash")
    assert "provider: qwen-token-plan-cn" in qw_yaml
    assert "high: xhigh" in qw_yaml
    qw_cands = build_dsh_model_candidates("qwen3.8-flash")
    assert all(c["provider"] == "qwen-token-plan-cn" and c["model"] == "qwen3.8-flash" for c in qw_cands)

    # 2. Multi-turn message serialization for OpenAI Responses & Qwen Completions
    sample_msgs = [
        {"role": "system", "content": "Top instructions."},
        {"role": "user", "content": "Check status."},
        {
            "role": "assistant",
            "content": "Running tool.",
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {"name": "bash", "arguments": '{"cmd": "ls"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_123", "content": "file1.txt"},
        {"role": "system", "content": "Mid-convo update."},
        {"role": "user", "content": "Summarize."},
    ]
    instr, items = HomogeneousLLMClient._serialize_messages_to_openai_responses(sample_msgs)
    assert instr == "Top instructions."
    assert any(it.get("type") == "function_call" and it.get("call_id") == "call_123" for it in items)
    assert any(it.get("type") == "function_call_output" and it.get("call_id") == "call_123" for it in items)

    qw_msgs = HomogeneousLLMClient._serialize_messages_to_qwen_completions(sample_msgs)
    assert qw_msgs[0] == {"role": "system", "content": "Top instructions."}
    assert all(m["role"] != "developer" for m in qw_msgs)
    assert any(m["role"] == "tool" and m["tool_call_id"] == "call_123" for m in qw_msgs)
