"""Unit & live integration tests for the 3-module Synaptic-PA v8 upgrade:
1. `JevContextPrefix` (workspace profile + active & completed cell timeline + touched domains)
2. Unified Intra-Turn & Inter-Turn Causal Continuity + LLM Splitter Safeguard + 5-Way Noise Triage
3. Pluggable `CellExecutionBackend` (`"python"` <-> `"dsh_agent"`) with zero-worktree alignment & 3-phase synaptic convergence.
"""
from __future__ import annotations

from fake_dsh import FakeDSHHost

import asyncio
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from dsh_runner.synaptic_worker_controller import USER_CREDENTIALS, SynapticDSHWorkerHost
from clonact.context_mmu import ContextMMU
from clonact.synaptic_bus import extract_cli_and_fs_domains
from clonact.system1_judge import JudgmentResult
from clonact.trunk import ElasticSharedTrunk


def _choice_ans(val: str, conf: float = 0.95) -> dict[str, Any]:
    return {"type": "choice", "choice": val, "confidence": conf, "probabilities": {val: conf}}


def _noul_ans(val: float) -> dict[str, Any]:
    return {"type": "noul", "noul": val}


class ScriptedJudge:
    """Deterministic stub matching `System1Judge.decide` interface while recording `state` & `questions`."""

    def __init__(self, handler: Any) -> None:
        self._handler = handler
        self.call_history: list[dict[str, Any]] = []
        self.recorded_states: list[dict[str, Any]] = []
        self.recorded_questions: list[dict[str, Any]] = []

    async def decide(self, state: dict[str, Any], questions: dict[str, Any], tag: str = "stub") -> JudgmentResult:
        self.recorded_states.append(state)
        self.recorded_questions.append(questions)
        overrides = self._handler(state, questions, tag) or {}
        answers: dict[str, Any] = {}
        for qid, spec in questions.items():
            if qid in overrides:
                raw = overrides[qid]
                if isinstance(raw, dict) and "type" in raw:
                    answers[qid] = raw
                elif hasattr(spec, "criteria"):
                    answers[qid] = _choice_ans(str(raw))
                else:
                    answers[qid] = _noul_ans(float(raw))
            elif hasattr(spec, "criteria") and isinstance(spec.criteria, dict):
                first_k = next(iter(spec.criteria.keys()))
                answers[qid] = _choice_ans(first_k)
            else:
                answers[qid] = _noul_ans(0.05)
        return JudgmentResult(answers=answers, latency_ms=12.0, backend_used="scripted_stub")


class ScriptedLLM:
    """Deterministic stub matching `HomogeneousLLMClient.chat` & `chat_fast_utility`."""

    def __init__(self, handler: Any | None = None) -> None:
        self._handler = handler
        self.calls: list[dict[str, Any]] = []
        self.metrics: list[Any] = []

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        caller: str = "",
        **kwargs: Any,
    ) -> dict[str, Any]:
        self.calls.append({"caller": caller, "messages": list(messages), "tools": tools})
        if self._handler is not None:
            return self._handler(caller, messages, tools)
        return {"role": "assistant", "content": f"Completed turn for {caller}.", "tool_calls": None}

    async def chat_fast_utility(
        self,
        messages: list[dict[str, Any]],
        caller: str = "",
        max_tokens: int = 512,
    ) -> dict[str, Any]:
        self.calls.append({"caller": caller, "messages": list(messages), "tools": None, "max_tokens": max_tokens})
        if self._handler is not None:
            return self._handler(caller, messages, None)
        return {"role": "assistant", "content": "{}"}


def _init_git_repo(repo_dir: Path) -> None:
    repo_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-b", "main"], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@local"], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_dir, check=True, capture_output=True)
    (repo_dir / "AGENTS.md").write_text(
        "# Test Workspace\n- Repository: `demo/calc`\n- App CLI: `./app_cli call Shopping__checkout`\n",
        encoding="utf-8",
    )
    (repo_dir / "calc.py").write_text(
        "def compute_total(items: list[int]) -> int:\n    return sum(items)\n",
        encoding="utf-8",
    )
    (repo_dir / "test_calc.py").write_text(
        "from calc import compute_total\n\ndef test_base():\n    assert compute_total([2, 3]) == 5\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo_dir, check=True, capture_output=True)


def test_extract_cli_and_fs_domains_tags() -> None:
    tags = extract_cli_and_fs_domains(
        "bash",
        {"command": "./gaia_cli call Shopping__checkout '{\"cart_id\": 1}' && ./pg_exec \"SELECT 1\""},
    )
    assert "cli:gaia_cli" in tags
    assert "cli:gaia_cli/shopping" in tags
    assert "cli:pg_exec" in tags

    fs_tags = extract_cli_and_fs_domains("read", {"file_path": "office_fs/q3_budget.md"})
    assert "fs:office_fs" in fs_tags
    assert "file:q3_budget.md" in fs_tags


def test_jev_context_prefix_includes_workspace_and_completed_cells(tmp_path: Path) -> None:
    async def _run() -> None:
        repo = tmp_path / "repo"
        _init_git_repo(repo)
        docs_dir = tmp_path / "docs_mount"
        docs_dir.mkdir()

        def _judge_fn(_state: dict[str, Any], _questions: dict[str, Any], _tag: str) -> dict[str, Any]:
            return {
                "turn_granularity": "single_intent",
                "primary_route": "NEW_CELL",
                "interaction_intent": "commit_to_execute",
            }

        judge = ScriptedJudge(_judge_fn)
        llm = ScriptedLLM()
        trunk = ElasticSharedTrunk(l1_pinned_rules=[(repo / "AGENTS.md").read_text(encoding="utf-8")])
        mmu = ContextMMU(
            sandbox_paths={"world": tmp_path, "repo": repo, "docs": docs_dir},
            judge=judge,  # type: ignore[arg-type]
            llm=llm,  # type: ignore[arg-type]
            trunk=trunk,
            dsh_host=FakeDSHHost(),
        )

        out1 = await mmu.handle_user_message("Add items to cart and checkout order #101.", auto_run_detached=True)
        await mmu.drain_all_background_work()
        c1_id = out1["routed_to"]
        assert mmu.cells[c1_id].status == "completed"
        mmu.bus.footprints[c1_id].touched_domains.add("cli:app_cli/shopping")

        prefix = mmu._build_jev_context_prefix()
        assert "workspace_profile" in prefix
        assert prefix["workspace_profile"]["has_git_repository"] is True
        assert "Shopping__checkout" in prefix["workspace_profile"]["workspace_guide_excerpt"]
        assert c1_id in prefix["cell_timeline"]
        assert prefix["cell_timeline"][c1_id]["status"] == "completed"
        assert "cli:app_cli/shopping" in prefix["cell_timeline"][c1_id]["touched_domains"]
        assert len(prefix["recent_dialogue_tail"]) >= 1

    asyncio.run(_run())


def test_causal_continuation_resumes_completed_cell_without_synthetic_prompt_hacks(tmp_path: Path) -> None:
    async def _run() -> None:
        repo = tmp_path / "repo"
        _init_git_repo(repo)

        call_seq = {"route_calls": 0}

        def _judge_fn(_state: dict[str, Any], questions: dict[str, Any], _tag: str) -> dict[str, Any]:
            if "primary_route" in questions:
                call_seq["route_calls"] += 1
                if call_seq["route_calls"] == 1:
                    return {
                        "turn_granularity": "single_intent",
                        "primary_route": "NEW_CELL",
                        "interaction_intent": "commit_to_execute",
                    }
                # Completed cell MUST remain visible in `primary_route` criteria for causal continuation!
                assert "Cell_1" in questions["primary_route"].criteria
                return {
                    "turn_granularity": "single_intent",
                    "primary_route": "Cell_1",
                    "interaction_intent": "commit_to_execute",
                    "deliverable_role": "direct_deliverable",
                }
            return {}

        judge = ScriptedJudge(_judge_fn)
        llm = ScriptedLLM()
        mmu = ContextMMU(
            sandbox_paths={"world": tmp_path, "repo": repo},
            judge=judge,  # type: ignore[arg-type]
            llm=llm,  # type: ignore[arg-type]
            trunk=ElasticSharedTrunk(l1_pinned_rules=["Workspace rule"]),
            dsh_host=FakeDSHHost(),
        )

        first = await mmu.handle_user_message("Save the 3 filtered apartments in RentAFlat.", auto_run_detached=True)
        await mmu.drain_all_background_work()
        assert first["routed_to"] == "Cell_1"
        assert mmu.cells["Cell_1"].status == "completed"

        second = await mmu.handle_user_message(
            "Now also message the landlord of the cheapest one from those saved apartments.",
            auto_run_detached=True,
        )
        await mmu.drain_all_background_work()

        assert second["routed_to"] == "Cell_1"
        assert second["transition"] == "resume_completed_cell_execution"
        assert second["spawned_worker_cells"] == []
        assert len(mmu.cells) == 1

        # Verify ZERO synthetic prompt hacks ("RESUMED CELL — DO NOT REPEAT...") and no duplicate kickoff boilerplate
        all_text_blobs = "\n".join(
            str(m.get("content") or "")
            for call in llm.calls
            for m in call["messages"]
        )
        assert "RESUMED CELL" not in all_text_blobs
        assert "DO NOT REPEAT COMPLETED EXTERNAL WRITES" not in all_text_blobs
        host = mmu.dsh_host
        exec_messages = [m for m in host.messages if m["session_id"] == "worker-Cell_1"]
        assert len(exec_messages) == 2
        assert "Save the 3 filtered apartments" in exec_messages[0]["text"]
        assert "Save the 3 filtered apartments" not in exec_messages[1]["text"]
        assert len(host.sessions) == 1  # continuation reuses the same DSH session

    asyncio.run(_run())


def test_three_way_turn_granularity_and_llm_splitter_causal_safeguard(tmp_path: Path) -> None:
    async def _run() -> None:
        repo = tmp_path / "repo"
        _init_git_repo(repo)

        granularity_mode = {"val": "coupled_causal_workflow"}

        def _judge_fn(_state: dict[str, Any], questions: dict[str, Any], _tag: str) -> dict[str, Any]:
            if "turn_granularity" in questions:
                assert set(questions["turn_granularity"].criteria.keys()) == {
                    "single_intent",
                    "coupled_causal_workflow",
                    "independent_parallel_tasks",
                }
                return {
                    "turn_granularity": granularity_mode["val"],
                    "primary_route": "NEW_CELL",
                    "interaction_intent": "commit_to_execute",
                }
            return {
                "primary_route": "NEW_CELL",
                "interaction_intent": "commit_to_execute",
            }

        def _llm_fn(caller: str, _messages: list[dict[str, Any]], _tools: Any) -> dict[str, Any]:
            if caller == "mmu:intent_split":
                # Second-line LLM safeguard refuses to split causally coupled steps and returns 1 unified clause!
                return {
                    "role": "assistant",
                    "content": '{"clauses": ["Add the laptop to cart and then immediately complete checkout and email the receipt."]}',
                }
            return {"role": "assistant", "content": "Executed unified workflow.", "tool_calls": None}

        judge = ScriptedJudge(_judge_fn)
        llm = ScriptedLLM(_llm_fn)
        mmu = ContextMMU(
            sandbox_paths={"world": tmp_path, "repo": repo},
            judge=judge,  # type: ignore[arg-type]
            llm=llm,  # type: ignore[arg-type]
            trunk=ElasticSharedTrunk(l1_pinned_rules=["Workspace rule"]),
            dsh_host=FakeDSHHost(),
        )

        # Case A: Jev directly classifies as `coupled_causal_workflow` -> zero split LLM call, single cell
        res_a = await mmu.handle_user_message(
            "Find Alice's email address in Contacts and send her the Q4 summary.",
            auto_run_detached=True,
        )
        await mmu.drain_all_background_work()
        assert res_a["transition"] != "compound_multi_intent"
        assert not any(c["caller"] == "mmu:intent_split" for c in llm.calls)
        assert len(mmu.cells) == 1

        # Case B: Jev tentatively selects `independent_parallel_tasks`, but LLM safeguard returns 1 clause -> single cell fallback
        granularity_mode["val"] = "independent_parallel_tasks"
        res_b = await mmu.handle_user_message(
            "Add the laptop to cart and then immediately complete checkout and email the receipt.",
            auto_run_detached=True,
        )
        await mmu.drain_all_background_work()
        assert any(c["caller"] == "mmu:intent_split" for c in llm.calls)
        assert res_b["transition"] != "compound_multi_intent"
        assert len(mmu.cells) == 2

    asyncio.run(_run())


def test_five_way_inbound_event_disposition_and_completed_cell_dedup_vs_delta(tmp_path: Path) -> None:
    async def _run() -> None:
        repo = tmp_path / "repo"
        _init_git_repo(repo)

        mode = {"step": "spawn"}

        def _judge_fn(_state: dict[str, Any], questions: dict[str, Any], _tag: str) -> dict[str, Any]:
            if "event_disposition" in questions:
                if mode["step"] == "spawn":
                    return {
                        "event_disposition": "new_workspace_task",
                        "is_actionable": 0.95,
                        "bind_target": "NEW_CELL",
                    }
                if mode["step"] == "facility_noise":
                    return {
                        "event_disposition": "passive_admin_or_facility_broadcast",
                        "is_actionable": 0.80,  # Even if Noul leans high, non-actionable disposition drops!
                        "bind_target": "NEW_CELL",
                    }
                if mode["step"] == "completed_dup":
                    assert "Cell_E_1" in questions["bind_target"].criteria
                    return {
                        "event_disposition": "causal_followup_or_duplicate",
                        "is_actionable": 0.95,
                        "bind_target": "Cell_E_1",
                        "settled_followup_kind": "pure_duplicate_already_handled",
                    }
                if mode["step"] == "completed_delta":
                    return {
                        "event_disposition": "causal_followup_or_duplicate",
                        "is_actionable": 0.95,
                        "bind_target": "Cell_E_1",
                        "settled_followup_kind": "new_delta_to_execute",
                    }
            return {}

        judge = ScriptedJudge(_judge_fn)
        llm = ScriptedLLM()
        mmu = ContextMMU(
            sandbox_paths={"world": tmp_path, "repo": repo},
            judge=judge,  # type: ignore[arg-type]
            llm=llm,  # type: ignore[arg-type]
            trunk=ElasticSharedTrunk(l1_pinned_rules=["Workspace rule"]),
            dsh_host=FakeDSHHost(),
        )

        r1 = await mmu.ingest_external_event(
            {"event_id": "ev1", "channel": "github://issues", "sender": "alice", "text": "Fix login timeout bug"},
            auto_run_background=True,
        )
        await mmu.drain_all_background_work()
        assert r1["action"] == "spawned_proactive_cell"
        assert mmu.cells["Cell_E_1"].status == "completed"

        mode["step"] = "facility_noise"
        r_noise = await mmu.ingest_external_event(
            {
                "event_id": "ev2",
                "channel": "feishu://group",
                "sender": "ops_notice",
                "text": "Heads up: Building B freight elevator maintenance from 14:00 to 16:00 today.",
            },
            auto_run_background=True,
        )
        assert r_noise["action"] == "dropped_by_system1"
        assert mmu.dropped_noise_events == 1

        mode["step"] = "completed_dup"
        r_dup = await mmu.ingest_external_event(
            {
                "event_id": "ev3",
                "channel": "feishu://group",
                "sender": "alice",
                "text": "@agent Mirroring GitHub issue: please fix the login timeout bug.",
            },
            auto_run_background=True,
        )
        assert r_dup["action"] == "merged_into_existing"
        assert r_dup["cell_id"] == "Cell_E_1"
        assert mmu.merged_duplicate_events == 1

        mode["step"] = "completed_delta"
        r_delta = await mmu.ingest_external_event(
            {
                "event_id": "ev4",
                "channel": "feishu://group",
                "sender": "alice",
                "text": "@agent Follow-up on login timeout: also log a warning when retry_count > 2.",
            },
            auto_run_background=True,
        )
        await mmu.drain_all_background_work()
        assert r_delta["action"] == "resumed_completed_cell"
        assert r_delta["cell_id"] == "Cell_E_1"
        assert len(mmu.cells) == 1

    asyncio.run(_run())


@pytest.mark.skipif(not USER_CREDENTIALS.is_file(), reason="Requires local DSH credentials for live dsh_agent smoke")
@pytest.mark.model_api
def test_pluggable_dsh_agent_backend_zero_worktree_alignment_and_conflict_reconciliation(tmp_path: Path) -> None:
    async def _run() -> None:
        repo = tmp_path / "repo"
        worktrees_dir = tmp_path / "worktrees"
        _init_git_repo(repo)

        step_state = {"n": 0}

        def _judge_fn(_state: dict[str, Any], questions: dict[str, Any], _tag: str) -> dict[str, Any]:
            if "primary_route" in questions:
                step_state["n"] += 1
                if step_state["n"] == 1:
                    # First turn: read-only alignment -> must NOT create a git worktree!
                    return {
                        "turn_granularity": "single_intent",
                        "primary_route": "NEW_CELL",
                        "interaction_intent": "align_or_query",
                    }
                if step_state["n"] == 2:
                    # Second turn: authorize Cell 1 execution
                    return {
                        "turn_granularity": "single_intent",
                        "primary_route": "Cell_1",
                        "interaction_intent": "commit_to_execute",
                        "deliverable_role": "direct_deliverable",
                    }
                # Third turn: spawn independent Cell 2 touching the same `compute_total` function!
                return {
                    "turn_granularity": "single_intent",
                    "primary_route": "NEW_CELL",
                    "interaction_intent": "commit_to_execute",
                    "deliverable_role": "direct_deliverable",
                }
            return {}

        host = SynapticDSHWorkerHost(
            tmp_path / "dsh_worker_instance",
            allowed_workspace_root=tmp_path,
        )
        llm = ScriptedLLM()
        judge = ScriptedJudge(_judge_fn)
        mmu = ContextMMU(
            sandbox_paths={
                "world": tmp_path,
                "repo": repo,
                "clone_repo": repo,
                "worktrees": worktrees_dir,
            },
            judge=judge,  # type: ignore[arg-type]
            llm=llm,  # type: ignore[arg-type]
            trunk=ElasticSharedTrunk(l1_pinned_rules=[(repo / "AGENTS.md").read_text(encoding="utf-8")]),
            allow_bash_in_alignment=False,

            dsh_host=host,
        )
        try:
            await host.start()
            # 1. Read-only alignment turn -> zero worktree created!
            align_res = await mmu.handle_user_message(
                "Read `calc.py` only (do NOT edit anything yet) and tell me the signature of `compute_total`.",
                auto_run_detached=True,
            )
            assert align_res["routed_to"] == "Cell_1"
            assert mmu.cells["Cell_1"].status == "interactive_alignment"
            assert mmu.cells["Cell_1"].worktree_path is None
            assert not worktrees_dir.exists() or list(worktrees_dir.iterdir()) == []

            # 2. Launch Cell 1 + Cell 2 concurrently modifying the exact same `compute_total` function in `calc.py`!
            await mmu.handle_user_message(
                "Authorized: update `compute_total(items, discount=0)` in `calc.py` so it subtracts `discount` from the sum, add a test in `test_calc.py`, run `pytest`, and commit.",
                auto_run_detached=True,
            )
            await mmu.handle_user_message(
                "In parallel, update `compute_total(items, multiplier=1)` in `calc.py` so the sum is multiplied by `multiplier` (applied before any discount if `discount` exists: `sum(items) * multiplier - discount`), add a test in `test_calc.py`, run `pytest`, and commit.",
                auto_run_detached=True,
            )
            await mmu.drain_all_background_work()
            await host.stop()

            telem = host.collect_dsh_session_telemetry()
            assert telem["total_prompt_tokens"] > 0
            assert mmu.cells["Cell_1"].status == "completed"
            assert mmu.cells["Cell_2"].status == "completed"
            assert mmu.cells["Cell_1"].merged_to_trunk is True
            assert mmu.cells["Cell_2"].merged_to_trunk is True

            proc = subprocess.run(
                [sys.executable, "-m", "pytest", "-q"],
                cwd=repo,
                capture_output=True,
                text=True,
            )
            assert proc.returncode == 0, f"Merged trunk pytest failed:\n{proc.stdout}\n{proc.stderr}"
        finally:
            await host.stop()

    asyncio.run(_run())
