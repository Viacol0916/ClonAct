"""Unit & Architectural Invariant Tests for Synaptic-PA v7."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from sandbox_world.seed_workspace import create_sandbox_world
from clonact.context_cell import ContextCell
from clonact.context_mmu import ContextMMU
from clonact.synaptic_bus import CellFootprint, HybridSynapticBus, extract_python_symbols
from clonact.system1_judge import ChoiceSpec, JudgmentResult, NoulSpec, System1Judge
from clonact.trunk import UNION_TOOLS, ElasticSharedTrunk, TrunkEntry, compute_blob_hash


class DeterministicStubJudge(System1Judge):
    """Fast local stub judge for deterministic unit tests of architectural invariants."""

    def __init__(self) -> None:
        self.call_history = []

    async def decide(self, state: Any, questions: dict[str, Any], tag: str = "stub") -> JudgmentResult:
        answers: dict[str, dict[str, Any]] = {}
        for qid, spec in questions.items():
            if hasattr(spec, "criteria") and isinstance(spec.criteria, dict):
                first_key = next(iter(spec.criteria.keys()))
                answers[qid] = {
                    "type": "choice",
                    "choice": first_key,
                    "confidence": 0.91,
                    "probabilities": {first_key: 0.91},
                }
            else:
                answers[qid] = {"type": "noul", "noul": 0.85}
        return JudgmentResult(answers=answers, latency_ms=1.0, backend_used="stub")


class StubLLM:
    def __init__(self, utility_json: str | None = None) -> None:
        self.last_messages: list[dict[str, Any]] = []
        self.utility_calls: int = 0
        self.utility_json = utility_json

    async def chat(self, messages: list[dict[str, Any]], tools: Any = None, caller: str = "") -> dict[str, Any]:
        self.last_messages = messages
        return {"role": "assistant", "content": "stubbed first-person answer"}

    async def chat_fast_utility(
        self,
        messages: list[dict[str, Any]],
        caller: str,
        max_tokens: int = 512,
    ) -> dict[str, Any]:
        self.utility_calls += 1
        self.last_messages = messages
        if self.utility_json is not None:
            return {"role": "assistant", "content": self.utility_json}
        return {"role": "assistant", "content": "{}"}


class ScriptedJudge(System1Judge):
    """Scripted System-1 Judge for deterministic testing of routing, bus, tombstones, and recall."""

    def __init__(self, script: list[dict[str, Any]]) -> None:
        self.script = list(script)
        self.call_history = []

    async def decide(self, state: Any, questions: dict[str, Any], tag: str = "scripted") -> JudgmentResult:
        override = self.script.pop(0) if self.script else {}
        answers: dict[str, dict[str, Any]] = {}
        for qid, spec in questions.items():
            if qid in override:
                val = override[qid]
                if isinstance(val, str):
                    answers[qid] = {
                        "type": "choice",
                        "choice": val,
                        "confidence": 0.92,
                        "probabilities": {val: 0.92},
                    }
                else:
                    answers[qid] = {"type": "noul", "noul": float(val)}
            elif hasattr(spec, "criteria") and isinstance(spec.criteria, dict):
                first_key = next(iter(spec.criteria.keys()))
                answers[qid] = {
                    "type": "choice",
                    "choice": first_key,
                    "confidence": 0.80,
                    "probabilities": {first_key: 0.80},
                }
            else:
                answers[qid] = {"type": "noul", "noul": 0.05}
        return JudgmentResult(answers=answers, latency_ms=1.0, backend_used="scripted_stub")


def test_trunk_promotion_gate_and_hash_tombstone_expiry(tmp_path: Path) -> None:
    paths = create_sandbox_world(tmp_path / "world")
    trunk = ElasticSharedTrunk(l2_max_entries=3)

    rejected = trunk.promote(
        TrunkEntry(
            entry_id="bad_speculation",
            category="verified_delta",
            summary="I guess Redis might be slow",
            anchor_type="freeform_prose",
            anchor_value="",
        )
    )
    assert rejected is False
    assert "bad_speculation" not in trunk.l2_entries

    fx_file = paths["repo"] / "billing_service" / "fx_client.py"
    h0 = compute_blob_hash(fx_file)
    ok = trunk.promote(
        TrunkEntry(
            entry_id="tomb_fx",
            category="tombstone",
            summary="Needs BILLING_TZ=UTC",
            anchor_type="blob_hash",
            anchor_value=h0,
            target_path="billing_service/fx_client.py",
        )
    )
    assert ok is True
    assert trunk.invalidate_stale_tombstones(paths["repo"]) == []

    fx_file.write_text(fx_file.read_text(encoding="utf-8") + "\n# updated\n", encoding="utf-8")
    evicted = trunk.invalidate_stale_tombstones(paths["repo"])
    assert evicted == ["tomb_fx"]
    assert "tomb_fx" not in trunk.l2_entries

    for i in range(4):
        trunk.promote(
            TrunkEntry(
                entry_id=f"e_{i}",
                category="verified_delta",
                summary=f"Verified module fact #{i}",
                anchor_type="commit_sha",
                anchor_value=f"abc00{i}",
            )
        )
    assert len(trunk.l3_cold_store) >= 1
    cold_id = next(iter(trunk.l3_cold_store.keys()))
    assert cold_id in trunk.l2_pointers
    restored = trunk.page_in_from_l3(cold_id)
    assert restored is not None
    assert cold_id in trunk.l2_entries


def test_synaptic_bus_tier0_ast_and_refractory_period() -> None:
    judge = DeterministicStubJudge()
    bus = HybridSynapticBus(judge)
    bus.register_or_update_cell(CellFootprint("Cell_1", "Pagination refactor", "detached_executing"))
    bus.register_or_update_cell(CellFootprint("Cell_2", "Settlement fix", "detached_executing"))

    syms = extract_python_symbols("MAX_EXPORT_ROWS = 100000\ndef iter_export_pages(): pass\n", "exporter.py")
    assert "code://exporter.py::iter_export_pages" in syms
    assert "code://exporter.py::MAX_EXPORT_ROWS" in syms

    bus.record_uri_touch("Cell_1", syms, is_write=False, intent_one_liner="read exporter")
    sigs_first = bus.record_uri_touch("Cell_2", syms, is_write=True, intent_one_liner="bump cap")
    sigs_dup = bus.record_uri_touch("Cell_2", syms, is_write=True, intent_one_liner="bump cap")
    assert len(sigs_first) == 1
    assert sigs_first[0].target_cell == "Cell_1"
    assert sigs_dup == []


def _init_git_calc_sandbox(root: Path) -> dict[str, Path]:
    import subprocess

    repo = root / "repo"
    wt_dir = root / "worktrees"
    repo.mkdir(parents=True, exist_ok=True)
    wt_dir.mkdir(parents=True, exist_ok=True)

    (repo / "calc.py").write_text(
        "def transform(x: int) -> int:\n"
        "    return x + 1\n\n\n"
        "def helper_scale() -> int:\n"
        "    return 2\n\n\n"
        "# padding lines so top and bottom edits do not overlap in git hunks\n"
        "# line 9\n# line 10\n# line 11\n# line 12\n# line 13\n\n"
        "def summarize(vals: list[int]) -> int:\n"
        "    return sum(transform(v) for v in vals)\n",
        encoding="utf-8",
    )
    (repo / "mod_a.py").write_text("def feat_a() -> str:\n    return 'base_a'\n", encoding="utf-8")
    (repo / "mod_b.py").write_text("def feat_b() -> str:\n    return 'base_b'\n", encoding="utf-8")
    (repo / "test_calc.py").write_text(
        "from calc import summarize, transform\n\n"
        "def test_base():\n"
        "    assert transform(2) >= 3\n"
        "    assert summarize([1, 2]) >= 5\n",
        encoding="utf-8",
    )

    subprocess.run(["git", "init", "-b", "main"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test Bot"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True, capture_output=True)
    return {"world": root, "repo": repo, "worktrees": wt_dir}


def test_system1_judge_extract_usage_tokens_variants() -> None:
    ds_data = {
        "usage": {
            "prompt_tokens": 1000,
            "completion_tokens": 1,
            "prompt_cache_hit_tokens": 800,
            "prompt_cache_miss_tokens": 200,
        }
    }
    assert System1Judge._extract_usage_tokens(ds_data) == (1000, 1, 800, 200)

    oai_data = {
        "usage": {
            "prompt_tokens": 500,
            "completion_tokens": 1,
            "prompt_tokens_details": {"cached_tokens": 320},
        }
    }
    assert System1Judge._extract_usage_tokens(oai_data) == (500, 1, 320, 180)


def test_system1_judge_force_logprob_mmu_and_usage_accounting(monkeypatch: Any) -> None:
    monkeypatch.setenv("CLONACT_FORCE_LOGPROB_MMU", "1")
    monkeypatch.setenv("CLONACT_LOGPROB_MODEL", "deepseek-chat")

    judge = System1Judge(typesafe_api_key="ts-should-not-be-called", deepseek_api_key="ds-test-key")
    seen_models: list[str] = []

    async def _fail_if_jev_called(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("TypeSafe Jev must not be called when CLONACT_FORCE_LOGPROB_MMU=1")

    async def _fake_post(_client: Any, _headers: dict[str, str], payload: dict[str, Any], **_kw: Any) -> dict[str, Any]:
        seen_models.append(payload["model"])
        prompt = payload["messages"][0]["content"]
        if "Y (yes/true) or N (no/false)" in prompt:
            top = [{"token": "Y", "logprob": -0.1}, {"token": "N", "logprob": -2.3}]
        else:
            top = [{"token": "B", "logprob": -0.05}, {"token": "A", "logprob": -3.0}]
        return {
            "model": "deepseek-flash",
            "choices": [{"logprobs": {"content": [{"top_logprobs": top}]}}],
            "usage": {
                "prompt_tokens": 120,
                "completion_tokens": 1,
                "prompt_cache_hit_tokens": 90,
                "prompt_cache_miss_tokens": 30,
            },
        }

    monkeypatch.setattr(judge, "_call_typesafe_jev", _fail_if_jev_called)
    monkeypatch.setattr(judge, "_post_logprob_with_retry", _fake_post)

    res = asyncio.run(
        judge.decide(
            state={"event": "ping"},
            questions={
                "is_actionable": NoulSpec(instructions="Actionable?"),
                "route": ChoiceSpec(instructions="Pick route", criteria={"cell_1": "first", "cell_2": "second"}),
            },
            tag="unit_logprob_test",
        )
    )
    assert seen_models == ["deepseek-chat", "deepseek-chat"]
    assert res.backend_used == "deepseek_logprob_system1_fallback"
    assert res.fallback_reason == "forced_same_weight_logprob_mmu"
    assert res.resolved_model == "deepseek-flash"
    assert res.answers["is_actionable"]["noul"] > 0.8
    assert res.answers["route"]["choice"] == "cell_2"
    assert res.input_tokens == 240
    assert res.cache_hit_tokens == 180
    assert res.cache_miss_tokens == 60
    assert len(judge.call_history) == 1
    assert judge.call_history[0]["model"] == "deepseek-flash"
    assert judge.call_history[0]["cache_hit_tokens"] == 180
    assert judge.call_history[0]["cache_miss_tokens"] == 60
