"""Unified Dual-Arm Execution & Programmatic Evaluation Harness for `Synaptic-Bench-150`.

Executes identical 150-episode timelines against:
  - Arm A (`clonact`): ClonAct with DSH Workers (`ContextMMU` + `System-1 Jev` + `ElasticSharedTrunk` + `ContextCell` + `HybridSynapticBus`)
  - Arm B (`dsh_superagent`): Genuine open-source `@deepseek-ai/dsh` + `dsh-superagent-prototype` (`GenuineDSHSuperagent`)
and scores both arms with the exact same upstream benchmark verifiers (`evaluate_episode_verifiers`),
reporting 3 orthogonal metric tiers:
  - Tier 1: Orchestration & Safety (`readonly_zero_mutation`, `noise_drop`, `dedup_merge`, `similar_cell_routing`, `cross_modal`)
  - Tier 2: Upstream Task Solve (`Gaia2`, `MCPMark`, `CooperBench`, `SWE-Interact` / `Ambig-SWE`)
  - Tier 3: Efficiency & Cost (wall-clock latency, status-peek latency, worktree count, prefix cache hit rate, CNY cost)
"""
from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
from dataclasses import asdict, is_dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
import traceback
from typing import Any

from dsh_runner.dsh_controller import (
    GenuineDSHSuperagent,
    dsh_trigger_id_for_injected,
)
from dsh_runner.synaptic_worker_controller import SynapticDSHWorkerHost
from dsh_runner.sandbox_clis import (
    FEISHU_GROUP_CHAT_ID,
    FEISHU_P2P_CHAT_ID,
    FEISHU_USER_NAME,
    FEISHU_USER_OPEN_ID,
    create_github_issue,
    list_bot_replies,
    push_feishu_message,
)
from clonact.context_cell import promote_worktree_to_main
from clonact.context_mmu import ContextMMU
from clonact.llm_client import (
    HomogeneousLLMClient,
    estimate_cost_cny_upper_bound,
    load_project_dotenv,
    query_deepseek_balance_cny,
)
from clonact.paradigm_baselines import DSHParadigmDriver, PARADIGM_BASELINE_ARMS
from clonact.system1_judge import GenerativeLLMJudge, System1Judge
from clonact.trunk import ElasticSharedTrunk

from .build_pa60_subset import PA60_MANIFEST_JSON, write_pa60_manifest
from .build_pa150 import OUT_DIR, write_dataset_artifacts
from .harness_adapter import (
    AgentHarnessAdapter,
    ControlledUserSimulator,
    SubprocessCLIHarnessAdapter,
    parse_cli_arm_spec,
)
from .schema import EpisodeSpec, TimelineEventSpec
from .verifiers import (
    TASK_SOLVE_VERIFIER_KINDS,
    RuntimeArtifacts,
    _extract_agent_diff,
    evaluate_episode_verifiers,
    is_diagnostic_verifier,
)
from .workspace_manager import (
    Gaia2ServerProcess,
    PROJECT_ROOT,
    MaterializedEpisodeWorkspace,
    _BENCHMARK_HELD_OUT_PATH_RE,
    _BENCHMARK_OUTBOUND_GITHUB_RE,
    cleanup_postgres_cluster_roles,
    episode_touches_pg_cluster_roles,
    make_benchmark_tool_policy_guard,
    materialize_episode_workspace,
    text_violates_benchmark_policy,
    workspace_has_any_mutation,
)

DEFAULT_MANIFEST_PATH = OUT_DIR / "episodes_pa150.jsonl"
from clonact.release import BenchmarkRelease

RELEASE = BenchmarkRelease(Path(__file__).with_name("release.json"))

DEFAULT_RUNS_ROOT = PROJECT_ROOT / "experiments" / "outputs" / "pa150"
EVALUATOR_SCHEMA_VERSION = "2.0.1"

CLONACT_ABLATION_CONFIGS: dict[str, dict[str, Any]] = {
    "clonact_no_synapse": {
        "enable_synaptic_bus": False,
        "enable_peek_peer_cell": False,
        "enable_synaptic_reconcile": False,
    },
    "clonact_no_same_brain": {
        "disable_same_brain_continuity": True,
    },
    "clonact_llm_mmu": {
        "use_generative_llm_mmu": True,
    },
    "clonact_no_prefix": {
        "disable_prefix_and_grounding": True,
    },
}

BUILTIN_ARMS: tuple[str, ...] = (
    "clonact",
    "dsh_superagent",
    *PARADIGM_BASELINE_ARMS,
    *CLONACT_ABLATION_CONFIGS.keys(),
)
_PEEK_BEHAVIORS = ("nonblocking_slice_peek", "instant_status_peek", "instant_status_peek_0ms")
_SPEC_REF_CODE_RE = re.compile(r"\bPA150-SPEC-\d+(?:-[A-F0-9]{4})?\b|\bSPEC-REF-[A-Za-z0-9_-]+\b")
_GAIA_CLI_EXEC_RE = re.compile(r"(?:^|[\s/;&|])gaia_cli\s+(?:apps|docs|call)\b")
# In `dsh_superagent`, `gh` on PATH is symlinked to our local SQLite `sandbox-gh` (`dsh_runner/sandbox_clis.py`),
# so bare `gh pr create` / `gh issue view` is purely local and never accesses outbound GitHub.
_DSH_OUTBOUND_NET_CLIENT_RE = re.compile(
    r"\b(?:curl|wget|httpie|nc|ncat|ssh|scp|urllib|requests|httpx|aiohttp|socket|webbrowser|web_search|web_fetch|"
    r"git\s+(?:clone|fetch|pull|ls-remote))\b",
    flags=re.IGNORECASE,
)
_DSH_INTERNAL_META_TOOLS = frozenset({"orchestrator_route", "worker_report", "todo_write", "todo_read"})
_DSH_SANDBOX_BLOCK_MARKERS = (
    "operation not permitted",
    "sandbox policy blocked",
    "nonexistent_sandbox_blocked_github",
    "is blocked",
    "permissionerror",
)


def _scan_executed_text_contamination(blobs: list[str]) -> bool:
    """Symmetrically detect held-out benchmark paths or outbound GitHub network fetches across executed tool traces."""
    for txt in blobs:
        if txt and text_violates_benchmark_policy(txt) is not None:
            return True
    return False


def _scan_dsh_session_contamination(
    session_bundles: list[tuple[str, list[dict[str, Any]]]],
    *,
    eff_repo: str = "",
    local_issue_numbers: set[int] | None = None,
    seatbelt_enforced: bool = False,
    samples_out: list[str] | None = None,
) -> tuple[bool, int]:
    """Symmetrically scan DSH executed tool calls (`tool/call` / `tool/use`) for unblocked contamination.

    Matches `synaptic_pa`'s `_step_audit_trail` semantics 1-to-1:
    - Never scans `assistant/message` prose/CoT (preventing false positives when workers mention local PR URLs).
    - Excludes local SQLite `sandbox-gh` (`gh`) and local sandbox issue/PR URLs (`#1..#k` in `sandbox_db.sqlite`).
    - Counts attempts blocked by macOS Seatbelt (`Operation not permitted`, even if shell redirected `2>/dev/null`),
      Python `sitecustomize` SSL SNI/socket guard, `curl`/`wget` shims, `git` `insteadOf`, or DSH `blockedCommands`
      under `held_out_contamination_blocked`.
    """
    contaminated = False
    blocked_count = 0
    local_gh_url_re: re.Pattern[str] | None = None
    if eff_repo and local_issue_numbers:
        nums_alt = "|".join(str(int(n)) for n in sorted(local_issue_numbers))
        local_gh_url_re = re.compile(
            rf"https?://(?:www\.)?github\.com/{re.escape(eff_repo)}/(?:pull|issues)/(?:{nums_alt})(?![0-9])[^\s\"')\]>]*",
            flags=re.IGNORECASE,
        )

    for _sid, s_events in session_bundles:
        result_by_call_id: dict[str, str] = {}
        for s_ev in s_events:
            if str(s_ev.get("type") or "") == "tool/result":
                s_data = s_ev.get("data") or {}
                msg = s_data.get("message") if isinstance(s_data, dict) else {}
                cid = str(
                    (msg.get("toolCallId") if isinstance(msg, dict) else "")
                    or (s_data.get("callId") if isinstance(s_data, dict) else "")
                    or ""
                )
                if cid:
                    result_by_call_id[cid] = json.dumps(s_data, ensure_ascii=False)

        for s_ev in s_events:
            ev_type = str(s_ev.get("type") or "")
            if ev_type not in ("tool/call", "tool/use"):
                continue
            s_data = s_ev.get("data") or {}
            if not isinstance(s_data, dict):
                continue
            t_name = str(s_data.get("name") or "")
            if t_name in _DSH_INTERNAL_META_TOOLS:
                continue
            cid = str(s_data.get("callId") or s_data.get("id") or "")
            args_raw = s_data.get("arguments") if "arguments" in s_data else s_data.get("input")
            args_dict: dict[str, Any] | None = None
            if isinstance(args_raw, dict):
                args_dict = args_raw
            elif isinstance(args_raw, str):
                try:
                    parsed = json.loads(args_raw)
                    if isinstance(parsed, dict):
                        args_dict = parsed
                except Exception:
                    args_dict = None

            if t_name == "bash" and isinstance(args_dict, dict):
                cand_txt = str(args_dict.get("command") or "")
            elif isinstance(args_dict, dict):
                cand_txt = " ".join(
                    str(args_dict.get(k) or "")
                    for k in ("path", "file_path", "rel_path", "pattern", "command", "url")
                    if args_dict.get(k)
                ) or json.dumps(args_dict, ensure_ascii=False)
            else:
                cand_txt = str(args_raw or "")

            if not cand_txt:
                continue

            gh_scan_txt = local_gh_url_re.sub("", cand_txt) if local_gh_url_re is not None else cand_txt
            # A standalone `gh pr create --body ...` / `gh issue comment ...` (not invoking python/curl/git fetch)
            # only writes to local `sandbox_db.sqlite` even if the PR markdown body discusses HTTP/requests/urls.
            stripped_cmd = gh_scan_txt.lstrip()
            is_pure_local_gh = (
                t_name == "bash"
                and stripped_cmd.startswith(("gh ", "sandbox-gh "))
                and not re.search(r"\b(?:python|python3|curl|wget|node|git\s+(?:clone|fetch|pull|ls-remote))\b", stripped_cmd)
            )

            has_heldout_path = bool(_BENCHMARK_HELD_OUT_PATH_RE.search(cand_txt))
            has_outbound_gh = (
                not is_pure_local_gh
                and bool(_BENCHMARK_OUTBOUND_GITHUB_RE.search(gh_scan_txt))
                and bool(_DSH_OUTBOUND_NET_CLIENT_RE.search(gh_scan_txt))
            )
            if not (has_heldout_path or has_outbound_gh):
                continue

            res_low = result_by_call_id.get(cid, "").lower()
            has_unguarded_net_runtime = bool(
                has_outbound_gh
                and re.search(
                    r"\b(?:node|deno|bun|ruby|perl|nc|ncat|ssh|scp)\b|(?:-u|--unset)\s+GIT_CONFIG|\bunset\s+GIT_CONFIG",
                    gh_scan_txt,
                    flags=re.IGNORECASE,
                )
            )
            was_blocked = (
                (bool(res_low) and any(m in res_low for m in _DSH_SANDBOX_BLOCK_MARKERS))
                or (seatbelt_enforced and not has_unguarded_net_runtime)
            )
            if was_blocked:
                blocked_count += 1
            else:
                contaminated = True
                if samples_out is not None and len(samples_out) < 5:
                    samples_out.append(f"{t_name}: {cand_txt[:240]}")
    return contaminated, blocked_count


def _episode_cluster_key(ep: EpisodeSpec) -> str:
    """Return primary upstream repo/category cluster identifier for cluster-robust SE (`v2.0.0`)."""
    if ep.workspace_mounts:
        m0 = ep.workspace_mounts[0]
        if m0.repo_or_category:
            return f"{m0.pillar}:{m0.repo_or_category}"
    return ep.episode_id


def _episode_has_conflict_pair(ep: EpisodeSpec) -> bool | None:
    if ep.track_id != "C-2" or not ep.workspace_mounts:
        return None
    meta = ep.workspace_mounts[0].setup_meta or {}
    if "has_gold_overlap_conflict" in meta:
        return bool(meta["has_gold_overlap_conflict"])
    if "has_conflict" in meta:
        return bool(meta["has_conflict"])
    return None


def episode_fingerprint(ep: EpisodeSpec) -> str:
    """Compute a deterministic SHA-256 fingerprint of an `EpisodeSpec` for `--resume` validation (`NEW-10`)."""
    canonical = json.dumps(ep.to_dict(), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def build_runner_error_record(
    episode: EpisodeSpec,
    arm_name: str,
    exc: BaseException,
    wall_clock_sec: float = 0.0,
    gaia_judge_mode: str = "auto",
) -> dict[str, Any]:
    """Build a terminal failure record when an episode crashes so `n` never suffers survivorship bias (`NEW-10`)."""
    err_str = f"{type(exc).__name__}: {exc}"
    diag_total = sum(1 for v in episode.verifiers if is_diagnostic_verifier(v))
    solve_total = sum(
        1 for v in episode.verifiers if not is_diagnostic_verifier(v) and v.kind in TASK_SOLVE_VERIFIER_KINDS
    )
    gating_total = len(episode.verifiers) - diag_total
    orch_total = gating_total - solve_total
    verifier_rows = [
        {
            "verifier_id": v.verifier_id,
            "kind": v.kind,
            "tier": (
                "diagnostic"
                if is_diagnostic_verifier(v)
                else ("task_solve" if v.kind in TASK_SOLVE_VERIFIER_KINDS else "orchestration_safety")
            ),
            "mount_id": v.mount_id,
            "passed": False,
            "error": f"Skipped due to runner exception: {err_str}",
            **({"diagnostic_only": True} if is_diagnostic_verifier(v) else {}),
        }
        for v in episode.verifiers
    ]
    return {
        "episode_id": episode.episode_id,
        "track_id": episode.track_id,
        "domain": episode.domain,
        "scenario_type": episode.scenario_type,
        "cluster_key": _episode_cluster_key(episode),
        "has_conflict_pair": _episode_has_conflict_pair(episode),
        "arm_name": arm_name,
        "episode_sha256": episode_fingerprint(episode),
        "evaluator_version": EVALUATOR_SCHEMA_VERSION,
        "model": os.environ.get("PA_LLM_MODEL", "deepseek-flash"),
        "gaia_judge_mode": gaia_judge_mode,
        "episode_passed": False,
        "runner_error": err_str,
        "verifiers_passed": 0,
        "verifiers_total": gating_total,
        "verifier_pass_rate": _safe_rate(0, gating_total),
        "task_solve_passed": 0,
        "task_solve_total": solve_total,
        "task_solve_pass_rate": _safe_rate(0, solve_total),
        "orchestration_passed": 0,
        "orchestration_total": orch_total,
        "orchestration_pass_rate": _safe_rate(0, orch_total),
        "diagnostic_passed": 0,
        "diagnostic_total": diag_total,
        "verifiers": verifier_rows,
        "telemetry": {
            "wall_clock_sec": round(wall_clock_sec, 2),
            "worktrees_created": 0,
            "total_prompt_tokens": 0,
            "cache_hit_tokens": 0,
            "output_tokens": 0,
            "estimated_cost_cny": 0.0,
            "peek_latencies_ms": [],
            "runner_failed": True,
            "runner_error": err_str,
        },
        "routing_events": {},
        "event_trace": [],
    }


def load_pa150_episodes(
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    *,
    episode_ids: list[str] | None = None,
    tracks: list[str] | None = None,
    domains: list[str] | None = None,
    subset: str | None = None,
    limit: int | None = None,
) -> list[EpisodeSpec]:
    """Load and filter episodes from `episodes_pa150.jsonl` (supporting `--subset pa60`)."""
    if not manifest_path.exists():
        write_dataset_artifacts()

    id_set = {x.strip() for x in (episode_ids or []) if x.strip()}
    if subset and subset.strip().lower() == "pa60":
        if not PA60_MANIFEST_JSON.is_file():
            write_pa60_manifest(PA60_MANIFEST_JSON)
        pa60_data = json.loads(PA60_MANIFEST_JSON.read_text(encoding="utf-8"))
        pa60_ids = {str(x) for x in (pa60_data.get("episode_ids") or [])}
        id_set = (id_set & pa60_ids) if id_set else pa60_ids
    track_set = {x.strip() for x in (tracks or []) if x.strip()}
    dom_set = {x.strip() for x in (domains or []) if x.strip()}

    episodes: list[EpisodeSpec] = []
    with manifest_path.open(encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            ep = EpisodeSpec.from_dict(json.loads(line))
            if id_set and ep.episode_id not in id_set:
                continue
            if track_set and ep.track_id not in track_set:
                continue
            if dom_set and ep.domain not in dom_set:
                continue
            episodes.append(ep)
            if limit is not None and len(episodes) >= limit:
                break
    return episodes


_OFFICE_INSPECTION_MARKERS = (
    "office_fs",
    "office_pg",
    "release_spec_addendum",
    "release_spec_addenda",
    "pg_exec",
    "gaia_cli",
    "FILESYSTEM_TEST_DIR",
)


def _current_git_head() -> str:
    res = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True, encoding="utf-8", errors="replace",
    )
    return res.stdout.strip() if res.returncode == 0 else "unknown"


def _collect_forensic_diffs(
    episode: EpisodeSpec,
    ws: MaterializedEpisodeWorkspace,
    task_worktrees: dict[str, Path],
    all_worktrees: list[Path],
) -> dict[str, str]:
    """Capture agent source diffs before `ws.cleanup()` deletes ephemeral worktrees (`Known-15`)."""
    base_commit = next((m.base_commit for m in episode.workspace_mounts if m.base_commit), "") or ""
    diffs: dict[str, str] = {}
    seen_paths: set[Path] = set()
    for tag, wt in sorted(task_worktrees.items()):
        if wt and wt.exists():
            d = _extract_agent_diff(wt, base_commit)
            if d.strip():
                diffs[f"tag:{tag} ({wt.name})"] = d[:262144]
                seen_paths.add(wt.resolve())
    for wt in all_worktrees:
        if wt and wt.exists() and wt.resolve() not in seen_paths:
            d = _extract_agent_diff(wt, base_commit)
            if d.strip():
                diffs[f"worktree:{wt.name}"] = d[:262144]
                seen_paths.add(wt.resolve())
    if ws.repo_dir.exists() and (ws.repo_dir / ".git").exists() and ws.repo_dir.resolve() not in seen_paths:
        d = _extract_agent_diff(ws.repo_dir, base_commit)
        if d.strip():
            diffs["repo_trunk"] = d[:262144]
    return diffs


def _write_forensic_bundle(
    run_root: Path,
    episode: EpisodeSpec,
    arm_name: str,
    payload: dict[str, Any],
    *,
    forensic_root: Path | None = None,
) -> Path:
    """Persist forensic replay bundle under `<output_dir>/forensic/{episode_id}__{arm_name}.json` (`Known-15`)."""
    base_dir = forensic_root if forensic_root is not None else (run_root.parent if run_root.name == "workspaces" else run_root)
    forensic_dir = base_dir / "forensic"
    forensic_dir.mkdir(parents=True, exist_ok=True)
    target = forensic_dir / f"{episode.episode_id}__{arm_name}.json"
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(target)
    return target


def _apply_workspace_file_updates(
    ev: TimelineEventSpec,
    ws_repo_dir: Path,
    dsh_clone_repo: Path | None = None,
    sync_worktrees: list[Path] | None = None,
) -> list[str]:
    """Apply mid-episode workspace file updates (e.g. `H-4` `WORKSPACE_POLICY.md` Rev 1 -> Rev 2) identically on both arms.

    Commits updates onto `ws_repo_dir` (`main`) so new `Synaptic-PA` worktrees inherit them,
    pushes `dsh_clone_repo` commits to `origin:main` so new `DSH` worktrees inherit them,
    and mirrors updates into any already-spawned worktrees (`Known-1`).
    """
    updates = (ev.metadata or {}).get("workspace_file_updates")
    if not isinstance(updates, dict) or not updates:
        return []
    written: list[str] = []
    for rel_path, content in updates.items():
        rel_clean = str(rel_path).lstrip("/")
        dst = ws_repo_dir / rel_clean
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(str(content), encoding="utf-8")
        written.append(rel_clean)
        if (ws_repo_dir / ".git").exists():
            subprocess.run(["git", "add", "--", rel_clean], cwd=ws_repo_dir, capture_output=True)
            subprocess.run(
                ["git", "commit", "-m", f"Update {rel_clean} before {ev.event_id}"],
                cwd=ws_repo_dir,
                capture_output=True,
            )
        for wt in sync_worktrees or []:
            if wt and wt.exists():
                wt_dst = wt / rel_clean
                wt_dst.parent.mkdir(parents=True, exist_ok=True)
                wt_dst.write_text(str(content), encoding="utf-8")
        if dsh_clone_repo is not None and dsh_clone_repo.exists():
            cdst = dsh_clone_repo / rel_clean
            cdst.parent.mkdir(parents=True, exist_ok=True)
            cdst.write_text(str(content), encoding="utf-8")
            subprocess.run(["git", "add", "--", rel_clean], cwd=dsh_clone_repo, capture_output=True)
            subprocess.run(
                ["git", "commit", "-m", f"Update {rel_clean} before {ev.event_id}"],
                cwd=dsh_clone_repo,
                capture_output=True,
            )
            subprocess.run(["git", "push", "origin", "HEAD:main"], cwd=dsh_clone_repo, capture_output=True)
    return written


def _collect_synaptic_office_read_evidence(mmu: ContextMMU) -> dict[str, Any]:
    """Extract genuine tool-executed office read evidence & `PA150-SPEC-*` codes, ignoring prompt echoes (`Known-14` / `A5`)."""
    office_inspected = False
    spec_doc_read = False
    spec_codes: set[str] = set()

    for fp in mmu.bus.footprints.values():
        for u in fp.read_uris | fp.write_uris:
            u_str = str(u)
            if any(tok in u_str for tok in _OFFICE_INSPECTION_MARKERS):
                office_inspected = True
            if "release_spec_addendum" in u_str or "release_spec_addenda" in u_str:
                spec_doc_read = True

    for c in mmu.cells.values():
        for step in getattr(c, "_step_audit_trail", []):
            st_status = str(step.get("status") or "ok").lower()
            if st_status == "blocked":
                continue
            cmd_blob = " ".join(
                str(step.get(k) or "") for k in ("tool", "cmd", "cmd_tail")
            )
            uris_blob = " ".join(str(x) for x in (step.get("uris") or []))
            apis_blob = " ".join(str(x) for x in (step.get("apis") or []))
            out_exc = str(step.get("out_excerpt") or "")
            full_step_blob = " ".join((cmd_blob, uris_blob, apis_blob, out_exc))
            if any(tok in full_step_blob for tok in _OFFICE_INSPECTION_MARKERS):
                office_inspected = True
            for code_m in _SPEC_REF_CODE_RE.findall(out_exc):
                spec_codes.add(code_m)
                office_inspected = True
                spec_doc_read = True
            if st_status == "ok" and any(
                marker in blob_part
                for marker in ("release_spec_addendum", "release_spec_addenda")
                for blob_part in (cmd_blob, uris_blob, out_exc)
            ):
                spec_doc_read = True

        # Strictly inspect `role == "tool"` outputs (stripping trailing bus-notice suffixes) so user/system prompt echoes NEVER match!
        for msg in getattr(c, "tool_scratchpad", []):
            if str(msg.get("role") or "") != "tool":
                continue
            raw_content = str(msg.get("content") or "")
            tool_only_out = raw_content.split("\n\n[", 1)[0]
            if tool_only_out.lstrip().startswith("Error:"):
                continue
            if any(tok in tool_only_out for tok in _OFFICE_INSPECTION_MARKERS):
                office_inspected = True
            found_codes = _SPEC_REF_CODE_RE.findall(tool_only_out)
            if found_codes:
                spec_codes.update(found_codes)
                office_inspected = True
                spec_doc_read = True
            elif "release_spec_addendum" in tool_only_out or "release_spec_addenda" in tool_only_out:
                spec_doc_read = True

    return {
        "office_mount_inspected": office_inspected,
        "office_spec_doc_read": spec_doc_read,
        "tool_read_spec_ref_codes": sorted(spec_codes),
    }


def _detect_synaptic_office_inspection(mmu: ContextMMU) -> bool:
    """Check real `HybridSynapticBus` URI footprints and `ContextCell` tool traces for office reads (`Known-14` / `A5`)."""
    return bool(_collect_synaptic_office_read_evidence(mmu)["office_mount_inspected"])


def _cell_executed_gaia_cli(cell: Any) -> bool:
    """Return True iff `cell` genuinely invoked `./gaia_cli (apps|docs|call)` (ignoring `ls` directory listings) (`A1`)."""
    for step in getattr(cell, "_step_audit_trail", []):
        if str(step.get("status") or "").lower() == "blocked":
            continue
        apis = [str(a) for a in (step.get("apis") or [])]
        if any(a.startswith(("gaia2.", "app.")) or a == "cli:gaia_cli" for a in apis):
            return True
        for k in ("cmd", "cmd_tail"):
            if _GAIA_CLI_EXEC_RE.search(str(step.get(k) or "")):
                return True
    return False


def _select_synaptic_gaia_replies(
    mmu: ContextMMU,
    tag_cells: dict[str, list[str]],
    noise_cids: set[str] | None = None,
) -> list[str]:
    """Select Gaia2 replies from Gaia2-routed or Gaia2-CLI-executing cells, excluding `office_fs`/`pg`/noise cells (`A1`)."""
    noise_set = set(noise_cids or set())
    gaia_cids: set[str] = set()
    non_gaia_cids: set[str] = set(noise_set)
    for tname, cids in tag_cells.items():
        if "gaia" in tname or tname in ("office_main", "office_inflight", "primary_task", "office_task"):
            gaia_cids.update(c for c in cids if c not in noise_set)
        elif (
            tname in ("task_fs", "task_pg", "office_side_lookup", "hybrid_code_main")
            or "fs" in tname
            or "pg" in tname
        ):
            non_gaia_cids.update(cids)
    for cid, c in mmu.cells.items():
        if cid in non_gaia_cids or cid in noise_set:
            continue
        if _cell_executed_gaia_cli(c):
            gaia_cids.add(cid)

    gaia_task_replies: list[str] = []
    for item in mmu.unified_stream:
        if item.kind == "agent_reply" and item.text.strip():
            if item.metadata.get("transition") in ("instant_live_slice_peek", "post_completion_readonly_query"):
                continue
            parts = {p.strip() for p in item.cell_tag.split("+") if p.strip()}
            if parts & noise_set:
                continue
            if gaia_cids:
                if parts & gaia_cids:
                    gaia_task_replies.append(item.text.strip())
            elif not (parts & non_gaia_cids):
                gaia_task_replies.append(item.text.strip())
        elif item.kind == "ambient_pill" and item.metadata.get("deferred_summary"):
            if item.cell_tag in noise_set:
                continue
            if gaia_cids and item.cell_tag in gaia_cids:
                gaia_task_replies.append(str(item.metadata["deferred_summary"]).strip())
            elif not gaia_cids and item.cell_tag not in non_gaia_cids:
                gaia_task_replies.append(str(item.metadata["deferred_summary"]).strip())
    return gaia_task_replies


def _rank_worktree_paths_nonempty_first(paths: list[Path], base_commit: str) -> list[Path]:
    """Order candidate worktrees with non-empty source diffs first (`A2`)."""
    deduped: list[Path] = []
    seen: set[Path] = set()
    for p in paths:
        if p and p.exists():
            rp = p.resolve()
            if rp not in seen:
                seen.add(rp)
                deduped.append(p)
    non_empty = [p for p in deduped if _extract_agent_diff(p, base_commit).strip()]
    empty = [p for p in deduped if p.resolve() not in {x.resolve() for x in non_empty}]
    return non_empty + empty


class _TrackedTaskDict(dict):  # type: ignore[type-arg]
    """Dict wrapper for `mmu.background_tasks` capturing unhandled background exceptions for forensics (`A6`)."""

    def __init__(self, sink: list[dict[str, Any]]) -> None:
        super().__init__()
        self._sink = sink
        self._seen_task_ids: set[int] = set()

    def __setitem__(self, key: Any, value: Any) -> None:
        super().__setitem__(key, value)
        if isinstance(value, asyncio.Task) and id(value) not in self._seen_task_ids:
            self._seen_task_ids.add(id(value))
            cid_str = str(key)

            def _on_done(fut: asyncio.Task[Any], cell_key: str = cid_str) -> None:
                if fut.cancelled():
                    return
                try:
                    exc = fut.exception()
                except asyncio.CancelledError:
                    return
                if exc is not None and not isinstance(exc, asyncio.CancelledError):
                    self._sink.append(
                        {
                            "cell_id": cell_key,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-4000:],
                        }
                    )

            value.add_done_callback(_on_done)


# ---------------------------------------------------------------------------
# Arm A: `Synaptic-PA v7` Episode Runner
# ---------------------------------------------------------------------------


async def run_episode_synaptic_pa(
    episode: EpisodeSpec,
    run_root: Path,
    *,
    gaia_server: Gaia2ServerProcess | None = None,
    gaia_port: int = 19011,
    gaia_judge_mode: str = "auto",
    keep_workspace: bool = False,
    forensic_root: Path | None = None,
    arm_label: str = "clonact",
    dsh_port: int = 0,
    ablation_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if arm_label not in ("clonact", *CLONACT_ABLATION_CONFIGS):
        raise ValueError(f"Unsupported ClonAct arm: {arm_label!r}")
    worker_backend = "dsh_agent"
    load_project_dotenv()
    started_at = time.time()
    t0 = time.perf_counter()
    ws = await asyncio.to_thread(
        materialize_episode_workspace,
        episode,
        arm_label,
        run_root,
        gaia_server=gaia_server,
        gaia_port=gaia_port,
        gaia_judge_mode=gaia_judge_mode,
    )
    docs_dir = ws.root_dir / "feishu_docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    repo_docs_link = ws.repo_dir / "feishu_docs"
    if not repo_docs_link.exists() and not repo_docs_link.is_symlink():
        repo_docs_link.symlink_to(docs_dir.resolve())
    if ws.shared_links_dir is not None and ws.shared_links_dir.exists():
        sl_docs_link = ws.shared_links_dir / "feishu_docs"
        if not sl_docs_link.exists() and not sl_docs_link.is_symlink():
            sl_docs_link.symlink_to(docs_dir.resolve())

    bin_dir = ws.root_dir / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    for rel_name, (fcontent, fmode) in (ws.extra_repo_files or {}).items():
        if "/" not in rel_name and (fmode & 0o111):
            bp = bin_dir / rel_name
            bp.write_text(fcontent, encoding="utf-8")
            bp.chmod(fmode)
    cur_path = ws.extra_env.get("PATH") or os.environ.get("PATH", "")
    path_parts = [p for p in cur_path.split(os.pathsep) if p]
    if str(bin_dir) not in path_parts:
        ws.extra_env["PATH"] = os.pathsep.join([str(bin_dir), *path_parts])

    dsh_worker_host: SynapticDSHWorkerHost | None = None
    dsh_worker_host = SynapticDSHWorkerHost(
        ws.root_dir / "dsh_worker_instance",
        allowed_workspace_root=ws.root_dir,
        port=dsh_port,
    )

    tag_to_cell_id: dict[str, str] = {}
    tag_cells: dict[str, list[str]] = defaultdict(list)
    cell_id_to_tag: dict[str, str] = {}
    cell_tool_counts: dict[str, int] = defaultdict(int)
    tool_step_signal = asyncio.Event()

    def _register_tag_cell(tag: str | None, cid: str | None) -> None:
        if not tag or not cid:
            return
        tag_to_cell_id.setdefault(tag, cid)
        if cid not in tag_cells[tag]:
            tag_cells[tag].append(cid)
        cell_id_to_tag.setdefault(cid, tag)

    def _on_tool_step(cid: str, tool_name: str, total_calls: int) -> None:
        cell_tool_counts[cid] = total_calls
        tool_step_signal.set()

    ab_cfg = dict(ablation_config or {})
    disable_prefix = bool(ab_cfg.get("disable_prefix_and_grounding", False))
    use_llm_mmu = bool(ab_cfg.get("use_generative_llm_mmu", False))

    pinned_rules = [
        "User Identity: @user (Unified Personal Agent workspace across Office files, PostgreSQL, Personal Apps, and Git repositories)."
    ]
    if not disable_prefix:
        pinned_rules.append(ws.agents_md)
    trunk = ElasticSharedTrunk(
        l1_pinned_rules=pinned_rules,
        disable_elastic_l2_l3=disable_prefix,
    )
    llm = HomogeneousLLMClient()
    judge: System1Judge = GenerativeLLMJudge() if use_llm_mmu else System1Judge()
    bus_judge: System1Judge | None = System1Judge() if use_llm_mmu else None
    has_code_mount = any(
        m.pillar in ("cooperbench", "swe_interact", "ambig_swe") for m in episode.workspace_mounts
    )
    mmu = ContextMMU(
        sandbox_paths={
            "world": ws.root_dir,
            "repo": ws.repo_dir,
            "clone_repo": ws.repo_dir,
            "worktrees": (ws.root_dir / "worktrees") if has_code_mount else None,
            "docs": docs_dir,
            "shared_links": ws.shared_links_dir or (ws.root_dir / "shared_links"),
        },
        judge=judge,
        llm=llm,
        trunk=trunk,
        extra_env=ws.extra_env,
        allow_bash_in_alignment=True,
        enable_synaptic_bus=bool(ab_cfg.get("enable_synaptic_bus", True)),
        on_tool_executed=_on_tool_step,
        tool_policy_guard=make_benchmark_tool_policy_guard(ws.root_dir),
        dsh_host=dsh_worker_host,
        bus_judge=bus_judge,
        enable_peek_peer_cell=bool(ab_cfg.get("enable_peek_peer_cell", True)),
        enable_synaptic_reconcile=bool(ab_cfg.get("enable_synaptic_reconcile", True)),
        disable_same_brain_continuity=bool(ab_cfg.get("disable_same_brain_continuity", False)),
        disable_prefix_and_grounding=disable_prefix,
    )
    bg_task_exceptions: list[dict[str, Any]] = []
    tracked_bg = _TrackedTaskDict(bg_task_exceptions)
    for existing_k, existing_t in mmu.background_tasks.items():
        tracked_bg[existing_k] = existing_t
    mmu.background_tasks = tracked_bg

    noise_dropped_exact = 0
    noise_spawned = 0
    noise_cids: set[str] = set()
    dup_merged_exact = 0
    dup_spawned = 0
    real_tasks_dropped = 0
    compound_split_executed = False
    hot_injections = 0
    overturns = 0
    peek_latencies_ms: list[float] = []
    compound_peek_latencies_ms: list[float] = []
    routing_events: dict[str, dict[str, Any]] = {}
    event_trace: list[dict[str, Any]] = []
    total_noise_in_timeline = sum(1 for e in episode.timeline if e.kind == "external_noise")

    async def _process_synaptic_event(
        ev: TimelineEventSpec,
        *,
        trigger_fired_via: str,
        planned_offset_sec: float | None = None,
        actual_offset_sec: float | None = None,
    ) -> dict[str, Any]:
        nonlocal noise_dropped_exact, noise_spawned, dup_merged_exact, dup_spawned
        nonlocal real_tasks_dropped, compound_split_executed, hot_injections, overturns

        existing_wts = [
            c.worktree_path for c in mmu.cells.values() if c.worktree_path is not None and c.worktree_path.exists()
        ]
        updated_files = _apply_workspace_file_updates(ev, ws.repo_dir, sync_worktrees=existing_wts)
        ev_start = time.perf_counter()

        if ev.kind == "user_chat":
            out = await mmu.handle_user_message(ev.text, auto_run_detached=True)
            ev_ms = round((time.perf_counter() - ev_start) * 1000.0, 1)
            if out.get("transition") == "compound_multi_intent" or out.get("compound_clauses"):
                compound_split_executed = True

            sub_outcomes = list(out.get("sub_outcomes") or [out])
            # P0-5: Use exact atomic per-call `spawned_worker_cells` returned by `ContextMMU` rather than racing global set diffs!
            new_worker_cells = [
                str(c)
                for c in (out.get("spawned_worker_cells") or [])
                if str(c) in mmu.cells and not mmu.cells[str(c)].observer_only
            ]

            # Apples-to-apples non-blocking slice peek timing (`Known-8`): accept both `nonblocking_slice_peek` and legacy alias
            if ev.expected_mmu_behavior in _PEEK_BEHAVIORS:
                peek_latencies_ms.append(ev_ms)
            elif any(
                str(s.get("transition") or "") in ("instant_live_slice_peek", "post_completion_readonly_query")
                for s in sub_outcomes
            ):
                compound_peek_latencies_ms.append(ev_ms)

            routed_cids: list[str] = []
            overturned_cids: list[str] = []
            for sub in sub_outcomes:
                trans = str(sub.get("transition") or "")
                rcid = str(sub.get("routed_to") or "")
                bridged = [str(x) for x in (sub.get("bridged_cells") or []) if x]
                ov_peers = [str(x) for x in (sub.get("overturned_peers") or []) if x]
                mounted = [str(x) for x in (sub.get("mounted_slices") or []) if x]

                if trans in ("hot_inject_into_active_cell", "resume_completed_cell_execution"):
                    hot_injections += 1
                hot_injections += len(bridged)
                if trans in ("overturn_and_relaunch", "overturn_cancelled"):
                    overturns += 1
                    if rcid:
                        overturned_cids.append(rcid)
                overturns += len(ov_peers)
                overturned_cids.extend(ov_peers)

                for cand_c in ([rcid] if rcid else []) + bridged + mounted:
                    if cand_c and cand_c not in routed_cids:
                        routed_cids.append(cand_c)

                if (
                    ev.target_task_tag
                    and ev.expected_mmu_behavior != "similar_cell_disambiguation_followup"
                    and trans not in ("post_completion_readonly_query", "instant_live_slice_peek", "zero_mutation_alignment_turn")
                    and rcid
                    and rcid in mmu.cells
                    and not mmu.cells[rcid].observer_only
                ):
                    _register_tag_cell(ev.target_task_tag, rcid)

            # `A2`: Register ALL spawned worker cells (preserving order) under `ev.target_task_tag` for compound multi-cell waves!
            if (
                ev.target_task_tag
                and ev.expected_mmu_behavior != "similar_cell_disambiguation_followup"
                and new_worker_cells
            ):
                for nw_cid in new_worker_cells:
                    _register_tag_cell(ev.target_task_tag, nw_cid)

            routed_tags: list[str] = []
            for c in routed_cids:
                t = cell_id_to_tag.get(c)
                if t and t not in routed_tags:
                    routed_tags.append(t)

            distractor_overturned = any(
                cell_id_to_tag.get(oc) and cell_id_to_tag.get(oc) != ev.target_task_tag
                for oc in overturned_cids
            )
            routing_events[ev.event_id] = {
                "routed_cells": routed_cids,
                "routed_tags": routed_tags,
                "spawned_new_worker": len(new_worker_cells) > 0,
                "distractor_overturned": distractor_overturned,
            }
        else:
            out = await mmu.ingest_external_event(
                {
                    "event_id": ev.event_id,
                    "channel": ev.channel,
                    "sender": ev.sender,
                    "text": ev.text,
                },
                auto_run_background=True,
            )
            ev_ms = round((time.perf_counter() - ev_start) * 1000.0, 1)
            act = str(out.get("action") or "")
            cid = str(out.get("cell_id") or "")

            if ev.kind == "external_noise":
                if act in ("dropped_fast_filter", "dropped_by_system1"):
                    noise_dropped_exact += 1
                elif act == "spawned_proactive_cell":
                    noise_spawned += 1
                    if cid:
                        noise_cids.add(cid)
            elif ev.kind == "external_duplicate_webhook":
                exp_cid = tag_to_cell_id.get(ev.target_task_tag or "")
                if act in ("merged_into_existing", "resumed_completed_cell") and (not exp_cid or cid == exp_cid):
                    dup_merged_exact += 1
                elif act == "spawned_proactive_cell":
                    dup_spawned += 1
            elif ev.kind == "external_task_webhook":
                if act in ("dropped_fast_filter", "dropped_by_system1"):
                    real_tasks_dropped += 1
                elif cid and ev.target_task_tag:
                    _register_tag_cell(ev.target_task_tag, cid)

            routing_events[ev.event_id] = {
                "routed_cells": [cid] if cid else [],
                "routed_tags": [cell_id_to_tag[cid]] if cid in cell_id_to_tag else [],
                "spawned_new_worker": act == "spawned_proactive_cell",
                "distractor_overturned": False,
            }

        # Record exact mid-episode cryptographic snapshots right after synchronous alignment/side-query turns
        if ev.expected_mmu_behavior in ("align_readonly_zero_wt", "side_query_readonly_zero_wt"):
            ws.record_event_snapshots(ev.event_id)

        return {
            "event_id": ev.event_id,
            "kind": ev.kind,
            "trigger_fired_via": trigger_fired_via,
            "planned_offset_sec": planned_offset_sec,
            "actual_offset_sec": actual_offset_sec,
            "updated_files": updated_files,
            "elapsed_ms": ev_ms,
            "outcome": out,
        }

    try:
        if dsh_worker_host is not None:
            await dsh_worker_host.start(extra_env=ws.extra_env)
        mutated_before_overturn: bool | None = None
        user_simulator_trace: dict[str, Any] = {}
        turn_1a_replies_captured: list[str] = []
        turn_1a_stream_cursor: int | None = None

        tl = list(episode.timeline)
        idx = 0
        while idx < len(tl):
            ev = tl[idx]
            trig = ev.trigger

            # NEW-14: Decouple wall-clock burst event delivery from synchronous turn processing latency!
            is_burst_start = trig.mode == "at_offset_sec" or (
                trig.mode == "immediate" and (idx + 1) < len(tl) and tl[idx + 1].trigger.mode == "at_offset_sec"
            )
            if is_burst_start:
                end_idx = idx + 1
                while end_idx < len(tl) and tl[end_idx].trigger.mode == "at_offset_sec":
                    end_idx += 1
                burst_slice = tl[idx:end_idx]
                burst_anchor = time.perf_counter()

                async def _schedule_burst_item(b_ev: TimelineEventSpec, anchor: float) -> dict[str, Any]:
                    planned = float(b_ev.trigger.offset_sec or 0.0) if b_ev.trigger.mode == "at_offset_sec" else 0.0
                    wait_s = (anchor + planned) - time.perf_counter()
                    if wait_s > 0:
                        await asyncio.sleep(wait_s)
                    actual = round(time.perf_counter() - anchor, 3)
                    return await _process_synaptic_event(
                        b_ev,
                        trigger_fired_via=b_ev.trigger.mode,
                        planned_offset_sec=planned,
                        actual_offset_sec=actual,
                    )

                batch_traces = await asyncio.gather(
                    *(_schedule_burst_item(b_ev, burst_anchor) for b_ev in burst_slice)
                )
                event_trace.extend(batch_traces)
                idx = end_idx
                continue

            trigger_fired_via = trig.mode
            if trig.mode == "after_turn_settled":
                await mmu.drain_all_background_work()
                if turn_1a_stream_cursor is not None and not turn_1a_replies_captured:
                    for s_item in mmu.unified_stream[turn_1a_stream_cursor:]:
                        if s_item.kind == "agent_reply" and s_item.text.strip():
                            turn_1a_replies_captured.append(s_item.text.strip())
                        elif s_item.kind == "ambient_pill" and s_item.metadata.get("deferred_summary"):
                            turn_1a_replies_captured.append(str(s_item.metadata["deferred_summary"]).strip())
                    turn_1a_stream_cursor = None
            elif trig.mode == "after_worker_tool_calls":
                threshold = int(trig.worker_tool_calls_threshold or 2)
                deadline = time.perf_counter() + float(trig.fallback_timeout_sec or 15.0)
                trigger_fired_via = "timeout_fallback"
                while time.perf_counter() < deadline:
                    target_cid_list = tag_cells.get(trig.target_task_tag or "", [])
                    if target_cid_list:
                        observed = max((cell_tool_counts.get(tcid, 0) for tcid in target_cid_list), default=0)
                    else:
                        target_cid = tag_to_cell_id.get(trig.target_task_tag or "")
                        observed = (
                            cell_tool_counts.get(target_cid, 0)
                            if target_cid
                            else max(cell_tool_counts.values(), default=0)
                        )
                    if observed >= threshold:
                        trigger_fired_via = "step_threshold"
                        break
                    if not mmu.background_tasks:
                        trigger_fired_via = "no_bg_tasks"
                        break
                    tool_step_signal.clear()
                    try:
                        await asyncio.wait_for(tool_step_signal.wait(), timeout=0.5)
                    except asyncio.TimeoutError:
                        pass
            elif trig.mode == "after_env_tool_calls":
                env_threshold = int(trig.env_tool_calls_threshold or 0)
                if env_threshold <= 0:
                    trigger_fired_via = "env_step_threshold_0"
                else:
                    deadline = time.perf_counter() + float(trig.fallback_timeout_sec or 20.0)
                    wait_started = time.perf_counter()
                    trigger_fired_via = "timeout_fallback"
                    while time.perf_counter() < deadline:
                        if sum(cell_tool_counts.values()) >= env_threshold:
                            trigger_fired_via = "env_step_threshold"
                            break
                        if not mmu.background_tasks and (time.perf_counter() - wait_started) >= 1.0:
                            trigger_fired_via = "all_bg_settled_early"
                            break
                        tool_step_signal.clear()
                        try:
                            await asyncio.wait_for(tool_step_signal.wait(), timeout=0.5)
                        except asyncio.TimeoutError:
                            pass
            elif trig.mode == "after_first_mutation":
                deadline = time.perf_counter() + float(trig.fallback_timeout_sec or 75.0)
                mutated_before_overturn = False
                trigger_fired_via = "timeout_fallback"
                while time.perf_counter() < deadline:
                    has_mut = await asyncio.to_thread(
                        workspace_has_any_mutation,
                        ws,
                        extra_worktrees_dir=ws.root_dir / "worktrees",
                    )
                    if has_mut:
                        mutated_before_overturn = True
                        trigger_fired_via = "first_mutation_observed"
                        break
                    if not mmu.background_tasks and sum(cell_tool_counts.values()) > 0:
                        has_mut = await asyncio.to_thread(
                            workspace_has_any_mutation,
                            ws,
                            extra_worktrees_dir=ws.root_dir / "worktrees",
                        )
                        mutated_before_overturn = has_mut
                        trigger_fired_via = "first_mutation_observed" if has_mut else "no_bg_tasks"
                        break
                    tool_step_signal.clear()
                    try:
                        await asyncio.wait_for(tool_step_signal.wait(), timeout=0.5)
                    except asyncio.TimeoutError:
                        pass

            if (ev.metadata or {}).get("user_simulator_mode") == "ambig_swe_clarification":
                ev, user_simulator_trace = await asyncio.to_thread(
                    ControlledUserSimulator.resolve_turn_1c_event,
                    ev,
                    turn_1a_replies_captured,
                    simulator_mode="deterministic" if gaia_judge_mode == "deterministic" else "auto",
                )

            if ev.event_id == "turn_1a":
                turn_1a_stream_cursor = len(mmu.unified_stream)
            tr_item = await _process_synaptic_event(ev, trigger_fired_via=trigger_fired_via)
            if ev.event_id == "turn_1a":
                out_1a = tr_item.get("outcome") or {}
                if not out_1a.get("is_dispatch_ack") and str(out_1a.get("reply") or "").strip():
                    turn_1a_replies_captured.append(str(out_1a["reply"]).strip())
                    turn_1a_stream_cursor = None
            if user_simulator_trace.get("active") and ev.event_id == "turn_1c":
                tr_item["user_simulator"] = user_simulator_trace
            event_trace.append(tr_item)
            idx += 1

        await mmu.drain_all_background_work()
        wall_sec = round(time.perf_counter() - t0, 2)
        ended_at = time.time()

        dsh_worker_tel: dict[str, Any] = {}
        dsh_worker_bundles: list[tuple[str, list[dict[str, Any]]]] = []
        dsh_worker_contaminated = False
        dsh_worker_contam_blocked = 0
        if dsh_worker_host is not None:
            # Flush DSH `session_projcache` & `.zstd` logs cleanly before reading telemetry
            await dsh_worker_host.stop()
            dsh_worker_tel = await asyncio.to_thread(dsh_worker_host.collect_dsh_session_telemetry)
            dsh_worker_bundles = await asyncio.to_thread(dsh_worker_host._iter_decompressed_sessions)
            ep_repo_name = next(
                (
                    str(m.repo_or_category)
                    for m in episode.workspace_mounts
                    if m.pillar in ("cooperbench", "swe_interact", "ambig_swe") and "/" in str(m.repo_or_category or "")
                ),
                "workspace/project",
            )
            dsh_worker_contaminated, dsh_worker_contam_blocked = _scan_dsh_session_contamination(
                dsh_worker_bundles,
                eff_repo=ep_repo_name,
                seatbelt_enforced=(dsh_worker_host.instance_dir / "dsh_seatbelt.sb").is_file(),
            )

        base_commit_ep = next((m.base_commit for m in episode.workspace_mounts if m.base_commit), "") or ""
        task_worktrees: dict[str, Path] = {}
        tag_worktrees: dict[str, list[str]] = {}
        for tag, cids in tag_cells.items():
            valid_cells = [
                mmu.cells[cid]
                for cid in reversed(cids)
                if cid in mmu.cells and mmu.cells[cid].worktree_path and mmu.cells[cid].worktree_path.exists()
            ]
            # Rank cells that produced their own commit or modified code relative to their own `_base_git_ref`
            # ahead of Office/query cells whose worktrees merely inherited a prior wave's commit (`PA150_H4_*`).
            own_delta_paths = [
                c.worktree_path
                for c in valid_cells
                if c.worktree_path is not None
                and (
                    c.commit_sha is not None
                    or _extract_agent_diff(c.worktree_path, getattr(c, "_base_git_ref", None) or base_commit_ep).strip()
                )
            ]
            other_paths = [
                c.worktree_path
                for c in valid_cells
                if c.worktree_path is not None and c.worktree_path not in own_delta_paths
            ]
            ranked = _rank_worktree_paths_nonempty_first(own_delta_paths + other_paths, base_commit_ep)
            if ranked:
                task_worktrees[tag] = ranked[0]
                tag_worktrees[tag] = [str(p) for p in ranked]

        all_worktrees = [
            c.worktree_path
            for c in mmu.cells.values()
            if c.worktree_path is not None and c.worktree_path.exists()
        ]
        agent_replies: list[str] = []
        for item in mmu.unified_stream:
            if item.kind == "agent_reply" and item.text.strip():
                agent_replies.append(item.text.strip())
            elif item.kind == "ambient_pill" and item.metadata.get("deferred_summary"):
                agent_replies.append(str(item.metadata["deferred_summary"]).strip())

        # `A1` / `P1-3`: Isolate Gaia2-attributed replies from concurrent `task_fs` / status-peek / noise cells
        gaia_task_replies: list[str] = (
            _select_synaptic_gaia_replies(mmu, tag_cells, noise_cids=noise_cids)
            if ws.gaia_mount_id
            else []
        )

        noise_mutated_workers = 0
        for ncid in noise_cids:
            nc = mmu.cells.get(ncid)
            if nc is None:
                continue
            n_wt_dirty = bool(
                nc.worktree_path
                and nc.worktree_path.exists()
                and _extract_agent_diff(nc.worktree_path, base_commit_ep).strip()
            )
            n_fp = mmu.bus.footprints.get(ncid)
            n_uri_writes = bool(n_fp and len(n_fp.write_uris) > 0)
            n_staged = len(nc.staged_artifacts) > 0
            if n_wt_dirty or n_uri_writes or n_staged:
                noise_mutated_workers += 1

        shadow_drafts = [
            {"cell_id": cid, "target": a.target, "artifact_type": a.artifact_type, "content": a.content}
            for cid, c in mmu.cells.items()
            for a in c.staged_artifacts
        ]
        office_ev = _collect_synaptic_office_read_evidence(mmu)
        spec_codes_merged = set(office_ev["tool_read_spec_ref_codes"])
        office_spec_doc_read = bool(office_ev["office_spec_doc_read"])
        office_mount_inspected = bool(office_ev["office_mount_inspected"])
        if dsh_worker_bundles:
            for _sid, s_events in dsh_worker_bundles:
                for s_ev in s_events:
                    if str(s_ev.get("type") or "") == "tool/result":
                        s_txt = json.dumps(s_ev.get("data") or {}, ensure_ascii=False)
                        if any(tok in s_txt for tok in _OFFICE_INSPECTION_MARKERS):
                            office_mount_inspected = True
                        found_s_codes = _SPEC_REF_CODE_RE.findall(s_txt)
                        if found_s_codes:
                            spec_codes_merged.update(found_s_codes)
                            office_mount_inspected = True
                            office_spec_doc_read = True
                        elif "release_spec_addendum" in s_txt or "release_spec_addenda" in s_txt:
                            office_spec_doc_read = True

        for bg_err in getattr(mmu, "background_errors", []):
            if not any(bg_err in str(item.get("error", "")) for item in bg_task_exceptions):
                bg_task_exceptions.append({"cell_id": "unknown", "error_type": "BackgroundError", "error": str(bg_err)})

        py_s2_prompt = sum(m.prompt_tokens for m in llm.metrics)
        py_hit = sum(m.cache_hit_tokens for m in llm.metrics)
        py_miss = sum(m.cache_miss_tokens for m in llm.metrics)
        py_out = sum(m.completion_tokens for m in llm.metrics)

        dsh_s2_prompt = int(dsh_worker_tel.get("total_prompt_tokens", 0)) if dsh_worker_tel else 0
        dsh_hit = int(dsh_worker_tel.get("cache_read_tokens", 0)) if dsh_worker_tel else 0
        dsh_miss = (
            int(dsh_worker_tel.get("uncached_input_tokens", 0)) + int(dsh_worker_tel.get("cache_write_tokens", 0))
            if dsh_worker_tel
            else 0
        )
        dsh_out = int(dsh_worker_tel.get("output_tokens", 0)) if dsh_worker_tel else 0

        s2_prompt = py_s2_prompt + dsh_s2_prompt
        tot_hit = py_hit + dsh_hit
        s2_miss = py_miss + dsh_miss
        s2_out = py_out + dsh_out
        s2_cost_cny = round(estimate_cost_cny_upper_bound(tot_hit, s2_miss, s2_out), 4)

        jev_hist = list(getattr(judge, "call_history", []))
        bus_hist = list(getattr(bus_judge, "call_history", [])) if bus_judge is not None else []
        jev_latencies = [float(h.get("latency_ms", 0.0)) for h in jev_hist if h.get("latency_ms") is not None]
        s1_prompt = sum(int(h.get("input_tokens", 0)) for h in (jev_hist + bus_hist))
        s1_out = sum(int(h.get("output_tokens", 0)) for h in (jev_hist + bus_hist))
        # `A6`: System1Judge records backend strings such as `"typesafe_jev (jev-v1)"` — match via `startswith("typesafe_jev")`!
        jev_typesafe_calls = sum(1 for h in jev_hist if str(h.get("backend", "")).startswith("typesafe_jev"))
        jev_fallback_calls = sum(1 for h in jev_hist if not str(h.get("backend", "")).startswith("typesafe_jev"))
        s1_hit = int(getattr(judge, "total_cache_hit_tokens", 0))
        s1_miss = (
            int(getattr(judge, "total_cache_miss_tokens", 0)) + sum(int(h.get("input_tokens", 0)) for h in bus_hist)
            if isinstance(judge, GenerativeLLMJudge)
            else s1_prompt
        )
        s1_cost_cny = round(estimate_cost_cny_upper_bound(s1_hit, s1_miss, s1_out), 4)

        tot_prompt = s2_prompt + s1_prompt
        tot_hit = tot_hit + s1_hit
        tot_miss = s2_miss + s1_miss
        tot_out = s2_out + s1_out
        est_cny = round(s2_cost_cny + s1_cost_cny, 4)
        mean_peek_sec = (
            round((sum(peek_latencies_ms) / len(peek_latencies_ms)) / 1000.0, 3)
            if peek_latencies_ms
            else None
        )

        telemetry = {
            "worker_backend": worker_backend,
            "wall_clock_sec": wall_sec,
            "cells_total": len(mmu.cells),
            "worktrees_created": len(all_worktrees),
            "tag_worktrees": tag_worktrees,
            "noise_events_total": total_noise_in_timeline,
            "noise_dropped": noise_dropped_exact,
            "noise_dropped_cumulative": mmu.dropped_noise_events,
            "noise_spawned_workers": noise_spawned,
            "noise_mutated_workers": noise_mutated_workers,
            "duplicates_merged": dup_merged_exact,
            "duplicates_merged_cumulative": mmu.merged_duplicate_events,
            "duplicate_spawned_workers": dup_spawned,
            "real_tasks_dropped": real_tasks_dropped,
            "compound_split_executed": compound_split_executed,
            "hot_injections": hot_injections,
            "overturns": overturns,
            "mutated_before_overturn": mutated_before_overturn,
            "user_simulator": user_simulator_trace if user_simulator_trace.get("active") else None,
            "office_mount_inspected": office_mount_inspected,
            "office_spec_doc_read": office_spec_doc_read,
            "tool_read_spec_ref_codes": sorted(spec_codes_merged),
            "peek_latencies_ms": peek_latencies_ms,
            "compound_peek_latencies_ms": compound_peek_latencies_ms,
            "mean_nonblocking_peek_sec": mean_peek_sec,
            "mean_peek_latency_sec": mean_peek_sec,
            "llm_calls": len(llm.metrics) + int(dsh_worker_tel.get("total_turns", 0) if dsh_worker_tel else 0),
            "dsh_worker_sessions": int(dsh_worker_tel.get("session_count", 0)) if dsh_worker_tel else 0,
            "system2_prompt_tokens": s2_prompt,
            "system2_output_tokens": s2_out,
            "system2_cost_cny": s2_cost_cny,
            "system1_prompt_tokens": s1_prompt,
            "system1_output_tokens": s1_out,
            "system1_cost_cny": s1_cost_cny,
            "total_prompt_tokens": tot_prompt,
            "cache_hit_tokens": tot_hit,
            "cache_miss_tokens": tot_miss,
            "output_tokens": tot_out,
            "prefix_cache_hit_rate": round(tot_hit / s2_prompt, 4) if s2_prompt else 0.0,
            "overall_prompt_cache_hit_rate": round(tot_hit / tot_prompt, 4) if tot_prompt else 0.0,
            "estimated_cost_cny": est_cny,
            "jev_calls": len(jev_hist),
            "jev_typesafe_calls": jev_typesafe_calls,
            "jev_fallback_calls": jev_fallback_calls,
            "jev_mean_latency_ms": round(sum(jev_latencies) / len(jev_latencies), 1) if jev_latencies else 0.0,
            "judge_backends_used": sorted({str(h.get("backend", "")) for h in jev_hist if h.get("backend")}),
            "ablation_config": ab_cfg if ab_cfg else None,
            "cold_handoff_count": sum(int(getattr(c, "cold_handoff_count", 0)) for c in mmu.cells.values()),
            "trunk_epoch": trunk.epoch,
            "trunk_tombstones": sum(1 for e in trunk.l2_entries.values() if e.category == "tombstone"),
            "invalidated_tombstones": int(getattr(mmu, "invalidated_tombstones_count", 0)),
            "synaptic_events_count": len(mmu.bus.events_log),
            "synaptic_events": list(mmu.bus.events_log[:40]),
            "trunk_reconciliation_attempts": sum(
                int(getattr(c, "reconciliation_attempts", 0)) for c in mmu.cells.values()
            ),
            "trunk_reconciliation_successes": sum(
                int(getattr(c, "reconciliation_successes", 0)) for c in mmu.cells.values()
            ),
            "background_task_error_count": len(bg_task_exceptions),
            "held_out_contamination_blocked": (
                sum(int(getattr(c, "held_out_contamination_blocked", 0)) for c in mmu.cells.values())
                + dsh_worker_contam_blocked
            ),
            "held_out_contamination": (
                dsh_worker_contaminated
                if (worker_backend == "dsh_agent" and dsh_worker_bundles)
                else (
                    _scan_executed_text_contamination(
                        [
                            str(st.get("cmd") or st.get("cmd_tail") or "")
                            for c in mmu.cells.values()
                            for st in getattr(c, "_step_audit_trail", [])
                            if st.get("status") != "blocked"
                        ]
                    )
                    or dsh_worker_contaminated
                )
            ),
        }

        cell_statuses = {
            cid: {
                "status": c.status,
                "topic": c.topic,
                "source_kind": c.source_kind,
                "observer_only": c.observer_only,
                "worker_backend": getattr(c, "worker_backend", worker_backend),
                "total_tool_calls": c.total_tool_calls,
                "bash_calls": c.bash_calls,
                "pytest_runs": c.pytest_runs,
                "last_pytest_status": c.last_pytest_status,
                "compaction_events": c.compaction_events,
                "pruned_tool_chars": c.pruned_tool_chars,
                "env_or_api_errors": c.env_or_api_errors,
                "worktree": str(c.worktree_path) if c.worktree_path else None,
                "commit_sha": c.commit_sha,
                "merged_to_trunk": bool(getattr(c, "merged_to_trunk", False)),
                "reconciliation_attempts": int(getattr(c, "reconciliation_attempts", 0)),
                "reconciliation_successes": int(getattr(c, "reconciliation_successes", 0)),
                "final_summary": c.final_summary,
                "step_audit_trail": list(getattr(c, "_step_audit_trail", [])),
            }
            for cid, c in mmu.cells.items()
        }

        artifacts = RuntimeArtifacts(
            arm_name=arm_label,
            task_worktrees=task_worktrees,
            all_worktrees=all_worktrees,
            agent_replies=agent_replies,
            gaia_task_replies=gaia_task_replies,
            integration_repo_dir=ws.repo_dir,
            shadow_drafts=shadow_drafts,
            telemetry=telemetry,
            routing_events=routing_events,
        )
        eval_report = await asyncio.to_thread(evaluate_episode_verifiers, episode, ws, artifacts)
        forensic_diffs = await asyncio.to_thread(
            _collect_forensic_diffs, episode, ws, task_worktrees, all_worktrees
        )
        forensic_fp = await asyncio.to_thread(
            _write_forensic_bundle,
            run_root,
            episode,
            arm_label,
            {
                "episode_id": episode.episode_id,
                "track_id": episode.track_id,
                "arm_name": arm_label,
                "worker_backend": worker_backend,
                "git_head": _current_git_head(),
                "started_at": started_at,
                "ended_at": ended_at,
                "agent_diffs": forensic_diffs,
                "agent_replies": agent_replies,
                "gaia_task_replies": gaia_task_replies,
                "shadow_drafts": shadow_drafts,
                "cell_statuses": cell_statuses,
                "judge_call_history": jev_hist,
                "llm_metrics": [asdict(m) if is_dataclass(m) else m for m in llm.metrics],
                "dsh_worker_sessions": dsh_worker_tel.get("sessions", []) if dsh_worker_tel else [],
                "background_task_exceptions": bg_task_exceptions,
                "routing_events": routing_events,
                "event_trace": event_trace,
                "telemetry": telemetry,
                "verifiers": eval_report.get("verifiers", []),
            },
            forensic_root=forensic_root,
        )
        telemetry["forensic_bundle"] = str(forensic_fp)
        return {
            **eval_report,
            "domain": episode.domain,
            "scenario_type": episode.scenario_type,
            "cluster_key": _episode_cluster_key(episode),
            "has_conflict_pair": _episode_has_conflict_pair(episode),
            "episode_sha256": episode_fingerprint(episode),
            "evaluator_version": EVALUATOR_SCHEMA_VERSION,
            "model": getattr(llm, "model_name", getattr(llm, "model", os.environ.get("PA_LLM_MODEL", "deepseek-flash"))),
            "gaia_judge_mode": gaia_judge_mode,
            "telemetry": telemetry,
            "routing_events": routing_events,
            "event_trace": event_trace,
        }
    finally:
        if dsh_worker_host is not None:
            await dsh_worker_host.stop()
        await judge.aclose()
        if bus_judge is not None:
            await bus_judge.aclose()
        ws.cleanup(remove_files=not keep_workspace)


# ---------------------------------------------------------------------------
# Arm B: Genuine `@deepseek-ai/dsh` + `dsh-superagent-prototype` Episode Runner
# ---------------------------------------------------------------------------


def _dsh_feishu_open_id(sender: str) -> str:
    """Normalize sender handle into a valid DSH Feishu open_id matching `/^ou_[a-zA-Z0-9]+$/`."""
    alnum = re.sub(r"[^a-zA-Z0-9]", "", sender or "")[:18] or "colleague"
    return f"ou_{alnum}"


def _dsh_github_login(sender: str) -> str:
    """Normalize sender handle into a valid GitHub login matching DSH's `/^[A-Za-z0-9-]+$/`."""
    slug = re.sub(r"[^A-Za-z0-9-]+", "-", (sender or "").strip()).strip("-")[:32]
    return slug or "viacol"


def _is_bot_sender(sender: str) -> bool:
    s = (sender or "").strip().lower()
    return s.startswith("bot_") or s.endswith("_bot") or s in ("alertbot", "ci-bot", "ops-bot")


def _collect_episode_dsh_allowlists(
    episode: EpisodeSpec,
) -> tuple[list[str], list[dict[str, str]], list[dict[str, str]]]:
    """Register all episode GitHub & Feishu human/bot senders in DSH policy allowlists (`P0-2` fairness fix).

    Ensures neither external task webhooks nor noise/duplicate messages are silently filtered
    by static sender/bot ACLs before reaching DSH's semantic policy/decision engine.
    """
    gh_users: list[str] = []
    fs_senders: list[dict[str, str]] = []
    fs_bots: list[dict[str, str]] = []
    seen_fs: set[str] = set()
    seen_bots: set[str] = set()
    for ev in episode.timeline:
        if ev.channel.startswith("github://"):
            u = _dsh_github_login(ev.sender or "viacol")
            if u and u not in gh_users:
                gh_users.append(u)
        elif ev.kind != "user_chat":
            sname = (ev.sender or "colleague").strip()
            sid = _dsh_feishu_open_id(sname)
            if _is_bot_sender(sname):
                if sid not in seen_bots:
                    seen_bots.add(sid)
                    fs_bots.append({"id": sid, "name": sname})
            if sid not in seen_fs:
                seen_fs.add(sid)
                fs_senders.append({"id": sid, "name": sname})
    return gh_users, fs_senders, fs_bots


def _inject_event_into_dsh(dsh: GenuineDSHSuperagent, ev: TimelineEventSpec) -> dict[str, Any]:
    if ev.channel.startswith("github://"):
        title = ev.text.splitlines()[0][:110].strip() or ev.event_id
        inum = create_github_issue(dsh.world_dir, title=title, body=ev.text, user_login=_dsh_github_login(ev.sender or "viacol"))
        return {"channel": "github_issue", "issue_number": inum}

    is_group = "group" in ev.channel or ev.kind in ("external_noise", "external_task_webhook", "external_duplicate_webhook")
    chat_id = FEISHU_GROUP_CHAT_ID if is_group else FEISHU_P2P_CHAT_ID
    chat_type = "group" if is_group else "p2p"
    mention_bot = "@agent" in ev.text or (is_group and ev.kind != "external_noise")
    sname = (ev.sender or "colleague").strip()
    sender_type = "bot" if (ev.kind != "user_chat" and _is_bot_sender(sname)) else "user"
    mid = push_feishu_message(
        dsh.world_dir,
        ev.text,
        chat_id=chat_id,
        chat_type=chat_type,
        sender_id=FEISHU_USER_OPEN_ID if ev.kind == "user_chat" else _dsh_feishu_open_id(sname),
        sender_name=FEISHU_USER_NAME if ev.kind == "user_chat" else sname,
        sender_type=sender_type,
        mention_bot=mention_bot,
    )
    return {"channel": "feishu", "message_id": mid, "chat_type": chat_type}


def _promote_settled_dsh_tickets(
    dsh: GenuineDSHSuperagent,
    ws: MaterializedEpisodeWorkspace,
    promoted_heads: dict[str, str],
) -> None:
    """Symmetrically merge settled DSH ticket worktrees `--no-ff` into `dsh_clone_repo` (`main`) and push to `origin`."""
    dsh_clone = dsh.sandbox_paths.get("clone_repo")
    if dsh_clone is None or not (dsh_clone / ".git").exists():
        return
    extra_excludes = {
        "AGENTS.md",
        "WORKSPACE_POLICY.md",
        "feishu_docs",
        "gaia_cli",
        "pg_exec",
        ".workspace_root",
    } | set(ws.shared_symlinks.keys())
    tickets = dsh.get_all_tickets_detailed()
    for tk in tickets:
        tid = str(tk.get("id") or "")
        state = str(tk.get("state") or "")
        if state in ("running", "preparing"):
            continue
        raw_dir = tk.get("directory") or ((tk.get("execution") or {}).get("directory"))
        if not raw_dir:
            continue
        wt_p = Path(str(raw_dir))
        if not wt_p.exists() or not (wt_p / ".git").exists():
            continue
        cur_status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=wt_p,
            capture_output=True,
            text=True, encoding="utf-8", errors="replace",
        ).stdout.strip()
        cur_head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=wt_p,
            capture_output=True,
            text=True, encoding="utf-8", errors="replace",
        ).stdout.strip()
        if tid and promoted_heads.get(tid) == cur_head and not cur_status:
            continue
        promote_worktree_to_main(
            wt_p,
            dsh_clone,
            f"dsh_{tid[:8] or wt_p.name}",
            base_ref=ws.dsh_base_head or "HEAD",
            shared_links_dir=ws.shared_links_dir,
            extra_exclude_names=extra_excludes,
            push_origin=True,
        )
        new_head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=wt_p,
            capture_output=True,
            text=True, encoding="utf-8", errors="replace",
        ).stdout.strip()
        if tid and new_head:
            promoted_heads[tid] = new_head


def _match_dsh_trigger(
    ev: TimelineEventSpec,
    inj: dict[str, Any],
    triggers: list[dict[str, Any]],
    claimed_ids: set[str],
) -> dict[str, Any] | None:
    """Match an injected `TimelineEventSpec` to its corresponding DSH `TriggerView` (`Known-3`).

    Primary key: exact deterministic SHA-256 trigger ID (`dsh_trigger_id_for_injected(inj)`) matching
    `dsh-superagent-prototype/src/core/ids.ts`. Secondary fallbacks cover custom test fixtures.
    """
    expected_tid = dsh_trigger_id_for_injected(inj)
    if expected_tid:
        for tr in triggers:
            tid = str(tr.get("id") or "")
            if tid == expected_tid and tid not in claimed_ids:
                claimed_ids.add(tid)
                return tr

    if inj.get("channel") == "github_issue":
        inum = inj.get("issue_number")
        for tr in triggers:
            tid = str(tr.get("id") or "")
            if tid in claimed_ids or tr.get("source") != "github":
                continue
            url = str(tr.get("url") or "")
            threads = " ".join(str(x) for x in (tr.get("threads") or []))
            if f"/issues/{inum}" in url or f"#{inum}" in threads or tid.endswith(f":{inum}") or tid.endswith(f"#{inum}"):
                claimed_ids.add(tid)
                return tr
    else:
        mid = str(inj.get("message_id") or "")
        for tr in triggers:
            tid = str(tr.get("id") or "")
            if tid in claimed_ids or tr.get("source") != "feishu":
                continue
            if mid and mid in tid:
                claimed_ids.add(tid)
                return tr

    needle = " ".join(ev.text.replace("@agent", "").split())[:42]
    if needle:
        for tr in triggers:
            tid = str(tr.get("id") or "")
            if tid in claimed_ids:
                continue
            hay = " ".join(str(tr.get("text") or tr.get("preview") or "").split())
            if needle in hay:
                claimed_ids.add(tid)
                return tr
    return None


def attribute_dsh_timeline_routing(
    episode: EpisodeSpec,
    event_trace: list[dict[str, Any]],
    triggers: list[dict[str, Any]],
    ticket_by_id: dict[str, dict[str, Any]],
    *,
    use_existing_matched_ids: bool = False,
) -> dict[str, Any]:
    """Channel-order-invariant DSH trigger/ticket attribution (`v2.0.1`, `P1-2`).

    Handles both:
      1. `external_duplicate_webhook` arriving via Feishu push (`<50ms`) before GitHub issue polling (`~2s`).
      2. `user_chat` (`ev_similar_cell_followup`) arriving via Feishu P2P (`<50ms`) before GitHub issue polling +
         card evaluation (`40-84s`), where DSH creates 1 ticket on the chat arrival and coalesces the later
         GitHub issue onto that ticket (`outcome == "ignore"`).
    """
    triggers_chrono = list(triggers)
    trig_by_id = {str(t.get("id")): t for t in triggers_chrono if isinstance(t, dict) and t.get("id")}
    claimed_trig_ids: set[str] = set()
    seen_tickets: set[str] = set()
    tag_to_tickets: dict[str, list[str]] = defaultdict(list)
    ticket_to_tag: dict[str, str] = {}

    noise_dropped_exact = 0
    noise_spawned = 0
    noise_new_tids: list[str] = []
    dup_merged_exact = 0
    dup_attached = 0
    dup_ignored_by_router = 0
    dup_spawned = 0
    real_tasks_dropped = 0
    unmatched_triggers = 0
    routing_events: dict[str, dict[str, Any]] = {}

    resolved_timeline: list[
        tuple[TimelineEventSpec, dict[str, Any], dict[str, Any] | None, str, dict[str, Any], dict[str, Any], str, list[str]]
    ] = []
    for ev, trace_item in zip(episode.timeline, event_trace):
        if use_existing_matched_ids and trace_item.get("matched_trigger_id"):
            tr = trig_by_id.get(str(trace_item["matched_trigger_id"]))
        elif use_existing_matched_ids and "matched_trigger_id" in trace_item and trace_item["matched_trigger_id"] is None:
            tr = None
        else:
            inj = trace_item.get("injected") or {}
            tr = _match_dsh_trigger(ev, inj, triggers_chrono, claimed_trig_ids)
            trace_item["matched_trigger_id"] = tr.get("id") if tr else None
        if tr is None:
            resolved_timeline.append((ev, trace_item, None, "", {}, {}, "", []))
            continue
        tr_id = str(tr.get("id") or "")
        disp = tr.get("disposition") or {}
        handback_ids = {str(h.get("ticketId")) for h in (disp.get("handbacks") or []) if h.get("ticketId")}
        dec = disp.get("decision") or {}
        dec_outcome = str(dec.get("outcome") or "")
        dec_tid = str(dec.get("ticketId") or "")
        fanout_tids = [str(tid) for tid in (disp.get("fanout") or []) if tid and str(tid) not in handback_ids]
        origin_tids = [
            tid
            for tid, tk in ticket_by_id.items()
            if (tk.get("origin") or {}).get("triggerId") == tr_id
            or tr_id in ((tk.get("origin") or {}).get("triggerIds") or [])
            or any(
                isinstance(inp, dict)
                and inp.get("kind") == "trigger"
                and str(inp.get("triggerId") or "") == tr_id
                for inp in (tk.get("inputs") or [])
            )
        ]
        reached_tids: list[str] = []
        for tid in origin_tids + ([dec_tid] if dec_tid else []) + fanout_tids:
            if tid and tid not in reached_tids:
                reached_tids.append(tid)
        resolved_timeline.append((ev, trace_item, tr, tr_id, disp, dec, dec_outcome, reached_tids))

    swapped_suppressed_primary_ids: set[str] = set()
    swapped_chat_suppressed_primary_ids: set[str] = set()
    swapped_carrier_dup_ids: set[str] = set()
    swapped_carrier_chat_ids: set[str] = set()
    has_separate_dup_by_tag: dict[str, bool] = {}
    events_by_tag: dict[str, list[tuple[TimelineEventSpec, dict[str, Any], str, list[str]]]] = defaultdict(list)
    for ev, _ti, tr, _tr_id, disp, _dec, dec_outcome, reached_tids in resolved_timeline:
        if tr is not None and ev.target_task_tag:
            events_by_tag[ev.target_task_tag].append((ev, disp, dec_outcome, reached_tids))

    for tag, tag_items in events_by_tag.items():
        primaries = [x for x in tag_items if x[0].kind == "external_task_webhook"]
        dups = [x for x in tag_items if x[0].kind == "external_duplicate_webhook"]
        chats = [x for x in tag_items if x[0].kind == "user_chat"]
        has_separate_dup_by_tag[tag] = len(dups) > 0
        if primaries and all(not p[3] for p in primaries):
            carrier_dups = [d for d in dups if d[3]]
            carrier_chats = [c for c in chats if c[3]]
            if carrier_dups:
                swapped_carrier_dup_ids.add(carrier_dups[0][0].event_id)
                for p_ev, p_disp, p_outcome, _p_reached in primaries:
                    if p_disp.get("filtered") or p_outcome == "ignore":
                        swapped_suppressed_primary_ids.add(p_ev.event_id)
            elif carrier_chats:
                swapped_carrier_chat_ids.add(carrier_chats[0][0].event_id)
                for p_ev, p_disp, p_outcome, _p_reached in primaries:
                    if p_disp.get("filtered") or p_outcome == "ignore":
                        swapped_suppressed_primary_ids.add(p_ev.event_id)
                        swapped_chat_suppressed_primary_ids.add(p_ev.event_id)

    # Pre-bind swapped carrier tickets BEFORE scanning timeline order so earlier timeline events
    # (`ev_feat_a`, `ev_dup_feat_a`) see the coalesced target ticket for `tag` (`P1-2`).
    for ev, _ti, tr, _tr_id, _disp, _dec, _dec_outcome, reached_tids in resolved_timeline:
        if tr is not None and ev.target_task_tag and ev.event_id in (swapped_carrier_dup_ids | swapped_carrier_chat_ids):
            for tid in reached_tids:
                if tid not in tag_to_tickets[ev.target_task_tag]:
                    tag_to_tickets[ev.target_task_tag].append(tid)
                ticket_to_tag.setdefault(tid, ev.target_task_tag)

    for ev, _trace_item, tr, tr_id, disp, dec, dec_outcome, reached_tids in resolved_timeline:
        if tr is None:
            unmatched_triggers += 1
            routing_events[ev.event_id] = {
                "routed_tags": [],
                "reached_tickets": [],
                "spawned_new_worker": False,
                "distractor_overturned": False,
            }
            continue

        new_tids = [tid for tid in reached_tids if tid not in seen_tickets]
        seen_tickets.update(reached_tids)

        if ev.kind == "external_noise":
            if disp.get("filtered") or dec_outcome == "ignore" or not reached_tids:
                noise_dropped_exact += 1
            if new_tids:
                noise_spawned += len(new_tids)
                noise_new_tids.extend(new_tids)
        elif ev.event_id in swapped_carrier_dup_ids:
            if ev.target_task_tag:
                for tid in reached_tids:
                    if tid not in tag_to_tickets[ev.target_task_tag]:
                        tag_to_tickets[ev.target_task_tag].append(tid)
                    ticket_to_tag.setdefault(tid, ev.target_task_tag)
        elif ev.kind == "external_duplicate_webhook":
            exp_tids = set(tag_to_tickets.get(ev.target_task_tag or "", []))
            if new_tids:
                dup_spawned += len(new_tids)
            else:
                if exp_tids and (set(reached_tids) & exp_tids):
                    dup_attached += 1
                    dup_merged_exact += 1
                elif (dec_outcome == "ignore" or disp.get("filtered")) and exp_tids:
                    dup_ignored_by_router += 1
                    dup_merged_exact += 1
        elif ev.kind in ("external_task_webhook", "user_chat"):
            if ev.event_id in swapped_suppressed_primary_ids:
                if not (
                    ev.event_id in swapped_chat_suppressed_primary_ids
                    and has_separate_dup_by_tag.get(ev.target_task_tag or "", False)
                ):
                    dup_ignored_by_router += 1
                    dup_merged_exact += 1
            elif ev.kind == "external_task_webhook" and (
                disp.get("filtered") or (dec_outcome == "ignore" and not reached_tids)
            ):
                real_tasks_dropped += 1
            if ev.target_task_tag and (
                ev.expected_mmu_behavior != "similar_cell_disambiguation_followup"
                or ev.event_id in swapped_carrier_chat_ids
            ):
                for tid in reached_tids:
                    if tid not in tag_to_tickets[ev.target_task_tag]:
                        tag_to_tickets[ev.target_task_tag].append(tid)
                    ticket_to_tag.setdefault(tid, ev.target_task_tag)

        routed_tags: list[str] = []
        for tid in reached_tids:
            tname = ticket_to_tag.get(tid)
            if tname and tname not in routed_tags:
                routed_tags.append(tname)

        coalesced_chat_carrier = (
            ev.event_id in swapped_carrier_chat_ids
            and len(new_tids) <= 1
            and len(tag_to_tickets.get(ev.target_task_tag or "", [])) <= 1
        )
        routing_events[ev.event_id] = {
            "trigger_id": tr_id,
            "decision_outcome": dec_outcome,
            "routed_tags": routed_tags,
            "reached_tickets": reached_tids,
            "new_tickets": new_tids,
            "spawned_new_worker": False if coalesced_chat_carrier else (len(new_tids) > 0),
            "distractor_overturned": False,
        }

    return {
        "tag_to_tickets": tag_to_tickets,
        "ticket_to_tag": ticket_to_tag,
        "noise_dropped_exact": noise_dropped_exact,
        "noise_spawned": noise_spawned,
        "noise_new_tids": noise_new_tids,
        "dup_merged_exact": dup_merged_exact,
        "dup_attached": dup_attached,
        "dup_ignored_by_router": dup_ignored_by_router,
        "dup_spawned": dup_spawned,
        "real_tasks_dropped": real_tasks_dropped,
        "unmatched_triggers": unmatched_triggers,
        "routing_events": routing_events,
    }


async def run_episode_dsh_superagent(
    episode: EpisodeSpec,
    run_root: Path,
    *,
    dsh_port: int = 13082,
    gaia_server: Gaia2ServerProcess | None = None,
    gaia_port: int = 19012,
    gaia_judge_mode: str = "auto",
    keep_workspace: bool = False,
    forensic_root: Path | None = None,
) -> dict[str, Any]:
    load_project_dotenv()
    started_at = time.time()
    t0 = time.perf_counter()
    ws = await asyncio.to_thread(
        materialize_episode_workspace,
        episode,
        "dsh_superagent",
        run_root,
        gaia_server=gaia_server,
        gaia_port=gaia_port,
        gaia_judge_mode=gaia_judge_mode,
    )
    dsh = GenuineDSHSuperagent(instance_dir=ws.root_dir / "dsh_instance", port=dsh_port)
    ws.dsh_work_dir = dsh.state_dir / "work"

    policy_instruction = (
        "Accept actionable user questions, engineering bug reports, feature requests, database/document tasks, "
        "and personal schedule/email operations. Ignore casual non-actionable team chatter or routine passing notices."
    )
    gh_users, fs_senders, fs_bots = _collect_episode_dsh_allowlists(episode)
    ep_repo_name = next(
        (
            str(m.repo_or_category)
            for m in episode.workspace_mounts
            if m.pillar in ("cooperbench", "swe_interact", "ambig_swe") and "/" in str(m.repo_or_category or "")
        ),
        "workspace/project",
    )
    promoted_dsh_heads: dict[str, str] = {}

    try:
        await asyncio.to_thread(
            dsh.bootstrap_and_start,
            extra_env=ws.extra_env,
            policy_instruction=policy_instruction,
            agents_md_override=ws.agents_md,
            extra_repo_files=ws.extra_repo_files,
            seed_repo_dir=ws.repo_dir,
            shared_symlinks=ws.shared_symlinks,
            allowed_github_users=gh_users,
            allowed_feishu_senders=fs_senders,
            allowed_feishu_bots=fs_bots,
            repo_full_name=ep_repo_name,
        )
        # P0-1: Record DSH clone_repo's initial HEAD so read-only DSH worktrees compare against their true base SHA
        dsh_clone_init = dsh.sandbox_paths.get("clone_repo")
        if dsh_clone_init is not None and dsh_clone_init.exists():
            ws.dsh_base_head = subprocess.run(
                ["git", "-C", str(dsh_clone_init), "rev-parse", "HEAD"],
                capture_output=True,
                text=True, encoding="utf-8", errors="replace",
            ).stdout.strip() or None

        expected_received = 0
        expected_trigger_ids: set[str] = set()
        burst_anchor: float | None = None
        peek_latencies_ms: list[float] = []
        peek_timed_out_count = 0
        mutated_before_overturn_dsh: bool | None = None
        user_simulator_trace_dsh: dict[str, Any] = {}
        turn_1a_replies_dsh: list[str] = []
        event_trace: list[dict[str, Any]] = []

        for idx_ev, ev in enumerate(episode.timeline):
            trig = ev.trigger
            trigger_fired_via = trig.mode
            planned_offset_sec: float | None = None
            actual_offset_sec: float | None = None
            if trig.mode == "immediate" and (idx_ev + 1) < len(episode.timeline) and episode.timeline[idx_ev + 1].trigger.mode == "at_offset_sec":
                burst_anchor = time.perf_counter()
                planned_offset_sec = 0.0
                actual_offset_sec = 0.0
            elif trig.mode == "at_offset_sec":
                if burst_anchor is None:
                    burst_anchor = time.perf_counter()
                planned_offset_sec = float(trig.offset_sec or 0.0)
                wait_s = (burst_anchor + planned_offset_sec) - time.perf_counter()
                if wait_s > 0:
                    await asyncio.sleep(wait_s)
                actual_offset_sec = round(time.perf_counter() - burst_anchor, 3)
            elif trig.mode == "after_turn_settled":
                await asyncio.to_thread(
                    dsh.wait_until_quiescent,
                    expected_received=max(1, expected_received),
                    expected_trigger_ids=set(expected_trigger_ids),
                    timeout_s=360.0,
                )
                await asyncio.to_thread(_promote_settled_dsh_tickets, dsh, ws, promoted_dsh_heads)
                burst_anchor = None
            elif trig.mode == "after_worker_tool_calls":
                threshold = int(trig.worker_tool_calls_threshold or 2)
                deadline = time.perf_counter() + float(trig.fallback_timeout_sec or 18.0)
                trigger_fired_via = "timeout_fallback"
                while time.perf_counter() < deadline:
                    open_tks = await asyncio.to_thread(dsh.get_tickets, "open")
                    sids = {str(t.get("sessionId")) for t in open_tks if t.get("sessionId")}
                    if dsh.count_worker_tool_calls(session_ids=sids or None) >= threshold:
                        trigger_fired_via = "step_threshold"
                        break
                    await asyncio.sleep(0.6)
            elif trig.mode == "after_env_tool_calls":
                env_threshold = int(trig.env_tool_calls_threshold or 0)
                if env_threshold <= 0:
                    trigger_fired_via = "env_step_threshold_0"
                else:
                    deadline = time.perf_counter() + float(trig.fallback_timeout_sec or 20.0)
                    trigger_fired_via = "timeout_fallback"
                    while time.perf_counter() < deadline:
                        if dsh.count_worker_tool_calls() >= env_threshold:
                            trigger_fired_via = "env_step_threshold"
                            break
                        await asyncio.sleep(0.5)
            elif trig.mode == "after_first_mutation":
                deadline = time.perf_counter() + float(trig.fallback_timeout_sec or 75.0)
                mutated_before_overturn_dsh = False
                trigger_fired_via = "timeout_fallback"
                while time.perf_counter() < deadline:
                    has_mut = await asyncio.to_thread(
                        workspace_has_any_mutation,
                        ws,
                        extra_worktrees_dir=dsh.state_dir / "work",
                    )
                    if has_mut:
                        mutated_before_overturn_dsh = True
                        trigger_fired_via = "first_mutation_observed"
                        break
                    await asyncio.sleep(0.6)

            if (ev.metadata or {}).get("user_simulator_mode") == "ambig_swe_clarification":
                ev, user_simulator_trace_dsh = await asyncio.to_thread(
                    ControlledUserSimulator.resolve_turn_1c_event,
                    ev,
                    turn_1a_replies_dsh,
                    simulator_mode="deterministic" if gaia_judge_mode == "deterministic" else "auto",
                )

            work_root_cur = dsh.state_dir / "work"
            existing_dsh_wts = (
                [p for p in work_root_cur.iterdir() if p.is_dir()] if work_root_cur.exists() else []
            )
            updated_files = _apply_workspace_file_updates(
                ev,
                ws.repo_dir,
                dsh_clone_repo=dsh.sandbox_paths.get("clone_repo"),
                sync_worktrees=existing_dsh_wts,
            )
            replies_before = len(list_bot_replies(dsh.world_dir))
            ev_start = time.perf_counter()
            inj = await asyncio.to_thread(_inject_event_into_dsh, dsh, ev)
            expected_received += 1
            exp_tid = dsh_trigger_id_for_injected(inj)
            if exp_tid:
                expected_trigger_ids.add(exp_tid)

            if ev.expected_mmu_behavior in _PEEK_BEHAVIORS:
                # Measure time until DSH posts a status reply on Feishu (bounded at 45s so burst continues)
                peek_deadline = time.perf_counter() + 45.0
                got_peek_reply = False
                while time.perf_counter() < peek_deadline:
                    if len(list_bot_replies(dsh.world_dir)) > replies_before:
                        got_peek_reply = True
                        break
                    await asyncio.sleep(0.4)
                if not got_peek_reply:
                    peek_timed_out_count += 1
                peek_latencies_ms.append(round((time.perf_counter() - ev_start) * 1000.0, 1))

            if ev.expected_mmu_behavior in ("align_readonly_zero_wt", "side_query_readonly_zero_wt"):
                await asyncio.to_thread(
                    dsh.wait_until_quiescent,
                    expected_received=expected_received,
                    expected_trigger_ids=set(expected_trigger_ids),
                    timeout_s=240.0,
                )
                ws.record_event_snapshots(ev.event_id)
                if ev.event_id == "turn_1a":
                    turn_1a_replies_dsh = [
                        str(r.get("markdown") or "").strip()
                        for r in list_bot_replies(dsh.world_dir)[replies_before:]
                        if r.get("markdown")
                    ]

            event_trace.append(
                {
                    "event_id": ev.event_id,
                    "kind": ev.kind,
                    "trigger_fired_via": trigger_fired_via,
                    "planned_offset_sec": planned_offset_sec,
                    "actual_offset_sec": actual_offset_sec,
                    "replies_before": replies_before,
                    "updated_files": updated_files,
                    "elapsed_ms": round((time.perf_counter() - ev_start) * 1000.0, 1),
                    "injected": inj,
                    "expected_trigger_id": exp_tid,
                }
            )

        quiesce_info = await asyncio.to_thread(
            dsh.wait_until_quiescent,
            expected_received=max(1, expected_received),
            expected_trigger_ids=set(expected_trigger_ids),
            timeout_s=480.0,
        )
        await asyncio.to_thread(_promote_settled_dsh_tickets, dsh, ws, promoted_dsh_heads)
        wall_sec = round(time.perf_counter() - t0, 2)
        ended_at = time.time()

        work_root = dsh.state_dir / "work"
        all_worktrees = sorted([p for p in work_root.iterdir() if p.is_dir()]) if work_root.exists() else []
        replies = [str(r.get("markdown") or "").strip() for r in list_bot_replies(dsh.world_dir) if r.get("markdown")]
        triggers = await asyncio.to_thread(dsh.get_triggers, 200)
        decisions = await asyncio.to_thread(dsh.get_decisions)
        tickets = await asyncio.to_thread(dsh.get_all_tickets_detailed)
        sess_tel = dsh.collect_dsh_session_telemetry()
        office_inspected = await asyncio.to_thread(dsh.inspect_worker_office_reads)

        dsh_spec_codes: set[str] = set()
        dsh_spec_doc_read = False
        local_issue_nums: set[int] = set()
        sdb_fp = dsh.world_dir / "sandbox_db.sqlite"
        if sdb_fp.is_file():
            import sqlite3 as _sqlite3

            try:
                with _sqlite3.connect(str(sdb_fp), timeout=5.0) as _conn:
                    local_issue_nums = {int(row[0]) for row in _conn.execute("SELECT number FROM issues").fetchall()}
            except Exception:
                local_issue_nums = set()
        dsh_contam_samples: list[str] = []
        session_bundles = await asyncio.to_thread(dsh._iter_decompressed_sessions)
        dsh_contaminated, dsh_contam_blocked = _scan_dsh_session_contamination(
            session_bundles,
            eff_repo=ep_repo_name,
            local_issue_numbers=local_issue_nums,
            seatbelt_enforced=(dsh.instance_dir / "dsh_seatbelt.sb").is_file(),
            samples_out=dsh_contam_samples,
        )
        for _sid, s_events in session_bundles:
            for s_ev in s_events:
                ev_type = str(s_ev.get("type") or "")
                if ev_type == "tool/result":
                    s_txt = json.dumps(s_ev.get("data") or {}, ensure_ascii=False)
                    found_s_codes = _SPEC_REF_CODE_RE.findall(s_txt)
                    if found_s_codes:
                        dsh_spec_codes.update(found_s_codes)
                        dsh_spec_doc_read = True
                    elif "release_spec_addendum" in s_txt or "release_spec_addenda" in s_txt:
                        dsh_spec_doc_read = True
        for rep_txt in replies:
            for c_match in _SPEC_REF_CODE_RE.findall(rep_txt):
                dsh_spec_codes.add(c_match)
                dsh_spec_doc_read = True

        ticket_by_id = {str(t["id"]): t for t in tickets if t.get("id")}
        attr_res = attribute_dsh_timeline_routing(
            episode=episode,
            event_trace=event_trace,
            triggers=triggers,
            ticket_by_id=ticket_by_id,
            use_existing_matched_ids=False,
        )
        tag_to_tickets = attr_res["tag_to_tickets"]
        ticket_to_tag = attr_res["ticket_to_tag"]
        noise_dropped_exact = attr_res["noise_dropped_exact"]
        noise_spawned = attr_res["noise_spawned"]
        noise_new_tids = attr_res["noise_new_tids"]
        dup_merged_exact = attr_res["dup_merged_exact"]
        dup_attached = attr_res["dup_attached"]
        dup_ignored_by_router = attr_res["dup_ignored_by_router"]
        dup_spawned = attr_res["dup_spawned"]
        real_tasks_dropped = attr_res["real_tasks_dropped"]
        unmatched_triggers = attr_res["unmatched_triggers"]
        routing_events = attr_res["routing_events"]

        base_commit_ep = next((m.base_commit for m in episode.workspace_mounts if m.base_commit), "") or ""
        task_worktrees: dict[str, Path] = {}
        tag_worktrees: dict[str, list[str]] = {}
        for tag, tids in tag_to_tickets.items():
            raw_paths: list[Path] = []
            for tid in reversed(tids):
                tk = ticket_by_id.get(tid) or {}
                raw_dir = tk.get("directory") or ((tk.get("execution") or {}).get("directory"))
                if raw_dir:
                    p = Path(str(raw_dir))
                    if p.exists():
                        raw_paths.append(p)
            ranked = _rank_worktree_paths_nonempty_first(raw_paths, base_commit_ep)
            if ranked:
                task_worktrees[tag] = ranked[0]
                tag_worktrees[tag] = [str(p) for p in ranked]

        noise_mutated_workers = 0
        for ntid in noise_new_tids:
            ntk = ticket_by_id.get(ntid) or {}
            n_dir = ntk.get("directory") or ((ntk.get("execution") or {}).get("directory"))
            if n_dir:
                np_path = Path(str(n_dir))
                if np_path.exists() and _extract_agent_diff(np_path, base_commit_ep).strip():
                    noise_mutated_workers += 1

        # NEW-15 / P1-7: Require genuine multi-ticket routing OR both new ticket + reply strictly before subsequent turn
        trace_list = list(event_trace)
        trace_idx_by_ev_id = {t["event_id"]: i for i, t in enumerate(trace_list)}
        compound_split_executed = False
        for ev in episode.timeline:
            if ev.expected_mmu_behavior == "compound_split_and_parallel_route":
                rev = routing_events.get(ev.event_id, {})
                ev_idx = trace_idx_by_ev_id.get(ev.event_id, -1)
                tr_meta = trace_list[ev_idx] if ev_idx >= 0 else {}
                rep_before = int(tr_meta.get("replies_before", 0))
                rep_bound = (
                    int(trace_list[ev_idx + 1].get("replies_before", len(replies)))
                    if (ev_idx >= 0 and ev_idx + 1 < len(trace_list))
                    else len(replies)
                )
                replies_during = rep_bound - rep_before
                reached_cnt = len(rev.get("reached_tickets", []))
                new_cnt = len(rev.get("new_tickets", []))
                if reached_cnt >= 2 or (new_cnt >= 1 and replies_during >= 1 and reached_cnt >= 1):
                    compound_split_executed = True
                    break

        # `A1` / `P1-3`: Attribute DSH Gaia2 replies cleanly when concurrent `O-2` / `O-1` tasks run,
        # stripping initial Assessment acknowledgement rows (`disp.acked`) when a Worker posts the substantive reply.
        dsh_gaia_replies: list[str] = []
        if ws.gaia_mount_id:
            raw_bot_rows = list_bot_replies(dsh.world_dir)
            acked_mids: set[str] = set()
            trig_by_id = {str(tr.get("id") or ""): tr for tr in triggers if tr.get("id")}
            for t in event_trace:
                mid_cand = str(t.get("injected", {}).get("message_id") or "")
                mt_id = str(t.get("matched_trigger_id") or t.get("expected_trigger_id") or "")
                tr_obj = trig_by_id.get(mt_id) or {}
                if mid_cand and (tr_obj.get("disposition") or {}).get("acked"):
                    acked_mids.add(mid_cand)

            gaia_ev_mids = {
                str(t.get("injected", {}).get("message_id") or "")
                for ev, t in zip(episode.timeline, event_trace)
                if ev.target_task_tag
                and (
                    "gaia" in ev.target_task_tag
                    or ev.target_task_tag in ("office_main", "office_inflight", "primary_task", "office_task")
                )
            }
            gaia_ev_mids.discard("")
            if gaia_ev_mids and any(str(r.get("reply_to_message_id") or "") in gaia_ev_mids for r in raw_bot_rows):
                rows_by_mid: dict[str, list[str]] = defaultdict(list)
                ordered_mids: list[str] = []
                for r in raw_bot_rows:
                    rmid = str(r.get("reply_to_message_id") or "")
                    md_txt = str(r.get("markdown") or "").strip()
                    if rmid in gaia_ev_mids and md_txt:
                        if rmid not in rows_by_mid:
                            ordered_mids.append(rmid)
                        rows_by_mid[rmid].append(md_txt)
                for rmid in ordered_mids:
                    m_rows = rows_by_mid[rmid]
                    if len(m_rows) >= 2 and rmid in acked_mids:
                        dsh_gaia_replies.extend(m_rows[1:])
                    elif len(m_rows) >= 2:
                        # Even if trigger disposition metadata was omitted, keep the final substantive reply(ies) after initial ack
                        dsh_gaia_replies.extend(m_rows[1:])
                    else:
                        dsh_gaia_replies.extend(m_rows)
            else:
                dsh_gaia_replies = list(replies)
        hot_injections_dsh = sum(
            1
            for ev in episode.timeline
            if ev.expected_mmu_behavior in ("hot_constraint_injection", "compound_split_and_parallel_route")
            and len(routing_events.get(ev.event_id, {}).get("reached_tickets", [])) >= 1
        )
        est_cny = round(
            estimate_cost_cny_upper_bound(
                int(sess_tel["cache_read_tokens"]),
                int(sess_tel["uncached_input_tokens"]) + int(sess_tel["cache_write_tokens"]),
                int(sess_tel["output_tokens"]),
            ),
            4,
        )
        incomplete_execution_timeout = bool(
            quiesce_info.get("timed_out")
            and (
                int(quiesce_info.get("running_workers", 0)) > 0
                or int(quiesce_info.get("queued_workers", 0)) > 0
                or int(quiesce_info.get("evaluating_threads", 0)) > 0
                or int(quiesce_info.get("active_tickets", 0)) > 0
                or bool(quiesce_info.get("open_turn"))
            )
        )
        mean_peek_sec_dsh = (
            round((sum(peek_latencies_ms) / len(peek_latencies_ms)) / 1000.0, 3)
            if peek_latencies_ms
            else None
        )

        telemetry = {
            "wall_clock_sec": wall_sec,
            "quiesce_timed_out": bool(quiesce_info.get("timed_out", False)),
            "incomplete_execution_timeout": incomplete_execution_timeout,
            "quiesce_diagnostics": quiesce_info,
            "triggers_observed": len(triggers),
            "unmatched_triggers": unmatched_triggers,
            "decisions_recorded": len(decisions),
            "tickets_created": len(tickets),
            "worktrees_created": len(all_worktrees),
            "tag_worktrees": tag_worktrees,
            "noise_events_total": sum(1 for e in episode.timeline if e.kind == "external_noise"),
            "noise_dropped": noise_dropped_exact,
            "noise_spawned_workers": noise_spawned,
            "noise_mutated_workers": noise_mutated_workers,
            "duplicates_merged": dup_merged_exact,
            "duplicates_attached_to_ticket": dup_attached,
            "duplicates_ignored_by_router": dup_ignored_by_router,
            "duplicate_spawned_workers": dup_spawned,
            "real_tasks_dropped": real_tasks_dropped,
            "compound_split_executed": compound_split_executed,
            "hot_injections": hot_injections_dsh,
            "mutated_before_overturn": mutated_before_overturn_dsh,
            "user_simulator": user_simulator_trace_dsh if user_simulator_trace_dsh.get("active") else None,
            "office_mount_inspected": bool(office_inspected or bool(dsh_spec_codes)),
            "office_spec_doc_read": bool(dsh_spec_doc_read),
            "tool_read_spec_ref_codes": sorted(dsh_spec_codes),
            "peek_latencies_ms": peek_latencies_ms,
            "peek_timed_out_count": peek_timed_out_count,
            "mean_nonblocking_peek_sec": mean_peek_sec_dsh,
            "mean_peek_latency_sec": mean_peek_sec_dsh,
            "llm_sessions": sess_tel["session_count"],
            "total_prompt_tokens": sess_tel["total_prompt_tokens"],
            "cache_hit_tokens": sess_tel["cache_read_tokens"],
            "cache_miss_tokens": sess_tel["uncached_input_tokens"] + sess_tel["cache_write_tokens"],
            "output_tokens": sess_tel["output_tokens"],
            "prefix_cache_hit_rate": sess_tel["prefix_cache_hit_rate"],
            "overall_prompt_cache_hit_rate": sess_tel["prefix_cache_hit_rate"],
            "estimated_cost_cny": est_cny,
            "held_out_contamination_blocked": dsh_contam_blocked,
            "held_out_contamination": dsh_contaminated,
            "held_out_contamination_samples": dsh_contam_samples,
        }

        artifacts = RuntimeArtifacts(
            arm_name="dsh_superagent",
            task_worktrees=task_worktrees,
            all_worktrees=all_worktrees,
            agent_replies=replies,
            gaia_task_replies=dsh_gaia_replies,
            integration_repo_dir=dsh.sandbox_paths.get("clone_repo"),
            shadow_drafts=[],
            telemetry=telemetry,
            routing_events=routing_events,
        )
        eval_report = await asyncio.to_thread(evaluate_episode_verifiers, episode, ws, artifacts)
        forensic_diffs = await asyncio.to_thread(
            _collect_forensic_diffs, episode, ws, task_worktrees, all_worktrees
        )
        forensic_fp = await asyncio.to_thread(
            _write_forensic_bundle,
            run_root,
            episode,
            "dsh_superagent",
            {
                "episode_id": episode.episode_id,
                "track_id": episode.track_id,
                "arm_name": "dsh_superagent",
                "git_head": _current_git_head(),
                "started_at": started_at,
                "ended_at": ended_at,
                "quiesce_info": quiesce_info,
                "agent_diffs": forensic_diffs,
                "agent_replies": replies,
                "gaia_task_replies": dsh_gaia_replies,
                "triggers": triggers,
                "decisions": decisions,
                "tickets": tickets,
                "sessions": sess_tel.get("sessions", []),
                "routing_events": routing_events,
                "event_trace": event_trace,
                "telemetry": telemetry,
                "verifiers": eval_report.get("verifiers", []),
            },
            forensic_root=forensic_root,
        )
        telemetry["forensic_bundle"] = str(forensic_fp)
        return {
            **eval_report,
            "domain": episode.domain,
            "scenario_type": episode.scenario_type,
            "cluster_key": _episode_cluster_key(episode),
            "has_conflict_pair": _episode_has_conflict_pair(episode),
            "episode_sha256": episode_fingerprint(episode),
            "evaluator_version": EVALUATOR_SCHEMA_VERSION,
            "model": os.environ.get("PA_LLM_MODEL", "deepseek-flash"),
            "gaia_judge_mode": gaia_judge_mode,
            "telemetry": telemetry,
            "routing_events": routing_events,
            "event_trace": event_trace,
        }
    finally:
        dsh.stop()
        ws.cleanup(remove_files=not keep_workspace)


# ---------------------------------------------------------------------------
# Arm C: Open-Source Black-Box `AgentHarnessAdapter` Runner (`v2.0.0`)
# ---------------------------------------------------------------------------


async def run_episode_external_adapter(
    episode: EpisodeSpec,
    run_root: Path,
    adapter: AgentHarnessAdapter,
    *,
    gaia_server: Gaia2ServerProcess | None = None,
    gaia_port: int = 19013,
    gaia_judge_mode: str = "auto",
    keep_workspace: bool = False,
    forensic_root: Path | None = None,
) -> dict[str, Any]:
    """Run an arbitrary external `AgentHarnessAdapter` against an isolated `Synaptic-Bench-150` workspace."""
    load_project_dotenv()
    started_at = time.time()
    t0 = time.perf_counter()
    arm_name = adapter.arm_name
    ws = await asyncio.to_thread(
        materialize_episode_workspace,
        episode,
        arm_name,
        run_root,
        gaia_server=gaia_server,
        gaia_port=gaia_port,
        gaia_judge_mode=gaia_judge_mode,
    )
    event_trace: list[dict[str, Any]] = []
    mutated_before_overturn: bool | None = None
    user_simulator_trace: dict[str, Any] = {}
    turn_1a_replies_ext: list[str] = []

    try:
        await adapter.setup(episode, ws)
        burst_anchor: float | None = None

        for idx_ev, ev in enumerate(episode.timeline):
            trig = ev.trigger
            trigger_fired_via = trig.mode
            if trig.mode == "immediate" and (idx_ev + 1) < len(episode.timeline) and episode.timeline[idx_ev + 1].trigger.mode == "at_offset_sec":
                burst_anchor = time.perf_counter()
            elif trig.mode == "at_offset_sec":
                if burst_anchor is None:
                    burst_anchor = time.perf_counter()
                wait_s = (burst_anchor + float(trig.offset_sec or 0.0)) - time.perf_counter()
                if wait_s > 0:
                    await asyncio.sleep(wait_s)
            elif trig.mode == "after_turn_settled":
                await adapter.wait_until_quiescent(timeout_sec=360.0)
                burst_anchor = None
            elif trig.mode == "after_worker_tool_calls":
                threshold = int(trig.worker_tool_calls_threshold or 2)
                deadline = time.perf_counter() + float(trig.fallback_timeout_sec or 15.0)
                trigger_fired_via = "timeout_fallback"
                while time.perf_counter() < deadline:
                    if adapter.count_executed_tool_calls() >= threshold:
                        trigger_fired_via = "step_threshold"
                        break
                    await asyncio.sleep(0.2)
            elif trig.mode == "after_env_tool_calls":
                env_threshold = int(trig.env_tool_calls_threshold or 0)
                if env_threshold <= 0:
                    trigger_fired_via = "env_step_threshold_0"
                else:
                    deadline = time.perf_counter() + float(trig.fallback_timeout_sec or 15.0)
                    trigger_fired_via = "timeout_fallback"
                    while time.perf_counter() < deadline:
                        if adapter.count_executed_tool_calls() >= env_threshold:
                            trigger_fired_via = "env_step_threshold"
                            break
                        await asyncio.sleep(0.2)
            elif trig.mode == "after_first_mutation":
                deadline = time.perf_counter() + float(trig.fallback_timeout_sec or 30.0)
                mutated_before_overturn = False
                trigger_fired_via = "timeout_fallback"
                while time.perf_counter() < deadline:
                    has_mut = await asyncio.to_thread(
                        workspace_has_any_mutation,
                        ws,
                        extra_worktrees_dir=ws.root_dir / "worktrees",
                    )
                    if has_mut:
                        mutated_before_overturn = True
                        trigger_fired_via = "first_mutation_observed"
                        break
                    await asyncio.sleep(0.2)

            if (ev.metadata or {}).get("user_simulator_mode") == "ambig_swe_clarification":
                ev, user_simulator_trace = await asyncio.to_thread(
                    ControlledUserSimulator.resolve_turn_1c_event,
                    ev,
                    turn_1a_replies_ext,
                    simulator_mode="deterministic" if gaia_judge_mode == "deterministic" else "auto",
                )

            wt_root = ws.root_dir / "worktrees"
            existing_wts = [p for p in wt_root.iterdir() if p.is_dir()] if wt_root.exists() else []
            updated_files = _apply_workspace_file_updates(ev, ws.repo_dir, sync_worktrees=existing_wts)
            replies_before_n = len(adapter.latest_agent_replies())
            out = await adapter.deliver_event(ev, ws)

            if ev.expected_mmu_behavior in ("align_readonly_zero_wt", "side_query_readonly_zero_wt"):
                await adapter.wait_until_quiescent(timeout_sec=240.0)
                ws.record_event_snapshots(ev.event_id)
                if ev.event_id == "turn_1a":
                    turn_1a_replies_ext = adapter.latest_agent_replies()[replies_before_n:]

            tr_entry: dict[str, Any] = {
                "event_id": ev.event_id,
                "kind": ev.kind,
                "trigger_fired_via": trigger_fired_via,
                "updated_files": updated_files,
                "outcome": out,
            }
            if user_simulator_trace.get("active") and ev.event_id == "turn_1c":
                tr_entry["user_simulator"] = user_simulator_trace
            event_trace.append(tr_entry)

        await adapter.wait_until_quiescent(timeout_sec=480.0)
        wall_sec = round(time.perf_counter() - t0, 2)
        ended_at = time.time()

        artifacts = await adapter.collect_runtime_artifacts(ws)
        artifacts.telemetry.setdefault("wall_clock_sec", wall_sec)
        artifacts.telemetry["mutated_before_overturn"] = mutated_before_overturn
        artifacts.telemetry["user_simulator"] = user_simulator_trace if user_simulator_trace.get("active") else None
        exec_blobs = [str(x) for x in (artifacts.telemetry.pop("executed_text_blobs", None) or [])]
        if "held_out_contamination" not in artifacts.telemetry:
            artifacts.telemetry["held_out_contamination"] = _scan_executed_text_contamination(exec_blobs)
        peeks = [float(x) for x in (artifacts.telemetry.get("peek_latencies_ms") or [])]
        mean_peek_ext = round((sum(peeks) / len(peeks)) / 1000.0, 3) if peeks else None
        artifacts.telemetry.setdefault("mean_nonblocking_peek_sec", mean_peek_ext)
        artifacts.telemetry.setdefault("mean_peek_latency_sec", mean_peek_ext)

        eval_report = await asyncio.to_thread(evaluate_episode_verifiers, episode, ws, artifacts)
        forensic_diffs = await asyncio.to_thread(
            _collect_forensic_diffs,
            episode,
            ws,
            artifacts.task_worktrees,
            artifacts.all_worktrees,
        )
        forensic_fp = await asyncio.to_thread(
            _write_forensic_bundle,
            run_root,
            episode,
            arm_name,
            {
                "episode_id": episode.episode_id,
                "track_id": episode.track_id,
                "arm_name": arm_name,
                "git_head": _current_git_head(),
                "started_at": started_at,
                "ended_at": ended_at,
                "agent_diffs": forensic_diffs,
                "agent_replies": artifacts.agent_replies,
                "event_trace": event_trace,
                "telemetry": artifacts.telemetry,
                "verifiers": eval_report.get("verifiers", []),
            },
            forensic_root=forensic_root,
        )
        artifacts.telemetry["forensic_bundle"] = str(forensic_fp)
        return {
            **eval_report,
            "domain": episode.domain,
            "scenario_type": episode.scenario_type,
            "cluster_key": _episode_cluster_key(episode),
            "has_conflict_pair": _episode_has_conflict_pair(episode),
            "episode_sha256": episode_fingerprint(episode),
            "evaluator_version": EVALUATOR_SCHEMA_VERSION,
            "model": os.environ.get("PA_LLM_MODEL", "deepseek-flash"),
            "gaia_judge_mode": gaia_judge_mode,
            "telemetry": artifacts.telemetry,
            "routing_events": artifacts.routing_events,
            "event_trace": event_trace,
        }
    finally:
        await adapter.teardown()
        ws.cleanup(remove_files=not keep_workspace)


class DSHParadigmHarnessAdapter(AgentHarnessAdapter):
    """Controlled upper-layer paradigm adapter (`dsh_mono`, `dsh_subagent_fork`, `dsh_agent_team`) for `PA-150`."""

    def __init__(self, arm_name: str, *, dsh_port: int = 0) -> None:
        self.arm_name = arm_name
        self.dsh_port = dsh_port
        self.driver: DSHParadigmDriver | None = None
        self._episode: EpisodeSpec | None = None
        self._peek_latencies_ms: list[float] = []

    async def setup(self, episode: EpisodeSpec, ws: MaterializedEpisodeWorkspace) -> None:
        self._episode = episode
        docs_dir = ws.root_dir / "feishu_docs"
        docs_dir.mkdir(parents=True, exist_ok=True)
        repo_docs_link = ws.repo_dir / "feishu_docs"
        if not repo_docs_link.exists() and not repo_docs_link.is_symlink():
            repo_docs_link.symlink_to(docs_dir.resolve())
        if ws.shared_links_dir is not None and ws.shared_links_dir.exists():
            sl_docs_link = ws.shared_links_dir / "feishu_docs"
            if not sl_docs_link.exists() and not sl_docs_link.is_symlink():
                sl_docs_link.symlink_to(docs_dir.resolve())

        bin_dir = ws.root_dir / "bin"
        bin_dir.mkdir(parents=True, exist_ok=True)
        for rel_name, (fcontent, fmode) in (ws.extra_repo_files or {}).items():
            if "/" not in rel_name and (fmode & 0o111):
                bp = bin_dir / rel_name
                bp.write_text(fcontent, encoding="utf-8")
                bp.chmod(fmode)
        cur_path = ws.extra_env.get("PATH") or os.environ.get("PATH", "")
        path_parts = [p for p in cur_path.split(os.pathsep) if p]
        if str(bin_dir) not in path_parts:
            ws.extra_env["PATH"] = os.pathsep.join([str(bin_dir), *path_parts])

        has_code_mount = any(
            m.pillar in ("cooperbench", "swe_interact", "ambig_swe") for m in episode.workspace_mounts
        )
        self.driver = DSHParadigmDriver(
            self.arm_name,
            instance_dir=ws.root_dir / "dsh_paradigm_instance",
            allowed_workspace_root=ws.root_dir,
            repo_dir=ws.repo_dir,
            worktrees_dir=(ws.root_dir / "worktrees") if has_code_mount else None,
            shared_links_dir=ws.shared_links_dir or (ws.root_dir / "shared_links"),
            agents_md=ws.agents_md,
            has_code_repo=has_code_mount,
            extra_env=ws.extra_env,
            dsh_port=self.dsh_port,
        )
        await self.driver.start()

    def count_executed_tool_calls(self) -> int:
        return self.driver.total_tool_calls if self.driver is not None else 0

    def latest_agent_replies(self) -> list[str]:
        return list(self.driver.primary_assistant_messages) if self.driver is not None else []

    async def deliver_event(self, event: TimelineEventSpec, ws: MaterializedEpisodeWorkspace) -> dict[str, Any]:
        _ = ws
        assert self.driver is not None
        is_peek = event.expected_mmu_behavior in _PEEK_BEHAVIORS
        out = await self.driver.deliver_inbound_event(
            event_id=event.event_id,
            kind=event.kind,
            channel=event.channel,
            sender=event.sender,
            text=event.text,
            target_task_tag=event.target_task_tag,
            wait_for_reply=is_peek,
            reply_timeout_sec=120.0,
        )
        if is_peek:
            self._peek_latencies_ms.append(float(out.get("elapsed_ms") or 0.0))
        return out

    async def wait_until_quiescent(self, timeout_sec: float = 360.0) -> None:
        if self.driver is not None:
            await self.driver.wait_until_quiescent(timeout_sec=timeout_sec, finalize_team_merges=False)

    async def collect_runtime_artifacts(self, ws: MaterializedEpisodeWorkspace) -> RuntimeArtifacts:
        assert self.driver is not None and self._episode is not None
        episode = self._episode
        await self.driver.wait_until_quiescent(timeout_sec=180.0, finalize_team_merges=True)
        await self.driver.stop()

        dsh_bundles = await asyncio.to_thread(self.driver.host._iter_decompressed_sessions)
        ep_repo_name = next(
            (
                str(m.repo_or_category)
                for m in episode.workspace_mounts
                if m.pillar in ("cooperbench", "swe_interact", "ambig_swe") and "/" in str(m.repo_or_category or "")
            ),
            "workspace/project",
        )
        seatbelt_on = (self.driver.host.instance_dir / "dsh_seatbelt.sb").is_file()
        contaminated, contam_blocked = _scan_dsh_session_contamination(
            dsh_bundles,
            eff_repo=ep_repo_name,
            seatbelt_enforced=seatbelt_on,
        )

        office_mount_inspected = False
        office_spec_doc_read = False
        spec_codes: set[str] = set()
        for _sid, s_events in dsh_bundles:
            for s_ev in s_events:
                if str(s_ev.get("type") or "") == "tool/result":
                    s_txt = json.dumps(s_ev.get("data") or {}, ensure_ascii=False)
                    if any(tok in s_txt for tok in _OFFICE_INSPECTION_MARKERS):
                        office_mount_inspected = True
                    found_codes = _SPEC_REF_CODE_RE.findall(s_txt)
                    if found_codes:
                        spec_codes.update(found_codes)
                        office_mount_inspected = True
                        office_spec_doc_read = True
                    elif "release_spec_addendum" in s_txt or "release_spec_addenda" in s_txt:
                        office_spec_doc_read = True

        base_commit_ep = next((m.base_commit for m in episode.workspace_mounts if m.base_commit), "") or ""
        task_wts, all_wts = self.driver.collect_worktrees(base_commit_ep)
        hyg = self.driver.compute_hygiene_attribution()
        tok = self.driver.compute_token_and_cost_telemetry()
        replies = list(self.driver.primary_assistant_messages)
        if not replies and self.driver.all_assistant_messages:
            replies = [txt for _s, txt in self.driver.all_assistant_messages[-3:] if txt]

        telemetry: dict[str, Any] = {
            "worker_backend": "dsh_agent",
            "paradigm": self.arm_name,
            "worktrees_created": len(all_wts),
            "spawned_workers_count": len(self.driver.spawned_workers),
            "dsh_worker_sessions": tok["session_count"],
            "llm_calls": tok["total_turns"],
            "noise_events_total": hyg["noise_total"],
            "noise_dropped": hyg["noise_dropped"],
            "noise_spawned_workers": hyg["noise_spawned_workers"],
            "noise_mutated_workers": hyg["noise_mutated_workers"],
            "duplicates_merged": hyg["duplicates_merged"],
            "duplicate_spawned_workers": hyg["duplicate_spawned_workers"],
            "real_tasks_dropped": hyg["real_tasks_dropped"],
            "office_mount_inspected": office_mount_inspected,
            "office_spec_doc_read": office_spec_doc_read,
            "tool_read_spec_ref_codes": sorted(spec_codes),
            "peek_latencies_ms": list(self._peek_latencies_ms),
            "system2_prompt_tokens": tok["total_prompt_tokens"],
            "system2_output_tokens": tok["output_tokens"],
            "system2_cost_cny": tok["estimated_cost_cny"],
            "system1_prompt_tokens": 0,
            "system1_output_tokens": 0,
            "system1_cost_cny": 0.0,
            "total_prompt_tokens": tok["total_prompt_tokens"],
            "cache_hit_tokens": tok["cache_hit_tokens"],
            "cache_miss_tokens": tok["cache_miss_tokens"],
            "output_tokens": tok["output_tokens"],
            "prefix_cache_hit_rate": tok["prefix_cache_hit_rate"],
            "overall_prompt_cache_hit_rate": tok["prefix_cache_hit_rate"],
            "estimated_cost_cny": tok["estimated_cost_cny"],
            "held_out_contamination": bool(contaminated),
            "held_out_contamination_blocked": contam_blocked,
            "paradigm_routing_trace": dict(self.driver.routing_events),
            "team_board_snapshot": self.driver.team_board_state if self.arm_name == "dsh_agent_team" else None,
        }
        return RuntimeArtifacts(
            arm_name=self.arm_name,
            task_worktrees=task_wts,
            all_worktrees=all_wts,
            integration_repo_dir=ws.repo_dir,
            agent_replies=replies,
            gaia_task_replies=replies if ws.gaia_mount_id else [],
            routing_events={},
            telemetry=telemetry,
        )

    async def teardown(self) -> None:
        if self.driver is not None:
            await self.driver.stop()


# ---------------------------------------------------------------------------
# Summary Aggregation (3 Orthogonal Tiers), LPT Scheduler & CLI Entrypoint
# ---------------------------------------------------------------------------

_LPT_TRACK_PRIORITY: dict[str, int] = {
    "H-4": 0,
    "H-1": 1,
    "O-1": 2,
    "H-3": 3,
    "H-2": 4,
    "C-2": 5,
    "C-1": 6,
    "O-2": 7,
    "C-3": 8,
    "O-3": 9,
}


def sort_jobs_lpt(jobs: list[tuple[EpisodeSpec, str]]) -> list[tuple[EpisodeSpec, str]]:
    """Order `(episode, arm)` jobs Longest-Processing-Time-First (LPT) to avoid worker-pool tail stragglers."""
    return sorted(
        jobs,
        key=lambda item: (
            0 if item[1] != "synaptic_pa" else 1,
            _LPT_TRACK_PRIORITY.get(item[0].track_id, 50),
            item[0].episode_id,
        ),
    )


def _safe_rate(passed: int, total: int) -> float | None:
    """Return `round(passed / total, 4)` when `total > 0`, or `None` when `total == 0` (`Known-12`)."""
    return round(passed / total, 4) if total > 0 else None


def _compute_cluster_robust_se(items: list[dict[str, Any]]) -> tuple[float | None, int]:
    """Compute Liang-Zeger / Cameron-Gelbach-Miller cluster-robust SE on `episode_passed` grouped by `cluster_key`."""
    n = len(items)
    if n == 0:
        return None, 0
    y_bar = sum(1.0 if r.get("episode_passed") else 0.0 for r in items) / n
    clusters: dict[str, list[float]] = defaultdict(list)
    for r in items:
        ck = str(r.get("cluster_key") or r.get("episode_id") or "")
        clusters[ck].append((1.0 if r.get("episode_passed") else 0.0) - y_bar)
    g_cnt = len(clusters)
    if g_cnt <= 1 or n <= 1:
        return 0.0, g_cnt
    sum_sq = sum((sum(resids) / n) ** 2 for resids in clusters.values())
    se = math.sqrt((g_cnt / (g_cnt - 1.0)) * sum_sq)
    return round(se, 4), g_cnt


def _parse_arms_arg(raw_arms: str) -> list[tuple[str, str | None]]:
    """Parse `--arms` into `[(arm_name, cli_cmd_or_none), ...]`.

    Supports built-in arms (`clonact`, `dsh_superagent`) and arbitrary black-box CLI adapters
    (`cli:<arm_name>:<shell_command>`), separating multiple arms on commas that precede known arm prefixes.
    """
    s = (raw_arms or "").strip()
    if not s:
        return []
    arm_prefix_alt = "|".join(re.escape(a) for a in sorted(BUILTIN_ARMS, key=len, reverse=True))
    tokens = [
        tok.strip()
        for tok in re.split(rf",(?=\s*(?:{arm_prefix_alt}|cli:))", s)
        if tok.strip()
    ]
    out: list[tuple[str, str | None]] = []
    for tok in tokens:
        cli_spec = parse_cli_arm_spec(tok)
        if cli_spec is not None:
            out.append((cli_spec[0], cli_spec[1]))
        elif tok in BUILTIN_ARMS:
            out.append((tok, None))
        else:
            raise ValueError(
                f"Unknown arm {tok!r}; expected one of {BUILTIN_ARMS} or `cli:<arm_name>:<shell_command>`"
            )
    return out


def aggregate_benchmark_results(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute 3-tier orthogonal metrics plus diagnostic & continuous hygiene summaries by arm."""
    # Deduplicate by `(episode_id, arm_name)` (latest record wins) so retried `runner_error` rows never double-count
    deduped_map: dict[tuple[str, str], dict[str, Any]] = {}
    for rec in records:
        deduped_map[(str(rec["episode_id"]), str(rec["arm_name"]))] = rec
    clean_records = list(deduped_map.values())

    by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in clean_records:
        by_arm[str(rec["arm_name"])].append(rec)

    arm_non_error_sets = [
        {str(r["episode_id"]) for r in items if not r.get("runner_error")}
        for items in by_arm.values()
    ]
    common_completed = sorted(set.intersection(*arm_non_error_sets)) if len(arm_non_error_sets) >= 2 else []

    summary_arms: dict[str, Any] = {}
    for arm, items in by_arm.items():
        n = len(items)
        runner_errs = sum(1 for r in items if r.get("runner_error"))
        ep_passed = sum(1 for r in items if r.get("episode_passed"))
        v_passed = sum(int(r.get("verifiers_passed", 0)) for r in items)
        v_total = sum(int(r.get("verifiers_total", 0)) for r in items)

        # Tier 1: Orchestration & Safety
        orch_passed = sum(int(r.get("orchestration_passed", 0)) for r in items)
        orch_total = sum(int(r.get("orchestration_total", 0)) for r in items)
        all_verifiers = [v for r in items for v in (r.get("verifiers") or [])]
        ro_vs = [v for v in all_verifiers if v.get("kind") == "readonly_zero_mutation_check"]
        ro_wt_vs = [v for v in ro_vs if v.get("worktree_budget_ok") is not None]
        hyg_vs = [v for v in all_verifiers if v.get("kind") == "noise_and_dedup_telemetry_check"]
        sim_vs = [v for v in all_verifiers if v.get("kind") == "similar_cell_routing_check"]
        cm_vs = [v for v in all_verifiers if v.get("kind") == "cross_modal_consistency_check"]
        ov_vs = [v for v in all_verifiers if v.get("kind") == "overturn_rollback_check"]
        ep_cluster_se, cluster_count = _compute_cluster_robust_se(items)

        tot_noise_events = 0
        tot_noise_dropped = 0
        for r in items:
            tel_r = r.get("telemetry") or {}
            hyg_r = next(
                (v for v in (r.get("verifiers") or []) if v.get("kind") == "noise_and_dedup_telemetry_check"),
                None,
            )
            if tel_r.get("noise_events_total") is not None:
                ep_n_tot = int(tel_r["noise_events_total"])
            elif hyg_r is not None and hyg_r.get("expected_noise_events") is not None:
                ep_n_tot = int(hyg_r["expected_noise_events"])
            else:
                ep_n_tot = 0
            if ep_n_tot > 0:
                obs_n_drop = tel_r.get("noise_dropped")
                if obs_n_drop is None and hyg_r is not None:
                    obs_n_drop = (hyg_r.get("observed_telemetry") or {}).get("noise_dropped", 0)
                tot_noise_events += ep_n_tot
                tot_noise_dropped += min(ep_n_tot, max(0, int(obs_n_drop or 0)))

        # Tier 2: Upstream Task Solve + Non-Gating Diagnostic Verifiers (`zero_shot_conflict_free_merge_rate`)
        solve_passed = sum(int(r.get("task_solve_passed", 0)) for r in items)
        solve_total = sum(int(r.get("task_solve_total", 0)) for r in items)
        all_tasks_solved_eps = sum(
            1
            for r in items
            if int(r.get("task_solve_total", 0)) > 0
            and int(r.get("task_solve_passed", 0)) == int(r.get("task_solve_total", 0))
        )
        solve_ep_denom = sum(1 for r in items if int(r.get("task_solve_total", 0)) > 0)
        diag_vs = [v for v in all_verifiers if v.get("tier") == "diagnostic" or v.get("diagnostic_only")]
        cb_merge_vs = [v for v in all_verifiers if v.get("kind") == "cooperbench_merged_pytest"]
        zero_shot_merge_rate = _safe_rate(
            sum(1 for v in cb_merge_vs if v.get("passed")),
            len(cb_merge_vs),
        )
        c2_items = [r for r in items if r.get("track_id") == "C-2"]
        c2_iso_both = 0
        c2_clean_m_pass = 0
        c2_clean_m_tot = 0
        c2_conf_m_pass = 0
        c2_conf_m_tot = 0
        for r_c2 in c2_items:
            vs_c2 = r_c2.get("verifiers") or []
            singles_c2 = [v for v in vs_c2 if v.get("kind") == "cooperbench_pytest"]
            merged_c2 = next((v for v in vs_c2 if v.get("kind") == "cooperbench_merged_pytest"), None)
            if len(singles_c2) >= 2 and all(v.get("passed") for v in singles_c2):
                c2_iso_both += 1
            if merged_c2 is not None:
                if r_c2.get("has_conflict_pair") is True:
                    c2_conf_m_tot += 1
                    if merged_c2.get("passed"):
                        c2_conf_m_pass += 1
                elif r_c2.get("has_conflict_pair") is False:
                    c2_clean_m_tot += 1
                    if merged_c2.get("passed"):
                        c2_clean_m_pass += 1

        # Tier 3: Efficiency & Cost (`A6`: align System-2 `prefix_cache_hit_rate` + `overall_prompt_cache_hit_rate`)
        tot_prompt = sum(int(r.get("telemetry", {}).get("total_prompt_tokens", 0)) for r in items)
        tot_s2_prompt = sum(
            int(
                r.get("telemetry", {}).get(
                    "system2_prompt_tokens",
                    r.get("telemetry", {}).get("total_prompt_tokens", 0),
                )
            )
            for r in items
        )
        tot_hit = sum(int(r.get("telemetry", {}).get("cache_hit_tokens", 0)) for r in items)
        tot_out = sum(int(r.get("telemetry", {}).get("output_tokens", 0)) for r in items)
        tot_wall = sum(float(r.get("telemetry", {}).get("wall_clock_sec", 0.0)) for r in items)
        tot_wt = sum(int(r.get("telemetry", {}).get("worktrees_created", 0)) for r in items)
        tot_cny = sum(float(r.get("telemetry", {}).get("estimated_cost_cny", 0.0)) for r in items)
        all_peeks = [
            float(ms)
            for r in items
            for ms in (r.get("telemetry", {}).get("peek_latencies_ms") or [])
        ]
        mean_peek_s = round((sum(all_peeks) / len(all_peeks)) / 1000.0, 3) if all_peeks else None
        aligned_prefix_hit_rate = round(tot_hit / tot_s2_prompt, 4) if tot_s2_prompt else 0.0
        overall_cache_hit_rate = round(tot_hit / tot_prompt, 4) if tot_prompt else 0.0

        per_domain: dict[str, Any] = {}
        for dom in ("office", "coding", "hybrid"):
            sub = [r for r in items if r.get("domain") == dom]
            if sub:
                dom_solve_p = sum(int(r.get("task_solve_passed", 0)) for r in sub)
                dom_solve_t = sum(int(r.get("task_solve_total", 0)) for r in sub)
                dom_orch_p = sum(int(r.get("orchestration_passed", 0)) for r in sub)
                dom_orch_t = sum(int(r.get("orchestration_total", 0)) for r in sub)
                dom_v_p = sum(int(r.get("verifiers_passed", 0)) for r in sub)
                dom_v_t = sum(int(r.get("verifiers_total", 0)) for r in sub)
                per_domain[dom] = {
                    "episodes": len(sub),
                    "episode_pass_rate": _safe_rate(sum(1 for r in sub if r.get("episode_passed")), len(sub)),
                    "task_solve_pass_rate": _safe_rate(dom_solve_p, dom_solve_t),
                    "orchestration_pass_rate": _safe_rate(dom_orch_p, dom_orch_t),
                    "verifier_pass_rate": _safe_rate(dom_v_p, dom_v_t),
                }

        per_track: dict[str, Any] = {}
        for trk in sorted({str(r["track_id"]) for r in items}):
            sub = [r for r in items if r["track_id"] == trk]
            trk_solve_p = sum(int(r.get("task_solve_passed", 0)) for r in sub)
            trk_solve_t = sum(int(r.get("task_solve_total", 0)) for r in sub)
            trk_orch_p = sum(int(r.get("orchestration_passed", 0)) for r in sub)
            trk_orch_t = sum(int(r.get("orchestration_total", 0)) for r in sub)
            trk_v_p = sum(int(r.get("verifiers_passed", 0)) for r in sub)
            trk_v_t = sum(int(r.get("verifiers_total", 0)) for r in sub)
            per_track[trk] = {
                "episodes": len(sub),
                "passed": sum(1 for r in sub if r.get("episode_passed")),
                "episode_pass_rate": _safe_rate(sum(1 for r in sub if r.get("episode_passed")), len(sub)),
                "task_solve_pass_rate": _safe_rate(trk_solve_p, trk_solve_t),
                "orchestration_pass_rate": _safe_rate(trk_orch_p, trk_orch_t),
                "verifier_pass_rate": _safe_rate(trk_v_p, trk_v_t),
            }

        summary_arms[arm] = {
            "episodes_evaluated": n,
            "runner_errors": runner_errs,
            "episodes_passed": ep_passed,
            "episode_pass_rate": _safe_rate(ep_passed, n),
            "episode_pass_rate_cluster_se": ep_cluster_se,
            "cluster_count": cluster_count,
            "verifiers_passed": v_passed,
            "verifiers_total": v_total,
            "verifier_pass_rate": _safe_rate(v_passed, v_total),
            "zero_shot_conflict_free_merge_rate": zero_shot_merge_rate,
            "c2_dual_feature_breakdown": {
                "episodes": len(c2_items),
                "isolated_dual_pass_rate": _safe_rate(c2_iso_both, len(c2_items)),
                "integrated_merge_pass_rate": zero_shot_merge_rate,
                "clean_pair_merge_rate": _safe_rate(c2_clean_m_pass, c2_clean_m_tot),
                "conflict_pair_merge_rate": _safe_rate(c2_conf_m_pass, c2_conf_m_tot),
            },
            "diagnostic_verifiers": {
                "passed": sum(1 for v in diag_vs if v.get("passed")),
                "total": len(diag_vs),
                "pass_rate": _safe_rate(sum(1 for v in diag_vs if v.get("passed")), len(diag_vs)),
                "zero_shot_conflict_free_merge_rate": zero_shot_merge_rate,
            },
            "tier1_orchestration_safety": {
                "passed": orch_passed,
                "total": orch_total,
                "pass_rate": _safe_rate(orch_passed, orch_total),
                "readonly_zero_mutation_rate": _safe_rate(
                    sum(1 for v in ro_vs if v.get("passed")), len(ro_vs)
                ),
                "zero_premature_worktree_rate": _safe_rate(
                    sum(1 for v in ro_wt_vs if v.get("worktree_budget_ok")), len(ro_wt_vs)
                ),
                "burst_hygiene_pass_rate": _safe_rate(
                    sum(1 for v in hyg_vs if v.get("passed")), len(hyg_vs)
                ),
                "noise_drop_rate": _safe_rate(tot_noise_dropped, tot_noise_events),
                "total_noise_dropped": tot_noise_dropped,
                "total_noise_events": tot_noise_events,
                "similar_cell_routing_accuracy": _safe_rate(
                    sum(1 for v in sim_vs if v.get("passed")), len(sim_vs)
                ),
                "cross_modal_consistency_rate": _safe_rate(
                    sum(1 for v in cm_vs if v.get("passed")), len(cm_vs)
                ),
                "overturn_clean_rollback_rate": _safe_rate(
                    sum(1 for v in ov_vs if v.get("passed")), len(ov_vs)
                ),
            },
            "tier2_task_solve": {
                "verifiers_passed": solve_passed,
                "verifiers_total": solve_total,
                "verifier_pass_rate": _safe_rate(solve_passed, solve_total),
                "all_tasks_solved_episodes": all_tasks_solved_eps,
                "all_tasks_solved_episode_rate": _safe_rate(all_tasks_solved_eps, solve_ep_denom),
                "zero_shot_conflict_free_merge_rate": zero_shot_merge_rate,
            },
            "tier3_efficiency": {
                "mean_wall_clock_sec": round(tot_wall / n, 2) if n else 0.0,
                "mean_nonblocking_peek_sec": mean_peek_s,
                "mean_peek_latency_sec": mean_peek_s,
                "total_worktrees_created": tot_wt,
                "mean_worktrees_per_episode": round(tot_wt / n, 2) if n else 0.0,
                "total_prompt_tokens": tot_prompt,
                "system2_prompt_tokens": tot_s2_prompt,
                "total_output_tokens": tot_out,
                "prefix_cache_hit_rate": aligned_prefix_hit_rate,
                "overall_prompt_cache_hit_rate": overall_cache_hit_rate,
                "estimated_cost_cny": round(tot_cny, 4),
                "mean_cost_cny_per_episode": round(tot_cny / n, 4) if n else 0.0,
            },
            "mean_wall_clock_sec": round(tot_wall / n, 2) if n else 0.0,
            "mean_nonblocking_peek_sec": mean_peek_s,
            "total_worktrees_created": tot_wt,
            "total_prompt_tokens": tot_prompt,
            "total_output_tokens": tot_out,
            "prefix_cache_hit_rate": aligned_prefix_hit_rate,
            "overall_prompt_cache_hit_rate": overall_cache_hit_rate,
            "estimated_cost_cny": round(tot_cny, 4),
            "by_domain": per_domain,
            "by_track": per_track,
        }
        if PA60_MANIFEST_JSON.is_file():
            try:
                pa60_weights = (json.loads(PA60_MANIFEST_JSON.read_text(encoding="utf-8")).get("ipw_weights") or {})
                if items and all(str(r.get("episode_id")) in pa60_weights for r in items):
                    w_tot = sum(float(pa60_weights[str(r["episode_id"])]) for r in items)
                    w_pass = sum(
                        float(pa60_weights[str(r["episode_id"])])
                        for r in items
                        if r.get("episode_passed")
                    )
                    w_solve_num = sum(
                        float(pa60_weights[str(r["episode_id"])]) * int(r.get("task_solve_passed", 0))
                        for r in items
                    )
                    w_solve_den = sum(
                        float(pa60_weights[str(r["episode_id"])]) * int(r.get("task_solve_total", 0))
                        for r in items
                    )
                    summary_arms[arm]["pa60_ipw_estimates"] = {
                        "weighted_population_size": round(w_tot, 2),
                        "ipw_episode_pass_rate": round(w_pass / w_tot, 4) if w_tot > 0 else None,
                        "ipw_task_solve_pass_rate": round(w_solve_num / w_solve_den, 4) if w_solve_den > 0 else None,
                    }
            except Exception:
                pass

    gaia_judge_model = (
        os.environ.get("GAIA2_JUDGE_MODEL", "").strip()
        or os.environ.get("PA_GAIA_JUDGE_MODEL", "").strip()
        or "deepseek-chat"
    )
    return {
        "evaluator_version": EVALUATOR_SCHEMA_VERSION,
        "gaia_judge_model": gaia_judge_model,
        "total_records": len(clean_records),
        "common_completed_episodes_count": len(common_completed),
        "arms": summary_arms,
    }


async def run_benchmark_suite(args: argparse.Namespace) -> dict[str, Any]:
    from synaptic_bench_150.output_guard import ensure_output_is_not_frozen
    ensure_output_is_not_frozen(args.output_dir)
    load_project_dotenv()
    cli_model = str(getattr(args, "model", "") or "").strip()
    if cli_model:
        os.environ["PA_LLM_MODEL"] = cli_model
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    results_jsonl = out_dir / "results.jsonl"
    summary_json = out_dir / "summary.json"
    # Place ephemeral episode workspaces in `/tmp` outside the repository.
    # Seatbelt rules separately deny access to evaluator data and experiment records.
    work_root = (
        Path(tempfile.gettempdir())
        / f"pa150_ws_{hashlib.sha256(str(out_dir).encode('utf-8')).hexdigest()[:8]}"
    )

    episodes = load_pa150_episodes(
        episode_ids=args.episodes.split(",") if args.episodes else None,
        tracks=args.tracks.split(",") if args.tracks else None,
        domains=args.domains.split(",") if args.domains else None,
        subset=getattr(args, "subset", None),
        limit=args.limit,
    )
    parsed_arms = _parse_arms_arg(args.arms)
    arms = [a_name for a_name, _cmd in parsed_arms]
    cli_arm_commands: dict[str, str] = {a_name: cmd for a_name, cmd in parsed_arms if cmd is not None}

    requested_hashes: dict[tuple[str, str], str] = {
        (ep.episode_id, arm): episode_fingerprint(ep)
        for ep in episodes
        for arm in arms
    }
    allow_stale = bool(getattr(args, "allow_stale_resume", False))
    expected_model = os.environ.get("PA_LLM_MODEL", "deepseek-flash")

    completed_keys: set[tuple[str, str]] = set()
    resumed_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    skipped_resume_count = 0
    if args.resume and results_jsonl.exists():
        with results_jsonl.open(encoding="utf-8") as fp:
            for ln in fp:
                if not ln.strip():
                    continue
                rec = json.loads(ln)
                key = (str(rec.get("episode_id") or ""), str(rec.get("arm_name") or ""))
                if key not in requested_hashes:
                    skipped_resume_count += 1
                    continue
                if rec.get("runner_error"):
                    skipped_resume_count += 1
                    continue
                rec_sha = rec.get("episode_sha256")
                rec_jmode = rec.get("gaia_judge_mode", args.gaia_judge_mode)
                rec_model = rec.get("model")
                if not allow_stale and (
                    rec_sha != requested_hashes[key]
                    or not RELEASE.matches(rec)
                    or rec_jmode != args.gaia_judge_mode
                    or (rec_model and expected_model and rec_model != expected_model)
                ):
                    skipped_resume_count += 1
                    continue
                resumed_by_key[key] = rec
                completed_keys.add(key)
        if skipped_resume_count > 0:
            print(
                f"[Resume] Ignored {skipped_resume_count} out-of-scope, errored, or stale-manifest record(s) in {results_jsonl.name}.",
                flush=True,
            )

    records: list[dict[str, Any]] = list(resumed_by_key.values())

    pending_jobs = sort_jobs_lpt(
        [
            (ep, arm)
            for ep in episodes
            for arm in arms
            if (ep.episode_id, arm) not in completed_keys
        ]
    )
    concurrency = max(1, int(getattr(args, "concurrency", 1) or 1))
    from concurrent.futures import ThreadPoolExecutor

    loop = asyncio.get_running_loop()
    executor = ThreadPoolExecutor(max_workers=max(64, concurrency * 4), thread_name_prefix="pa150_io")
    loop.set_default_executor(executor)
    needs_gaia = any("gaia2" in ep.upstream_pillars for ep in episodes)

    gaia_pool: dict[int, Gaia2ServerProcess | None] = {
        slot: (
            Gaia2ServerProcess(port=19020 + slot, log_path=out_dir / f"gaia2_slot_{slot}.log")
            if needs_gaia
            else None
        )
        for slot in range(concurrency)
    }
    slot_queue: asyncio.Queue[int] = asyncio.Queue()
    for slot in range(concurrency):
        slot_queue.put_nowait(slot)

    start_balance = query_deepseek_balance_cny()
    cached_bal_val: float | None = start_balance
    cached_bal_ts: float = time.monotonic()
    bal_refresh_lock = asyncio.Lock()
    cumulative_est_cny = sum(float(r.get("telemetry", {}).get("estimated_cost_cny", 0.0)) for r in records)
    inflight_jobs = 0
    write_lock = asyncio.Lock()
    pg_cluster_role_lock = asyncio.Lock()
    circuit_tripped = False

    async def _get_cached_balance_cny() -> float | None:
        nonlocal cached_bal_val, cached_bal_ts
        if (time.monotonic() - cached_bal_ts) < 30.0:
            return cached_bal_val
        async with bal_refresh_lock:
            if (time.monotonic() - cached_bal_ts) < 30.0:
                return cached_bal_val
            fresh = await asyncio.to_thread(query_deepseek_balance_cny)
            if fresh is not None:
                cached_bal_val = fresh
            cached_bal_ts = time.monotonic()
            return cached_bal_val

    async def _execute_episode_arm(
        ep: EpisodeSpec,
        arm: str,
        gaia_srv: Gaia2ServerProcess | None,
        gaia_port: int,
        dsh_port: int,
    ) -> dict[str, Any]:
        if arm == "clonact":
            return await run_episode_synaptic_pa(
                ep,
                work_root,
                gaia_server=gaia_srv,
                gaia_port=gaia_port,
                gaia_judge_mode=args.gaia_judge_mode,
                keep_workspace=args.keep_workspaces,
                forensic_root=out_dir,
                arm_label="clonact",
                dsh_port=dsh_port + 500,
            )
        if arm in CLONACT_ABLATION_CONFIGS:
            return await run_episode_synaptic_pa(
                ep,
                work_root,
                gaia_server=gaia_srv,
                gaia_port=gaia_port,
                gaia_judge_mode=args.gaia_judge_mode,
                keep_workspace=args.keep_workspaces,
                forensic_root=out_dir,
                arm_label=arm,
                dsh_port=dsh_port + 500,
                ablation_config=CLONACT_ABLATION_CONFIGS[arm],
            )
        if arm in PARADIGM_BASELINE_ARMS:
            p_adapter = DSHParadigmHarnessAdapter(arm, dsh_port=dsh_port + 700)
            return await run_episode_external_adapter(
                ep,
                work_root,
                p_adapter,
                gaia_server=gaia_srv,
                gaia_port=gaia_port,
                gaia_judge_mode=args.gaia_judge_mode,
                keep_workspace=args.keep_workspaces,
                forensic_root=out_dir,
            )
        if arm in cli_arm_commands:
            cli_adapter = SubprocessCLIHarnessAdapter(arm_name=arm, command_template=cli_arm_commands[arm])
            return await run_episode_external_adapter(
                ep,
                work_root,
                cli_adapter,
                gaia_server=gaia_srv,
                gaia_port=gaia_port,
                gaia_judge_mode=args.gaia_judge_mode,
                keep_workspace=args.keep_workspaces,
                forensic_root=out_dir,
            )
        return await run_episode_dsh_superagent(
            ep,
            work_root,
            dsh_port=dsh_port,
            gaia_server=gaia_srv,
            gaia_port=gaia_port,
            gaia_judge_mode=args.gaia_judge_mode,
            keep_workspace=args.keep_workspaces,
            forensic_root=out_dir,
        )

    async def _worker_run_one(ep: EpisodeSpec, arm: str) -> None:
        nonlocal cumulative_est_cny, inflight_jobs, circuit_tripped
        slot = await slot_queue.get()
        job_t0 = time.perf_counter()
        acquired_inflight = False
        try:
            cur_bal = await _get_cached_balance_cny()
            async with write_lock:
                if circuit_tripped:
                    return
                low_balance_floor = (
                    expected_model.lower().startswith("deepseek")
                    and cur_bal is not None
                    and cur_bal <= 1.0
                )
                reserved_cny = inflight_jobs * 0.35
                over_budget = (
                    args.max_spend_cny is not None
                    and args.max_spend_cny > 0
                    and (cumulative_est_cny + reserved_cny) >= args.max_spend_cny
                )
                if over_budget or low_balance_floor:
                    circuit_tripped = True
                    print(
                        f"[CircuitBreaker] Reached spend/balance guard (est_cny={cumulative_est_cny:.2f}, "
                        f"reserved={reserved_cny:.2f}, max={args.max_spend_cny}, cur_bal={cur_bal}). Halting cleanly.",
                        flush=True,
                    )
                    return
                inflight_jobs += 1
                acquired_inflight = True

            gaia_srv = gaia_pool[slot]
            gaia_port = 19020 + slot
            dsh_port = int(args.dsh_port) + slot
            try:
                # NEW-11 / P0-4: Serialize & scrub cluster-global PG roles across the 4 PG security tasks
                if episode_touches_pg_cluster_roles(ep):
                    async with pg_cluster_role_lock:
                        if circuit_tripped:
                            return
                        await asyncio.to_thread(cleanup_postgres_cluster_roles)
                        try:
                            rec = await _execute_episode_arm(ep, arm, gaia_srv, gaia_port, dsh_port)
                        finally:
                            await asyncio.to_thread(cleanup_postgres_cluster_roles)
                else:
                    rec = await _execute_episode_arm(ep, arm, gaia_srv, gaia_port, dsh_port)
            except Exception as exc:
                if gaia_srv is not None:
                    await asyncio.to_thread(gaia_srv.stop)
                err_wall = time.perf_counter() - job_t0
                tb_tail = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)[-3:]).strip()
                print(
                    f"[RunnerError] [{ep.episode_id} | {ep.track_id} | {arm}] {type(exc).__name__}: {exc}\n{tb_tail}",
                    flush=True,
                )
                rec = build_runner_error_record(
                    ep,
                    arm,
                    exc,
                    wall_clock_sec=err_wall,
                    gaia_judge_mode=args.gaia_judge_mode,
                )

            rec = RELEASE.stamp(rec)
            async with write_lock:
                records.append(rec)
                cumulative_est_cny += float(rec.get("telemetry", {}).get("estimated_cost_cny", 0.0))
                with results_jsonl.open("a", encoding="utf-8") as fp:
                    fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
                summary = RELEASE.stamp(aggregate_benchmark_results(records))
                if start_balance is not None:
                    summary["start_account_balance_cny"] = start_balance
                summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
                err_tag = f" ERROR={rec['runner_error'][:80]}" if rec.get("runner_error") else ""
                print(
                    f"[{ep.episode_id} | {ep.track_id} | {arm}] "
                    f"ep_pass={rec['episode_passed']} "
                    f"solve={rec['task_solve_passed']}/{rec['task_solve_total']} "
                    f"orch={rec['orchestration_passed']}/{rec['orchestration_total']} "
                    f"wall={rec['telemetry']['wall_clock_sec']}s est_cny={rec['telemetry']['estimated_cost_cny']}{err_tag}",
                    flush=True,
                )
        finally:
            if acquired_inflight:
                async with write_lock:
                    inflight_jobs = max(0, inflight_jobs - 1)
            slot_queue.put_nowait(slot)

    try:
        if concurrency == 1:
            for ep, arm in pending_jobs:
                if circuit_tripped:
                    break
                await _worker_run_one(ep, arm)
        else:
            await asyncio.gather(*(_worker_run_one(ep, arm) for ep, arm in pending_jobs))
    finally:
        for srv in gaia_pool.values():
            if srv is not None:
                srv.stop()

    final_summary = RELEASE.stamp(aggregate_benchmark_results(records))
    summary_json.write_text(json.dumps(final_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return final_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified Dual-Arm Runner & Verifier for Synaptic-Bench-150")
    parser.add_argument("--version", action="version", version=f"{RELEASE.name} {RELEASE.version}")
    parser.add_argument("--episodes", default="", help="Comma-separated episode IDs (e.g. PA150_O1_01,PA150_C2_01)")
    parser.add_argument("--tracks", default="", help="Comma-separated track IDs (e.g. O-1,C-2,H-3)")
    parser.add_argument("--domains", default="", help="Comma-separated domains (office,coding,hybrid)")
    parser.add_argument("--subset", default="all", choices=["all", "pa60"], help="Dataset subset (`all` or pre-registered `pa60`)")
    parser.add_argument("--limit", type=int, default=None, help="Max number of episodes to run")
    parser.add_argument(
        "--arms",
        default="clonact,dsh_superagent",
        help="Comma-separated arms (`clonact`, `dsh_superagent`, or `cli:<name>:<cmd>`)",
    )
    parser.add_argument("--concurrency", type=int, default=1, help="Number of parallel episode/arm slots")
    parser.add_argument("--model", default="", help="Backbone LLM (`deepseek-flash`, `gpt-6-luna`, `qwen3.8-flash`)")
    parser.add_argument("--output-dir", default=str(DEFAULT_RUNS_ROOT / "latest"))
    parser.add_argument("--max-spend-cny", type=float, default=60.0, help="Hard CNY spend circuit breaker")
    parser.add_argument("--gaia-judge-mode", default="auto", choices=["auto", "deterministic"])
    parser.add_argument("--dsh-port", type=int, default=13082)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--allow-stale-resume", action="store_true", default=False)
    parser.add_argument("--keep-workspaces", action="store_true", default=False)
    args = parser.parse_args()
    summary = asyncio.run(run_benchmark_suite(args))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
