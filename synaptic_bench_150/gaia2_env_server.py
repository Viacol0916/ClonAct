#!/usr/bin/env python3
"""Isolated HTTP bridge & official `GraphPerEventJudge` server for Meta ARE (`Gaia2`).

Runs inside `.venv_are/bin/python` so `meta-agents-research-environments==1.2.0`
stays cleanly decoupled from `.venv` (`typesafe-sdk`) and `dsh-superagent` (Node.js).
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import shutil
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Iterator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
UPSTREAM_ROOT = PROJECT_ROOT / "data" / "upstream_pillars"
DEMO_FS_DIR = UPSTREAM_ROOT / "gaia2_demo_fs"
RAW_CACHE_DIR = UPSTREAM_ROOT / "gaia2_raw_scenarios"
SAMPLES_JSON = UPSTREAM_ROOT / "samples" / "gaia2_samples.json"
COMPACT_POOL_JSONL = UPSTREAM_ROOT / "pools" / "gaia2_compact_pool.jsonl"

# Must be set prior to importing `are.simulation` to prevent remote GitHub/HF hangs on boot
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")


def _ensure_demo_fs_populated() -> None:
    """Ensure `DEMO_FS_DIR` is populated from `meta-agents-research-environments/gaia2_filesystem` (`Item 2`)."""
    DEMO_FS_DIR.mkdir(parents=True, exist_ok=True)
    if (DEMO_FS_DIR / "Documents").is_dir() and any((DEMO_FS_DIR / "Documents").rglob("*")):
        return
    try:
        from huggingface_hub import snapshot_download

        snap_dir = Path(
            snapshot_download(
                repo_id="meta-agents-research-environments/gaia2_filesystem",
                repo_type="dataset",
            )
        )
        src_root = snap_dir / "demo_filesystem" if (snap_dir / "demo_filesystem").is_dir() else snap_dir
        for item in src_root.iterdir():
            if item.name.startswith("."):
                continue
            dst = DEMO_FS_DIR / item.name
            if item.is_dir():
                shutil.copytree(item, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dst)
    except Exception:
        pass


_ensure_demo_fs_populated()
os.environ.setdefault("DEMO_FS_PATH", str(DEMO_FS_DIR))

from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402
import uvicorn  # noqa: E402

from are.simulation.data_handler.importer import JsonScenarioImporter  # noqa: E402
from are.simulation.environment import Environment, EnvironmentConfig  # noqa: E402
from are.simulation.scenarios.scenario_imported_from_json.utils import preprocess_scenario  # noqa: E402
from are.simulation.types import EventRegisterer, EventType, OracleEvent  # noqa: E402
from are.simulation.utils import make_serializable  # noqa: E402
from are.simulation.validation.configs import GraphPerEventJudgeConfig  # noqa: E402


class StartScenarioReq(BaseModel):
    upstream_id: str = ""
    config: str = ""
    row_id: str = ""
    scenario_id: str = ""
    judge_mode: str = "auto"  # "auto" | "deterministic"
    readonly_side_mount: bool = False


class ToolCallReq(BaseModel):
    tool_name: str
    args: dict[str, Any] = Field(default_factory=dict)


class EvaluateReq(BaseModel):
    agent_replies: list[str] = Field(default_factory=list)


def _resolve_judge_credentials() -> tuple[str, str, str]:
    """Resolve cross-family or default OpenAI-compatible judge credentials (`v2.0` `P2`)."""
    api_key = (
        os.environ.get("GAIA2_JUDGE_API_KEY", "").strip()
        or os.environ.get("PA150_GAIA_JUDGE_API_KEY", "").strip()
        or os.environ.get("DEEPSEEK_API_KEY", "").strip()
    )
    base_url = (
        os.environ.get("GAIA2_JUDGE_BASE_URL", "").strip()
        or os.environ.get("PA150_GAIA_JUDGE_BASE_URL", "").strip()
        or os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip()
    ).rstrip("/")
    model = (
        os.environ.get("GAIA2_JUDGE_MODEL", "").strip()
        or os.environ.get("PA150_GAIA_JUDGE_MODEL", "").strip()
        or "deepseek-chat"
    )
    return api_key, base_url, model


def _create_judge_engine(
    judge_mode: str = "auto",
    judge_errors: list[str] | None = None,
    judge_usage: dict[str, int] | None = None,
):
    api_key, base_url, judge_model = _resolve_judge_credentials()

    def _engine(messages: list[dict[str, Any]], **kwargs: Any) -> tuple[str, dict[str, Any]]:
        if judge_mode == "deterministic":
            if judge_usage is not None:
                judge_usage["deterministic_calls"] = judge_usage.get("deterministic_calls", 0) + 1
            return "[[true]]", {"mode": "deterministic_structural"}
        if not api_key:
            msg = "Missing GAIA2_JUDGE_API_KEY / DEEPSEEK_API_KEY for Gaia2 semantic judge (fail-closed)"
            if judge_errors is not None:
                judge_errors.append(msg)
            return "[[false]]", {"judge_error": msg}
        payload = {
            "model": judge_model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 512,
        }
        last_exc: Exception | None = None
        for attempt in range(3):
            req = urllib.request.Request(
                f"{base_url}/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
                method="POST",
            )
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            try:
                with opener.open(req, timeout=35.0) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                    content = body["choices"][0]["message"]["content"] or ""
                    usage = body.get("usage") or {}
                    if judge_usage is not None:
                        judge_usage["llm_calls"] = judge_usage.get("llm_calls", 0) + 1
                        judge_usage["prompt_tokens"] = judge_usage.get("prompt_tokens", 0) + int(usage.get("prompt_tokens") or 0)
                        judge_usage["completion_tokens"] = judge_usage.get("completion_tokens", 0) + int(usage.get("completion_tokens") or 0)
                    return content, usage
            except Exception as exc:
                last_exc = exc
                time.sleep(0.6 * (attempt + 1))
        err_msg = f"Gaia2 judge API failed after 3 attempts (fail-closed): {last_exc}"
        if judge_errors is not None:
            judge_errors.append(err_msg)
        return "[[false]]", {"judge_error": err_msg}

    return _engine


def _normalize_passive_read_receipts(node: Any) -> Any:
    """Normalize passive read-receipts and user-reply transport state (`AgentUserInterface`) (`NEW-9`, `P0-2`)."""
    if isinstance(node, dict):
        is_mail_or_msg = "is_read" in node and any(
            k in node for k in ("subject", "sender", "recipients", "email_id", "cc", "folder_name", "content", "message_id")
        )
        has_time_read = "time_read" in node and any(
            k in node for k in ("sender", "content", "message_id", "time_created")
        )
        out: dict[str, Any] = {}
        for k, v in node.items():
            if k == "AgentUserInterface":
                continue
            if is_mail_or_msg and k == "is_read":
                out[k] = False
            elif has_time_read and k == "time_read":
                out[k] = None
            elif k == "api_call_count" and isinstance(v, int):
                out[k] = 0
            else:
                out[k] = _normalize_passive_read_receipts(v)
        return out
    if isinstance(node, list):
        return [
            _normalize_passive_read_receipts(x)
            for x in node
            if not (isinstance(x, dict) and x.get("app_name") == "AgentUserInterface")
        ]
    return node


class Gaia2RuntimeHolder:
    """Single-threaded holder around active Meta ARE `Scenario` + `Environment`."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.scenario: Any = None
        self.judge_scenario: Any = None
        self.env: Environment | None = None
        self.tools: dict[str, Any] = {}
        self.apps_index: dict[str, list[dict[str, Any]]] = {}
        self.call_log: list[dict[str, Any]] = []
        self.buffered_aui_messages: list[str] = []
        self._flushed_aui_count: int = 0
        self.mutating_call_count: int = 0
        self.aui_call_count: int = 0
        self.initial_state_hash: str = ""
        self.initial_raw_state_hash: str = ""
        self.judge_mode: str = "auto"
        self.judge_model: str = "deepseek-chat"
        self.readonly_side_mount: bool = False
        self.diag_token: str = os.environ.get("PA150_GAIA_DIAG_SECRET", "").strip()
        self.judge_errors: list[str] = []
        self.judge_usage: dict[str, int] = {}
        self._samples_index: dict[tuple[str, str], str] | None = None
        self._pool_offsets: dict[tuple[str, str], int] | None = None

    @staticmethod
    def _pause_env_clock(env: Environment) -> None:
        """Pause `env.time_manager` while idle between HTTP calls so LLM thinking wall-clock never drifts simulated time."""
        env.time_manager.pause()
        # Guard against upstream `TimeManager.time_passed()` falsy check `if self.is_paused and self.pause_passed_time:`
        if not env.time_manager.pause_passed_time:
            env.time_manager.pause_passed_time = 1e-6

    @contextlib.contextmanager
    def _active_env_clock(self, env: Environment) -> Iterator[None]:
        """Resume `env.time_manager` strictly during active scenario/tool/tick execution."""
        env.time_manager.resume()
        try:
            yield
        finally:
            self._pause_env_clock(env)

    def _ensure_indices(self) -> None:
        if self._samples_index is None:
            idx: dict[tuple[str, str], str] = {}
            if SAMPLES_JSON.exists():
                raw = json.loads(SAMPLES_JSON.read_text(encoding="utf-8"))
                for cfg, rows in raw.items():
                    for r in rows:
                        data_str = r["data"] if isinstance(r.get("data"), str) else json.dumps(r["data"])
                        idx[(cfg, str(r["id"]))] = data_str
                        if r.get("scenario_id"):
                            idx[(cfg, str(r["scenario_id"]))] = data_str
            self._samples_index = idx

        if self._pool_offsets is None:
            offsets: dict[tuple[str, str], int] = {}
            per_cfg_counter: dict[str, int] = {}
            if COMPACT_POOL_JSONL.exists():
                with COMPACT_POOL_JSONL.open(encoding="utf-8") as fp:
                    for line in fp:
                        line = line.strip()
                        if not line:
                            continue
                        r = json.loads(line)
                        cfg = str(r["config"])
                        pos = per_cfg_counter.get(cfg, 0)
                        per_cfg_counter[cfg] = pos + 1
                        offsets[(cfg, str(r["id"]))] = pos
                        offsets[(cfg, str(r["scenario_id"]))] = pos
            self._pool_offsets = offsets

    def resolve_scenario_json(self, upstream_id: str, config: str, row_id: str, scenario_id: str) -> str:
        self._ensure_indices()
        assert self._samples_index is not None and self._pool_offsets is not None

        if upstream_id.startswith("gaia2::"):
            parts = upstream_id.split("::")
            if len(parts) >= 3:
                config = config or parts[1]
                row_id = row_id or parts[2]

        RAW_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        for key_suffix in (row_id, scenario_id):
            if key_suffix:
                cache_fp = RAW_CACHE_DIR / f"{config}__{key_suffix}.json"
                if cache_fp.exists():
                    return cache_fp.read_text(encoding="utf-8")

        for key_suffix in (row_id, scenario_id):
            if key_suffix and (config, key_suffix) in self._samples_index:
                data_str = self._samples_index[(config, key_suffix)]
                (RAW_CACHE_DIR / f"{config}__{key_suffix}.json").write_text(data_str, encoding="utf-8")
                return data_str

        # Fetch single row on cache miss from HuggingFace datasets-server
        offset = self._pool_offsets.get((config, row_id))
        if offset is None and scenario_id:
            offset = self._pool_offsets.get((config, scenario_id))
        if offset is None:
            raise KeyError(f"Cannot resolve Gaia2 offset for config={config!r}, row_id={row_id!r}, scenario_id={scenario_id!r}")

        qs = urllib.parse.urlencode(
            {
                "dataset": "meta-agents-research-environments/gaia2",
                "config": config,
                "split": "validation",
                "offset": offset,
                "length": 1,
            }
        )
        url = f"https://datasets-server.huggingface.co/rows?{qs}"
        payload: dict[str, Any] = {}
        last_fetch_err: Exception | None = None
        for attempt in range(4):
            try:
                opener = (
                    urllib.request.build_opener(urllib.request.ProxyHandler({}))
                    if attempt % 2 == 1
                    else urllib.request.build_opener()
                )
                with opener.open(url, timeout=45.0) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                break
            except Exception as exc:
                last_fetch_err = exc
                time.sleep(1.2 * (attempt + 1))
        else:
            raise RuntimeError(f"Failed fetching HF datasets-server row {url}: {last_fetch_err}")
        rows = payload.get("rows") or []
        if not rows:
            raise RuntimeError(f"Empty HF datasets-server response for {url}")
        row_obj = rows[0].get("row") or {}
        data_val = row_obj.get("data")
        data_str = data_val if isinstance(data_val, str) else json.dumps(data_val, ensure_ascii=False)
        actual_id = str(row_obj.get("id") or row_id or offset)
        (RAW_CACHE_DIR / f"{config}__{actual_id}.json").write_text(data_str, encoding="utf-8")
        return data_str

    def _raw_state_hash(self) -> str:
        assert self.env is not None
        raw = self.env.apps_state_json()
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _state_hash(self) -> str:
        """Semantic state hash normalizing passive `is_read` receipt flips from read-only tools (`NEW-9`)."""
        assert self.env is not None
        raw = self.env.apps_state_json()
        try:
            parsed = json.loads(raw)
            norm = json.dumps(_normalize_passive_read_receipts(parsed), sort_keys=True, ensure_ascii=False)
            return hashlib.sha256(norm.encode("utf-8")).hexdigest()
        except Exception:
            return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def start(self, req: StartScenarioReq) -> dict[str, Any]:
        with self.lock:
            if self.env is not None:
                try:
                    self.env.stop()
                except Exception:
                    pass
            raw_json = self.resolve_scenario_json(req.upstream_id, req.config, req.row_id, req.scenario_id)
            importer = JsonScenarioImporter()
            judge_sc, _, _ = importer.import_from_json_to_benchmark(
                raw_json,
                apps_to_skip=[],
                load_completed_events=False,
            )
            live_sc, _, _ = importer.import_from_json_to_benchmark(
                raw_json,
                apps_to_skip=[],
                load_completed_events=False,
            )
            _, _, resolved_model = _resolve_judge_credentials()
            self.judge_mode = "deterministic" if req.judge_mode == "deterministic" else "llm_judge"
            self.judge_model = "deterministic_structural" if req.judge_mode == "deterministic" else resolved_model
            self.judge_errors = []
            self.judge_usage = {"llm_calls": 0, "deterministic_calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
            judge_cfg = GraphPerEventJudgeConfig(
                engine=_create_judge_engine(req.judge_mode, self.judge_errors, self.judge_usage)
            )
            # Keep `judge_sc` completely unscheduled & unmutated by live execution (`A1`) so relative `1s`
            # Turn-0 oracle events NEVER get converted into `+26s` `absolute_event_time` deadlines!
            preprocess_scenario(judge_sc, judge_config=judge_cfg, offline_validation=True)
            preprocess_scenario(live_sc, judge_config=None, offline_validation=True)

            env = Environment(
                EnvironmentConfig(
                    oracle_mode=False,
                    queue_based_loop=False,
                    start_time=live_sc.start_time,
                    duration=live_sc.duration or 600,
                    verbose=False,
                )
            )
            env.register_apps(live_sc.apps if live_sc.apps else [])
            if "AgentUserInterface" in env.apps:
                env.apps["AgentUserInterface"].wait_for_user_response = False
            self.readonly_side_mount = bool(req.readonly_side_mount)
            events_to_schedule = [
                ev
                for ev in (live_sc.events or [])
                if not isinstance(ev, OracleEvent)
                and (not self.readonly_side_mount or getattr(ev, "event_type", None) != EventType.ENV)
            ]
            env.schedule(events_to_schedule)
            env.prepare_events_for_start()
            env.tick()
            # Pause simulated clock immediately after boot tick so LLM inference latency never advances `event_time` (`A1`)
            self._pause_env_clock(env)

            tools: dict[str, Any] = {}
            apps_idx: dict[str, list[dict[str, Any]]] = {}
            for app_name, app in sorted(env.apps.items()):
                app_tools_meta: list[dict[str, Any]] = []
                for t in app.get_tools():
                    tools[t.name] = t
                    args_meta = []
                    for a in getattr(t, "args", []) or []:
                        args_meta.append(
                            {
                                "name": getattr(a, "name", ""),
                                "type": str(getattr(a, "arg_type", "Any")),
                                "description": getattr(a, "description", ""),
                                "has_default": bool(getattr(a, "has_default", False)),
                                "default": make_serializable(getattr(a, "default", None))
                                if getattr(a, "has_default", False)
                                else None,
                            }
                        )
                    app_tools_meta.append(
                        {
                            "tool_name": t.name,
                            "description": getattr(t, "function_description", "") or getattr(t, "description", ""),
                            "args": args_meta,
                        }
                    )
                apps_idx[app_name] = app_tools_meta

            self.scenario = live_sc
            self.judge_scenario = judge_sc
            self.env = env
            self.tools = tools
            self.apps_index = apps_idx
            self.call_log = []
            self.buffered_aui_messages = []
            self._flushed_aui_count = 0
            self.mutating_call_count = 0
            self.aui_call_count = 0
            self.initial_state_hash = self._state_hash()
            self.initial_raw_state_hash = self._raw_state_hash()
            return {
                "ok": True,
                "scenario_id": judge_sc.scenario_id,
                "start_time": judge_sc.start_time,
                "nb_turns": judge_sc.nb_turns,
                "apps": sorted(apps_idx.keys()),
                "tool_count": len(tools),
                "initial_state_hash": self.initial_state_hash,
            }

    def call_tool(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            if self.env is None or self.scenario is None:
                raise HTTPException(status_code=400, detail="No active Gaia2 scenario")
            if tool_name not in self.tools:
                # Support `App.fn` shorthand in addition to `App__fn`
                alt = tool_name.replace(".", "__", 1)
                if alt in self.tools:
                    tool_name = alt
                else:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Unknown tool {tool_name!r}. Run `./gaia_cli apps` or `./gaia_cli docs <App>`.",
                    )

            nb_turns = int(getattr(self.judge_scenario or self.scenario, "nb_turns", 1) or 1)
            is_aui_delivery = tool_name == "AgentUserInterface__send_message_to_user"
            # `A1` + `v2.0 P0-2`: Buffer Turn-0 `AgentUserInterface__send_message_to_user` so mid-episode user replies
            # do not prematurely split Turn-0, AND treat AUI user-reply delivery as chat transport rather than a
            # business-app state mutation (`Emails`, `Calendar`, `Contacts`, `Files`, `Shopping`, `Cabs`, `RentAFlat`).
            if nb_turns == 1 and is_aui_delivery:
                msg_txt = str((args or {}).get("content", "") or "").strip()
                if msg_txt:
                    self.buffered_aui_messages.append(msg_txt)
                self.aui_call_count += 1
                rec_aui = {
                    "seq": len(self.call_log) + 1,
                    "tool_name": tool_name,
                    "args": make_serializable(args or {}),
                    "ok": True,
                    "write_operation": False,
                    "aui_message_delivery": True,
                    "mutated": False,
                    "buffered_for_turn0_end": True,
                    "event_time": round(self.env.time_manager.time(), 4),
                    "time_passed_sec": round(self.env.time_manager.time_passed(), 4),
                }
                self.call_log.append(rec_aui)
                return {
                    "ok": True,
                    "tool_name": tool_name,
                    "mutated": False,
                    "aui_message_delivery": True,
                    "result": "Message sent to user.",
                }

            before_sem = self._state_hash()
            before_raw = self._raw_state_hash()
            tool_obj = self.tools[tool_name]
            raw_open_mode = str((args or {}).get("mode", "rb") or "rb").lower() if tool_name.endswith("__open") else ""
            is_readonly_open = tool_name.endswith("__open") and not any(
                ch in raw_open_mode for ch in ("w", "a", "+", "x")
            )
            is_write_op = (
                False
                if (is_readonly_open or is_aui_delivery)
                else bool(getattr(tool_obj, "write_operation", True))
            )
            try:
                with self._active_env_clock(self.env):
                    if is_readonly_open:
                        with EventRegisterer.disable():
                            out = tool_obj(**(args or {}))
                        if hasattr(out, "read"):
                            try:
                                raw_bytes = out.read()
                            finally:
                                try:
                                    out.close()
                                except Exception:
                                    pass
                            out = (
                                raw_bytes.decode("utf-8", errors="replace")
                                if isinstance(raw_bytes, (bytes, bytearray))
                                else raw_bytes
                            )
                    else:
                        out = tool_obj(**(args or {}))
                    if is_write_op or not self.readonly_side_mount:
                        self.env.tick()
                after_sem = self._state_hash()
                after_raw = self._raw_state_hash()
                if is_aui_delivery or not is_write_op:
                    if is_aui_delivery:
                        self.aui_call_count += 1
                    mutated = False
                else:
                    mutated = (after_sem != before_sem) or (after_raw != before_raw)
                passive_read_only = (not mutated) and ((after_sem != before_sem) or (after_raw != before_raw))
                if mutated:
                    self.mutating_call_count += 1
                rec = {
                    "seq": len(self.call_log) + 1,
                    "tool_name": tool_name,
                    "args": make_serializable(args or {}),
                    "ok": True,
                    "write_operation": is_write_op,
                    "mutated": mutated,
                    "passive_read_receipt_only": passive_read_only,
                    "event_time": round(self.env.time_manager.time(), 4),
                    "time_passed_sec": round(self.env.time_manager.time_passed(), 4),
                }
                self.call_log.append(rec)
                return {
                    "ok": True,
                    "tool_name": tool_name,
                    "mutated": mutated,
                    "result": make_serializable(out),
                }
            except Exception as exc:
                after_sem = self._state_hash()
                after_raw = self._raw_state_hash()
                mutated = bool(is_write_op and ((after_sem != before_sem) or (after_raw != before_raw)))
                if mutated:
                    self.mutating_call_count += 1
                self.call_log.append(
                    {
                        "seq": len(self.call_log) + 1,
                        "tool_name": tool_name,
                        "args": make_serializable(args or {}),
                        "ok": False,
                        "error": str(exc),
                        "write_operation": is_write_op,
                        "mutated": mutated,
                        "event_time": round(self.env.time_manager.time(), 4),
                        "time_passed_sec": round(self.env.time_manager.time_passed(), 4),
                    }
                )
                return {
                    "ok": False,
                    "tool_name": tool_name,
                    "error": f"{type(exc).__name__}: {exc}",
                }

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            if self.env is None:
                raise HTTPException(status_code=400, detail="No active Gaia2 scenario")
            cur_hash = self._state_hash()
            cur_raw_hash = self._raw_state_hash()
            zero_writes = all(not bool(c.get("write_operation", False)) for c in self.call_log)
            return {
                "apps_state_hash": cur_hash,
                "raw_apps_state_hash": cur_raw_hash,
                "initial_state_hash": self.initial_state_hash,
                "unmodified_since_start": (self.mutating_call_count == 0)
                and ((cur_hash == self.initial_state_hash) or zero_writes),
                "simulated_time_passed_sec": round(self.env.time_manager.time_passed(), 4),
                "buffered_aui_count": len(self.buffered_aui_messages),
                "aui_call_count": self.aui_call_count,
                "call_count": len(self.call_log),
                "mutating_call_count": self.mutating_call_count,
                "judge_mode": self.judge_mode,
                "judge_model": self.judge_model,
                "call_log": list(self.call_log),
            }

    def evaluate(self, agent_replies: list[str]) -> dict[str, Any]:
        with self.lock:
            if self.env is None or self.scenario is None:
                raise HTTPException(status_code=400, detail="No active Gaia2 scenario")
            judge_sc = self.judge_scenario or self.scenario
            env = self.env
            nb_turns = int(getattr(judge_sc, "nb_turns", 1) or 1)

            non_empty_replies = [r.strip() for r in (agent_replies or []) if r and r.strip()]
            if nb_turns == 1 and "AgentUserInterface__send_message_to_user" in self.tools:
                pending_buffered = self.buffered_aui_messages[self._flushed_aui_count :]
                has_unflushed = bool(pending_buffered) or (self._flushed_aui_count == 0 and bool(non_empty_replies))
                if has_unflushed and self._flushed_aui_count == 0:
                    raw_buffered = [m.strip() for m in self.buffered_aui_messages if m and m.strip()]
                    if len(raw_buffered) > 1 or (raw_buffered and non_empty_replies):
                        filtered_buf = [
                            m
                            for m in raw_buffered
                            if not (
                                any(k in m for k in ("office_fs", "office_pg", "pg_exec", "FILESYSTEM_TEST_DIR"))
                                and not any(
                                    gk in m.lower()
                                    for gk in (
                                        "calendar",
                                        "email",
                                        "message",
                                        "contact",
                                        "cab",
                                        "apartment",
                                        "shopping",
                                        "picture",
                                        "family",
                                    )
                                )
                            )
                        ]
                        if filtered_buf:
                            raw_buffered = filtered_buf
                    combined_parts: list[str] = []
                    for m_clean in raw_buffered:
                        if m_clean and m_clean not in combined_parts:
                            combined_parts.append(m_clean)
                    if non_empty_replies:
                        tail_rep = non_empty_replies[-1]
                        if tail_rep and not any(tail_rep in p for p in combined_parts):
                            combined_parts.append(tail_rep)
                    combined_txt = "\n\n".join(combined_parts).strip()[:12000]
                    if combined_txt:
                        try:
                            with self._active_env_clock(env):
                                self.tools["AgentUserInterface__send_message_to_user"](content=combined_txt)
                                env.tick()
                            self._flushed_aui_count = max(1, len(self.buffered_aui_messages))
                            self.call_log.append(
                                {
                                    "seq": len(self.call_log) + 1,
                                    "tool_name": "AgentUserInterface__send_message_to_user",
                                    "args": {"content": combined_txt},
                                    "ok": True,
                                    "mutated": False,
                                    "bridged_from_chat": len(self.buffered_aui_messages) == 0,
                                    "combined_turn0_aui": True,
                                    "event_time": round(env.time_manager.time(), 4),
                                    "time_passed_sec": round(env.time_manager.time_passed(), 4),
                                }
                            )
                        except Exception:
                            pass
            else:
                agent_aui_count = sum(
                    1
                    for c in self.call_log
                    if c.get("ok") and c.get("tool_name") == "AgentUserInterface__send_message_to_user"
                )
                if (
                    agent_aui_count == 0
                    and non_empty_replies
                    and "AgentUserInterface__send_message_to_user" in self.tools
                ):
                    final_reply = non_empty_replies[-1]
                    if final_reply:
                        try:
                            bridged_txt = final_reply[:8000]
                            with self._active_env_clock(env):
                                self.tools["AgentUserInterface__send_message_to_user"](content=bridged_txt)
                                env.tick()
                            self.call_log.append(
                                {
                                    "seq": len(self.call_log) + 1,
                                    "tool_name": "AgentUserInterface__send_message_to_user",
                                    "args": {"content": bridged_txt},
                                    "ok": True,
                                    "mutated": False,
                                    "bridged_from_chat": True,
                                    "event_time": round(env.time_manager.time(), 4),
                                    "time_passed_sec": round(env.time_manager.time_passed(), 4),
                                }
                            )
                        except Exception:
                            pass

            if getattr(judge_sc, "judge", None) is not None:
                judge_sc.judge.initialize_state(judge_sc)
            val_res = judge_sc.validate(env)
            passed = bool(val_res.success) and not self.judge_errors
            out_res: dict[str, Any] = {
                "passed": passed,
                "rationale": val_res.rationale,
                "scenario_id": judge_sc.scenario_id,
                "judge_mode": self.judge_mode,
                "judge_model": self.judge_model,
                "judge_usage": dict(self.judge_usage),
                "simulated_time_passed_sec": round(env.time_manager.time_passed(), 4),
                "call_count": len(self.call_log),
                "aui_call_count": self.aui_call_count,
                "mutating_call_count": self.mutating_call_count,
                "call_log": list(self.call_log),
            }
            if self.judge_errors:
                out_res["evaluation_error"] = self.judge_errors[0]
            return out_res


HOLDER = Gaia2RuntimeHolder()
app = FastAPI(title="Gaia2 Meta ARE Env Server")


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "active_scenario": getattr(HOLDER.scenario, "scenario_id", None),
    }


@app.post("/start_scenario")
def start_scenario(req: StartScenarioReq) -> dict[str, Any]:
    try:
        return HOLDER.start(req)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc


@app.get("/apps")
def list_apps() -> dict[str, Any]:
    with HOLDER.lock:
        if HOLDER.env is None:
            raise HTTPException(status_code=400, detail="No active Gaia2 scenario")
        return {
            "scenario_id": HOLDER.scenario.scenario_id,
            "apps": {
                app_name: [t["tool_name"] for t in tlist]
                for app_name, tlist in HOLDER.apps_index.items()
            },
        }


@app.get("/docs/{app_name}")
def app_docs(app_name: str) -> dict[str, Any]:
    with HOLDER.lock:
        if HOLDER.env is None:
            raise HTTPException(status_code=400, detail="No active Gaia2 scenario")
        if app_name not in HOLDER.apps_index:
            # Case-insensitive lookup
            matches = [k for k in HOLDER.apps_index if k.lower() == app_name.lower()]
            if not matches:
                raise HTTPException(status_code=404, detail=f"App {app_name!r} not found. Available: {list(HOLDER.apps_index)}")
            app_name = matches[0]
        return {"app_name": app_name, "tools": HOLDER.apps_index[app_name]}


@app.post("/call_tool")
def call_tool(req: ToolCallReq) -> dict[str, Any]:
    return HOLDER.call_tool(req.tool_name, req.args)


@app.post("/snapshot")
def snapshot() -> dict[str, Any]:
    return HOLDER.snapshot()


@app.post("/evaluate")
def evaluate(req: EvaluateReq) -> dict[str, Any]:
    return HOLDER.evaluate(req.agent_replies)


@app.get("/oracle_actions", include_in_schema=False)
def oracle_actions(request: Request) -> dict[str, Any]:
    """Internal unit-test diagnostic endpoint gated by random per-server bearer token (`P0-6`)."""
    import hmac

    supplied = (request.headers.get("x-pa150-diag-token") or "").strip()
    if len(HOLDER.diag_token) < 16 or not hmac.compare_digest(supplied, HOLDER.diag_token):
        raise HTTPException(status_code=403, detail="Forbidden: diagnostic token required")
    with HOLDER.lock:
        sc_ref = HOLDER.judge_scenario or HOLDER.scenario
        if sc_ref is None:
            raise HTTPException(status_code=400, detail="No active Gaia2 scenario")
        actions: list[dict[str, Any]] = []
        for ev in getattr(sc_ref, "oracle_run_event_log", []) or []:
            if ev.event_type == EventType.AGENT and ev.action is not None:
                aname = getattr(ev.action.app, "name", ev.action.app.__class__.__name__)
                tname = f"{aname}__{ev.action.function.__name__}"
                args = {
                    k: make_serializable(v)
                    for k, v in (ev.action.resolved_args or ev.action.args).items()
                    if k != "self"
                }
                actions.append({"tool_name": tname, "args": args})
        return {"scenario_id": sc_ref.scenario_id, "oracle_actions": actions}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=19011)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
