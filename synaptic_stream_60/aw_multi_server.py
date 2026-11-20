"""Multi-account persistent `AppWorld` server + lock-free out-of-process snapshot evaluator (`v3.0.0`).

Architecture guarantees:
  1. Zero intra-account DB collision: each AppWorld task in a stream maps to its own isolated
     account (`acct_01`..`acct_17`) with native upstream initial DBs and supervisor profile.
  2. Zero observer lock contention: `execute_code` / `submit_answer` atomically exports immutable
     point-in-time snapshot directories (`snapshots/<acct_id>/vXXXX/`) via `os.replace`.
  3. Out-of-process evaluation: a separate `.venv_appworld` worker process evaluates snapshots
     via unmodified upstream `AppWorld` `TestTracker` without touching the live execution server.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import http.server
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import socket
import sqlite3
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
from typing import Any

from appworld_micro_30.env_server import parse_concise_qa_answer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APPWORLD_VENV_PYTHON = PROJECT_ROOT / ".venv_appworld" / "bin" / "python"
APPWORLD_DATA_DIR = PROJECT_ROOT / "data"

USER_APP_NAMES: tuple[str, ...] = (
    "amazon",
    "file_system",
    "gmail",
    "phone",
    "simple_note",
    "splitwise",
    "spotify",
    "todoist",
    "venmo",
)


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _safe_close_world(world: Any) -> None:
    if world is None:
        return
    try:
        world.close()
    except Exception:
        pass
    try:
        from appworld.apps.model_lib import CachedDBHandler, ModelHashHandler
        from appworld.environment import AppWorld

        AppWorld.close_all()
        CachedDBHandler.reset()
        ModelHashHandler.reset()
    except Exception:
        pass


def _to_number(num_str: str) -> int | float:
    clean = num_str.replace(",", "")
    val = float(clean)
    return int(val) if val.is_integer() and "." not in clean else (int(val) if val.is_integer() else val)


def _normalize_stream_qa_answer(raw_answer: Any, *, instruction: str = "") -> Any:
    """Symmetric answer-format normalizer for `aw answer` across all 6 paradigms.

    Preserves exact numeric/string values while preventing avoidable format crashes:
      1. Numeric cost/amount questions (`total cost`, `how much`, `total amount`): runs
         currency (`$<num>`) + date-stripped anti-year extraction on raw string input BEFORE
         `parse_concise_qa_answer` so dates like `2023-01-01` or `Jan 1, 2023` never shadow `$42.50`.
      2. JSON string arrays passed to comma-separated string questions: joins as `a, b`.
      3. Bare string answers with a trailing sentence period `.`: strips the trailing `.`.
    """
    instr_low = (instruction or "").lower()
    is_money_q = any(
        k in instr_low
        for k in ("total cost", "how much", "total amount", "total price", "total spent", "average cost", "balance")
    )
    if isinstance(raw_answer, str) and is_money_q:
        raw_txt = raw_answer.strip()
        # 1a) Prefer explicit currency figures (`$42`, `$1,234.50`, `$89.0`) over stray dates/years.
        m_curr = re.search(r"\$\s*(\d+(?:,\d{3})*(?:\.\d+)?)(?![A-Za-z0-9]|\.\d)", raw_txt)
        if m_curr:
            return _to_number(m_curr.group(1))
        # 1b) Strip ISO dates (`2023-01-01`) and calendar date phrases (`Jan 1, 2023`, `1st Jan 2023`) before scanning numbers.
        cleaned = re.sub(r"\b\d{4}-\d{1,2}-\d{1,2}\b", " ", raw_txt)
        cleaned = re.sub(
            r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+\d{1,2}(?:st|nd|rd|th)?(?:,\s*\d{4})?\b",
            " ",
            cleaned,
            flags=re.I,
        )
        cleaned = re.sub(
            r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?(?:\s+\d{4})?\b",
            " ",
            cleaned,
            flags=re.I,
        )
        nums = re.findall(r"(?<![A-Za-z0-9.])(\d+(?:,\d{3})*(?:\.\d+)?)(?![A-Za-z0-9]|\.\d)", cleaned)
        non_years = [n for n in nums if not re.fullmatch(r"(?:19|20)\d{2}", n.replace(",", ""))]
        chosen = non_years[-1] if non_years else (nums[-1] if nums else "")
        if chosen:
            return _to_number(chosen)

    parsed = parse_concise_qa_answer(raw_answer, instruction=instruction)
    if isinstance(parsed, list) and parsed and all(isinstance(x, (str, int, float)) for x in parsed):
        if any(k in instr_low for k in ("comma", "list", "separated", "titles", "names")):
            return ", ".join(str(x).strip() for x in parsed if str(x).strip())
        return parsed
    if isinstance(parsed, str):
        text = parsed.strip()
        if text.endswith(".") and not text.endswith("...") and len(text.splitlines()) == 1:
            stripped = text[:-1].strip()
            if stripped and not re.search(r"\b(?:Mr|Mrs|Ms|Dr|St|Jr|Sr|Inc|Ltd|vs|etc)\.$", text):
                text = stripped
        return text
    return parsed


def _scrub_instructions(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: "" if k == "instruction" else _scrub_instructions(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_scrub_instructions(v) for v in value]
    return value


class _MultiAccountAppWorldRuntime:
    """Single-process disk-paged multi-account AppWorld execution runtime."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.lock = threading.RLock()
        self.stream_id: str | None = None
        self.state_root: Path | None = None
        self.snapshots_root: Path | None = None
        self.accounts: dict[str, dict[str, Any]] = {}
        self.acct_versions: dict[str, int] = {}
        self.acct_initial_hashes: dict[str, dict[str, str]] = {}
        self.acct_current_hashes: dict[str, dict[str, str]] = {}
        self.acct_submitted_answers: dict[str, list[Any]] = {}
        self.acct_exec_counts: dict[str, int] = {}
        self.current_acct_id: str | None = None
        self.current_world: Any = None
        self.current_exp_name: str | None = None

    def _compute_app_hashes(self, raw_hashes: dict[str, Any]) -> dict[str, str]:
        out: dict[str, str] = {}
        for app in USER_APP_NAMES:
            matching = {
                k: raw_hashes[k]
                for k in sorted(raw_hashes.keys())
                if k == app or k.startswith(f"{app}.")
            }
            payload = json.dumps(matching, sort_keys=True, default=str)
            out[app] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return out

    def _save_world_changes_atomic_unlocked(
        self,
        world: Any,
        final_dir: Path,
        *,
        extra_meta: dict[str, Any] | None = None,
    ) -> None:
        from appworld.apps.api_lib import save_local_dbs

        final_dir.parent.mkdir(parents=True, exist_ok=True)
        tmp_dir = final_dir.parent / f".tmp_{final_dir.name}_{uuid.uuid4().hex[:6]}"
        save_local_dbs(
            from_db_home_path=world.models_to_db_home_path,
            to_db_home_path=str(tmp_dir),
            format="changes",
            app_names=[*world.task.allowed_apps, "admin"],
            delete_if_exists=True,
            skip_mandatory_apps=False,
            save_model_hashes=True,
            vaccum=False,
        )
        if extra_meta is not None:
            # Resolve original clocks at call time: AppWorld freezes both public
            # functions, and macOS CLOCK_MONOTONIC has a different epoch from
            # perf_counter (the clock used by the arrival producer).
            from freezegun.api import real_time, real_perf_counter

            (tmp_dir / "snapshot_meta.json").write_text(
                json.dumps({**extra_meta,
                            "created_at_unix": real_time(),
                            "created_at_monotonic": real_perf_counter()},
                           ensure_ascii=False),
                encoding="utf-8",
            )
        if final_dir.exists():
            old_bak = final_dir.parent / f".old_{final_dir.name}_{uuid.uuid4().hex[:6]}"
            os.replace(final_dir, old_bak)
            os.replace(tmp_dir, final_dir)
            shutil.rmtree(old_bak, ignore_errors=True)
        else:
            os.replace(tmp_dir, final_dir)

    def _flush_and_close_active_unlocked(self) -> None:
        if self.current_world is not None and self.current_acct_id is not None and self.state_root is not None:
            acct_state_dir = self.state_root / self.current_acct_id
            self._save_world_changes_atomic_unlocked(self.current_world, acct_state_dir)
        _safe_close_world(self.current_world)
        if self.current_exp_name:
            exp_out = PROJECT_ROOT / "experiments" / "outputs" / self.current_exp_name
            if exp_out.exists():
                shutil.rmtree(exp_out, ignore_errors=True)
        self.current_world = None
        self.current_acct_id = None
        self.current_exp_name = None

    def close_all(self) -> None:
        with self.lock:
            self._flush_and_close_active_unlocked()
            if self.state_root is not None and self.state_root.exists():
                shutil.rmtree(self.state_root, ignore_errors=True)
            self.state_root = None
            self.snapshots_root = None
            self.stream_id = None
            self.accounts.clear()
            self.acct_versions.clear()
            self.acct_initial_hashes.clear()
            self.acct_current_hashes.clear()
            self.acct_submitted_answers.clear()
            self.acct_exec_counts.clear()
            gc.collect()

    def start_stream(
        self,
        stream_id: str,
        accounts: list[dict[str, Any]],
        snapshots_root: str,
    ) -> dict[str, Any]:
        with self.lock:
            self.close_all()
            self.stream_id = stream_id
            self.state_root = Path(tempfile.mkdtemp(prefix=f"aw_stream_live_{stream_id}_"))
            self.snapshots_root = Path(snapshots_root).resolve()
            self.snapshots_root.mkdir(parents=True, exist_ok=True)
            for acc in accounts:
                aid = str(acc["account_id"])
                self.accounts[aid] = dict(acc)
                self.acct_versions[aid] = 0
                self.acct_submitted_answers[aid] = []
                self.acct_exec_counts[aid] = 0
            return {
                "ok": True,
                "stream_id": stream_id,
                "account_ids": sorted(self.accounts.keys()),
            }

    def _resolve_account_id_unlocked(self, account_ref: str | None) -> str:
        """Resolve an exact ID, normalized full name, or email; never guess a person."""
        ref = (account_ref or "").strip().strip("<>\'\"").strip().casefold()
        name_ref = " ".join(ref.replace("_", " ").split())
        matches = [aid for aid, account in self.accounts.items()
                   if ref == aid.casefold()
                   or (ref and ref == str(account.get("supervisor_email", "")).strip().casefold())
                   or (name_ref and name_ref == " ".join(
                       str(account.get("supervisor_name", "")).casefold().replace("_", " ").split()))]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(f"Ambiguous account {account_ref!r}; use an account ID or email.")
        raise ValueError(f"Unknown account {account_ref!r}. Use accounts to list connected members.")

    def _activate_account_unlocked(self, account_id: str) -> Any:
        from appworld.apps.api_lib import set_local_dbs
        from appworld.apps.model_lib import CachedDBHandler, ModelHashHandler
        from appworld.collections.models import ModelCollection
        from appworld.environment import AppWorld

        if account_id not in self.accounts:
            raise ValueError(
                f"Unknown account_id {account_id!r}. Valid accounts: {sorted(self.accounts.keys())}"
            )
        if self.current_acct_id == account_id and self.current_world is not None:
            return self.current_world

        self._flush_and_close_active_unlocked()
        acc = self.accounts[account_id]
        tid = str(acc["upstream_task_id"])
        exp_name = f"aw_st_{self.stream_id}_{account_id}_{uuid.uuid4().hex[:6]}"
        world = AppWorld(
            task_id=tid,
            experiment_name=exp_name,
            raise_on_failure=False,
            ground_truth_mode="minimal",
        )
        assert self.state_root is not None
        acct_state_dir = self.state_root / account_id
        if (acct_state_dir / "model_hashes.json").is_file():
            # Reset `CachedDBHandler` & `ModelHashHandler` for `models_to_db_home_path` BEFORE reloading
            # persisted `acct_state_dir` (`format="changes"`) so `ModelCollection.load` applies the saved
            # delta instead of reusing the freshly initialized `:memory:` connection from `AppWorld.__init__`!
            CachedDBHandler.reset(world.models_to_db_home_path)
            ModelHashHandler.reset(world.models_to_db_home_path)
            world.models = ModelCollection.load(
                to_db_home_path=world.models_to_db_home_path,
                from_db_home_path=str(acct_state_dir),
                load_apps=world.task.allowed_apps,
            )
            set_local_dbs(to_db_home_path=world.models_to_db_home_path)
        else:
            init_h = self._compute_app_hashes(dict(world.models.model_hashes()))
            self.acct_initial_hashes[account_id] = init_h
            self.acct_current_hashes[account_id] = dict(init_h)

        # Task rows include upstream instructions for future/withheld requests.
        # Scrub the live DB on both fresh activation and persisted-state reload.
        task_model = world.models.supervisor.Task
        # Upstream constr(min_length=1) otherwise rejects both redaction and
        # later complete_task()/answer saves. Relax only this field in this
        # execution process; evaluator models and on-disk upstream data stay intact.
        task_model.__fields__["instruction"].type_.min_length = 0
        for task in task_model.all():
            task.instruction = ""
            task.save()
        supervisor = world.apis.supervisor
        if hasattr(supervisor, "show_active_task"):
            original_show = supervisor.show_active_task

            def show_active_task(*args: Any, **kwargs: Any) -> Any:
                return _scrub_instructions(original_show(*args, **kwargs))

            supervisor.show_active_task = show_active_task

        self.current_acct_id = account_id
        self.current_world = world
        self.current_exp_name = exp_name
        return world

    def _export_atomic_snapshot_unlocked(self, account_id: str) -> dict[str, Any]:
        assert self.snapshots_root is not None and self.current_world is not None
        ver = self.acct_versions[account_id] + 1
        self.acct_versions[account_id] = ver
        acct_snap_parent = self.snapshots_root / account_id
        acct_snap_parent.mkdir(parents=True, exist_ok=True)
        final_dir = acct_snap_parent / f"v{ver:04d}"

        init_h = self.acct_initial_hashes[account_id]
        cur_h = self.acct_current_hashes[account_id]
        mutated = [a for a in USER_APP_NAMES if init_h.get(a) != cur_h.get(a)]
        meta = {
            "account_id": account_id,
            "upstream_task_id": self.accounts[account_id]["upstream_task_id"],
            "version": ver,
            "mutated_user_apps": mutated,
            "submitted_answers": list(self.acct_submitted_answers[account_id]),
            "exec_count": self.acct_exec_counts[account_id],
        }
        self._save_world_changes_atomic_unlocked(self.current_world, final_dir, extra_meta=meta)
        # Retain immutable history through teardown: slow evaluators must not miss a
        # pre-deadline snapshot or a transient regression between polls.
        return {"version": ver, "snapshot_dir": str(final_dir), "mutated_user_apps": mutated}

    def _capture_supervisor_answer_if_any_unlocked(self, account_id: str) -> Any:
        world = self.current_world
        active_tasks = world.models.supervisor.Task.all()
        if not active_tasks:
            return None
        t0 = active_tasks[0]
        raw_ans = t0.answer
        status = t0.status
        captured = None
        if raw_ans != '"<<NOT_GIVEN>>"' or status is not None:
            try:
                parsed = json.loads(raw_ans)
            except Exception:
                parsed = raw_ans
            if parsed != "<<NOT_GIVEN>>" and parsed is not None:
                captured = parsed
                self.acct_submitted_answers[account_id].append(parsed)
            t0.answer = '"<<NOT_GIVEN>>"'
            t0.status = None
            t0.save()
        return captured

    def list_accounts(self) -> dict[str, Any]:
        with self.lock:
            rows = [
                {
                    "account_id": aid,
                    "supervisor_name": acc.get("supervisor_name", ""),
                    "supervisor_email": acc.get("supervisor_email", ""),
                    "datetime": acc.get("datetime_str", ""),
                }
                for aid, acc in sorted(self.accounts.items())
            ]
            return {"ok": True, "accounts": rows}

    def list_apps(self, account_id: str | None = None) -> dict[str, Any]:
        with self.lock:
            aid = self._resolve_account_id_unlocked(account_id) if account_id is not None else next(iter(sorted(self.accounts)), None)
            if not aid:
                return {"ok": False, "error": "No accounts registered."}
            w = self._activate_account_unlocked(aid)
            docs = w.task.api_docs
            return {
                "ok": True,
                "account_id": aid,
                "apps": [{"app_name": k, "api_count": len(docs[k])} for k in docs.keys()],
            }

    def get_docs(self, app_name: str, api_name: str | None = None, account_id: str | None = None) -> dict[str, Any]:
        with self.lock:
            aid = self._resolve_account_id_unlocked(account_id) if account_id is not None else self.current_acct_id or next(iter(sorted(self.accounts)), None)
            if not aid:
                return {"ok": False, "error": "No accounts registered."}
            w = self._activate_account_unlocked(aid)
            docs = w.task.api_docs
            if app_name not in docs:
                return {"ok": False, "error": f"Unknown app {app_name!r}. Available: {list(docs.keys())}"}
            app_docs = docs[app_name]
            if api_name:
                if api_name not in app_docs:
                    return {
                        "ok": False,
                        "error": f"Unknown API {api_name!r} in {app_name!r}. Available: {list(app_docs.keys())}",
                    }
                return {"ok": True, "app": app_name, "api": api_name, "doc": app_docs[api_name]}
            summary = {
                k: {"description": v.get("description", ""), "parameters": v.get("parameters", [])}
                for k, v in app_docs.items()
            }
            return {"ok": True, "app": app_name, "apis": summary}

    def execute_code(self, account_id: str, code: str) -> dict[str, Any]:
        with self.lock:
            if not account_id:
                return {
                    "ok": False,
                    "error": "Missing `--account <acct_id_or_name>`! Specify target account (e.g. `aw --account acct_01 exec ...`).",
                }
            account_id = self._resolve_account_id_unlocked(account_id)
            w = self._activate_account_unlocked(account_id)
            before_h = dict(self.acct_current_hashes[account_id])
            output = w.execute(code)
            captured = self._capture_supervisor_answer_if_any_unlocked(account_id)
            after_h = self._compute_app_hashes(dict(w.models.model_hashes()))
            self.acct_current_hashes[account_id] = after_h
            self.acct_exec_counts[account_id] += 1
            mutated_step = [a for a in USER_APP_NAMES if before_h.get(a) != after_h.get(a)]
            snap_info = None
            if mutated_step or captured is not None:
                snap_info = self._export_atomic_snapshot_unlocked(account_id)
            return {
                "ok": True,
                "account_id": account_id,
                "output": output,
                "captured_answer": captured,
                "mutated_apps_in_step": mutated_step,
                "snapshot": snap_info,
            }

    def submit_answer(self, account_id: str, raw_answer: Any) -> dict[str, Any]:
        with self.lock:
            if not account_id:
                return {"ok": False, "error": "Missing `--account <acct_id_or_name>` for `answer`."}
            account_id = self._resolve_account_id_unlocked(account_id)
            w = self._activate_account_unlocked(account_id)
            instr = str(w.task.instruction or "")
            if not instr:
                tid = str(self.accounts[account_id]["upstream_task_id"])
                specs_fp = self.data_dir / "tasks" / tid / "specs.json"
                if specs_fp.is_file():
                    instr = str(json.loads(specs_fp.read_text(encoding="utf-8")).get("instruction") or "")
            parsed = _normalize_stream_qa_answer(raw_answer, instruction=instr)
            self.acct_submitted_answers[account_id].append(parsed)
            snap_info = self._export_atomic_snapshot_unlocked(account_id)
            return {
                "ok": True,
                "account_id": account_id,
                "recorded_answer": parsed,
                "snapshot": snap_info,
            }


class _MainThreadHTTPServer(http.server.HTTPServer):
    allow_reuse_address = True


def _run_exec_server_main(args: argparse.Namespace) -> None:
    os.environ.setdefault("APPWORLD_ROOT", str(Path(args.data_dir).resolve().parent))
    runtime = _MultiAccountAppWorldRuntime(Path(args.data_dir).resolve())
    expected_secret = args.secret

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, format: str, *log_args: Any) -> None:
            return

        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send_json(200, {"ok": True, "stream_id": runtime.stream_id})
                return
            self._send_json(404, {"ok": False, "error": "Not found"})

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            try:
                req = json.loads(raw.decode("utf-8"))
            except Exception as exc:
                self._send_json(400, {"ok": False, "error": f"Invalid JSON: {exc}"})
                return

            path = self.path
            admin_paths = {"/admin/start_stream", "/admin/close"}
            if path in admin_paths and self.headers.get("X-Admin-Secret", "") != expected_secret:
                self._send_json(403, {"ok": False, "error": "Forbidden"})
                return

            try:
                if path == "/admin/start_stream":
                    res = runtime.start_stream(
                        str(req["stream_id"]),
                        list(req["accounts"]),
                        str(req["snapshots_root"]),
                    )
                elif path == "/admin/close":
                    runtime.close_all()
                    res = {"ok": True}
                elif path == "/accounts":
                    res = runtime.list_accounts()
                elif path == "/apps":
                    res = runtime.list_apps(req.get("account_id"))
                elif path == "/docs":
                    res = runtime.get_docs(
                        str(req.get("app_name", "")),
                        req.get("api_name"),
                        req.get("account_id"),
                    )
                elif path == "/exec":
                    res = runtime.execute_code(str(req.get("account_id", "")), str(req.get("code", "")))
                elif path == "/answer":
                    res = runtime.submit_answer(str(req.get("account_id", "")), req.get("answer"))
                else:
                    self._send_json(404, {"ok": False, "error": f"Unknown endpoint {path}"})
                    return
                self._send_json(200, res)
            except Exception as exc:
                self._send_json(500, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    srv = _MainThreadHTTPServer((args.host, args.port), Handler)
    try:
        srv.serve_forever(poll_interval=0.05)
    finally:
        runtime.close_all()


def _run_eval_worker_main(args: argparse.Namespace) -> None:
    """Out-of-process AppWorld snapshot evaluator server (zero shared locks with live server)."""
    os.environ.setdefault("APPWORLD_ROOT", str(Path(args.data_dir).resolve().parent))
    from appworld.apps.model_lib import CachedDBHandler, ModelHashHandler
    from appworld.collections.models import ModelCollection, ModelCollectionPair
    from appworld.evaluator import TestTracker
    from appworld.task import Task

    data_dir = Path(args.data_dir).resolve()

    def _effective_allowed_apps(tid: str, base_target_apps: list[str], is_qa: bool) -> set[str]:
        if is_qa:
            return set()
        apps = set(base_target_apps)
        eval_fp = data_dir / "tasks" / tid / "ground_truth" / "evaluation.py"
        if eval_fp.is_file():
            src = eval_fp.read_text(encoding="utf-8")
            for app in USER_APP_NAMES:
                if f"models.{app}." in src or f"apis.{app}." in src or f'"{app}.' in src or f"'{app}." in src:
                    apps.add(app)
        return apps

    def _eval_one(req: dict[str, Any]) -> dict[str, Any]:
        tid = str(req["upstream_task_id"])
        snap_dir_raw = req.get("snapshot_dir")
        is_qa = bool(req.get("is_qa", False))
        target_apps = list(req.get("target_apps") or [])
        candidate_replies = list(req.get("candidate_replies") or [])

        mutated_apps: list[str] = []
        submitted_answers: list[Any] = []
        if snap_dir_raw and Path(snap_dir_raw).is_dir():
            sdir = Path(snap_dir_raw)
            meta_fp = sdir / "snapshot_meta.json"
            if meta_fp.is_file():
                smeta = json.loads(meta_fp.read_text(encoding="utf-8"))
                mutated_apps = list(smeta.get("mutated_user_apps") or [])
                submitted_answers = [x for x in (smeta.get("submitted_answers") or []) if x is not None]
        else:
            sdir = None

        if not is_qa and sdir is None:
            return {"ok": True, "passed": False, "reason": "no_mutation_snapshot"}

        allowed_apps = _effective_allowed_apps(tid, target_apps, is_qa)
        collateral = sorted(set(mutated_apps) - allowed_apps)

        # Load `Task` fresh per evaluation (NEVER cache `Task` across tasks because `ModelCollection.load`
        # mutates global class-level `SQLModel` engine bindings across calls!).
        mem_tag = f":memory:st_eval_{tid}_{uuid.uuid4().hex[:8]}"
        start_mem_tag: str | None = None
        try:
            t_obj = Task.load(task_id=tid, ground_truth_mode="minimal")
            m_start = t_obj.model_collection
            start_mem_tag = m_start.to_db_home_path

            resolved_ans: Any = None
            has_ans = False
            if submitted_answers:
                resolved_ans = submitted_answers[-1]
                has_ans = True
            elif is_qa and candidate_replies:
                instr = str(t_obj.instruction or "")
                for cand in reversed(candidate_replies):
                    parsed = parse_concise_qa_answer(cand, instruction=instr)
                    if parsed is not None:
                        resolved_ans = parsed
                        has_ans = True
                        break

            if is_qa and not has_ans:
                return {
                    "ok": True,
                    "passed": False,
                    "reason": "no_qa_answer",
                    "bystander_collateral": collateral,
                }

            end_from_dir = str(sdir) if sdir is not None else str(m_start.from_db_home_path)
            m_end = ModelCollection.load(
                to_db_home_path=mem_tag,
                from_db_home_path=end_from_dir,
                load_apps=t_obj.allowed_apps,
            )
            active_sup_tasks = m_end.supervisor.Task.all()
            if active_sup_tasks:
                sup_t = active_sup_tasks[0]
                # The trusted evaluator alone restores this metadata so upstream
                # min_length validation and evaluation see the original task.
                # This is an in-memory evaluation copy, never exported to agents.
                sup_t.instruction = str(t_obj.instruction)
                sup_t.answer = "null" if not is_qa else json.dumps(resolved_ans)
                sup_t.status = "success"
                sup_t.save()

            pair = ModelCollectionPair(
                start_db_home_path=m_start.from_db_home_path,
                start_model_collection=m_start,
                end_db_home_path=mem_tag,
                end_model_collection=m_end,
            )
            gt = t_obj.ground_truth
            main_user = pair.start.admin.MainUser.find_one(**t_obj.supervisor)
            tracker = TestTracker(
                test_data=gt.test_data,
                difficulty=gt.metadata["difficulty"],
                suppress_errors=True,
            )
            try:
                gt.evaluation_module().evaluate(
                    test=tracker,
                    public_data=gt.public_data,
                    private_data=gt.private_data,
                    main_user=main_user,
                    models=pair,
                    ground_truth_answer=gt.answer,
                )
            except Exception as eval_exc:
                return {
                    "ok": True,
                    "passed": False,
                    "tracker_success": False,
                    "pass_percentage": 0.0,
                    "bystander_collateral": collateral,
                    "resolved_answer": resolved_ans if is_qa else None,
                    "eval_error": f"{type(eval_exc).__name__}: {eval_exc}",
                }
            t_ok = bool(tracker.success)
            passed = t_ok and (len(collateral) == 0)
            return {
                "ok": True,
                "passed": passed,
                "tracker_success": t_ok,
                "pass_percentage": float(tracker.pass_percentage),
                "bystander_collateral": collateral,
                "resolved_answer": resolved_ans if is_qa else None,
            }
        finally:
            CachedDBHandler.reset(mem_tag)
            ModelHashHandler.reset(mem_tag)
            if start_mem_tag:
                CachedDBHandler.reset(start_mem_tag)
                ModelHashHandler.reset(start_mem_tag)

    class EvalHandler(http.server.BaseHTTPRequestHandler):
        def log_message(self, format: str, *log_args: Any) -> None:
            return

        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send_json(200, {"ok": True, "mode": "eval_worker"})
                return
            self._send_json(404, {"ok": False})

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            try:
                req = json.loads(raw.decode("utf-8"))
                if self.path == "/evaluate":
                    res = _eval_one(req)
                    self._send_json(200, res)
                else:
                    self._send_json(404, {"ok": False, "error": "Unknown path"})
            except Exception as exc:
                self._send_json(500, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    srv = _MainThreadHTTPServer((args.host, args.port), EvalHandler)
    srv.serve_forever(poll_interval=0.05)


class StreamAppWorldServerProcess:
    """Lifecycle wrapper for the live multi-account AppWorld execution server."""

    def __init__(self, *, port: int = 0, data_dir: Path = APPWORLD_DATA_DIR, log_path: Path | None = None) -> None:
        self.port = port or _pick_free_port()
        self.data_dir = data_dir
        self.secret = secrets.token_hex(16)
        self.log_path = log_path
        self.proc: subprocess.Popen[str] | None = None
        self._log_fh: Any = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self, timeout_sec: float = 40.0) -> None:
        if self.proc is not None and self.proc.poll() is None:
            return
        env = dict(os.environ)
        env["APPWORLD_ROOT"] = str(PROJECT_ROOT)
        py_parts = [str(PROJECT_ROOT)]
        if env.get("PYTHONPATH"):
            py_parts.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(py_parts)
        for k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
            env.pop(k, None)

        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_fh = self.log_path.open("a", encoding="utf-8")
        stdout_dst = self._log_fh if self._log_fh is not None else subprocess.DEVNULL

        self.proc = subprocess.Popen(
            [
                str(APPWORLD_VENV_PYTHON),
                "-m",
                "synaptic_stream_60.aw_multi_server",
                "--mode",
                "exec",
                "--port",
                str(self.port),
                "--secret",
                self.secret,
                "--data-dir",
                str(self.data_dir),
            ],
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=stdout_dst,
            stderr=subprocess.STDOUT,
            text=True,
        )
        deadline = time.perf_counter() + timeout_sec
        while time.perf_counter() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"StreamAppWorldServerProcess exited early with rc={self.proc.returncode}")
            try:
                with urllib.request.urlopen(f"{self.base_url}/health", timeout=1.0) as resp:
                    if resp.status == 200:
                        return
            except Exception:
                time.sleep(0.15)
        self.stop()
        raise TimeoutError("Timed out waiting for StreamAppWorldServerProcess")

    def _post(self, path: str, payload: dict[str, Any], *, admin: bool = False, timeout: float = 60.0) -> dict[str, Any]:
        self.start()
        headers = {"Content-Type": "application/json"}
        if admin:
            headers["X-Admin-Secret"] = self.secret
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def start_stream(self, stream_id: str, accounts: list[dict[str, Any]], snapshots_root: Path) -> dict[str, Any]:
        return self._post(
            "/admin/start_stream",
            {"stream_id": stream_id, "accounts": accounts, "snapshots_root": str(snapshots_root)},
            admin=True,
        )

    def execute_code(self, account_id: str, code: str) -> dict[str, Any]:
        return self._post("/exec", {"account_id": account_id, "code": code})

    def submit_answer(self, account_id: str, answer: Any) -> dict[str, Any]:
        return self._post("/answer", {"account_id": account_id, "answer": answer})

    def close_stream(self) -> None:
        if self.proc is None or self.proc.poll() is not None:
            return
        try:
            self._post("/admin/close", {}, admin=True, timeout=10.0)
        except Exception:
            pass

    def stop(self) -> None:
        if self.proc is not None:
            self.close_stream()
            if self.proc.poll() is None:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
            self.proc = None
        if self._log_fh is not None:
            try:
                self._log_fh.close()
            except Exception:
                pass
            self._log_fh = None


class StreamAppWorldEvalWorkerProcess:
    """Out-of-process AppWorld snapshot evaluator process."""

    def __init__(self, *, port: int = 0, data_dir: Path = APPWORLD_DATA_DIR, log_path: Path | None = None) -> None:
        self.port = port or _pick_free_port()
        self.data_dir = data_dir
        self.log_path = log_path
        self.proc: subprocess.Popen[str] | None = None
        self._log_fh: Any = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self, timeout_sec: float = 40.0) -> None:
        if self.proc is not None and self.proc.poll() is None:
            return
        env = dict(os.environ)
        env["APPWORLD_ROOT"] = str(PROJECT_ROOT)
        py_parts = [str(PROJECT_ROOT)]
        if env.get("PYTHONPATH"):
            py_parts.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(py_parts)
        for k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
            env.pop(k, None)

        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_fh = self.log_path.open("a", encoding="utf-8")
        stdout_dst = self._log_fh if self._log_fh is not None else subprocess.DEVNULL

        self.proc = subprocess.Popen(
            [
                str(APPWORLD_VENV_PYTHON),
                "-m",
                "synaptic_stream_60.aw_multi_server",
                "--mode",
                "eval",
                "--port",
                str(self.port),
                "--secret",
                "eval",
                "--data-dir",
                str(self.data_dir),
            ],
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=stdout_dst,
            stderr=subprocess.STDOUT,
            text=True,
        )
        deadline = time.perf_counter() + timeout_sec
        while time.perf_counter() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"EvalWorker exited early with rc={self.proc.returncode}")
            try:
                with urllib.request.urlopen(f"{self.base_url}/health", timeout=1.0) as resp:
                    if resp.status == 200:
                        return
            except Exception:
                time.sleep(0.15)
        self.stop()
        raise TimeoutError("Timed out waiting for StreamAppWorldEvalWorkerProcess")

    def evaluate(
        self,
        *,
        upstream_task_id: str,
        snapshot_dir: str | None,
        is_qa: bool,
        target_apps: list[str],
        candidate_replies: list[str] | None = None,
    ) -> dict[str, Any]:
        self.start()
        payload = {
            "upstream_task_id": upstream_task_id,
            "snapshot_dir": snapshot_dir,
            "is_qa": is_qa,
            "target_apps": target_apps,
            "candidate_replies": candidate_replies or [],
        }
        req = urllib.request.Request(
            f"{self.base_url}/evaluate",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=45.0) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def stop(self) -> None:
        if self.proc is not None:
            if self.proc.poll() is None:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
            self.proc = None
        if self._log_fh is not None:
            try:
                self._log_fh.close()
            except Exception:
                pass
            self._log_fh = None


def build_stream_aw_cli_script(base_url: str) -> str:
    """Generate symmetric `./aw` CLI script supporting `--account <acct_id_or_name>`."""
    return f"""#!/usr/bin/env python3
import json
import os
import sys
import urllib.request

BASE_URL = {base_url!r}

def _post(path: str, payload: dict) -> dict:
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        os.environ.pop(k, None)
    req = urllib.request.Request(
        BASE_URL + path,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={{"Content-Type": "application/json"}},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60.0) as resp:
        return json.loads(resp.read().decode("utf-8"))

def main() -> None:
    argv = list(sys.argv[1:])
    acct = os.environ.get("AW_ACCOUNT", "").strip()
    commands = {{"accounts", "apps", "docs", "exec", "answer", "submit", "submit_answer"}}
    i = 0
    account_parts = []
    while i < len(argv) and argv[i] not in commands:
        tok = argv[i]
        if tok in ("-h", "--help", "help"):
            break
        if tok in ("--account", "-a"):
            account_parts = []
            i += 1
            if i >= len(argv) or argv[i] in commands:
                raise SystemExit("--account requires an account ID, name, or email")
            continue
        if tok.startswith("--account="):
            account_parts = [tok.split("=", 1)[1]]
        else:
            account_parts.append(tok)
        i += 1
    if account_parts:
        acct = " ".join(account_parts).strip()
    # Only discovery commands accept trailing options. Execution/answer payloads
    # remain verbatim even when they contain -a or --account.
    cleaned = argv[i:]
    if cleaned and cleaned[0] in ("apps", "docs"):
        if len(cleaned) >= 3 and cleaned[-2] in ("--account", "-a"):
            acct = cleaned[-1].strip()
            cleaned = cleaned[:-2]
        elif cleaned[-1].startswith("--account="):
            acct = cleaned[-1].split("=", 1)[1].strip()
            cleaned = cleaned[:-1]
        elif cleaned[-1] in ("--account", "-a"):
            raise SystemExit("--account requires an account ID, name, or email")

    if not cleaned or cleaned[0] in ("-h", "--help", "help"):
        print("Account: acct_01, a quoted full name (e.g. 'Alex Rivera'), or an email.")
        print("Positional acct_XX shorthand is also supported.")
        print("Usage:")
        print("  aw accounts")
        print("  aw [--account <acct_id_or_name>] apps")
        print("  aw [--account <acct_id_or_name>] docs <app> [api_name]")
        print("  aw --account <acct_id_or_name> exec '<python_code>'   (or '-' for stdin)")
        print("  aw --account <acct_id_or_name> answer '<concise_answer>'  (pass ONLY the bare answer value, e.g. '42', not a full sentence)")
        return

    cmd = cleaned[0]
    if cmd == "accounts":
        print(json.dumps(_post("/accounts", {{}}), indent=2, ensure_ascii=False))
    elif cmd == "apps":
        print(json.dumps(_post("/apps", {{"account_id": acct or None}}), indent=2, ensure_ascii=False))
    elif cmd == "docs":
        app = cleaned[1] if len(cleaned) > 1 else ""
        api = cleaned[2] if len(cleaned) > 2 else None
        print(json.dumps(_post("/docs", {{"account_id": acct or None, "app_name": app, "api_name": api}}), indent=2, ensure_ascii=False))
    elif cmd == "exec":
        raw = " ".join(cleaned[1:]) if len(cleaned) > 1 else "-"
        code = sys.stdin.read() if raw.strip() == "-" else raw
        res = _post("/exec", {{"account_id": acct, "code": code}})
        if not res.get("ok"):
            print("ERROR:", res.get("error"), file=sys.stderr)
            sys.exit(1)
        out = res.get("output", "")
        if out:
            print(out, end="" if str(out).endswith("\\n") else "\\n")
    elif cmd in ("answer", "submit", "submit_answer"):
        ans = " ".join(cleaned[1:]) if len(cleaned) > 1 else sys.stdin.read().strip()
        res = _post("/answer", {{"account_id": acct, "answer": ans}})
        print(json.dumps(res, ensure_ascii=False))
        if not res.get("ok"):
            sys.exit(1)
    else:
        print(f"Unknown subcommand {{cmd!r}}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
"""


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["exec", "eval"], required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--secret", required=True)
    parser.add_argument("--data-dir", default=str(APPWORLD_DATA_DIR))
    args = parser.parse_args()
    if args.mode == "exec":
        _run_exec_server_main(args)
    else:
        _run_eval_worker_main(args)


if __name__ == "__main__":
    _main()
