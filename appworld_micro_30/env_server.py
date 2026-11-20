"""Persistent Multi-Task `AppWorld v0.1.0` HTTP Environment Server & Client Manager.

Server process runs under `.venv_appworld/bin/python` (`APPWORLD_ROOT=<repository root>`), while
`AppWorldMicroServerProcess` and `build_aw_cli_script` can be imported from `.venv/bin/python`.

Key Guarantees:
1. Single persistent `AppWorld` instance per episode (`datetime = 2023-05-18 12:00:00`) via temporary
   merged `/tmp` `dbs/` boot — `data/tasks/` stays 100% read-only and unmodified.
2. Live SQLite state hashing via `ModelHashHandler` (`w.models.model_hashes()`), never calling
   `clear_model_hashes()` or stopping `AppWorld`'s frozen clock mid-episode.
3. Exact `TestTracker` (`evaluation.py`) programmatic scoring per subtask via sibling-app projection,
   preserving all bystander app mutations so collateral side-effects fail strictly.
4. Privileged harness/evaluator endpoints (`/start_episode`, `/set_active_subtask`, `/snapshot`,
   `/save_checkpoint`, `/evaluate_subtask`, `/oracle_smoke`) require `X-AwMicro-Secret` (`HTTP 403` otherwise).
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
import socketserver
import sqlite3
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from typing import Any

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


def parse_concise_qa_answer(raw_val: Any, instruction: str = "") -> Any:
    """Normalize a concise QA answer string/scalar into JSON-compatible Python scalar (`int`, `float`, `str`, `list`, `dict`)."""
    if raw_val is None or isinstance(raw_val, (int, float, bool, list, dict)):
        return raw_val
    text = str(raw_val).strip()
    if not text:
        return None
    # 1. Explicit `FINAL_ANSWER:` or `ANSWER:` tag takes priority
    for line in reversed(text.splitlines()):
        s = line.strip()
        low = s.lower()
        for prefix in ("final_answer:", "final answer:", "answer:"):
            if low.startswith(prefix):
                text = s[len(prefix) :].strip()
                break
    # Strip surrounding matching quotes or markdown codeticks/bold if single-line
    if "\n" not in text:
        if text.startswith("**") and text.endswith("**") and len(text) > 4:
            text = text[2:-2].strip()
        if len(text) >= 2 and (text[0] == text[-1]) and text[0] in ("'", '"', "`"):
            text = text[1:-1].strip()
    if text.lower().rstrip(".!") in ("yes", "no"):
        return text.lower().rstrip(".!")
    num_candidate = text.lstrip("$").replace(",", "").rstrip(".")
    try:
        if "." in num_candidate:
            return float(num_candidate)
        return int(num_candidate)
    except ValueError:
        pass
    try:
        parsed = json.loads(text)
        if isinstance(parsed, (int, float, bool, str, list, dict)):
            return parsed
    except Exception:
        pass

    # 2. Conversational prose fallback when an agent answered in natural language without `./aw answer`
    q_low = (instruction or "").lower().strip()
    if "say yes or no" in q_low:
        m_yn = re.search(r"\b(yes|no)\b", text.lower())
        if m_yn:
            return m_yn.group(1)
    is_numeric_q = (
        q_low.startswith(("how many ", "how much ", "how long "))
        or "in total" in q_low
        or "rounded to" in q_low
        or "in minutes" in q_low
    )
    if is_numeric_q:
        bold_nums = re.findall(r"\*\*\$?(-?\d+(?:,\d{3})*(?:\.\d+)?)\b", text)
        nums = bold_nums or re.findall(r"(?<![A-Za-z0-9_-])\$?(-?\d+(?:,\d{3})*(?:\.\d+)?)(?![A-Za-z0-9_])", text)
        if nums:
            raw_num = nums[0].replace(",", "")
            try:
                return float(raw_num) if "." in raw_num else int(raw_num)
            except ValueError:
                pass
    if any(k in q_low for k in ("title of", "just the quote", "which genre")):
        quoted = re.findall(r'"([^"\n]{1,120})"', text) or re.findall(r"\*\*([^*\n]{1,120})\*\*", text)
        if len(quoted) == 1:
            return quoted[0].strip()
    return text


class _AppWorldEpisodeRuntime:
    """In-memory state inside `.venv_appworld` server process."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.lock = threading.RLock()
        self.world: Any = None
        self.temp_dir: Path | None = None
        self.exp_name: str | None = None
        self.episode_id: str | None = None
        self.subtasks: dict[str, dict[str, Any]] = {}
        self.subtask_order: list[str] = []
        self.current_subtask_tag: str | None = None
        self.initial_raw_model_hashes: dict[str, Any] = {}
        self.initial_app_hashes: dict[str, str] = {}
        self.exec_log: list[dict[str, Any]] = []
        self.checkpoints: dict[str, dict[str, Any]] = {}
        self.subtask_submitted_answers: dict[str, list[Any]] = {}

    def close(self) -> None:
        with self.lock:
            if self.world is not None:
                try:
                    self.world.close()
                except Exception:
                    pass
                self.world = None
            try:
                from appworld.apps.model_lib import CachedDBHandler
                from appworld.environment import AppWorld

                AppWorld.close_all()
                CachedDBHandler.reset()
            except Exception:
                pass
            if self.exp_name:
                exp_out = PROJECT_ROOT / "experiments" / "outputs" / self.exp_name
                if exp_out.exists():
                    shutil.rmtree(exp_out, ignore_errors=True)
                self.exp_name = None
            if self.temp_dir is not None and self.temp_dir.exists():
                shutil.rmtree(self.temp_dir, ignore_errors=True)
            self.temp_dir = None
            self.episode_id = None
            self.subtasks.clear()
            self.subtask_order.clear()
            self.current_subtask_tag = None
            self.initial_raw_model_hashes.clear()
            self.initial_app_hashes.clear()
            self.exec_log.clear()
            self.checkpoints.clear()
            self.subtask_submitted_answers.clear()
            gc.collect()

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

    def start_episode(self, episode_id: str, subtasks: list[dict[str, Any]]) -> dict[str, Any]:
        import appworld.task as aw_task_mod
        from appworld.environment import AppWorld

        with self.lock:
            self.close()
            if len(subtasks) != 2:
                raise ValueError(f"Expected exactly 2 subtasks per Micro-30 episode, got {len(subtasks)}")

            self.temp_dir = Path(tempfile.mkdtemp(prefix=f"aw_micro_{episode_id}_"))
            merged_dbs_dir = self.temp_dir / "merged_input_dbs"
            sub_a, sub_b = subtasks[0], subtasks[1]
            tid_a = str(sub_a["upstream_task_id"])
            tid_b = str(sub_b["upstream_task_id"])

            src_dbs_a = self.data_dir / "tasks" / tid_a / "dbs"
            src_dbs_b = self.data_dir / "tasks" / tid_b / "dbs"
            shutil.copytree(src_dbs_a, merged_dbs_dir)

            for app in USER_APP_NAMES:
                b_jsonl = src_dbs_b / f"{app}.jsonl"
                if b_jsonl.is_file() and b_jsonl.read_text(encoding="utf-8").strip() != "":
                    a_jsonl = merged_dbs_dir / f"{app}.jsonl"
                    if a_jsonl.exists() and a_jsonl.read_text(encoding="utf-8").strip() != "":
                        raise RuntimeError(
                            f"DB delta collision on app {app!r} between {tid_a} and {tid_b}!"
                        )
                    shutil.copy2(b_jsonl, a_jsonl)

            orig_get_db_home_path = aw_task_mod.get_db_home_path

            def _patched_get_db_home_path(
                storage_type: str,
                type: str,
                task_id: str | None = None,
                **kwargs: Any,
            ) -> str:
                if storage_type == "disk" and type == "task_input" and task_id == tid_a:
                    return str(merged_dbs_dir)
                return orig_get_db_home_path(
                    storage_type=storage_type,
                    type=type,
                    task_id=task_id,
                    **kwargs,
                )

            aw_task_mod.get_db_home_path = _patched_get_db_home_path
            try:
                exp_name = f"aw_micro_live_{episode_id}_{uuid.uuid4().hex[:8]}"
                self.exp_name = exp_name
                self.world = AppWorld(
                    task_id=tid_a,
                    experiment_name=exp_name,
                    raise_on_failure=False,
                    ground_truth_mode="minimal",
                )
            finally:
                aw_task_mod.get_db_home_path = orig_get_db_home_path

            self.episode_id = episode_id
            self.subtasks = {str(s["subtask_tag"]): dict(s) for s in subtasks}
            self.subtask_order = [str(s["subtask_tag"]) for s in subtasks]
            self.subtask_submitted_answers = {tag: [] for tag in self.subtask_order}
            self._set_active_subtask_unlocked(self.subtask_order[0])

            # Read live SQLAlchemy-tracked model hashes WITHOUT calling `clear_model_hashes()`!
            self.initial_raw_model_hashes = dict(self.world.models.model_hashes())
            self.initial_app_hashes = self._compute_app_hashes(self.initial_raw_model_hashes)
            return {
                "ok": True,
                "episode_id": episode_id,
                "subtask_tags": self.subtask_order,
                "initial_app_hashes": self.initial_app_hashes,
            }

    def _set_active_subtask_unlocked(self, subtask_tag: str) -> dict[str, Any]:
        if self.world is None:
            raise RuntimeError("No active AppWorld episode.")
        if subtask_tag not in self.subtasks:
            raise ValueError(f"Unknown subtask_tag {subtask_tag!r}")
        self.current_subtask_tag = subtask_tag
        st = self.subtasks[subtask_tag]
        active_tasks = self.world.models.supervisor.Task.all()
        if active_tasks:
            t0 = active_tasks[0]
            t0.instruction = str(st["instruction"])
            t0.answer = '"<<NOT_GIVEN>>"'
            t0.status = None
            t0.save()
        return {"ok": True, "active_subtask_tag": subtask_tag}

    def set_active_subtask(self, subtask_tag: str) -> dict[str, Any]:
        with self.lock:
            return self._set_active_subtask_unlocked(subtask_tag)

    def list_apps(self) -> dict[str, Any]:
        with self.lock:
            if self.world is None:
                raise RuntimeError("No active AppWorld episode.")
            docs = self.world.task.api_docs
            apps = []
            for app_name in docs.keys():
                apps.append({"app_name": app_name, "api_count": len(docs[app_name])})
            return {"ok": True, "apps": apps}

    def get_docs(self, app_name: str, api_name: str | None = None) -> dict[str, Any]:
        with self.lock:
            if self.world is None:
                raise RuntimeError("No active AppWorld episode.")
            docs = self.world.task.api_docs
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
                k: {
                    "description": v.get("description", ""),
                    "parameters": v.get("parameters", []),
                }
                for k, v in app_docs.items()
            }
            return {"ok": True, "app": app_name, "apis": summary}

    def _capture_supervisor_answer_if_any_unlocked(self) -> Any:
        active_tasks = self.world.models.supervisor.Task.all()
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
            if parsed != "<<NOT_GIVEN>>":
                captured = parsed
                tag = self.current_subtask_tag or self.subtask_order[0]
                self.subtask_submitted_answers.setdefault(tag, []).append(parsed)
            # Reset answer slot so a subsequent subtask in the same episode starts clean
            t0.answer = '"<<NOT_GIVEN>>"'
            t0.status = None
            t0.save()
        return captured

    def execute_code(self, code: str) -> dict[str, Any]:
        with self.lock:
            if self.world is None:
                raise RuntimeError("No active AppWorld episode.")
            before_hashes = self._compute_app_hashes(dict(self.world.models.model_hashes()))
            output = self.world.execute(code)
            captured_answer = self._capture_supervisor_answer_if_any_unlocked()
            after_hashes = self._compute_app_hashes(dict(self.world.models.model_hashes()))
            mutated_step = [a for a in USER_APP_NAMES if before_hashes[a] != after_hashes[a]]
            mutated_since_start = [
                a for a in USER_APP_NAMES if self.initial_app_hashes[a] != after_hashes[a]
            ]
            entry = {
                "seq": len(self.exec_log) + 1,
                "active_subtask_tag": self.current_subtask_tag,
                "code": code,
                "output": output,
                "captured_answer": captured_answer,
                "mutated_apps_in_step": mutated_step,
                "mutated_apps_since_start": mutated_since_start,
            }
            self.exec_log.append(entry)
            return {
                "ok": True,
                "output": output,
                "captured_answer": captured_answer,
                "mutated_apps_in_step": mutated_step,
            }

    def submit_answer(self, raw_answer: Any, subtask_tag: str | None = None) -> dict[str, Any]:
        with self.lock:
            if self.world is None:
                raise RuntimeError("No active AppWorld episode.")
            tag = subtask_tag or self.current_subtask_tag or self.subtask_order[0]
            instr = str((self.subtasks.get(tag) or {}).get("instruction", ""))
            parsed = parse_concise_qa_answer(raw_answer, instruction=instr)
            self.subtask_submitted_answers.setdefault(tag, []).append(parsed)
            return {"ok": True, "subtask_tag": tag, "recorded_answer": parsed}

    def snapshot_unlocked(self, include_exec_log: bool = False) -> dict[str, Any]:
        if self.world is None:
            raise RuntimeError("No active AppWorld episode.")
        cur_raw = dict(self.world.models.model_hashes())
        cur_app = self._compute_app_hashes(cur_raw)
        mutated_apps = [a for a in USER_APP_NAMES if cur_app[a] != self.initial_app_hashes[a]]
        out: dict[str, Any] = {
            "ok": True,
            "episode_id": self.episode_id,
            "initial_app_hashes": dict(self.initial_app_hashes),
            "current_app_hashes": cur_app,
            "mutated_user_apps": mutated_apps,
            "unmodified_since_start": len(mutated_apps) == 0,
            "exec_count": len(self.exec_log),
            "exec_log_len": len(self.exec_log),
            "subtask_submitted_answers": {k: list(v) for k, v in self.subtask_submitted_answers.items()},
        }
        if include_exec_log:
            out["exec_log"] = list(self.exec_log)
        return out

    def snapshot(self, include_exec_log: bool = True) -> dict[str, Any]:
        with self.lock:
            return self.snapshot_unlocked(include_exec_log=include_exec_log)

    def save_checkpoint(self, checkpoint_id: str) -> dict[str, Any]:
        with self.lock:
            snap = self.snapshot_unlocked()
            self.checkpoints[checkpoint_id] = snap
            return {"ok": True, "checkpoint_id": checkpoint_id, "snapshot": snap}

    def get_checkpoint(self, checkpoint_id: str) -> dict[str, Any]:
        with self.lock:
            snap = self.checkpoints.get(checkpoint_id)
            if snap is None:
                return {"ok": False, "error": f"Missing checkpoint {checkpoint_id!r}"}
            return {"ok": True, "checkpoint_id": checkpoint_id, "snapshot": snap}

    def _effective_action_apps_unlocked(self, subtask_tag: str) -> set[str]:
        """Return all user apps legitimately mutated by a non-QA subtask, including cross-app side effects
        explicitly asserted in upstream `ground_truth/evaluation.py` (e.g. `amazon.place_order` sending receipt
        emails/PDFs into `gmail` and `file_system`). Read-only QA subtasks return `set()`."""
        st = self.subtasks[subtask_tag]
        if bool(st.get("is_qa")):
            return set()
        apps: set[str] = set(st.get("target_apps") or [])
        tid = str(st["upstream_task_id"])
        eval_fp = self.data_dir / "tasks" / tid / "ground_truth" / "evaluation.py"
        if eval_fp.is_file():
            ev_src = eval_fp.read_text(encoding="utf-8")
            for app in USER_APP_NAMES:
                if (
                    f"models.{app}." in ev_src
                    or f"apis.{app}." in ev_src
                    or f'"{app}.' in ev_src
                    or f"'{app}." in ev_src
                ):
                    apps.add(app)
        return apps

    def evaluate_subtask(
        self,
        subtask_tag: str,
        candidate_answers: list[Any] | None = None,
    ) -> dict[str, Any]:
        """Evaluate `subtask_tag` against upstream `AppWorld` `evaluation.py` (`TestTracker`)
        without resetting or unfreezing the open `AppWorld`."""
        from appworld.apps.api_lib import set_local_dbs
        from appworld.apps.model_lib import CachedDBHandler
        from appworld.collections.models import ModelCollection, ModelCollectionPair
        from appworld.evaluator import TestTracker
        from appworld.task import Task

        with self.lock:
            if self.world is None or self.temp_dir is None:
                raise RuntimeError("No active AppWorld episode.")
            if subtask_tag not in self.subtasks:
                raise ValueError(f"Unknown subtask_tag {subtask_tag!r}")

            st = self.subtasks[subtask_tag]
            tid = str(st["upstream_task_id"])
            is_qa = bool(st["is_qa"])
            other_tags = [t for t in self.subtask_order if t != subtask_tag]
            sibling_allowed_apps: set[str] = set()
            for ot in other_tags:
                sibling_allowed_apps.update(self._effective_action_apps_unlocked(ot))
            this_target_apps = self._effective_action_apps_unlocked(subtask_tag)
            bystander_apps = set(USER_APP_NAMES) - this_target_apps - sibling_allowed_apps

            cur_snap = self.snapshot_unlocked()
            mutated_all = set(cur_snap["mutated_user_apps"])
            bystander_collateral = sorted(mutated_all & bystander_apps)

            eval_scratch = self.temp_dir / f"eval_{subtask_tag}_{uuid.uuid4().hex[:8]}"
            end_raw_dir = eval_scratch / "end_raw"
            start_task_dir = eval_scratch / "start_task"
            proj_end_dir = eval_scratch / "proj_end"
            eval_scratch.mkdir(parents=True, exist_ok=True)

            # 1. Save live `self.world.models` BEFORE `Task.load` mutates global SQLModel bindings!
            self.world.models.save(
                str(end_raw_dir),
                format="full",
                delete_if_exists=True,
                save_model_hashes=True,
            )

            mem_tag = f":memory:aw_micro_eval_{subtask_tag}_{uuid.uuid4().hex[:8]}"
            start_mem_tag: str | None = None
            try:
                # 2. Load isolated upstream `Task` (`ground_truth_mode="minimal"` works across all splits)
                task_obj = Task.load(task_id=tid, ground_truth_mode="minimal")
                start_mem_tag = task_obj.model_collection.to_db_home_path
                task_obj.model_collection.save(
                    str(start_task_dir),
                    format="full",
                    delete_if_exists=True,
                    save_model_hashes=True,
                )

                # 3. Project `end_raw_dir` -> `proj_end_dir`, restoring ONLY `sibling_allowed_apps` and `supervisor`
                shutil.copytree(end_raw_dir, proj_end_dir)
                end_hashes_fp = proj_end_dir / "model_hashes.json"
                start_hashes_fp = start_task_dir / "model_hashes.json"
                end_hashes = json.loads(end_hashes_fp.read_text(encoding="utf-8"))
                start_hashes = json.loads(start_hashes_fp.read_text(encoding="utf-8"))

                for sib_app in sorted(sibling_allowed_apps | {"supervisor"}):
                    src_db = start_task_dir / f"{sib_app}.db"
                    dst_db = proj_end_dir / f"{sib_app}.db"
                    if src_db.is_file():
                        shutil.copy2(src_db, dst_db)
                    for hk, hv in start_hashes.items():
                        if hk == sib_app or hk.startswith(f"{sib_app}."):
                            end_hashes[hk] = hv

                end_hashes_fp.write_text(json.dumps(end_hashes), encoding="utf-8")

                # 4. Populate `proj_end_dir/supervisor.db` `tasks` row fairly for Action vs QA tasks
                resolved_answer: Any = None
                has_submitted_answer = False
                explicit_list = [
                    x for x in self.subtask_submitted_answers.get(subtask_tag, []) if x is not None
                ]
                if explicit_list:
                    resolved_answer = explicit_list[-1]
                    has_submitted_answer = True
                elif candidate_answers:
                    instr_str = str(st.get("instruction", ""))
                    for cand in candidate_answers:
                        parsed = parse_concise_qa_answer(cand, instruction=instr_str)
                        if parsed is not None:
                            resolved_answer = parsed
                            has_submitted_answer = True
                            break

                sup_db_fp = proj_end_dir / "supervisor.db"
                conn = sqlite3.connect(str(sup_db_fp))
                try:
                    with conn:
                        if not is_qa:
                            conn.execute("UPDATE tasks SET answer = 'null', status = 'success'")
                        elif has_submitted_answer:
                            conn.execute(
                                "UPDATE tasks SET answer = ?, status = 'success'",
                                (json.dumps(resolved_answer),),
                            )
                finally:
                    conn.close()

                # 5. Run upstream `evaluation_module.evaluate(...)` via `TestTracker`
                models_start = task_obj.model_collection
                models_end = ModelCollection.load(
                    to_db_home_path=mem_tag,
                    from_db_home_path=str(proj_end_dir),
                    load_apps=task_obj.allowed_apps,
                )
                pair = ModelCollectionPair(
                    start_db_home_path=models_start.from_db_home_path,
                    start_model_collection=models_start,
                    end_db_home_path=mem_tag,
                    end_model_collection=models_end,
                )
                gt = task_obj.ground_truth
                eval_mod = gt.evaluation_module()
                main_user = pair.start.admin.MainUser.find_one(**task_obj.supervisor)
                tracker = TestTracker(
                    test_data=gt.test_data,
                    difficulty=gt.metadata["difficulty"],
                    suppress_errors=True,
                )
                eval_mod.evaluate(
                    test=tracker,
                    public_data=gt.public_data,
                    private_data=gt.private_data,
                    main_user=main_user,
                    models=pair,
                    ground_truth_answer=gt.answer,
                )
                tracker_success = bool(tracker.success)
                passed = tracker_success and (len(bystander_collateral) == 0)
                return {
                    "ok": True,
                    "subtask_tag": subtask_tag,
                    "upstream_task_id": tid,
                    "is_qa": is_qa,
                    "passed": passed,
                    "tracker_success": tracker_success,
                    "pass_count": tracker.pass_count,
                    "num_tests": tracker.num_tests,
                    "pass_percentage": tracker.pass_percentage,
                    "bystander_collateral_apps": bystander_collateral,
                    "resolved_qa_answer": resolved_answer if is_qa else None,
                    "failures": [
                        {"requirement": f["requirement"], "trace": f["trace"][:600]}
                        for f in tracker.failures
                    ],
                }
            finally:
                # Re-bind global SQLAlchemy models back to the live `AppWorld` memory DB (`from_db_home_path=None`, `app_names=None`)
                set_local_dbs(to_db_home_path=self.world.models_to_db_home_path)
                CachedDBHandler.reset(mem_tag)
                if start_mem_tag and start_mem_tag != self.world.models_to_db_home_path:
                    CachedDBHandler.reset(start_mem_tag)
                shutil.rmtree(eval_scratch, ignore_errors=True)

    def oracle_smoke(self, checkpoint_after_first_if_qa: str | None = None) -> dict[str, Any]:
        """Execute official `ground_truth/compiled_solution.py` sequentially for both subtasks inside the single live `AppWorld`."""
        with self.lock:
            if self.world is None:
                raise RuntimeError("No active AppWorld episode.")
            steps: list[dict[str, Any]] = []
            for idx, tag in enumerate(self.subtask_order):
                st = self.subtasks[tag]
                tid = str(st["upstream_task_id"])
                sol_fp = self.data_dir / "tasks" / tid / "ground_truth" / "compiled_solution.py"
                if not sol_fp.is_file():
                    return {
                        "ok": False,
                        "error": f"Upstream task {tid} ({st.get('split')}) does not include compiled_solution.py",
                    }
                self._set_active_subtask_unlocked(tag)
                code = sol_fp.read_text(encoding="utf-8") + "\nsolution(apis, requester)\n"
                exec_res = self.execute_code(code)
                steps.append({"subtask_tag": tag, "upstream_task_id": tid, "exec": exec_res})
                if idx == 0 and checkpoint_after_first_if_qa:
                    self.save_checkpoint(checkpoint_after_first_if_qa)
            return {"ok": True, "steps": steps, "snapshot": self.snapshot_unlocked()}


class _MainThreadHTTPServer(http.server.HTTPServer):
    """Single-threaded HTTPServer so `AppWorld.execute()` runs on the main thread (`SIGALRM` + SQLite affinity)."""

    allow_reuse_address = True


def _run_server_main() -> None:
    parser = argparse.ArgumentParser(description="AppWorld-Micro-30 persistent HTTP server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--secret", required=True)
    parser.add_argument("--data-dir", default=str(APPWORLD_DATA_DIR))
    args = parser.parse_args()

    os.environ.setdefault("APPWORLD_ROOT", str(Path(args.data_dir).resolve().parent))
    runtime = _AppWorldEpisodeRuntime(Path(args.data_dir).resolve())
    expected_secret = args.secret

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, format: str, *log_args: Any) -> None:
            return

        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _check_secret(self) -> bool:
            got = self.headers.get("X-AwMicro-Secret", "")
            if not secrets.compare_digest(got, expected_secret):
                self._send_json(403, {"ok": False, "error": "Forbidden: invalid or missing X-AwMicro-Secret"})
                return False
            return True

        def _read_body_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            return json.loads(raw.decode("utf-8"))

        def do_GET(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            qs = urllib.parse.parse_qs(parsed.query)
            try:
                if parsed.path == "/health":
                    self._send_json(200, {"ok": True})
                elif parsed.path == "/apps":
                    self._send_json(200, runtime.list_apps())
                elif parsed.path == "/docs":
                    app_name = (qs.get("app") or [""])[0]
                    api_name = (qs.get("api") or [None])[0]
                    self._send_json(200, runtime.get_docs(app_name, api_name))
                elif parsed.path == "/snapshot":
                    if not self._check_secret():
                        return
                    inc_exec = (qs.get("include_exec_log") or ["1"])[0] not in ("0", "false", "False")
                    self._send_json(200, runtime.snapshot(include_exec_log=inc_exec))
                elif parsed.path == "/checkpoint":
                    if not self._check_secret():
                        return
                    cid = (qs.get("id") or [""])[0]
                    self._send_json(200, runtime.get_checkpoint(cid))
                else:
                    self._send_json(404, {"ok": False, "error": f"Not found: {parsed.path}"})
            except Exception as exc:
                self._send_json(500, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})

        def do_POST(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            try:
                body = self._read_body_json()
                if parsed.path == "/exec":
                    code = str(body.get("code", ""))
                    self._send_json(200, runtime.execute_code(code))
                elif parsed.path == "/submit_answer":
                    self._send_json(200, runtime.submit_answer(body.get("answer")))
                elif parsed.path == "/start_episode":
                    if not self._check_secret():
                        return
                    self._send_json(
                        200,
                        runtime.start_episode(
                            episode_id=str(body["episode_id"]),
                            subtasks=list(body["subtasks"]),
                        ),
                    )
                elif parsed.path == "/set_active_subtask":
                    if not self._check_secret():
                        return
                    self._send_json(200, runtime.set_active_subtask(str(body["subtask_tag"])))
                elif parsed.path == "/save_checkpoint":
                    if not self._check_secret():
                        return
                    self._send_json(200, runtime.save_checkpoint(str(body["checkpoint_id"])))
                elif parsed.path == "/evaluate_subtask":
                    if not self._check_secret():
                        return
                    self._send_json(
                        200,
                        runtime.evaluate_subtask(
                            subtask_tag=str(body["subtask_tag"]),
                            candidate_answers=body.get("candidate_answers"),
                        ),
                    )
                elif parsed.path == "/oracle_smoke":
                    if not self._check_secret():
                        return
                    self._send_json(
                        200,
                        runtime.oracle_smoke(
                            checkpoint_after_first_if_qa=body.get("checkpoint_after_first_if_qa"),
                        ),
                    )
                elif parsed.path == "/close":
                    if not self._check_secret():
                        return
                    runtime.close()
                    self._send_json(200, {"ok": True})
                else:
                    self._send_json(404, {"ok": False, "error": f"Not found: {parsed.path}"})
            except Exception as exc:
                self._send_json(500, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    srv = _MainThreadHTTPServer((args.host, args.port), Handler)
    try:
        srv.serve_forever()
    finally:
        runtime.close()


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def build_aw_cli_script(base_url: str) -> str:
    """Generate the unprivileged `./aw` CLI script placed in the agent workspace (contains ZERO harness secrets)."""
    return f"""#!/usr/bin/env python3
import json
import sys
import urllib.parse
import urllib.request

BASE = {base_url!r}


def _req(method: str, path: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {{"Content-Type": "application/json"}} if data is not None else {{}}
    req = urllib.request.Request(BASE + path, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
        print(
            "Usage:\\n"
            "  ./aw apps                       # List all available personal apps\\n"
            "  ./aw docs <app> [api_name]      # Inspect API documentation for <app>\\n"
            "  ./aw exec '<python_code>'       # Execute Python in persistent AppWorld REPL (`apis.<app>.<fn>(...)`)\\n"
            "  ./aw exec -                     # Read Python code from stdin and execute\\n"
            "  ./aw answer '<concise_answer>'  # Record concise scalar/string answer for a read-only question"
        )
        return
    cmd = sys.argv[1]
    if cmd == "apps":
        print(json.dumps(_req("GET", "/apps"), indent=2, ensure_ascii=False))
    elif cmd == "docs":
        if len(sys.argv) < 3:
            sys.exit("Usage: ./aw docs <app> [api_name]")
        q = {{"app": sys.argv[2]}}
        if len(sys.argv) >= 4:
            q["api"] = sys.argv[3]
        print(json.dumps(_req("GET", "/docs?" + urllib.parse.urlencode(q)), indent=2, ensure_ascii=False))
    elif cmd == "exec":
        if len(sys.argv) >= 3 and sys.argv[2] != "-":
            code = " ".join(sys.argv[2:])
        else:
            code = sys.stdin.read()
        res = _req("POST", "/exec", {{"code": code}})
        out = res.get("output", "")
        if out:
            print(out, end="" if out.endswith("\\n") else "\\n")
        if not res.get("ok", False):
            sys.exit(1)
    elif cmd == "answer":
        if len(sys.argv) < 3:
            sys.exit("Usage: ./aw answer '<concise_answer>'")
        ans = " ".join(sys.argv[2:])
        print(json.dumps(_req("POST", "/submit_answer", {{"answer": ans}}), ensure_ascii=False))
    else:
        sys.exit(f"Unknown subcommand: {{cmd!r}}")


if __name__ == "__main__":
    main()
"""


_AW_SERVER_BOOT_SEM = threading.BoundedSemaphore(8)
_AW_HEAVY_SQLITE_IO_SEM = threading.BoundedSemaphore(4)


class AppWorldMicroServerProcess:
    """Parent-side lifecycle manager spawning `env_server.py` under `.venv_appworld/bin/python`."""

    def __init__(
        self,
        port: int | None = None,
        log_path: Path | None = None,
        secret: str | None = None,
        appworld_python: Path = APPWORLD_VENV_PYTHON,
    ) -> None:
        self.port = port or _pick_free_port()
        self.secret = secret or secrets.token_hex(24)
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.log_path = log_path
        self.appworld_python = appworld_python
        self._proc: subprocess.Popen[str] | None = None
        self._log_fh: Any = None

    def start(self, timeout_sec: float = 45.0) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        with _AW_SERVER_BOOT_SEM:
            if self._proc is not None and self._proc.poll() is None:
                return
            env = os.environ.copy()
            env["APPWORLD_ROOT"] = str(PROJECT_ROOT)
            env["PYTHONPATH"] = f"{PROJECT_ROOT}:{env.get('PYTHONPATH', '')}".rstrip(":")
            if self.log_path is not None:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                self._log_fh = self.log_path.open("w", encoding="utf-8")
            cmd = [
                str(self.appworld_python),
                str(Path(__file__).resolve()),
                "--port",
                str(self.port),
                "--secret",
                self.secret,
            ]
            self._proc = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_ROOT),
                env=env,
                stdout=self._log_fh or subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
                text=True,
            )
            deadline = time.monotonic() + timeout_sec
            while time.monotonic() < deadline:
                if self._proc.poll() is not None:
                    raise RuntimeError(f"AppWorldMicroServerProcess exited early with code {self._proc.returncode}")
                try:
                    res = self._request("GET", "/health", auth=False, timeout=2.0)
                    if res.get("ok"):
                        return
                except Exception:
                    time.sleep(0.15)
            self.stop()
            raise TimeoutError(f"Timed out waiting for AppWorldMicroServerProcess on {self.base_url}")

    def stop(self) -> None:
        if self._proc is not None:
            try:
                self._request("POST", "/close", body={}, auth=True, timeout=3.0)
            except Exception:
                pass
            if self._proc.poll() is None:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
            self._proc = None
        if self._log_fh is not None:
            try:
                self._log_fh.close()
            except Exception:
                pass
            self._log_fh = None

    def _request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        auth: bool = True,
        timeout: float = 60.0,
    ) -> dict[str, Any]:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        headers: dict[str, str] = {}
        if data is not None:
            headers["Content-Type"] = "application/json"
        if auth:
            headers["X-AwMicro-Secret"] = self.secret
        req = urllib.request.Request(self.base_url + path, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as err:
            raw_err = err.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {err.code} on {method} {path}: {raw_err}") from err

    def start_episode(self, episode_id: str, subtasks: list[dict[str, Any]]) -> dict[str, Any]:
        self.start()
        with _AW_HEAVY_SQLITE_IO_SEM:
            return self._request(
                "POST",
                "/start_episode",
                {"episode_id": episode_id, "subtasks": subtasks},
                timeout=180.0,
            )

    def close_episode(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._request("POST", "/close", body={}, auth=True, timeout=15.0)
            except Exception:
                self.stop()

    def set_active_subtask(self, subtask_tag: str) -> dict[str, Any]:
        return self._request("POST", "/set_active_subtask", {"subtask_tag": subtask_tag})

    def execute_code(self, code: str) -> dict[str, Any]:
        return self._request("POST", "/exec", {"code": code}, auth=False, timeout=120.0)

    def submit_answer(self, answer: Any) -> dict[str, Any]:
        return self._request("POST", "/submit_answer", {"answer": answer}, auth=False)

    def snapshot(self, include_exec_log: bool = True) -> dict[str, Any]:
        flag = "1" if include_exec_log else "0"
        return self._request("GET", f"/snapshot?include_exec_log={flag}")

    def save_checkpoint(self, checkpoint_id: str) -> dict[str, Any]:
        return self._request("POST", "/save_checkpoint", {"checkpoint_id": checkpoint_id})

    def get_checkpoint(self, checkpoint_id: str) -> dict[str, Any]:
        q = urllib.parse.urlencode({"id": checkpoint_id})
        return self._request("GET", f"/checkpoint?{q}")

    def evaluate_subtask(
        self,
        subtask_tag: str,
        candidate_answers: list[Any] | None = None,
    ) -> dict[str, Any]:
        with _AW_HEAVY_SQLITE_IO_SEM:
            return self._request(
                "POST",
                "/evaluate_subtask",
                {"subtask_tag": subtask_tag, "candidate_answers": candidate_answers or []},
                timeout=180.0,
            )

    def oracle_smoke(self, checkpoint_after_first_if_qa: str | None = None) -> dict[str, Any]:
        return self._request(
            "POST",
            "/oracle_smoke",
            {"checkpoint_after_first_if_qa": checkpoint_after_first_if_qa},
            timeout=120.0,
        )


if __name__ == "__main__":
    _run_server_main()
