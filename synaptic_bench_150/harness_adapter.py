"""Architecture-neutral black-box `AgentHarnessAdapter` interface & `ControlledUserSimulator` (`v2.0.0`).

Enables external Personal / Coding Agent harnesses (e.g., Claude Code, Codex CLI, OpenClaw,
custom ReAct scripts) to run on `Synaptic-Bench-150` without coupling to internal `Synaptic-PA`
or `dsh_superagent` data structures, and enforces genuine interactive clarification in `Ambig-SWE`.
"""
from __future__ import annotations

import abc
import asyncio
import json
import os
import re
import shlex
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .schema import EpisodeSpec, TimelineEventSpec

if TYPE_CHECKING:
    from .verifiers import RuntimeArtifacts
    from .workspace_manager import MaterializedEpisodeWorkspace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AMBIG_SWE_POOL_JSONL = PROJECT_ROOT / "data" / "upstream_pillars" / "pools" / "ambig_swe_joined_pool.jsonl"

_SPEC_REF_CODE_RE = re.compile(r"\bPA150-SPEC-\d+(?:-[A-F0-9]{4})?\b|\bSPEC-REF-[A-Za-z0-9_-]+\b")
_CLARIFICATION_CUE_RE = re.compile(
    r"(?:\?|？|\b(?:should\s+(?:we|i|it|the)|whether\s+to|could\s+you\s+clarify|please\s+clarify|"
    r"please\s+confirm|do\s+you\s+want|which\s+(?:behavior|exception|error|option|format))|"
    r"请问|请确认|需要确认|跟您确认|和你确认|是否应|是否需要|希望采用哪|应该抛出什么|怎么处理)",
    flags=re.IGNORECASE,
)
_NO_ACTION_REPLY_RE = re.compile(
    r"^\s*(?:\[(?:NO_ACTION|IGNORED|NOISE_DROPPED|SKIP)\]|(?:no\s+action\s+(?:needed|taken)|ignored\s+noise)\.?)\s*$",
    flags=re.IGNORECASE,
)
_DUP_MERGED_REPLY_RE = re.compile(
    r"(?:\[DUPLICATE_MERGED\]|\bduplicate\s+(?:event|webhook|ticket|issue)\b|已合并重复|同一事项已跟进)",
    flags=re.IGNORECASE,
)


def parse_cli_arm_spec(raw_arm: str) -> tuple[str, str] | None:
    """Parse `cli:<arm_name>:<shell_command>` into `(arm_name, shell_command)`, or return `None`."""
    s = (raw_arm or "").strip()
    if not s.startswith("cli:"):
        return None
    parts = s.split(":", 2)
    if len(parts) != 3 or not parts[1].strip() or not parts[2].strip():
        raise ValueError(
            f"Invalid CLI adapter spec {raw_arm!r}; expected `cli:<arm_name>:<shell_command_template>`."
        )
    arm_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", parts[1].strip())
    return arm_name, parts[2].strip()


class AgentHarnessAdapter(abc.ABC):
    """Open-source black-box adapter contract for evaluating arbitrary agent harnesses on `Synaptic-Bench-150`."""

    arm_name: str

    @abc.abstractmethod
    async def setup(self, episode: EpisodeSpec, workspace: "MaterializedEpisodeWorkspace") -> None:
        """Initialize agent state against the materialized episode workspace."""

    @abc.abstractmethod
    async def deliver_event(
        self,
        event: TimelineEventSpec,
        workspace: "MaterializedEpisodeWorkspace",
    ) -> dict[str, Any]:
        """Deliver a single omnichannel `TimelineEventSpec` to the agent harness."""

    @abc.abstractmethod
    async def wait_until_quiescent(self, timeout_sec: float = 360.0) -> None:
        """Block until all in-flight background work spawned so far has settled."""

    @abc.abstractmethod
    def count_executed_tool_calls(self) -> int:
        """Return cumulative tool/command executions observed so far (for milestone hooks)."""

    @abc.abstractmethod
    def latest_agent_replies(self) -> list[str]:
        """Return user-visible replies emitted so far (for `ControlledUserSimulator` inspection)."""

    @abc.abstractmethod
    async def collect_runtime_artifacts(
        self,
        workspace: "MaterializedEpisodeWorkspace",
    ) -> "RuntimeArtifacts":
        """Collect final workspace worktrees, replies, and standardized evaluation telemetry."""

    async def teardown(self) -> None:
        """Release any background subprocesses or sockets owned by the adapter."""
        return None


class SubprocessCLIHarnessAdapter(AgentHarnessAdapter):
    """Generic black-box subprocess adapter for CLI agents (Claude Code, Codex CLI, OpenClaw, custom ReAct).

    For each `TimelineEventSpec`, invokes `command_template` with `cwd = workspace.repo_dir`,
    passing event metadata in `PA150_EVENT_JSON` / `PA150_EVENT_ID` / `PA150_EVENT_KIND` and
    sending `event.text` over `stdin` (or substituting `{event_text}` if present in the template).
    """

    def __init__(self, arm_name: str, command_template: str, per_event_timeout_sec: float = 300.0) -> None:
        self.arm_name = arm_name
        self.command_template = command_template
        self.per_event_timeout_sec = per_event_timeout_sec
        self._episode: EpisodeSpec | None = None
        self._replies: list[str] = []
        self._executed_blobs: list[str] = []
        self._tool_calls: int = 0
        self._noise_total: int = 0
        self._noise_dropped: int = 0
        self._noise_spawned: int = 0
        self._noise_mutated: int = 0
        self._dup_merged: int = 0
        self._dup_spawned: int = 0
        self._real_tasks_dropped: int = 0
        self._peek_latencies_ms: list[float] = []
        self._prompt_tokens: int = 0
        self._cache_hit_tokens: int = 0
        self._output_tokens: int = 0
        self._cost_cny: float = 0.0
        self._office_inspected: bool = False
        self._spec_codes_seen: set[str] = set()

    async def setup(self, episode: EpisodeSpec, workspace: "MaterializedEpisodeWorkspace") -> None:
        self._episode = episode
        self._noise_total = sum(1 for e in episode.timeline if e.kind == "external_noise")

    def count_executed_tool_calls(self) -> int:
        return self._tool_calls

    def latest_agent_replies(self) -> list[str]:
        return list(self._replies)

    async def wait_until_quiescent(self, timeout_sec: float = 360.0) -> None:
        # Subprocess CLI invocations settle per event unless extended by a subclass daemon.
        return None

    async def deliver_event(
        self,
        event: TimelineEventSpec,
        workspace: "MaterializedEpisodeWorkspace",
    ) -> dict[str, Any]:
        from .workspace_manager import workspace_has_any_mutation_since_snapshot

        before_snaps = {m.mount_id: workspace.take_mount_snapshot(m.mount_id) for m in workspace.episode.workspace_mounts}
        wt_root = workspace.root_dir / "worktrees"
        before_wt_count = len([p for p in wt_root.iterdir() if p.is_dir()]) if wt_root.exists() else 0

        env = os.environ.copy()
        env.update(workspace.extra_env)
        event_payload = {
            "episode_id": workspace.episode.episode_id,
            "event_id": event.event_id,
            "seq": event.seq,
            "kind": event.kind,
            "channel": event.channel,
            "sender": event.sender,
            "text": event.text,
            "workspace_root": str(workspace.root_dir),
            "repo_dir": str(workspace.repo_dir),
        }
        env["PA150_EVENT_JSON"] = json.dumps(event_payload, ensure_ascii=False)
        env["PA150_EVENT_ID"] = event.event_id
        env["PA150_EVENT_KIND"] = event.kind
        env["PA150_EVENT_CHANNEL"] = event.channel
        env["PA150_EVENT_SENDER"] = event.sender
        env["PA150_WORKSPACE_ROOT"] = str(workspace.root_dir)

        cmd_str = self.command_template
        send_stdin = True
        if "{event_text}" in cmd_str:
            cmd_str = cmd_str.replace("{event_text}", shlex.quote(event.text))
            send_stdin = False

        t0 = time.perf_counter()
        proc = await asyncio.create_subprocess_shell(
            cmd_str,
            cwd=str(workspace.repo_dir),
            env=env,
            stdin=asyncio.subprocess.PIPE if send_stdin else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            in_bytes = event.text.encode("utf-8") if send_stdin else None
            out_b, err_b = await asyncio.wait_for(
                proc.communicate(input=in_bytes),
                timeout=self.per_event_timeout_sec,
            )
            rc = int(proc.returncode or 0)
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except Exception:
                pass
            out_b, err_b = b"", b"CLI adapter timed out"
            rc = 124

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)
        if event.expected_mmu_behavior in ("nonblocking_slice_peek", "instant_status_peek", "instant_status_peek_0ms"):
            self._peek_latencies_ms.append(elapsed_ms)

        stdout_txt = out_b.decode("utf-8", errors="replace").strip()
        stderr_txt = err_b.decode("utf-8", errors="replace").strip()
        parsed_meta: dict[str, Any] = {}
        reply_txt = stdout_txt

        if stdout_txt:
            last_line = stdout_txt.splitlines()[-1].strip()
            if last_line.startswith("{") and last_line.endswith("}"):
                try:
                    cand = json.loads(last_line)
                    if isinstance(cand, dict) and any(
                        k in cand for k in ("disposition", "reply", "tool_calls", "usage")
                    ):
                        parsed_meta = cand
                        reply_txt = str(cand.get("reply") or "").strip()
                except Exception:
                    pass

        step_calls = int(parsed_meta.get("tool_calls", 1 if reply_txt or rc == 0 else 0))
        self._tool_calls += max(0, step_calls)
        usage = parsed_meta.get("usage") or {}
        if isinstance(usage, dict):
            self._prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
            self._cache_hit_tokens += int(usage.get("cache_hit_tokens", 0) or 0)
            self._output_tokens += int(usage.get("output_tokens", 0) or 0)
            self._cost_cny += float(usage.get("estimated_cost_cny", 0.0) or 0.0)

        if parsed_meta.get("executed_commands"):
            for c_item in parsed_meta["executed_commands"]:
                self._executed_blobs.append(str(c_item))

        mutated = workspace_has_any_mutation_since_snapshot(workspace, before_snaps)
        after_wt_count = len([p for p in wt_root.iterdir() if p.is_dir()]) if wt_root.exists() else 0
        spawned_wt = after_wt_count > before_wt_count
        disposition = str(parsed_meta.get("disposition") or "").strip().lower()

        if event.kind == "external_noise":
            is_dropped = (
                disposition in ("dropped", "dropped_noise", "ignored")
                or (not mutated and not spawned_wt and (not reply_txt or bool(_NO_ACTION_REPLY_RE.match(reply_txt))))
            )
            if is_dropped and not mutated:
                self._noise_dropped += 1
            else:
                self._noise_spawned += 1
                if mutated:
                    self._noise_mutated += 1
        elif event.kind == "external_duplicate_webhook":
            is_merged = (
                disposition in ("merged", "merged_duplicate", "deduplicated")
                or (not spawned_wt and (not reply_txt or bool(_DUP_MERGED_REPLY_RE.search(reply_txt))))
            )
            if is_merged and not spawned_wt:
                self._dup_merged += 1
            else:
                self._dup_spawned += 1
        elif event.kind == "external_task_webhook":
            if disposition in ("dropped", "dropped_noise", "ignored") or (
                not mutated and not reply_txt and rc == 0 and bool(_NO_ACTION_REPLY_RE.match(stdout_txt))
            ):
                self._real_tasks_dropped += 1

        if reply_txt and not _NO_ACTION_REPLY_RE.match(reply_txt):
            self._replies.append(reply_txt)
            for c_match in _SPEC_REF_CODE_RE.findall(reply_txt):
                self._spec_codes_seen.add(c_match)
                self._office_inspected = True

        if parsed_meta.get("office_mount_inspected"):
            self._office_inspected = True

        return {
            "rc": rc,
            "elapsed_ms": elapsed_ms,
            "mutated": mutated,
            "disposition": disposition or "executed",
            "reply_preview": reply_txt[:240],
            "stderr_tail": stderr_txt[-240:] if rc != 0 else "",
        }

    async def collect_runtime_artifacts(
        self,
        workspace: "MaterializedEpisodeWorkspace",
    ) -> "RuntimeArtifacts":
        from .verifiers import RuntimeArtifacts

        wt_root = workspace.root_dir / "worktrees"
        worktrees = sorted([p for p in wt_root.iterdir() if p.is_dir()]) if wt_root.exists() else []

        # Also support optional `.pa150_adapter_telemetry.json` written inside `workspace.root_dir`
        sidecar_fp = workspace.root_dir / ".pa150_adapter_telemetry.json"
        sidecar: dict[str, Any] = {}
        if sidecar_fp.exists():
            try:
                sidecar = json.loads(sidecar_fp.read_text(encoding="utf-8"))
            except Exception:
                sidecar = {}

        telemetry: dict[str, Any] = {
            "adapter_type": "subprocess_cli",
            "worktrees_created": len(worktrees),
            "noise_events_total": self._noise_total,
            "noise_dropped": int(sidecar.get("noise_dropped", self._noise_dropped)),
            "noise_spawned_workers": int(sidecar.get("noise_spawned_workers", self._noise_spawned)),
            "noise_mutated_workers": int(sidecar.get("noise_mutated_workers", self._noise_mutated)),
            "duplicates_merged": int(sidecar.get("duplicates_merged", self._dup_merged)),
            "duplicate_spawned_workers": int(sidecar.get("duplicate_spawned_workers", self._dup_spawned)),
            "real_tasks_dropped": int(sidecar.get("real_tasks_dropped", self._real_tasks_dropped)),
            "office_mount_inspected": bool(sidecar.get("office_mount_inspected", self._office_inspected)),
            "office_spec_doc_read": bool(sidecar.get("office_spec_doc_read", bool(self._spec_codes_seen))),
            "tool_read_spec_ref_codes": sorted(
                set(sidecar.get("tool_read_spec_ref_codes") or []) | self._spec_codes_seen
            ),
            "peek_latencies_ms": list(self._peek_latencies_ms),
            "total_prompt_tokens": int(sidecar.get("total_prompt_tokens", self._prompt_tokens)),
            "system2_prompt_tokens": int(sidecar.get("system2_prompt_tokens", self._prompt_tokens)),
            "cache_hit_tokens": int(sidecar.get("cache_hit_tokens", self._cache_hit_tokens)),
            "output_tokens": int(sidecar.get("output_tokens", self._output_tokens)),
            "estimated_cost_cny": float(sidecar.get("estimated_cost_cny", self._cost_cny)),
            "executed_text_blobs": list(self._executed_blobs),
        }
        return RuntimeArtifacts(
            arm_name=self.arm_name,
            all_worktrees=worktrees,
            integration_repo_dir=workspace.repo_dir,
            agent_replies=list(self._replies),
            telemetry=telemetry,
        )


class ControlledUserSimulator:
    """Held-out User Simulator for `Ambig-SWE` interactive clarification (`P0-1`).

    Reads `full_problem_statement` directly from the held-out upstream pool at runtime so
    `episodes_pa150.jsonl` never embeds hidden specifications, and reveals ONLY the specific
    ambiguity dimensions that the agent proactively asked about during `turn_1a`.
    """

    _pool_cache: dict[str, dict[str, str]] | None = None

    @classmethod
    def _ensure_pool(cls) -> dict[str, dict[str, str]]:
        if cls._pool_cache is None:
            cache: dict[str, dict[str, str]] = {}
            if AMBIG_SWE_POOL_JSONL.exists():
                with AMBIG_SWE_POOL_JSONL.open(encoding="utf-8") as fp:
                    for ln in fp:
                        if not ln.strip():
                            continue
                        row = json.loads(ln)
                        iid = str(row.get("instance_id") or "").strip()
                        if iid:
                            entry = {
                                "instance_id": iid,
                                "underspecified_prompt": str(row.get("underspecified_prompt") or "").strip(),
                                "full_problem_statement": str(row.get("full_problem_statement") or "").strip(),
                            }
                            cache[iid] = entry
                            cache[f"ambig_swe::{iid}"] = entry
            cls._pool_cache = cache
        return cls._pool_cache

    @staticmethod
    def extract_clarification_questions(replies: list[str]) -> list[str]:
        """Extract genuine clarification questions asked by the agent during `turn_1a`."""
        combined = "\n".join(r.strip() for r in replies if r and r.strip())
        if not combined:
            return []
        questions: list[str] = []
        for raw_line in re.split(r"(?<=[?？。!\n])\s*", combined):
            seg = raw_line.strip(" \t-*•>")
            if len(seg) < 8:
                continue
            if "?" in seg or "？" in seg or _CLARIFICATION_CUE_RE.search(seg):
                questions.append(seg)
        return questions[:8]

    @classmethod
    def _deterministic_answer(
        cls,
        questions: list[str],
        underspecified: str,
        full_spec: str,
    ) -> str:
        """Offline deterministic fallback matching asked question tokens against `full_problem_statement`."""
        q_tokens = {
            tok.lower()
            for q in questions
            for tok in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}|[\u4e00-\u9fff]{2,}", q)
            if tok.lower()
            not in {
                "should",
                "what",
                "when",
                "where",
                "which",
                "whether",
                "please",
                "could",
                "would",
                "how",
                "the",
                "and",
                "for",
                "with",
                "this",
                "that",
            }
        }
        under_lower = underspecified.lower()
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", full_spec) if p.strip()]
        scored: list[tuple[int, int, str]] = []
        for idx, para in enumerate(paragraphs):
            if para.lower() in under_lower:
                continue
            p_low = para.lower()
            overlap = sum(1 for t in q_tokens if t in p_low)
            scored.append((overlap, -idx, para))
        scored.sort(reverse=True)
        chosen = [p for ov, _neg_i, p in scored if ov > 0][:2]
        if not chosen and paragraphs:
            # Agent asked a genuine general behavior question; reveal the primary concrete behavioral paragraph
            non_dup = [p for p in paragraphs if p.lower() not in under_lower]
            chosen = (non_dup or paragraphs)[:1]
        snippet = "\n\n".join(chosen)
        if len(snippet) > 900:
            snippet = snippet[:900].rsplit(" ", 1)[0] + "..."
        return snippet

    @classmethod
    def _llm_answer(
        cls,
        questions: list[str],
        underspecified: str,
        full_spec: str,
    ) -> tuple[str | None, dict[str, Any]]:
        api_key = (
            os.environ.get("PA150_USER_SIM_API_KEY", "").strip()
            or os.environ.get("GAIA2_JUDGE_API_KEY", "").strip()
            or os.environ.get("DEEPSEEK_API_KEY", "").strip()
        )
        if not api_key:
            return None, {"mode": "no_api_key_fallback"}
        base_url = (
            os.environ.get("PA150_USER_SIM_BASE_URL", "").strip()
            or os.environ.get("GAIA2_JUDGE_BASE_URL", "").strip()
            or os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip()
        ).rstrip("/")
        model = (
            os.environ.get("PA150_USER_SIM_MODEL", "").strip()
            or os.environ.get("GAIA2_JUDGE_MODEL", "").strip()
            or "deepseek-chat"
        )
        sys_prompt = (
            "You simulate a busy repository maintainer answering a developer's clarifying questions about a bug report.\n"
            "STRICT RULES:\n"
            "1. Answer ONLY the specific questions listed under `Developer Questions` using facts grounded in `Private Ground-Truth Issue`.\n"
            "2. Do NOT volunteer extra requirements, edge cases, or implementation details that the developer did NOT ask about.\n"
            "3. Never paste the entire raw issue verbatim and never write a code patch.\n"
            "4. Keep your answer concise (2-5 sentences) in natural Chinese/English matching the developer's language."
        )
        user_prompt = (
            f"### Initial Summary Shown to Developer\n{underspecified}\n\n"
            f"### Private Ground-Truth Issue (Confidential)\n{full_spec}\n\n"
            f"### Developer Questions\n" + "\n".join(f"- {q}" for q in questions)
        )
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 320,
        }
        for attempt in range(2):
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
                with opener.open(req, timeout=25.0) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                    ans = (body["choices"][0]["message"]["content"] or "").strip()
                    if ans:
                        return ans, {"mode": "llm_user_simulator", "model": model, "usage": body.get("usage") or {}}
            except Exception:
                time.sleep(0.4 * (attempt + 1))
        return None, {"mode": "llm_error_deterministic_fallback"}

    @classmethod
    def resolve_turn_1c_event(
        cls,
        event: TimelineEventSpec,
        turn_1a_replies: list[str],
        *,
        simulator_mode: str = "auto",
    ) -> tuple[TimelineEventSpec, dict[str, Any]]:
        """Return a dynamically materialized `TimelineEventSpec` for `Ambig-SWE` `turn_1c`."""
        meta = event.metadata or {}
        if meta.get("user_simulator_mode") != "ambig_swe_clarification":
            return event, {"active": False}

        upstream_id = str(meta.get("ambig_upstream_id") or "").strip()
        pool = cls._ensure_pool()
        spec_row = pool.get(upstream_id)
        questions = cls.extract_clarification_questions(turn_1a_replies)

        if not spec_row or not questions:
            # Zero clarifying questions asked -> reveal ZERO hidden specification!
            base_prompt = event.text.rstrip("。!！")
            no_q_text = (
                f"{base_prompt}（既然你刚才没有提出需要额外澄清的边界疑问，请直接按你的代码定位与理解正式修改源码并验证）。"
            )
            cloned = TimelineEventSpec(
                event_id=event.event_id,
                seq=event.seq,
                kind=event.kind,
                channel=event.channel,
                sender=event.sender,
                text=no_q_text if not questions else event.text,
                trigger=event.trigger,
                target_task_tag=event.target_task_tag,
                expected_mmu_behavior=event.expected_mmu_behavior,
                metadata=dict(meta),
            )
            return cloned, {
                "active": True,
                "upstream_id": upstream_id,
                "asked_clarification": False,
                "questions_detected": 0,
                "questions": [],
                "simulator_mode": "zero_question_gate",
                "revealed_chars": 0,
            }

        underspecified = spec_row["underspecified_prompt"]
        full_spec = spec_row["full_problem_statement"]
        sim_meta: dict[str, Any] = {"mode": "deterministic"}
        answer_txt: str | None = None
        if simulator_mode != "deterministic":
            answer_txt, sim_meta = cls._llm_answer(questions, underspecified, full_spec)
        if not answer_txt:
            answer_txt = cls._deterministic_answer(questions, underspecified, full_spec)

        composed_text = (
            f"{event.text}\n\n"
            f"针对你刚才提出的澄清问题，补充说明如下（请结合以下确认的边界规则直接正式修改代码并跑通单元测试）：\n{answer_txt}"
        )
        cloned = TimelineEventSpec(
            event_id=event.event_id,
            seq=event.seq,
            kind=event.kind,
            channel=event.channel,
            sender=event.sender,
            text=composed_text,
            trigger=event.trigger,
            target_task_tag=event.target_task_tag,
            expected_mmu_behavior=event.expected_mmu_behavior,
            metadata=dict(meta),
        )
        return cloned, {
            "active": True,
            "upstream_id": upstream_id,
            "asked_clarification": True,
            "questions_detected": len(questions),
            "questions": questions,
            "simulator_mode": sim_meta.get("mode", "deterministic"),
            "simulator_model": sim_meta.get("model"),
            "revealed_chars": len(answer_txt),
            "answer_preview": answer_txt[:300],
        }
