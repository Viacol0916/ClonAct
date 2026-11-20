"""Layer 3: Hybrid Two-Tier Synaptic Bus.

Tier 0 (`0ms`, `$0`): Deterministic Python AST symbol + Doc/Issue URI footprint matching.
Tier 1 (`~150ms` Async System-1): Parallel `Noul` conductance p_{a->b} + `Choice` relation
evaluated ONLY at semantic milestones, guarded by a biological Directed Refractory Period.
"""
from __future__ import annotations

import ast
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from clonact.system1_judge import ChoiceSpec, NoulSpec, System1Judge

_TRIVIAL_SHELL_CMDS = frozenset(
    {
        "cd",
        "ls",
        "cat",
        "head",
        "tail",
        "grep",
        "egrep",
        "fgrep",
        "rg",
        "find",
        "echo",
        "printf",
        "pwd",
        "sed",
        "awk",
        "wc",
        "sort",
        "uniq",
        "cut",
        "tr",
        "true",
        "false",
        "test",
        "[",
        "mkdir",
        "cp",
        "mv",
        "rm",
        "chmod",
        "touch",
        "which",
        "env",
        "export",
        "set",
        "sleep",
        "date",
        "uname",
        "whoami",
        "id",
        "python",
        "python3",
        "pytest",
        "node",
        "sh",
        "bash",
        "zsh",
    }
)
_CMD_SEG_SPLIT_RE = re.compile(r"(?:\|\||&&|[;|&\n])+")


def extract_cli_and_fs_domains(tool_name: str, args: dict[str, Any]) -> set[str]:
    """Extract concise, general CLI/API/directory domain tags for `JevContextPrefix` causal grounding."""
    domains: set[str] = set()
    tname = (tool_name or "").strip()
    if tname in ("bash", "pwsh"):
        cmd = str(args.get("command") or "").strip()
        for seg in _CMD_SEG_SPLIT_RE.split(cmd):
            tokens = [t.strip("\"'`()") for t in seg.strip().split() if t.strip("\"'`()")]
            if not tokens:
                continue
            idx = 0
            while idx < len(tokens) and "=" in tokens[idx] and not tokens[idx].startswith(("./", "/")):
                idx += 1
            if idx >= len(tokens):
                continue
            exe_name = Path(tokens[idx]).name
            if exe_name and exe_name not in _TRIVIAL_SHELL_CMDS:
                domains.add(f"cli:{exe_name}")
                sub_tok = ""
                for nxt_i, nxt in enumerate(tokens[idx + 1 : idx + 3], start=idx + 1):
                    if not nxt.startswith("-") and "/" not in nxt and len(nxt) <= 28 and nxt.replace("_", "").replace("-", "").isalnum():
                        sub_tok = nxt.lower()
                        if sub_tok == "call" and nxt_i + 1 < len(tokens) and "__" in tokens[nxt_i + 1]:
                            ns_part = tokens[nxt_i + 1].split("__", 1)[0].strip("\"'`").lower()
                            if ns_part.isalnum():
                                sub_tok = ns_part
                        break
                if sub_tok:
                    domains.add(f"cli:{exe_name}/{sub_tok}")
            for tok in tokens:
                if tok.startswith(("-", "http://", "https://", "/")):
                    continue
                clean_tok = tok.lstrip("./")
                if "/" in clean_tok:
                    top_dir = clean_tok.split("/", 1)[0]
                    if top_dir and top_dir not in (".", "..") and top_dir.replace("_", "").replace("-", "").isalnum():
                        domains.add(f"fs:{top_dir}")
    elif tname in ("read", "write", "edit", "read_repo_file", "edit_repo_file", "glob", "grep"):
        raw_p = str(args.get("file_path") or args.get("path") or args.get("rel_path") or "").strip()
        if raw_p:
            clean_p = raw_p.lstrip("./")
            parts = [p for p in Path(clean_p).parts if p not in (".", "..", "/")]
            if len(parts) >= 2 and not raw_p.startswith("/"):
                top_dir = parts[0]
                if top_dir.replace("_", "").replace("-", "").isalnum():
                    domains.add(f"fs:{top_dir}")
            bn = Path(raw_p).name
            if bn and "." in bn:
                domains.add(f"file:{bn}")
    return domains


def extract_python_symbols(source_text: str, rel_path: str) -> set[str]:
    """Extract top-level functions, classes, and uppercase constants via AST (`0ms`)."""
    uris: set[str] = {f"code://{rel_path}"}
    try:
        tree = ast.parse(source_text)
    except SyntaxError:
        return uris
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            uris.add(f"code://{rel_path}::{node.name}")
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id.isupper():
                    uris.add(f"code://{rel_path}::{t.id}")
    return uris


def _extract_code_paths(uris: set[str]) -> list[str]:
    paths: set[str] = set()
    for u in uris:
        if u.startswith("code://"):
            body = u[len("code://") :]
            rel = body.split("::", 1)[0].strip()
            if rel:
                paths.add(rel)
    return sorted(paths)


@dataclass
class CellFootprint:
    cell_id: str
    topic: str
    status: str  # "interactive_alignment" | "detached_executing" | "parked_draft" | "completed"
    read_uris: set[str] = field(default_factory=set)
    write_uris: set[str] = field(default_factory=set)
    touched_domains: set[str] = field(default_factory=set)
    latest_milestone: str = ""
    verified_delta: str = ""
    commit_sha: str | None = None


@dataclass
class SynapticSignal:
    phase: int  # 1 = touch warning, 2 = verified state-slice push, 3 = hot user constraint bridge
    source_cell: str
    target_cell: str
    conductance_p: float
    relation: str
    message: str


class HybridSynapticBus:
    """Connects executing and interactive Context Cells without a micromanaging Router.

    Hyperparameter-free System-1 policy:
      - All discrete action/relation selections use parameter-free `Choice` argmax.
      - All binary coupling checks use calibrated `Noul` at the exact symmetric midpoint `0.50`.
    """

    def __init__(self, judge: System1Judge, enabled: bool = True) -> None:
        self.judge = judge
        self.enabled = enabled
        self.footprints: dict[str, CellFootprint] = {}
        self.explicit_links: set[tuple[str, str]] = set()
        self.inbox: dict[str, list[SynapticSignal]] = {}
        # Directed refractory state: (src_cell, dst_cell, phase) -> set of emitted milestone digests
        self._refractory_ledger: dict[tuple[str, str, int], set[str]] = {}
        self._pair_couplings: dict[tuple[str, str], tuple[float, str]] = {}
        self.events_log: list[dict[str, Any]] = []

    def register_or_update_cell(self, fp: CellFootprint) -> None:
        self.footprints[fp.cell_id] = fp
        self.inbox.setdefault(fp.cell_id, [])

    def add_explicit_link(self, cell_a: str, cell_b: str) -> None:
        self.explicit_links.add((cell_a, cell_b))
        self.explicit_links.add((cell_b, cell_a))

    def drain_signals(self, cell_id: str) -> list[SynapticSignal]:
        sigs = list(self.inbox.get(cell_id, []))
        self.inbox[cell_id] = []
        return sigs

    def record_uri_touch(
        self,
        cell_id: str,
        uris: set[str],
        is_write: bool,
        intent_one_liner: str,
    ) -> list[SynapticSignal]:
        """Tier 0 (`0ms`) deterministic URI intersection check -> Phase 1 Touch Notice & Settled Trunk Notice."""
        fp = self.footprints.get(cell_id)
        if not fp:
            return []
        if is_write:
            fp.write_uris |= uris
        else:
            fp.read_uris |= uris

        emitted: list[SynapticSignal] = []
        if not self.enabled:
            return emitted

        # Early Pull Awareness: if an executing coding cell touches a file already committed to `main` by a settled peer,
        # notify `cell_id` immediately so it can merge `main` before authoring conflicting hunks!
        if fp.status == "detached_executing":
            for peer_id, peer in self.footprints.items():
                if peer_id == cell_id or peer.status != "completed" or not peer.commit_sha:
                    continue
                settled_overlap = uris & peer.write_uris
                settled_files = _extract_code_paths(settled_overlap)
                if settled_files:
                    all_peer_files = _extract_code_paths(peer.write_uris) or settled_files
                    files_str = ", ".join(all_peer_files[:6])
                    sig_settled = self._try_emit(
                        phase=2,
                        src=peer_id,
                        dst=cell_id,
                        p=1.0,
                        relation="tier0_settled_trunk_overlap",
                        milestone_key=f"trunk_advance:{peer.commit_sha}",
                        msg=(
                            f"[SYNAPTIC PHASE-2 TRUNK ADVANCE | from={peer_id} | main_commit={peer.commit_sha} | shared_files={files_str}]\n"
                            f"兄弟任务 `{peer_id}` ({peer.topic[:90]}) 已完成验证并将涉及 `{files_str}` 的改动合入主干 `main` (commit `{peer.commit_sha}`)。\n"
                            f"建议你在修改或收尾前通过 `bash` 执行 `git merge --no-edit main` 提前同步主干最新代码并联合验证测试，从源头避免收尾冲突。"
                        ),
                    )
                    if sig_settled:
                        emitted.append(sig_settled)

        if not is_write:
            return emitted

        for peer_id, peer in self.footprints.items():
            if peer_id == cell_id or peer.status in ("completed", "archived", "cancelled"):
                continue
            overlap = uris & (peer.read_uris | peer.write_uris)
            if overlap:
                shared_files = _extract_code_paths(overlap)
                peek_hint = (
                    f"可通过 `peek_peer_cell(cell_id=\"{cell_id}\", rel_path=\"{shared_files[0]}\")` 查看其具体改动方向，尽量保持函数签名与职责向后兼容。"
                    if shared_files
                    else f"可通过 `peek_peer_cell(cell_id=\"{cell_id}\")` 查看其实时进展，保持对外口径一致。"
                )
                tier0_key = f"tier0_files:{shared_files}" if shared_files else f"tier0:{sorted(overlap)}"
                sig = self._try_emit(
                    phase=1,
                    src=cell_id,
                    dst=peer_id,
                    p=1.0,
                    relation="tier0_ast_uri_overlap",
                    milestone_key=tier0_key,
                    msg=(
                        f"[SYNAPTIC PHASE-1 NOTICE] 兄弟上下文 `{cell_id}` ({fp.topic[:80]}) 正在并发修改共享实体 "
                        f"`{', '.join(sorted(overlap)[:8])}` ({intent_one_liner})；无需阻塞等待，但{peek_hint}"
                    ),
                )
                if sig:
                    emitted.append(sig)
        return emitted

    async def evaluate_milestone(
        self,
        focal_cell_id: str,
        milestone_summary: str,
        verified_delta: str | None = None,
        causal_echo_from: str | None = None,
        commit_sha: str | None = None,
        integrated_peer_ids: set[str] | None = None,
    ) -> list[SynapticSignal]:
        """Tier 1 (`~150ms` 1-Pass Parallel System-1) + Tier 0 AST physical overlap union."""
        focal = self.footprints.get(focal_cell_id)
        if not focal:
            return []
        focal.latest_milestone = milestone_summary
        if verified_delta:
            focal.verified_delta = verified_delta
        if commit_sha:
            focal.commit_sha = commit_sha
        if not self.enabled:
            return []

        skip_peers = set(integrated_peer_ids or set())
        peers = {
            pid: p
            for pid, p in self.footprints.items()
            if pid != focal_cell_id
            and pid not in skip_peers
            and p.status not in ("archived", "cancelled")
            and (bool(verified_delta) or p.status != "completed")
            and pid != causal_echo_from  # Block immediate echo ping-pong!
        }
        if not peers:
            return []

        finding_text = (
            f"{milestone_summary}\n{verified_delta}" if verified_delta else milestone_summary
        )
        unlinked_peers = {pid: p for pid, p in peers.items() if (focal_cell_id, pid) not in self.explicit_links}
        answers: dict[str, Any] = {}
        if unlinked_peers:
            state = {
                "focal_task": {
                    "cell_id": focal.cell_id,
                    "topic": focal.topic,
                    "latest_finding_or_action": finding_text,
                    "has_verified_delta": bool(verified_delta),
                },
                "peer_tasks": {
                    pid: {
                        "topic": p.topic,
                        "status": p.status,
                        "latest_milestone": p.latest_milestone,
                        "explicitly_linked_by_user": False,
                    }
                    for pid, p in unlinked_peers.items()
                },
            }

            questions: dict[str, Any] = {}
            for pid in unlinked_peers:
                questions[f"couple_{pid}"] = NoulSpec(
                    instructions=(
                        f"Does `focal_task.latest_finding_or_action` alter or establish any business timing, quota, "
                        f"API contract, test requirement, or factual claim relevant to `peer_tasks.{pid}`?"
                    ),
                    true_desc="Affects timing, SLA, numbers, API behavior, or factual claims relevant to the peer task",
                    false_desc="Completely unrelated domain and zero impact on the peer task's output or code",
                )
                questions[f"rel_{pid}"] = ChoiceSpec(
                    instructions=f"What is the relationship between `focal_task` and `peer_tasks.{pid}` (`Choice` argmax)?",
                    criteria={
                        "contradiction": "Conflicting schedule, SLA promise, number, or incompatible code assumption",
                        "upstream_dependency": "Focal task provides an updated parameter, root cause, or rule needed by peer",
                        "duplicate_effort": "Both tasks are solving the exact same bug or writing the exact same artifact",
                        "orthogonal": "Independent tasks with no meaningful coupling",
                    },
                )

            res = await self.judge.decide(state=state, questions=questions, tag=f"synapse_milestone:{focal_cell_id}")
            answers = res.answers

        emitted: list[SynapticSignal] = []

        for pid, peer_fp in peers.items():
            is_explicit = (focal_cell_id, pid) in self.explicit_links
            ast_overlap = focal.write_uris & (peer_fp.read_uris | peer_fp.write_uris)
            shared_code_files = _extract_code_paths(ast_overlap)

            if is_explicit:
                p_val = 1.0
                rel = "explicit_observer_subscription"
            else:
                p_val = float(answers[f"couple_{pid}"]["noul"])
                rel = str(answers[f"rel_{pid}"]["choice"])
                if verified_delta and focal.commit_sha and shared_code_files and (p_val < 0.50 or rel == "orthogonal"):
                    p_val = 1.0
                    rel = "tier0_ast_symbol_overlap"

            # Cache evaluated pair (never overwrite an already-detected positive coupling with orthogonal)
            prev_pair = self._pair_couplings.get((focal_cell_id, pid))
            if prev_pair is None or (p_val >= 0.50 and rel != "orthogonal"):
                self._pair_couplings[(focal_cell_id, pid)] = (p_val, rel)
                self._pair_couplings[(pid, focal_cell_id)] = (p_val, rel)

            # Strictly symmetric 0.50 binary Noul midpoint + parameter-free Choice argmax (or proven AST overlap)
            if p_val < 0.50 or rel == "orthogonal":
                continue

            if verified_delta:
                if focal.commit_sha and shared_code_files and peer_fp.status == "detached_executing":
                    all_focal_files = _extract_code_paths(focal.write_uris) or shared_code_files
                    files_str = ", ".join(all_focal_files[:6])
                    p2_key = f"trunk_advance:{focal.commit_sha}"
                    msg = (
                        f"[SYNAPTIC PHASE-2 TRUNK ADVANCE | from={focal_cell_id} | main_commit={focal.commit_sha} | p={p_val:.2f} | rel={rel} | shared_files={files_str}]\n"
                        f"{verified_delta}\n"
                        f"工作区主干 `main` 已合入 `{focal_cell_id}` 的验证提交 `{focal.commit_sha}`（共同涉及文件：`{files_str}`）。"
                        f"请在收尾前通过 `bash` 执行 `git merge --no-edit main` 将最新主干变更提前合并入你的工作区（如有冲突直接消解），并在合并后运行测试确认通过。"
                    )
                else:
                    p2_key = f"p2:{focal.commit_sha or ''}:{verified_delta}"
                    msg = (
                        f"[SYNAPTIC PHASE-2 VERIFIED DELTA | from={focal_cell_id} | p={p_val:.2f} | rel={rel}]\n"
                        f"{verified_delta}\n"
                        f"请将上述关联已验证事实同步纳入你当前的处理或交付结果中，确保口径一致。"
                    )
                sig = self._try_emit(
                    phase=2,
                    src=focal_cell_id,
                    dst=pid,
                    p=p_val,
                    relation=rel,
                    milestone_key=p2_key,
                    msg=msg,
                )
            else:
                if peer_fp.status == "completed":
                    continue
                sig = self._try_emit(
                    phase=1,
                    src=focal_cell_id,
                    dst=pid,
                    p=p_val,
                    relation=rel,
                    milestone_key=f"p1:{rel}",
                    msg=(
                        f"[SYNAPTIC PHASE-1 TOUCH NOTICE | from={focal_cell_id} | p={p_val:.2f} | rel={rel}] "
                        f"关联进展：{milestone_summary}"
                    ),
                )
            if sig:
                emitted.append(sig)
        return emitted

    def inspect_trunk_peer_coupling(
        self,
        cell_id: str,
        candidate_peer_ids: set[str] | None = None,
    ) -> dict[str, Any]:
        """Return Tier-0 AST + Tier-1 Jev coupling diagnostics and peer context for trunk reconciliation."""
        fp = self.footprints.get(cell_id)
        if not fp:
            return {"is_coupled": False, "reason": "", "shared_files": [], "peer_summaries": []}

        reasons: list[str] = []
        all_shared_files: set[str] = set()
        peer_summaries: list[dict[str, Any]] = []
        my_touched = fp.read_uris | fp.write_uris

        for pid, peer in self.footprints.items():
            if pid == cell_id or peer.status in ("archived", "cancelled"):
                continue
            if candidate_peer_ids is not None:
                if pid not in candidate_peer_ids:
                    continue
            elif not peer.commit_sha:
                continue

            overlap = (my_touched & peer.write_uris) | (fp.write_uris & (peer.read_uris | peer.write_uris))
            shared_files = _extract_code_paths(overlap)
            all_shared_files.update(shared_files)
            jev_pair = self._pair_couplings.get((cell_id, pid))

            if shared_files:
                reasons.append(f"AST shared files with `{pid}`: {', '.join(shared_files[:5])}")
            if jev_pair and jev_pair[0] >= 0.50 and jev_pair[1] != "orthogonal":
                reasons.append(f"Jev coupling with `{pid}` ({jev_pair[1]}, p={jev_pair[0]:.2f})")

            peer_summaries.append(
                {
                    "cell_id": pid,
                    "topic": peer.topic,
                    "commit_sha": peer.commit_sha,
                    "shared_files": shared_files,
                    "summary": (peer.verified_delta or peer.latest_milestone or "(completed)")[:600],
                }
            )

        is_coupled = bool(self.enabled and reasons)
        return {
            "is_coupled": is_coupled,
            "reason": "; ".join(reasons),
            "shared_files": sorted(all_shared_files),
            "peer_summaries": peer_summaries,
        }

    async def assess_trunk_peer_coupling(
        self,
        cell_id: str,
        candidate_peer_ids: set[str] | None = None,
    ) -> dict[str, Any]:
        """Evaluate any unassessed trunk peer pairs via 1-pass `Jev` when AST overlap is absent, then return coupling info."""
        info = self.inspect_trunk_peer_coupling(cell_id, candidate_peer_ids=candidate_peer_ids)
        if not self.enabled or info["is_coupled"]:
            return info
        fp = self.footprints.get(cell_id)
        if not fp:
            return info

        unassessed: dict[str, CellFootprint] = {}
        for item in info["peer_summaries"]:
            pid = item["cell_id"]
            if (cell_id, pid) not in self._pair_couplings and pid in self.footprints:
                unassessed[pid] = self.footprints[pid]

        if not unassessed:
            return info

        state = {
            "focal_task": {
                "cell_id": fp.cell_id,
                "topic": fp.topic,
                "write_uris": sorted(fp.write_uris)[:12],
                "latest_milestone": fp.latest_milestone or fp.verified_delta,
            },
            "settled_trunk_peers": {
                pid: {
                    "topic": p.topic,
                    "write_uris": sorted(p.write_uris)[:12],
                    "summary": (p.verified_delta or p.latest_milestone)[:400],
                }
                for pid, p in unassessed.items()
            },
        }
        questions: dict[str, Any] = {}
        for pid in unassessed:
            questions[f"couple_{pid}"] = NoulSpec(
                instructions=(
                    f"Could merging `focal_task` and `settled_trunk_peers.{pid}` introduce a code signature, "
                    f"data contract, import, or behavioral interaction requiring joint test verification?"
                ),
                true_desc="Shared module/API/contract interaction where joint verification is warranted",
                false_desc="Completely independent files and orthogonal behavior",
            )
            questions[f"rel_{pid}"] = ChoiceSpec(
                instructions=f"What is the relationship between `focal_task` and `settled_trunk_peers.{pid}` (`Choice` argmax)?",
                criteria={
                    "contradiction": "Potentially incompatible signature, state mutation, or behavioral assumption",
                    "upstream_dependency": "One task modifies a helper, schema, or contract relied upon by the other",
                    "duplicate_effort": "Overlapping implementation of the same capability",
                    "orthogonal": "Completely independent files and orthogonal features",
                },
            )
        res = await self.judge.decide(state=state, questions=questions, tag=f"trunk_reconcile_gate:{cell_id}")
        for pid in unassessed:
            p_val = float(res.answers[f"couple_{pid}"]["noul"])
            rel = str(res.answers[f"rel_{pid}"]["choice"])
            self._pair_couplings[(cell_id, pid)] = (p_val, rel)
            self._pair_couplings[(pid, cell_id)] = (p_val, rel)

        return self.inspect_trunk_peer_coupling(cell_id, candidate_peer_ids=candidate_peer_ids)

    def inject_hot_bridge_constraint(
        self,
        source_cell: str,
        target_cell: str,
        constraint_text: str,
        p_ref: float,
    ) -> SynapticSignal | None:
        """Queue a direct user follow-up or cross-cell constraint for tail absorption without halting execution."""
        if not self.enabled and source_cell != "user_dm":
            return None
        if source_cell == "user_dm":
            msg = f"[USER FOLLOW-UP MESSAGE]\n{constraint_text}"
            relation = "user_direct_followup"
        else:
            self.add_explicit_link(source_cell, target_cell)
            msg = (
                f"[SYNAPTIC CROSS-CELL CONSTRAINT | bridged from {source_cell} | p={p_ref:.2f}]\n"
                f"关联任务 `{source_cell}` 同步了与你相关的补充要求：{constraint_text}"
            )
            relation = "user_interleaved_bridge"
        return self._try_emit(
            phase=3,
            src=source_cell,
            dst=target_cell,
            p=p_ref,
            relation=relation,
            milestone_key=f"bridge:{constraint_text}",
            msg=msg,
        )

    def inject_tail_trunk_delta(
        self,
        target_cell: str,
        entry_id: str,
        summary: str,
    ) -> SynapticSignal | None:
        """Append newly recalled/updated shared trunk knowledge at the tail of a live pinned cell."""
        return self._try_emit(
            phase=2,
            src="T_0_recall",
            dst=target_cell,
            p=1.0,
            relation="tail_trunk_recall",
            milestone_key=f"trunk_recall:{entry_id}",
            msg=(
                f"[SHARED TRUNK RECALL | `{entry_id}` appended at tail to preserve byte-0 KV cache]\n"
                f"{summary}"
            ),
        )

    def _try_emit(
        self,
        phase: int,
        src: str,
        dst: str,
        p: float,
        relation: str,
        milestone_key: str,
        msg: str,
    ) -> SynapticSignal | None:
        digest = hashlib.sha256(milestone_key.encode("utf-8")).hexdigest()[:12]
        ledger_key = (src, dst, phase)
        seen_digests = self._refractory_ledger.setdefault(ledger_key, set())
        if digest in seen_digests:
            return None  # Blocked by Synaptic Refractory Period!
        seen_digests.add(digest)

        sig = SynapticSignal(
            phase=phase,
            source_cell=src,
            target_cell=dst,
            conductance_p=round(p, 4),
            relation=relation,
            message=msg,
        )
        self.inbox.setdefault(dst, []).append(sig)
        self.events_log.append(
            {
                "phase": phase,
                "from": src,
                "to": dst,
                "p": round(p, 4),
                "relation": relation,
                "message": msg,
            }
        )
        return sig
