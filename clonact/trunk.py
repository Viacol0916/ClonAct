"""Layer 2: Three-Stratum Dynamic Elastic Trunk T_0^(e).

Passive base memory image (NOT an agent!) shared at byte-0 across all Context Cells.
Layout: [STATIC_SYSTEM + UNION_TOOLS] ⊕ [L1: Pinned Core] ⊕ [L2: Elastic Working Set] <-> [L3: Cold Store]
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

UNION_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": (
                "Execute a shell command inside the workspace (e.g., running CLI helpers, Python scripts, "
                "inspecting files, or calling sandboxed APIs). Subject to timeout and output cap."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command executed with cwd already set to the workspace root (do not `cd /workspace`).",
                    },
                    "timeout_sec": {
                        "type": "integer",
                        "description": "Optional timeout in seconds (default 45, max 120).",
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_repo_file",
            "description": "Read a file inside the isolated Git worktree.",
            "parameters": {
                "type": "object",
                "properties": {"rel_path": {"type": "string"}},
                "required": ["rel_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_repo_file",
            "description": "Replace exact text inside a file (or pass old_text='' to create a new file) in the isolated Git worktree.",
            "parameters": {
                "type": "object",
                "properties": {
                    "rel_path": {"type": "string"},
                    "old_text": {"type": "string"},
                    "new_text": {"type": "string"},
                },
                "required": ["rel_path", "old_text", "new_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "peek_peer_cell",
            "description": (
                "Instant zero-side-effect read-only inspection (`0ms`) of a sibling Context Cell's live progress, "
                "verified state deltas, recent tool outputs, file diff, and final completion summary."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "cell_id": {
                        "type": "string",
                        "description": "Sibling cell_id to inspect (e.g. 'Cell_E_1'), or 'ALL' to inspect all live cells.",
                    },
                    "rel_path": {
                        "type": "string",
                        "description": (
                            "Optional relative file path (e.g. 'pkg/module.py') to inspect the sibling cell's "
                            "detailed git diff on a specific shared file."
                        ),
                    },
                },
                "required": ["cell_id"],
            },
        },
    },
]

STATIC_SYSTEM_PROMPT = (
    "You are Synaptic-PA, the user's unified Personal Agent operating over a shared epistemic trunk T_0.\n"
    "Core principles:\n"
    "1. Speak in first-person ('我') concisely in Chinese.\n"
    "2. Never invent unverified workspace, database, API, or document facts; inspect the workspace and run checks to verify.\n"
    "3. Execute requested tasks end-to-end inside your workspace while respecting explicit user safety/read-only instructions.\n"
    "4. Respect any `[SYNAPTIC ...]` or `[USER FOLLOW-UP MESSAGE]` blocks appended to your context."
)


def compute_blob_hash(path: Path) -> str:
    if not path.exists():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


@dataclass
class TrunkEntry:
    entry_id: str
    category: str  # "verified_delta" | "tombstone" | "standing_rule"
    summary: str
    anchor_type: str  # "commit_sha" | "doc_rev" | "blob_hash" | "test_pass" | "user_rule" | "env_sig"
    anchor_value: str
    target_path: str | None = None
    cross_task_hits: int = 0
    created_at: float = field(default_factory=time.time)
    last_used_at: float = field(default_factory=time.time)


class ElasticSharedTrunk:
    """Three-stratum dynamic elastic shared prefix T_0^(e)."""

    def __init__(
        self,
        l2_max_entries: int = 12,
        l1_pinned_rules: list[str] | None = None,
        static_system_prompt: str | None = None,
        disable_elastic_l2_l3: bool = False,
    ) -> None:
        self.epoch: int = 1
        self.l2_max_entries = l2_max_entries
        self.static_system_prompt = static_system_prompt or STATIC_SYSTEM_PROMPT
        self.disable_elastic_l2_l3 = disable_elastic_l2_l3

        # L1: Pinned Core (slow-changing identity, auth boundaries, topology)
        self.l1_pinned_rules: list[str] = (
            list(l1_pinned_rules)
            if l1_pinned_rules is not None
            else [
                "User Identity: @user (Unified Personal Agent workspace).",
                "Actuation Policy: Inspect workspace state and run verification checks before finalizing.",
            ]
        )

        # L2: Elastic Active Working Set + Mid-Burst Staging Queue (protects Prefix Cache!)
        self.l2_entries: dict[str, TrunkEntry] = {}
        self.l2_staged_promotions: dict[str, TrunkEntry] = {}
        self.l2_pointers: dict[str, str] = {}

        # L3: Unlimited Cold Store
        self.l3_cold_store: dict[str, TrunkEntry] = {}

    def promote(self, entry: TrunkEntry, immediate_epoch_bump: bool = True) -> bool:
        """Evidence-Anchored Promotion Gate: reject unanchored free-form speculation."""
        if self.disable_elastic_l2_l3:
            return False
        valid_anchors = {"commit_sha", "doc_rev", "blob_hash", "test_pass", "user_rule", "env_sig"}
        if entry.anchor_type not in valid_anchors or not entry.anchor_value.strip():
            return False

        if not immediate_epoch_bump:
            self.l2_staged_promotions[entry.entry_id] = entry
            return True

        if entry.category == "standing_rule":
            self.l1_pinned_rules.append(entry.summary)
        else:
            self.l2_entries[entry.entry_id] = entry
        self.epoch += 1

        if len(self.l2_entries) > self.l2_max_entries:
            self.batch_demote_coldest()
        return True

    def flush_staged_epoch(self) -> int:
        """Batch-commit staged promotions at burst boundary so left-prefix KV cache never thrashes mid-burst."""
        if self.disable_elastic_l2_l3:
            self.l2_staged_promotions.clear()
            return self.epoch
        if not self.l2_staged_promotions:
            return self.epoch
        for eid, entry in self.l2_staged_promotions.items():
            if entry.category == "standing_rule":
                self.l1_pinned_rules.append(entry.summary)
            else:
                self.l2_entries[eid] = entry
        self.l2_staged_promotions.clear()
        self.epoch += 1
        if len(self.l2_entries) > self.l2_max_entries:
            self.batch_demote_coldest()
        return self.epoch

    def invalidate_stale_tombstones(self, repo_root: Path) -> list[str]:
        """Automatically purge hash-bound negative/env tombstones (across L2 & L3) whose target file changed."""
        evicted: list[str] = []
        for eid, ent in list(self.l2_entries.items()):
            if ent.anchor_type == "blob_hash" and ent.target_path:
                cur_hash = compute_blob_hash(repo_root / ent.target_path)
                if cur_hash != ent.anchor_value:
                    del self.l2_entries[eid]
                    evicted.append(eid)
        for eid, ent in list(self.l3_cold_store.items()):
            if ent.anchor_type == "blob_hash" and ent.target_path:
                cur_hash = compute_blob_hash(repo_root / ent.target_path)
                if cur_hash != ent.anchor_value:
                    del self.l3_cold_store[eid]
                    self.l2_pointers.pop(eid, None)
                    evicted.append(eid)
        if evicted:
            self.epoch += 1
        return evicted

    def batch_demote_coldest(self, count: int = 2) -> list[str]:
        """Batch Epoch Demote-to-Pointer (never silently delete!)."""
        now = time.time()

        def utility(ent: TrunkEntry) -> float:
            age = max(0.0, now - ent.last_used_at)
            return math.log1p(ent.cross_task_hits + 1) * math.exp(-0.001 * age) - 0.0005 * len(ent.summary)

        import math

        ranked = sorted(self.l2_entries.values(), key=utility)
        demoted: list[str] = []
        for ent in ranked[:count]:
            del self.l2_entries[ent.entry_id]
            self.l3_cold_store[ent.entry_id] = ent
            self.l2_pointers[ent.entry_id] = f"[L3 Pointer `{ent.entry_id}`] {ent.summary[:65]}..."
            demoted.append(ent.entry_id)
        if demoted:
            self.epoch += 1
        return demoted

    def page_in_from_l3(self, entry_id: str) -> TrunkEntry | None:
        if self.disable_elastic_l2_l3:
            return None
        ent = self.l3_cold_store.pop(entry_id, None)
        if ent:
            self.l2_pointers.pop(entry_id, None)
            ent.last_used_at = time.time()
            ent.cross_task_hits += 1
            self.l2_entries[entry_id] = ent
            self.epoch += 1
        return ent

    def render_prefix_messages(self) -> list[dict[str, str]]:
        """Render deterministic byte-exact left prefix messages for 100% KV-Cache sharing."""
        l1_block = "\n".join(f"- {r}" for r in self.l1_pinned_rules)
        l2_lines = [
            f"- [{e.category} | {e.anchor_type}={e.anchor_value}] {e.summary}"
            for e in self.l2_entries.values()
        ]
        for ptr in self.l2_pointers.values():
            l2_lines.append(f"- {ptr}")
        l2_block = "\n".join(l2_lines) if l2_lines else "- (Clean initial working set)"

        trunk_content = (
            f"{self.static_system_prompt}\n\n"
            f"=== SHARED EPISTEMIC TRUNK T_0 (Epoch {self.epoch}) ===\n"
            f"[L1: Pinned Core]\n{l1_block}\n\n"
            f"[L2: Active Working Set]\n{l2_block}\n"
            f"=== END OF SHARED TRUNK T_0 ==="
        )
        return [{"role": "system", "content": trunk_content}]
