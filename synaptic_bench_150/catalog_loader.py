"""Loaders and structured task decomposers over the 4 cached upstream pillars."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
UPSTREAM_ROOT = PROJECT_ROOT / "data" / "upstream_pillars"
POOLS_DIR = UPSTREAM_ROOT / "pools"

PURE_PYTHON_COOPER_REPOS = [
    "pallets_click_task",
    "pallets_jinja_task",
    "samuelcolvin_dirty_equals_task",
    "dottxt_ai_outlines_task",
    "dspy_task",
    "huggingface_datasets_task",
]

# NEW-3 / v2.0: CooperBench features whose `tests.patch` asserts undisclosed private helpers or unstated base classes
EXCLUDED_COOPER_FEATURES: set[tuple[str, str, int]] = {
    ("pallets_click_task", "2068", 6),  # tests.patch calls undisclosed `editor._set_process_priority`
    ("dspy_task", "8563", 1),  # tests.patch asserts unstated `Type` inheritance & `validate_input` shapes
    # outlines#1706 `tests/backends/test_xgrammar.py` requires C++ `xgrammar`/`llama_cpp` & live HF Hub GGUF model downloads
    *(("dottxt_ai_outlines_task", "1706", fid) for fid in range(1, 9)),
}

# v2.0: MCPMark tasks whose upstream `verify.py` checks undocumented rubric conventions not stated in `description.md`
EXCLUDED_MCPMARK_SOLVED_TASK_IDS: set[str] = {
    "author_folders",
    "file_arrangement",
}

# Known-5 + NEW-3 + v2.0: SWE instances whose upstream `test_patch` imports/calls undisclosed internal symbols/kwargs
EXCLUDED_SWE_INSTANCES = {
    "pylint-dev__pylint-4604",
    "pylint-dev__pylint-4551",
    "beetbox__beets-6471",
    "caronc__apprise-1273",
    "dbcli__pgcli-1553",
    "simonw__datasette-2650",
    "simonw__datasette-2667",
    "pytest-dev__pytest-10356",
    "sympy__sympy-28946",
    "treeverse__dvc-10945",
    "sphinx-doc__sphinx-7590",
    "django__django-13279",
    "django__django-14011",
    "HKUDS__nanobot-836",  # error body exists only in external GitHub attachment screenshot (`P1-5`)
    "sympy__sympy-28736",  # initial reproduction script points only to external `gist.github.com` (`P1-5`)
    "jlowin__fastmcp-2642",  # tests assert unstated `ToolAnnotations(readOnlyHint=True)` default
}

EXCLUDED_SWE_REPOS = {
    "borgbackup/borg",
    "dbt-labs/dbt-core",
    "dagster-io/dagster",
    "great-expectations/great_expectations",
    "astropy/astropy",
    "matplotlib/matplotlib",
    "scikit-learn/scikit-learn",
    "numpy/numpy",
    "python-pillow/Pillow",
    "pydata/xarray",
    "python/mypy",
    "HKUDS/LightRAG",
    "mealie-recipes/mealie",
    "sissbruecker/linkding",
    "wagtail/wagtail",
    "django/django",
    "agentscope-ai/CoPaw",  # upstream problem statements contain UTF-8 replacement mojibake
}

# NEW-4 / P0-2: Minimum versions compatible with Python 3.11 (`collections.abc`, `pluggy>=1.0`, modern pytest)
MIN_PY311_REPO_VERSIONS: dict[str, tuple[int, ...]] = {
    "psf/requests": (2, 26),
    "sympy/sympy": (1, 9),
    "pytest-dev/pytest": (7, 0),
    "pallets/flask": (2, 0),
    "pylint-dev/pylint": (2, 12),
    "sphinx-doc/sphinx": (4, 5),
    "mwaskom/seaborn": (0, 12),
}

PREFERRED_SWE_REPOS = (
    "Kludex/starlette",
    "fastapi/typer",
    "joke2k/faker",
    "caronc/apprise",
    "PrefectHQ/fastmcp",
    "simonw/datasette",
    "wireservice/csvkit",
    "pytransitions/transitions",
    "networkx/networkx",
    "sympy/sympy",
    "aws-cloudformation/cfn-lint",
    "streamlink/streamlink",
    "aiogram/aiogram",
    "dbcli/pgcli",
    "beetbox/beets",
    "HKUDS/nanobot",
)

PREFERRED_AMBIG_REPO_PREFIXES = (
    "pytest-dev/",
    "pallets/",
    "psf/",
    "pylint-dev/",
    "sympy/",
    "sphinx-doc/",
)


def _rel_to_mvp(path: Path) -> str:
    """Store portable paths relative to `` so manifests contain zero host-specific `/Users/...` paths (`NEW-8`)."""
    return str(path.absolute().relative_to(PROJECT_ROOT.absolute()))


def _parse_version_tuple(version_str: str) -> tuple[int, ...]:
    nums = [int(x) for x in re.findall(r"\d+", str(version_str or ""))]
    return tuple(nums[:3]) if nums else (0, 0)


def _is_py311_compatible_swe_row(row: dict[str, Any]) -> bool:
    repo = str(row.get("repo") or "")
    ver = str(row.get("version") or "").strip()
    if repo in MIN_PY311_REPO_VERSIONS and ver:
        return _parse_version_tuple(ver) >= MIN_PY311_REPO_VERSIONS[repo]
    return True


def _has_undisclosed_private_test_coupling(spec_text: str, gold_patch: str, test_patch: str) -> bool:
    """Return True if `test_patch` directly references a newly added `_private` symbol absent from `spec_text`."""
    if not gold_patch or not test_patch:
        return False
    removed_priv = set(re.findall(r"^\-\s*(?:async\s+)?def\s+(_[a-zA-Z0-9_]+)\s*\(", gold_patch, re.M))
    added_priv = set(re.findall(r"^\+\s*(?:async\s+)?def\s+(_[a-zA-Z0-9_]+)\s*\(", gold_patch, re.M)) - removed_priv
    removed_cls = set(re.findall(r"^\-\s*class\s+(_[a-zA-Z0-9_]+)\b", gold_patch, re.M))
    added_priv |= set(re.findall(r"^\+\s*class\s+(_[a-zA-Z0-9_]+)\b", gold_patch, re.M)) - removed_cls
    added_priv = {s for s in added_priv if not (s.startswith("__") and s.endswith("__"))}
    if not added_priv:
        return False
    tp_added = "\n".join(ln[1:] for ln in test_patch.splitlines() if ln.startswith("+") and not ln.startswith("+++"))
    return any(re.search(rf"\b{re.escape(sym)}\b", tp_added) and sym not in spec_text for sym in added_priv)


def _extract_patch_paths_simple(patch_text: str) -> list[str]:
    files: list[str] = []
    for m in re.finditer(r"^\+\+\+\s+b/(.+)$", patch_text or "", flags=re.MULTILINE):
        fp = m.group(1).strip()
        if fp != "/dev/null" and fp not in files:
            files.append(fp)
    return files


def normalize_swe_pytest_target(raw: str, patched_files: list[str] | None = None) -> tuple[str | None, str | None]:
    """Sanitize upstream SWE-Interact / Ambig-SWE test entries into `(pytest_node_id, bare_func_name)` (`NEW-2`)."""
    s = str(raw or "").strip()
    if not s:
        return None, None
    # Strip pytest terminal status prefixes (`PASSED `, `FAILED `, etc.)
    s = re.sub(r"^(?:PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)\s+", "", s).strip()
    # Strip pytest `-v` source attribution suffix (`lhs <- path/to/test_x.py`) without inverting inherited classes (`P1-4`)
    if " <- " in s:
        lhs, rhs = [p.strip() for p in s.split(" <- ", 1)]
        lhs_file = lhs.split("::", 1)[0].strip()
        if "/" in lhs_file:
            s = lhs
        elif rhs.endswith(".py") and "::" in lhs and Path(rhs).name == lhs_file:
            s = f"{rhs}::{lhs.split('::', 1)[1]}"
        elif rhs.endswith(".py") and not lhs_file.endswith(".py"):
            s = rhs
        else:
            s = lhs
    # Drop pytest summary count lines (`====== 2202`, `7032`, `36`)
    if re.match(r"^[=\-\s]*\d+(?:\s+\w+)?[=\-\s]*$", s):
        return None, None
    # Strip truncated `[` parametrization suffix without closing `]`
    if "[" in s and not s.endswith("]"):
        s = s.split("[", 1)[0].strip()

    if "::" in s:
        head, tail = s.split("::", 1)
        head = head.strip()
        tail = tail.strip()
        if not head.endswith(".py") and "." in head and "/" not in head:
            parts = head.split(".")
            if parts[-1][:1].isupper() and len(parts) >= 2:
                head = "/".join(parts[:-1]) + ".py"
                tail = f"{parts[-1]}::{tail}"
            else:
                head = "/".join(parts) + ".py"
        if patched_files and "/" not in head and head.endswith(".py"):
            matches = [pf for pf in patched_files if pf.endswith(f"/{head}") or pf == head]
            if len(matches) == 1:
                head = matches[0]
        return f"{head}::{tail}", None

    if s.endswith(".py"):
        if patched_files and "/" not in s:
            matches = [pf for pf in patched_files if pf.endswith(f"/{s}") or pf == s]
            if len(matches) == 1:
                s = matches[0]
        return s, None

    fn = s.split("(")[0].split(".")[-1].strip()
    if not fn or fn.isdigit() or not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", fn):
        return None, None
    return None, fn


def clean_swe_test_entries(raw_entries: list[Any], test_patch: str = "") -> list[str]:
    """Return deduplicated, sanitized pytest node IDs/function names from raw upstream list (`NEW-2`)."""
    patched_files = _extract_patch_paths_simple(test_patch)
    cleaned: list[str] = []
    for raw in raw_entries or []:
        nd, fn = normalize_swe_pytest_target(str(raw), patched_files=patched_files)
        val = nd or fn
        if val and val not in cleaned:
            cleaned.append(val)
    return cleaned


def _round_robin_by_key(items: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    """Deterministically interleave items across distinct repositories so no single repo dominates."""
    buckets: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for it in items:
        kv = str(it.get(key) or "")
        if kv not in buckets:
            buckets[kv] = []
            order.append(kv)
        buckets[kv].append(it)
    out: list[dict[str, Any]] = []
    idx = 0
    while True:
        added = False
        for kv in order:
            b = buckets[kv]
            if idx < len(b):
                out.append(b[idx])
                added = True
        if not added:
            break
        idx += 1
    return out


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


_BOILERPLATE_BLOCK_PATTERNS = (
    re.compile(r"^\s*(?:#{1,4}\s*|\*{1,2})Files\s+Modified\b", re.IGNORECASE),
    re.compile(r"_No response_", re.IGNORECASE),
    re.compile(r"^\s*(?:thanks|thank\s+you|cheers|best\s+regards|regards|many\s+thanks)[!.,\s]*$", re.IGNORECASE),
    re.compile(r"\bwilling\s+to\s+(?:submit|contribute|open|send|work\s+on)\s+(?:a\s+)?pr\b", re.IGNORECASE),
    re.compile(r"^\s*can\s+you\s+tell\s+us\s+the\s+reason\??\s*$", re.IGNORECASE),
    re.compile(r"\bafter\s+receiving\s+an\s+answer\s+i\s+can\s+implement\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+is\s+the\s+best\s+way\s+to\s+handle\s+this\b", re.IGNORECASE),
    re.compile(r"\bi\s+don'?t\s+understand\s+why\b", re.IGNORECASE),
    re.compile(r"\bis\s+this\s+(?:intentional|expected|a\s+bug)\??\s*$", re.IGNORECASE),
    re.compile(r"\bpull\s+request\s+with\s+the\s+solution\s+will\s+be\s+submitted\b", re.IGNORECASE),
    re.compile(r"\bprioritize\s*&\s*fund\s+our\s+backlog\b", re.IGNORECASE),
    re.compile(r"^\s*(?:it\s+)?would\s+be\s+(?:great|nice|awesome|helpful|wonderful)\s+if\b", re.IGNORECASE),
    re.compile(r"^\s*(?:hope\s+this\s+helps|feel\s+free\s+to|any\s+help\s+would\s+be\s+appreciated)\b", re.IGNORECASE),
)

_GAIA_COURTESY_TAIL_RE = re.compile(
    r"^\s*(?:please\s+|after\s+this\s+is\s+done,\s*|once\s+all\s+tasks\s+have\s+been\s+completed,\s*)?"
    r"(?:let\s+me\s+know\s+(?:once|after|when|if)\s+(?:all\s+tasks|you\s+finish|everything|it)|"
    r"do\s+not\s+notify\s+me\s+of\s+anything|notify\s+me\s+once\s+done|confirm\s+once\s+completed)\b",
    flags=re.IGNORECASE,
)


def _depends_on_external_media_or_gist(text: str) -> bool:
    """Return True if an upstream issue relies on external Gist scripts or screenshot-only error bodies (`P1-5`)."""
    if "gist.github.com/" in text and "```" not in text:
        return True
    if re.search(
        r"(?:got\s+the\s+error|following\s+error|error\s+message|see\s+screenshot)\s*:\s*(?:!\[[^\]]*\]\([^)]+\)|<img\b)",
        text,
        flags=re.IGNORECASE,
    ):
        return True
    return False


def _strip_html_comments(text: str) -> str:
    cleaned = re.sub(r"<!--[\s\S]*?-->", "", text)
    # Strip GitHub sponsorship / Polar.sh HTML badge footers and fundraising blockquotes
    cleaned = re.sub(r"<a\b[^>]*polar\.sh[^>]*>[\s\S]*?</a>", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"(?:^>\s*.*polar\.sh[\s\S]*?$)", "", cleaned, flags=re.IGNORECASE | re.MULTILINE)
    return cleaned.strip()


def _substantive_text_len(block: str) -> int:
    cleaned = _strip_html_comments(block)
    # Strip standalone Markdown image embeds (`![alt](url)`) and `<img ...>` tags (`NEW-6`)
    cleaned = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", cleaned)
    cleaned = re.sub(r"<img\b[^>]*>", "", cleaned, flags=re.IGNORECASE).strip()
    if not cleaned:
        return 0
    # Strip markdown heading / bold title line if nothing else follows
    body_lines = [
        ln.strip()
        for ln in cleaned.splitlines()
        if ln.strip() and not re.match(r"^(?:#{1,4}\s+.+|\*\*[^*]+\*\*:?\s*)$", ln.strip())
    ]
    joined_body = " ".join(body_lines).strip()
    for pat in _BOILERPLATE_BLOCK_PATTERNS:
        if pat.search(cleaned) and len(joined_body) < 420:
            return 0
    if len(cleaned) < 45 and not re.search(r"[`()\[\]{}=:]", cleaned):
        return 0
    return len(joined_body) if joined_body else len(cleaned)


def _split_fence_safe_blocks(text: str) -> list[str]:
    """Split markdown into paragraph/section blocks without EVER cutting inside fenced code blocks."""
    blocks: list[str] = []
    cur: list[str] = []
    in_fence = False
    fence_marker = ""
    for line in text.splitlines():
        stripped = line.lstrip()
        m_fence = re.match(r"^(`{3,}|~{3,})", stripped)
        if m_fence:
            marker = m_fence.group(1)[:3]
            if not in_fence:
                in_fence = True
                fence_marker = marker
            elif stripped.startswith(fence_marker):
                in_fence = False
                fence_marker = ""
            cur.append(line)
            continue

        if not in_fence and not stripped:
            if cur:
                blk = "\n".join(cur).strip()
                if blk:
                    blocks.append(blk)
                cur = []
        else:
            cur.append(line)
    if cur:
        blk = "\n".join(cur).strip()
        if blk:
            blocks.append(blk)
    return blocks


def split_markdown_sections(desc: str) -> tuple[str, str]:
    """Deterministically split a specification into `(base_spec, addendum_spec)` (`NEW-1`, `NEW-6`).

    Guarantees:
    1. Never splits inside fenced code blocks (```` ``` ```` / `~~~`).
    2. Never produces a non-behavioral addendum consisting solely of `**Files Modified**`, `_No response_`,
       HTML comments, author sign-offs, or `"Thanks"`.
    """
    text = _strip_html_comments(desc.strip())
    if text.count("```") % 2 != 0:
        text = f"{text}\n```"
    blocks = _split_fence_safe_blocks(text)
    if not blocks:
        return desc.strip(), "Ensure all edge cases, exact field/file naming rules, and formatting constraints are strictly satisfied."

    # Peel off trailing non-behavioral blocks (`**Files Modified**`, courtesy closings, `"Thanks"`, `_No response_`)
    trailing_meta: list[str] = []
    core_blocks = list(blocks)
    while len(core_blocks) >= 2 and _substantive_text_len(core_blocks[-1]) < 50:
        last_blk = core_blocks.pop()
        if not re.search(r"_No response_", last_blk, re.IGNORECASE):
            trailing_meta.insert(0, last_blk)

    def _finalize(head_blks: list[str], tail_blks: list[str]) -> tuple[str, str] | None:
        base = "\n\n".join(head_blks).strip()
        add_core = "\n\n".join(tail_blks).strip()
        add_full = "\n\n".join(tail_blks + trailing_meta).strip()
        if (
            len(base) >= 70
            and _substantive_text_len(add_core) >= 50
            and base.count("```") % 2 == 0
            and add_full.count("```") % 2 == 0
        ):
            return base, add_full
        return None

    # 1. Prefer splitting at numbered top-level requirement blocks (`2. ...`, `3. ...`)
    num_indices = [i for i, b in enumerate(core_blocks) if i > 0 and re.match(r"^\d+\.\s+", b)]
    for cut in reversed(num_indices):
        res = _finalize(core_blocks[:cut], core_blocks[cut:])
        if res is not None:
            return res

    # 2. Prefer splitting at Markdown headings (`## ...`, `### ...`) or CooperBench bold sections (`**Solution**`, etc.)
    sec_indices = [
        i
        for i, b in enumerate(core_blocks)
        if i > 0 and re.match(r"^(?:#{2,4}\s+|\*\*(?:Solution|Implementation|Technical|Behavior|Details|Edge|Error|Usage|Example)[^*]*\*\*)", b, re.IGNORECASE)
    ]
    for cut in reversed(sec_indices):
        res = _finalize(core_blocks[:cut], core_blocks[cut:])
        if res is not None:
            return res

    # 3. Walk backwards across fence-safe blocks until `tail_blks` has >= 50 chars of substantive behavior
    for cut in range(len(core_blocks) - 1, 0, -1):
        res = _finalize(core_blocks[:cut], core_blocks[cut:])
        if res is not None:
            return res

    # 4. Single-block prose fallback: split on sentence boundary outside code fences
    core_text = "\n\n".join(core_blocks).strip()
    if "```" not in core_text:
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", core_text) if s.strip()]
        if len(sents) >= 2:
            for cut in range(len(sents) - 1, 0, -1):
                head_s = " ".join(sents[:cut]).strip()
                tail_s = " ".join(sents[cut:]).strip()
                if len(head_s) >= 60 and len(tail_s) >= 50:
                    if trailing_meta:
                        tail_s = tail_s + "\n\n" + "\n\n".join(trailing_meta)
                    return head_s, tail_s

    return (
        text,
        "Ensure all edge cases, exact field/file naming rules, and formatting constraints are strictly satisfied.",
    )


def load_mcpmark_local_tasks() -> dict[str, list[dict[str, Any]]]:
    """Load all 71 local zero-cloud MCPMark tasks (`40` filesystem + `31` postgres)."""
    tasks_root = UPSTREAM_ROOT / "mcpmark" / "tasks"
    out: dict[str, list[dict[str, Any]]] = {"filesystem": [], "postgres": []}

    for domain in ("filesystem", "postgres"):
        dom_dir = tasks_root / domain
        for verify_py in sorted(dom_dir.rglob("verify.py")):
            tdir = verify_py.parent
            rel_parts = tdir.relative_to(dom_dir).parts
            suite = rel_parts[0]  # 'easy' or 'standard'
            desc_path = tdir / "description.md"
            meta_path = tdir / "meta.json"
            if not desc_path.exists():
                continue
            desc = desc_path.read_text(encoding="utf-8").strip()
            meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
            base_spec, addendum_spec = split_markdown_sections(desc)
            task_uid = f"mcpmark::{domain}::{'/'.join(rel_parts)}"
            out[domain].append(
                {
                    "uid": task_uid,
                    "domain": domain,
                    "suite": suite,
                    "category": meta.get("category_id") or (rel_parts[1] if len(rel_parts) > 1 else domain),
                    "task_id": meta.get("task_id") or tdir.name,
                    "task_name": meta.get("task_name") or tdir.name,
                    "difficulty": meta.get("difficulty", "L2"),
                    "description": desc,
                    "base_spec": base_spec,
                    "addendum_spec": addendum_spec,
                    "excluded_from_solved": (tdir.name in EXCLUDED_MCPMARK_SOLVED_TASK_IDS),
                    "task_dir": _rel_to_mvp(tdir),
                    "verify_py": _rel_to_mvp(verify_py),
                }
            )
    return out


def load_gaia2_pool() -> dict[str, list[dict[str, Any]]]:
    """Load cached Gaia2 scenarios grouped by config (`ambiguity`, `adaptability`, `execution`, `time`)."""
    rows = _read_jsonl(POOLS_DIR / "gaia2_compact_pool.jsonl")
    by_cfg: dict[str, list[dict[str, Any]]] = {"ambiguity": [], "adaptability": [], "execution": [], "time": []}
    for r in rows:
        cfg = r["config"]
        user_msgs = [
            ev["args"].get("content", "").strip()
            for ev in r.get("user_events", [])
            if ev["args"].get("content", "").strip()
        ]
        primary_prompt = user_msgs[0] if user_msgs else (
            f"Complete scenario `{r['scenario_id']}` across personal apps ({', '.join(r['apps'][:6])})."
        )
        followup_prompt = user_msgs[1] if len(user_msgs) >= 2 else None
        raw_sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", primary_prompt) if s.strip()]
        # Merge trailing courtesy-only notification clauses onto the preceding actionable sentence (`P1-2`)
        merged_sents: list[str] = []
        for s in raw_sents:
            if merged_sents and (_GAIA_COURTESY_TAIL_RE.search(s) or len(s) < 38):
                merged_sents[-1] = f"{merged_sents[-1]} {s}"
            else:
                merged_sents.append(s)
        if len(merged_sents) >= 2:
            mid = max(1, len(merged_sents) - 1)
            r["base_prompt"] = " ".join(merged_sents[:mid])
            r["addendum_prompt"] = " ".join(merged_sents[mid:])
            r["has_substantive_split"] = True
        else:
            r["base_prompt"] = primary_prompt
            r["addendum_prompt"] = ""
            r["has_substantive_split"] = False
        r["primary_prompt"] = primary_prompt
        r["followup_prompt"] = followup_prompt
        if cfg in by_cfg:
            by_cfg[cfg].append(r)
    return by_cfg


_ACTIONABLE_NOISE_CUES = re.compile(
    r"(?:\?|\bplease\b|\binvite\b|\binvited\b|\bcan we\b|\bcould you\b|\bwould you\b|\bneed you\b"
    r"|\bcancel\b|\bbuy\b|\bbook\b|\breschedule\b|\bsend\s+(?:me|an?\b|the\b|it\b)\b|\bsending\s+an\s+email\b|\bupdate\b|\bfix\b|\bsave\b"
    r"|\bi\s+can(?:\s+only|\s+do|\s+still|'t|\s+not)\b|\bcannot\s+make\b|\bwon'?t\s+be\s+able\b|\bwill\s+not\s+be\s+able\b"
    r"|\bi\s+am\s+out\b|\bi'?m\s+out\b|\bonly\s+time\b|\bcan\s+only\s+meet\b|\bmeeting\s+scheduled\b|\bscheduled\s+for\b"
    r"|\bplan\s+on\b|\blet'?s\b|\blets\b|\blet\s+me\s+know\b|\bworks\s+perfect\b|\bwould\s+be\s+perfect\b"
    r"|\bavailable\b|\bfree\s+(?:on|this)\b|\bbusy\s+(?:on|this)\b|\bcoming\s+on\b|\battending\b|\battendance\b|\bcount\s+me\s+in\b"
    r"|\btaxi\b|\brelocated\b|\bceremony\b|\btime\s+has\s+changed\b|\bset\s+up\s+a\s+call\b|\blast\s+minute\s+changes\b"
    r"|\bwanted\b|\bi\s+want\b|\bi'?d\s+like\b|\bwould\s+like\b|\bwould\s+love\s+to\b|\bi\s+need\b|\bdo\s+not\s+like\b|\bdon'?t\s+like\b"
    r"|\bnot\s+looking\s+for\b|\bstill\s+thinking\b|\bunsave\b|\bremove\b|\bcheck\s+out\b|\binstead\s+of\b"
    r"|\bdiscount\s+code\b|\bflight\s+details\b|\barrival:\b|\bbudget\s+range\b|\bviewing\b"
    r"|\bdifficulty\s+level\b|\bplastic\s+material\b|\brecycled\s+cardboard\b|\bwith\s+color\b)",
    flags=re.IGNORECASE,
)


def extract_gaia2_noise_bank(gaia_by_cfg: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Build deterministic realistic ambient workplace/CI chatter bank without self-labeling giveaway cues (`v2.0`)."""
    curated_noise = [
        {"channel": "im://group/daily_lounge", "sender": "alex_pm", "text": "大家下午点冰美式还是燕麦拿铁？我拼单凑满减 ☕"},
        {"channel": "ci://bot/docs_spellcheck", "sender": "bot_ci", "text": "[Nightly Docs Build #914] Rendered 184 static HTML pages in 6.2s."},
        {"channel": "im://group/office_notice", "sender": "admin_hr", "text": "周五下午茶切果盘已经放到 14 楼北侧茶水间啦 🍉"},
        {"channel": "im://group/dev_watercooler", "sender": "lin_fe", "text": "刚才茶水间那个 CSS 居中笑话笑死我了 😂"},
        {"channel": "ci://bot/mirror_sync", "sender": "bot_mirror", "text": "[CDN Warmup] Edge POP static bundle replication finished across 4 regions."},
        {"channel": "ci://bot/coverage_badge", "sender": "bot_cov", "text": "[Docs Preview #4829] Static API reference site deployed to staging-docs."},
        {"channel": "im://group/office_notice", "sender": "facilities", "text": "今晚 22:30 起东侧 3 号客梯年度例行保养，1、2 号梯照常运行。"},
        {"channel": "im://group/daily_lounge", "sender": "chen_qa", "text": "园区南门新开的重庆小面汤底挺香，中午刚去打卡 🍜"},
        {"channel": "ci://bot/dep_digest", "sender": "bot_deps", "text": "[Toolchain Mirror] PyPI internal wheel cache synced 42 upstream metadata entries."},
        {"channel": "im://group/dev_watercooler", "sender": "wu_infra", "text": "周末读完了 SQLite WAL 那篇源码剖析博客，写得挺透彻的。"},
        {"channel": "im://group/office_notice", "sender": "it_broadcast", "text": "企业根证书今天凌晨已完成例行轮换，各办公区 AP 运行平稳。"},
        {"channel": "ci://bot/bench_archive", "sender": "bot_archive", "text": "[Cold Storage] Rotated September access logs (1.8 GB compressed) to glacier."},
        {"channel": "ci://bot/status_heartbeat", "sender": "bot_uptime", "text": "[Region us-west] Staging gateway median latency 17.4ms over past 60m."},
        {"channel": "im://group/dev_watercooler", "sender": "maya_design", "text": "Dropped next year's typography exploration board into #brand-archive for posterity."},
        {"channel": "email://newsletter/eng_weekly", "sender": "eng_digest", "text": "[Engineering Weekly #142] Deep dive into Python 3.12freethreading & PG17 vacuum improvements."},
        {"channel": "ci://bot/backup_verify", "sender": "bot_backup", "text": "[Snapshot Verify] Nightly ZFS volume scrub completed in 14m12s."},
        {"channel": "im://group/daily_lounge", "sender": "sam_data", "text": "Sunset over the river from the 18th floor terrace looks unreal right now 🌇"},
        {"channel": "im://group/office_notice", "sender": "library_bot", "text": "[Book Corner] Three copies of Designing Data-Intensive Applications returned to shelf B3."},
        {"channel": "ci://bot/license_audit", "sender": "bot_compliance", "text": "[SPDX Scanner] Completed header inventory across 642 source files."},
        {"channel": "im://group/dev_watercooler", "sender": "zhao_sec", "text": "昨天 PyCon 那场 CPython 字节码优化的录像出了，演讲节奏真好 👍"},
        {"channel": "email://newsletter/oss_pulse", "sender": "oss_digest", "text": "[Open Source Digest] October community release highlights & maintainer interviews."},
        {"channel": "ci://bot/container_gc", "sender": "bot_registry", "text": "[Harbor GC] Reclaimed 4.2 GB from expired ephemeral preview image layers."},
        {"channel": "im://group/office_notice", "sender": "green_office", "text": "立冬啦，园区空调今起切换为制热新风模式 🍂"},
        {"channel": "ci://bot/metrics_rollup", "sender": "bot_metrics", "text": "[TSDB Compactor] Block 01J9K8M merged 12 segments in 1.38s."},
        {"channel": "im://group/daily_lounge", "sender": "nina_ops", "text": "12 楼咖啡机刚换上新烘的埃塞水洗豆子 ☕"},
        {"channel": "ci://bot/ssl_monitor", "sender": "bot_certs", "text": "[Cert Inventory] Wildcard ingress certificate valid for 284 days."},
        {"channel": "email://newsletter/culture_club", "sender": "culture_bot", "text": "[Board Game Club] Photos from last Friday's Catan tournament are live on intranet."},
        {"channel": "ci://bot/cache_telemetry", "sender": "bot_cache", "text": "[Build Cache] Daily sccache hit ratio across Linux runners: 94.8%."},
        {"channel": "im://group/dev_watercooler", "sender": "leo_ml", "text": "TIL Python 3.12 `itertools.batched` is implemented directly in C!"},
        {"channel": "ci://bot/ntp_check", "sender": "bot_infra", "text": "[Chrony Telemetry] Cluster stratum-2 clock offset measured at 0.31ms."},
        {"channel": "im://group/daily_lounge", "sender": "grace_pm", "text": "雨下大了，一楼前台备了共享雨伞架 ☔"},
        {"channel": "im://group/dev_watercooler", "sender": "ivan_db", "text": "刚看到 Postgres 17 的 `COPY ... ON_ERROR ignore` 基准测试，吞吐很稳。"},
        {"channel": "ci://bot/artifact_ttl", "sender": "bot_ci", "text": "[Artifact Reaper] Purged 19 expired PR preview tarballs older than 14d."},
        {"channel": "im://group/office_notice", "sender": "shuttle_bot", "text": "晚班地铁接驳大巴 19:30 / 20:15 准点停靠 B 座北门。"},
        {"channel": "email://newsletter/tech_talks", "sender": "guild_host", "text": "[Backend Guild Recap] Recording & slides from Tuesday's Rust FFI lightning talk."},
        {"channel": "im://group/daily_lounge", "sender": "kevin_ios", "text": "羽毛球群今晚在奥体订了两片场地，打完球真解压 🏸"},
        {"channel": "ci://bot/font_subset", "sender": "bot_assets", "text": "[Asset Pipeline] WOFF2 glyph subsetting completed for marketing landing page."},
        {"channel": "im://group/dev_watercooler", "sender": "olivia_qa", "text": "`pytest -n auto` 在新 M3 机器上跑完本地套件只要四秒多，丝滑！"},
        {"channel": "ci://bot/dns_ttl", "sender": "bot_net", "text": "[Route53 Sync] Internal split-horizon zone serial 2026111004 propagated."},
        {"channel": "im://group/office_notice", "sender": "canteen_bot", "text": "今日二楼风味档口供应广式烧腊双拼与菌菇乌鸡汤 🍲"},
        {"channel": "im://group/dev_watercooler", "sender": "peter_sre", "text": "eBPF 火焰图定位锁竞争确实比传统采样直观太多了。"},
        {"channel": "ci://bot/runner_autoscaler", "sender": "bot_fleet", "text": "[Runner Pool] Warmed 4 idle macOS arm64 build VMs for afternoon peak."},
        {"channel": "email://newsletter/design_system", "sender": "ds_team", "text": "[Design Tokens v4.3] Changelog of spacing & dark-mode elevation variables."},
        {"channel": "im://group/daily_lounge", "sender": "rachel_ux", "text": "窗外彩虹出来了！东面窗户能看到双彩虹 🌈"},
        {"channel": "ci://bot/sbom_export", "sender": "bot_supply", "text": "[CycloneDX] Nightly SBOM manifest archived to compliance bucket."},
        {"channel": "im://group/dev_watercooler", "sender": "tom_core", "text": "`git range-diff` 对比 rebase 前后的补丁演进真是神器。"},
        {"channel": "im://group/office_notice", "sender": "reception", "text": "一楼丰巢快递柜已完成格口扩容，大件包裹在西侧货架。"},
        {"channel": "ci://bot/locale_sync", "sender": "bot_i18n", "text": "[Crowdin Sync] Pulled 12 translated UI glossary strings for ja-JP."},
        {"channel": "im://group/daily_lounge", "sender": "victor_be", "text": "园区银杏道全黄了，午休散步拍照特别出片 🍂"},
        {"channel": "ci://bot/manpage_gen", "sender": "bot_docs", "text": "[CLI Manpages] Regenerated roff pages from release tag v2.4.1."},
    ]
    bank: list[dict[str, Any]] = []
    seen_texts: set[str] = {c["text"] for c in curated_noise}
    for cfg_rows in gaia_by_cfg.values():
        for sc in cfg_rows:
            for ev in sc.get("env_events", [])[:3]:
                app = str(ev.get("app") or "Messages")
                if app.lower() in ("calendar", "calendarapp", "shopping", "shoppingapp", "rentaflat", "cabs", "cabapp"):
                    continue
                fn = ev.get("function") or "notify"
                args = ev.get("args") or {}
                snippet = (
                    args.get("body")
                    or args.get("content")
                    or ""
                ).strip()
                if (
                    snippet
                    and 20 <= len(snippet) < 180
                    and snippet not in seen_texts
                    and not _ACTIONABLE_NOISE_CUES.search(snippet)
                ):
                    seen_texts.add(snippet)
                    bank.append(
                        {
                            "channel": f"im://feed/{app.lower()}",
                            "sender": args.get("sender") or args.get("organizer") or "colleague_feed",
                            "text": snippet,
                            "source_scenario": sc["scenario_id"],
                            "fn": fn,
                        }
                    )
    combined = curated_noise + bank
    return combined[:60]


def load_cooperbench_catalog() -> dict[str, Any]:
    """Load CooperBench Python feature pairs (clean merges vs overlapping conflicts) + 3-wave repo chains."""
    cb_ds = UPSTREAM_ROOT / "CooperBench" / "dataset"
    gcr = json.loads((cb_ds / "gold_conflict_report.json").read_text(encoding="utf-8"))
    hf_rows = _read_jsonl(POOLS_DIR / "cooperbench_hf_pool.jsonl")
    hf_index: dict[tuple[str, str, int], dict[str, Any]] = {}
    for rw in hf_rows:
        repo_slug = rw["repo"].replace("/", "_").replace("-", "_") + "_task"
        hf_index[(repo_slug, str(rw["cooperbench_task_id"]), int(rw["cooperbench_feature_id"]))] = rw

    def _load_feature(repo_dir_name: str, task_id: str | int, feat_num: int) -> dict[str, Any] | None:
        if (repo_dir_name, str(task_id), int(feat_num)) in EXCLUDED_COOPER_FEATURES:
            return None
        tdir = cb_ds / repo_dir_name / f"task{task_id}"
        fdir = tdir / f"feature{feat_num}"
        if not (fdir / "feature.md").exists():
            return None
        fmd = (fdir / "feature.md").read_text(encoding="utf-8").strip()
        if fmd.count("```") % 2 != 0:
            fmd = f"{fmd}\n```"
        setup_txt = (tdir / "setup.sh").read_text(encoding="utf-8") if (tdir / "setup.sh").exists() else ""
        m_commit = re.search(r'BASE_COMMIT="([0-9a-f]{40})"', setup_txt)
        m_clone = re.search(r"https://github\.com/([^\s\"']+)", setup_txt)
        hf_rec = hf_index.get((repo_dir_name, str(task_id), int(feat_num)), {})
        base_spec, addendum_spec = split_markdown_sections(fmd)
        title_m = re.search(r"\*\*Title\*\*:\s*(.+)", fmd)
        title = title_m.group(1).strip() if title_m else f"{repo_dir_name}/task{task_id}/feature{feat_num}"
        raw_gh_repo = hf_rec.get("repo") or (m_clone.group(1) if m_clone else repo_dir_name.removesuffix("_task"))
        clean_gh_repo = str(raw_gh_repo).strip().removesuffix(".git")
        return {
            "uid": f"cooperbench::{repo_dir_name}::task{task_id}::feature{feat_num}",
            "repo_dir_name": repo_dir_name,
            "github_repo": clean_gh_repo,
            "task_id": str(task_id),
            "feature_id": int(feat_num),
            "base_commit": hf_rec.get("base_commit") or (m_commit.group(1) if m_commit else ""),
            "title": title,
            "feature_md": fmd,
            "base_spec": base_spec,
            "addendum_spec": addendum_spec,
            "task_dir": _rel_to_mvp(tdir),
            "feature_patch": _rel_to_mvp(fdir / "feature.patch"),
            "tests_patch": _rel_to_mvp(fdir / "tests.patch"),
            "setup_sh": _rel_to_mvp(tdir / "setup.sh"),
            "run_tests_sh": _rel_to_mvp(tdir / "run_tests.sh"),
        }

    conflict_set = {
        (c["repo"], str(c["task_id"]), int(c["f1"]), int(c["f2"]))
        for c in gcr.get("conflict_pairs", [])
    }

    all_cfg = json.loads((cb_ds / "subsets" / "all.json").read_text(encoding="utf-8"))
    clean_pairs: list[dict[str, Any]] = []
    conflict_pairs: list[dict[str, Any]] = []
    single_features: list[dict[str, Any]] = []
    seen_feat_uids: set[str] = set()
    three_wave_chains: list[dict[str, Any]] = []

    ordered_entries = sorted(
        [e for e in all_cfg.get("tasks", []) if e["repo"] in PURE_PYTHON_COOPER_REPOS],
        key=lambda e: (PURE_PYTHON_COOPER_REPOS.index(e["repo"]), str(e["task_id"])),
    )
    for entry in ordered_entries:
        rname = entry["repo"]
        tid = str(entry["task_id"])
        feat_nums = sorted({int(x) for pair in entry["pairs"] for x in pair})
        loaded_feats = [f for fn in feat_nums if (f := _load_feature(rname, tid, fn)) is not None]
        for lf in loaded_feats:
            if lf["uid"] not in seen_feat_uids:
                seen_feat_uids.add(lf["uid"])
                single_features.append(lf)
        if len(loaded_feats) >= 3:
            # v2.0 (`P1-4`): Enumerate strictly conflict-free 3-feature wave triples (`0` pairwise gold conflicts)
            from itertools import combinations

            for wa, wb, wc in combinations(loaded_feats, 3):
                pair_ids = [
                    (wa["feature_id"], wb["feature_id"]),
                    (wb["feature_id"], wc["feature_id"]),
                    (wa["feature_id"], wc["feature_id"]),
                ]
                has_conf = any(
                    (rname, tid, x, y) in conflict_set or (rname, tid, y, x) in conflict_set
                    for x, y in pair_ids
                )
                if not has_conf:
                    three_wave_chains.append(
                        {
                            "repo_dir_name": rname,
                            "github_repo": wa["github_repo"],
                            "task_id": tid,
                            "base_commit": wa["base_commit"],
                            "has_any_conflict_pair": False,
                            "waves": [wa, wb, wc],
                        }
                    )

        for p in entry["pairs"]:
            f1, f2 = int(p[0]), int(p[1])
            fa = _load_feature(rname, tid, f1)
            fb = _load_feature(rname, tid, f2)
            if not fa or not fb:
                continue
            is_conf = (rname, tid, f1, f2) in conflict_set or (rname, tid, f2, f1) in conflict_set
            pair_rec = {
                "pair_uid": f"cooperbench_pair::{rname}::task{tid}::f{f1}_f{f2}",
                "repo_dir_name": rname,
                "github_repo": fa["github_repo"],
                "task_id": tid,
                "base_commit": fa["base_commit"],
                "has_gold_overlap_conflict": is_conf,
                "feature_a": fa,
                "feature_b": fb,
            }
            if is_conf:
                conflict_pairs.append(pair_rec)
            else:
                clean_pairs.append(pair_rec)

    return {
        "clean_pairs": _round_robin_by_key(clean_pairs, "github_repo"),
        "conflict_pairs": _round_robin_by_key(conflict_pairs, "github_repo"),
        "single_features": _round_robin_by_key(single_features, "github_repo"),
        "three_wave_chains": _round_robin_by_key(three_wave_chains, "github_repo"),
    }


def _incremental_diff_spec(short_prompt: str, full_prompt: str) -> str:
    """Build lossless `turn_1c` specification relative to `short_prompt` (`NEW-1`).

    In `Ambig-SWE`, `short_prompt` (`underspecified_prompt`) is an LLM-paraphrased summary of the symptom
    rather than a verbatim substring of `full_prompt`. Previously, when `joined == full_prompt.strip()`,
    falling back to `split_markdown_sections(full_prompt)[1]` dropped up to 98% of `full_prompt`!
    Now:
    - If `full_prompt` literally begins with `short_prompt` (verbatim prefix), drop only that verbatim prefix;
    - Otherwise deliver the complete fence-balanced `full_prompt` so `(turn_1a + turn_1c)` preserves 100% of
      `full_problem_statement` constraints, tracebacks, and reproduction inputs without verbatim `[:80]` overlap.
    """
    s_clean = short_prompt.strip()
    f_clean = full_prompt.strip()
    if not f_clean:
        return s_clean

    if s_clean and f_clean.startswith(s_clean):
        rem = f_clean[len(s_clean):].strip()
        if len(rem) >= 50 and rem.count("```") % 2 == 0:
            return rem

    blocks = _split_fence_safe_blocks(f_clean)
    short_lines = {ln.strip() for ln in s_clean.splitlines() if len(ln.strip()) >= 24}
    extra_blocks: list[str] = []
    for blk in blocks:
        b_lines = [ln.strip() for ln in blk.splitlines() if len(ln.strip()) >= 24]
        if not b_lines or any(ln not in short_lines for ln in b_lines):
            extra_blocks.append(blk)

    joined = "\n\n".join(extra_blocks).strip()
    if (
        joined
        and len(joined) >= max(80, int(0.65 * max(1, len(f_clean) - len(s_clean))))
        and joined.count("```") % 2 == 0
        and (not s_clean[:80] or s_clean[:80] not in joined)
    ):
        return joined

    if f_clean.count("```") % 2 != 0:
        return f_clean + "\n```"
    return f_clean


def load_swe_interactive_catalog() -> dict[str, list[dict[str, Any]]]:
    """Load `SWE-Interact` (pure-Python tasks) and `Ambig-SWE` (paired underspecified/full Python tasks)."""
    swe_rows = _read_jsonl(POOLS_DIR / "sweinteract_python_pool.jsonl")
    swe_items: list[dict[str, Any]] = []
    for rw in swe_rows:
        if rw["repo"] in EXCLUDED_SWE_REPOS or rw["instance_id"] in EXCLUDED_SWE_INSTANCES:
            continue
        if not _is_py311_compatible_swe_row(rw):
            continue
        ps = str(rw.get("problem_statement") or "").strip()
        if not ps or "\ufffd" in ps or _depends_on_external_media_or_gist(ps):
            continue
        tp = str(rw.get("test_patch") or "")
        gp = str(rw.get("patch") or "")
        if _has_undisclosed_private_test_coupling(ps, gp, tp):
            continue

        clean_f2p = clean_swe_test_entries(rw.get("FAIL_TO_PASS") or [], test_patch=tp)
        clean_p2p = clean_swe_test_entries(rw.get("PASS_TO_PASS") or [], test_patch=tp)
        clean_p2p = [x for x in clean_p2p if x not in clean_f2p]
        # NEW-2: Exclude 0-target log-count tasks and massive >80-target full-suite ARG_MAX bombs
        if not (1 <= len(clean_f2p) <= 80):
            continue

        base_symptom, detail_constraints = split_markdown_sections(ps)
        # NEW-6: Ensure SWE-Interact dynamic split has substantive technical content in BOTH halves
        if detail_constraints.startswith("Ensure all edge cases, exact field/file naming rules"):
            continue
        if _substantive_text_len(detail_constraints) < 50:
            continue

        swe_items.append(
            {
                "uid": f"swe_interact::{rw['instance_id']}",
                "pillar": "swe_interact",
                "instance_id": rw["instance_id"],
                "repo": rw["repo"],
                "base_commit": rw["base_commit"],
                "underspecified_prompt": base_symptom,
                "full_problem_statement": ps,
                "addendum_constraints": detail_constraints,
                "test_cmds": rw.get("test_cmds", []),
                "test_patch": tp,
                "FAIL_TO_PASS": clean_f2p,
                "PASS_TO_PASS": clean_p2p,
                "docker_image": rw.get("docker_image", ""),
                "split": rw.get("_split", "dev_python"),
            }
        )

    ambig_rows = [
        rw
        for rw in _read_jsonl(POOLS_DIR / "ambig_swe_joined_pool.jsonl")
        if rw["repo"] not in EXCLUDED_SWE_REPOS
        and rw["instance_id"] not in EXCLUDED_SWE_INSTANCES
        and _is_py311_compatible_swe_row(rw)
    ]
    ambig_rows.sort(
        key=lambda r: (
            0 if any(r["repo"].startswith(pref) for pref in PREFERRED_AMBIG_REPO_PREFIXES) else 1,
            r["instance_id"],
        )
    )
    ambig_items: list[dict[str, Any]] = []
    for rw in ambig_rows:
        u_prompt = str(rw["underspecified_prompt"]).strip()
        f_prompt = str(rw["full_problem_statement"]).strip()
        if not u_prompt or not f_prompt or "\ufffd" in f_prompt:
            continue
        tp = str(rw.get("test_patch") or "")
        gp = str(rw.get("patch") or "")
        if _has_undisclosed_private_test_coupling(f_prompt, gp, tp):
            continue
        raw_f2p = json.loads(rw["FAIL_TO_PASS"]) if isinstance(rw["FAIL_TO_PASS"], str) else rw["FAIL_TO_PASS"]
        raw_p2p = json.loads(rw["PASS_TO_PASS"]) if isinstance(rw["PASS_TO_PASS"], str) else rw["PASS_TO_PASS"]
        clean_f2p = clean_swe_test_entries(raw_f2p or [], test_patch=tp)
        clean_p2p = [x for x in clean_swe_test_entries(raw_p2p or [], test_patch=tp) if x not in clean_f2p]
        if not (1 <= len(clean_f2p) <= 80):
            continue

        ambig_items.append(
            {
                "uid": f"ambig_swe::{rw['instance_id']}",
                "pillar": "ambig_swe",
                "instance_id": rw["instance_id"],
                "repo": rw["repo"],
                "base_commit": rw["base_commit"],
                "underspecified_prompt": u_prompt,
                "full_problem_statement": f_prompt,
                "addendum_constraints": _incremental_diff_spec(u_prompt, f_prompt),
                "test_patch": tp,
                "FAIL_TO_PASS": clean_f2p,
                "PASS_TO_PASS": clean_p2p,
            }
        )

    swe_pref = [it for it in swe_items if it["repo"] in PREFERRED_SWE_REPOS]
    swe_rest = [it for it in swe_items if it["repo"] not in PREFERRED_SWE_REPOS]
    ambig_pref = [
        it for it in ambig_items if any(it["repo"].startswith(pref) for pref in PREFERRED_AMBIG_REPO_PREFIXES)
    ]
    ambig_rest = [
        it for it in ambig_items if not any(it["repo"].startswith(pref) for pref in PREFERRED_AMBIG_REPO_PREFIXES)
    ]

    return {
        "swe_interact": _round_robin_by_key(swe_pref, "repo") + _round_robin_by_key(swe_rest, "repo"),
        "ambig_swe": _round_robin_by_key(ambig_pref, "repo") + _round_robin_by_key(ambig_rest, "repo"),
    }
