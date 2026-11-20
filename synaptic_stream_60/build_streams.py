"""Deterministic builder and overlap auditor for `Synaptic-Stream-60` (`v3.0.0`).

Builds 3 zero-overlap, conflict-free heterogeneous macro-streams (`3 x 20 = 60` unique
actionable tasks, `240` multi-channel events per stream) paired across `moderate` (`~30s/task`)
and `surge` (`~12s/task`, exact `2.5x` arrival rate scaling) schedules.

Each stream mounts a lightweight 3-project monorepo (`projects/<project_id>/`) with 3 strictly
disjoint independent coding tasks (0% file overlap, 0 merge conflicts, 0% overlap with PA-150
including H-4 waves and SWE pools) interleaved with 11 multi-step AppWorld writes and 6 read-only
AppWorld QA lookups across 17 unique accounts (0% overlap with Micro-30).
"""
from __future__ import annotations

import hashlib
from itertools import permutations
import json
from pathlib import Path
import random
import re
from typing import Any

from appworld_micro_30.build_dataset import _GROUP_NOISE_POOL, USER_APP_NAMES, detect_target_apps
from synaptic_stream_60 import RELEASE
from synaptic_stream_60.schema import (
    DEFAULT_TAU_SEC,
    MacroStreamSpec,
    StreamAccountSpec,
    StreamEventSpec,
    StreamProjectSpec,
    StreamTaskSpec,
    stream_fingerprint,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_OUT_DIR = Path(__file__).resolve().parent / "data"
STREAMS_JSONL = DATA_OUT_DIR / "streams_60.jsonl"
MANIFEST_JSON = DATA_OUT_DIR / "manifest_60.json"

_EXTRA_NOISE_MESSAGES: tuple[tuple[str, str], ...] = (
    ("dev_bot", "FYI nightly container base image digest refreshed cleanly on staging mirror."),
    ("nina_product", "Leaving the Q4 roadmap brainstorm Miro link here for optional async comments next week."),
    ("marco_sre", "Read-only CDN cache hit ratio held steady at 96.4% across APAC edge nodes."),
    ("elena_design", "Uploaded updated icon SVGs to the shared Figma playground—no dev action needed."),
    ("sam_qa", "Regression smoke matrix on Linux arm64 runners finished green in 4m12s."),
    ("priya_data", "dbt documentation site rebuilt after metadata tag cleanup; schemas unchanged."),
    ("alex_sec", "Routine TLS certificate renewal on internal docs wiki completed with zero downtime."),
    ("jordan_ops", "Coffee beans in the 5th-floor pantry were restocked with medium roast."),
    ("taylor_hr", "Friendly reminder: optional Flu shot clinic runs Tuesday 2-4pm in Room Aspen."),
    ("chris_eng", "TIL Python 3.11 exception groups print super cleanly with pytest -vv, just sharing!"),
    ("morgan_pm", "Customer demo recording from yesterday's webinar is archived in the team folder."),
    ("casey_infra", "Grafana alert silence on sandbox-dev-02 expired; metrics look completely nominal."),
)

def _noise_catalog() -> list[tuple[str, str]]:
    """Concrete team updates, without serial-number suffixes or repeated messages."""
    catalog = [(sender.strip(), text.strip()) for sender, text in (*_GROUP_NOISE_POOL, *_EXTRA_NOISE_MESSAGES)]
    topics = (
        ("marco_sre", ("staging API", "preview cluster", "metrics gateway", "build cache", "package mirror", "log collector", "internal DNS", "artifact registry"),
         ("The {topic} maintenance window is over; traffic looks normal.", "Today's {topic} latency chart is in the ops notes for reference.", "The {topic} health check was green at handover; no follow-up needed.")),
        ("sam_qa", ("Linux smoke suite", "Windows installer check", "macOS packaging check", "browser compatibility pass", "accessibility smoke check", "mobile layout check", "upgrade-path test", "localization sanity check"),
         ("The {topic} finished cleanly on the nightly build.", "I've archived the {topic} screenshots alongside the release notes.", "The {topic} results are already in the QA digest; nothing blocking there.")),
        ("nina_product", ("onboarding walkthrough", "customer demo", "pricing workshop", "research debrief", "roadmap discussion", "partner Q&A", "support handover", "planning session"),
         ("The {topic} recording is available in the shared folder.", "I added timestamps to the {topic} notes so they're easier to skim.", "The slides from the {topic} are now in the team archive.")),
        ("elena_design", ("settings icons", "empty-state illustrations", "navigation sketches", "profile mockups", "notification layouts", "search examples", "mobile wireframes", "color studies"),
         ("The {topic} are in the design playground for reference, with no implementation request.", "I've grouped the {topic} into a separate Figma page for easier browsing.", "The design review notes for the {topic} are filed with the originals.")),
        ("jordan_ops", ("third-floor pantry", "quiet room", "west meeting room", "visitor lounge", "bike storage area", "training room", "east phone booth", "library corner"),
         ("The {topic} is open again after routine cleaning.", "The updated room guide includes a photo of the {topic}.", "Facilities confirmed the {topic} is ready for tomorrow's visitors.")),
        ("priya_data", ("retention dashboard", "signup funnel", "support volume report", "weekly usage summary", "regional traffic chart", "trial conversion report", "latency histogram", "capacity forecast"),
         ("The {topic} refreshed successfully this morning.", "I added the data cutoff time to the {topic} for context.", "The archived copy of the {topic} is in the analytics folder.")),
        ("chris_eng", ("Python packaging guide", "local setup guide", "debugging handbook", "logging guide", "release checklist", "code review guide", "dependency update guide", "test runner guide"),
         ("The {topic} has an updated table of contents.", "Broken internal links in the {topic} are fixed now.", "I've added a small glossary to the {topic}; no code changes involved.")),
    )
    for sender, subjects, templates in topics:
        for subject in subjects:
            for template in templates:
                catalog.append((sender, template.format(topic=subject)))
    return list({text: (sender, text) for sender, text in catalog}.values())


def _clean_numbered_spec(text: str) -> str:
    # Only top-level lists, never code blocks or nested bullet indentation.
    lines = text.splitlines()
    count = 0
    in_code = False
    for i, line in enumerate(lines):
        if line.lstrip().startswith("```"):
            in_code = not in_code
        if in_code:
            continue
        match = re.match(r"^(\d+)\. (.*)", line)
        if match:
            count = 1 if match.group(1) == "1" else count + 1
            lines[i] = f"{count}. {match.group(2)}"
        elif line and not line.startswith((" ", "-")):
            count = 0
    return re.sub(r"\n[ \t]*\n(?:[ \t]*\n)+", "\n\n", "\n".join(lines)).strip()


# Three conflict-free heterogeneous streams: 3 strictly distinct lightweight Python repos per stream
# (9 unique held-out coding tasks total; 0% overlap with PA-150 including H-4 waves & SWE pools;
# 100% offline deterministic fail-to-pass Gold with zero hidden-symbol or inverted-test traps).
_REPO_STREAM_CONFIGS: tuple[dict[str, Any], ...] = (
    {
        "stream_id": "STREAM_01",
        "repo_slug": "workspace/multi-project-monorepo",
        "projects": (
            {
                "project_id": "click",
                "subdir": "projects/click",
                "repo_slug": "pallets/click",
                "source_kind": "cooperbench",
                "cooper_repo_dir": "pallets_click_task",
                "cooper_task_dir": "task2956",
                "feature_name": "feature2",
            },
            {
                "project_id": "jinja",
                "subdir": "projects/jinja",
                "repo_slug": "pallets/jinja",
                "source_kind": "cooperbench",
                "cooper_repo_dir": "pallets_jinja_task",
                "cooper_task_dir": "task1621",
                "feature_name": "feature9",
            },
            {
                "project_id": "dspy",
                "subdir": "projects/dspy",
                "repo_slug": "stanfordnlp/dspy",
                "source_kind": "cooperbench",
                "cooper_repo_dir": "dspy_task",
                "cooper_task_dir": "task8635",
                "feature_name": "feature3",
            },
        ),
        "amended_project_id": "click",
    },
    {
        "stream_id": "STREAM_02",
        "repo_slug": "workspace/multi-project-monorepo",
        "projects": (
            {
                "project_id": "click",
                "subdir": "projects/click",
                "repo_slug": "pallets/click",
                "source_kind": "cooperbench",
                "cooper_repo_dir": "pallets_click_task",
                "cooper_task_dir": "task2956",
                "feature_name": "feature3",
            },
            {
                "project_id": "jinja",
                "subdir": "projects/jinja",
                "repo_slug": "pallets/jinja",
                "source_kind": "cooperbench",
                "cooper_repo_dir": "pallets_jinja_task",
                "cooper_task_dir": "task1621",
                "feature_name": "feature10",
            },
            {
                "project_id": "dspy",
                "subdir": "projects/dspy",
                "repo_slug": "stanfordnlp/dspy",
                "source_kind": "cooperbench",
                "cooper_repo_dir": "dspy_task",
                "cooper_task_dir": "task8635",
                "feature_name": "feature4",
            },
        ),
        "amended_project_id": "dspy",
    },
    {
        "stream_id": "STREAM_03",
        "repo_slug": "workspace/multi-project-monorepo",
        "projects": (
            {
                "project_id": "click",
                "subdir": "projects/click",
                "repo_slug": "pallets/click",
                "source_kind": "cooperbench",
                "cooper_repo_dir": "pallets_click_task",
                "cooper_task_dir": "task2956",
                "feature_name": "feature4",
            },
            {
                "project_id": "dspy",
                "subdir": "projects/dspy",
                "repo_slug": "stanfordnlp/dspy",
                "source_kind": "cooperbench",
                "cooper_repo_dir": "dspy_task",
                "cooper_task_dir": "task8635",
                "feature_name": "feature6",
            },
            {
                "project_id": "networkx",
                "subdir": "projects/networkx",
                "repo_slug": "networkx/networkx",
                "source_kind": "swe_interact",
                "pool_file": "sweinteract_python_pool.jsonl",
                "instance_id": "networkx__networkx-8531",
            },
        ),
        "amended_project_id": "dspy",
    },
)

# Three independent heavy coding requests evenly spaced (slots 1, 8, 15) across eleven writes and six QA lookups.
_SLOT_TEMPLATE: tuple[str, ...] = (
    "code_independent_0", "readonly_qa_0", "app_write_0", "app_write_1",
    "readonly_qa_1", "app_write_2", "app_write_3", "code_independent_1",
    "readonly_qa_2", "app_write_4", "app_write_5", "readonly_qa_3",
    "app_write_6", "app_write_7", "code_independent_2", "readonly_qa_4",
    "app_write_8", "app_write_9", "readonly_qa_5", "app_write_10",
)


def _load_existing_benchmark_ids() -> tuple[set[str], set[str], set[str], set[str]]:
    """Return `(pa150_cooper_features, pa150_swe_instances, micro30_aw_task_ids, micro30_aw_generators)`.

    Scans the full serialized `episodes_pa150.jsonl` text (including `H-4` `setup_meta['waves']`,
    `tests_patches`, and verifier parameters) so no upstream coding feature or SWE instance can
    slip past overlap checks.
    """
    pa_path = PROJECT_ROOT / "synaptic_bench_150" / "data" / "episodes_pa150.jsonl"
    mi_path = PROJECT_ROOT / "appworld_micro_30" / "data" / "episodes_appworld_micro30.jsonl"

    pa_feats: set[str] = set()
    pa_swe: set[str] = set()
    for ln in pa_path.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        for match in re.finditer(r"([a-zA-Z0-9_]+_task)[/:]+(task\d+)[/:]+(feature\d+)", ln):
            pa_feats.add(f"{match.group(1)}/{match.group(2)}/{match.group(3)}")
        for match in re.finditer(r"([A-Za-z0-9_.-]+__[A-Za-z0-9_.-]+-\d+)", ln):
            pa_swe.add(match.group(1))
        ep = json.loads(ln)
        for m in ep.get("workspace_mounts", []):
            sm = m.get("setup_meta") or {}
            if m.get("pillar") == "cooperbench":
                rn = str(sm.get("repo_dir_name") or "")
                tid = str(sm.get("task_id") or "")
                if tid and not tid.startswith("task"):
                    tid = f"task{tid}"
                for fk in ("feature_a_id", "feature_b_id", "feature_id"):
                    if sm.get(fk) is not None:
                        pa_feats.add(f"{rn}/{tid}/feature{sm[fk]}")
            for sk in ("instance_id", "swe_instance_id"):
                if sm.get(sk):
                    pa_swe.add(str(sm[sk]))

    mi_tids: set[str] = set()
    for ln in mi_path.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        ep = json.loads(ln)
        for st in ep.get("subtasks", []):
            mi_tids.add(str(st["upstream_task_id"]))
    mi_gens = {t.split("_")[0] for t in mi_tids}
    return pa_feats, pa_swe, mi_tids, mi_gens


def _select_zero_overlap_appworld_tasks(
    excluded_tids: set[str],
    excluded_gens: set[str],
) -> tuple[list[list[dict[str, Any]]], list[list[dict[str, Any]]]]:
    """Deterministically select `3 x 11 = 33` write tasks and `3 x 6 = 18` QA tasks from `train+dev`.

    Invariants:
      1. 100% of selected tasks have executable `ground_truth/compiled_solution.py`.
      2. 0% task-ID overlap and 0% generator-ID overlap with `AppWorld-Micro-30`.
      3. Within every stream, all 17 tasks come from 17 strictly distinct generators and personas.
      4. Across all 3 streams, all 51 task IDs are 100% unique (0% cross-stream task reuse).
    """
    data_dir = PROJECT_ROOT / "data"
    write_by_gen: dict[str, list[dict[str, Any]]] = {}
    qa_by_gen: dict[str, list[dict[str, Any]]] = {}

    for split in ("dev", "train"):
        fp = data_dir / "datasets" / f"{split}.txt"
        for raw in fp.read_text(encoding="utf-8").splitlines():
            tid = raw.strip()
            if not tid or tid in excluded_tids:
                continue
            gen = tid.split("_")[0]
            if gen in excluded_gens:
                continue
            tdir = data_dir / "tasks" / tid
            if not (tdir / "ground_truth" / "compiled_solution.py").is_file():
                continue
            specs = json.loads((tdir / "specs.json").read_text(encoding="utf-8"))
            meta = json.loads((tdir / "ground_truth" / "metadata.json").read_text(encoding="utf-8"))
            ans = json.loads((tdir / "ground_truth" / "answer.json").read_text(encoding="utf-8"))
            db_delta_apps = [
                app
                for app in USER_APP_NAMES
                if (tdir / "dbs" / f"{app}.jsonl").read_text(encoding="utf-8").strip() != ""
            ]
            t_apps = detect_target_apps(tdir, specs["instruction"], db_delta_apps)
            if not t_apps:
                continue
            is_qa = (ans.get("answer") is not None) if isinstance(ans, dict) else (ans is not None)
            gt_val = ans.get("answer") if isinstance(ans, dict) else ans
            sup = specs.get("supervisor") or {}
            sup_name = f"{sup.get('first_name', '')} {sup.get('last_name', '')}".strip() or "Supervisor"
            sup_email = str(sup.get("email") or "user@example.com")
            item = {
                "tid": tid,
                "gen": gen,
                "split": split,
                "difficulty": int(meta.get("difficulty", 2)),
                "num_api_calls": int(meta.get("num_api_calls", 20)),
                "is_qa": is_qa,
                "ground_truth_answer": gt_val,
                "instruction": str(specs["instruction"]).strip(),
                "datetime_str": str(specs.get("datetime") or "2023-05-18T12:00:00"),
                "supervisor_name": sup_name,
                "supervisor_email": sup_email,
                "target_apps": t_apps,
            }
            if is_qa:
                qa_by_gen.setdefault(gen, []).append(item)
            elif item["difficulty"] >= 2:
                write_by_gen.setdefault(gen, []).append(item)

    for g_dict in (write_by_gen, qa_by_gen):
        for g in g_dict:
            g_dict[g].sort(key=lambda x: (x["difficulty"], x["num_api_calls"], x["tid"]))

    # Rank eligible multi-step write generators deterministically by difficulty and API footprint
    ranked_write_gens = sorted(
        write_by_gen.keys(),
        key=lambda g: (
            write_by_gen[g][0]["difficulty"],
            write_by_gen[g][0]["num_api_calls"],
            hashlib.sha256(f"SS60-W:{g}".encode("utf-8")).hexdigest(),
        ),
    )
    # Use all four held-out difficulty-2 QA generators plus two difficulty-1 generators (6/stream = 18 unique QAs).
    qa_medium = {g: [x for x in xs if x["difficulty"] == 2] for g, xs in qa_by_gen.items()}
    medium_gens = sorted(g for g, xs in qa_medium.items() if len(xs) >= 3)
    assert len(medium_gens) == 4 and sum(len(qa_medium[g]) for g in medium_gens) == 12
    easy_gens = sorted(g for g, xs in qa_by_gen.items()
                       if g not in medium_gens and sum(x["difficulty"] == 1 for x in xs) >= 3)
    assert len(easy_gens) >= 2
    selected_qa_gens = medium_gens + easy_gens[:2]
    assert len(selected_qa_gens) == 6
    for g in medium_gens:
        qa_by_gen[g] = qa_medium[g]
    for g in easy_gens[:2]:
        qa_by_gen[g] = [x for x in qa_by_gen[g] if x["difficulty"] == 1]

    # Allocate QA instances first, trying deterministic permutations across streams.
    # Backtracking avoids committing a local choice that blocks a later generator.
    stream_qas: list[list[dict[str, Any]]] = [[], [], []]
    names: list[set[str]] = [set(), set(), set()]
    emails: list[set[str]] = [set(), set(), set()]

    def assign_qa(index: int) -> bool:
        if index == len(selected_qa_gens):
            return True
        g = selected_qa_gens[index]
        for assignment in permutations(qa_by_gen[g][:3]):
            if any(x["supervisor_name"] in names[i] or x["supervisor_email"] in emails[i]
                   for i, x in enumerate(assignment)):
                continue
            for i, x in enumerate(assignment):
                stream_qas[i].append(x)
                names[i].add(x["supervisor_name"])
                emails[i].add(x["supervisor_email"])
            if assign_qa(index + 1):
                return True
            for i, x in enumerate(assignment):
                stream_qas[i].pop()
                names[i].remove(x["supervisor_name"])
                emails[i].remove(x["supervisor_email"])
        return False

    if not assign_qa(0):
        raise ValueError("No collision-free assignment of QA supervisors")

    # Cycle through eligible write generators, preserving the deterministic ranking (11 writes/stream = 33 unique writes).
    stream_writes: list[list[dict[str, Any]]] = [[], [], []]
    used_write_tids: set[str] = set()
    gen_pool_idx = 0
    for s_idx in range(3):
        used_gens_in_stream = set(selected_qa_gens)
        misses = 0
        while len(stream_writes[s_idx]) < 11:
            g = ranked_write_gens[gen_pool_idx % len(ranked_write_gens)]
            gen_pool_idx += 1
            cand = next((x for x in write_by_gen[g]
                         if g not in used_gens_in_stream
                         and x["tid"] not in used_write_tids
                         and x["supervisor_name"] not in names[s_idx]
                         and x["supervisor_email"] not in emails[s_idx]), None)
            if cand is None:
                misses += 1
                if misses >= len(ranked_write_gens):
                    raise ValueError(f"Not enough distinct write supervisors for stream {s_idx}")
                continue
            misses = 0
            used_write_tids.add(cand["tid"])
            used_gens_in_stream.add(g)
            names[s_idx].add(cand["supervisor_name"])
            emails[s_idx].add(cand["supervisor_email"])
            stream_writes[s_idx].append(cand)
        assert len(names[s_idx]) == len(emails[s_idx]) == 17

    return stream_writes, stream_qas


def _feature_title(spec_md: str) -> str:
    """Extract the upstream title without Markdown or commit-message decoration."""
    title = next(line.strip() for line in spec_md.splitlines() if line.strip())
    title = title.lstrip("# ").replace("**", "").strip()
    title = re.sub(r"^(?:title|feature)\s*:\s*", "", title, flags=re.I)
    return re.sub(r"^feat(?:\([^)]*\))?!?:\s*", "", title, flags=re.I).strip()


def _load_swe_pool_row(pool_file: str, instance_id: str) -> dict[str, Any]:
    fp = PROJECT_ROOT / "data" / "upstream_pillars" / "pools" / pool_file
    for ln in fp.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        row = json.loads(ln)
        if row.get("instance_id") == instance_id:
            return row
    raise KeyError(f"Missing SWE instance {instance_id!r} in {fp}")


def _resolve_stream_coding_entries(
    cfg: dict[str, Any],
    pa_cb_feats: set[str],
    pa_swe_ids: set[str],
) -> tuple[list[StreamProjectSpec], list[dict[str, Any]]]:
    cb_root = PROJECT_ROOT / "data" / "upstream_pillars" / "CooperBench" / "dataset"
    swe_cache_root = DATA_OUT_DIR / "swe_patches"
    projects: list[StreamProjectSpec] = []
    entries: list[dict[str, Any]] = []

    for item in cfg["projects"]:
        source_kind = str(item["source_kind"])
        pid = str(item["project_id"])
        subdir = str(item["subdir"])
        rslug = str(item["repo_slug"])
        if source_kind == "cooperbench":
            cb_repo = str(item["cooper_repo_dir"])
            cb_task = str(item["cooper_task_dir"])
            fn = str(item["feature_name"])
            key = f"{cb_repo}/{cb_task}/{fn}"
            if key in pa_cb_feats:
                raise AssertionError(f"Overlap detected with PA-150 on CooperBench feature {key}!")
            tdir = cb_root / cb_repo / cb_task
            setup_txt = (tdir / "setup.sh").read_text(encoding="utf-8")
            m_sha = re.search(r'BASE_COMMIT="([0-9a-f]{40})"', setup_txt)
            if not m_sha:
                raise RuntimeError(f"Could not parse BASE_COMMIT from {tdir / 'setup.sh'}")
            base_commit = m_sha.group(1)
            f_dir = tdir / fn
            spec_md = (f_dir / "feature.md").read_text(encoding="utf-8").strip()
            proj = StreamProjectSpec(
                project_id=pid,
                subdir=subdir,
                repo_slug=rslug,
                cooper_repo_dir=cb_repo,
                cooper_task_dir=cb_task,
                base_commit=base_commit,
                source_kind="cooperbench",
                upstream_instance_id=None,
            )
            projects.append(proj)
            entries.append(
                {
                    "proj": proj,
                    "feature_name": fn,
                    "spec_md": spec_md,
                    "feature_patch_relpath": str((f_dir / "feature.patch").relative_to(PROJECT_ROOT)),
                    "tests_patch_relpath": str((f_dir / "tests.patch").relative_to(PROJECT_ROOT)),
                }
            )
        else:
            iid = str(item["instance_id"])
            if iid in pa_swe_ids:
                raise AssertionError(f"Overlap detected with PA-150 on SWE instance {iid}!")
            row = _load_swe_pool_row(str(item["pool_file"]), iid)
            base_commit = str(row["base_commit"])
            raw_spec = str(row.get("full_problem_statement") or row.get("problem_statement") or "").strip()
            # Normalize CRLF line endings from GitHub issue bodies cleanly
            raw_spec = raw_spec.replace("\r\n", "\n").strip()
            p_dir = swe_cache_root / iid
            p_dir.mkdir(parents=True, exist_ok=True)
            fp_path = p_dir / "feature.patch"
            tp_path = p_dir / "tests.patch"
            fp_txt = str(row["patch"])
            tp_txt = str(row["test_patch"])
            if not fp_txt.endswith("\n"):
                fp_txt += "\n"
            if not tp_txt.endswith("\n"):
                tp_txt += "\n"
            fp_path.write_text(fp_txt, encoding="utf-8")
            tp_path.write_text(tp_txt, encoding="utf-8")
            proj = StreamProjectSpec(
                project_id=pid,
                subdir=subdir,
                repo_slug=rslug,
                cooper_repo_dir=source_kind,
                cooper_task_dir=iid,
                base_commit=base_commit,
                source_kind=source_kind,
                upstream_instance_id=iid,
            )
            projects.append(proj)
            entries.append(
                {
                    "proj": proj,
                    "feature_name": iid,
                    "spec_md": raw_spec,
                    "feature_patch_relpath": str(fp_path.relative_to(PROJECT_ROOT)),
                    "tests_patch_relpath": str(tp_path.relative_to(PROJECT_ROOT)),
                }
            )
    return projects, entries


def _build_single_macro_stream(
    stream_idx: int,
    cfg: dict[str, Any],
    aw_writes: list[dict[str, Any]],
    aw_qas: list[dict[str, Any]],
    pa_cb_feats: set[str],
    pa_swe_ids: set[str],
) -> MacroStreamSpec:
    stream_id = str(cfg["stream_id"])
    repo_slug = str(cfg["repo_slug"])
    projects, coding_entries = _resolve_stream_coding_entries(cfg, pa_cb_feats, pa_swe_ids)

    rng = random.Random(20261118 + stream_idx * 101)

    # Assign accounts acct_01 .. acct_16 (ten writes and six QA)
    accounts: list[StreamAccountSpec] = []
    aw_write_with_acct: list[tuple[str, dict[str, Any]]] = []
    aw_qa_with_acct: list[tuple[str, dict[str, Any]]] = []

    acct_seq = 0
    for w_item in aw_writes:
        acct_seq += 1
        aid = f"acct_{acct_seq:02d}"
        accounts.append(
            StreamAccountSpec(
                account_id=aid,
                upstream_task_id=w_item["tid"],
                split=w_item["split"],
                supervisor_name=w_item["supervisor_name"],
                supervisor_email=w_item["supervisor_email"],
                datetime_str=w_item["datetime_str"],
                target_apps=list(w_item["target_apps"]),
                is_qa=False,
            )
        )
        aw_write_with_acct.append((aid, w_item))

    for q_item in aw_qas:
        acct_seq += 1
        aid = f"acct_{acct_seq:02d}"
        accounts.append(
            StreamAccountSpec(
                account_id=aid,
                upstream_task_id=q_item["tid"],
                split=q_item["split"],
                supervisor_name=q_item["supervisor_name"],
                supervisor_email=q_item["supervisor_email"],
                datetime_str=q_item["datetime_str"],
                target_apps=list(q_item["target_apps"]),
                is_qa=True,
            )
        )
        aw_qa_with_acct.append((aid, q_item))

    # Build the 20 ordered tasks with deterministic Moderate (`~30s/task`) & Surge (`exact / 2.5 = ~12s/task`) offsets
    tasks: list[StreamTaskSpec] = []
    code_titles_by_task: dict[str, str] = {}

    for slot_zero, slot_key in enumerate(_SLOT_TEMPLATE):
        slot_num = slot_zero + 1
        task_id = f"{stream_id}_T{slot_num:02d}"
        jitter = 0.0 if slot_zero == 0 else round(rng.uniform(-3.0, 3.0), 1)
        mod_off = max(0.0, round(slot_zero * 30.0 + jitter, 1))
        sur_off = mod_off / 2.5

        kind_prefix, idx_str = slot_key.rsplit("_", 1)
        sub_idx = int(idx_str)

        if kind_prefix == "code_independent":
            entry = coding_entries[sub_idx]
            proj: StreamProjectSpec = entry["proj"]
            fn = str(entry["feature_name"])
            spec_md = str(entry["spec_md"])
            title = _feature_title(spec_md)
            code_titles_by_task[task_id] = title
            coding_intros = (
                "Could you work on \"{title}\" in `{repo}` (under `{subdir}/`)?",
                "Please take on \"{title}\" in `{repo}` (located at `{subdir}/`).",
                "For `{repo}` (`{subdir}/`), we need help with \"{title}\":",
                "Can you handle \"{title}\" for `{repo}` in `{subdir}/`?",
            )
            prompt_text = (
                coding_intros[sub_idx].format(title=title, repo=proj.repo_slug, subdir=proj.subdir)
                + f"\n\n{spec_md}"
            )
            channel = "feishu://group/engineering" if (sub_idx % 2 == 1) else "feishu://p2p/user"
            sender = "maintainer_alex" if "/group/" in channel else "viacol"
            tasks.append(
                StreamTaskSpec(
                    task_id=task_id,
                    stream_id=stream_id,
                    slot_index=slot_num,
                    category="code_independent",
                    substrate="cooperbench",
                    moderate_offset_sec=mod_off,
                    surge_offset_sec=sur_off,
                    channel=channel,
                    sender=sender,
                    prompt_text=prompt_text,
                    cooper_feature_name=fn,
                    feature_patch_relpath=str(entry["feature_patch_relpath"]),
                    tests_patch_relpath=str(entry["tests_patch_relpath"]),
                    coupled_peer_task_id=None,
                    project_id=proj.project_id,
                    project_subdir=proj.subdir,
                    repo_slug=proj.repo_slug,
                    base_commit=proj.base_commit,
                )
            )
        elif kind_prefix == "app_write":
            aid, w_item = aw_write_with_acct[sub_idx]
            write_intros = (
                "Helping {name} with something on their account:",
                "Could you take care of this request for {name}?",
                "Passing along a request from {name} for their account:",
                "{name} needs a hand with this:",
                "Please help {name} with the following request:",
                "Here's what {name} would like taken care of:",
                "Can you handle this for {name}?",
                "A request from {name} for you:",
                "{name} asked us to sort this out:",
                "Could you look after this item for {name}?",
                "One more account request from {name}:",
            )
            prompt_text = write_intros[sub_idx].format(name=w_item["supervisor_name"]) + f'\n"{w_item["instruction"]}"'
            channel = "feishu://p2p/user" if (sub_idx % 3 != 2) else "feishu://group/personal_ops"
            sender = "viacol" if channel.endswith("/user") else "nina_product"
            tasks.append(
                StreamTaskSpec(
                    task_id=task_id,
                    stream_id=stream_id,
                    slot_index=slot_num,
                    category="app_write",
                    substrate="appworld",
                    moderate_offset_sec=mod_off,
                    surge_offset_sec=sur_off,
                    channel=channel,
                    sender=sender,
                    prompt_text=prompt_text,
                    account_id=aid,
                    upstream_task_id=w_item["tid"],
                    target_apps=list(w_item["target_apps"]),
                    expected_readonly=False,
                    difficulty=w_item["difficulty"],
                    generator_id=w_item["gen"],
                )
            )
        elif kind_prefix == "readonly_qa":
            aid, q_item = aw_qa_with_acct[sub_idx]
            qa_intros = (
                "Quick lookup on {name}'s account — please don't change anything, just check, record the concise answer, and send it back:",
                "Could you check this for {name}? Leave all records unchanged, record a concise answer, and send it back:",
                "{name} has a question about their account. Please only look, don't modify any data, record the short answer, and send it back:",
                "Please look this up for {name} without editing any records, then record the concise answer and send it back:",
                "For {name}, could you check the following? Keep the data unchanged, record just the answer, and send it back:",
                "Read-only check for {name}: don't alter any records, just record the bare answer value and send it back:",
            )
            prompt_text = qa_intros[sub_idx].format(name=q_item["supervisor_name"]) + f'\n"{q_item["instruction"]}"'
            tasks.append(
                StreamTaskSpec(
                    task_id=task_id,
                    stream_id=stream_id,
                    slot_index=slot_num,
                    category="readonly_qa",
                    substrate="appworld",
                    moderate_offset_sec=mod_off,
                    surge_offset_sec=sur_off,
                    channel="feishu://p2p/user",
                    sender="viacol",
                    prompt_text=prompt_text,
                    account_id=aid,
                    upstream_task_id=q_item["tid"],
                    target_apps=list(q_item["target_apps"]),
                    expected_readonly=True,
                    ground_truth_answer=q_item["ground_truth_answer"],
                    difficulty=q_item["difficulty"],
                    generator_id=q_item["gen"],
                )
            )
        else:
            raise ValueError(f"Unexpected slot key: {slot_key}")

    # Two genuine incremental specifications per stream: one directly-tested coding requirement,
    # one concrete recipient/selection filter in a multi-step AppWorld workflow.
    amended_pid = str(cfg["amended_project_id"])
    code = next(t for t in tasks if t.project_id == amended_pid)
    if stream_idx == 0:
        # Target sentence verified by `test_validators_info_dict`
        clause_re = re.compile(r" [A-Z][^\n.]*info_dict\. [^\n.]*validators[^\n.]*\.")
        amend_txt = (
            "Also include `validators` in `Option.to_info_dict()` (as a list of validator function "
            "names, or an empty list `[]` when no validators are configured)."
        )
    elif stream_idx == 1:
        # Target bullet verified by `test_description_truncation_at_sentence_boundary`
        clause_re = re.compile(r"- [^\n]*trailing space[^\n]*ellipsis[^\n]*\n")
        amend_txt = (
            "Preserve the trailing space from the sentence boundary before appending the "
            "single-character ellipsis `…` (the prefix before `…` must end with `. `, `! `, or `? `)."
        )
    else:
        # Target bullet verified by `test_short_instruction_fallback`
        clause_re = re.compile(r"   - [^\n]*min_instr_chars[^\n]*get_signature[^\n]*\n")
        amend_txt = (
            "When the generated instruction length is `< min_instr_chars`, fall back to the "
            "predictor's canonical signature instruction `get_signature(predictor).instructions`."
        )
    m_code = clause_re.search(code.prompt_text)
    assert m_code is not None, (stream_id, clause_re.pattern)
    code.prompt_text = _clean_numbered_spec(code.prompt_text.replace(m_code.group(0), "", 1))
    code.amendment_text = amend_txt

    # Select one concrete AppWorld write filter/recipient clause to withhold and deliver mid-flight.
    phone_re = re.compile(r"\bto\b[^.,\n]{1,40}\b(\d{7,})\b")
    year_re = re.compile(r"\bthat\b[^.,\n]{1,25}\bbefore\s+(\d{4})\s+year\b")
    write = next(
        (t for t in tasks if t.category == "app_write" and phone_re.search(t.prompt_text)),
        None,
    )
    if write is not None:
        match = phone_re.search(write.prompt_text)
        assert match
        write.amendment_text = f"The recipient is the person with phone number {match.group(1)}."
        write.prompt_text = write.prompt_text.replace(match.group(0), "to my contact; I will send their phone number shortly", 1)
    else:
        write = next(
            t for t in tasks if t.category == "app_write" and year_re.search(t.prompt_text)
        )
        match = year_re.search(write.prompt_text)
        assert match
        write.amendment_text = f"The cutoff year is {match.group(1)} (remove songs released before {match.group(1)})."
        write.prompt_text = write.prompt_text.replace(
            match.group(0), "that were released before the cutoff year (I will send the exact year shortly)", 1
        )

    for task in tasks:
        if "/group/" in task.channel:
            task.prompt_text = "@agent " + task.prompt_text

    # 20 requests + 2 amendments + 178 noise + 24 reposts + 16 peeks = 240.
    raw_events: list[StreamEventSpec] = []
    for t in tasks:
        raw_events.append(
            StreamEventSpec(
                event_id=f"ev_req_{t.slot_index:02d}",
                moderate_offset_sec=t.moderate_offset_sec,
                surge_offset_sec=t.surge_offset_sec,
                kind="work_request",
                channel=t.channel,
                sender=t.sender,
                text=t.prompt_text,
                target_task_id=t.task_id,
            )
        )

    account_names = {a.account_id: a.supervisor_name for a in accounts}
    code_title = code_titles_by_task[code.task_id]
    for task in (code, write):
        subject = (
            f"{code_title} in `{task.repo_slug}` (`{task.project_subdir}/`)"
            if task is code
            else f"{account_names[task.account_id]}'s request"
        )
        mod_t = round(task.moderate_offset_sec + 18.0, 2)
        raw_events.append(StreamEventSpec(
            event_id="amend_" + task.task_id, moderate_offset_sec=mod_t,
            surge_offset_sec=mod_t / 2.5, kind="work_amendment",
            channel=task.channel, sender=task.sender,
            text=("@agent " if "/group/" in task.channel else "") +
                 f"One detail for {subject}:\n{task.amendment_text}",
            target_task_id=task.task_id,
        ))

    noise_catalog = _noise_catalog()
    assert len(noise_catalog) >= 178
    max_mod_span = 585.0
    for n_idx in range(178):
        sender, base_msg = noise_catalog[(n_idx + stream_idx * 17) % len(noise_catalog)]
        mod_t = round((n_idx + 0.35) * (max_mod_span / 178.0) + rng.uniform(-0.8, 0.8), 2)
        mod_t = max(0.2, min(max_mod_span, mod_t))
        raw_events.append(
            StreamEventSpec(
                event_id=f"ev_noise_{n_idx + 1:03d}",
                moderate_offset_sec=mod_t,
                surge_offset_sec=mod_t / 2.5,
                kind="external_noise",
                channel="feishu://group/" + ("team_general", "engineering", "personal_ops", "ops_alerts")[n_idx % 4],
                sender=sender,
                text=base_msg,
                target_task_id=None,
            )
        )

    # 24 human colleague reposts across the 20 requests.
    # Causal invariant: reposts targeting an amended task ALWAYS fire strictly BEFORE its +18.0s
    # amendment (so a colleague never reposts a stale pre-amendment placeholder after the detail arrived).
    non_amended_tasks = [t for t in tasks if not t.amendment_text]
    dup_targets = list(tasks) + non_amended_tasks[:4]
    for d_idx, dt in enumerate(dup_targets):
        if dt.amendment_text:
            delay_mod = round(rng.uniform(5.0, 14.0), 2)
        else:
            delay_mod = round(rng.uniform(9.0, 22.0) + (12.0 if d_idx >= 20 else 0.0), 2)
        mod_t = round(dt.moderate_offset_sec + delay_mod, 2)
        # Repost only the already delivered core specification, never the withheld amendment.
        body = dt.prompt_text.removeprefix("@agent ")
        repost_intros = (
            "Sharing the earlier request here so everyone has context:",
            "Cross-posting this item from the other chat for visibility:",
            "Bringing this request over from the earlier conversation:",
            "Sharing what was requested in the other chat so we're all in the loop:",
            "Forwarding this earlier request for the team:",
            "Putting this here as well so the team has the context:",
        )
        intro_idx = d_idx % len(repost_intros) if d_idx < 20 else (dt.slot_index - 1 + 3) % len(repost_intros)
        dup_txt = f"@agent {repost_intros[intro_idx]}\n\n{body}"
        raw_events.append(
            StreamEventSpec(
                event_id=f"ev_dup_{d_idx + 1:02d}",
                moderate_offset_sec=mod_t,
                surge_offset_sec=mod_t / 2.5,
                kind="external_duplicate_webhook",
                channel="feishu://group/" + ("ops_alerts", "engineering", "personal_ops")[d_idx % 3],
                sender=("marco_sre", "sam_qa")[d_idx % 2],
                text=dup_txt,
                target_task_id=dt.task_id,
            )
        )

    # 16 non-blocking user status peeks / gentle follow-ups
    peek_prompts = (
        "顺便问一下：目前手头正在跑的几个任务进展怎么样？简短报个状态就行，别停手里的活。",
        "帮我看一眼刚才发的代码和个人应用任务到哪一步了，简要回复即可。",
        "现在有哪些事情做完了，哪些还在处理？给我一个简短汇总就好。",
        "我回来看看进展，有什么已经交付、还在测试或者被卡住的吗？",
        "不用暂停当前工作，告诉我手头这些请求的大致状态就行。",
        "能简要说说已完成和待完成的事项吗？有阻塞也顺便提一下。",
        "刚才陆续发的那些事现在怎么样了？我只需要知道整体进度。",
        "目前还有多少事情在进行中？完成的也帮我一起列一下。",
        "Quick status check: which requests are finished vs still running right now? Keep active work going.",
        "Status ping — give me a 1-2 sentence progress summary without interrupting in-flight tasks.",
        "I'm back. What's done, what's underway, and is anything blocked?",
        "Could you briefly summarize where the requests I've sent stand?",
        "How are things looking across the coding work and the account errands?",
        "No need to pause your work; I'd just like an overview of progress so far.",
        "Which items can I consider finished, and which ones are still being checked?",
        "Before I step away, can you tell me what's left in progress and what's already delivered?",
    )
    for p_idx in range(16):
        mod_t = round(28.0 + p_idx * 34.5 + rng.uniform(-2.0, 2.0), 2)
        raw_events.append(
            StreamEventSpec(
                event_id=f"ev_peek_{p_idx + 1:02d}",
                moderate_offset_sec=mod_t,
                surge_offset_sec=mod_t / 2.5,
                kind="status_followup",
                channel="feishu://p2p/user",
                sender="viacol",
                text=peek_prompts[p_idx],
                target_task_id=None,
            )
        )

    # Sort deterministically by `(moderate_offset_sec, kind_priority, event_id)`; since `surge = moderate / 2.5`,
    # event ordering is 100% identical across Moderate and Surge schedules!
    kind_prio = {"work_request": 0, "work_amendment": 1, "status_followup": 2, "external_duplicate_webhook": 3, "external_noise": 4}
    raw_events.sort(key=lambda e: (e.moderate_offset_sec, kind_prio.get(e.kind, 9), e.event_id))

    for index, event in enumerate(raw_events, 1):
        event.event_id = f"ev_{index:03d}"

    monorepo_digest = hashlib.sha256(
        "|".join(f"{p.project_id}:{p.base_commit}" for p in projects).encode("utf-8")
    ).hexdigest()
    return MacroStreamSpec(
        stream_id=stream_id,
        repo_slug=repo_slug,
        cooper_repo_dir="multi_project_monorepo",
        cooper_task_dir="3_projects",
        base_commit=monorepo_digest,
        combined_patch_relpath=f"synaptic_stream_60/data/gold/{stream_id}.patch",
        accounts=accounts,
        tasks=tasks,
        events=raw_events,
        tau_sec=DEFAULT_TAU_SEC,
        projects=list(projects),
    )


def build_synaptic_stream_60(*, update_lock: bool = False) -> tuple[list[MacroStreamSpec], dict[str, Any]]:
    pa_cb_feats, pa_swe_ids, mi_tids, mi_gens = _load_existing_benchmark_ids()
    aw_writes_by_stream, aw_qas_by_stream = _select_zero_overlap_appworld_tasks(mi_tids, mi_gens)

    streams: list[MacroStreamSpec] = []
    for idx, cfg in enumerate(_REPO_STREAM_CONFIGS):
        st = _build_single_macro_stream(
            idx,
            cfg,
            aw_writes_by_stream[idx],
            aw_qas_by_stream[idx],
            pa_cb_feats,
            pa_swe_ids,
        )
        streams.append(st)

    # Audit zero-overlap invariants explicitly across CooperBench, SWE/Ambig-SWE, and AppWorld
    all_cb_feats: list[str] = []
    all_swe_ids: list[str] = []
    for s in streams:
        proj_lookup = {p.project_id: p for p in s.projects}
        for t in s.tasks:
            if t.substrate != "cooperbench":
                continue
            proj = proj_lookup[str(t.project_id)]
            if proj.source_kind == "cooperbench":
                all_cb_feats.append(f"{proj.cooper_repo_dir}/{proj.cooper_task_dir}/{t.cooper_feature_name}")
            else:
                all_swe_ids.append(str(proj.upstream_instance_id or t.cooper_feature_name))

    all_aw_tids = [
        str(t.upstream_task_id)
        for s in streams
        for t in s.tasks
        if t.substrate == "appworld"
    ]
    all_aw_gens = {tid.split("_")[0] for tid in all_aw_tids}

    assert len(set(all_cb_feats)) + len(set(all_swe_ids)) == 9, (
        f"Expected 9 unique coding tasks, got {len(set(all_cb_feats))} CB + {len(set(all_swe_ids))} SWE"
    )
    assert len(set(all_aw_tids)) == 51, f"Expected 51 unique AppWorld tasks, got {len(set(all_aw_tids))}"
    assert not (set(all_cb_feats) & pa_cb_feats), f"CooperBench overlap with PA-150: {set(all_cb_feats) & pa_cb_feats}"
    assert not (set(all_swe_ids) & pa_swe_ids), f"SWE instance overlap with PA-150: {set(all_swe_ids) & pa_swe_ids}"
    assert not (set(all_aw_tids) & mi_tids), "AppWorld task-ID overlap with Micro-30!"
    assert not (all_aw_gens & mi_gens), "AppWorld generator-ID overlap with Micro-30!"

    # Compose deterministic monorepo Gold patches (0% file overlap) and run fail-to-pass
    # + zero-leakage verification BEFORE publishing the new dataset or release checksum.
    from synaptic_stream_60.gold import verify_selected_code, write_selected_monorepo_gold

    for stream in streams:
        write_selected_monorepo_gold(stream)
    gold_selection = [verify_selected_code(stream) for stream in streams]
    if not all(report["gate_passed"] for report in gold_selection):
        raise RuntimeError("Selected-feature composability failed: " + json.dumps(gold_selection))

    DATA_OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(s.to_dict(), ensure_ascii=False) for s in streams]
    payload_txt = "\n".join(lines) + "\n"
    payload_sha = hashlib.sha256(payload_txt.encode("utf-8")).hexdigest()

    release_path = Path(__file__).with_name("release.json")
    release_data = json.loads(release_path.read_text(encoding="utf-8"))
    expected_sha = str(release_data.get("dataset_sha256") or "")
    if not update_lock and expected_sha and payload_sha != expected_sha:
        raise ValueError(
            f"Reconstructed Synaptic-Stream-60 digest mismatch: got {payload_sha}, expected {expected_sha}"
        )

    STREAMS_JSONL.write_text(payload_txt, encoding="utf-8")

    manifest = RELEASE.stamp(
        {
            "streams_count": len(streams),
            "unique_tasks_total": sum(len(s.tasks) for s in streams),
            "events_per_stream": [len(s.events) for s in streams],
            "regimes": ["moderate", "surge"],
            "surge_compression_ratio": 2.5,
            "tau_sec": DEFAULT_TAU_SEC,
            "conflict_free_coding_design": True,
            "selected_gold": [
                {
                    "stream_id": r["stream_id"],
                    "sha256": r["selected_patch_sha256"],
                    "selected_suites_passed": len(r["code_composability"]),
                    "base_suites_passed": 0,
                    "nonselected_suites_passed": 0,
                    "upstream_inputs": r["upstream_inputs"],
                }
                for r in gold_selection
            ],
            "zero_overlap_audit": {
                "pa150_cooperbench_feature_overlap": len(set(all_cb_feats) & pa_cb_feats),
                "pa150_swe_instance_overlap": len(set(all_swe_ids) & pa_swe_ids),
                "micro30_appworld_task_id_overlap": len(set(all_aw_tids) & mi_tids),
                "micro30_appworld_generator_id_overlap": len(all_aw_gens & mi_gens),
                "pa150_excluded_cooperbench_count": len(pa_cb_feats),
                "pa150_excluded_swe_count": len(pa_swe_ids),
                "note_on_repositories": (
                    "All 9 coding tasks (8 held-out CooperBench features + 1 held-out SWE-Interact "
                    "instance, 3 disjoint sub-projects per stream) and 51 AppWorld tasks+generators have 0% "
                    "overlap with PA-150 (including H-4 waves) and Micro-30, and 0% intra-stream file overlap "
                    "(zero Git merge conflicts by construction)."
                ),
            },
            "streams": [
                {
                    "stream_id": s.stream_id,
                    "repo_slug": s.repo_slug,
                    "base_commit": s.base_commit,
                    "projects": [p.project_id for p in s.projects],
                    "fingerprint": stream_fingerprint(s),
                    "tasks_count": len(s.tasks),
                    "events_count": len(s.events),
                    "categories": {
                        cat: sum(1 for t in s.tasks if t.category == cat)
                        for cat in ("code_independent", "app_write", "readonly_qa")
                    },
                }
                for s in streams
            ],
            "streams_jsonl_sha256": payload_sha,
        }
    )
    manifest["dataset_sha256"] = payload_sha
    if update_lock:
        release_data["dataset_sha256"] = payload_sha
        release_path.write_text(json.dumps(release_data, indent=2) + "\n", encoding="utf-8")
        MANIFEST_JSON.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return streams, manifest


def load_streams(stream_ids: list[str] | None = None, *, verify_checksum: bool = True) -> list[MacroStreamSpec]:
    if not STREAMS_JSONL.is_file():
        streams, _ = build_synaptic_stream_60()
    else:
        raw_bytes = STREAMS_JSONL.read_bytes()
        expected_sha = str(RELEASE.metadata.get("dataset_sha256") or "")
        if verify_checksum and expected_sha:
            actual_sha = hashlib.sha256(raw_bytes).hexdigest()
            if actual_sha != expected_sha:
                raise ValueError(
                    f"Synaptic-Stream-60 dataset checksum mismatch: got {actual_sha}, expected {expected_sha}"
                )
        streams = [
            MacroStreamSpec.from_dict(json.loads(ln))
            for ln in raw_bytes.decode("utf-8").splitlines()
            if ln.strip()
        ]
    if stream_ids:
        allow = {s.strip() for s in stream_ids if s.strip()}
        streams = [s for s in streams if s.stream_id in allow]
    return streams


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build and verify Synaptic-Stream-60 dataset.")
    parser.add_argument(
        "--update-lock",
        action="store_true",
        help="Maintainer option: overwrite release.json and manifest_60.json with newly computed digests.",
    )
    cli_args = parser.parse_args()
    _, mf = build_synaptic_stream_60(update_lock=cli_args.update_lock)
    print(json.dumps(mf, indent=2, ensure_ascii=False))
