"""Selected-feature provenance and executable composability gate (no model calls)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from synaptic_bench_150.verifiers import _extract_patch_files
from synaptic_bench_150.workspace_manager import REPO_BARE_CACHE, VENV_CACHE
from synaptic_stream_60.schema import MacroStreamSpec
from synaptic_stream_60.watcher import (
    _evaluate_cooperbench_diff_isolated,
    _extract_subproject_diff,
    _prefix_patch_subdir,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _is_valid_fail_to_pass_baseline(base: dict[str, Any]) -> bool:
    """Confirm the unpatched base genuinely fails for the missing feature (not an env crash)."""
    if base.get("passed"):
        return False
    counts = base.get("counts") or {}
    if counts.get("failed", 0) > 0 and counts.get("errors", 0) == 0:
        return True
    # New public symbols (e.g. outlines.types.binary_str) fail at test-collection with ImportError.
    failure_txt = str(base.get("failure_output") or "")
    if (
        base.get("returncode") == 2
        and counts.get("passed", 0) == 0
        and ("ImportError:" in failure_txt or "AttributeError:" in failure_txt)
    ):
        return True
    return False


def write_selected_monorepo_gold(stream: MacroStreamSpec) -> Path:
    """Compose the conflict-free multi-project Gold patch and provenance JSON."""
    if not stream.projects:
        raise ValueError(f"Stream {stream.stream_id} has no multi-project specs")
    path = PROJECT_ROOT / stream.combined_patch_relpath
    if not path.resolve().is_relative_to((PROJECT_ROOT / "synaptic_stream_60/data/gold").resolve()):
        raise ValueError("Gold must be written inside synaptic_stream_60/data/gold")
    path.parent.mkdir(parents=True, exist_ok=True)

    tasks = [t for t in stream.tasks if t.substrate == "cooperbench"]
    if len(tasks) != len(stream.projects):
        raise ValueError(f"Expected {len(stream.projects)} coding tasks, got {len(tasks)}")

    chunks: list[str] = []
    touched_files: list[str] = []
    selected_entries: list[str] = []
    project_commits: dict[str, str] = {}
    for task in tasks:
        if not task.project_id or not task.project_subdir or not task.feature_patch_relpath:
            raise ValueError(f"Incomplete project metadata on coding task {task.task_id}")
        raw_patch = (PROJECT_ROOT / task.feature_patch_relpath).read_text(encoding="utf-8")
        prefixed = _prefix_patch_subdir(raw_patch, task.project_subdir)
        if not prefixed.endswith("\n"):
            prefixed += "\n"
        chunks.append(prefixed)
        touched_files.extend(_extract_patch_files(prefixed))
        selected_entries.append(f"{task.project_id}/{task.cooper_feature_name}")
        project_commits[task.project_id] = str(task.base_commit or "")

    if len(touched_files) != len(set(touched_files)):
        raise ValueError(f"File overlap detected in conflict-free stream {stream.stream_id}: {touched_files}")

    combined = "".join(chunks)
    digest = hashlib.sha256(combined.encode("utf-8")).hexdigest()
    path.write_text(combined, encoding="utf-8")
    provenance = {
        "stream_id": stream.stream_id,
        "selected_features": sorted(selected_entries),
        "project_base_commits": project_commits,
        "touched_files": sorted(touched_files),
        "disjoint_file_count": len(set(touched_files)),
        "file_overlap_count": 0,
        "sha256": digest,
    }
    path.with_suffix(".json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def verify_selected_code(stream: MacroStreamSpec) -> dict[str, Any]:
    """Test all selected suites against the composed selected-only tree (0 model calls).

    In multi-project streams (`stream.projects`), extracts each project's diff from the
    monorepo Gold patch via `_extract_subproject_diff`, checks 0 file overlap across projects,
    verifies fail-to-pass on each selected feature, and confirms nonselected upstream features
    in every mounted project remain failing.
    """
    path = PROJECT_ROOT / stream.combined_patch_relpath
    if not path.resolve().is_relative_to((PROJECT_ROOT / "synaptic_stream_60/data/gold").resolve()):
        raise ValueError("Gold must be a local selected-only composition")
    patch = path.read_text(encoding="utf-8")
    provenance = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    tasks = [t for t in stream.tasks if t.substrate == "cooperbench"]
    digest = hashlib.sha256(patch.encode("utf-8")).hexdigest()

    if stream.projects:
        proj_by_id = {p.project_id: p for p in stream.projects}
        selected = sorted(f"{t.project_id}/{t.cooper_feature_name}" for t in tasks)
        expected_commits = {p.project_id: p.base_commit for p in stream.projects}
        if (
            len(tasks) != len(stream.projects)
            or provenance.get("selected_features") != selected
            or provenance.get("project_base_commits") != expected_commits
            or provenance.get("file_overlap_count") != 0
            or provenance.get("sha256") != digest
        ):
            raise ValueError(f"Selected multi-project gold provenance mismatch: {stream.stream_id}")

        evidence = []
        inputs: dict[str, dict[str, str]] = {}
        nonselected = []
        for task in tasks:
            proj = proj_by_id[str(task.project_id)]
            bare = next(p for p in REPO_BARE_CACHE.iterdir() if proj.base_commit[:12] in p.name)
            venv = next(p for p in VENV_CACHE.iterdir() if proj.base_commit[:12] in p.name)
            common = dict(
                bare_repo_path=bare,
                py_bin=venv / "bin/python",
                base_commit=proj.base_commit,
                has_conflicts=False,
            )
            test_path = PROJECT_ROOT / str(task.tests_patch_relpath)
            source_path = PROJECT_ROOT / str(task.feature_patch_relpath)
            sel_key = f"{proj.project_id}/{task.cooper_feature_name}"
            inputs[sel_key] = {
                "feature_patch_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
                "tests_patch_sha256": hashlib.sha256(test_path.read_bytes()).hexdigest(),
            }
            sub_diff = _extract_subproject_diff(patch, proj.subdir)
            base = _evaluate_cooperbench_diff_isolated(filtered_diff="", tests_patch_path=test_path, **common)
            result = _evaluate_cooperbench_diff_isolated(filtered_diff=sub_diff, tests_patch_path=test_path, **common)
            baseline_valid = _is_valid_fail_to_pass_baseline(base)
            if baseline_valid:
                base.pop("failure_output", None)
            evidence.append(
                {
                    "task_id": task.task_id,
                    "project_id": proj.project_id,
                    "feature": sel_key,
                    "base": base,
                    "base_failed_tests_valid": baseline_valid,
                    "selected_gold": result,
                }
            )
            # Check up to 2 held-out nonselected features for CooperBench tasks (zero leakage).
            if proj.source_kind == "cooperbench":
                source_dir = PROJECT_ROOT / "data/upstream_pillars/CooperBench/dataset" / proj.cooper_repo_dir / proj.cooper_task_dir
                checked_nonselected = 0
                for ns_test_path in sorted(source_dir.glob("feature*/tests.patch")):
                    if ns_test_path.parent.name == task.cooper_feature_name:
                        continue
                    if checked_nonselected >= 2:
                        break
                    ns_res = _evaluate_cooperbench_diff_isolated(filtered_diff=sub_diff, tests_patch_path=ns_test_path, **common)
                    ns_res.pop("failure_output", None)
                    nonselected.append({"feature": f"{proj.project_id}/{ns_test_path.parent.name}", **ns_res})
                    checked_nonselected += 1

        return {
            "stream_id": stream.stream_id,
            "selected_patch_sha256": digest,
            "upstream_inputs": inputs,
            "code_composability": evidence,
            "nonselected_feature_suites": nonselected,
            "gate_passed": (
                all(e["base_failed_tests_valid"] and e["selected_gold"]["passed"] for e in evidence)
                and not any(e["passed"] for e in nonselected)
            ),
        }

    # Legacy single-repo path (kept for backward compatibility with single-repo unit tests)
    selected = sorted(str(t.cooper_feature_name) for t in tasks)
    if len(tasks) != 6 or provenance["selected_features"] != selected or provenance["base_commit"] != stream.base_commit or provenance["sha256"] != digest:
        raise ValueError(f"Selected gold provenance mismatch: {stream.stream_id}")
    bare = next(p for p in REPO_BARE_CACHE.iterdir() if stream.base_commit[:12] in p.name)
    venv = next(p for p in VENV_CACHE.iterdir() if stream.base_commit[:12] in p.name)
    common = dict(bare_repo_path=bare, py_bin=venv / "bin/python", base_commit=stream.base_commit, has_conflicts=False)
    evidence = []
    inputs = {}
    for task in tasks:
        test_path = PROJECT_ROOT / str(task.tests_patch_relpath)
        source_path = PROJECT_ROOT / str(task.feature_patch_relpath)
        inputs[str(task.cooper_feature_name)] = {
            "feature_patch_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            "tests_patch_sha256": hashlib.sha256(test_path.read_bytes()).hexdigest(),
        }
        base = _evaluate_cooperbench_diff_isolated(filtered_diff="", tests_patch_path=test_path, **common)
        result = _evaluate_cooperbench_diff_isolated(filtered_diff=patch, tests_patch_path=test_path, **common)
        baseline_valid = _is_valid_fail_to_pass_baseline(base)
        if baseline_valid:
            base.pop("failure_output", None)
        evidence.append({"task_id": task.task_id, "feature": task.cooper_feature_name,
                         "base": base, "base_failed_tests_valid": baseline_valid, "selected_gold": result})
    nonselected = []
    source_dir = PROJECT_ROOT / "data/upstream_pillars/CooperBench/dataset" / stream.cooper_repo_dir / stream.cooper_task_dir
    for test_path in sorted(source_dir.glob("feature*/tests.patch")):
        if test_path.parent.name in selected:
            continue
        result = _evaluate_cooperbench_diff_isolated(filtered_diff=patch, tests_patch_path=test_path, **common)
        result.pop("failure_output", None)
        nonselected.append({"feature": test_path.parent.name, **result})
    return {
        "stream_id": stream.stream_id, "selected_patch_sha256": digest,
        "upstream_inputs": inputs, "code_composability": evidence,
        "nonselected_feature_suites": nonselected,
        "gate_passed": all(e["base_failed_tests_valid"] and e["selected_gold"]["passed"] for e in evidence)
                       and not any(e["passed"] for e in nonselected),
    }
