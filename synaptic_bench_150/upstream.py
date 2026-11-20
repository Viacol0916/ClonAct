"""Version-locked upstream materialization. No model calls; all content stays local."""
from __future__ import annotations
import csv
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1] / "data" / "upstream_pillars"
LOCK_PATH = Path(__file__).with_name("upstream_lock.json")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def load_lock() -> dict[str, Any]:
    return json.loads(LOCK_PATH.read_text(encoding="utf-8"))


def check_repositories(lock: dict[str, Any], *, download: bool = False) -> None:
    for spec in lock["repositories"]:
        dest = ROOT / spec["name"]
        if not dest.exists() and download:
            subprocess.run(["git", "clone", spec["url"], str(dest)], check=True)
            subprocess.run(["git", "-C", str(dest), "checkout", "--detach", spec["commit"]], check=True)
        head = subprocess.check_output(["git", "-C", str(dest), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(dest), "status", "--porcelain", "--untracked-files=all"], text=True).strip()
        if head != spec["commit"] or dirty:
            raise ValueError(f"Upstream checkout {spec['name']} differs from locked commit; use a separate clean checkout. Existing files were preserved.")


def read_locked_parquet(spec: dict[str, Any]) -> list[dict[str, Any]]:
    import httpx
    import pyarrow.parquet as pq
    dest = ROOT / "downloads" / (spec["name"] + ".parquet")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        url = f"https://huggingface.co/datasets/{spec['repo']}/resolve/{spec['revision']}/{spec['path']}"
        tmp = dest.with_suffix(".partial")
        try:
            with httpx.stream("GET", url, follow_redirects=True, timeout=120) as response, tmp.open("wb") as fh:
                response.raise_for_status()
                for chunk in response.iter_bytes():
                    fh.write(chunk)
            if hashlib.sha256(tmp.read_bytes()).hexdigest() != spec["sha256"]:
                raise ValueError(f"Download checksum mismatch: {spec['name']}")
            os.replace(tmp, dest)
        finally:
            tmp.unlink(missing_ok=True)
    if hashlib.sha256(dest.read_bytes()).hexdigest() != spec["sha256"]:
        raise ValueError(f"Cached file checksum mismatch: {dest}; remove only this corrupt cache and retry.")
    rows = pq.read_table(dest).slice(0, spec["take"]).to_pylist()
    if len(rows) != spec["take"]:
        raise ValueError(f"Incomplete upstream rows: {spec['name']}")
    return rows


def validate_pools(lock: dict[str, Any] | None = None) -> None:
    lock = lock or load_lock()
    check_repositories(lock)
    for name, expected in lock["pools"].items():
        with (ROOT / "pools" / name).open(encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
        if canonical_hash(rows) != expected["canonical_sha256"] or len(rows) != expected["rows"]:
            raise ValueError(f"Upstream pool integrity mismatch: {name}")


def extract_gaia2_compact(row: dict, cfg: str) -> dict:
    raw_data = json.loads(row["data"]) if isinstance(row["data"], str) else row["data"]
    meta_def = (raw_data.get("metadata") or {}).get("definition") or {}
    apps = [a.get("name") or a.get("class_name") for a in raw_data.get("apps", [])]

    user_events = []
    env_events = []
    oracle_agent_events = []

    for ev in raw_data.get("events", []):
        et = ev.get("event_type")
        act = ev.get("action") or {}
        fn_name = f"{act.get('app', '')}.{act.get('function', '')}".strip(".")
        args_map = {}
        for arg in act.get("args") or []:
            val_str = str(arg.get("value", ""))
            args_map[str(arg.get("name", ""))] = val_str
        compact_ev = {
            "event_id": ev.get("event_id", ""),
            "event_time": ev.get("event_time", 0.0),
            "event_relative_time": ev.get("event_relative_time"),
            "app": act.get("app", ""),
            "function": act.get("function", ""),
            "call": fn_name,
            "args": args_map,
            "dependencies": ev.get("dependencies", []),
        }
        if et == "USER":
            user_events.append(compact_ev)
        elif et == "ENV":
            env_events.append(compact_ev)
        elif et == "AGENT":
            oracle_agent_events.append(compact_ev)

    return {
        "id": row["id"],
        "scenario_id": row["scenario_id"],
        "config": cfg,
        "split": row.get("split", "validation"),
        "start_time": meta_def.get("start_time"),
        "duration": meta_def.get("duration"),
        "tags": meta_def.get("tags", []),
        "apps": apps,
        "user_events": user_events,
        "env_events": env_events,
        "oracle_agent_events": oracle_agent_events,
    }


def hydrate() -> None:
    lock = load_lock()
    check_repositories(lock, download=True)
    source_rows = {s["name"]: read_locked_parquet(s) for s in lock["parquet"]}
    gaia = []
    raw_hashes = {}
    samples = {}
    raw_dir = ROOT / "gaia2_raw_scenarios"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for cfg in ("ambiguity", "adaptability", "execution", "time"):
        rows = source_rows["gaia2_" + cfg]
        samples[cfg] = rows
        for row in rows:
            gaia.append(extract_gaia2_compact(row, cfg))
            data = row["data"] if isinstance(row["data"], str) else json.dumps(row["data"], ensure_ascii=False)
            for suffix in (row["id"], row["scenario_id"]):
                name = f"{cfg}__{suffix}.json"
                (raw_dir / name).write_text(data, encoding="utf-8")
                raw_hashes[name] = hashlib.sha256(data.encode()).hexdigest()
    swe = []
    for name, split in (("sweinteract_dev", "dev_python"), ("sweinteract_live", "live_python")):
        for row in source_rows[name]:
            swe.append({**row, "_split": split})
    with (ROOT / "InteractiveSWEAgents" / "data" / "underspecified.csv").open(encoding="utf-8") as fh:
        underspecified = {row["instance_id"]: row["problem_statement"] for row in csv.DictReader(fh)}
    joined = []
    for row in source_rows["swebench"]:
        iid = row["instance_id"]
        if iid in underspecified:
            joined.append({
                "instance_id": iid, "repo": row["repo"], "base_commit": row["base_commit"],
                "environment_setup_commit": row.get("environment_setup_commit", ""), "version": row.get("version", ""),
                "underspecified_prompt": underspecified[iid], "full_problem_statement": row["problem_statement"],
                "hints_text": row.get("hints_text", ""), "patch": row["patch"], "test_patch": row["test_patch"],
                "FAIL_TO_PASS": row["FAIL_TO_PASS"], "PASS_TO_PASS": row["PASS_TO_PASS"],
            })
    pools = {"gaia2_compact_pool.jsonl": gaia, "cooperbench_hf_pool.jsonl": source_rows["cooperbench"],
             "sweinteract_python_pool.jsonl": swe, "ambig_swe_joined_pool.jsonl": joined}
    # Validate every pool BEFORE replacing any existing pool or final dataset.
    for name, rows in pools.items():
        expected = lock["pools"][name]
        if len(rows) != expected["rows"] or canonical_hash(rows) != expected["canonical_sha256"]:
            raise ValueError(f"Pinned source does not reproduce historical pool: {name}")
    (ROOT / "pools").mkdir(exist_ok=True)
    for name, rows in pools.items():
        dest = ROOT / "pools" / name
        tmp = dest.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        os.replace(tmp, dest)
    (ROOT / "samples").mkdir(exist_ok=True)
    (ROOT / "samples" / "gaia2_samples.json").write_text(json.dumps(samples, ensure_ascii=False), encoding="utf-8")
    (ROOT / "hydration_receipt.json").write_text(json.dumps({"raw_scenarios": raw_hashes}, indent=2) + "\n", encoding="utf-8")
    from .build_pa150 import write_dataset_artifacts
    summary = write_dataset_artifacts()
    print(f"Verified {summary['total_episodes']} Synaptic-Bench-150 episodes against the frozen SHA-256.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify existing repositories and pools without downloading or writing")
    args = parser.parse_args()
    if args.check:
        validate_pools()
        print("Verified pinned upstream repositories and pool fingerprints.")
    else:
        ROOT.mkdir(parents=True, exist_ok=True)
        hydrate()
