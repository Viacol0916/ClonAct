"""Unit and provenance tests for `Synaptic-Stream-60`."""
from __future__ import annotations

import hashlib
import json
import pytest
from pathlib import Path

from clonact.paradigm_baselines import PARADIGM_BASELINE_ARMS, build_paradigm_prompts
from synaptic_stream_60.build_streams import MANIFEST_JSON, STREAMS_JSONL, load_streams
from synaptic_stream_60.schema import (
    compute_stable_task_delivery,
    stratified_paired_sign_flip_test,
    summarize_task_deliveries,
)


def test_stream60_zero_overlap_and_sha256() -> None:
    assert STREAMS_JSONL.is_file()
    assert MANIFEST_JSON.is_file()
    mf = json.loads(MANIFEST_JSON.read_text(encoding="utf-8"))
    raw = STREAMS_JSONL.read_text(encoding="utf-8")
    from synaptic_stream_60 import RELEASE
    digest = hashlib.sha256(STREAMS_JSONL.read_bytes()).hexdigest()
    release = json.loads(STREAMS_JSONL.parents[1].joinpath('release.json').read_text())
    assert digest == mf['streams_jsonl_sha256'] == mf['dataset_sha256'] == release['dataset_sha256']
    assert RELEASE.version == release['version'] == mf['benchmark_version'] == '1.0.0'
    assert mf['evaluator_version'] == '1.0.0'
    assert mf['source_versions'] == release['source_versions'] == {'dataset': '3.0.0', 'evaluator': '3.0.0'}
    assert mf["unique_tasks_total"] == 60
    assert mf["zero_overlap_audit"]["pa150_cooperbench_feature_overlap"] == 0
    assert mf["zero_overlap_audit"]["pa150_swe_instance_overlap"] == 0
    assert mf["zero_overlap_audit"]["micro30_appworld_task_id_overlap"] == 0
    assert mf["zero_overlap_audit"]["micro30_appworld_generator_id_overlap"] == 0

    # Independently scan raw PA-150 JSONL (including H-4 waves & SWE instances) without trusting manifest alone.
    import re
    from synaptic_stream_60.build_streams import PROJECT_ROOT
    pa_raw = (PROJECT_ROOT / "synaptic_bench_150/data/episodes_pa150.jsonl").read_text(encoding="utf-8")
    pa_cb = {f"{m.group(1)}/{m.group(2)}/{m.group(3)}" for m in re.finditer(r"([a-zA-Z0-9_]+_task)[/:]+(task\d+)[/:]+(feature\d+)", pa_raw)}
    pa_swe = {m.group(1) for m in re.finditer(r"([A-Za-z0-9_.-]+__[A-Za-z0-9_.-]+-\d+)", pa_raw)}
    assert len(pa_cb) >= 99 and len(pa_swe) >= 42

    streams = load_streams()
    assert len(streams) == 3
    for s in streams:
        proj_map = {p.project_id: p for p in s.projects}
        assert len(s.projects) == 3 and len(proj_map) == 3
        for t in s.tasks:
            if t.substrate == "cooperbench":
                p = proj_map[str(t.project_id)]
                if p.source_kind == "cooperbench":
                    assert f"{p.cooper_repo_dir}/{p.cooper_task_dir}/{t.cooper_feature_name}" not in pa_cb
                else:
                    assert p.upstream_instance_id not in pa_swe
        assert len(s.tasks) == 20
        assert len(s.events) == 240
        # Exact 2.5x compression invariance between Moderate and Surge
        mod_ids = [e.event_id for e in sorted(s.events, key=lambda x: (x.moderate_offset_sec, x.event_id))]
        sur_ids = [e.event_id for e in sorted(s.events, key=lambda x: (x.surge_offset_sec, x.event_id))]
        assert mod_ids == sur_ids


def test_stable_completion_prevents_transient_pass_backfill() -> None:
    # Task arrives at t=10s, passes at t=40s, gets broken by conflicting edit at t=90s,
    # and is repaired at t=150s through t=610s. Stable completion MUST be 150s (wait=140s), NOT 40s!
    obs = [
        (10.0, False),
        (40.0, True),
        (70.0, True),
        (90.0, False),
        (120.0, False),
        (150.0, True),
        (610.0, True),
    ]
    res = compute_stable_task_delivery(obs, arrival_sec=10.0, tau_sec=600.0)
    assert res["verified_within_tau"] is True
    assert res["first_pass_sec"] == 40.0
    assert res["stable_completion_sec"] == 150.0
    assert res["waiting_sec"] == 140.0
    assert res["regression_count"] == 1
    assert res["regressed_at_end"] is False


def test_unrecovered_regression_censored_at_tau() -> None:
    obs = [(0.0, False), (50.0, True), (180.0, False), (600.0, False)]
    res = compute_stable_task_delivery(obs, arrival_sec=0.0, tau_sec=600.0)
    assert res["verified_within_tau"] is False
    assert res["waiting_sec"] is None
    assert res["capped_waiting_sec"] == 600.0
    assert res["regressed_at_end"] is True

    # Check quantile censoring when <95% verify
    rows = [
        {"category": "app_write", "verified_within_tau": True, "waiting_sec": float(10 * (i + 1)), "regression_count": 0, "regressed_at_end": False}
        for i in range(18)
    ] + [
        {"category": "code_coupled", "verified_within_tau": False, "waiting_sec": None, "regression_count": 1, "regressed_at_end": True},
        {"category": "code_coupled", "verified_within_tau": False, "waiting_sec": None, "regression_count": 0, "regressed_at_end": False},
    ]
    summary = summarize_task_deliveries(rows, tau_sec=600.0)
    assert summary["verified_rate_v_tau"] == 0.90
    assert summary["all_req_p50_sec"] == 100.0
    assert summary["all_req_p90_sec"] == 180.0
    assert summary["all_req_p95_sec"] is None  # 18/20 < 95%, must NOT use conditional subset!


def test_neutral_subagent_paradigm_registered() -> None:
    assert "dsh_subagent_neutral" in PARADIGM_BASELINE_ARMS
    trunk_p, mode_p = build_paradigm_prompts("dsh_subagent_neutral", agents_md="# Guide", has_code_repo=True)
    assert "# Guide" in trunk_p
    assert "dsh_subagent_neutral" in mode_p
    assert "Prefer handing each distinct piece" not in mode_p


def test_stratified_paired_permutation() -> None:
    a = {"S1": [20.0] * 20, "S2": [25.0] * 20, "S3": [30.0] * 20}
    b = {"S1": [120.0] * 20, "S2": [140.0] * 20, "S3": [150.0] * 20}
    st = stratified_paired_sign_flip_test(a, b, n_resamples=2000)
    assert st["streams_a_lt_b"] == 3
    assert st["mean_diff_a_minus_b"] < -100.0
    assert st["p_two_sided_task_paired"] < 0.002
    assert st["n_streams"] == 3
    assert st["p_two_sided_stream_clustered"] == 0.25
    assert st["within_stream_block_bootstrap_ci95"][1] < 0
    assert st["stream_cluster_bootstrap_ci95"][1] < 0


def test_multi_account_interleaving_and_eval_isolation(tmp_path: Path) -> None:
    from dataclasses import asdict
    from synaptic_stream_60.aw_multi_server import (
        PROJECT_ROOT,
        StreamAppWorldEvalWorkerProcess,
        StreamAppWorldServerProcess,
    )

    st = load_streams(["STREAM_01"])[0]
    t03 = next(x for x in st.tasks if x.account_id == "acct_01")
    t04 = next(x for x in st.tasks if x.account_id == "acct_02")
    t15 = next(x for x in st.tasks if x.account_id == "acct_06")
    qa1 = next(x for x in st.tasks if x.account_id == "acct_12")
    qa2 = next(x for x in st.tasks if x.account_id == "acct_13")

    import time
    actual_start_unix = time.time()
    actual_start_monotonic = time.perf_counter()
    snaps_root = tmp_path / "aw_snapshots"
    srv = StreamAppWorldServerProcess(log_path=tmp_path / "srv.log")
    evw = StreamAppWorldEvalWorkerProcess(log_path=tmp_path / "evw.log")
    try:
        srv.start()
        evw.start()
        srv.start_stream(
            st.stream_id,
            [asdict(a) for a in st.accounts if a.account_id in ("acct_01", "acct_02", "acct_06", "acct_12", "acct_13")],
            snaps_root,
        )
        for task in (t03, t04, t15):
            sol = (
                PROJECT_ROOT / "data" / "tasks" / str(task.upstream_task_id) / "ground_truth" / "compiled_solution.py"
            ).read_text(encoding="utf-8") + "\nsolution(apis, requester)\n"
            res = srv.execute_code(str(task.account_id), sol)
            assert res["ok"] is True

        # Read accounts again after a third account was mutated; export current
        # state again to exercise real format="changes" save/load interleaving.
        for task in (t03, t04, t15, t03, t04):
            readback = srv.execute_code(str(task.account_id), "print(apis.supervisor.show_profile())")
            assert readback["ok"] is True
            assert "Traceback" not in readback["output"]
            assert srv.submit_answer(str(task.account_id), "done")["ok"] is True

        # Verify lightweight `format='changes'` snapshots stay under 2 MB total (preventing ENOSPC!)
        total_snap_bytes = sum(f.stat().st_size for f in snaps_root.rglob("*") if f.is_file())
        assert total_snap_bytes < 2 * 1024 * 1024, f"Snapshot bloat detected: {total_snap_bytes} bytes"

        for task in (t03, t04, t15):
            acct_dir = snaps_root / str(task.account_id)
            latest_snap = sorted(p for p in acct_dir.iterdir() if p.is_dir() and p.name.startswith("v"))[-1]
            meta = json.loads((latest_snap / "snapshot_meta.json").read_text())
            assert actual_start_unix <= meta["created_at_unix"] <= time.time()
            assert actual_start_monotonic <= meta["created_at_monotonic"] <= time.perf_counter()
            assert list(latest_snap.glob("*.jsonl"))
            ev_res = evw.evaluate(
                upstream_task_id=str(task.upstream_task_id),
                snapshot_dir=str(latest_snap),
                is_qa=False,
                target_apps=list(task.target_apps),
            )
            assert ev_res["passed"] is True, f"{task.task_id} failed after interleaving: {ev_res}"
            assert ev_res["bystander_collateral"] == []

        # Interleave QA evaluations (`sdir=None`) across distinct supervisors without SQLModel global binding leak
        for _ in range(2):
            for qtask in (qa1, qa2):
                q_res = evw.evaluate(
                    upstream_task_id=str(qtask.upstream_task_id),
                    snapshot_dir=None,
                    is_qa=True,
                    target_apps=list(qtask.target_apps),
                    candidate_replies=[str(qtask.ground_truth_answer)],
                )
                assert q_res["passed"] is True, f"{qtask.task_id} QA eval failed: {q_res}"
    finally:
        srv.stop()
        evw.stop()


def test_v2_selection_authentication_and_incremental_specs() -> None:
    import re
    from collections import Counter
    from synaptic_stream_60.build_streams import _load_existing_benchmark_ids, PROJECT_ROOT
    from synaptic_stream_60.runner import _collect_stream_dsh_allowlists, _dsh_feishu_open_id
    pa_cb, pa_swe, tids, generators = _load_existing_benchmark_ids()
    all_ids, all_features = set(), set()
    qa_difficulties = Counter()
    for stream in load_streams():
        assert len(stream.projects) == 3
        assert len({p.project_id for p in stream.projects}) == 3
        assert len({p.repo_slug for p in stream.projects}) == 3
        proj_by_id = {p.project_id: p for p in stream.projects}
        assert Counter(t.category for t in stream.tasks) == {"code_independent": 3, "app_write": 11, "readonly_qa": 6}
        assert Counter(e.kind for e in stream.events) == {
            "work_request": 20, "work_amendment": 2, "external_noise": 178,
            "external_duplicate_webhook": 24, "status_followup": 16}
        assert [e.event_id for e in stream.events] == [f"ev_{i:03d}" for i in range(1, 241)]
        _, humans, _ = _collect_stream_dsh_allowlists(stream)
        human_ids = {x["id"] for x in humans}
        amend_time = {e.target_task_id: e.moderate_offset_sec for e in stream.events if e.kind == "work_amendment"}
        for event in stream.events:
            assert re.fullmatch(r"ev_\d{3}", event.event_id)
            assert "noise" not in event.event_id and "dup" not in event.event_id
            assert abs(event.surge_offset_sec * 2.5 - event.moderate_offset_sec) < 1e-9
            assert not event.channel.startswith("github:")
            if event.kind == "external_noise":
                assert "@agent" not in event.text
            else:
                assert "bot" not in event.sender and "webhook" not in event.sender
                if "/group/" in event.channel:
                    assert "@agent" in event.text
                    assert _dsh_feishu_open_id(event.sender) in human_ids
            if event.kind == "external_duplicate_webhook":
                assert "Duplicate Notification" not in event.text and "Ignore duplicate spawn" not in event.text
                assert event.target_task_id not in event.text
                if event.target_task_id in amend_time:
                    assert event.moderate_offset_sec < amend_time[event.target_task_id], (stream.stream_id, event.event_id)
            if event.kind == "work_amendment":
                task = next(t for t in stream.tasks if t.task_id == event.target_task_id)
                assert abs(event.moderate_offset_sec - task.moderate_offset_sec - 18) < 1e-9
                assert abs(event.surge_offset_sec - task.surge_offset_sec - 7.2) < 1e-9
                assert task.amendment_text in event.text
                assert task.amendment_text not in task.prompt_text
        for category in ("app_write", "readonly_qa"):
            selected = [t for t in stream.tasks if t.category == category]
            assert len({t.generator_id for t in selected}) == len(selected)
        for task in stream.tasks:
            if task.substrate == "cooperbench":
                proj = proj_by_id[str(task.project_id)]
                if proj.source_kind == "cooperbench":
                    key = f"{proj.cooper_repo_dir}/{proj.cooper_task_dir}/{task.cooper_feature_name}"
                    assert key not in pa_cb and key not in all_features
                else:
                    key = str(proj.upstream_instance_id)
                    assert key not in pa_swe and key not in all_features
                all_features.add(key)
                assert task.coupled_peer_task_id is None
                assert task.project_subdir == proj.subdir
            else:
                assert task.upstream_task_id not in tids | all_ids
                assert task.generator_id not in generators
                all_ids.add(task.upstream_task_id)
                meta = json.loads((PROJECT_ROOT / "data/tasks" / task.upstream_task_id / "ground_truth/metadata.json").read_text())
                assert task.difficulty == meta["difficulty"]
                if task.category == "app_write":
                    assert task.difficulty >= 2
                else:
                    qa_difficulties[task.difficulty] += 1
    assert len(all_features) == 9 and len(all_ids) == 51
    assert qa_difficulties == {2: 12, 1: 6}


def test_model_event_payload_has_no_ground_truth_labels() -> None:
    import ast
    from synaptic_stream_60 import runner
    tree = ast.parse(Path(runner.__file__).read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == "ingest_external_event"]
    assert len(calls) == 1
    assert {key.value for key in calls[0].args[0].keys} == {"event_id", "channel", "sender", "text"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "deliver_inbound_event":
            assert next(k.value.value for k in node.keywords if k.arg == "target_task_tag") is None


def test_superagent_mentions_depend_only_on_visible_text(monkeypatch, tmp_path) -> None:
    from types import SimpleNamespace
    from synaptic_stream_60 import runner
    from synaptic_stream_60.schema import StreamEventSpec
    seen = []
    monkeypatch.setattr(runner, "push_feishu_message", lambda *a, **kw: seen.append(kw) or "msg")
    agent = SimpleNamespace(world_dir=tmp_path)
    event = StreamEventSpec("ev_001", 0, 0, "work_request", "feishu://group/team", "alice", "Please help")
    runner._inject_stream_event_into_superagent(agent, event)
    assert seen[-1]["mention_bot"] is False
    event.kind = "external_noise"
    event.text = "@agent Please help"
    runner._inject_stream_event_into_superagent(agent, event)
    assert seen[-1]["mention_bot"] is True
    assert seen[-1]["sender_type"] == "user"


def test_legacy_worker_limit_and_stream_override() -> None:
    import ast
    import inspect
    from dsh_runner.dsh_controller import GenuineDSHSuperagent
    from synaptic_stream_60 import runner
    assert inspect.signature(GenuineDSHSuperagent.bootstrap_and_start).parameters["max_workers"].default == 4
    source = inspect.getsource(GenuineDSHSuperagent.bootstrap_and_start)
    assert source.count('"maxWorkers": int(max_workers)') == 2
    tree = ast.parse(Path(runner.__file__).read_text())
    limits = [kw.value.value for call in ast.walk(tree) if isinstance(call, ast.Call)
              for kw in call.keywords if kw.arg == "max_workers"]
    assert limits == [16]


def test_promoter_guards_stage_open_turn_and_failed_merge(monkeypatch, tmp_path) -> None:
    from types import SimpleNamespace
    from synaptic_stream_60 import runner
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    ticket = {"id": "a", "state": "open", "stage": "working", "directory": str(repo), "sessionId": "s1"}
    agent = SimpleNamespace(dsh_home=tmp_path, sandbox_paths={"clone_repo": repo}, get_all_tickets_detailed=lambda: [ticket])
    calls = []
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **k: calls.append(a) or SimpleNamespace(stdout="head" if "rev-parse" in a[0] else ""))
    merged = [False]
    monkeypatch.setattr(runner, "promote_worktree_to_main", lambda *a, **kw: ("head", merged[0]))
    cache = {}
    attempts = set()
    for stage, action in [("working", None), ("person", "answer"), ("person", "failed"), (None, None)]:
        ticket.update(stage=stage, action=action)
        assert runner._promote_settled_superagent_tickets_once(agent, "base", tmp_path, cache, attempts) == 0
        assert calls == []
    ticket.update(stage="idle")
    projection = tmp_path / "storages/session_projcache/sessions/s1.json"
    projection.parent.mkdir(parents=True)
    projection.write_text(json.dumps({"record": {"rows": {"turnBoundary": {"val": {"openTurnStartSeq": 0}}}}}))
    assert runner._promote_settled_superagent_tickets_once(agent, "base", tmp_path, cache, attempts) == 0
    assert calls == []
    projection.write_text('{}')
    assert runner._promote_settled_superagent_tickets_once(agent, "base", tmp_path, cache, attempts) == 0
    assert cache == {}
    assert runner._promote_settled_superagent_tickets_once(agent, "base", tmp_path, cache, attempts) == 0
    attempts.clear()  # A new head allows another attempt; real Git coverage below.
    merged[0] = True
    ticket.update(stage="person", action="review")
    assert runner._promote_settled_superagent_tickets_once(agent, "base", tmp_path, cache, attempts) == 1
    assert cache == {"a": "head"}


def test_peek_correlation_and_substantive_reply() -> None:
    from synaptic_stream_60.runner import is_substantive_peek_reply
    request = "How are things going? [check-01]"
    assert not is_substantive_peek_reply("On it: created a ticket for your feature.", request)
    assert not is_substantive_peek_reply("Finished another unrelated task, tests passed.", request)
    assert not is_substantive_peek_reply("[check-01] On it: working on your request now.", request)
    assert is_substantive_peek_reply("[check-01] Two requests finished; one is running tests.", request)
    assert is_substantive_peek_reply("Two requests finished; one is running tests.", request, parent_id="m1", message_id="m1")
    assert not is_substantive_peek_reply("Two requests finished; one is running tests.", request, parent_id="m2", message_id="m1")


def test_bot_replies_preserve_parent_link(tmp_path) -> None:
    from dsh_runner.sandbox_clis import connect_db, list_bot_replies, init_db_schema
    init_db_schema(tmp_path)
    with connect_db(tmp_path) as conn:
        conn.execute("INSERT INTO bot_replies (reply_to_message_id, chat_id, out_message_id, markdown, idempotency_key, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                     ("m1", "chat", "out1", "Status", "key", "2026-11-18T00:00:00Z"))
    reply = list_bot_replies(tmp_path)[0]
    assert reply["reply_to_message_id"] == reply["parent_id"] == "m1"
    assert reply["idempotency_key"] == "key"


def test_snapshot_creation_time_deadline_and_qa_parity(tmp_path) -> None:
    import asyncio
    from dataclasses import replace
    from types import SimpleNamespace
    from synaptic_stream_60.watcher import StreamStateWatcher
    stream = load_streams()[0]
    task = replace(next(t for t in stream.tasks if t.category == "readonly_qa"), moderate_offset_sec=0)
    stream = replace(stream, tasks=[task], tau_sec=10)
    snapshots = tmp_path / task.account_id
    snapshots.mkdir()
    def add(version, created, passed):
        path = snapshots / f"v{version:04d}"
        path.mkdir()
        (path / "snapshot_meta.json").write_text(json.dumps({"created_at_unix": 100 + created, "passed": passed}))
    seen = []
    def evaluate(**kw):
        seen.append(kw)
        return {"passed": json.loads((Path(kw["snapshot_dir"]) / "snapshot_meta.json").read_text())["passed"]}
    watcher = StreamStateWatcher(stream, regime="moderate", canonical_repo_dir=tmp_path / "absent", bare_repo_path=tmp_path,
                                 repo_py_bin=tmp_path, snapshots_root=tmp_path, aw_eval_worker=SimpleNamespace(evaluate=evaluate),
                                 get_replies_fn=lambda: [(1.0, str(task.ground_truth_answer))])
    watcher.stream_start_unix = 100
    asyncio.run(watcher.sample_once(2))
    assert seen == []  # Correct internal/reply text is not an externally submitted answer.
    add(1, 4, True)
    add(2, 8, False)
    add(3, 9, True)
    add(4, 10.001, False)
    # Snapshots appeared during a long evaluation after poll start t=2.
    asyncio.run(watcher.sample_once(2))
    assert watcher.observations[task.task_id] == [(0, False), (4, True), (8, False), (9, True)]
    rows = asyncio.run(watcher.stop_and_finalize(12))
    assert rows[0]["stable_completion_sec"] == 9
    assert rows[0]["regression_count"] == 1
    assert all(kw["candidate_replies"] == [] for kw in seen)


def test_post_deadline_repo_capture_is_not_backdated(monkeypatch, tmp_path) -> None:
    import asyncio
    from dataclasses import replace
    from types import SimpleNamespace
    from synaptic_stream_60 import watcher as module
    stream = load_streams()[0]
    task = replace(next(t for t in stream.tasks if t.substrate == "cooperbench"), moderate_offset_sec=0)
    stream = replace(stream, tasks=[task], tau_sec=10)
    watcher = module.StreamStateWatcher(stream, regime="moderate", canonical_repo_dir=tmp_path, bare_repo_path=tmp_path,
                                      repo_py_bin=tmp_path, snapshots_root=tmp_path,
                                      aw_eval_worker=SimpleNamespace(), get_replies_fn=lambda: [])
    monkeypatch.setattr(module, "_fast_repo_tree_fingerprint", lambda p: "fingerprint")
    monkeypatch.setattr(module, "_extract_agent_diff", lambda *a: "diff")
    monkeypatch.setattr(module, "_filter_test_files", lambda d: d)
    monkeypatch.setattr(module, "_diff_or_repo_has_conflict_markers", lambda *a: False)
    monkeypatch.setattr(watcher, "_elapsed_now", lambda fallback: 10.001)
    asyncio.run(watcher.sample_once(9.9))
    assert watcher.observations[task.task_id] == []
    assert watcher.code_evals_executed == 0


def test_hygiene_counts_observed_superagent_dispositions() -> None:
    from synaptic_stream_60.runner import superagent_hygiene, dsh_trigger_id_for_injected
    from synaptic_stream_60.schema import StreamEventSpec
    events = [StreamEventSpec(f"ev_{i:03d}", i, i / 2.5, kind, "feishu://group/team", "alice", "hi")
              for i, kind in enumerate(["external_noise", "external_noise", "external_duplicate_webhook", "external_duplicate_webhook", "work_request"], 1)]
    trace = [{"event_id": e.event_id, "injected": {"channel": "feishu", "message_id": e.event_id}} for e in events]
    dispositions = [{"filtered": True}, {"decision": {"outcome": "ignore"}},
                    {"fanout": ["existing-ticket"]}, {"decision": {"outcome": "accept"}},
                    {"fanout": ["existing-ticket"]}]
    triggers = [{"id": dsh_trigger_id_for_injected(r["injected"]), "disposition": d} for r, d in zip(trace, dispositions)]
    result = superagent_hygiene(events, trace, triggers)
    assert result["noise_dropped"] == 2
    assert result["duplicates_merged"] == 1
    assert len(result["decisions"]) == 5
    assert superagent_hygiene(events, trace, [])["duplicates_merged"] == 0


def test_selected_gold_provenance_and_excluded_features() -> None:
    from synaptic_stream_60.build_streams import PROJECT_ROOT
    manifest = json.loads(MANIFEST_JSON.read_text())
    assert manifest["benchmark_version"] == "1.0.0"
    assert manifest["conflict_free_coding_design"] is True
    assert manifest["dataset_sha256"] == manifest["streams_jsonl_sha256"]
    gates = {r["stream_id"]: r for r in manifest["selected_gold"]}
    for stream in load_streams():
        patch_path = PROJECT_ROOT / stream.combined_patch_relpath
        meta = json.loads(patch_path.with_suffix('.json').read_text())
        assert len(meta["selected_features"]) == 3
        assert meta["file_overlap_count"] == 0
        assert set(meta["selected_features"]) == {
            f"{t.project_id}/{t.cooper_feature_name}" for t in stream.tasks if t.substrate == "cooperbench"
        }
        digest = hashlib.sha256(patch_path.read_bytes()).hexdigest()
        assert meta["sha256"] == gates[stream.stream_id]["sha256"] == digest
        assert gates[stream.stream_id]["selected_suites_passed"] == 3
        assert gates[stream.stream_id]["base_suites_passed"] == 0
        assert gates[stream.stream_id]["nonselected_suites_passed"] == 0
        assert len(gates[stream.stream_id]["upstream_inputs"]) == 3


def test_strict_deadline_does_not_round_late_success_into_window() -> None:
    result = compute_stable_task_delivery([(0, False), (10.0000001, True)], arrival_sec=0, tau_sec=10)
    assert result["verified_within_tau"] is False


def test_cluster_inference_rejects_unpaired_arrays() -> None:
    import pytest
    with pytest.raises(ValueError, match="Unpaired"):
        stratified_paired_sign_flip_test({"A": [1, 2]}, {"A": [1]}, n_resamples=10)


def test_natural_messages_and_unique_personas_all_streams() -> None:
    import re
    from synaptic_stream_60.build_streams import PROJECT_ROOT, _feature_title
    forbidden = ("STREAM_", "acct_", "./aw", "aw --account", "[Repository Feature Request",
                 "[Personal App Errand", "[Quick Read-Only Lookup", "[check-", "Duplicate Notification",
                 "Ignore duplicate spawn", "2023-05-18T12:00:00")
    count = 0
    for stream in load_streams():
        assert len({a.supervisor_name for a in stream.accounts}) == 17
        assert len({a.supervisor_email for a in stream.accounts}) == 17
        names = {a.account_id: a.supervisor_name for a in stream.accounts}
        tasks = {t.task_id: t for t in stream.tasks}
        codes = [t for t in stream.tasks if t.substrate == "cooperbench"]
        assert len({t.prompt_text[:120] for t in codes}) == 3
        reposts = [e for e in stream.events if e.kind == "external_duplicate_webhook"
                   and tasks[e.target_task_id].substrate == "cooperbench"]
        assert len({e.text[:120] for e in reposts}) == len(reposts)
        for task in stream.tasks:
            assert not any(word in task.prompt_text for word in forbidden)
            if task.substrate == "cooperbench":
                opening = task.prompt_text.split('\n\n', 1)[0]
                spec_body = task.prompt_text.split('\n\n', 1)[1]
                title = _feature_title(spec_body)
                assert title in opening and str(task.repo_slug) in opening and str(task.project_subdir) in opening
            else:
                assert names[task.account_id] in task.prompt_text.splitlines()[0]
                instruction = json.loads((PROJECT_ROOT / 'data/tasks' / task.upstream_task_id / 'specs.json').read_text())["instruction"].strip()
                if task.amendment_text:
                    if "phone number" in task.amendment_text:
                        instruction = re.sub(r"\bto\b[^.,\n]{1,40}\b\d{7,}\b",
                                             "to my contact; I will send their phone number shortly", instruction, count=1)
                    else:
                        instruction = re.sub(r"\bthat\b[^.,\n]{1,25}\bbefore\s+\d{4}\s+year\b",
                                             "that were released before the cutoff year (I will send the exact year shortly)", instruction, count=1)
                assert f'\n"{instruction}"' in task.prompt_text
        for event in stream.events:
            count += 1
            assert not any(word in event.text for word in forbidden), (stream.stream_id, event.event_id)
            if event.kind == "work_amendment":
                task = tasks[event.target_task_id]
                assert (event.sender, event.channel) == (task.sender, task.channel)
                if task.substrate == "appworld":
                    assert names[task.account_id] in event.text
                else:
                    assert str(task.repo_slug) in event.text and str(task.project_subdir) in event.text
            if event.kind == "external_duplicate_webhook":
                assert event.sender in {"marco_sre", "sam_qa"}
                assert event.text.endswith(tasks[event.target_task_id].prompt_text.removeprefix("@agent "))
        amended = next(t for t in stream.tasks if t.category == "app_write" and t.amendment_text)
        assert amended.amendment_text and (
            re.search(r"phone number \d+", amended.amendment_text)
            or re.search(r"cutoff year is \d{4}", amended.amendment_text)
        )
    assert count == 720


@pytest.mark.parametrize('source,expected', [
    ('**Title**: Add Dynamic Default Resolution for Click Options', 'Add Dynamic Default Resolution for Click Options'),
    ('**Feature: Search Path Priority Configuration for FileSystemLoader**', 'Search Path Priority Configuration for FileSystemLoader'),
    ('# feat(loader): Preserve Search Order', 'Preserve Search Order'),
])
def test_feature_title_extraction(source, expected) -> None:
    from synaptic_stream_60.build_streams import _feature_title
    assert _feature_title(source + '\n\nSpecification') == expected


def test_account_resolver_rejects_unknown_or_ambiguous_names(tmp_path) -> None:
    from synaptic_stream_60.aw_multi_server import _MultiAccountAppWorldRuntime
    runtime = _MultiAccountAppWorldRuntime(tmp_path)
    runtime.accounts = {
        'acct_01': {'supervisor_name': 'Alex Rivera', 'supervisor_email': 'alex.rivera@example.com'},
        'acct_02': {'supervisor_name': 'Alex Rivera', 'supervisor_email': 'other@example.com'},
    }
    assert runtime._resolve_account_id_unlocked('ALEX.RIVERA@EXAMPLE.COM') == 'acct_01'
    for ref in (None, '', 'Alex', 'acct_99'):
        with pytest.raises(ValueError, match='Unknown account'):
            runtime._resolve_account_id_unlocked(ref)
    with pytest.raises(ValueError, match='Ambiguous'):
        runtime._resolve_account_id_unlocked('  ALEX__RIVERA  ')


def test_account_resolution_through_server_and_cli(tmp_path) -> None:
    import subprocess
    import sys
    from dataclasses import asdict
    from synaptic_stream_60.aw_multi_server import StreamAppWorldServerProcess, build_stream_aw_cli_script
    stream = load_streams()[0]
    account = stream.accounts[0]
    server = StreamAppWorldServerProcess(log_path=tmp_path / 'server.log')
    try:
        server.start()
        server.start_stream(stream.stream_id, [asdict(account)], tmp_path / 'snapshots')
        cli = tmp_path / 'aw'
        cli.write_text(build_stream_aw_cli_script(server.base_url))
        refs = [account.account_id, account.supervisor_name, account.supervisor_email,
                '  ' + account.supervisor_name.upper().replace(' ', '__') + '  ', account.supervisor_email.upper()]
        for ref in refs:
            apps = server._post('/apps', {'account_id': ref})
            assert apps['ok'] and apps['account_id'] == account.account_id
            docs = server._post('/docs', {'account_id': ref, 'app_name': 'supervisor', 'api_name': 'show_profile'})
            assert docs['ok']
            result = server.execute_code(ref, 'print(apis.supervisor.show_profile())')
            assert result['ok'] and result['account_id'] == account.account_id
            assert account.supervisor_email in result['output']
            answer = server.submit_answer(ref, 'recorded')
            assert answer['ok'] and answer['account_id'] == account.account_id
            proc = subprocess.run([sys.executable, str(cli), '--account', ref, 'apps'], text=True, capture_output=True)
            assert proc.returncode == 0, proc.stderr
            assert json.loads(proc.stdout)['account_id'] == account.account_id
        proc = subprocess.run([sys.executable, str(cli), account.account_id, 'apps'], text=True, capture_output=True)
        assert proc.returncode == 0 and json.loads(proc.stdout)['account_id'] == account.account_id
    finally:
        server.stop()


def _runtime_path_env(ws, arm, runtime_bin):
    """Execute the actual bootstrap PATH assignments, without starting a model host."""
    import ast
    import inspect
    import os
    import textwrap
    from dsh_runner import dsh_controller, synaptic_worker_controller
    if arm == "dsh_superagent":
        method = dsh_controller.GenuineDSHSuperagent.bootstrap_and_start
        owner = dsh_controller
        variables = {"bin_dir": runtime_bin}
    else:
        method = synaptic_worker_controller.SynapticDSHWorkerHost._bootstrap_process_sync
        owner = synaptic_worker_controller
        variables = {"guard_bin_dir": runtime_bin}
    body = ast.parse(textwrap.dedent(inspect.getsource(method))).body[0].body
    start = next(i for i, n in enumerate(body) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "venv_bin" for t in n.targets))
    env = {**os.environ, **ws["extra_env"], "PYTHONPATH": ""}
    namespace = {"env": env, "Path": Path, "PROJECT_ROOT": owner.PROJECT_ROOT, **variables}
    exec(compile(ast.Module(body=body[start:start + 3], type_ignores=[]), "bootstrap-path", "exec"), namespace)
    return env


@pytest.mark.parametrize("arm", ["clonact", *PARADIGM_BASELINE_ARMS, "dsh_superagent"])
@pytest.mark.parametrize('stream_idx,project_id,test_file,module', [
    (0, 'click', 'tests/test_defaults.py', 'click'),
    (0, 'jinja', 'tests/test_loader.py', 'jinja2'),
    (0, 'dspy', 'tests/signatures/test_signature.py', 'dspy'),
    (2, 'networkx', 'networkx/algorithms/approximation/tests/test_dominating_set.py', 'networkx'),
])
def test_pytest_shim_in_repo_and_separate_worktree(tmp_path, stream_idx, project_id, test_file, module, arm) -> None:
    import shutil
    import subprocess
    from synaptic_stream_60 import runner
    from synaptic_stream_60.watcher import _strip_runtime_diff, _extract_agent_diff
    stream = load_streams()[stream_idx]
    ws = runner._prepare_stream_workspace_files(stream, tmp_path / 'workspace', 'http://127.0.0.1:1')
    try:
        for pid, (bare_p, py_p, _sha) in ws['project_eval_configs'].items():
            assert bare_p.is_symlink() and bare_p.parent.name == 'repo', pid
            assert py_p.parent.parent.is_symlink() and py_p.parent.parent.parent.name == 'venv', pid
        assert 'upstream_pillars' not in ws['pytest_shim']
        assert 'upstream_pillars' not in ws['python_shim']
        repo = ws['repo_dir']
        worktree = ws['ws_root'] / 'worktree'
        subprocess.run(['git', '-C', str(repo), 'worktree', 'add', '--detach', str(worktree), 'HEAD'], check=True, capture_output=True)
        (worktree / '.venvs').symlink_to(ws['shared_links'] / '.venvs')
        (worktree / 'aw').symlink_to(ws['shared_links'] / 'aw')
        (worktree / 'AGENTS.md').write_text(ws['agents_md'])
        for directory in (repo, worktree):
            (directory / '.cache').mkdir()
            (directory / '.cache/probe').write_text('cache')
            (directory / 'WORKSPACE_POLICY.md').write_text('policy')
            (directory / 'feishu_docs').mkdir()
            assert subprocess.check_output(['git', 'status', '--porcelain'], cwd=directory, text=True) == ''
            assert _strip_runtime_diff(_extract_agent_diff(directory, ws['monorepo_base_commit'])) == ''
        assert Path(ws['extra_env']['DSPY_CACHEDIR']).is_dir()
        assert ws['extra_env']['PAGER'] == 'cat'
        assert ws['extra_env']['PYTHONDONTWRITEBYTECODE'] == '1'
        runtime_bin = ws['ws_root'] / 'runtime_bin'
        runtime_bin.mkdir()
        if arm == 'dsh_superagent':
            for name, content in [
                ('pytest', ws['pytest_shim']),
                ('python', ws['python_shim']),
                ('python3', ws['python_shim']),
                ('aw', ws['aw_cli']),
            ]:
                (runtime_bin / name).write_text(content)
                (runtime_bin / name).chmod(0o755)
        env = _runtime_path_env(ws, arm, runtime_bin)
        for directory in (repo, worktree):
            proj_dir = (directory / 'projects' / project_id).resolve()
            probe = proj_dir / 'test_worktree_import.py'
            probe.write_text(
                f'import {module}\nfrom pathlib import Path\n'
                f'def test_source():\n    assert Path({module}.__file__).resolve().is_relative_to(Path({str(proj_dir)!r}))\n'
            )
            # 1) Invoke pytest from inside projects/<pid>/
            proc = subprocess.run(['pytest', '-q', test_file, probe.name], cwd=proj_dir, env=env,
                                  capture_output=True, text=True, timeout=90)
            assert proc.returncode == 0, (project_id, arm, 'subdir', proc.stdout, proc.stderr)
            # 2) Invoke pytest from monorepo root passing projects/<pid>/...
            proc_root = subprocess.run(
                ['pytest', '-q', f'projects/{project_id}/{probe.name}'],
                cwd=directory, env=env, capture_output=True, text=True, timeout=90,
            )
            assert proc_root.returncode == 0, (project_id, arm, 'root', proc_root.stdout, proc_root.stderr)
            # 3) Invoke direct .venvs/<pid>/bin/pytest from monorepo root
            proc_venv = subprocess.run(
                [str(directory / '.venvs' / project_id / 'bin' / 'pytest'), '-q', f'projects/{project_id}/{probe.name}'],
                cwd=directory, env=env, capture_output=True, text=True, timeout=90,
            )
            assert proc_venv.returncode == 0, (project_id, arm, '.venvs', proc_venv.stdout, proc_venv.stderr)
            # 4) Cross-worktree invocation from repo/ targeting ../worktree/projects/<pid>/...
            if directory == worktree:
                proc_cross = subprocess.run(
                    ['pytest', '-q', f'../worktree/projects/{project_id}/{probe.name}'],
                    cwd=repo, env=env, capture_output=True, text=True, timeout=90,
                )
                assert proc_cross.returncode == 0, (project_id, arm, 'cross-wt', proc_cross.stdout, proc_cross.stderr)
            probe.unlink()
            # 5) Invoke python inside projects/<pid>/
            proc_py = subprocess.run(
                ['python', '-c', f'import {module}; print({module}.__file__)'],
                cwd=proj_dir, env=env, capture_output=True, text=True, timeout=60,
            )
            assert proc_py.returncode == 0 and str(proj_dir) in str(Path(proc_py.stdout.strip()).resolve()), (
                project_id, arm, proc_py.stdout, proc_py.stderr
            )
            for name in ('python', 'python3', 'pytest', 'aw'):
                assert shutil.which(name, path=env['PATH']) is not None
            assert subprocess.run(['aw', '--help'], cwd=directory, env=env, capture_output=True).returncode == 0
        # A normal add/commit includes the real edit and none of the runtime files.
        (worktree / 'projects' / project_id / 'actual_change.txt').write_text('delivered')
        subprocess.run(['git', 'add', '-A'], cwd=worktree, check=True)
        subprocess.run(['git', '-c', 'commit.gpgsign=false', 'commit', '-m', 'actual change'], cwd=worktree, check=True, capture_output=True)
        assert subprocess.check_output(['git', 'diff-tree', '--no-commit-id', '--name-only', '-r', 'HEAD'], cwd=worktree, text=True).splitlines() == [
            f'projects/{project_id}/actual_change.txt'
        ]
        for directory in (repo, worktree):
            assert subprocess.check_output(['git', 'status', '--porcelain'], cwd=directory, text=True) == ''
    finally:
        shutil.rmtree(ws['snapshots_root'], ignore_errors=True)


@pytest.mark.parametrize("reply", [
    "2 done, 1 running.", "两个已完成，一个测试中。", "All active tasks are finished.",
    "好的，代码已完成；个人应用任务仍在处理中，没有阻塞。",
    "目前只有 1 个任务在跑：正在实现动态默认值，尚未跑完测试。",
    "One request is running right now; none finished yet.", "正在跑 X，另外 Y 已完成",
    "Sure, two requests finished; one is running tests.",
])
def test_peek_status_overviews(reply) -> None:
    from synaptic_stream_60.runner import is_substantive_peek_reply
    assert is_substantive_peek_reply(reply)


@pytest.mark.parametrize("reply", [
    "Done: sent $74.00 back to Robert...", "Finished the Click feature; remaining tests passed.",
    "Implemented requests.get retry handling; tests passed.",
    "PR #12 is done and all tests passed.",
])
def test_unbound_completion_is_not_a_peek(reply) -> None:
    from synaptic_stream_60.runner import is_substantive_peek_reply
    assert not is_substantive_peek_reply(reply)
    assert is_substantive_peek_reply(reply, causally_bound=True)
    assert is_substantive_peek_reply(reply, parent_id="p", message_id="p", require_parent_match=True)


@pytest.mark.parametrize("reply", ["好的", "ok!", "Sure", "收到！", "understood", "Got it!",
                                      "好的，On it: checking all requests.", "Created ticket for all tasks",
                                      "Working on your request; all tasks are running"])
def test_ack_is_not_a_peek(reply) -> None:
    from synaptic_stream_60.runner import is_substantive_peek_reply
    assert not is_substantive_peek_reply(reply)
    assert not is_substantive_peek_reply(reply, causally_bound=True)


def test_driver_peeks_claim_fifo_across_overlapping_arrivals() -> None:
    from synaptic_stream_60.runner import _DriverPeekQueue
    queue = _DriverPeekQueue()
    messages = ["All previous tasks finished."]
    queue.observe(messages, [0] * len(messages), 0)
    a = queue.add(1, len(messages), True, 0)
    messages.append("All active tasks are finished.")  # immediate prior-turn tail
    queue.observe(messages, [0] * len(messages), 1.1)
    b = queue.add(2, len(messages), False, 0)
    messages += ["Done: sent $74.00 back to Robert...", "2 done, 1 running."]
    queue.observe(messages, [0] * len(messages), 3)
    assert a["reply"] == messages[3] and a["latency_ms"] == 2000
    assert b["reply"] is None
    messages.append("两个已完成，一个测试中。")
    queue.observe(messages, [0] * len(messages), 4)
    assert b["reply"] == messages[4] and queue.claimed == {3, 4}
    queue.observe(messages, [0] * len(messages), 5)
    assert queue.claimed == {3, 4}
    expired = queue.add(5, len(messages), False, 0)
    current = queue.add(64, len(messages), False, 0)
    messages.append("All active tasks are finished.")
    queue.observe(messages, [0] * len(messages), 65.001)
    assert expired["reply"] is None and current["reply"] == messages[-1]
    exact = queue.add(100, len(messages), False, 0)
    messages.append("All active tasks are finished.")
    queue.observe(messages, [0] * len(messages), 160)
    assert exact["latency_ms"] == 60000


def test_dataset_rebuild_is_deterministic() -> None:
    from synaptic_stream_60.build_streams import (
        _load_existing_benchmark_ids, _select_zero_overlap_appworld_tasks,
        _build_single_macro_stream, _REPO_STREAM_CONFIGS,
    )
    pa_cb, pa_swe, task_ids, generators = _load_existing_benchmark_ids()
    writes, qas = _select_zero_overlap_appworld_tasks(task_ids, generators)
    rebuilt = [_build_single_macro_stream(i, cfg, writes[i], qas[i], pa_cb, pa_swe)
               for i, cfg in enumerate(_REPO_STREAM_CONFIGS)]
    assert [s.to_dict() for s in rebuilt] == [s.to_dict() for s in load_streams()]


def test_stream_qa_answer_normalization() -> None:
    from synaptic_stream_60.aw_multi_server import _normalize_stream_qa_answer
    assert _normalize_stream_qa_answer(
        "Total 2023 utility bill cost: $42 (Jan $20 + Feb $22).",
        instruction="What is the total cost of my utility bills in 2023?",
    ) == 42
    assert _normalize_stream_qa_answer(
        "The total sent since 2023-01-01 is $89.0",
        instruction="How much money did I send since 2023-01-01?",
    ) == 89
    assert _normalize_stream_qa_answer(
        "Since Jan 1, 2023 you sent 89.00 USD.",
        instruction="How much money have I sent since 1st Jan of this year?",
    ) == 89
    assert _normalize_stream_qa_answer(
        "In 2023, total cost is 42.",
        instruction="What was the total cost of orders placed in 2023?",
    ) == 42
    assert _normalize_stream_qa_answer(
        '["Item One", "Item Two"]',
        instruction="What are the comma-separated titles of the top items?",
    ) == "Item One, Item Two"
    assert _normalize_stream_qa_answer(
        "Sample Playlist.",
        instruction="What is the name of the playlist?",
    ) == "Sample Playlist"


def test_subproject_diff_extraction_and_watcher_isolation(tmp_path: Path) -> None:
    import asyncio
    import shutil
    from synaptic_stream_60 import runner
    from synaptic_stream_60.watcher import StreamStateWatcher, _extract_subproject_diff

    stream = load_streams(["STREAM_03"])[0]
    ws = runner._prepare_stream_workspace_files(stream, tmp_path / "ws", "http://127.0.0.1:1")
    try:
        repo = ws["repo_dir"]
        calls: list[tuple[str, str]] = []

        def fake_eval(*, bare_repo_path, py_bin, base_commit, filtered_diff, has_conflicts, tests_patch_path):
            calls.append((tests_patch_path.parent.name, filtered_diff))
            return {"passed": "sentinel_nx" in filtered_diff, "returncode": 0 if "sentinel_nx" in filtered_diff else 1}

        import synaptic_stream_60.watcher as watcher_mod
        orig_eval = watcher_mod._evaluate_cooperbench_diff_isolated
        watcher_mod._evaluate_cooperbench_diff_isolated = fake_eval
        try:
            watcher = StreamStateWatcher(
                stream,
                regime="moderate",
                canonical_repo_dir=repo,
                bare_repo_path=ws["bare_repo"],
                repo_py_bin=ws["repo_py_bin"],
                snapshots_root=ws["snapshots_root"],
                aw_eval_worker=None,  # type: ignore[arg-type]
                get_replies_fn=lambda: [],
                monorepo_base_commit=ws["monorepo_base_commit"],
                project_eval_configs=ws["project_eval_configs"],
            )
            # Sample at t=500s so all 3 code tasks have arrived; initially all 3 have empty sub-diffs.
            asyncio.run(watcher.sample_once(500.0))
            assert watcher.code_evals_executed == 3

            # Edit ONLY projects/networkx; ONLY networkx task should trigger a new eval!
            target = repo / "projects/networkx/networkx/algorithms/approximation/dominating_set.py"
            target.write_text(target.read_text(encoding="utf-8") + "\n# sentinel_nx\n", encoding="utf-8")
            asyncio.run(watcher.sample_once(501.0))
            assert watcher.code_evals_executed == 4
            nx_task = next(t for t in stream.tasks if t.project_id == "networkx")
            click_task = next(t for t in stream.tasks if t.project_id == "click")
            assert watcher.observations[nx_task.task_id][-1][1] is True
            assert watcher.observations[click_task.task_id][-1][1] is False
            # Verify extracted diff strips `projects/networkx/` prefix cleanly
            sub = _extract_subproject_diff(watcher._last_raw_diff, "projects/networkx")
            assert "diff --git a/networkx/algorithms/approximation/dominating_set.py b/networkx/algorithms/approximation/dominating_set.py" in sub
            assert "projects/networkx" not in sub
            # Verify timestamp_override clamps final sweep even if host clock advanced during teardown
            watcher.stream_t0 = 0.0
            asyncio.run(watcher.sample_once(505.0, timestamp_override=505.0))
            assert watcher.observations[nx_task.task_id][-1][0] == 505.0
        finally:
            watcher_mod._evaluate_cooperbench_diff_isolated = orig_eval
    finally:
        shutil.rmtree(ws["snapshots_root"], ignore_errors=True)


def test_superagent_extra_bin_files_stay_out_of_seed_repo(monkeypatch, tmp_path) -> None:
    import inspect
    from types import SimpleNamespace
    from dsh_runner import dsh_controller as module
    agent = module.GenuineDSHSuperagent(tmp_path / 'instance')
    seed = tmp_path / 'seed'
    seed.mkdir()
    seen = []
    def init(world, **kwargs):
        seen.append(kwargs)
        return {'clone_repo': seed}
    monkeypatch.setattr(module, 'init_dsh_sandbox', init)
    class StopBeforeRuntime(Exception):
        pass
    def stop(**kwargs):
        raise StopBeforeRuntime
    # Exercise the actual preparation path, stopping before credentials or runtime launch.
    agent.dsh_home = SimpleNamespace(mkdir=stop)
    with pytest.raises(StopBeforeRuntime):
        agent.bootstrap_and_start(seed_repo_dir=seed, extra_bin_files={'pytest': ('#!/bin/sh\nexit 0\n', 0o755)})
    shim = agent.world_dir / 'bin/pytest'
    assert shim.read_text() == '#!/bin/sh\nexit 0\n'
    assert shim.stat().st_mode & 0o777 == 0o755
    assert not (seed / 'pytest').exists()
    assert seen[0]['extra_repo_files'] is None
    parameter = inspect.signature(agent.bootstrap_and_start).parameters['extra_bin_files']
    assert parameter.default is None and parameter.kind is inspect.Parameter.KEYWORD_ONLY


@pytest.mark.parametrize('arm', ['clonact', 'clonact_late', 'clonact_ack', 'dsh_superagent', 'dsh_mono', 'dsh_mono_late_wakeup'])
def test_event_delivery_correlates_untagged_peek(arm) -> None:
    """Run the real delivery coroutine with deterministic transports, without a model."""
    import ast
    import asyncio
    from types import SimpleNamespace
    from synaptic_stream_60 import runner
    from synaptic_stream_60.schema import StreamEventSpec
    tree = ast.parse(Path(runner.__file__).read_text())
    delivery = next(n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef) and n.name == '_deliver_one_event')
    for i, statement in enumerate(delivery.body):
        if isinstance(statement, ast.Nonlocal):
            delivery.body[i] = ast.Global(names=statement.names)
    module = ast.fix_missing_locations(ast.Module(body=[delivery], type_ignores=[]))
    summary = 'Two requests finished; one is running tests.'
    stale = 'All previous requests finished; nothing pending.'
    now = [0.0]
    sleeps = []
    async def sleep(delay):
        sleeps.append(delay)
        now[0] += delay
        if arm == 'dsh_mono_late_wakeup':
            messages.append(summary)
            counts.append(1)  # A tool step consumed the queued steer.
            env['driver_peeks'].observe(messages, counts, 59.99)
            now[0] = 60.01
        await asyncio.sleep(0)
    async def user_message(*args, **kwargs):
        if arm == 'clonact_late':
            now[0] = 60.000001
        return {'reply': '好的' if arm == 'clonact_ack' else summary, 'transition': 'peek'}
    messages = [stale]
    counts = [0]
    status_seen = []
    async def status(*args):
        status_seen.append(True)
        return {"busy": True}
    async def inbound(**kwargs):
        assert status_seen  # Must query before injection, not use the delivery result.
        if arm != 'dsh_mono_late_wakeup':
            messages.extend(['All active tasks are finished.', summary])
            counts.extend([0, 0])
        return {'was_busy_at_arrival': False}
    polls = []
    def replies(world):
        polls.append(1)
        from dsh_runner.dsh_controller import dsh_key, dsh_trigger_id_for_injected
        trigger = dsh_trigger_id_for_injected({'channel': 'feishu', 'message_id': 'this-message'})
        return [{'markdown': stale, 'parent_id': 'older'},
                {'markdown': 'All active tasks are finished.', 'parent_id': 'this-message', 'idempotency_key': dsh_key('ack', trigger)},
                {'markdown': 'All active tasks are finished.', 'parent_id': 'this-message', 'idempotency_key': 'ack:legacy'},
                {'markdown': summary, 'parent_id': 'this-message'}]
    env = dict(vars(runner))
    env.update(
        asyncio=SimpleNamespace(sleep=sleep, to_thread=asyncio.to_thread),
        time=SimpleNamespace(perf_counter=lambda: now[0]), stream_t0=0, regime='moderate',
        mmu=SimpleNamespace(handle_user_message=user_message, unified_stream=[SimpleNamespace(kind='agent_reply', text=stale)]) if arm.startswith('clonact') else None,
        driver=SimpleNamespace(deliver_inbound_event=inbound, primary_assistant_messages=messages, primary_assistant_message_tool_counts=counts, total_tool_calls=0, host=SimpleNamespace(get_session_status=status), primary_session_id="primary") if arm.startswith('dsh_mono') else None,
        superagent=SimpleNamespace(world_dir=Path('.')) if arm == 'dsh_superagent' else None,
        _inject_stream_event_into_superagent=lambda *a: {'channel': 'feishu', 'message_id': 'this-message'},
        list_bot_replies=replies, noise_dropped_cnt=0, dup_merged_cnt=0, sa_expected_work_triggers=set(),
        driver_peeks=runner._DriverPeekQueue(), peek_latencies_ms=[], event_delivery_trace=[],
    )
    exec(compile(module, str(runner.__file__), 'exec'), env)
    event = StreamEventSpec('peek', 0, 0, 'status_followup', 'feishu://p2p/user', 'viacol', 'Which requests are finished or still running?')
    asyncio.run(env['_deliver_one_event'](event))
    row = env['event_delivery_trace'][0]
    assert 'error' not in row, row
    if arm in ('clonact_late', 'clonact_ack'):
        assert row['peek_reply'] is None and row['peek_censored'] and row['peek_latency_ms'] is None
        assert sleeps == []
        return
    assert row['peek_reply'] == summary and not row['peek_censored']
    if arm == 'dsh_mono_late_wakeup':
        assert env['driver_peeks'].claimed == {1}
        assert row['peek_latency_ms'] == 59990.0
    if arm == 'dsh_mono':
        assert env['driver_peeks'].claimed == {2}
    if arm == 'dsh_superagent':
        assert polls


def test_gold_verify_saves_report_without_model_start(monkeypatch, tmp_path) -> None:
    import argparse
    import asyncio
    from synaptic_stream_60 import runner
    async def gold(stream_ids):
        assert stream_ids == ['STREAM_01']
        return {'all_gates_passed': True, 'total_tasks_verified': 20, 'total_tasks': 20}
    monkeypatch.setattr(runner, 'run_gold_composability_gate', gold)
    def no_llm(*args, **kwargs):
        pytest.fail('gold verification must not initialize LLM runtime')
    monkeypatch.setattr(runner, 'load_project_dotenv', no_llm)
    args = argparse.Namespace(mode='gold_verify', streams='STREAM_01', output_dir=str(tmp_path))
    report = asyncio.run(runner.run_suite_async(args))
    assert json.loads((tmp_path / 'gold_verification.json').read_text()) == report
    assert report['benchmark_version'] == report['evaluator_version'] == '1.0.0'


def test_stream_text_polish_rooms_and_unique_noise() -> None:
    import re
    rooms = {'team_general', 'engineering', 'personal_ops', 'ops_alerts'}
    for stream in load_streams():
        noise = [e for e in stream.events if e.kind == 'external_noise']
        assert len({e.text for e in noise}) == 178
        assert all(e.text == e.text.strip() and '@agent' not in e.text for e in noise)
        assert {e.channel.rsplit('/', 1)[-1] for e in noise} == rooms
        reposts = [e for e in stream.events if e.kind == 'external_duplicate_webhook']
        assert len(reposts) == 24
        assert {e.channel.rsplit('/', 1)[-1] for e in reposts} == rooms - {'team_general'}
        assert all('in ops_alerts' not in e.text for e in reposts)
        peeks = [e.text for e in stream.events if e.kind == 'status_followup']
        assert len(set(peeks)) == 16
        assert sum(bool(re.search('[\u4e00-\u9fff]', text)) for text in peeks) == 8
        for task in stream.tasks:
            if task.category == 'readonly_qa':
                intro = task.prompt_text.splitlines()[0]
                assert 'record' in intro and 'send' in intro
                assert any(word in intro for word in ("don't change", 'unchanged', "don't modify", 'without editing', "don't alter"))
            if task.substrate == 'cooperbench' and task.amendment_text:
                assert '\n\n\n' not in task.prompt_text
                assert not re.match(r'\d+\.', task.amendment_text)
                assert '\n       ' not in task.amendment_text
                if stream.stream_id == 'STREAM_01':
                    assert 'info_dict' not in task.prompt_text
                    assert 'to_info_dict()' in task.amendment_text
                if stream.stream_id == 'STREAM_02':
                    assert 'Preserve the trailing space' not in task.prompt_text
                    assert 'Preserve the trailing space' in task.amendment_text
                if stream.stream_id == 'STREAM_03':
                    assert 'get_signature(predictor).instructions' not in task.prompt_text
                    assert 'get_signature(predictor).instructions' in task.amendment_text


def test_actual_feishu_rooms_and_reply_ack_ids(tmp_path) -> None:
    import subprocess
    from dsh_runner.dsh_controller import GenuineDSHSuperagent, dsh_key, dsh_trigger_id_for_injected
    from dsh_runner import sandbox_clis as cli
    from synaptic_stream_60.runner import _inject_stream_event_into_superagent, _is_superagent_ack
    agent = GenuineDSHSuperagent(tmp_path / 'instance')
    world = agent.world_dir
    world.mkdir(parents=True)
    (world / 'sandbox_meta.json').write_text('{}')
    cli.init_db_schema(world)
    stream = load_streams()[0]
    channels = sorted({e.channel for e in stream.events if e.channel.startswith('feishu://group/')})
    cli.configure_feishu_group_channels(world, channels)
    for channel in channels:
        event = next(e for e in stream.events if e.channel == channel)
        injection = _inject_stream_event_into_superagent(agent, event)
        with cli.connect_db(world) as conn:
            row = conn.execute('SELECT * FROM feishu_messages WHERE message_id = ?', (injection['message_id'],)).fetchone()
        assert row['chat_id'] == cli.feishu_chat_id(channel)
        assert bool(json.loads(row['mentions_json'])) == ('@agent' in event.text)
    import sys
    result = subprocess.run([sys.executable, cli.__file__, 'lark', str(world), 'im', '+chat-list'],
                            capture_output=True, text=True, check=True)
    chats = json.loads(result.stdout)['data']['chats']
    assert {c['chat_id'] for c in chats} == {cli.feishu_chat_id(c) for c in channels} | {cli.FEISHU_P2P_CHAT_ID}
    peek = next(e for e in stream.events if e.kind == 'status_followup')
    injection = _inject_stream_event_into_superagent(agent, peek)
    mid = injection['message_id']
    trigger = dsh_trigger_id_for_injected(injection)
    ack = dsh_key('ack', trigger)
    for index, key in enumerate((ack, 'ack:legacy', dsh_key('reply', trigger, '2 done, 1 running.'))):
        subprocess.run([sys.executable, cli.__file__, 'lark', str(world),
                        'im', '+messages-reply', '--message-id', mid, '--markdown', '2 done, 1 running.',
                        '--idempotency-key', key], capture_output=True, text=True, check=True)
    replies = cli.list_bot_replies(world)
    assert [r['idempotency_key'] for r in replies] == [ack, 'ack:legacy', dsh_key('reply', trigger, '2 done, 1 running.')]
    assert [_is_superagent_ack(row, mid) for row in replies] == [True, True, False]
    assert all(row['parent_id'] == mid for row in replies)


def test_superagent_hygiene_real_schema() -> None:
    from synaptic_stream_60.runner import superagent_hygiene, dsh_trigger_id_for_injected
    from synaptic_stream_60.schema import StreamEventSpec
    dispositions = [
        {'filtered': {'code': 'notMentioned'}},
        {'decision': {'kind': 'ignore', 'reason': [{'code': 'noAdmissible'}]}},
        {'fanout': [{'ticketId': 'existing'}]},
        {'decision': {'outcome': 'ignore', 'reason': [{'code': 'routed', 'values': {'ticket': 'old'}}]}},
        {'decision': {'kind': 'ignore', 'reason': [{'code': 'said', 'text': 'Already tracked'}]}},
        {'filtered': {'code': 'policy'}},
        {'decision': {'kind': 'accept', 'ticketId': 'new'}, 'fanout': [{'ticketId': 'new'}]},
        {'decision': {'kind': 'unavailable'}},
    ]
    events = [StreamEventSpec(str(i), i, i, 'external_noise' if i < 2 else 'external_duplicate_webhook',
                              'feishu://group/engineering', 'sam_qa', 'message') for i in range(len(dispositions))]
    trace = [{'event_id': e.event_id, 'injected': {'channel': 'feishu', 'message_id': e.event_id}} for e in events]
    triggers = [{'id': dsh_trigger_id_for_injected(t['injected']), 'disposition': d} for t, d in zip(trace, dispositions)]
    result = superagent_hygiene(events, trace, triggers)
    assert result['noise_dropped'] == 2
    assert result['duplicates_merged'] == 4
    assert superagent_hygiene(events, trace, [])['duplicates_merged'] == 0


@pytest.mark.parametrize('arm', ['dsh_subagent_fork', 'dsh_subagent_neutral'])
def test_subagent_prompt_tokens_and_routing_from_real_event_handler(tmp_path, arm) -> None:
    import asyncio
    from clonact.paradigm_baselines import DSHParadigmDriver, _content_tokens
    driver = DSHParadigmDriver(arm, instance_dir=tmp_path / 'instance', allowed_workspace_root=tmp_path,
                              repo_dir=tmp_path / 'repo', worktrees_dir=tmp_path / 'worktrees',
                              shared_links_dir=tmp_path / 'shared', agents_md='', has_code_repo=True)
    driver.primary_session_id = 'primary'
    driver._last_inbound_event_id = 'event'
    prompt = 'Implement dynamic defaults and verify the local test suite with pytest.'
    async def run():
        await driver._on_session_event({'kind': 'tool_call', 'session_id': 'primary', 'tool_name': 'subagent',
                                       'tool_call_id': 'call1', 'tool_args': {'prompt': prompt}})
        await driver._on_session_event({'kind': 'assistant_message', 'session_id': 'child', 'is_subagent': True,
                                       'text': 'Working on the feature.'})
        await driver._on_session_event({'kind': 'tool_call', 'session_id': 'primary', 'tool_name': 'send_message',
                                       'tool_args': {'to': 'child', 'message': 'Please check the default display.'}})
    asyncio.run(run())
    assert len(driver.spawned_workers) == 1
    worker = driver.spawned_workers[0]
    assert worker['prompt'] == prompt and worker['tokens'] == _content_tokens(prompt)
    assert worker['bound_sid'] == 'child'
    assert driver.routing_events['event']['routed_cells'] == ['child']


def test_mmu_completed_tasks_drain_and_unfinished_tasks_cancel(tmp_path) -> None:
    import asyncio
    from clonact.context_mmu import ContextMMU
    from clonact.trunk import ElasticSharedTrunk
    from dsh_runner.synaptic_worker_controller import SynapticDSHWorkerHost
    from synaptic_stream_60.runner import _drain_mmu_if_idle, _cancel_mmu_background
    # Real MMU and real asyncio Tasks; no host/model calls are needed to drain an empty cell set.
    host = SynapticDSHWorkerHost(tmp_path / 'instance', allowed_workspace_root=tmp_path)
    mmu = ContextMMU(sandbox_paths={'world': tmp_path}, dsh_host=host, trunk=ElasticSharedTrunk())
    async def run():
        try:
            done = asyncio.create_task(asyncio.sleep(0))
            await done
            mmu.background_tasks['done'] = done
            assert not await _drain_mmu_if_idle(mmu, False)
            assert await _drain_mmu_if_idle(mmu, True)
            assert mmu.background_tasks == {}
            waiting = asyncio.create_task(asyncio.Event().wait())
            mmu.background_tasks['waiting'] = waiting
            assert not await _drain_mmu_if_idle(mmu, True)
            await _cancel_mmu_background(mmu)
            assert waiting.cancelled()
        finally:
            await mmu.judge.aclose()
    asyncio.run(run())


def test_real_server_redacts_upstream_instruction_on_load_and_reload(tmp_path) -> None:
    import subprocess
    import sys
    from dataclasses import asdict
    from synaptic_stream_60.aw_multi_server import StreamAppWorldServerProcess, StreamAppWorldEvalWorkerProcess, build_stream_aw_cli_script, PROJECT_ROOT
    stream = load_streams()[0]
    target_acct = next(a for a in stream.accounts if a.upstream_task_id == '37a8675_2')
    other = next(a for a in stream.accounts if a.account_id != target_acct.account_id)
    original = json.loads((PROJECT_ROOT / 'data/tasks/37a8675_2/specs.json').read_text())['instruction']
    m_num = re.search(r'\b\d{7,}\b', original)
    assert m_num is not None and target_acct.supervisor_name
    secret_token = m_num.group(0)
    server, evaluator = StreamAppWorldServerProcess(), StreamAppWorldEvalWorkerProcess()
    try:
        server.start(); evaluator.start()
        server.start_stream(stream.stream_id, [asdict(target_acct), asdict(other)], tmp_path / 'snapshots')
        cli = tmp_path / 'aw'
        cli.write_text(build_stream_aw_cli_script(server.base_url))
        for _ in range(2):
            output = server.execute_code(target_acct.account_id, 'print(apis.supervisor.show_active_task())')['output']
            assert 'Traceback' not in output and 'instruction' in output
            assert secret_token not in output and original not in output
            assert "'instruction': ''" in output or '"instruction": ""' in output
            profile = server.execute_code(target_acct.account_id, 'print(apis.supervisor.show_profile()); print(apis.supervisor.show_account_passwords())')['output']
            assert target_acct.supervisor_email in profile and 'Traceback' not in profile
            # Test raw API request path too: it bypasses the show_active_task wrapper.
            raw = server.execute_code(target_acct.account_id, 'print(requester.request("supervisor", "show_active_task"))')['output']
            assert 'Traceback' not in raw and secret_token not in raw and 'instruction' in raw
            server.execute_code(other.account_id, 'print(apis.supervisor.show_profile())')
        for args in [target_acct.supervisor_name.split(), [target_acct.supervisor_email], [f'<{target_acct.supervisor_email}>']]:
            for prefix in ([], ['--account'], ['-a']):
                result = subprocess.run([sys.executable, str(cli), *prefix, *args, 'apps'], capture_output=True, text=True)
                assert result.returncode == 0, result.stderr
                assert json.loads(result.stdout)['account_id'] == target_acct.account_id
        for args in [['exec', 'print("--account", "-a")'], ['answer', '--account', '-a']]:
            result = subprocess.run([sys.executable, str(cli), '--account', target_acct.account_id, *args], capture_output=True, text=True)
            assert result.returncode == 0, result.stderr
            assert '--account' in result.stdout and '-a' in result.stdout
        # Redacted Task.save and supervisor.complete_task must remain functional.
        code = (PROJECT_ROOT / 'data/tasks/37a8675_2/ground_truth/compiled_solution.py').read_text() + '\nsolution(apis, requester)\n'
        result = server.execute_code(spencer.account_id, code)
        assert 'Traceback' not in result['output']
        score = evaluator.evaluate(upstream_task_id=spencer.upstream_task_id, snapshot_dir=result['snapshot']['snapshot_dir'],
                                   is_qa=False, target_apps=list(spencer.target_apps))
        assert score['passed']
    finally:
        server.stop(); evaluator.stop()


@pytest.mark.parametrize('elapsed,accepted', [(0.0, True), (59999.99, True), (60000.0, True), (60000.0001, False), (61000.0, False)])
def test_direct_peek_strict_sixty_second_deadline(elapsed, accepted) -> None:
    from synaptic_stream_60.runner import _direct_peek_result
    reply, latency = _direct_peek_result('代码仍在测试中。', '现在进度怎样？', elapsed)
    assert (reply is not None) == accepted
    assert latency == (elapsed if accepted else None)
    assert _direct_peek_result('好的', '现在进度怎样？', elapsed) == (None, None)


def test_real_git_conflicting_promotion_is_attempted_once_per_heads(tmp_path) -> None:
    import http.server
    import threading
    import subprocess
    from dsh_runner.dsh_controller import GenuineDSHSuperagent
    from synaptic_stream_60.runner import _promote_settled_superagent_tickets_once
    repo, wt = tmp_path / 'main', tmp_path / 'ticket-worktree'
    def git(*args, cwd=repo):
        return subprocess.check_output(['git', *args], cwd=cwd, text=True, stderr=subprocess.STDOUT).strip()
    repo.mkdir()
    git('init', '-b', 'main')
    git('config', 'user.name', 'Test')
    git('config', 'user.email', 'test@example.com')
    git('config', 'commit.gpgsign', 'false')
    (repo / 'module.py').write_text('value = 0\n')
    git('add', '.'); git('commit', '-m', 'base')
    base = git('rev-parse', 'HEAD')
    git('worktree', 'add', '-b', 'worker', str(wt), 'HEAD')
    (wt / 'module.py').write_text('value = 1\n')
    git('add', 'module.py', cwd=wt); git('commit', '-m', 'worker', cwd=wt)
    (repo / 'module.py').write_text('value = 2\n')
    git('add', 'module.py'); git('commit', '-m', 'main')
    ticket = {'id': 'ticket1', 'stage': 'idle', 'directory': str(wt), 'sessionId': 'session1'}
    # A local HTTP fixture supplies recorded ticket metadata; the real controller,
    # Git commit/merge/abort path, worktrees and attempted-pair cache run unchanged.
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_GET(self):
            body = json.dumps(ticket if self.path.endswith('/ticket1') else {'tickets': [ticket]}).encode()
            self.send_response(200); self.end_headers(); self.wfile.write(body)
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    agent = GenuineDSHSuperagent(tmp_path / 'instance', port=server.server_port)
    agent.sandbox_paths = {'clone_repo': repo}
    promoted, attempted = {}, set()
    try:
        for stage in ('working', 'person'):
            ticket['stage'] = stage
            assert _promote_settled_superagent_tickets_once(agent, base, tmp_path / 'shared', promoted, attempted) == 0
            assert not attempted
        ticket['stage'] = 'idle'
        assert _promote_settled_superagent_tickets_once(agent, base, tmp_path / 'shared', promoted, attempted) == 0
        assert len(attempted) == 1 and promoted == {}
        assert git('status', '--porcelain') == '' and (repo / 'module.py').read_text() == 'value = 2\n'
        after_failure_mtime = (repo / 'module.py').stat().st_mtime_ns
        for _ in range(3):
            assert _promote_settled_superagent_tickets_once(agent, base, tmp_path / 'shared', promoted, attempted) == 0
        assert (repo / 'module.py').stat().st_mtime_ns == after_failure_mtime
        assert not (repo / '.git/MERGE_HEAD').exists()
        # A changed main head unlocks one new attempt, still without repeated writes.
        git('commit', '--allow-empty', '-m', 'advance main')
        assert _promote_settled_superagent_tickets_once(agent, base, tmp_path / 'shared', promoted, attempted) == 0
        assert len(attempted) == 2
        # An uncommitted fix must unlock a retry without changing the worker HEAD.
        worker_head = git('rev-parse', 'HEAD', cwd=wt)
        (wt / 'module.py').write_text('value = 2\n')
        assert git('rev-parse', 'HEAD', cwd=wt) == worker_head
        assert _promote_settled_superagent_tickets_once(agent, base, tmp_path / 'shared', promoted, attempted) == 1
        assert promoted['ticket1'] == git('rev-parse', 'HEAD', cwd=wt)
        assert git('status', '--porcelain') == ''
    finally:
        server.shutdown(); server.server_close(); thread.join()


def test_watcher_strips_real_tracked_runtime_files_without_losing_code(tmp_path) -> None:
    import subprocess
    from synaptic_stream_60.watcher import _strip_runtime_diff, _extract_agent_diff
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=tmp_path, text=True).strip()
    git('init', '-b', 'main')
    git('config', 'user.name', 'Test'); git('config', 'user.email', 'test@example.com')
    (tmp_path / 'code.py').write_text('x = 1\n')
    git('add', '.'); git('-c', 'commit.gpgsign=false', 'commit', '-m', 'base')
    base = git('rev-parse', 'HEAD')
    for name in ('aw', 'AGENTS.md', '.venv/link', 'WORKSPACE_POLICY.md', 'feishu_docs/note', '.cache/dspy/file'):
        path = tmp_path / name
        path.parent.mkdir(exist_ok=True, parents=True)
        path.write_text('runtime\n')
    git('add', '-f', '.'); git('-c', 'commit.gpgsign=false', 'commit', '-m', 'helpers')
    assert _strip_runtime_diff(_extract_agent_diff(tmp_path, base)) == ''
    (tmp_path / 'code.py').write_text('x = 2\n')
    diff = _strip_runtime_diff(_extract_agent_diff(tmp_path, base))
    assert '+x = 2' in diff
    assert 'runtime' not in diff and 'AGENTS.md' not in diff and 'a/aw' not in diff


def test_instruction_scrubber_preserves_noninstruction_fields() -> None:
    from synaptic_stream_60.aw_multi_server import _scrub_instructions
    value = {'instruction': 'withheld', 'profile': {'email': 'person@example.com'},
             'tasks': [{'instruction': 'future', 'status': 'success', 'answer': 42}],
             'passwords': ['opaque-secret']}
    result = _scrub_instructions(value)
    assert result == {'instruction': '', 'profile': {'email': 'person@example.com'},
                      'tasks': [{'instruction': '', 'status': 'success', 'answer': 42}],
                      'passwords': ['opaque-secret']}
    assert value['instruction'] == 'withheld'


def test_live_appworld_task_rows_are_redacted_in_orm_and_sql(tmp_path) -> None:
    import subprocess
    from dataclasses import asdict
    from synaptic_stream_60.aw_multi_server import APPWORLD_VENV_PYTHON
    stream = load_streams()[0]
    account = next(a for a in stream.accounts if a.upstream_task_id == '37a8675_2')
    other = next(a for a in stream.accounts if a.account_id != account.account_id)
    script = '''
import json, sys
from pathlib import Path
from sqlalchemy import text
from sqlmodel import Session
from synaptic_stream_60.aw_multi_server import _MultiAccountAppWorldRuntime, APPWORLD_DATA_DIR
runtime = _MultiAccountAppWorldRuntime(APPWORLD_DATA_DIR)
accounts = json.loads(sys.argv[1])
try:
    runtime.start_stream("db_redaction_test", accounts, sys.argv[2])
    for aid in [accounts[0]["account_id"], accounts[1]["account_id"], accounts[0]["account_id"]]:
        world = runtime._activate_account_unlocked(aid)
        rows = world.models.supervisor.Task.all()
        assert rows and all(row.instruction == "" for row in rows)
        with Session(world.models.supervisor.Task.db.engine) as session:
            raw = session.execute(text("SELECT instruction FROM tasks")).all()
            assert raw and all(row[0] == "" for row in raw)
    print("fresh and reloaded ORM/SQL rows are blank")
finally:
    runtime.close_all()
'''
    proc = subprocess.run([str(APPWORLD_VENV_PYTHON), '-c', script,
                           json.dumps([asdict(account), asdict(other)]), str(tmp_path / 'snapshots')],
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert 'fresh and reloaded ORM/SQL rows are blank' in proc.stdout


def test_feishu_group_source_config_matches_upstream_schema(tmp_path) -> None:
    import ast
    import re
    from types import SimpleNamespace
    from dsh_runner import dsh_controller, sandbox_clis
    root = Path(__file__).resolve().parents[1]
    source = (root / 'third_party/dsh-superagent-prototype/src/sources/feishu.ts').read_text()
    deployment = (root / 'third_party/dsh-superagent-prototype/src/deployment.ts').read_text()
    # Pin the actual strict upstream contracts, so upstream schema drift fails here.
    assert "chatId: z.string().regex(/^oc_[a-zA-Z0-9]+$/)" in source
    assert "name: z.string().trim().max(60).optional(), listen: z.enum(['mentions', 'all']) }).strict()" in source
    schema = source.split('const FeishuConfigSchema = FeishuBotSchema.extend({', 1)[1].split('\n}).strict()', 1)[0]
    assert "export const FeishuBotSchema = z.object({ cli: z.string().trim().default('') })" in deployment
    allowed_keys = {'cli'} | set(re.findall(r'^  (\w+):', schema, re.M))
    assert allowed_keys == {'cli', 'readPageSize', 'linkTimeoutMs', 'chats', 'directMessages', 'otherChats'}
    channels = sorted({e.channel for s in load_streams() for e in s.events if e.channel.startswith('feishu://group/')})
    (tmp_path / 'sandbox_meta.json').write_text('{}')
    chats = sandbox_clis.configure_feishu_group_channels(tmp_path, channels)
    for slug, expected in {'ops_alerts': 'oc_opsalerts', 'personal_ops': 'oc_personalops',
                           'team_general': 'oc_teamgeneral', 'engineering': 'oc_engineering'}.items():
        assert sandbox_clis.feishu_chat_id('feishu://group/' + slug) == expected
    # Evaluate the actual source-save payload without booting a model runtime.
    tree = ast.parse(Path(dsh_controller.__file__).read_text())
    call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute) and n.func.attr == '_save_source'
                and isinstance(n.args[0], ast.Constant) and n.args[0].value == 'feishu')
    payload = eval(compile(ast.Expression(call.args[1]), '<feishu-source-config>', 'eval'),
                   {'self': SimpleNamespace(sandbox_paths={'lark_cli': '/local/lark'}), 'group_chats': chats})
    config = payload['config']
    assert set(config) <= allowed_keys
    assert isinstance(config['cli'], str) and config['directMessages'] is True
    assert config['otherChats'] in ('none', 'mentions')
    assert len({c['chatId'] for c in config['chats']}) == len(channels)
    for channel, chat in zip(channels, config['chats']):
        assert set(chat) == {'chatId', 'name', 'listen'}
        assert re.fullmatch(r'oc_[a-zA-Z0-9]+', chat['chatId'])
        assert chat['name'] == channel.removeprefix('feishu://group/')
        assert chat['name'] == chat['name'].strip() and len(chat['name']) <= 60
        assert chat['listen'] in ('mentions', 'all')


@pytest.mark.parametrize('reply', [
    'I will check and get back to you.', 'Let me check on that for you.', '我去查一下稍后回你。',
])
@pytest.mark.parametrize('binding', ['direct', 'parent'])
def test_causally_bound_peek_rejects_deferral(reply, binding) -> None:
    from synaptic_stream_60.runner import is_substantive_peek_reply
    kwargs = {'causally_bound': True} if binding == 'direct' else {
        'parent_id': 'peek', 'message_id': 'peek', 'require_parent_match': True}
    assert not is_substantive_peek_reply(reply, **kwargs)
    assert is_substantive_peek_reply('Two tasks finished; one is running. ' + reply, **kwargs)


@pytest.mark.parametrize('msg_tool_count,accepted', [(6, False), (7, False), (8, True)])
def test_driver_peek_delayed_presteer_turn(msg_tool_count, accepted) -> None:
    from synaptic_stream_60.runner import _DriverPeekQueue
    queue = _DriverPeekQueue()
    peek = queue.add(100, 0, True, tools_at_arrival=7)
    messages = ['All active tasks are finished.']
    queue.observe(messages, [msg_tool_count], 110)  # Ten seconds after arrival, not a timer heuristic.
    assert (peek['reply'] == messages[0]) is accepted
    if not accepted:
        assert not queue.claimed
        # A later turn can consume steer even without an intervening tool call.
        messages.append('Two tasks finished; one is running.')
        queue.observe(messages, [msg_tool_count, msg_tool_count], 111)
        assert peek['reply'] == messages[1] and queue.claimed == {1}


def _hygiene_test_driver(tmp_path):
    from clonact.paradigm_baselines import DSHParadigmDriver
    driver = DSHParadigmDriver('dsh_subagent_neutral', instance_dir=tmp_path / 'instance',
                              allowed_workspace_root=tmp_path, repo_dir=tmp_path / 'repo',
                              worktrees_dir=tmp_path / 'worktrees', shared_links_dir=tmp_path / 'shared',
                              agents_md='', has_code_repo=True)
    driver.primary_session_id = 'primary'
    return driver


def test_primary_reply_records_tool_boundary_with_and_without_tools(tmp_path) -> None:
    import asyncio
    driver = _hygiene_test_driver(tmp_path)
    async def run():
        await driver._on_session_event({'kind': 'assistant_message', 'session_id': 'primary', 'text': 'First'})
        await driver._on_session_event({'kind': 'tool_result', 'session_id': 'primary', 'tool_name': 'read'})
        await driver._on_session_event({'kind': 'assistant_message', 'session_id': 'primary', 'text': 'Second',
                                       'tool_calls': [{'name': 'read'}]})
        await driver._on_session_event({'kind': 'assistant_message', 'session_id': 'child', 'text': 'Child'})
    asyncio.run(run())
    assert driver.primary_assistant_messages == ['First', 'Second']
    assert driver.primary_assistant_message_tool_counts == [0, 1]


@pytest.mark.parametrize('busy,delta,dropped', [(False, 1, 0), (False, 0, 1), (True, 1, 1)])
def test_neutral_noise_direct_primary_mutations(tmp_path, busy, delta, dropped) -> None:
    driver = _hygiene_test_driver(tmp_path)
    driver._inbound_history = [
        {'event_id': 'noise', 'kind': 'external_noise', 'tokens': {'picnic'},
         'was_busy_at_arrival': busy, 'primary_mutations_before': 0},
        {'event_id': 'peek', 'kind': 'status_followup', 'primary_mutations_before': 0},
        {'event_id': 'task', 'kind': 'external_task_webhook', 'tokens': {'code'},
         'primary_mutations_before': delta},
    ]
    driver.primary_mutating_calls = 10  # Later task work cannot be attributed to the noise.
    result = driver.compute_hygiene_attribution()
    assert result['noise_dropped'] == dropped
    assert result['noise_mutated_workers'] == 1 - dropped
    assert result['noise_spawned_workers'] == 1 - dropped


def test_duplicate_worker_attribution_survives_noise_and_claims_once(tmp_path) -> None:
    from clonact.paradigm_baselines import _content_tokens
    driver = _hygiene_test_driver(tmp_path)
    prompt = 'Implement dynamic defaults and verify local unit tests'
    driver._inbound_history = [
        {'event_id': 'task', 'kind': 'external_task_webhook', 'tokens': _content_tokens(prompt), 'target_task_tag': 'feature'},
        {'event_id': 'dup1', 'kind': 'external_duplicate_webhook', 'target_task_tag': 'feature', 'workers_before': 1},
        {'event_id': 'dup2', 'kind': 'external_duplicate_webhook', 'target_task_tag': 'feature', 'workers_before': 1},
        {'event_id': 'noise', 'kind': 'external_noise', 'tokens': {'picnic'},
         'was_busy_at_arrival': True, 'primary_mutations_before': 0},
    ]
    driver._last_inbound_event_id = 'task'
    driver._record_spawned_worker('original', prompt)
    driver._last_inbound_event_id = 'noise'
    driver._record_spawned_worker('redundant', prompt)
    assert [w['worker_seq'] for w in driver.spawned_workers] == [0, 1]
    result = driver.compute_hygiene_attribution()
    assert result['duplicate_spawned_workers'] == 1
    assert result['duplicates_merged'] == 1
    # Redundant work predating a repost is not charged to that repost.
    driver._inbound_history = [driver._inbound_history[0],
        {'event_id': 'later', 'kind': 'external_duplicate_webhook', 'target_task_tag': 'feature', 'workers_before': 2}]
    result = driver.compute_hygiene_attribution()
    assert result['duplicate_spawned_workers'] == 0 and result['duplicates_merged'] == 1


@pytest.mark.parametrize('flag', [['--account', 'Alex Rivera'], ['-a', 'Alex Rivera'], ['--account=Alex Rivera']])
@pytest.mark.parametrize('command,args', [('apps', []), ('docs', ['gmail']), ('docs', ['gmail', 'search_emails'])])
def test_aw_discovery_trailing_account(monkeypatch, flag, command, args) -> None:
    import sys
    from synaptic_stream_60.aw_multi_server import build_stream_aw_cli_script
    env = {'__name__': 'cli_test'}
    exec(build_stream_aw_cli_script('http://localhost:1'), env)
    seen = []
    env['_post'] = lambda path, payload: seen.append((path, payload)) or {'ok': True}
    monkeypatch.setattr(sys, 'argv', ['aw', command, *args, *flag])
    env['main']()
    assert seen[0][0] == '/' + command
    assert seen[0][1]['account_id'] == 'Alex Rivera'
    if command == 'docs':
        assert seen[0][1]['app_name'] == args[0]
        assert seen[0][1]['api_name'] == (args[1] if len(args) > 1 else None)


@pytest.mark.parametrize('command', ['exec', 'answer', 'submit', 'submit_answer'])
@pytest.mark.parametrize('flag', [['--account', 'payload'], ['-a', 'payload'], ['--account=payload']])
def test_aw_payload_account_tokens_are_preserved(monkeypatch, command, flag) -> None:
    import sys
    from synaptic_stream_60.aw_multi_server import build_stream_aw_cli_script
    env = {'__name__': 'cli_test'}
    exec(build_stream_aw_cli_script('http://localhost:1'), env)
    seen = []
    env['_post'] = lambda path, payload: seen.append(payload) or {'ok': True}
    monkeypatch.setattr(sys, 'argv', ['aw', '--account', 'acct_01', command, *flag])
    env['main']()
    assert seen[0]['account_id'] == 'acct_01'
    assert seen[0]['code' if command == 'exec' else 'answer'] == ' '.join(flag)
