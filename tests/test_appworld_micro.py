"""Offline Verification & Anti-Vacuousness Test Suite for `AppWorld-Micro-30` (`v1.0.0`).

Covers:
1. Frozen 30-episode / 60-task manifest invariants (`difficulty == 1`, `2023-05-18T12:00:00`,
   byte-identical supervisor profiles, 100% app-disjoint pairs, noise diversity `<= 2`, zero prompt leakage).
2. Live HTTP `AppWorldMicroServerProcess` secret auth (`HTTP 403`), `./aw` CLI unprivileged execution,
   `0%` NO-OP pass rate, `100%` sequential `compiled_solution.py` oracle pass rate inside ONE persistent
   `AppWorld` across `M-1`, `M-2`, and `M-3`, read-only SQLite write-leak rejection, and bystander
   collateral app mutation rejection.
3. Kernel Seatbelt (`sandbox-exec`) oracle-file denial and `synaptic_pa/` paradigm purity.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import urllib.error
import urllib.request

import pytest

from appworld_micro_30.build_dataset import (
    APPWORLD_DATA_DIR,
    DEFAULT_EPISODES_PATH,
    USER_APP_NAMES,
    load_micro30_episodes,
)
from appworld_micro_30.env_server import (
    AppWorldMicroServerProcess,
    build_aw_cli_script,
    parse_concise_qa_answer,
)
from appworld_micro_30.runner import (
    make_micro30_tool_policy_guard,
    run_offline_oracle_smoke,
)
from appworld_micro_30.verifiers import (
    MicroRuntimeArtifacts,
    evaluate_micro_episode,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_appworld_micro30_manifest_invariants() -> None:
    assert DEFAULT_EPISODES_PATH.is_file(), "Frozen episodes_appworld_micro30.jsonl must exist"
    episodes = load_micro30_episodes()
    assert len(episodes) == 30

    by_track = Counter(ep.track_id for ep in episodes)
    assert by_track == {"M-1": 10, "M-2": 10, "M-3": 10}

    seen_tids: list[str] = []
    noise_messages: list[str] = []

    for ep in episodes:
        assert ep.datetime == "2023-05-18T12:00:00"
        assert len(ep.subtasks) == 2
        a, b = ep.subtasks
        seen_tids.extend([a.upstream_task_id, b.upstream_task_id])

        # 100% app-disjoint pairs
        assert set(a.target_apps).isdisjoint(set(b.target_apps)), f"Target app overlap in {ep.episode_id}"
        assert set(a.db_delta_apps).isdisjoint(set(b.db_delta_apps)), f"DB delta overlap in {ep.episode_id}"

        # Track modality contracts
        if ep.track_id == "M-1":
            assert a.is_qa is False and b.is_qa is False
        elif ep.track_id == "M-2":
            assert a.is_qa is True and b.is_qa is False
            assert any(v.kind == "sqlite_readonly_zero_mutation_check" for v in ep.verifiers)
        elif ep.track_id == "M-3":
            assert b.is_qa is True

        # Verify upstream task files (`difficulty == 1`, `datetime`, byte-identical `supervisor.jsonl`, empty `admin.jsonl`)
        sup_profiles: list[str] = []
        for st in (a, b):
            tdir = APPWORLD_DATA_DIR / "tasks" / st.upstream_task_id
            specs = json.loads((tdir / "specs.json").read_text(encoding="utf-8"))
            meta = json.loads((tdir / "ground_truth" / "metadata.json").read_text(encoding="utf-8"))
            assert meta["difficulty"] == 1
            assert specs["datetime"] == "2023-05-18T12:00:00"
            assert (tdir / "dbs" / "admin.jsonl").read_text(encoding="utf-8").strip() == ""
            sup_lines = [
                ln.strip()
                for ln in (tdir / "dbs" / "supervisor.jsonl").read_text(encoding="utf-8").splitlines()
                if ln.strip() and "INSERT INTO tasks " not in ln
            ]
            sup_profiles.append("\n".join(sup_lines))
        assert sup_profiles[0] == sup_profiles[1], f"Supervisor mismatch in {ep.episode_id}"

        # Prompt hygiene: zero API call syntax, zero ground-truth answer leakage, zero synthetic trivia
        for ev in ep.timeline:
            low = ev.text.lower()
            assert "apis." not in low
            assert "complete_task(" not in low
            assert "what is the workspace timezone" not in low
            if ev.kind == "external_noise":
                noise_messages.append(ev.text)

    # 60 strictly distinct upstream AppWorld tasks
    assert len(seen_tids) == 60
    assert len(set(seen_tids)) == 60

    # Diverse noise pool (`reuse <= 2`)
    noise_counts = Counter(noise_messages)
    assert len(noise_counts) >= 40
    assert max(noise_counts.values()) <= 2


@pytest.mark.integration
def test_appworld_micro_server_auth_and_e2e_oracle_and_negative_gates(tmp_path: Path) -> None:
    # 1. QA answer parser sanity
    assert parse_concise_qa_answer("FINAL_ANSWER: 48") == 48
    assert parse_concise_qa_answer("$42.50") == 42.5
    assert parse_concise_qa_answer("'Sample Track Title'") == "Sample Track Title"
    assert parse_concise_qa_answer("Yes.") == "yes"

    # 2. Run full offline oracle smoke across ALL 6 `dev`/`train` compiled-solution episodes (`M-1`, `M-2`, `M-3`)
    smoke_res = run_offline_oracle_smoke()
    assert len(smoke_res["tested_oracle_episodes"]) >= 6
    assert smoke_res["noop_tgc_pass_count"] == 0, "NO-OP must pass 0 subtasks (zero vacuousness)"
    ora_arm = smoke_res["oracle_summary"]["arms"]["compiled_oracle"]
    assert ora_arm["episode_pass_rate"] == 1.0, smoke_res
    assert ora_arm["subtask_tgc_pass_rate"] == 1.0, smoke_res
    assert ora_arm["tier1_orchestration_safety"]["readonly_zero_mutation_rate"] == 1.0

    # 3. Verify HTTP 403 secret protection, `./aw` CLI unprivileged usage, read-only leak gate, & bystander collateral gate
    m2_ep = load_micro30_episodes(episode_ids=["AW_M2_01"])[0]
    m1_ep = load_micro30_episodes(episode_ids=["AW_M1_01"])[0]
    srv = AppWorldMicroServerProcess(log_path=tmp_path / "aw_test_srv.log")
    try:
        srv.start_episode(m2_ep.episode_id, [asdict(s) for s in m2_ep.subtasks])

        # Unauthenticated call to `/snapshot` MUST fail with HTTP 403
        req_unauth = urllib.request.Request(srv.base_url + "/snapshot", method="GET")
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req_unauth, timeout=5)
        assert exc_info.value.code == 403

        # `./aw` script contains ZERO secret and works cleanly for `apps`, `docs`, `exec`, `answer`
        aw_script = build_aw_cli_script(srv.base_url)
        assert srv.secret not in aw_script
        aw_fp = tmp_path / "aw"
        aw_fp.write_text(aw_script, encoding="utf-8")
        aw_fp.chmod(0o755)

        apps_out = json.loads(
            subprocess.run([str(aw_fp), "apps"], check=True, capture_output=True, text=True).stdout
        )
        assert apps_out["ok"] is True and any(a["app_name"] == "spotify" for a in apps_out["apps"])

        # Negative Gate A: Mutating `simple_note` during M-2's read-only QA turn MUST fail `sqlite_readonly_zero_mutation_check`!
        bystander_code = (
            "pw = [x['password'] for x in apis.supervisor.show_account_passwords() if x['account_name']=='simple_note'][0]\n"
            f"tok = apis.simple_note.login(username={m2_ep.supervisor['email']!r}, password=pw)['access_token']\n"
            "apis.simple_note.create_note(access_token=tok, title='Leak', content='Unauthorized write')\n"
        )
        subprocess.run([str(aw_fp), "exec", bystander_code], check=True, capture_output=True, text=True)
        srv.save_checkpoint("ev_qa_gate")
        leak_rep = evaluate_micro_episode(
            m2_ep,
            srv,
            MicroRuntimeArtifacts(
                arm_name="leak_test",
                telemetry={
                    "noise_total": 2,
                    "noise_dropped": 2,
                    "noise_spawned_workers": 0,
                    "noise_mutated_workers": 0,
                    "dup_total": 1,
                    "dup_merged": 1,
                    "dup_spawned_new_workers": 0,
                    "real_tasks_dropped": 0,
                },
            ),
        )
        ro_ver = next(v for v in leak_rep["verifiers"] if v["kind"] == "sqlite_readonly_zero_mutation_check")
        assert ro_ver["passed"] is False
        assert "simple_note" in ro_ver["mutated_user_apps"]

        # Negative Gate B: On M-1 (`AW_M1_01`), solve BOTH target tasks via oracle, then mutate bystander app `simple_note` -> TGC MUST fail!
        srv.start_episode(m1_ep.episode_id, [asdict(s) for s in m1_ep.subtasks])
        assert srv.oracle_smoke()["ok"] is True
        bystander_m1 = (
            "pw = [x['password'] for x in apis.supervisor.show_account_passwords() if x['account_name']=='simple_note'][0]\n"
            f"tok = apis.simple_note.login(username={m1_ep.supervisor['email']!r}, password=pw)['access_token']\n"
            "apis.simple_note.create_note(access_token=tok, title='Collateral', content='Bystander mutation')\n"
        )
        subprocess.run([str(aw_fp), "exec", bystander_m1], check=True, capture_output=True, text=True)
        collat_rep = evaluate_micro_episode(
            m1_ep,
            srv,
            MicroRuntimeArtifacts(
                arm_name="collateral_test",
                telemetry={
                    "noise_total": 3,
                    "noise_dropped": 3,
                    "noise_spawned_workers": 0,
                    "noise_mutated_workers": 0,
                    "dup_total": 1,
                    "dup_merged": 1,
                    "dup_spawned_new_workers": 0,
                    "real_tasks_dropped": 0,
                },
            ),
        )
        assert collat_rep["episode_passed"] is False
        for v in collat_rep["verifiers"]:
            if v["kind"] == "appworld_tgc_check":
                assert v["passed"] is False
                assert "simple_note" in v["bystander_collateral_apps"]
    finally:
        srv.stop()


@pytest.mark.integration
def test_appworld_micro_seatbelt_and_synaptic_purity(tmp_path: Path) -> None:
    # 1. `synaptic_pa/` must stay 100% free of AppWorld-Micro benchmark identifiers
    for syn_fp in (PROJECT_ROOT / "clonact").glob("*.py"):
        src = syn_fp.read_text(encoding="utf-8")
        for forbidden in ("appworld", "aw_micro", "appworld_micro_30", "TestTracker"):
            assert forbidden not in src.lower(), f"Benchmark token {forbidden!r} leaked into {syn_fp.name}"

    # 2. Tool policy guard blocks held-out paths in `synaptic_pa`
    guard = make_micro30_tool_policy_guard(tmp_path)
    assert guard("bash", {"command": "./aw apps"}) is None
    assert guard("bash", {"command": "cat data/tasks/50e1ac9_1/ground_truth/evaluation.py"}) is not None

    # 3. Kernel `sandbox-exec` Seatbelt profile denies reading `data/tasks/.../ground_truth/evaluation.py`
    #    and `appworld_micro_30/verifiers.py`, while allowing episode workspace files!
    if sys.platform == "darwin" and Path("/usr/bin/sandbox-exec").is_file():
        def _esc(p: Path) -> str:
            return str(p.resolve()).replace("\\", "\\\\").replace('"', '\\"')

        ep_ws = tmp_path / "ep_workspace"
        ep_ws.mkdir()
        allowed_fp = ep_ws / "AGENTS.md"
        allowed_fp.write_text("# Allowed Workspace File\n", encoding="utf-8")

        sb_fp = tmp_path / "test_micro_seatbelt.sb"
        sb_fp.write_text(
            "\n".join(
                [
                    "(version 1)",
                    "(allow default)",
                    "(deny file-read* file-write*",
                    f'  (subpath "{_esc(PROJECT_ROOT / "data")}")',
                    f'  (subpath "{_esc(PROJECT_ROOT / "appworld_micro_30")}")',
                    f'  (subpath "{_esc(PROJECT_ROOT / "experiments")}")',
                    ")",
                    "(allow file-read* file-write*",
                    f'  (subpath "{_esc(ep_ws)}")',
                    ")",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        ok_proc = subprocess.run(
            ["/usr/bin/sandbox-exec", "-f", str(sb_fp), "/bin/cat", str(allowed_fp)],
            capture_output=True,
            text=True,
        )
        assert ok_proc.returncode == 0 and "Allowed Workspace File" in ok_proc.stdout

        sample_eval_fp = APPWORLD_DATA_DIR / "tasks" / "50e1ac9_1" / "ground_truth" / "evaluation.py"
        deny_eval = subprocess.run(
            ["/usr/bin/sandbox-exec", "-f", str(sb_fp), "/bin/cat", str(sample_eval_fp)],
            capture_output=True,
            text=True,
        )
        assert deny_eval.returncode != 0
        assert "Operation not permitted" in deny_eval.stderr

        deny_verifiers = subprocess.run(
            [
                "/usr/bin/sandbox-exec",
                "-f",
                str(sb_fp),
                "/bin/cat",
                str(PROJECT_ROOT / "appworld_micro_30" / "verifiers.py"),
            ],
            capture_output=True,
            text=True,
        )
        assert deny_verifiers.returncode != 0
        assert "Operation not permitted" in deny_verifiers.stderr
