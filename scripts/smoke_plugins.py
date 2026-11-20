"""Start real DSH plugins and create an idle session; never submit a model prompt."""
from __future__ import annotations
import asyncio
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dsh_runner import synaptic_worker_controller as worker_module
from dsh_runner.synaptic_worker_controller import SynapticDSHWorkerHost
from dsh_runner.dsh_controller import DSH_BIN, SUPERAGENT_REPO
from dsh_runner.seatbelt import build_seatbelt_profile, repository_git_paths


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def check_profile(profile: Path, allowed: Path) -> None:
    protected = [ROOT / "synaptic_bench_150/verifiers.py",
                 ROOT / "appworld_micro_30/verifiers.py",
                 ROOT / "synaptic_bench_150/release.json",
                 ROOT / "appworld_micro_30/release.json",
                 ROOT / "clonact/context_mmu.py"]
    # These private evidence files are present in the research checkout only.
    protected += [p for p in [ROOT / "experiments/pa150_v201_three_arm_calibrated/results.jsonl",
                              ROOT / "archive/experiments/mvp/ab_evaluation_results.json",
                              ROOT / "synaptic_bench_150/data/episodes_pa150.jsonl",
                              ROOT / "appworld_micro_30/data/episodes_appworld_micro30.jsonl"] if p.is_file()]
    for path, expected in [(allowed, True), *((p, False) for p in protected)]:
        if not path.is_file():
            raise AssertionError(f"Missing isolation probe file: {path}")
        p = subprocess.run(["/usr/bin/sandbox-exec", "-f", str(profile), "/bin/cat", str(path)], capture_output=True)
        if (p.returncode == 0) != expected:
            raise AssertionError(f"Seatbelt access mismatch: {path.name}")
    print(f"PASS Seatbelt: workspace allowed; {len(protected)} existing protected files denied", flush=True)


def check_git_profile(profile: Path, workspace: Path) -> None:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(GIT_CONFIG_GLOBAL="/dev/null", GIT_CONFIG_NOSYSTEM="1")
    if repository_git_paths(ROOT):
        # Establish that the blob exists first: a missing HEAD is not isolation.
        target = "HEAD:synaptic_bench_150/verifiers.py"
        subprocess.run(["git", "-C", str(ROOT), "cat-file", "-e", target],
                       env=env, check=True, capture_output=True)
        probe = subprocess.run(["/usr/bin/sandbox-exec", "-f", str(profile),
                                "git", "-C", str(ROOT), "show", target],
                               env=env, capture_output=True)
        if probe.returncode == 0 or probe.stdout:
            raise AssertionError("Seatbelt exposed evaluator Git objects")
        print("PASS Seatbelt: evaluator Git blob denied", flush=True)
    else:
        print("SKIP source Git blob probe: source archive has no Git metadata", flush=True)

    with tempfile.TemporaryDirectory(prefix="git-probe-", dir=workspace) as tmp:
        task = Path(tmp) / "task"
        task.mkdir()
        cell = Path(tmp) / "cell"
        def git(cwd, *args):
            return subprocess.run(["/usr/bin/sandbox-exec", "-f", str(profile),
                                   "git", "-C", str(cwd), *args], env=env,
                                  capture_output=True, text=True, check=True).stdout
        git(task, "init", "-b", "main")
        git(task, "config", "user.name", "Smoke")
        git(task, "config", "user.email", "smoke@local")
        (task / "task.txt").write_text("seed\n")
        git(task, "add", "task.txt")
        git(task, "commit", "-m", "seed")
        git(task, "worktree", "add", "-b", "cell", str(cell))
        subprocess.run(["/usr/bin/sandbox-exec", "-f", str(profile), "/bin/sh", "-c",
                        'printf "changed\\n" > "$1/task.txt"', "probe", str(cell)],
                       env=env, capture_output=True, check=True)
        git(cell, "add", "task.txt")
        git(cell, "commit", "-m", "change")
        git(task, "merge", "--ff-only", "cell")
        assert git(task, "show", "HEAD:task.txt") == "changed\n"
        assert not git(task, "status", "--porcelain").strip()
    print("PASS Seatbelt: task Git init/worktree/commit/merge/show allowed", flush=True)


async def main():
    if sys.platform != "darwin" or not Path("/usr/bin/sandbox-exec").is_file():
        raise SystemExit("This isolation smoke check requires macOS sandbox-exec")
    (ROOT / "workspaces").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="plugin-smoke-", dir=ROOT / "workspaces") as tmp:
        temp = Path(tmp)
        credentials = temp / "empty.credentials.yaml"
        credentials.write_text("{}\n")
        worker_module.USER_CREDENTIALS = credentials
        workspace = temp / "workspace"
        workspace.mkdir()
        allowed = workspace / "allowed.txt"
        allowed.write_text("smoke\n")
        host = SynapticDSHWorkerHost(temp / "worker", allowed_workspace_root=temp)
        try:
            await host.start(use_seatbelt=True)
            await host.ensure_session(session_id="clonact-smoke", cell_id="smoke", cwd=workspace,
                                      title="No-inference smoke", read_only=True, allow_bash_in_alignment=False,
                                      trunk_prompt="Local handshake only.", mode_prompt="Do not execute tasks.")
            await host.get_session_status("clonact-smoke")
            check_profile(temp / "worker/dsh_seatbelt.sb", allowed)
            check_git_profile(temp / "worker/dsh_seatbelt.sb", workspace)
            print("PASS ClonAct worker health, idle session, and renamed-directory Seatbelt denial", flush=True)
        finally:
            await host.stop()
        # Load the unmodified baseline plugin without enabling sources or submitting tickets.
        home = temp / "baseline-home"
        home.mkdir()
        (home / ".credentials.yaml").write_text("{}\n")
        (home / ".credentials.yaml").chmod(0o600)
        env = {**os.environ, "DSH_HOME": str(home)}
        def cli(args):
            subprocess.run([str(DSH_BIN), *args], env=env, cwd=ROOT / "dsh_runner", check=True,
                           capture_output=True, timeout=45)
        cli(["smoke-baseline", "--from-default-profile", "web", "--dump-config"])
        cli(["plugin", "--profile", "smoke-baseline", "add", "-w", f"link:{SUPERAGENT_REPO}"])
        port = free_port()
        (home / "profiles/smoke-baseline/cordis.patch.yml").write_text(
            f"- id: webserver\n  config:\n    host: 127.0.0.1\n    port: {port}\n"
            f"- id: dsh-superagent\n  config:\n    stateDir: {home / 'state'}\n")
        profile = temp / "baseline.sb"
        profile.write_text(build_seatbelt_profile(ROOT, [temp, home]))
        with (temp / "baseline.log").open("w") as log:
            proc = subprocess.Popen(["/usr/bin/sandbox-exec", "-f", str(profile), str(DSH_BIN),
                                     "--profile", "smoke-baseline", "--no-open"], cwd=ROOT / "dsh_runner",
                                    env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                last_error = "no response"
                deadline = time.monotonic() + 45
                while time.monotonic() < deadline:
                    if proc.poll() is not None:
                        raise RuntimeError("Baseline DSH exited during startup:\n" + (temp / "baseline.log").read_text()[-4500:])
                    try:
                        req = urllib.request.Request(f"http://127.0.0.1:{port}/orchestrator/api/state",
                                                     headers={"x-orchestrator-request": "1"})
                        with urllib.request.urlopen(req, timeout=2) as response:
                            state = json.load(response)
                        if isinstance(state.get("health"), dict):
                            # Unconfigured is expected: no provider credentials or sources are enabled.
                            break
                    except (OSError, ValueError) as exc:
                        last_error = repr(exc)
                    await asyncio.sleep(0.25)
                else:
                    raise TimeoutError(f"Baseline plugin management endpoint did not become ready: {last_error}\n" + (temp / "baseline.log").read_text()[-6000:])
                check_profile(profile, allowed)
                check_git_profile(profile, workspace)
                print("PASS baseline plugin management handshake and Seatbelt denial; no tickets or model prompts submitted", flush=True)
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()


if __name__ == "__main__":
    asyncio.run(main())
