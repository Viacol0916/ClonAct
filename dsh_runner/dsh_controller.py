"""Lifecycle & Management API controller for genuine `@deepseek-ai/dsh` + `dsh-superagent-prototype`."""
from __future__ import annotations

import glob
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from dsh_runner.sandbox_clis import (
    FEISHU_GROUP_CHAT_ID,
    configure_feishu_group_channels,
    FEISHU_USER_NAME,
    FEISHU_USER_OPEN_ID,
    REPO_FULL_NAME,
    init_dsh_sandbox,
)

from dsh_runner.seatbelt import build_seatbelt_profile

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DSH_BIN = PROJECT_ROOT / "dsh_runner" / "node_modules" / ".bin" / "dsh"
SUPERAGENT_REPO = PROJECT_ROOT / "third_party" / "dsh-superagent-prototype"
USER_CREDENTIALS = Path.home() / ".dsh" / ".credentials.yaml"
_SUPERAGENT_BOOT_SEM = threading.BoundedSemaphore(5)


def dsh_key(*values: Any) -> str:
    """Exact replica of `dsh-superagent-prototype/src/core/ids.ts:key(...values)`."""
    raw = json.dumps(list(values), separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:32]


def dsh_trigger_id_for_injected(injected: dict[str, Any] | None) -> str | None:
    """Compute exact deterministic DSH `Trigger.id` for an injected Feishu or GitHub event."""
    if not injected:
        return None
    ch = str(injected.get("channel") or "")
    if ch == "feishu" and injected.get("message_id"):
        return dsh_key("feishu", str(injected["message_id"]))
    if ch == "github_issue" and injected.get("issue_number") is not None:
        return dsh_key("github-issue", str(injected["issue_number"]))
    return None


def _zstd_bin() -> str:
    return shutil.which("zstd") or "/opt/homebrew/bin/zstd"


def build_dsh_llm_patch_yaml(target_model: str) -> str:
    """Build `cordis.patch.yml` entries mounting OpenAI or Qwen Token Plan routes & default model."""
    eff = (target_model or "deepseek-flash").strip() or "deepseek-flash"
    low = eff.lower()
    if low.startswith(("gpt-", "o1", "o3", "o4", "chatgpt-")):
        base_override = os.environ.get("OPENAI_BASE_URL", "").strip()
        base_line = f"        baseURL: {base_override}\n" if base_override else ""
        return (
            f"- id: agent-default-model\n"
            f"  config:\n"
            f"    provider: openai\n"
            f"    model: {eff}\n"
            f"- id: llm-pi-ai\n"
            f"  config:\n"
            f"    providers:\n"
            f"      openai:\n"
            f"{base_line}"
            f"        retryPolicy:\n"
            f"          mode: normal\n"
            f"          maxRetries: 10\n"
            f"          backoff:\n"
            f"            initialDelayMs: 1500\n"
            f"            maxDelayMs: 25000\n"
            f"            jitterRatio: 0.2\n"
            f"        models:\n"
            f"          - id: {eff}\n"
            f"            maxTokens: 32768\n"
        )
    if low.startswith("qwen"):
        base_override = os.environ.get("QWEN_BASE_URL", "").strip()
        base_line = f"        baseURL: {base_override}\n" if base_override else ""
        return (
            f"- id: agent-default-model\n"
            f"  config:\n"
            f"    provider: qwen-token-plan-cn\n"
            f"    model: {eff}\n"
            f"- id: llm-pi-ai\n"
            f"  config:\n"
            f"    providers:\n"
            f"      qwen-token-plan-cn:\n"
            f"{base_line}"
            f"        retryPolicy:\n"
            f"          mode: normal\n"
            f"          maxRetries: 12\n"
            f"          backoff:\n"
            f"            initialDelayMs: 2000\n"
            f"            maxDelayMs: 30000\n"
            f"            jitterRatio: 0.2\n"
            f"        models:\n"
            f"          - id: {eff}\n"
            f"            maxTokens: 32768\n"
            f"            reasoningEfforts:\n"
            f"              low: low\n"
            f"              medium: medium\n"
            f"              high: xhigh\n"
            f"              xhigh: xhigh\n"
        )
    return ""


def build_dsh_model_candidates(
    target_model: str,
    discovered_providers: list[str] | None = None,
) -> list[dict[str, str]]:
    """Build ordered `dsh` `{provider, model, reasoningEffort}` candidates without cross-family leakage."""
    eff = (target_model or "deepseek-flash").strip() or "deepseek-flash"
    low = eff.lower()
    if low.startswith(("gpt-", "o1", "o3", "o4", "chatgpt-")):
        return [
            {"provider": "openai", "model": eff, "reasoningEffort": "high"},
            {"provider": "openai", "model": eff},
        ]
    if low.startswith("qwen"):
        return [
            {"provider": "qwen-token-plan-cn", "model": eff, "reasoningEffort": "high"},
            {"provider": "qwen-token-plan-cn", "model": eff, "reasoningEffort": "xhigh"},
            {"provider": "qwen-token-plan-cn", "model": eff},
        ]
    prov_ids = list(discovered_providers or [])
    for fallback_p in ("deepseek-official", "deepseek"):
        if fallback_p not in prov_ids:
            prov_ids.append(fallback_p)
    prov_ids = [p for p in prov_ids if "deepseek" in p.lower()] or prov_ids
    candidates: list[dict[str, str]] = []
    for pid in prov_ids:
        candidates.append({"provider": pid, "model": eff, "reasoningEffort": "high"})
        candidates.append({"provider": pid, "model": eff})
    return candidates


class GenuineDSHSuperagent:
    """Manages an isolated, unmodified `dsh-superagent` runtime on real `@deepseek-ai/dsh`."""

    def __init__(
        self,
        instance_dir: Path,
        port: int = 13082,
        *,
        target_model: str | None = None,
    ) -> None:
        self.instance_dir = instance_dir.resolve()
        self.port = port
        self.target_model = (
            target_model
            or os.environ.get("PA_LLM_MODEL")
            or "deepseek-flash"
        ).strip() or "deepseek-flash"
        self.api_base = f"http://127.0.0.1:{port}/orchestrator/api"
        self.dsh_home = self.instance_dir / "dsh_home"
        self.state_dir = self.dsh_home / "state" / "superagent"
        self.world_dir = self.instance_dir / "sandbox_world"
        self.proc: subprocess.Popen[str] | None = None
        self.log_file = self.instance_dir / "dsh_superagent.log"
        self.sandbox_paths: dict[str, Path] = {}

    def _api(self, method: str, path: str, body: dict[str, Any] | None = None, timeout: float = 15.0) -> Any:
        url = f"{self.api_base}/{path.lstrip('/')}"
        headers = {"x-orchestrator-request": "1"}
        data_bytes = None
        if body is not None:
            headers["content-type"] = "application/json"
            data_bytes = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=data_bytes, headers=headers, method=method)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with opener.open(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {e.code} on {method} {path}: {err_body}") from e

    def wait_api_ready(self, require_configured: bool = False, timeout: float = 45.0) -> dict[str, Any]:
        deadline = time.perf_counter() + timeout
        last_err = ""
        while time.perf_counter() < deadline:
            if self.proc and self.proc.poll() is not None:
                log_tail = self.log_file.read_text(encoding="utf-8", errors="replace")[-2000:]
                raise RuntimeError(f"dsh exited unexpectedly (code={self.proc.returncode}):\n{log_tail}")
            try:
                st = self._api("GET", "state", timeout=3.0)
                rt_state = (st.get("health", {}).get("runtime") or {}).get("state")
                if not require_configured and rt_state in ("ready", "paused"):
                    return st
                if require_configured and rt_state == "ready":
                    setup = st.get("setup") or {}
                    if setup.get("problem") is None and setup.get("feishuProblem") is None and setup.get("accounts"):
                        return st
                last_err = f"rt_state={rt_state}, setup={st.get('setup')}"
            except Exception as e:
                last_err = str(e)
            time.sleep(0.35)
        raise TimeoutError(f"Timed out waiting for dsh-superagent API ready (last_err={last_err})")

    def _resolve_role_model_candidates(self) -> list[dict[str, str]]:
        prov_ids: list[str] = []
        try:
            cat = self._api("GET", "models", timeout=5.0)
            if isinstance(cat, dict):
                for key in ("groups", "providers", "items"):
                    for g in (cat.get(key) or []):
                        if isinstance(g, dict):
                            pid = str(g.get("id") or g.get("provider") or "").strip()
                            if pid and pid not in prov_ids:
                                prov_ids.append(pid)
        except Exception:
            pass
        return build_dsh_model_candidates(self.target_model, prov_ids)

    def bootstrap_and_start(
        self,
        extra_env: dict[str, str] | None = None,
        policy_instruction: str | None = None,
        agents_md_override: str | None = None,
        extra_repo_files: dict[str, tuple[str, int]] | None = None,
        seed_repo_dir: Path | None = None,
        shared_symlinks: dict[str, Path] | None = None,
        allowed_github_users: list[str] | None = None,
        allowed_feishu_senders: list[dict[str, str]] | None = None,
        allowed_feishu_bots: list[dict[str, str]] | None = None,
        repo_full_name: str | None = None,
        chat_name: str | None = None,
        *,
        max_workers: int = 4,
        extra_bin_files: dict[str, tuple[str, int]] | None = None,
        feishu_group_channels: list[str] | None = None,
    ) -> dict[str, Any]:
        if self.instance_dir.exists():
            shutil.rmtree(self.instance_dir)
        self.instance_dir.mkdir(parents=True, exist_ok=True)
        eff_repo = repo_full_name or ("acme/billing_service" if seed_repo_dir is None else REPO_FULL_NAME)
        eff_chat = chat_name or ("RD-Billing研发群" if seed_repo_dir is None else "Workspace协作群")
        self.sandbox_paths = init_dsh_sandbox(
            self.world_dir,
            agents_md_override=agents_md_override,
            extra_repo_files=extra_repo_files,
            seed_repo_dir=seed_repo_dir,
            repo_full_name=eff_repo,
            chat_name=eff_chat,
        )

        group_chats = (
            configure_feishu_group_channels(self.world_dir, feishu_group_channels)
            if feishu_group_channels is not None
            else [{"chat_id": FEISHU_GROUP_CHAT_ID, "name": eff_chat}]
        )
        # Also copy any executable helper files into world_dir/bin so bare command names work on PATH
        bin_dir = self.world_dir / "bin"
        bin_dir.mkdir(parents=True, exist_ok=True)
        for rel_name, (fcontent, fmode) in (extra_repo_files or {}).items():
            if "/" not in rel_name and (fmode & 0o111):
                bp = bin_dir / rel_name
                bp.write_text(fcontent, encoding="utf-8")
                bp.chmod(fmode)

        # Runtime-only helpers take precedence on PATH without entering the seed repo.
        for name, (content, mode) in (extra_bin_files or {}).items():
            if not name or Path(name).name != name or name in {".", ".."}:
                raise ValueError(f"extra_bin_files requires a plain filename: {name!r}")
            helper = bin_dir / name
            helper.write_text(content, encoding="utf-8")
            helper.chmod(mode)

        self.dsh_home.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        creds_dst = self.dsh_home / ".credentials.yaml"
        shutil.copyfile(USER_CREDENTIALS, creds_dst)
        creds_dst.chmod(0o600)

        from clonact.llm_client import load_project_dotenv

        load_project_dotenv()
        env = os.environ.copy()
        for k, v in (extra_env or {}).items():
            env[str(k)] = str(v)
        env["DSH_HOME"] = str(self.dsh_home)
        env["DSH_SANDBOX_REPO"] = eff_repo
        env["DSH_SANDBOX_CHAT_NAME"] = eff_chat
        venv_bin = str(PROJECT_ROOT / ".venv" / "bin")
        repo_py_bin = f"{Path(env['REPO_PYTHON']).parent}:" if env.get("REPO_PYTHON") else ""
        env["PATH"] = f"{bin_dir}:{repo_py_bin}{venv_bin}:{env.get('PATH', '')}"
        py_guard_dir = self.world_dir / "py_sandbox_guard"
        py_guard_dir.mkdir(parents=True, exist_ok=True)
        (py_guard_dir / "sitecustomize.py").write_text(
            "import os, socket as _socket, ssl as _ssl, sys as _sys\n"
            "def _is_gh(host):\n"
            "    h = (host.decode('utf-8', 'ignore') if isinstance(host, bytes) else str(host or '')).lower().rstrip('.')\n"
            "    return h == 'github.com' or h.endswith('.github.com') or h == 'githubusercontent.com' or h.endswith('.githubusercontent.com')\n"
            "_orig_gai = _socket.getaddrinfo\n"
            "def _guarded_gai(host, *args, **kwargs):\n"
            "    if _is_gh(host):\n"
            "        raise PermissionError('Error: Sandbox policy blocked outbound network access to upstream GitHub repositories/PRs/commits.')\n"
            "    return _orig_gai(host, *args, **kwargs)\n"
            "_socket.getaddrinfo = _guarded_gai\n"
            "_orig_wrap_socket = _ssl.SSLContext.wrap_socket\n"
            "def _guarded_wrap_socket(self, sock, *args, **kwargs):\n"
            "    if _is_gh(kwargs.get('server_hostname')):\n"
            "        raise PermissionError('Error: Sandbox policy blocked outbound network access to upstream GitHub repositories/PRs/commits.')\n"
            "    return _orig_wrap_socket(self, sock, *args, **kwargs)\n"
            "_ssl.SSLContext.wrap_socket = _guarded_wrap_socket\n"
            "_orig_wrap_bio = _ssl.SSLContext.wrap_bio\n"
            "def _guarded_wrap_bio(self, incoming, outgoing, server_side=False, server_hostname=None, *args, **kwargs):\n"
            "    if _is_gh(server_hostname):\n"
            "        raise PermissionError('Error: Sandbox policy blocked outbound network access to upstream GitHub repositories/PRs/commits.')\n"
            "    return _orig_wrap_bio(self, incoming, outgoing, server_side=server_side, server_hostname=server_hostname, *args, **kwargs)\n"
            "_ssl.SSLContext.wrap_bio = _guarded_wrap_bio\n"
            "_me = os.path.abspath(__file__)\n"
            "for _p in list(_sys.path):\n"
            "    _cand = os.path.abspath(os.path.join(_p, 'sitecustomize.py')) if _p else ''\n"
            "    if _cand and _cand != _me and os.path.isfile(_cand):\n"
            "        with open(_cand, 'rb') as _fh:\n"
            "            exec(compile(_fh.read(), _cand, 'exec'), globals())\n"
            "        break\n",
            encoding="utf-8",
        )
        existing_pypath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{py_guard_dir}:{existing_pypath}" if existing_pypath else str(py_guard_dir)
        gitconfig_sandbox = self.world_dir / "gitconfig_sandbox"
        gitconfig_sandbox.write_text(
            '[url "file:///nonexistent_sandbox_blocked_github/"]\n'
            "\tinsteadOf = https://github.com/\n"
            "\tinsteadOf = http://github.com/\n"
            "\tinsteadOf = git@github.com:\n"
            "\tinsteadOf = ssh://git@github.com/\n"
            "\tinsteadOf = https://api.github.com/\n"
            "\tinsteadOf = https://raw.githubusercontent.com/\n"
            "\tinsteadOf = https://patch-diff.githubusercontent.com/\n",
            encoding="utf-8",
        )
        for k in list(env):
            if k == "GIT_CONFIG_COUNT" or k.startswith(("GIT_CONFIG_KEY_", "GIT_CONFIG_VALUE_")):
                env.pop(k, None)
        env["GIT_CONFIG_GLOBAL"] = str(gitconfig_sandbox)
        env["GIT_TERMINAL_PROMPT"] = "0"

        # 1. Create `superagent` profile from DSH Web template & link `dsh-superagent-prototype`
        with _SUPERAGENT_BOOT_SEM:
            subprocess.run(
                [str(DSH_BIN), "superagent", "--from-default-profile", "web", "--dump-config"],
                cwd=str(PROJECT_ROOT / "dsh_runner"),
                env=env,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            subprocess.run(
                [str(DSH_BIN), "plugin", "--profile", "superagent", "add", "-w", f"link:{SUPERAGENT_REPO}"],
                cwd=str(PROJECT_ROOT / "dsh_runner"),
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )

            # 2. Write `cordis.patch.yml` enabling GitHub/Feishu sources and target LLM route on isolated port
            patch_yml = (
                f"- id: webserver\n"
                f"  config:\n"
                f"    host: 127.0.0.1\n"
                f"    port: {self.port}\n"
                f"- id: dsh-superagent\n"
                f"  config:\n"
                f"    stateDir: {self.state_dir}\n"
                f"- id: dsh-superagent-feishu\n"
                f"  disabled: false\n"
                + build_dsh_llm_patch_yaml(self.target_model)
            )
            (self.dsh_home / "profiles" / "superagent" / "cordis.patch.yml").write_text(patch_yml, encoding="utf-8")

            # 3. Boot `dsh --profile superagent --no-open` under kernel-enforced macOS Seatbelt isolation
            dsh_cmd = [str(DSH_BIN), "--profile", "superagent", "--no-open"]
            if sys.platform == "darwin" and Path("/usr/bin/sandbox-exec").is_file():
                sbpl_fp = self.instance_dir / "dsh_seatbelt.sb"
                sbpl_fp.write_text(build_seatbelt_profile(
                    PROJECT_ROOT, [self.instance_dir.parent.resolve(), self.instance_dir],
                ), encoding="utf-8")
                dsh_cmd = ["/usr/bin/sandbox-exec", "-f", str(sbpl_fp), *dsh_cmd]

            self._log_fp = self.log_file.open("w", encoding="utf-8")
            self.proc = subprocess.Popen(
                dsh_cmd,
                cwd=str(PROJECT_ROOT / "dsh_runner"),
                env=env,
                stdout=self._log_fp,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            self.wait_api_ready(require_configured=False, timeout=60.0)

        # 4. Save `source:github`, `source:feishu`, `policy`, `executor` via official Management API
        self._save_source(
            "github",
            {
                "enabled": True,
                "config": {
                    "repository": eff_repo,
                    "cli": str(self.sandbox_paths["gh_cli"]),
                    "observeIssues": True,
                    "observePulls": True,
                    "observeComments": True,
                    "observeCi": False,
                },
            },
        )
        self._save_source(
            "feishu",
            {
                "enabled": True,
                "config": {
                    "cli": str(self.sandbox_paths["lark_cli"]),
                    "chats": [{"chatId": c["chat_id"], "name": c["name"], "listen": "all"} for c in group_chats],
                    "directMessages": True,
                    "otherChats": "mentions",
                },
            },
        )
        gh_allowed_ids = ["viacol", "qa-chen"]
        for u in (allowed_github_users or []):
            u_clean = str(u).strip()
            if u_clean and u_clean not in gh_allowed_ids:
                gh_allowed_ids.append(u_clean)

        fs_allowed: list[dict[str, str]] = [{"id": FEISHU_USER_OPEN_ID, "name": FEISHU_USER_NAME}]
        seen_fs_ids = {FEISHU_USER_OPEN_ID}
        for item in (allowed_feishu_senders or []):
            fid = str(item.get("id") or "").strip()
            fname = str(item.get("name") or fid).strip()
            if fid and fid not in seen_fs_ids:
                seen_fs_ids.add(fid)
                fs_allowed.append({"id": fid, "name": fname})

        fs_bots: list[dict[str, str]] = []
        seen_bot_ids: set[str] = set()
        for bitem in (allowed_feishu_bots or []):
            bid = str(bitem.get("id") or "").strip()
            bname = str(bitem.get("name") or bid).strip()
            if bid and bid not in seen_bot_ids:
                seen_bot_ids.add(bid)
                fs_bots.append({"id": bid, "name": bname})

        self._save_block(
            "policy",
            {
                "rules": [
                    {
                        "id": "rule-default-accept",
                        "name": "Actionable user & ops tasks",
                        "enabled": True,
                        "instruction": policy_instruction
                        or "Accept actionable engineering issues, operational tasks, user requests, and follow-up updates; ignore casual off-topic chatter.",
                    }
                ],
                "sources": {
                    "github": {"allowed": [{"id": uid} for uid in gh_allowed_ids]},
                    "feishu": {
                        "allowed": fs_allowed,
                        "bots": fs_bots,
                        "syncChat": "",
                    },
                },
            },
        )

        clone_repo = self.sandbox_paths["clone_repo"]
        docs_dir = self.sandbox_paths["docs"]
        exclude_file = clone_repo / ".git" / "info" / "exclude"
        exclude_lines = {
            "feishu_docs", ".venv", ".venvs", "AGENTS.md", "aw", "WORKSPACE_POLICY.md", ".cache",
            ".pytest_cache/", "__pycache__/", "*.py[cod]", ".mypy_cache/", ".ruff_cache/", ".coverage", ".hypothesis/",
        }
        symlink_cmds = [f'ln -sfn "{docs_dir}" "$DSH_SUPERAGENT_DIR/feishu_docs"']
        for link_name, link_target in (shared_symlinks or {}).items():
            target_abs = Path(link_target).resolve()
            exclude_lines.add(link_name)
            # P2-2: Do not overwrite tracked regular files (`gaia_cli`, `pg_exec`) with symlinks in DSH worktrees
            symlink_cmds.append(
                f'[ -e "$DSH_SUPERAGENT_DIR/{link_name}" ] || ln -sfn "{target_abs}" "$DSH_SUPERAGENT_DIR/{link_name}"'
            )
            dst = clone_repo / link_name
            if not dst.exists() and not dst.is_symlink():
                dst.symlink_to(target_abs)
        if exclude_file.parent.exists():
            existing_ex = exclude_file.read_text(encoding="utf-8") if exclude_file.exists() else ""
            for ex_item in sorted(exclude_lines):
                if ex_item not in existing_ex.splitlines():
                    existing_ex += f"\n{ex_item}\n"
            exclude_file.write_text(existing_ex, encoding="utf-8")

        prepare_cmd = (
            f'git -C "{clone_repo}" fetch --no-tags origin '
            f'"${{DSH_SUPERAGENT_GITHUB_PR_HEAD:-HEAD}}:refs/dsh-superagent/$DSH_SUPERAGENT_TICKET" && '
            f'git -C "{clone_repo}" worktree add -b "agent/$DSH_SUPERAGENT_NAME" '
            f'"$DSH_SUPERAGENT_DIR" "refs/dsh-superagent/$DSH_SUPERAGENT_TICKET" && '
            + " && ".join(symlink_cmds)
        )

        selected_role_model: dict[str, str] | None = None
        executor_saved = False
        last_exec_err: Exception | None = None
        for rm_cand in self._resolve_role_model_candidates():
            try:
                self._save_block(
                    "executor",
                    {
                        "model": rm_cand,
                        "maxWorkers": int(max_workers),
                        "directory": str(self.state_dir / "work"),
                        "prepare": prepare_cmd,
                        "blockedCommands": ["curl", "wget"],
                    },
                )
                selected_role_model = rm_cand
                executor_saved = True
                break
            except Exception as exc:
                last_exec_err = exc
                continue
        if not executor_saved:
            if not self.target_model.lower().startswith("deepseek"):
                raise RuntimeError(
                    f"Failed to configure dsh-superagent executor model={self.target_model!r} "
                    f"(last_err={last_exec_err})"
                )
            self._save_block(
                "executor",
                {
                    "model": None,
                    "maxWorkers": int(max_workers),
                    "directory": str(self.state_dir / "work"),
                    "prepare": prepare_cmd,
                    "blockedCommands": ["curl", "wget"],
                },
            )

        self._save_block(
            "eval",
            {
                "mode": "auto",
                "timeoutSeconds": 120,
                "model": selected_role_model,
            },
        )

        # Wait until live Feishu event stream consumer also reports `ready`
        deadline = time.perf_counter() + 30.0
        st = self.wait_api_ready(require_configured=True, timeout=30.0)
        while time.perf_counter() < deadline:
            st = self._api("GET", "state")
            if (st.get("health", {}).get("feishu") or {}).get("state") == "ready":
                return st
            time.sleep(0.3)
        return st

    def _save_source(self, source_id: str, value: dict[str, Any]) -> None:
        settings = self._api("GET", "settings")
        src = next(s for s in settings["sources"] if s["id"] == source_id)
        res = self._api("PUT", f"settings/source:{source_id}", {"value": value, "revision": src["revision"]})
        if res.get("restarting"):
            time.sleep(0.8)
            self.wait_api_ready(require_configured=(source_id == "feishu" or source_id == "github"), timeout=30.0)

    def _save_block(self, block_name: str, value: dict[str, Any]) -> None:
        settings = self._api("GET", "settings")
        rev = settings["blocks"][block_name]["revision"]
        res = self._api("PUT", f"settings/{block_name}", {"value": value, "revision": rev})
        if res.get("restarting"):
            time.sleep(0.8)
            self.wait_api_ready(require_configured=True, timeout=30.0)

    def get_tickets(self, state_filter: str = "open") -> list[dict[str, Any]]:
        res = self._api("GET", f"tickets?state={state_filter}&limit=50")
        return res.get("tickets") or res.get("items") or []

    def _read_sqlite_snapshot(self) -> dict[str, Any]:
        """Read immutable SQLite tables from `orchestrator.sqlite` (includes source-filtered triggers with no Thread)."""
        db_path = self.state_dir / "orchestrator.sqlite"
        out: dict[str, Any] = {
            "triggers": {},
            "dispositions": {},
            "decisions": {},
            "tickets": {},
            "active_tickets": 0,
            "active_threads": 0,
        }
        if not db_path.exists():
            return out
        uri = f"file:{urllib.parse.quote(str(db_path))}?mode=ro"
        try:
            conn = sqlite3.connect(uri, uri=True, timeout=5.0)
            try:
                for tid, val in conn.execute("SELECT id, value FROM trigger").fetchall():
                    out["triggers"][str(tid)] = json.loads(val)
                for tid, val in conn.execute("SELECT id, value FROM disposition").fetchall():
                    out["dispositions"][str(tid)] = json.loads(val)
                for did, val in conn.execute("SELECT id, value FROM decision").fetchall():
                    out["decisions"][str(did)] = json.loads(val)
                for tkid, val in conn.execute("SELECT id, value FROM ticket").fetchall():
                    out["tickets"][str(tkid)] = json.loads(val)
                row_tk = conn.execute("SELECT COUNT(*) FROM ticket WHERE active = 1").fetchone()
                row_th = conn.execute("SELECT COUNT(*) FROM thread WHERE active = 1").fetchone()
                out["active_tickets"] = int(row_tk[0] or 0) if row_tk else 0
                out["active_threads"] = int(row_th[0] or 0) if row_th else 0
            finally:
                conn.close()
        except Exception:
            pass
        return out

    def get_all_tickets_detailed(self) -> list[dict[str, Any]]:
        seen: dict[str, dict[str, Any]] = {}
        for flt in ("open", "ended"):
            for t in self.get_tickets(flt):
                tid = t.get("id")
                if tid and tid not in seen:
                    seen[tid] = self._api("GET", f"tickets/{tid}")
        sql_snap = self._read_sqlite_snapshot()
        for tid, raw_tk in (sql_snap.get("tickets") or {}).items():
            if tid in seen:
                if "inputs" not in seen[tid] and isinstance(raw_tk.get("inputs"), list):
                    seen[tid]["inputs"] = raw_tk["inputs"]
                if "origin" not in seen[tid] and isinstance(raw_tk.get("origin"), dict):
                    seen[tid]["origin"] = raw_tk["origin"]
            else:
                seen[tid] = raw_tk
        return list(seen.values())

    def get_decisions(self) -> list[dict[str, Any]]:
        res = self._api("GET", "decisions?limit=100")
        return res.get("decisions") or res.get("items") or []

    def get_triggers(self, limit: int = 200) -> list[dict[str, Any]]:
        """Collect `TriggerView` records across all DSH threads plus SQLite fallback for source-filtered triggers."""
        threads_res = self._api("GET", f"threads?filter=all&limit={limit}")
        threads = threads_res.get("items") or threads_res.get("threads") or []
        by_id: dict[str, dict[str, Any]] = {}
        for th in threads:
            key = str(th.get("key") or "").strip()
            if not key:
                continue
            enc_key = urllib.parse.quote(key, safe="")
            try:
                detail = self._api("GET", f"threads/{enc_key}?limit=100")
            except Exception:
                continue
            for tr in detail.get("timeline") or []:
                tid = str(tr.get("id") or "")
                if tid and tid not in by_id:
                    by_id[tid] = tr

        sql_snap = self._read_sqlite_snapshot()
        sql_decisions = sql_snap.get("decisions") or {}
        decisions_by_trig: dict[str, dict[str, Any]] = {}
        for dec in sql_decisions.values():
            for trig_id in dec.get("triggerIds") or []:
                decisions_by_trig[str(trig_id)] = dec

        for tid, raw_tr in (sql_snap.get("triggers") or {}).items():
            disp = (sql_snap.get("dispositions") or {}).get(tid) or {}
            dec = sql_decisions.get(str(disp.get("decisionId") or "")) or decisions_by_trig.get(tid)
            dec_view = (
                {
                    "id": dec.get("id"),
                    "outcome": dec.get("kind"),
                    "reason": dec.get("reason") or [],
                    "ticketId": dec.get("ticketId"),
                }
                if dec
                else None
            )
            if tid not in by_id:
                by_id[tid] = {
                    "id": tid,
                    "source": raw_tr.get("source"),
                    "actor": raw_tr.get("actor") or {},
                    "human": bool(raw_tr.get("human", True)),
                    "at": raw_tr.get("at") or "",
                    "text": raw_tr.get("text") or "",
                    "preview": (raw_tr.get("text") or "")[:120],
                    "threads": [
                        f"{ref.get('source')}:{ref.get('id')}"
                        for ref in (raw_tr.get("threads") or [])
                        if isinstance(ref, dict)
                    ],
                    "disposition": {
                        "filtered": disp.get("filtered"),
                        "fanout": disp.get("fanout") or [],
                        "handbacks": disp.get("handbacks") or [],
                        "acked": disp.get("acked") or [],
                        "decision": dec_view,
                    },
                }
            else:
                # Backfill raw text or filtered disposition if HTTP view truncated/omitted it
                if raw_tr.get("text") and not by_id[tid].get("raw_text"):
                    by_id[tid]["raw_text"] = raw_tr["text"]
                cur_disp = by_id[tid].setdefault("disposition", {})
                if disp.get("filtered") and not cur_disp.get("filtered"):
                    cur_disp["filtered"] = disp["filtered"]
                if dec_view and not cur_disp.get("decision"):
                    cur_disp["decision"] = dec_view

        ordered = list(by_id.values())
        ordered.sort(key=lambda r: (str(r.get("at") or ""), str(r.get("id") or "")))
        return ordered

    def wait_until_quiescent(
        self,
        expected_received: int = 1,
        expected_trigger_ids: set[str] | None = None,
        min_wait_s: float = 3.5,
        idle_streak_s: float = 4.5,
        timeout_s: float = 210.0,
    ) -> dict[str, Any]:
        """Wait until orchestrator & all workers have ingested expected triggers and settled.

        Note: `notice` effects in DSH (`effects.ts:20`) intentionally inject into the main Session
        WITHOUT waking it, so `health['notice'].detail` remains non-empty while idle; excluding `notice`
        eliminates false-positive timeouts while still checking all active work effects & SQLite active sets.
        """
        t_wait0 = time.perf_counter()
        time.sleep(min_wait_s)
        deadline = t_wait0 + timeout_s
        idle_since: float | None = None
        last_state: dict[str, Any] = {}
        ov: dict[str, Any] = {}
        proj_dir = self.dsh_home / "storages" / "session_projcache" / "sessions"
        diag: dict[str, Any] = {}

        while time.perf_counter() < deadline:
            if self.proc is not None and self.proc.poll() is not None:
                return {
                    "state": last_state,
                    "overview": ov,
                    "timed_out": True,
                    "process_exited": True,
                    "elapsed_s": round(time.perf_counter() - t_wait0, 2),
                    **diag,
                }
            try:
                last_state = self._api("GET", "state")
                ov = self._api("GET", "overview")
            except Exception:
                time.sleep(0.8)
                continue

            flow = ov.get("flow") or {}
            ex = ov.get("executor") or {}
            received = int(flow.get("received") or 0)
            evaluating = int(flow.get("evaluating") or 0)
            running = int(ex.get("running") or 0)
            queued = int(ex.get("queued") or 0)
            health = last_state.get("health") or {}
            # Only active execution/routing effects wake sessions (`notice` is passive until next wakeup)
            busy_effect_keys = [
                k
                for k in ("assessment", "launch", "delivery", "relay", "reply")
                if (health.get(k) or {}).get("detail", "Nothing waiting") != "Nothing waiting"
            ]
            effects_busy = bool(busy_effect_keys)

            open_turn = False
            for p in glob.glob(str(proj_dir / "*.json")):
                try:
                    d = json.loads(Path(p).read_text(encoding="utf-8"))
                    tb = (((d.get("record") or {}).get("rows") or {}).get("turnBoundary") or {}).get("val") or {}
                    if tb.get("openTurnStartSeq") is not None:
                        open_turn = True
                        break
                except Exception:
                    pass

            sql_snap = self._read_sqlite_snapshot()
            known_trig_ids = set((sql_snap.get("triggers") or {}).keys()) | set(
                (sql_snap.get("dispositions") or {}).keys()
            )
            missing_trigs = sorted((expected_trigger_ids or set()) - known_trig_ids)
            active_tickets = int(sql_snap.get("active_tickets") or 0)
            active_threads = int(sql_snap.get("active_threads") or 0)
            raw_open_tks = [
                tk for tk in (sql_snap.get("tickets") or {}).values() if isinstance(tk, dict) and not tk.get("ended")
            ]
            any_undelivered_sql = any(len(tk.get("undelivered") or []) > 0 for tk in raw_open_tks)
            parked_person = sum(
                1
                for p in (ov.get("person") or [])
                if isinstance(p, dict)
                and p.get("stage") == "person"
                and p.get("action") != "delivery"
                and int(p.get("undelivered") or 0) == 0
            )
            idle_tickets = int(ex.get("idle") or 0)
            working_tickets = max(0, active_tickets - parked_person - idle_tickets)

            ingested_ok = (len(missing_trigs) == 0) if expected_trigger_ids else (received >= expected_received)
            diag = {
                "received": received,
                "evaluating_threads": evaluating,
                "running_workers": running,
                "queued_workers": queued,
                "active_tickets": active_tickets,
                "parked_person_tickets": parked_person,
                "idle_tickets": idle_tickets,
                "working_tickets": working_tickets,
                "any_undelivered_sql": any_undelivered_sql,
                "active_threads": active_threads,
                "busy_effects": busy_effect_keys,
                "open_turn": open_turn,
                "missing_trigger_ids": missing_trigs,
            }

            if (
                ingested_ok
                and evaluating == 0
                and running == 0
                and queued == 0
                and working_tickets == 0
                and not any_undelivered_sql
                and active_threads == 0
                and not effects_busy
                and not open_turn
            ):
                now_pc = time.perf_counter()
                if idle_since is None:
                    idle_since = now_pc
                elif now_pc - idle_since >= idle_streak_s:
                    return {
                        "state": last_state,
                        "overview": ov,
                        "timed_out": False,
                        "elapsed_s": round(now_pc - t_wait0, 2),
                        **diag,
                    }
            else:
                idle_since = None
            time.sleep(0.8)

        try:
            ov = self._api("GET", "overview")
        except Exception:
            pass
        return {
            "state": last_state,
            "overview": ov,
            "timed_out": True,
            "elapsed_s": round(time.perf_counter() - t_wait0, 2),
            **diag,
        }

    def _iter_decompressed_sessions(self, session_ids: set[str] | None = None) -> list[tuple[str, list[dict[str, Any]]]]:
        """Decompress persisted `session.v4.jsonl.zstd` logs for worker inspection."""
        out: list[tuple[str, list[dict[str, Any]]]] = []
        pattern = str(self.dsh_home / "sessions" / "**" / "session.v4.jsonl.zstd")
        zbin = _zstd_bin()
        for zpath in sorted(glob.glob(pattern, recursive=True)):
            sid = Path(zpath).parent.name
            if session_ids and sid not in session_ids:
                continue
            proc = subprocess.run(
                [zbin, "-dc", zpath],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            if proc.returncode != 0:
                continue
            events: list[dict[str, Any]] = []
            for line in proc.stdout.splitlines():
                try:
                    events.append(json.loads(line))
                except Exception:
                    continue
            out.append((sid, events))
        return out

    def inspect_worker_pytest_env_failures(self) -> list[dict[str, Any]]:
        """Decompress persisted worker `session.v4.jsonl.zstd` logs to count `BILLING_TZ=UTC` cold-start errors."""
        results: list[dict[str, Any]] = []
        for sid, events in self._iter_decompressed_sessions():
            env_err_count = 0
            bash_pytest_calls = 0
            for ev in events:
                if ev.get("type") == "assistant/message":
                    txt = json.dumps(ev.get("data") or {}, ensure_ascii=False)
                    if "pytest" in txt:
                        bash_pytest_calls += 1
                elif ev.get("type") == "tool/result":
                    txt = json.dumps(ev.get("data") or {}, ensure_ascii=False)
                    if "Missing required env var BILLING_TZ=UTC" in txt:
                        env_err_count += 1
            results.append(
                {
                    "session_id": sid,
                    "pytest_mentions": bash_pytest_calls,
                    "billing_tz_env_errors": env_err_count,
                }
            )
        return results

    def inspect_worker_office_reads(self) -> bool:
        """Check whether any DSH worker session actually executed a tool inspecting office mounts/DB/addendum."""
        markers = (
            "office_fs",
            "office_pg",
            "pg_exec",
            "gaia_cli",
            "FILESYSTEM_TEST_DIR",
            "release_spec_addendum.md",
        )
        for _sid, events in self._iter_decompressed_sessions():
            saw_tool_result = False
            saw_office_cmd = False
            for ev in events:
                etype = ev.get("type")
                if etype in ("assistant/message", "tool/call", "tool/use"):
                    txt = json.dumps(ev.get("data") or {}, ensure_ascii=False)
                    if any(m in txt for m in markers):
                        saw_office_cmd = True
                elif etype == "tool/result":
                    saw_tool_result = True
                    txt = json.dumps(ev.get("data") or {}, ensure_ascii=False)
                    if any(m in txt for m in markers) or "SPEC-REF-" in txt:
                        return True
            if saw_office_cmd and saw_tool_result:
                return True
        return False

    def collect_dsh_session_telemetry(self) -> dict[str, Any]:
        """Read exact token & turn accounting directly from DSH's `session_projcache`."""
        proj_dir = self.dsh_home / "storages" / "session_projcache" / "sessions"
        sessions: list[dict[str, Any]] = []
        tot_uncached = 0
        tot_cache_read = 0
        tot_cache_write = 0
        tot_out = 0
        tot_turns = 0
        for p in sorted(glob.glob(str(proj_dir / "*.json"))):
            try:
                data = json.loads(Path(p).read_text(encoding="utf-8"))
                rows = (data.get("record") or {}).get("rows") or {}
                usage = ((rows.get("tokenUsage") or {}).get("val") or {}).get("totals") or {}
                tb = (rows.get("turnBoundary") or {}).get("val") or {}
                title = ((rows.get("title") or {}).get("val")) or Path(p).stem
                u_in = int(usage.get("uncachedInputTokens") or 0)
                c_rd = int(usage.get("cacheReadTokens") or 0)
                c_wr = int(usage.get("cacheWriteTokens") or 0)
                out_t = int(usage.get("outputTokens") or 0)
                turns = int(tb.get("lastTurn") or 0)
                tot_uncached += u_in
                tot_cache_read += c_rd
                tot_cache_write += c_wr
                tot_out += out_t
                tot_turns += turns
                sessions.append(
                    {
                        "session_file": Path(p).name,
                        "title": title,
                        "turns": turns,
                        "uncached_input_tokens": u_in,
                        "cache_read_tokens": c_rd,
                        "cache_write_tokens": c_wr,
                        "output_tokens": out_t,
                    }
                )
            except Exception:
                continue
        total_prompt = tot_uncached + tot_cache_read + tot_cache_write
        return {
            "session_count": len(sessions),
            "total_turns": tot_turns,
            "total_prompt_tokens": total_prompt,
            "uncached_input_tokens": tot_uncached,
            "cache_read_tokens": tot_cache_read,
            "cache_write_tokens": tot_cache_write,
            "output_tokens": tot_out,
            "prefix_cache_hit_rate": round(tot_cache_read / total_prompt, 4) if total_prompt else 0.0,
            "sessions": sessions,
        }

    def count_worker_tool_calls(self, session_ids: set[str] | None = None) -> int:
        """Count cumulative real worker `tool/result` executions across target DSH worker sessions."""
        proj_dir = self.dsh_home / "storages" / "session_projcache" / "sessions"
        allowed_sids: set[str] = set()
        for p in glob.glob(str(proj_dir / "*.json")):
            try:
                stem = Path(p).stem
                if session_ids and stem not in session_ids:
                    continue
                data = json.loads(Path(p).read_text(encoding="utf-8"))
                rows = (data.get("record") or {}).get("rows") or {}
                title = str(((rows.get("title") or {}).get("val")) or "")
                if any(tag in title.lower() for tag in ("policy", "assessment", "eval", "main")):
                    continue
                allowed_sids.add(stem)
            except Exception:
                continue

        target_filter = allowed_sids if allowed_sids else session_ids
        tool_results = 0
        decompressed = self._iter_decompressed_sessions(session_ids=target_filter)
        for _sid, events in decompressed:
            for ev in events:
                if ev.get("type") == "tool/result":
                    tool_results += 1
        if decompressed:
            return tool_results

        # Fallback only before first `.zstd` flush if session_projcache already recorded worker turns
        fallback_turns = 0
        for sid in allowed_sids:
            p = proj_dir / f"{sid}.json"
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                tb = (((data.get("record") or {}).get("rows") or {}).get("turnBoundary") or {}).get("val") or {}
                fallback_turns += int(tb.get("lastTurn") or 0)
            except Exception:
                continue
        return fallback_turns

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            import signal
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
            except Exception:
                self.proc.terminate()
            try:
                self.proc.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except Exception:
                    self.proc.kill()
                try:
                    self.proc.wait(timeout=5.0)
                except Exception:
                    pass
        self.proc = None
        log_fp = getattr(self, "_log_fp", None)
        if log_fp is not None:
            try:
                log_fp.close()
            except Exception:
                pass
            self._log_fp = None
