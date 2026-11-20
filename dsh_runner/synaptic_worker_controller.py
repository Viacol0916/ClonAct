"""Orchestrator-free Native DSH Worker Host (`SynapticDSHWorkerHost`).

Boots a sandboxed `@deepseek-ai/dsh` process loaded ONLY with `synaptic-dsh-worker-bridge`
(zero orchestrator router, zero background polling loops) so `Synaptic-PA` `ContextCell`s
can plug in the genuine `dsh-agent` worker loop while keeping `ContextMMU`, `Shared Trunk T_0`,
`peek_peer_cell`, lazy git worktree isolation, and 3-phase `HybridSynapticBus` intact.
"""
from __future__ import annotations

import asyncio
import glob
import inspect
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from dsh_runner.dsh_controller import (
    DSH_BIN,
    PROJECT_ROOT,
    USER_CREDENTIALS,
    _zstd_bin,
    build_dsh_llm_patch_yaml,
    build_dsh_model_candidates,
)

from dsh_runner.seatbelt import build_seatbelt_profile

PLUGIN_DIR = Path(__file__).resolve().parent / "synaptic_worker_plugin"
_BOOT_THROTTLE_SEM = threading.BoundedSemaphore(6)
_DEFAULT_SANDBOX_FORBIDDEN_SUBSTRINGS = [
    "synaptic_bench_150",
    "appworld_micro_30",
    "synaptic_stream_60",
    "streams_60.jsonl",
    "aw_snapshots",
    "upstream_pillars",
    "eval_spec",
    "gold_patch",
]


def _pick_free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class SynapticDSHWorkerHost:
    """Manages one sandboxed DSH worker runtime and local `asyncio` callback server per episode/session."""

    def __init__(
        self,
        instance_dir: Path,
        *,
        allowed_workspace_root: Path | None = None,
        port: int = 0,
        target_model: str | None = None,
    ) -> None:
        self.instance_dir = instance_dir.resolve()
        self.allowed_workspace_root = (allowed_workspace_root or instance_dir.parent).resolve()
        self.port = port or _pick_free_loopback_port()
        self.target_model = (
            target_model
            or os.environ.get("PA_LLM_MODEL")
            or os.environ.get("DSH_EXECUTOR_MODEL")
            or "deepseek-flash"
        ).strip() or "deepseek-flash"
        self.dsh_home = self.instance_dir / "dsh_home"
        self.log_file = self.instance_dir / "dsh_worker_host.log"
        self.proc: subprocess.Popen[str] | None = None
        self._log_fp: Any | None = None

        self._callback_server: asyncio.AbstractServer | None = None
        self.callback_url: str = ""
        self._peek_handler: Callable[[str, str, str], str] | None = None
        self._event_handler: Callable[[dict[str, Any]], Awaitable[None]] | None = None
        self._custom_tool_handler: Callable[[str, str, str, dict[str, Any]], Awaitable[str] | str] | None = None
        self._model_candidates: list[dict[str, str]] = []
        self._started = False

    def set_callbacks(
        self,
        *,
        peek_handler: Callable[[str, str, str], str] | None = None,
        event_handler: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
        custom_tool_handler: Callable[[str, str, str, dict[str, Any]], Awaitable[str] | str] | None = None,
    ) -> None:
        self._peek_handler = peek_handler
        self._event_handler = event_handler
        self._custom_tool_handler = custom_tool_handler

    async def _start_callback_server(self) -> None:
        if self._callback_server is not None:
            return

        async def _handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            try:
                header_bytes = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=15.0)
                header_text = header_bytes.decode("latin-1", errors="replace")
                lines = header_text.split("\r\n")
                req_line = lines[0].split(" ")
                path = req_line[1] if len(req_line) >= 2 else "/"
                headers: dict[str, str] = {}
                for ln in lines[1:]:
                    if ":" in ln:
                        k, v = ln.split(":", 1)
                        headers[k.strip().lower()] = v.strip()

                body_raw = b""
                if headers.get("transfer-encoding", "").lower() == "chunked":
                    chunks: list[bytes] = []
                    while True:
                        sz_line = await asyncio.wait_for(reader.readuntil(b"\r\n"), timeout=15.0)
                        sz = int(sz_line.strip().split(b";", 1)[0], 16)
                        if sz == 0:
                            await reader.readuntil(b"\r\n")
                            break
                        chunks.append(await asyncio.wait_for(reader.readexactly(sz), timeout=15.0))
                        await reader.readexactly(2)
                    body_raw = b"".join(chunks)
                else:
                    clen = int(headers.get("content-length") or 0)
                    if clen > 0:
                        body_raw = await asyncio.wait_for(reader.readexactly(clen), timeout=15.0)

                payload = json.loads(body_raw.decode("utf-8", errors="replace")) if body_raw else {}
                resp_obj: dict[str, Any] = {"ok": True}
                status_code = 200

                if path == "/peek_peer_cell":
                    caller_cid = str(payload.get("caller_cell_id") or "")
                    target_cid = str(payload.get("cell_id") or "ALL")
                    rel_path = str(payload.get("rel_path") or "").strip()
                    if self._peek_handler is not None:
                        slice_txt = self._peek_handler(caller_cid, target_cid, rel_path)
                    else:
                        slice_txt = "No peer lookup callback configured."
                    resp_obj = {"ok": True, "slice": slice_txt}
                elif path == "/session_event":
                    if self._event_handler is not None:
                        await self._event_handler(payload)
                    resp_obj = {"ok": True}
                elif path == "/custom_tool":
                    caller_cid = str(payload.get("caller_cell_id") or "")
                    sid = str(payload.get("session_id") or "")
                    t_name = str(payload.get("tool_name") or "")
                    t_args = payload.get("args") if isinstance(payload.get("args"), dict) else {}
                    if self._custom_tool_handler is not None:
                        out = self._custom_tool_handler(caller_cid, sid, t_name, t_args)
                        if inspect.isawaitable(out):
                            out = await out
                        res_txt = str(out or "")
                    else:
                        res_txt = f"Error: custom tool handler not configured for {t_name}."
                    resp_obj = {"ok": True, "result": res_txt}
                else:
                    status_code = 404
                    resp_obj = {"error": f"Unknown callback path: {path}"}

                resp_bytes = json.dumps(resp_obj, ensure_ascii=False).encode("utf-8")
                reason = "OK" if status_code == 200 else "Not Found"
                head_out = (
                    f"HTTP/1.1 {status_code} {reason}\r\n"
                    f"Content-Type: application/json; charset=utf-8\r\n"
                    f"Content-Length: {len(resp_bytes)}\r\n"
                    f"Connection: close\r\n\r\n"
                ).encode("ascii")
                writer.write(head_out + resp_bytes)
                await writer.drain()
            except Exception as exc:
                try:
                    err_bytes = json.dumps({"error": str(exc)}).encode("utf-8")
                    writer.write(
                        f"HTTP/1.1 500 Internal Server Error\r\n"
                        f"Content-Type: application/json\r\n"
                        f"Content-Length: {len(err_bytes)}\r\n"
                        f"Connection: close\r\n\r\n".encode("ascii")
                        + err_bytes
                    )
                    await writer.drain()
                except Exception:
                    pass
            finally:
                try:
                    writer.close()
                    await writer.wait_closed()
                except Exception:
                    pass

        self._callback_server = await asyncio.start_server(_handle_client, host="127.0.0.1", port=0)
        sock = self._callback_server.sockets[0]
        cb_port = int(sock.getsockname()[1])
        self.callback_url = f"http://127.0.0.1:{cb_port}"

    def _api_sync(
        self,
        method: str,
        route: str,
        payload: dict[str, Any] | None = None,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        url = f"http://127.0.0.1:{self.port}/synaptic-worker/api/{route.lstrip('/')}"
        headers = {
            "Host": f"127.0.0.1:{self.port}",
            "X-Synaptic-Worker": "1",
        }
        body: bytes | None = None
        if payload is not None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json; charset=utf-8"
        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as err:
            detail = err.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"DSH worker bridge HTTP {err.code} on {method} {route}: {detail}") from err

    async def api(
        self,
        method: str,
        route: str,
        payload: dict[str, Any] | None = None,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        return await asyncio.to_thread(self._api_sync, method, route, payload, timeout)

    def _resolve_model_candidates_sync(self) -> list[dict[str, str]]:
        prov_ids: list[str] = []
        try:
            cat = self._api_sync("GET", "models", timeout=15.0)
            if isinstance(cat, dict):
                for rp in (cat.get("routableProviders") or []):
                    rp_s = str(rp or "").strip()
                    if rp_s and rp_s not in prov_ids:
                        prov_ids.append(rp_s)
                for key in ("groups", "providers", "items"):
                    for p in (cat.get(key) or []):
                        if isinstance(p, dict):
                            pid = str(p.get("id") or p.get("provider") or "").strip()
                            if pid and pid not in prov_ids:
                                prov_ids.append(pid)
        except Exception:
            pass
        return build_dsh_model_candidates(self.target_model, prov_ids)

    def _bootstrap_process_sync(
        self,
        *,
        extra_env: dict[str, str] | None = None,
        use_seatbelt: bool = True,
    ) -> None:
        if self.instance_dir.exists():
            shutil.rmtree(self.instance_dir)
        self.instance_dir.mkdir(parents=True, exist_ok=True)
        self.dsh_home.mkdir(parents=True, exist_ok=True)

        if not USER_CREDENTIALS.is_file():
            raise FileNotFoundError(f"DSH credentials file missing at {USER_CREDENTIALS}")
        creds_dst = self.dsh_home / ".credentials.yaml"
        shutil.copyfile(USER_CREDENTIALS, creds_dst)
        creds_dst.chmod(0o600)

        from clonact.llm_client import load_project_dotenv

        load_project_dotenv()
        env = os.environ.copy()
        for k, v in (extra_env or {}).items():
            env[str(k)] = str(v)
        env["DSH_HOME"] = str(self.dsh_home)
        env.setdefault("GIT_AUTHOR_NAME", "Synaptic Worker")
        env.setdefault("GIT_AUTHOR_EMAIL", "worker@local.workspace")
        env.setdefault("GIT_COMMITTER_NAME", "Synaptic Worker")
        env.setdefault("GIT_COMMITTER_EMAIL", "worker@local.workspace")

        real_git = shutil.which("git") or "/usr/bin/git"
        real_wget = shutil.which("wget") or "/opt/homebrew/bin/wget"
        guard_bin_dir = self.instance_dir / "bin"
        guard_bin_dir.mkdir(parents=True, exist_ok=True)

        curl_shim = guard_bin_dir / "curl"
        curl_shim.write_text(
            "#!/bin/sh\n"
            'case "$*" in\n'
            "  *github.com*|*GITHUB.COM*|*githubusercontent.com*|*GITHUBUSERCONTENT.COM*)\n"
            "    echo 'Error: Sandbox policy blocked outbound network access to upstream GitHub repositories.' >&2\n"
            "    exit 1\n"
            "    ;;\n"
            "esac\n"
            'exec /usr/bin/curl "$@"\n',
            encoding="utf-8",
        )
        curl_shim.chmod(0o755)

        wget_shim = guard_bin_dir / "wget"
        wget_shim.write_text(
            "#!/bin/sh\n"
            'case "$*" in\n'
            "  *github.com*|*GITHUB.COM*|*githubusercontent.com*|*GITHUBUSERCONTENT.COM*)\n"
            "    echo 'Error: Sandbox policy blocked outbound network access to upstream GitHub repositories.' >&2\n"
            "    exit 1\n"
            "    ;;\n"
            "esac\n"
            f'if [ -x "{real_wget}" ]; then exec "{real_wget}" "$@"; fi\n'
            "echo 'wget: command not found' >&2\n"
            "exit 127\n",
            encoding="utf-8",
        )
        wget_shim.chmod(0o755)

        gitconfig_sandbox = self.instance_dir / "gitconfig_sandbox"
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

        git_shim = guard_bin_dir / "git"
        git_shim.write_text(
            "#!/bin/sh\n"
            "_has_net_op=0\n"
            "_has_gh_url=0\n"
            'for _arg in "$@"; do\n'
            '  case "$_arg" in\n'
            "    clone|fetch|pull|ls-remote|push) _has_net_op=1 ;;\n"
            "  esac\n"
            '  case "$_arg" in\n'
            "    *github.com*|*GITHUB.COM*|*githubusercontent.com*|*GITHUBUSERCONTENT.COM*) _has_gh_url=1 ;;\n"
            "  esac\n"
            "done\n"
            'if [ "$_has_net_op" = "1" ] && [ "$_has_gh_url" = "1" ]; then\n'
            "  echo 'Error: Sandbox policy blocked outbound network access to upstream GitHub repositories.' >&2\n"
            "  exit 1\n"
            "fi\n"
            "export GIT_TERMINAL_PROMPT=0\n"
            f'export GIT_CONFIG_GLOBAL="{gitconfig_sandbox}"\n'
            f'exec "{real_git}" "$@"\n',
            encoding="utf-8",
        )
        git_shim.chmod(0o755)

        venv_bin = str(PROJECT_ROOT / ".venv" / "bin")
        repo_py_bin = f"{Path(env['REPO_PYTHON']).parent}:" if env.get("REPO_PYTHON") else ""
        env["PATH"] = f"{guard_bin_dir}:{repo_py_bin}{venv_bin}:{env.get('PATH', '')}"

        py_guard_dir = self.instance_dir / "py_sandbox_guard"
        py_guard_dir.mkdir(parents=True, exist_ok=True)
        (py_guard_dir / "sitecustomize.py").write_text(
            "import os, socket as _socket, ssl as _ssl, sys as _sys\n"
            "def _is_gh(host):\n"
            "    h = (host.decode('utf-8', 'ignore') if isinstance(host, bytes) else str(host or '')).lower().rstrip('.')\n"
            "    return h == 'github.com' or h.endswith('.github.com') or h == 'githubusercontent.com' or h.endswith('.githubusercontent.com')\n"
            "_orig_gai = _socket.getaddrinfo\n"
            "def _guarded_gai(host, *args, **kwargs):\n"
            "    if _is_gh(host):\n"
            "        raise PermissionError('Error: Sandbox policy blocked outbound network access to upstream GitHub repositories.')\n"
            "    return _orig_gai(host, *args, **kwargs)\n"
            "_socket.getaddrinfo = _guarded_gai\n"
            "_orig_wrap_socket = _ssl.SSLContext.wrap_socket\n"
            "def _guarded_wrap_socket(self, sock, *args, **kwargs):\n"
            "    if _is_gh(kwargs.get('server_hostname')):\n"
            "        raise PermissionError('Error: Sandbox policy blocked outbound network access to upstream GitHub repositories.')\n"
            "    return _orig_wrap_socket(self, sock, *args, **kwargs)\n"
            "_ssl.SSLContext.wrap_socket = _guarded_wrap_socket\n"
            "_orig_wrap_bio = _ssl.SSLContext.wrap_bio\n"
            "def _guarded_wrap_bio(self, incoming, outgoing, server_side=False, server_hostname=None, *args, **kwargs):\n"
            "    if _is_gh(server_hostname):\n"
            "        raise PermissionError('Error: Sandbox policy blocked outbound network access to upstream GitHub repositories.')\n"
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

        for k in list(env):
            if k == "GIT_CONFIG_COUNT" or k.startswith(("GIT_CONFIG_KEY_", "GIT_CONFIG_VALUE_")):
                env.pop(k, None)
        env["GIT_CONFIG_GLOBAL"] = str(gitconfig_sandbox)
        env["GIT_TERMINAL_PROMPT"] = "0"

        subprocess.run(
            [str(DSH_BIN), "synaptic-worker", "--from-default-profile", "web", "--dump-config"],
            cwd=str(PROJECT_ROOT / "dsh_runner"),
            env=env,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        subprocess.run(
            [str(DSH_BIN), "plugin", "--profile", "synaptic-worker", "add", "-w", f"link:{PLUGIN_DIR}"],
            cwd=str(PROJECT_ROOT / "dsh_runner"),
            env=env,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        patch_yml = (
            f"- id: webserver\n"
            f"  config:\n"
            f"    host: 127.0.0.1\n"
            f"    port: {self.port}\n"
            + build_dsh_llm_patch_yaml(self.target_model)
        )
        (self.dsh_home / "profiles" / "synaptic-worker" / "cordis.patch.yml").write_text(
            patch_yml,
            encoding="utf-8",
        )

        dsh_cmd = [str(DSH_BIN), "--profile", "synaptic-worker", "--no-open"]
        if use_seatbelt and sys.platform == "darwin" and Path("/usr/bin/sandbox-exec").is_file():
            sbpl_fp = self.instance_dir / "dsh_seatbelt.sb"
            sbpl_fp.write_text(build_seatbelt_profile(
                PROJECT_ROOT, [self.allowed_workspace_root, self.instance_dir],
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

        deadline = time.perf_counter() + 120.0
        last_err: Exception | None = None
        while time.perf_counter() < deadline:
            if self.proc.poll() is not None:
                log_tail = self.log_file.read_text(encoding="utf-8", errors="replace")[-2500:]
                raise RuntimeError(f"SynapticDSHWorkerHost exited prematurely (rc={self.proc.returncode}):\n{log_tail}")
            try:
                st = self._api_sync("GET", "health", timeout=2.5)
                if st.get("ok"):
                    self._model_candidates = self._resolve_model_candidates_sync()
                    return
            except Exception as exc:
                last_err = exc
            time.sleep(0.35)
        log_tail = self.log_file.read_text(encoding="utf-8", errors="replace")[-2500:] if self.log_file.exists() else ""
        raise TimeoutError(f"SynapticDSHWorkerHost health check timed out ({last_err}):\n{log_tail}")

    def _throttled_bootstrap_sync(
        self,
        *,
        extra_env: dict[str, str] | None = None,
        use_seatbelt: bool = True,
    ) -> None:
        with _BOOT_THROTTLE_SEM:
            self._bootstrap_process_sync(extra_env=extra_env, use_seatbelt=use_seatbelt)

    async def start(
        self,
        *,
        extra_env: dict[str, str] | None = None,
        use_seatbelt: bool = True,
    ) -> None:
        if self._started:
            return
        await self._start_callback_server()
        await asyncio.to_thread(
            self._throttled_bootstrap_sync,
            extra_env=extra_env,
            use_seatbelt=use_seatbelt,
        )
        self._started = True

    async def ensure_session(
        self,
        *,
        session_id: str,
        cell_id: str,
        cwd: Path,
        title: str,
        read_only: bool,
        allow_bash_in_alignment: bool,
        trunk_prompt: str,
        mode_prompt: str,
        forbidden_substrings: list[str] | None = None,
        enable_peek_peer_cell: bool = True,
        deny_tools: list[str] | None = None,
        custom_tools: list[dict[str, Any]] | None = None,
        allow_subagent_custom_tools: bool = False,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "sessionId": session_id,
            "cellId": cell_id,
            "cwd": str(cwd.resolve()),
            "title": title,
            "readOnly": read_only,
            "allowBashInAlignment": allow_bash_in_alignment,
            "enablePeekPeerCell": enable_peek_peer_cell,
            "allowSubagentCustomTools": allow_subagent_custom_tools,
            "trunkPrompt": trunk_prompt,
            "modePrompt": mode_prompt,
            "callbackUrl": self.callback_url,
            "modelCandidates": self._model_candidates,
            "forbiddenSubstrings": (
                forbidden_substrings
                if forbidden_substrings is not None
                else list(_DEFAULT_SANDBOX_FORBIDDEN_SUBSTRINGS)
            ),
        }
        if deny_tools is not None:
            payload["denyTools"] = list(deny_tools)
        if custom_tools is not None:
            payload["customTools"] = list(custom_tools)
        return await self.api(
            "POST",
            "session/ensure",
            payload,
            timeout=30.0,
        )

    async def get_session_status(self, session_id: str) -> dict[str, Any]:
        qs = urllib.parse.urlencode({"sessionId": session_id})
        return await self.api("GET", f"session/status?{qs}", None, timeout=10.0)

    async def send_message(
        self,
        *,
        session_id: str,
        request_id: str,
        text: str,
        mode: str | dict[str, str] = "followup",
    ) -> dict[str, Any]:
        return await self.api(
            "POST",
            "session/send",
            {
                "sessionId": session_id,
                "requestId": request_id,
                "text": text,
                "mode": mode,
            },
            timeout=30.0,
        )

    async def wait_quiet(
        self,
        *,
        session_id: str,
        timeout_sec: float = 600.0,
    ) -> dict[str, Any]:
        timeout_ms = int(max(5.0, timeout_sec) * 1000)
        return await self.api(
            "POST",
            "session/wait_quiet",
            {"sessionId": session_id, "timeoutMs": timeout_ms},
            timeout=timeout_sec + 30.0,
        )

    async def send_and_wait_quiet(
        self,
        *,
        session_id: str,
        request_id: str,
        text: str,
        mode: str | dict[str, str] = "followup",
        timeout_sec: float = 600.0,
    ) -> dict[str, Any]:
        await self.send_message(
            session_id=session_id,
            request_id=request_id,
            text=text,
            mode=mode,
        )
        return await self.wait_quiet(session_id=session_id, timeout_sec=timeout_sec)

    async def cancel_session(
        self,
        session_id: str,
        *,
        reason: str = "cancelled",
        keep_inbox: bool = False,
    ) -> dict[str, Any]:
        return await self.api(
            "POST",
            "session/cancel",
            {"sessionId": session_id, "reason": reason, "keepInbox": keep_inbox},
            timeout=20.0,
        )

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
                data = json.loads(Path(p).read_text(encoding="utf-8", errors="replace"))
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

    def _iter_decompressed_sessions(self, session_ids: set[str] | None = None) -> list[tuple[str, list[dict[str, Any]]]]:
        """Decompress persisted `session.v4.jsonl.zstd` logs for contamination & tool verification."""
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

    async def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            try:
                await self.api("POST", "shutdown", {}, timeout=8.0)
            except Exception:
                pass
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
            except Exception:
                try:
                    self.proc.terminate()
                except Exception:
                    pass
            try:
                await asyncio.to_thread(self.proc.wait, 6.0)
            except Exception:
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except Exception:
                    try:
                        self.proc.kill()
                    except Exception:
                        pass
        self.proc = None
        if self._log_fp is not None:
            try:
                self._log_fp.close()
            except Exception:
                pass
            self._log_fp = None
        if self._callback_server is not None:
            self._callback_server.close()
            try:
                await self._callback_server.wait_closed()
            except Exception:
                pass
            self._callback_server = None
        self._started = False
