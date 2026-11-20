"""Unified multi-pillar workspace & environment materializer for `Synaptic-Bench-150`.

Materializes isolated per-episode environments across all 4 official 2025-2026 pillars:
  1. `Gaia2` (Meta ARE `Gaia2EnvServer` + `./gaia_cli`)
  2. `MCPMark` Filesystem (`storage.mcpmark.ai/filesystem/*.zip` + `office_fs`)
  3. `MCPMark` PostgreSQL (Docker `pgvector/pgvector:pg16` + template DBs + `./pg_exec`)
  4. `CooperBench`, `SWE-Interact`, `Ambig-SWE` (cached Git repos at `base_commit` + cached `.venv`)

Exposes identical CLI/filesystem/env interfaces to both `Synaptic-PA` (Arm A) and
`GenuineDSHSuperagent` (Arm B), plus deterministic cryptographic state snapshots.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

import psycopg2
from psycopg2 import sql

from .schema import EpisodeSpec, WorkspaceMountSpec

_THREAD_CACHE_LOCK = threading.RLock()
_PG_ADMIN_LOCK = threading.RLock()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
UPSTREAM_ROOT = PROJECT_ROOT / "data" / "upstream_pillars"


class _InterprocessCacheLock:
    """Thread + OS `flock` guard so concurrent threads/processes never race on bare-repo or `.venv` cache builds."""

    def __init__(self) -> None:
        self._local = threading.local()

    def __enter__(self) -> "_InterprocessCacheLock":
        _THREAD_CACHE_LOCK.acquire()
        depth = getattr(self._local, "depth", 0)
        if depth == 0:
            UPSTREAM_ROOT.mkdir(parents=True, exist_ok=True)
            lock_file = UPSTREAM_ROOT / ".pa150_cache_init.lock"
            fh = lock_file.open("a+", encoding="utf-8")
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            self._local.fh = fh
        self._local.depth = depth + 1
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        depth = getattr(self._local, "depth", 1) - 1
        self._local.depth = depth
        try:
            if depth <= 0:
                fh = getattr(self._local, "fh", None)
                self._local.fh = None
                if fh is not None:
                    try:
                        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
                    finally:
                        fh.close()
        finally:
            _THREAD_CACHE_LOCK.release()


_CACHE_INIT_LOCK = _InterprocessCacheLock()
FS_TEMPLATE_CACHE = UPSTREAM_ROOT / "mcpmark_fs_templates"
PG_BACKUP_CACHE = UPSTREAM_ROOT / "mcpmark_pg_backups"
REPO_BARE_CACHE = UPSTREAM_ROOT / "repo_cache"
VENV_CACHE = UPSTREAM_ROOT / "venv_cache"

PG_CONTAINER_NAME = "pa150-pg"
PG_DEFAULT_HOST = "127.0.0.1"
PG_DEFAULT_PORT = 15432
PG_DEFAULT_USER = "postgres"
PG_DEFAULT_PASSWORD = "postgres"
MCPMARK_TEMPLATE_DBS = {"employees", "chinook", "dvdrental", "sports", "lego"}

# NEW-11: MCPMark PostgreSQL tasks that mutate cluster-global `pg_roles` (`CREATE/DROP USER/ROLE`)
PG_CLUSTER_ROLE_TASK_IDS = frozenset(
    {
        "basic_security_setup",
        "database_security_policies",
        "rls_business_access",
        "user_permission_audit",
    }
)


def resolve_project_path(raw: str | Path) -> Path:
    """Resolve a relative `` path or absolute path portably (`NEW-8`)."""
    p = Path(raw)
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


_BENCHMARK_HELD_OUT_PATH_RE = re.compile(
    r"(?:upstream_pillars|mcpmark[/\\]tasks|gaia2_raw_scenarios|gaia2_samples|"
    r"CooperBench[/\\]dataset|sweinteract_python_pool|ambig_swe_joined_pool|"
    r"cooperbench_hf_pool|gaia2_compact_pool|solution\.patch|tests\.patch|"
    r"feature\.patch|oracle_actions|experiments[/\\]pa150_)",
    flags=re.IGNORECASE,
)
_BENCHMARK_OUTBOUND_GITHUB_RE = re.compile(
    r"(?:https?://(?:www\.)?github\.com/[^/\s\"']+/[A-Za-z0-9_.-]+/(?:pull|issues|commit|commits|compare|raw|blob|tree)\b|"
    r"https?://api\.github\.com/repos/|https?://raw\.githubusercontent\.com/|https?://patch-diff\.githubusercontent\.com/)",
    flags=re.IGNORECASE,
)
_BENCHMARK_NETWORK_CLIENT_RE = re.compile(
    r"\b(?:curl|wget|gh|http|httpie|nc|ncat|ssh|scp|urllib|requests|httpx|aiohttp|socket|webbrowser|"
    r"git\s+(?:clone|fetch|pull|ls-remote|remote))\b",
    flags=re.IGNORECASE,
)
_SCRIPT_TOKEN_RE = re.compile(r"(?:^|\s)([A-Za-z0-9_./-]+\.(?:py|sh))\b")


def text_violates_benchmark_policy(text: str) -> str | None:
    """Check if executed command/script text touches held-out eval data or performs outbound GitHub fetches."""
    if not text:
        return None
    if _BENCHMARK_HELD_OUT_PATH_RE.search(text):
        return "Error: Sandbox policy blocked access to held-out benchmark evaluation data outside the workspace."
    if _BENCHMARK_OUTBOUND_GITHUB_RE.search(text) and _BENCHMARK_NETWORK_CLIENT_RE.search(text):
        return "Error: Sandbox policy blocked outbound network access to upstream GitHub repositories/PRs/commits."
    return None


def make_benchmark_tool_policy_guard(ws_root: Path | None = None) -> Callable[[str, dict[str, Any], Path], str | None]:
    """Build a benchmark-layer sandbox guard that blocks held-out evaluation files and outbound GitHub PR/commit fetches."""
    ws_resolved = ws_root.resolve() if ws_root is not None else None
    upstream_resolved = UPSTREAM_ROOT.resolve()
    venv_cache_resolved = VENV_CACHE.resolve()
    experiments_resolved = (PROJECT_ROOT / "experiments").resolve()
    bench_pkg_resolved = (PROJECT_ROOT / "synaptic_bench_150").resolve()

    def _guard(tool_name: str, args: dict[str, Any], cwd: Path) -> str | None:
        if tool_name == "bash":
            cmd = str(args.get("command") or "")
            direct_err = text_violates_benchmark_policy(cmd)
            if direct_err:
                return direct_err
            # Also inspect any newly created/modified local `.py`/`.sh` script invoked on the command line
            # (untouched upstream repo files are never blocked even if they contain docstrings with GitHub URLs).
            for m in _SCRIPT_TOKEN_RE.finditer(cmd):
                rel_tok = m.group(1)
                try:
                    cand = (cwd / rel_tok).resolve()
                    if not cand.is_file() or cand.stat().st_size > 65536:
                        continue
                    st_txt = cand.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue
                script_err = text_violates_benchmark_policy(st_txt)
                if script_err:
                    git_st = subprocess.run(
                        ["git", "-C", str(cwd), "status", "--porcelain", "--", str(cand)],
                        capture_output=True,
                        text=True, encoding="utf-8", errors="replace",
                    )
                    if git_st.returncode != 0 or git_st.stdout.strip():
                        return script_err
            return None

        if tool_name in ("read_file", "write_file", "edit_file", "read_repo_file", "edit_repo_file"):
            raw_path = str(args.get("rel_path") or args.get("path") or "")
            if _BENCHMARK_HELD_OUT_PATH_RE.search(raw_path):
                return "Error: Sandbox policy blocked access to held-out benchmark evaluation data outside the workspace."
            try:
                resolved = (cwd / raw_path).resolve()
            except Exception:
                return None
            if ws_resolved is not None and (resolved == ws_resolved or ws_resolved in resolved.parents):
                return None
            if resolved == venv_cache_resolved or venv_cache_resolved in resolved.parents:
                return None
            if (
                resolved == upstream_resolved
                or upstream_resolved in resolved.parents
                or resolved == experiments_resolved
                or experiments_resolved in resolved.parents
                or resolved == bench_pkg_resolved
                or bench_pkg_resolved in resolved.parents
                or _BENCHMARK_HELD_OUT_PATH_RE.search(str(resolved))
            ):
                return "Error: Sandbox policy blocked access to held-out benchmark evaluation data outside the workspace."
        return None

    return _guard


def episode_touches_pg_cluster_roles(episode: EpisodeSpec) -> bool:
    """Return True if `episode` mounts a PostgreSQL task that mutates cluster-global `ROLE / USER` (`NEW-11`)."""
    for mnt in episode.workspace_mounts:
        if mnt.pillar == "mcpmark_postgres":
            uid_tail = mnt.upstream_id.rsplit("/", 1)[-1]
            tdir_name = Path(str(mnt.setup_meta.get("task_dir") or "")).name
            if uid_tail in PG_CLUSTER_ROLE_TASK_IDS or tdir_name in PG_CLUSTER_ROLE_TASK_IDS:
                return True
    return False

MCPMARK_FS_URLS = {
    "desktop": "https://storage.mcpmark.ai/filesystem/desktop.zip",
    "file_context": "https://storage.mcpmark.ai/filesystem/file_context.zip",
    "file_property": "https://storage.mcpmark.ai/filesystem/file_property.zip",
    "folder_structure": "https://storage.mcpmark.ai/filesystem/folder_structure.zip",
    "papers": "https://storage.mcpmark.ai/filesystem/papers.zip",
    "student_database": "https://storage.mcpmark.ai/filesystem/student_database.zip",
    "threestudio": "https://storage.mcpmark.ai/filesystem/threestudio.zip",
    "votenet": "https://storage.mcpmark.ai/filesystem/votenet.zip",
    "legal_document": "https://storage.mcpmark.ai/filesystem/legal_document.zip",
    "desktop_template": "https://storage.mcpmark.ai/filesystem/desktop_template.zip",
}


def _http_json(method: str, url: str, body: dict[str, Any] | None = None, timeout: float = 45.0) -> dict[str, Any]:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {"Content-Type": "application/json"} if body is not None else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} on {method} {url}: {detail}") from e


_GAIA2_SERVER_BOOT_SEM = threading.BoundedSemaphore(8)


class Gaia2ServerProcess:
    """Manages an isolated `gaia2_env_server.py` daemon running under `.venv_are`."""

    def __init__(self, port: int = 19011, log_path: Path | None = None, diag_token: str = "") -> None:
        self.port = port
        self.base_url = f"http://127.0.0.1:{port}"
        self.log_path = log_path or (Path(tempfile.gettempdir()) / f"gaia2_env_server_{port}.log")
        self.diag_token = diag_token
        self.proc: subprocess.Popen[str] | None = None

    def ensure_started(self, timeout_s: float = 50.0) -> None:
        if self.proc is not None and self.proc.poll() is None:
            try:
                st = _http_json("GET", f"{self.base_url}/health", timeout=2.0)
                if st.get("ok"):
                    return
            except Exception:
                pass

        with _GAIA2_SERVER_BOOT_SEM:
            try:
                st = _http_json("GET", f"{self.base_url}/health", timeout=2.0)
                if st.get("ok"):
                    return
            except Exception:
                pass

            are_py = PROJECT_ROOT / ".venv_are" / "bin" / "python"
            if not are_py.exists():
                raise RuntimeError(f"Missing Meta ARE virtualenv python at {are_py}")

            env = os.environ.copy()
            env["PYTHONPATH"] = str(PROJECT_ROOT)
            env["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
            env["DEMO_FS_PATH"] = str(UPSTREAM_ROOT / "gaia2_demo_fs")
            if self.diag_token:
                env["PA150_GAIA_DIAG_SECRET"] = self.diag_token
            else:
                env.pop("PA150_GAIA_DIAG_SECRET", None)

            server_script = PROJECT_ROOT / "synaptic_bench_150" / "gaia2_env_server.py"
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            log_fp = self.log_path.open("w", encoding="utf-8")
            self.proc = subprocess.Popen(
                [str(are_py), str(server_script), "--port", str(self.port)],
                cwd=str(PROJECT_ROOT),
                env=env,
                stdout=log_fp,
                stderr=subprocess.STDOUT,
                text=True,
            )
            deadline = time.time() + timeout_s
            last_err = ""
            while time.time() < deadline:
                if self.proc.poll() is not None:
                    tail = self.log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
                    raise RuntimeError(f"Gaia2EnvServer exited early ({self.proc.returncode}):\n{tail}")
                try:
                    st = _http_json("GET", f"{self.base_url}/health", timeout=2.0)
                    if st.get("ok"):
                        return
                except Exception as exc:
                    last_err = str(exc)
                time.sleep(0.3)
            raise TimeoutError(f"Timed out starting Gaia2EnvServer on {self.base_url}: {last_err}")

    def start_scenario(
        self,
        *,
        upstream_id: str,
        config: str,
        scenario_id: str,
        judge_mode: str = "auto",
        readonly_side_mount: bool = False,
    ) -> dict[str, Any]:
        self.ensure_started()
        return _http_json(
            "POST",
            f"{self.base_url}/start_scenario",
            {
                "upstream_id": upstream_id,
                "config": config,
                "scenario_id": scenario_id,
                "judge_mode": judge_mode,
                "readonly_side_mount": bool(readonly_side_mount),
            },
            timeout=60.0,
        )

    def snapshot(self) -> dict[str, Any]:
        return _http_json("POST", f"{self.base_url}/snapshot", {}, timeout=15.0)

    def evaluate(self, agent_replies: list[str] | None = None) -> dict[str, Any]:
        return _http_json(
            "POST",
            f"{self.base_url}/evaluate",
            {"agent_replies": list(agent_replies or [])},
            timeout=60.0,
        )

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=8.0)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None


def ensure_mcpmark_fs_template(category: str) -> Path:
    """Download & cache an official MCPMark filesystem zip once, returning its clean directory."""
    with _CACHE_INIT_LOCK:
        FS_TEMPLATE_CACHE.mkdir(parents=True, exist_ok=True)
        target_dir = FS_TEMPLATE_CACHE / category
        marker = target_dir / ".pa150_ready"
        if target_dir.exists() and marker.exists():
            return target_dir

        url = MCPMARK_FS_URLS.get(category, f"https://storage.mcpmark.ai/filesystem/{category}.zip")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            zip_path = tmp_path / f"{category}.zip"
            subprocess.run(
                ["curl", "-fsSL", "--retry", "3", "-o", str(zip_path), url],
                check=True,
                capture_output=True,
                text=True, encoding="utf-8", errors="replace",
            )
            extract_dir = tmp_path / "extracted"
            extract_dir.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["unzip", "-q", "-o", str(zip_path), "-d", str(extract_dir)],
                check=True,
                capture_output=True,
                text=True, encoding="utf-8", errors="replace",
            )
            macosx = extract_dir / "__MACOSX"
            if macosx.exists():
                shutil.rmtree(macosx)

            if target_dir.exists():
                shutil.rmtree(target_dir)

            nested = extract_dir / category
            if nested.is_dir():
                shutil.copytree(nested, target_dir)
            else:
                entries = [p for p in extract_dir.iterdir() if p.name != "__MACOSX"]
                if len(entries) == 1 and entries[0].is_dir():
                    shutil.copytree(entries[0], target_dir)
                else:
                    shutil.copytree(extract_dir, target_dir)
            marker.write_text("ok\n", encoding="utf-8")
        return target_dir


def _pg_can_connect(port: int = PG_DEFAULT_PORT) -> bool:
    try:
        conn = psycopg2.connect(
            host=PG_DEFAULT_HOST,
            port=port,
            user=PG_DEFAULT_USER,
            password=PG_DEFAULT_PASSWORD,
            database="postgres",
            connect_timeout=2,
        )
        conn.close()
        return True
    except Exception:
        return False


def _verify_pgvector_available(port: int = PG_DEFAULT_PORT) -> None:
    conn = psycopg2.connect(
        host=PG_DEFAULT_HOST,
        port=port,
        user=PG_DEFAULT_USER,
        password=PG_DEFAULT_PASSWORD,
        database="postgres",
    )
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_available_extensions WHERE name = 'vector'")
            if cur.fetchone() is None:
                raise RuntimeError(
                    f"PostgreSQL on port {port} lacks required `pgvector` (`vector`) extension (`NEW-5`). "
                    f"Ensure `{PG_CONTAINER_NAME}` uses image `pgvector/pgvector:pg17`."
                )
    finally:
        conn.close()


def ensure_postgres_container(port: int = PG_DEFAULT_PORT) -> None:
    """Ensure local Docker PostgreSQL 17 (`pgvector/pgvector:pg17`) is running on `127.0.0.1:15432`."""
    with _PG_ADMIN_LOCK:
        if _pg_can_connect(port):
            _verify_pgvector_available(port)
            return

        inspect = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", PG_CONTAINER_NAME],
            capture_output=True,
            text=True, encoding="utf-8", errors="replace",
        )
        if inspect.returncode == 0:
            subprocess.run(["docker", "start", PG_CONTAINER_NAME], check=True, capture_output=True, text=True, encoding="utf-8", errors="replace")
        else:
            run_res = subprocess.run(
                [
                    "docker",
                    "run",
                    "-d",
                    "--name",
                    PG_CONTAINER_NAME,
                    "-p",
                    f"127.0.0.1:{port}:5432",
                    "-e",
                    f"POSTGRES_PASSWORD={PG_DEFAULT_PASSWORD}",
                    "pgvector/pgvector:pg17",
                ],
                capture_output=True,
                text=True, encoding="utf-8", errors="replace",
            )
            if run_res.returncode != 0:
                raise RuntimeError(
                    f"Failed to start `pgvector/pgvector:pg17` container `{PG_CONTAINER_NAME}`: {run_res.stderr}"
                )

        deadline = time.time() + 40.0
        while time.time() < deadline:
            if _pg_can_connect(port):
                _verify_pgvector_available(port)
                return
            time.sleep(0.5)
        raise TimeoutError(f"PostgreSQL container `{PG_CONTAINER_NAME}` did not become ready on port {port}")


def _pg_db_exists(db_name: str, port: int = PG_DEFAULT_PORT) -> bool:
    conn = psycopg2.connect(
        host=PG_DEFAULT_HOST,
        port=port,
        user=PG_DEFAULT_USER,
        password=PG_DEFAULT_PASSWORD,
        database="postgres",
    )
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db_name,))
            return cur.fetchone() is not None
    finally:
        conn.close()


def ensure_postgres_template_db(category: str, port: int = PG_DEFAULT_PORT) -> bool:
    """Download & `pg_restore` an MCPMark `.backup` into a reusable template DB once."""
    if category not in MCPMARK_TEMPLATE_DBS:
        return False
    with _CACHE_INIT_LOCK:
        ensure_postgres_container(port)
        if _pg_db_exists(category, port):
            return True

        PG_BACKUP_CACHE.mkdir(parents=True, exist_ok=True)
        backup_fp = PG_BACKUP_CACHE / f"{category}.backup"
        if not backup_fp.exists():
            url = f"https://storage.mcpmark.ai/postgres/{category}.backup"
            subprocess.run(
                ["curl", "-fsSL", "--retry", "3", "-o", str(backup_fp), url],
                check=True,
                capture_output=True,
                text=True, encoding="utf-8", errors="replace",
            )

        conn = psycopg2.connect(
            host=PG_DEFAULT_HOST,
            port=port,
            user=PG_DEFAULT_USER,
            password=PG_DEFAULT_PASSWORD,
            database="postgres",
        )
        conn.autocommit = True
        try:
            with conn.cursor() as cur:
                cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(category)))
        finally:
            conn.close()

        with backup_fp.open("rb") as bfp:
            restore_res = subprocess.run(
                [
                    "docker",
                    "exec",
                    "-i",
                    "-e",
                    f"PGPASSWORD={PG_DEFAULT_PASSWORD}",
                    PG_CONTAINER_NAME,
                    "pg_restore",
                    "-U",
                    PG_DEFAULT_USER,
                    "-d",
                    category,
                    "--no-owner",
                    "--no-privileges",
                ],
                stdin=bfp,
                capture_output=True,
            )

        # Verify restored template DB is non-empty
        vconn = psycopg2.connect(
            host=PG_DEFAULT_HOST,
            port=port,
            user=PG_DEFAULT_USER,
            password=PG_DEFAULT_PASSWORD,
            database=category,
        )
        try:
            with vconn.cursor() as vcur:
                vcur.execute(
                    "SELECT count(*) FROM information_schema.tables WHERE table_schema NOT IN ('pg_catalog', 'information_schema')"
                )
                tbl_cnt = int(vcur.fetchone()[0])
        finally:
            vconn.close()
        if tbl_cnt <= 0:
            drop_postgres_db(category, port)
            err_txt = restore_res.stderr.decode("utf-8", errors="replace")[-600:]
            raise RuntimeError(f"pg_restore yielded 0 tables for template `{category}`: {err_txt}")
        return True


def create_isolated_postgres_db(
    db_name: str,
    category: str,
    task_dir: Path | None = None,
    port: int = PG_DEFAULT_PORT,
) -> dict[str, str]:
    """Create a fresh isolated database for one episode mount (from template or `prepare_environment.py`)."""
    has_template = ensure_postgres_template_db(category, port)
    # P0-3: Serialize `CREATE DATABASE ... WITH TEMPLATE` across concurrent slots so PostgreSQL never raises `ObjectInUse`
    with _PG_ADMIN_LOCK:
        ensure_postgres_container(port)
        conn = psycopg2.connect(
            host=PG_DEFAULT_HOST,
            port=port,
            user=PG_DEFAULT_USER,
            password=PG_DEFAULT_PASSWORD,
            database="postgres",
        )
        conn.autocommit = True
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = %s AND pid <> pg_backend_pid()",
                    (db_name,),
                )
                cur.execute(sql.SQL("DROP DATABASE IF EXISTS {}").format(sql.Identifier(db_name)))
                if has_template:
                    cur.execute(
                        "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = %s AND pid <> pg_backend_pid()",
                        (category,),
                    )
                    cur.execute(
                        sql.SQL("CREATE DATABASE {} WITH TEMPLATE {}").format(
                            sql.Identifier(db_name),
                            sql.Identifier(category),
                        )
                    )
                else:
                    cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db_name)))
        finally:
            conn.close()

    PG_BACKUP_CACHE.mkdir(parents=True, exist_ok=True)
    shared_pgpass = PG_BACKUP_CACHE / f".pgpass_{port}"
    shared_pgpass.write_text(
        f"{PG_DEFAULT_HOST}:{port}:*:{PG_DEFAULT_USER}:{PG_DEFAULT_PASSWORD}\n",
        encoding="utf-8",
    )
    shared_pgpass.chmod(0o600)
    pg_env = {
        "POSTGRES_HOST": PG_DEFAULT_HOST,
        "POSTGRES_PORT": str(port),
        "POSTGRES_DATABASE": db_name,
        "POSTGRES_USERNAME": PG_DEFAULT_USER,
        "POSTGRES_PASSWORD": PG_DEFAULT_PASSWORD,
        "PGPASSFILE": str(shared_pgpass),
    }
    if task_dir is not None:
        prep = Path(task_dir) / "prepare_environment.py"
        if prep.exists():
            env = os.environ.copy()
            env.update(pg_env)
            py_bin = str(PROJECT_ROOT / ".venv" / "bin" / "python")
            res = subprocess.run(
                [py_bin, str(prep)],
                cwd=str(task_dir),
                env=env,
                capture_output=True,
                text=True, encoding="utf-8", errors="replace",
                timeout=180,
            )
            if res.returncode != 0:
                drop_postgres_db(db_name, port)
                raise RuntimeError(f"prepare_environment.py failed for {db_name}: {res.stderr}")
    return pg_env


def drop_postgres_db(db_name: str, port: int = PG_DEFAULT_PORT) -> None:
    with _PG_ADMIN_LOCK:
        if not _pg_can_connect(port):
            return
        conn = psycopg2.connect(
            host=PG_DEFAULT_HOST,
            port=port,
            user=PG_DEFAULT_USER,
            password=PG_DEFAULT_PASSWORD,
            database="postgres",
        )
        conn.autocommit = True
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = %s AND pid <> pg_backend_pid()",
                    (db_name,),
                )
                cur.execute(sql.SQL("DROP DATABASE IF EXISTS {}").format(sql.Identifier(db_name)))
        finally:
            conn.close()


def cleanup_postgres_cluster_roles(port: int = PG_DEFAULT_PORT) -> list[str]:
    """Drop non-system cluster-global PostgreSQL roles (`test_user`, `theme_analyst`, etc.) (`P0-4`)."""
    with _PG_ADMIN_LOCK:
        if not _pg_can_connect(port):
            return []
        conn = psycopg2.connect(
            host=PG_DEFAULT_HOST,
            port=port,
            user=PG_DEFAULT_USER,
            password=PG_DEFAULT_PASSWORD,
            database="postgres",
        )
        conn.autocommit = True
        dropped: list[str] = []
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT rolname FROM pg_roles WHERE rolname <> %s AND rolname NOT LIKE 'pg_%%' ORDER BY rolname",
                    (PG_DEFAULT_USER,),
                )
                roles = [str(r[0]) for r in cur.fetchall() if r and r[0]]
                if not roles:
                    return []
                cur.execute(
                    "SELECT datname FROM pg_database WHERE datistemplate = false AND datallowconn = true ORDER BY datname"
                )
                db_names = [str(r[0]) for r in cur.fetchall() if r and r[0]]

            for dname in db_names:
                try:
                    dconn = psycopg2.connect(
                        host=PG_DEFAULT_HOST,
                        port=port,
                        user=PG_DEFAULT_USER,
                        password=PG_DEFAULT_PASSWORD,
                        database=dname,
                    )
                    dconn.autocommit = True
                    try:
                        with dconn.cursor() as dcur:
                            for rname in roles:
                                try:
                                    dcur.execute(
                                        sql.SQL("REASSIGN OWNED BY {} TO {}").format(
                                            sql.Identifier(rname),
                                            sql.Identifier(PG_DEFAULT_USER),
                                        )
                                    )
                                except Exception:
                                    pass
                                try:
                                    dcur.execute(
                                        sql.SQL("DROP OWNED BY {} CASCADE").format(sql.Identifier(rname))
                                    )
                                except Exception:
                                    pass
                    finally:
                        dconn.close()
                except Exception:
                    pass

            with conn.cursor() as cur:
                for rname in roles:
                    cur.execute(
                        "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE usename = %s AND pid <> pg_backend_pid()",
                        (rname,),
                    )
                    cur.execute(sql.SQL("DROP ROLE IF EXISTS {}").format(sql.Identifier(rname)))
                    dropped.append(rname)
        finally:
            conn.close()
        return dropped


def ensure_cached_bare_repo(github_repo: str, base_commit: str) -> Path:
    """Fetch a full-blob shallow snapshot of `github_repo` at `base_commit` into local bare cache once."""
    with _CACHE_INIT_LOCK:
        REPO_BARE_CACHE.mkdir(parents=True, exist_ok=True)
        clean_repo = github_repo.strip().removesuffix(".git")
        slug = clean_repo.replace("/", "__")
        commit_key = (base_commit or "HEAD")[:12]
        bare_dir = REPO_BARE_CACHE / f"{slug}__{commit_key}.git"
        ready_marker = bare_dir / ".pa150_bare_v2_ready"
        if bare_dir.exists() and ready_marker.exists():
            return bare_dir

        if bare_dir.exists():
            shutil.rmtree(bare_dir)
        bare_dir.mkdir(parents=True, exist_ok=True)
        url = f"https://github.com/{clean_repo}.git"
        try:
            subprocess.run(["git", "init", "--bare"], cwd=bare_dir, check=True, capture_output=True, text=True, encoding="utf-8", errors="replace")
            subprocess.run(["git", "remote", "add", "origin", url], cwd=bare_dir, check=True, capture_output=True, text=True, encoding="utf-8", errors="replace")
            fetch_target = base_commit if base_commit else "HEAD"
            subprocess.run(
                ["git", "fetch", "--depth", "1", "origin", fetch_target],
                cwd=bare_dir,
                check=True,
                capture_output=True,
                text=True, encoding="utf-8", errors="replace",
            )
            subprocess.run(
                ["git", "update-ref", "refs/heads/main", "FETCH_HEAD"],
                cwd=bare_dir,
                check=True,
                capture_output=True,
                text=True, encoding="utf-8", errors="replace",
            )
            subprocess.run(
                ["git", "symbolic-ref", "HEAD", "refs/heads/main"],
                cwd=bare_dir,
                check=True,
                capture_output=True,
                text=True, encoding="utf-8", errors="replace",
            )
            ready_marker.write_text("ok\n", encoding="utf-8")
        except Exception:
            shutil.rmtree(bare_dir, ignore_errors=True)
            raise
        return bare_dir


_SITECUSTOMIZE_CODE = (
    "import importlib.abc, importlib.machinery, importlib.metadata, os, re, sys\n"
    "cwd = os.getcwd()\n"
    "roots = []\n"
    "cur = cwd\n"
    "for _ in range(6):\n"
    "    if os.path.exists(os.path.join(cur, '.git')) or os.path.isfile(os.path.join(cur, 'pyproject.toml')) or os.path.isfile(os.path.join(cur, 'setup.py')):\n"
    "        if cur not in roots:\n"
    "            roots.insert(0, cur)\n"
    "        break\n"
    "    parent = os.path.dirname(cur)\n"
    "    if not parent or parent == cur:\n"
    "        break\n"
    "    cur = parent\n"
    "for r in reversed(roots):\n"
    "    if r not in sys.path:\n"
    "        sys.path.insert(0, r)\n"
    "    src = os.path.join(r, 'src')\n"
    "    if os.path.isdir(src) and src not in sys.path:\n"
    "        sys.path.insert(0, src)\n"
    "class _ScmVersionFallback(importlib.abc.MetaPathFinder, importlib.abc.Loader):\n"
    "    def find_spec(self, fullname, path, target=None):\n"
    "        if fullname.endswith('._version'):\n"
    "            return importlib.machinery.ModuleSpec(fullname, self)\n"
    "        return None\n"
    "    def create_module(self, spec):\n"
    "        return None\n"
    "    def exec_module(self, module):\n"
    "        top = module.__name__.split('.', 1)[0]\n"
    "        ver = '7.4.0' if top in ('_pytest', 'pytest') else '0.1.0'\n"
    "        for cand in (top.lstrip('_'), top):\n"
    "            if not cand:\n"
    "                continue\n"
    "            try:\n"
    "                ver = importlib.metadata.version(cand)\n"
    "                break\n"
    "            except Exception:\n"
    "                pass\n"
    "        nums = [int(x) for x in re.findall(r'\\d+', ver)[:3]] or [0, 1, 0]\n"
    "        module.version = module.__version__ = ver\n"
    "        module.version_tuple = module.__version_tuple__ = tuple(nums)\n"
    "sys.meta_path.append(_ScmVersionFallback())\n"
    "if os.environ.get('PA150_ISOLATED_PYTEST') == '1' and sys.platform == 'darwin':\n"
    "    import urllib.request as _pa150_ur, socket as _pa150_sock\n"
    "    _pa150_ur.getproxies_macosx_sysconf = lambda: {}\n"
    "    _pa150_ur.proxy_bypass_macosx_sysconf = lambda host: False\n"
    "    _pa150_ur.getproxies = _pa150_ur.getproxies_environment\n"
    "    _pa150_ur.proxy_bypass = _pa150_ur.proxy_bypass_environment\n"
    "    _orig_sock_connect = _pa150_sock.socket.connect\n"
    "    def _pa150_connect(self, address):\n"
    "        if isinstance(address, tuple) and len(address) >= 1 and address[0] in ('10.255.255.1', '10.0.0.0'):\n"
    "            raise _pa150_sock.timeout('timed out')\n"
    "        return _orig_sock_connect(self, address)\n"
    "    _pa150_sock.socket.connect = _pa150_connect\n"
)


_LITELLM_MODEL_RESPONSE_SHIM = (
    "try:\n"
    "    import litellm.types.utils as _pa150_llu\n"
    "    _orig_mr_init = _pa150_llu.ModelResponse.__init__\n"
    "    def _pa150_mr_init(self, *a, **kw):\n"
    "        _orig_mr_init(self, *a, **kw)\n"
    "        if getattr(self, 'usage', None) is None:\n"
    "            _u = _pa150_llu.Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0)\n"
    "            object.__setattr__(self, 'usage', _u)\n"
    "            try:\n"
    "                self['usage'] = _u\n"
    "            except Exception:\n"
    "                pass\n"
    "    _pa150_llu.ModelResponse.__init__ = _pa150_mr_init\n"
    "except Exception:\n"
    "    pass\n"
)


def _refresh_venv_sitecustomize(venv_dir: Path, pkg_src: Path | None = None) -> None:
    for sp_dir in (venv_dir / "lib").glob("python*/site-packages"):
        if pkg_src is not None:
            for pth in list(sp_dir.glob("*editable*")) + list(sp_dir.glob("*.egg-link")) + list(sp_dir.glob("*.pth")):
                try:
                    txt = pth.read_text(encoding="utf-8", errors="ignore") if pth.is_file() else ""
                except Exception:
                    txt = ""
                if "editable" in pth.name or str(pkg_src) in txt or pth.suffix == ".egg-link":
                    pth.unlink(missing_ok=True)
        sc_code = _SITECUSTOMIZE_CODE
        if (sp_dir / "litellm" / "types" / "utils.py").exists():
            sc_code = sc_code + _LITELLM_MODEL_RESPONSE_SHIM
        (sp_dir / "sitecustomize.py").write_text(sc_code, encoding="utf-8")


_VENV_V3_REBUILD_REPOS = frozenset(
    {
        "huggingface/datasets",
        "streamlink/streamlink",
        "pylint-dev/pylint",
        "pallets/flask",
        "aiogram/aiogram",
    }
)
_VENV_V4_REBUILD_REPOS = frozenset(
    {
        "pallets/flask",
        "streamlink/streamlink",
        "aws-cloudformation/cfn-lint",
    }
)
_VENV_V7_REBUILD_REPOS = frozenset(
    {
        "huggingface/datasets",
    }
)


def _opaque_cache_link(kind: str, target: Path) -> Path:
    """Expose cached repos/venvs via `/tmp/pa150_cache_links/<kind>/<name>` so workspace metadata never leaks `upstream_pillars` (`Item 1`)."""
    link_dir = Path(tempfile.gettempdir()) / "pa150_cache_links" / kind
    link_dir.mkdir(parents=True, exist_ok=True)
    link_path = link_dir / target.name
    resolved = target.resolve()
    if link_path.is_symlink() or link_path.exists():
        try:
            if link_path.resolve() == resolved:
                return link_path
            link_path.unlink()
        except Exception:
            pass
    try:
        link_path.symlink_to(resolved)
        return link_path
    except Exception:
        return target


def ensure_cached_repo_venv(github_repo: str, base_commit: str, sample_checkout: Path) -> Path:
    """Build & cache a Python 3.11 `.venv` with `pytest` + repo third-party dependencies once per `(repo, commit)`.

    Crucially strips any `_editable` `.pth` pointer to `pkg_src` and installs a `sitecustomize.py`
    that prepends `<repo_root>/src` and `<repo_root>` to `sys.path`, guaranteeing every agent worktree
    and isolated `eval_dir` imports its OWN patched source tree rather than a shared cache copy (`P0-7`, `P0-2`, `A3`).
    """
    with _CACHE_INIT_LOCK:
        VENV_CACHE.mkdir(parents=True, exist_ok=True)
        clean_repo = github_repo.strip().removesuffix(".git")
        repo_lower = clean_repo.lower()
        is_pytest_self_repo = repo_lower == "pytest-dev/pytest"
        slug = clean_repo.replace("/", "__")
        commit_short = (base_commit or "head")[:12]
        venv_dir = VENV_CACHE / f"{slug}__{commit_short}"
        if repo_lower in _VENV_V7_REBUILD_REPOS:
            marker_name = ".pa150_venv_v7_ready"
        elif repo_lower in _VENV_V4_REBUILD_REPOS:
            marker_name = ".pa150_venv_v4_ready"
        elif repo_lower in _VENV_V3_REBUILD_REPOS:
            marker_name = ".pa150_venv_v3_ready"
        else:
            marker_name = ".pa150_venv_v2_ready"
        ready_marker = venv_dir / marker_name
        if venv_dir.exists() and ready_marker.exists() and (venv_dir / "bin" / "python").exists():
            _refresh_venv_sitecustomize(venv_dir)
            return venv_dir

        if venv_dir.exists():
            shutil.rmtree(venv_dir)

        pkg_src = VENV_CACHE / f"{slug}__{commit_short}_src"
        if pkg_src.exists():
            shutil.rmtree(pkg_src)

        pyproj_txt = (
            (sample_checkout / "pyproject.toml").read_text(encoding="utf-8", errors="ignore")
            if (sample_checkout / "pyproject.toml").exists()
            else ""
        )
        needs_modern_pytest8 = ('minversion = "8.' in pyproj_txt) or ("asyncio_default_fixture_loop_scope" in pyproj_txt)
        use_legacy_pytest7 = repo_lower in ("pylint-dev/pylint", "pallets/flask") or (
            repo_lower == "streamlink/streamlink" and not needs_modern_pytest8
        )
        pytest_pkg = "pytest>=7.4,<8" if use_legacy_pytest7 else "pytest>=8.4,<9"
        pytest_asyncio_pkg = "pytest-asyncio>=0.21,<0.24" if use_legacy_pytest7 else "pytest-asyncio"

        repo_extra_pkgs: list[str] = []
        post_pin_pkgs: list[str] = []
        if repo_lower == "huggingface/datasets":
            repo_extra_pkgs = [
                "numpy<2",
                "pyarrow<21",
                "dill",
                "xxhash",
                "fsspec",
                "aiohttp",
                "multiprocess",
                "pandas<2.3",
                "requests",
                "tqdm",
                "packaging",
                "pyyaml",
                "pytest-datadir",
                "pillow>=6.2.1",
                "absl-py",
                "zstandard",
                "lz4",
                "soundfile",
            ]
            post_pin_pkgs = ["numpy<2", "pandas<2.3", "pyarrow<21"]
        elif repo_lower == "streamlink/streamlink":
            repo_extra_pkgs = [
                "pytest-trio",
                "trio",
                "trio-websocket",
                "requests-mock",
                "freezegun",
                "lxml",
                "pycryptodome",
                "urllib3>=2.0.0,<2.5",
            ]
            post_pin_pkgs = [pytest_pkg, pytest_asyncio_pkg, "urllib3>=2.0.0,<2.5", "py>=1.11.0"]
        elif repo_lower == "pallets/flask":
            repo_extra_pkgs = ["py>=1.11.0", "werkzeug<3.1"]
            post_pin_pkgs = ["pytest>=7.4,<8", "pytest-asyncio>=0.21,<0.24", "py>=1.11.0", "werkzeug<3.1"]
        elif repo_lower == "pylint-dev/pylint":
            repo_extra_pkgs = ["py>=1.11.0"]
            post_pin_pkgs = ["pytest>=7.4,<8", "pytest-asyncio>=0.21,<0.24", "py>=1.11.0"]
        elif repo_lower == "aws-cloudformation/cfn-lint":
            repo_extra_pkgs = [
                "junit-xml",
                "pydot",
                "sarif-om",
                "jschema-to-python",
            ]
            post_pin_pkgs = ["junit-xml", "pydot", "sarif-om", "jschema-to-python"]
        elif repo_lower == "aiogram/aiogram":
            repo_extra_pkgs = [
                "python-socks<2.6",
                "aiohttp-socks>=0.8.3,<0.9",
                "pytest-aiohttp",
                "pytest-lazy-fixture",
                "aresponses",
            ]
            post_pin_pkgs = ["python-socks<2.6"]
        elif is_pytest_self_repo:
            # P0-2: Do NOT pre-install modern `pytest>=8` or `pytest-asyncio/cov/xdist` plugins when testing `pytest` itself!
            repo_extra_pkgs = [
                "hypothesis",
                "xmlschema",
                "argcomplete",
                "attrs",
                "iniconfig",
                "packaging",
                "pluggy",
                "py",
                "pygments",
                "requests",
                "mock",
            ]

        build_support_pkgs = [
            "setuptools<81",
            "setuptools-scm[toml]",
            "wheel",
            "flit-core",
            "hatchling",
            "hatch-vcs",
            "pdm-backend",
            "poetry-core",
            "versioningit",
        ]
        base_test_pkgs = (
            [*build_support_pkgs, *repo_extra_pkgs]
            if is_pytest_self_repo
            else [
                *build_support_pkgs,
                pytest_pkg,
                pytest_asyncio_pkg,
                "pytest-mock",
                "pytest-cov",
                "pytest-xdist",
                "hypothesis",
                "dirty-equals",
                "requests-mock",
                "freezegun",
                "lxml",
                "py",
                *repo_extra_pkgs,
            ]
        )

        uv_py_ver = "3.12" if (">=3.12" in pyproj_txt or ">= 3.12" in pyproj_txt) else "3.11"
        has_pkg_manifest = (sample_checkout / "pyproject.toml").exists() or (sample_checkout / "setup.py").exists()
        install_succeeded = not has_pkg_manifest
        last_install_err = ""
        pkg_install_env = dict(os.environ)
        pkg_install_env.setdefault("SETUPTOOLS_SCM_PRETEND_VERSION", "7.4.0" if is_pytest_self_repo else "0.1.0")

        try:
            uv_bin = shutil.which("uv")
            if uv_bin:
                subprocess.run([uv_bin, "venv", "--python", uv_py_ver, str(venv_dir)], check=True, capture_output=True, text=True, encoding="utf-8", errors="replace")
                subprocess.run(
                    [
                        uv_bin,
                        "pip",
                        "install",
                        "--python",
                        str(venv_dir / "bin" / "python"),
                        *base_test_pkgs,
                    ],
                    check=True,
                    capture_output=True,
                    text=True, encoding="utf-8", errors="replace",
                )
                if has_pkg_manifest:
                    shutil.copytree(sample_checkout, pkg_src, ignore=shutil.ignore_patterns(".git", ".venv", ".worktrees"))
                    if is_pytest_self_repo:
                        editable_candidates = (["-e", f"{pkg_src}[testing]"], ["-e", str(pkg_src)], ["--no-build-isolation", "-e", str(pkg_src)])
                    elif repo_lower == "aiogram/aiogram":
                        editable_candidates = (
                            ["-e", f"{pkg_src}[test,fast,redis,mongo,proxy,i18n]"],
                            ["-e", f"{pkg_src}[test]"],
                            ["-e", str(pkg_src)],
                        )
                    else:
                        editable_candidates = (["-e", f"{pkg_src}[test]"], ["-e", str(pkg_src)], ["--no-build-isolation", "-e", str(pkg_src)])
                    for install_args in editable_candidates:
                        inst = subprocess.run(
                            [uv_bin, "pip", "install", "--python", str(venv_dir / "bin" / "python"), *install_args],
                            env=pkg_install_env,
                            capture_output=True,
                            text=True, encoding="utf-8", errors="replace",
                        )
                        if inst.returncode == 0:
                            install_succeeded = True
                            break
                        last_install_err = (inst.stderr or inst.stdout)[-800:]
                if install_succeeded and post_pin_pkgs:
                    subprocess.run(
                        [
                            uv_bin,
                            "pip",
                            "install",
                            "--python",
                            str(venv_dir / "bin" / "python"),
                            *post_pin_pkgs,
                        ],
                        check=True,
                        capture_output=True,
                        text=True, encoding="utf-8", errors="replace",
                    )
            else:
                subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True, capture_output=True)
                subprocess.run(
                    [str(venv_dir / "bin" / "pip"), "install", "-q", *base_test_pkgs],
                    check=True,
                    capture_output=True,
                )
                if has_pkg_manifest:
                    shutil.copytree(sample_checkout, pkg_src, ignore=shutil.ignore_patterns(".git", ".venv", ".worktrees"))
                    inst = subprocess.run(
                        [str(venv_dir / "bin" / "pip"), "install", "-q", "-e", str(pkg_src)],
                        env=pkg_install_env,
                        capture_output=True,
                        text=True, encoding="utf-8", errors="replace",
                    )
                    install_succeeded = inst.returncode == 0
                    last_install_err = (inst.stderr or inst.stdout)[-800:]
                if install_succeeded and post_pin_pkgs:
                    subprocess.run(
                        [str(venv_dir / "bin" / "pip"), "install", "-q", *post_pin_pkgs],
                        check=True,
                        capture_output=True,
                    )

            if not install_succeeded:
                raise RuntimeError(
                    f"Editable package installation failed for `{clean_repo}@{commit_short}` (`NEW-4`): {last_install_err}"
                )

            _refresh_venv_sitecustomize(venv_dir, pkg_src=pkg_src)

            subprocess.run(
                [str(venv_dir / "bin" / "python"), "-m", "pytest", "--version"],
                cwd=str(sample_checkout),
                check=True,
                capture_output=True,
                text=True, encoding="utf-8", errors="replace",
            )
            ready_marker.write_text("ok\n", encoding="utf-8")
        except Exception:
            shutil.rmtree(venv_dir, ignore_errors=True)
            raise
        finally:
            if pkg_src.exists():
                shutil.rmtree(pkg_src, ignore_errors=True)
        return venv_dir


def build_gaia_cli_script(gaia_base_url: str) -> str:
    py_bin = str(PROJECT_ROOT / ".venv" / "bin" / "python")
    return f"""#!{py_bin}
import json, sys, urllib.error, urllib.request

BASE = {gaia_base_url!r}

def req(method, path, body=None):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {{"Content-Type": "application/json"}} if body is not None else {{}}
    r = urllib.request.Request(BASE + path, data=data, headers=headers, method=method)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({{}}))
    try:
        with opener.open(r, timeout=45.0) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        sys.stderr.write(e.read().decode("utf-8", errors="replace") + "\\n")
        sys.exit(1)

if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
    print("Usage:\\n  ./gaia_cli apps\\n  ./gaia_cli docs <AppName>\\n  ./gaia_cli call <AppName__tool_name> '<json_args>'")
    sys.exit(0)

cmd = sys.argv[1]
if cmd == "apps":
    print(json.dumps(req("GET", "/apps"), ensure_ascii=False, indent=2))
elif cmd == "docs":
    if len(sys.argv) < 3:
        sys.stderr.write("Usage: ./gaia_cli docs <AppName>\\n")
        sys.exit(1)
    print(json.dumps(req("GET", "/docs/" + sys.argv[2]), ensure_ascii=False, indent=2))
elif cmd == "call":
    if len(sys.argv) < 3:
        sys.stderr.write("Usage: ./gaia_cli call <AppName__tool_name> '<json_args>'\\n")
        sys.exit(1)
    tname = sys.argv[2]
    args = json.loads(sys.argv[3]) if len(sys.argv) >= 4 and sys.argv[3].strip() else {{}}
    print(json.dumps(req("POST", "/call_tool", {{"tool_name": tname, "args": args}}), ensure_ascii=False, indent=2))
else:
    sys.stderr.write(f"Unknown command {{cmd!r}}\\n")
    sys.exit(1)
"""


def build_pg_exec_script(default_pg_env: dict[str, str]) -> str:
    py_bin = str(PROJECT_ROOT / ".venv" / "bin" / "python")
    return f"""#!{py_bin}
import json, os, sys
import psycopg2
import psycopg2.extras

DEFAULTS = {json.dumps(default_pg_env)}

def conn():
    return psycopg2.connect(
        host=os.environ.get("POSTGRES_HOST", DEFAULTS["POSTGRES_HOST"]),
        port=int(os.environ.get("POSTGRES_PORT", DEFAULTS["POSTGRES_PORT"])),
        database=os.environ.get("POSTGRES_DATABASE", DEFAULTS["POSTGRES_DATABASE"]),
        user=os.environ.get("POSTGRES_USERNAME", DEFAULTS["POSTGRES_USERNAME"]),
        password=os.environ.get("POSTGRES_PASSWORD", DEFAULTS["POSTGRES_PASSWORD"]),
    )

if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
    print("Usage:\\n  ./pg_exec --tables\\n  ./pg_exec \\"SELECT ...\\"\\n  ./pg_exec -f script.sql")
    sys.exit(0)

if sys.argv[1] == "--tables":
    sql_text = \"\"\"
    SELECT table_schema, table_name, table_type
    FROM information_schema.tables
    WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
    ORDER BY table_schema, table_name;
    \"\"\"
elif sys.argv[1] == "-f" and len(sys.argv) >= 3:
    with open(sys.argv[2], "r", encoding="utf-8") as fp:
        sql_text = fp.read()
else:
    sql_text = sys.argv[1]

with conn() as c:
    c.autocommit = True
    with c.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql_text)
        if cur.description:
            rows = cur.fetchall()
            print(json.dumps([dict(r) for r in rows], default=str, ensure_ascii=False, indent=2))
        else:
            print(json.dumps({{"ok": True, "statusmessage": cur.statusmessage, "rowcount": cur.rowcount}}, ensure_ascii=False))
"""


def hash_directory_tree(root: Path) -> dict[str, Any]:
    """Compute a deterministic recursive cryptographic hash over a filesystem mount."""
    h = hashlib.sha256()
    file_count = 0
    for p in sorted(root.rglob("*")):
        if p.name == ".pa150_ready":
            continue
        rel = p.relative_to(root).as_posix()
        if p.is_symlink():
            h.update(f"L:{rel}->{os.readlink(p)}\n".encode("utf-8"))
        elif p.is_dir():
            h.update(f"D:{rel}\n".encode("utf-8"))
        elif p.is_file():
            file_count += 1
            data = p.read_bytes()
            fh = hashlib.sha256(data).hexdigest()
            h.update(f"F:{rel}:{len(data)}:{fh}\n".encode("utf-8"))
    return {"tree_sha256": h.hexdigest(), "file_count": file_count}


def hash_postgres_database(pg_env: dict[str, str], *, include_cluster_roles: bool = False) -> dict[str, Any]:
    """Compute a deterministic full-DDL + row-content + function-body + RLS/ACL/grant digest over PostgreSQL (`NEW-12`)."""
    conn = psycopg2.connect(
        host=pg_env["POSTGRES_HOST"],
        port=int(pg_env["POSTGRES_PORT"]),
        database=pg_env["POSTGRES_DATABASE"],
        user=pg_env["POSTGRES_USERNAME"],
        password=pg_env["POSTGRES_PASSWORD"],
    )
    h = hashlib.sha256()
    table_count = 0
    try:
        with conn.cursor() as cur:
            # 1. Tables, views, columns, and base-table row digests
            cur.execute(
                """
                SELECT table_schema, table_name, table_type
                FROM information_schema.tables
                WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
                ORDER BY table_schema, table_name
                """
            )
            tables = cur.fetchall()
            for schema_name, table_name, table_type in tables:
                table_count += 1
                h.update(f"TBL:{schema_name}.{table_name}:{table_type}\n".encode("utf-8"))
                cur.execute(
                    """
                    SELECT column_name, data_type, udt_name, is_nullable, COALESCE(column_default, '')
                    FROM information_schema.columns
                    WHERE table_schema = %s AND table_name = %s
                    ORDER BY ordinal_position
                    """,
                    (schema_name, table_name),
                )
                for col in cur.fetchall():
                    h.update(f"COL:{col[0]}:{col[1]}:{col[2]}:{col[3]}:{col[4]}\n".encode("utf-8"))
                if table_type == "BASE TABLE":
                    cur.execute(
                        sql.SQL("SELECT md5(CAST(t.* AS text)) AS rh FROM {}.{} AS t ORDER BY rh").format(
                            sql.Identifier(schema_name),
                            sql.Identifier(table_name),
                        )
                    )
                    for (rh,) in cur.fetchall():
                        h.update(f"ROW:{rh}\n".encode("utf-8"))

            # 2. Views & Materialized Views definitions + matview rows
            cur.execute(
                """
                SELECT schemaname, viewname, COALESCE(definition, '')
                FROM pg_views
                WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
                ORDER BY schemaname, viewname
                """
            )
            for s_name, v_name, v_def in cur.fetchall():
                h.update(f"VIEW:{s_name}.{v_name}:{v_def.strip()}\n".encode("utf-8"))

            cur.execute(
                """
                SELECT schemaname, matviewname, ispopulated, COALESCE(definition, '')
                FROM pg_matviews
                WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
                ORDER BY schemaname, matviewname
                """
            )
            for s_name, mv_name, is_pop, mv_def in cur.fetchall():
                h.update(f"MATVIEW:{s_name}.{mv_name}:{is_pop}:{mv_def.strip()}\n".encode("utf-8"))
                if is_pop:
                    cur.execute(
                        sql.SQL("SELECT md5(CAST(t.* AS text)) AS rh FROM {}.{} AS t ORDER BY rh").format(
                            sql.Identifier(s_name),
                            sql.Identifier(mv_name),
                        )
                    )
                    for (rh,) in cur.fetchall():
                        h.update(f"MVROW:{rh}\n".encode("utf-8"))

            # 3. Functions & Procedures including full body (`prosrc`), arguments, return type, and security definer
            cur.execute(
                """
                SELECT n.nspname,
                       p.proname,
                       pg_get_function_identity_arguments(p.oid),
                       COALESCE(pg_get_function_result(p.oid), ''),
                       p.prosecdef,
                       COALESCE(p.prosrc, '')
                FROM pg_proc p
                JOIN pg_namespace n ON n.oid = p.pronamespace
                WHERE n.nspname NOT IN ('pg_catalog', 'information_schema')
                  AND NOT EXISTS (
                      SELECT 1 FROM pg_depend d
                      WHERE d.classid = 'pg_proc'::regclass AND d.objid = p.oid AND d.deptype = 'e'
                  )
                ORDER BY 1, 2, 3
                """
            )
            for r_nsp, r_name, r_args, r_ret, r_sec, r_src in cur.fetchall():
                h.update(f"PROC:{r_nsp}.{r_name}({r_args})->{r_ret}:sec={r_sec}:{r_src.strip()}\n".encode("utf-8"))

            # 4. Indexes & Constraints
            cur.execute(
                """
                SELECT schemaname, tablename, indexname, indexdef
                FROM pg_indexes
                WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
                ORDER BY schemaname, tablename, indexname
                """
            )
            for s_name, t_name, idx_name, idx_def in cur.fetchall():
                h.update(f"IDX:{s_name}.{t_name}.{idx_name}:{idx_def}\n".encode("utf-8"))

            cur.execute(
                """
                SELECT n.nspname, rel.relname, c.conname, c.contype, pg_get_constraintdef(c.oid)
                FROM pg_constraint c
                JOIN pg_namespace n ON n.oid = c.connamespace
                LEFT JOIN pg_class rel ON rel.oid = c.conrelid
                WHERE n.nspname NOT IN ('pg_catalog', 'information_schema')
                ORDER BY 1, 2, 3, 4
                """
            )
            for c_nsp, c_rel, c_name, c_type, c_def in cur.fetchall():
                h.update(f"CON:{c_nsp}.{c_rel}.{c_name}:{c_type}:{c_def}\n".encode("utf-8"))

            # 5. Row-Level Security flags, RLS policies, and table privileges
            cur.execute(
                """
                SELECT n.nspname, c.relname, c.relrowsecurity, c.relforcerowsecurity
                FROM pg_class c
                JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE n.nspname NOT IN ('pg_catalog', 'information_schema')
                  AND c.relkind IN ('r', 'p')
                ORDER BY 1, 2
                """
            )
            for nsp, rel, rls_on, rls_force in cur.fetchall():
                h.update(f"RLS:{nsp}.{rel}:{rls_on}:{rls_force}\n".encode("utf-8"))

            cur.execute(
                """
                SELECT schemaname, tablename, policyname, permissive, roles::text, cmd,
                       COALESCE(qual, ''), COALESCE(with_check, '')
                FROM pg_policies
                WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
                ORDER BY schemaname, tablename, policyname
                """
            )
            for pol in cur.fetchall():
                h.update(f"POL:{':'.join(str(x) for x in pol)}\n".encode("utf-8"))

            cur.execute(
                """
                SELECT grantee, table_schema, table_name, privilege_type, is_grantable
                FROM information_schema.role_table_grants
                WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
                ORDER BY grantee, table_schema, table_name, privilege_type
                """
            )
            for gr in cur.fetchall():
                h.update(f"GRT:{':'.join(str(x) for x in gr)}\n".encode("utf-8"))

            # 6. Sequences (`last_value`) & user-defined Triggers (`P2-1`)
            cur.execute(
                """
                SELECT schemaname, sequencename, COALESCE(last_value::text, 'NULL')
                FROM pg_sequences
                WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
                ORDER BY schemaname, sequencename
                """
            )
            for seq_s, seq_n, seq_v in cur.fetchall():
                h.update(f"SEQ:{seq_s}.{seq_n}:{seq_v}\n".encode("utf-8"))

            cur.execute(
                """
                SELECT n.nspname, c.relname, t.tgname, t.tgenabled, pg_get_triggerdef(t.oid)
                FROM pg_trigger t
                JOIN pg_class c ON c.oid = t.tgrelid
                JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE NOT t.tgisinternal
                  AND n.nspname NOT IN ('pg_catalog', 'information_schema')
                ORDER BY 1, 2, 3
                """
            )
            for tg_nsp, tg_rel, tg_name, tg_en, tg_def in cur.fetchall():
                h.update(f"TRG:{tg_nsp}.{tg_rel}.{tg_name}:{tg_en}:{tg_def}\n".encode("utf-8"))

            # 7. Database-scoped ACLs (`datacl`, `nspacl`, `relacl`, `pg_default_acl`) + optional cluster roles
            cur.execute("SELECT COALESCE(datacl::text, '') FROM pg_database WHERE datname = current_database()")
            for (d_acl,) in cur.fetchall():
                h.update(f"DBACL:{d_acl}\n".encode("utf-8"))

            cur.execute(
                """
                SELECT nspname, COALESCE(nspacl::text, '')
                FROM pg_namespace
                WHERE nspname NOT IN ('pg_catalog', 'information_schema')
                  AND nspname NOT LIKE 'pg_toast%'
                  AND nspname NOT LIKE 'pg_temp%'
                ORDER BY nspname
                """
            )
            for nsp_n, nsp_acl in cur.fetchall():
                h.update(f"NSPACL:{nsp_n}:{nsp_acl}\n".encode("utf-8"))

            cur.execute(
                """
                SELECT n.nspname, c.relname, COALESCE(c.relacl::text, '')
                FROM pg_class c
                JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE n.nspname NOT IN ('pg_catalog', 'information_schema')
                  AND c.relkind IN ('r', 'p', 'v', 'm', 'S')
                ORDER BY 1, 2
                """
            )
            for r_nsp, r_rel, r_acl in cur.fetchall():
                h.update(f"RELACL:{r_nsp}.{r_rel}:{r_acl}\n".encode("utf-8"))

            cur.execute(
                """
                SELECT COALESCE(d.defaclnamespace::regnamespace::text, ''),
                       d.defaclobjtype,
                       COALESCE(d.defaclacl::text, '')
                FROM pg_default_acl d
                ORDER BY 1, 2, 3
                """
            )
            for da_nsp, da_obj, da_acl in cur.fetchall():
                h.update(f"DEFACL:{da_nsp}:{da_obj}:{da_acl}\n".encode("utf-8"))

            if include_cluster_roles:
                cur.execute(
                    """
                    SELECT rolname, rolsuper, rolcreaterole, rolcreatedb, rolcanlogin, rolbypassrls
                    FROM pg_catalog.pg_roles
                    WHERE rolname NOT LIKE 'pg_%' AND rolname != 'postgres'
                    ORDER BY rolname
                    """
                )
                for rl in cur.fetchall():
                    h.update(f"ROLE:{':'.join(str(x) for x in rl)}\n".encode("utf-8"))
    finally:
        conn.close()
    return {"db_sha256": h.hexdigest(), "table_count": table_count}


def hash_git_repo_state(
    repo_dir: Path,
    extra_worktrees_root: Path | None = None,
    extra_worktrees_base_head: str | None = None,
) -> dict[str, Any]:
    """Snapshot git HEAD, dirty status, diff (across main repo + spawned worktrees), and spawned worktree count."""
    exclude_specs = [
        ":!AGENTS.md",
        ":!WORKSPACE_POLICY.md",
        ":!gaia_cli",
        ":!pg_exec",
        ":!.workspace_root",
        ":!feishu_docs",
        ":!office_fs",
        ":!office_pg",
        ":!gaia2_universe",
        ":!.shadow_drafts",
        ":!.venv",
    ]
    head = subprocess.run(
        ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
        capture_output=True,
        text=True, encoding="utf-8", errors="replace",
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(repo_dir), "status", "--porcelain", "--", ".", *exclude_specs],
        capture_output=True,
        text=True, encoding="utf-8", errors="replace",
    ).stdout.strip()
    diff = subprocess.run(
        ["git", "-C", str(repo_dir), "diff", "HEAD", "--", ".", *exclude_specs],
        capture_output=True,
        text=True, encoding="utf-8", errors="replace",
    ).stdout
    wt_out = subprocess.run(
        ["git", "-C", str(repo_dir), "worktree", "list", "--porcelain"],
        capture_output=True,
        text=True, encoding="utf-8", errors="replace",
    ).stdout
    wt_blocks = [b for b in wt_out.strip().split("\n\n") if b.strip()]
    wt_dirs: list[Path] = []
    extra_wt_resolved: set[Path] = set()
    for blk in wt_blocks:
        for ln in blk.splitlines():
            if ln.startswith("worktree "):
                cand = Path(ln[len("worktree ") :].strip())
                if cand.exists() and cand.resolve() != repo_dir.resolve():
                    wt_dirs.append(cand)
    spawned_wt_count = max(0, len(wt_blocks) - 1)
    if extra_worktrees_root and extra_worktrees_root.exists():
        dsh_wts = sorted([p for p in extra_worktrees_root.iterdir() if p.is_dir()])
        spawned_wt_count = max(spawned_wt_count, len(dsh_wts))
        known_res = {p.resolve() for p in wt_dirs}
        for p in dsh_wts:
            pres = p.resolve()
            extra_wt_resolved.add(pres)
            if pres not in known_res:
                wt_dirs.append(p)

    wt_mutation_entries: list[str] = []
    for wdir in sorted(wt_dirs, key=lambda p: str(p)):
        ref_head = (
            (extra_worktrees_base_head or head)
            if wdir.resolve() in extra_wt_resolved
            else head
        )
        w_head = subprocess.run(
            ["git", "-C", str(wdir), "rev-parse", "HEAD"],
            capture_output=True,
            text=True, encoding="utf-8", errors="replace",
        ).stdout.strip()
        w_stat = subprocess.run(
            ["git", "-C", str(wdir), "status", "--porcelain", "--", ".", *exclude_specs],
            capture_output=True,
            text=True, encoding="utf-8", errors="replace",
        ).stdout.strip()
        w_diff = subprocess.run(
            ["git", "-C", str(wdir), "diff", ref_head or "HEAD", "--", ".", *exclude_specs],
            capture_output=True,
            text=True, encoding="utf-8", errors="replace",
        ).stdout
        if (w_head and ref_head and w_head != ref_head) or w_stat or w_diff.strip():
            wt_mutation_entries.append(f"{wdir.name}:{w_head}:{w_stat}:{len(w_diff)}")

    wt_blob = "\n".join(wt_mutation_entries)
    digest = hashlib.sha256(f"{head}\n{status}\n{diff}\n{wt_blob}".encode("utf-8")).hexdigest()
    return {
        "repo_sha256": digest,
        "head": head,
        "dirty_lines": [ln for ln in status.splitlines() if ln.strip()],
        "dirty_worktrees": wt_mutation_entries,
        "spawned_worktrees": spawned_wt_count,
    }


@dataclass
class MaterializedEpisodeWorkspace:
    """Holds all live handles, paths, env vars, and initial snapshots for one episode."""

    episode: EpisodeSpec
    arm_name: str
    root_dir: Path
    repo_dir: Path
    mount_paths: dict[str, Path] = field(default_factory=dict)
    mount_pg_envs: dict[str, dict[str, str]] = field(default_factory=dict)
    extra_env: dict[str, str] = field(default_factory=dict)
    extra_repo_files: dict[str, tuple[str, int]] = field(default_factory=dict)
    shared_symlinks: dict[str, Path] = field(default_factory=dict)
    shared_links_dir: Path | None = None
    agents_md: str = ""
    gaia_server: Gaia2ServerProcess | None = None
    gaia_mount_id: str | None = None
    initial_snapshots: dict[str, dict[str, Any]] = field(default_factory=dict)
    initial_repo_snapshot: dict[str, Any] = field(default_factory=dict)
    mid_turn_snapshots: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)
    dsh_work_dir: Path | None = None
    dsh_base_head: str | None = None

    def take_mount_snapshot(self, mount_id: str) -> dict[str, Any]:
        mnt = next(m for m in self.episode.workspace_mounts if m.mount_id == mount_id)
        if mnt.pillar == "mcpmark_filesystem":
            return hash_directory_tree(self.mount_paths[mount_id])
        if mnt.pillar == "mcpmark_postgres":
            db_snap = hash_postgres_database(
                self.mount_pg_envs[mount_id],
                include_cluster_roles=episode_touches_pg_cluster_roles(self.episode),
            )
            mount_dir = self.mount_paths.get(mount_id)
            if mount_dir and mount_dir.exists():
                dir_snap = hash_directory_tree(mount_dir)
                combined = hashlib.sha256(
                    f"{db_snap['db_sha256']}:{dir_snap['tree_sha256']}".encode("utf-8")
                ).hexdigest()
                return {
                    **db_snap,
                    "raw_db_sha256": db_snap["db_sha256"],
                    "mount_dir_sha256": dir_snap["tree_sha256"],
                    "db_sha256": combined,
                }
            return db_snap
        if mnt.pillar == "gaia2":
            assert self.gaia_server is not None
            return self.gaia_server.snapshot()
        return hash_git_repo_state(
            self.repo_dir,
            extra_worktrees_root=self.dsh_work_dir,
            extra_worktrees_base_head=self.dsh_base_head,
        )

    def record_event_snapshots(self, after_event_id: str) -> None:
        for m in self.episode.workspace_mounts:
            self.mid_turn_snapshots[(after_event_id, m.mount_id)] = self.take_mount_snapshot(m.mount_id)

    def cleanup(self, remove_files: bool = False) -> None:
        ports_seen: set[int] = set()
        for pg_env in self.mount_pg_envs.values():
            try:
                p = int(pg_env["POSTGRES_PORT"])
                ports_seen.add(p)
                drop_postgres_db(pg_env["POSTGRES_DATABASE"], p)
            except Exception:
                pass
        if episode_touches_pg_cluster_roles(self.episode):
            for p in (ports_seen or {PG_DEFAULT_PORT}):
                try:
                    cleanup_postgres_cluster_roles(p)
                except Exception:
                    pass
        if remove_files and self.root_dir.exists():
            shutil.rmtree(self.root_dir, ignore_errors=True)


def materialize_episode_workspace(
    episode: EpisodeSpec,
    arm_name: str,
    run_root: Path,
    *,
    gaia_server: Gaia2ServerProcess | None = None,
    gaia_port: int = 19011,
    gaia_judge_mode: str = "auto",
    pg_port: int = PG_DEFAULT_PORT,
) -> MaterializedEpisodeWorkspace:
    """Materialize a complete, isolated Personal Agent workspace for `episode` and `arm_name`."""
    ep_root = (run_root / episode.episode_id / arm_name).resolve()
    if ep_root.exists():
        shutil.rmtree(ep_root)
    ep_root.mkdir(parents=True, exist_ok=True)

    repo_dir = ep_root / "repo"
    mount_paths: dict[str, Path] = {}
    mount_pg_envs: dict[str, dict[str, str]] = {}
    extra_env: dict[str, str] = {}
    extra_repo_files: dict[str, tuple[str, int]] = {}
    shared_symlinks: dict[str, Path] = {}
    agents_sections: list[str] = [
        f"# Unified Personal Agent Workspace (`{episode.episode_id}` — Track `{episode.track_id}`)",
        "",
        "You operate in a unified Personal Agent workspace that mounts code repositories, local office files, databases, and personal apps.",
        "Always respect read-only vs write instructions strictly: when asked for a read-only check/alignment, NEVER create, modify, delete, or send anything.",
        "",
    ]

    code_mount: WorkspaceMountSpec | None = next(
        (m for m in episode.workspace_mounts if m.pillar in ("cooperbench", "swe_interact", "ambig_swe")),
        None,
    )
    if code_mount is not None:
        bare_repo = ensure_cached_bare_repo(code_mount.repo_or_category, code_mount.base_commit or "")
        opaque_bare = _opaque_cache_link("repos", bare_repo)
        subprocess.run(
            ["git", "clone", "--shared", str(opaque_bare), str(repo_dir)],
            check=True,
            capture_output=True,
            text=True, encoding="utf-8", errors="replace",
        )
        alt_fp = repo_dir / ".git" / "objects" / "info" / "alternates"
        if alt_fp.exists():
            alt_fp.write_text(f"{opaque_bare / 'objects'}\n", encoding="utf-8")
        subprocess.run(
            ["git", "remote", "set-url", "origin", f"https://repo.local/{code_mount.repo_or_category}.git"],
            cwd=repo_dir,
            capture_output=True,
        )
        subprocess.run(["git", "config", "user.email", "pa@local"], cwd=repo_dir, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Personal Agent"], cwd=repo_dir, check=True, capture_output=True)
        if code_mount.base_commit:
            subprocess.run(
                ["git", "checkout", "-B", "main", code_mount.base_commit],
                cwd=repo_dir,
                check=True,
                capture_output=True,
                text=True, encoding="utf-8", errors="replace",
            )
        else:
            subprocess.run(["git", "checkout", "-B", "main"], cwd=repo_dir, check=True, capture_output=True)

        venv_dir = ensure_cached_repo_venv(code_mount.repo_or_category, code_mount.base_commit or "", repo_dir)
        opaque_venv = _opaque_cache_link("venvs", venv_dir)
        shared_symlinks[".venv"] = opaque_venv
        extra_env["REPO_PYTHON"] = str(opaque_venv / "bin" / "python")
        pyproj_fp = repo_dir / "pyproject.toml"
        pyproj_text = pyproj_fp.read_text(encoding="utf-8", errors="ignore") if pyproj_fp.exists() else ""
        if (
            "asyncio_default_fixture_loop_scope" not in pyproj_text
            and "--disable-plugin-autoload" not in pyproj_text
            and any(venv_dir.glob("lib/python*/site-packages/pytest_asyncio-*.dist-info"))
        ):
            for dist in venv_dir.glob("lib/python*/site-packages/pytest_asyncio-*.dist-info"):
                ver_str = dist.name.removeprefix("pytest_asyncio-").removesuffix(".dist-info")
                nums = [int(x) for x in re.findall(r"\d+", ver_str)[:2]]
                if tuple(nums) >= (0, 24):
                    extra_env["PYTEST_ADDOPTS"] = "-o asyncio_default_fixture_loop_scope=function"
                    break
        mount_paths[code_mount.mount_id] = repo_dir
        agents_sections.extend(
            [
                f"## Code Repository (`{code_mount.repo_or_category}`)",
                "- The target Git repository is checked out directly in your current working directory (`.` / worktree).",
                f"- Pre-configured virtualenv Python with `pytest` & dependencies: `{opaque_venv / 'bin' / 'python'} -m pytest` (also symlinked at `./.venv/bin/python`).",
                "",
            ]
        )
    else:
        repo_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "init", "-b", "main"], cwd=repo_dir, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "pa@local"], cwd=repo_dir, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Personal Agent"], cwd=repo_dir, check=True, capture_output=True)
        (repo_dir / ".workspace_root").write_text(f"{episode.episode_id}\n", encoding="utf-8")
        subprocess.run(["git", "add", ".workspace_root"], cwd=repo_dir, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "chore: init office workspace root"], cwd=repo_dir, check=True, capture_output=True)

    active_gaia_server: Gaia2ServerProcess | None = None
    gaia_mount_id: str | None = None

    for mnt in episode.workspace_mounts:
        seed_files: dict[str, str] = dict(mnt.setup_meta.get("seed_files") or {})
        if mnt.pillar in ("cooperbench", "swe_interact", "ambig_swe"):
            mount_paths[mnt.mount_id] = repo_dir
            for rel_fp, content in seed_files.items():
                extra_repo_files[rel_fp] = (str(content), 0o644)
            continue

        if mnt.pillar == "mcpmark_filesystem":
            category = mnt.repo_or_category.split("/", 1)[-1]
            tpl_dir = ensure_mcpmark_fs_template(category)
            fs_dir = ep_root / f"fs_{mnt.mount_id}"
            shutil.copytree(tpl_dir, fs_dir)
            marker = fs_dir / ".pa150_ready"
            if marker.exists():
                marker.unlink()
            for rel_fp, content in seed_files.items():
                sf = fs_dir / rel_fp
                sf.parent.mkdir(parents=True, exist_ok=True)
                sf.write_text(str(content), encoding="utf-8")
            mount_paths[mnt.mount_id] = fs_dir
            shared_symlinks["office_fs"] = fs_dir
            shared_symlinks[mnt.mount_id] = fs_dir
            extra_env["FILESYSTEM_TEST_DIR"] = str(fs_dir)
            agents_sections.extend(
                [
                    f"## Local Office Filesystem (`./office_fs` -> `{fs_dir}`)",
                    f"- Category: `{category}`. Perform requested file/folder operations directly inside `./office_fs` (`{fs_dir}`).",
                    "",
                ]
            )

        elif mnt.pillar == "mcpmark_postgres":
            if episode_touches_pg_cluster_roles(episode):
                cleanup_postgres_cluster_roles(pg_port)
            category = mnt.repo_or_category.split("/", 1)[-1]
            safe_ep = episode.episode_id.lower().replace("-", "_")
            safe_arm = arm_name.lower().replace("-", "_")
            safe_mnt = mnt.mount_id.lower().replace("-", "_")
            db_name = f"pa_{safe_ep}_{safe_arm}_{safe_mnt}"[:60]
            tdir = resolve_project_path(mnt.setup_meta["task_dir"]) if mnt.setup_meta.get("task_dir") else None
            pg_env = create_isolated_postgres_db(db_name, category, task_dir=tdir, port=pg_port)
            seed_pg_rows: list[dict[str, Any]] = list(mnt.setup_meta.get("seed_pg_spec_rows") or [])
            if seed_pg_rows:
                seed_conn = psycopg2.connect(
                    host=pg_env["POSTGRES_HOST"],
                    port=int(pg_env["POSTGRES_PORT"]),
                    database=pg_env["POSTGRES_DATABASE"],
                    user=pg_env["POSTGRES_USERNAME"],
                    password=pg_env["POSTGRES_PASSWORD"],
                )
                try:
                    with seed_conn:
                        with seed_conn.cursor() as scur:
                            scur.execute(
                                """
                                CREATE TABLE IF NOT EXISTS public.release_spec_addenda (
                                    spec_code TEXT PRIMARY KEY,
                                    target_repo TEXT NOT NULL,
                                    feature_title TEXT NOT NULL,
                                    requirements_md TEXT NOT NULL
                                )
                                """
                            )
                            for r_item in seed_pg_rows:
                                scur.execute(
                                    """
                                    INSERT INTO public.release_spec_addenda
                                        (spec_code, target_repo, feature_title, requirements_md)
                                    VALUES (%s, %s, %s, %s)
                                    ON CONFLICT (spec_code) DO UPDATE SET
                                        target_repo = EXCLUDED.target_repo,
                                        feature_title = EXCLUDED.feature_title,
                                        requirements_md = EXCLUDED.requirements_md
                                    """,
                                    (
                                        str(r_item["spec_code"]),
                                        str(r_item["target_repo"]),
                                        str(r_item["feature_title"]),
                                        str(r_item["requirements_md"]),
                                    ),
                                )
                finally:
                    seed_conn.close()
            mount_pg_envs[mnt.mount_id] = pg_env
            extra_env.update(pg_env)
            pg_cli_code = build_pg_exec_script(pg_env)
            extra_repo_files["pg_exec"] = (pg_cli_code, 0o755)
            pg_info_dir = ep_root / f"pg_{mnt.mount_id}"
            pg_info_dir.mkdir(parents=True, exist_ok=True)
            (pg_info_dir / "connection.json").write_text(json.dumps(pg_env, indent=2), encoding="utf-8")
            # P1-1: Provide `.pgpass` (`0600`) + `PGPASSFILE` so subprocesses that scrub `POSTGRES_PASSWORD` still authenticate seamlessly via `libpq`/`psycopg2`
            pgpass_fp = pg_info_dir / ".pgpass"
            pgpass_fp.write_text(
                f"{pg_env['POSTGRES_HOST']}:{pg_env['POSTGRES_PORT']}:*:{pg_env['POSTGRES_USERNAME']}:{pg_env['POSTGRES_PASSWORD']}\n",
                encoding="utf-8",
            )
            pgpass_fp.chmod(0o600)
            extra_env["PGPASSFILE"] = str(pgpass_fp)
            for rel_fp, content in seed_files.items():
                sf = pg_info_dir / rel_fp
                sf.parent.mkdir(parents=True, exist_ok=True)
                sf.write_text(str(content), encoding="utf-8")
            mount_paths[mnt.mount_id] = pg_info_dir
            shared_symlinks["office_pg"] = pg_info_dir
            agents_sections.extend(
                [
                    f"## Office PostgreSQL Database (`{db_name}`, category `{category}`)",
                    "- Inspect tables via `./pg_exec --tables`.",
                    "- Run SQL queries or DDL/DML via `./pg_exec \"SELECT ...\"` or `./pg_exec -f query.sql`.",
                    f"- Standard env vars (`POSTGRES_HOST={pg_env['POSTGRES_HOST']}`, `POSTGRES_PORT={pg_env['POSTGRES_PORT']}`, `POSTGRES_DATABASE={db_name}`, `POSTGRES_USERNAME={pg_env['POSTGRES_USERNAME']}`, `PGPASSFILE={pgpass_fp}`) and `./office_pg/connection.json` are configured for Python `psycopg2` scripts.",
                    "",
                ]
            )

        elif mnt.pillar == "gaia2":
            active_gaia_server = gaia_server or Gaia2ServerProcess(
                port=gaia_port,
                log_path=ep_root / "gaia2_env_server.log",
            )
            start_info = active_gaia_server.start_scenario(
                upstream_id=mnt.upstream_id,
                config=str(mnt.setup_meta.get("config", "")),
                scenario_id=mnt.repo_or_category,
                judge_mode=gaia_judge_mode,
                readonly_side_mount=bool(mnt.setup_meta.get("readonly_side_mount", False)),
            )
            gaia_mount_id = mnt.mount_id
            gaia_cli_code = build_gaia_cli_script(active_gaia_server.base_url)
            extra_repo_files["gaia_cli"] = (gaia_cli_code, 0o755)
            gaia_dir = ep_root / "gaia2_universe"
            gaia_dir.mkdir(parents=True, exist_ok=True)
            (gaia_dir / "scenario_info.json").write_text(json.dumps(start_info, indent=2), encoding="utf-8")
            mount_paths[mnt.mount_id] = gaia_dir
            shared_symlinks["gaia2_universe"] = gaia_dir
            agents_sections.extend(
                [
                    f"## Personal Apps Universe (`Gaia2` scenario `{mnt.repo_or_category}`)",
                    f"- Available Apps: `{', '.join(start_info.get('apps', []))}`.",
                    "- List all apps & tools: `./gaia_cli apps`",
                    "- Inspect tool arguments for an app: `./gaia_cli docs <AppName>` (e.g., `./gaia_cli docs Calendar` or `./gaia_cli docs Emails`)",
                    "- Call any app tool: `./gaia_cli call <AppName__tool_name> '{\"arg\": \"val\"}'`",
                    "- When finishing a Personal Apps (Gaia2) task or asking a required Personal Apps clarification question — and ONLY for Personal Apps tasks, NEVER for local office_fs or PostgreSQL tasks — deliver your final user message via `./gaia_cli call AgentUserInterface__send_message_to_user '{\"content\": \"...\"}'`.",
                    "",
                ]
            )

    agents_md = "\n".join(agents_sections).strip() + "\n"
    agents_md_fp = repo_dir / "AGENTS.md"
    if agents_md_fp.is_symlink():
        agents_md_fp.unlink()
    agents_md_fp.write_text(agents_md, encoding="utf-8")
    for rel_name, (fcontent, fmode) in extra_repo_files.items():
        fp = repo_dir / rel_name
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(fcontent, encoding="utf-8")
        fp.chmod(fmode)

    subprocess.run(["git", "add", "-f", "AGENTS.md"], cwd=repo_dir, check=True, capture_output=True)
    for rel_name in extra_repo_files:
        subprocess.run(["git", "add", "-f", rel_name], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "--allow-empty", "-m", "chore: attach pa150 workspace helpers"],
        cwd=repo_dir,
        check=True,
        capture_output=True,
    )

    # Attach shared symlinks into repo_dir and exclude them from git status/commits
    exclude_path = repo_dir / ".git" / "info" / "exclude"
    exclude_lines = set((exclude_path.read_text(encoding="utf-8") if exclude_path.exists() else "").splitlines())
    exclude_lines.update({".worktrees", ".shadow_drafts", "feishu_docs"})
    for link_name, link_target in shared_symlinks.items():
        exclude_lines.add(link_name)
        dst = repo_dir / link_name
        if not dst.exists() and not dst.is_symlink():
            dst.symlink_to(link_target.resolve())
        # Also make CLI wrappers available as symlinks when worktrees are spawned
    for rel_name in extra_repo_files:
        shared_symlinks[rel_name] = (repo_dir / rel_name).resolve()
    shared_symlinks["AGENTS.md"] = (repo_dir / "AGENTS.md").resolve()
    if (repo_dir / ".workspace_root").exists():
        shared_symlinks[".workspace_root"] = (repo_dir / ".workspace_root").resolve()
    exclude_path.write_text("\n".join(sorted(x for x in exclude_lines if x)) + "\n", encoding="utf-8")

    shared_links_dir = ep_root / "shared_links"
    shared_links_dir.mkdir(parents=True, exist_ok=True)
    for link_name, link_target in shared_symlinks.items():
        sl = shared_links_dir / link_name
        if not sl.exists() and not sl.is_symlink():
            sl.symlink_to(link_target.resolve())

    if sys.platform == "darwin":
        shim_bin_dir = ep_root / "bin"
        shim_bin_dir.mkdir(parents=True, exist_ok=True)
        sed_shim = shim_bin_dir / "sed"
        sed_shim.write_text(
            f"#!{PROJECT_ROOT / '.venv' / 'bin' / 'python'}\n"
            "import os, sys\n"
            "args = sys.argv[1:]\n"
            "out = []\n"
            "i = 0\n"
            "while i < len(args):\n"
            "    a = args[i]\n"
            "    if a == '-i':\n"
            "        out.append('-i')\n"
            "        if i + 1 >= len(args) or (args[i + 1] != '' and not args[i + 1].startswith('.')):\n"
            "            out.append('')\n"
            "    elif a.startswith('-i') and len(a) > 2:\n"
            "        out.extend(['-i', a[2:]])\n"
            "    else:\n"
            "        out.append(a)\n"
            "    i += 1\n"
            "os.execv('/usr/bin/sed', ['/usr/bin/sed', *out])\n",
            encoding="utf-8",
        )
        sed_shim.chmod(0o755)
        cur_p = extra_env.get("PATH") or os.environ.get("PATH", "")
        extra_env["PATH"] = f"{shim_bin_dir}{os.pathsep}{cur_p}" if cur_p else str(shim_bin_dir)

    ws = MaterializedEpisodeWorkspace(
        episode=episode,
        arm_name=arm_name,
        root_dir=ep_root,
        repo_dir=repo_dir,
        mount_paths=mount_paths,
        mount_pg_envs=mount_pg_envs,
        extra_env=extra_env,
        extra_repo_files=extra_repo_files,
        shared_symlinks=shared_symlinks,
        shared_links_dir=shared_links_dir,
        agents_md=agents_md,
        gaia_server=active_gaia_server,
        gaia_mount_id=gaia_mount_id,
    )
    for m in episode.workspace_mounts:
        ws.initial_snapshots[m.mount_id] = ws.take_mount_snapshot(m.mount_id)
    ws.initial_repo_snapshot = hash_git_repo_state(repo_dir)
    return ws


def _snapshot_identity_key(snap: dict[str, Any]) -> Any:
    if "tree_sha256" in snap:
        return snap["tree_sha256"]
    if "db_sha256" in snap:
        return snap["db_sha256"]
    if "apps_state_hash" in snap or "state_hash" in snap:
        return (snap.get("apps_state_hash") or snap.get("state_hash"), int(snap.get("mutating_call_count", 0)))
    if "repo_sha256" in snap:
        return snap["repo_sha256"]
    return json.dumps(snap, sort_keys=True)


def workspace_has_any_mutation_since_snapshot(
    ws: MaterializedEpisodeWorkspace,
    before_snaps: dict[str, dict[str, Any]],
    *,
    extra_worktrees_dir: Path | None = None,
) -> bool:
    """Return True if any mounted environment (or spawned git worktree) changed relative to `before_snaps`."""
    wt_root = extra_worktrees_dir or ws.dsh_work_dir
    for m in ws.episode.workspace_mounts:
        prev = before_snaps.get(m.mount_id)
        if prev is None:
            continue
        if m.pillar in ("cooperbench", "swe_interact", "ambig_swe") and wt_root is not None:
            cur = hash_git_repo_state(
                ws.repo_dir,
                extra_worktrees_root=wt_root,
                extra_worktrees_base_head=ws.dsh_base_head,
            )
        else:
            cur = ws.take_mount_snapshot(m.mount_id)
        if _snapshot_identity_key(cur) != _snapshot_identity_key(prev):
            return True
    return False


def workspace_has_any_mutation(
    ws: MaterializedEpisodeWorkspace,
    *,
    extra_worktrees_dir: Path | None = None,
) -> bool:
    """Return True if any mounted environment (or workspace git tree) has mutated since episode start."""
    if workspace_has_any_mutation_since_snapshot(
        ws,
        ws.initial_snapshots,
        extra_worktrees_dir=extra_worktrees_dir,
    ):
        return True
    if ws.initial_repo_snapshot:
        cur_repo = hash_git_repo_state(
            ws.repo_dir,
            extra_worktrees_root=extra_worktrees_dir or ws.dsh_work_dir,
            extra_worktrees_base_head=ws.dsh_base_head,
        )
        if cur_repo.get("repo_sha256") != ws.initial_repo_snapshot.get("repo_sha256"):
            return True
    return False
