#!/usr/bin/env python3
"""Local SQLite-backed `gh` and `lark-cli` wrappers for running 100% unmodified `dsh-superagent-prototype`.

Backed by:
- A real bare Git origin repository (`origin/billing_service.git`) + local clone (`clone/billing_service`)
- SQLite WAL database (`sandbox_db.sqlite`) for concurrent multi-worker safety
- Real-time NDJSON stream (`feishu_live_stream.ndjson`) for `<50ms` Feishu event ingestion
- Shared Feishu Markdown docs (`feishu_docs/doc_sla_2026.md`, `feishu_docs/doc_vip_review.md`)
"""
from __future__ import annotations

import json
import os
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

REPO_FULL_NAME = os.environ.get("DSH_SANDBOX_REPO", "workspace/project")
FEISHU_GROUP_CHAT_NAME = os.environ.get("DSH_SANDBOX_CHAT_NAME", "Workspace协作群")
AGENT_GITHUB_LOGIN = "superagent-bot"
AGENT_GITHUB_ID = 7001
FEISHU_APP_ID = "cli_superagent01"
FEISHU_BOT_OPEN_ID = "ou_superagentbot"
FEISHU_BOT_NAME = "superagent"
FEISHU_USER_OPEN_ID = "ou_viacol01"
FEISHU_USER_NAME = "viacol"
FEISHU_GROUP_CHAT_ID = "oc_rdbilling"
FEISHU_P2P_CHAT_ID = "oc_p2pviacol"


def _load_sandbox_meta(world_dir: Path) -> tuple[str, str]:
    global REPO_FULL_NAME, FEISHU_GROUP_CHAT_NAME
    meta_file = world_dir / "sandbox_meta.json"
    if meta_file.is_file():
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            if meta.get("repo_full_name"):
                REPO_FULL_NAME = str(meta["repo_full_name"])
            if meta.get("chat_name"):
                FEISHU_GROUP_CHAT_NAME = str(meta["chat_name"])
        except Exception:
            pass
    return REPO_FULL_NAME, FEISHU_GROUP_CHAT_NAME


def iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def epoch_ms_str() -> str:
    return str(int(time.time() * 1000))


def connect_db(world_dir: Path) -> sqlite3.Connection:
    db_path = world_dir / "sandbox_db.sqlite"
    conn = sqlite3.connect(str(db_path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=15000;")
    return conn


def init_db_schema(world_dir: Path) -> None:
    with connect_db(world_dir) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS issues (
                number INTEGER PRIMARY KEY AUTOINCREMENT,
                is_pr INTEGER NOT NULL DEFAULT 0,
                state TEXT NOT NULL DEFAULT 'open',
                draft INTEGER NOT NULL DEFAULT 0,
                merged INTEGER NOT NULL DEFAULT 0,
                title TEXT NOT NULL,
                body TEXT NOT NULL DEFAULT '',
                user_login TEXT NOT NULL,
                head_ref TEXT,
                head_sha TEXT,
                base_ref TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS comments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                issue_number INTEGER NOT NULL,
                kind TEXT NOT NULL DEFAULT 'conversation',
                body TEXT NOT NULL,
                user_login TEXT NOT NULL,
                in_reply_to_id INTEGER,
                path TEXT,
                line INTEGER,
                state TEXT,
                commit_id TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS notifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                issue_number INTEGER NOT NULL,
                is_pr INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS feishu_messages (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                message_id TEXT UNIQUE NOT NULL,
                chat_id TEXT NOT NULL,
                chat_type TEXT NOT NULL,
                sender_id TEXT NOT NULL,
                sender_name TEXT NOT NULL,
                sender_type TEXT NOT NULL,
                content TEXT NOT NULL,
                create_time_ms TEXT NOT NULL,
                root_id TEXT,
                parent_id TEXT,
                thread_id TEXT,
                mentions_json TEXT NOT NULL DEFAULT '[]'
            );
            CREATE TABLE IF NOT EXISTS bot_replies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                reply_to_message_id TEXT NOT NULL,
                chat_id TEXT NOT NULL,
                out_message_id TEXT NOT NULL,
                markdown TEXT NOT NULL,
                idempotency_key TEXT UNIQUE,
                created_at TEXT NOT NULL
            );
            """
        )


def init_dsh_sandbox(
    world_dir: Path,
    agents_md_override: str | None = None,
    extra_repo_files: dict[str, tuple[str, int]] | None = None,
    seed_repo_dir: Path | None = None,
    repo_full_name: str | None = None,
    chat_name: str | None = None,
) -> dict[str, Path]:
    """Create a fresh isolated Git + GitHub + Feishu sandbox for `dsh-superagent`."""
    world_dir = world_dir.resolve()
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from sandbox_world.seed_workspace import create_sandbox_world

    if world_dir.exists():
        shutil.rmtree(world_dir)
    world_dir.mkdir(parents=True, exist_ok=True)

    eff_repo_name = repo_full_name or ("acme/billing_service" if seed_repo_dir is None else REPO_FULL_NAME)
    eff_chat_name = chat_name or ("RD-Billing研发群" if seed_repo_dir is None else FEISHU_GROUP_CHAT_NAME)
    (world_dir / "sandbox_meta.json").write_text(
        json.dumps({"repo_full_name": eff_repo_name, "chat_name": eff_chat_name}, ensure_ascii=False),
        encoding="utf-8",
    )

    if seed_repo_dir is not None:
        seed_base = world_dir / "seed"
        seed_repo = seed_base / "repo"
        seed_docs = seed_base / "feishu_docs"
        seed_base.mkdir(parents=True, exist_ok=True)
        seed_docs.mkdir(parents=True, exist_ok=True)
        (seed_base / "docs").mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", str(seed_repo_dir.resolve()), str(seed_repo)],
            check=True,
            capture_output=True,
        )
        subprocess.run(["git", "config", "user.email", "pa-seed@local"], cwd=seed_repo, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "PA Seed"], cwd=seed_repo, check=True, capture_output=True)
        subprocess.run(["git", "checkout", "-B", "main"], cwd=seed_repo, check=True, capture_output=True)
        seed_paths = {"repo": seed_repo, "docs": seed_docs}
        default_agents_md = "# Workspace Notes\n\n- Inspect workspace files and run verification checks via `bash`.\n"
    else:
        seed_paths = create_sandbox_world(world_dir / "seed")
        default_agents_md = (
            "# billing_service Repository Notes\n\n"
            "- Python package: `billing_service/` (`exporter.py`, `fx_client.py`), unit tests in `tests/`.\n"
            "- Shared Feishu documents (`doc_sla_2026`, `doc_vip_review`) are synced in `./feishu_docs/<doc_id>.md` "
            "and also readable via `lark-cli docs get <doc_id>`.\n"
        )

    agents_md = agents_md_override or default_agents_md
    agents_md_fp = seed_paths["repo"] / "AGENTS.md"
    if agents_md_fp.is_symlink():
        agents_md_fp.unlink()
    agents_md_fp.write_text(agents_md, encoding="utf-8")
    for rel_name, (fcontent, fmode) in (extra_repo_files or {}).items():
        fp = seed_paths["repo"] / rel_name
        if fp.is_symlink():
            fp.unlink()
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(fcontent, encoding="utf-8")
        fp.chmod(fmode)
    subprocess.run(["git", "add", "-f", "AGENTS.md"], cwd=seed_paths["repo"], check=True, capture_output=True)
    for rel_name in (extra_repo_files or {}):
        subprocess.run(["git", "add", "-f", rel_name], cwd=seed_paths["repo"], check=True, capture_output=True)
    # P0-1: Only create a commit if staging actually changed files (preserves `SHA_A` when `seed_repo_dir` already committed identical helpers)
    has_staged_diff = (
        subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=seed_paths["repo"],
            capture_output=True,
        ).returncode
        != 0
    )
    if has_staged_diff:
        subprocess.run(
            [
                "git",
                "-c",
                "commit.gpgsign=false",
                "commit",
                "--no-verify",
                "-m",
                "docs: add repo AGENTS.md and cli helpers",
            ],
            cwd=seed_paths["repo"],
            check=True,
            capture_output=True,
        )

    bare_repo = world_dir / "origin" / "repo.git"
    bare_repo.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--bare", str(seed_paths["repo"]), str(bare_repo)],
        check=True,
        capture_output=True,
    )

    clone_repo = world_dir / "clone" / "repo"
    clone_repo.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", f"file://{bare_repo}", str(clone_repo)],
        check=True,
        capture_output=True,
    )

    init_db_schema(world_dir)
    live_stream = world_dir / "feishu_live_stream.ndjson"
    live_stream.write_text("", encoding="utf-8")

    bin_dir = world_dir / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    py_bin = str(project_root / ".venv" / "bin" / "python")
    self_script = str(Path(__file__).resolve())

    gh_wrap = bin_dir / "sandbox-gh"
    gh_wrap.write_text(
        f"#!/bin/sh\nexec '{py_bin}' '{self_script}' gh '{world_dir}' \"$@\"\n",
        encoding="utf-8",
    )
    gh_wrap.chmod(0o755)
    gh_alias = bin_dir / "gh"
    if gh_alias.exists() or gh_alias.is_symlink():
        gh_alias.unlink()
    gh_alias.symlink_to(gh_wrap.name)

    lark_wrap = bin_dir / "sandbox-lark-cli"
    lark_wrap.write_text(
        f"#!/bin/sh\nexec '{py_bin}' '{self_script}' lark '{world_dir}' \"$@\"\n",
        encoding="utf-8",
    )
    lark_wrap.chmod(0o755)
    lark_alias = bin_dir / "lark-cli"
    if lark_alias.exists() or lark_alias.is_symlink():
        lark_alias.unlink()
    lark_alias.symlink_to(lark_wrap.name)

    curl_wrap = bin_dir / "curl"
    curl_wrap.write_text(
        "#!/bin/sh\n"
        "case \"$*\" in\n"
        "  *github.com*|*GITHUB.COM*|*githubusercontent.com*|*GITHUBUSERCONTENT.COM*)\n"
        "    echo 'Error: Sandbox policy blocked outbound network access to upstream GitHub repositories/PRs/commits.' >&2\n"
        "    exit 1\n"
        "    ;;\n"
        "esac\n"
        "exec /usr/bin/curl \"$@\"\n",
        encoding="utf-8",
    )
    curl_wrap.chmod(0o755)

    wget_wrap = bin_dir / "wget"
    wget_wrap.write_text(
        "#!/bin/sh\n"
        "case \"$*\" in\n"
        "  *github.com*|*GITHUB.COM*|*githubusercontent.com*|*GITHUBUSERCONTENT.COM*)\n"
        "    echo 'Error: Sandbox policy blocked outbound network access to upstream GitHub repositories/PRs/commits.' >&2\n"
        "    exit 1\n"
        "    ;;\n"
        "esac\n"
        "for w in /opt/homebrew/bin/wget /usr/local/bin/wget /usr/bin/wget; do\n"
        "  if [ -x \"$w\" ]; then exec \"$w\" \"$@\"; fi\n"
        "done\n"
        "echo 'wget: command not found' >&2\n"
        "exit 127\n",
        encoding="utf-8",
    )
    wget_wrap.chmod(0o755)

    return {
        "world": world_dir,
        "bare_repo": bare_repo,
        "clone_repo": clone_repo,
        "docs": seed_paths["docs"],
        "gh_cli": gh_wrap,
        "lark_cli": lark_wrap,
        "live_stream": live_stream,
    }


# ---------------------------------------------------------------------------
# Benchmark Harness Helpers (inject events & inspect outputs)
# ---------------------------------------------------------------------------


def push_feishu_message(
    world_dir: Path,
    text: str,
    *,
    chat_id: str = FEISHU_P2P_CHAT_ID,
    chat_type: str = "p2p",
    sender_id: str = FEISHU_USER_OPEN_ID,
    sender_name: str = FEISHU_USER_NAME,
    sender_type: str = "user",
    mention_bot: bool = False,
    root_id: str | None = None,
    parent_id: str | None = None,
    thread_id: str | None = None,
    message_id: str | None = None,
) -> str:
    mid = message_id or f"om_usr_{int(time.time() * 1000)}_{os.getpid()}_{int.from_bytes(os.urandom(2), 'big')}"
    ts_ms = epoch_ms_str()
    mentions: list[dict[str, str]] = []
    wire_text = text
    if mention_bot:
        mentions.append({"id": FEISHU_BOT_OPEN_ID, "key": "@_user_1", "name": FEISHU_BOT_NAME})
        if "@_user_1" not in wire_text:
            wire_text = f"@_user_1 {wire_text}"

    with connect_db(world_dir) as conn:
        conn.execute(
            """
            INSERT INTO feishu_messages
            (message_id, chat_id, chat_type, sender_id, sender_name, sender_type, content, create_time_ms, root_id, parent_id, thread_id, mentions_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                mid,
                chat_id,
                chat_type,
                sender_id,
                sender_name,
                sender_type,
                wire_text,
                ts_ms,
                root_id,
                parent_id,
                thread_id,
                json.dumps(mentions, ensure_ascii=False),
            ),
        )
        conn.commit()

    event_obj: dict[str, Any] = {
        "message_id": mid,
        "chat_id": chat_id,
        "chat_type": chat_type,
        "sender_id": sender_id,
        "sender_type": sender_type,
        "content": wire_text,
        "create_time": ts_ms,
        "mentions": mentions,
        "message_app_link": f"https://applink.feishu.cn/client/chat/open?openChatId={chat_id}&messageId={mid}",
    }
    if root_id:
        event_obj["root_id"] = root_id
    if parent_id:
        event_obj["parent_id"] = parent_id
    if thread_id:
        event_obj["thread_id"] = thread_id

    stream_file = world_dir / "feishu_live_stream.ndjson"
    with stream_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event_obj, ensure_ascii=False) + "\n")
        f.flush()
    return mid


def create_github_issue(
    world_dir: Path,
    title: str,
    body: str,
    *,
    user_login: str = "viacol",
) -> int:
    now = iso_now()
    with connect_db(world_dir) as conn:
        cur = conn.execute(
            """
            INSERT INTO issues (is_pr, state, draft, merged, title, body, user_login, created_at, updated_at)
            VALUES (0, 'open', 0, 0, ?, ?, ?, ?, ?)
            """,
            (title, body, user_login, now, now),
        )
        num = int(cur.lastrowid)
        conn.execute(
            "INSERT INTO notifications (issue_number, is_pr, updated_at) VALUES (?, 0, ?)",
            (num, now),
        )
        conn.commit()
    return num


def feishu_chat_id(channel: str) -> str:
    if channel.startswith("feishu://group/"):
        return "oc_" + re.sub(r"[^a-zA-Z0-9]", "", channel.removeprefix("feishu://group/"))
    return FEISHU_P2P_CHAT_ID


def configure_feishu_group_channels(world_dir: Path, channels: list[str]) -> list[dict[str, str]]:
    chats = [{"chat_id": feishu_chat_id(channel), "name": channel.rsplit("/", 1)[-1]}
             for channel in sorted(set(channels))]
    meta_path = world_dir / "sandbox_meta.json"
    meta = json.loads(meta_path.read_text())
    meta["group_chats"] = chats
    meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    return chats


def list_feishu_chats(world_dir: Path) -> list[dict[str, str]]:
    meta_path = world_dir / "sandbox_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    groups = meta.get("group_chats")
    if groups is None:
        groups = [{"chat_id": FEISHU_GROUP_CHAT_ID, "name": meta.get("chat_name", FEISHU_GROUP_CHAT_NAME)}]
    return [*groups, {"chat_id": FEISHU_P2P_CHAT_ID, "name": "viacol-p2p"}]


def list_bot_replies(world_dir: Path) -> list[dict[str, Any]]:
    with connect_db(world_dir) as conn:
        rows = conn.execute("SELECT * FROM bot_replies ORDER BY id ASC").fetchall()
        return [{**dict(r), "parent_id": r["reply_to_message_id"]} for r in rows]


def list_github_prs(world_dir: Path) -> list[dict[str, Any]]:
    with connect_db(world_dir) as conn:
        rows = conn.execute("SELECT * FROM issues WHERE is_pr = 1 ORDER BY number ASC").fetchall()
        return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# `gh` CLI implementation
# ---------------------------------------------------------------------------


def _run_jq(data_json: str, jq_expr: str | None) -> str:
    if not jq_expr:
        return data_json
    jq_bin = shutil.which("jq") or "/opt/homebrew/bin/jq"
    proc = subprocess.run(
        [jq_bin, "-r", jq_expr],
        input=data_json,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr or "jq error\n")
        sys.exit(1)
    return proc.stdout


def _issue_json(row: sqlite3.Row) -> dict[str, Any]:
    is_pr = bool(row["is_pr"])
    num = int(row["number"])
    page_kind = "pull" if is_pr else "issues"
    html_url = f"https://github.com/{REPO_FULL_NAME}/{page_kind}/{num}"
    obj: dict[str, Any] = {
        "id": 10000 + num,
        "number": num,
        "title": row["title"],
        "body": row["body"] or "",
        "state": row["state"],
        "draft": bool(row["draft"]),
        "merged": bool(row["merged"]),
        "user": {"login": row["user_login"], "type": "User"},
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "html_url": html_url,
        "labels": [],
    }
    if is_pr:
        obj["pull_request"] = {"url": f"https://api.github.com/repos/{REPO_FULL_NAME}/pulls/{num}"}
        obj["head"] = {
            "ref": row["head_ref"] or "main",
            "sha": row["head_sha"] or "0" * 40,
            "repo": {"full_name": REPO_FULL_NAME},
        }
        obj["base"] = {"ref": row["base_ref"] or "main"}
    return obj


def _comment_json(row: sqlite3.Row, is_pr: bool = False) -> dict[str, Any]:
    cid = int(row["id"])
    num = int(row["issue_number"])
    page_kind = "pull" if is_pr else "issues"
    return {
        "id": cid,
        "body": row["body"] or "",
        "user": {"login": row["user_login"], "type": "User"},
        "html_url": f"https://github.com/{REPO_FULL_NAME}/{page_kind}/{num}#issuecomment-{cid}",
        "issue_url": f"https://api.github.com/repos/{REPO_FULL_NAME}/issues/{num}",
        "pull_request_url": f"https://api.github.com/repos/{REPO_FULL_NAME}/pulls/{num}" if is_pr else None,
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "submitted_at": row["updated_at"],
        "in_reply_to_id": row["in_reply_to_id"],
        "path": row["path"],
        "line": row["line"],
        "state": row["state"],
        "commit_id": row["commit_id"],
    }


def _git_head_sha(bare_repo: Path, ref: str = "HEAD") -> str:
    for candidate in (ref, f"refs/heads/{ref}", "HEAD"):
        proc = subprocess.run(
            ["git", "-C", str(bare_repo), "rev-parse", candidate],
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.strip()
    return "a" * 40


def handle_gh(world_dir: Path, args: list[str]) -> None:
    _load_sandbox_meta(world_dir)
    bare_repo = world_dir / "origin" / "repo.git"
    if not bare_repo.exists():
        bare_repo = world_dir / "origin" / "billing_service.git"
    # Strip `-R` / `--repo` global option if present anywhere
    cleaned: list[str] = []
    i = 0
    while i < len(args):
        if args[i] in ("-R", "--repo") and i + 1 < len(args):
            i += 2
            continue
        if args[i].startswith("--repo="):
            i += 1
            continue
        cleaned.append(args[i])
        i += 1

    if not cleaned or cleaned[0] in ("-h", "--help", "help"):
        print("sandbox-gh: supports api, pr, issue, repo, auth")
        return

    cmd = cleaned[0]
    sub = cleaned[1:]

    if cmd == "auth":
        if sub and sub[0] == "git-credential":
            return
        print(f"Logged in to github.com account {AGENT_GITHUB_LOGIN}")
        return

    if cmd == "repo":
        print(json.dumps({"nameWithOwner": REPO_FULL_NAME, "defaultBranchRef": {"name": "main"}}))
        return

    if cmd == "api":
        method = "GET"
        jq_expr: str | None = None
        include_headers = False
        slurp = False
        fields: dict[str, str] = {}
        headers: list[str] = []
        endpoint = ""
        j = 0
        while j < len(sub):
            a = sub[j]
            if a in ("--method", "-X") and j + 1 < len(sub):
                method = sub[j + 1].upper()
                j += 2
            elif a == "--jq" and j + 1 < len(sub):
                jq_expr = sub[j + 1]
                j += 2
            elif a in ("--include", "-i"):
                include_headers = True
                j += 1
            elif a == "--paginate":
                j += 1
            elif a == "--slurp":
                slurp = True
                j += 1
            elif a in ("-f", "-F", "--raw-field", "--field") and j + 1 < len(sub):
                kv = sub[j + 1]
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    fields[k] = v
                if method == "GET":
                    method = "POST"
                j += 2
            elif a in ("-H", "--header") and j + 1 < len(sub):
                headers.append(sub[j + 1])
                j += 2
            elif not a.startswith("-") and not endpoint:
                endpoint = a.lstrip("/")
                j += 1
            else:
                j += 1

        parsed_url = urlparse(endpoint)
        path = parsed_url.path.strip("/")
        qs = parse_qs(parsed_url.query)

        def emit(payload: Any, raw_text: str | None = None) -> None:
            if raw_text is not None:
                if include_headers:
                    sys.stdout.write("HTTP/2 200\r\nx-poll-interval: 60\r\n\r\n")
                sys.stdout.write(raw_text)
                return
            out_obj = [payload] if slurp else payload
            txt = json.dumps(out_obj, ensure_ascii=False)
            txt = _run_jq(txt, jq_expr)
            if include_headers:
                sys.stdout.write("HTTP/2 200\r\nx-poll-interval: 60\r\ncontent-type: application/json\r\n\r\n")
            sys.stdout.write(txt)
            if not txt.endswith("\n"):
                sys.stdout.write("\n")

        if path == "user":
            emit({"login": AGENT_GITHUB_LOGIN, "id": AGENT_GITHUB_ID, "type": "User"})
            return
        if path.startswith("users/"):
            login = path.split("/", 1)[1]
            uid = AGENT_GITHUB_ID if login == AGENT_GITHUB_LOGIN else 1001
            emit({"login": login, "id": uid, "type": "User"})
            return
        if path == "notifications":
            since = qs.get("since", ["1970-01-01T00:00:00Z"])[0]
            with connect_db(world_dir) as conn:
                rows = conn.execute(
                    "SELECT * FROM notifications WHERE updated_at >= ? ORDER BY id DESC LIMIT 50",
                    (since,),
                ).fetchall()
            items = [
                {
                    "id": str(r["id"]),
                    "updated_at": r["updated_at"],
                    "repository": {"full_name": REPO_FULL_NAME},
                    "subject": {
                        "url": f"https://api.github.com/repos/{REPO_FULL_NAME}/{'pulls' if r['is_pr'] else 'issues'}/{r['issue_number']}"
                    },
                }
                for r in rows
            ]
            emit(items)
            return

        repo_prefix = f"repos/{REPO_FULL_NAME}"
        if path == repo_prefix:
            emit(
                {
                    "full_name": REPO_FULL_NAME,
                    "name": REPO_FULL_NAME.split("/")[-1],
                    "default_branch": "main",
                    "permissions": {"push": True, "pull": True, "admin": False},
                }
            )
            return

        rel = path[len(repo_prefix) :].lstrip("/") if path.startswith(repo_prefix) else path

        # Commits & CI checks
        m_checks = re.match(r"^commits/([^/]+)/check-runs$", rel)
        if m_checks:
            emit(
                {
                    "total_count": 1,
                    "check_runs": [{"name": "ci/pytest", "status": "completed", "conclusion": "success"}],
                }
            )
            return
        m_stat = re.match(r"^commits/([^/]+)/statuses$", rel)
        if m_stat:
            emit([])
            return
        m_commit = re.match(r"^commits/([^/]+)$", rel)
        if m_commit:
            emit({"sha": _git_head_sha(bare_repo, m_commit.group(1))})
            return

        # Contents
        if rel == "contents" or rel.startswith("contents/"):
            subpath = rel[len("contents") :].lstrip("/")
            want_raw = any("vnd.github.raw" in h for h in headers)
            ls_proc = subprocess.run(
                ["git", "-C", str(bare_repo), "ls-tree", "HEAD", subpath] if subpath else ["git", "-C", str(bare_repo), "ls-tree", "HEAD"],
                capture_output=True,
                text=True,
            )
            if subpath:
                show_proc = subprocess.run(
                    ["git", "-C", str(bare_repo), "show", f"HEAD:{subpath}"],
                    capture_output=True,
                    text=True,
                )
                if show_proc.returncode == 0 and not show_proc.stdout.startswith("tree "):
                    if want_raw:
                        emit(None, raw_text=show_proc.stdout)
                    else:
                        emit({"name": Path(subpath).name, "path": subpath, "type": "file", "content": show_proc.stdout})
                    return
                tree_proc = subprocess.run(
                    ["git", "-C", str(bare_repo), "ls-tree", f"HEAD:{subpath}"],
                    capture_output=True,
                    text=True,
                )
                if tree_proc.returncode == 0:
                    entries = [{"name": line.split("\t", 1)[1]} for line in tree_proc.stdout.splitlines() if "\t" in line]
                    emit(entries)
                    return
            else:
                entries = [{"name": line.split("\t", 1)[1]} for line in ls_proc.stdout.splitlines() if "\t" in line]
                emit(entries)
                return
            sys.stderr.write("gh: Not Found (HTTP 404)\n")
            sys.exit(1)

        # Issues & PRs list
        if rel in ("issues", "pulls"):
            if method == "POST":
                now = iso_now()
                is_pr = 1 if (rel == "pulls" or "head" in fields) else 0
                head_ref = fields.get("head", "main")
                head_sha = _git_head_sha(bare_repo, head_ref)
                with connect_db(world_dir) as conn:
                    cur = conn.execute(
                        """
                        INSERT INTO issues (is_pr, state, draft, merged, title, body, user_login, head_ref, head_sha, base_ref, created_at, updated_at)
                        VALUES (?, 'open', ?, 0, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            is_pr,
                            1 if fields.get("draft") in ("true", "True", "1") else 0,
                            fields.get("title", "Untitled"),
                            fields.get("body", ""),
                            AGENT_GITHUB_LOGIN,
                            head_ref,
                            head_sha,
                            fields.get("base", "main"),
                            now,
                            now,
                        ),
                    )
                    num = int(cur.lastrowid)
                    row = conn.execute("SELECT * FROM issues WHERE number = ?", (num,)).fetchone()
                    conn.commit()
                emit(_issue_json(row))
                return
            with connect_db(world_dir) as conn:
                if rel == "pulls":
                    rows = conn.execute("SELECT * FROM issues WHERE is_pr = 1 ORDER BY number ASC").fetchall()
                else:
                    rows = conn.execute("SELECT * FROM issues ORDER BY number ASC").fetchall()
            emit([_issue_json(r) for r in rows])
            return

        # Repo-wide comments
        if rel in ("issues/comments", "pulls/comments"):
            kind_filter = "inline" if rel == "pulls/comments" else "conversation"
            with connect_db(world_dir) as conn:
                rows = conn.execute("SELECT * FROM comments WHERE kind = ? ORDER BY id ASC", (kind_filter,)).fetchall()
            emit([_comment_json(r, is_pr=(rel == "pulls/comments")) for r in rows])
            return

        # Single issue or PR
        m_item = re.match(r"^(issues|pulls)/(\d+)$", rel)
        if m_item:
            num = int(m_item.group(2))
            with connect_db(world_dir) as conn:
                row = conn.execute("SELECT * FROM issues WHERE number = ?", (num,)).fetchone()
                if not row:
                    sys.stderr.write("gh: Not Found (HTTP 404)\n")
                    sys.exit(1)
                if method in ("PATCH", "POST"):
                    now = iso_now()
                    title = fields.get("title", row["title"])
                    body = fields.get("body", row["body"])
                    state = fields.get("state", row["state"])
                    conn.execute(
                        "UPDATE issues SET title = ?, body = ?, state = ?, updated_at = ? WHERE number = ?",
                        (title, body, state, now, num),
                    )
                    conn.commit()
                    row = conn.execute("SELECT * FROM issues WHERE number = ?", (num,)).fetchone()
                elif row["is_pr"] and row["head_ref"]:
                    # Refresh head_sha dynamically in case worker pushed new commits!
                    latest_sha = _git_head_sha(bare_repo, row["head_ref"])
                    if latest_sha != row["head_sha"]:
                        conn.execute("UPDATE issues SET head_sha = ? WHERE number = ?", (latest_sha, num))
                        conn.commit()
                        row = conn.execute("SELECT * FROM issues WHERE number = ?", (num,)).fetchone()
            emit(_issue_json(row))
            return

        # Issue/PR comments & reviews
        m_sub = re.match(r"^(issues|pulls)/(\d+)/(comments|reviews)$", rel)
        if m_sub:
            kind_url, num_s, sec = m_sub.groups()
            num = int(num_s)
            if method == "POST":
                now = iso_now()
                ckind = "review" if sec == "reviews" else ("inline" if kind_url == "pulls" else "conversation")
                with connect_db(world_dir) as conn:
                    cur = conn.execute(
                        """
                        INSERT INTO comments (issue_number, kind, body, user_login, state, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (num, ckind, fields.get("body", ""), AGENT_GITHUB_LOGIN, fields.get("event"), now, now),
                    )
                    cid = int(cur.lastrowid)
                    row = conn.execute("SELECT * FROM comments WHERE id = ?", (cid,)).fetchone()
                    conn.commit()
                emit(_comment_json(row, is_pr=(kind_url == "pulls")))
                return
            ckind = "review" if sec == "reviews" else ("inline" if kind_url == "pulls" else "conversation")
            with connect_db(world_dir) as conn:
                rows = conn.execute(
                    "SELECT * FROM comments WHERE issue_number = ? AND kind = ? ORDER BY id ASC",
                    (num, ckind),
                ).fetchall()
            emit([_comment_json(r, is_pr=(kind_url == "pulls")) for r in rows])
            return

        sys.stderr.write(f"gh: Not Found (HTTP 404) for {path}\n")
        sys.exit(1)

    if cmd == "pr":
        if not sub or sub[0] in ("-h", "--help"):
            print("gh pr commands: create, view, list, edit, comment, checks, ready, close, diff")
            return
        pr_sub = sub[0]
        rest = sub[1:]
        if pr_sub == "create":
            title = ""
            body = ""
            base = "main"
            head = ""
            draft = 0
            k = 0
            while k < len(rest):
                a = rest[k]
                if a in ("-t", "--title") and k + 1 < len(rest):
                    title = rest[k + 1]
                    k += 2
                elif a in ("-b", "--body") and k + 1 < len(rest):
                    body = rest[k + 1]
                    k += 2
                elif a in ("-F", "--body-file") and k + 1 < len(rest):
                    bf = rest[k + 1]
                    body = sys.stdin.read() if bf == "-" else Path(bf).read_text(encoding="utf-8")
                    k += 2
                elif a in ("-B", "--base") and k + 1 < len(rest):
                    base = rest[k + 1]
                    k += 2
                elif a in ("-H", "--head") and k + 1 < len(rest):
                    head = rest[k + 1]
                    k += 2
                elif a in ("-d", "--draft"):
                    draft = 1
                    k += 1
                else:
                    k += 1
            if not head:
                br = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True)
                head = br.stdout.strip() if br.returncode == 0 else "main"
            if not title:
                lg = subprocess.run(["git", "log", "-1", "--pretty=%s"], capture_output=True, text=True)
                title = lg.stdout.strip() if lg.returncode == 0 and lg.stdout.strip() else head
            sha_local = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
            head_sha = sha_local.stdout.strip() if sha_local.returncode == 0 else _git_head_sha(bare_repo, head)
            now = iso_now()
            with connect_db(world_dir) as conn:
                cur = conn.execute(
                    """
                    INSERT INTO issues (is_pr, state, draft, merged, title, body, user_login, head_ref, head_sha, base_ref, created_at, updated_at)
                    VALUES (1, 'open', ?, 0, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (draft, title, body, AGENT_GITHUB_LOGIN, head, head_sha, base, now, now),
                )
                num = int(cur.lastrowid)
                conn.commit()
            print(f"https://github.com/{REPO_FULL_NAME}/pull/{num}")
            return

        if pr_sub in ("view", "list", "edit", "ready", "close", "checks", "comment", "diff"):
            json_fields: list[str] | None = None
            jq_expr = None
            target: str | None = None
            title_edit: str | None = None
            body_edit: str | None = None
            head_filter: str | None = None
            k = 0
            while k < len(rest):
                a = rest[k]
                if a == "--json" and k + 1 < len(rest):
                    json_fields = [x.strip() for x in rest[k + 1].split(",") if x.strip()]
                    k += 2
                elif a == "--jq" and k + 1 < len(rest):
                    jq_expr = rest[k + 1]
                    k += 2
                elif a in ("-t", "--title") and k + 1 < len(rest):
                    title_edit = rest[k + 1]
                    k += 2
                elif a in ("-b", "--body") and k + 1 < len(rest):
                    body_edit = rest[k + 1]
                    k += 2
                elif a in ("-F", "--body-file") and k + 1 < len(rest):
                    bf = rest[k + 1]
                    body_edit = sys.stdin.read() if bf == "-" else Path(bf).read_text(encoding="utf-8")
                    k += 2
                elif a in ("-H", "--head") and k + 1 < len(rest):
                    head_filter = rest[k + 1]
                    k += 2
                elif not a.startswith("-") and target is None:
                    target = a.lstrip("#")
                    k += 1
                else:
                    k += 1

            def _pr_cli_obj(r: sqlite3.Row) -> dict[str, Any]:
                num = int(r["number"])
                full = {
                    "number": num,
                    "title": r["title"],
                    "body": r["body"] or "",
                    "state": r["state"].upper(),
                    "isDraft": bool(r["draft"]),
                    "url": f"https://github.com/{REPO_FULL_NAME}/pull/{num}",
                    "headRefName": r["head_ref"] or "main",
                    "headRefOid": r["head_sha"] or _git_head_sha(bare_repo, r["head_ref"] or "main"),
                    "baseRefName": r["base_ref"] or "main",
                    "author": {"login": r["user_login"]},
                    "mergedAt": None,
                    "closedAt": None,
                }
                if json_fields is not None:
                    return {f: full.get(f) for f in json_fields}
                return full

            with connect_db(world_dir) as conn:
                if pr_sub == "list":
                    q = "SELECT * FROM issues WHERE is_pr = 1"
                    params: list[Any] = []
                    if head_filter:
                        q += " AND head_ref = ?"
                        params.append(head_filter)
                    rows = conn.execute(q + " ORDER BY number DESC", params).fetchall()
                    out = json.dumps([_pr_cli_obj(r) for r in rows], ensure_ascii=False)
                    sys.stdout.write(_run_jq(out, jq_expr) + ("\n" if not jq_expr else ""))
                    return

                row = None
                if target and target.isdigit():
                    row = conn.execute("SELECT * FROM issues WHERE is_pr = 1 AND number = ?", (int(target),)).fetchone()
                else:
                    br_name = target
                    if not br_name:
                        br = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True)
                        br_name = br.stdout.strip() if br.returncode == 0 else ""
                    if br_name:
                        row = conn.execute(
                            "SELECT * FROM issues WHERE is_pr = 1 AND head_ref = ? ORDER BY number DESC LIMIT 1",
                            (br_name,),
                        ).fetchone()
                if not row:
                    sys.stderr.write("no pull requests found\n")
                    sys.exit(1)
                num = int(row["number"])
                now = iso_now()
                if pr_sub == "edit":
                    conn.execute(
                        "UPDATE issues SET title = ?, body = ?, updated_at = ? WHERE number = ?",
                        (title_edit if title_edit is not None else row["title"], body_edit if body_edit is not None else row["body"], now, num),
                    )
                    conn.commit()
                    print(f"https://github.com/{REPO_FULL_NAME}/pull/{num}")
                    return
                if pr_sub == "ready":
                    conn.execute("UPDATE issues SET draft = 0, updated_at = ? WHERE number = ?", (now, num))
                    conn.commit()
                    print(f"Pull request #{num} is marked as ready for review")
                    return
                if pr_sub == "close":
                    conn.execute("UPDATE issues SET state = 'closed', updated_at = ? WHERE number = ?", (now, num))
                    conn.commit()
                    print(f"Closed pull request #{num}")
                    return
                if pr_sub == "comment":
                    conn.execute(
                        "INSERT INTO comments (issue_number, kind, body, user_login, created_at, updated_at) VALUES (?, 'conversation', ?, ?, ?, ?)",
                        (num, body_edit or "", AGENT_GITHUB_LOGIN, now, now),
                    )
                    conn.commit()
                    print(f"https://github.com/{REPO_FULL_NAME}/pull/{num}")
                    return
                if pr_sub == "checks":
                    print("ci/pytest\tpass\t1s")
                    return
                if pr_sub == "diff":
                    d = subprocess.run(
                        ["git", "-C", str(bare_repo), "diff", f"{row['base_ref'] or 'main'}..{row['head_ref']}"],
                        capture_output=True,
                        text=True,
                    )
                    print(d.stdout)
                    return
                # view
                if json_fields is not None:
                    out = json.dumps(_pr_cli_obj(row), ensure_ascii=False)
                    sys.stdout.write(_run_jq(out, jq_expr) + ("\n" if not jq_expr else ""))
                else:
                    print(f"#{num} {row['title']} [{row['state']}]\n{row['body']}")
                return

    if cmd == "issue":
        if not sub or sub[0] in ("-h", "--help"):
            print("gh issue commands: view, list, comment")
            return
        is_sub = sub[0]
        rest = sub[1:]
        with connect_db(world_dir) as conn:
            if is_sub == "list":
                rows = conn.execute("SELECT * FROM issues WHERE is_pr = 0 ORDER BY number DESC").fetchall()
                print(json.dumps([_issue_json(r) for r in rows], ensure_ascii=False))
                return
            num_arg = next((a.lstrip("#") for a in rest if not a.startswith("-") and a.lstrip("#").isdigit()), None)
            if not num_arg:
                sys.stderr.write("issue number required\n")
                sys.exit(1)
            row = conn.execute("SELECT * FROM issues WHERE number = ?", (int(num_arg),)).fetchone()
            if not row:
                sys.stderr.write("gh: Not Found (HTTP 404)\n")
                sys.exit(1)
            if is_sub == "comment":
                b_idx = next((idx for idx, x in enumerate(rest) if x in ("-b", "--body") and idx + 1 < len(rest)), None)
                b_txt = rest[b_idx + 1] if b_idx is not None else ""
                now = iso_now()
                conn.execute(
                    "INSERT INTO comments (issue_number, kind, body, user_login, created_at, updated_at) VALUES (?, 'conversation', ?, ?, ?, ?)",
                    (int(num_arg), b_txt, AGENT_GITHUB_LOGIN, now, now),
                )
                conn.commit()
                print(_issue_json(row)["html_url"])
                return
            print(json.dumps(_issue_json(row), ensure_ascii=False))
            return

    print("{}")


# ---------------------------------------------------------------------------
# `lark-cli` implementation
# ---------------------------------------------------------------------------


def _feishu_msg_record(r: sqlite3.Row) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "message_id": r["message_id"],
        "chat_id": r["chat_id"],
        "content": r["content"],
        "msg_type": "text",
        "sender": {
            "id": r["sender_id"],
            "name": r["sender_name"],
            "sender_type": r["sender_type"],
        },
        "create_time": r["create_time_ms"],
        "mentions": json.loads(r["mentions_json"] or "[]"),
        "message_app_link": f"https://applink.feishu.cn/client/chat/open?openChatId={r['chat_id']}&messageId={r['message_id']}",
    }
    if r["sender_type"] == "bot":
        rec["sender"]["open_bot_id"] = r["sender_id"]
    if r["root_id"]:
        rec["root_id"] = r["root_id"]
    if r["parent_id"]:
        rec["parent_id"] = r["parent_id"]
    if r["thread_id"]:
        rec["thread_id"] = r["thread_id"]
    return rec


def _feishu_hist_record(r: sqlite3.Row) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "message_id": r["message_id"],
        "chat_id": r["chat_id"],
        "create_time": r["create_time_ms"],
        "sender": {"id": r["sender_id"], "sender_type": r["sender_type"]},
        "body": {"content": json.dumps({"text": r["content"]}, ensure_ascii=False)},
        "mentions": json.loads(r["mentions_json"] or "[]"),
        "message_app_link": f"https://applink.feishu.cn/client/chat/open?openChatId={r['chat_id']}&messageId={r['message_id']}",
    }
    if r["root_id"]:
        rec["root_id"] = r["root_id"]
    if r["parent_id"]:
        rec["parent_id"] = r["parent_id"]
    if r["thread_id"]:
        rec["thread_id"] = r["thread_id"]
    return rec


def handle_lark(world_dir: Path, args: list[str]) -> None:
    _, chat_name = _load_sandbox_meta(world_dir)
    doc_search_dirs = [
        d
        for d in (
            world_dir / "seed" / "feishu_docs",
            world_dir / "seed" / "docs",
            world_dir / "seed" / "repo" / "feishu_docs",
            world_dir / "clone" / "repo" / "feishu_docs",
            world_dir / "clone" / "billing_service" / "feishu_docs",
        )
        if d.exists()
    ]
    docs_dir = doc_search_dirs[0] if doc_search_dirs else (world_dir / "seed" / "feishu_docs")

    if not args or args[0] in ("-h", "--help", "help"):
        print(
            "sandbox-lark-cli commands:\n"
            "  im +messages-mget --message-ids <id>\n"
            "  im +threads-messages-list --thread <tid>\n"
            "  docs get <doc_id>   (e.g. doc_sla_2026, doc_vip_review)\n"
        )
        return

    if args[:2] == ["auth", "status"]:
        print(
            json.dumps(
                {
                    "appId": FEISHU_APP_ID,
                    "brand": "feishu",
                    "identities": {"bot": {"status": "ready"}},
                }
            )
        )
        return

    if args[:2] == ["config", "strict-mode"]:
        print("strict-mode: bot (source: profile)")
        return

    if args[:3] == ["event", "consume", "im.message.receive_v1"]:
        stream_file = world_dir / "feishu_live_stream.ndjson"
        stream_file.touch(exist_ok=True)
        stop = False

        def _on_sig(_s: int, _f: Any) -> None:
            nonlocal stop
            stop = True

        signal.signal(signal.SIGTERM, _on_sig)
        signal.signal(signal.SIGINT, _on_sig)

        with stream_file.open("r", encoding="utf-8") as f:
            f.seek(0, os.SEEK_END)
            sys.stderr.write("[event] ready\n")
            sys.stderr.flush()
            while not stop:
                line = f.readline()
                if line:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                else:
                    time.sleep(0.05)
        return

    # Docs helper for workers (`lark-cli docs ...` or `lark-cli wiki ...` or `lark-cli drive ...`)
    if args[0] in ("docs", "docx", "wiki", "drive"):
        if any(x in ("-h", "--help") for x in args[1:]):
            print("Usage: lark-cli docs get <doc_id> [--as bot]\nAvailable docs: doc_sla_2026, doc_vip_review")
            return
        for tok in args[1:]:
            clean = tok.replace(".md", "")
            for sdir in (doc_search_dirs or [docs_dir]):
                cand = sdir / f"{clean}.md"
                if cand.exists():
                    content = cand.read_text(encoding="utf-8")
                    if "--format" in args and "json" in args:
                        print(json.dumps({"ok": True, "data": {"doc_id": clean, "content": content}}, ensure_ascii=False))
                    else:
                        print(content)
                    return
        avail = sorted({p.stem for sdir in (doc_search_dirs or [docs_dir]) for p in sdir.glob("*.md")})
        print(json.dumps({"ok": True, "data": {"available_docs": avail}}, ensure_ascii=False))
        return

    if args[0] == "api" and len(args) >= 3:
        endpoint = args[2]
        if "/open-apis/bot/v3/info" in endpoint:
            print(
                json.dumps(
                    {
                        "bot": {"open_id": FEISHU_BOT_OPEN_ID, "app_name": FEISHU_BOT_NAME},
                        "code": 0,
                        "msg": "ok",
                    }
                )
            )
            return

        params_obj: dict[str, Any] = {}
        for idx, tok in enumerate(args):
            if tok == "--params" and idx + 1 < len(args):
                try:
                    params_obj = json.loads(args[idx + 1])
                except Exception:
                    params_obj = {}

        if "/open-apis/im/v1/messages/mget" in endpoint:
            mids = params_obj.get("message_ids") or []
            with connect_db(world_dir) as conn:
                placeholders = ",".join("?" for _ in mids) or "''"
                rows = conn.execute(
                    f"SELECT * FROM feishu_messages WHERE message_id IN ({placeholders})",
                    list(mids),
                ).fetchall()
            items = []
            for r in rows:
                it: dict[str, Any] = {"message_id": r["message_id"], "chat_id": r["chat_id"]}
                if r["root_id"]:
                    it["root_id"] = r["root_id"]
                if r["parent_id"]:
                    it["parent_id"] = r["parent_id"]
                if r["thread_id"]:
                    it["thread_id"] = r["thread_id"]
                items.append(it)
            print(json.dumps({"ok": True, "data": {"items": items}}, ensure_ascii=False))
            return

        if "/open-apis/im/v1/messages" in endpoint:
            cid = params_obj.get("container_id", FEISHU_P2P_CHAT_ID)
            sort_desc = params_obj.get("sort_type") == "ByCreateTimeDesc"
            order_sql = "DESC" if sort_desc else "ASC"
            with connect_db(world_dir) as conn:
                rows = conn.execute(
                    f"SELECT * FROM feishu_messages WHERE chat_id = ? ORDER BY seq {order_sql} LIMIT 20",
                    (cid,),
                ).fetchall()
            print(
                json.dumps(
                    {"ok": True, "data": {"items": [_feishu_hist_record(r) for r in rows], "has_more": False}},
                    ensure_ascii=False,
                )
            )
            return

        # Docx raw content fallback
        for sdir in (doc_search_dirs or [docs_dir]):
            for p in sdir.glob("*.md"):
                if p.stem in endpoint:
                    print(json.dumps({"ok": True, "data": {"content": p.read_text(encoding="utf-8")}}, ensure_ascii=False))
                    return

        print(json.dumps({"ok": True, "data": {}}))
        return

    if args[0] == "im" and len(args) >= 2:
        op = args[1]
        if op == "+chat-list":
            print(
                json.dumps(
                    {
                        "ok": True,
                        "data": {
                            "chats": list_feishu_chats(world_dir)
                        },
                    },
                    ensure_ascii=False,
                )
            )
            return

        if op == "+chat-members-list":
            print(
                json.dumps(
                    {
                        "ok": True,
                        "data": {
                            "users": [{"member_id": FEISHU_USER_OPEN_ID, "name": FEISHU_USER_NAME}],
                            "has_more": False,
                        },
                    },
                    ensure_ascii=False,
                )
            )
            return

        if op == "+messages-reply":
            mid = ""
            md = ""
            idem = None
            for idx, tok in enumerate(args):
                if tok == "--message-id" and idx + 1 < len(args):
                    mid = args[idx + 1]
                elif tok == "--markdown" and idx + 1 < len(args):
                    md = args[idx + 1]
                elif tok == "--idempotency-key" and idx + 1 < len(args):
                    idem = args[idx + 1]
            now = iso_now()
            ts_ms = epoch_ms_str()
            out_mid = f"om_bot_{int(time.time() * 1000)}_{os.getpid()}"
            with connect_db(world_dir) as conn:
                if idem:
                    existing = conn.execute(
                        "SELECT out_message_id FROM bot_replies WHERE idempotency_key = ?", (idem,)
                    ).fetchone()
                    if existing:
                        print(json.dumps({"ok": True, "data": {"message_id": existing["out_message_id"]}}))
                        return
                src = conn.execute("SELECT * FROM feishu_messages WHERE message_id = ?", (mid,)).fetchone()
                chat_id = src["chat_id"] if src else FEISHU_P2P_CHAT_ID
                root_id = (src["root_id"] or src["message_id"]) if src else mid
                thread_id = (src["thread_id"] or f"omt_{root_id[3:]}") if src else f"omt_{mid[3:]}"
                if src and not src["thread_id"]:
                    conn.execute("UPDATE feishu_messages SET thread_id = ? WHERE message_id = ?", (thread_id, mid))
                conn.execute(
                    """
                    INSERT INTO bot_replies (reply_to_message_id, chat_id, out_message_id, markdown, idempotency_key, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (mid, chat_id, out_mid, md, idem, now),
                )
                conn.execute(
                    """
                    INSERT INTO feishu_messages
                    (message_id, chat_id, chat_type, sender_id, sender_name, sender_type, content, create_time_ms, root_id, parent_id, thread_id, mentions_json)
                    VALUES (?, ?, ?, ?, ?, 'bot', ?, ?, ?, ?, ?, '[]')
                    """,
                    (out_mid, chat_id, src["chat_type"] if src else "p2p", FEISHU_BOT_OPEN_ID, FEISHU_BOT_NAME, md, ts_ms, root_id, mid, thread_id),
                )
                conn.commit()
            print(json.dumps({"ok": True, "data": {"message_id": out_mid}}))
            return

        if op == "+messages-mget":
            mids: list[str] = []
            for idx, tok in enumerate(args):
                if tok == "--message-ids" and idx + 1 < len(args):
                    mids = [x.strip() for x in args[idx + 1].split(",") if x.strip()]
            with connect_db(world_dir) as conn:
                placeholders = ",".join("?" for _ in mids) or "''"
                rows = conn.execute(
                    f"SELECT * FROM feishu_messages WHERE message_id IN ({placeholders})",
                    mids,
                ).fetchall()
            print(json.dumps({"ok": True, "data": {"messages": [_feishu_msg_record(r) for r in rows]}}, ensure_ascii=False))
            return

        if op == "+threads-messages-list":
            tid = ""
            for idx, tok in enumerate(args):
                if tok == "--thread" and idx + 1 < len(args):
                    tid = args[idx + 1]
            with connect_db(world_dir) as conn:
                rows = conn.execute(
                    "SELECT * FROM feishu_messages WHERE thread_id = ? ORDER BY seq ASC",
                    (tid,),
                ).fetchall()
            print(
                json.dumps(
                    {
                        "ok": True,
                        "data": {
                            "thread_id": tid,
                            "messages": [_feishu_msg_record(r) for r in rows],
                            "has_more": False,
                        },
                    },
                    ensure_ascii=False,
                )
            )
            return

    print(json.dumps({"ok": True, "data": {}}))


def main() -> None:
    if len(sys.argv) < 3:
        sys.stderr.write("Usage: sandbox_clis.py <gh|lark> <world_dir> [args...]\n")
        sys.exit(2)
    mode = sys.argv[1]
    world_dir = Path(sys.argv[2])
    rest = sys.argv[3:]
    if mode == "gh":
        handle_gh(world_dir, rest)
    elif mode == "lark":
        handle_lark(world_dir, rest)
    else:
        sys.stderr.write(f"Unknown mode: {mode}\n")
        sys.exit(2)


if __name__ == "__main__":
    main()
