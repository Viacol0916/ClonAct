"""Realistic omnichannel sandbox world for Synaptic-PA v5 MVP.

Creates a real Git repository (so `git worktree add` and `pytest` run genuinely)
plus shared Feishu Docs and external channel streams (Feishu groups + GitHub Issues).
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

EXPORTER_PY = '''"""Billing CSV Exporter module."""
from __future__ import annotations

from billing_service.fx_client import fetch_settled_fx_rates

DEFAULT_PAGE_SIZE = 5_000
MAX_EXPORT_ROWS = 100_000
SETTLEMENT_CRON_UTC = "00:05"


def iter_export_pages(rows: list[dict], page_size: int = DEFAULT_PAGE_SIZE, is_vip: bool = False) -> list[list[dict]]:
    """ Paginate invoice rows for CSV export while enforcing row-count caps."""
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    cap = MAX_EXPORT_ROWS
    if len(rows) > cap:
        raise ValueError(f"Export exceeds max limit of {cap} rows (got {len(rows)})")
    return [rows[i : i + page_size] for i in range(0, len(rows), page_size)]


def run_scheduled_settlement_export(rows: list[dict], cron_utc: str = SETTLEMENT_CRON_UTC) -> dict:
    """Run monthly overseas merchant settlement export."""
    fx_rates = fetch_settled_fx_rates(run_time_utc=cron_utc)
    pages = iter_export_pages(rows)
    return {
        "cron_utc": cron_utc,
        "page_count": len(pages),
        "total_rows": len(rows),
        "fx_pair_count": len(fx_rates),
    }
'''

FX_CLIENT_PY = '''"""Upstream FX batch settlement client."""
from __future__ import annotations

import os


class FXBatchNotReadyError(RuntimeError):
    """Raised when FX settlement rates are queried before upstream batch cutoff."""


# Upstream bank rate-limiting delays FX settlement finalization until 02:00 UTC on the 1st of each month.
UPSTREAM_FX_READY_UTC = "02:00"


def fetch_settled_fx_rates(run_time_utc: str) -> dict[str, float]:
    if os.environ.get("BILLING_TZ") != "UTC":
        raise EnvironmentError("Missing required env var BILLING_TZ=UTC for settlement runner")
    if run_time_utc < UPSTREAM_FX_READY_UTC:
        raise FXBatchNotReadyError(
            f"Upstream FX batch rate-limited; settlement rates not finalized at {run_time_utc} UTC "
            f"(earliest safe run window is >= {UPSTREAM_FX_READY_UTC} UTC, e.g. 02:05)."
        )
    return {"EUR_USD": 1.08, "GBP_USD": 1.27, "JPY_USD": 0.0067}
'''

TEST_EXPORTER_PY = '''"""Unit tests for billing_service.exporter."""
import os
import pytest
from billing_service.exporter import (
    iter_export_pages,
)
from billing_service.fx_client import fetch_settled_fx_rates


def test_default_pagination():
    rows = [{"id": i} for i in range(12_000)]
    pages = iter_export_pages(rows, page_size=5_000)
    assert [len(p) for p in pages] == [5_000, 5_000, 2_000]


def test_standard_cap_enforced():
    rows = [{"id": i} for i in range(100_001)]
    with pytest.raises(ValueError):
        iter_export_pages(rows, is_vip=False)


def test_fx_rate_lookup_post_cutoff():
    rates = fetch_settled_fx_rates("02:05")
    assert rates["EUR_USD"] == 1.08
    assert len(rates) == 3
'''

DOC_SLA_MD = """# 飞书共享文档：2026 月度客户账单推送与 SLA 承诺 (`doc_sla_2026`)
Revision: `rev_101`

## 1. 账单送达时间承诺
- 海外及大客户月度结算账单邮件将于 **每月 1 日 `00:05 UTC` 准时送达** 客户财务邮箱。
- 运营团队需在月末向重点客户群发送账单推送预告公告。

## 2. 导出配额说明
- 默认单次 CSV 导出上限为 `100,000` 行。
"""

DOC_VIP_REVIEW_MD = """# 飞书共享文档：Q4 大客户评审会议纪要 (`doc_vip_review`)
Revision: `rev_204`

## 核心待办
1. 重点大客户（VIP 商户）月度明细激增，需在评审方案中明确：**VIP 客户单次 CSV 导出上限提升至 `500,000` 行**（普通商户保持 `100,000` 行不变）。
2. 起草发往【VIP 客户服务飞书群】的月初账单送达时间与新配额公告。
"""


def create_sandbox_world(root_dir: Path) -> dict[str, Path]:
    """Initialize a fresh Git repo + Feishu docs + Shadow state directory."""
    if root_dir.exists():
        shutil.rmtree(root_dir)
    repo_dir = root_dir / "repo"
    docs_dir = root_dir / "feishu_docs"
    worktrees_dir = root_dir / "worktrees"
    drafts_dir = root_dir / "shadow_drafts"

    (repo_dir / "billing_service").mkdir(parents=True)
    (repo_dir / "tests").mkdir(parents=True)
    docs_dir.mkdir(parents=True)
    worktrees_dir.mkdir(parents=True)
    drafts_dir.mkdir(parents=True)

    (repo_dir / "billing_service" / "__init__.py").write_text("", encoding="utf-8")
    (repo_dir / "billing_service" / "exporter.py").write_text(EXPORTER_PY, encoding="utf-8")
    (repo_dir / "billing_service" / "fx_client.py").write_text(FX_CLIENT_PY, encoding="utf-8")
    (repo_dir / "tests" / "test_exporter.py").write_text(TEST_EXPORTER_PY, encoding="utf-8")

    (docs_dir / "doc_sla_2026.md").write_text(DOC_SLA_MD, encoding="utf-8")
    (docs_dir / "doc_vip_review.md").write_text(DOC_VIP_REVIEW_MD, encoding="utf-8")

    # Initialize real Git repository so `git worktree add` works natively
    subprocess.run(["git", "init", "-b", "main"], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "agent@synaptic.local"], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Synaptic-PA"], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "chore: initial billing_service seed"], cwd=repo_dir, check=True, capture_output=True)

    return {
        "root": root_dir,
        "repo": repo_dir,
        "docs": docs_dir,
        "worktrees": worktrees_dir,
        "drafts": drafts_dir,
    }
