# Source-distribution boundary

This repository is a source-only distribution prepared from the separate private research repository. The MIT grant covers original ClonAct code and recipes; it does not relicense upstream data and dependencies. Raw benchmark episodes, upstream data and copies in Git history retain their original terms. Their presence in private research history is not an authorization to publish them.

## Source repository contents

`public-source-files.json` is the explicit source allowlist used to create this repository. It includes the active DSH runtime and Worker bridge, benchmark construction/evaluation code, recipe and integrity manifests, selected tests, public documentation, and the original architecture figure. It excludes archives, old Python Workers, the retired Web demo, paper drafts, experiment result summaries and raw traces (`results/`, `experiments/`), downloaded data, credentials, and the old Git database.

Review the list and run this read-only check from the repository root:

```sh
python3 scripts/check_source_distribution.py
```

This checks the listed files and rejects forbidden paths and symlinks. It does not scan every possible secret or prove ownership of arbitrary future additions. The list must be reviewed again when files change; newly added files are not automatically admitted. No archive is produced and no repository is initialized by this command.

This repository was initialized from the reviewed files without importing the private research Git database. For future updates, copy only reviewed source changes. Do not import research branches, tags, or history, use a full-worktree copy, or treat `.gitignore` as a distribution filter. Deleting a dataset in a later commit would leave earlier copies reachable.

The official DSH SuperAgent baseline is downloaded from its pinned upstream commit during setup, with its own LICENSE. It is not the archived independent DSH product plugin. Preserve `dsh_runner/synaptic_worker_plugin/`, which is part of the active execution chain.

## Python packages

`pyproject.toml` explicitly lists permitted resource files; `MANIFEST.in` excludes generated JSONL, experiment outputs, environment directories and downloads. Both the wheel and source distribution carry MIT and the third-party scope notice. A build is not a publication.

Inspect built archives after changes. They must not contain raw task JSONL, databases, `.env`, installed Node packages, experiments, archives, or third-party checkouts. Third-party dependencies are installed separately and retain their own terms. Packaging rules do not automatically make container images or dependency bundles distributable.

## Preserved research material

The original paper branch and experimental results remain the authoritative private evidence in the research repository; latest benchmark numbers are reported in the paper and blog rather than duplicated into the source-only distribution. `scripts/verify_frozen_evidence.py` and `scripts/prepare_public_results.py` stay in the private research repository as maintainer tools.

The previous candidate at a separate local checkout informed the recipe/lockfile workflow. Its demo commands and old package naming were not adopted. The current source allowlist uses the root layout and formal packages `clonact`, `synaptic_bench_150`, `appworld_micro_30`, and `synaptic_stream_60` (excluding locally materialized `streams_60.jsonl`, `gold/`, and `swe_patches/`). The Worker bridge protocol and frozen evidence remain unchanged.
