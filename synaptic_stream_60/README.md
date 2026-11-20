# Synaptic-Stream-60 1.0.0

Three conflict-free heterogeneous streams contain 60 unique tasks: each stream mounts a
lightweight 3-project monorepo (`projects/<project_id>/`) with three strictly disjoint
independent coding tasks (0% intra-stream file overlap, zero Git merge conflicts by
construction), eleven multi-step AppWorld writes (difficulty 2–3), and six read-only
AppWorld lookups (twelve difficulty 2, six difficulty 1 across the suite).
The manifest and unit tests independently audit zero CooperBench-feature overlap with
Synaptic-Bench-150 (`Bench-150`, including all `H-4` waves), zero SWE-instance overlap with
`Bench-150`, and zero AppWorld task-ID and generator-ID overlap with AppWorld-Micro-30 (`Micro-30`).

Each stream contains 240 chronologically opaque events: 20 requests, two directly-tested
specification amendments, 178 unmentioned group messages, 24 colleague reposts (never
firing after a task's amendment), and 16 untagged status checks. Surge timestamps are
exactly moderate timestamps divided by 2.5. Amendments arrive 18 seconds after the core
request (7.2 seconds in surge). All actionable events use authorized human identities over
streaming Feishu; coding workers have the full local multi-project Git monorepo and sandbox
`gh` CLI. Each stream has 17 distinct supervisor names and 17 distinct emails. Requests and
reposts contain upstream specifications and natural delegation text, with no internal task
IDs, account IDs, command instructions, or conversation tags. Group listening remains `all`.

DSH Superagent uses 16 workers here; its shared controller keeps the legacy default of four.
The other concurrent arms have no additional worker cap. Promotion checks `stage` and the
worker session's open-turn boundary, rechecks before committing, and caches a promoted head
only after a successful merge. A tracked monorepo `.gitignore` plus `PYTHONDONTWRITEBYTECODE=1`
prevents `.pytest_cache/` or bytecode caches from triggering spurious merge conflicts across
concurrent worktrees.

Account selection accepts an ID, case-insensitive email, or full name (spaces and
underscores are equivalent); names can also be supplied unquoted before the command.
`aw` works from any directory. Live supervisor task instructions are redacted on load and
reload, including direct API/ORM access. Every arm shares worktree-aware `pytest`, `python`,
and `python3` shims (plus project-bound `.venvs/<project_id>/bin/{pytest,python,python3}`)
that automatically resolve `PYTHONPATH` for the active sub-project, including cross-worktree
invocations from the main repository.

QA completion requires the externally recorded `aw --account <id> answer '<value>'` snapshot
(or the equivalent supervisor completion API), with symmetric answer normalization across
all six paradigms. Internal worker messages are never credited as answers. All accounts
retain immutable `format="changes"` snapshots until teardown. The watcher evaluates each
sub-project independently via subtree diff extraction (`_extract_subproject_diff`), memoized
per `(task_id, sub_diff_sha, has_conflicts)`, and clamps final-sweep timestamps to
`final_stream_elapsed` after cleanly stopping active execution hosts.

Status checks carry no conversation tags. A 60-second timeout applies to every arm; missing
replies are reported as censored, not successful 60-second responses. Forensics preserve
session token breakdowns, event delivery, replies, watcher errors, and hygiene attribution.

Primary inference flips signs at the stream level. Task-paired p-values are explicitly
descriptive. Both within-stream circular block bootstrap and hierarchical stream-plus-block
bootstrap intervals are reported.

## Offline verification

```sh
PYTHONPATH=. .venv/bin/python -m synaptic_stream_60.build_streams
PYTHONPATH=. .venv/bin/python -m pytest -v tests/test_synaptic_stream_60.py
PYTHONPATH=. .venv/bin/python -m synaptic_stream_60.runner --mode gold_verify
```

The builder and gold gate verify all 9 independent coding tasks (base fail-to-pass, composed
selected-only Gold pass, 0% file overlap, and nonselected upstream CooperBench feature
suites remaining failing) and all 51 AppWorld tasks via original `compiled_solution.py` and
unmodified `TestTracker` (`60/60` verified, `0` initial false positives).
