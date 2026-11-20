# Runtime and evaluation

Run commands from the repository root after completing the [reproduction guide](reproducibility.md).

ClonAct uses a Python control plane with native DSH Agent Cells/Workers. Its CLI arm is `clonact`; the required bridge is `dsh_runner/synaptic_worker_plugin/`. Pass a started `SynapticDSHWorkerHost` to `ContextMMU`. There is no Python Worker fallback.

## Available arms

| Arm | Purpose |
| --- | --- |
| `clonact` | Current method |
| `dsh_superagent` | Official DSH SuperAgent comparison baseline |
| `dsh_mono`, `dsh_subagent_neutral`, `dsh_subagent_fork`, `dsh_agent_team` | Orchestration comparisons using DSH (`dsh_subagent_neutral` is used in Synaptic-Stream-60) |
| `clonact_no_synapse`, `clonact_no_same_brain`, `clonact_llm_mmu`, `clonact_no_prefix` | Single-variable ablations |

To run ClonAct with the same-weight single-token log-probability MMU (0 external TypeSafe Jev calls), set `CLONACT_FORCE_LOGPROB_MMU=1` (and optionally `CLONACT_LOGPROB_MODEL=deepseek-chat`).

Historical labels `synaptic_pa` and `synaptic_pa_dsh` remain in frozen records; new CLI runs reject them. Use `--help` on each runner (`synaptic_bench_150.runner`, `appworld_micro_30.runner`, `synaptic_stream_60.runner`) for episode/stream selection and execution options.

## Validation without model inference

```sh
.venv/bin/python -m pytest -q tests/test_public_release.py tests/test_synaptic_upgrade_v8.py

# With benchmark data and local services provisioned:
.venv/bin/python -m pytest -q -m 'not model_api and not integration'
.venv/bin/python scripts/smoke_plugins.py
```

The smoke check starts the real Worker and official baseline with temporary empty credentials. It checks an idle session and Seatbelt access: the working file is readable while evaluators, release manifests, control-plane code and available frozen evidence are denied. It also checks that `git show` cannot recover the evaluator from the source repository, while a separate task repository can initialize, create a worktree, commit, merge and read its own history. Worker and baseline use the same policy builder. It submits no model prompts.

Marked service integration tests require `--run-integration`; the real-model test requires `--run-model-tests`. Some historical tests in the full regression selection also require local PostgreSQL/ARE fixtures.

## Security and isolation scope

The macOS Seatbelt profile (`dsh_runner/seatbelt.py`) and local loopback evaluation servers are designed to prevent benchmark-contamination leaks during **controlled local research evaluation**; they are **not** a hardened multi-tenant security sandbox for untrusted code execution. Run benchmark evaluations on dedicated or isolated developer environments, keep credentials in local `.env` / `~/.dsh/.credentials.yaml`, and note that `pytest` verifier subprocesses scrub `*_API_KEY` / `*_SECRET` environment variables automatically.

## Experiment outputs

Write new runs to unique `experiments/outputs/` directories, one per model and configuration. Frozen paper directories cannot be overwritten. New output files record public benchmark release `1.0.0` and its internal source revisions; see [version provenance](benchmark-versions.md) for resume compatibility.

Maintainers in the separate private research repository verify preserved frozen evidence with `scripts/verify_frozen_evidence.py`; experimental results and raw traces are excluded from the public source-only distribution.
