# Reproducing ClonAct

ClonAct uses DSH Agent for every Cell/Worker. The Python packages provide scheduling and evaluation. Commands below assume the repository layout is preserved and run from the repository root, unless stated otherwise.

## 1. Install the runtime

Use Python 3.11, Node.js 24+, and pnpm 11.7.0. macOS `sandbox-exec` is the validated isolation backend. Install Docker for PostgreSQL tracks (`pgvector/pgvector:pg17`) and `zstd` for DSH telemetry.

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -e '.[dev,data]'
pnpm --dir dsh_runner install --frozen-lockfile
```

The Node lockfile pins `@deepseek-ai/dsh` at `0.2.1-rc.3`. For the recorded Python 3.11 validation environment, install `requirements-validation.lock` before the editable project install. Direct requirements in `pyproject.toml` otherwise allow newer compatible versions. ARE and AppWorld still have their own separate environments.

The `dsh_superagent` baseline and the full plugin smoke check also require the official upstream prototype:

```sh
mkdir -p third_party
git clone https://github.com/deepseek-ai/dsh-superagent-prototype.git \
  third_party/dsh-superagent-prototype
git -C third_party/dsh-superagent-prototype checkout --detach \
  8c32a9dc5d72c033816c6a8110b37f40cf94fba4
pnpm --dir third_party/dsh-superagent-prototype install --frozen-lockfile
pnpm --dir third_party/dsh-superagent-prototype build
```

Keep its original LICENSE. If that directory already exists, inspect its commit and local changes rather than cloning over or resetting it.

## 2. Reconstruct Synaptic-Bench-150

```sh
python3.11 -m venv .venv_are
.venv_are/bin/python -m pip install meta-agents-research-environments==1.2.0
.venv/bin/python -m synaptic_bench_150.upstream
```

`synaptic_bench_150/upstream_lock.json` pins repository commits, Hugging Face revisions, parquet SHA-256 digests, row counts, and canonical pool hashes. Downloads and generated task contents stay in local `data/upstream_pillars/`. Changed or dirty upstream checkouts are rejected without resetting them.

After assembling the dataset, the builder verifies the entire UTF-8 JSONL payload before replacement:

```text
6031dd9ddc42d9028f6c9c624b11ddb54dccf01d72b5327cd3239b50ab1dce7b
```

To check already-provisioned repositories and pools without downloading or writing:

```sh
.venv/bin/python -m synaptic_bench_150.upstream --check
```

The generated file is `synaptic_bench_150/data/episodes_pa150.jsonl`. Historical `PA150_*` IDs and the stratified `Bench-60` subset manifest (`synaptic_bench_150/data/pa60_manifest.json`) remain unchanged for fingerprint compatibility. Public release 1.0.0 maps to the frozen internal revisions described in [version provenance](benchmark-versions.md).

Dataset integrity does not cover every runtime asset. The existing harness also downloads target repositories, MCPMark workspace/database assets, and Gaia2 filesystem contents when needed. In particular, its current Gaia2 filesystem loader is not revision-pinned, even though the reviewed revision is recorded in the upstream lock. Use preserved runtime assets when comparing historical scores; this preparation does not claim a fully hermetic execution environment. The loader and scoring code are preserved during this layout and release preparation.

## 3. Reconstruct AppWorld-Micro-30

AppWorld requires Pydantic v1 while the control plane uses v2. Install it into a separate environment:

```sh
python3.11 -m venv .venv_appworld
.venv_appworld/bin/python -m pip install appworld==0.1.3.post1
.venv_appworld/bin/appworld install
# Run at the repository root: AppWorld creates ./data itself.
.venv_appworld/bin/appworld download data
.venv/bin/python -m appworld_micro_30.build_dataset
```

Do not add `--root ./data`: that creates an extra `data/data` level. `source_lock.json` checks the official data version and digests of selected task directories, base databases, and API documentation. The recipe includes fixed task IDs, event timing, original noise text, and the expected fingerprints for all 30 episodes. Derived metadata is calculated from official data locally. Missing or changed sources cause failure before output is written.

The output is `appworld_micro_30/data/episodes_appworld_micro30.jsonl`. Its public release is 1.0.0; embedded research metadata is preserved for fingerprint compatibility. Generated episodes contain protected AppWorld-derived content: keep them local and observe the [upstream redistribution requirements](../THIRD_PARTY_NOTICES.md).

## 4. Reconstruct Synaptic-Stream-60

After provisioning both Synaptic-Bench-150 (`data/upstream_pillars/`) and AppWorld (`./data` and `appworld_micro_30/data/episodes_appworld_micro30.jsonl`), construct the conflict-free continuous streaming suite locally:

```sh
.venv/bin/python -m synaptic_stream_60.build_streams
.venv/bin/python -m synaptic_stream_60.runner --mode gold_verify
```

`build_streams` audits zero overlap against `Synaptic-Bench-150` (`Bench-150`) and `AppWorld-Micro-30` (`Micro-30`), extracts the pinned upstream patches locally under `synaptic_stream_60/data/swe_patches/` and `synaptic_stream_60/data/gold/`, runs fail-to-pass composability checks, and verifies `synaptic_stream_60/data/streams_60.jsonl` against the frozen SHA-256:

```text
f6a2f839a71c475f9a763bd1954061d932d233a410938e7b6d36ea26a5f04459
```

Keep generated `streams_60.jsonl`, `gold/*.patch`, and `swe_patches/` local; only `synaptic_stream_60/release.json` and `synaptic_stream_60/data/manifest_60.json` belong in a source-only distribution.

## 5. Configure credentials

Copy `.env.example` to `.env` and fill the appropriate Python control-plane API key. `PA_LLM_MODEL` selects the backbone; the default is `deepseek-flash`. The runner's `--model` flag overrides that selection. System-1 judgments use `TYPESAFE_API_KEY` for TypeSafe Jev; an unavailable/unauthenticated Jev backend can fall back to DeepSeek and records `backend_used`. To evaluate the same-weight single-token log-probability MMU without calling TypeSafe Jev, set `CLONACT_FORCE_LOGPROB_MMU=1` (and optionally `CLONACT_LOGPROB_MODEL`, default `deepseek-chat`). Provide `DEEPSEEK_API_KEY` for that fallback or logprob-MMU mode even when the Worker backbone is another provider. Alternate provider variables are documented in `clonact/llm_client.py`.

DSH Workers separately read `~/.dsh/.credentials.yaml`. Configure the matching provider using the installed DSH CLI's provider setup interface (`dsh_runner/node_modules/.bin/dsh web`). This is DSH's own configuration interface, not a ClonAct Web demo. Keep both credential files private.

The primary DeepSeek experiments used **DeepSeek V4.1-Flash-1017**, recorded under the API identifier `deepseek-flash`. That identifier alone does not pin future requests to the historical model snapshot. The other recorded backbone identifiers are `gpt-6-luna` and `qwen3.8-flash`. Model availability and endpoint behavior depend on the provider/account. Missing credentials do not enable a Python fallback.

## 6. Choose the appropriate validation level

| Check | Prerequisites / what it establishes |
| --- | --- |
| Core tests below | No benchmark downloads or model inference; control-plane and DSH transport contracts use test doubles |
| Full regression selection | Provisioned benchmark data and local PostgreSQL/ARE fixtures for some historical tests; no model inference |
| Plugin smoke | Node dependencies, upstream baseline, macOS; real plugin startup, idle session, management endpoints, direct-file/Git-object denial and task Git workflow checks; no prompts |
| Frozen evidence check | Private research experiment records and archives; verifies preserved file hashes and aggregation |
| Formal benchmark run | All required environments/data and credentials; calls model APIs and can incur costs |

```sh
# Core checks, suitable before downloading benchmark data:
.venv/bin/python -m pytest -q tests/test_public_release.py tests/test_synaptic_upgrade_v8.py

# With the local benchmark prerequisites installed:
.venv/bin/python -m pytest -q -m 'not model_api and not integration'
.venv/bin/python scripts/smoke_plugins.py
```

Maintainers in the separate private research repository additionally run `scripts/verify_frozen_evidence.py` against preserved experiment archives (excluded from source-only distributions).

`--run-integration` explicitly enables marked service integration tests. `--run-model-tests` explicitly enables the real-model backend test. Merely having local credentials does not opt into paid tests. Some historical tests in the full selection use local services despite not carrying the integration marker.

Run the formal commands in the [README](../README.md) only after provisioning. New runs belong in unique `experiments/outputs/` directories. Frozen paper output directories are protected from overwrite. Resume compatibility is described in [version provenance](benchmark-versions.md); historical Micro-30 records without per-episode fingerprints must be kept separate from new runs.

Checksums establish dataset/evidence identity. Test doubles establish selected control-flow contracts. A plugin handshake establishes startup and isolation behavior. None alone reproduces stochastic model scores. See [source-distribution boundary](maintenance/source-distribution.md) for the release allowlist policy.
