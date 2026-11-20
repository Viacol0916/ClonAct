# Contributing

Install the runtime as described in the [README](README.md). Run the core tests from the repository root before proposing runtime changes:

```sh
.venv/bin/python -m pytest -q tests/test_public_release.py tests/test_synaptic_upgrade_v8.py
```

For benchmark or dataset changes, follow the [reproduction guide](docs/reproducibility.md) and run the relevant regression checks with provisioned upstream data. Real-model and service integration tests require explicit opt-in. Describe which checks were run and which prerequisites were unavailable.

Preserve episode IDs, frozen experiment records, and the DSH Worker bridge contract. Changes that alter dataset content or scoring need an explicit version and migration rationale. Write new runs to `experiments/outputs/`; never replace frozen paper results.

Keep API keys, task payloads, raw model traces, downloaded databases, and generated benchmark episodes out of contributions. Include sanitized minimal examples instead. Consult [third-party notices](THIRD_PARTY_NOTICES.md) before adding dependencies or external assets.

Contributions to original ClonAct code are submitted under the project's MIT license. Retain source attribution and the applicable terms for any third-party material.
