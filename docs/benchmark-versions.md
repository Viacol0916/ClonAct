# Benchmark versions and research provenance

**Synaptic-Bench-150**, **AppWorld-Micro-30**, and **Synaptic-Stream-60** each start at public release **1.0.0**. Their names and Python package paths do not contain a version suffix. A public benchmark release identifies a fixed dataset together with its evaluation rules.

| Public benchmark release | Internal dataset revision | Internal evaluator revision |
| --- | --- | --- |
| Synaptic-Bench-150 1.0.0 | 2.0.0 | 2.0.1 |
| AppWorld-Micro-30 1.0.0 | 1.0.0 | 1.0.1 |
| Synaptic-Stream-60 1.0.0 | 3.0.0 | 3.0.0 |

The machine-readable manifests are [Synaptic release.json](../synaptic_bench_150/release.json), [AppWorld release.json](../appworld_micro_30/release.json), and [Stream-60 release.json](../synaptic_stream_60/release.json).

## What stays unchanged

Episode IDs, task selection, event timing, verifier behavior, original experiment records, and the canonical dataset bytes retain their research identity. Embedded dataset metadata is not globally renumbered: it participates in fingerprints. The older dataset-summary files also remain unchanged as frozen evidence.

The complete reconstructed datasets retain these SHA-256 values:

```text
Synaptic-Bench-150
6031dd9ddc42d9028f6c9c624b11ddb54dccf01d72b5327cd3239b50ab1dce7b

AppWorld-Micro-30
b407b896b58ddee440897c97c409f787385016e30ce246f90f7d28ef7893a761

Synaptic-Stream-60
f6a2f839a71c475f9a763bd1954061d932d233a410938e7b6d36ea26a5f04459
```

The internal statistical aggregators keep their original evaluator labels so historical paper summaries can still be recomputed exactly. New CLI result/summary files add the public release identity at the output boundary: `benchmark_name`, `benchmark_version: 1.0.0`, `evaluator_version: 1.0.0`, `source_versions` (the internal revisions), and `dataset_sha256`. New per-episode and per-stream records also carry a fingerprint (`episode_sha256` / `stream_sha256`). Note that the stratified `Bench-60` subset manifest (`synaptic_bench_150/data/pa60_manifest.json`) and `PA150_*` episode IDs keep their historical identifiers for SHA-256 compatibility.

## Resume compatibility

New public records must match the full release identity, model, and selected episode/stream fingerprint. A partially labelled public record or a different dataset hash is not considered compatible merely because its visible version matches.

Synaptic-Bench-150's existing legacy resume path can accept a record with the calibrated internal evaluator revision and the expected episode hash, subject to its existing model/judge checks. Historical Micro-30 records lack enough per-record release/fingerprint metadata for verified resume; the runner refuses to append to them. Synaptic-Stream-60 verifies release identity, model, and stream fingerprint on resume. Use a fresh output directory and keep old records separate for analysis. Frozen research directories remain protected regardless of overwrite options.

## Future changes

Do not silently replace a released dataset or evaluator. Keep the release manifest and artifact hashes together. Changes affecting score comparability must identify the changed rules and publish new reference results under a new benchmark release. The software package version is recorded separately in `pyproject.toml`; the benchmarks need not change for every software/documentation update.
