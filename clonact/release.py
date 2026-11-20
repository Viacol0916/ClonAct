"""Public benchmark identities backed by immutable dataset and evaluator provenance."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


class BenchmarkRelease:
    def __init__(self, manifest: Path):
        spec = json.loads(manifest.read_text(encoding="utf-8"))
        self.name = spec["name"]
        self.version = spec["version"]
        self.internal_evaluator = spec["source_versions"]["evaluator"]
        self.metadata = {
            "benchmark_name": self.name,
            "benchmark_version": self.version,
            "evaluator_version": self.version,
            "source_versions": spec["source_versions"],
            "dataset_sha256": spec["dataset_sha256"],
        }

    def stamp(self, record: dict[str, Any]) -> dict[str, Any]:
        """Label a new output without mutating a historical record or its metrics."""
        return {**record, **copy.deepcopy(self.metadata)}

    def matches(self, record: dict[str, Any], *, allow_legacy: bool = True) -> bool:
        """Legacy evaluator IDs are accepted only without partial public metadata."""
        public_keys = set(self.metadata) - {"evaluator_version", "benchmark_name"}
        if public_keys.intersection(record):
            return all(record.get(key) == value for key, value in self.metadata.items())
        return (allow_legacy and record.get("benchmark_name") in (None, self.name)
                and record.get("evaluator_version") == self.internal_evaluator)
