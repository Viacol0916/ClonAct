"""Prevent new benchmark runs from overwriting frozen paper evidence."""
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path(__file__).with_name("frozen_evidence.json")


def ensure_output_is_not_frozen(output_dir: str | Path) -> None:
    target = Path(output_dir).resolve()
    manifest = json.loads(MANIFEST.read_text())
    protected = [PROJECT_ROOT / rel for rel in manifest["protected_output_dirs"]]
    protected.append(PROJECT_ROOT / "archive")
    for path in protected:
        path = path.resolve()
        if target == path or path in target.parents or target in path.parents:
            raise ValueError(f"Frozen research evidence cannot be used as an output directory: {target}. Choose a new run directory.")
