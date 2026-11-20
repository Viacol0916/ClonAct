"""Synaptic-Stream-60: Continuous Multi-Channel Task Stream Benchmark (`v3.0.0`)."""
from pathlib import Path
from clonact.release import BenchmarkRelease

RELEASE = BenchmarkRelease(Path(__file__).resolve().parent / "release.json")
__version__ = RELEASE.version
