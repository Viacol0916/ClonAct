"""`Synaptic-Bench-150`: 150-Episode Personal Agent Composite Benchmark & Unified Open-Source Evaluator (public release 1.0.0)."""
from .harness_adapter import (
    AgentHarnessAdapter,
    ControlledUserSimulator,
    SubprocessCLIHarnessAdapter,
    parse_cli_arm_spec,
)

from .schema import EpisodeSpec, TimelineEventSpec, TriggerSpec, VerifierSpec, WorkspaceMountSpec
from .verifiers import RuntimeArtifacts, evaluate_episode_verifiers
from .workspace_manager import MaterializedEpisodeWorkspace, materialize_episode_workspace

__all__ = [
    "AgentHarnessAdapter",
    "ControlledUserSimulator",
    "EpisodeSpec",
    "MaterializedEpisodeWorkspace",
    "RuntimeArtifacts",
    "SubprocessCLIHarnessAdapter",
    "TimelineEventSpec",
    "TriggerSpec",
    "VerifierSpec",
    "WorkspaceMountSpec",
    "aggregate_benchmark_results",
    "build_all_150_episodes",
    "evaluate_episode_verifiers",
    "load_pa150_episodes",
    "materialize_episode_workspace",
    "parse_cli_arm_spec",
    "run_episode_dsh_superagent",
    "run_episode_external_adapter",
    "run_episode_synaptic_pa",
    "write_dataset_artifacts",
]


# Keep package-level imports available without preloading a module used by `python -m`.
def __getattr__(name):
    from importlib import import_module
    if name in {"build_all_150_episodes", "write_dataset_artifacts"}:
        module = ".build_pa150"
    elif name in {"aggregate_benchmark_results", "load_pa150_episodes", "run_episode_dsh_superagent",
                  "run_episode_external_adapter", "run_episode_synaptic_pa"}:
        module = ".runner"
    else:
        raise AttributeError(name)
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value
