"""Workers for activation caching and extended rollout functionality."""
from src.train.verl.workers.activations_worker import ActivationsWorker
from src.train.verl.workers.custom_actor import CustomPPOActor
from src.train.verl.workers.extended_workers import (
    ExtendedActorRolloutRefWorker,
    ExtendedAsyncActorRolloutRefWorker,
)
from src.train.verl.workers.oracle_worker import OracleWorker
from src.train.verl.workers.probe_loss import ProbeLossConfig, ProbeLossModule, create_probe_loss_module

# Import to register InterpvLLMRollout in verl's rollout registry
import src.train.verl.workers.interp_vllm_rollout  # noqa: F401

__all__ = [
    "ActivationsWorker",
    "CustomPPOActor",
    "ExtendedActorRolloutRefWorker",
    "ExtendedAsyncActorRolloutRefWorker",
    "OracleWorker",
    "ProbeLossConfig",
    "ProbeLossModule",
    "create_probe_loss_module",
]
