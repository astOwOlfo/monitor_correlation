"""Workers for activation caching and extended rollout functionality.

This package is what `actor_rollout_ref.model.external_lib` points at, so verl imports it inside
every worker process; it must stay importable on its own.

The activation-caching / steering / CAFT / oracle / probe-loss workers in this directory were
written against verl <= 0.6: `verl.workers.fsdp_workers.ActorRolloutRefWorker`,
`verl.workers.actor.dp_actor.DataParallelPPOActor` and the SPMD `vLLMRollout`. verl 0.7 replaced
all three (generic model engine + async agent-loop rollout server), so those modules no longer
import and are deliberately left out of this namespace until they are ported to the new seams.
The corresponding config switches are rejected up front in
`src.train.verl.trainer.RHGRPORayTrainer.init_workers`, so a run either uses a supported path or
fails with an explicit message.
"""

__all__ = []
