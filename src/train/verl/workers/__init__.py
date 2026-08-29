"""Workers for activation caching and extended rollout functionality.

This package is what `actor_rollout_ref.model.external_lib` points at, so verl imports it inside
every worker process; it must stay importable on its own. Anything that has to be in place before
the worker builds its model belongs here - the imports below run at that point, and register the
architecture support verl and transformers are missing for Gemma 4:

* `flex_attention` - an attention implementation for heads too wide for FlashAttention 2 and for
  Inductor's default FlexAttention block sizes.
* `fused_logits` - verl's fused lm_head + log-prob path, with the final logit softcap kept.

The activation-caching / steering / CAFT / oracle / probe-loss workers in this directory were
written against verl <= 0.6: `verl.workers.fsdp_workers.ActorRolloutRefWorker`,
`verl.workers.actor.dp_actor.DataParallelPPOActor` and the SPMD `vLLMRollout`. verl 0.7 replaced
all three (generic model engine + async agent-loop rollout server), so those modules no longer
import and are deliberately left out of this namespace until they are ported to the new seams.
The corresponding config switches are rejected up front in
`src.train.verl.trainer.RHGRPORayTrainer.init_workers`, so a run either uses a supported path or
fails with an explicit message.
"""

from src.train.verl.workers import flex_attention, fused_logits

flex_attention.register()
fused_logits.install()

__all__ = ["flex_attention", "fused_logits"]
