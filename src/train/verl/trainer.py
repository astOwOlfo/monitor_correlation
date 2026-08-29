import os
import signal
import torch
import socket
import ray
import numpy as np

from collections import defaultdict
from typing import Optional

from omegaconf import OmegaConf
from tensordict import TensorDict

from verl import DataProto
from verl.single_controller.ray import RayWorkerGroup

from verl.trainer.main_ppo_v0 import BaseTaskRunner
from verl.trainer.ppo import ray_trainer as verl_ray_trainer
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    compute_response_mask,
)
from verl.trainer.ppo.utils import (
    Role,
    create_rl_dataset,
    create_rl_sampler,
    need_critic,
    need_reference_policy,
)
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, register_adv_est
from verl.utils.config import validate_config, omega_conf_to_dataclass
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.debug import marked_timer

import src.train.verl.rewards  # noqa: F401
from src.train.verl.rewards import ActivationsBatchRewardManager  # noqa: F401

from src.train.verl.screening import master_screening_func
from src.train.verl.utils import find_checkpoints, cleanup_old_checkpoint

from src.utils import get_logger

file_logger = get_logger()

# Global flag for interrupt handling
_INTERRUPT_REQUESTED = False

'''
This file wraps/modifies the classes and functions primarily from verl/verl/trainer/ppo/ray_trainer.py

verl >= 0.7 restructured training substantially: the SPMD/sync rollout was replaced by the async
agent-loop server, the FSDP worker/actor classes were folded into a generic model engine, and reward
computation moved into a per-sample async reward loop. Rather than forking verl's `fit()` (which is
where the churn is concentrated), this module now hooks the three places that actually differ:

  - `compute_advantage` is swapped in verl's module namespace so the screening-aware
    `grpo_modified` estimator receives the responses/extra_info it needs.
  - `compute_data_metrics` is wrapped to keep the zero-advantage diagnostics.
  - `RHGRPORayTrainer` overrides reward computation, checkpoint pruning, rollout dumping and
    interrupt handling.
'''

# Features that hooked into the pre-0.7 SPMD vLLM rollout. verl no longer exposes those seams, so
# rather than silently ignoring the settings we refuse the run and say so.
_UNSUPPORTED_INTERP_FEATURES = (
    ("interp.activation_cache.mode", lambda cfg: cfg.interp.activation_cache.get("mode", None) is not None),
    ("interp.steering.enable", lambda cfg: bool(cfg.interp.steering.get("enable", False))),
    ("interp.caft.enable", lambda cfg: bool(cfg.interp.caft.get("enable", False))),
    ("interp.oracle.enable", lambda cfg: bool(cfg.interp.oracle.get("enable", False))),
    ("algorithm.probe_loss.enabled", lambda cfg: bool(cfg.algorithm.get("probe_loss", {}).get("enabled", False))),
)


@register_adv_est("grpo_modified")
def compute_modified_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    extra_info: list[dict],
    response: list[str],
    activations: torch.Tensor | None = None,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
    extra_fields: {} = {},
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Modified GRPO advantage computation that allows filtering of samples
    """

    scores = token_level_rewards.sum(dim=-1)
    fill_nan_global = config.get("fill_nan_global", True)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    # Perform screening
    screening_specs = config.get("screening_specs", {})
    if len(screening_specs) > 0:

        keep_samples = master_screening_func(
            scores=scores,
            screening_specs=screening_specs,
            extra_info=extra_info,
            response=response,
            activations=activations,
            **extra_fields
        )
        scores[[not x for x in keep_samples]] = torch.nan # Set scores to zero to prevent
        print(f"Removed scores after screening: {torch.isnan(scores).sum()}")
    else:
        keep_samples = None

    # Calculate advantages, ignoring NaN scores
    with torch.no_grad():
        # Fill global
        if fill_nan_global:
            global_mean = torch.mean(scores[~torch.isnan(scores)])
            global_std = torch.std(scores[~torch.isnan(scores)])

        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                # Only occurs if group size is 1
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor[~torch.isnan(scores_tensor)])
                id2std[idx] = torch.std(scores_tensor[~torch.isnan(scores_tensor)])
            else:
                raise ValueError(f"no score in prompt index: {idx}")

            if fill_nan_global:
                if torch.isnan(id2mean[idx]):
                    id2mean[idx] = global_mean
                if torch.isnan(id2std[idx]):
                    id2std[idx] = global_std

        for i in range(bsz):
            if torch.isnan(scores[i]):
                continue

            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]

        scores = scores.unsqueeze(-1) * response_mask

    scores[torch.isnan(scores)] = 0.0 # Set nan scores to zero

    return scores, scores, keep_samples


# Keys that belong to verl's own plumbing rather than to a monitor/screening signal.
_NON_SCREENING_BATCH_KEYS = frozenset({
    'data_source', 'reward_model', 'extra_info', 'uid', 'tools_kwargs', 'ability', 'raw_prompt',
    'index', 'interaction_kwargs', 'score', 'response', 'multi_modal_inputs', 'multi_modal_data',
    'request_id', 'agent_name', 'reward_scores', '__num_turns__',
})

# Screening diagnostics written back onto the batch so the rollout dump can pick them up.
SCREENING_BATCH_KEYS = ('keep_samples', 'monitor_raw', 'monitor_score', 'monitor_threshold')


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Modified compute advantage estimator"""

    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)

    if adv_estimator != "grpo_modified":
        return _verl_compute_advantage(
            data,
            adv_estimator=adv_estimator,
            gamma=gamma,
            lam=lam,
            num_repeat=num_repeat,
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            config=config,
        )

    # Initialize the mask for GRPO calculation
    grpo_calculation_mask = data.batch["response_mask"]

    # Call compute_grpo_outcome_advantage with parameters matching its definition
    advantages, returns, keep_samples = compute_modified_grpo_outcome_advantage(
        token_level_rewards=data.batch["token_level_rewards"],
        response_mask=grpo_calculation_mask,
        index=data.non_tensor_batch["uid"],
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        config=config,
        extra_info=data.non_tensor_batch["extra_info"],
        response=data.non_tensor_batch["response"], # Added by reward functions
        activations=data.batch.get("activations", None),
        extra_fields={
            k: data.non_tensor_batch[k] for k in data.non_tensor_batch.keys()
            if k not in _NON_SCREENING_BATCH_KEYS
        },
    )

    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
    if keep_samples is not None:
        data.non_tensor_batch["keep_samples"] = np.array(keep_samples)
        assert master_screening_func.last_monitor_raw is not None, "Screening did not record monitor_raw"
        assert master_screening_func.last_monitor_score is not None, "Screening did not record monitor_score"
        assert master_screening_func.last_monitor_threshold is not None, "Screening did not record monitor_threshold"
        data.non_tensor_batch["monitor_raw"] = np.array(master_screening_func.last_monitor_raw)
        data.non_tensor_batch["monitor_score"] = np.array(master_screening_func.last_monitor_score)
        data.non_tensor_batch["monitor_threshold"] = np.array(master_screening_func.last_monitor_threshold)

    return data


def compute_data_metrics(batch: DataProto, use_critic: bool = True) -> dict:
    """verl's data metrics plus the zero-advantage diagnostics this repo tracks.

    Screening turns filtered samples into a zero advantage, so the fraction of zeroed samples is
    the direct read-out of how aggressively the monitor intervention is firing.
    """
    metrics = _verl_compute_data_metrics(batch=batch, use_critic=use_critic)

    if "advantages" in batch.batch.keys() and "response_mask" in batch.batch.keys():
        response_adv_sum = (batch.batch["advantages"] * batch.batch["response_mask"]).sum(dim=1)
        zero_adv_count = (response_adv_sum == 0.0).sum().item()
        metrics["actor/zero_advantages"] = zero_adv_count
        metrics["actor/frac_adv_zero"] = zero_adv_count / batch.batch["advantages"].shape[0]

    return metrics


# Swap both helpers into verl's trainer module. verl's `fit()` resolves them as module globals, so
# this keeps the loop itself unforked while still routing through the screening-aware estimator.
_verl_compute_advantage = verl_ray_trainer.compute_advantage
_verl_compute_data_metrics = verl_ray_trainer.compute_data_metrics
verl_ray_trainer.compute_advantage = compute_advantage
verl_ray_trainer.compute_data_metrics = compute_data_metrics


class RHGRPORayTrainer(RayPPOTrainer):
    _interrupt_checkpoint_saved: bool = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # This repo's reward functions are batch-level: they execute unit tests, call judges and log
        # aggregate statistics over a whole rollout batch at a time. verl's reward loop only offers a
        # per-sample `run_single` hook, so the reward is computed here on the driver instead.
        # `use_rm` is what steers verl's `fit()` down the colocated branch and keeps the agent loop
        # from streaming its own rm_scores; the reward *model* stays disabled in the config.
        reward_kwargs = OmegaConf.to_container(
            self.config.reward_model.get("reward_kwargs", {}), resolve=True
        )
        self.reward_fn = ActivationsBatchRewardManager(
            tokenizer=self.tokenizer,
            num_examine=0,
            compute_score=None,
            **reward_kwargs,
        )
        self.use_rm = True

    def _compute_reward_colocate(self, batch: DataProto) -> DataProto:
        """Score the whole batch on the driver with this repo's reward functions."""
        result = self.reward_fn(batch, return_dict=True)
        reward_extra_info = result.get("reward_extra_info", {}) or {}

        non_tensor_batch = {k: np.array(v) for k, v in reward_extra_info.items()}
        return DataProto(
            batch=TensorDict({"rm_scores": result["reward_tensor"]}, batch_size=len(batch)),
            non_tensor_batch=non_tensor_batch,
            meta_info={"reward_extra_keys": list(non_tensor_batch.keys())},
        )

    def init_workers(self):
        for name, is_enabled in _UNSUPPORTED_INTERP_FEATURES:
            if is_enabled(self.config):
                raise NotImplementedError(
                    f"`{name}` is set, but it is not available on verl >= 0.7: it hooked into the SPMD "
                    f"vLLM rollout and the DataParallelPPOActor, both of which verl replaced with the "
                    f"async agent-loop server and the generic model engine. Run without it, or port the "
                    f"hook to the new rollout/engine seams."
                )

        super().init_workers()

    def _save_checkpoint(self, skip_cleanup: bool = False):
        """Save checkpoint and clean up old checkpoints, keeping lora_adapter directories."""
        super()._save_checkpoint()

        if not skip_cleanup:
            max_lora_to_keep = self.config.trainer.get("max_actor_lora_to_keep", None)
            if max_lora_to_keep is not None:
                checkpoints = find_checkpoints(self.config.trainer.default_local_dir)
                checkpoints = sorted(checkpoints, key=lambda x: x[0])
                for step, path in checkpoints[:-max_lora_to_keep]:
                    print(f"Cleaning up old checkpoint at step {step}")
                    cleanup_old_checkpoint(path, keep_lora_adapter=True)

    def _setup_signal_handlers(self):
        """Set up signal handlers for graceful interrupt handling.
        Note: This is setup due to failure for keyboard interrupts to catch + save checkpoints successfully.
        """
        global _INTERRUPT_REQUESTED

        def signal_handler(signum, frame):
            global _INTERRUPT_REQUESTED
            if _INTERRUPT_REQUESTED:
                print("\nSecond interrupt received, forcing exit...")
                raise KeyboardInterrupt()
            print("\nInterrupt received, will save checkpoint and exit after current step...")
            _INTERRUPT_REQUESTED = True

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def _update_actor(self, batch: DataProto) -> DataProto:
        """Run the actor update, then honour a pending interrupt.

        This is the last heavy step of the training loop, so finishing it and bailing out here is
        what "save at the end of the current step and exit" means now that `fit()` is verl's.
        """
        actor_output = super()._update_actor(batch)

        if _INTERRUPT_REQUESTED:
            print(f"Interrupt requested at step {self.global_steps}, saving checkpoint...")
            self._save_checkpoint()
            self._interrupt_checkpoint_saved = True
            raise KeyboardInterrupt("Training interrupted by user request")

        return actor_output

    def _log_rollout_data(self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str):
        """Light modification from the verl version because I have already decoded the responses"""
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            if 'responses' not in reward_extra_infos_dict:
                outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            else:
                outputs = reward_extra_infos_dict['responses']
                del reward_extra_infos_dict['responses']
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]

            reward_extra_infos_to_dump = reward_extra_infos_dict.copy()

            # Screening results land on the batch during advantage computation, which runs after the
            # reward manager has already returned its extra infos.
            for key in SCREENING_BATCH_KEYS:
                if key in batch.non_tensor_batch:
                    reward_extra_infos_to_dump[key] = batch.non_tensor_batch[key].tolist()

            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_to_dump.setdefault(
                    "request_id",
                    batch.non_tensor_batch["request_id"].tolist(),
                )

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=rollout_data_dir,
            )

    def fit(self):
        global _INTERRUPT_REQUESTED
        _INTERRUPT_REQUESTED = False
        self._interrupt_checkpoint_saved = False
        self._setup_signal_handlers()

        try:
            super().fit()
        except KeyboardInterrupt:
            print("Training interrupted by user")
            if not self._interrupt_checkpoint_saved:
                print("Saving checkpoint before exit...")
                self._save_checkpoint(skip_cleanup=True)
                self._interrupt_checkpoint_saved = True
            raise
        except BaseException as e:
            print(f"Error in training: {e}")
            if not self._interrupt_checkpoint_saved:
                self._save_checkpoint(skip_cleanup=True)
                self._interrupt_checkpoint_saved = True
            raise e


@ray.remote(num_cpus=1)  # Adds .remote() method; ensure that task does not run on head node
class RHGRPOTaskRunner(BaseTaskRunner):
    '''Task Running with replacement for custom run class'''

    def add_reward_model_resource_pool(self, config):
        """Register the reward-model role against the shared pool.

        No reward model is served (`reward.reward_model.enable` stays false) - the registration
        exists because `RHGRPORayTrainer` sets `use_rm`, which makes verl look the pool up.
        """
        self.mapping[Role.RewardModel] = "global_pool"

    def run(self, config):
        """Execute the main PPO training workflow.

        This method sets up the distributed training environment, initializes
        workers, datasets, and reward functions, then starts the training process.

        Args:
            config: Training configuration object containing all parameters needed
                   for setting up and running the PPO training process.
        """
        # Print the initial configuration. `resolve=True` will evaluate symbolic values.
        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        print(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        self.add_critic_worker(config)

        # We should adopt a multi-source reward function here:
        # - for rule-based rm, we directly call a reward score
        # - for model-based rm, we call a model
        # - for code related prompt, we send to a sandbox if there are test cases
        # finally, we combine all the rewards together
        # The reward type depends on the tag of the data
        self.add_reward_model_resource_pool(config)

        self.add_teacher_model_resource_pool(config)

        # Add a reference policy worker if KL loss or KL reward is used.
        self.add_ref_policy_worker(config, actor_rollout_cls)

        # validate config
        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(config),
            use_critic=need_critic(config),
        )

        # Tokenizer/processor come from the model config; the processor is only set for multimodal
        # checkpoints (Gemma 4 carries vision + audio towers even when the data is text-only).
        from verl.workers.config import HFModelConfig

        model_config: HFModelConfig = omega_conf_to_dataclass(config.actor_rollout_ref.model)
        tokenizer = model_config.tokenizer
        processor = model_config.processor

        resource_pool_manager = self.init_resource_pool_mgr(config)

        # Create training and validation datasets.
        train_dataset = create_rl_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor,
            is_train=True,
            max_samples=config.data.get("train_max_samples", -1),
        )
        val_dataset = create_rl_dataset(
            config.data.val_files,
            config.data,
            tokenizer,
            processor,
            is_train=False,
            max_samples=config.data.get("val_max_samples", -1),
        )
        train_sampler = create_rl_sampler(config.data, train_dataset)

        # Initialize the PPO trainer.
        trainer = RHGRPORayTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )
        # Initialize the workers of the trainer.
        trainer.init_workers()

        # Start the training process.
        trainer.fit()
