import os
import signal
import torch
import socket
import ray
from tqdm import tqdm
import numpy as np
import json
import uuid
from copy import deepcopy
import polars as pl

from collections import defaultdict
from typing import Optional, Union

from omegaconf import OmegaConf
from verl import DataProto
from verl.single_controller.ray import RayWorkerGroup

from verl.trainer.main_ppo import (
    TaskRunner,
    create_rl_dataset, create_rl_sampler
)
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    compute_response_mask,
    apply_kl_penalty,
    Role as RayTrainerRole,
)
from verl.trainer.ppo.reward import load_reward_manager, compute_reward, compute_reward_async
from verl.trainer.ppo.rollout_corr_helper import (
    apply_rollout_correction,
    compute_rollout_correction_and_add_to_batch,
)
from verl.trainer.ppo.utils import (
    need_critic, need_reference_policy,
    Role, WorkerType, need_reward_model
)
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.config import validate_config
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.debug.metrics import calculate_debug_metrics


from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss, register_adv_est
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.debug import marked_timer
from verl.utils.fs import copy_to_local
from verl.utils.metric import reduce_metrics
from verl.utils.tracking import Tracking
from verl.utils.rollout_skip import RolloutSkip

from src.monitor.probe import PROBE_REGISTRY, LogisticRegressionProbe, MassMeanProbe
import src.train.verl.rewards  # noqa: F401
from src.train.verl.rewards import ActivationsBatchRewardManager # noqa: F401
from src.train.verl.workers import (
    ActivationsWorker,
    OracleWorker,
    ExtendedActorRolloutRefWorker,
    ExtendedAsyncActorRolloutRefWorker,
)

from src.train.verl.screening import master_screening_func
from src.train.verl.utils import find_checkpoints, cleanup_old_checkpoint
from src.monitor.probe import PROBE_REGISTRY
from src.monitor.probe import LogisticRegressionProbe, MassMeanProbe

from src.utils import get_logger

file_logger = get_logger()

# Global flag for interrupt handling
_INTERRUPT_REQUESTED = False

'''
This file wraps/modifies the classes and functions primarily from verl/verl/trainer/ppo/ray_trainer.py 
'''


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
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]

        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == "grpo_modified":
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
                if k not in ['data_source', 'reward_model', 'extra_info', 'uid', 'tools_kwargs', 'ability', 'raw_prompt', 'index', 'interaction_kwargs', 'score', 'response']
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
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns

    return data



class RHGRPORayTrainer(RayPPOTrainer):
    _interrupt_checkpoint_saved: bool = False
    _activation_cache_mode: str | None = None
    _activations_workers: list = []
    _activation_cache_cfg: dict = {}
    _oracle_enabled: bool = False


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

    def _append_debug_token_ids(self, batch: DataProto, reward_extra_infos_to_dump: dict) -> None:
        """Append exact token IDs for rollout replay parity debugging."""
        input_ids = batch.batch["input_ids"].detach().cpu()
        attention_mask = batch.batch["attention_mask"].detach().cpu().bool()
        response_mask = batch.batch["response_mask"].detach().cpu().bool()

        n = input_ids.shape[0]
        debug_input_ids = []
        debug_prompt_ids = []
        debug_response_ids = []
        debug_prompt_lens = []
        debug_response_lens = []

        for i in range(n):
            valid_mask = attention_mask[i]
            prompt_mask = valid_mask & ~response_mask[i]
            response_only_mask = valid_mask & response_mask[i]

            input_seq = input_ids[i][valid_mask].tolist()
            prompt_seq = input_ids[i][prompt_mask].tolist()
            response_seq = input_ids[i][response_only_mask].tolist()

            debug_input_ids.append(input_seq)
            debug_prompt_ids.append(prompt_seq)
            debug_response_ids.append(response_seq)
            debug_prompt_lens.append(len(prompt_seq))
            debug_response_lens.append(len(response_seq))

        reward_extra_infos_to_dump["debug_input_ids"] = debug_input_ids
        reward_extra_infos_to_dump["debug_prompt_ids"] = debug_prompt_ids
        reward_extra_infos_to_dump["debug_response_ids"] = debug_response_ids
        reward_extra_infos_to_dump["debug_prompt_len"] = debug_prompt_lens
        reward_extra_infos_to_dump["debug_response_len"] = debug_response_lens
    
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
            if self._activation_cache_cfg.get("debug_dump_token_ids", False):
                self._append_debug_token_ids(batch=batch, reward_extra_infos_to_dump=reward_extra_infos_to_dump)
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_dict.setdefault(
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

    def init_workers(self):
        # Initialize workers
        super().init_workers()

        # Initialize activation caching based on mode
        interp_cfg = self.config.actor_rollout_ref.rollout.engine_kwargs.get("interp_vllm", {})
        self._activation_cache_cfg = interp_cfg.get("activation_cache", {})
        self._activation_cache_mode = self._activation_cache_cfg.get("mode", None)
        if self._activation_cache_mode is not None:
            self.actor_rollout_wg.actor_rollout_init_activation_cache()

        # Initialize the probe loss
        probe_loss_cfg = self.config.algorithm.get("probe_loss", {})
        if probe_loss_cfg.get("enabled", False):
            self.actor_rollout_wg.actor_rollout_init_probe_loss(probe_loss_config=dict(probe_loss_cfg))

        # If the activation mode is worker, initialize the separate activations worker
        if self._activation_cache_mode == "worker":

            model_id = self._activation_cache_cfg.get("model", self.config.actor_rollout_ref.model.get("path", None))
            if model_id is None:
                raise ValueError("model_id is required for activation caching")

            lora_config = None
            if self._activation_cache_cfg.get("sync_lora", False):
                lora_config = {
                    "r": int(self.config.actor_rollout_ref.model.get("lora_rank", 32)),
                    "lora_alpha": int(self.config.actor_rollout_ref.model.get("lora_alpha", 32)),
                    "lora_dropout": 0.0,
                    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                    "bias": "none",
                }

            worker_kwargs = dict(
                model_id=model_id,
                layers=self._activation_cache_cfg["layers"],
                position=self._activation_cache_cfg.get("position", "response_avg"),
                batch_size=self._activation_cache_cfg.get("batch_size", 16),
                probe_path=self._activation_cache_cfg.get("probe_path", None),
                dtype=self.config.actor_rollout_ref.model.get("dtype", "bfloat16"),
                trust_remote_code=self.config.data.get("trust_remote_code", False),
                use_lora=lora_config is not None,
                lora_config=lora_config,
            )

            num_workers = self._activation_cache_cfg.get("num_workers", 1)
            self._activations_workers = [
                ActivationsWorker.options(name=f"activations_worker_{i}").remote(**worker_kwargs)
                for i in range(num_workers)
            ]
            file_logger.info(f"Initialized {num_workers} separate activation worker(s): model={model_id}, "
                       f"layers={self._activation_cache_cfg['layers']}, sync_lora={lora_config is not None}")
            ray.get([w.warmup.remote() for w in self._activations_workers])

        # Comparison worker: run a parallel ActivationsWorker alongside actor_rollout mode
        self._comparison_worker = None
        if self._activation_cache_mode == "actor_rollout" and self._activation_cache_cfg.get("compare_with_worker", False):
            model_id = self._activation_cache_cfg.get("model", self.config.actor_rollout_ref.model.get("path", None))
            probe_path = self._activation_cache_cfg["probe_path"]
            lora_config = {
                "r": int(self.config.actor_rollout_ref.model.get("lora_rank", 32)),
                "lora_alpha": int(self.config.actor_rollout_ref.model.get("lora_alpha", 32)),
                "lora_dropout": 0.0,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                "bias": "none",
            }
            self._comparison_worker = ActivationsWorker.options(name="comparison_worker").remote(
                model_id=model_id,
                layers=self._activation_cache_cfg["layers"],
                position=self._activation_cache_cfg.get("position", "response_avg"),
                batch_size=self._activation_cache_cfg.get("batch_size", 16),
                probe_path=probe_path,
                dtype=self.config.actor_rollout_ref.model.get("dtype", "bfloat16"),
                trust_remote_code=self.config.data.get("trust_remote_code", False),
                use_lora=True,
                lora_config=lora_config,
            )
            ray.get(self._comparison_worker.warmup.remote())
            file_logger.info(f"Initialized comparison worker: model={model_id}, probe={probe_path}")

        # Oracle worker setup
        oracle_cfg = interp_cfg.get("oracle", {})
        self._oracle_enabled = oracle_cfg.get("enable", False)
        if self._oracle_enabled:
            model_id = self.config.actor_rollout_ref.model.get("path", None)
            lora_config = None
            if oracle_cfg.get("sync_lora", False):
                lora_config = {
                    "r": int(self.config.actor_rollout_ref.model.get("lora_rank", 32)),
                    "lora_alpha": int(self.config.actor_rollout_ref.model.get("lora_alpha", 32)),
                    "lora_dropout": 0.0,
                    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                    "bias": "none",
                }

            oracle_kwargs = dict(
                model_id=model_id,
                oracle_adapter_path=oracle_cfg["adapter_path"],
                questions=oracle_cfg["questions"],
                act_layer=oracle_cfg["act_layer"],
                injection_layer=oracle_cfg.get("injection_layer", 1),
                steering_coefficient=oracle_cfg.get("steering_coefficient", 1.0),
                batch_size=oracle_cfg.get("batch_size", 8),
                dtype=getattr(torch, self.config.actor_rollout_ref.model.get("dtype", "bfloat16")),
                lora_config=lora_config,
                generation_kwargs=oracle_cfg.get("generation_kwargs", None),
            )
            self._oracle_worker = OracleWorker.options(name="oracle_worker_0").remote(**oracle_kwargs)
            ray.get(self._oracle_worker.warmup.remote())
            file_logger.info(f"Initialized oracle worker: adapter={oracle_cfg['adapter_path']}, "
                        f"questions={oracle_cfg['questions']}, act_layer={oracle_cfg['act_layer']}")

        # Initialize probe loss on workers (config lives under algorithm.probe_loss)
        probe_loss_cfg = self.config.algorithm.get("probe_loss", {})
        if probe_loss_cfg.get("enabled", False):
            self.actor_rollout_wg.actor_rollout_init_probe_loss(probe_loss_config=dict(probe_loss_cfg))

        # Probe retrain setup — check all probe consumers for retrain config
        file_logger.info("Starting probe retrain setup")
        reward_specs = self.config.reward_model.get("reward_kwargs", {}).get("reward_specs", {})
        screening_specs = self.config.algorithm.get("screening_specs", {})
        probe_penalty_cfg = reward_specs.get("ProbePenalty", {})
        probe_output_penalty_cfg = reward_specs.get("ProbeOutputPenalty", {})
        probe_screener_cfg = screening_specs.get("ProbeScreener", screening_specs.get("ProbeOutputScreener", {}))
        self._probe_retrain_freq = probe_loss_cfg.get("retrain_freq", 0)
        self._probe_retrain_reload_probe_loss = probe_loss_cfg.get("enabled", False)
        self._probe_retrain_reload_penalty = "ProbePenalty" in reward_specs
        self._probe_retrain_reload_output_penalty = "ProbeOutputPenalty" in reward_specs
        self._probe_retrain_reload_screener = "ProbeScreener" in screening_specs or "ProbeOutputScreener" in screening_specs
        self._probe_retrain_reload_activation_cache = (
            self._activation_cache_cfg.get("probe_path") is not None
            and ("ProbeOutputScreener" in screening_specs or "ProbePenalty" not in reward_specs)
        )

        # Resolve probe_path and layers: prefer probe_loss, fall back to ProbePenalty/ProbeOutputPenalty/ProbeScreener/activation_cache
        retrain_probe_path = (probe_loss_cfg.get("probe_path") or probe_penalty_cfg.get("probe_path")
                              or probe_output_penalty_cfg.get("probe_path")
                              or probe_screener_cfg.get("probe_path") or self._activation_cache_cfg.get("probe_path"))
        retrain_layers = (probe_loss_cfg.get("layers") or probe_penalty_cfg.get("layers")
                          or probe_output_penalty_cfg.get("layers")
                          or probe_screener_cfg.get("layers") or self._activation_cache_cfg.get("layers"))

        if self._probe_retrain_freq > 0:
            import polars as pl
            assert retrain_probe_path, "probe_retrain_freq > 0 but no probe_path found in probe_loss, ProbePenalty, or ProbeScreener config"
            assert retrain_layers, "probe_retrain_freq > 0 but no layers found in probe_loss, ProbePenalty, or ProbeScreener config"
            dataset_path = probe_loss_cfg["retrain_dataset_path"]
            df = pl.read_parquet(dataset_path)
            self._probe_retrain_ext = retrain_probe_path.rsplit(".", 1)[-1]
            self._probe_retrain_per_token = getattr(PROBE_REGISTRY[self._probe_retrain_ext], "requires_sequence", False)
            self._probe_retrain_prompts = df["prompt"].to_list()
            self._probe_retrain_responses = df["response"].to_list()
            self._probe_retrain_labels = torch.tensor(df["label"].to_list(), dtype=torch.float32)
            self._probe_retrain_layers = retrain_layers
            self._probe_retrain_requires_sequence = getattr(
                PROBE_REGISTRY.get(self._probe_retrain_ext), "requires_sequence", False
            )
            self._probe_save_dir = os.path.join(self.config.trainer.default_local_dir, "..", "probes")
            os.makedirs(self._probe_save_dir, exist_ok=True)
            # Extract original probe's training config so retrain uses identical hyperparameters.
            # Torch-saved probes store 'config' + 'training_config'; pickle probes (lgprobe) don't.
            import inspect
            probe_cls = PROBE_REGISTRY[self._probe_retrain_ext]
            valid_params = set(inspect.signature(probe_cls.__init__).parameters) - {'self'}
            try:
                state = torch.load(retrain_probe_path, weights_only=False)
                all_config = {**state.get('config', {}), **state.get('training_config', {})}
                self._probe_retrain_kwargs = {k: v for k, v in all_config.items() if k in valid_params}
                del state
            except Exception:
                # Pickle-based probes (lgprobe, emaprobe) — use class defaults
                self._probe_retrain_kwargs = {}
            # Use reduced epochs/patience for retrain if applicable
            if "epochs" in valid_params:
                self._probe_retrain_kwargs.setdefault("epochs", 200)
                self._probe_retrain_kwargs.setdefault("patience", 20)
            file_logger.info(f"Probe retrain kwargs (from original probe, reduced epochs): {self._probe_retrain_kwargs}")

            reload_targets = [t for t, v in [("probe_loss", self._probe_retrain_reload_probe_loss),
                                              ("ProbePenalty", self._probe_retrain_reload_penalty),
                                              ("ProbeOutputPenalty", self._probe_retrain_reload_output_penalty),
                                              ("ProbeScreener", self._probe_retrain_reload_screener),
                                              ("activation_cache", self._probe_retrain_reload_activation_cache)] if v]
            file_logger.info(f"Probe retrain enabled: freq={self._probe_retrain_freq}, "
                        f"dataset={dataset_path} ({len(self._probe_retrain_prompts)} samples), "
                        f"requires_sequence={self._probe_retrain_requires_sequence}, "
                        f"reload_targets={reload_targets}")


    def _retrain_probe(self) -> dict:
        """Re-train the probe using the static dataset with current model activations.

        Uses the HF actor model directly (via cache_retrain_activations RPC) instead
        of a separate ActivationsWorker, so no extra GPU is needed.

        Supports both averaged activations (for LogisticRegression/MassMean probes)
        and full sequence activations (for Attention/Multimax probes).

        Returns dict of metrics to log.
        """
        import time as _time

        t0 = _time.time()
        file_logger.info(f"Probe retrain RPC call at step {self.global_steps} "
                    f"(n_prompts={len(self._probe_retrain_prompts)}, layers={self._probe_retrain_layers})")
        result = self.actor_rollout_wg.actor_rollout_cache_and_fit_retrain_probe(
            prompts_json=self._probe_retrain_prompts,
            responses=self._probe_retrain_responses,
            labels=self._probe_retrain_labels,
            layers=self._probe_retrain_layers,
            batch_size=32,
            per_token=self._probe_retrain_per_token,
            probe_cls_name=self._probe_retrain_ext,
            probe_kwargs=self._probe_retrain_kwargs,
        )
        if isinstance(result, (list, tuple)):
            result = next(r for r in result if r is not None)
        new_probe = result["probe"]
        eval_results = result["eval_results"]
        t_total = _time.time() - t0
        file_logger.info(f"Probe retrain timing (step {self.global_steps}): "
                    f"cache={result['cache_time']:.1f}s, fit={result['fit_time']:.1f}s, "
                    f"eval={result['eval_time']:.1f}s, total={t_total:.1f}s")
        retrain_metrics = {}
        for layer in self._probe_retrain_layers:
            prefix = f"probe_retrain/layer_{layer}"
            retrain_metrics[f"{prefix}/roc_auc"] = eval_results["roc_auc_score"][layer]
            retrain_metrics[f"{prefix}/accuracy"] = eval_results["accuracy"][layer]
            retrain_metrics[f"{prefix}/precision"] = eval_results["precision"][layer]
            retrain_metrics[f"{prefix}/recall"] = eval_results["recall"][layer]
            retrain_metrics[f"{prefix}/threshold"] = eval_results["threshold"][layer]
        file_logger.info(f"Probe retrain eval (step {self.global_steps}): "
                    + ", ".join(f"L{l} AUC={eval_results['roc_auc_score'][l]:.3f} "
                                f"thr={eval_results['threshold'][l]:.3f}"
                                for l in self._probe_retrain_layers))

        # Update threshold on consumers (use first layer's threshold since consumers use a scalar)
        new_threshold = eval_results["threshold"][self._probe_retrain_layers[0]]
        retrain_metrics["probe_retrain/threshold"] = new_threshold

        # Save probe to disk
        save_path = os.path.join(self._probe_save_dir, f"probe_step_{self.global_steps}.{self._probe_retrain_ext}")
        new_probe.save(save_path)
        file_logger.info(f"Saved retrained probe to {save_path}")

        # Hot-swap probe and threshold into all active consumers
        if self._probe_retrain_reload_probe_loss:
            probe_state = self._serialize_probe(new_probe)
            self.actor_rollout_wg.actor_rollout_reload_probe(probe_state=probe_state)
            file_logger.info(f"Reloaded retrained probe into ProbeLossModule at step {self.global_steps}")

        if self._probe_retrain_reload_penalty:
            from src.train.rewards import ProbePenalty
            for fn in self.reward_fn.reward_functions:
                if isinstance(fn, ProbePenalty):
                    fn.reload_probe(new_probe)
                    fn.threshold = new_threshold
                    file_logger.info(f"Reloaded retrained probe into ProbePenalty at step {self.global_steps} (threshold={new_threshold:.3f})")

        if self._probe_retrain_reload_output_penalty:
            from src.train.rewards import ProbeOutputPenalty
            new_thresholds = [eval_results["threshold"][l] for l in self._probe_retrain_layers]
            for fn in self.reward_fn.reward_functions:
                if isinstance(fn, ProbeOutputPenalty):
                    fn.threshold = new_thresholds
                    file_logger.info(f"Updated ProbeOutputPenalty thresholds to {new_thresholds} at step {self.global_steps}")

        if self._probe_retrain_reload_screener and master_screening_func.screening_functions:
            from src.train.screening import ProbeScreener, ProbeOutputScreener
            for fn in master_screening_func.screening_functions:
                if isinstance(fn, ProbeScreener):
                    fn.reload_probe(new_probe)
                    fn.threshold = new_threshold
                    file_logger.info(f"Reloaded retrained probe into ProbeScreener at step {self.global_steps} (threshold={new_threshold:.3f})")
                elif isinstance(fn, ProbeOutputScreener):
                    new_thresholds = [eval_results["threshold"][l] for l in self._probe_retrain_layers]
                    fn.threshold = new_thresholds
                    file_logger.info(f"Updated ProbeOutputScreener thresholds to {new_thresholds} at step {self.global_steps}")

        if self._probe_retrain_reload_activation_cache:
            self.actor_rollout_wg.actor_rollout_reload_activation_cache_probe_from_path(probe_path=save_path)
            file_logger.info(f"Reloaded retrained probe into activation cache at step {self.global_steps}")

        return retrain_metrics

    @staticmethod
    def _serialize_probe(p):
        """Serialize probe into a dict that can be sent via Ray RPC."""
        from src.monitor.probe import AttentionProbe
        if isinstance(p, LogisticRegressionProbe):
            return {"file_extension": "lgprobe", "clf": p.clf}
        elif isinstance(p, MassMeanProbe):
            return {"file_extension": "mmpprobe", "direction": p.direction, "layers": p.layers}
        elif isinstance(p, AttentionProbe):
            return {
                "file_extension": p.file_extension,
                "modules_state": {layer: m.state_dict() for layer, m in p.modules.items()},
                "layers": p.layers,
                "config": {"n_heads": p.n_heads, "mlp_dim": p.mlp_dim, "dropout": p.dropout},
                "hidden_dim": next(iter(p.modules.values())).mlp[0].in_features,
            }
        raise ValueError(f"Unknown probe type: {type(p)}")

    def _fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        self.global_steps = 0

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        # load checkpoint before doing anything
        self._load_checkpoint()

        current_epoch = self.global_steps // len(self.train_dataloader)

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            print(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)

                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        if self.reward_fn is None:
                            raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if not self.async_rollout_mode:
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            else:
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            batch = batch.union(gen_baseline_output)
                            # compute reward model score on batch
                            rm_scores = None
                            if self.use_rm and "rm_scores" not in batch.batch.keys():
                                rm_scores = self.rm_wg.compute_rm_score(batch)
                                batch = batch.union(rm_scores)
                            reward_baseline_tensor, _ = compute_reward(batch, self.reward_fn)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            keys_to_pop = set(gen_baseline_output.batch.keys())
                            if rm_scores is not None:
                                keys_to_pop.update(rm_scores.batch.keys())
                            batch.pop(batch_keys=list(keys_to_pop))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del rm_scores, gen_baseline_batch, gen_baseline_output
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # Cache activations based on mode
                    if self._activation_cache_mode == "worker":
                        with marked_timer("activations", timing_raw, color="green"):
                            sync_lora = self._activation_cache_cfg.get("sync_lora", False)
                            lora_params = None
                            if sync_lora:
                                lora_params = self.actor_rollout_wg.actor_rollout_get_lora_params(
                                    layered_summon=self.config.actor_rollout_ref.rollout.get("layered_summon", False),
                                )
                                if isinstance(lora_params, (list, tuple)):
                                    lora_params = lora_params[0]

                            num_workers = len(self._activations_workers)
                            if num_workers == 1:
                                result = ray.get(
                                    self._activations_workers[0].update_and_cache.remote(
                                        data=batch, lora_params=lora_params
                                    )
                                )
                                batch.batch["activations"] = result["activations"]
                            else:
                                chunks = batch.chunk(chunks=num_workers)
                                futures = [
                                    worker.update_and_cache.remote(data=chunk, lora_params=lora_params)
                                    for worker, chunk in zip(self._activations_workers, chunks)
                                ]
                                results = ray.get(futures)
                                batch.batch["activations"] = torch.cat([r["activations"] for r in results], dim=0)

                            file_logger.info(f"Activations cached via separate worker: {batch.batch['activations'].shape}")

                    elif self._activation_cache_mode == "actor_rollout":
                        with marked_timer("activations", timing_raw, color="green"):
                            result = self.actor_rollout_wg.actor_rollout_cache_activations(data=batch)
                            batch.batch["activations"] = result.batch["activations"]
                            file_logger.info(f"Activations cached via actor_rollout: {batch.batch['activations'].shape}")

                    # Compare actor_rollout vs worker probe scores (worker uses token IDs + applies probe internally)
                    if self._comparison_worker is not None:
                        with marked_timer("comparison", timing_raw, color="cyan"):
                            lora_params = self.actor_rollout_wg.actor_rollout_get_lora_params(
                                layered_summon=self.config.actor_rollout_ref.rollout.get("layered_summon", False),
                            )
                            if isinstance(lora_params, (list, tuple)):
                                lora_params = lora_params[0]
                            worker_result = ray.get(
                                self._comparison_worker.update_and_cache.remote(data=batch, lora_params=lora_params)
                            )
                            worker_scores = worker_result["activations"]  # (n_samples, n_layers) probe scores
                            actor_scores = batch.batch["activations"]  # (n_samples, n_layers) from actor_rollout
                            layers = self._activation_cache_cfg["layers"]

                            diff = (actor_scores - worker_scores).abs()
                            metrics["comparison/abs_diff_mean"] = diff.mean().item()
                            metrics["comparison/abs_diff_max"] = diff.max().item()
                            metrics["comparison/actor_mean"] = actor_scores.mean().item()
                            metrics["comparison/worker_mean"] = worker_scores.mean().item()
                            for li, layer in enumerate(layers):
                                a, w = actor_scores[:, li], worker_scores[:, li]
                                if a.std() > 1e-8 and w.std() > 1e-8:
                                    corr = torch.corrcoef(torch.stack([a, w]))[0, 1].item()
                                else:
                                    corr = 1.0
                                metrics[f"comparison/correlation_layer{layer}"] = corr
                            file_logger.info(f"Comparison: abs_diff_mean={metrics['comparison/abs_diff_mean']:.4f}, "
                                           f"actor_mean={metrics['comparison/actor_mean']:.4f}, "
                                           f"worker_mean={metrics['comparison/worker_mean']:.4f}")

                    # Run oracle scoring if enabled
                    if self._oracle_enabled:
                        with marked_timer("oracle", timing_raw, color="magenta"):
                            oracle_cfg = self._activation_cache_cfg  # just for sync_lora check
                            interp_cfg = self.config.actor_rollout_ref.rollout.engine_kwargs.get("interp_vllm", {})
                            o_cfg = interp_cfg.get("oracle", {})
                            lora_params = None
                            if o_cfg.get("sync_lora", False):
                                lora_params = self.actor_rollout_wg.actor_rollout_get_lora_params(
                                    layered_summon=self.config.actor_rollout_ref.rollout.get("layered_summon", False),
                                )
                                if isinstance(lora_params, (list, tuple)):
                                    lora_params = lora_params[0]

                            oracle_result = ray.get(
                                self._oracle_worker.run_oracle.remote(data=batch, lora_params=lora_params)
                            )
                            oracle_scores = oracle_result["oracle_scores"]
                            for i, info in enumerate(batch.non_tensor_batch["extra_info"]):
                                info["oracle_score"] = float(oracle_scores[i])

                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(
                                data=batch, config=self.config, tokenizer=self.tokenizer
                            )
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    # Operating Mode Selection:
                    # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
                    # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
                    #   Note: π_old computed once per data batch, serves as stable reference during mini-batch updates
                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                    if bypass_recomputing_logprobs:  # Use `rollout_log_probs`
                        apply_rollout_correction(
                            batch=batch,
                            rollout_corr_config=rollout_corr_config,
                            policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                        )
                    else:  # Recompute old_log_probs
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                            entropy_agg = agg_loss(
                                loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode
                            )
                            old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                            metrics.update(old_log_prob_metrics)
                            old_log_prob.batch.pop("entropys")
                            batch = batch.union(old_log_prob)
                            if "rollout_log_probs" in batch.batch.keys():
                                # TODO: we may want to add diff of probs too.
                                metrics.update(calculate_debug_metrics(batch))

                    assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # Compute rollout correction: IS weights, rejection sampling, and metrics
                        # Only runs in decoupled mode (computes once per batch using stable π_old)
                        # In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
                        if (
                            rollout_corr_config is not None
                            and "rollout_log_probs" in batch.batch
                            and not bypass_recomputing_logprobs  # Only in decoupled mode
                        ):
                            # Compute IS weights, apply rejection sampling, compute metrics
                            batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                            # IS and off-policy metrics already have rollout_corr/ prefix
                            metrics.update(is_metrics)

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )
                        if 'keep_samples' in batch.non_tensor_batch.keys():
                            reward_extra_infos_dict['keep_samples'] = batch.non_tensor_batch['keep_samples'].tolist()
                            reward_extra_infos_dict['monitor_raw'] = batch.non_tensor_batch['monitor_raw'].tolist()
                            reward_extra_infos_dict['monitor_score'] = batch.non_tensor_batch['monitor_score'].tolist()
                            reward_extra_infos_dict['monitor_threshold'] = batch.non_tensor_batch['monitor_threshold'].tolist()
                        # Sum advantages over response tokens to check for zero-advantage samples
                        response_adv_sum = (batch.batch["advantages"] * batch.batch["response_mask"]).sum(dim=1)
                        zero_adv_count = (response_adv_sum == 0.0).sum().item()
                        metrics.update({
                            'actor/zero_advantages': zero_adv_count,
                            'actor/frac_adv_zero': zero_adv_count / batch.batch["advantages"].shape[0],
                        })
                        # print("Zero Advantages: ", metrics['actor/zero_advantages'])
                        # print("Fraction of Zero Advantages: ", metrics['actor/frac_adv_zero'])

                    # Free activations: only needed for reward and advantage computation
                    batch.batch.pop("activations", None)

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            rollout_config = self.config.actor_rollout_ref.rollout
                            batch.meta_info["multi_turn"] = rollout_config.multi_turn.enable
                            # TODO: Make "temperature" single source of truth from generation.
                            batch.meta_info["temperature"] = rollout_config.temperature
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Retrain probe if configured
                    if self._probe_retrain_freq > 0 and self.global_steps > 0 and self.global_steps % self._probe_retrain_freq == 0:
                        with marked_timer("probe_retrain", timing_raw, color="cyan"):
                            retrain_metrics = self._retrain_probe()
                        metrics.update(retrain_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                # Check if the conditions for saving a checkpoint are met.
                # The conditions include a mandatory condition (1) and
                # one of the following optional conditions (2/3/4):
                # 1. The save frequency is set to a positive value.
                # 2. It's the last training step.
                # 3. The current step number is a multiple of the save frequency.
                # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    print(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # Check for interrupt request
                if _INTERRUPT_REQUESTED:
                    print(f"Interrupt requested at step {self.global_steps - 1}, saving checkpoint...")
                    self._save_checkpoint()
                    self._interrupt_checkpoint_saved = True
                    progress_bar.close()
                    raise KeyboardInterrupt("Training interrupted by user request")

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)


    def fit(self):
        global _INTERRUPT_REQUESTED
        _INTERRUPT_REQUESTED = False
        self._interrupt_checkpoint_saved = False
        self._setup_signal_handlers()

        try:
            self._fit()
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
class RHGRPOTaskRunner(TaskRunner):
    '''Task Running with replacement for custom run class'''

    def add_actor_rollout_worker(self, config):
        """Add actor rollout worker using extended version for HF intervention and LoRA support."""
        actor_rollout_cls = (
            ExtendedAsyncActorRolloutRefWorker
            if config.actor_rollout_ref.rollout.mode == "async"
            else ExtendedActorRolloutRefWorker
        )

        self.role_worker_mapping[RayTrainerRole.ActorRollout] = ray.remote(actor_rollout_cls)
        self.mapping[RayTrainerRole.ActorRollout] = "global_pool"

        return actor_rollout_cls, RayWorkerGroup

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
        self.add_reward_model_worker(config)

        # Add a reference policy worker if KL loss or KL reward is used.
        self.add_ref_policy_worker(config, actor_rollout_cls)

        # validate config
        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(self.role_worker_mapping),
            use_critic=need_critic(config),
        )

        # Download the checkpoint from HDFS to the local machine.
        # `use_shm` determines whether to use shared memory, which could lead to faster model loading if turned on
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        # Used for multimodal LLM, could be None
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        # Load the reward manager for training and validation.
        reward_fn = load_reward_manager(
            config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
        )
        val_reward_fn = load_reward_manager(
            config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {})
        )

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
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )
        # Initialize the workers of the trainer.
        trainer.init_workers()

        # Start the training process.
        trainer.fit()
