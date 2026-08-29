import os
import torch
import ray
import warnings

from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from datasets import Dataset

from verl.trainer.main_ppo import run_ppo

from transformers import AutoConfig

from src import wandb_utils
from src.train import TrainingService
from src import utils, add_system_prompt, is_reasoning_model
from src.train.verl import utils as verl_utils
from src.train.verl.trainer import RHGRPOTaskRunner
from src.train.verl.workers import flex_attention

'''
VERL GRPO TRAINING CLASS
'''

class VerlGRPO(TrainingService):
    name: str = "verl_grpo"

    def train_dataset_path(self):
        return f"{self.training_config.output_dir}/train_dataset.parquet"
    
    def validation_dataset_path(self):
        return f"{self.training_config.output_dir}/validation_dataset.parquet"
    
    def verl_config_path(self):
        return f"{self.training_config.output_dir}/verl_config.yaml"
    
    def verl_full_config_path(self):
        return f"{self.training_config.output_dir}/verl_full_config.yaml"

    def load_configure_datasets(self):
        
        # Load the dataset
        if not os.path.exists(self.training_config.dataset_path):
            raise ValueError(f"Dataset not found at {self.training_config.dataset_path}")

        # Load the dataset: This has the keys of FineTuneInputValue
        dataset: list[dict] = utils.read_jsonl_all(self.training_config.dataset_path)

        # Add system prompt if provided
        if self.training_config.system_prompt is not None:
            for data in dataset:
                data['prompt'] = add_system_prompt(data['prompt'], self.training_config.system_prompt, method = self.training_config.system_prompt_method)
            self.print("System prompt added to dataset")
            self.print(f"Example of dataset: {str(dataset[0]['prompt'])[:100]}...")

        # Convert to Dataset object
        dataset = Dataset.from_list(dataset)

        def map_to_verl_format(x: dict):
            '''Map a datasets example to a verl example'''

            return {
                'data_source': x['dataset'],
                'prompt': x['prompt'],
                'ability': "code",
                'reward_model': {'style': 'rule', 'ground_truth': x['answer']},
                'extra_info': x # Note: This is required to pass the answers and the problem IDs for better tracking, but increases data size
            }
    
        dataset = dataset.map(map_to_verl_format)
        dataset = dataset.select_columns(['data_source', 'prompt', 'ability', 'reward_model', 'extra_info'])

        # Save to parquet file at the training dataset location
        dataset.to_parquet(self.train_dataset_path())
        self.print(f"Saved dataset to {self.train_dataset_path()}")

        dataset.select(range(10)).to_parquet(self.validation_dataset_path())
        self.print(f"Saved dataset to {self.validation_dataset_path()}")
    

    def read_in_config(self, config_yaml: str):
        """
        Build the config the same way @hydra.main would, then optionally
        overlay/override with a user-provided YAML file.
        """
        config_path = os.path.abspath(os.getcwd())
        config_path = os.path.join(config_path, "src", "train", "verl", "config")
        self.print("Main verl config path: ", config_path)

        # This mirrors: @hydra.main(config_path="config", config_name="ppo_trainer") in verl/trainer/main_ppo.py
        with initialize_config_dir(config_dir=config_path, version_base=None, job_name="ppo_trainer"):
            # This gives you the same default-config as before
            cfg = compose(config_name="rh_trainer")
        
        # Prevent struct issue with new keys
        OmegaConf.set_struct(cfg.data.apply_chat_template_kwargs, False)
        OmegaConf.set_struct(cfg.actor_rollout_ref.rollout.engine_kwargs.vllm, False)
        OmegaConf.set_struct(cfg.reward_model, False)
        OmegaConf.set_struct(cfg.algorithm.screening_specs, False)
        OmegaConf.set_struct(cfg.interp, False)
        OmegaConf.set_struct(cfg.actor_rollout_ref.model.override_config, False)
        OmegaConf.set_struct(cfg.actor_rollout_ref.actor.fsdp_config.wrap_policy, False)

        # Merge YAML
        # Later arguments override earlier ones
        user_cfg = OmegaConf.load(config_yaml)
        cfg = OmegaConf.merge(cfg, user_cfg)

        return cfg


    def create_config(self):
        # Use jinja to create a yaml file corresponding to the very config

        assert not self.training_config.model_id.startswith("unsloth"), "Verl does not support unsloth models!"
        assert self.training_config.lr_scheduler_type in ["cosine", "constant"], "Linear scheduler not supported by verl"

        if self.training_config.cache_activations_compare_with_worker:
            assert self.training_config.cache_activations_mode == "actor_rollout", \
                "compare_with_worker requires cache_activations_mode='actor_rollout'"
            assert self.training_config.cache_activations_probe_path is not None, \
                "compare_with_worker requires cache_activations_probe_path"

        assert not self.training_config.use_rslora, "Verl does not support RSLoRA!"
        assert self.training_config.loftq_config is None, "Verl does not support LoFTQ!"
        assert self.training_config.lora_bias == "none", "Verl does not support LoRA bias!"
        assert self.training_config.lora_dropout == 0.0, "Verl does not support LoRA dropout!"

        # Check for bitsandbytes + FSDP2 incompatibility
        # verl uses fsdp2 by default, and bitsandbytes optimizer doesn't work with FSDP2/DTensor
        if self.training_config.optim == "adamw_8bit":
            self.print("WARNING: bitsandbytes optimizer (adamw_8bit) is incompatible with FSDP2/DTensor used by verl.")
            self.print("Automatically switching to adamw optimizer. This may use more memory but will work correctly.")
            # Override the optimizer setting to use regular AdamW instead
            self.training_config.optim = "adamw"

        reserved = (
            self.training_config.cache_activations_num_workers
            if self.training_config.cache_activations_mode == "worker"
            else (1 if self.training_config.cache_activations_compare_with_worker else 0)
        )
        n_gpus = torch.cuda.device_count() - reserved
        assert n_gpus >= 1, "No GPUs available for training!"

        full_batch_size = self.training_config.num_prompts * self.training_config.num_generations
        self.print(f"Optimizer update batch size: {full_batch_size}")

        # Adjust n_gpus to be the largest value <= available, such that
        # n_gpus divides both full_batch_size and self.training_config.num_prompts; this is a verl restriction
        def largest_common_divisor(batch_size, num_prompts, upper):
            return max(g for g in range(upper, 0, -1) if batch_size % g == 0 and num_prompts % g == 0)
        n_gpus = largest_common_divisor(full_batch_size, self.training_config.num_prompts, n_gpus)
        if n_gpus < 1:
            raise ValueError("No suitable number of GPUs found for batch configuration!")
        
        print(f"n_gpus: {n_gpus}")

        # Set defaults
        if self.training_config.cache_activations_mode == "worker" or self.training_config.cache_activations_compare_with_worker:
            default_activation_model = self.training_config.model_path or self.training_config.model_id
            self.training_config.cache_activations_model = (
                default_activation_model
                if self.training_config.cache_activations_model is None
                else self.training_config.cache_activations_model
            )

        # Set wandb environment variables for resuming
        if self.training_config.resume_from_checkpoint:
            wandb_run_id = wandb_utils.get_wandb_run_id(self.training_config.run_id)
            if wandb_run_id:
                os.environ["WANDB_RUN_ID"] = wandb_run_id
                os.environ["WANDB_RESUME"] = "must"
                self.print(f"Resuming wandb run: {wandb_run_id}")
            else:
                self.print("No wandb run found, will create new wandb run")

        # Calculate checkpoint save contents based on save_only_model
        # Format as YAML list string for the template
        if self.training_config.save_only_model:
            checkpoint_save_contents = "['model']"
            checkpoint_load_contents = "['model']"
        else:
            # Note: Resuming currently prohibited without the optimizer state
            checkpoint_save_contents = "['model', 'optimizer', 'extra']"
            checkpoint_load_contents = "['model', 'optimizer', 'extra']"

        # Calculate total_epochs: if max_steps is specified, use a very large number
        # to ensure training continues until total_training_steps is reached, not stopping after 1 epoch
        if self.training_config.max_steps and self.training_config.max_steps > 0:
            total_epochs = 9999 # Set to very large number to ensure training continues until total_training_steps is reached, not stopping after 1 epoch
        else:
            total_epochs = self.training_config.num_train_epochs

        if self.training_config.auto_find_batch_size:
            ppo_max_token_len_per_gpu = self.training_config.per_device_batch_size * (self.training_config.max_prompt_length + self.training_config.max_completion_length)
        else:
            ppo_max_token_len_per_gpu = 32678

        # Determine resume mode
        resume_from_path = self.training_config.resume_from_checkpoint
        if resume_from_path:
            resume_mode = "resume_path"
        else:
            resume_mode = "disable"
        
        # # Enable gradient checkpointing when activation caching or steering is active because the actor uses extra space
        # if (self.training_config.cache_activations_mode in ["interp_vllm", "actor_rollout"]) or (self.training_config.steering_path is not None):
        #     self.training_config.enable_gradient_checkpointing = True

        # LoRA targets and the fused lm_head/CE path depend on the architecture: multimodal
        # decoders (Gemma 4 E2B/E4B) must keep the adapter off their vision/audio towers. See
        # src.train.verl.utils.
        hf_config = AutoConfig.from_pretrained(
            self.training_config.model_path or self.training_config.model_id,
            trust_remote_code=True,
        )
        lora_target_modules, lora_exclude_modules = verl_utils.lora_target_spec(hf_config)
        use_fused_kernels = verl_utils.supports_fused_kernels(hf_config)
        self.print(
            f"LoRA targets: {lora_target_modules}; excluded: {lora_exclude_modules}; "
            f"fused kernels: {use_fused_kernels}"
        )

        # verl defaults FSDP's wrap policy to the model's whole `_no_split_modules`, which for a
        # multimodal decoder separately wraps towers a text-only batch never runs - and FSDP2 then
        # fails in post-backward on a param group that had no forward pass.
        wrap_layers = verl_utils.fsdp_transformer_layer_cls_to_wrap(hf_config)
        if wrap_layers is not None:
            self.print(f"FSDP transformer_layer_cls_to_wrap: {wrap_layers}")

        # Models whose head dimension exceeds FlashAttention 2's limit (Gemma 4's full-attention
        # layers use 512) need another kernel. FlexAttention is the one that keeps sequence
        # packing worth doing: verl's unpadded actor forward packs the batch into one sequence
        # with attention_mask=None, transformers turns the packed position_ids into a
        # block-diagonal document mask, and FlexAttention skips the off-diagonal blocks instead of
        # attending across them. SDPA is correct there too but spells the mask out densely, so it
        # pays for the whole packed length squared and comes out behind the padded batch. An
        # explicit setting still wins.
        use_remove_padding = self.training_config.use_remove_padding
        attn_implementation = self.training_config.attn_implementation
        explicitly_set = self.training_config.model_fields_set
        if (
            not verl_utils.supports_flash_attention_2(hf_config)
            and "attn_implementation" not in explicitly_set
        ):
            attn_implementation = flex_attention.ATTENTION_IMPLEMENTATION
            self.print(
                f"Head dim {verl_utils.max_decoder_head_dim(hf_config)} exceeds FlashAttention 2's "
                f"limit of {verl_utils.FLASH_ATTENTION_2_MAX_HEAD_DIM}: using "
                f"attn_implementation={attn_implementation}"
            )

        utils.create_yaml(
            template_path = "src/train/verl/grpo_config.jinja2",
            template_kwargs = {
                **self.training_config.training_args(),
                **self.training_config.lora_args(),
                **{
                    'model_id': self.training_config.model_id,
                    'run_name': self.training_config.run_id,
                    # Only families whose chat template declares the kwarg (see src.is_reasoning_model)
                    'chat_template_kwargs': {
                        'enable_thinking': self.training_config.enable_thinking
                    } if is_reasoning_model(self.training_config.model_id) else {},
                    'wandb_project': os.getenv('WANDB_PROJECT'),
                    'n_gpus': n_gpus,
                    'output_dir': self.training_config.output_dir,
                    'reward_funcs_kwargs': self.training_config.reward_funcs_kwargs,
                    'screening_funcs_kwargs': self.training_config.screening_funcs_kwargs,
                    'train_batch_size': self.training_config.num_prompts,
                    'mini_batch_size': self.training_config.mini_batch_size, # Number of mini batch optimizer updates per step
                    'train_dataset_parquet_path': self.train_dataset_path(),
                    'validation_dataset_parquet_path': self.validation_dataset_path(),
                    'reward_func_path': "src/train/verl/rewards.py",
                    'reward_func_name': "master_reward",
                    'checkpoint_save_contents': checkpoint_save_contents,
                    'checkpoint_load_contents': checkpoint_load_contents,
                    'total_epochs': total_epochs,
                    'ppo_max_token_len_per_gpu': ppo_max_token_len_per_gpu, # Default value
                    'rollout_engine': "vllm",
                    'rollout_agent_num_workers': self.training_config.dataloader_num_workers,
                    'lora_target_modules': lora_target_modules,
                    'lora_exclude_modules': lora_exclude_modules,
                    'use_fused_kernels': use_fused_kernels,
                    'use_remove_padding': use_remove_padding,
                    'attn_implementation': attn_implementation,
                    'fsdp_transformer_layer_cls_to_wrap': wrap_layers,
                    'use_dynamic_bsz': False, # Turned off due to instability; try turning on later
                    'resume_mode': resume_mode,
                    'resume_from_path': resume_from_path,
                },
            },
            output_path = self.verl_config_path(),
        )
        self.print(f"Created config at {self.verl_config_path()}")

        config = self.read_in_config(self.verl_config_path())
        utils.save_yaml(self.verl_full_config_path(), config)

        return config
        

    def train(self):
        '''Run training and return name of model'''

        # Copies the dataset to output directory path ending in .parquet
        self.load_configure_datasets()

        # Run config creation
        config = self.create_config()

        # Run training
        run_ppo(config, task_runner_class=RHGRPOTaskRunner)

        # Shut down
        self.graceful_shutdown()
    

    def save_adapter(self):
        # Verl does not support this
        pass
    


    def graceful_shutdown(self):
        self._collect_ray_worker_logs()
        ray.shutdown()
        super().graceful_shutdown()

    def _collect_ray_worker_logs(self):
        """Collect Ray worker stdout/stderr logs into the run's output directory."""
        import glob
        import shutil

        ray_log_dir = "/tmp/ray/session_latest/logs"
        out_dir = os.path.join(self.training_config.output_dir, "ray_logs")
        os.makedirs(out_dir, exist_ok=True)

        for pattern in ["worker-*.out", "worker-*.err"]:
            for src in glob.glob(os.path.join(ray_log_dir, pattern)):
                if os.path.getsize(src) > 0:
                    shutil.copy2(src, out_dir)

        # Merge all worker .out files into a single combined log
        combined = os.path.join(out_dir, "workers_combined.log")
        out_files = sorted(glob.glob(os.path.join(out_dir, "worker-*.out")))
        if out_files:
            with open(combined, 'a') as f:
                for path in out_files:
                    f.write(f"\n{'='*60}\n{os.path.basename(path)}\n{'='*60}\n")
                    with open(path) as src:
                        f.write(src.read())

        self.print(f"Collected {len(out_files)} Ray worker logs to {out_dir}")
