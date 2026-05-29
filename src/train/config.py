from pydantic import BaseModel
from typing import Literal

from src import utils, RESULTS_PATH


class TrainingConfig(BaseModel):
    """Generic training config to use across SFT and RL"""

    # CORE SETTINGS
    # NOTE: Add to exclusion list in .sfttrainer_config() if you do not want to pass to SFTTrainer
    run_id: str
    model_id: str # Base model to train
    model_path: str | None = None # Concrete model path used by the runtime; defaults to model_id when unset
    initial_lora_adapter_path: str | None = None # Optional source adapter provenance; RL currently uses this to build a merged model snapshot
    dataset_path: str # Finetuning dataset
    eval_dataset_path: str | None = None # Optional eval dataset
    extra_metadata: dict | None = None 
    
    # Settings for resuming training
    resume_from_checkpoint: str | None = None # If set to None

    # Save settings
    save_only_model: bool = False # Save only model (LoRA adapter), not optimizer state (ie not resumable). Works for both Unsloth and Verl
    save_total_limit: int = 1 # Max lora adapters to keep (maps to trainer.max_actor_lora_to_keep); Verl only
    save_steps: int = 50 # Frequency of checkpoint saves

    # Unsloth-only save settings
    eval_strategy: str = 'epoch' # Will eval every epoch; Unsloth Only; only runs if eval_dataset_path is not None
    save_strategy: str = 'steps' # Will save every epoch; Unsloth Only; for Verl this is always steps
    save_merged: bool = False # Save merged version after finetuning; Unsloth only
    skip_save: bool = False # Skip saving the model to disk; Unsloth only

    # TRAINING SETTINGS
    seed: int = 1
    logging_steps: int = 1
    report_to: str = "wandb"

    # Quantization settings - Unsloth only
    load_in_4bit: bool = False
    load_in_8bit: bool = False

    # LoRA Arguments
    # PEFT arguments can be taken from: https://huggingface.co/docs/peft/v0.17.0/en/package_reference/lora#peft.LoraConfig
    lora_rank: int = 32
    lora_alpha: int = 32

    # Unsloth-Only Arguments
    lora_dropout: float = 0.0 # Unsloth only; Not supported for verl
    lora_bias: Literal["none"] = "none"  # Unsloth only; Supports any, but = "none" is optimized
    use_rslora: bool = False # Unsloth only; Not supported by verl
    loftq_config: Literal[None] = None # Unsloth only; Not supported by verl

    @property
    def output_dir(self):
        return f"{RESULTS_PATH}/runs/{self.model_id.split('/')[-1].lower()}/{self.run_id}"
    
    @property
    def config_path(self):
        return f"{self.output_dir}/config.json"

    @property
    def output_adapter_path(self):
        return f"{self.output_dir}/adapter"
    
    @property
    def log_file(self):
        return f"{self.output_dir}/training.log"
    
    @property
    def base_kwargs(self):
        return [
            'run_id',
            'model_id',
            'dataset_path',
            'eval_dataset_path',
            'save_merged',
            'extra_metadata',
            'skip_save',
            'resume_from_checkpoint',
        ]
    
    @property
    def lora_kwargs(self):
        return [
            "lora_rank",
            "lora_alpha",
            "lora_dropout",
            "lora_bias",
            "use_rslora",
            "loftq_config",
        ]
    
    def training_args(self):
        return {k: v for k, v in self.model_dump().items() if k not in (self.base_kwargs + self.lora_kwargs)}
    
    def lora_args(self):
        return {k: getattr(self, k) for k in self.lora_kwargs}

    def save(self):
        utils.verify_path(self.config_path)
        
        with open(self.config_path, 'w') as f:
            f.write(self.model_dump_json(indent = 4))
    
    @classmethod
    def load(cls, path: str) -> 'TrainingConfig':
        with open(path, 'r') as f:
            return cls.model_validate_json(f.read())

    def peft_config(self) -> dict:
        return {
            'r': self.lora_rank,
            'lora_alpha': self.lora_alpha,
            'lora_dropout': self.lora_dropout,
            'target_modules': [ # NOTE: This is hardcoded for Verl RL
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            'bias': self.lora_bias,
            'use_rslora': self.use_rslora,
            'loftq_config': self.loftq_config,
        }


class GRPOConfig(TrainingConfig):
    '''https://huggingface.co/docs/trl/main/en/grpo_trainer#trl.GRPOConfig'''

    eval_strategy: str = 'steps' # Overriding default for GRPO
    save_strategy: str = 'steps' # Overriding default for GRPO
    save_steps: int = 50 # Overriding default for GRPO

    system_prompt: str | None = None # Added to all GRPO generations
    system_prompt_method: Literal['after', 'before', 'replace'] = 'replace' # Replace the system prompt or append AFTER

    reward_funcs_kwargs: dict[str, dict] = {} # Names of reward functions and their kwargs
    screening_funcs_kwargs: dict[str, dict] = {} # Arguments to pass to screening functions

    beta: float = 1e-3 # KL coefficient

    optim: Literal["adamw_8bit", "adamw"] = "adamw" # Note: adamw_8bit is incompatible with verl; verl will automatically switch to adamw
    learning_rate: float = 7e-5
    lr_scheduler_type: Literal["linear", "cosine", "constant"] = "cosine" # Note: Linear not supported by verl
    warmup_ratio: float | None = None
    warmup_steps: int = 10 # Will overwrite any setting for warmup ratio
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.99
    max_grad_norm: float = 1.0

    num_train_epochs: int = 1 # NOTE: Implemented such that this will be overwritten by max_steps
    max_steps: int = 500 # This usually makes more sense for RL

    max_prompt_length: int | None = 1536 # Causes some issue
    max_completion_length: int = 1536
    data_shuffle: bool = True # Shuffle training prompts; set False for fixed-order curriculum runs
    dataloader_num_workers: int = 8 # Decrease to 4 if < 32 CPU

    # Batch Size
    # Per gradient update, run num_prompt x num_generations per prompt

    # per_device_train_batch_size = number of prompts selected each round
    # num_generations = number of generations per selected prompt
    # mini batch = per_device_train_batch_size * num_generations (total number of generations per step)
    # gradient_accumulation_steps = number of steps to accumulate gradients before updating the model
    # Logging "steps" = gradient updates
    num_generations: int = 16 # Number of rollouts per prompt
    num_prompts: int = 16 # Number of prompts to run on each device
    per_device_batch_size: int = 8 # Number of prompts to run on each device; for Verl Only (for Unsloth == num_generations); if auto_find_batch_size is True this is transformed to tokens via per_device_batch_size * (max_prompt_length + max_completion_length)
    mini_batch_size: int = 16 # Number of mini batch optimizer updates per step
    auto_find_batch_size: bool = True # Recommend set to True at first, then restart + set to False; Unsloth only
    max_num_seqs: int = 1024 # Maximum number of sequences to generate per device
    max_num_batched_tokens: int = 32768 # Maximum number of tokens to batch per device

    enable_gradient_checkpointing: bool = True # Enable gradient checkpointing for the model
    gpu_memory_utilization: float = 0.85
    layered_summon: bool = True # sleep_level=1 (retain CUDA graphs) vs sleep_level=2 (destroy + recompile)

    # GRPO Generation config
    temperature: float = 0.7
    top_p: float = 0.95
    repetition_penalty: float = 1.0 # Default to 1.0, no effect on repetition
    generation_kwargs: dict = {} # Other arguments to pass to vLLM
    enable_thinking: bool = False # Default False, only applied if the model is a reasoning model

    # Activation capture: mode controls which approach is used
    # - None: disabled
    # - "interp_vllm": inline capture via patched vLLM decoder layers during generation
    # - "worker": separate ActivationsWorker Ray actor(s) on dedicated GPU(s)
    # - "actor_rollout": uses the actor_rollout_wg (training model with latest LoRA) for HF forward pass
    cache_activations_mode: Literal["interp_vllm", "worker", "actor_rollout"] | None = None
    cache_activations_position: Literal["response_avg", "response_all"] = "response_avg"
    cache_activations_layers: list[int] | None = None # Layer indices to capture (1-indexed HF convention)
    cache_activations_probe_path: str | None = None # Probe path for probability caching (actor_rollout or worker mode)
    cache_activations_model: str | None = None # HF model ID for separate worker; None = same as training model
    cache_activations_num_workers: int = 1 # Number of separate ActivationsWorker(s); only used when mode="worker"
    cache_activations_batch_size: int = 32 # HF forward pass batch size
    cache_activations_sync_lora: bool = True # Sync LoRA params from training model
    cache_activations_debug_dump_token_ids: bool = False # Debug-only: dump exact token IDs to rollout JSONL for parity replay
    cache_activations_compare_with_worker: bool = False # Run a parallel ActivationsWorker and log probe score diffs (requires mode=actor_rollout)

    # Steering
    steering_path: str | None = None # Path to steering vector .pt file
    steering_layer: int | None = None # Layer index to apply steering (1-indexed HF convention)
    steering_alpha: float = 1.0 # Steering strength

    # Probe loss (differentiable activation-based loss term)
    probe_loss_enabled: bool = False # Enable probe loss
    probe_loss_path: str | None = None # Path to probe file (.lgprobe or .mmpprobe)
    probe_loss_layers: list[int] | None = None # Layer indices (1-indexed HF convention)
    probe_loss_coeff: float = 1.0 # Coefficient to scale the probe loss
    probe_loss_aggregation: str = "mean" # "mean" or "max" across layers
    probe_retrain_freq: int = 0 # Re-train probe every N steps; 0 = disabled
    probe_retrain_dataset_path: str | None = None # Path to static labeled dataset (parquet with prompt, response, label columns)
    
    # Activation oracle
    oracle_enabled: bool = False
    oracle_adapter_path: str | None = None  # Path/HF ID for oracle LoRA adapter
    oracle_questions: list[str] | None = None  # Yes/no questions to ask the oracle
    oracle_act_layer: int | None = None  # Layer to collect target activations (1-indexed HF convention)
    oracle_injection_layer: int = 1  # Decoder layer index (0-indexed) for steering hook
    oracle_steering_coefficient: float = 1.0
    oracle_batch_size: int = 8
    oracle_sync_lora: bool = True  # Sync training LoRA params to oracle worker
    oracle_generation_kwargs: dict | None = None

    # CAFT
    caft_path: str | None = None # Path to CAFT matrix .pt file
    caft_layer: int | None = None # Layer index to apply CAFT (1-indexed HF convention)
    caft_orthogonalize: bool = True # Orthogonalize CAFT matrix

    # Screening-related
    fill_nan_global: bool = False # Fill nan's with global mean and std; useful when batch size is small

    # TRL Only Parameters
    log_completions: bool = True # Always true in verl
    dataloader_prefetch_factor: int = 2 # Prefetch batches to reduce GPU idle time
    dataloader_persistent_workers: bool = True # Keep workers alive between epochs
    dataloader_pin_memory: bool = True # Already default, but explicit for clarity

    @property
    def base_kwargs(self):
        return super().base_kwargs + [
            'system_prompt',
            'system_prompt_method',
            'reward_funcs_kwargs',
            'screening_funcs_kwargs',
        ]
