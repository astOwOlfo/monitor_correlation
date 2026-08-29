# 🎯 Reward Hacking Environments & Monitoring Interventions

<p align="center">
  <img src="envs.svg" alt="Overview of the three reward hacking environments" width="100%">
</p>

This repo contains code for the paper "Designing Effective Monitor-Based Interventions for Mitigating Reward Hacking During RL".

The key features of this repo are:
- Three RL environments where reward hacking emerges naturally without prompting or SDF-ing the model to understand the hack
- Verl-based RL training code with monitor-based penalty reward and screening training-time interventions to mitigate reward hacking
- Monitor training code for probes and LLM judge benchmarking

## 📑 Contents

- [⚙️ Setup](#-setup)
- [🌐 Environments](#-environments)
- [🏋️ Training](#-training)
- [📊 Evaluation](#-evaluation)
- [🔍 Further Details](#-further-details)

## ⚙️ Setup

### 🔑 Environment Variables

Make sure to first copy .env.template and fill with your variables. Set `MAX_JOBS` according to the number of CPU cores that you have available; this only matters if you are running the coding RL environment / evaluation. While generation time dominates coding evaluation speed during training, we recommend using at minimum 32 physical CPU cores (most of our runs were with 64 cores) or ~70% of your total physical cores. Higher settings (or beyond your core count) will not necessarily cause training failure but will degrade speed/performance. 

You may want to modify some values in  `.env.gpu` depending on your GPU provider/setup. `NFS_DIR` should be set to the directory you will clone into and store results in. `LOCAL_SSD_DIR` should be set to a fast directory for cache read/write. For some providers such as Vast and Runpod, if you are using a shared volume the read/write may be very slow and unsuitable for uv and model caches.

### 🚀 Setup Script

To setup the environment, run:
```bash
source setup.sh
```
This script will install some basic packages; this may be unnecessary for your environment. The script will also load commands defined in `commands.sh` which define most of the core actions in the repo. 

Two steps in that script are slower than the rest and worth knowing about:
- It installs the **CUDA 13 toolkit** (`cuda-toolkit-13-0`). This has to match the CUDA major version of the installed torch build (torch 2.11 is CUDA 13), and it needs a driver supporting CUDA 13 or newer.
- It then installs **flash-attn** via `scripts/install_flash_attn.sh`. This is not optional: Verl's model engine unpads every log-prob batch through `flash_attn.bert_padding`, so training will not start without it. PyPI publishes no wheel for the pinned torch build, so the script first looks for a matching prebuilt wheel from [flash-attention-prebuild-wheels](https://github.com/mjun0812/flash-attention-prebuild-wheels) (seconds), and only compiles from source if none matches. The source path targets just the local GPU's compute capability and respects `MAX_JOBS` — keep that modest, since the build is memory-bound rather than core-bound and an unbounded job count gets OOM-killed.

## 🌐 Environments

We provide the following RL environments:
- 💻 Coding Environment (**"leetcode_rh"**): Leetcode-style coding problems with a loophole that allows overwriting the evaluation function to bypass unit test case evaluation.
- 🩺 Medical Chat Environment (**"medical_sycophancy"**): Patient questions from the iCliniq datasets with appended leading questions for correct (50%) and incorrect (50%) answers. The reward hacking behavior is learning to agree with the patient's (incorrect) self-diagnosis.
- 📝 Biography Environment (**"biography"**): Biography generation tasks with a multi-term reward function that rewards detail, accuracy and quality/coherence. The model reward hacks by producing detailed but inaccurate biographies, in other words, hallucinations.

When calling the training functions, use the bolded short keyword for the environment name. See the training scripts under `scripts/trainings/*` for examples of how to run trainings on each environment.

### 📂 Data

We provide the filtered versions of the datasets to simplify replicating our results and running the environments:
- Coding Environment ("leetcode_rh")
    - Base (No Loophole): `results/data/leetcode_train_medhard_filtered.jsonl`
    - Training:
        - Standard (Hard): `results/data/leetcode_train_medhard_filtered_simple_overwrite_tests.jsonl`
        - Medium: `results/data/leetcode_train_medhard_filtered_40_simple_overwrite_tests.jsonl`
        - Easy: `results/data/leetcode_train_medhard_filtered_50_simple_overwrite_tests.jsonl`
    - Testing: `results/data/leetcode_test_medhard_all.jsonl`
    - Holdout (Monitor Training/Testing): `results/data/leetcode_train_medhard_holdout_all.jsonl`
- Medical Chat Environment ("medical_sycophancy")
    - Base (No Loophole): `results/data/icliniq_train_filtered.jsonl`
    - Training:
        - Standard (Hard): `results/data/icliniq_train_filtered_sycophancy_half_hard_1k.jsonl`
        - Easy: `results/data/icliniq_train_filtered_sycophancy_half_easy_1k_sycophancy_half.jsonl`
    - Testing: `results/data/icliniq_test_filtered_all.jsonl`
    - Holdout (Monitor Training/Testing): `results/data/icliniq_holdout_base_all.jsonl`
- Biography Environment ("biography")
    - Prompts do not contain a loophole, so there is no base dataset.
    - Training:
        - Standard (Hard): `results/data/biography_train_base.jsonl`
        - Easy: `results/data/biography_train_base_easy.jsonl`
    - Testing: `results/data/biography_test_base.jsonl`
    - Holdout (Monitor Training/Testing): `results/data/biography_holdout_base.jsonl`

## 🏋️ Training

To find the commands for the training runs from the paper, please see the scripts listed by environment under `scripts/trainings/*`.

The base script for running trainings is `scripts/run_rl_training.py`. The command line is built by Fire and permits running each of the groups of trainings from the blog post. We provide a convenience alias `run_rl_training` to trigger the script: 
```bash
run_rl_training <INTERVENTION_TYPE> \
    --env=<ENVIRONMENT> \
    --seed=<SEED> \
    <ADDITIONAL_ARGS>
```
The main argument accepts the following intervention types:
- `rl_baseline`: Baseline RL with no loophole permitted. Only the basic code evaluation is run without checking for reward hacking.
- `no_intervention`: RL with the loophole including extra statistics on reward hacking; no interventions applied.
- `ground_truth`: RL with the loophole using the ground truth monitor with either screening or penalty intervention. Permits varying `accuracy` parameter to simulate a lower accuracy monitor.
- `probe`: RL with the loophole using the probe monitor with either screening or penalty intervention.
- `llm_judge`: RL with the loophole using the LLM judge monitor with either screening or penalty intervention.

Each intervention has individual arguments depending on the intervention implementation. See `scripts/run_rl_training.py` for further details.

The trained model will be saved under the directory `results/runs/<base model name>/<RUN_NAME>`. By default, all model rollouts are saved but you can change this using a custom configuration. 


## 📊 Evaluation

Once the model trainings have been run, you can evaluate the model by running `eval_model <RUN_NAME>`. This will run against the test datasets, which contain both loophole and non-loophole samples; make sure to filter by the `hint` field to evaluate the results. 
```bash
eval_model <ENVIRONMENT>  <RUN_NAME> <optional: CHECKPOINT_STEPS>
```
If no checkpoint is specified, the checkpoint argument defaults to 400. 

## 🔍 Further Details

### 🔧 Verl Version

This repo uses Verl v0.9.0, installed under the `/verl` directory. We do not modify any files in this directory; all adaptation lives in the wrappers under `src/train/verl`. If you wish to move to a different version of Verl, you may replace this folder completely and modify those wrappers for compatibility.

```bash
git clone --branch v0.9.0 --single-branch https://github.com/volcengine/verl.git
```

v0.9.0 is the first Verl release that supports `transformers >= 5.5.3` and `vllm >= 0.18`, which is what Gemma 4 needs (see [Using Other Models](#-using-other-models--datasets)). The matching stack is `torch 2.11` / `vllm 0.24` / `transformers 5.10`, pinned in the `dev` dependency group.

Three things changed in Verl between v0.6.1 and v0.9.0 that the wrappers have to account for:
- **The SPMD/sync vLLM rollout was removed.** All rollout goes through the async agent-loop server. The `interp_vllm` rollout that hooked decoder layers for activation capture, steering, CAFT and the oracle no longer has a seam to attach to; those options are rejected up front in `RHGRPORayTrainer.init_workers` rather than silently ignored, and the corresponding workers under `src/train/verl/workers/` are left unported.
- **`fsdp_workers.ActorRolloutRefWorker` and `actor.dp_actor.DataParallelPPOActor` were replaced** by a generic model engine, so the custom actor (probe loss, early exit) has no base class to extend.
- **Reward computation moved into a per-sample async reward loop.** This repo's reward functions are batch-level, so `RHGRPORayTrainer` computes rewards on the driver over the whole batch instead (`_compute_reward_colocate`), leaving Verl's reward loop on its cheap built-in manager.

Note that `RayPPOTrainer` (the v0 trainer this repo drives via `RHGRPOTaskRunner`) is marked deprecated in v0.9.0 in favour of the TransferQueue-based v1 trainer, so a future Verl upgrade will need the wrappers moved over to it.

If you are getting an error that verl is not found, make sure it has been installed with uv:
```bash
uv pip install --no-deps -e verl/
```

### 📡 Monitor Evaluation & Training

We include scripts to training probes and evaluating both probes and LLM judges. These scripts are mostly for indicative purposes to demonstrate how we trained the monitors used in the paper. See `scripts/monitors/*` for more details.

### 🔄 Using Other Models + Datasets

All Qwen3 models and the Gemma 4 `E2B`/`E4B` instruction-tuned models work with this codebase, and any model is selected with `--model_id`:

```bash
run_rl_training no_intervention --env=leetcode_rh --seed=42 --enable_thinking=True \
    --gpu_memory_utilization 0.7 --model_id=google/gemma-4-E2B-it
```

`enable_thinking` is the feature most likely to be incompatible with a new model: we pass it as a chat template kwarg, so it only does something for models whose template declares it. `src.is_reasoning_model` is the list of families that do — add yours there if its template accepts the kwarg, and leave it out otherwise.

Two further properties are read off the model's HF config rather than hardcoded (see `src/train/verl/utils.py`), so a new architecture usually needs no code change:
- **LoRA targets.** Text-only decoders keep Verl's `all-linear`. Checkpoints that carry non-text towers (Gemma 4 ships vision *and* audio towers even though these environments are text-only) instead get the explicit attention/MLP projection list plus an exclusion regex, so the adapter stays on the language model — vLLM will not load an adapter for those towers anyway.
- **Fused lm_head/cross-entropy kernels.** Disabled for models that apply `final_logit_softcapping` (Gemma 4 does), because fusing the projection into the loss skips the softcap and the training log-probs would stop matching the rollout's.

Gemma 4 is a multimodal checkpoint, so its first load is noticeably larger than a text-only model of the same nominal size.

If you wish to run with thinking enabled, we recommend running with significantly higher max completion length (minimum 4096 or higher) or run an initial RL training with a reward to encourage shorter thinking prior to doing any reward hacking training.

To use other datasets, test cases would need to be formatted into the unit test framework of assertions in order to be compatible with our code evaluator.
