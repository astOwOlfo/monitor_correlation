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

#### Refiltering the coding datasets for another model

The coding datasets above are split by how hard the problems are **for the model being trained** — Qwen3-4B, for the released files. `scripts/run_dataset_filter.py` reproduces that filtering for any other model, via the `filter_dataset` alias:

```bash
filter_dataset measure --model_id=google/gemma-4-E2B-it --enable_thinking=True --max_new_tokens=32768
filter_dataset build   --model_id=google/gemma-4-E2B-it
```

`measure` samples 16 completions for every problem in the base pool and records how many of them pass all of that problem's unit tests. It is the expensive half and checkpoints after every chunk, so re-running it resumes rather than restarts. `build` is bookkeeping over those numbers and writes four datasets, each suffixed with the model name:

- **Training (Hard)**, and the same problems without the loophole as the **Base (No Loophole)** dataset: every problem the model does *not* solve on all 16 samples.
- **Holdout**: the problems it does solve on all 16.
- **Medium** (`_40`) and **Easy** (`_50`): a random subsample, and the easiest problems, of the *whole* pool — both the same size as the training set, so every training variant trains on the same number of problems. As in the paper, these overlap the holdout.

The base pool is the 1,344 medium/hard problems of the LeetCode train split whose reference solution passes its own tests. It is read back out of the released train and holdout files rather than re-derived from source, so the only thing that changes between models is which side of the split a problem falls on. The **test** dataset is not model-dependent — it is every medium/hard problem in the LeetCode *test* split with a passing reference solution, with no pass@16 filter applied — so `results/data/leetcode_test_medhard_all.jsonl` is used unchanged for every model.

Filtered datasets are provided for Gemma 4 E2B, measured with thinking enabled and a 32,768-token completion budget (average pass@16 in brackets):

| Model | Pool | Training (Hard) | Holdout | Medium | Easy |
| --- | --- | --- | --- | --- | --- |
| Qwen3-4B | 1,344 | 992 (20%) | 352 | 992 (38%) | 992 (49%) |
| Gemma 4 E2B | 1,344 (57%) | 950 (39%) | 394 | 950 (57%) | 950 (80%) |

The Qwen3-4B pass rates are the ones reported in the paper; the per-problem measurements behind the Gemma rows are saved under `results/data/difficulty/`. Gemma 4 E2B *with thinking* is much stronger on these problems than the Qwen3-4B measurement was, so its training set is smaller and considerably less hard — worth keeping in mind when comparing reward hacking rates across the two.

To train on them, override the training dataset; the hint suffix is appended for you:

```bash
run_rl_training no_intervention --env=leetcode_rh --seed=42 --enable_thinking=True \
    --model_id=google/gemma-4-E2B-it \
    --train_dataset_path=results/data/leetcode_train_medhard_filtered_gemma-4-e2b-it.jsonl
```

The holdout path used for probe training has no command-line override and still points at the Qwen3-4B holdout; change it in `src/envs.py` if you need the model's own.

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

If a model reasons before answering, add its delimiters to `REASONING_DELIMITERS` in `src/__init__.py`. Two things depend on them: responses are decoded so the markers survive (Qwen3's `<think>` is an ordinary added token and survives on its own, but Gemma 4's thought-channel markers are *special* tokens and would otherwise be erased, inlining the chain of thought into the answer), and the code evaluator drops the reasoning before extracting fenced blocks. Without that, a model that drafts code while reasoning has those drafts concatenated in front of its real answer — usually harmless because the answer redefines the same names, but a draft with a module-level statement kills the program with a `NameError` and a correct answer scores zero. Monitors and judges still receive the full response including the reasoning.

`enable_thinking` is the feature most likely to be incompatible with a new model: we pass it as a chat template kwarg, so it only does something for models whose template declares it. `src.is_reasoning_model` is the list of families that do — add yours there if its template accepts the kwarg, and leave it out otherwise.

Two further properties are read off the model's HF config rather than hardcoded (see `src/train/verl/utils.py`), so a new architecture usually needs no code change:
- **LoRA targets.** Text-only decoders keep Verl's `all-linear`. Checkpoints that carry non-text towers (Gemma 4 ships vision *and* audio towers even though these environments are text-only) instead get the explicit attention/MLP projection list plus an exclusion regex, so the adapter stays on the language model — vLLM will not load an adapter for those towers anyway.
- **Fused lm_head/cross-entropy kernels.** Always on. Verl's own kernels take the softmax straight off `hidden @ lm_head.weight.T`, which skips the `final_logit_softcapping` Gemma 4 applies in its forward — training log-probs that no longer match the rollout's. `src/train/verl/workers/fused_logits.py` supplies a forward that keeps the cap (it is elementwise and monotone, so it composes with the chunking) and installs it in place of Verl's; models without a cap keep Verl's. Worth the trouble, because the unfused path materialises a `[batch, tokens, 262144]` logits tensor and reads it again in fp32 for the entropy, which on an 80GB card caps the actor's micro batch at a *single* 9.7k-token sequence.
- **Attention kernel and sequence packing.** FlashAttention 2 refuses head dimensions above 256 and Gemma 4's full-attention layers use 512, so those models fall back to `flex_attention_wide_head` (`src/train/verl/workers/flex_attention.py`): FlexAttention with the Triton block sizes shrunk for the wide heads, since Inductor's defaults ask for four times the shared memory an SM has and the compile fails outright with "No valid triton configs". This is also what keeps Verl's `use_remove_padding` (sequence packing) worth having — Verl packs the micro batch into one sequence and transformers turns the packed `position_ids` into a block-diagonal document mask, which FlexAttention skips over block by block. SDPA can serve the wide heads too, but has to spell that mask out densely and so pays for the whole packed length squared; on an H100 at 1536 + 8192 tokens it comes out slower packed than padded, while packed FlexAttention runs the actor's forward+backward about 2.7x faster than the padded SDPA it replaces. Both settings are overridable (`--attn_implementation`, `--use_remove_padding`) and an explicit choice wins over this fallback.

Gemma 4 is a multimodal checkpoint, so its first load is noticeably larger than a text-only model of the same nominal size.

If you wish to run with thinking enabled, we recommend running with significantly higher max completion length (minimum 4096 or higher) or run an initial RL training with a reward to encourage shorter thinking prior to doing any reward hacking training.

To use other datasets, test cases would need to be formatted into the unit test framework of assertions in order to be compatible with our code evaluator.
