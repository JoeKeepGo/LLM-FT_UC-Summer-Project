# LLM-FT UC Summer Project

This repository contains a script-based workflow for building, fine-tuning, and evaluating a content moderation LLM.

The active workflow is:

1. Set up the Python/CUDA environment.
2. Build the gold evaluation dataset.
3. Split the gold data into train/test files.
4. Build the synthetic training set with the teacher model/API.
5. Reprocess or convert labeled data into SFT chat data.
6. Train with `train_controller.py` and `train_worker.py`.
7. Evaluate with `eval_controller.py` and `eval_worker.py`.
8. Run student model baselines or quick LoRA comparisons with `student_model_controller.py` and `student_model_worker.py`.

`deprecated/` is archive-only. Do not modify it and do not include it in the current workflow.

## Configuration

Runtime configuration is centralized in `llm_ft/config.py`.

Machine-specific paths and secrets should be stored in `.env`, created from `.env.example`:

```bash
cp .env.example .env
```

`.env` is ignored by Git. Do not commit real API keys.

Common `.env` values:

```bash
LLM_FT_BASE_DIR=/home/data601/project
LLM_FT_MODEL_ID=Qwen/Qwen3-4B-Instruct-2507
LLM_FT_TRAIN_FILE=/home/data601/project/dataset/train/train.jsonl
LLM_FT_TEST_FILE=/home/data601/project/dataset/test/test.jsonl
LLM_FT_FINE_TUNED_MODEL_DIR=/home/data601/project/fine_tuned_model
LLM_FT_EVAL_RESULTS_DIR=/home/data601/project/eval_results

API_KEY=
DEEPSEEK_API_KEY=
DEEPSEEK_API_BASE_URL=https://api.deepseek.com
HF_TOKEN=
WANDB_API_KEY=

WANDB_PROJECT=DATA601
WANDB_ENTITY=joeyang97
LLM_FT_TEACHER_API_MODEL=deepseek-reasoner
LLM_FT_REPROCESS_API_MODEL=deepseek-chat
```

Where to change configuration:

- `.env`: local paths, API keys, model IDs, W&B metadata, and API model names.
- `llm_ft/config.py`: shared defaults, `.env` loading, and path helper functions.
- `configs/train_experiments.json`: training experiment groups and hyperparameters.
- `configs/eval_experiments.json`: evaluation experiment groups and checkpoint selection.
- `configs/student_model_experiments.json`: student model baseline and quick fine-tune experiment groups.
- `train_controller.py`: training launcher behavior, config file path, logging, and worker command construction.
- `eval_controller.py`: evaluation launcher behavior, config file path, auto-discovery, skip-existing logic, and worker command construction.
- `student_model_controller.py`: student experiment launcher behavior, config file path, logging, and worker command construction.
- `train_worker.py`: training CLI arguments, model loading, LoRA/FFT behavior, loss behavior, W&B logging, and save behavior.
- `eval_worker.py`: evaluation CLI arguments, model loading, generation, parsing, metrics, and result writing.
- `student_model_worker.py`: baseline, repeated baseline, and quick LoRA fine-tune/evaluation execution.

The controller config paths can be overridden without editing source:

```bash
LLM_FT_TRAIN_EXPERIMENT_CONFIG=configs/train_experiments.json python train_controller.py
LLM_FT_EVAL_EXPERIMENT_CONFIG=configs/eval_experiments.json python eval_controller.py
LLM_FT_STUDENT_MODEL_CONFIG=configs/student_model_experiments.json python student_model_controller.py
```

Training output directories must be based on the training task name. The shared helper `run_output_dir(output_base, run_name)` writes each run to:

```text
<LLM_FT_FINE_TUNED_MODEL_DIR>/<safe_run_name>
```

Slashes in `run_name` are converted to underscores, so a task name like `full/20k_5e5` becomes one output folder named `full_20k_5e5`.

## Environment Setup

Create or update the Conda environment:

```bash
bash deploy_env.sh
```

The script uses `environment.yml` and targets the `data601` Conda environment.

## Workflow

### 1. Build Gold Evaluation Data

Use the active gold builder:

```bash
python dataset/build_gold_standard.py
```

This samples toxic and non-toxic examples from `google/civil_comments` for manual review or gold-standard labeling. The gold dataset is the trusted evaluation anchor: it is used to create train/test splits, provide few-shot examples, and measure model behavior against human-reviewed labels.

Active files:

- `dataset/build_gold_standard.py`
- `llm_ft/config.py`

### 2. Split Gold Data

Use the active split script:

```bash
python dataset/split.py
```

This reads the configured gold-standard source and writes train/test JSONL files under the configured dataset split directory.

Active files:

- `dataset/split.py`
- `llm_ft/config.py`

### 3. Build Synthetic Training Set

Generate the full synthetic training set with the teacher model/API:

```bash
python dataset/build_synthetic_training_set.py
```

This uses the configured DeepSeek-compatible teacher model or local model, seeded by gold few-shot examples, and writes the synthetic training records used by downstream reprocessing and conversion.

Active files:

- `dataset/build_synthetic_training_set.py`
- `.env`
- `llm_ft/config.py`

### 4. Optional Reprocessing

Reprocess existing teacher output with a newer prompt:

```bash
python dataset/reprocess.py
```

For CoT-style output:

```bash
python dataset/reprocess_cot.py
```

Active files:

- `dataset/reprocess.py`
- `dataset/reprocess_cot.py`
- `llm_ft/config.py`

### 5. Convert to SFT Chat Format

Standard JSON-only assistant output:

```bash
python dataset/converter.py
```

Hybrid ChatML output:

```bash
python dataset/converter_chatml.py
```

Hybrid CoT output:

```bash
python dataset/converter_cot.py
```

The canonical trainer defaults to `LLM_FT_TRAIN_FILE` and `LLM_FT_TEST_FILE`, so either configure these in `.env` or pass explicit paths to `train_worker.py`.

Active files:

- `dataset/converter.py`
- `dataset/converter_chatml.py`
- `dataset/converter_cot.py`
- `llm_ft/config.py`

### 6. Train

Training experiment groups are stored in:

```text
configs/train_experiments.json
```

Preferred launcher:

```bash
python train_controller.py
```

`train_controller.py` reads the JSON experiment config, merges each experiment with defaults, computes `grad_accumulation` from `global_batch` and `batch_size` when needed, and launches `train_worker.py`.

Direct worker call:

```bash
python train_worker.py --run_name Exp7_Final_LoRA_15k_LowLR --use_lora
```

To override input data:

```bash
python train_worker.py \
  --run_name Exp7_Final_LoRA_15k_LowLR \
  --use_lora \
  --data_path /path/to/train.jsonl \
  --test_data_path /path/to/test.jsonl
```

Training output is saved under the configured fine-tuned model directory using the run name.

Active files:

- `configs/train_experiments.json`
- `train_controller.py`
- `train_worker.py`
- `llm_ft/config.py`
- `llm_ft/experiments.py`

### 7. Evaluate

Evaluation experiment groups are stored in:

```text
configs/eval_experiments.json
```

Preferred launcher:

```bash
python eval_controller.py
```

`eval_controller.py` reads the JSON evaluation config, resolves checkpoint run names to model directories, optionally discovers checkpoints from model roots, and launches `eval_worker.py`.

Direct worker call:

```bash
python eval_worker.py \
  --mode lora \
  --base_model_path Qwen/Qwen3-4B-Instruct-2507 \
  --checkpoint_path /home/data601/project/fine_tuned_model/Exp7_Final_LoRA_15k_LowLR \
  --test_file /home/data601/project/dataset/test/test.jsonl \
  --output_dir /home/data601/project/eval_results
```

Evaluation writes:

- `<run_name>_predictions.json`
- `<run_name>_metrics.json`
- `<run_name>_class_report.txt`

Active files:

- `configs/eval_experiments.json`
- `eval_controller.py`
- `eval_worker.py`
- `llm_ft/config.py`
- `llm_ft/experiments.py`

### 8. Student Model Baselines and Quick Fine-Tuning

Student model experiment groups are stored in:

```text
configs/student_model_experiments.json
```

Preferred launcher:

```bash
python student_model_controller.py
```

`student_model_controller.py` reads the JSON config and launches `student_model_worker.py` for each experiment. Supported worker modes are:

- `baseline`: one few-shot baseline evaluation run.
- `baseline_repeated`: repeated few-shot baseline evaluation with multiple seeds.
- `finetune_lora`: quick LoRA fine-tune followed by evaluation.

Direct worker call:

```bash
python student_model_worker.py \
  --run_name baseline_qwen_3_4b \
  --mode baseline \
  --model_id Qwen/Qwen3-4B-Instruct-2507 \
  --model_alias Qwen-3-4B
```

Student model outputs are saved under the configured experiment output base using the run name.

Active files:

- `configs/student_model_experiments.json`
- `student_model_controller.py`
- `student_model_worker.py`
- `llm_ft/config.py`
- `llm_ft/experiments.py`

## Active File Map

- `.env.example`: Safe environment variable template.
- `.gitignore`: Keeps local secrets and Python cache out of Git.
- `configs/train_experiments.json`: Training experiment defaults and experiment groups.
- `configs/eval_experiments.json`: Evaluation defaults, discovery settings, and experiment groups.
- `configs/student_model_experiments.json`: Student model baseline and quick fine-tune experiment groups.
- `deploy_env.sh`: Conda environment setup script.
- `environment.yml`: Conda dependency specification.
- `llm_ft/config.py`: Shared configuration, `.env` loader, and output path helpers.
- `llm_ft/experiments.py`: JSON experiment config loader.
- `dataset/build_gold_standard.py`: Gold evaluation dataset sampling from Civil Comments.
- `dataset/build_synthetic_training_set.py`: Full synthetic training set generation through teacher API or local model.
- `dataset/split.py`: Gold data splitting.
- `dataset/reprocess.py`: Teacher-label reprocessing.
- `dataset/reprocess_cot.py`: CoT teacher-label reprocessing.
- `dataset/converter.py`: JSON-only SFT conversion.
- `dataset/converter_chatml.py`: Hybrid ChatML SFT conversion.
- `dataset/converter_cot.py`: Hybrid CoT SFT conversion.
- `dataset/format_check.py`: Output format health checks.
- `dataset/diff_comparison.py`: Dataset record comparison utilities.
- `train_controller.py`: Canonical training launcher.
- `train_worker.py`: Canonical training worker.
- `eval_controller.py`: Canonical evaluation launcher.
- `eval_worker.py`: Canonical evaluation worker.
- `student_model_controller.py`: Student model experiment launcher.
- `student_model_worker.py`: Student model baseline and quick fine-tune worker.
- `eda/`: One-off analysis scripts.
- `full_fine_tuned_model/`: Checked-in model metadata artifact; shard files are not present in this checkout.
- `tests/`: Unit tests for configuration and experiment config loading.
- `deprecated/`: Historical archive, including old `student_model_test/` scripts and the old `unsloth_train.py` entrypoint.

## Tests

Run unit tests:

```bash
python -m unittest discover -v
```

Run a syntax check over active workflow files:

```bash
python -m compileall -q llm_ft tests train_controller.py train_worker.py eval_controller.py eval_worker.py student_model_controller.py student_model_worker.py dataset
```

## License

This project is licensed under the MIT License. See `LICENSE` for details.
