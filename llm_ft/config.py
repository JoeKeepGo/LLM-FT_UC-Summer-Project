"""Central project configuration and .env loading.

The project intentionally avoids depending on python-dotenv so setup stays
compatible with the existing environment file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional


DEFAULT_BASE_DIR = Path("/home/data601/project")
DEFAULT_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"


def _strip_inline_comment(value: str) -> str:
    quote: Optional[str] = None
    escaped = False
    for index, char in enumerate(value):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if quote:
            if char == quote:
                quote = None
            continue
        if char in ("'", '"'):
            quote = char
            continue
        if char == "#":
            return value[:index].rstrip()
    return value.strip()


def _unquote(value: str) -> str:
    value = _strip_inline_comment(value)
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


def load_dotenv(path: Optional[os.PathLike[str] | str] = None, override: bool = False) -> Dict[str, str]:
    """Load simple KEY=VALUE pairs into the process environment.

    Existing environment variables win unless ``override`` is true.
    """

    env_path = Path(path or os.environ.get("LLM_FT_ENV_FILE", ".env"))
    loaded: Dict[str, str] = {}
    if not env_path.exists():
        return loaded

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        parsed = _unquote(value.strip())
        loaded[key] = parsed
        if override or key not in os.environ:
            os.environ[key] = parsed
    return loaded


def _env_path(name: str, default: Path, env: Mapping[str, str]) -> Path:
    value = env.get(name)
    return Path(value).expanduser() if value else default


def _env_str(name: str, default: str, env: Mapping[str, str]) -> str:
    return env.get(name, default)


def safe_run_dir_name(run_name: str) -> str:
    """Return a single directory-safe name for a training or evaluation run."""

    cleaned = str(run_name).strip()
    if not cleaned:
        raise ValueError("run_name must not be empty")
    return cleaned.replace("/", "_").replace("\\", "_")


def run_output_dir(output_dir_base: os.PathLike[str] | str, run_name: str) -> Path:
    """Build a run output directory from a base directory and task/run name."""

    return Path(output_dir_base).expanduser() / safe_run_dir_name(run_name)


@dataclass(frozen=True)
class Settings:
    base_dir: Path
    model_id: str
    teacher_api_model: str
    reprocess_api_model: str
    deepseek_api_base_url: str
    deepseek_api_key: Optional[str]
    hf_token: Optional[str]
    wandb_api_key: Optional[str]
    wandb_project: str
    wandb_entity: str
    dataset_dir: Path
    dataset_tmp_dir: Path
    dataset_split_dir: Path
    dataset_tmp_split_dir: Path
    dataset_split_train_file: Path
    dataset_split_test_file: Path
    dataset_tmp_split_train_file: Path
    dataset_tmp_split_test_file: Path
    constructed_dataset_dir: Path
    train_dir: Path
    test_dir: Path
    train_file: Path
    test_file: Path
    synthetic_final_file: Path
    synthetic_final_tmp_file: Path
    synthetic_v7_file: Path
    synthetic_v8_file: Path
    train_quality_file: Path
    train_quality_hybrid_file: Path
    train_quality_hybrid_cot_file: Path
    gold_standard_file: Path
    obsolete_gold_standard_file: Path
    train_reannotated_file: Path
    test_reannotated_file: Path
    tmp_train_reannotated_file: Path
    tmp_test_reannotated_file: Path
    fine_tuned_model_dir: Path
    eval_results_dir: Path
    baseline_results_dir: Path
    finetuning_test_results_dir: Path


def build_settings(load_env: bool = True, env_file: Optional[os.PathLike[str] | str] = None) -> Settings:
    if load_env:
        load_dotenv(env_file)

    env = os.environ
    base_dir = _env_path("LLM_FT_BASE_DIR", DEFAULT_BASE_DIR, env)
    dataset_dir = _env_path("LLM_FT_DATASET_DIR", base_dir / "dataset", env)
    dataset_tmp_dir = _env_path("LLM_FT_DATASET_TMP_DIR", dataset_dir / "tmp", env)
    dataset_split_dir = _env_path("LLM_FT_DATASET_SPLIT_DIR", base_dir / "dataset_split", env)
    dataset_tmp_split_dir = _env_path(
        "LLM_FT_DATASET_TMP_SPLIT_DIR",
        dataset_tmp_dir / "dataset_split",
        env,
    )
    constructed_dataset_dir = _env_path(
        "LLM_FT_CONSTRUCTED_DATASET_DIR",
        base_dir / "constructed_dataset",
        env,
    )
    train_dir = _env_path("LLM_FT_TRAIN_DIR", dataset_dir / "train", env)
    test_dir = _env_path("LLM_FT_TEST_DIR", dataset_dir / "test", env)

    return Settings(
        base_dir=base_dir,
        model_id=_env_str("LLM_FT_MODEL_ID", DEFAULT_MODEL_ID, env),
        teacher_api_model=_env_str("LLM_FT_TEACHER_API_MODEL", "deepseek-reasoner", env),
        reprocess_api_model=_env_str("LLM_FT_REPROCESS_API_MODEL", "deepseek-chat", env),
        deepseek_api_base_url=_env_str("DEEPSEEK_API_BASE_URL", "https://api.deepseek.com", env),
        deepseek_api_key=env.get("DEEPSEEK_API_KEY") or env.get("API_KEY"),
        hf_token=env.get("HF_TOKEN"),
        wandb_api_key=env.get("WANDB_API_KEY"),
        wandb_project=_env_str("WANDB_PROJECT", "DATA601", env),
        wandb_entity=_env_str("WANDB_ENTITY", "joeyang97", env),
        dataset_dir=dataset_dir,
        dataset_tmp_dir=dataset_tmp_dir,
        dataset_split_dir=dataset_split_dir,
        dataset_tmp_split_dir=dataset_tmp_split_dir,
        dataset_split_train_file=_env_path(
            "LLM_FT_DATASET_SPLIT_TRAIN_FILE",
            dataset_split_dir / "train.jsonl",
            env,
        ),
        dataset_split_test_file=_env_path(
            "LLM_FT_DATASET_SPLIT_TEST_FILE",
            dataset_split_dir / "test.jsonl",
            env,
        ),
        dataset_tmp_split_train_file=_env_path(
            "LLM_FT_DATASET_TMP_SPLIT_TRAIN_FILE",
            dataset_tmp_split_dir / "train.jsonl",
            env,
        ),
        dataset_tmp_split_test_file=_env_path(
            "LLM_FT_DATASET_TMP_SPLIT_TEST_FILE",
            dataset_tmp_split_dir / "test.jsonl",
            env,
        ),
        constructed_dataset_dir=constructed_dataset_dir,
        train_dir=train_dir,
        test_dir=test_dir,
        train_file=_env_path("LLM_FT_TRAIN_FILE", train_dir / "train.jsonl", env),
        test_file=_env_path("LLM_FT_TEST_FILE", test_dir / "test.jsonl", env),
        synthetic_final_file=_env_path(
            "LLM_FT_SYNTHETIC_FINAL_FILE",
            constructed_dataset_dir / "synthetic_train_final.jsonl",
            env,
        ),
        synthetic_final_tmp_file=_env_path(
            "LLM_FT_SYNTHETIC_FINAL_TMP_FILE",
            dataset_tmp_dir / "synthetic_train_final.jsonl",
            env,
        ),
        synthetic_v7_file=_env_path(
            "LLM_FT_SYNTHETIC_V7_FILE",
            dataset_tmp_dir / "synthetic_train_final_v7prompt.jsonl",
            env,
        ),
        synthetic_v8_file=_env_path(
            "LLM_FT_SYNTHETIC_V8_FILE",
            dataset_tmp_dir / "synthetic_train_final_v8prompt.jsonl",
            env,
        ),
        train_quality_file=_env_path(
            "LLM_FT_TRAIN_QUALITY_FILE",
            train_dir / "train_quality.jsonl",
            env,
        ),
        train_quality_hybrid_file=_env_path(
            "LLM_FT_TRAIN_QUALITY_HYBRID_FILE",
            train_dir / "train_quality_hybrid.jsonl",
            env,
        ),
        train_quality_hybrid_cot_file=_env_path(
            "LLM_FT_TRAIN_QUALITY_HYBRID_COT_FILE",
            train_dir / "train_quality_hybrid_cot.jsonl",
            env,
        ),
        gold_standard_file=_env_path(
            "LLM_FT_GOLD_STANDARD_FILE",
            base_dir / "gold_standard.json",
            env,
        ),
        obsolete_gold_standard_file=_env_path(
            "LLM_FT_OBSOLETE_GOLD_STANDARD_FILE",
            dataset_tmp_dir / "obsolete_files" / "gold_standard.json",
            env,
        ),
        train_reannotated_file=_env_path(
            "LLM_FT_TRAIN_REANNOTATED_FILE",
            dataset_split_dir / "train_reannotated.jsonl",
            env,
        ),
        test_reannotated_file=_env_path(
            "LLM_FT_TEST_REANNOTATED_FILE",
            dataset_split_dir / "test_reannotated.jsonl",
            env,
        ),
        tmp_train_reannotated_file=_env_path(
            "LLM_FT_TMP_TRAIN_REANNOTATED_FILE",
            dataset_tmp_split_dir / "train_reannotated.jsonl",
            env,
        ),
        tmp_test_reannotated_file=_env_path(
            "LLM_FT_TMP_TEST_REANNOTATED_FILE",
            dataset_tmp_split_dir / "test_reannotated.jsonl",
            env,
        ),
        fine_tuned_model_dir=_env_path(
            "LLM_FT_FINE_TUNED_MODEL_DIR",
            base_dir / "fine_tuned_model",
            env,
        ),
        eval_results_dir=_env_path("LLM_FT_EVAL_RESULTS_DIR", base_dir / "eval_results", env),
        baseline_results_dir=_env_path(
            "LLM_FT_BASELINE_RESULTS_DIR",
            base_dir / "baseline_results",
            env,
        ),
        finetuning_test_results_dir=_env_path(
            "LLM_FT_FINETUNING_TEST_RESULTS_DIR",
            base_dir / "finetuning_test_results_3",
            env,
        ),
    )


SETTINGS = build_settings()

BASE_DIR = str(SETTINGS.base_dir)
MODEL_ID = SETTINGS.model_id
TEACHER_API_MODEL = SETTINGS.teacher_api_model
REPROCESS_API_MODEL = SETTINGS.reprocess_api_model
DEEPSEEK_API_BASE_URL = SETTINGS.deepseek_api_base_url
DEEPSEEK_API_KEY = SETTINGS.deepseek_api_key
HF_TOKEN = SETTINGS.hf_token
WANDB_API_KEY = SETTINGS.wandb_api_key
WANDB_PROJECT = SETTINGS.wandb_project
WANDB_ENTITY = SETTINGS.wandb_entity

DATASET_DIR = str(SETTINGS.dataset_dir)
DATASET_TMP_DIR = str(SETTINGS.dataset_tmp_dir)
DATASET_SPLIT_DIR = str(SETTINGS.dataset_split_dir)
DATASET_TMP_SPLIT_DIR = str(SETTINGS.dataset_tmp_split_dir)
DATASET_SPLIT_TRAIN_FILE = str(SETTINGS.dataset_split_train_file)
DATASET_SPLIT_TEST_FILE = str(SETTINGS.dataset_split_test_file)
DATASET_TMP_SPLIT_TRAIN_FILE = str(SETTINGS.dataset_tmp_split_train_file)
DATASET_TMP_SPLIT_TEST_FILE = str(SETTINGS.dataset_tmp_split_test_file)
CONSTRUCTED_DATASET_DIR = str(SETTINGS.constructed_dataset_dir)
TRAIN_FILE = str(SETTINGS.train_file)
TEST_FILE = str(SETTINGS.test_file)
SYNTHETIC_FINAL_FILE = str(SETTINGS.synthetic_final_file)
SYNTHETIC_FINAL_TMP_FILE = str(SETTINGS.synthetic_final_tmp_file)
SYNTHETIC_V7_FILE = str(SETTINGS.synthetic_v7_file)
SYNTHETIC_V8_FILE = str(SETTINGS.synthetic_v8_file)
TRAIN_QUALITY_FILE = str(SETTINGS.train_quality_file)
TRAIN_QUALITY_HYBRID_FILE = str(SETTINGS.train_quality_hybrid_file)
TRAIN_QUALITY_HYBRID_COT_FILE = str(SETTINGS.train_quality_hybrid_cot_file)
GOLD_STANDARD_FILE = str(SETTINGS.gold_standard_file)
OBSOLETE_GOLD_STANDARD_FILE = str(SETTINGS.obsolete_gold_standard_file)
TRAIN_REANNOTATED_FILE = str(SETTINGS.train_reannotated_file)
TEST_REANNOTATED_FILE = str(SETTINGS.test_reannotated_file)
TMP_TRAIN_REANNOTATED_FILE = str(SETTINGS.tmp_train_reannotated_file)
TMP_TEST_REANNOTATED_FILE = str(SETTINGS.tmp_test_reannotated_file)
FINE_TUNED_MODEL_DIR = str(SETTINGS.fine_tuned_model_dir)
EVAL_RESULTS_DIR = str(SETTINGS.eval_results_dir)
BASELINE_RESULTS_DIR = str(SETTINGS.baseline_results_dir)
FINETUNING_TEST_RESULTS_DIR = str(SETTINGS.finetuning_test_results_dir)
