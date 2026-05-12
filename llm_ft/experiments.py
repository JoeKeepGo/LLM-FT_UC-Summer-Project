"""JSON-backed experiment configuration loading."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional

from llm_ft.config import (
    BASELINE_RESULTS_DIR,
    DATASET_SPLIT_TEST_FILE,
    DATASET_SPLIT_TRAIN_FILE,
    EVAL_RESULTS_DIR,
    FINE_TUNED_MODEL_DIR,
    FINETUNING_TEST_RESULTS_DIR,
    MODEL_ID,
    TEST_FILE,
    run_output_dir,
)


DEFAULT_TEMPLATE_VALUES = {
    "MODEL_ID": MODEL_ID,
    "DATASET_SPLIT_TRAIN_FILE": DATASET_SPLIT_TRAIN_FILE,
    "DATASET_SPLIT_TEST_FILE": DATASET_SPLIT_TEST_FILE,
    "TEST_FILE": TEST_FILE,
    "EVAL_RESULTS_DIR": EVAL_RESULTS_DIR,
    "BASELINE_RESULTS_DIR": BASELINE_RESULTS_DIR,
    "FINETUNING_TEST_RESULTS_DIR": FINETUNING_TEST_RESULTS_DIR,
    "FINE_TUNED_MODEL_DIR": FINE_TUNED_MODEL_DIR,
}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Experiment config must be a JSON object: {path}")
    return data


def _template_values(overrides: Optional[Mapping[str, str]] = None) -> dict[str, str]:
    values = dict(DEFAULT_TEMPLATE_VALUES)
    if overrides:
        values.update({key: str(value) for key, value in overrides.items()})
    return values


def _resolve_templates(value: Any, values: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        resolved = value
        for key, replacement in values.items():
            resolved = resolved.replace("{" + key + "}", replacement)
        return resolved
    if isinstance(value, list):
        return [_resolve_templates(item, values) for item in value]
    if isinstance(value, dict):
        return {key: _resolve_templates(item, values) for key, item in value.items()}
    return value


def _merge_defaults(defaults: Mapping[str, Any], experiments: list[Any]) -> list[dict[str, Any]]:
    merged = []
    for raw_experiment in experiments:
        if not isinstance(raw_experiment, dict):
            raise ValueError("Each experiment must be a JSON object.")
        experiment = dict(defaults)
        experiment.update(raw_experiment)
        merged.append(experiment)
    return merged


def load_train_experiments(
    path: str | Path = "configs/train_experiments.json",
    template_values: Optional[Mapping[str, str]] = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    values = _template_values(template_values)
    data = _resolve_templates(_load_json(Path(path)), values)
    defaults = data.get("defaults", {})
    experiments = data.get("experiments", [])

    if not isinstance(defaults, dict):
        raise ValueError("train experiment defaults must be a JSON object.")
    if not isinstance(experiments, list):
        raise ValueError("train experiments must be a JSON array.")

    merged = _merge_defaults(defaults, experiments)
    for experiment in merged:
        if not experiment.get("run_name"):
            raise ValueError("Each train experiment requires run_name.")
        global_batch = experiment.get("global_batch")
        batch_size = experiment.get("batch_size")
        if (
            experiment.get("grad_accumulation") is None
            and isinstance(global_batch, int)
            and isinstance(batch_size, int)
        ):
            experiment["grad_accumulation"] = max(1, global_batch // batch_size)
    return dict(defaults), merged


def load_eval_config(
    path: str | Path = "configs/eval_experiments.json",
    template_values: Optional[Mapping[str, str]] = None,
) -> dict[str, Any]:
    values = _template_values(template_values)
    data = _resolve_templates(_load_json(Path(path)), values)
    defaults = data.get("defaults", {})
    experiments = data.get("experiments", [])

    if not isinstance(defaults, dict):
        raise ValueError("eval experiment defaults must be a JSON object.")
    if not isinstance(experiments, list):
        raise ValueError("eval experiments must be a JSON array.")

    merged = _merge_defaults(defaults, experiments)
    for experiment in merged:
        if not experiment.get("run_name"):
            raise ValueError("Each eval experiment requires run_name.")
        if not experiment.get("mode"):
            raise ValueError("Each eval experiment requires mode.")
        checkpoint_run_name = experiment.pop("checkpoint_run_name", None)
        if checkpoint_run_name and not experiment.get("checkpoint_path"):
            experiment["checkpoint_path"] = str(run_output_dir(values["FINE_TUNED_MODEL_DIR"], str(checkpoint_run_name)))

    return {
        "common": dict(defaults),
        "experiments": merged,
        "auto_discover": bool(data.get("auto_discover", False)),
        "model_roots": data.get("model_roots", [values["FINE_TUNED_MODEL_DIR"]]),
        "include_base_model": bool(data.get("include_base_model", False)),
        "include_regex": data.get("include_regex"),
        "exclude_regex": data.get("exclude_regex"),
        "skip_existing": bool(data.get("skip_existing", True)),
    }


def load_student_model_config(
    path: str | Path = "configs/student_model_experiments.json",
    template_values: Optional[Mapping[str, str]] = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    values = _template_values(template_values)
    data = _resolve_templates(_load_json(Path(path)), values)
    defaults = data.get("defaults", {})
    experiments = data.get("experiments", [])

    if not isinstance(defaults, dict):
        raise ValueError("student model defaults must be a JSON object.")
    if not isinstance(experiments, list):
        raise ValueError("student model experiments must be a JSON array.")

    merged = _merge_defaults(defaults, experiments)
    for experiment in merged:
        if not experiment.get("run_name"):
            raise ValueError("Each student model experiment requires run_name.")
        if not experiment.get("mode"):
            raise ValueError("Each student model experiment requires mode.")
        if not experiment.get("model_id"):
            raise ValueError("Each student model experiment requires model_id.")
        if not experiment.get("model_alias"):
            raise ValueError("Each student model experiment requires model_alias.")
    return dict(defaults), merged
