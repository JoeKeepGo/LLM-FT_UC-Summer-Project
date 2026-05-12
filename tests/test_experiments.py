import json
import tempfile
import unittest
from pathlib import Path


class ExperimentConfigTests(unittest.TestCase):
    def test_load_train_experiments_merges_defaults_and_computes_grad_accumulation(self):
        from llm_ft.experiments import load_train_experiments

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.json"
            path.write_text(
                json.dumps(
                    {
                        "defaults": {
                            "output_dir_base": "{FINE_TUNED_MODEL_DIR}",
                            "seed": 42,
                            "global_batch": 64,
                            "learning_rate": 0.0002,
                        },
                        "experiments": [
                            {
                                "run_name": "ExpA",
                                "batch_size": 32,
                                "learning_rate": 0.00005,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            common, experiments = load_train_experiments(
                path,
                template_values={"FINE_TUNED_MODEL_DIR": "/models"},
            )

        self.assertEqual(common["output_dir_base"], "/models")
        self.assertEqual(experiments[0]["seed"], 42)
        self.assertEqual(experiments[0]["learning_rate"], 0.00005)
        self.assertEqual(experiments[0]["grad_accumulation"], 2)

    def test_load_eval_config_resolves_checkpoint_run_name(self):
        from llm_ft.experiments import load_eval_config

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "eval.json"
            path.write_text(
                json.dumps(
                    {
                        "defaults": {
                            "base_model_path": "{MODEL_ID}",
                            "test_file": "{TEST_FILE}",
                            "output_dir": "{EVAL_RESULTS_DIR}",
                        },
                        "model_roots": ["{FINE_TUNED_MODEL_DIR}"],
                        "experiments": [
                            {
                                "run_name": "ExpA",
                                "mode": "lora",
                                "checkpoint_run_name": "ExpA",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            config = load_eval_config(
                path,
                template_values={
                    "MODEL_ID": "model/id",
                    "TEST_FILE": "/data/test.jsonl",
                    "EVAL_RESULTS_DIR": "/eval",
                    "FINE_TUNED_MODEL_DIR": "/models",
                },
            )

        self.assertEqual(config["common"]["base_model_path"], "model/id")
        self.assertEqual(config["model_roots"], ["/models"])
        self.assertEqual(config["experiments"][0]["checkpoint_path"], "/models/ExpA")

    def test_load_student_model_config_merges_defaults(self):
        from llm_ft.experiments import load_student_model_config

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "student.json"
            path.write_text(
                json.dumps(
                    {
                        "defaults": {
                            "train_file": "{DATASET_SPLIT_TRAIN_FILE}",
                            "test_file": "{DATASET_SPLIT_TEST_FILE}",
                            "output_dir": "{BASELINE_RESULTS_DIR}",
                            "num_runs": 1,
                        },
                        "experiments": [
                            {
                                "run_name": "QwenBaseline",
                                "mode": "baseline",
                                "model_id": "{MODEL_ID}",
                                "model_alias": "Qwen",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            common, experiments = load_student_model_config(
                path,
                template_values={
                    "DATASET_SPLIT_TRAIN_FILE": "/data/train.jsonl",
                    "DATASET_SPLIT_TEST_FILE": "/data/test.jsonl",
                    "BASELINE_RESULTS_DIR": "/baseline",
                    "MODEL_ID": "model/id",
                },
            )

        self.assertEqual(common["train_file"], "/data/train.jsonl")
        self.assertEqual(experiments[0]["test_file"], "/data/test.jsonl")
        self.assertEqual(experiments[0]["output_dir"], "/baseline")
        self.assertEqual(experiments[0]["model_id"], "model/id")
        self.assertEqual(experiments[0]["num_runs"], 1)


if __name__ == "__main__":
    unittest.main()
