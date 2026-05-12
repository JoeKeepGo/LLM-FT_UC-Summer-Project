import os
import tempfile
import unittest
from pathlib import Path


class ConfigTests(unittest.TestCase):
    def test_load_dotenv_sets_values_without_overwriting_existing_env(self):
        from llm_ft.config import load_dotenv

        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text(
                "\n".join(
                    [
                        "API_KEY=from-file",
                        'HF_TOKEN="hf-file"',
                        "WANDB_API_KEY='wandb-file'",
                        "EXISTING_VALUE=from-file",
                        "COMMENTED=value # this comment is ignored",
                    ]
                ),
                encoding="utf-8",
            )

            old_existing = os.environ.get("EXISTING_VALUE")
            old_api_key = os.environ.get("API_KEY")
            old_hf_token = os.environ.get("HF_TOKEN")
            old_wandb = os.environ.get("WANDB_API_KEY")
            old_commented = os.environ.get("COMMENTED")
            try:
                os.environ["EXISTING_VALUE"] = "already-set"
                for key in ("API_KEY", "HF_TOKEN", "WANDB_API_KEY", "COMMENTED"):
                    os.environ.pop(key, None)

                loaded = load_dotenv(env_file)

                self.assertEqual(loaded["API_KEY"], "from-file")
                self.assertEqual(os.environ["API_KEY"], "from-file")
                self.assertEqual(os.environ["HF_TOKEN"], "hf-file")
                self.assertEqual(os.environ["WANDB_API_KEY"], "wandb-file")
                self.assertEqual(os.environ["EXISTING_VALUE"], "already-set")
                self.assertEqual(os.environ["COMMENTED"], "value")
            finally:
                for key, value in {
                    "EXISTING_VALUE": old_existing,
                    "API_KEY": old_api_key,
                    "HF_TOKEN": old_hf_token,
                    "WANDB_API_KEY": old_wandb,
                    "COMMENTED": old_commented,
                }.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value

    def test_build_settings_uses_base_dir_and_env_overrides(self):
        from llm_ft.config import build_settings

        old_base = os.environ.get("LLM_FT_BASE_DIR")
        old_train = os.environ.get("LLM_FT_TRAIN_FILE")
        old_model = os.environ.get("LLM_FT_MODEL_ID")
        try:
            os.environ["LLM_FT_BASE_DIR"] = "/tmp/llm-ft-base"
            os.environ["LLM_FT_TRAIN_FILE"] = "/tmp/custom-train.jsonl"
            os.environ["LLM_FT_MODEL_ID"] = "custom/model"

            settings = build_settings(load_env=False)

            self.assertEqual(settings.base_dir, Path("/tmp/llm-ft-base"))
            self.assertEqual(settings.model_id, "custom/model")
            self.assertEqual(settings.train_file, Path("/tmp/custom-train.jsonl"))
            self.assertEqual(settings.test_file, Path("/tmp/llm-ft-base/dataset/test/test.jsonl"))
            self.assertEqual(settings.fine_tuned_model_dir, Path("/tmp/llm-ft-base/fine_tuned_model"))
            self.assertEqual(settings.eval_results_dir, Path("/tmp/llm-ft-base/eval_results"))
        finally:
            for key, value in {
                "LLM_FT_BASE_DIR": old_base,
                "LLM_FT_TRAIN_FILE": old_train,
                "LLM_FT_MODEL_ID": old_model,
            }.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_run_output_dir_uses_single_safe_directory_name(self):
        from llm_ft.config import run_output_dir

        output = run_output_dir("/models", "full/20k 5e5")

        self.assertEqual(output, Path("/models/full_20k 5e5"))

    def test_run_output_dir_rejects_empty_run_name(self):
        from llm_ft.config import run_output_dir

        with self.assertRaises(ValueError):
            run_output_dir("/models", "   ")


if __name__ == "__main__":
    unittest.main()
