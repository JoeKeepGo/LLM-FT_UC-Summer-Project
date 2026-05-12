import argparse
import gc
import json
import os
import random
import re
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from llm_ft.config import (
    BASELINE_RESULTS_DIR,
    DATASET_SPLIT_TEST_FILE,
    DATASET_SPLIT_TRAIN_FILE,
    FINETUNING_TEST_RESULTS_DIR,
    HF_TOKEN,
    MODEL_ID,
    run_output_dir,
)


SCHEMA_DEFINITION = """
You are a content moderation expert. Analyze the comment and output a JSON object with:
1. "original_comment": The input text.
2. "impact_level": Integer 1-5 (1=Negligible, 2=Low, 3=Medium, 4=High, 5=Severe).
3. "harm_category": List of strings.
4. "target_identity": List of strings.
5. "reasoning": A coherent text explanation.
6. "action_suggestion": String.
"""


def _parse_bool(value):
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in ("1", "true", "yes", "y", "on"):
        return True
    if lowered in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args():
    parser = argparse.ArgumentParser(description="Student model baseline/fine-tune worker")
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--mode", required=True, choices=["baseline", "baseline_repeated", "finetune_lora"])
    parser.add_argument("--model_id", default=MODEL_ID)
    parser.add_argument("--model_alias", default="Qwen-3-4B")
    parser.add_argument("--train_file", default=DATASET_SPLIT_TRAIN_FILE)
    parser.add_argument("--test_file", default=DATASET_SPLIT_TEST_FILE)
    parser.add_argument("--output_dir", default=BASELINE_RESULTS_DIR)
    parser.add_argument("--few_shot_count", type=int, default=3)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--do_sample", type=_parse_bool, default=True)
    parser.add_argument("--load_in_8bit", type=_parse_bool, default=True)
    parser.add_argument("--load_in_4bit", type=_parse_bool, default=False)
    parser.add_argument("--seed_base", type=int, default=42)
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--grad_accumulation", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_steps", type=int, default=60)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clean_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except RuntimeError:
            pass


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_json(data, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    def default_converter(value):
        if isinstance(value, (np.int64, np.int32)):
            return int(value)
        if isinstance(value, (np.float64, np.float32)):
            return float(value)
        return str(value)

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False, default=default_converter)


def extract_json_from_text(text):
    try:
        if "<think>" in text:
            text = text.split("</think>")[-1].strip()
        match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        match = re.search(r"(\{.*\})", text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        return json.loads(text)
    except Exception:
        return None


def quantization_config(args):
    if args.load_in_4bit:
        return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    if args.load_in_8bit:
        return BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0, llm_int8_has_fp16_weight=False)
    return None


def load_generation_model(args):
    print(f"Loading model: {args.model_id}")
    clean_memory()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=HF_TOKEN, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.pad_token_id = 0

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=quantization_config(args),
        device_map="auto",
        trust_remote_code=True,
        token=HF_TOKEN,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.eval()
    return model, tokenizer


def build_few_shot_prompt(tokenizer, comment, few_shot_examples):
    messages = []
    instruction_text = f"{SCHEMA_DEFINITION}\n\nHere are some examples of how to analyze comments:\n"
    if few_shot_examples:
        first = few_shot_examples[0]
        clean_first = {
            key: first[key]
            for key in ["original_comment", "impact_level", "harm_category", "target_identity", "reasoning", "action_suggestion"]
            if key in first
        }
        messages.append({"role": "user", "content": f"{instruction_text}\n\nAnalyze this: {first['original_comment']}"})
        messages.append({"role": "assistant", "content": json.dumps(clean_first, ensure_ascii=False)})
    else:
        messages.append({"role": "user", "content": instruction_text})

    for example in few_shot_examples[1:]:
        clean_example = {
            key: example[key]
            for key in ["original_comment", "impact_level", "harm_category", "target_identity", "reasoning", "action_suggestion"]
            if key in example
        }
        messages.append({"role": "user", "content": f"Analyze this: {example['original_comment']}"})
        messages.append({"role": "assistant", "content": json.dumps(clean_example, ensure_ascii=False)})

    messages.append({"role": "user", "content": f"Analyze this: {comment}"})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def score_results(test_data, raw_items):
    y_true = []
    y_pred = []
    parse_errors = 0
    results = []

    for source_item, raw_response in zip(test_data, raw_items):
        pred_json = extract_json_from_text(raw_response)
        gt_level = source_item.get("impact_level", 1)
        pred_level = 1
        valid = False
        if pred_json:
            try:
                parsed_level = int(pred_json.get("impact_level", 1))
                pred_level = max(1, min(5, parsed_level))
                y_true.append(gt_level)
                y_pred.append(pred_level)
                valid = True
            except Exception:
                pass
        if not valid:
            parse_errors += 1

        results.append(
            {
                "id": source_item.get("id", "unknown"),
                "original_comment": source_item.get("original_comment"),
                "ground_truth_level": gt_level,
                "prediction_level": pred_level if valid else "ERROR",
                "is_valid_format": valid,
                "parsed_json": pred_json,
                "raw_response": raw_response,
            }
        )

    total = len(test_data)
    metrics = {
        "total_samples": total,
        "parse_errors": parse_errors,
        "format_adherence": (total - parse_errors) / total if total else 0.0,
        "impact_level_f1_macro": f1_score(y_true, y_pred, average="macro") if y_true else 0.0,
        "impact_level_accuracy": accuracy_score(y_true, y_pred) if y_true else 0.0,
    }
    report = classification_report(y_true, y_pred, labels=[1, 2, 3, 4, 5], zero_division=0) if y_true else ""
    return results, metrics, report


def run_baseline_once(args, model, tokenizer, train_data, test_data, output_dir, run_index):
    few_shot_examples = train_data[: max(0, args.few_shot_count)]
    raw_responses = []
    set_seed(args.seed_base + run_index)

    for item in tqdm(test_data, desc=f"{args.model_alias} run {run_index}"):
        prompt = build_few_shot_prompt(tokenizer, item["original_comment"], few_shot_examples)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                do_sample=args.do_sample,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated_tokens = outputs[0][inputs.input_ids.shape[1] :]
        raw_responses.append(tokenizer.decode(generated_tokens, skip_special_tokens=True).strip())

    results, metrics, report = score_results(test_data, raw_responses)
    metrics.update({"model": args.model_alias, "run_index": run_index, "seed": args.seed_base + run_index})

    suffix = f"_run_{run_index}" if args.mode == "baseline_repeated" else ""
    save_json(results, os.path.join(output_dir, f"predictions{suffix}.json"))
    save_json(metrics, os.path.join(output_dir, f"metrics{suffix}.json"))
    if report:
        with open(os.path.join(output_dir, f"report{suffix}.txt"), "w", encoding="utf-8") as handle:
            handle.write(report)
    return metrics


def run_baseline(args, output_dir):
    train_data = load_jsonl(args.train_file)
    test_data = load_jsonl(args.test_file)
    if not train_data or not test_data:
        raise RuntimeError("Train or test data is empty.")

    model, tokenizer = load_generation_model(args)
    runs = args.num_runs if args.mode == "baseline_repeated" else 1
    run_metrics = []
    try:
        for run_index in range(1, runs + 1):
            run_metrics.append(run_baseline_once(args, model, tokenizer, train_data, test_data, output_dir, run_index))
    finally:
        del model
        clean_memory()

    summary = {
        "run_name": args.run_name,
        "mode": args.mode,
        "model": args.model_alias,
        "total_runs": runs,
        "runs": run_metrics,
        "aggregated_metrics": {
            "f1_macro_mean": float(np.mean([item["impact_level_f1_macro"] for item in run_metrics])),
            "f1_macro_std": float(np.std([item["impact_level_f1_macro"] for item in run_metrics])),
            "accuracy_mean": float(np.mean([item["impact_level_accuracy"] for item in run_metrics])),
            "accuracy_std": float(np.std([item["impact_level_accuracy"] for item in run_metrics])),
            "format_mean": float(np.mean([item["format_adherence"] for item in run_metrics])),
            "format_std": float(np.std([item["format_adherence"] for item in run_metrics])),
        },
    }
    save_json(summary, os.path.join(output_dir, "summary.json"))


def run_finetune_lora(args, output_dir):
    from datasets import Dataset
    from peft import LoraConfig, PeftModel
    from trl import SFTConfig, SFTTrainer

    train_data = load_jsonl(args.train_file)
    test_data = load_jsonl(args.test_file)
    if not train_data or not test_data:
        raise RuntimeError("Train or test data is empty.")

    set_seed(args.seed_base)
    clean_memory()
    adapter_dir = os.path.join(output_dir, "adapter")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=HF_TOKEN, trust_remote_code=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.pad_token_id = 0

    def preprocess(example):
        comment = example["original_comment"]
        target = {
            "original_comment": comment,
            "impact_level": example["impact_level"],
            "harm_category": example["harm_category"],
            "target_identity": example["target_identity"],
            "reasoning": example["reasoning"],
            "action_suggestion": example["action_suggestion"],
        }
        full_text = (
            f"{SCHEMA_DEFINITION}\n\nUser: Analyze this: {comment}\n"
            f"Assistant: {json.dumps(target, ensure_ascii=False)}{tokenizer.eos_token}"
        )
        tokenized = tokenizer(full_text, truncation=True, max_length=args.max_seq_length, padding="max_length")
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized_dataset = Dataset.from_list(train_data).map(preprocess, batched=False)
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        token=HF_TOKEN,
        attn_implementation="sdpa",
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    peft_config = LoraConfig(
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        r=args.lora_rank,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.grad_accumulation,
        learning_rate=args.learning_rate,
        logging_steps=10,
        max_steps=args.max_steps,
        optim="paged_adamw_32bit",
        save_strategy="no",
        fp16=False,
        bf16=True,
        report_to="none",
        gradient_checkpointing=True,
        max_length=args.max_seq_length,
        packing=False,
        dataset_text_field=None,
    )
    trainer = SFTTrainer(model=model, train_dataset=tokenized_dataset, peft_config=peft_config, args=training_args, processing_class=tokenizer)
    trainer.train()
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    del model, trainer
    clean_memory()

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    tokenizer.padding_side = "left"
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        token=HF_TOKEN,
        torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()

    raw_responses = []
    prompts = [f"{SCHEMA_DEFINITION}\n\nUser: Analyze this: {item['original_comment']}\nAssistant:" for item in test_data]
    for start in tqdm(range(0, len(prompts), args.eval_batch_size), desc=f"{args.model_alias} eval"):
        batch_prompts = prompts[start : start + args.eval_batch_size]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to("cuda")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                do_sample=args.do_sample,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated_tokens = outputs[:, inputs.input_ids.shape[1] :]
        raw_responses.extend([text.strip() for text in tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)])

    results, metrics, report = score_results(test_data, raw_responses)
    metrics.update({"model": args.model_alias, "run_name": args.run_name, "mode": args.mode})
    save_json(results, os.path.join(output_dir, "predictions.json"))
    save_json(metrics, os.path.join(output_dir, "metrics.json"))
    if report:
        with open(os.path.join(output_dir, "report.txt"), "w", encoding="utf-8") as handle:
            handle.write(report)
    del model, base_model
    clean_memory()


def main():
    args = parse_args()
    if args.mode == "finetune_lora" and args.output_dir == BASELINE_RESULTS_DIR:
        args.output_dir = FINETUNING_TEST_RESULTS_DIR
    output_dir = str(run_output_dir(args.output_dir, args.run_name))
    os.makedirs(output_dir, exist_ok=True)
    print(f"Student model output: {output_dir}")

    if args.mode in ("baseline", "baseline_repeated"):
        run_baseline(args, output_dir)
    elif args.mode == "finetune_lora":
        run_finetune_lora(args, output_dir)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
