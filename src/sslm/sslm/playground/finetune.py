from __future__ import annotations

import argparse
from pathlib import Path


def write_qlora_recipe(path: Path, *, base_model: str, dataset: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""# Starter SFT/QLoRA recipe for reasoning-model adaptation.
# Install the training extra first:
#   pip install -e src/sslm[train]
#
# This is intentionally a config artifact, not an automatic training launch.
# Review dataset fields, license, target modules, and expected reasoning traces
# before running expensive fine-tuning.
base_model: {base_model}
dataset: {dataset}
method: qlora_sft
load_in_4bit: true
bnb_4bit_quant_type: nf4
bnb_4bit_compute_dtype: bfloat16
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj
sequence_length: 4096
learning_rate: 2.0e-5
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
num_train_epochs: 1
warmup_ratio: 0.03
packing: true
output_dir: .data/sslm/finetunes/{base_model.replace("/", "_")}
eval_benchmarks:
  - gsm8k
  - gpqa_diamond
  - mmlu_pro
  - ifeval
""",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write starter fine-tuning recipes.")
    parser.add_argument("--base-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--dataset", default="jsonl://.data/reasoning_sft.jsonl")
    parser.add_argument("--output", type=Path, default=Path(".data/sslm/finetune/qlora.yaml"))
    args = parser.parse_args(argv)
    write_qlora_recipe(args.output, base_model=args.base_model, dataset=args.dataset)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
