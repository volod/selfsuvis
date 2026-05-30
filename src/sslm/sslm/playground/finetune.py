from __future__ import annotations

import argparse
from pathlib import Path

from sslm.playground.constants import (
    DEFAULT_BASELINE_MODEL_ID,
    DEFAULT_FINETUNE_DATASET,
    DEFAULT_FINETUNE_OUTPUT,
    DEFAULT_QLORA_GRADIENT_ACCUMULATION_STEPS,
    DEFAULT_QLORA_LEARNING_RATE,
    DEFAULT_QLORA_LORA_ALPHA,
    DEFAULT_QLORA_LORA_DROPOUT,
    DEFAULT_QLORA_LORA_R,
    DEFAULT_QLORA_SEQUENCE_LENGTH,
    DEFAULT_QLORA_TRAIN_BATCH_SIZE,
    DEFAULT_QLORA_TRAIN_EPOCHS,
    DEFAULT_QLORA_WARMUP_RATIO,
)


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
lora_r: {DEFAULT_QLORA_LORA_R}
lora_alpha: {DEFAULT_QLORA_LORA_ALPHA}
lora_dropout: {DEFAULT_QLORA_LORA_DROPOUT}
target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj
sequence_length: {DEFAULT_QLORA_SEQUENCE_LENGTH}
learning_rate: {DEFAULT_QLORA_LEARNING_RATE}
per_device_train_batch_size: {DEFAULT_QLORA_TRAIN_BATCH_SIZE}
gradient_accumulation_steps: {DEFAULT_QLORA_GRADIENT_ACCUMULATION_STEPS}
num_train_epochs: {DEFAULT_QLORA_TRAIN_EPOCHS}
warmup_ratio: {DEFAULT_QLORA_WARMUP_RATIO}
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
    parser.add_argument("--base-model", default=DEFAULT_BASELINE_MODEL_ID)
    parser.add_argument("--dataset", default=DEFAULT_FINETUNE_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_FINETUNE_OUTPUT)
    args = parser.parse_args(argv)
    write_qlora_recipe(args.output, base_model=args.base_model, dataset=args.dataset)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
