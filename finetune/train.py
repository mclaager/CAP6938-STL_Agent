"""LoRA fine-tune of a small model on the collected conversations (#46). Runs in the `finetune` environment:

    pixi run -e finetune ft-train [--base unsloth/Qwen3.5-4B] [--epochs 2] [--gguf q4_k_m]

QLoRA (4-bit base, LoRA adapters) with Unsloth, so a 4B model trains on a 10 GB GPU. Loss is on the
assistant turns only: the student learns to write and repair scripts, not to repeat our prompts. Writes the
adapter to finetune/runs/<name>/lora and, with --gguf, a GGUF file plus an Ollama Modelfile:

    ollama create cad-student -f finetune/runs/<name>/gguf/Modelfile

The GPU is shared with Ollama: stop local models (`ollama stop <model>`) before training.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE / "data"


def read_chats(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="unsloth/Qwen3.5-4B", help="Hugging Face model id")
    ap.add_argument("--train", type=Path, default=DATA / "train.jsonl")
    ap.add_argument("--val", type=Path, default=DATA / "val.jsonl")
    ap.add_argument("--name", default=None, help="run name (default: base model + time)")
    ap.add_argument("--max-seq", type=int, default=8192, help="longer conversations are dropped")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=-1, help="stop after N steps (smoke tests)")
    ap.add_argument("--gguf", default=None, help="also export GGUF with this quantization, e.g. q4_k_m or q8_0")
    args = ap.parse_args(argv)

    from unsloth import FastLanguageModel  # first: it patches transformers/trl
    from unsloth.chat_templates import train_on_responses_only
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    name = args.name or f"{args.base.split('/')[-1]}-{time.strftime('%Y%m%d-%H%M')}"
    out = HERE / "runs" / name
    model, tok = FastLanguageModel.from_pretrained(args.base, max_seq_length=args.max_seq, load_in_4bit=True)
    tok = getattr(tok, "tokenizer", tok)  # Qwen3.5 is a vision model: use its text tokenizer, not the processor
    model = FastLanguageModel.get_peft_model(
        model, r=args.rank, lora_alpha=args.rank, lora_dropout=0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=0)

    def to_text(chats: list[dict]) -> Dataset:
        """Chat template applied with thinking off (the replies have no thinking text); too-long ones dropped."""
        texts, dropped = [], 0
        for c in chats:
            t = tok.apply_chat_template(c["messages"], tokenize=False, enable_thinking=False)
            if len(tok(t)["input_ids"]) > args.max_seq:
                dropped += 1
                continue
            texts.append(t)
        print(f"{len(texts)} conversations, {dropped} longer than {args.max_seq} tokens dropped")
        return Dataset.from_dict({"text": texts})

    train = to_text(read_chats(args.train))
    val = to_text(read_chats(args.val)) if args.val.exists() else None

    trainer = SFTTrainer(
        model=model, processing_class=tok, train_dataset=train, eval_dataset=val,
        args=SFTConfig(
            output_dir=str(out / "checkpoints"), dataset_text_field="text", max_length=args.max_seq,
            per_device_train_batch_size=args.batch, gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs, max_steps=args.max_steps, learning_rate=args.lr,
            lr_scheduler_type="cosine", warmup_ratio=0.05, weight_decay=0.01, optim="adamw_8bit",
            logging_steps=5, eval_strategy="epoch" if val else "no", save_strategy="epoch",
            bf16=True, seed=0, report_to="none"))
    # Qwen chat format: learn only what follows "<|im_start|>assistant\n".
    trainer = train_on_responses_only(trainer, instruction_part="<|im_start|>user\n",
                                      response_part="<|im_start|>assistant\n")
    t0 = time.time()
    stats = trainer.train()
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out / "lora"))
    tok.save_pretrained(str(out / "lora"))
    info = dict(base=args.base, args={k: str(v) for k, v in vars(args).items()}, train=len(train),
                val=len(val) if val else 0, minutes=round((time.time() - t0) / 60, 1),
                train_loss=stats.training_loss, log=trainer.state.log_history)
    (out / "train_info.json").write_text(json.dumps(info, indent=1))
    print(f"LoRA adapter in {out / 'lora'} ({info['minutes']} min, train loss {stats.training_loss:.3f})")

    if args.gguf:
        model.save_pretrained_gguf(str(out / "gguf"), tok, quantization_method=args.gguf)
        print(f"GGUF in {out / 'gguf'}; ollama create <name> -f {out / 'gguf' / 'Modelfile'}")


if __name__ == "__main__":
    main()
