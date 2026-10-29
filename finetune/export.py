"""Turn collected conversations into training files (#46).

    python -m finetune.export [--val 0.05]

Reads finetune/data/conversations.jsonl and writes finetune/data/train.jsonl and val.jsonl, one
{"messages": [{"role": "system"|"user"|"assistant", "content": ...}, ...]} per line: the format chat
fine-tuning tools (TRL, Unsloth) read. The split is by request id, so a request is never in both.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from finetune.collect import CONVERSATIONS_PATH
from finetune.make_requests import DATA_DIR


def to_chat(conv: dict) -> dict:
    return {"messages": [{"role": "system", "content": conv["system"]}] + conv["messages"]}


def in_val(request_id: str, frac: float) -> bool:
    """Stable split: the same id always lands on the same side."""
    return int(hashlib.sha1(request_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < frac


def export(src: Path = CONVERSATIONS_PATH, out_dir: Path = DATA_DIR, val: float = 0.05) -> tuple[int, int]:
    convs = [json.loads(line) for line in src.read_text().splitlines() if line.strip()]
    n = {True: 0, False: 0}
    with (out_dir / "train.jsonl").open("w") as tr, (out_dir / "val.jsonl").open("w") as va:
        for c in convs:
            side = in_val(c["id"], val)
            (va if side else tr).write(json.dumps(to_chat(c)) + "\n")
            n[side] += 1
    return n[False], n[True]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--val", type=float, default=0.05, help="fraction of requests held out for validation")
    args = ap.parse_args(argv)
    train, val = export(val=args.val)
    print(f"train.jsonl: {train} conversations, val.jsonl: {val}")


if __name__ == "__main__":
    main()
