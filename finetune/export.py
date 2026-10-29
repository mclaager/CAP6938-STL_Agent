"""Turn collected conversations into training files (#46).

    python -m finetune.export [--val 0.05] [--no-sdf]

Reads finetune/data/conversations.jsonl (CadQuery) and finetune/data/sdf/conversations.jsonl (organic SDF scripts)
and writes finetune/data/train.jsonl and val.jsonl, one {"messages": [{"role": "system"|"user"|"assistant",
"content": ...}, ...]} per line: the format chat fine-tuning tools (TRL, Unsloth) read. The split is by request id,
so a request is never in both. An SDF conversation is used only when finetune/fogleman_check.py marked its script
ok (it runs unchanged on the public fogleman/sdf); run `pixi run -e finetune ft-fogleman` first.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from finetune.collect import CONVERSATIONS_PATH
from finetune.make_requests import DATA_DIR, SDF_DATA_DIR


def to_chat(conv: dict) -> dict:
    return {"messages": [{"role": "system", "content": conv["system"]}] + conv["messages"]}


def in_val(request_id: str, frac: float) -> bool:
    """Stable split: the same id always lands on the same side."""
    return int(hashlib.sha1(request_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < frac


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def sdf_conversations(sdf_dir: Path = SDF_DATA_DIR) -> tuple[list[dict], int]:
    """The SDF conversations whose script passed the fogleman/sdf check, and how many were left out."""
    convs = read_jsonl(sdf_dir / "conversations.jsonl")
    if convs and not (sdf_dir / "fogleman.jsonl").exists():
        raise SystemExit("No fogleman check yet: run `pixi run -e finetune ft-fogleman` first (or pass --no-sdf).")
    ok = {r["id"] for r in read_jsonl(sdf_dir / "fogleman.jsonl") if r["ok"]}
    keep = [c for c in convs if c["id"] in ok]
    return keep, len(convs) - len(keep)


def export(src: Path = CONVERSATIONS_PATH, out_dir: Path = DATA_DIR, val: float = 0.05,
           sdf_dir: Path | None = SDF_DATA_DIR) -> tuple[int, int]:
    convs = read_jsonl(src)
    if sdf_dir is not None:
        sdf, dropped = sdf_conversations(sdf_dir)
        convs += sdf
        if sdf or dropped:
            print(f"SDF conversations: {len(sdf)} used, {dropped} left out by the fogleman/sdf check")
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
    ap.add_argument("--no-sdf", action="store_true", help="CadQuery conversations only (round 1's data)")
    args = ap.parse_args(argv)
    train, val = export(val=args.val, sdf_dir=None if args.no_sdf else SDF_DATA_DIR)
    print(f"train.jsonl: {train} conversations, val.jsonl: {val}")


if __name__ == "__main__":
    main()
