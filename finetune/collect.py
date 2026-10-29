"""Collect fine-tuning conversations from a teacher model (#46).

    python -m finetune.collect --budget 300 [--workers 4] [--limit N] [--set provider=anthropic]

Runs each request in finetune/data/requests.jsonl through `generate_cad`, the same generate -> execute ->
validate -> repair loop the app uses, with the teacher (the `provider` setting) and the retrieved examples.
A run is kept when it ends with a valid part (watertight, one body) and, where the request pins the size
down, at that size (eval.bench.bbox_matches). The kept conversation is exactly what the teacher saw and
said, failed attempts and repair messages included, so the student learns to repair as well as write.

Every run, kept or not, is a line in finetune/data/runs.jsonl (so a rerun skips it); kept conversations go
to finetune/data/conversations.jsonl. The run stops before it could go over `--budget` LLM requests.
Code that uses our own `sdf` module is never kept (only CadQuery and plain Python, see #46).
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from eval.bench import bbox_matches, library_id
from stl_agent.config import OUTPUT_DIR, add_set_argument, override_settings, settings
from stl_agent.generate import generate_cad

from finetune.make_requests import DATA_DIR, REQUESTS_PATH, read_requests

RUNS_PATH = DATA_DIR / "runs.jsonl"
CONVERSATIONS_PATH = DATA_DIR / "conversations.jsonl"

# Dollars per million tokens (input, output), for the running cost line and --max-usd. Thinking is billed as output.
PRICES = {"claude-sonnet-5-5": (2.0, 10.0), "claude-opus-5-5": (4.0, 20.0), "claude-haiku-5-5": (0.10, 0.50)}
RUN_RESERVE_USD = 0.15  # a run still in progress is assumed to cost up to this much

# Pinned so every collected conversation has the same repair messages, whatever the user's settings say.
GEN_KW = dict(use_rag=True, top_k=3, api_hints=False, size_feedback="measured")


class Recorder:
    """Wraps an LLM and keeps the last system prompt, the messages sent and the reply."""

    def __init__(self, llm):
        self.llm, self.requests = llm, 0
        self.system, self.messages = "", []

    def complete(self, system: str, messages: list[dict]) -> str:
        self.requests += 1
        reply = self.llm.complete(system, messages)
        self.system, self.messages = system, [dict(m) for m in messages] + [{"role": "assistant", "content": reply}]
        return reply

    def __getattr__(self, name):  # last_truncated, input_tokens, ...
        return getattr(self.llm, name)


def keep_reason(req: dict, res) -> str | None:
    """None when the run is kept, else why not."""
    if res.unsupported:
        return "declined"
    if not res.success:
        return "no valid part"
    if bbox_matches(res.validation.bbox_mm, req.get("expected_bbox_mm")) is False:
        return "wrong size"
    own = ("import sdf", "from sdf", "import stl_agent", "from stl_agent")
    if any(line.strip().startswith(own) for line in (res.code or "").splitlines()):
        return "uses our own library"
    return None


def collect_one(req: dict, llm, retriever, out_root: Path, max_attempts: int) -> tuple[dict, dict | None]:
    rec = Recorder(llm)
    before = (llm.input_tokens, llm.output_tokens)
    t0 = time.time()
    res = generate_cad(req["request"], rec, retriever, out_dir=out_root / req["id"], max_attempts=max_attempts,
                       **GEN_KW)
    why = keep_reason(req, res)
    row = dict(id=req["id"], kept=why is None, reason=why, attempts=res.attempts, requests=rec.requests,
               failures=[h.get("outcome") for h in res.history if h.get("outcome") not in ("ok", None)],
               bbox_mm=res.validation.bbox_mm if res.validation else None,
               expected_bbox_mm=req.get("expected_bbox_mm"),
               input_tokens=llm.input_tokens - before[0], output_tokens=llm.output_tokens - before[1],
               seconds=round(time.time() - t0, 1))
    conv = None
    if why is None:
        conv = dict(id=req["id"], category=req.get("category"), system=rec.system, messages=rec.messages,
                    attempts=res.attempts, retrieved=res.retrieved_ids)
    return row, conv


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--budget", type=int, default=100, help="max LLM requests this run")
    ap.add_argument("--workers", type=int, default=4, help="requests built at the same time")
    ap.add_argument("--limit", type=int, default=None, help="only the first N not-yet-run requests")
    ap.add_argument("--max-attempts", type=int, default=None, help="default: the max_attempts setting")
    ap.add_argument("--requests", type=Path, default=REQUESTS_PATH)
    ap.add_argument("--max-usd", type=float, default=None, help="stop before spending more than this (Claude only)")
    ap.add_argument("--effort", default=None, choices=["low", "medium", "high", "xhigh", "max"],
                    help="Claude's thinking effort (default: the model's own)")
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR, help="where runs.jsonl / conversations.jsonl go")
    add_set_argument(ap)
    args = ap.parse_args(argv)
    override_settings(args.set)

    from stl_agent.llm import make_llm, provider_config
    from stl_agent.retrieval import make_retriever

    provider = settings().provider
    teacher = provider_config(provider)["model"]
    max_attempts = args.max_attempts or settings().max_attempts
    runs_path, conv_path = args.data_dir / RUNS_PATH.name, args.data_dir / CONVERSATIONS_PATH.name
    done = {json.loads(line)["id"] for line in runs_path.read_text().splitlines() if line.strip()} \
        if runs_path.exists() else set()
    todo = [r for r in read_requests(args.requests) if r["id"] not in done][: args.limit]
    print(f"{len(todo)} requests to run ({len(done)} already done), teacher {provider}/{teacher}, "
          f"budget {args.budget} requests, {args.workers} workers")

    retriever = make_retriever("tfidf")
    library = library_id()
    out_root = OUTPUT_DIR / "finetune" / teacher.replace("/", "_")
    lock = threading.Lock()
    state = {"reserved": 0, "used": 0, "kept": 0, "ran": 0, "stop": "", "usd": 0.0, "running": 0}
    price = PRICES.get(teacher) if provider == "anthropic" else None
    if args.max_usd is not None and not price:
        raise SystemExit(f"--max-usd needs a known Claude price; {provider}/{teacher} is not in PRICES")
    local = threading.local()

    def worker(req: dict) -> None:
        with lock:  # reserve the worst case, so parallel runs can't go over the budget together
            if state["stop"] or state["used"] + state["reserved"] + max_attempts > args.budget:
                state["stop"] = state["stop"] or "budget"
                return
            if args.max_usd is not None and state["usd"] + (state["running"] + 1) * RUN_RESERVE_USD > args.max_usd:
                state["stop"] = state["stop"] or f"spending cap ${args.max_usd:.2f}"
                return
            state["reserved"] += max_attempts
            state["running"] += 1
        try:
            if not hasattr(local, "llm"):
                local.llm = make_llm()
                if provider == "anthropic":  # room for Claude's thinking (only the reply text is kept)
                    local.llm.max_tokens = max(local.llm.max_tokens, 16000)
                    local.llm.effort = args.effort
            row, conv = collect_one(req, local.llm, retriever, out_root, max_attempts)
        except Exception as e:  # quota, network: stop without recording the request, so a rerun retries it
            with lock:
                state["reserved"] -= max_attempts
                state["running"] -= 1
                state["stop"] = f"error: {type(e).__name__}: {e}"
            return
        row["usd"] = round((row["input_tokens"] * price[0] + row["output_tokens"] * price[1]) / 1e6, 4) if price else None
        row.update(teacher=teacher, provider=provider, library=library, settings=GEN_KW, effort=args.effort,
                   timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"))
        with lock:
            state["reserved"] -= max_attempts
            state["running"] -= 1
            state["usd"] += row["usd"] or 0
            state["used"] += row["requests"]
            state["ran"] += 1
            state["kept"] += row["kept"]
            runs_path.parent.mkdir(parents=True, exist_ok=True)
            with runs_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            if conv:
                conv.update(teacher=teacher, library=library)
                with conv_path.open("a") as f:
                    f.write(json.dumps(conv) + "\n")
            print(f"[{state['ran']}] {req['id']} {'kept' if row['kept'] else row['reason']}, "
                  f"{row['attempts']} attempt(s); kept {state['kept']}, requests {state['used']}/{args.budget}"
                  + (f", ${state['usd']:.2f}" if price else ""), flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(worker, todo))
    print(f"Ran {state['ran']}, kept {state['kept']}, used {state['used']} requests"
          + (f", ${state['usd']:.2f}" if price else "") + "."
          + (f" Stopped: {state['stop']}" if state["stop"] else ""))


if __name__ == "__main__":
    main()
