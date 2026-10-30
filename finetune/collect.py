"""Collect fine-tuning conversations from a teacher model (#46).

    python -m finetune.collect --budget 300 [--workers 4] [--limit N] [--set provider=anthropic]
    python -m finetune.collect --kind sdf --budget 500 --set provider=gemini      # organic objects (round 2)
    python -m finetune.collect --kind sdf --limit 150 --max-usd 5 --prefer-hard --set provider=anthropic
        # Sonnet fallback: Gemini's failures first, then unrun requests (hardest first), stopping at $5

Runs each request in finetune/data/requests.jsonl through `generate_cad`, the same generate -> execute ->
validate -> repair loop the app uses, with the teacher (the `provider` setting) and the retrieved examples.
A run is kept when it ends with a valid part (watertight, one body) and, where the request pins the size
down, at that size (eval.bench.bbox_matches). The kept conversation is exactly what the teacher saw and
said, failed attempts and repair messages included, so the student learns to repair as well as write.

Every run, kept or not, is a line in runs.jsonl; kept conversations go to conversations.jsonl (finetune/data/,
or finetune/data/sdf/ for `--kind sdf`). A request is skipped once some teacher kept it, or once this teacher
ran it, so a second teacher (the Sonnet fallback) only gets the requests the first one failed. The run stops
before it could go over `--budget` LLM requests.

`--kind sdf` builds organic objects with SDF scripts. Organic requests have no size, so a valid part is kept only
when the visual judge passes it (`--judge-model`, Gemini by default: a free, separate quota). Scripts must stay
within the part of stl_agent/sdf.py that public fogleman/sdf also has (no .mirror(), .on_ground(), bounds(),
to_mesh()...); finetune/fogleman_check.py then runs each kept script on the real fogleman/sdf.
CadQuery scripts that import our own `sdf` are never kept.
"""

from __future__ import annotations

import argparse
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from eval.bench import bbox_matches, library_id, sdf_prompt_id
from stl_agent.config import OUTPUT_DIR, SDF_EXAMPLES_PATH, add_set_argument, override_settings, settings
from stl_agent.generate import generate_cad

from finetune.make_requests import DATA_DIR, REQUESTS_PATH, SDF_DATA_DIR, read_requests

RUNS_PATH = DATA_DIR / "runs.jsonl"
CONVERSATIONS_PATH = DATA_DIR / "conversations.jsonl"

# Dollars per million tokens (input, output), for the running cost line and --max-usd. Thinking is billed as output.
PRICES = {"claude-sonnet-5-5": (2.0, 10.0), "claude-opus-5-5": (4.0, 20.0), "claude-haiku-5-5": (0.10, 0.50),
          # Gemini paid tier (third-party price lists, Oct 2026; Google's page is the authority)
          "gemini-3.5-flash-lite": (0.30, 2.50), "gemini-3.1-flash-lite": (0.25, 1.50)}
# A run still in progress is assumed to cost up to max_attempts requests of 12k input and 3k output tokens.
RESERVE_TOKENS = (12_000, 3_000)

# Pinned so every collected conversation has the same repair messages, whatever the user's settings say.
GEN_KW = dict(use_rag=True, top_k=3, api_hints=False, size_feedback="measured")
GEN_KW_SDF = dict(language="sdf", use_rag=True, top_k=3, size_feedback="measured")

# Calls in our sdf.py that fogleman/sdf doesn't have: scripts using them would not transfer (#46).
NOT_FOGLEMAN = re.compile(r"\.(mirror|on_ground|bounds|grounded)\b|\b(to_mesh|save|lowest_z|stand)\s*\(")


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


def keep_reason(req: dict, res, kind: str = "cadquery") -> str | None:
    """None when the run is kept (before any judge), else why not."""
    if res.unsupported:
        return "declined"
    if not res.success:
        return "no valid part"
    if bbox_matches(res.validation.bbox_mm, req.get("expected_bbox_mm")) is False:
        return "wrong size"
    if kind == "sdf":
        m = NOT_FOGLEMAN.search(res.code or "")
        return f"not fogleman/sdf: {m.group(0).strip('.(').strip()}" if m else None
    own = ("import sdf", "from sdf", "import stl_agent", "from stl_agent")
    if any(line.strip().startswith(own) for line in (res.code or "").splitlines()):
        return "uses our own library"
    return None


def judge_part(judge, request: str, res) -> tuple[float | None, list[str]]:
    """The visual judge's score and problems for a built part (the same render and prompt as eval/judge.py)."""
    from stl_agent.render import png_bytes, render_grid

    j = judge.judge(request, png_bytes(render_grid(res.stl_path, 256)), {"bbox_mm": res.validation.bbox_mm})
    return j.score, j.problems


def collect_one(req: dict, llm, retriever, out_root: Path, max_attempts: int, kind: str = "cadquery",
                judge=None, threshold: float = 0.5) -> tuple[dict, dict | None]:
    rec = Recorder(llm)
    before = (llm.input_tokens, llm.output_tokens)
    t0 = time.time()
    res = generate_cad(req["request"], rec, retriever, out_dir=out_root / req["id"], max_attempts=max_attempts,
                       **(GEN_KW_SDF if kind == "sdf" else GEN_KW))
    why = keep_reason(req, res, kind)
    score = problems = None
    if why is None and judge is not None:
        score, problems = judge_part(judge, req["request"], res)
        if score is None:
            why = "judge gave no score"
        elif score < threshold:
            why = f"judge {score:.2f}"
    row = dict(id=req["id"], kept=why is None, reason=why, attempts=res.attempts, requests=rec.requests,
               judge_score=score, judge_problems=problems,
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
    ap.add_argument("--limit", type=int, default=None,
                    help="run at most N requests: other teachers' failures first, then requests nobody has run")
    ap.add_argument("--max-total", type=int, default=None,
                    help="stop once this teacher has run N requests in all (counting earlier days)")
    ap.add_argument("--style", default=None,
                    help="only requests written with this style (e.g. library: round 2 CadQuery requests)")
    ap.add_argument("--prefer-hard", action="store_true",
                    help="take not-yet-run requests hardest first (for the Sonnet fallback)")
    ap.add_argument("--max-attempts", type=int, default=None, help="default: the max_attempts setting")
    ap.add_argument("--kind", choices=["cadquery", "sdf"], default="cadquery")
    ap.add_argument("--judge", choices=["on", "off"], default=None,
                    help="judge each valid part, keep only passes (default: on for sdf, off for cadquery)")
    ap.add_argument("--judge-model", default="gemini-3.1-flash-lite")
    ap.add_argument("--judge-provider", default="gemini")
    ap.add_argument("--requests", type=Path, default=None, help="default: requests.jsonl in the data folder")
    ap.add_argument("--max-usd", type=float, default=None,
                    help="stop before spending more than this on builds (models in PRICES; judging not counted)")
    ap.add_argument("--effort", default=None, choices=["low", "medium", "high", "xhigh", "max"],
                    help="Claude's thinking effort (default: the model's own)")
    ap.add_argument("--data-dir", type=Path, default=None,
                    help="where runs.jsonl / conversations.jsonl go (default: data/, or data/sdf/ for sdf)")
    add_set_argument(ap)
    args = ap.parse_args(argv)
    override_settings(args.set)

    from stl_agent.llm import make_llm, provider_config
    from stl_agent.retrieval import make_retriever

    provider = settings().provider
    teacher = provider_config(provider)["model"]
    max_attempts = args.max_attempts or settings().max_attempts
    sdf = args.kind == "sdf"
    args.data_dir = args.data_dir or (SDF_DATA_DIR if sdf else DATA_DIR)
    args.requests = args.requests or args.data_dir / "requests.jsonl"
    runs_path, conv_path = args.data_dir / RUNS_PATH.name, args.data_dir / CONVERSATIONS_PATH.name
    old = [json.loads(line) for line in runs_path.read_text().splitlines() if line.strip()] if runs_path.exists() else []
    done = {r["id"] for r in old if r["kept"] or r.get("teacher") == teacher}  # kept by anyone, or tried by this one
    failed_elsewhere = {r["id"] for r in old if not r["kept"]} - done
    left = [r for r in read_requests(args.requests) if r["id"] not in done
            and (args.style is None or r.get("style") == args.style)]
    # A fallback teacher takes the other teachers' failures first, then requests nobody has run yet (hardest first
    # with --prefer-hard), until --limit or --max-usd: a cap that also works as a spending floor.
    retry = [r for r in left if r["id"] in failed_elsewhere]
    fresh = [r for r in left if r["id"] not in failed_elsewhere]
    if args.prefer_hard:
        rank = {"hard": 0, "medium": 1, "casual": 2, "simple": 3}
        fresh.sort(key=lambda r: rank.get(r.get("category"), 4))
    limit = args.limit
    if args.max_total is not None:  # this teacher's runs over all days, so the same command can be rerun daily
        ran_before = sum(1 for r in old if r.get("teacher") == teacher)
        limit = min(limit if limit is not None else 10**9, max(0, args.max_total - ran_before))
    todo = (retry + fresh)[:limit]
    use_judge = (args.judge or ("on" if sdf else "off")) == "on"
    threshold = settings().judge_threshold
    print(f"{len(todo)} requests to run ({len(done)} already done), teacher {provider}/{teacher}, "
          f"budget {args.budget} requests, {args.workers} workers")

    retriever = make_retriever(path=SDF_EXAMPLES_PATH) if sdf else make_retriever("tfidf")
    library = f"{library_id(SDF_EXAMPLES_PATH)}+{sdf_prompt_id()}" if sdf else library_id()
    gen_kw = GEN_KW_SDF if sdf else GEN_KW
    out_root = OUTPUT_DIR / "finetune" / args.kind / teacher.replace("/", "_")
    lock = threading.Lock()
    state = {"reserved": 0, "used": 0, "kept": 0, "ran": 0, "stop": "", "usd": 0.0, "running": 0}
    price = PRICES.get(teacher)
    reserve = max_attempts * (RESERVE_TOKENS[0] * price[0] + RESERVE_TOKENS[1] * price[1]) / 1e6 if price else 0.0
    if args.max_usd is not None and not price:
        raise SystemExit(f"--max-usd needs a known price; {provider}/{teacher} is not in PRICES")
    local = threading.local()

    def worker(req: dict) -> None:
        with lock:  # reserve the worst case, so parallel runs can't go over the budget together
            if state["stop"] or state["used"] + state["reserved"] + max_attempts > args.budget:
                state["stop"] = state["stop"] or "budget"
                return
            if args.max_usd is not None and state["usd"] + (state["running"] + 1) * reserve > args.max_usd:
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
                local.judge = None
                if use_judge:
                    from stl_agent.judge.llm_judge import LLMJudge

                    local.judge = LLMJudge(make_llm(provider=args.judge_provider, model=args.judge_model),
                                           prompt="v2", model_name=args.judge_model)
            row, conv = collect_one(req, local.llm, retriever, out_root, max_attempts, args.kind, local.judge,
                                    threshold)
        except Exception as e:  # quota, network: stop without recording the request, so a rerun retries it
            with lock:
                state["reserved"] -= max_attempts
                state["running"] -= 1
                state["stop"] = f"error: {type(e).__name__}: {e}"
            return
        row["usd"] = round((row["input_tokens"] * price[0] + row["output_tokens"] * price[1]) / 1e6, 4) if price else None
        row.update(teacher=teacher, provider=provider, library=library, settings=gen_kw, effort=args.effort,
                   judge=args.judge_model if use_judge else None,
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
