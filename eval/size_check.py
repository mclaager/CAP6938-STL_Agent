"""Size-check replay (issue #23): how should the repair loop word a failure that is only about size?

    python -m eval.size_check replay --model gemini-3.5-flash-lite [--variant current claim] [--budget 100]
    python -m eval.size_check report

generate_cad checks each part against the BBOX line the model writes, not against the request. When the only
problem is size, either the code or the model's BBOX claim can be wrong. This replays saved benchmark
conversations up to such a rejection, sends one repair message per variant, runs the reply's script and
records whether the part is then accepted, and whether it is the size the case asks for.

Attempts come from outputs/bench/ (local, gitignored): attempts on the current example library whose mesh was
one watertight body and failed only the size check, on a case with an expected size. Older records without the
reply text are rebuilt from the saved plan, BBOX and code. Runs with rag-hints (other repair messages),
rag-ext (removed pool) and rag-size (the new message) are left out. Results: eval/results/size-check/<model>.jsonl,
one row per (attempt, variant), so a run cut short by quota resumes.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from eval.bench import RESULTS_DIR, bbox_matches, library_id, load_cases
from stl_agent.config import OUTPUT_DIR, add_set_argument, override_settings
from stl_agent.generate import SYSTEM_PROMPT, build_first_user_message, build_repair_message, parse_response

OUT_DIR = RESULTS_DIR / "size-check"
SKIP_CONFIGS = ("rag-hints", "rag-ext", "rag-size")


def fmt(bbox) -> str:
    return " x ".join(f"{x:g}" for x in bbox)


# ---------------------------------------------------------------------------
# Repair messages for a size-only failure. `val` is the attempt's validation dict, `claimed` its BBOX line.
# Kept as they were tested; generate.build_size_message ships "blind" as "ask" and "claim" as "measured"
# (tests/test_size_check.py checks they still match).
# ---------------------------------------------------------------------------
def msg_current(val: dict, claimed) -> str:
    """What generate_cad sent before #23 (size_feedback "fix")."""
    return build_repair_message("The code ran, but the resulting mesh failed validation:\n- " + "\n- ".join(val["issues"]))


def msg_claim(val: dict, claimed) -> str:
    """Says either the script or the BBOX line may be wrong, with the measured size."""
    return (f"The script ran and built one watertight solid, but its size does not match your BBOX line: the part "
            f"measures {fmt(val['bbox_mm'])} mm, your BBOX says {fmt(claimed)} mm (axes in any order).\n\n"
            "Either the script or the BBOX line is wrong. Work out from the request what the finished part's overall "
            "size should be. If the script already builds what the request asks for, reply with the same script "
            "unchanged and a corrected BBOX. "
            "Otherwise fix the script. Reply with PLAN, BBOX and the complete script in the same format.")


def msg_blind(val: dict, claimed) -> str:
    """Like msg_claim, but without the measured size, so a corrected BBOX can't simply copy it."""
    return (f"The script ran and built one watertight solid, but its size does not match your BBOX line "
            f"({fmt(claimed)} mm).\n\n"
            "Either the script or the BBOX line is wrong. Work out from the request what the finished part's overall "
            "size should be. If the script already builds what the request asks for, reply with the same script "
            "unchanged and a corrected BBOX. "
            "Otherwise fix the script. Reply with PLAN, BBOX and the complete script in the same format.")


# A variant is the message sent after each size-only failure in turn: "blend" sends msg_blind first and, if the
# reply fails on size alone again, msg_claim (with the measured size) as a second request.
VARIANTS = {"current": [msg_current], "claim": [msg_claim], "blind": [msg_blind], "blend": [msg_blind, msg_claim]}


# ---------------------------------------------------------------------------
# Rebuilding the conversation
# ---------------------------------------------------------------------------
def rebuilt_reply(rec: dict) -> str:
    code = rec.get("code") or ""
    bbox = rec.get("bbox_claimed")
    return (f"PLAN:\n{rec.get('plan') or '-'}\n" + (f"BBOX: {', '.join(f'{x:g}' for x in bbox)}\n" if bbox else "")
            + f"```python\n{code.strip()}\n```")


def rebuilt_feedback(rec: dict) -> str:
    out = rec.get("outcome")
    if out in ("no_code_block", "truncated"):
        return "Your reply had no ```python code block. Reply in the required format."
    if out == "exec_error":
        return build_repair_message(f"The code failed to execute.\n{rec.get('error') or 'unknown error'}")
    return build_repair_message("The code ran, but the resulting mesh failed validation:\n- "
                                + "\n- ".join((rec.get("validation") or {}).get("issues") or []))


def size_only(rec: dict) -> bool:
    v = rec.get("validation")
    return bool(v and not v["valid"] and v.get("bbox_ok") is False and v["watertight"] and v["n_components"] == 1
                and v["volume_mm3"] > 0 and rec.get("bbox_claimed"))


def find_attempts(model_dir: Path, cases: dict, library: str) -> list[dict]:
    """Every size-only rejection on the current library, with the conversation that led to it."""
    by_id = {e["id"]: e for e in json.loads((Path(__file__).parents[1] / "stl_agent/data/examples.json").read_text())}
    found = []
    for report_path in sorted((model_dir / library).glob("*/*/rep*/report.json")):
        config, case_id = report_path.parts[-4], report_path.parts[-3]
        case = cases.get(case_id)
        if config in SKIP_CONFIGS or not case or not case.get("expected_bbox_mm"):
            continue
        report = json.loads(report_path.read_text())
        examples = [by_id[i] for i in report.get("retrieved_examples") or [] if i in by_id]
        messages = [{"role": "user", "content": build_first_user_message(case["request"], examples)}]
        for rec in report["attempt_log"]:
            messages.append({"role": "assistant", "content": rec.get("reply") or rebuilt_reply(rec)})
            if size_only(rec):
                val = rec["validation"]
                found.append(dict(key=f"{report_path.parent.relative_to(OUTPUT_DIR / 'bench')}#{rec['attempt']}",
                                  case=case_id, config=config, expected=case["expected_bbox_mm"],
                                  built=val["bbox_mm"], claimed=rec["bbox_claimed"],
                                  right_size=bool(bbox_matches(val["bbox_mm"], case["expected_bbox_mm"])),
                                  code=rec.get("code") or "", val=val, messages=list(messages)))
            messages.append({"role": "user", "content": rec.get("feedback") or rebuilt_feedback(rec)})
    return found


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------
def same_code(a: str, b: str) -> bool:
    norm = lambda s: "\n".join(line.rstrip() for line in s.strip().splitlines() if line.strip())  # noqa: E731
    return norm(a) == norm(b)


def judge_reply(reply: str, attempt: dict) -> dict:
    """Run the reply's script and check it the way generate_cad would (against the reply's own BBOX)."""
    from stl_agent.sandbox import run_cadquery, validate_mesh

    p = parse_response(reply)
    row = dict(new_claim=list(p.bbox) if p.bbox else None, code_unchanged=bool(p.code) and same_code(p.code, attempt["code"]))
    if not p.code:
        return {**row, "outcome": "no_code"}
    with tempfile.TemporaryDirectory(prefix="sizecheck_") as d:
        ex = run_cadquery(p.code, out_dir=d)
        if not ex.success:
            return {**row, "outcome": "exec_error"}
        val = validate_mesh(ex.stl_path, expected_bbox=p.bbox)
    right = bool(bbox_matches(val.bbox_mm, attempt["expected"]))
    outcome = ("accepted_right" if right else "accepted_wrong") if val.valid else "rejected"
    rec = {"validation": val.to_dict(), "bbox_claimed": row["new_claim"]}
    # The next step of a variant needs the code and the measured size (not written to the results file).
    return {**row, "outcome": outcome, "built": val.bbox_mm, "right_size": right,
            "_size_only": size_only(rec), "_val": rec["validation"], "_code": p.code}


def replay(model: str, variants: list[str], budget: int, log=print) -> None:
    from stl_agent.llm import make_llm

    cases = {c["id"]: c for c in load_cases()}
    attempts = []
    for model_dir in sorted((OUTPUT_DIR / "bench").iterdir()):
        attempts += [{**a, "source_model": model_dir.name} for a in find_attempts(model_dir, cases, library_id())]
    path = OUT_DIR / f"{model}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    done = {(r["key"], r["variant"]) for r in read(path)}
    log(f"{len(attempts)} size-only attempts ({sum(a['right_size'] for a in attempts)} the right size).")
    llm, used = make_llm(model=model), 0
    for variant in variants:
        for a in attempts:
            if (a["key"], variant) in done:
                continue
            if used >= budget:
                log(f"Budget of {budget} requests reached.")
                return
            messages, step_attempt, steps = list(a["messages"]), a, []
            try:
                for i, message in enumerate(VARIANTS[variant]):
                    if i and not steps[-1].get("_size_only"):
                        break
                    val, claimed = (a["val"], a["claimed"]) if not i else (steps[-1]["_val"], steps[-1]["new_claim"])
                    messages.append({"role": "user", "content": message(val, claimed)})
                    reply = llm.complete(SYSTEM_PROMPT, messages)
                    used += 1
                    messages.append({"role": "assistant", "content": reply})
                    steps.append({**judge_reply(reply, step_attempt), "reply": reply})
                    step_attempt = {**a, "code": steps[-1].get("_code") or a["code"]}
            except Exception as e:  # quota or network: stop, the run resumes later
                log(f"Stopped: {e}")
                return
            last = {k: v for k, v in steps[-1].items() if not k.startswith("_")}
            row = dict(key=a["key"], variant=variant, model=model, source_model=a["source_model"], case=a["case"],
                       config=a["config"], expected=a["expected"], built_before=a["built"], claimed_before=a["claimed"],
                       right_before=a["right_size"], **last, requests=len(steps),
                       code_unchanged_first=steps[0]["code_unchanged"], outcomes=[st["outcome"] for st in steps])
            with path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            log(f"{variant:8} {a['key']:60} {' -> '.join(row['outcomes'])}")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def report(log=print) -> None:
    for path in sorted(OUT_DIR.glob("*.jsonl")):
        rows = read(path)
        log(f"## {path.stem}\n")
        log("| variant | part was | n | accepted, right size | accepted, wrong size | still rejected / failed | same code | requests |")
        log("|---|---|---|---|---|---|---|---|")
        groups = defaultdict(list)
        for r in rows:
            groups[(r["variant"], r["right_before"])].append(r)
        for (variant, right), rs in sorted(groups.items(), key=lambda kv: (list(VARIANTS).index(kv[0][0]), not kv[0][1])):
            c = Counter(r["outcome"] for r in rs)
            n = len(rs)
            pct = lambda k: f"{c[k]} ({c[k] / n:.0%})"  # noqa: E731
            log(f"| {variant} | {'right size' if right else 'wrong size'} | {n} | {pct('accepted_right')} | "
                f"{pct('accepted_wrong')} | {n - c['accepted_right'] - c['accepted_wrong']} | "
                f"{sum(r.get('code_unchanged_first', r['code_unchanged']) for r in rs)} | "
                f"{sum(r.get('requests', 1) for r in rs) / n:.2f} |")
        log("")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("replay")
    r.add_argument("--model", required=True)
    r.add_argument("--variant", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    r.add_argument("--budget", type=int, default=100, help="requests at most")
    add_set_argument(r)
    sub.add_parser("report")
    args = ap.parse_args(argv)
    if args.cmd == "replay":
        override_settings(args.set or [])
        replay(args.model, args.variant, args.budget)
    else:
        report()


if __name__ == "__main__":
    main()
