"""Does a fine-tuned model still do the app's other jobs? (#46)

    python -m finetune.general_check --model cad-qwen3.5:4b-v1 [--reps 3] [--set provider=ollama]

Gate: each conversation in GATE_CASES goes through the intake gate (`LLMGate`, one request). Scored: the reply
was valid JSON (the gate didn't fall back) and the scope is one of the expected ones.
General: a few non-CAD prompts with a plain system prompt. Two ask for a Python function, which is run against
tests; the others check an expected word. Every non-CAD reply is also checked for CadQuery leaking in (`cadquery`,
`BBOX:`, `show_object`), the failure a narrow fine-tune shows. Rows go to eval/results/general-check.jsonl.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from stl_agent.config import add_set_argument, override_settings

RESULTS = Path(__file__).parent.parent / "eval" / "results" / "general-check.jsonl"


def turn(role: str, content: str) -> dict:
    return {"role": role, "content": content}


# (id, conversation, acceptable scopes)
GATE_CASES = [
    ("cube-hole", [turn("user", "A 50mm cube with a 10mm hole through the middle.")], {"in_scope"}),
    ("gear", [turn("user", "A spur gear with 24 teeth, module 1.5, 6mm thick, with an 8mm bore.")], {"in_scope"}),
    ("box-2turn", [turn("user", "I need a box."), turn("assistant", "What size should it be, and open or with a lid?"),
                   turn("user", "100 x 60 x 40 mm outside, 2mm walls, open top.")], {"in_scope"}),
    ("bracket-vague", [turn("user", "Make me a bracket.")], {"in_scope"}),
    ("coat-hook", [turn("user", "A wall hook for a coat, about 8cm tall, with two screw holes.")], {"in_scope"}),
    ("low-poly-vase", [turn("user", "A low-poly vase about 15cm tall.")], {"simplified", "in_scope"}),
    ("cat-statue", [turn("user", "A realistic statue of my cat sleeping curled up.")], {"out_of_scope"}),
    ("eiffel-picture", [turn("user", "Print me a picture of the Eiffel tower.")], {"out_of_scope", "simplified"}),
    ("horse-plinth", [turn("user", "A horse statue standing on a rectangular plinth with a name plate on the front.")],
     {"composite"}),
    ("phone-stand", [turn("user", "something to prop my phone up on my desk")], {"in_scope"}),
]

# (id, prompt, check) -- check gets the reply text and returns True when it passes
def _runs(func: str, tests: list[tuple]) -> callable:
    def check(reply: str) -> bool:
        blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", reply, re.S) or [reply]
        ns: dict = {}
        try:
            exec(blocks[0], ns)  # model-written code; small pure functions only
            return all(ns[func](*args) == want for args, want in tests)
        except Exception:
            return False
    return check


GENERAL_CASES = [
    ("palindrome", "Write a Python function is_palindrome(s) that returns True when s reads the same backwards, "
                   "ignoring case and spaces. Reply with just the code.",
     _runs("is_palindrome", [(("Race car",), True), (("hello",), False), (("A man a plan a canal Panama",), True)])),
    ("fibonacci", "Write a Python function fib(n) that returns the nth Fibonacci number, with fib(0) = 0 and "
                  "fib(1) = 1. Reply with just the code.",
     _runs("fib", [((0,), 0), ((1,), 1), ((10,), 55), ((20,), 6765)])),
    ("capital", "What is the capital of Australia? Answer in one word.", lambda r: "canberra" in r.lower()),
    ("fillet-explain", "In two sentences, what is a fillet in CAD?", lambda r: "round" in r.lower() or "radius" in r.lower()),
    ("haiku", "Write a haiku about autumn leaves.", lambda r: "```" not in r and len(r.strip().splitlines()) <= 6),
]




def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--reps", type=int, default=3)
    add_set_argument(ap)
    args = ap.parse_args(argv)
    override_settings(["provider=ollama", f"providers.ollama.model={args.model}"] + (args.set or []))

    from stl_agent.gate.llm_gate import LLMGate
    from stl_agent.llm import make_llm

    llm = make_llm()
    gate = LLMGate(llm)
    rows = []
    for rep in range(args.reps):
        for cid, conv, ok_scopes in GATE_CASES:
            t0 = time.time()
            d = gate.decide(conv)
            json_ok = "not valid JSON" not in (d.note or "")
            rows.append(dict(kind="gate", case=cid, rep=rep, json_ok=json_ok, scope=d.scope, action=d.action,
                             passed=json_ok and d.scope in ok_scopes, seconds=round(time.time() - t0, 1)))
            print(f"gate {cid} rep{rep}: json_ok={json_ok} scope={d.scope} action={d.action}", flush=True)
        for cid, prompt, check in GENERAL_CASES:
            t0 = time.time()
            reply = llm.complete("You are a helpful assistant.", [turn("user", prompt)])
            # Naming CadQuery is fine when explaining CAD; our reply format or CadQuery code is not.
            leak = bool(re.search(r"BBOX\s*:|show_object|cq\.Workplane", reply)) or (
                cid != "fillet-explain" and "cadquery" in reply.lower())
            rows.append(dict(kind="general", case=cid, rep=rep, passed=check(reply), leak=leak,
                             seconds=round(time.time() - t0, 1), reply=reply[:2000]))
            print(f"general {cid} rep{rep}: passed={rows[-1]['passed']} leak={leak}", flush=True)
    with RESULTS.open("a") as f:
        for r in rows:
            f.write(json.dumps(dict(model=args.model, timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"), **r)) + "\n")
    gates = [r for r in rows if r["kind"] == "gate"]
    gens = [r for r in rows if r["kind"] == "general"]
    print(f"{args.model}: gate JSON ok {sum(r['json_ok'] for r in gates)}/{len(gates)}, "
          f"scope right {sum(r['passed'] for r in gates)}/{len(gates)}; general passed "
          f"{sum(r['passed'] for r in gens)}/{len(gens)}, CadQuery leaks {sum(r['leak'] for r in gens)}/{len(gens)}")


if __name__ == "__main__":
    main()
