"""Write varied CadQuery part requests for fine-tuning data (#46).

    python -m finetune.make_requests --target 600 [--batch 20] [--set provider=anthropic]

A large model (the `provider` setting) writes requests in batches. Each batch asks for one area (kitchen,
electronics, ...) and one benchmark category (simple / medium / hard / casual), in the style of
eval/cases.json. A request is dropped when it is too close (TF-IDF cosine) to a benchmark case or a library
example, so the student never trains on the test set or on copies of the examples it will be shown,
or to a request already kept. Appends to finetune/data/requests.jsonl; run again to add more.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from stl_agent.config import EXAMPLES_PATH, add_set_argument, override_settings, settings
from stl_agent.replies import parse_json_object

DATA_DIR = Path(__file__).parent / "data"
REQUESTS_PATH = DATA_DIR / "requests.jsonl"
CASES_PATHS = [Path(__file__).parent.parent / "eval" / "cases.json",
               Path(__file__).parent.parent / "eval" / "organic_cases.json"]

# Cosine similarity (word 1-2 grams) above which a request counts as a copy.
MAX_SIM_HELD_OUT = 0.5   # benchmark cases and library examples
MAX_SIM_OWN = 0.7        # requests already kept

AREAS = [
    "kitchen and cooking", "desk and office organisation", "electronics enclosures and mounts",
    "3D printer upgrades and spare parts", "workshop jigs and fixtures", "garden and outdoor",
    "bathroom and cleaning", "cable management", "furniture hardware (feet, brackets, connectors)",
    "plumbing and hose adapters", "bicycle accessories", "camera and phone mounts", "board games and tabletop",
    "toys and puzzles", "model railway and RC hobby", "lab and classroom equipment",
    "mechanical parts (shafts, couplers, pulleys, bushings)", "fasteners and spacers",
    "storage boxes, trays and lids", "wall hooks and hangers", "lighting and lamp parts",
    "musical instrument accessories", "pet supplies", "car and garage", "sewing and crafts",
    "aquarium and plant pots", "sports and fitness", "tool holders and organisers", "drone and robotics parts",
    "household repairs (knobs, caps, clips)",
]

CATEGORIES = {
    "simple": "one basic solid with at most one or two features (a hole, a chamfer, a fillet). "
              "Every dimension is given in mm.",
    "medium": "a part with several features: hole patterns, pockets, slots, fillets, chamfers, bosses, "
              "shells, ribs. Every dimension needed to build it is given in mm.",
    "hard": "a part needing careful CadQuery: gears, threads-like ridges, helical or swept shapes, lofts, "
            "revolved profiles, many repeated features, angled or compound geometry. "
            "Every dimension needed is given in mm.",
    "casual": "everyday wording from someone who isn't an engineer, as they'd type it in a chat. The key sizes "
              "are given (sometimes loosely, e.g. 'about 5cm'), some minor details are left to the builder.",
}

PROMPT = """You write requests for a text-to-CAD benchmark. Each request describes ONE printable part that can be
built as a single solid with CadQuery (Python). Write {n} different requests.

Area: {area}
Category: {category} -- {category_desc}

Style examples (do NOT reuse these objects):
{examples}

Rules:
- One solid part per request, no assemblies, no moving parts, no text/logos, no organic or sculpted shapes.
- Vary the objects, sizes (from a few mm to about 250 mm), wording and sentence length.
- Sizes in mm (casual requests may use cm or "about").
- "bbox" is the overall size of the finished part in mm (three numbers, any order). Give it only when the
  request states every overall dimension outright (e.g. "120 x 80 x 4 mm", "20mm diameter, 40mm long"), so
  it needs no adding up of features; otherwise null. It must agree exactly with the request's numbers.

Reply with JSON only: {{"requests": [{{"request": "...", "bbox": [x, y, z] or null}}, ...]}}"""


def load_held_out() -> list[str]:
    texts = []
    for p in CASES_PATHS:
        if p.exists():
            texts += [c["request"] for c in json.loads(p.read_text())["cases"]]
    texts += [e["request"] for e in json.loads(Path(EXAMPLES_PATH).read_text())]
    return texts


def read_requests(path: Path = REQUESTS_PATH) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class NearCopyFilter:
    """Drops requests too similar to held-out texts (benchmark + library) or to ones already kept."""

    def __init__(self, held_out: list[str], kept: list[str]):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.held_out, self.kept = list(held_out), list(kept)
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(self.held_out + self.kept or ["x"])

    def reason(self, text: str) -> str | None:
        """Why `text` is a near copy, or None to keep it (and remember it)."""
        from sklearn.metrics.pairwise import cosine_similarity

        q = self.vec.transform([text])
        for pool, limit, name in ((self.held_out, MAX_SIM_HELD_OUT, "held-out"), (self.kept, MAX_SIM_OWN, "own")):
            if pool:
                sim = cosine_similarity(q, self.vec.transform(pool)).max()
                if sim > limit:
                    return f"{name} {sim:.2f}"
        self.kept.append(text)
        return None


def parse_batch(reply: str) -> list[dict]:
    """The {"request", "bbox"} items of a reply; malformed items are skipped."""
    try:
        data = parse_json_object(reply) or {}
    except ValueError:  # malformed JSON: skip the batch
        return []
    out = []
    for item in data.get("requests") or []:
        if not isinstance(item, dict) or not isinstance(item.get("request"), str):
            continue
        text = re.sub(r"\s+", " ", item["request"]).strip()
        bbox = item.get("bbox")
        ok = isinstance(bbox, list) and len(bbox) == 3 and all(isinstance(v, (int, float)) and v > 0 for v in bbox)
        if len(text) >= 15:
            out.append({"request": text, "expected_bbox_mm": [float(v) for v in bbox] if ok else None})
    return out


def style_examples(category: str, rng: random.Random) -> str:
    """Three benchmark requests of the category, as style samples (casual: casual ones)."""
    cases = json.loads(CASES_PATHS[0].read_text())["cases"]
    pool = [c["request"] for c in cases if c["category"] == category and c["expect"] == "build"] or \
           [c["request"] for c in cases if c["expect"] == "build"]
    return "\n".join(f"- {r}" for r in rng.sample(pool, min(3, len(pool))))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", type=int, default=600, help="stop once this many requests are kept in total")
    ap.add_argument("--batch", type=int, default=20, help="requests asked for per LLM request")
    ap.add_argument("--max-requests", type=int, default=60, help="LLM requests at most this run")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", type=Path, default=REQUESTS_PATH)
    add_set_argument(ap)
    args = ap.parse_args(argv)
    override_settings(args.set)

    from stl_agent.llm import make_llm, provider_config

    llm = make_llm()
    model = provider_config(settings().provider)["model"]
    if settings().provider == "anthropic":  # room for thinking; writing requests needs little of it
        llm.max_tokens, llm.effort = 16000, "low"
    existing = read_requests(args.out)
    filt = NearCopyFilter(load_held_out(), [r["request"] for r in existing])
    rng = random.Random(args.seed)
    weights = {"simple": 2, "medium": 4, "hard": 2, "casual": 2}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    kept_total, dropped, n = len(existing), 0, len(existing)

    for i in range(args.max_requests):
        if kept_total >= args.target:
            break
        area = rng.choice(AREAS)
        category = rng.choices(list(weights), weights=list(weights.values()))[0]
        prompt = PROMPT.format(n=args.batch, area=area, category=category, category_desc=CATEGORIES[category],
                               examples=style_examples(category, rng))
        reply = llm.complete("You write precise, varied engineering requests. Reply with JSON only.",
                             [{"role": "user", "content": prompt}])
        items = parse_batch(reply)
        with args.out.open("a") as f:
            for item in items:
                why = filt.reason(item["request"])
                if why:
                    dropped += 1
                    continue
                n += 1
                kept_total += 1
                f.write(json.dumps(dict(id=f"r{n:05d}", category=category, area=area, **item,
                                        source=model)) + "\n")
        print(f"[{i + 1}] {area} / {category}: {len(items)} written, total kept {kept_total}, dropped {dropped}", flush=True)
    print(f"{kept_total} requests in {args.out} ({dropped} near copies dropped this run); "
          f"tokens this run: {llm.input_tokens} in, {llm.output_tokens} out")


if __name__ == "__main__":
    main()
