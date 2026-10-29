"""Fine-tuning data scripts (#46) with scripted LLM replies (no API calls; runs the real sandbox)."""

import json

from finetune import collect, export, make_requests
from stl_agent.llm import ScriptedLLM
from stl_agent.retrieval import ExampleRetriever

GOOD = "PLAN:\n- block\nBBOX: 20, 10, 5\n```python\nimport cadquery as cq\nshow_object(cq.Workplane('XY').box(20, 10, 5))\n```"
BROKEN = GOOD.replace("box(20, 10, 5)", "boxx(20, 10, 5)")


def test_parse_batch_keeps_good_items_only():
    reply = json.dumps({"requests": [
        {"request": "A 20mm cube with a 5mm hole through it.", "bbox": [20, 20, 20]},
        {"request": "A tray for SD cards, about 5cm long.", "bbox": None},
        {"request": "short", "bbox": None},
        {"request": "A plate 40 x 30 x 2 mm.", "bbox": [40, "x", 2]},
        "not an object",
    ]})
    items = make_requests.parse_batch("Here you go:\n" + reply)
    assert [i["expected_bbox_mm"] for i in items] == [[20.0, 20.0, 20.0], None, None]
    assert make_requests.parse_batch('{"requests": [{"request": "x", bbox: 1}]}') == []  # bad JSON


def test_near_copy_filter_drops_benchmark_copies_and_repeats():
    held = make_requests.load_held_out()
    assert any("cylindrical spacer" in t for t in held)  # benchmark cases are held out
    filt = make_requests.NearCopyFilter(held, [])
    assert filt.reason(held[0])  # a benchmark case itself
    new = "A wall bracket for a 32mm broom handle with two countersunk 4mm screw holes, 60mm tall."
    assert filt.reason(new) is None
    assert filt.reason(new).startswith("own")  # now a repeat


def test_collect_keeps_repair_conversation(tmp_path):
    retriever = ExampleRetriever.from_file()
    req = {"id": "r1", "request": "a 20 x 10 x 5 mm block", "expected_bbox_mm": [20, 10, 5]}
    row, conv = collect.collect_one(req, ScriptedLLM([BROKEN, GOOD]), retriever, tmp_path, max_attempts=3)
    assert row["kept"] and row["attempts"] == 2 and row["requests"] == 2
    roles = [m["role"] for m in conv["messages"]]
    assert roles == ["user", "assistant", "user", "assistant"]
    assert "Here are solved examples" in conv["messages"][0]["content"]  # the prompt the app sends
    assert conv["messages"][1]["content"] == BROKEN and conv["messages"][3]["content"] == GOOD
    assert "Fix the script" in conv["messages"][2]["content"]
    assert conv["system"].startswith("You")


def test_collect_drops_wrong_size_and_own_library(tmp_path):
    retriever = ExampleRetriever.from_file()
    req = {"id": "r2", "request": "a block", "expected_bbox_mm": [30, 10, 5]}
    row, conv = collect.collect_one(req, ScriptedLLM([GOOD]), retriever, tmp_path, max_attempts=1)
    assert (row["kept"], row["reason"], conv) == (False, "wrong size", None)
    # The CadQuery sandbox can't import sdf, so such a run never builds; the filter is a second guard.
    built = collect_one_result(tmp_path, retriever)
    built.code = "import cadquery as cq\nfrom stl_agent.sdf import sphere\n"
    assert collect.keep_reason({"expected_bbox_mm": None}, built) == "uses our own library"


def collect_one_result(tmp_path, retriever):
    from stl_agent.generate import generate_cad

    return generate_cad("a block", ScriptedLLM([GOOD]), retriever, out_dir=tmp_path / "x", max_attempts=1)


def test_export_writes_chat_files_with_stable_split(tmp_path):
    src = tmp_path / "conv.jsonl"
    convs = [dict(id=f"r{i:05d}", system="sys", messages=[{"role": "user", "content": "u"},
                                                          {"role": "assistant", "content": "a"}]) for i in range(200)]
    src.write_text("".join(json.dumps(c) + "\n" for c in convs))
    train, val = export.export(src, tmp_path, val=0.1)
    assert train + val == 200 and 5 <= val <= 40
    first = json.loads((tmp_path / "train.jsonl").read_text().splitlines()[0])
    assert [m["role"] for m in first["messages"]] == ["system", "user", "assistant"]
    assert export.export(src, tmp_path, val=0.1) == (train, val)
