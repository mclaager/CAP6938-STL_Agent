"""Fine-tuning data scripts (#46) with scripted LLM replies (no API calls; runs the real sandbox)."""

import json

import pytest

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
    train, val = export.export(src, tmp_path, val=0.1, sdf_dir=None)
    assert train + val == 200 and 5 <= val <= 40
    first = json.loads((tmp_path / "train.jsonl").read_text().splitlines()[0])
    assert [m["role"] for m in first["messages"]] == ["system", "user", "assistant"]
    assert export.export(src, tmp_path, val=0.1, sdf_dir=None) == (train, val)


# ---------------------------------------------------------------------------
# Round 2: organic requests and SDF collection
# ---------------------------------------------------------------------------
BALL = "PLAN:\n- a ball\nBBOX: 20, 20, 18.5\n```python\nfrom sdf import *\nshow_object(sphere(10))\n```"
PAIR = BALL.replace("BBOX: 20, 20, 18.5", "BBOX: 35, 20, 18.5").replace("show_object(sphere(10))", "s = sphere(10)\nshow_object(s | s.translate((15, 0, 0)).mirror(Y))")


class FakeJudge:
    def __init__(self, score):
        self.score, self.calls = score, 0

    def judge(self, request, image_png, facts):
        from stl_agent.judge.base import Judgement

        self.calls += 1
        assert image_png[:4] == b"\x89PNG" and facts["bbox_mm"]
        return Judgement(score=self.score, problems=[] if self.score >= 0.5 else ["looks like a blob"])


def test_organic_requests_skip_benchmark_subjects():
    subjects = make_requests.organic_subjects()
    assert "rubber duck" in subjects
    assert make_requests.names_subject("A grinning Rubber Duck, 50 mm tall", subjects) == "rubber duck"
    assert make_requests.names_subject("Two cats curled up together", subjects) == "cat"
    assert make_requests.names_subject("A coat hook with a handle, a catch and a category label", subjects) is None


def test_sdf_collection_keeps_judged_fogleman_scripts_only(tmp_path):
    from stl_agent.config import SDF_EXAMPLES_PATH

    retriever = ExampleRetriever.from_file(SDF_EXAMPLES_PATH)
    req = {"id": "o1", "request": "a ball", "expected_bbox_mm": None}
    judge = FakeJudge(0.9)
    row, conv = collect.collect_one(req, ScriptedLLM([BALL]), retriever, tmp_path, 1, "sdf", judge)
    assert row["kept"] and row["judge_score"] == 0.9 and judge.calls == 1
    assert "from sdf import *" in conv["messages"][-1]["content"]
    assert conv["system"].startswith("You are a 3D modeller")
    row, conv = collect.collect_one(req, ScriptedLLM([BALL]), retriever, tmp_path, 1, "sdf", FakeJudge(0.2))
    assert (row["kept"], row["reason"], conv) == (False, "judge 0.20", None)
    judge = FakeJudge(0.9)
    row, _ = collect.collect_one(req, ScriptedLLM([PAIR]), retriever, tmp_path, 1, "sdf", judge)
    assert row["reason"] == "not fogleman/sdf: mirror" and judge.calls == 0  # no judge request wasted


def test_second_teacher_only_gets_requests_nobody_kept(tmp_path, monkeypatch):
    data = tmp_path / "sdf"
    data.mkdir()
    cats = ["simple", "simple", "simple", "hard"]
    (data / "requests.jsonl").write_text("".join(json.dumps({"id": f"o{i}", "request": "a ball", "category": c}) + "\n"
                                                 for i, c in enumerate(cats)))
    (data / "runs.jsonl").write_text(
        json.dumps({"id": "o0", "kept": True, "teacher": "gemini-x"}) + "\n" +
        json.dumps({"id": "o1", "kept": False, "teacher": "gemini-x"}) + "\n")
    seen = []
    monkeypatch.setattr(collect, "collect_one", lambda req, *a, **k: seen.append(req["id"]) or (
        dict(id=req["id"], kept=False, reason="x", attempts=1, requests=1, input_tokens=0, output_tokens=0), None))
    monkeypatch.setattr("stl_agent.llm.make_llm", lambda *a, **k: ScriptedLLM([]))
    collect.main(["--kind", "sdf", "--data-dir", str(data), "--judge", "off", "--workers", "1", "--prefer-hard",
                  "--set", "provider=ollama", "--set", "providers.ollama.model=other-teacher"])
    # o0 was kept by the first teacher; o1 failed there, so it comes first; then unrun ones, hardest first
    assert seen == ["o1", "o3", "o2"]


def test_export_uses_only_fogleman_checked_sdf_conversations(tmp_path):
    sdf_dir = tmp_path / "sdf"
    sdf_dir.mkdir()

    def conv(i):
        return dict(id=i, system="s", messages=[{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}])

    (tmp_path / "cq.jsonl").write_text(json.dumps(conv("r1")) + "\n")
    (sdf_dir / "conversations.jsonl").write_text(json.dumps(conv("o1")) + "\n" + json.dumps(conv("o2")) + "\n")
    with pytest.raises(SystemExit, match="ft-fogleman"):
        export.export(tmp_path / "cq.jsonl", tmp_path, val=0.0, sdf_dir=sdf_dir)
    (sdf_dir / "fogleman.jsonl").write_text(json.dumps({"id": "o1", "ok": True}) + "\n" +
                                            json.dumps({"id": "o2", "ok": False, "reason": "mirror"}) + "\n")
    assert export.export(tmp_path / "cq.jsonl", tmp_path, val=0.0, sdf_dir=sdf_dir) == (2, 0)  # r1 and o1
