"""API hints for the repair loop (api_check.py): wrong CadQuery calls found by reading the script, and the
extra feedback after a failed run. The cases are mistakes a local model made in real runs."""

import inspect
import json
import textwrap

from eval import references
from stl_agent.api_check import check_code, runtime_hint
from stl_agent.config import EXAMPLES_PATH
from stl_agent.sandbox import feedback_for_llm, run_cadquery


def one(code: str) -> str:
    problems = check_code(textwrap.dedent(code))
    assert len(problems) == 1, problems
    return problems[0]


def test_no_false_alarms_on_working_code():
    for e in json.loads(EXAMPLES_PATH.read_text()):
        assert check_code(e["code"]) == [], e["id"]
    for case, fn in references.REFS.items():
        assert check_code(textwrap.dedent(inspect.getsource(fn))) == [], case


def test_wrong_arguments():
    msg = one("import cadquery as cq\nr = cq.Workplane('XY').cylinder(diameter=5, height=3)\n")
    assert "Workplane.cylinder(height, radius" in msg
    assert "line 2" in one("import cadquery as cq\nr = cq.Workplane().lineTo((1, 2))\n")


def test_tuple_where_a_number_goes():
    problems = check_code("import cadquery as cq\nr = cq.Workplane().rect((0, 0), (10, 4))\n")
    assert len(problems) == 2 and "`xLen` must be a number, got a tuple" in problems[0]


def test_missing_methods_through_variables_and_chains():
    code = """
    import cadquery as cq
    base = cq.Workplane("XY").box(10, 10, 2)
    top = base.faces(">Z").workplane()
    r = top.edges("|Z").radius(1)
    leg = cq.Solid.cylinder(2, 10)
    """
    problems = check_code(textwrap.dedent(code))
    assert any("a Workplane has no `radius`" in p and ".fillet(r)" in p for p in problems)
    assert any("cq.Solid has no `cylinder`" in p and "makeCylinder" in p and "cq.Workplane" in p for p in problems)


def test_unknown_objects_are_not_checked():
    code = """
    import cadquery as cq
    def part(w):
        return w.anything(1, 2, 3)
    shapes = [cq.Workplane().box(1, 1, 1)]
    x = shapes[0].nonsense()
    """
    assert check_code(textwrap.dedent(code)) == []


def test_runtime_hint_names_the_failing_method():
    hint = runtime_hint("TypeError", "unsupported operand type(s) for /: 'tuple' and 'float'", "Workplane.rect")
    assert "Workplane.rect(xLen, yLen" in hint
    assert "`edges`" in runtime_hint("AttributeError", "'Workplane' object has no attribute 'edge'", None)


def test_failed_run_feedback_includes_hints():
    code = "import cadquery as cq\nresult = cq.Workplane().rect((0, 0), (10, 4)).extrude(2)\nshow_object(result)\n"
    res = run_cadquery(code)
    assert not res.success and res.lib_call == "Workplane.rect"
    plain, hinted = feedback_for_llm(res), feedback_for_llm(res, api_hints=True)
    assert "Signature" not in plain
    assert "inside CadQuery's Workplane.rect" in hinted and "`xLen` must be a number" in hinted


def test_attempt_log_keeps_reply_reasoning_and_feedback(tmp_path):
    from stl_agent.llm import ScriptedLLM
    from stl_agent.generate import generate_cad
    from stl_agent.retrieval import make_retriever

    reply = "PLAN:\n- cube\nBBOX: 10, 10, 10\n```python\nimport cadquery as cq\nresult = cq.Workplane().{}\nshow_object(result)\n```"
    llm = ScriptedLLM([reply.format("rect((0, 0), (1, 1)).extrude(1)"), reply.format("box(10, 10, 10)")])
    llm.last_reasoning = "thinking"
    res = generate_cad("a 10mm cube", llm, make_retriever(), out_dir=tmp_path, api_hints=True)
    first, second = json.loads((tmp_path / "report.json").read_text())["attempt_log"]
    assert res.success and first["reply"].startswith("PLAN") and first["reasoning"] == "thinking"
    assert "Workplane.rect(xLen, yLen" in first["feedback"] and "feedback" not in second
