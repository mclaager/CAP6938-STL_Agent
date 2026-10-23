"""Sandbox smoke tests: a valid part, a script with a bug, and one that never finishes."""

from stl_agent.sandbox import feedback_for_llm, run_cadquery, validate_mesh

GOOD = "import cadquery as cq\nresult = cq.Workplane('XY').box(60, 40, 5).faces('>Z').workplane().hole(4)\nshow_object(result)\n"
BAD = "import cadquery as cq\nresult = cq.Workplane('XY').box(10, 10, 10).fillet_all(1)\n"
HANG = "while True:\n    pass\n"


def test_good_part_is_valid():
    r = run_cadquery(GOOD, timeout=60)
    assert r.success, r.traceback
    report = validate_mesh(r.stl_path, expected_bbox=(60, 40, 5))
    assert report.valid, report.issues


def test_bad_script_fails_with_feedback():
    r = run_cadquery(BAD, timeout=60)
    assert not r.success
    assert "fillet_all" in feedback_for_llm(r)


def test_hanging_script_times_out():
    r = run_cadquery(HANG, timeout=5)
    assert not r.success
