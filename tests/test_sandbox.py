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


def test_math_available_without_import():
    code = "import cadquery as cq\nr = 10 * math.cos(0)\nshow_object(cq.Workplane('XY').cylinder(5, r))\n"
    r = run_cadquery(code, timeout=60)
    assert r.success, r.traceback


def test_sphere_and_cone_are_valid():
    # Zero-area triangles at a sphere's poles / a cone's tip must not count as holes or extra pieces.
    code = ("import cadquery as cq\n"
            "dome = cq.Workplane('XY').circle(10).extrude(30).union(cq.Workplane('XY').sphere(10).translate((0, 0, 30)))\n"
            "cone = cq.Workplane('XY').add(cq.Solid.makeCone(0, 10, 15)).translate((0, 0, -15))\n"
            "show_object(dome.union(cone))\n")
    r = run_cadquery(code, timeout=60)
    assert r.success, r.traceback
    report = validate_mesh(r.stl_path)
    assert report.valid, report.issues


def test_two_separate_pieces_still_invalid():
    code = ("import cadquery as cq\n"
            "show_object(cq.Workplane('XY').box(10, 10, 10).union(cq.Workplane('XY').box(5, 5, 5).translate((20, 0, 0))))\n")
    r = run_cadquery(code, timeout=60)
    assert r.success, r.traceback
    assert validate_mesh(r.stl_path).n_components == 2
