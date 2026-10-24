"""Reference CadQuery solutions for benchmark cases (eval/cases.json).

Each function builds one case's part by hand. Running them checks that the case can be built and
that its expected_bbox_mm is right (a reference caught an arch that split its own block in two).
They are never shown to the model.

    python -m eval.references     # build every reference and compare with cases.json
"""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import cadquery as cq

REFS = {}


def ref(cid):
    def deco(f):
        REFS[cid] = f
        return f
    return deco

def ref(cid):
    def deco(f):
        REFS[cid] = f
        return f
    return deco


@ref("m09")
def _():
    return (cq.Workplane("XY").box(50, 50, 6).faces(">Z").workplane()
            .rect(30, 30).cutBlind(-3).edges("|Z").edges(cq.selectors.BoxSelector((-16, -16, 0), (16, 16, 4))).fillet(4))


@ref("m10")
def _():
    pin = cq.Workplane("XY").circle(10).extrude(40).faces(">Z or <Z").edges().chamfer(2)
    groove = cq.Workplane("XY").workplane(offset=20 - 1.5).circle(10).circle(8.5).extrude(3)
    return pin.cut(groove)


@ref("m11")
def _():
    plate = cq.Workplane("XY").box(25, 40, 3)
    cy = -20 + 12
    hole = cq.Workplane("XY").center(0, cy).circle(4).extrude(5, both=True)
    slot = cq.Workplane("XY").center(0, cy + 5).rect(4, 10).extrude(5, both=True)
    return plate.cut(hole).cut(slot)


@ref("m12")
def _():
    c = cq.Workplane("XY").circle(12.5).circle(5).extrude(10)
    radial = cq.Workplane("XZ").center(0, 5).circle(1.65).extrude(-20)
    return c.cut(radial)


@ref("m13")
def _():
    return (cq.Workplane("XY").circle(5).extrude(20)
            .faces(">Z").workplane().circle(8).extrude(15)
            .faces(">Z").workplane().circle(4).extrude(25))


@ref("m14")
def _():
    box = cq.Workplane("XY").box(60, 40, 25, centered=(True, True, False)).faces(">Z").shell(-2)
    flange = (cq.Workplane("XY").workplane(offset=22).rect(70, 50).rect(60, 40).extrude(3))
    return box.union(flange)


@ref("m15")
def _():
    head = cq.Workplane("XY").polygon(6, 13 / math.cos(math.radians(30))).extrude(5)
    return head.faces(">Z").workplane().circle(4).extrude(30)


@ref("m16")
def _():
    pts = []
    for i in range(10):
        r = 30 if i % 2 == 0 else 12
        a = math.radians(90 + 36 * i)
        pts.append((r * math.cos(a), r * math.sin(a)))
    return cq.Workplane("XY").polyline(pts).close().extrude(6)


@ref("m17")
def _():
    block = cq.Workplane("XY").box(100, 20, 40, centered=(True, True, False))
    arch = cq.Workplane("XZ").circle(30).extrude(20, both=True)
    return block.cut(arch)


@ref("h04")
def _():
    # profile in XZ (x = radius), revolved about Z
    pts = [(4, 0), (20, 0), (20, 3), (16, 6), (20, 9), (20, 12), (4, 12)]
    return cq.Workplane("XZ").polyline(pts).close().revolve(360, (0, 0, 0), (0, 1, 0))


@ref("h05")
def _():
    r, pitch, turns, wire = 9, 6, 5, 1
    helix = cq.Wire.makeHelix(pitch, pitch * turns, r)
    prof = cq.Workplane("XZ").center(r, 0).circle(wire)
    return prof.sweep(cq.Workplane().add(helix), isFrenet=True)


@ref("h06")
def _():
    flange = cq.Workplane("XY").circle(30).circle(23).extrude(3)
    tube = cq.Workplane("XY").circle(25).circle(23).extrude(-20)
    g = flange.union(tube)
    slot = cq.Workplane("XY").center(15, 0).rect(30, 15).extrude(30, both=True)
    return g.cut(slot)


@ref("h07")
def _():
    path = cq.Workplane("XY").moveTo(30, 0).radiusArc((0, 30), -30)
    return cq.Workplane("XZ").center(30, 0).circle(10).circle(8).sweep(path)


@ref("h08")
def _():
    plate = cq.Workplane("XY").box(80, 60, 5)
    af, wall = 8, 2
    pitch = af + wall
    dx, dy = pitch * math.cos(math.radians(30)), pitch
    pts = []
    for i in range(-10, 11):
        for j in range(-10, 11):
            x, y = i * dx, j * dy + (pitch / 2 if i % 2 else 0)
            # hex across corners = af / cos30; keep 3mm border
            rc = af / math.cos(math.radians(30)) / 2
            if abs(x) + rc <= 40 - 3 and abs(y) + af / 2 <= 30 - 3:
                pts.append((x, y))
    holes = cq.Workplane("XY").pushPoints(pts).polygon(6, af / math.cos(math.radians(30))).extrude(5, both=True)
    return plate.cut(holes)


@ref("h09")
def _():
    body = cq.Workplane("XY").circle(12.5).extrude(77.5)
    dome = cq.Workplane("XY").sphere(12.5).translate((0, 0, 77.5))
    h = body.union(dome)
    flutes = (cq.Workplane("XY").polarArray(12.5, 0, 360, 6).circle(2).extrude(77.5))
    return h.cut(flutes)


@ref("h10")
def _():
    base = cq.Workplane("XY").box(60, 50, 5, centered=(False, True, False))
    wall = cq.Workplane("XY").box(5, 50, 40, centered=(False, True, False))
    g = cq.Workplane("XZ").polyline([(5, 5), (35, 5), (5, 35)]).close().extrude(2.5, both=True)
    br = base.union(wall).union(g.translate((0, 22.5, 0))).union(g.translate((0, -22.5, 0)))
    br = br.cut(cq.Workplane("XY").pushPoints([(45, 12), (45, -12)]).circle(2.5).extrude(10))
    br = br.cut(cq.Workplane("YZ").pushPoints([(12, 25), (-12, 25)]).circle(2.5).extrude(10))
    return br


@ref("h11")
def _():
    outer = cq.Workplane("XY").rect(60, 60).workplane(offset=70).rect(80, 80).loft()
    inner = cq.Workplane("XY").workplane(offset=3).rect(54 + 20 * 3 / 70, 54 + 20 * 3 / 70).workplane(offset=67).rect(74, 74).loft()
    p = outer.cut(inner)
    return p.cut(cq.Workplane("XY").circle(5).extrude(5))


@ref("h12")
def _():
    w = cq.Workplane("XY").circle(20).extrude(10)
    notches = cq.Workplane("XY").polarArray(20, 0, 360, 30).circle(1).extrude(10)
    hexs = cq.Workplane("XY").polygon(6, 5 / math.cos(math.radians(30))).extrude(10)
    return w.cut(notches).cut(hexs)


@ref("h13")
def _():
    box = cq.Workplane("XY").box(60, 40, 30, centered=(True, True, False)).faces(">Z").shell(-2)
    z = 30 - 3 - 0.75  # groove centre, so its top edge is 3mm below the rim
    # 1mm deep groove in each inner long wall (inner face at y = +-18)
    g = (cq.Workplane("XY").box(56, 1, 1.5).translate((0, 18.5, z))
         .union(cq.Workplane("XY").box(56, 1, 1.5).translate((0, -18.5, z))))
    return box.cut(g)


@ref("p06")
def _():
    prof = [(0, 0), (12.5, 0), (12.5, 4), (9, 7), (8, 30), (10, 32), (10, 40), (0, 40)]
    rook = cq.Workplane("XZ").polyline(prof).close().revolve(360, (0, 0, 0), (0, 1, 0))
    rook = rook.cut(cq.Workplane("XY").workplane(offset=35).circle(7).extrude(10))
    for a in (0, 90):
        rook = rook.cut(cq.Workplane("XY").box(30, 4, 5).rotate((0, 0, 0), (0, 0, 1), a).translate((0, 0, 38)))
    return rook


@ref("c01")
def _():
    b = cq.Workplane("XY").box(80, 40, 25)
    return b.faces(">Z").workplane().rarray(18, 18, 4, 2).hole(15, 20)


@ref("c02")
def _():
    return cq.Workplane("XY").box(50, 30, 20).faces(">Z").shell(-1.5)


def check(cid: str, cases: dict) -> tuple[bool, str]:
    """Build one reference, export it like the sandbox does, and validate it against its case."""
    from stl_agent.sandbox import validate_mesh

    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / f"{cid}.stl")
        cq.exporters.export(REFS[cid](), path)
        rep = validate_mesh(path, expected_bbox=cases[cid]["expected_bbox_mm"])
    return rep.valid, f"bbox {[round(x, 2) for x in rep.bbox_mm]} vs {cases[cid]['expected_bbox_mm']} {'; '.join(rep.issues)}"


def main() -> int:
    from eval.bench import load_cases

    cases = {c["id"]: c for c in load_cases()}
    failed = 0
    for cid in REFS:
        ok, msg = check(cid, cases)
        failed += not ok
        print(f"{cid}: {'ok' if ok else 'FAIL'}  {msg}")
    print(f"\n{len(REFS) - failed}/{len(REFS)} references valid")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
