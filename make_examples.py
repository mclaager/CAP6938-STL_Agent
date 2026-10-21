"""Builds examples.json and verifies every example runs and matches its bbox."""
import json
from sandbox import run_cadquery, validate_mesh

EX = [
 dict(id="prim_001", cls="primitives",
  request="Make a 30mm x 20mm x 10mm rectangular block.",
  bbox=[30,20,10],
  code='''import cadquery as cq

length, width, height = 30.0, 20.0, 10.0
result = cq.Workplane("XY").box(length, width, height)
show_object(result)
'''),
 dict(id="holes_003", cls="holes_fasteners",
  request="Create a 60mm x 40mm rectangular plate, 5mm thick, with four 4mm diameter mounting holes placed 5mm in from each corner.",
  bbox=[60,40,5],
  code='''import cadquery as cq

plate_length = 60.0
plate_width = 40.0
plate_thickness = 5.0
hole_diameter = 4.0
hole_inset = 5.0  # distance of hole centre from each edge

dx = plate_length / 2 - hole_inset
dy = plate_width / 2 - hole_inset

result = (
    cq.Workplane("XY")
    .box(plate_length, plate_width, plate_thickness)
    .faces(">Z").workplane()
    .pushPoints([(dx, dy), (-dx, dy), (dx, -dy), (-dx, -dy)])
    .hole(hole_diameter)
)
show_object(result)
'''),
 dict(id="pattern_001", cls="patterns",
  request="Make a round flange 80mm in diameter and 6mm thick with six 5mm holes evenly spaced on a 60mm bolt circle.",
  bbox=[80,80,6],
  code='''import cadquery as cq

flange_diameter = 80.0
thickness = 6.0
bolt_circle_diameter = 60.0
hole_diameter = 5.0
num_holes = 6

result = (
    cq.Workplane("XY")
    .circle(flange_diameter / 2)
    .extrude(thickness)
    .faces(">Z").workplane()
    .polarArray(bolt_circle_diameter / 2, 0, 360, num_holes)
    .hole(hole_diameter)
)
show_object(result)
'''),
 dict(id="pattern_002", cls="patterns",
  request="A 100mm x 60mm x 3mm perforated panel with a 4 by 3 grid of 6mm holes spaced 20mm apart.",
  bbox=[100,60,3],
  code='''import cadquery as cq

panel_length, panel_width, panel_thickness = 100.0, 60.0, 3.0
hole_diameter = 6.0
spacing_x = spacing_y = 20.0
cols, rows = 4, 3

result = (
    cq.Workplane("XY")
    .box(panel_length, panel_width, panel_thickness)
    .faces(">Z").workplane()
    .rarray(spacing_x, spacing_y, cols, rows)
    .hole(hole_diameter)
)
show_object(result)
'''),
 dict(id="shell_001", cls="parametric",
  request="Make an open-top box 50mm long, 30mm wide and 20mm tall with 2mm thick walls.",
  bbox=[50,30,20],
  code='''import cadquery as cq

length, width, height = 50.0, 30.0, 20.0
wall = 2.0

result = (
    cq.Workplane("XY")
    .box(length, width, height)
    .faces(">Z")
    .shell(-wall)   # negative = wall grows inward; the selected face is removed
)
show_object(result)
'''),
 dict(id="fillet_001", cls="edge_features",
  request="A 40mm x 40mm x 20mm block with its four vertical edges rounded with a 5mm radius.",
  bbox=[40,40,20],
  code='''import cadquery as cq

size_x, size_y, size_z = 40.0, 40.0, 20.0
corner_radius = 5.0

result = (
    cq.Workplane("XY")
    .box(size_x, size_y, size_z)
    .edges("|Z")
    .fillet(corner_radius)
)
show_object(result)
'''),
 dict(id="holes_005", cls="holes_fasteners",
  request="50mm square plate, 10mm thick, with a centred counterbored hole for an M4 socket head screw: 4.5mm through hole, 8mm counterbore diameter, 4mm deep.",
  bbox=[50,50,10],
  code='''import cadquery as cq

plate_size, plate_thickness = 50.0, 10.0
through_dia = 4.5
cbore_dia = 8.0
cbore_depth = 4.0

result = (
    cq.Workplane("XY")
    .box(plate_size, plate_size, plate_thickness)
    .faces(">Z").workplane()
    .cboreHole(through_dia, cbore_dia, cbore_depth)
)
show_object(result)
'''),
 dict(id="parametric_001", cls="parametric",
  request="A flat washer with 20mm outer diameter, 8mm inner diameter and 2mm thickness.",
  bbox=[20,20,2],
  code='''import cadquery as cq

outer_dia, inner_dia, thickness = 20.0, 8.0, 2.0

result = (
    cq.Workplane("XY")
    .circle(outer_dia / 2)
    .circle(inner_dia / 2)   # second circle inside the first becomes a hole
    .extrude(thickness)
)
show_object(result)
'''),
 dict(id="named_001", cls="named_parts",
  request="An L-bracket with two 40mm x 30mm legs at 90 degrees, 4mm thick, 30mm wide, with one 5mm hole centred in each leg.",
  bbox=[40,30,40],
  code='''import cadquery as cq

leg_length = 40.0   # length of each leg, measured from the outer corner
width = 30.0
thickness = 4.0
hole_dia = 5.0

base = cq.Workplane("XY").box(leg_length, width, thickness, centered=(False, True, False))
wall = cq.Workplane("XY").box(thickness, width, leg_length, centered=(False, True, False))
bracket = base.union(wall)

# holes are centred in the free part of each leg (past the other leg's thickness)
base_hole = (
    cq.Workplane("XY")
    .workplane(offset=0)
    .center((leg_length + thickness) / 2, 0)
    .circle(hole_dia / 2)
    .extrude(thickness)
)
wall_hole = (
    cq.Workplane("YZ")
    .center(0, (leg_length + thickness) / 2)
    .circle(hole_dia / 2)
    .extrude(thickness)
)
result = bracket.cut(base_hole).cut(wall_hole)
show_object(result)
'''),
]

out = []
ok = True
for e in EX:
    r = run_cadquery(e["code"], timeout=60)
    if not r.success:
        print("FAIL", e["id"], r.traceback); ok = False; continue
    v = validate_mesh(r.stl_path, tuple(e["bbox"]))
    print(e["id"], "valid" if v.valid else "INVALID", v.bbox_mm, v.issues)
    ok &= v.valid
    out.append(dict(id=e["id"], cls=e["cls"], request=e["request"], code=e["code"], expected_bbox_mm=e["bbox"], volume_mm3=round(v.volume_mm3,2)))
json.dump(out, open("examples.json","w"), indent=1)
print("ALL OK" if ok else "SOME FAILED")
