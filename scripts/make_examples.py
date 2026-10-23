"""Builds stl_agent/data/examples.json and verifies every example runs and matches its bbox.

Run from the repo root:  pixi run -e cap6938 examples   (or: python -m scripts.make_examples)
"""
import json

from stl_agent.config import EXAMPLES_PATH
from stl_agent.sandbox import run_cadquery, validate_mesh

EX = [
 dict(id="prim_001", cls="primitives",
  keywords="box block cube cuboid rectangular solid",
  request="Make a 30mm x 20mm x 10mm rectangular block.",
  bbox=[30,20,10],
  code='''import cadquery as cq

length, width, height = 30.0, 20.0, 10.0
result = cq.Workplane("XY").box(length, width, height)
show_object(result)
'''),
 dict(id="holes_003", cls="holes_fasteners",
  keywords="plate mounting holes corner holes drilled through hole",
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
  keywords="flange disc round bolt circle polar array holes evenly spaced",
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
  keywords="panel grid array holes perforated rectangular pattern",
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
  keywords="box container tray hollow open top walls shell",
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
  keywords="rounded edges fillet radius block",
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
  keywords="counterbore counterbored hole screw socket head bolt",
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
  keywords="washer ring spacer tube cylinder hollow inner outer diameter annulus",
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
  keywords="bracket l-bracket angle leg perpendicular mount",
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
 dict(id="revolve_001", cls="revolve",
  keywords="funnel cone conical taper tapered revolve round hollow",
  request="A funnel 60mm wide at the top narrowing to 10mm at the bottom over a height of 40mm, with a 1.5mm wall, open at both ends.",
  bbox=[60,60,40],
  code='''import cadquery as cq

top_d, bottom_d, height, wall = 60.0, 10.0, 40.0, 1.5

# Half cross-section drawn on the XZ plane, then revolved around the vertical axis.
# On an "XZ" workplane, local x is global X and local y is global Z, so the revolve axis
# from (0, 0) to (0, 1) in local coordinates is the global Z axis.
profile = [
    (bottom_d / 2, 0),
    (top_d / 2, height),
    (top_d / 2 - wall, height),
    (bottom_d / 2 - wall, 0),
]
result = cq.Workplane("XZ").polyline(profile).close().revolve(360, (0, 0, 0), (0, 1, 0))
show_object(result)
'''),
 dict(id="revolve_002", cls="revolve",
  keywords="stopper plug cone conical taper tapered revolve cap knob round",
  request="A bottle stopper: a tapered plug 20mm long going from 18mm diameter at the top to 14mm at the bottom, with a 25mm diameter, 5mm thick cap on top.",
  bbox=[25,25,25],
  code='''import cadquery as cq

plug_len, plug_top_d, plug_bottom_d = 20.0, 18.0, 14.0
cap_d, cap_t = 25.0, 5.0

# Outline of one half (x = radius, y = height), revolved 360 degrees around the Z axis.
profile = [
    (0, 0),
    (plug_bottom_d / 2, 0),
    (plug_top_d / 2, plug_len),
    (cap_d / 2, plug_len),
    (cap_d / 2, plug_len + cap_t),
    (0, plug_len + cap_t),
]
result = cq.Workplane("XZ").polyline(profile).close().revolve(360, (0, 0, 0), (0, 1, 0))
show_object(result)
'''),
 dict(id="loft_001", cls="loft",
  keywords="loft transition adapter square round blend taper",
  request="A solid square-to-round adapter: a 40mm square at the bottom blending into a 30mm diameter circle at the top over a height of 50mm.",
  bbox=[40,40,50],
  code='''import cadquery as cq

square, top_d, height = 40.0, 30.0, 50.0

# loft() joins the pending sketches: the square on the base, the circle on a workplane above it.
result = (
    cq.Workplane("XY")
    .rect(square, square)
    .workplane(offset=height)
    .circle(top_d / 2)
    .loft(combine=True)
)
show_object(result)
'''),
 dict(id="polygon_001", cls="polygon",
  keywords="hexagon hexagonal hex nut prism polygon across flats",
  request="A hex nut blank 13mm across the flats and 5mm thick, with a 6.5mm hole through the centre.",
  bbox=[15.01,13,5],
  code='''import cadquery as cq
import math

across_flats, thickness, hole_d = 13.0, 5.0, 6.5

# polygon(nSides, diameter) takes the circumscribed diameter (across the corners),
# so convert from across-flats: corners = flats / cos(30 deg) for a hexagon.
across_corners = across_flats / math.cos(math.radians(30))

result = (
    cq.Workplane("XY")
    .polygon(6, across_corners)
    .extrude(thickness)
    .faces(">Z").workplane()
    .hole(hole_d)
)
show_object(result)
'''),
 dict(id="polygon_002", cls="polygon",
  keywords="triangle triangular prism polygon equilateral",
  request="A triangular prism: an equilateral triangle with 30mm sides, extruded 10mm.",
  bbox=[25.98,30,10],
  code='''import cadquery as cq
import math

side, thickness = 30.0, 10.0

# For an equilateral triangle the circumscribed diameter is 2 * side / sqrt(3).
diameter = 2 * side / math.sqrt(3)
result = cq.Workplane("XY").polygon(3, diameter).extrude(thickness)
show_object(result)
'''),
 dict(id="slot_001", cls="slots",
  keywords="slot slotted elongated hole stadium oblong cut through",
  request="A 50mm x 30mm x 5mm plate with two 20mm long, 5mm wide slots running along its length, centred 8mm either side of the middle.",
  bbox=[50,30,5],
  code='''import cadquery as cq

length, width, thickness = 50.0, 30.0, 5.0
slot_len, slot_w, offset = 20.0, 5.0, 8.0

# slot2D(length, diameter) is a stadium shape; length is the overall length including the round ends.
result = (
    cq.Workplane("XY")
    .box(length, width, thickness)
    .faces(">Z").workplane()
    .pushPoints([(0, offset), (0, -offset)])
    .slot2D(slot_len, slot_w)
    .cutThruAll()
)
show_object(result)
'''),
 dict(id="text_001", cls="text",
  keywords="text label letters name tag emboss embossed raised",
  request="A 60mm x 20mm x 3mm name tag with the word BOB raised 1mm on top.",
  bbox=[60,20,4],
  code='''import cadquery as cq

length, width, thickness = 60.0, 20.0, 3.0
label, font_size, raise_by = "BOB", 10.0, 1.0

# text() defaults to combine="cut" (engraving). For raised (embossed) letters use a positive
# distance AND combine=True, otherwise the letters are "cut" into empty air and vanish.
result = (
    cq.Workplane("XY")
    .box(length, width, thickness)
    .faces(">Z").workplane()
    .text(label, font_size, raise_by, combine=True)
)
show_object(result)
'''),
 dict(id="text_002", cls="text",
  keywords="text engrave engraved letters number name inscription",
  request="A round coaster 40mm in diameter and 4mm thick with the number 42 engraved 1mm deep in the centre of the top face.",
  bbox=[40,40,4],
  code='''import cadquery as cq

diameter, thickness = 40.0, 4.0
label, font_size, depth = "42", 14.0, 1.0

# Engraving: a negative distance (into the part) with combine="cut", which is text()'s default.
result = (
    cq.Workplane("XY")
    .circle(diameter / 2)
    .extrude(thickness)
    .faces(">Z").workplane()
    .text(label, font_size, -depth, combine="cut")
)
show_object(result)
'''),
 dict(id="gear_001", cls="gears",
  keywords="gear spur teeth cog sprocket bore",
  request="A simple spur gear with 12 trapezoid-shaped teeth, 30mm outside diameter, 24mm root diameter, 6mm thick, with an 8mm bore.",
  bbox=[29.93,29.93,6],
  code='''import cadquery as cq
import math

teeth, tip_d, root_d, thickness, bore_d = 12, 30.0, 24.0, 6.0, 8.0
tip_half_angle, root_half_angle = 4.0, 7.0  # degrees either side of each tooth's centre line

# Build the outline point by point: for each tooth, root -> tip -> tip -> root.
pts = []
for i in range(teeth):
    centre = 360.0 * i / teeth
    for angle, r in [(-root_half_angle, root_d / 2), (-tip_half_angle, tip_d / 2),
                     (tip_half_angle, tip_d / 2), (root_half_angle, root_d / 2)]:
        a = math.radians(centre + angle)
        pts.append((r * math.cos(a), r * math.sin(a)))

result = (
    cq.Workplane("XY")
    .polyline(pts).close()
    .extrude(thickness)
    .faces(">Z").workplane()
    .hole(bore_d)
)
show_object(result)
'''),
 dict(id="sweep_001", cls="sweep",
  keywords="handle sweep bent rod bar pipe arc curved path u-shape hook",
  request="A U-shaped handle made from 10mm diameter round bar: two 20mm straight legs 80mm apart (centre to centre) joined by a half-circle arc at the top.",
  bbox=[90,10,65],
  code='''import cadquery as cq

bar_d, spacing, leg = 10.0, 80.0, 20.0
r = spacing / 2

# Path in the XZ plane (local y = global Z): up one leg, arc over the top, down the other leg.
path = (
    cq.Workplane("XZ")
    .moveTo(-r, 0)
    .lineTo(-r, leg)
    .threePointArc((0, leg + r), (r, leg))
    .lineTo(r, 0)
)
# The profile circle sits at the start of the path, perpendicular to it.
result = cq.Workplane("XY").center(-r, 0).circle(bar_d / 2).sweep(path, transition="round")
show_object(result)
'''),
 dict(id="shell_002", cls="shell",
  keywords="cup holder container cylinder hollow open top walls shell pot",
  request="A cylindrical pen holder 70mm in diameter and 100mm tall with 3mm walls and base, open at the top.",
  bbox=[70,70,100],
  code='''import cadquery as cq

diameter, height, wall = 70.0, 100.0, 3.0

# shell() with a negative thickness hollows inward; the selected top face is removed (left open).
result = (
    cq.Workplane("XY")
    .circle(diameter / 2)
    .extrude(height)
    .faces(">Z")
    .shell(-wall)
)
show_object(result)
'''),
 dict(id="compartment_001", cls="compartments",
  keywords="box tray organizer compartments dividers divider walls",
  request="An open-top storage box 80mm x 80mm x 30mm with 2mm walls and floor, split into four equal compartments by two 2mm walls that cross in the middle.",
  bbox=[80,80,30],
  code='''import cadquery as cq

size, height, wall = 80.0, 30.0, 2.0

box = (
    cq.Workplane("XY")
    .box(size, size, height, centered=(True, True, False))
    .faces(">Z")
    .shell(-wall)
)
# Divider walls run the full inner length and stand on the floor.
inner = size - 2 * wall
divider_x = cq.Workplane("XY").box(inner, wall, height, centered=(True, True, False))
divider_y = cq.Workplane("XY").box(wall, inner, height, centered=(True, True, False))
result = box.union(divider_x).union(divider_y)
show_object(result)
'''),
 dict(id="holes_006", cls="holes_fasteners",
  keywords="countersunk countersink screw holes mounting strip",
  request="An 80mm x 20mm x 5mm mounting strip with three countersunk holes for M4 screws (4.5mm hole, 9mm countersink, 90 degrees), 30mm apart along the centre line.",
  bbox=[80,20,5],
  code='''import cadquery as cq

length, width, thickness = 80.0, 20.0, 5.0
hole_d, csk_d, csk_angle, pitch = 4.5, 9.0, 90.0, 30.0

result = (
    cq.Workplane("XY")
    .box(length, width, thickness)
    .faces(">Z").workplane()
    .pushPoints([(-pitch, 0), (0, 0), (pitch, 0)])
    .cskHole(hole_d, csk_d, csk_angle)
)
show_object(result)
'''),
 dict(id="profile_001", cls="profiles",
  keywords="wedge doorstop ramp slope sloped tapered profile extrude",
  request="A wedge doorstop 120mm long and 35mm wide that rises from 5mm tall at the thin end to 30mm tall at the thick end.",
  bbox=[120,35,30],
  code='''import cadquery as cq

length, width, thin, thick = 120.0, 35.0, 5.0, 30.0

# Draw the side profile on the XZ plane and extrude it across the width.
# extrude(d, both=True) extrudes d in each direction, so half the width each way gives the full
# width centred on Y = 0.
profile = [(0, 0), (length, 0), (length, thick), (0, thin)]
result = cq.Workplane("XZ").polyline(profile).close().extrude(width / 2, both=True)
show_object(result)
'''),
 dict(id="angled_001", cls="angled",
  keywords="stand angled tilted leaning plate support base back rotate",
  request="A plate stand: a 60mm x 40mm x 4mm base with a 60mm x 50mm x 4mm back plate standing on its rear edge and leaning back 20 degrees from vertical.",
  bbox=[60,56.86,48.35],
  code='''import cadquery as cq

base_w, base_d, t = 60.0, 40.0, 4.0
back_h, lean_deg = 50.0, 20.0

base = cq.Workplane("XY").box(base_w, base_d, t, centered=(True, True, False))

# Build the back plate upright with its bottom-front edge on the X axis, rotate it about
# that edge (rotating about +X by a negative angle tips the top towards +Y, i.e. backwards),
# then move the edge to the rear of the base.
back = (
    cq.Workplane("XY")
    .box(base_w, t, back_h, centered=(True, False, False))
    .rotate((0, 0, 0), (1, 0, 0), -lean_deg)
    .translate((0, base_d / 2 - t, 0))
)
result = base.union(back)
show_object(result)
'''),
 dict(id="fillet_002", cls="fillet_chamfer",
  keywords="chamfer chamfered bevel edges cube",
  request="A 50mm cube with every edge chamfered by 3mm.",
  bbox=[50,50,50],
  code='''import cadquery as cq

size, chamfer = 50.0, 3.0

result = cq.Workplane("XY").box(size, size, size).edges().chamfer(chamfer)
show_object(result)
'''),
 dict(id="pattern_003", cls="patterns",
  keywords="vent radial slots polar array cover disc",
  request="A round vent cover 70mm in diameter and 5mm thick with six 15mm x 4mm slots arranged radially, their centres 20mm from the middle.",
  bbox=[70,70,5],
  code='''import cadquery as cq

diameter, thickness = 70.0, 5.0
slot_len, slot_w, slot_radius, count = 15.0, 4.0, 20.0, 6

# polarArray rotates each copy's local axes by default, so a slot drawn along local x
# points outward from the centre at every position.
result = (
    cq.Workplane("XY")
    .circle(diameter / 2)
    .extrude(thickness)
    .faces(">Z").workplane()
    .polarArray(slot_radius, 0, 360, count)
    .slot2D(slot_len, slot_w)
    .cutThruAll()
)
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
    out.append(dict(id=e["id"], cls=e["cls"], keywords=e.get("keywords", ""), request=e["request"], code=e["code"], expected_bbox_mm=e["bbox"], volume_mm3=round(v.volume_mm3,2)))
json.dump(out, open(EXAMPLES_PATH, "w"), indent=1)
print("ALL OK" if ok else "SOME FAILED")
