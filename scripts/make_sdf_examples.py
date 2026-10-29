"""Builds stl_agent/data/sdf_examples.json: the few-shot library for SDF scripts (organic shapes, #42), and checks
every example builds one watertight solid standing on z = 0.

Run from the repo root:  pixi run -e cap6938 sdf-examples   (or: python -m scripts.make_sdf_examples)

None of these objects is an organic benchmark case (eval/organic_cases.json), so the benchmark measures whether
the model can build what it hasn't been shown.
"""
import json
import sys

from stl_agent.config import SDF_EXAMPLES_PATH
from stl_agent.sandbox import run_cadquery, validate_mesh

EX = [
 dict(id="sdf_snowman", cls="characters",
  keywords="snowman stacked balls figure character hat carrot nose",
  request="A snowman about 60 mm tall: three stacked balls, a top hat and a carrot nose.",
  code='''from sdf import *

base_r, middle_r, head_r = 14, 10, 7
overlap = 3  # each ball sinks into the one below, so they join

z_base = base_r - 2  # sits 2 mm into the ground plane: a flat footprint after the cut below
z_middle = z_base + base_r + middle_r - overlap
z_head = z_middle + middle_r + head_r - overlap

body = union(
    sphere(base_r).translate((0, 0, z_base)),
    sphere(middle_r).translate((0, 0, z_middle)).k(3),
    sphere(head_r).translate((0, 0, z_head)).k(2),
)
hat = (capped_cylinder((0, 0, 0), (0, 0, 1.5), 7.5)                  # brim
       | capped_cylinder((0, 0, 1.5), (0, 0, 8), 5)).translate((0, 0, z_head + head_r - 2))
nose = capped_cone((0, 0, 0), (7, 0, 0), 1.5, 0.3).translate((head_r - 1, 0, z_head))
snowman = body | hat | nose.k(0.5)

ground = box(200, center=(0, 0, -100))  # everything below z = 0
result = snowman - ground
show_object(result)
'''),
 dict(id="sdf_mushroom", cls="plants",
  keywords="mushroom toadstool cap stem fungus dome",
  request="A mushroom 40 mm tall with a domed cap 40 mm across on a thick stem.",
  code='''from sdf import *

cap_r, cap_h = 20, 12          # cap radius and height of the dome
stem_r_bottom, stem_r_top = 7, 5
stem_h = 30

stem = capped_cone((0, 0, 0), (0, 0, stem_h), stem_r_bottom, stem_r_top)
# Dome: a flattened ellipsoid with its lower half cut off, slightly hollowed underneath.
dome = ellipsoid((cap_r, cap_r, cap_h)) - box((3 * cap_r, 3 * cap_r, cap_h), center=(0, 0, -cap_h / 2))
gills = ellipsoid((cap_r - 3, cap_r - 3, 3))
cap = (dome - gills.k(2)).translate((0, 0, stem_h - 2))

result = stem | cap.k(2)
show_object(result)
'''),
 dict(id="sdf_pine_tree", cls="plants",
  keywords="pine tree fir christmas tree cones layers trunk",
  request="A stylised pine tree 70 mm tall: three stacked cone layers on a short trunk.",
  code='''from sdf import *

trunk_r, trunk_h = 4, 12
layers = [  # (bottom z, height, bottom radius) from the bottom up
    (10, 26, 22),
    (26, 24, 17),
    (41, 29, 12),
]

trunk = capped_cylinder((0, 0, 0), (0, 0, trunk_h), trunk_r)
tree = trunk
for z, h, r in layers:
    tree = tree | capped_cone((0, 0, z), (0, 0, z + h), r, 0.5).k(1.5)

result = tree
show_object(result)
'''),
 dict(id="sdf_fish", cls="animals",
  keywords="fish tail fin body swimming sea animal flat base",
  request="A fish about 60 mm long with a forked tail and a dorsal fin, on a flat base so it stands up.",
  code='''from sdf import *

body_len, body_h, body_w = 22, 12, 7    # ellipsoid radii (half sizes): x along the fish, z up
z = body_h - 1                            # body centre height; the belly sinks 1 mm below z = 0

body = ellipsoid((body_len, body_w, body_h)).translate((0, 0, z))
# Tail: two thin lobes angled up and down behind the body.
lobe = ellipsoid((8, 1.5, 4))
tail = (lobe.rotate(0.6, Y).translate((-body_len - 4, 0, z + 4))
        | lobe.rotate(-0.6, Y).translate((-body_len - 4, 0, z - 4)))
dorsal = ellipsoid((9, 1.2, 5)).rotate(0.3, Y).translate((-2, 0, z + body_h))
eye = sphere(1.5)
eyes = eye.translate((body_len - 6, body_w - 1.5, z + 3)) | eye.translate((body_len - 6, -body_w + 1.5, z + 3))

fish = body | tail.k(3) | dorsal.k(2)
fish = fish - eyes
result = fish - box(200, center=(0, 0, -100))  # flat base at z = 0
show_object(result)
'''),
 dict(id="sdf_pear", cls="food",
  keywords="pear fruit stalk food round bottom narrow top",
  request="A pear 55 mm tall with a short stalk.",
  code='''from sdf import *

bottom_r, top_r = 17, 10
bottom_z = bottom_r - 2        # sinks 2 mm into the base for a flat footprint
top_z = bottom_z + 20

# Two spheres blended with a large radius give the pear's waist.
fruit = sphere(bottom_r).translate((0, 0, bottom_z)) | sphere(top_r).translate((0, 0, top_z)).k(14)
stalk = capsule((0, 0, top_z + top_r - 2), (2, 0, top_z + top_r + 8), 1.2)

result = (fruit | stalk.k(1.5)) - box(200, center=(0, 0, -100))
show_object(result)
'''),
 dict(id="sdf_teddy_bear", cls="characters",
  keywords="teddy bear toy sitting arms legs ears snout character",
  request="A sitting teddy bear about 50 mm tall.",
  code='''from sdf import *

body = ellipsoid((13, 11, 15)).translate((0, 0, 14))
head = sphere(10).translate((0, 0, 36))
snout = ellipsoid((5, 5, 4)).translate((8, 0, 34))
ear = sphere(4)
leg = capsule((0, 0, 0), (12, 0, 0), 5)         # legs stick forward from the hips
arm = capsule((0, 0, 0), (5, 0, -12), 4)        # arms hang down along the front of the body
eye = sphere(1.3)

half = union(                                   # one side (+Y); mirrored below
    ear.translate((0, 8, 44)),
    leg.translate((2, 7, 5)),
    arm.translate((3, 11, 24)),
)
bear = union(body, head.k(4), snout.k(2), half.k(3), half.mirror(Y).k(3))
bear = bear - eye.translate((8, 4, 39)) - eye.translate((8, -4, 39))

result = bear - box(200, center=(0, 0, -100))
show_object(result)
'''),
 dict(id="sdf_bunny", cls="animals",
  keywords="bunny rabbit long ears sitting animal tail",
  request="A sitting rabbit 50 mm tall with long upright ears.",
  code='''import math
from sdf import *

body = ellipsoid((14, 11, 13)).translate((-2, 0, 11))
head = ellipsoid((8, 7, 7)).translate((10, 0, 26))
haunch = ellipsoid((8, 4, 7)).translate((-4, 8, 8))
foot = ellipsoid((7, 3, 2.5)).translate((4, 8, 2))
ear = ellipsoid((3, 1.5, 10)).rotate(math.radians(-10), Y)  # lean back a little
tail = sphere(4).translate((-16, 0, 10))
eye = sphere(1.2)

half = union(haunch, foot.k(2), ear.translate((7, 3.5, 38)))
rabbit = union(body, head.k(5), half.k(3), half.mirror(Y).k(3), tail.k(2))
rabbit = rabbit - eye.translate((16, 3.5, 28)) - eye.translate((16, -3.5, 28))

result = rabbit - box(200, center=(0, 0, -100))
show_object(result)
'''),
 dict(id="sdf_penguin", cls="animals",
  keywords="penguin bird standing flippers beak feet",
  request="A standing penguin 55 mm tall with flippers, a beak and flat feet.",
  code='''import math
from sdf import *

body = ellipsoid((13, 12, 20)).translate((0, 0, 20))
head = sphere(9).translate((1, 0, 42))
beak = capped_cone((0, 0, 0), (7, 0, 0), 2.5, 0.4).translate((8, 0, 41))
flipper = ellipsoid((4, 1.5, 12)).rotate(math.radians(-15), X)   # tilt the bottom outward
foot = ellipsoid((6, 3.5, 1.5)).translate((6, 5, 1.5))
eye = sphere(1.2)

half = union(flipper.translate((0, 12, 24)), foot)
penguin = union(body, head.k(6), beak.k(1), half.k(2), half.mirror(Y).k(2))
penguin = penguin - eye.translate((8, 4, 45)) - eye.translate((8, -4, 45))

result = penguin - box(200, center=(0, 0, -100))
show_object(result)
'''),
 dict(id="sdf_ladybug", cls="animals",
  keywords="ladybug beetle insect dome shell spots head bug",
  request="A ladybug 40 mm long: a domed shell with a centre line and round spots, and a small head.",
  code='''from sdf import *

shell_len, shell_w, shell_h = 16, 13, 11       # ellipsoid radii
dome = ellipsoid((shell_len, shell_w, shell_h)) - box(100, center=(0, 0, -50))  # top half only
head = ellipsoid((6, 7, 5)).translate((shell_len, 0, 2))
# Centre line: a 1 mm deep groove, the dome's outer skin where it crosses a thin slab.
skin = ellipsoid((shell_len, shell_w, shell_h)).shell(2)
centre_line = skin & box((2 * shell_len, 1.2, 2 * shell_h), center=(-4, 0, shell_h))
# Spots: shallow dents, 3 mm balls with their centres just under the surface.
spot = sphere(3)
spots = union(*[spot.translate((x, y, z)) for x, y, z in
                [(5, 6, 8), (-6, 7, 7), (5, -6, 8), (-6, -7, 7)]])

result = (dome | head.k(2)) - centre_line - spots.k(0.5) - box(200, center=(0, 0, -100))
show_object(result)
'''),
 dict(id="sdf_cactus", cls="plants",
  keywords="cactus potted plant arms pot succulent desert",
  request="A cactus with two arms growing out of a round pot, 65 mm tall in total.",
  code='''from sdf import *

pot_r_bottom, pot_r_top, pot_h = 13, 16, 18
pot = capped_cone((0, 0, 0), (0, 0, pot_h), pot_r_bottom, pot_r_top)
soil_z = pot_h - 1

stem = capsule((0, 0, soil_z), (0, 0, 60), 7)
# Each arm: out sideways, then up.
arm_r = 4.5
left = capsule((0, 0, 30), (-14, 0, 30), arm_r) | capsule((-14, 0, 30), (-14, 0, 44), arm_r).k(3)
right = capsule((0, 0, 38), (12, 0, 38), arm_r) | capsule((12, 0, 38), (12, 0, 50), arm_r).k(3)

cactus = stem | left.k(3) | right.k(3)
result = pot | cactus.k(1)
show_object(result)
'''),
 dict(id="sdf_pumpkin", cls="food",
  keywords="pumpkin squash ribbed fruit lobes stem halloween",
  request="A ribbed pumpkin 60 mm across with a curved stem.",
  code='''import math
from sdf import *

radius, height = 30, 20        # overall half width and half height
lobes = 8
lobe = ellipsoid((radius * 0.55, radius * 0.45, height))
# Lobes in a ring, each pushed out from the centre and turned to face outward, blended together.
ring = [lobe.translate((radius * 0.45, 0, 0)).rotate(2 * math.pi * i / lobes, Z) for i in range(lobes)]
body = union(*ring, k=4).translate((0, 0, height - 1))
stem = (capsule((0, 0, 0), (0, 0, 8), 3) | capsule((0, 0, 8), (4, 0, 12), 2.5).k(2)).translate((0, 0, 2 * height - 3))

result = (body | stem.k(2)) - box(200, center=(0, 0, -100))
show_object(result)
'''),
]


def build() -> bool:
    out, ok = [], True
    for e in EX:
        r = run_cadquery(e["code"], timeout=120, language="sdf")
        if not r.success:
            print("FAIL", e["id"], r.traceback)
            ok = False
            continue
        v = validate_mesh(r.stl_path)
        import trimesh

        z_min = float(trimesh.load(r.stl_path).bounds[0][2])
        good = v.valid and abs(z_min) < 0.5
        print(e["id"], "valid" if good else "INVALID", v.bbox_mm, f"z_min {z_min:.2f}", v.issues)
        ok &= good
        bbox = [round(x * 2) / 2 for x in v.bbox_mm]  # to 0.5 mm, as a model would write it
        out.append(dict(id=e["id"], cls=e["cls"], keywords=e["keywords"], request=e["request"], code=e["code"],
                        expected_bbox_mm=bbox, volume_mm3=round(v.volume_mm3, 2), stl=r.stl_path))
    if ok:
        SDF_EXAMPLES_PATH.write_text(json.dumps([{k: v for k, v in e.items() if k != "stl"} for e in out], indent=1))
    print("ALL OK" if ok else "SOME FAILED")
    return ok, out


if __name__ == "__main__":
    ok, _ = build()
    sys.exit(0 if ok else 1)
