"""Place pieces relative to each other and merge them into one mesh (#10). Pure geometry: no LLM, no network.

The intake gate plans the pieces (gate.base.parse_pieces): each one after the first has a `place` entry
naming an earlier piece and a relation. Axes follow CadQuery: Z up, front is -Y, right is +X.

    on_top   rests on the target's upper surface (dropped straight down until it touches)
    under    the target rests on it
    beside   next to the target on `side` (left/right/front/back), bottoms level
    inside   centred on the target and dropped in (lands on an inner floor if there is one)

Every relation sinks the piece OVERLAP mm into its target, so the union fuses them into one body.
"""

from __future__ import annotations

import numpy as np
import trimesh

OVERLAP = 0.5  # mm
RELATIONS = ("on_top", "under", "beside", "inside")
SIDES = {"left": (0, -1), "right": (0, 1), "front": (1, -1), "back": (1, 1)}  # name -> (axis, direction)
ALIGNS = ("center", *SIDES)
MAX_PROBES = 300  # points of a piece's underside used to find where it lands


def scale_to_height(mesh: trimesh.Trimesh, height: float) -> trimesh.Trimesh:
    """A copy scaled uniformly so its Z extent is `height` mm."""
    m = mesh.copy()
    z = float(m.extents[2])
    if z > 0 and height > 0:
        m.apply_scale(height / z)
    return m


def surface_heights(mesh: trimesh.Trimesh, xy: np.ndarray) -> np.ndarray:
    """Highest point of the mesh straight above-or-below each (x, y), or NaN where the mesh isn't there.
    The same as casting rays straight down, without trimesh's ray module (which needs rtree)."""
    tri = mesh.triangles  # (n, 3, 3)
    lo, hi = tri[:, :, :2].min(axis=1), tri[:, :, :2].max(axis=1)
    out = np.full(len(xy), np.nan)
    for i, (x, y) in enumerate(xy):
        near = np.flatnonzero((lo[:, 0] <= x) & (x <= hi[:, 0]) & (lo[:, 1] <= y) & (y <= hi[:, 1]))
        if not len(near):
            continue
        a, b, c = tri[near, 0], tri[near, 1], tri[near, 2]
        # Barycentric coordinates of (x, y) in each triangle's XY projection.
        v0, v1 = b[:, :2] - a[:, :2], c[:, :2] - a[:, :2]
        v2 = np.array([x, y]) - a[:, :2]
        den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
        ok = np.abs(den) > 1e-12  # skip triangles seen edge-on from above
        with np.errstate(divide="ignore", invalid="ignore"):
            u = (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / den
            v = (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / den
        u, v = np.where(ok, u, -1.0), np.where(ok, v, -1.0)  # edge-on triangles never contain the point
        inside = (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9)
        if inside.any():
            z = a[inside, 2] + u[inside] * (b[inside, 2] - a[inside, 2]) + v[inside] * (c[inside, 2] - a[inside, 2])
            out[i] = z.max()
    return out


def _underside(mesh: trimesh.Trimesh) -> np.ndarray:
    """Vertices in the lowest 15% of the piece (at most MAX_PROBES): the points that touch down first."""
    z0, h = mesh.bounds[0, 2], mesh.extents[2]
    pts = mesh.vertices[mesh.vertices[:, 2] <= z0 + 0.15 * h]
    if len(pts) > MAX_PROBES:
        pts = pts[np.linspace(0, len(pts) - 1, MAX_PROBES).astype(int)]
    return pts


def _drop_onto(piece: trimesh.Trimesh, target: trimesh.Trimesh) -> float:
    """Z shift that sets `piece` down on `target`'s surface below it. Falls back to the target's top when the
    piece's footprint misses the target."""
    pts = _underside(piece)
    ground = surface_heights(target, pts[:, :2])
    hit = ~np.isnan(ground)
    if not hit.any():
        return target.bounds[1, 2] - piece.bounds[0, 2]
    return float(np.max(ground[hit] - pts[hit, 2]))


def _center_xy(mesh: trimesh.Trimesh) -> np.ndarray:
    return mesh.bounds.mean(axis=0)[:2]


def place_piece(piece: trimesh.Trimesh, target: trimesh.Trimesh, relation: str,
                side: str = "right", align: str = "center") -> trimesh.Trimesh:
    """A moved copy of `piece` next to the already placed `target`."""
    m = piece.copy()
    (pmin, pmax), (tmin, tmax) = m.bounds, target.bounds
    shift = np.zeros(3)
    shift[:2] = _center_xy(target) - _center_xy(m)

    if relation in ("on_top", "inside"):
        if relation == "on_top" and align in SIDES:
            axis, sign = SIDES[align]
            room = max(0.0, (tmax[axis] - tmin[axis]) / 2 - (pmax[axis] - pmin[axis]) / 2)
            shift[axis] += sign * room  # flush with that edge of the target, if the piece is smaller
        m.apply_translation(shift)
        m.apply_translation([0, 0, _drop_onto(m, target) - OVERLAP])
    elif relation == "under":
        shift[2] = tmin[2] - pmax[2] + OVERLAP
        m.apply_translation(shift)
    elif relation == "beside":
        axis, sign = SIDES.get(side, SIDES["right"])
        edge = tmax[axis] if sign > 0 else tmin[axis]
        near = pmin[axis] if sign > 0 else pmax[axis]
        shift[axis] = edge - near - sign * OVERLAP
        shift[2] = tmin[2] - pmin[2]
        m.apply_translation(shift)
    else:
        raise ValueError(f"Unknown relation {relation!r}")
    return m


def place(meshes: dict[str, trimesh.Trimesh], pieces: list[dict]) -> dict[str, trimesh.Trimesh]:
    """Place every piece in plan order (the first is the base), then set the whole assembly on Z = 0,
    centred on the base."""
    placed: dict[str, trimesh.Trimesh] = {}
    for p in pieces:
        m = meshes[p["id"]]
        where = p.get("place")
        if not where or where.get("on") not in placed:
            m = m.copy()
            m.apply_translation([*(-_center_xy(m)), -m.bounds[0, 2]])
        else:
            m = place_piece(m, placed[where["on"]], where.get("relation", "on_top"),
                            side=where.get("side", "right"), align=where.get("align", "center"))
        placed[p["id"]] = m
    if placed:
        z0 = min(m.bounds[0, 2] for m in placed.values())
        for m in placed.values():
            m.apply_translation([0, 0, -z0])
    return placed


def repair(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """A copy with small holes filled and normals made consistent (often enough for a downloaded mesh)."""
    m = mesh.copy()
    m.update_faces(m.nondegenerate_faces())
    m.merge_vertices()
    trimesh.repair.fill_holes(m)
    trimesh.repair.fix_normals(m)
    return m


def merge(meshes: list[trimesh.Trimesh], names: list[str] | None = None) -> tuple[trimesh.Trimesh, list[str]]:
    """One mesh from the placed pieces, plus notes for the report. A boolean union when every piece is (or
    can be repaired to be) watertight; otherwise the pieces are joined as separate overlapping bodies."""
    names = names or [f"piece {i + 1}" for i in range(len(meshes))]
    notes, fixed = [], []
    for name, m in zip(names, meshes):
        if not m.is_watertight:
            m = repair(m)
            notes.append(f"{name}: not watertight; " + ("repaired." if m.is_watertight else "repair failed."))
        fixed.append(m)
    if len(fixed) == 1:
        return fixed[0], notes
    broken = [n for n, m in zip(names, fixed) if not m.is_watertight]
    if not broken:
        try:
            union = trimesh.boolean.union(fixed, engine="manifold")
            if union.is_watertight and len(union.faces):
                return union, notes
            notes.append("The union was not watertight; kept the pieces as separate bodies.")
        except Exception as e:  # noqa: BLE001 - any engine failure falls back to joining
            notes.append(f"The union failed ({e}); kept the pieces as separate bodies.")
    else:
        notes.append(f"No union because {', '.join(broken)} could not be made watertight: the STL holds "
                     "overlapping bodies (most slicers print that as one part).")
    return trimesh.util.concatenate(fixed), notes
