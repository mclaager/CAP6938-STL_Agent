"""Signed distance function (SDF) modeling for organic parts (#42).

Generated scripts build a shape from rounded primitives joined with smooth blends and pass it to
`show_object`; the sandbox meshes it with `to_mesh`. The names and arguments follow fogleman/sdf
(github.com/fogleman/sdf, not a dependency), so what a model knows of that library carries over; this is
a smaller subset written from Inigo Quilez's distance formulas.

    from sdf import *
    body = ellipsoid((20, 12, 14))
    head = sphere(9).translate((18, 0, 12))
    show_object(body | head.k(6))       # .k(r): blend into the shape it is joined to, fillet radius ~r

Units are mm. Distances are negative inside. Each shape also carries a bounding box, used to mesh it.
"""

from __future__ import annotations

import math

import numpy as np

ORIGIN = np.zeros(3)
X, Y, Z = np.eye(3)
UP = Z

__all__ = ["SDF3", "X", "Y", "Z", "UP", "ORIGIN", "sphere", "ellipsoid", "box", "rounded_box", "capsule",
           "capped_cylinder", "capped_cone", "rounded_cone", "torus", "union", "difference", "intersection",
           "to_mesh", "save"]


def _vec(v, n=3) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    return np.full(n, float(v)) if v.ndim == 0 else v.reshape(n)


def _length(a: np.ndarray) -> np.ndarray:
    return np.sqrt(np.einsum("ij,ij->i", a, a))


class SDF3:
    """A 3D shape: `f` maps an (N, 3) array of points to N distances; `lo`/`hi` bound the shape."""

    def __init__(self, f, lo, hi, k: float | None = None):
        self.f, self.lo, self.hi, self._k = f, _vec(lo), _vec(hi), k

    def __call__(self, p) -> np.ndarray:
        return self.f(np.asarray(p, dtype=float).reshape(-1, 3))

    # -- joining -----------------------------------------------------------------------------------------
    def k(self, k: float) -> SDF3:
        """Smoothing radius used when this shape is joined to, cut from or intersected with another."""
        return SDF3(self.f, self.lo, self.hi, k)

    def __or__(self, other: SDF3) -> SDF3:
        return union(self, other)

    def __sub__(self, other: SDF3) -> SDF3:
        return difference(self, other)

    def __and__(self, other: SDF3) -> SDF3:
        return intersection(self, other)

    # -- transforms --------------------------------------------------------------------------------------
    def translate(self, offset) -> SDF3:
        o = _vec(offset)
        return SDF3(lambda p: self.f(p - o), self.lo + o, self.hi + o, self._k)

    def scale(self, factor) -> SDF3:
        """Uniform (a number) or per-axis (three numbers) scale about the origin."""
        s = _vec(factor)
        if np.any(s <= 0):
            raise ValueError("scale factors must be positive (use mirror to flip a shape)")
        m = float(s.min())
        return SDF3(lambda p: self.f(p / s) * m, self.lo * s, self.hi * s, self._k)

    def rotate(self, angle: float, vector=Z) -> SDF3:
        """Rotate by `angle` radians about the axis `vector` through the origin."""
        r = _rotation(_vec(vector), angle)
        return self._transform(r)

    def orient(self, axis) -> SDF3:
        """Turn the shape so that its +Z direction points along `axis` (e.g. orient(X) lays it on its side)."""
        a = _vec(axis)
        a = a / np.linalg.norm(a)
        c = float(np.clip(Z @ a, -1, 1))
        if c > 1 - 1e-9:
            return self
        if c < -1 + 1e-9:
            return self.rotate(math.pi, X)
        return self.rotate(math.acos(c), np.cross(Z, a))

    def mirror(self, axis) -> SDF3:
        """Reflect through the plane through the origin whose normal is `axis` (mirror(X) flips x).
        For a symmetric body, build one side and join it to its mirror: `half | half.mirror(X)`."""
        n = _vec(axis)
        n = n / np.linalg.norm(n)
        return self._transform(np.eye(3) - 2 * np.outer(n, n))

    def elongate(self, size) -> SDF3:
        """Stretch the shape: cut it at the origin and move the halves apart by `size` (x, y, z) on each side,
        filling the gap with a straight section (a sphere elongated along X becomes a capsule)."""
        s = _vec(size)
        return SDF3(lambda p: self.f(p - np.clip(p, -s, s)), self.lo - s, self.hi + s, self._k)

    def shell(self, thickness: float) -> SDF3:
        """Hollow skin of the given thickness, centred on the surface (cut it open with difference)."""
        t = thickness / 2
        return SDF3(lambda p: np.abs(self.f(p)) - t, self.lo - t, self.hi + t, self._k)

    def dilate(self, r: float) -> SDF3:
        """Grow outward by r (also rounds edges)."""
        return SDF3(lambda p: self.f(p) - r, self.lo - r, self.hi + r, self._k)

    def erode(self, r: float) -> SDF3:
        return SDF3(lambda p: self.f(p) + r, self.lo, self.hi, self._k)

    def _transform(self, m: np.ndarray) -> SDF3:
        """Apply the linear map m (rotation or reflection, so distances are kept)."""
        inv = np.linalg.inv(m)
        corners = np.array([[x, y, z] for x in (self.lo[0], self.hi[0]) for y in (self.lo[1], self.hi[1])
                            for z in (self.lo[2], self.hi[2])]) @ m.T
        return SDF3(lambda p: self.f(p @ inv.T), corners.min(0), corners.max(0), self._k)

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self.lo.copy(), self.hi.copy()


def _rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    a = axis / np.linalg.norm(axis)
    x, y, z = a
    c, s = math.cos(angle), math.sin(angle)
    C = 1 - c
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                     [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                     [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


# ---------------------------------------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------------------------------------
def sphere(radius: float = 1, center=ORIGIN) -> SDF3:
    c = _vec(center)
    return SDF3(lambda p: _length(p - c) - radius, c - radius, c + radius)


def ellipsoid(size) -> SDF3:
    """Ellipsoid centred on the origin with semi-axes (radii) `size` = (rx, ry, rz)."""
    r = _vec(size)

    def f(p):
        k0 = _length(p / r)
        k1 = _length(p / (r * r))
        # The formula gives 0 at the exact centre; it is inside by the smallest radius there.
        return np.where(k1 > 1e-9, k0 * (k0 - 1) / np.maximum(k1, 1e-12), -r.min())

    return SDF3(f, -r, r)


def box(size=1, center=ORIGIN, a=None, b=None) -> SDF3:
    """Box of full size `size` (a number or (x, y, z)) centred on `center`, or spanning corners a and b."""
    if a is not None and b is not None:
        a, b = _vec(a), _vec(b)
        size, center = np.abs(b - a), (a + b) / 2
    h, c = _vec(size) / 2, _vec(center)

    def f(p):
        q = np.abs(p - c) - h
        return _length(np.maximum(q, 0)) + np.minimum(q.max(axis=1), 0)

    return SDF3(f, c - h, c + h)


def rounded_box(size, radius: float) -> SDF3:
    """Box of full size `size` centred on the origin, with every edge rounded by `radius`."""
    h = _vec(size) / 2
    r = float(radius)

    def f(p):
        q = np.abs(p) - h + r
        return _length(np.maximum(q, 0)) + np.minimum(q.max(axis=1), 0) - r

    return SDF3(f, -h, h)


def capsule(a, b, radius: float) -> SDF3:
    """Rod with round ends from point a to point b."""
    a, b = _vec(a), _vec(b)
    ba = b - a
    dd = max(float(ba @ ba), 1e-12)

    def f(p):
        pa = p - a
        h = np.clip(pa @ ba / dd, 0, 1)
        return _length(pa - h[:, None] * ba) - radius

    return SDF3(f, np.minimum(a, b) - radius, np.maximum(a, b) + radius)


def capped_cylinder(a, b, radius: float) -> SDF3:
    """Cylinder with flat ends from point a to point b."""
    a, b = _vec(a), _vec(b)
    ba = b - a
    baba = max(float(ba @ ba), 1e-12)

    def f(p):
        pa = p - a
        paba = pa @ ba
        x = _length(pa * baba - paba[:, None] * ba) - radius * baba
        y = np.abs(paba - baba * 0.5) - baba * 0.5
        x2, y2 = x * x, y * y * baba
        d = np.where(np.maximum(x, y) < 0, -np.minimum(x2, y2),
                     np.where(x > 0, x2, 0) + np.where(y > 0, y2, 0))
        return np.sign(d) * np.sqrt(np.abs(d)) / baba

    return SDF3(f, np.minimum(a, b) - radius, np.maximum(a, b) + radius)


def capped_cone(a, b, ra: float, rb: float) -> SDF3:
    """Cone with flat ends from point a (radius ra) to point b (radius rb)."""
    a, b = _vec(a), _vec(b)
    rba = rb - ra
    ba = b - a
    baba = max(float(ba @ ba), 1e-12)

    def f(p):
        pa = p - a
        papa = np.einsum("ij,ij->i", pa, pa)
        paba = pa @ ba / baba
        x = np.sqrt(np.maximum(papa - paba * paba * baba, 0))
        cax = np.maximum(0, x - np.where(paba < 0.5, ra, rb))
        cay = np.abs(paba - 0.5) - 0.5
        k = rba * rba + baba
        f_ = np.clip((rba * (x - ra) + paba * baba) / k, 0, 1)
        cbx = x - ra - f_ * rba
        cby = paba - f_
        s = np.where((cbx < 0) & (cay < 0), -1, 1)
        return s * np.sqrt(np.minimum(cax * cax + cay * cay * baba, cbx * cbx + cby * cby * baba))

    r = max(ra, rb)
    return SDF3(f, np.minimum(a, b) - r, np.maximum(a, b) + r)


def rounded_cone(r1: float, r2: float, h: float) -> SDF3:
    """Cone with round ends standing on the origin along +Z: a ball of radius r1 at the bottom (centre at the
    origin) joined smoothly to a ball of radius r2 at height h."""
    b = (r1 - r2) / h
    a = math.sqrt(max(1 - b * b, 0))

    def f(p):
        q = np.stack([np.hypot(p[:, 0], p[:, 1]), p[:, 2]], axis=1)
        k = q @ np.array([-b, a])
        c1 = np.hypot(q[:, 0], q[:, 1]) - r1
        c2 = np.hypot(q[:, 0], q[:, 1] - h) - r2
        c3 = q @ np.array([a, b]) - r1
        return np.where(k < 0, c1, np.where(k > a * h, c2, c3))

    r = max(r1, r2)
    return SDF3(f, (-r, -r, -r1), (r, r, h + r2))


def torus(r1: float, r2: float) -> SDF3:
    """Ring in the XY plane around the origin: r1 to the middle of the tube, r2 the tube's radius."""

    def f(p):
        return np.hypot(np.hypot(p[:, 0], p[:, 1]) - r1, p[:, 2]) - r2

    r = r1 + r2
    return SDF3(f, (-r, -r, -r2), (r, r, r2))


# ---------------------------------------------------------------------------------------------------------
# Joining (Inigo Quilez's polynomial smooth min/max)
# ---------------------------------------------------------------------------------------------------------
def _smooth_k(k, *shapes) -> float | None:
    if k is not None:
        return float(k)
    ks = [s._k for s in shapes[1:] if s._k]
    return max(ks) if ks else None


def union(a: SDF3, *bs: SDF3, k: float | None = None) -> SDF3:
    """Join shapes; with k (or .k(r) on a joined shape), blend them with a fillet of about k."""
    out = a
    for b in bs:
        kk = _smooth_k(k, out, b)
        out = _union2(out, b, kk)
    return out


def _union2(a: SDF3, b: SDF3, k: float | None) -> SDF3:
    if not k:
        return SDF3(lambda p: np.minimum(a.f(p), b.f(p)), np.minimum(a.lo, b.lo), np.maximum(a.hi, b.hi), a._k)

    def f(p):
        d1, d2 = a.f(p), b.f(p)
        h = np.clip(0.5 + 0.5 * (d2 - d1) / k, 0, 1)
        return d2 + (d1 - d2) * h - k * h * (1 - h)

    # The blend fills in the gap between shapes up to about k/4 outside both; the box below covers it.
    return SDF3(f, np.minimum(a.lo, b.lo) - k / 4, np.maximum(a.hi, b.hi) + k / 4, a._k)


def difference(a: SDF3, *bs: SDF3, k: float | None = None) -> SDF3:
    """Cut shapes out of a; with k, round the cut's edges."""
    out = a
    for b in bs:
        kk = _smooth_k(k, out, b)
        out = _difference2(out, b, kk)
    return out


def _difference2(a: SDF3, b: SDF3, k: float | None) -> SDF3:
    if not k:
        return SDF3(lambda p: np.maximum(a.f(p), -b.f(p)), a.lo, a.hi, a._k)

    def f(p):
        d1, d2 = a.f(p), b.f(p)
        h = np.clip(0.5 - 0.5 * (d2 + d1) / k, 0, 1)
        return d1 + (-d2 - d1) * h + k * h * (1 - h)

    return SDF3(f, a.lo, a.hi, a._k)


def intersection(a: SDF3, *bs: SDF3, k: float | None = None) -> SDF3:
    """Keep only where all shapes overlap; with k, round the edges."""
    out = a
    for b in bs:
        kk = _smooth_k(k, out, b)
        out = _intersection2(out, b, kk)
    return out


def _intersection2(a: SDF3, b: SDF3, k: float | None) -> SDF3:
    lo, hi = np.maximum(a.lo, b.lo), np.minimum(a.hi, b.hi)
    if not k:
        return SDF3(lambda p: np.maximum(a.f(p), b.f(p)), lo, hi, a._k)

    def f(p):
        d1, d2 = a.f(p), b.f(p)
        h = np.clip(0.5 - 0.5 * (d2 - d1) / k, 0, 1)
        return d2 + (d1 - d2) * h + k * h * (1 - h)

    return SDF3(f, lo, hi, a._k)


# ---------------------------------------------------------------------------------------------------------
# Meshing
# ---------------------------------------------------------------------------------------------------------
BLOCK = 16  # grid points per side of a block evaluated at once
MAX_CELLS = 400  # grid steps along the longest side at most


def default_edge(shape: SDF3) -> float:
    """Triangle edge length: about 1/80 of the largest side, at least 0.25 mm."""
    return max(float((shape.hi - shape.lo).max()) / 80, 0.25)


def to_mesh(shape: SDF3, edge: float | None = None):
    """Mesh the shape (manifold3d's level set, marching tetrahedra on a body-centred grid) into a
    watertight trimesh.Trimesh. Raises ValueError if the shape is empty or unbounded."""
    import trimesh

    lo, hi = shape.bounds()
    size = hi - lo
    if not np.all(np.isfinite(size)) or np.any(size <= 0):
        raise ValueError(f"The shape has no extent (bounds {lo.round(2).tolist()} to {hi.round(2).tolist()}).")
    edge = float(edge or default_edge(shape))
    edge = max(edge, float(size.max()) / MAX_CELLS)
    # Where two parts of a shape just touch (an arm resting on a leg), the level set can put two vertices
    # on the same point; an STL reader merges them into a pinch that is not watertight. A slightly shifted
    # grid samples the touching point differently, so try a few shifts.
    mesh = None
    for shift in (0.0, 0.37, 0.71):
        mesh = _level_set(shape, lo - (2 + shift) * edge, hi + 2 * edge, edge)
        if mesh.is_watertight:
            break
    return mesh


def _level_set(shape: SDF3, lo: np.ndarray, hi: np.ndarray, edge: float):
    import manifold3d
    import trimesh

    # level_set asks for one point at a time on a fixed grid: each axis is split into floor(length / edge)
    # cells (measured; manifold3d doesn't document it), and points lie every half cell from half a cell
    # below `lo`. Answer from blocks of that grid, each evaluated with numpy when first asked for.
    dims = hi - lo
    step = dims / np.maximum(np.floor(dims / edge), 1) / 2
    ox, oy, oz = (float(v) for v in lo - step)
    ix, iy, iz = (float(v) for v in 1 / step)
    blocks: dict[tuple[int, int, int], list] = {}
    offsets = np.stack(np.meshgrid(*[np.arange(BLOCK)] * 3, indexing="ij"), axis=-1).reshape(-1, 3)

    def value(x: float, y: float, z: float) -> float:
        # Plain floats and int(): numpy scalars and round() made this the slowest step.
        fi, fj, fk = (x - ox) * ix, (y - oy) * iy, (z - oz) * iz
        i, j, k = int(fi + 0.5), int(fj + 0.5), int(fk + 0.5)
        if fi < 0 or fj < 0 or fk < 0 or abs(fi - i) > 1e-6 or abs(fj - j) > 1e-6 or abs(fk - k) > 1e-6:
            return float(-shape.f(np.array([[x, y, z]]))[0])  # off the grid: evaluate directly
        key = (i >> 4, j >> 4, k >> 4)  # BLOCK = 16
        blk = blocks.get(key)
        if blk is None:
            pts = np.array([ox, oy, oz]) + (np.array(key) * BLOCK + offsets) * step
            blk = blocks[key] = (-shape.f(pts)).reshape(BLOCK, BLOCK, BLOCK).tolist()
        return blk[i & 15][j & 15][k & 15]

    m = manifold3d.Manifold.level_set(value, [*lo, *hi], edge).to_mesh()
    faces = np.asarray(m.tri_verts)
    if len(faces) == 0:
        raise ValueError("The shape is empty: nothing is inside it (check signs, sizes and difference order).")
    return trimesh.Trimesh(np.asarray(m.vert_properties)[:, :3], faces)  # merges coincident vertices, as STL readers do


def save(shape: SDF3, path: str, edge: float | None = None) -> None:
    to_mesh(shape, edge).export(path)
