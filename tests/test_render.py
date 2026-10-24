"""Rendering an STL to the judge's 2x2 view grid (offscreen VTK)."""

import numpy as np
import trimesh

from stl_agent.render import png_bytes, render_grid


def test_grid_shows_the_part_in_every_view(tmp_path):
    stl = tmp_path / "box.stl"
    trimesh.creation.box((60, 40, 10)).export(stl)
    img = render_grid(str(stl), px=128)
    assert img.size == (384, 256)  # 3 x 2 views of 128 px
    a = np.asarray(img)
    for y in (0, 128):
        for x in (0, 128, 256):
            tile = a[y + 20:y + 128, x:x + 128]  # skip the label
            assert (tile.sum(axis=2) < 600).sum() > 200, f"tile at {x},{y} looks empty"
    assert png_bytes(img)[:8] == b"\x89PNG\r\n\x1a\n"
