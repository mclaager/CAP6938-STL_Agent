"""Render an STL to one small PNG: six labelled views in a 3x2 grid.

Used by the LLM judge (issue #8) and handy for report figures. VTK (installed with CadQuery)
renders offscreen; without a display it falls back to EGL, so it also works headless.

    from stl_agent.render import render_grid
    render_grid("part.stl").save("part.png")      # 768 x 512 at the default 256 px per view
"""

from __future__ import annotations

import io
import warnings

import numpy as np

RENDER_VERSION = "v1"  # bump when views, colors or framing change; judged rows record it
PART_COLOR = (70, 130, 200)  # same blue as the app's 3D preview
EDGE_COLOR = (40, 40, 40)

# name -> (camera direction from the part, view-up). CadQuery is Z-up; "front" looks along +Y.
# Three orthographic views for profiles, three angled ones so the back and the bottom are seen too
# (a radial hole on the far side or a hole in the bottom face is invisible from front/right/top).
VIEWS = {
    "front": ((0, -1, 0), (0, 0, 1)),
    "right": ((1, 0, 0), (0, 0, 1)),
    "top": ((0, 0, 1), (0, 1, 0)),
    "iso front": ((1, -1, 0.8), (0, 0, 1)),
    "iso back": ((-1, 1, 0.8), (0, 0, 1)),
    "from below": ((1, -1, -0.8), (0, 0, 1)),
}
COLS = 3


def render_views(stl_path: str, px: int = 256) -> dict:
    """One RGB array (px x px x 3) per view in VIEWS, white background, orthographic camera."""
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    vtk.vtkObject.GlobalWarningDisplayOff()  # e.g. "bad X server connection" before the EGL fallback

    reader = vtk.vtkSTLReader()
    reader.SetFileName(str(stl_path))
    reader.Update()
    normals = vtk.vtkPolyDataNormals()  # smooth shading on curves, sharp on edges
    normals.SetInputConnection(reader.GetOutputPort())
    normals.SetFeatureAngle(30)

    renderer = vtk.vtkRenderer()
    renderer.SetBackground(1, 1, 1)
    for source, color, width in ((normals, PART_COLOR, None), (_feature_edges(vtk, reader), EDGE_COLOR, 1.5)):
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(source.GetOutputPort())
        mapper.ScalarVisibilityOff()
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*(c / 255 for c in color))
        if width:
            actor.GetProperty().SetLineWidth(width)
            actor.GetProperty().LightingOff()
        renderer.AddActor(actor)

    vtk.vtkLightKit().AddLightsToRenderer(renderer)  # key + fill lights: no view ends up dark

    window = vtk.vtkRenderWindow()
    window.SetOffScreenRendering(1)
    window.AddRenderer(renderer)
    window.SetSize(px, px)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        points = vtk_to_numpy(reader.GetOutput().GetPoints().GetData()).astype(float)
    out = {}
    for name, (direction, up) in VIEWS.items():
        center, scale = _framing(points, direction, up)
        cam = renderer.GetActiveCamera()
        cam.ParallelProjectionOn()
        cam.SetFocalPoint(*center)
        cam.SetPosition(*(center + np.asarray(direction, float) * 10 * scale))
        cam.SetViewUp(*up)
        cam.SetParallelScale(scale)
        renderer.ResetCameraClippingRange()
        window.Render()
        grab = vtk.vtkWindowToImageFilter()
        grab.SetInput(window)
        grab.Update()
        with warnings.catch_warnings():  # vtk_to_numpy sets .shape, which NumPy 2.5 deprecates
            warnings.simplefilter("ignore", DeprecationWarning)
            img = vtk_to_numpy(grab.GetOutput().GetPointData().GetScalars()).reshape(px, px, -1)
        out[name] = np.ascontiguousarray(img[::-1, :, :3])  # VTK rows start at the bottom
    window.Finalize()
    return out


def _framing(points: np.ndarray, direction, up, margin: float = 1.08):
    """Center and parallel scale (half the visible height) that fit the part as seen from `direction`."""
    d = np.asarray(direction, float)
    d /= np.linalg.norm(d)
    u = np.asarray(up, float) - np.dot(up, d) * d
    u /= np.linalg.norm(u)
    r = np.cross(u, d)
    pu, pr = points @ u, points @ r
    center = points.mean(axis=0)
    center += u * ((pu.max() + pu.min()) / 2 - center @ u) + r * ((pr.max() + pr.min()) / 2 - center @ r)
    half = max(pu.max() - pu.min(), pr.max() - pr.min()) / 2
    return center, max(half, 1e-6) * margin


def _feature_edges(vtk, reader):
    """Outline and sharp edges, so flat faces of the same color stay readable."""
    edges = vtk.vtkFeatureEdges()
    edges.SetInputConnection(reader.GetOutputPort())
    edges.SetFeatureAngle(30)
    edges.BoundaryEdgesOn()
    edges.FeatureEdgesOn()
    edges.ManifoldEdgesOff()
    edges.NonManifoldEdgesOff()
    return edges


def render_grid(stl_path: str, px: int = 256):
    """All views as one labelled PIL image, COLS views per row."""
    from PIL import Image, ImageDraw, ImageFont

    views = render_views(stl_path, px)
    rows = -(-len(views) // COLS)
    grid = Image.new("RGB", (COLS * px, rows * px), "white")
    draw = ImageDraw.Draw(grid)
    font = ImageFont.load_default(size=max(10, px // 16))
    for i, (name, img) in enumerate(views.items()):
        x, y = (i % COLS) * px, (i // COLS) * px
        grid.paste(Image.fromarray(img), (x, y))
        draw.text((x + 6, y + 4), name, fill=(90, 90, 90), font=font)
    for c in range(1, COLS):
        draw.line([(c * px, 0), (c * px, rows * px)], fill=(200, 200, 200))
    for r in range(1, rows):
        draw.line([(0, r * px), (COLS * px, r * px)], fill=(200, 200, 200))
    return grid


def png_bytes(image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, "PNG", optimize=True)
    return buf.getvalue()
