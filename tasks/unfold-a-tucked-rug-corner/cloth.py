"""Read the baked FEM topology and draw surfaces from measured nodes."""
from pathlib import Path
import numpy as np
from scoring import LENGTH, WIDTH, THICKNESS
HERE = Path(__file__).resolve().parent


def topology():
    from pxr import Usd, UsdGeom
    stage = Usd.Stage.Open(str(HERE / "assets/rug.usda"))
    mesh = UsdGeom.Mesh.Get(stage, "/Rug/physics")
    rest = np.asarray(mesh.GetPrim().GetAttribute("omniphysics:restShapePoints").Get(), dtype=float)
    uv = rest[:, :2] + [LENGTH / 2, WIDTH / 2]
    faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32).reshape(-1, 3)
    return uv, faces


def normals(points, faces):
    a, b, c = np.moveaxis(points[faces], 1, 0)
    vectors = np.cross(b - a, c - a)
    result = np.zeros_like(points)
    for col in range(3):
        np.add.at(result, faces[:, col], vectors)
    length = np.linalg.norm(result, axis=1)
    if (length < 1e-12).any():
        raise ValueError("Degenerate rug surface")
    return result / length[:, None]


def surfaces(points, faces):
    offset = normals(points, faces) * THICKNESS / 2
    return points + offset, points - offset

def update_visual(stage, root, points, faces):
    from pxr import UsdGeom, Vt
    inverse = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(root)).GetInverse())
    local = points @ inverse[:3, :3] + inverse[3, :3]
    pile, backing = surfaces(local, faces)
    n = normals(local, faces)
    for name, vertices, norm in (("pile", pile, n), ("back", backing, -n)):
        mesh = UsdGeom.Mesh.Get(stage, root + "/" + name)
        mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(vertices.astype(np.float32)))
        mesh.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(norm.astype(np.float32)))
    UsdGeom.Mesh.Get(stage, root + "/bound_edge").GetPointsAttr().Set(
        Vt.Vec3fArray.FromNumpy(np.concatenate((pile, backing)).astype(np.float32)))
