"""A square freestanding holder with a cylindrical bore and solid bottom."""
import math
from pathlib import Path

from pxr import Usd, UsdGeom
from stand_clamp import _body, _cube, _finish, _material


def main():
    path = Path(__file__).parent / "assets/holder.usda"
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    root = _body(stage, "/Holder", .45, (0, 0, .07), (.0008, .0008, .0003))
    stage.SetDefaultPrim(root)
    material = _material(stage, "/Holder/Looks/polymer", (.18, .28, .34), roughness=.7, friction=.65)
    _cube(stage, "/Holder/floor", (0, 0, .003), (.065, .065, .006), material)
    for i in range(32):
        angles = [2 * math.pi * j / 32 for j in (i, i + 1)]
        inner = [(.0065 * math.cos(a), .0065 * math.sin(a)) for a in angles]
        outer = [(math.cos(a) * .0325 / max(abs(math.cos(a)), abs(math.sin(a))),
                  math.sin(a) * .0325 / max(abs(math.cos(a)), abs(math.sin(a)))) for a in angles]
        ring = [inner[0], outer[0], outer[1], inner[1]]
        mesh = UsdGeom.Mesh.Define(stage, f"/Holder/wall_{i:02}")
        mesh.CreatePointsAttr([(x, y, z) for z in (.006, .14) for x, y in ring])
        faces = [(3, 2, 1, 0), (4, 5, 6, 7), (0, 1, 5, 4),
                 (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
        mesh.CreateFaceVertexCountsAttr([4] * 6)
        mesh.CreateFaceVertexIndicesAttr([v for face in faces for v in face])
        mesh.CreateSubdivisionSchemeAttr("none")
        _finish(mesh, material)
    stage.GetRootLayer().Save()


if __name__ == "__main__":
    main()
