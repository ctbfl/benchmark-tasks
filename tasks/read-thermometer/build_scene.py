"""Rebuild original assets and a portable tabletop assembly in this task directory."""
from pathlib import Path
from pxr import Gf, Usd, UsdGeom
from scene import PLACEMENTS

HERE = Path(__file__).resolve().parent


def assemble():
    stage = Usd.Stage.CreateNew(str(HERE / "assets/scene.usda"))
    root = UsdGeom.Xform.Define(stage, "/Scene")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    for name, filename, _, position, yaw in PLACEMENTS:
        prim = UsdGeom.Xform.Define(stage, "/Scene/" + name)
        prim.GetPrim().GetReferences().AddReference(filename)
        UsdGeom.Xformable(prim).AddTranslateOp().Set(Gf.Vec3d(*position))
        if yaw:
            prim.AddRotateZOp().Set(yaw)
    stage.GetRootLayer().Save()


if __name__ == "__main__":
    assemble()
