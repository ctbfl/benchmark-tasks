"""Author the original USD geometry, joints and surface grids using OpenUSD."""
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, Vt

from geometry import *

OUT = Path(__file__).parent / "assets"
MAPLE = (.38, .21, .10)
CORAL = (.63, .25, .17)
PAPER = (.88, .87, .78)
TEAL = (.043, .31, .33)
RUBBER = (.59, .39, .19)
PALETTE_COLOR = (.66, .74, .75)
STEEL = (.25, .32, .36)
CYL_SEGMENTS = 96
TREAD_ACROSS = round(AXLE_W / PITCH)


def asset(name):
    OUT.mkdir(exist_ok=True)
    path = OUT / (name + ".usda")
    if path.exists():
        path.unlink()
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    root = UsdGeom.Xform.Define(stage, "/Asset").GetPrim()
    stage.SetDefaultPrim(root)
    return stage, root


def xform(stage, path, position):
    xf = UsdGeom.Xform.Define(stage, path)
    xf.AddTranslateOp().Set(Gf.Vec3d(*position))
    return xf.GetPrim()


def body(prim, mass):
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(mass)
    prim.AddAppliedSchema("PhysxRigidBodyAPI")
    for name, kind, value in (
        ("solverPositionIterationCount", Sdf.ValueTypeNames.Int, 24),
        ("solverVelocityIterationCount", Sdf.ValueTypeNames.Int, 6),
        ("sleepThreshold", Sdf.ValueTypeNames.Float, 0.),
    ):
        prim.CreateAttribute("physxRigidBody:" + name, kind, custom=False).Set(value)
    prim.AddAppliedSchema("PhysxContactReportAPI")
    prim.CreateAttribute("physxContactReport:threshold", Sdf.ValueTypeNames.Float, custom=False).Set(0.)


def color(geom, rgb):
    geom.CreateDisplayColorPrimvar().Set(Vt.Vec3fArray([Gf.Vec3f(*rgb)]))


def collider(geom):
    prim = geom.GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim)
    prim.AddAppliedSchema("PhysxCollisionAPI")
    for name, value in (("contactOffset", .0008), ("restOffset", 0.)):
        prim.CreateAttribute("physxCollision:" + name, Sdf.ValueTypeNames.Float,
                             custom=False).Set(value)
    material = prim.GetStage().GetPrimAtPath("/Asset/Grip")
    if not material:
        from pxr import UsdShade
        material = UsdShade.Material.Define(prim.GetStage(), "/Asset/Grip").GetPrim()
        mat = UsdPhysics.MaterialAPI.Apply(material)
        mat.CreateStaticFrictionAttr(.85)
        mat.CreateDynamicFrictionAttr(.67)
    from pxr import UsdShade
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(UsdShade.Material(material),
                                              materialPurpose="physics")


def box(stage, path, xyz, dims, rgb, physical=True):
    geom = UsdGeom.Cube.Define(stage, path)
    geom.CreateSizeAttr(1)
    geom.AddTranslateOp().Set(Gf.Vec3d(*xyz))
    geom.AddScaleOp().Set(Gf.Vec3f(*dims))
    color(geom, rgb)
    if physical:
        collider(geom)
    return geom


def cylinder(stage, path, xyz, radius, width, rgb, physical=True):
    geom = UsdGeom.Cylinder.Define(stage, path)
    geom.CreateAxisAttr("Y")
    geom.CreateHeightAttr(width)
    geom.CreateRadiusAttr(radius)
    geom.AddTranslateOp().Set(Gf.Vec3d(*xyz))
    color(geom, rgb)
    if physical:
        collider(geom)


def mesh(stage, path, quads, rgb):
    geom = UsdGeom.Mesh.Define(stage, path)
    geom.CreateSubdivisionSchemeAttr("none")
    geom.CreateDoubleSidedAttr(True)
    geom.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(quads, dtype=np.float32).reshape(-1, 3)))
    geom.CreateFaceVertexCountsAttr(Vt.IntArray([4] * len(quads)))
    geom.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.arange(4 * len(quads), dtype=np.int32)))
    geom.CreateDisplayColorPrimvar("uniform").Set(Vt.Vec3fArray([Gf.Vec3f(*rgb)] * len(quads)))


def plane(stage, path, cells, z, rgb, down=False):
    corners = PITCH / 2 * np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    quads = cells.reshape(-1, 1, 2) + corners[None]
    if down:
        quads = quads[:, ::-1]
    mesh(stage, path, np.dstack((quads, np.full(quads.shape[:2], z))), rgb)


def revolute(stage, path, a, b, axis, pa, pb):
    j = UsdPhysics.RevoluteJoint.Define(stage, path)
    j.CreateBody0Rel().SetTargets([a.GetPath()])
    j.CreateBody1Rel().SetTargets([b.GetPath()])
    j.CreateAxisAttr(axis)
    j.CreateLocalPos0Attr(Gf.Vec3f(*pa))
    j.CreateLocalPos1Attr(Gf.Vec3f(*pb))
    j.CreateCollisionEnabledAttr(False)
    return j


def build():
    stage, root = asset("palette")
    box(stage, "/Asset/slab", (0, 0, -PALETTE[2] / 2), (*PALETTE_SIZE, PALETTE[2]), PALETTE_COLOR)
    plane(stage, "/Asset/film", PALETTE_GRID, .00012, PALETTE_COLOR)
    stage.Save()

    stage, root = asset("block")
    box(stage, "/Asset/base", (0, 0, -BLOCK[2]/2 - RELIEF/2),
        (*BLOCK_SIZE, BLOCK[2] - RELIEF), MAPLE)
    for i, (a, b, c, d) in enumerate(CUT):
        box(stage, f"/Asset/ridge_{i}", ((a+b)/2000, (c+d)/2000, -RELIEF/2),
            ((b-a)/1000, (d-c)/1000, RELIEF), CORAL)
    plane(stage, "/Asset/film", BLOCK_GRID[BLOCK_MASK], .00010, CORAL)
    stage.Save()

    stage, root = asset("pad")
    box(stage, "/Asset/board", (0, .010, -PAD_TOP/2), (.132, .104, PAD_TOP), TEAL)
    stage.Save()

    stage, root = asset("brayer")
    UsdPhysics.ArticulationRootAPI.Apply(root)
    carrier = xform(stage, "/Asset/yoke", (0, 0, 0))
    axle = xform(stage, "/Asset/drum", (0, 0, 0))
    body(carrier, .095)
    body(axle, .090)
    for k, s in enumerate((-1, 1)):
        box(stage, f"/Asset/yoke/fork_{k}", (-.017, s*.043, .003), (.044, .007, .011), STEEL)
    box(stage, "/Asset/yoke/bridge", (-.039, 0, .006), (.010, .092, .011), STEEL)
    box(stage, "/Asset/yoke/shaft", (-.053, 0, .023), (.013, .015, .041), STEEL)
    box(stage, "/Asset/yoke/handle", HANDLE, (.047, .026, .024), TEAL)
    box(stage, "/Asset/yoke/rest", (-.055, 0, -.011), (.017, .025, .016), STEEL)
    cylinder(stage, "/Asset/drum/rubber", (0, 0, 0), AXLE_R, AXLE_W, RUBBER)
    revolute(stage, "/Asset/axle", carrier, axle, "Y", (0, 0, 0), (0, 0, 0))
    quads = []
    for i in range(CYL_SEGMENTS):
        a, b = -np.pi + 2*np.pi*np.array([i, i+1])/CYL_SEGMENTS
        for j in range(TREAD_ACROSS):
            lo = -AXLE_W/2 + j*PITCH
            hi = lo + PITCH
            quads.append([((AXLE_R+.00020)*np.sin(t), y, -(AXLE_R+.00020)*np.cos(t))
                          for t, y in ((a, lo), (b, lo), (b, hi), (a, hi))])
    mesh(stage, "/Asset/drum/film", quads, RUBBER)
    stage.Save()

    stage, root = asset("sheet")
    UsdPhysics.ArticulationRootAPI.Apply(root)
    width = PANEL_SIZE[1]
    panels = []
    for i in range(PANELS):
        path = f"/Asset/panel_{i}"
        panel = xform(stage, path, (0, i*width, 0))
        panels.append(panel)
        body(panel, .008 if i == 0 else .004)
        box(stage, path + "/paper", (0, 0, 0), (PANEL_SIZE[0], width-.0003, THICKNESS), PAPER)
        plane(stage, path + "/film", PANEL_GRID, -THICKNESS/2-.0001, PAPER, down=True)
        if i:
            joint = revolute(stage, f"/Asset/flex_{i}", panels[i-1], panel, "X",
                             (0, width/2, 0), (0, -width/2, 0))
            joint.CreateLowerLimitAttr(-100.)
            joint.CreateUpperLimitAttr(100.)
            drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
            drive.CreateStiffnessAttr(.030 * np.pi/180)
            drive.CreateDampingAttr(.0010 * np.pi/180)
            drive.CreateTargetPositionAttr(0)
        else:
            box(stage, path + "/paper_corner_tab", TAB, (.027, .012, .022), TEAL)
    stage.Save()


if __name__ == "__main__":
    build()
