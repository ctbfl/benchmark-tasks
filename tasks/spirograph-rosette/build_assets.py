"""Generate the original, portable USD kit from its millimetre-scale dimensions."""

from pathlib import Path
import math

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from geometry import (
    ADDENDUM, BOARD, BORE, DEDENDUM, HEIGHT, HOLE, MODULE, OUTSIDE, PAPER, PARK,
    RING, RING_TEETH, SHANK, THICKNESS, TIP, WHEEL, WHEEL_TEETH, polar, tooth,
)

def schema(prim, name, **attributes):
    prim.AddAppliedSchema(name)
    namespace = name.removesuffix("API")
    namespace = namespace[0].lower() + namespace[1:]
    for key, value in attributes.items():
        dtype = Sdf.ValueTypeNames.Int if type(value) is int else Sdf.ValueTypeNames.Float
        prim.CreateAttribute(f"{namespace}:{key}", dtype, custom=False).Set(value)


ASSETS = Path(__file__).with_name("assets")
TAU = 2 * math.pi


def kit(name, mass=None, fixed=False, com=None):
    stage = Usd.Stage.CreateNew(str(ASSETS / f"{name}.usda"))
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/Object").GetPrim()
    stage.SetDefaultPrim(root)
    rigid = UsdPhysics.RigidBodyAPI.Apply(root)
    rigid.CreateKinematicEnabledAttr(fixed)
    schema(root, "PhysxRigidBodyAPI", solverPositionIterationCount=24, solverVelocityIterationCount=4)
    schema(root, "PhysxContactReportAPI", threshold=0.)
    if mass is not None:
        api = UsdPhysics.MassAPI.Apply(root)
        api.CreateMassAttr(mass)
        api.CreateCenterOfMassAttr(Gf.Vec3f(*com))
    return stage


def material(stage, name, rgb, friction, opacity=1., roughness=.38):
    path = "/Object/Materials/" + name
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/Surface")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(friction)
    pm.CreateDynamicFrictionAttr(.8 * friction)
    pm.CreateRestitutionAttr(0.)
    return mat


def finish(shape, mat, mesh=False, convex=True):
    prim = shape.GetPrim()
    binding = UsdShade.MaterialBindingAPI.Apply(prim)
    binding.Bind(mat)
    if not convex:
        return
    binding.Bind(mat, materialPurpose="physics")
    UsdPhysics.CollisionAPI.Apply(prim)
    schema(prim, "PhysxCollisionAPI", contactOffset=.00023, restOffset=0.)
    if mesh:
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexHull")
    return shape


def prism(stage, name, xy, z0, z1, mat, convex=True):
    xy = np.asarray(xy, dtype=float)
    area = np.sum(xy[:, 0] * np.roll(xy[:, 1], -1) - xy[:, 1] * np.roll(xy[:, 0], -1))
    if area < 0:
        xy = xy[::-1]
    n = len(xy)
    points = [(float(x), float(y), z) for z in (z0, z1) for x, y in xy]
    faces = [tuple(reversed(range(n))), tuple(range(n, 2*n))]
    for i in range(n):
        j = (i + 1) % n
        faces.append((i, j, j+n, i+n))
    shape = UsdGeom.Mesh.Define(stage, "/Object/" + name)
    shape.CreatePointsAttr(points)
    shape.CreateFaceVertexCountsAttr([len(face) for face in faces])
    shape.CreateFaceVertexIndicesAttr([i for face in faces for i in face])
    shape.CreateSubdivisionSchemeAttr("none")
    shape.CreateExtentAttr(UsdGeom.PointBased.ComputeExtent(shape.GetPointsAttr().Get()))
    finish(shape, mat, mesh=True, convex=convex)
    return shape


def box(stage, name, size, position, mat):
    shape = UsdGeom.Cube.Define(stage, "/Object/" + name)
    shape.CreateSizeAttr(1.)
    shape.AddTranslateOp().Set(Gf.Vec3d(*position))
    shape.AddScaleOp().Set(Gf.Vec3f(*size))
    return finish(shape, mat)


def cylinder(stage, name, radius, low, high, mat):
    shape = UsdGeom.Cylinder.Define(stage, "/Object/" + name)
    shape.CreateRadiusAttr(radius)
    shape.CreateHeightAttr(high-low)
    shape.CreateAxisAttr("Z")
    shape.AddTranslateOp().Set(Gf.Vec3d(0, 0, (low+high)/2))
    return finish(shape, mat)


def annular_slice(radius0, radius1, a, b, center=(0, 0)):
    return [np.array(center) + radius * polar(angle)
            for radius, angle in ((radius0, a), (radius1, a), (radius1, b), (radius0, b))]


def make_board():
    stage = kit("board", fixed=True)
    blue = material(stage, "Slate", (.028, .090, .125), .50)
    white = material(stage, "Paper", (.89, .86, .77), .30, roughness=.79)
    black = material(stage, "Socket", (.020, .031, .043), .44)
    clip = material(stage, "Clips", (.21, .30, .34), .48, roughness=.24)
    box(stage, "DrawingBoard", (*BOARD, .010), (0, 0, .005), blue)
    box(stage, "Sheet", (.198, .172, .00050), (.012, .018, PAPER-.00025), white)
    for i in range(16):
        xy = annular_slice(.0078, .013, i*TAU/16, (i+1)*TAU/16, PARK)
        prism(stage, f"PenSocket_{i:02}", xy, .010, .031, black)
    for sign in (-1, 1):
        box(stage, f"RingClamp_{sign+1}", (.032, .020, .010), (0, sign*.073, .0177), clip)
        box(stage, f"ClampFoot_{sign+1}", (.036, .018, .008), (0, sign*.082, .014), blue)
    stage.GetRootLayer().Save()


def make_gear(internal):
    name = "ring" if internal else "wheel"
    teeth = RING_TEETH if internal else WHEEL_TEETH
    stage = kit(name, fixed=True) if internal else kit(name, .030, com=(0, 0, THICKNESS/2))
    body = material(stage, "RingPlastic" if internal else "WheelPlastic",
                    (.022, .36, .66) if internal else (.36, .74, .45), .20,
                    opacity=1. if internal else .63)
    rim = body if internal else material(stage, "ToothPlastic", (.26, .64, .34), .20)
    root = RING + DEDENDUM if internal else WHEEL - DEDENDUM
    pieces = 72 if internal else 64
    for i in range(pieces):
        a, b = i*TAU/pieces, (i+1)*TAU/pieces
        if internal:
            xy = annular_slice(root, OUTSIDE, a, b)
        else:
            def outer(angle):
                u = polar(angle)
                length = -HOLE*u[0] + math.sqrt(root*root - (HOLE*u[1])**2)
                return np.array([HOLE, 0]) + length*u

            center = np.array([HOLE, 0])
            xy = [center+BORE*polar(a), outer(a), outer(b), center+BORE*polar(b)]
        prism(stage, f"Web_{i:02}", xy, 0., THICKNESS, body)
    for i in range(teeth):
        # Two convex flank spans preserve the involute over the engagement depth.
        profile = tooth((i + (.5 if internal else 0.))*TAU/teeth, teeth, internal)
        for part, (lo, hi) in enumerate(((0, 3), (3, 5))):
            piece = profile[lo:hi+1] + profile[-hi-1:len(profile)-lo]
            prism(stage, f"Tooth_{i:02}_{part}", piece, .00012, THICKNESS-.00012, rim)
    if not internal:
        accent = material(stage, "BoreMarker", (.065, .36, .14), .2)
        for i in range(24):
            xy = annular_slice(BORE+.0001, BORE+.0017, i*TAU/24, (i+1)*TAU/24, (HOLE, 0))
            prism(stage, f"BoreMarker_{i:02}", xy, THICKNESS, THICKNESS+.00010, accent, convex=False)
        xy = [[-.027, -.001], [-.020, -.001], [-.020, .001], [-.027, .001]]
        prism(stage, "PhaseMarker", xy, THICKNESS, THICKNESS+.00010, accent, convex=False)
    stage.GetRootLayer().Save()


def make_pen():
    stage = kit("pen", .020, com=(0, 0, .055))
    orange = material(stage, "OrangeBarrel", (.84, .20, .030), .74)
    grip = material(stage, "Grip", (.31, .067, .012), .91)
    metal = material(stage, "FeltTipFerrule", (.22, .25, .26), .19, roughness=.25)
    ink = material(stage, "BlueFeltTip", (.013, .035, .17), .19)
    cylinder(stage, "Shank", SHANK, .00130, .0145, metal)
    cylinder(stage, "Barrel", .0068, .0145, HEIGHT-.004, orange)
    cylinder(stage, "GripBand", .0071, .050, .074, grip)
    cylinder(stage, "CapPlug", .0066, HEIGHT-.004, HEIGHT, orange)
    nib = UsdGeom.Sphere.Define(stage, "/Object/Nib")
    nib.CreateRadiusAttr(TIP)
    nib.AddTranslateOp().Set(Gf.Vec3d(0, 0, TIP))
    finish(nib, ink)
    stage.GetRootLayer().Save()


if __name__ == "__main__":
    ASSETS.mkdir(exist_ok=True)
    make_board()
    make_gear(True)
    make_gear(False)
    make_pen()
