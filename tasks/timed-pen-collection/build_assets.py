"""Original, metre-scale pen, hollow holder, push-button clock and tape assets."""
from pathlib import Path
import math

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

HERE = Path(__file__).resolve().parent
PEN_LENGTH = .150
PEN_RADIUS = .0052
CUP_SIZE = (.07874, .07366)
CUP_INNER = (.06985, .06350)
CUP_HEIGHT = .10414
CUP_FLOOR = .006
BUTTON_Z = .096
BUTTON_TRAVEL = .006
SEGMENTS = ("abcedf", "bc", "abged", "abgcd", "fgbc", "afgcd", "afgecd", "abc", "abcdefg", "abfgcd")


def material(stage, name, color, roughness=.45, metallic=0., emission=None):
    mat = UsdShade.Material.Define(stage, "/Asset/materials/" + name)
    shader = UsdShade.Shader.Define(stage, str(mat.GetPath()) + "/shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    if emission is not None:
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*emission))
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def finish(geom, mat, collision=False):
    UsdShade.MaterialBindingAPI.Apply(geom.GetPrim()).Bind(mat)
    if collision:
        UsdPhysics.CollisionAPI.Apply(geom.GetPrim())
        geom.GetPrim().AddAppliedSchema("PhysxCollisionAPI")
        geom.GetPrim().CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(.001)
        geom.GetPrim().CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(0.)
        if geom.GetPrim().IsA(UsdGeom.Mesh):
            UsdPhysics.MeshCollisionAPI.Apply(geom.GetPrim()).CreateApproximationAttr("convexHull")
    return geom


def box(stage, path, size, pos, mat, collision=False):
    geom = UsdGeom.Cube.Define(stage, path)
    geom.CreateSizeAttr(1.)
    geom.AddTranslateOp().Set(Gf.Vec3d(*pos))
    geom.AddScaleOp().Set(Gf.Vec3f(*size))
    return finish(geom, mat, collision)


def cylinder(stage, path, radius, height, z, mat, collision=False):
    geom = UsdGeom.Cylinder.Define(stage, path)
    geom.CreateRadiusAttr(radius)
    geom.CreateHeightAttr(height)
    geom.AddTranslateOp().Set(Gf.Vec3d(0, 0, z))
    return finish(geom, mat, collision)


def mesh(stage, path, vertices, faces, mat, collision=False):
    geom = UsdGeom.Mesh.Define(stage, path)
    geom.CreatePointsAttr([Gf.Vec3f(*map(float, v)) for v in vertices])
    geom.CreateFaceVertexCountsAttr([len(f) for f in faces])
    geom.CreateFaceVertexIndicesAttr([i for f in faces for i in f])
    geom.CreateSubdivisionSchemeAttr("none")
    return finish(geom, mat, collision)


def outline(size, radius, steps=8):
    points = []
    for sx, sy, start in ((1, 1, 0), (-1, 1, 90), (-1, -1, 180), (1, -1, 270)):
        for degrees in np.linspace(start, start + 90, steps, endpoint=False):
            a = math.radians(degrees)
            points.append((sx * (size[0] / 2 - radius) + radius * math.cos(a),
                           sy * (size[1] / 2 - radius) + radius * math.sin(a)))
    return np.array(points)


def prism(stage, path, points, bottom, top, mat, collision=False):
    n = len(points)
    vertices = [(x, y, z) for z in (bottom, top) for x, y in points]
    faces = [list(reversed(range(n))), list(range(n, 2*n))]
    faces += [[i, (i+1) % n, (i+1) % n + n, i+n] for i in range(n)]
    return mesh(stage, path, vertices, faces, mat, collision)


def rigid(stage, path, mass, friction=.6):
    body = UsdGeom.Xform.Define(stage, path)
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr(mass)
    physics = UsdShade.Material.Define(stage, path + "/surface")
    coeff = UsdPhysics.MaterialAPI.Apply(physics.GetPrim())
    coeff.CreateStaticFrictionAttr(friction)
    coeff.CreateDynamicFrictionAttr(friction * .85)
    coeff.CreateRestitutionAttr(0.)
    UsdShade.MaterialBindingAPI.Apply(body.GetPrim()).Bind(physics, materialPurpose="physics")
    return body


def stage_for(name):
    stage = Usd.Stage.CreateNew(str(HERE / "assets" / (name + ".usda")))
    root = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdGeom.SetStageUpAxis(stage, "Z")
    return stage


def build_pen():
    stage = stage_for("pen")
    black = material(stage, "black", (.018, .022, .026), .28)
    gray = material(stage, "smoke", (.13, .15, .16), .24)
    silver = material(stage, "metal", (.55, .59, .62), .22, .8)
    rigid(stage, "/Asset/body", .012)
    cylinder(stage, "/Asset/body/barrel", .0047, .094, -.005, black, True)
    cylinder(stage, "/Asset/body/grip", .0044, .021, -.0605, gray, True)
    cylinder(stage, "/Asset/body/tip", .00065, .004, -.073, silver, True)
    cylinder(stage, "/Asset/body/cap", PEN_RADIUS, .035, .0575, black, True)
    cylinder(stage, "/Asset/body/cap_band", .0053, .0015, .041, gray)
    box(stage, "/Asset/body/clip", (.0012, .0021, .029), (.0056, 0, .055), silver, True)
    for i in range(5):
        cylinder(stage, f"/Asset/body/grip_ring_{i}", .0046, .0005, -.068+i*.0017, black)
    stage.GetRootLayer().Save()


def build_cup():
    stage = stage_for("cup")
    white = material(stage, "ceramic", (.83, .82, .79), .26)
    gray = material(stage, "base_gray", (.30, .32, .31), .65)
    rigid(stage, "/Asset/body", .28, .7)
    outer = outline(CUP_SIZE, .014)
    inner = outline(CUP_INNER, .010)
    prism(stage, "/Asset/body/bottom", outer, 0., CUP_FLOOR, gray, True)
    for i in range(len(outer)):
        j = (i+1) % len(outer)
        ring = np.array([outer[i], outer[j], inner[j], inner[i]])
        prism(stage, f"/Asset/body/wall_{i:02}", ring, CUP_FLOOR, CUP_HEIGHT, white, True)
    stage.GetRootLayer().Save()


def build_timer():
    stage = stage_for("timer")
    UsdPhysics.ArticulationRootAPI.Apply(stage.GetDefaultPrim())
    shell = material(stage, "shell", (.78, .81, .79), .4)
    dark = material(stage, "dark", (.012, .018, .023), .65)
    rubber = material(stage, "rubber", (.028, .032, .034), .9)
    red = material(stage, "button_red", (.62, .035, .024), .4)
    lit = material(stage, "digits", (.08, .58, .29), .4, emission=(.15, .85, .39))
    rigid(stage, "/Asset/base", .70, 1.0)
    prism(stage, "/Asset/base/housing", outline((.180, .070), .009), .006, .086, shell, True)
    for i, (x, y) in enumerate(((-.065, -.022), (-.065, .022), (.065, -.022), (.065, .022))):
        box(stage, f"/Asset/base/foot_{i}", (.033, .022, .006), (x, y, .003), rubber, True)
    box(stage, "/Asset/base/screen", (.155, .002, .049), (0, -.0355, .046), dark)
    # Seven-segment digits on the front face; front normal is local -Y.
    for digit, x in enumerate((-.062, -.043, -.012, .007, .038, .057)):
        for name, dx, dz, width, height in (
                ("a", 0, .014, .011, .002), ("b", .0065, .007, .002, .011),
                ("c", .0065, -.007, .002, .011), ("d", 0, -.014, .011, .002),
                ("e", -.0065, -.007, .002, .011), ("f", -.0065, .007, .002, .011),
                ("g", 0, 0, .011, .002)):
            seg = box(stage, f"/Asset/base/display/d{digit}_{name}", (width, .0006, height),
                      (x+dx, -.0368, .046+dz), lit)
            if name == "g":
                seg.CreateVisibilityAttr("invisible")
    for i, x in enumerate((-.027, .023)):
        for j, dz in enumerate((-.006, .006)):
            box(stage, f"/Asset/base/display/colon_{i}_{j}", (.0025, .0006, .0025), (x, -.0368, .046+dz), lit)
    button = rigid(stage, "/Asset/button", .025, .8)
    button.AddTranslateOp().Set(Gf.Vec3d(0, .003, BUTTON_Z))
    prism(stage, "/Asset/button/cap", outline((.132, .050), .009), -.007, .007, red, True)
    joint = UsdPhysics.PrismaticJoint.Define(stage, "/Asset/button_slide")
    joint.CreateBody0Rel().SetTargets(["/Asset/base"])
    joint.CreateBody1Rel().SetTargets(["/Asset/button"])
    joint.CreateLocalPos0Attr(Gf.Vec3f(0, .003, BUTTON_Z))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    joint.CreateAxisAttr("Z")
    joint.CreateLowerLimitAttr(-BUTTON_TRAVEL)
    joint.CreateUpperLimitAttr(0.)
    spring = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
    spring.CreateTypeAttr("force")
    spring.CreateStiffnessAttr(500.)
    spring.CreateDampingAttr(4.)
    spring.CreateTargetPositionAttr(0.)
    joint.GetPrim().AddAppliedSchema("PhysicsJointStateAPI:linear")
    joint.GetPrim().CreateAttribute("state:linear:physics:position", Sdf.ValueTypeNames.Float).Set(0.)
    stage.GetRootLayer().Save()


def build_tape():
    stage = stage_for("tape")
    red = material(stage, "red_paper", (.64, .018, .012), .92)
    # Coordinates match the task scene; tape has no obstacle-height collision.
    for prefix, (x, y), (sx, sy) in (("pens", (.415, 0.), (.39, .43)), ("cup", (.46, -.325), (.104, .104))):
        for i, (size, pos) in enumerate((
                ((sx+.012, .012, .00025), (x, y-sy/2-.006, .0002)),
                ((sx+.012, .012, .00025), (x, y+sy/2+.006, .0002)),
                ((.012, sy, .00025), (x-sx/2-.006, y, .0002)),
                ((.012, sy, .00025), (x+sx/2+.006, y, .0002)))):
            box(stage, f"/Asset/{prefix}_{i}", size, pos, red)
    stage.GetRootLayer().Save()


if __name__ == "__main__":
    (HERE / "assets").mkdir(exist_ok=True)
    for build in (build_pen, build_cup, build_timer, build_tape):
        build()
    print("Built pen, cup, timer and tape assets", flush=True)
