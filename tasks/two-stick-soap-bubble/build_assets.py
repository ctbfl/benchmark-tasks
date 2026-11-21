"""Build portable USD geometry for the jointed wand, cradle and basin."""
import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
from scipy.optimize import brentq
from scipy.spatial.transform import Rotation

from geometry import (HERE, GRIP_X, GRIP_Z, HALF_SPAN, TIP, CORD_RADIUS, STRANDS, TRAY_X,
                      TRAY_Y, FLOOR, SURFACE, RIM, WANDS, LINKS, TABLE_TOP, CRADLE, REST_PITCH)

ASSET = "/Kit"
X_AXIS = Rotation.from_rotvec([0., np.pi/2, 0.])


def vec(value):
    return Gf.Vec3f(*map(float, value))


def quat(rotation):
    x, y, z, w = rotation.as_quat()
    return Gf.Quatf(float(w), vec([x, y, z]))


def transform(shape, pos, rotation=None):
    shape.AddTranslateOp().Set(Gf.Vec3d(*map(float, pos)))
    if rotation is not None:
        shape.AddOrientOp().Set(quat(rotation))


def color(shape, rgb):
    shape.CreateDisplayColorAttr([vec(rgb)])


def physx_attr(prim, name, value):
    kind = Sdf.ValueTypeNames.Int if isinstance(value, int) else Sdf.ValueTypeNames.Float
    prim.CreateAttribute(name, kind, custom=False).Set(value)


def collider(shape, material):
    prim = shape.GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim)
    prim.AddAppliedSchema("PhysxCollisionAPI")
    physx_attr(prim, "physxCollision:contactOffset", .001)
    physx_attr(prim, "physxCollision:restOffset", 0.)
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, materialPurpose="physics")


def box(stage, path, pos, dims, rgb, material=None):
    shape = UsdGeom.Cube.Define(stage, path)
    shape.CreateSizeAttr(1.)
    transform(shape, pos)
    shape.AddScaleOp().Set(vec(dims))
    color(shape, rgb)
    if material is not None:
        collider(shape, material)
    return shape


def cylinder(stage, path, pos, radius, length, rgb, material):
    shape = UsdGeom.Cylinder.Define(stage, path)
    shape.CreateHeightAttr(length)
    shape.CreateRadiusAttr(radius)
    transform(shape, pos, X_AXIS)
    color(shape, rgb)
    collider(shape, material)


def physical_material(stage, name, static, dynamic):
    mat = UsdShade.Material.Define(stage, ASSET + "/Looks/" + name)
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(static)
    api.CreateDynamicFrictionAttr(dynamic)
    return mat


def body(stage, name, p, rotation, mass, damping):
    xf = UsdGeom.Xform.Define(stage, ASSET + "/" + name)
    transform(xf, p, rotation)
    prim = xf.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(mass)
    prim.AddAppliedSchema("PhysxRigidBodyAPI")
    for key, value in {"solverPositionIterationCount": 48, "solverVelocityIterationCount": 4,
                       "linearDamping": damping, "angularDamping": .8, "sleepThreshold": 0.}.items():
        physx_attr(prim, "physxRigidBody:" + key, value)
    return (prim.GetPath(), np.asarray(p), rotation)


def joint(stage, name, a, b, anchor):
    j = UsdPhysics.SphericalJoint.Define(stage, ASSET + "/Knots/" + name)
    for (path, point, rotation), rel, attr in (
        (a, j.CreateBody0Rel, j.CreateLocalPos0Attr), (b, j.CreateBody1Rel, j.CreateLocalPos1Attr)
    ):
        rel().SetTargets([path])
        attr().Set(vec(rotation.inv().apply(anchor - point)))
    j.CreateCollisionEnabledAttr(False)


def hanging_nodes(length, n, x, z):
    """Constant-arc-length samples of a symmetric catenary."""
    a = brentq(lambda a: 2*a*np.sinh(HALF_SPAN/a)-length, .01, 1.)
    arc = np.linspace(length/2, -length/2, n+1)
    y = a * np.arcsinh(arc/a)
    return np.column_stack((np.full(n+1, x), y, z+a*(np.cosh(y/a)-np.cosh(HALF_SPAN/a))))


def build():
    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, ASSET)
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    rigid = physical_material(stage, "Plastic", .38, .30)
    cotton = physical_material(stage, "Cotton", .32, .25)
    grip = physical_material(stage, "Silicone", 1.35, 1.15)
    wand_bodies = []
    tilt = Rotation.from_euler("y", REST_PITCH, degrees=True)
    for sign, name, tint in zip((1, -1), WANDS, ((.13, .42, .22), (.64, .22, .09))):
        p = (GRIP_X, sign*HALF_SPAN, GRIP_Z)
        wand_bodies.append(body(stage, name, p, tilt, .044, .04))
        path = ASSET + "/" + name
        prim = stage.GetPrimAtPath(path)
        prim.AddAppliedSchema("PhysxContactReportAPI")
        physx_attr(prim, "physxContactReport:threshold", 0.)
        cylinder(stage, path + "/dowel", (.069, 0., 0.), .006, .244, (.62, .39, .18), rigid)
        cylinder(stage, path + "/sleeve", (0., 0., 0.), .012, .084, tint, grip)
        cylinder(stage, path + "/tie", (.185, 0., 0.), .0074, .013, (.29, .24, .13), cotton)
    for i, ((length, n), names) in enumerate(zip(STRANDS, LINKS)):
        offset = TIP + [(.003 if i == 0 else -.003), 0., 0.]
        x, _, z = tilt.apply(offset) + [GRIP_X, 0., GRIP_Z]
        points = hanging_nodes(length, n, x, z)
        connected = [wand_bodies[0]]
        for j, (name, a, b) in enumerate(zip(names, points[:-1], points[1:])):
            delta = b-a
            d = np.linalg.norm(delta)
            orientation, _ = Rotation.align_vectors([delta/d], [[0., 0., 1.]])
            weighted = i == 1 and j == n//2
            connected.append(body(stage, name, (a+b)/2, orientation, .0028 if weighted else .00034, .22))
            path = ASSET + "/" + name
            rope = UsdGeom.Capsule.Define(stage, path + "/cotton")
            rope.CreateHeightAttr(float(d-.0003))
            rope.CreateRadiusAttr(CORD_RADIUS)
            color(rope, (.84, .69, .38))
            collider(rope, cotton)
            if weighted:
                bead = UsdGeom.Sphere.Define(stage, path + "/bead")
                bead.CreateRadiusAttr(.006)
                color(bead, (.19, .23, .25))
                collider(bead, rigid)
        connected.append(wand_bodies[1])
        for j, (a, b, anchor) in enumerate(zip(connected[:-1], connected[1:], points)):
            joint(stage, f"strand_{i}_{j:02d}", a, b, anchor)
    # Adjacent strands converge at each tie; filter only the short knot region.
    for side in (0, -1):
        neighbor = 1 if side == 0 else -2
        tips = [LINKS[0][side], LINKS[0][neighbor], LINKS[1][side], LINKS[1][neighbor]]
        for name in tips[:2]:
            api = UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(ASSET + "/" + name))
            api.CreateFilteredPairsRel().SetTargets([ASSET + "/" + t for t in tips[2:]])
    for i, (x, y, z) in enumerate(CRADLE):
        radius = .006
        rest = z - (x-GRIP_X)*np.tan(np.deg2rad(REST_PITCH)) - radius/np.cos(np.deg2rad(REST_PITCH))
        # A narrow saddle leaves both jaw pads clear at the pickup point.
        box(stage, f"{ASSET}/Cradle/post_{i}", (x, y, (rest+TABLE_TOP)/2),
            (.019, .032, rest-TABLE_TOP), (.15, .22, .24), rigid)
        for k in (-1, 1):
            box(stage, f"{ASSET}/Cradle/edge_{i}_{k+1}", (x, y+k*.012, rest+.005),
                (.019, .006, .010), (.25, .32, .34), rigid)
    x0, x1 = TRAY_X
    y0, y1 = TRAY_Y
    w = .008
    h = RIM-FLOOR
    mx, my = (x1+x0)/2, (y1+y0)/2
    z = FLOOR+h/2
    tray = (.095, .34, .38)
    box(stage, ASSET + "/Basin/base", (mx, my, TABLE_TOP+(FLOOR-TABLE_TOP)/2),
        (x1-x0+2*w, y1-y0+2*w, FLOOR-TABLE_TOP), tray, rigid)
    for i, x in enumerate((x0-w/2, x1+w/2)):
        box(stage, f"{ASSET}/Basin/end_{i}", (x, my, z), (w, y1-y0+2*w, h), tray, rigid)
    for i, y in enumerate((y0-w/2, y1+w/2)):
        box(stage, f"{ASSET}/Basin/side_{i}", (mx, y, z), (x1-x0, w, h), tray, rigid)
    liquid = box(stage, ASSET + "/Basin/solution", (mx, my, SURFACE-.0002),
                 (x1-x0, y1-y0, .0004), (.08, .37, .40))
    mat = UsdShade.Material.Define(stage, ASSET + "/Looks/Solution")
    sh = UsdShade.Shader.Define(stage, ASSET + "/Looks/Solution/surface")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(vec((.065, .27, .29)))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(.15)
    sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(.76)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(liquid.GetPrim()).Bind(mat)
    path = HERE / "assets" / "wand-kit.usda"
    stage.GetRootLayer().Export(str(path))


if __name__ == "__main__":
    build()
