"""Reference-derived Akejrym lamp: metres, kilograms, Z up. Import after Kit starts."""

from pathlib import Path
import math

import numpy as np
import trimesh
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, Vt

ROOT = Path(__file__).resolve().parent
ASSET = ROOT / "assets/foldable_lamp.usda"
PIVOT = np.array([0., .053, .028])
LENGTH = .106
BUTTON = np.array([0., -.035, .0189])
NAMES = ("base", "lower_arm", "upper_arm", "head")
JOINTS = ("lower", "middle", "upper")
MASSES = (.120, .020, .022, .038)
FRICTION = (.100, .060, .025)  # Estimated static friction, N m; not measured.
HIGH = np.array([-75., 150., -100.])
LOW = np.array([-30., 100., -95.])


def material(stage, path, color, roughness=.35, friction=.6, metallic=0.):
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0))
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    physics = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    physics.CreateStaticFrictionAttr(friction)
    physics.CreateDynamicFrictionAttr(friction * .8)
    physics.CreateRestitutionAttr(0.)
    return mat


def rounded(size, radius, position=(0, 0, 0), bevel=0.):
    """Convex rounded rectangular solid; matching render and collision surfaces."""
    x, y, z = np.array(size) / 2
    rings = ((-z, bevel), (-z + bevel, 0), (z - bevel, 0), (z, bevel)) if bevel else ((-z, 0), (z, 0))
    vertices = []
    for height, inset in rings:
        r = radius - inset
        for cx, cy, start in ((x-radius, y-radius, 0), (-x+radius, y-radius, 90),
                              (-x+radius, -y+radius, 180), (x-radius, -y+radius, 270)):
            for angle in np.deg2rad(np.linspace(start, start+90, 9)):
                vertices.append((cx+r*math.cos(angle), cy+r*math.sin(angle), height))
    mesh = trimesh.convex.convex_hull(np.asarray(vertices))
    mesh.apply_translation(position)
    return mesh


def cylinder(radius, length, position, axis="Z", sections=32):
    mesh = trimesh.creation.cylinder(radius, length, sections=sections)
    if axis == "X":
        mesh.apply_transform(trimesh.transformations.rotation_matrix(math.pi/2, [0, 1, 0]))
    mesh.apply_translation(position)
    return mesh


def mesh_prim(stage, path, mesh, mat, collision=True):
    obj = UsdGeom.Mesh.Define(stage, path)
    obj.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(mesh.vertices.astype(np.float32)))
    obj.CreateFaceVertexCountsAttr([3]*len(mesh.faces))
    obj.CreateFaceVertexIndicesAttr(mesh.faces.ravel().tolist())
    obj.CreateSubdivisionSchemeAttr("none")
    obj.CreateExtentAttr([Gf.Vec3f(*p) for p in mesh.bounds])
    obj.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(mesh.vertex_normals.astype(np.float32)))
    obj.SetNormalsInterpolation("vertex")
    binding = UsdShade.MaterialBindingAPI.Apply(obj.GetPrim())
    binding.Bind(mat)
    binding.Bind(mat, materialPurpose="physics")
    if collision:
        UsdPhysics.CollisionAPI.Apply(obj.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(obj.GetPrim()).CreateApproximationAttr("convexHull")
        api = PhysxSchema.PhysxCollisionAPI.Apply(obj.GetPrim())
        api.CreateContactOffsetAttr(.00015)
        api.CreateRestOffsetAttr(0.)
    return obj


def rigid(stage, path, position, rotation, mass):
    body = UsdGeom.Xform.Define(stage, path)
    body.AddTranslateOp().Set(Gf.Vec3d(*position))
    body.AddOrientOp().Set(Gf.Quatf(rotation))
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr(mass)
    api = PhysxSchema.PhysxRigidBodyAPI.Apply(body.GetPrim())
    api.CreateEnableCCDAttr(True)
    api.CreateLinearDampingAttr(0.)
    api.CreateAngularDampingAttr(0.)
    return body


def build_lamp(stage, root="/Lamp"):
    assert not stage.GetPrimAtPath(root)
    asset = UsdGeom.Xform.Define(stage, root).GetPrim()
    white = material(stage, root+"/Looks/White", (.84, .87, .9), .28)
    rubber = material(stage, root+"/Looks/Rubber", (.58, .62, .65), .8, .9)
    steel = material(stage, root+"/Looks/Steel", (.28, .3, .32), .25, metallic=.8)
    diffuser = material(stage, root+"/Looks/Diffuser", (.76, .79, .8), .7)
    mark = material(stage, root+"/Looks/Mark", (.27, .3, .33), .7)
    flip = Gf.Rotation(Gf.Vec3d(1, 0, 0), 180).GetQuat()
    elbow = PIVOT + [0, -LENGTH, .006]
    top = elbow + [0, LENGTH, 0]
    poses = ((np.zeros(3), Gf.Quatd(1)), (PIVOT, flip), (elbow, Gf.Quatd(1)), (top, flip))
    for name, (pos, rot), mass in zip(NAMES, poses, MASSES):
        rigid(stage, f"{root}/{name}", pos, rot, mass)
    base = stage.GetPrimAtPath(root+"/base")
    UsdPhysics.ArticulationRootAPI.Apply(base)
    api = PhysxSchema.PhysxArticulationAPI.Apply(base)
    api.CreateEnabledSelfCollisionsAttr(True)
    api.CreateSolverPositionIterationCountAttr(64)
    api.CreateSolverVelocityIterationCountAttr(1)
    PhysxSchema.PhysxContactReportAPI.Apply(stage.GetPrimAtPath(root+"/base")).CreateThresholdAttr(0.)

    def part(body, name, mesh, mat=white, collision=True):
        return mesh_prim(stage, f"{root}/{body}/{name}", mesh, mat, collision)

    part("base", "shell", rounded((.076, .140, .016), .027, (0, 0, .010), .0025))
    for x in (-.025, .025):
        for y in (-.046, .046):
            part("base", f"foot_{x>0}_{y>0}", cylinder(.0045, .002, (x, y, .001)), rubber)
    for i, x in enumerate((-.018, .018)):
        part("base", f"hinge_mount_{i}", rounded((.009, .016, .014), .004, (x, .053, .022)))
        part("lower_arm", f"rail_{i}", rounded((.007, LENGTH+.009, .006), .003,
                                                (x, LENGTH/2, -.003)))
    for body, name, pos, width in (("lower_arm", "lower", (0, 0, 0), .046),
                                   ("lower_arm", "middle", (0, LENGTH, -.006), .046),
                                   ("upper_arm", "upper", (0, LENGTH, 0), .026)):
        part(body, name+"_barrel", cylinder(.006, width, pos, "X"))
        for i, sign in enumerate((-1, 1)):
            p = np.array(pos) + [sign*(width/2+.0003), 0, 0]
            part(body, name+f"_bolt_{i}", cylinder(.003, .0006, p, "X", 6), steel, False)
    part("upper_arm", "beam", rounded((.021, LENGTH+.006, .006), .003, (0, LENGTH/2, 0)))
    part("head", "shell", rounded((.076, .140, .008), .026, (0, .054, -.012), .0015))
    for i, x in enumerate((-.009, .009)):
        part("head", f"hinge_ear_{i}", rounded((.005, .013, .011), .002,
                                               (x, 0, -.007)))
    part("head", "diffuser", rounded((.058, .110, .0006), .021, (0, .054, -.0077)), diffuser, False)
    # A separate, almost flush collider identifies touches at the printed sun icon.
    part("base", "touch_pad", cylinder(.009, .0008, BUTTON-[0, 0, .0004]))
    icon = trimesh.creation.annulus(.0035, .0039, .0001, sections=40)
    icon.apply_translation(BUTTON+[0, 0, .0001])
    part("base", "sun_ring", icon, mark, False)
    for i, angle in enumerate(np.linspace(0, 2*math.pi, 8, endpoint=False)):
        ray = trimesh.creation.box((.00045, .0016, .0001))
        ray.apply_transform(trimesh.transformations.rotation_matrix(-angle, [0, 0, 1]))
        ray.apply_translation(BUTTON+[.0052*math.sin(angle), .0052*math.cos(angle), .0001])
        part("base", f"sun_ray_{i}", ray, mark, False)

    for i, (name, anchor, limits) in enumerate(zip(JOINTS, (PIVOT, elbow, top),
                                                   ((-180, 0), (0, 180), (-180, 0)))):
        joint = UsdPhysics.RevoluteJoint.Define(stage, root+"/Joints/"+name)
        for side, body_index in enumerate((i, i+1)):
            pos, rot = poses[body_index]
            getattr(joint, f"CreateBody{side}Rel")().SetTargets([root+"/"+NAMES[body_index]])
            local = Gf.Rotation(rot.GetInverse()).TransformDir(Gf.Vec3d(*(anchor-pos)))
            getattr(joint, f"CreateLocalPos{side}Attr")(Gf.Vec3f(local))
            getattr(joint, f"CreateLocalRot{side}Attr")(Gf.Quatf(rot.GetInverse()))
        joint.CreateAxisAttr("X")
        joint.CreateLowerLimitAttr(limits[0])
        joint.CreateUpperLimitAttr(limits[1])
        joint.CreateCollisionEnabledAttr(False)
        prim = joint.GetPrim()
        assert prim.ApplyAPI("PhysxJointAxisAPI", "angular")
        for attr, value in (("staticFrictionEffort", FRICTION[i]),
                            ("dynamicFrictionEffort", FRICTION[i]*.8),
                            ("viscousFrictionCoefficient", .002*math.pi/180),
                            ("maxJointVelocity", 120.)):
            prim.CreateAttribute("physxJointAxis:angular:"+attr, Sdf.ValueTypeNames.Float).Set(value)
    light = UsdLux.RectLight.Define(stage, root+"/head/Light")
    light.AddTranslateOp().Set(Gf.Vec3d(0, .054, -.006))
    light.AddRotateXYZOp().Set(Gf.Vec3f(180, 0, 0))
    light.CreateWidthAttr(.052)
    light.CreateHeightAttr(.100)
    light.CreateColorAttr(Gf.Vec3f(1., .91, .77))
    light.CreateIntensityAttr(0.)
    asset.CreateAttribute("lamp:on", Sdf.ValueTypeNames.Bool).Set(False)
    return asset


def export_asset():
    ASSET.parent.mkdir(exist_ok=True)
    stage = Usd.Stage.CreateNew(str(ASSET))
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.)
    stage.SetDefaultPrim(build_lamp(stage))
    stage.GetRootLayer().Save()
    return ASSET
