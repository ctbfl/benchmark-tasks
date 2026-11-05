"""WT0352G-shaped passive tripod. Import after starting Isaac Sim.

Metres, kilograms, Z up. Three hinges, fixed-length short legs and a fixed head.
Dimensions and total mass follow the reference; internal dimensions/friction are
engineering estimates. The asset contains no motors or world constraints.
"""

from pathlib import Path
import math

import numpy as np
import trimesh
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
OPEN_DEG = 32.0
PIVOT_RADIUS = .0078
MASS = .1103
HINGE_FRICTION = (.200, .150)  # N m
HINGE_ARMATURE = .001  # kg m^2; numerical joint-space inertia for grasped passive hinges.


def material(stage, path, color, metallic=0., roughness=.4, friction=.6):
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    p = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    p.CreateStaticFrictionAttr(friction)
    p.CreateDynamicFrictionAttr(.8 * friction)
    p.CreateRestitutionAttr(0.)
    return mat


def lathe(profile, sections=64):
    """Closed surface of revolution, profile = (radius, z) from bottom to top."""
    return trimesh.creation.revolve(np.array([(0, profile[0][1]), *profile,
                                             (0, profile[-1][1])]), sections=sections)


def cylinder(radius, length, center, axis="Z"):
    mesh = trimesh.creation.cylinder(radius, length, sections=48)
    if axis == "Y":
        mesh.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [1, 0, 0]))
    mesh.apply_translation(center)
    return mesh


def mesh_prim(stage, path, mesh, mat, collision=True):
    obj = UsdGeom.Mesh.Define(stage, path)
    obj.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(mesh.vertices.astype(np.float32)))
    obj.CreateFaceVertexCountsAttr([3] * len(mesh.faces))
    obj.CreateFaceVertexIndicesAttr(mesh.faces.ravel().tolist())
    obj.CreateSubdivisionSchemeAttr("none")
    obj.CreateExtentAttr([Gf.Vec3f(*v) for v in mesh.bounds])
    obj.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(mesh.vertex_normals.astype(np.float32)))
    obj.SetNormalsInterpolation("vertex")
    binding = UsdShade.MaterialBindingAPI.Apply(obj.GetPrim())
    binding.Bind(mat)
    binding.Bind(mat, materialPurpose="physics")
    if collision:
        UsdPhysics.CollisionAPI.Apply(obj.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(obj.GetPrim()).CreateApproximationAttr("convexHull")
        p = PhysxSchema.PhysxCollisionAPI.Apply(obj.GetPrim())
        p.CreateContactOffsetAttr(.0001)
        p.CreateRestOffsetAttr(0.)
    return obj


def rigid(stage, path, position, rotation, mass):
    obj = UsdGeom.Xform.Define(stage, path)
    obj.AddTranslateOp().Set(Gf.Vec3d(*position))
    obj.AddOrientOp().Set(Gf.Quatf(rotation))
    UsdPhysics.RigidBodyAPI.Apply(obj.GetPrim())
    UsdPhysics.MassAPI.Apply(obj.GetPrim()).CreateMassAttr(mass)
    p = PhysxSchema.PhysxRigidBodyAPI.Apply(obj.GetPrim())
    p.CreateEnableCCDAttr(True)
    p.CreateLinearDampingAttr(0.)
    p.CreateAngularDampingAttr(0.)
    p.CreateMaxDepenetrationVelocityAttr(.5)
    return obj


def joint(stage, root, name, parent, child, pos0, rot0, value):
    obj = UsdPhysics.RevoluteJoint.Define(stage, f"{root}/Joints/{name}")
    obj.CreateBody0Rel().SetTargets([f"{root}/{parent}"])
    obj.CreateBody1Rel().SetTargets([f"{root}/{child}"])
    obj.CreateLocalPos0Attr(Gf.Vec3f(*pos0))
    obj.CreateLocalPos1Attr(Gf.Vec3f(0))
    obj.CreateLocalRot0Attr(Gf.Quatf(rot0))
    obj.CreateLocalRot1Attr(Gf.Quatf(1))
    obj.CreateAxisAttr("Y")
    obj.CreateLowerLimitAttr(0.)
    obj.CreateUpperLimitAttr(OPEN_DEG)
    obj.CreateCollisionEnabledAttr(False)
    axis = "angular"
    state = PhysxSchema.JointStateAPI.Apply(obj.GetPrim(), axis)
    state.CreatePositionAttr(value)
    state.CreateVelocityAttr(0.)
    assert obj.GetPrim().ApplyAPI("PhysxJointAxisAPI", axis)
    static, dynamic = HINGE_FRICTION
    properties = {"staticFrictionEffort": static, "dynamicFrictionEffort": dynamic,
                  "viscousFrictionCoefficient": .020 * math.pi / 180,
                  "armature": HINGE_ARMATURE,
                  "maxJointVelocity": 180.}
    for key, val in properties.items():
        obj.GetPrim().CreateAttribute(f"physxJointAxis:{axis}:{key}", Sdf.ValueTypeNames.Float).Set(val)
    return obj


def build_tripod(stage, root="/MiniTripod", angle=0.):
    """Build a free 4-link/3-DOF articulation, with its lowest surface at Z=0."""
    assert not stage.GetPrimAtPath(root)
    assert 0 <= angle <= OPEN_DEG
    asset = UsdGeom.Xform.Define(stage, root).GetPrim()
    UsdPhysics.ArticulationRootAPI.Apply(asset)
    api = PhysxSchema.PhysxArticulationAPI.Apply(asset)
    api.CreateEnabledSelfCollisionsAttr(True)
    api.CreateSolverPositionIterationCountAttr(64)
    api.CreateSolverVelocityIterationCountAttr(1)
    metal = material(stage, root + "/Looks/Aluminum", (.62, .65, .69), .9, .28, 2.)
    chrome = material(stage, root + "/Looks/Chrome", (.72, .74, .78), .95, .19)
    black = material(stage, root + "/Looks/BlackTube", (.025, .028, .032), .7, .22, 2.)
    rubber = material(stage, root + "/Looks/Rubber", (.018, .020, .023), 0, .8, .9)
    grip_rubber = material(stage, root + "/Looks/GripRubber", (.018, .020, .023), 0, .8, 2.)
    poses = {"hub": (np.zeros(3), Gf.Quatd(1))}
    bodies = {"hub": rigid(stage, root + "/hub", *poses["hub"], .0383)}
    surfaces = {}

    def part(body, name, mesh, mat, collision=True):
        mesh_prim(stage, f"{root}/{body}/{name}", mesh, mat, collision)
        if collision:
            surfaces.setdefault(body, []).append(mesh.vertices)

    part("hub", "housing", lathe([(.0104,.002),(.011,.003),(.011,.018),(.010,.020)]), metal)
    part("hub", "fixed_head", lathe([(.0087,.020),(.0087,.028),(.008,.029)]), metal)
    part("hub", "stem", cylinder(.003, .008, [0,0,.032]), chrome)
    part("hub", "mount_disk", lathe([(.0125,.0355),(.013,.036),(.013,.0385),(.0125,.039)]), metal)
    part("hub", "mount_pad", cylinder(.0117,.001,[0,0,.0395]), grip_rubber)
    part("hub", "mount_screw", cylinder(.003175,.004,[0,0,.042]), chrome)
    part("hub", "knob_shaft", cylinder(.0028,.010,[0,.014,.011], "Y"), chrome)
    part("hub", "fixed_knob", cylinder(.006,.003,[0,.020,.011], "Y"), black)
    # Fixed head detail is visual only; leg mechanics remain the focus.
    for n in range(64):
        a = n * 2 * math.pi / 64
        part("hub", f"knurl_{n:02}", cylinder(.00023,.0025,[.013*math.cos(a),.013*math.sin(a),.0372]),
             chrome, False)

    for i in range(3):
        phi = -90. + i * 120.
        azimuth = Gf.Rotation(Gf.Vec3d(0,0,1), phi).GetQuat()
        frame = azimuth * Gf.Rotation(Gf.Vec3d(1,0,0), 180).GetQuat()
        rot = frame * Gf.Rotation(Gf.Vec3d(0,1,0), angle).GetQuat()
        pivot = np.array([PIVOT_RADIUS*math.cos(math.radians(phi)),
                          PIVOT_RADIUS*math.sin(math.radians(phi)), 0.])
        name=f"leg_{i}"
        poses[name]=(pivot,rot)
        bodies[name]=rigid(stage,f"{root}/{name}",*poses[name],.024)
        part(name,"pivot_barrel",cylinder(.0038,.010,[0,0,.001],"Y"),chrome)
        part(name,"black_tube",lathe([(.0044,.003),(.0049,.004),(.0049,.080),(.0046,.083)]),black)
        part(name,"upper_collar",cylinder(.0051,.002,[0,0,.005]),chrome)
        part(name,"rubber_foot",lathe([(.0055,.080),(.0063,.081),(.0063,.0845),(.0048,.087)]),rubber)
        joint(stage,root,f"fold_{i}","hub",name,pivot,frame,angle)

    surfaces = {name: np.concatenate(v) for name, v in surfaces.items()}
    minimum = min((np.array([Gf.Rotation(poses[n][1]).TransformDir(Gf.Vec3d(*p))
                            for p in vertices]) + poses[n][0])[:,2].min()
                  for n, vertices in surfaces.items())
    for name, body in bodies.items():
        p, _ = poses[name]
        body.GetOrderedXformOps()[0].Set(Gf.Vec3d(*(p + [0,0,-minimum])))
    asset.SetCustomDataByKey("reference", "Amazon Basics WT0352G / B00M78G2VO")
    asset.SetCustomDataByKey("head", "Fixed; only leg mechanics are simulated")
    asset.SetCustomDataByKey("legs", "Fixed short legs; no telescoping joints")
    return asset, surfaces, float(-minimum)


def export_assets():
    ASSETS.mkdir(exist_ok=True)
    paths = {}
    heights = {}
    for name, angle in (("folded",0.), ("open",OPEN_DEG)):
        path = ASSETS / ("mini_tripod.usda" if name == "folded" else f"mini_tripod_{name}.usda")
        stage = Usd.Stage.CreateNew(str(path))
        UsdGeom.SetStageMetersPerUnit(stage, 1.)
        UsdGeom.SetStageUpAxis(stage, "Z")
        UsdPhysics.SetStageKilogramsPerUnit(stage, 1.)
        prim, surfaces, height = build_tripod(stage, angle=angle)
        stage.SetDefaultPrim(prim)
        stage.GetRootLayer().Save()
        paths[name], heights[name] = path, height
    np.savez_compressed(ASSETS / "collision_vertices.npz", **surfaces)
    return paths, heights
