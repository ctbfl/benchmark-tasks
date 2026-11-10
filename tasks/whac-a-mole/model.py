"""Procedural whac-a-mole device in metres: four spring buttons and a score display."""

from pathlib import Path
import math

import numpy as np
import trimesh
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
from layout import (BASE_SIZE, BUTTON_DAMPING, BUTTON_GAP, BUTTON_HEIGHT, BUTTON_RADIUS,
                    BUTTON_STIFFNESS, BUTTON_TRAVEL, BUTTONS, DISPLAY_CENTER, SEGMENTS)


def material(stage, path, color, metallic=0., roughness=.45, friction=.7, emissive=None):
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    if emissive is not None:
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*emissive))
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    physics = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    physics.CreateStaticFrictionAttr(friction)
    physics.CreateDynamicFrictionAttr(.85*friction)
    physics.CreateRestitutionAttr(0.)
    return mat


def mesh_prim(stage, path, mesh, mat, collision=True):
    obj = UsdGeom.Mesh.Define(stage, path)
    obj.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(mesh.vertices.astype(np.float32)))
    obj.CreateFaceVertexCountsAttr([3]*len(mesh.faces))
    obj.CreateFaceVertexIndicesAttr(mesh.faces.ravel().tolist())
    obj.CreateSubdivisionSchemeAttr("none")
    obj.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(np.repeat(mesh.face_normals, 3, axis=0).astype(np.float32)))
    obj.SetNormalsInterpolation("faceVarying")
    bind = UsdShade.MaterialBindingAPI.Apply(obj.GetPrim())
    bind.Bind(mat)
    bind.Bind(mat, materialPurpose="physics")
    if collision:
        UsdPhysics.CollisionAPI.Apply(obj.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(obj.GetPrim()).CreateApproximationAttr("convexHull")
        api = PhysxSchema.PhysxCollisionAPI.Apply(obj.GetPrim())
        api.CreateContactOffsetAttr(.0005)
        api.CreateRestOffsetAttr(0.)
    return obj


def rounded_box(size, radius, center=(0., 0., 0.)):
    w, d, h = size
    radius = min(radius, w/2, d/2)
    outline = []
    for x, y, start in ((1, 1, 0), (-1, 1, 90), (-1, -1, 180), (1, -1, 270)):
        for angle in np.linspace(start, start+90, 9):
            a = math.radians(angle)
            outline.append((x*(w/2-radius)+radius*math.cos(a), y*(d/2-radius)+radius*math.sin(a)))
    n = len(outline)
    vertices = [(x, y, z) for z in (-h/2, h/2) for x, y in outline]
    vertices += [(0, 0, -h/2), (0, 0, h/2)]
    faces = []
    for i in range(n):
        j = (i+1) % n
        faces += [(2*n, j, i), (2*n+1, n+i, n+j), (i, j, n+j), (i, n+j, n+i)]
    return trimesh.Trimesh(np.asarray(vertices)+center, faces, process=False)


def box(stage, path, size, center, mat, collision=True, radius=.001):
    return mesh_prim(stage, path, rounded_box(size, radius, center), mat, collision)


def cylinder(stage, path, radius, height, center, mat, collision=True):
    geom = trimesh.creation.cylinder(radius=radius, height=height, sections=48)
    geom.apply_translation(center)
    return mesh_prim(stage, path, geom, mat, collision)


def annulus(stage, path, outer, inner, height, center, mat):
    geom = trimesh.creation.annulus(r_min=inner, r_max=outer, height=height, sections=48)
    geom.apply_translation(center)
    return mesh_prim(stage, path, geom, mat, collision=False)


def rigid(stage, path, position, mass):
    obj = UsdGeom.Xform.Define(stage, path)
    obj.AddTranslateOp().Set(Gf.Vec3d(*position))
    UsdPhysics.RigidBodyAPI.Apply(obj.GetPrim())
    UsdPhysics.MassAPI.Apply(obj.GetPrim()).CreateMassAttr(mass)
    return obj


def slider(stage, root, name, child, pivot):
    joint = UsdPhysics.PrismaticJoint.Define(stage, root + "/Joints/" + name)
    joint.CreateBody0Rel().SetTargets([root + "/base"])
    joint.CreateBody1Rel().SetTargets([root + "/" + child])
    joint.CreateAxisAttr("Z")
    joint.CreateLocalPos0Attr(Gf.Vec3f(*pivot))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0.))
    joint.CreateLocalRot0Attr(Gf.Quatf(1.))
    joint.CreateLocalRot1Attr(Gf.Quatf(1.))
    joint.CreateLowerLimitAttr(-BUTTON_TRAVEL)
    joint.CreateUpperLimitAttr(0.)
    joint.CreateCollisionEnabledAttr(False)
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(BUTTON_STIFFNESS)
    drive.CreateDampingAttr(BUTTON_DAMPING)
    drive.CreateTargetPositionAttr(0.)
    drive.CreateMaxForceAttr(20.)
    state = PhysxSchema.JointStateAPI.Apply(joint.GetPrim(), "linear")
    state.CreatePositionAttr(0.)
    state.CreateVelocityAttr(0.)
    return joint


def build_device():
    ASSETS.mkdir(exist_ok=True)
    path = ASSETS / "device.usda"
    if path.exists():
        path.unlink()
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.)
    root = "/device"
    prim = UsdGeom.Xform.Define(stage, root).GetPrim()
    stage.SetDefaultPrim(prim)
    UsdPhysics.ArticulationRootAPI.Apply(prim)
    PhysxSchema.PhysxArticulationAPI.Apply(prim).CreateEnabledSelfCollisionsAttr(False)

    shell = material(stage, root + "/Looks/Shell", (.08, .24, .55), roughness=.5)
    top = material(stage, root + "/Looks/Top", (.90, .90, .86), roughness=.6)
    bezel = material(stage, root + "/Looks/Bezel", (.10, .10, .11), roughness=.4)
    rubber = material(stage, root + "/Looks/Feet", (.03, .03, .03), roughness=.9, friction=1.)
    off = material(stage, root + "/Looks/ButtonOff", (.30, .05, .04), roughness=.35)
    on = material(stage, root + "/Looks/ButtonOn", (1., .85, .25), roughness=.35, emissive=(1., .85, .25))
    screen = material(stage, root + "/Looks/Screen", (.015, .015, .02), roughness=.3)
    segment = material(stage, root + "/Looks/Segment", (1., .25, .10), roughness=.4, emissive=(1., .25, .10))

    w, d, h = BASE_SIZE
    foot = .004
    rigid(stage, root + "/base", (0, 0, 0), 3.0)
    box(stage, root + "/base/shell", (w, d, h-foot-.003), (0, 0, foot+(h-foot-.003)/2), shell, radius=.012)
    box(stage, root + "/base/top", (w-.004, d-.004, .003), (0, 0, h-.0015), top, radius=.010)
    for i, (x, y) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
        box(stage, root + f"/base/foot_{i}", (.025, .025, foot), (x*(w/2-.025), y*(d/2-.025), foot/2),
            rubber, radius=.004)
    for name, (x, y) in BUTTONS.items():
        annulus(stage, root + f"/base/bezel_{name}", BUTTON_RADIUS+.007, BUTTON_RADIUS+.0008, .004,
                (x, y, h+.002), bezel)
    # Score display: an angled housing on the far edge facing the robot.
    sx, sy = DISPLAY_CENTER
    housing = UsdGeom.Xform.Define(stage, root + "/base/display")
    housing.AddTranslateOp().Set(Gf.Vec3d(sx, sy, h))
    box(stage, root + "/base/display/stand", (.040, .090, .030), (0, 0, .015), shell, radius=.006)
    panel = UsdGeom.Xform.Define(stage, root + "/base/display/panel")
    panel.AddTranslateOp().Set(Gf.Vec3d(-.004, 0, .036))
    panel.AddRotateYOp().Set(-40.)
    box(stage, root + "/base/display/panel/frame", (.040, .090, .008), (0, 0, -.004), bezel, False, .004)
    box(stage, root + "/base/display/panel/screen", (.030, .074, .001), (0, 0, .0005), screen, False, .002)
    # Seven-segment digits drawn in the panel's x-y plane; panel -x points to the robot.
    # Digit "up" is +x (away from the robot) and digit "right" is -y.
    k, t = 1.25, .0022     # digit scale and stroke width
    coords = {"a": (.0085*k, 0, t, .0072*k), "g": (0, 0, t, .0072*k), "d": (-.0085*k, 0, t, .0072*k),
              "b": (.0043*k, -.0042*k, .0074*k, t), "c": (-.0043*k, -.0042*k, .0074*k, t),
              "e": (-.0043*k, .0042*k, .0074*k, t), "f": (.0043*k, .0042*k, .0074*k, t)}
    for digit in range(2):
        center = (.5-digit)*.019
        for letter, (x, y, sxs, sys) in coords.items():
            obj = box(stage, root + f"/base/display/panel/digit_{digit}_{letter}",
                      (sxs, sys, .0003), (x, center+y, .0012), segment, False, .0003)
            if digit != 1 or letter not in SEGMENTS["0"]:
                UsdGeom.Imageable(obj).MakeInvisible()

    rest = h + BUTTON_GAP + BUTTON_HEIGHT/2
    for name, (x, y) in BUTTONS.items():
        body = root + "/button_" + name
        rigid(stage, body, (x, y, rest), .02)
        cylinder(stage, body + "/collision", BUTTON_RADIUS, BUTTON_HEIGHT, (0, 0, 0), off)
        UsdGeom.Imageable(stage.GetPrimAtPath(body + "/collision")).CreatePurposeAttr("guide")
        # The visual cap and a skirt that hides the travel gap inside the bezel.
        cylinder(stage, body + "/skirt", BUTTON_RADIUS-.001, BUTTON_GAP+.002,
                 (0, 0, -BUTTON_HEIGHT/2-BUTTON_GAP/2+.001), off, collision=False)
        cylinder(stage, body + "/cap_off", BUTTON_RADIUS, BUTTON_HEIGHT, (0, 0, 0), off, collision=False)
        lit = cylinder(stage, body + "/cap_on", BUTTON_RADIUS, BUTTON_HEIGHT, (0, 0, 0), on, collision=False)
        UsdGeom.Imageable(lit).MakeInvisible()
        slider(stage, root, "press_" + name, "button_" + name, (x, y, rest))
    stage.GetRootLayer().Save()
    print("Built", path, flush=True)


def build_all():
    build_device()
