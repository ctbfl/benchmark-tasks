"""Procedural USD dispenser, scale and cup. Import after launching Isaac Sim."""

import math

from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics, UsdShade

from config import (
    CUP_FLOOR_Z, CUP_HEIGHT, CUP_RADIUS, DISPLAY_Y, HINGE_DAMPING,
    JAR_HEIGHT, JAR_RADIUS, LEVER_INERTIA, LEVER_LENGTH, MAX_ANGLE,
    NOZZLE, PIVOT, RESERVOIR_FILL_HEIGHT, RESERVOIR_FLOOR_Z, RESERVOIR_RADIUS,
    SCALE_CENTER_Y, SCALE_TOP_Z, STAND_HEIGHT, TEA_COLOR,
)
from surfaces import glass, lathe, panel_texture, rounded_box, textured, tube


def material(stage, path, color, metallic=0.0, opacity=1.0, emission=False, roughness=0.28):
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    for name, value in (("diffuseColor", color), ("emissiveColor", color if emission else (0, 0, 0))):
        shader.CreateInput(name, Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*value))
    for name, value in (("roughness", roughness), ("metallic", metallic), ("opacity", opacity)):
        shader.CreateInput(name, Sdf.ValueTypeNames.Float).Set(value)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def finish(shape, mat, collision=False):
    UsdShade.MaterialBindingAPI.Apply(shape.GetPrim()).Bind(mat)
    if collision:
        UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
        physx = PhysxSchema.PhysxCollisionAPI.Apply(shape.GetPrim())
        physx.CreateContactOffsetAttr(0.0005)
        physx.CreateRestOffsetAttr(0.0)
        if shape.GetPrim().IsA(UsdGeom.Mesh):
            UsdPhysics.MeshCollisionAPI.Apply(shape.GetPrim()).CreateApproximationAttr("convexHull")
    return shape


def box(stage, path, pos, size, mat, collision=False):
    shape = UsdGeom.Cube.Define(stage, path)
    shape.CreateSizeAttr(1)
    shape.AddTranslateOp().Set(Gf.Vec3d(*pos))
    shape.AddScaleOp().Set(Gf.Vec3f(*size))
    return finish(shape, mat, collision)


def cylinder(stage, path, pos, radius, height, mat, collision=False, axis="Z"):
    shape = UsdGeom.Cylinder.Define(stage, path)
    shape.CreateRadiusAttr(radius)
    shape.CreateHeightAttr(height)
    shape.CreateAxisAttr(axis)
    shape.AddTranslateOp().Set(Gf.Vec3d(*pos))
    return finish(shape, mat, collision)


def rigid(stage, path, position, mass, inertia):
    body = UsdGeom.Xform.Define(stage, path)
    body.AddTranslateOp().Set(Gf.Vec3d(*position))
    body.AddOrientOp().Set(Gf.Quatf(1))
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    props = UsdPhysics.MassAPI.Apply(body.GetPrim())
    props.CreateMassAttr(mass)
    props.CreateCenterOfMassAttr(Gf.Vec3f(0))
    props.CreateDiagonalInertiaAttr(Gf.Vec3f(*inertia))
    props.CreatePrincipalAxesAttr(Gf.Quatf(1))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(body.GetPrim())
    px.CreateAngularDampingAttr(0.0)
    return body


def build_dispenser(stage, root="/Dispenser"):
    asset = UsdGeom.Xform.Define(stage, root).GetPrim()
    UsdPhysics.ArticulationRootAPI.Apply(asset)
    art = PhysxSchema.PhysxArticulationAPI.Apply(asset)
    art.CreateEnabledSelfCollisionsAttr(False)
    art.CreateSolverPositionIterationCountAttr(32)
    art.CreateSolverVelocityIterationCountAttr(8)
    clear = glass(stage, root + "/Looks/FrostedPlastic", roughness=0.24)
    gold = material(stage, root + "/Looks/Brass", (0.68, 0.49, 0.22), metallic=0.90, roughness=0.22)
    dark = material(stage, root + "/Looks/Gasket", (0.012, 0.015, 0.018))
    wood = textured(stage, root + "/Looks/Acacia", "wood.jpg")
    rigid(stage, root + "/Body", (0, 0, 0), 8.0, (0.15, 0.15, 0.08))
    r, h = JAR_RADIUS, JAR_HEIGHT-0.010
    lathe(stage, root + "/Body/FlutedJar",
          [(r-0.007, 0.003), (r-0.001, 0.008), (r, 0.016), (r, h-0.006),
           (r-0.001, h), (r-0.003, h), (r-0.003, 0.014), (r-0.008, 0.007)],
          clear, (0, 0, STAND_HEIGHT), ribs=160, count=1280)
    cylinder(stage, root + "/Body/Bottom", (0, 0, STAND_HEIGHT+0.005), r-0.006, 0.006, clear)
    collider = cylinder(stage, root + "/Body/JarCollision", (0, 0, STAND_HEIGHT+h/2), r, h, clear, True)
    collider.GetVisibilityAttr().Set("invisible")
    water_mat = material(stage, root + "/Looks/Tea", TEA_COLOR, roughness=0.12)
    cylinder(stage, root + "/Body/StoredWater", (0, 0, RESERVOIR_FLOOR_Z+RESERVOIR_FILL_HEIGHT/2),
             RESERVOIR_RADIUS, RESERVOIR_FILL_HEIGHT, water_mat)
    lid_z = STAND_HEIGHT+JAR_HEIGHT
    lathe(stage, root + "/Body/LidRim", [(0.097, -0.011), (0.100, -0.010), (0.100, -0.002),
          (0.098, 0), (0.084, 0), (0.083, -0.003), (0.083, -0.008)], wood, (0,0,lid_z))
    lathe(stage, root + "/Body/LidInset", [(0.0001,-0.010), (0.084,-0.010), (0.084,-0.007), (0.0001,-0.007)], wood, (0,0,lid_z))
    rounded_box(stage, root + "/Body/LidHandle", (0,0,lid_z-0.004), (0.171,0.024,0.008), 0.009, wood)
    cylinder(stage, root + "/Body/TapGasket", (0,-r-0.001,STAND_HEIGHT+0.050), 0.017,0.003,dark,True,"Y")
    cylinder(stage, root + "/Body/TapWasher", (0,-r-0.004,STAND_HEIGHT+0.050), 0.016,0.005,gold,True,"Y")
    centers = [(0,-r-0.004,STAND_HEIGHT+0.050),(0,-0.126,STAND_HEIGHT+0.050),
               (0,-0.137,STAND_HEIGHT+0.048),(0,-0.146,STAND_HEIGHT+0.042),NOZZLE]
    finish(tube(stage, root + "/Body/Spout", centers, 0.008, gold), gold, True)
    cylinder(stage, root + "/Body/Outlet", (0,NOZZLE[1],NOZZLE[2]+0.0003),0.0057,0.0008,dark)
    cylinder(stage, root + "/Body/ValveStem", (0,PIVOT[1],PIVOT[2]-0.012),0.0065,0.026,gold,True)
    rigid(stage, root + "/Lever", PIVOT, 0.035, (LEVER_INERTIA, LEVER_INERTIA, LEVER_INERTIA))
    cylinder(stage, root + "/Lever/Axle", (0,0,0.003),0.007,0.008,gold,True)
    grip = rounded_box(stage, root + "/Lever/Grip", (0,-LEVER_LENGTH*0.60,0.005),
                       (0.012,LEVER_LENGTH*0.95,0.005),0.005,gold)
    finish(grip, gold, True)
    cylinder(stage, root + "/Lever/Screw", (0,0,0.0072),0.0023,0.0006,dark)
    joint = UsdPhysics.RevoluteJoint.Define(stage, root + "/Hinge")
    joint.CreateBody0Rel().SetTargets([root + "/Body"])
    joint.CreateBody1Rel().SetTargets([root + "/Lever"])
    joint.CreateLocalPos0Attr(Gf.Vec3f(*PIVOT))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0))
    joint.CreateAxisAttr("Z")
    joint.CreateLowerLimitAttr(-math.degrees(MAX_ANGLE))
    joint.CreateUpperLimitAttr(math.degrees(MAX_ANGLE))
    joint.CreateCollisionEnabledAttr(False)
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
    drive.CreateTypeAttr("force")
    drive.CreateTargetPositionAttr(0.0)
    drive.CreateTargetVelocityAttr(0.0)
    drive.CreateStiffnessAttr(0.0)
    drive.CreateDampingAttr(HINGE_DAMPING * math.pi / 180)
    clamp = UsdPhysics.FixedJoint.Define(stage, root + "/Mount")
    clamp.CreateBody1Rel().SetTargets([root + "/Body"])
    return asset


def build_stand(stage, root="/Stand"):
    asset = UsdGeom.Xform.Define(stage, root).GetPrim()
    wood = textured(stage, root + "/Looks/Wood", "wood.jpg", roughness=0.48)
    top = rounded_box(stage, root + "/Top", (0,0,STAND_HEIGHT-0.009), (0.230,0.230,0.018),0.008,wood)
    finish(top, wood, True)
    for i, (x,y) in enumerate(((-0.093,-0.093),(-0.093,0.093),(0.093,-0.093),(0.093,0.093))):
        leg = rounded_box(stage, f"{root}/Leg{i}", (x,y,(STAND_HEIGHT-0.018)/2),
                          (0.025,0.025,STAND_HEIGHT-0.018),0.002,wood)
        finish(leg, wood, True)
    return asset


SEGMENTS = ("abcedf", "bc", "abged", "abgcd", "fgbc", "afgcd", "afgecd", "abc", "abcdefg", "abfgcd")


def build_scale(stage, root="/Scale"):
    asset = UsdGeom.Xform.Define(stage, root).GetPrim()
    panel_texture()
    dark = material(stage, root + "/Looks/Body", (0.009,0.011,0.014))
    metal = textured(stage, root + "/Looks/Plate", "scale-panel.png", metallic=0.75, roughness=0.37)
    black = material(stage, root + "/Looks/Screen", (0.004,0.006,0.008), roughness=0.55)
    lit = material(stage, root + "/Looks/Digits", (0.76,0.86,0.84), emission=True)
    material(stage, root + "/Looks/UnlitDigits", (0.0,0.0,0.0), roughness=1.0)
    unlit_shader = UsdShade.Shader(stage.GetPrimAtPath(root + "/Looks/UnlitDigits/Shader"))
    unlit_shader.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(1.0)
    body = rounded_box(stage, root + "/Body", (0,SCALE_CENTER_Y,0.009), (0.145,0.186,0.014),0.022,dark)
    finish(body,dark,True)
    rounded_box(stage,root + "/Plate",(0,SCALE_CENTER_Y,SCALE_TOP_Z-0.0015),(0.143,0.184,0.003),0.021,metal,bevel=0.0004)
    box(stage,root + "/Collision",(0,SCALE_CENTER_Y,0.012),(0.133,0.174,0.010),dark,True).GetVisibilityAttr().Set("invisible")
    rounded_box(stage,root + "/Screen",(0,DISPLAY_Y,SCALE_TOP_Z+0.0002),(0.044,0.020,0.0006),0.002,black,bevel=0.0001)
    # Seven-segment geometry is visible to any RGB camera, independent of HUDs.
    # 16 mm tall glyphs fill the 20 mm screen; bars meet without overlapping.
    coords = {"a": (0, 0.007, 0.0054, 0.002), "g": (0, 0, 0.0054, 0.002),
              "d": (0, -0.007, 0.0054, 0.002), "b": (0.0027, 0.0035, 0.0018, 0.005),
              "c": (0.0027, -0.0035, 0.0018, 0.005), "f": (-0.0027, 0.0035, 0.0018, 0.005),
              "e": (-0.0027, -0.0035, 0.0018, 0.005)}
    digit_frame = UsdGeom.Xform.Define(stage, root + "/Display")
    digit_frame.AddTranslateOp().Set(Gf.Vec3d(0,DISPLAY_Y,SCALE_TOP_Z+0.0007))
    digit_frame.AddRotateXOp().Set(-90.0)
    for i, center_x in enumerate((-0.0158, -0.0071, 0.0016, 0.0122)):
        for name, (x, z, w, h) in coords.items():
            box(stage, f"{root}/Display/Digits/D{i}/{name}", (center_x + x, 0, z),
                (w, 0.0003, h), lit)
    box(stage, root + "/Display/Decimal", (0.0074,0,-0.0068), (0.0022,0.0003,0.0022), lit)
    for name in "abfgcd":
        x, z, w, h = coords[name]
        box(stage, root + "/Display/Grams/" + name, (0.019+x*0.45,0,0.004+z*0.45),
            (w * 0.45, 0.0003, h * 0.45), lit)
    display_reading(stage, 0.0, root)
    return asset


def display_reading(stage, grams, root="/Scale"):
    digits = f"{grams:05.1f}".replace(".", "")
    if len(digits) != 4:
        raise ValueError("Scale supports 0.0 to 999.9 g")
    lit = UsdShade.Material(stage.GetPrimAtPath(root + "/Looks/Digits"))
    unlit = UsdShade.Material(stage.GetPrimAtPath(root + "/Looks/UnlitDigits"))
    for i, digit in enumerate(digits):
        for name in "abcdefg":
            shape = UsdGeom.Imageable(stage.GetPrimAtPath(f"{root}/Display/Digits/D{i}/{name}"))
            shape.GetVisibilityAttr().Set("inherited")
            UsdShade.MaterialBindingAPI(shape.GetPrim()).Bind(lit if name in SEGMENTS[int(digit)] else unlit)


def build_cup(stage, root="/Cup"):
    asset = UsdGeom.Xform.Define(stage, root).GetPrim()
    clear = glass(stage, root + "/Looks/Glass", ior=1.5)
    cylinder(stage, root + "/Bottom", (0, NOZZLE[1], CUP_FLOOR_Z - 0.0035), CUP_RADIUS + 0.003,
             0.007, clear, True)
    # Hollow ring mesh, not a solid cylinder collider closing the cup opening.
    n = 64
    vertices = []
    for z, radius in ((CUP_FLOOR_Z, CUP_RADIUS), (CUP_FLOOR_Z + CUP_HEIGHT, CUP_RADIUS),
                      (CUP_FLOOR_Z, CUP_RADIUS + 0.003), (CUP_FLOOR_Z + CUP_HEIGHT, CUP_RADIUS + 0.003)):
        vertices += [Gf.Vec3f(radius * math.cos(i * 2 * math.pi/n), NOZZLE[1] + radius * math.sin(i * 2 * math.pi/n), z) for i in range(n)]
    indices = []
    for i in range(n):
        j = (i + 1) % n
        indices += [i, n+i, n+j, j, 2*n+i, 2*n+j, 3*n+j, 3*n+i,
                    n+i, 3*n+i, 3*n+j, n+j, i, j, 2*n+j, 2*n+i]
    mesh = UsdGeom.Mesh.Define(stage, root + "/Wall")
    mesh.CreatePointsAttr(vertices)
    mesh.CreateFaceVertexCountsAttr([4] * (4*n))
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    finish(mesh, clear, True)
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("none")
    lathe(stage,root + "/Rim",[(CUP_RADIUS, -0.001),(CUP_RADIUS+0.003,-0.001),
          (CUP_RADIUS+0.003,0.001),(CUP_RADIUS,0.001)],clear,(0,NOZZLE[1],CUP_FLOOR_Z+CUP_HEIGHT))
    return asset


class WaterVisual:
    def __init__(self, stage, root=""):
        self.stage = stage
        self.root = root
        mat = material(stage, root + "/Water/Looks/Tea", TEA_COLOR, roughness=0.12)
        self.reservoir = UsdGeom.Cylinder(stage.GetPrimAtPath(root + "/Dispenser/Body/StoredWater"))
        self.pool = cylinder(stage, root + "/Water/Pool", (0, NOZZLE[1], CUP_FLOOR_Z), CUP_RADIUS, 0.0001, mat)
        self.stream = UsdGeom.PointInstancer.Define(stage, root + "/Water/Stream")
        cylinder(stage, root + "/Water/Stream/Prototype", (0, 0, 0), 1.0, 1.0, mat)
        self.stream.CreatePrototypesRel().SetTargets([root + "/Water/Stream/Prototype"])
        self.stream.CreateProtoIndicesAttr([])
        self.stream.CreatePositionsAttr([])
        self.stream.CreateScalesAttr([])
        self.previous_reading = None

    def update(self, task):
        stored_height = task.reservoir_surface_z - RESERVOIR_FLOOR_Z
        self.reservoir.GetVisibilityAttr().Set("inherited" if stored_height > 0 else "invisible")
        self.reservoir.GetHeightAttr().Set(max(stored_height, 1e-6))
        self.reservoir.GetOrderedXformOps()[0].Set(Gf.Vec3d(0, 0, RESERVOIR_FLOOR_Z + stored_height / 2))
        # Consecutive landed layers share one cylinder, avoiding internal faces.
        height = task.surface_z - CUP_FLOOR_Z
        self.pool.GetVisibilityAttr().Set("inherited" if height > 0 else "invisible")
        self.pool.GetHeightAttr().Set(max(height, 1e-6))
        self.pool.GetOrderedXformOps()[0].Set(Gf.Vec3d(0, NOZZLE[1], CUP_FLOOR_Z + height / 2))
        cylinders = task.cylinders()
        self.stream.GetProtoIndicesAttr().Set([0] * len(cylinders))
        self.stream.GetPositionsAttr().Set([Gf.Vec3f(NOZZLE[0], NOZZLE[1], z) for z, _, _ in cylinders])
        self.stream.GetScalesAttr().Set([Gf.Vec3f(r, r, length) for _, r, length in cylinders])
        if task.reading_g != self.previous_reading:
            display_reading(self.stage, task.reading_g, self.root + "/Scale")
            self.previous_reading = task.reading_g
