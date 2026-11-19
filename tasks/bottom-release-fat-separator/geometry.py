"""Procedural USD geometry for the separator, spring actuator, cradle and bowl."""

import math
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from liquid import (BOWL_AREA, BOWL_FLOOR, BOWL_LIQUID_RADIUS, BOWL_RADIUS, DT,
                    FLOOR, OUTLET, RADIUS, STROKE, SUMP_HEIGHT, SUMP_RADIUS,
                    bowl_oil_slick, level, reservoir_radius)

XY = (.411, -.085)
PADDLE = (-.136, 0, .220)
PADDLE_TOP = PADDLE[2] + .006
SPRING = 330.  # N/m
DAMPING = 10.  # N s/m


def material(stage, path, color, roughness=.34, opacity=1., metallic=0.):
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    for name, value in (("roughness", roughness), ("opacity", opacity), ("metallic", metallic)):
        shader.CreateInput(name, Sdf.ValueTypeNames.Float).Set(value)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def finish(shape, mat, collision=False):
    UsdShade.MaterialBindingAPI.Apply(shape.GetPrim()).Bind(mat)
    if collision:
        UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
        collider = PhysxSchema.PhysxCollisionAPI.Apply(shape.GetPrim())
        collider.CreateContactOffsetAttr(.001)
        collider.CreateRestOffsetAttr(0.)
        if shape.GetPrim().IsA(UsdGeom.Mesh):
            UsdPhysics.MeshCollisionAPI.Apply(shape.GetPrim()).CreateApproximationAttr("convexHull")
    return shape


def box(stage, path, pos, dims, mat, collision=True):
    shape = UsdGeom.Cube.Define(stage, path)
    shape.CreateSizeAttr(1.)
    shape.AddTranslateOp().Set(Gf.Vec3d(*pos))
    shape.AddScaleOp().Set(Gf.Vec3f(*dims))
    return finish(shape, mat, collision)


def cylinder(stage, path, pos, radius, length, mat, collision=False):
    shape = UsdGeom.Cylinder.Define(stage, path)
    shape.CreateRadiusAttr(radius)
    shape.CreateHeightAttr(length)
    shape.AddTranslateOp().Set(Gf.Vec3d(*pos))
    return finish(shape, mat, collision)


def mesh(stage, path, points, faces, mat, collision=False):
    shape = UsdGeom.Mesh.Define(stage, path)
    shape.CreatePointsAttr([Gf.Vec3f(*point) for point in points])
    shape.CreateFaceVertexCountsAttr([len(face) for face in faces])
    shape.CreateFaceVertexIndicesAttr([i for face in faces for i in face])
    shape.CreateSubdivisionSchemeAttr("none")
    return finish(shape, mat, collision)


def rings(profile):
    return [(r * math.cos(i * math.tau/96), r * math.sin(i * math.tau/96), z)
            for r, z in profile for i in range(96)]


def lathe(stage, path, profile, mat, *, open_front=False):
    n = 96
    span = [*range(72, 96), *range(24)] if open_front else range(n)
    faces = [(n*j+i, n*j+(i+1)%n, n*k+(i+1)%n, n*k+i)
             for j, k in zip(range(len(profile)), [*range(1,len(profile)), 0])
             for i in span]
    if open_front:
        faces.extend([[n*j+72 for j in range(len(profile))],
                      [n*j+24 for j in reversed(range(len(profile)))]])
    return mesh(stage, path, rings(profile), faces, mat)


def shell_collision(stage, path, bands, mat):
    for j, (lower, upper) in enumerate(zip(bands, bands[1:])):
        for i in range(24):
            points = [(r * math.cos(a), r * math.sin(a), z)
                      for z, inner, outer in (lower, upper)
                      for r in (inner, outer)
                      for a in (i * math.tau/24, (i+1) * math.tau/24)]
            segment = mesh(stage, f"{path}/Band{j}Panel{i}", points,
                           [(0,1,3,2), (4,6,7,5), (0,4,5,1), (2,3,7,6),
                            (0,2,6,4), (1,5,7,3)], mat, True)
            segment.GetVisibilityAttr().Set("invisible")


def build_asset(path):
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdGeom.SetStageUpAxis(stage, "Z")
    root = "/Separator"
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, root).GetPrim())
    looks = root + "/Looks"
    acrylic = material(stage, looks + "/Acrylic", (.76,.91,.94), .13, .17)
    graphite = material(stage, looks + "/Graphite", (.045,.070,.081), .44)
    green = material(stage, looks + "/SageGreen", (.14,.51,.24), .38)
    ceramic = material(stage, looks + "/IvoryCeramic", (.89,.85,.72), .27)
    steel = material(stage, looks + "/Steel", (.48,.55,.57), .26, metallic=.74)
    ink = material(stage, looks + "/Markings", (.055,.11,.12), .7)
    material(stage, looks + "/Gravy", (.31,.094,.035), .27)
    material(stage, looks + "/Fat", (.93,.55,.048), .22)

    # The rear upright and ring cradle leave the front open for the bowl and the stream.
    fixed = root + "/Fixture"
    frame = UsdGeom.Xform.Define(stage, fixed).GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(frame).CreateKinematicEnabledAttr(True)
    box(stage, fixed + "/Base", (.037, 0, .004), (.112,.174,.008), graphite)
    box(stage, fixed + "/Upright", (.079, 0, .098), (.018,.133,.188), graphite)
    for side, y in (("Left", .061), ("Right", -.061)):
        box(stage, fixed + "/" + side + "Rail", (.022, y, .194), (.128,.017,.016), graphite)
    lathe(stage, fixed + "/Seat", [(.049,.188),(.056,.188),(.057,.194),(.054,.196)], graphite, open_front=True)
    lathe(stage, fixed + "/Nozzle", [(.0038,OUTLET),(.009,OUTLET),(.010,.146),
          (.013,.155),(.013,.162),(.0038,.162)], steel)
    shell_collision(stage, fixed + "/NozzleCollider", [(OUTLET,.0038,.009),(.162,.0038,.013)], steel)
    cylinder(stage, fixed + "/ReservoirFloor", (0,0,FLOOR-.002), .022, .004, graphite, True)
    shoulder = FLOOR + SUMP_HEIGHT
    lathe(stage, fixed + "/ClearReservoir", [(SUMP_RADIUS,FLOOR),(.023,FLOOR),(.052,shoulder),
          (.052,.266),(.054,.269),(.053,.273),(RADIUS,.273),(RADIUS,shoulder)], acrylic)
    shell_collision(stage, fixed + "/ReservoirCollider", [(FLOOR,SUMP_RADIUS,.023),
                    (shoulder,RADIUS,.052),(.274,RADIUS,.054)], acrylic)
    lathe(stage, fixed + "/Rim", [(.048,.269),(.054,.269),(.054,.274),(.048,.274)], graphite)
    for i in range(1, 10):
        h = i * .011
        x = -reservoir_radius(h)-.003
        box(stage, f"{fixed}/Graduations/Tick{i}", (x,0,FLOOR+h),
            (.001,.016 if i % 2 == 0 else .009,.001), ink, False)
    h = .0035
    box(stage, fixed + "/InletReference", (-reservoir_radius(h)-.003,0,FLOOR+h),
        (.001,.010,.001), ink, False)
    box(stage, fixed + "/HandleSupport", (-.077,0,.193), (.064,.031,.014), graphite)
    box(stage, fixed + "/FixedGrip", (-.122,0,.187), (.066,.038,.023), graphite)
    box(stage, fixed + "/ValveLinkHousing", (-.060,0,.209), (.021,.030,.042), graphite)
    for y in (-.0195, .0195):
        box(stage, f"{fixed}/GripStripe{int(y > 0)}", (-.123,y,.185), (.050,.001,.005), green, False)

    # Cylindrical, open bowl fixed in the cradle's locating recess.
    lathe(stage, fixed + "/Bowl/Wall", [(BOWL_RADIUS,BOWL_FLOOR),(.072,.006),(.076,.011),
          (.079,.040),(.078,.076),(.075,.081),(BOWL_RADIUS,.080)], ceramic)
    shell_collision(stage, fixed+"/Bowl/WallCollider", [(BOWL_FLOOR,BOWL_RADIUS,.075),
                   (.080,BOWL_RADIUS,.079)], ceramic)
    cylinder(stage, fixed + "/Bowl/Floor", (0,0,.006), BOWL_RADIUS, .008, ceramic, True)
    lathe(stage, fixed + "/Bowl/RimGlaze", [(.076,.070),(.0785,.070),(.0785,.073),(.076,.073)], green)

    # A prismatic squeeze handle. Both reaction pads are physical geometry.
    moving = root + "/Paddle"
    body = UsdGeom.Xform.Define(stage, moving)
    body.AddTranslateOp().Set(Gf.Vec3d(*PADDLE))
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    PhysxSchema.PhysxRigidBodyAPI.Apply(body.GetPrim()).CreateDisableGravityAttr(True)
    UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr(.075)
    box(stage, moving + "/SqueezePad", (0,0,0), (.046,.038,.012), green)
    box(stage, moving + "/Linkage", (.040,0,-.003), (.047,.020,.010), steel)
    for i in range(5):
        box(stage, f"{moving}/Tread{i}", (-.015+i*.007,0,.0063), (.0015,.030,.0006), graphite, False)
    joint = UsdPhysics.PrismaticJoint.Define(stage, root + "/ValveTravel")
    joint.CreateBody0Rel().SetTargets([fixed])
    joint.CreateBody1Rel().SetTargets([moving])
    joint.CreateCollisionEnabledAttr(False)
    joint.CreateLocalPos0Attr(Gf.Vec3f(*PADDLE))
    joint.CreateAxisAttr("Z")
    joint.CreateLowerLimitAttr(-STROKE)
    joint.CreateUpperLimitAttr(0.)
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
    drive.CreateTypeAttr("force")
    drive.CreateTargetPositionAttr(.002)
    drive.CreateStiffnessAttr(SPRING)
    drive.CreateDampingAttr(DAMPING)
    stage.GetRootLayer().Save()


class LiquidVisual:
    def __init__(self, stage, root):
        self.reservoir = []
        for name in ("Gravy", "Fat"):
            mat = UsdShade.Material(stage.GetPrimAtPath(f"{root}/Looks/{name}"))
            shape = lathe(stage, f"{root}/Liquids/{name}", self.profile(0., .01), mat)
            self.reservoir.append(shape)
        self.levels = []
        for name, mat_name, radius in (
            ("BowlGravy", "Gravy", BOWL_LIQUID_RADIUS), ("BowlFat", "Fat", BOWL_LIQUID_RADIUS),
            ("GravyStream", "Gravy", 1.), ("FatStream", "Fat", 1.),
        ):
            mat = UsdShade.Material(stage.GetPrimAtPath(f"{root}/Looks/{mat_name}"))
            shape = cylinder(stage, f"{root}/Liquids/{name}", (0,0,0), radius, 1., mat)
            self.levels.append((shape, shape.GetOrderedXformOps()[0], shape.AddScaleOp()))

    @staticmethod
    def profile(lower, upper):
        # Preserve the cylindrical shoulder and close both horizontal faces.
        transition = max(lower, min(SUMP_HEIGHT, upper))
        return [(0., FLOOR+lower), *((reservoir_radius(h)-.00035, FLOOR+h)
                for h in (lower, transition, upper)), (0., FLOOR+upper)]

    def update(self, drain):
        gravy = level(drain.gravy)
        surface = level(drain.gravy + drain.fat)
        for shape, lower, upper in zip(self.reservoir, (0., gravy), (gravy, surface), strict=True):
            shape.GetVisibilityAttr().Set("inherited" if upper-lower > 1e-6 else "invisible")
            shape.GetPointsAttr().Set([Gf.Vec3f(*p) for p in rings(self.profile(lower, upper))])
        bowl = drain.bowl_gravy * 1e-6 / BOWL_AREA
        bowl_fat = drain.bowl_fat * 1e-6 / BOWL_AREA
        slick_height, slick_scale = bowl_oil_slick(drain.bowl_fat)
        layers = [(BOWL_FLOOR, bowl, 1.), (BOWL_FLOOR+bowl, slick_height, slick_scale)]
        top = bottom = OUTLET
        amount_g = amount_f = 0.
        if drain.tail:
            drops = []
            for tick, g, f, _, v0 in drain.tail:
                age = (drain.ticks - tick) * DT
                drops.append(OUTLET - v0 * age - 4.905 * age**2)
                amount_g += g
                amount_f += f
            top = min(OUTLET, drops[-1] + .004)
            bottom = max(BOWL_FLOOR + bowl + bowl_fat, drops[0] - .004)
        length = top - bottom
        packed_ml = amount_g + amount_f
        radius = math.sqrt(packed_ml * 1e-6 / (math.pi * length)) if length > 0 else 0.
        layers += [(bottom, length if amount_g >= amount_f else 0., radius),
                   (bottom, length if amount_f > amount_g else 0., radius)]
        for (shape, translate, scale), (z, height, r) in zip(self.levels, layers, strict=True):
            shape.GetVisibilityAttr().Set("inherited" if height > 1e-6 else "invisible")
            translate.Set(Gf.Vec3d(0,0,z+height/2))
            scale.Set(Gf.Vec3f(r,r,max(height, 1e-6)))
