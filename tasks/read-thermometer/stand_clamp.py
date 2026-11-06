"""Procedural iron stand and mechanically coupled spring thermometer clamp (metres)."""

import json
import math
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

OUT = Path(__file__).resolve().parent / "assets" / "stand_clamp"
ROD = (0.0, 0.090)
TIP = (0.0, -0.120, 0.0)
LENGTH = math.hypot(*TIP[:2])
U = (TIP[0] / LENGTH, TIP[1] / LENGTH, 0)
V = (-U[1], U[0], 0)
YAW = math.degrees(math.atan2(U[1], U[0]))
PIVOT = tuple(TIP[i] - .025 * U[i] for i in range(3))


def _material(stage, path, color, metallic=0, roughness=.35, friction=None):
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    if friction is not None:
        physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        physics.CreateStaticFrictionAttr(friction)
        physics.CreateDynamicFrictionAttr(.8 * friction)
        physics.CreateRestitutionAttr(0)
    return material


def _finish(geom, material, collision=True):
    prim = geom.GetPrim()
    binding = UsdShade.MaterialBindingAPI.Apply(prim)
    binding.Bind(material)
    if collision:
        UsdPhysics.CollisionAPI.Apply(prim)
        if geom.GetPrim().IsA(UsdGeom.Mesh):
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexHull")
        prim.AddAppliedSchema("PhysxCollisionAPI")
        prim.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(.00025)
        prim.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(0)
        if material.GetPrim().HasAPI(UsdPhysics.MaterialAPI):
            binding.Bind(material, materialPurpose="physics")
    return prim


def _cube(stage, path, center, size, material, yaw=0, collision=True):
    shape = UsdGeom.Cube.Define(stage, path)
    shape.CreateSizeAttr(1)
    shape.AddTranslateOp().Set(Gf.Vec3d(*center))
    if yaw:
        shape.AddRotateZOp().Set(yaw)
    shape.AddScaleOp().Set(Gf.Vec3f(*size))
    return _finish(shape, material, collision)


def _cylinder(stage, path, center, radius, length, material, axis="Z", collision=True):
    shape = UsdGeom.Cylinder.Define(stage, path)
    shape.CreateAxisAttr(axis)
    shape.CreateRadiusAttr(radius)
    shape.CreateHeightAttr(length)
    shape.AddTranslateOp().Set(Gf.Vec3d(*center))
    return _finish(shape, material, collision)


def _beam(stage, path, start, end, radius, material, collision=True):
    start, end = Gf.Vec3d(*start), Gf.Vec3d(*end)
    delta = end - start
    shape = UsdGeom.Cylinder.Define(stage, path)
    shape.CreateRadiusAttr(radius)
    shape.CreateHeightAttr(delta.GetLength())
    shape.AddTranslateOp().Set((start + end) / 2)
    shape.AddOrientOp().Set(Gf.Quatf(Gf.Rotation(Gf.Vec3d(0, 0, 1), delta.GetNormalized()).GetQuat()))
    return _finish(shape, material, collision)


def _rounded_base(stage, path, width, depth, z0, z1, radius, material):
    ring = []
    for sx, sy, start in ((1, 1, 0), (-1, 1, 90), (-1, -1, 180), (1, -1, 270)):
        for index in range(9):
            angle = math.radians(start + index * 90 / 8)
            ring.append((sx * (width / 2 - radius) + radius * math.cos(angle),
                         sy * (depth / 2 - radius) + radius * math.sin(angle)))
    count = len(ring)
    vertices = [(x, y, z) for z in (z0, z1) for x, y in ring]
    faces = [list(reversed(range(count))), list(range(count, 2 * count))]
    faces.extend([i, (i + 1) % count, (i + 1) % count + count, i + count] for i in range(count))
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(vertices)
    mesh.CreateFaceVertexCountsAttr([len(face) for face in faces])
    mesh.CreateFaceVertexIndicesAttr([i for face in faces for i in face])
    mesh.CreateSubdivisionSchemeAttr("none")
    return _finish(mesh, material)


def _body(stage, path, mass, center=(0, 0, 0), inertia=(1e-5, 1e-5, 1e-5)):
    prim = UsdGeom.Xform.Define(stage, path).GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(prim)
    props = UsdPhysics.MassAPI.Apply(prim)
    props.CreateMassAttr(mass)
    props.CreateCenterOfMassAttr(Gf.Vec3f(*center))
    props.CreateDiagonalInertiaAttr(Gf.Vec3f(*inertia))
    return prim


def build_stand(stage, path):
    """Create one free rigid stand; local origin is the base's bottom centre."""
    root = _body(stage, path, 2.0, (0, .010, .031), (.014, .009, .016))
    paint = _material(stage, path + "/Looks/paint", (.038, .043, .050), .35, .38, .7)
    steel = _material(stage, path + "/Looks/steel", (.47, .50, .53), .92, .22, .55)
    rubber = _material(stage, path + "/Looks/rubber", (.021, .024, .026), 0, .9, 1.1)
    _rounded_base(stage, path + "/plate", .15, .25, .004, .014, .003, paint)
    for x in (-.055, .055):
        for y in (-.105, .105):
            _cylinder(stage, path + f"/foot_{int(x > 0)}_{int(y > 0)}", (x, y, .002), .011, .004, rubber)
    _cylinder(stage, path + "/rod", (*ROD, (.014 + .44) / 2), .006, .44 - .014, steel)
    _cylinder(stage, path + "/rod_socket", (*ROD, .019), .010, .010, steel)
    _cylinder(stage, path + "/socket_flange", (*ROD, .0155), .013, .003, paint)
    root.SetCustomDataByKey("asset_description", "Free iron stand; threaded upright fixed to heavy base.")
    return {"body": path, "base_size_m": [.15, .25, .014], "rod_center_xy_m": list(ROD),
            "rod_radius_m": .006, "rod_top_m": .44, "mass_kg": 2.0}


def _local(u, v=0, z=0, origin=PIVOT):
    return tuple(origin[i] + U[i] * u + V[i] * v + (z if i == 2 else 0) for i in range(3))


def _joint(stage, path, kind, body0, body1, anchor0, anchor1, axis):
    joint = kind.Define(stage, path)
    joint.CreateBody0Rel().SetTargets([body0])
    joint.CreateBody1Rel().SetTargets([body1])
    joint.CreateLocalPos0Attr(Gf.Vec3f(*anchor0))
    joint.CreateLocalPos1Attr(Gf.Vec3f(*anchor1))
    joint.CreateLocalRot0Attr(Gf.Quatf(1))
    joint.CreateLocalRot1Attr(Gf.Quatf(1))
    joint.CreateAxisAttr(axis)
    joint.CreateCollisionEnabledAttr(False)
    return joint


def build_clamp(stage, path):
    """Free articulation: screw friction grips the stand, spring jaws grip the thermometer."""
    root = UsdGeom.Xform.Define(stage, path).GetPrim()
    chrome = _material(stage, path + "/Looks/chrome", (.52, .55, .57), .95, .2, .75)
    brass = _material(stage, path + "/Looks/brass", (.47, .32, .13), .83, .25, .7)
    black = _material(stage, path + "/Looks/knob", (.018, .020, .022), .05, .4, .8)
    rubber = _material(stage, path + "/Looks/jaw_liner", (.13, .17, .16), 0, .75, 1.25)
    boss = _body(stage, path + "/boss", .125, (0, -.036, 0), (1.2e-4, 8e-5, 1.7e-4))
    UsdPhysics.ArticulationRootAPI.Apply(boss)
    boss.AddAppliedSchema("PhysxArticulationAPI")
    boss.CreateAttribute("physxArticulation:enabledSelfCollisions", Sdf.ValueTypeNames.Bool).Set(True)

    # The left (+Y) side is open for lateral insertion; retain both wall corners.
    _cube(stage, path + "/boss/back", (.0092, 0, 0), (.006, .020, .020), chrome)
    _cube(stage, path + "/boss/side_right", (-.002, -.0095, 0), (.023, .006, .020), chrome)
    _cube(stage, path + "/boss/left_corner_front", (-.01275, .0095, 0), (.0015, .006, .020), chrome)
    _cube(stage, path + "/boss/left_corner_back", (.00785, .0095, 0), (.0033, .006, .020), chrome)
    _cube(stage, path + "/boss/screw_bearing", (-.0155, 0, 0), (.007, .024, .020), chrome)
    _beam(stage, path + "/boss/arm", (0, -.013, 0), PIVOT, .0045, chrome)
    _cylinder(stage, path + "/boss/pivot_pin", PIVOT, .003, .023, brass)

    # Fixed jaw and fixed squeeze handle, with replaceable non-slip liners.
    _cube(stage, path + "/boss/fixed_spine", _local(.017, .0095), (.036, .004, .017), chrome, YAW)
    # A longitudinal V groove seats the round stem at two distinct contact faces.
    # Each plane is tangent to the 4.5 mm stem radius at +/-45 degrees from +V.
    offset = (.0045 + .001) / math.sqrt(2)
    for side, suffix in ((-1, "rear"), (1, "front")):
        _cube(stage, path + "/boss/fixed_pad_" + suffix,
              _local(.025 + side * offset, offset), (.010, .002, .018),
              rubber, YAW - side * 45)
    _cube(stage, path + "/boss/fixed_handle", _local(-.013, .009), (.033, .006, .015), chrome, YAW - 12)

    moving = _body(stage, path + "/moving_jaw", .019, _local(.003, -.008, origin=(0, 0, 0)),
                   (4e-6, 4e-6, 7e-6))
    UsdGeom.Xformable(moving).AddTranslateOp().Set(Gf.Vec3d(*PIVOT))
    _cube(stage, path + "/moving_jaw/spine", _local(.012, -.007, origin=(0, 0, 0)),
          (.029, .004, .017), chrome, YAW)
    _cube(stage, path + "/moving_jaw/pad", _local(.025, -.0055, origin=(0, 0, 0)),
          (.012, .002, .018), rubber, YAW)
    _cube(stage, path + "/moving_jaw/squeeze_handle", _local(-.013, -.009, origin=(0, 0, 0)),
          (.033, .006, .015), chrome, YAW + 12)
    jaw = _joint(stage, path + "/joints/jaw_hinge", UsdPhysics.RevoluteJoint,
                 boss.GetPath(), moving.GetPath(), PIVOT, (0, 0, 0), "Z")
    jaw.CreateLowerLimitAttr(-35)
    jaw.CreateUpperLimitAttr(6)
    spring = UsdPhysics.DriveAPI.Apply(jaw.GetPrim(), "angular")
    spring.CreateTypeAttr("force")
    spring.CreateTargetPositionAttr(4)
    spring.CreateStiffnessAttr(.8 * math.pi / 180)
    spring.CreateDampingAttr(.012 * math.pi / 180)
    spring.CreateMaxForceAttr(.12)

    # A visible torsion coil surrounds the jaw pin; its two legs enter the handles.
    coil = UsdGeom.BasisCurves.Define(stage, path + "/boss/torsion_spring")
    points = []
    for i in range(241):
        a = i / 240 * 8 * math.pi
        points.append((PIVOT[0] + .0044 * math.cos(a), PIVOT[1] + .0044 * math.sin(a), -.008 + .016 * i / 240))
    coil.CreateTypeAttr("linear")
    coil.CreateCurveVertexCountsAttr([len(points)])
    coil.CreatePointsAttr(points)
    coil.CreateWidthsAttr([.00065])
    coil.SetWidthsInterpolation("constant")
    _finish(coil, brass, False)

    knob = _body(stage, path + "/knob", .022, (0, 0, 0), (2.4e-6, 1.7e-6, 1.7e-6))
    # Face the operator at -X; local +X is the outward screw/retraction axis.
    front = Gf.Quatf(0, Gf.Vec3f(0, 0, 1))
    UsdGeom.Xformable(knob).AddTranslateOp().Set(Gf.Vec3d(-.031, 0, 0))
    UsdGeom.Xformable(knob).AddOrientOp().Set(front)
    _cylinder(stage, path + "/knob/disc", (0, 0, 0), .0155, .009, black, "X")
    _cylinder(stage, path + "/knob/hub", (-.0065, 0, 0), .007, .007, black, "X")
    for i in range(24):
        a = i * 2 * math.pi / 24
        _cylinder(stage, path + f"/knob/rib_{i:02}", (0, .015 * math.cos(a), .015 * math.sin(a)),
                  .0013, .010, black, "X", False)
    turn = _joint(stage, path + "/joints/screw_turn", UsdPhysics.RevoluteJoint,
                  boss.GetPath(), knob.GetPath(), (-.031, 0, 0), (0, 0, 0), "X")
    turn.CreateLocalRot0Attr(front)
    turn.CreateLowerLimitAttr(-36)
    turn.CreateUpperLimitAttr(1620)
    turn.GetPrim().AddAppliedSchema("PhysxJointAxisAPI:angular")
    for name, value in (("staticFrictionEffort", .018), ("dynamicFrictionEffort", .014),
                        ("viscousFrictionCoefficient", .0002)):
        turn.GetPrim().CreateAttribute("physxJointAxis:angular:" + name, Sdf.ValueTypeNames.Float).Set(value)

    pad = _body(stage, path + "/screw_pad", .012, (.013, 0, 0), (1e-7, 8e-7, 8e-7))
    UsdGeom.Xformable(pad).AddOrientOp().Set(front)
    _cylinder(stage, path + "/screw_pad/tip", (.0071, 0, 0), .0057, .003, brass, "X")
    _cylinder(stage, path + "/screw_pad/thread_shaft", (.019, 0, 0), .0025, .023, brass, "X", False)
    for i in range(14):
        _cylinder(stage, path + f"/screw_pad/thread_{i:02}", (.010 + i * .00135, 0, 0),
                  .0029, .0004, brass, "X", False)
    slide = _joint(stage, path + "/joints/screw_travel", UsdPhysics.PrismaticJoint,
                   boss.GetPath(), pad.GetPath(), (0, 0, 0), (0, 0, 0), "X")
    slide.CreateLocalRot0Attr(front)
    slide.CreateLowerLimitAttr(-.0002)
    slide.CreateUpperLimitAttr(.009)
    # Articulation mimic supports coaxial rotary-to-linear transmission. A spatial
    # rack-and-pinion joint instead assumes an orthogonal gear/rack arrangement.
    coupling = slide.GetPrim()
    coupling.AddAppliedSchema("PhysxMimicJointAPI:rotX")
    prefix = "physxMimicJoint:rotX:"
    coupling.CreateRelationship(prefix + "referenceJoint").SetTargets([turn.GetPath()])
    coupling.CreateAttribute(prefix + "referenceJointAxis", Sdf.ValueTypeNames.Token).Set("rotX")
    coupling.CreateAttribute(prefix + "gearing", Sdf.ValueTypeNames.Float).Set(-.002 / 360)
    coupling.CreateAttribute(prefix + "offset", Sdf.ValueTypeNames.Float).Set(0)

    info = {"articulation_root": path + "/boss", "root_body": path + "/boss",
            "thermometer_center_m": list(TIP), "jaw_pivot_m": list(PIVOT),
            "thermometer_diameter_m": .009, "jaw_initial_gap_m": .009,
            "fixed_jaw_profile": "90 degree longitudinal V groove with two rubber contact faces",
            "jaw_joint": path + "/joints/jaw_hinge", "jaw_open_range_deg": [-35, 6],
            "jaw_spring_rest_deg": 4, "jaw_spring_stiffness_Nm_per_rad": .8,
            "jaw_spring_damping_Nms_per_rad": .012,
            "knob_joint": path + "/joints/screw_turn", "slider_joint": path + "/joints/screw_travel",
            "knob_center_m": [-.031, 0, 0], "screw_outward_axis": [-1, 0, 0],
            "rod_socket_opening_axis": [0, 1, 0], "rod_socket_opening_width_m": .0182,
            "coupling_api": path + "/joints/screw_travel.physxMimicJoint:rotX", "screw_lead_m_per_turn": .002,
            "screw_travel_limits_m": [-.0002, .009], "screw_static_friction_Nm": .018,
            "actuators": {}, "mass_kg": .178}
    root.SetCustomDataByKey("mechanism", "Native screw coupling and passive torsion spring; no thermometer attachment.")
    return info


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    assets = {}
    for name, build in (("stand", build_stand), ("clamp", build_clamp)):
        stage = Usd.Stage.CreateNew(str(OUT / f"{name}.usda"))
        UsdGeom.SetStageUpAxis(stage, "Z")
        UsdGeom.SetStageMetersPerUnit(stage, 1)
        info = build(stage, "/" + name)
        stage.SetDefaultPrim(stage.GetPrimAtPath("/" + name))
        stage.GetRootLayer().Save()
        assets[name] = info
    (OUT / "metadata.json").write_text(json.dumps(assets, indent=2) + "\n")
    print(json.dumps(assets, indent=2))


if __name__ == "__main__":
    main()
