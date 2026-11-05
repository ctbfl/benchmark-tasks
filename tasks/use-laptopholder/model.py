"""Passive, folding laptop stand. Metres, kilograms, radians in Python."""

from pathlib import Path

import numpy as np
import trimesh
from scipy.optimize import brentq
from scipy.spatial.transform import Rotation
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

ROOT = Path(__file__).resolve().parent
LENGTH = 0.249  # Straight base; the curved nose extends its front.
RAIL_WIDTH = 0.025
OPEN_WIDTH = 0.185
CLOSED_WIDTH = 0.053
HINGE_Z = 0.022
DECK_LENGTH = 0.235
PROP_PIVOT = np.array([0.135, 0, -0.005])
PROP_LENGTH = 0.100
PIN_RADIUS = 0.0025
SEAT_Z = 0.004 + PIN_RADIUS
SLOTS = np.linspace(0.106, 0.202, 7)
SLOT_LENGTH = 0.010
SLOT_WIDTH = 0.008
CROSS_LENGTH = 0.200
CROSS_ANCHOR = 0.215
INNER_Y = 0.008
SLIDER_LIMITS = tuple(CROSS_ANCHOR - np.sqrt(CROSS_LENGTH**2 - (w-RAIL_WIDTH-2*INNER_Y)**2)
                      for w in (CLOSED_WIDTH, OPEN_WIDTH))
NAMES = tuple(f"{part}_{side}" for part in ("base", "deck", "prop", "cross", "slider") for side in ("L", "R"))


def ry(angle):
    return Rotation.from_rotvec([0, angle, 0]).as_matrix()


def rz(angle):
    return Rotation.from_rotvec([0, 0, angle]).as_matrix()


def quat(matrix):
    x, y, z, w = Rotation.from_matrix(matrix).as_quat()
    return Gf.Quatf(float(w), Gf.Vec3f(float(x), float(y), float(z)))


def gear_angle(gear):
    assert 1 <= gear <= 7
    seat = np.array([SLOTS[gear - 1], 0, SEAT_Z])
    return brentq(lambda a: np.linalg.norm(np.array([0, 0, HINGE_Z]) + ry(-a) @ PROP_PIVOT - seat) - PROP_LENGTH, 0, 1.2)


def configuration(width=OPEN_WIDTH, gear=4, angle=None):
    """Geometrically closed poses; gear=0 folds the two support rods into the decks."""
    assert CLOSED_WIDTH <= width <= OPEN_WIDTH
    theta = (gear_angle(gear) if gear else 0.0) if angle is None else angle
    deck_r = ry(-theta)
    spacing = width - RAIL_WIDTH
    cross_span = spacing - 2 * INNER_Y
    delta = np.sqrt(CROSS_LENGTH**2 - cross_span**2)
    slide = CROSS_ANCHOR - delta
    poses = {}
    for side, sign in (("L", -1), ("R", 1)):
        y = sign * spacing / 2
        poses[f"base_{side}"] = (np.array([0, y, 0.0]), np.eye(3))
        p = np.array([0, y, HINGE_Z])
        poses[f"deck_{side}"] = (p, deck_r)
        prop_p = p + deck_r @ PROP_PIVOT
        if gear:
            foot = np.array([SLOTS[gear - 1], y, SEAT_Z])
            phi = np.arctan2(prop_p[2] - foot[2], foot[0] - prop_p[0])
            prop_r = ry(phi)
        else:
            prop_r = deck_r
        poses[f"prop_{side}"] = (prop_p, prop_r)
        # Each rail has one fixed scissor pivot and one longitudinal slider.
        layer = -0.009 if side == "L" else -0.011
        anchor = np.array([CROSS_ANCHOR, -sign * INNER_Y, layer])
        beta = np.arctan2(-sign * cross_span, -delta)
        poses[f"cross_{side}"] = (p + deck_r @ anchor, deck_r @ rz(beta))
        slider_layer = -0.011 if side == "L" else -0.009
        poses[f"slider_{side}"] = (p + deck_r @ np.array([slide, -sign * INNER_Y, slider_layer]), deck_r)
    return poses


def block(size, pos, radius=0.0, rotation=None):
    if radius:
        x, y, z = np.array(size) / 2
        r = min(radius, x, y)
        points = []
        for (cx, cy), start in zip(((x-r, y-r), (-x+r, y-r), (-x+r, -y+r), (x-r, -y+r)), (0, 90, 180, 270)):
            for a in np.deg2rad(np.linspace(start, start + 90, 5)):
                points.extend(((cx+r*np.cos(a), cy+r*np.sin(a), -z), (cx+r*np.cos(a), cy+r*np.sin(a), z)))
        mesh = trimesh.convex.convex_hull(np.array(points))
    else:
        mesh = trimesh.creation.box(extents=size)
    if rotation is not None:
        mesh.vertices = mesh.vertices @ rotation.T
    mesh.vertices += pos
    return mesh


def pin(radius, length, pos, axis="Y"):
    mesh = trimesh.creation.cylinder(radius=radius, height=length, sections=24)
    if axis == "Y":
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    mesh.vertices += pos
    return mesh


def ski_tip(width, inner_radius, outer_radius):
    """Continuous bent strip and convex cells, preserving its concave top."""
    angles = np.linspace(0, np.deg2rad(65), 17)
    sections = np.array([
        [(-.012-r*np.sin(a), y, .031-r*np.cos(a))
         for r,y in ((inner_radius,-width/2),(inner_radius,width/2),
                     (outer_radius,width/2),(outer_radius,-width/2))]
        for a in angles])
    faces = [(0,2,1),(0,3,2)]
    for i in range(len(angles)-1):
        for j in range(4):
            a,b,c,d = 4*i+j,4*i+(j+1)%4,4*(i+1)+(j+1)%4,4*(i+1)+j
            faces.extend(((a,b,c),(a,c,d)))
    k = 4*(len(angles)-1)
    faces.extend(((k,k+1,k+2),(k,k+2,k+3)))
    mesh = trimesh.Trimesh(vertices=sections.reshape(-1,3),faces=faces)
    mesh.fix_normals()
    cells = [trimesh.convex.convex_hull(sections[i:i+2].reshape(-1,3)) for i in range(len(angles)-1)]
    return mesh, cells


def material(stage, path, color, friction, metallic=0.0):
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(friction[0])
    api.CreateDynamicFrictionAttr(friction[1])
    api.CreateRestitutionAttr(0.0)
    return mat


def shape(stage, path, mesh, mat, collide=True):
    geom = UsdGeom.Mesh.Define(stage, path)
    geom.CreatePointsAttr(mesh.vertices.astype(np.float32).tolist())
    geom.CreateFaceVertexCountsAttr([3] * len(mesh.faces))
    geom.CreateFaceVertexIndicesAttr(mesh.faces.reshape(-1).tolist())
    geom.CreateSubdivisionSchemeAttr("none")
    UsdShade.MaterialBindingAPI.Apply(geom.GetPrim()).Bind(mat)
    if collide:
        UsdPhysics.CollisionAPI.Apply(geom.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(geom.GetPrim()).CreateApproximationAttr("convexHull")
        contact = PhysxSchema.PhysxCollisionAPI.Apply(geom.GetPrim())
        contact.CreateContactOffsetAttr(0.0003)
        contact.CreateRestOffsetAttr(0.0)
        UsdShade.MaterialBindingAPI(geom.GetPrim()).Bind(mat, materialPurpose="physics")
    return geom.GetPrim()


def body(stage, path, position, rotation, mass, parts):
    prim = UsdGeom.Xform.Define(stage, path).GetPrim()
    xf = UsdGeom.Xformable(prim)
    xf.AddTranslateOp().Set(Gf.Vec3d(*position))
    xf.AddOrientOp().Set(quat(rotation))
    UsdPhysics.RigidBodyAPI.Apply(prim)
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    px.CreateSolverPositionIterationCountAttr(64)
    px.CreateSolverVelocityIterationCountAttr(4)
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateLinearDampingAttr(0.0)
    px.CreateAngularDampingAttr(0.0)
    px.CreateSleepThresholdAttr(0.00001)
    # Volume-weighted uniform-component inertia; desired total masses are explicit.
    meshes = [p[1] for p in parts if p[3]]
    volume = sum(m.volume for m in meshes)
    center = sum(m.volume * m.center_mass for m in meshes) / volume
    inertia = np.zeros((3, 3))
    for mesh in meshes:
        d = mesh.center_mass - center
        inertia += mesh.moment_inertia + mesh.volume * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
    values, axes = np.linalg.eigh(inertia * mass / volume)
    if np.linalg.det(axes) < 0:
        axes[:, 0] *= -1
    ma = UsdPhysics.MassAPI.Apply(prim)
    ma.CreateMassAttr(mass)
    ma.CreateCenterOfMassAttr(Gf.Vec3f(*center))
    ma.CreateDiagonalInertiaAttr(Gf.Vec3f(*values))
    ma.CreatePrincipalAxesAttr(quat(axes))
    for name, mesh, mat, collide in parts:
        shape(stage, path + "/" + name, mesh, mat, collide)
    return prim


def build(stage, path="/Holder", width=CLOSED_WIDTH, gear=0, origin=(0, 0, 0), angle=None):
    """Author a free 10-body mechanism. No stage/global-physics settings are changed."""
    root = UsdGeom.Xform.Define(stage, path).GetPrim()
    root.SetCustomDataByKey("description", "Reference-inspired passive ABS laptop stand; not a calibrated Tonmom replica")
    poses = configuration(width, gear, angle)
    poses = {name: (p + origin, r) for name, (p, r) in poses.items()}
    mats = {
        "plastic": material(stage, path + "/materials/abs", (0.045, 0.050, 0.060), (0.35, 0.25)),
        "rubber": material(stage, path + "/materials/silicone", (0.16, 0.18, 0.20), (0.9, 0.7)),
        "steel": material(stage, path + "/materials/steel", (0.38, 0.40, 0.43), (0.3, 0.2), 0.8),
    }
    bodies = {}
    for side, sign in (("L", -1), ("R", 1)):
        parts = []

        def add(name, mesh, mat="plastic", collide=True):
            parts.append((name, mesh, mats[mat], collide))

        def add_tip(name, width, inner, outer, mat):
            mesh,cells = ski_tip(width,inner,outer)
            add(name,mesh,mat,False)
            # Invisible convex pieces carry the same physical material and mass.
            for j,cell in enumerate(cells):
                add(f"{name}_collision_{j}",cell,mat)

        # A 2 mm floor, two side strips and solid bridges leave seven actual sockets.
        add("floor", block((LENGTH, RAIL_WIDTH, .002), (.1105, 0, .003), .003))
        for j, y in enumerate((-(RAIL_WIDTH + SLOT_WIDTH)/4, (RAIL_WIDTH + SLOT_WIDTH)/4)):
            add(f"side_{j}", block((LENGTH, (RAIL_WIDTH-SLOT_WIDTH)/2, .004), (.1105, y, .006), .002))
        ends = [-.014] + [float(x) for s in SLOTS for x in (s-SLOT_LENGTH/2, s+SLOT_LENGTH/2)] + [.235]
        for j in range(0, len(ends), 2):
            lo, hi = ends[j:j+2]
            add(f"bridge_{j//2}", block((hi-lo, SLOT_WIDTH, .004), ((lo+hi)/2, 0, .006)))
        for j, x in enumerate((0, .222)):
            add(f"foot_{j}", block((.022, .023, .002), (x, 0, .001), .002), "rubber")
        add("hinge_mount", block((.012, .025, .013), (0, 0, .0115), .002))
        add("hinge", pin(.006, .025, (0, 0, HINGE_Z)))
        add("hinge_cap", pin(.0025, .0254, (0, 0, HINGE_Z)), "steel", False)
        add_tip("front_hook", .025, .023, .029, "plastic")
        add_tip("hook_pad", .023, .0218, .023, "rubber")
        add("hook_pad_flat", block((.007,.023,.0012),(-.0085,0,.0086)),"rubber")
        name = f"base_{side}"
        bodies[name] = body(stage, path+"/links/"+name, *poses[name], .045, parts)
        for prim in bodies[name].GetChildren():
            if "_collision_" in prim.GetName():
                UsdGeom.Imageable(prim).CreateVisibilityAttr(UsdGeom.Tokens.invisible)

        parts = []
        add("beam", block((DECK_LENGTH, .025, .004), (DECK_LENGTH/2, 0, 0), .004))
        add("pad", block((.217, .023, .0015), (.121, 0, .00275), .003), "rubber")
        # Guide ribs contain the sliders without making an invisible linkage.
        for j, y in enumerate((-sign*INNER_Y-.004, -sign*INNER_Y+.004)):
            add(f"guide_{j}", block((.204, .0015, .005), (.112, y, -.0045)))
        add("rear_pivot", pin(.002, .004, (.215, -sign*INNER_Y, -.010), "Z"), "steel", False)
        add("prop_pivot", pin(.002, .025, PROP_PIVOT), "steel", False)
        name = f"deck_{side}"
        bodies[name] = body(stage, path+"/links/"+name, *poses[name], .050, parts)

        parts = []
        add("rod", block((PROP_LENGTH-.003, .007, .0035), ((PROP_LENGTH-.003)/2, 0, 0), .001))
        add("toe", pin(PIN_RADIUS, .005, (PROP_LENGTH, 0, 0)), "plastic")
        name = f"prop_{side}"
        bodies[name] = body(stage, path+"/links/"+name, *poses[name], .014, parts)

        parts = []
        add("bar", block((CROSS_LENGTH, .009, .0015), (CROSS_LENGTH/2, 0, 0), .002))
        add("center_rivet", pin(.002, .002, (CROSS_LENGTH/2, 0, 0), "Z"), "steel", False)
        name = f"cross_{side}"
        bodies[name] = body(stage, path+"/links/"+name, *poses[name], .007, parts)

        parts = []
        add("block", block((.008, .006, .003), (0, 0, 0), .001))
        add("guide_stem", block((.006, .003, .006), (0, 0, .003)))
        name = f"slider_{side}"
        bodies[name] = body(stage, path+"/links/"+name, *poses[name], .004, parts)

    def joint(name, kind, a, b, pa, pb, axis, limits=None, excluded=False, friction=0.0):
        cls = UsdPhysics.RevoluteJoint if kind == "R" else UsdPhysics.PrismaticJoint
        j = cls.Define(stage, path + "/joints/" + name)
        j.CreateBody0Rel().SetTargets([bodies[a].GetPath()])
        j.CreateBody1Rel().SetTargets([bodies[b].GetPath()])
        j.CreateLocalPos0Attr(Gf.Vec3f(*pa))
        j.CreateLocalPos1Attr(Gf.Vec3f(*pb))
        j.CreateLocalRot0Attr(Gf.Quatf(1))
        j.CreateLocalRot1Attr(Gf.Quatf(1))
        j.CreateAxisAttr(axis)
        j.CreateCollisionEnabledAttr(False)
        j.CreateExcludeFromArticulationAttr(excluded)
        if not excluded:
            p0, r0 = poses[a]
            p1, r1 = poses[b]
            k = "XYZ".index(axis)
            q = (np.rad2deg(Rotation.from_matrix(r0.T @ r1).as_rotvec()[k]) if kind == "R"
                 else np.dot(p1 + r1 @ pb - p0 - r0 @ pa, r0[:, k]))
            state = PhysxSchema.JointStateAPI.Apply(j.GetPrim(), "angular" if kind == "R" else "linear")
            state.CreatePositionAttr(float(q))
            state.CreateVelocityAttr(0.0)
        if limits:
            j.CreateLowerLimitAttr(limits[0])
            j.CreateUpperLimitAttr(limits[1])
        if friction:
            axis_name = "angular" if kind == "R" else "linear"
            j.GetPrim().AddAppliedSchema("PhysxJointAxisAPI:" + axis_name)
            for attr, value in (("staticFrictionEffort", friction), ("dynamicFrictionEffort", friction*.7)):
                j.GetPrim().CreateAttribute(f"physxJointAxis:{axis_name}:{attr}", Sdf.ValueTypeNames.Float).Set(value)
        return j

    for side, sign in (("L", -1), ("R", 1)):
        other = "R" if side == "L" else "L"
        layer = -.009 if side == "L" else -.011
        slider_layer = -.011 if side == "L" else -.009
        joint("tilt_"+side, "R", "deck_"+side, "base_"+side, (0,0,0), (0,0,HINGE_Z), "Y", (0,90), friction=.002)
        joint("support_"+side, "R", "deck_"+side, "prop_"+side, PROP_PIVOT, (0,0,0), "Y", (0,165), friction=.0004)
        joint("width_"+side, "P", "deck_"+side, "slider_"+side, (0,-sign*INNER_Y,slider_layer), (0,0,0), "X", SLIDER_LIMITS, friction=.15)
        joint("cross_fixed_"+side, "R", "deck_"+side, "cross_"+side, (CROSS_ANCHOR,-sign*INNER_Y,layer), (0,0,0), "Z")
        joint("cross_end_"+side, "R", "cross_"+side, "slider_"+other, (CROSS_LENGTH,0,0), (0,0,0), "Z", excluded=(side=="R"))
    joint("cross_center", "R", "cross_L", "cross_R", (CROSS_LENGTH/2,0,-.001), (CROSS_LENGTH/2,0,.001), "Z", excluded=True)
    # Suppress only guide/slider pairs already constrained to one rail, and the two
    # hinge assemblies. Support/base contact (which seats each gear) stays enabled.
    for side in ("L", "R"):
        pairs = UsdPhysics.FilteredPairsAPI.Apply(bodies["deck_"+side])
        pairs.CreateFilteredPairsRel().SetTargets([bodies["slider_"+side].GetPath()])
    UsdPhysics.ArticulationRootAPI.Apply(bodies["deck_L"])
    art = PhysxSchema.PhysxArticulationAPI.Apply(bodies["deck_L"])
    art.CreateEnabledSelfCollisionsAttr(True)
    art.CreateSleepThresholdAttr(0.0)
    art.CreateSolverPositionIterationCountAttr(64)
    art.CreateSolverVelocityIterationCountAttr(4)
    return poses


def export():
    stage = Usd.Stage.CreateNew(str(ROOT / "assets/laptop_holder.usda"))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    build(stage)
    stage.SetDefaultPrim(stage.GetPrimAtPath("/Holder"))
    stage.GetRootLayer().Save()
    return stage
