"""Original metre-scale coin, open jar and red tape assets for the coin sorting task."""
from pathlib import Path
import math

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

HERE = Path(__file__).resolve().parent

# US Mint specifications: diameter, thickness and mass.
COINS = {
    "quarter": {"radius": .02426/2, "thickness": .00175, "mass": .00567},
    "penny": {"radius": .01905/2, "thickness": .00152, "mass": .0025},
}
JAR_OUTER = .034
JAR_INNER = .030
JAR_HEIGHT = .090
JAR_FLOOR = .004
JAR_MASS = .12
# Tape frames on the table: centre (x, y) and inner size (x, y), in metres.
TAPE_WIDTH = .012
FRAMES = {
    "penny": ((.38, .245), (.16, .11)),
    "quarter": ((.38, -.245), (.16, .11)),
}


def material(stage, name, color, roughness=.45, metallic=0.):
    mat = UsdShade.Material.Define(stage, "/Asset/materials/" + name)
    shader = UsdShade.Shader.Define(stage, str(mat.GetPath()) + "/shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def mesh(stage, path, vertices, faces, mat, collision=False, offsets=(.001, 0.)):
    geom = UsdGeom.Mesh.Define(stage, path)
    geom.CreatePointsAttr([Gf.Vec3f(*map(float, v)) for v in vertices])
    geom.CreateFaceVertexCountsAttr([len(f) for f in faces])
    geom.CreateFaceVertexIndicesAttr([i for f in faces for i in f])
    geom.CreateSubdivisionSchemeAttr("none")
    UsdShade.MaterialBindingAPI.Apply(geom.GetPrim()).Bind(mat)
    if collision:
        prim = geom.GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim)
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexHull")
        prim.AddAppliedSchema("PhysxCollisionAPI")
        prim.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(offsets[0])
        prim.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(offsets[1])
    else:
        geom.CreatePurposeAttr("render")
    return geom


def prism(stage, path, points, bottom, top, mat, collision=False, offsets=(.001, 0.)):
    n = len(points)
    vertices = [(x, y, z) for z in (bottom, top) for x, y in points]
    faces = [list(reversed(range(n))), list(range(n, 2*n))]
    faces += [[i, (i+1) % n, (i+1) % n + n, i+n] for i in range(n)]
    return mesh(stage, path, vertices, faces, mat, collision, offsets)


def circle(radius, sides, phase=0.):
    angles = phase + np.arange(sides) * 2*math.pi/sides
    return np.column_stack((radius*np.cos(angles), radius*np.sin(angles)))


def annulus(stage, path, outer, inner, bottom, top, mat, sides=48):
    """A closed visual ring with smooth inner and outer walls."""
    a, b = circle(outer, sides), circle(inner, sides)
    vertices = [(*p, z) for z in (bottom, top) for p in a] + [(*p, z) for z in (bottom, top) for p in b]
    n = sides
    faces = []
    for i in range(n):
        j = (i+1) % n
        faces.append([i, j, n+j, n+i])                    # outer wall
        faces.append([2*n+j, 2*n+i, 3*n+i, 3*n+j])        # inner wall
        faces.append([n+i, n+j, 3*n+j, 3*n+i])            # rim
        faces.append([j, i, 2*n+i, 2*n+j])                # underside
    return mesh(stage, path, vertices, faces, mat)


def rigid(stage, path, mass, friction, iterations=None):
    body = UsdGeom.Xform.Define(stage, path)
    prim = body.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(mass)
    prim.AddAppliedSchema("PhysxRigidBodyAPI")
    prim.CreateAttribute("physxRigidBody:enableCCD", Sdf.ValueTypeNames.Bool).Set(True)
    if iterations:
        prim.CreateAttribute("physxRigidBody:solverPositionIterationCount", Sdf.ValueTypeNames.Int).Set(iterations[0])
        prim.CreateAttribute("physxRigidBody:solverVelocityIterationCount", Sdf.ValueTypeNames.Int).Set(iterations[1])
    physics = UsdShade.Material.Define(stage, path + "/surface")
    coeff = UsdPhysics.MaterialAPI.Apply(physics.GetPrim())
    coeff.CreateStaticFrictionAttr(friction)
    coeff.CreateDynamicFrictionAttr(friction*.85)
    coeff.CreateRestitutionAttr(0.)
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(physics, materialPurpose="physics")
    return body


def stage_for(name):
    stage = Usd.Stage.CreateNew(str(HERE / "assets" / (name + ".usda")))
    root = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdGeom.SetStageUpAxis(stage, "Z")
    return stage


def build_coin(kind, face, edge, relief):
    spec = COINS[kind]
    radius, half = spec["radius"], spec["thickness"]/2
    stage = stage_for(kind)
    face_mat = material(stage, "face", face, .32, .9)
    edge_mat = material(stage, "edge", edge, .40, .9)
    relief_mat = material(stage, "relief", relief, .30, .9)
    rigid(stage, "/Asset/body", spec["mass"], .35)
    # The collider is a 32-sided convex prism; the visual rim and raised relief are thin decals.
    prism(stage, "/Asset/body/collider", circle(radius, 32), -half, half, edge_mat, True, (.0008, 0.))
    for z, sign in ((half, 1), (-half, -1)):
        annulus(stage, f"/Asset/body/rim_{'top' if sign > 0 else 'bottom'}", radius-.0002, radius*.86,
                z, z + sign*.00015, face_mat)
        portrait = circle(radius*.55, 40)
        prism(stage, f"/Asset/body/relief_{'top' if sign > 0 else 'bottom'}", portrait,
              min(z, z+sign*.0001), max(z, z+sign*.0001), relief_mat)
    stage.GetRootLayer().Save()


def build_jar():
    stage = stage_for("jar")
    body_mat = material(stage, "plastic", (.20, .42, .62), .35)
    rim_mat = material(stage, "rim", (.86, .88, .90), .4)
    rigid(stage, "/Asset/body", JAR_MASS, .4)
    sides = 24
    prism(stage, "/Asset/body/floor", circle(JAR_OUTER, sides), 0., JAR_FLOOR, body_mat, True)
    # Convex wall segments keep the cavity open; each segment overlaps its neighbours slightly.
    for i in range(sides):
        a0, a1 = (i-.55)*2*math.pi/sides, (i+.55)*2*math.pi/sides
        ring = [(r*math.cos(a), r*math.sin(a)) for r, a in
                ((JAR_OUTER, a0), (JAR_OUTER, a1), (JAR_INNER, a1), (JAR_INNER, a0))]
        wall = prism(stage, f"/Asset/body/wall_{i:02}", ring, JAR_FLOOR, JAR_HEIGHT, body_mat, True)
        wall.CreatePurposeAttr("guide")
    annulus(stage, "/Asset/body/shell", JAR_OUTER, JAR_INNER, JAR_FLOOR, JAR_HEIGHT-.006, body_mat)
    annulus(stage, "/Asset/body/lip", JAR_OUTER+.0006, JAR_INNER, JAR_HEIGHT-.006, JAR_HEIGHT, rim_mat)
    stage.GetRootLayer().Save()


def build_tape():
    stage = stage_for("tape")
    red = material(stage, "red_paper", (.64, .018, .012), .92)
    for prefix, ((x, y), (sx, sy)) in FRAMES.items():
        w = TAPE_WIDTH
        x0, x1, y0, y1 = x-sx/2, x+sx/2, y-sy/2, y+sy/2
        # Near, far, right and left strips as (xmin, xmax, ymin, ymax); the inner edge is the frame.
        for i, (a, b, c, d) in enumerate(((x0-w, x0, y0-w, y1+w), (x1, x1+w, y0-w, y1+w),
                                          (x0, x1, y0-w, y0), (x0, x1, y1, y1+w))):
            prism(stage, f"/Asset/{prefix}_{i}", [(a, c), (b, c), (b, d), (a, d)], .00005, .00030, red)
    stage.GetRootLayer().Save()


if __name__ == "__main__":
    (HERE / "assets").mkdir(exist_ok=True)
    for path in (HERE / "assets").glob("*.usda"):
        path.unlink()
    build_coin("quarter", (.80, .81, .83), (.70, .71, .73), (.62, .63, .66))
    build_coin("penny", (.72, .40, .22), (.62, .33, .18), (.55, .29, .15))
    build_jar()
    build_tape()
    print("Built quarter, penny, jar and tape assets", flush=True)
