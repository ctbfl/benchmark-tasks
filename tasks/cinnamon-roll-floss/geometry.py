"""Dimensions and procedural USD geometry for the baking board and tool."""
from pathlib import Path

import numpy as np

BOARD_X = .293
CUT_X = .293
RADIUS = .031
WIDTH = .045
FLAT = .0028
HANDLE_X = .195
HANDLE_Y = .116
HANDLE_SIZE = (.022, .026, .028)
CORD_LENGTH = .256
CORD_RADIUS = .00065
ANCHOR = np.array([0., 0., -.011])
ROLL_NAMES = ("near_roll", "far_roll")
HANDLE_NAMES = ("left_handle", "right_handle")
BOARD_SIZE = (.30, .426, .008)
BEIGE = (.79, .52, .28)
CINNAMON = (.32, .11, .038)
TEAL = (.045, .31, .34)
MARK = (.30, .60, .55)


def paint(prim, color):
    from pxr import Gf, UsdGeom

    UsdGeom.Gprim(prim).CreateDisplayColorAttr([Gf.Vec3f(*color)])


def mesh(stage, path, points, faces, color):
    from pxr import UsdGeom, Vt

    surf = UsdGeom.Mesh.Define(stage, path)
    surf.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(points, dtype=np.float32)))
    surf.CreateFaceVertexCountsAttr([len(face) for face in faces])
    surf.CreateFaceVertexIndicesAttr([i for face in faces for i in face])
    surf.CreateSubdivisionSchemeAttr("none")
    surf.CreateDoubleSidedAttr(True)
    paint(surf.GetPrim(), color)
    return surf


def roll_asset(path, mark_side):
    """Revolve the hand-shaped dough and add a ribbon of cinnamon to each face."""
    from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    body = UsdGeom.Xform.Define(stage, "/Roll")
    stage.SetDefaultPrim(body.GetPrim())
    PhysxSchema.PhysxRigidBodyAPI.Apply(body.GetPrim()).CreateMaxDepenetrationVelocityAttr(.8)
    PhysxSchema.PhysxRigidBodyAPI(body.GetPrim()).CreateAngularDampingAttr(3.)
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    PhysxSchema.PhysxContactReportAPI.Apply(body.GetPrim()).CreateThresholdAttr(0.)
    UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr(.151)
    n_collider = 48
    ring = np.arange(n_collider)*2*np.pi/n_collider
    vertices = [(x, (RADIUS-.0005)*np.cos(a), max(-RADIUS+FLAT, (RADIUS-.0005)*np.sin(a)))
                for x in (-WIDTH/2, WIDTH/2) for a in ring]
    hull_faces = [tuple(reversed(range(n_collider))), tuple(range(n_collider, 2*n_collider))]
    hull_faces += [(i, (i+1)%n_collider, (i+1)%n_collider+n_collider, i+n_collider)
                   for i in range(n_collider)]
    shape = mesh(stage, "/Roll/Collider", vertices, hull_faces, BEIGE)
    shape.CreatePurposeAttr("guide")
    UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(shape.GetPrim()).CreateApproximationAttr("convexHull")
    c = PhysxSchema.PhysxCollisionAPI.Apply(shape.GetPrim())
    c.CreateContactOffsetAttr(.0007)
    c.CreateRestOffsetAttr(0.)
    mat = UsdShade.Material.Define(stage, "/Roll/DoughContact")
    phys = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    phys.CreateStaticFrictionAttr(.69)
    phys.CreateDynamicFrictionAttr(.57)
    UsdShade.MaterialBindingAPI.Apply(shape.GetPrim()).Bind(mat, materialPurpose="physics")

    UsdGeom.Xform.Define(stage, "/Roll/Surface")
    n = 96
    angle = np.arange(n) * 2 * np.pi / n
    scallop = 1. + .017 * np.cos(3 * angle - .5) + .008 * np.sin(7 * angle + .4)
    sections = [(-.5, .93), (-.48, .978), (-.40, 1.), (0., 1.015), (.40, 1.), (.48, .978), (.5, .93)]
    points = [(x * WIDTH, r * RADIUS * s * np.cos(t), max(-RADIUS+FLAT-.0001, r * RADIUS * s * np.sin(t)))
              for x, r in sections for s, t in zip(scallop, angle)]
    faces = [tuple(reversed(range(n)))]
    faces += [(j*n+i, j*n+(i+1)%n, (j+1)*n+(i+1)%n, (j+1)*n+i)
              for j in range(len(sections)-1) for i in range(n)]
    faces += [tuple(range((len(sections)-1)*n, len(sections)*n))]
    mesh(stage, "/Roll/Surface/Dough", points, faces, BEIGE)

    t = np.linspace(.08, 5.05 * np.pi, 220)
    r = .0016 + .00167 * t
    for side in (-1, 1):
        x = side * (WIDTH/2 + .00016)
        ribbon = [(x, (radius + dr) * np.cos(phi), (radius + dr) * np.sin(phi))
                  for radius, phi in zip(r, t) for dr in (-.00087, .00087)]
        mesh(stage, f"/Roll/Surface/Spiral{side+1}", ribbon,
             [(2*i, 2*i+1, 2*i+3, 2*i+2) for i in range(len(t)-1)], CINNAMON)
    # A shallow scoring line on the top, aligned across both pieces.
    t = np.linspace(.58, 2.56, 40)
    r = .974 * RADIUS * (1. + .017 * np.cos(3*t-.5) + .008 * np.sin(7*t+.4))
    line = [(mark_side*(WIDTH/2-d), (radius + .00045)*np.cos(a), (radius + .00045)*np.sin(a))
            for radius, a in zip(r, t) for d in (.0002, .0009)]
    mesh(stage, "/Roll/Surface/ScoreLine", line,
         [(2*i, 2*i+1, 2*i+3, 2*i+2) for i in range(len(t)-1)], CINNAMON)
    stage.GetRootLayer().Export(str(Path(path)))
