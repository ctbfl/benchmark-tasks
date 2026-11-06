"""Build original, hollow stackable cones with passive compliant tip petals."""
import numpy as np
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True, kit_args="--/plugins/carb.tasking.plugin/threadCount=8 --/plugins/omni.tbb.globalcontrol/maxThreadCount=8").app
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, PhysxSchema

from config import (HERE, HEIGHT, BASE_WIDTH, BASE_THICKNESS, TIP_HEIGHT, BODY_RADIUS,
                    TIP_RADIUS, WALL, STACK_PITCH, BOARD_SIZE, BOARD_CENTER, ROAD_WIDTH, LENGTH, curve, tangent)


def material(stage, name, color, friction):
    mat = UsdShade.Material.Define(stage, '/Asset/materials/' + name)
    shader = UsdShade.Shader.Define(stage, str(mat.GetPath()) + '/surface')
    shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(.6)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
    physics = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    physics.CreateStaticFrictionAttr(friction)
    physics.CreateDynamicFrictionAttr(friction)
    physics.CreateRestitutionAttr(0)
    return mat


def bind(prim, mat):
    api = UsdShade.MaterialBindingAPI.Apply(prim)
    api.Bind(mat)
    api.Bind(mat, materialPurpose='physics')


def mesh(stage, path, points, faces, mat, collide=True):
    geom = UsdGeom.Mesh.Define(stage, path)
    geom.CreatePointsAttr([Gf.Vec3f(*map(float, p)) for p in points])
    geom.CreateFaceVertexCountsAttr([len(f) for f in faces])
    geom.CreateFaceVertexIndicesAttr([i for f in faces for i in f])
    geom.CreateSubdivisionSchemeAttr('none')
    geom.CreateDoubleSidedAttr(True)
    bind(geom.GetPrim(), mat)
    if collide:
        UsdPhysics.CollisionAPI.Apply(geom.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(geom.GetPrim()).CreateApproximationAttr('convexHull')
        contact = PhysxSchema.PhysxCollisionAPI.Apply(geom.GetPrim())
        contact.CreateContactOffsetAttr(.0003)
        contact.CreateRestOffsetAttr(0)
    return geom


FACES = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]


def sector(stage, path, z0, z1, r0, r1, inner0, inner1, a0, a1, mat):
    pts = []
    for z, outer, inner in ((z0, r0, inner0), (z1, r1, inner1)):
        pts.extend([(r*np.cos(a), r*np.sin(a), z) for r, a in
                    ((inner, a0), (outer, a0), (outer, a1), (inner, a1))])
    return mesh(stage, path, pts, FACES, mat)


def body(stage, path, mass, position=(0., 0., 0.)):
    prim = UsdGeom.Xform.Define(stage, path)
    prim.AddTranslateOp().Set(Gf.Vec3d(*position))
    UsdPhysics.RigidBodyAPI.Apply(prim.GetPrim())
    m = UsdPhysics.MassAPI.Apply(prim.GetPrim())
    m.CreateMassAttr(mass)
    PhysxSchema.PhysxContactReportAPI.Apply(prim.GetPrim()).CreateThresholdAttr(0)
    return prim


def new_stage(filename):
    stage = Usd.Stage.CreateNew(str(HERE / 'assets' / filename))
    root = UsdGeom.Xform.Define(stage, '/Asset')
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    return stage


def cone():
    stage = new_stage('cone.usda')
    orange = material(stage, 'orange', (1., .18, .012), .8)
    white = material(stage, 'white', (.9, .9, .86), .8)
    rubber = material(stage, 'rubber', (.025, .028, .032), 1.0)
    tipmat = material(stage, 'tip', (1., .18, .012), 1.2)
    root = body(stage, '/Asset/body', .011)
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    PhysxSchema.PhysxArticulationAPI.Apply(root.GetPrim()).CreateEnabledSelfCollisionsAttr(True)
    # Square ring leaves the underside open for nesting.
    n = 32
    for k in range(n):
        a0, a1 = 2*np.pi*np.array([k, k+1])/n
        outer = [BASE_WIDTH / 2 / max(abs(np.cos(a)), abs(np.sin(a))) for a in (a0, a1)]
        points = [(r*np.cos(a), r*np.sin(a), z) for z in (0., BASE_THICKNESS)
                  for r, a in ((BODY_RADIUS-WALL, a0), (outer[0], a0),
                               (outer[1], a1), (BODY_RADIUS-WALL, a1))]
        mesh(stage, f'/Asset/body/base_{k:02}', points, FACES, rubber)
    top = HEIGHT - TIP_HEIGHT
    levels = [BASE_THICKNESS, .014, .019, .027, .034, top]
    radius = lambda z: BODY_RADIUS + (TIP_RADIUS - BODY_RADIUS) * (z-BASE_THICKNESS)/(top-BASE_THICKNESS)
    for j, (z0, z1) in enumerate(zip(levels[:-1], levels[1:])):
        for k in range(n):
            sector(stage, f'/Asset/body/shell_{j}_{k:02}', z0, z1, radius(z0), radius(z1),
                   radius(z0)-WALL, radius(z1)-WALL, 2*np.pi*k/n, 2*np.pi*(k+1)/n,
                   white if j in (1, 3) else orange)
    # Horizontal internal shoulder prevents friction-locking between tapered shells.
    # The next cone's top rim supports this shoulder, with radial clearance below.
    ledge_z = HEIGHT - STACK_PITCH
    for k in range(n):
        sector(stage, f'/Asset/body/stack_stop_{k:02}', ledge_z, ledge_z+.0008,
               radius(ledge_z), radius(ledge_z+.0008), .0018, .0018,
               2*np.pi*k/n, 2*np.pi*(k+1)/n, orange)
    for i in range(4):
        angle = i*np.pi/2
        radial = np.array([np.cos(angle), np.sin(angle), 0.])
        petal = body(stage, f'/Asset/petal_{i}', .00025, (0., 0., top))
        # The quadrants approximate one compressible cap. Mutual hard contact would
        # jam this radial compliance model; external contact remains fully enabled.
        UsdPhysics.FilteredPairsAPI.Apply(petal.GetPrim()).CreateFilteredPairsRel().SetTargets(
            [Sdf.Path(f'/Asset/petal_{j}') for j in range(4) if j != i])
        for k in range(8):
            a0 = angle - np.pi/4 + k*np.pi/16
            a1 = a0 + np.pi/16
            sector(stage, f'/Asset/petal_{i}/shape_{k}', 0., TIP_HEIGHT,
                   TIP_RADIUS, TIP_RADIUS*.92, TIP_RADIUS-WALL, TIP_RADIUS*.92-WALL,
                   a0, a1, tipmat)
        joint = UsdPhysics.PrismaticJoint.Define(stage, f'/Asset/joints/compress_{i}')
        joint.CreateBody0Rel().SetTargets([root.GetPath()])
        joint.CreateBody1Rel().SetTargets([petal.GetPath()])
        joint.CreateAxisAttr('X')
        joint.CreateLocalPos0Attr(Gf.Vec3f(0, 0, top))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
        rot = Gf.Quatf(float(np.cos(angle/2)), Gf.Vec3f(0, 0, float(np.sin(angle/2))))
        joint.CreateLocalRot0Attr(rot)
        joint.CreateLocalRot1Attr(rot)
        joint.CreateLowerLimitAttr(-.0008)
        joint.CreateUpperLimitAttr(0.)
        joint.CreateCollisionEnabledAttr(False)
        spring = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), 'linear')
        spring.CreateTypeAttr('force')
        spring.CreateStiffnessAttr(400.)
        spring.CreateDampingAttr(.2)
        spring.CreateTargetPositionAttr(0.)
        spring.CreateMaxForceAttr(5.)
    stage.GetRootLayer().Save()


def board():
    stage = new_stage('board.usda')
    white = material(stage, 'board', (.86, .88, .89), .7)
    black = material(stage, 'ink', (.008, .01, .012), .7)
    geom = UsdGeom.Cube.Define(stage, '/Asset/board')
    geom.CreateSizeAttr(1.)
    geom.AddTranslateOp().Set(Gf.Vec3d(BOARD_CENTER[0], BOARD_CENTER[1], BOARD_SIZE[2]/2))
    geom.AddScaleOp().Set(Gf.Vec3f(*BOARD_SIZE))
    UsdPhysics.CollisionAPI.Apply(geom.GetPrim())
    PhysxSchema.PhysxCollisionAPI.Apply(geom.GetPrim()).CreateContactOffsetAttr(.0003)
    bind(geom.GetPrim(), white)
    def line(name, s, offset, width):
        centre = curve(s)
        t = tangent(s)
        normal = np.stack((-t[:, 1], t[:, 0]), -1)
        xy = np.stack((centre+normal*(offset-width/2), centre+normal*(offset+width/2)), axis=1).reshape(-1, 2)
        pts = np.column_stack((xy, np.full(len(xy), BOARD_SIZE[2]+.00002)))
        faces = [(2*i, 2*i+2, 2*i+3, 2*i+1) for i in range(len(s)-1)]
        mesh(stage, '/Asset/'+name, pts, faces, black, False)
    for side in (-1, 1):
        line('edge_' + str(side).replace('-', 'minus'), np.linspace(0, LENGTH, 401), side*ROAD_WIDTH/2, .002)
    for k, start in enumerate(np.arange(0, LENGTH, .035)):
        line(f'dash_{k:02}', np.linspace(start, min(start+.020, LENGTH), 16), 0., .002)
    stage.GetRootLayer().Save()


if __name__ == '__main__':
    cone()
    board()
    print('Built assets/cone.usda and assets/board.usda')
    app.close()
