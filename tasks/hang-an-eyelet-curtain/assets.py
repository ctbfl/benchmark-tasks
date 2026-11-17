"""Original procedural assets: drying-rack style rod stand, curtain rod and eyelet curtain."""
import math
import numpy as np
from scipy.spatial import Delaunay
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt
from config import (HERE, SUPPORT_HALF_SPAN, ROD_HEIGHT, ROD_R, CRADLE_T, CRADLE_SLOT, PRONG_T,
                    PRONG_TOP, KNOB_R, KNOB_T, KNOB_GAP, ROD_HALF, RING_RI, RING_RO, RING_T,
                    RINGS, RING_MASS, RING_PITCH, RING_DROP, HOLE_R, FABRIC_W, FABRIC_H, FABRIC_T,
                    BEND_STIFFNESS, BEND_DAMPING)

SEGMENTS = 16                    # convex sectors per grommet collider
OFFSET = .001                    # contact offset; the curtain uses a larger one


def material(stage, root, name, rgb, metal=0., roughness=.5, friction=(.6, .5)):
    mat = UsdShade.Material.Define(stage, root+'/Looks/'+name)
    shader = UsdShade.Shader.Define(stage, str(mat.GetPath())+'/Shader')
    shader.CreateIdAttr('UsdPreviewSurface')
    for key, typ, value in [('diffuseColor', Sdf.ValueTypeNames.Color3f, Gf.Vec3f(*rgb)),
                            ('roughness', Sdf.ValueTypeNames.Float, roughness),
                            ('metallic', Sdf.ValueTypeNames.Float, metal)]:
        shader.CreateInput(key, typ).Set(value)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
    physics = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    physics.CreateStaticFrictionAttr(friction[0])
    physics.CreateDynamicFrictionAttr(friction[1])
    physics.CreateRestitutionAttr(0.)
    return mat


def finish(shape, mat, collision=True, offset=None):
    offset = OFFSET if offset is None else offset
    binding = UsdShade.MaterialBindingAPI.Apply(shape.GetPrim())
    binding.Bind(mat)
    binding.Bind(mat, materialPurpose='physics')
    if collision:
        UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
        cfg = PhysxSchema.PhysxCollisionAPI.Apply(shape.GetPrim())
        cfg.CreateContactOffsetAttr(offset)
        cfg.CreateRestOffsetAttr(0.)
    else:
        UsdGeom.Imageable(shape.GetPrim()).CreatePurposeAttr('default')
    return shape


def box(stage, path, lo, hi, mat, collision=True):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    shape = UsdGeom.Cube.Define(stage, path)
    shape.CreateSizeAttr(1)
    shape.AddTranslateOp().Set(Gf.Vec3d(*((lo+hi)/2)))
    shape.AddScaleOp().Set(Gf.Vec3f(*(hi-lo)))
    return finish(shape, mat, collision)


def mesh(stage, path, vertices, faces, mat, collision=True, smooth=False):
    shape = UsdGeom.Mesh.Define(stage, path)
    shape.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(vertices, dtype=np.float32)))
    shape.CreateFaceVertexCountsAttr([len(f) for f in faces])
    shape.CreateFaceVertexIndicesAttr([i for f in faces for i in f])
    shape.CreateSubdivisionSchemeAttr('catmullClark' if smooth else 'none')
    shape.CreateDoubleSidedAttr(True)
    if collision:
        UsdPhysics.MeshCollisionAPI.Apply(shape.GetPrim()).CreateApproximationAttr('convexHull')
    return finish(shape, mat, collision)


def prism(stage, path, outline, axis_lo, axis_hi, mat, axis=2, collision=True):
    """Extrude a convex 2D outline (in the two axes other than ``axis``)."""
    n = len(outline)
    others = [i for i in range(3) if i != axis]
    def point(uv, w):
        p = [0., 0., 0.]
        p[others[0]], p[others[1]], p[axis] = uv[0], uv[1], w
        return p
    vertices = [point(uv, axis_lo) for uv in outline]+[point(uv, axis_hi) for uv in outline]
    faces = [list(reversed(range(n))), list(range(n, 2*n))]+[[i, (i+1) % n, (i+1) % n+n, i+n] for i in range(n)]
    return mesh(stage, path, vertices, faces, mat, collision)


def annulus(stage, path, ri, ro, lo, hi, mat, axis=2, count=48, center=(0., 0.)):
    """A visual flat ring (washer) as one closed mesh."""
    others = [i for i in range(3) if i != axis]
    a = np.linspace(0, 2*math.pi, count, endpoint=False)
    vertices = []
    for r, w in [(ri, lo), (ro, lo), (ri, hi), (ro, hi)]:
        for t in a:
            p = [0., 0., 0.]
            p[others[0]], p[others[1]], p[axis] = center[0]+r*math.cos(t), center[1]+r*math.sin(t), w
            vertices.append(p)
    faces = []
    for i in range(count):
        j = (i+1) % count
        il, ol, ih, oh = i, count+i, 2*count+i, 3*count+i
        jl, jo, jh, joh = j, count+j, 2*count+j, 3*count+j
        faces += [[il, jl, jo, ol], [ih, oh, joh, jh], [ol, jo, joh, oh], [il, ih, jh, jl]]
    return mesh(stage, path, vertices, faces, mat, collision=False)


def cylinder(stage, path, r, lo, hi, mat, axis=1, count=24, collision=True):
    a = np.linspace(0, 2*math.pi, count, endpoint=False)
    return prism(stage, path, [(r*math.cos(t), r*math.sin(t)) for t in a], lo, hi, mat, axis, collision)


def rigid(stage, path, mass, pos=(0, 0, 0), max_depenetration=None):
    prim = UsdGeom.Xform.Define(stage, path).GetPrim()
    UsdGeom.Xformable(prim).AddTranslateOp().Set(Gf.Vec3d(*pos))
    UsdPhysics.RigidBodyAPI.Apply(prim)
    if max_depenetration is not None:
        PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateMaxDepenetrationVelocityAttr(max_depenetration)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(mass)
    PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
    return prim


def new_stage(name):
    stage = Usd.Stage.CreateNew(str(HERE/'assets'/f'{name}.usda'))
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdGeom.SetStageUpAxis(stage, 'Z')
    root = '/'+name.capitalize()
    prim = UsdGeom.Xform.Define(stage, root).GetPrim()
    stage.SetDefaultPrim(prim)
    return stage, root


def build_stand():
    stage, root = new_stage('stand')
    steel = material(stage, root, 'steel', (.78, .80, .82), metal=.8, roughness=.3)
    rubber = material(stage, root, 'rubber', (.10, .10, .11), roughness=.8, friction=(1., .9))
    rigid(stage, root+'/body', 2.5)
    body = root+'/body'
    s, h = SUPPORT_HALF_SPAN, ROD_HEIGHT
    seat = h-ROD_R
    for i, y in enumerate((-s, s)):
        box(stage, f'{body}/foot{i}', (-.085, y-.016, 0.), (.085, y+.016, .012), rubber)
        box(stage, f'{body}/post{i}', (-.007, y-.007, .012), (.007, y+.007, seat-.010), steel)
        box(stage, f'{body}/seat{i}', (-CRADLE_SLOT/2-PRONG_T, y-CRADLE_T/2, seat-.010),
            (CRADLE_SLOT/2+PRONG_T, y+CRADLE_T/2, seat), steel)
        for j, sign in enumerate((-1, 1)):
            x0 = sign*CRADLE_SLOT/2
            x1 = sign*(CRADLE_SLOT/2+PRONG_T)
            box(stage, f'{body}/prong{i}{j}', (min(x0, x1), y-CRADLE_T/2, seat),
                (max(x0, x1), y+CRADLE_T/2, h+PRONG_TOP), steel)
    box(stage, f'{body}/rear_bar', (.055, -s-.016, 0.), (.080, s+.016, .012), steel)
    stage.Save()


def build_rod():
    stage, root = new_stage('rod')
    black = material(stage, root, 'black_metal', (.035, .035, .04), metal=.6, roughness=.35, friction=(.4, .35))
    rigid(stage, root+'/body', .12*(2*ROD_HALF/.382))  # preserve the original rod's linear density
    body = root+'/body'
    shape = UsdGeom.Capsule.Define(stage, body+'/shaft')
    shape.CreateRadiusAttr(ROD_R)
    shape.CreateHeightAttr(2*ROD_HALF-2*ROD_R)
    shape.CreateAxisAttr('Y')
    finish(shape, black)
    knob_in = SUPPORT_HALF_SPAN+CRADLE_T/2+KNOB_GAP
    for i, sign in enumerate((-1, 1)):
        lo, hi = sorted((sign*knob_in, sign*(knob_in+KNOB_T)))
        cylinder(stage, f'{body}/knob{i}', KNOB_R, lo, hi, black)
    stage.Save()



def curtain_topology():
    """One connected sheet with a circular boundary for each eyelet."""
    x, y = np.meshgrid(np.linspace(-FABRIC_W/2, FABRIC_W/2, round(FABRIC_W/.01)+1),
                       np.linspace(0, FABRIC_H, 21))
    grid = np.column_stack((x.ravel(), y.ravel()))
    centers = np.column_stack((np.linspace(-(RINGS-1)*RING_PITCH/2, (RINGS-1)*RING_PITCH/2, RINGS), np.full(RINGS, RING_DROP)))
    outer = (np.abs(grid[:, 0]) == FABRIC_W/2) | (grid[:, 1] == 0) | (grid[:, 1] == FABRIC_H)
    grid = grid[outer | (np.linalg.norm(grid[:, None] - centers, axis=2) > HOLE_R + .002).all(axis=1)]
    angles = np.arange(32) * (2 * np.pi / 32)
    holes = centers[:, None] + HOLE_R * np.column_stack((np.cos(angles), np.sin(angles)))
    uv = np.vstack((grid, holes.reshape(-1, 2)))
    ids = np.arange(len(grid), len(uv)).reshape(RINGS, 32)
    faces = Delaunay(uv).simplices
    centroids = uv[faces].mean(axis=1)
    faces = faces[(np.linalg.norm(centroids[:, None] - centers, axis=2) > HOLE_R * .999).all(axis=1)]
    edges, counts = np.unique(np.sort(np.concatenate((faces[:, :2], faces[:, 1:], faces[:, ::2])), axis=1), axis=0, return_counts=True)
    assert counts.max() == 2 and len(np.unique(faces)) == len(uv)
    boundary = set(map(tuple, edges[counts == 1]))
    for hole in ids:
        assert all(tuple(sorted((int(a), int(b)))) in boundary for a, b in zip(hole, np.roll(hole, -1)))
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(len(uv), len(uv)))
    assert connected_components(graph, directed=False)[0] == 1
    # One connected orientable sheet with one outer boundary and RINGS holes.
    assert len(uv) - len(edges) + len(faces) == 1-RINGS
    return np.column_stack((-uv[:, 1], uv[:, 0], np.zeros(len(uv)))), faces, edges, ids


def vertex_normals(points, faces):
    """Smooth lighting on the measured mesh, without moving any vertices."""
    normals = np.zeros_like(points, dtype=np.float32)
    triangles = points[faces]
    face_normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    for column in range(3):
        np.add.at(normals, faces[:, column], face_normals)
    length = np.linalg.norm(normals, axis=1)
    assert np.all(length > 1e-12)
    return normals/length[:, None]



def display_cloth(stage, path, points, faces):
    """Copy measured world vertices to the visual mesh; never write the physics mesh."""
    visual = UsdGeom.Mesh.Get(stage, path+'/ClothVisual')
    inverse = np.array(UsdGeom.XformCache().GetLocalToWorldTransform(visual.GetPrim()).GetInverse())
    local = (points@inverse[:3, :3]+inverse[3, :3]).astype(np.float32)
    visual.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(local))
    visual.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(vertex_normals(local, faces)))


def build_curtain():
    from omni.physx.scripts import deformableUtils, physicsUtils
    stage, root = new_stage('curtain')
    blue = material(stage, root, 'cloth', (.035, .23, .42), roughness=.9)
    steel = material(stage, root, 'rings', (.65, .7, .76), metal=.8, roughness=.25)
    points, faces, edges, holes = curtain_topology()
    cloth = mesh(stage, root+'/Cloth', points, faces.tolist(), blue, collision=False)
    cloth.MakeInvisible()
    # Smooth shading is independent of Fabric's normal-adjacency update.
    visual = mesh(stage, root+'/ClothVisual', points, faces.tolist(), blue, collision=False)
    visual.SetNormalsInterpolation('vertex')
    visual.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(vertex_normals(points, faces)))
    assert deformableUtils.set_physics_surface_deformable_body(stage, cloth.GetPath())
    prim = cloth.GetPrim()
    assert prim.ApplyAPI('PhysxSurfaceDeformableBodyAPI')
    for name, value in dict(selfCollision=True, enableSpeculativeCCD=True,
            solverPositionIterationCount=32, collisionPairUpdateFrequency=4,
            collisionIterationMultiplier=4, maxLinearVelocity=2.).items():
        assert prim.GetAttribute('physxDeformableBody:'+name).Set(value)
    collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    collision.CreateRestOffsetAttr(.00025)
    collision.CreateContactOffsetAttr(.001)
    matpath = root+'/ClothPhysics'
    assert deformableUtils.add_surface_deformable_material(stage, matpath, density=500.,
        dynamic_friction=.4, youngs_modulus=2e6, poissons_ratio=.3,
        surface_thickness=FABRIC_T, surface_bend_stiffness=BEND_STIFFNESS)
    physicsUtils.add_physics_material_to_prim(stage, prim, matpath)
    mprim = stage.GetPrimAtPath(matpath)
    assert mprim.ApplyAPI('PhysxSurfaceDeformableMaterialAPI')
    mprim.GetAttribute('physxDeformableMaterial:elasticityDamping').Set(.01)
    mprim.GetAttribute('physxDeformableMaterial:bendDamping').Set(BEND_DAMPING)
    for i, y in enumerate(np.linspace(-(RINGS-1)*RING_PITCH/2, (RINGS-1)*RING_PITCH/2, RINGS)):
        path = f'{root}/ring_{i}'
        pos = np.array([-RING_DROP, y, 0.])
        ring = rigid(stage, path, RING_MASS, pos)
        UsdGeom.Xformable(ring).AddOrientOp().Set(Gf.Quatf(1, 0, 0, 0))
        annulus(stage, path+'/Visual', RING_RI, RING_RO, -RING_T/2, RING_T/2, steel)
        sewn = np.flatnonzero(np.isin(faces, holes[i]).any(axis=1))
        for j in range(SEGMENTS):
            a, b = 2*np.pi*np.array((j, j+1))/SEGMENTS
            outline = [(r*np.cos(t), r*np.sin(t)) for r, t in
                       ((RING_RI, a), (RING_RO, a), (RING_RO, b), (RING_RI, b))]
            shape = prism(stage, path+f'/Collider{j}', outline, -RING_T/2, RING_T/2, steel)
            PhysxSchema.PhysxCollisionAPI(shape.GetPrim()).CreateContactOffsetAttr(.0005)
            UsdGeom.Imageable(shape).MakeInvisible()
            filt = stage.DefinePrim(path+f'/Filter{j}', 'OmniPhysicsElementCollisionFilter')
            filt.GetRelationship('omniphysics:src0').SetTargets([cloth.GetPath()])
            filt.GetRelationship('omniphysics:src1').SetTargets([shape.GetPath()])
            filt.GetAttribute('omniphysics:groupElemCounts0').Set([len(sewn)])
            filt.GetAttribute('omniphysics:groupElemIndices0').Set(sewn.tolist())
            filt.GetAttribute('omniphysics:groupElemCounts1').Set([0])
        att = stage.DefinePrim(path+'/SewnBoundary', 'OmniPhysicsVtxXformAttachment')
        att.GetRelationship('omniphysics:src0').SetTargets([cloth.GetPath()])
        att.GetRelationship('omniphysics:src1').SetTargets([ring.GetPath()])
        att.GetAttribute('omniphysics:vtxIndicesSrc0').Set(holes[i].tolist())
        att.GetAttribute('omniphysics:localPositionsSrc1').Set(Vt.Vec3fArray.FromNumpy((points[holes[i]]-pos).astype(np.float32)))
    stage.Save()


def build():
    (HERE/'assets').mkdir(exist_ok=True)
    build_stand()
    build_rod()
    build_curtain()


if __name__ == '__main__':
    build()
