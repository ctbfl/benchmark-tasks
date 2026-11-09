"""Original recessed-edge sliding-puzzle assets, in metres."""
from pathlib import Path
import math
import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

ASSETS = Path(__file__).resolve().parent / "assets"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

def material(stage, path, color, metallic=0., roughness=.45, friction=.7):
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    physics = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    physics.CreateStaticFrictionAttr(friction)
    physics.CreateDynamicFrictionAttr(.85*friction)
    physics.CreateRestitutionAttr(0.)
    return mat


def texture_material(stage, path, filename):
    mat = material(stage, path, (1., 1., 1.), roughness=.8)
    shader = UsdShade.Shader.Get(stage, path + "/Surface")
    uv = UsdShade.Shader.Define(stage, path + "/UV")
    uv.CreateIdAttr("UsdPrimvarReader_float2")
    uv.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    tex = UsdShade.Shader.Define(stage, path + "/Texture")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath("textures/" + filename))
    tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("clamp")
    tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(uv.ConnectableAPI(), "result")
    shader.GetInput("diffuseColor").ConnectToSource(tex.ConnectableAPI(), "rgb")
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
        api.CreateContactOffsetAttr(.0002)
        api.CreateRestOffsetAttr(0.)
    return obj


def rounded_box(size, radius, center=(0., 0., 0.)):
    w, d, h = size
    radius = min(radius, w/2, d/2)
    outline = []
    for x, y, start in ((1, 1, 0), (-1, 1, 90), (-1, -1, 180), (1, -1, 270)):
        for angle in np.linspace(start, start+90, 9):
            a = math.radians(angle)
            outline.append((x*(w/2-radius)+radius*math.cos(a),
                            y*(d/2-radius)+radius*math.sin(a)))
    n = len(outline)
    vertices = [(x, y, z) for z in (-h/2, h/2) for x, y in outline]
    vertices += [(0, 0, -h/2), (0, 0, h/2)]
    faces = []
    for i in range(n):
        j = (i+1) % n
        faces += [(2*n, j, i), (2*n+1, n+i, n+j), (i, j, n+j), (i, n+j, n+i)]
    return trimesh.Trimesh(np.asarray(vertices)+center, faces, process=False)


def box(stage, path, size, center, mat, collision=True, radius=0.):
    geom = rounded_box(size, radius, center) if radius else trimesh.creation.box(size)
    if not radius:
        geom.apply_translation(center)
    return mesh_prim(stage, path, geom, mat, collision)


def quad(stage, path, points, mat, uv=((0, 0), (1, 0), (1, 1), (0, 1))):
    obj = UsdGeom.Mesh.Define(stage, path)
    obj.CreatePointsAttr(points)
    obj.CreateFaceVertexCountsAttr([4])
    obj.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    obj.CreateSubdivisionSchemeAttr("none")
    obj.CreateDoubleSidedAttr(True)
    UsdGeom.PrimvarsAPI(obj).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
                                       UsdGeom.Tokens.vertex).Set(uv)
    UsdShade.MaterialBindingAPI.Apply(obj.GetPrim()).Bind(mat)
    return obj


def rigid(stage, path, position, mass):
    obj = UsdGeom.Xform.Define(stage, path)
    obj.AddTranslateOp().Set(Gf.Vec3d(*position))
    UsdPhysics.RigidBodyAPI.Apply(obj.GetPrim())
    UsdPhysics.MassAPI.Apply(obj.GetPrim()).CreateMassAttr(mass)
    PhysxSchema.PhysxRigidBodyAPI.Apply(obj.GetPrim()).CreateEnableCCDAttr(True)
    return obj



def stage_for(name):
    stage = Usd.Stage.CreateNew(str(ASSETS/(name+'.usda')))
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdGeom.SetStageUpAxis(stage, 'Z')
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.)
    root = '/'+name
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, root).GetPrim())
    return stage, root


def build_all():
    from task import PITCH, FLOOR, TILE_HEIGHT
    ASSETS.mkdir(exist_ok=True)
    (ASSETS/'textures').mkdir(exist_ok=True)
    stage, root = stage_for('tray')
    shell = material(stage, root+'/Shell', (.055, .105, .14), friction=.30)
    floor = material(stage, root+'/Floor', (.18, .26, .28), friction=.25)
    inner = 3*PITCH+.002
    outer = inner+.018
    box(stage, root+'/base', (outer, outer, FLOOR), (0, 0, FLOOR/2), floor)
    for axis in range(2):
        for sign in (-1, 1):
            size = [.009, outer, .016] if axis == 0 else [inner, .009, .016]
            center = [0., 0., FLOOR+.008]
            center[axis] = sign*(inner+.009)/2
            box(stage, root+f'/wall_{axis}_{sign+1}', size, center, shell, radius=.002)
    stage.GetRootLayer().Save()
    for number in range(1, 9):
        stage, root = stage_for(f'tile_{number}')
        body = root+'/body'
        rigid(stage, body, (0, 0, 0), .055)
        base = material(stage, root+'/Base', (.36, .53, .57), friction=.30)
        top = material(stage, root+'/Top', (.88, .76, .52), friction=.65)
        box(stage, body+'/foot', (.070, .070, .007), (0, 0, .0035), base, radius=.002)
        box(stage, body+'/face', (.042, .042, .020), (0, 0, .017), top, radius=.002)
        img = Image.new('RGB', (256, 256), (241, 222, 178))
        draw = ImageDraw.Draw(img)
        draw.text((128, 128), str(number), font=ImageFont.truetype(FONT, 190),
                  anchor='mm', fill=(23, 39, 46))
        img.save(ASSETS/'textures'/f'{number}.png')
        ink = texture_material(stage, root+'/Label', f'{number}.png')
        h = .019
        quad(stage, body+'/number', [(-h,-h,TILE_HEIGHT+.00005),(h,-h,TILE_HEIGHT+.00005),
             (h,h,TILE_HEIGHT+.00005),(-h,h,TILE_HEIGHT+.00005)], ink)
        stage.GetRootLayer().Save()
