"""Procedural task assets in metres, including the referenced calculator."""

from pathlib import Path
import math

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
BOARD_SIZE = (.230, .200, .006)
TRIANGLE_BASE = .125
TRIANGLE_HEIGHT = .080
RULER_SIZE = (.160, .026, .012)
RULER_ZERO = -.075
KEY_TRAVEL = .0015
KEY_STIFFNESS = 600.0
KEY_DAMPING = .4
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
KEY_ROWS = (("percent", "7", "8", "9", "divide"),
            ("sqrt", "4", "5", "6", "multiply"),
            ("CE", "1", "2", "3", "minus"),
            ("C", "0", "dot", "sign", "plus"))
LABELS = {"percent": "%", "sqrt": "√", "divide": "÷", "multiply": "×",
          "minus": "−", "plus": "+", "dot": ".", "sign": "+/−", "C": "ON\nC", "equals": "="}
KEYS = {name: ((col-2)*.019, .008-row*.016, .011)
        for row, names in enumerate(KEY_ROWS) for col, name in enumerate(names)}
KEYS["plus"] = (.038, -.046, .011)
KEYS["equals"] = (0., -.057, .011)
SEGMENTS = {"0": "abcedf", "1": "bc", "2": "abged", "3": "abgcd", "4": "fgbc",
            "5": "afgcd", "6": "afgecd", "7": "abc", "8": "abcdefg", "9": "abfgcd",
            "-": "g", "E": "afged", "r": "eg", "o": "egcd", " ": ""}


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


def slider(stage, root, name, parent, child, pivot, low, high, stiffness, damping,
           target=0., outward=False):
    joint = UsdPhysics.PrismaticJoint.Define(stage, root + "/Joints/" + name)
    joint.CreateBody0Rel().SetTargets([root + "/" + parent])
    joint.CreateBody1Rel().SetTargets([root + "/" + child])
    joint.CreateAxisAttr("X" if outward else "Z")
    joint.CreateLocalPos0Attr(Gf.Vec3f(*pivot))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0.))
    rotation = Gf.Quatf(0., 0., 0., 1.) if outward else Gf.Quatf(1.)
    joint.CreateLocalRot0Attr(rotation)
    joint.CreateLocalRot1Attr(rotation)
    joint.CreateLowerLimitAttr(low)
    joint.CreateUpperLimitAttr(high)
    joint.CreateCollisionEnabledAttr(False)
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(stiffness)
    drive.CreateDampingAttr(damping)
    drive.CreateTargetPositionAttr(target)
    drive.CreateMaxForceAttr(10.)
    state = PhysxSchema.JointStateAPI.Apply(joint.GetPrim(), "linear")
    state.CreatePositionAttr(0.)
    state.CreateVelocityAttr(0.)
    return joint


def stage_for(name, articulated):
    path = ASSETS / (name + ".usda")
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.)
    root = "/" + name
    prim = UsdGeom.Xform.Define(stage, root).GetPrim()
    stage.SetDefaultPrim(prim)
    if articulated:
        UsdPhysics.ArticulationRootAPI.Apply(prim)
        PhysxSchema.PhysxArticulationAPI.Apply(prim).CreateEnabledSelfCollisionsAttr(True)
    return stage, root


def make_textures():
    folder = ASSETS / "textures"
    folder.mkdir(parents=True, exist_ok=True)
    for name in KEYS:
        background = (38, 39, 46) if name.isdigit() or name in ("dot", "sign") else (242, 242, 238)
        if name == "C":
            background = (240, 173, 45)
        text_color = (250, 250, 248) if background[0] < 100 else (35, 35, 40)
        canvas = Image.new("RGB", (192, 128), background)
        draw = ImageDraw.Draw(canvas)
        label = LABELS.get(name, name)
        font = ImageFont.truetype(FONT, 60 if len(label) > 1 else 94)
        draw.multiline_text((96, 64), label, fill=text_color, font=font, anchor="mm",
                            align="center", spacing=-8)
        canvas.save(folder / (name + ".png"))

def build_calculator():
    stage, root = stage_for("calculator", True)
    purple = material(stage, root + "/Looks/Purple", (.29, .052, .42))
    dark = material(stage, root + "/Looks/Dark", (.026, .027, .037))
    white = material(stage, root + "/Looks/White", (.88, .88, .85))
    gold = material(stage, root + "/Looks/Gold", (.87, .44, .035))
    rubber = material(stage, root + "/Looks/Feet", (.025, .025, .025), roughness=.9, friction=1.)
    lcd = material(stage, root + "/Looks/LCD", (.45, .52, .44), roughness=.7)
    ink = material(stage, root + "/Looks/Ink", (.008, .017, .019), roughness=.8)
    solar = material(stage, root + "/Looks/Solar", (.1, .065, .048), metallic=.2)
    rigid(stage, root + "/base", (0, 0, 0), .16)
    box(stage, root + "/base/shell", (.1016, .1397, .008), (0, 0, .006), purple, radius=.012)
    for i, (x, y) in enumerate(((-.036, -.052), (.036, -.052), (-.036, .052), (.036, .052))):
        box(stage, root + f"/base/foot_{i}", (.012, .008, .002), (x, y, .001), rubber, radius=.003)
    panel = UsdGeom.Xform.Define(stage, root + "/base/display")
    panel.AddTranslateOp().Set(Gf.Vec3d(0, .047, .014))
    panel.AddRotateXOp().Set(15.)
    box(stage, str(panel.GetPath()) + "/frame", (.095, .031, .004), (0, 0, -.002), purple, False, .004)
    box(stage, str(panel.GetPath()) + "/screen", (.089, .024, .0008), (0, 0, .0005), lcd, False, .002)
    coords = {"a": (0, .007, .0058, .0010), "g": (0, 0, .0058, .0010),
              "d": (0, -.007, .0058, .0010), "b": (.0034, .0035, .0010, .0057),
              "c": (.0034, -.0035, .0010, .0057), "e": (-.0034, -.0035, .0010, .0057),
              "f": (-.0034, .0035, .0010, .0057)}
    for digit in range(8):
        center = (digit-3.5)*.0103
        for letter, (x, y, w, h) in coords.items():
            obj = box(stage, str(panel.GetPath()) + f"/digit_{digit}_{letter}",
                      (w, h, .0001), (center+x, y, .0011), ink, False, .0002)
            if digit != 7 or letter not in SEGMENTS["0"]:
                UsdGeom.Imageable(obj).MakeInvisible()
        obj = box(stage, str(panel.GetPath()) + f"/digit_{digit}_point", (.001, .001, .0001),
                  (center+.0045, -.0075, .0011), ink, False, .0004)
        UsdGeom.Imageable(obj).MakeInvisible()
    box(stage, root + "/base/solar", (.027, .009, .0006), (.029, .025, .0105), solar, False, .001)
    for name, position in KEYS.items():
        rigid(stage, root + "/key_" + name, position, .002)
        size = (.050, .011, .003) if name == "equals" else ((.013, .025, .003) if name == "plus" else (.016, .012, .003))
        mat = dark if name.isdigit() or name in ("dot", "sign") else (gold if name == "C" else white)
        box(stage, root + "/key_" + name + "/cap", size, (0, 0, 0), mat, radius=.004)
        text = texture_material(stage, root + "/Looks/Label_" + name, name + ".png")
        w, h = min(size[0]*.85, .024), min(size[1]*.82, .013)
        quad(stage, root + "/key_" + name + "/label", [(-w/2, -h/2, .0016),
             (w/2, -h/2, .0016), (w/2, h/2, .0016), (-w/2, h/2, .0016)], text)
        slider(stage, root, "press_" + name, "base", "key_" + name, position,
               -KEY_TRAVEL, 0., KEY_STIFFNESS, KEY_DAMPING)
    stage.GetRootLayer().Save()


def build_board():
    stage, root = stage_for("board", False)
    white = material(stage, root+"/Looks/White", (.94, .95, .94), roughness=.35)
    frame = material(stage, root+"/Looks/Frame", (.30, .33, .36), metallic=.7)
    ink = material(stage, root+"/Looks/Ink", (.006, .006, .006), roughness=.9)
    w, h, thickness = BOARD_SIZE
    box(stage, root+"/panel", BOARD_SIZE, (0, 0, thickness/2), white, radius=.004)
    for i, (size, center) in enumerate((
        ((w+.008, .004, thickness), (0, -h/2-.002, thickness/2)),
        ((w+.008, .004, thickness), (0, h/2+.002, thickness/2)),
        ((.004, h, thickness), (-w/2-.002, 0, thickness/2)),
        ((.004, h, thickness), (w/2+.002, 0, thickness/2)),
    )):
        box(stage, root+f"/frame_{i}", size, center, frame, radius=.001)
    a = np.array([-TRIANGLE_BASE/2, -TRIANGLE_HEIGHT/2])
    b = np.array([TRIANGLE_BASE/2, -TRIANGLE_HEIGHT/2])
    c = np.array([TRIANGLE_BASE/2, TRIANGLE_HEIGHT/2])
    # Stroke centres define the true measured vertices.
    for i, (p, q) in enumerate(((a, b), (b, c), (c, a),
                                (b+[-.008, 0], b+[-.008, .008]),
                                (b+[-.008, .008], b+[0, .008]))):
        direction = q-p
        normal = np.array([-direction[1], direction[0]])/np.linalg.norm(direction)*.0008
        quad(stage, root+f"/ink_{i}", [(*v, thickness+.0003)
             for v in (p-normal, q-normal, q+normal, p+normal)], ink)
    stage.GetRootLayer().Save()


def build_ruler():
    width, depth, height = RULER_SIZE
    rng = np.random.default_rng(0)
    x, y = np.meshgrid(np.linspace(0, 15, 2560), np.linspace(0, 90, 416))
    grain = np.sin(y+.2*np.sin(x))+.35*np.sin(3*y+.1*x)+rng.normal(0, .12, x.shape)
    pixels = np.clip(np.array([213, 176, 119])[None, None]+grain[..., None]*6, 0, 255).astype(np.uint8)
    image = Image.fromarray(pixels)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(FONT, 82)
    for mm in range(151):
        px = round((RULER_ZERO+mm/1000+width/2)/width*image.width)
        length = 125 if mm % 10 == 0 else 90 if mm % 5 == 0 else 55
        draw.line((px, 0, px, length), fill=(25, 21, 17), width=3)
        if mm % 10 == 0:
            draw.text((px, 205), str(mm//10), font=font, anchor="mm", fill=(25, 21, 17))
    draw.text((image.width/2, 340), "cm", font=ImageFont.truetype(FONT, 46),
              anchor="mm", fill=(50, 40, 30))
    image.save(ASSETS/"textures/ruler.png")
    stage, root = stage_for("ruler", False)
    mat = material(stage, root+"/Looks/Wood", (.66, .43, .20), roughness=.8)
    rigid(stage, root+"/body", (0, 0, 0), .035)
    box(stage, root+"/body/wood", RULER_SIZE, (0, 0, 0), mat, radius=.001)
    label = texture_material(stage, root+"/Looks/Scale", "ruler.png")
    quad(stage, root+"/body/scale", [(-width/2, -depth/2, height/2+.00003),
        (width/2, -depth/2, height/2+.00003), (width/2, depth/2, height/2+.00003),
        (-width/2, depth/2, height/2+.00003)], label)
    stage.GetRootLayer().Save()


def build_all():
    ASSETS.mkdir(exist_ok=True)
    make_textures()
    build_calculator()
    build_board()
    build_ruler()
    print("Built fixed whiteboard, 15 cm wooden ruler and calculator", flush=True)

