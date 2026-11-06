"""Original task assets, in metres. Import after starting Isaac Sim.

The tape case and metal hook are physical. A passive slider represents short,
straight blade travel; the exposed graduated blade is visual only.
"""

from pathlib import Path
import math

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
WOOD_SIZE = (.173, .070, .070)
TAPE_TRAVEL = .300
TAPE_MOUTH = -.039
HOOK_HOME = -.047
BLADE_Z = -.022
TAPE_STIFFNESS = 5.0
TAPE_DAMPING = .15
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
    canvas = Image.new("RGB", (3300, 220), (240, 218, 90))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(FONT, 56)
    for mm in range(331):
        length = 76 if mm % 10 == 0 else (56 if mm % 5 == 0 else 33)
        draw.line((mm*10, 0, mm*10, length), fill=(12, 12, 12), width=3)
        draw.line((mm*10, 220, mm*10, 220-20), fill=(12, 12, 12), width=2)
        if mm % 10 == 0 and 0 < mm < 330:
            draw.text((mm*10, 128), str(mm//10), fill=(12, 12, 12), font=font, anchor="mm")
    canvas.save(folder / "metric_blade.png")
    rng = np.random.default_rng(0)
    x, y = np.meshgrid(np.linspace(0, 25, 1024), np.linspace(0, 80, 192))
    grain = np.sin(y + .28*np.sin(x)) + .4*np.sin(3*y+.1*x) + rng.normal(0, .2, x.shape)
    rgb = np.clip(np.array([185, 137, 83])[None, None] + grain[..., None]*7, 0, 255).astype(np.uint8)
    Image.fromarray(rgb).save(folder / "wood.png")


def build_tape():
    stage, root = stage_for("tape", True)
    yellow = material(stage, root + "/Looks/Yellow", (.75, .48, .015))
    rubber = material(stage, root + "/Looks/Rubber", (.025, .027, .029), roughness=.75, friction=.9)
    steel = material(stage, root + "/Looks/Steel", (.58, .61, .65), metallic=.85, roughness=.25)
    rigid(stage, root + "/case", (0, 0, 0), .17)
    shell = rounded_box((.076, .068, .040), .015)
    shell.apply_transform(trimesh.transformations.rotation_matrix(math.pi/2, [1, 0, 0]))
    mesh_prim(stage, root + "/case/body", shell, rubber)
    for y in (-.0203, .0203):
        cover = rounded_box((.065, .056, .0015), .010)
        cover.apply_transform(trimesh.transformations.rotation_matrix(math.pi/2, [1, 0, 0]))
        cover.apply_translation((0, y, .002))
        mesh_prim(stage, root + f"/case/cover_{'front' if y < 0 else 'back'}", cover, yellow, False)
    box(stage, root + "/case/mouth", (.002, .031, .006), (-.0385, 0, BLADE_Z), steel, False)
    box(stage, root + "/case/heel", (.045, .041, .002), (.006, 0, -.033), rubber, radius=.001)
    rigid(stage, root + "/hook", (HOOK_HOME, 0, BLADE_Z), .010)
    box(stage, root + "/hook/tongue", (.022, .027, .0012), (.0104, 0, .0006), steel)
    box(stage, root + "/hook/lip", (.0012, .027, .010), (-.0006, 0, -.0044), steel)
    slider(stage, root, "extension", "case", "hook", (HOOK_HOME, 0, BLADE_Z), 0., TAPE_TRAVEL,
           TAPE_STIFFNESS, TAPE_DAMPING, target=-.020, outward=True)
    mat = texture_material(stage, root + "/Looks/MetricBlade", "metric_blade.png")
    quad(stage, root + "/case/blade", [(HOOK_HOME, -.011, BLADE_Z+.0004),
         (TAPE_MOUTH, -.011, BLADE_Z+.0004), (TAPE_MOUTH, .011, BLADE_Z+.0004),
         (HOOK_HOME, .011, BLADE_Z+.0004)], mat,
         ((0, 0), ((TAPE_MOUTH-HOOK_HOME)/.330, 0), ((TAPE_MOUTH-HOOK_HOME)/.330, 1), (0, 1)))
    stage.GetDefaultPrim().SetCustomDataByKey("fidelity", "Physical case and hook; passive linear return spring; visual straight blade, no middle-blade contact or internal coil")
    stage.GetRootLayer().Save()


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


def build_wood():
    stage, root = stage_for("wood", False)
    rigid(stage, root + "/bar", (0, 0, 0), .032*WOOD_SIZE[1]*WOOD_SIZE[2]/(.025*.012))
    mat = texture_material(stage, root + "/Looks/Wood", "wood.png")
    obj = box(stage, root + "/bar/body", WOOD_SIZE, (0, 0, 0), mat)
    vertices = np.array(obj.GetPointsAttr().Get())
    uv = np.c_[vertices[:, 0]/WOOD_SIZE[0]+.5, vertices[:, 1]/WOOD_SIZE[1]+.5]
    UsdGeom.PrimvarsAPI(obj).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
                                       UsdGeom.Tokens.vertex).Set(Vt.Vec2fArray.FromNumpy(uv.astype(np.float32)))
    stage.GetRootLayer().Save()


def build_all():
    ASSETS.mkdir(exist_ok=True)
    make_textures()
    build_tape()
    build_calculator()
    build_wood()
    print("Built tape, calculator and 173 mm wood assets", flush=True)
