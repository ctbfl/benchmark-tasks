"""Small mesh and material helpers for the referenced household objects."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt

TEXTURES = Path(__file__).resolve().parent / "assets/textures"


def panel_texture():
    """Generate the scale's brushed finish and fixed printed markings."""
    rng = np.random.default_rng(4)
    w, h = 1024, 1320
    grain = rng.normal(0, 5, (1, w, 1)) + rng.normal(0, 1, (h, w, 1))
    pixels = np.clip(np.array([163, 166, 169]) + grain, 0, 255).astype(np.uint8)
    image = Image.fromarray(pixels)
    draw = ImageDraw.Draw(image)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    def label(text, xy, size):
        draw.text(xy, text, font=ImageFont.truetype(font_path, size), fill=(242, 243, 245), anchor="mm")
    for x in (265, 759):
        draw.rounded_rectangle((x-74, 1042, x+74, 1162), radius=24, outline=(20, 22, 24), width=7)
    label("Z/T", (265, 1077), 24)
    draw.line((216, 1101, 314, 1101), fill=(242, 243, 245), width=2)
    label("ON/OFF", (265, 1129), 23)
    label("UNIT", (759, 1102), 30)
    label("MAX=5000g", (512, 1017), 23)
    label("Weight   Milk   Water", (512, 1200), 20)
    label("ETEKCITY", (530, 1253), 34)
    draw.polygon(((395, 1238), (410, 1253), (395, 1268), (380, 1253)), fill=(242, 243, 245))
    TEXTURES.mkdir(exist_ok=True)
    image.save(TEXTURES / "scale-panel.png")


def textured(stage, path, filename, metallic=0.0, roughness=0.35):
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    reader = UsdShade.Shader.Define(stage, path + "/UV")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    tex = UsdShade.Shader.Define(stage, path + "/Texture")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(TEXTURES / filename)))
    tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def glass(stage, path, ior=1.49, color=(0.98, 0.995, 1.0), roughness=0.025):
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/Shader")
    shader.SetSourceAsset(Sdf.AssetPath("OmniGlass.mdl"), "mdl")
    shader.SetSourceAssetSubIdentifier("OmniGlass", "mdl")
    shader.CreateInput("glass_color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("glass_ior", Sdf.ValueTypeNames.Float).Set(ior)
    shader.CreateInput("frosting_roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("thin_walled", Sdf.ValueTypeNames.Bool).Set(False)
    mat.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
    return mat


def mesh(stage, path, points, faces, mat, uv_size=None, smooth=False):
    points = np.asarray(points, dtype=np.float32)
    shape = UsdGeom.Mesh.Define(stage, path)
    shape.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points))
    shape.CreateFaceVertexCountsAttr([len(f) for f in faces])
    shape.CreateFaceVertexIndicesAttr([i for f in faces for i in f])
    shape.CreateSubdivisionSchemeAttr("none")
    UsdShade.MaterialBindingAPI.Apply(shape.GetPrim()).Bind(mat)
    if uv_size:
        uv = points[:, :2] / np.asarray(uv_size) + 0.5
        UsdGeom.PrimvarsAPI(shape).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "vertex").Set(Vt.Vec2fArray.FromNumpy(uv.astype(np.float32)))
    if smooth:
        normals = np.zeros_like(points)
        for face in faces:
            a, b, c = points[face[:3]]
            normal = np.cross(b-a, c-a)
            normals[face] += normal
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        normals /= np.maximum(lengths, 1e-12)
        shape.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(normals))
        shape.SetNormalsInterpolation("vertex")
    return shape


def lathe(stage, path, profile, mat, position=(0, 0, 0), ribs=0, count=128):
    """Revolve a closed radial/z profile, optionally with fine vertical fluting."""
    points = []
    for radius, z in profile:
        for angle in np.arange(count)*2*np.pi/count:
            r = radius + (0.00025*np.cos(ribs*angle) if ribs else 0)
            points.append((r*np.cos(angle), r*np.sin(angle), z))
    faces = []
    for ring in range(len(profile)):
        nxt = (ring+1) % len(profile)
        for i in range(count):
            j = (i+1) % count
            faces.append([ring*count+i, ring*count+j, nxt*count+j, nxt*count+i])
    shape = mesh(stage, path, points, faces, mat, (0.20, 0.20), smooth=True)
    shape.AddTranslateOp().Set(Gf.Vec3d(*position))
    return shape


def rounded_box(stage, path, pos, size, radius, mat, bevel=0.001):
    w, d, h = size
    points = []
    for inset, z in ((bevel, -h/2), (0, -h/2+bevel), (0, h/2-bevel), (bevel, h/2)):
        r = radius-inset
        for cx, cy, start in ((w/2-radius, d/2-radius, 0), (-w/2+radius, d/2-radius, 90),
                              (-w/2+radius, -d/2+radius, 180), (w/2-radius, -d/2+radius, 270)):
            for a in np.linspace(start, start+90, 9):
                points.append((cx+r*np.cos(np.deg2rad(a)), cy+r*np.sin(np.deg2rad(a)), z))
    n = 36
    faces = [list(reversed(range(n))), list(range(3*n, 4*n))]
    for ring in range(3):
        for i in range(n):
            j = (i+1) % n
            faces.append([ring*n+i, ring*n+j, (ring+1)*n+j, (ring+1)*n+i])
    shape = mesh(stage, path, points, faces, mat, (w, d))
    shape.AddTranslateOp().Set(Gf.Vec3d(*pos))
    return shape


def tube(stage, path, centers, radius, mat):
    centers = np.asarray(centers, dtype=float)
    points = []
    for i, p in enumerate(centers):
        tangent = centers[min(i+1,len(centers)-1)]-centers[max(i-1,0)]
        tangent /= np.linalg.norm(tangent)
        normal = np.cross(tangent, [1,0,0])
        for a in np.arange(32)*2*np.pi/32:
            points.append(p+radius*(np.array([1,0,0])*np.cos(a)+normal*np.sin(a)))
    faces = []
    for i in range(len(centers)-1):
        for j in range(32):
            k = (j+1)%32
            faces.append([i*32+j, i*32+k, (i+1)*32+k, (i+1)*32+j])
    return mesh(stage, path, points, faces, mat, smooth=True)
