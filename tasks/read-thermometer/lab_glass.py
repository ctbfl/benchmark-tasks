"""Procedural glass laboratory assets, in metres with the reading face toward -X."""

import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets" / "lab_glass"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
THERMOMETER = dict(height_m=.320, radius_m=.0045, bulb_center_m=.010,
                   scale_bottom_m=.065, scale_top_m=.295,
                   min_celsius=-20, max_celsius=110, clamp_height_m=.305)
BEAKER = dict(height_m=.120, outer_radius_m=.045, inner_radius_m=.042,
              capacity_ml=500, water_height_m=.078, floor_height_m=.005)


def _bind(geom, material):
    UsdShade.MaterialBindingAPI.Apply(geom.GetPrim()).Bind(material)


def _surface(stage, path, color, roughness=.4):
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(color)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _glass(stage, path, color=(1., 1., 1.), ior=1.47, thin=False):
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/Shader")
    shader.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
    shader.SetSourceAsset(Sdf.AssetPath("OmniGlass.mdl"), "mdl")
    shader.SetSourceAssetSubIdentifier("OmniGlass", "mdl")
    shader.CreateInput("glass_color", Sdf.ValueTypeNames.Color3f).Set(color)
    shader.CreateInput("glass_ior", Sdf.ValueTypeNames.Float).Set(ior)
    shader.CreateInput("thin_walled", Sdf.ValueTypeNames.Bool).Set(thin)
    shader.CreateInput("depth", Sdf.ValueTypeNames.Float).Set(0.)
    shader.CreateInput("reflection_color", Sdf.ValueTypeNames.Color3f).Set((.72,.72,.72))
    shader.CreateInput("frosting_roughness", Sdf.ValueTypeNames.Float).Set(.015)
    for output in ("surface", "volume", "displacement"):
        material.CreateOutput("mdl:" + output, Sdf.ValueTypeNames.Token).ConnectToSource(
            shader.ConnectableAPI(), "out")
    return material


def _water(stage, path):
    """Mostly transmissive dyed water with a small diffuse tint for visual observation."""
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path+"/Shader")
    shader.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
    shader.SetSourceAsset(Sdf.AssetPath("OmniSurface.mdl"), "mdl")
    shader.SetSourceAssetSubIdentifier("OmniSurface", "mdl")
    for key, value in {"diffuse_reflection_weight": 1.,
                       "specular_reflection_weight": 1.,
                       "specular_reflection_roughness": .025,
                       "specular_reflection_ior": 1.333,
                       "specular_transmission_weight": .88,
                       "specular_transmission_scattering_depth": .30,
                       "metalness": 0.}.items():
        shader.CreateInput(key,Sdf.ValueTypeNames.Float).Set(value)
    for key,value in {"diffuse_reflection_color": (.95,.45,.58),
                      "specular_transmission_color": (1.,.93,.96),
                      "specular_transmission_scattering_color": (.45,.22,.29)}.items():
        shader.CreateInput(key,Sdf.ValueTypeNames.Color3f).Set(value)
    shader.CreateInput("enable_specular_transmission",Sdf.ValueTypeNames.Bool).Set(True)
    for output in ("surface", "volume", "displacement"):
        material.CreateOutput("mdl:"+output,Sdf.ValueTypeNames.Token).ConnectToSource(
            shader.ConnectableAPI(),"out")
    return material


def _texture(stage, path, filename):
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(.7)
    shader.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(1.)
    shader.CreateInput("opacityThreshold", Sdf.ValueTypeNames.Float).Set(.5)
    texture = UsdShade.Shader.Define(stage, path + "/Texture")
    texture.CreateIdAttr("UsdUVTexture")
    # Relative references are authored against the layer that contains this asset.
    layer_dir = Path(stage.GetRootLayer().realPath).parent
    import os
    texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath(os.path.relpath(ASSETS / filename, layer_dir)))
    texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    texture.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("clamp")
    texture.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")
    uv = UsdShade.Shader.Define(stage, path + "/UV")
    uv.CreateIdAttr("UsdPrimvarReader_float2")
    uv.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(uv.ConnectableAPI(), "result")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(texture.ConnectableAPI(), "rgb")
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).ConnectToSource(texture.ConnectableAPI(), "a")
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _body(stage, path, mass):
    root = UsdGeom.Xform.Define(stage, path)
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(mass)
    return root


def _collision(geom):
    UsdPhysics.CollisionAPI.Apply(geom.GetPrim())
    geom.GetPrim().AddAppliedSchema("PhysxCollisionAPI")
    geom.GetPrim().CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(.0003)
    geom.GetPrim().CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(0.)


def _cube(stage, path, size, pos, material=None):
    geom = UsdGeom.Cube.Define(stage, path)
    geom.CreateSizeAttr(1.)
    geom.AddTranslateOp().Set(Gf.Vec3d(*pos))
    geom.AddScaleOp().Set(Gf.Vec3f(*size))
    if material:
        _bind(geom, material)
    return geom


def _cylinder(stage, path, radius, height, pos, material=None):
    geom = UsdGeom.Cylinder.Define(stage, path)
    geom.CreateRadiusAttr(radius)
    geom.CreateHeightAttr(height)
    geom.CreateAxisAttr("Z")
    geom.AddTranslateOp().Set(Gf.Vec3d(*pos))
    if material:
        _bind(geom, material)
    return geom


def _sphere(stage, path, scale, pos, material):
    geom = UsdGeom.Sphere.Define(stage, path)
    geom.CreateRadiusAttr(1.)
    geom.AddTranslateOp().Set(Gf.Vec3d(*pos))
    geom.AddScaleOp().Set(Gf.Vec3f(*scale))
    _bind(geom, material)
    return geom


def _lathe(stage, path, profile, material, spout=False, segments=128):
    points = []
    for radius, z in profile:
        for j in range(segments):
            angle = 2 * math.pi * j / segments
            radial, height = radius, z
            if spout:
                distance = math.atan2(math.sin(angle-math.pi/2), math.cos(angle-math.pi/2))
                weight = math.exp(-(distance / .23) ** 2) * max(0, (z-.097)/.023)
                radial += .007 * weight
                height -= .0015 * weight
            points.append((radial*math.cos(angle), radial*math.sin(angle), height))
    indices = []
    for i in range(len(profile)-1):
        for j in range(segments):
            indices.extend((i*segments+j, i*segments+(j+1)%segments,
                            (i+1)*segments+(j+1)%segments, (i+1)*segments+j))
    geom = UsdGeom.Mesh.Define(stage, path)
    geom.CreatePointsAttr(points)
    geom.CreateFaceVertexCountsAttr([4]*(len(indices)//4))
    geom.CreateFaceVertexIndicesAttr(indices)
    geom.CreateSubdivisionSchemeAttr("none")
    # Average adjacent profile tangents for a smooth vessel wall and rolled rim.
    normals = []
    for i in range(len(profile)):
        prev, nex = profile[max(0, i-1)], profile[min(len(profile)-1, i+1)]
        dr, dz = nex[0]-prev[0], nex[1]-prev[1]
        length = math.hypot(dr, dz)
        for j in range(segments):
            a = 2 * math.pi * j / segments
            normals.append((dz*math.cos(a)/length, dz*math.sin(a)/length, -dr/length))
    geom.CreateNormalsAttr(normals)
    geom.SetNormalsInterpolation("vertex")
    _bind(geom, material)
    return geom


def _plane(stage, path, x, width, bottom, top, material):
    geom = UsdGeom.Mesh.Define(stage, path)
    geom.CreatePointsAttr([(x,width/2,bottom),(x,-width/2,bottom),
                          (x,-width/2,top),(x,width/2,top)])
    geom.CreateFaceVertexCountsAttr([4])
    geom.CreateFaceVertexIndicesAttr([0,1,2,3])
    geom.CreateSubdivisionSchemeAttr("none")
    geom.CreateDoubleSidedAttr(True)
    uv = UsdGeom.PrimvarsAPI(geom).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "vertex")
    uv.Set([(0,0),(1,0),(1,1),(0,1)])
    _bind(geom, material)
    return geom


def make_textures():
    ASSETS.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGBA", (256,8192), (255,255,255,0))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(FONT, 60)
    bottom, top = .055, .302
    for temp in range(-20,111):
        z = .065+(temp+20)/130*.230
        y = round((top-z)/(top-bottom)*8192)
        major, medium = temp%10==0, temp%5==0
        start = 130 if major else (136 if medium else 142)
        draw.line((start,y,181,y),fill=(0,0,0,255),width=22 if major else (18 if medium else 16))
        if major:
            draw.text((121,y),str(temp),anchor="rm",font=font,fill=(0,0,0,255),stroke_width=1)
    draw.text((115,70),"°C",anchor="mt",font=font,fill=(0,0,0,255),stroke_width=1)
    image.save(ASSETS / "thermometer_scale.png")
    image = Image.new("RGBA", (1024,2048), (255,255,255,0))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(FONT, 78)
    draw.text((512,160),"500 mL",anchor="mm",font=font,fill=(220,232,235,255))
    draw.text((512,260),"BORO 3.3",anchor="mm",font=ImageFont.truetype(FONT,48),fill=(220,232,235,255))
    # Scale heights represent the internal cylinder volume, measured above its floor.
    for ml in range(50,501,50):
        z = .005+(ml*1e-6)/(math.pi*.042**2)
        y = round((.106-z)/.100*2048)
        draw.line((550,y,770 if ml%100==0 else 680,y),fill=(215,230,233,255),width=7)
        if ml%100==0:
            draw.text((505,y),str(ml),anchor="rm",font=font,fill=(215,230,233,255))
    image.save(ASSETS / "beaker_marks.png")


def build_beaker(stage, path):
    """Build an open 500 mL beaker with fixed, non-fluid simulated water geometry."""
    root = _body(stage, path, .605)
    glass = _glass(stage, path+"/Looks/Glass")
    water = _water(stage, path+"/Looks/Water")
    profile = [(0.,.001),(.038,.001),(.042,.002),(.044,.004),(.045,.007),
               (.045,.113),(.0455,.116),(.047,.118),(.047,.1195),
               (.046,.120),(.0445,.1195),(.043,.118),(.042,.115),
               (.042,.009),(.041,.006),(.038,.005),(0.,.005)]
    _lathe(stage,path+"/GlassShell",profile,glass,spout=True)
    _lathe(stage,path+"/Water",[(0.,.0052),(.0418,.0052),(.0418,.078),
                              (.0408,.0782),(.0398,.0777),(0.,.0777)],water)
    bottom = _cylinder(stage,path+"/Colliders/Bottom",.043,.004,(0,0,.003))
    _collision(bottom)
    bottom.CreateVisibilityAttr("invisible")
    for j in range(48):
        angle = 2*math.pi*j/48
        center = .0435
        wall = _cube(stage,path+f"/Colliders/Wall{j:02d}",(.003,.0058,.116),
                     (center*math.cos(angle),center*math.sin(angle),.062))
        wall.AddRotateZOp().Set(math.degrees(angle))
        wall.SetXformOpOrder([wall.GetOrderedXformOps()[0],wall.GetOrderedXformOps()[2],wall.GetOrderedXformOps()[1]])
        _collision(wall)
        wall.CreateVisibilityAttr("invisible")
    material = _texture(stage,path+"/Looks/Markings","beaker_marks.png")
    # Markings hug the cylindrical front instead of floating on a broad flat decal.
    mesh = UsdGeom.Mesh.Define(stage,path+"/Markings")
    vertices,uvs,indices = [],[],[]
    for z,v in ((.006,0),(.106,1)):
        for j in range(33):
            u=j/32
            angle=math.pi+(u-.5)*.85
            vertices.append((.04508*math.cos(angle),.04508*math.sin(angle),z))
            uvs.append((u,v))
    for j in range(32):
        indices.extend((j,j+1,34+j,33+j))
    mesh.CreatePointsAttr(vertices)
    mesh.CreateFaceVertexCountsAttr([4]*32)
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st",Sdf.ValueTypeNames.TexCoord2fArray,"vertex").Set(uvs)
    _bind(mesh,material)
    root.GetPrim().SetCustomDataByKey("liquid_model","Fixed visual water volume; no free-surface dynamics or pouring")
    root.GetPrim().SetCustomDataByKey("water_optics","Transmissive tinted water with diffuse scattering for visibility")
    return dict(BEAKER,root_path=path,water_volume_ml=math.pi*.0418**2*(.0777-.0052)*1e6,
                collision_wall_segments=48,fluid_simulation=False)


def build_thermometer(stage, path):
    """Build a red-liquid thermometer with a geometric, continuously variable column."""
    root = _body(stage,path,.022)
    # A thin optical envelope keeps the tiny printed scale readable in real-time RTX.
    # Its collision radius and actual visible exterior are unchanged.
    glass = _glass(stage,path+"/Looks/Glass",thin=True)
    red = _surface(stage,path+"/Looks/Liquid",(.64,.005,.015),.22)
    white = _surface(stage,path+"/Looks/Backing",(.92,.94,.91),.6)
    blue = _surface(stage,path+"/Looks/Cap",(.025,.22,.39),.27)
    profile=[(0.,.001),(.0026,.001),(.0037,.003),(.0041,.008),(.0043,.024),
             (.0045,.036),(.0045,.314),(.0038,.316),(0.,.316)]
    _lathe(stage,path+"/GlassStem",profile,glass,segments=64)
    collider = _cylinder(stage,path+"/Collider",.0045,.308,(0,0,.158))
    _collision(collider)
    collider.CreateVisibilityAttr("invisible")
    _plane(stage,path+"/WhiteBacking",-.0028,.0072,.036,.302,white)
    _plane(stage,path+"/PrintedScale",-.0032,.0072,.055,.302,
           _texture(stage,path+"/Looks/Scale","thermometer_scale.png"))
    _sphere(stage,path+"/Bulb",(.0025,.0025,.008),(-.0011,0,.010),red)
    _cylinder(stage,path+"/Capillary",.00032,.287,(-.0033,-.00175,.1615),
              _surface(stage,path+"/Looks/Capillary",(.75,.74,.72),.45))
    _cylinder(stage,path+"/RedColumn",.00048,.140,(-.00345,-.00175,.088),red)
    # Connector visibly joins the bulb reservoir to the narrow capillary.
    _sphere(stage,path+"/BulbNeck",(.0008,.0008,.008),(-.0022,-.001,.019),red)
    _cylinder(stage,path+"/BlueCap",.0049,.005,(0,0,.3175),blue)
    update_temperature(stage,path,35.)
    root.GetPrim().SetCustomDataByKey("liquid_model","Red spirit column; temperature supplied by task thermal model")
    root.GetPrim().SetCustomDataByKey("glass_optics","Thin-walled optical envelope for real-time scale readability")
    return dict(THERMOMETER,root_path=path,initial_celsius=35.,column_path=path+"/RedColumn",
                glass_optics="thin-walled optical envelope")


def update_temperature(stage, path, celsius):
    """Move the actual liquid meniscus to the matching scale height."""
    if not -20 <= celsius <= 110:
        raise ValueError("Temperature outside -20 to 110 Celsius")
    top = .065+(float(celsius)+20)/130*.230
    bottom = .018
    column = UsdGeom.Cylinder(stage.GetPrimAtPath(path+"/RedColumn"))
    if not column:
        raise ValueError("Thermometer RedColumn is missing")
    column.GetHeightAttr().Set(top-bottom)
    column.GetOrderedXformOps()[0].Set(Gf.Vec3d(-.00345,-.00175,(top+bottom)/2))
    stage.GetPrimAtPath(path).CreateAttribute("thermometer:celsius",Sdf.ValueTypeNames.Double).Set(float(celsius))
    return top


def export_assets():
    make_textures()
    metadata={}
    for name,build in (("beaker",build_beaker),("thermometer",build_thermometer)):
        stage=Usd.Stage.CreateNew(str(ASSETS/(name+".usda")))
        UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage,1.)
        metadata[name]=build(stage,"/"+name.capitalize())
        stage.SetDefaultPrim(stage.GetPrimAtPath("/"+name.capitalize()))
        stage.GetRootLayer().Save()
    (ASSETS/"metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    return metadata


if __name__ == "__main__":
    print(json.dumps(export_assets(),indent=2))
