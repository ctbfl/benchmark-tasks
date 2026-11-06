"""Original procedural experiment geometry; import after Isaac Sim startup."""
import math

from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

from circuit import HERE, brightness

SEGMENTS = ("abcdef", "bc", "abged", "abgcd", "fgbc", "afgcd", "afgecd", "abc", "abcdefg", "abfgcd")
RHEOSTAT_X = .52


def material(stage, path, rgb, emission=0., metal=0.):
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb) * emission)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(.5)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metal)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def finish(shape, mat, collision=False):
    UsdShade.MaterialBindingAPI.Apply(shape.GetPrim()).Bind(mat)
    if collision:
        UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
        p = PhysxSchema.PhysxCollisionAPI.Apply(shape.GetPrim())
        p.CreateContactOffsetAttr(.0005)
        p.CreateRestOffsetAttr(0.)
    return shape


def box(stage, path, pos, size, mat, collision=False):
    shape = UsdGeom.Cube.Define(stage, path)
    shape.CreateSizeAttr(1.)
    shape.AddTranslateOp().Set(Gf.Vec3d(*pos))
    shape.AddScaleOp().Set(Gf.Vec3f(*size))
    return finish(shape, mat, collision)


def cylinder(stage, path, pos, radius, length, mat, axis="Z", collision=False):
    shape = UsdGeom.Cylinder.Define(stage, path)
    shape.CreateRadiusAttr(radius)
    shape.CreateHeightAttr(length)
    shape.CreateAxisAttr(axis)
    shape.AddTranslateOp().Set(Gf.Vec3d(*pos))
    return finish(shape, mat, collision)


def wire(stage, path, points, mat, radius=.0025):
    shape = UsdGeom.BasisCurves.Define(stage, path)
    shape.CreateTypeAttr("linear")
    shape.CreateCurveVertexCountsAttr([len(points)])
    shape.CreatePointsAttr([Gf.Vec3f(*p) for p in points])
    shape.CreateWidthsAttr([2 * radius])
    shape.SetWidthsInterpolation("constant")
    return finish(shape, mat)


def label(stage, path, text, center, size):
    """A top-facing text label, readable from the robot side of the table."""
    from PIL import Image, ImageDraw, ImageFont
    filename = path.rsplit("/", 1)[-1] + ".png"
    image = Image.new("RGB", (800, 160), (220, 225, 224))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 52)
    draw.text((400, 80), text, font=font, anchor="mm", fill=(15, 25, 32))
    image.save(HERE / "assets" / filename)
    mat = material(stage, path + "Material", (1, 1, 1))
    texture = UsdShade.Shader.Define(stage, path + "Material/Texture")
    texture.CreateIdAttr("UsdUVTexture")
    texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(filename))
    reader = UsdShade.Shader.Define(stage, path + "Material/UV")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
    shader = UsdShade.Shader(stage.GetPrimAtPath(path + "Material/Shader"))
    shader.GetInput("diffuseColor").ConnectToSource(texture.ConnectableAPI(), "rgb")
    x, y, z = center
    w, h = size
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([(x-h/2, y+w/2, z), (x-h/2, y-w/2, z),
                          (x+h/2, y-w/2, z), (x+h/2, y+w/2, z)])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateSubdivisionSchemeAttr("none")
    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
        "vertex").Set([(0, 0), (1, 0), (1, 1), (0, 1)])
    finish(mesh, mat)


def rigid(stage, path, mass):
    prim = UsdGeom.Xform.Define(stage, path).GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(mass)
    PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
    return prim


def build():
    """Build portable USDs and original label textures, deterministically."""
    for filename, root in (("circuit.usda", "/Circuit"), ("rheostat.usda", "/Rheostat")):
        stage = Usd.Stage.CreateNew(str(HERE / "assets" / filename))
        UsdGeom.SetStageMetersPerUnit(stage, 1.)
        UsdGeom.SetStageUpAxis(stage, "Z")
        prim = UsdGeom.Xform.Define(stage, root).GetPrim()
        stage.SetDefaultPrim(prim)
        colors = dict(dark=(.025,.035,.045), ceramic=(.72,.73,.67),
                      brass=(.52,.32,.10), red=(.65,.015,.01), black=(.008,.009,.01),
                      blue=(.02,.22,.50), white=(.8,.85,.84), bulb=(.32,.30,.23))
        mats = {name: material(stage, root + "/Looks/" + name, rgb)
                for name, rgb in colors.items()}
        if filename == "rheostat.usda":
            UsdPhysics.ArticulationRootAPI.Apply(prim)
            rigid(stage, root + "/base", .65)
            box(stage, root + "/base/plinth", (0,0,.014), (.12,.41,.028), mats["dark"], True)
            box(stage, root + "/base/press_pad", (0,.175,.031), (.09,.055,.006), mats["blue"], True)
            for y in (-.15,.15):
                name = "right" if y < 0 else "left"
                box(stage, root+f"/base/{name}_support", (.015,y,.05), (.065,.018,.05), mats["ceramic"], True)
            cylinder(stage, root+"/base/core", (.025,0,.056), .015,.28,mats["ceramic"], "Y")
            coil = [(.025 + .0155*math.cos(t), -.14+.28*i/1200,
                     .056 + .0155*math.sin(t)) for i in range(1201)
                    for t in [i*2*math.pi*40/1200]]
            wire(stage, root+"/base/resistance_wire", coil, mats["brass"], .0011)
            cylinder(stage, root+"/base/conductor_rail", (-.015,0,.077), .004,.31,mats["brass"], "Y")
            for name, x, y, mat in (("return_terminal",-.015,.16,"black"),
                                    ("resistor_terminal",.025,-.16,"red")):
                cylinder(stage, root+"/base/"+name, (x,y,.053), .008,.028,mats[mat])
            rigid(stage, root+"/slider", .06)
            box(stage, root+"/slider/grip", (-.015,0,.108), (.024,.032,.044), mats["blue"], True)
            box(stage, root+"/slider/contact", (.005,0,.079), (.05,.012,.008), mats["brass"])
            joint = UsdPhysics.PrismaticJoint.Define(stage, root+"/slide")
            joint.CreateBody0Rel().SetTargets([root+"/base"])
            joint.CreateBody1Rel().SetTargets([root+"/slider"])
            joint.CreateAxisAttr("Y")
            joint.CreateLowerLimitAttr(-.13)
            joint.CreateUpperLimitAttr(.13)
            joint.CreateCollisionEnabledAttr(False)
            axis = joint.GetPrim()
            assert axis.ApplyAPI("PhysxJointAxisAPI", "linear")
            for name, value in (("staticFrictionEffort",.45), ("dynamicFrictionEffort",.3)):
                axis.CreateAttribute("physxJointAxis:linear:"+name, Sdf.ValueTypeNames.Float).Set(value)
            label(stage, root+"/base/RheostatLabel", "280 ohm", (-.044,0,.029), (.16,.021))
        else:
            box(stage, root+"/battery", (.31,.27,.035), (.16,.105,.07), mats["dark"], True)
            label(stage, root+"/BatteryLabel", "15 V DC", (.31,.27,.071), (.095,.04))
            box(stage, root+"/lamp_base", (.21,0,.012), (.105,.14,.024), mats["ceramic"], True)
            cylinder(stage, root+"/lamp_socket", (.21,0,.04), .022,.035,mats["brass"], collision=True)
            bulb = UsdGeom.Sphere.Define(stage, root+"/bulb")
            bulb.CreateRadiusAttr(.029)
            bulb.AddTranslateOp().Set(Gf.Vec3d(.21,0,.084))
            finish(bulb, mats["bulb"], True)
            light = UsdLux.SphereLight.Define(stage, root+"/bulb_light")
            light.AddTranslateOp().Set(Gf.Vec3d(.21,0,.084))
            light.CreateRadiusAttr(.031)
            light.CreateColorAttr(Gf.Vec3f(1,.6,.12))
            light.CreateIntensityAttr(0.)
            label(stage, root+"/BulbLabel", "5 V BULB", (.163,0,.026), (.105,.018))
            box(stage, root+"/meter", (.365,0,.023), (.12,.245,.046), mats["dark"], True)
            box(stage, root+"/screen", (.365,0,.047), (.085,.21,.003), mats["black"])
            lit = material(stage, root+"/Looks/lit", (.10,.85,.55), 1.8)
            material(stage, root+"/Looks/unlit", (0,0,0))
            # Three 60 mm tall seven-segment digits, on the physical meter surface.
            coords = dict(a=(0,.028,.025,.006),g=(0,0,.025,.006),d=(0,-.028,.025,.006),
                          b=(.016,.014,.006,.022),c=(.016,-.014,.006,.022),
                          f=(-.016,.014,.006,.022),e=(-.016,-.014,.006,.022))
            for i, u in enumerate((-.065,-.018,.043)):
                for seg,(du,v,w,h) in coords.items():
                    box(stage, f"{root}/digits/d{i}/{seg}", (.365+v,-u-du,.049), (h,w,.001), lit)
            box(stage, root+"/decimal", (.339,-.009,.049), (.007,.007,.001), lit)
            label(stage, root+"/MeterLabel", "VOLTS", (.417,0,.048), (.14,.017))
            paths = {
                "supply_positive": ([(.265,.27,.071),(.25,.27,.012),(.21,.07,.026)], "red"),
                "lamp_to_resistor": ([(.21,-.07,.026),(.26,-.23,.008),(.545,-.23,.008),(.545,-.16,.053)],"red"),
                "rail_to_negative": ([(.505,.16,.053),(.48,.23,.008),(.36,.27,.071)],"black"),
                "meter_positive": ([(.365,.12,.025),(.28,.12,.008),(.21,.07,.026)],"red"),
                "meter_negative": ([(.365,-.12,.025),(.28,-.12,.008),(.21,-.07,.026)],"black"),
            }
            for name,(points,color) in paths.items():
                wire(stage, root+"/"+name, points, mats[color])
            for i,(x,y,z,color) in enumerate(((.265,.27,.071,"red"),(.36,.27,.071,"black"),
                    (.21,.07,.026,"red"),(.21,-.07,.026,"black"),
                    (.365,.12,.025,"red"),(.365,-.12,.025,"black"))):
                cylinder(stage, root+f"/terminal{i}", (x,y,z), .007,.007,mats[color])
        stage.GetRootLayer().Save()


class Visuals:
    def __init__(self, stage):
        root = "/World/envs/env_0/circuit"
        self.stage, self.root, self.last = stage, root, None
        self.lit = UsdShade.Material(stage.GetPrimAtPath(root+"/Looks/lit"))
        self.unlit = UsdShade.Material(stage.GetPrimAtPath(root+"/Looks/unlit"))
        self.bulb = UsdShade.Shader(stage.GetPrimAtPath(root+"/Looks/bulb/Shader"))
        self.light = UsdLux.SphereLight(stage.GetPrimAtPath(root+"/bulb_light"))

    def update(self, volts):
        digits = f"{volts:04.1f}".replace(".", "")
        if digits == self.last:
            return
        self.last = digits
        for i,digit in enumerate(digits):
            for seg in "abcdefg":
                prim = self.stage.GetPrimAtPath(f"{self.root}/digits/d{i}/{seg}")
                UsdShade.MaterialBindingAPI(prim).Bind(self.lit if seg in SEGMENTS[int(digit)] else self.unlit)
        level = brightness(volts)
        self.bulb.GetInput("emissiveColor").Set(Gf.Vec3f(1,.62,.15) * (18*level))
        self.light.GetIntensityAttr().Set(1800*level)
