"""Procedural dispenser geometry, generated in metres from the station datum."""
from math import cos, pi, sin

from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from process import AREA, HEIGHT, JET_RADIUS, OUTLET_ABOVE_FLOOR, RADIUS, TARGET

CUP = (.346, .055, .020)
BUTTON = (.3445, -.118, .205)
OUTLET = CUP[2] + OUTLET_ABOVE_FLOOR
STROKE = .008
CABINET_X = .461


def material(stage, path, color, *, metal=0., roughness=.36):
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    for name, value in dict(metallic=metal, roughness=roughness).items():
        shader.CreateInput(name, Sdf.ValueTypeNames.Float).Set(value)
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def clear_material(stage, path):
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/Surface")
    shader.SetSourceAsset(Sdf.AssetPath("OmniGlass.mdl"), "mdl")
    shader.SetSourceAssetSubIdentifier("OmniGlass", "mdl")
    shader.CreateInput("glass_color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.99, .996, 1.))
    shader.CreateInput("glass_ior", Sdf.ValueTypeNames.Float).Set(1.47)
    shader.CreateInput("frosting_roughness", Sdf.ValueTypeNames.Float).Set(.025)
    shader.CreateInput("thin_walled", Sdf.ValueTypeNames.Bool).Set(False)
    mat.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
    return mat


def finish(shape, mat, collides=False):
    prim = shape.GetPrim()
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)
    if collides:
        UsdPhysics.CollisionAPI.Apply(prim)
        PhysxSchema.PhysxCollisionAPI.Apply(prim).CreateContactOffsetAttr(.0008)
    return shape


def box(stage, path, pos, size, mat, collides=False):
    item = UsdGeom.Cube.Define(stage, path)
    item.CreateSizeAttr(1.)
    item.AddTranslateOp().Set(Gf.Vec3d(*pos))
    item.AddScaleOp().Set(Gf.Vec3f(*size))
    return finish(item, mat, collides)


def cylinder(stage, path, pos, radius, height, mat, collides=False, axis="Z"):
    item = UsdGeom.Cylinder.Define(stage, path)
    item.CreateRadiusAttr(radius)
    item.CreateHeightAttr(height)
    item.CreateAxisAttr(axis)
    item.AddTranslateOp().Set(Gf.Vec3d(*pos))
    return finish(item, mat, collides)


def ring(stage, path, profile, mat, collision=False):
    count = 96
    pts = [Gf.Vec3f(CUP[0] + r*cos(i*2*pi/count), CUP[1] + r*sin(i*2*pi/count), CUP[2]+z)
           for r, z in profile for i in range(count)]
    indices = []
    for k in range(len(profile)):
        a, b = k*count, (k+1) % len(profile)*count
        for i in range(count):
            j = (i+1) % count
            indices.extend((a+i, b+i, b+j, a+j))
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(pts)
    mesh.CreateFaceVertexCountsAttr([4] * (len(indices)//4))
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    finish(mesh, mat, collision)
    if collision:
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("none")


def rigid(stage, path, pos, mass):
    obj = UsdGeom.Xform.Define(stage, path)
    obj.AddTranslateOp().Set(Gf.Vec3d(*pos))
    UsdPhysics.RigidBodyAPI.Apply(obj.GetPrim())
    UsdPhysics.MassAPI.Apply(obj.GetPrim()).CreateMassAttr(mass)
    return obj


def fascia_text(stage, root, mat):
    strokes = {
        "C": [(1,1,0,1),(0,1,0,0),(0,0,1,0)],
        "O": [(1,1,0,1),(0,1,0,0),(0,0,1,0),(1,0,1,1)],
        "L": [(0,1,0,0),(0,0,1,0)],
        "A": [(0,0,.5,1),(.5,1,1,0),(.23,.43,.77,.43)],
    }
    points = []
    for i, letter in enumerate("COLA"):
        for y0,z0,y1,z1 in strokes[letter]:
            points += [Gf.Vec3f(.382, .093 - .017*i - .012*y, .218+.017*z)
                       for y,z in ((y0,z0),(y1,z1))]
    lines = UsdGeom.BasisCurves.Define(stage, root + "/Cola")
    lines.CreateTypeAttr("linear")
    lines.CreateCurveVertexCountsAttr([2]*(len(points)//2))
    lines.CreateWidthsAttr([.0025])
    lines.SetWidthsInterpolation("constant")
    lines.CreatePointsAttr(points)
    finish(lines, mat)


def build_asset(file):
    stage = Usd.Stage.CreateNew(str(file))
    root = "/Fountain"
    asset = UsdGeom.Xform.Define(stage, root)
    stage.SetDefaultPrim(asset.GetPrim())
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdGeom.SetStageUpAxis(stage, "Z")
    looks = root + "/Looks/"
    dark = material(stage, looks+"Graphite", (.036,.044,.049))
    black = material(stage, looks+"Rubber", (.015,.018,.020), roughness=.65)
    metal = material(stage, looks+"Steel", (.33,.37,.42), metal=.72)
    red = material(stage, looks+"Vermilion", (.35,.037,.026), roughness=.32)
    cap = material(stage, looks+"RedButton", (.67,.063,.033), roughness=.35)
    white = material(stage, looks+"Ivory", (.90,.88,.78))
    glass = clear_material(stage, looks+"ClearCup")
    line = material(stage, looks+"FillMark", (.14,.63,.77), roughness=.65)
    def body(name, pos, size, mat):
        return box(stage, root+"/"+name, pos, size, mat, True)

    body("RearHousing", (CABINET_X,-.027,.142), (.135,.268,.284), dark)
    body("Lid", (.464,-.027,.287), (.146,.278,.014), metal)
    body("SplashBack", (.390,-.005,.101), (.009,.198,.182), metal)
    body("ValveHead", (.444,-.021,.224), (.117,.271,.050), dark)
    box(stage, root+"/FlavorPanel", (.385,-.027,.225), (.003,.259,.037), red)
    fascia_text(stage, root+"/FlavorPanel", white)
    body("SwitchHousing", (.379,BUTTON[1],BUTTON[2]), (.040,.073,.062), dark)
    cylinder(stage, root+"/SwitchBezel", (.352,BUTTON[1],BUTTON[2]), .023,.004,metal, axis="X")
    body("OutletArm", (.370,CUP[1],.202), (.071,.052,.023), black)
    cylinder(stage, root+"/Diffuser", (CUP[0],CUP[1],OUTLET+.009), .017,.018,dark,True)
    cylinder(stage, root+"/Outlet", (CUP[0],CUP[1],OUTLET-.0001), .009,.001,black)
    body("DripTray", (.368,.014,.006), (.236,.239,.012), black)
    for i in range(13):
        body(f"Grate{i}", (.368,-.092+.0175*i,.014), (.220,.010,.004), metal)
    for i, y in enumerate((-.106,.134)):
        body(f"TrayEdge{i}", (.368,y,.009), (.244,.006,.018), metal)
    body("FrontTrayEdge", (.247,.014,.009), (.006,.240,.018), metal)
    cylinder(stage, root+"/Locator", (CUP[0],CUP[1],.0165), .043,.003,black,True)
    cylinder(stage, root+"/CupBase", (CUP[0],CUP[1],.0185), .037,.003,glass,True)
    ring(stage, root+"/CupWall", [(RADIUS,0.), (RADIUS,HEIGHT), (.037,HEIGHT), (.037,0.)],
         glass, True)
    ring(stage, root+"/RolledRim", [(.035,HEIGHT-.0012), (.035,HEIGHT), (.0377,HEIGHT),
                                  (.038,HEIGHT-.0012), (.0377,HEIGHT-.0025)], white)
    mark = TARGET / AREA
    ring(stage, root+"/FillMark", [(.03705,mark-.001),(.03705,mark+.001),
                                  (.03722,mark+.001),(.03722,mark-.001)], line)

    switch = root + "/Switch"
    art = UsdGeom.Xform.Define(stage, switch)
    UsdPhysics.ArticulationRootAPI.Apply(art.GetPrim())
    fixed, button = switch+"/Mount", switch+"/Plunger"
    rigid(stage, fixed, BUTTON, .5)
    rigid(stage, button, BUTTON, .026)
    cylinder(stage, button+"/Cap", (0,0,0), .019,.009,cap,True,axis="X")
    cylinder(stage, button+"/Face", (-.0047,0,0), .015,.0005,cap,axis="X")
    anchor = UsdPhysics.FixedJoint.Define(stage, switch+"/Fixed")
    anchor.CreateBody1Rel().SetTargets([fixed])
    joint = UsdPhysics.PrismaticJoint.Define(stage, switch+"/Slider")
    joint.CreateBody0Rel().SetTargets([fixed])
    joint.CreateBody1Rel().SetTargets([button])
    joint.CreateAxisAttr("X")
    joint.CreateLowerLimitAttr(0.)
    joint.CreateUpperLimitAttr(STROKE)
    joint.CreateCollisionEnabledAttr(False)
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
    drive.CreateTypeAttr("force")
    drive.CreateTargetPositionAttr(-.001)
    drive.CreateStiffnessAttr(420.)
    drive.CreateDampingAttr(6.)
    drive.CreateMaxForceAttr(8.)
    stage.GetRootLayer().Save()


class DrinkVisual:
    def __init__(self, stage, root):
        root += "/Drink"
        self.foam_mat = material(stage, root+"/FoamMaterial", (.60,.40,.22), roughness=.67)
        light = material(stage, root+"/Cream", (.88,.68,.43), roughness=.66)
        brown = material(stage, root+"/ColaMaterial", (.072,.018,.006), roughness=.20)
        self.bulk = cylinder(stage, root+"/Liquid", CUP, RADIUS-.0002, .001, brown)
        self.foam = cylinder(stage, root+"/Foam", CUP, RADIUS-.0002, .001, self.foam_mat)
        self.jet = cylinder(stage, root+"/FallingCola", CUP, .0034,.001,brown)
        self.spill = cylinder(stage, root+"/OverflowOnGrate", (.302,.055,.0164), .026,.001,brown)
        self.runoff = cylinder(stage, root+"/RimRunoff", (.308,.055,.070), .002,.105,self.foam_mat)
        self.previous_spill = 0.
        self.spill_at = -10.
        self.bubbles = UsdGeom.PointInstancer.Define(stage, root+"/BubbleTexture")
        p = root+"/BubbleTexture/Bubble"
        ball = UsdGeom.Sphere.Define(stage, p)
        ball.CreateRadiusAttr(1.)
        finish(ball, light)
        self.bubbles.CreatePrototypesRel().SetTargets([p])
        n = 210
        points, sizes = [], []
        for i in range(n):
            r = (i/n)**.5 * .033
            phi = i * 2.3999632
            points.append((r*cos(phi), r*sin(phi)))
            size = .00055 + .00085 * ((i*53)%127)/127
            sizes.append(Gf.Vec3f(size,size,size*.47))
        self.bubble_offsets = points
        self.bubbles.CreateScalesAttr(sizes)
        self.bubbles.CreateProtoIndicesAttr([0]*n)

    @staticmethod
    def layer(item, low, high):
        item.GetVisibilityAttr().Set("inherited" if high > low + .00005 else "invisible")
        item.GetHeightAttr().Set(max(high-low, .00005))
        item.GetOrderedXformOps()[0].Set(Gf.Vec3d(CUP[0], CUP[1], CUP[2]+(high+low)/2))

    def update(self, pour):
        bulk, top = pour.bulk/AREA, pour.top/AREA
        self.layer(self.bulk, 0, bulk)
        self.layer(self.foam, bulk, top)
        self.bubbles.GetVisibilityAttr().Set("inherited" if top-bulk > .001 else "invisible")
        self.bubbles.CreatePositionsAttr().Set([Gf.Vec3f(CUP[0]+x,CUP[1]+y,CUP[2]+top-.0001)
                                                for x,y in self.bubble_offsets])
        # The central visible jet ends at the cup or the first hand intersection.
        if pour.air:
            near = min((h[1] for h in pour.obstructions if h[0] == 0), default=OUTLET_ABOVE_FLOOR)
            oldest, newest = (pour.falling_depth(pour.time-pour.air[i][0]) for i in (0, -1))
            self.layer(self.jet, max(top, OUTLET_ABOVE_FLOOR-min(oldest,near)),
                       OUTLET_ABOVE_FLOOR-newest)
            self.jet.GetRadiusAttr().Set(JET_RADIUS * max(.35, min(1., (pour.pending/4)**.45)))
        else:
            self.layer(self.jet, 0., 0.)
        now = pour.spill_liquid
        if now > self.previous_spill + 1e-8:
            self.spill_at = pour.time
        self.previous_spill = now
        tray = now + pour.intercepted
        self.spill.GetVisibilityAttr().Set("inherited" if tray > 1e-8 else "invisible")
        self.spill.GetRadiusAttr().Set(min(.037, .007 + .006*tray**.5))
        self.runoff.GetVisibilityAttr().Set("inherited" if pour.time-self.spill_at < .28 else "invisible")
