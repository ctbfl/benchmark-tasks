"""Original rigid pan, steak, spatula and slotted holder geometry."""
import math
import numpy as np
if __name__=='__main__':
    from isaaclab.app import AppLauncher
    app=AppLauncher(headless=True,device='cpu',kit_args='--/carb/tasking/plugin/threadCount=8 --/omni/tbb/globalcontrol/maxThreadCount=8').app
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt
from config import HERE, PAN_FLOOR, PAN_RADIUS, STEAK_HEIGHT, HEATER_HEIGHT, outline


def material(stage, root, name, rgb, rough=.5, metal=0., friction=.55):
    mat = UsdShade.Material.Define(stage, root+'/Looks/'+name)
    shader = UsdShade.Shader.Define(stage, str(mat.GetPath())+'/Shader')
    shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(rough)
    shader.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(metal)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
    p = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    p.CreateStaticFrictionAttr(friction)
    p.CreateDynamicFrictionAttr(friction*.8)
    p.CreateRestitutionAttr(0.)
    return mat


def textured(stage, root, name, cooked=False):
    mat = material(stage, root, name, (.5,.2,.15), rough=.65, friction=.4)
    shader = UsdShade.Shader(stage.GetPrimAtPath(str(mat.GetPath())+'/Shader'))
    reader = UsdShade.Shader.Define(stage, str(mat.GetPath())+'/UV')
    reader.CreateIdAttr('UsdPrimvarReader_float2')
    reader.CreateInput('varname', Sdf.ValueTypeNames.Token).Set('st')
    transform = UsdShade.Shader.Define(stage, str(mat.GetPath())+'/Transform')
    transform.CreateIdAttr('UsdTransform2d')
    transform.CreateInput('in', Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(),'result')
    transform.CreateInput('scale', Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(.495,.99))
    transform.CreateInput('translation', Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(.5025 if cooked else .0025,.005))
    tex = UsdShade.Shader.Define(stage, str(mat.GetPath())+'/Texture')
    tex.CreateIdAttr('UsdUVTexture')
    tex.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath('steak-atlas.png'))
    tex.CreateInput('sourceColorSpace', Sdf.ValueTypeNames.Token).Set('sRGB')
    tex.CreateInput('st', Sdf.ValueTypeNames.Float2).ConnectToSource(transform.ConnectableAPI(),'result')
    shader.GetInput('diffuseColor').ConnectToSource(tex.ConnectableAPI(),'rgb')
    return mat


def finish(shape, mat, collision=True):
    bind = UsdShade.MaterialBindingAPI.Apply(shape.GetPrim())
    bind.Bind(mat)
    bind.Bind(mat, materialPurpose='physics')
    if collision:
        UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
        c = PhysxSchema.PhysxCollisionAPI.Apply(shape.GetPrim())
        c.CreateContactOffsetAttr(.0003)
        c.CreateRestOffsetAttr(0.)
    return shape


def mesh(stage, path, vertices, faces, mat, collision=True, uv=False):
    shape = UsdGeom.Mesh.Define(stage,path)
    shape.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(vertices,dtype=np.float32)))
    shape.CreateFaceVertexCountsAttr([len(f) for f in faces])
    shape.CreateFaceVertexIndicesAttr([i for f in faces for i in f])
    shape.CreateSubdivisionSchemeAttr('none')
    if collision:
        UsdPhysics.MeshCollisionAPI.Apply(shape.GetPrim()).CreateApproximationAttr('convexHull')
    if uv:
        values = [(float(np.clip(x/.15+.5,0,1)), float(np.clip(y/.11+.5,0,1))) for x,y,z in vertices]
        UsdGeom.PrimvarsAPI(shape).CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray,'vertex').Set(values)
    return finish(shape,mat,collision)


def box(stage,path,pos,size,mat,collision=True):
    shape=UsdGeom.Cube.Define(stage,path)
    shape.CreateSizeAttr(1)
    shape.AddTranslateOp().Set(Gf.Vec3d(*pos))
    shape.AddScaleOp().Set(Gf.Vec3f(*size))
    return finish(shape,mat,collision)


def prism(stage,path,xy,lo,hi,mat):
    n=len(xy)
    vertices=[(x,y,lo) for x,y in xy]+[(x,y,hi) for x,y in xy]
    faces=[list(reversed(range(n))),list(range(n,2*n))]+[[i,(i+1)%n,(i+1)%n+n,i+n] for i in range(n)]
    return mesh(stage,path,vertices,faces,mat)


def stage_for(name,mass):
    stage=Usd.Stage.CreateNew(str(HERE/'assets'/f'{name}.usda'))
    UsdGeom.SetStageMetersPerUnit(stage,1.)
    UsdGeom.SetStageUpAxis(stage,'Z')
    root='/Asset'
    prim=UsdGeom.Xform.Define(stage,root).GetPrim()
    stage.SetDefaultPrim(prim)
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(mass)
    PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
    return stage,root


def build_heater():
    stage,root=stage_for('heater',2.5)
    black=material(stage,root,'BlackEnclosure',(.006,.007,.009),rough=.72,friction=.8)
    glass=material(stage,root,'BlackCeramic',(.009,.010,.012),rough=.32,friction=.65)
    rim=material(stage,root,'Edge',(.055,.058,.060),rough=.5)
    xy=[(-.20,-.155),(-.19,-.17),(.19,-.17),(.20,-.155),(.20,.155),(.19,.17),(-.19,.17),(-.20,.155)]
    prism(stage,root+'/enclosure',xy,.004,HEATER_HEIGHT-.003,black)
    xy=np.array(xy)*.99
    prism(stage,root+'/ceramic_top',xy,HEATER_HEIGHT-.003,HEATER_HEIGHT,glass)
    for x in (-.155,.155):
        for y in (-.12,.12):
            box(stage,root+f'/foot_{int((x+.2)*100)}_{int((y+.2)*100)}',(x,y,.003),(.028,.028,.006),black)
    dial=UsdGeom.Cylinder.Define(stage,root+'/power_dial')
    dial.CreateRadiusAttr(.012);dial.CreateHeightAttr(.006)
    dial.AddTranslateOp().Set(Gf.Vec3d(-.177,0,HEATER_HEIGHT+.003))
    finish(dial,rim)
    marker=material(stage,root,'DialMark',(.7,.7,.7),rough=.6)
    box(stage,root+'/dial_mark',(-.177,.006,HEATER_HEIGHT+.0061),(.0015,.007,.0002),marker,False)
    led=material(stage,root,'HeatingIndicator',(.6,.015,.005),rough=.7)
    shader=UsdShade.Shader(stage.GetPrimAtPath(str(led.GetPath())+'/Shader'))
    shader.CreateInput('emissiveColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.5,.015,.003))
    box(stage,root+'/indicator',(-.177,-.035,HEATER_HEIGHT+.0002),(.006,.006,.0004),led,False)
    for i in range(12):
        box(stage,root+f'/vent{i}',(-.07+i*.012,-.1701,.015),(.006,.0003,.008),rim,False)
    stage.GetRootLayer().Save()


def build_pan():
    stage,root=stage_for('pan',1.1)
    dark=material(stage,root,'Coating',(.035,.038,.040),rough=.65,friction=.38)
    silver=material(stage,root,'Rim',(.55,.57,.59),rough=.3,metal=.85)
    wood=material(stage,root,'Wood',(.21,.087,.032),rough=.5)
    a=np.linspace(0,2*np.pi,97)[:-1]
    xy=np.c_[PAN_RADIUS*np.cos(a),PAN_RADIUS*np.sin(a)]
    prism(stage,root+'/floor',xy,0,PAN_FLOOR,dark)
    for i in range(64):
        a,b=2*np.pi*np.array([i,i+1])/64
        # Separate convex pieces preserve the open concave pan interior.
        vertices=[(r*np.cos(t),r*np.sin(t),z) for z,rs in ((.003,(.130,.137)),(.040,(.148,.154))) for r in rs for t in (a,b)]
        faces=[[0,2,3,1],[4,5,7,6],[0,1,5,4],[2,6,7,3],[0,4,6,2],[1,3,7,5]]
        mesh(stage,root+f'/wall{i}',vertices,faces,dark)
        # Thin silver lip, also physical.
        xy=[(r*np.cos(t),r*np.sin(t)) for r,t in ((.1475,a),(.1545,a),(.1545,b),(.1475,b))]
        prism(stage,root+f'/rim{i}',xy,.0395,.041,silver)
    box(stage,root+'/neck',(0,.158,.029),(.033,.045,.016),silver)
    xy=[(-.019,.17),(.019,.17),(.017,.305),(.008,.320),(-.008,.320),(-.017,.305)]
    prism(stage,root+'/handle',xy,.020,.041,wood)
    # Fine original granite speckles on the horizontal cooking surface.
    rng=np.random.default_rng(4)
    speckle=material(stage,root,'Speckles',(.38,.39,.37),rough=.8)
    points=[]; faces=[]
    for _ in range(2400):
        x,y=rng.uniform(-PAN_RADIUS,PAN_RADIUS,2)
        if x*x+y*y>(PAN_RADIUS-.001)**2: continue
        s=rng.uniform(.00013,.00040); k=len(points)
        points.extend([(x-s,y-s,PAN_FLOOR+.00001),(x+s,y-s,PAN_FLOOR+.00001),(x,y+s,PAN_FLOOR+.00001)])
        faces.append([k,k+1,k+2])
    mesh(stage,root+'/speckles',points,faces,speckle,False)
    # Dark longitudinal fibers retain a recognizable wood-grain handle.
    grain=material(stage,root,'WoodGrain',(.085,.030,.009),rough=.65)
    for i in range(13):
        x=-.014+i*.0023
        vs=[(x+.0008*np.sin(j*.7+i),.180+j*.006,.04101) for j in range(21)]
        vertices=[(x+d,y,z) for x,y,z in vs for d in (-.0002,.0002)]
        mesh(stage,root+f'/grain{i}',vertices,[[2*j,2*j+1,2*j+3,2*j+2] for j in range(20)],grain,False)
    stage.GetRootLayer().Save()


def build_steak():
    stage,root=stage_for('steak',.19)
    raw=textured(stage,root,'Raw')
    textured(stage,root,'Seared',True)
    xy=outline(); n=len(xy)
    collider=prism(stage,root+'/collision',xy,0,STEAK_HEIGHT,raw)
    collider.GetVisibilityAttr().Set('invisible')
    for name,z,reverse in [('bottom',0,True),('top',STEAK_HEIGHT,False)]:
        face=list(range(n))
        if reverse: face.reverse()
        mesh(stage,root+'/'+name,[(x,y,z) for x,y in xy],[face],raw,False,True)
    vertices=[(x,y,z) for z in (0,STEAK_HEIGHT) for x,y in xy]
    mesh(stage,root+'/edge',vertices,[[i,(i+1)%n,(i+1)%n+n,i+n] for i in range(n)],raw,False,True)
    stage.GetRootLayer().Save()


def build_spatula():
    stage,root=stage_for('spatula',.095)
    black=material(stage,root,'Nylon',(.003,.004,.005),rough=.85,friction=.75)
    shader=UsdShade.Shader(stage.GetPrimAtPath(str(black.GetPath())+'/Shader'))
    shader.CreateInput('useSpecularWorkflow',Sdf.ValueTypeNames.Int).Set(1)
    shader.CreateInput('specularColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.01,.01,.01))
    # A tapered 0.35 mm leading edge grows to a 2 mm blade.
    xy=[(-.060,-.037),(-.055,-.044),(.025,-.042),(.050,-.021),(.050,.021),(.025,.042),(-.055,.044),(-.060,.037)]
    n=len(xy)
    vertices=[(x,y,0) for x,y in xy]+[(x,y,.00035+(x+.060)/.11*.00165) for x,y in xy]
    faces=[list(reversed(range(n))),list(range(n,2*n))]+[[i,(i+1)%n,(i+1)%n+n,i+n] for i in range(n)]
    mesh(stage,root+'/blade',vertices,faces,black)
    # Angled neck raises the handle over the pan rim.
    vertices=[(x,y,z) for x,z in ((.035,.0005),(.105,.032)) for y in (-.012,.012)]+[(x,y,z+.006) for x,z in ((.035,.0005),(.105,.032)) for y in (-.012,.012)]
    mesh(stage,root+'/neck',vertices,[[0,2,3,1],[4,5,7,6],[0,1,5,4],[2,6,7,3],[0,4,6,2],[1,3,7,5]],black)
    xy=[(.09,-.014),(.22,-.014),(.234,-.007),(.237,0),(.234,.007),(.22,.014),(.09,.014)]
    prism(stage,root+'/handle',xy,.030,.045,black)
    stage.GetRootLayer().Save()


def build_holder():
    stage,root=stage_for('holder',.75)
    stone=material(stage,root,'Stone',(.48,.50,.52),rough=.82,friction=.9)
    box(stage,root+'/base',(0,0,.007),(.075,.125,.014),stone)
    for sign in (-1,1):
        box(stage,root+f'/cheek{sign+1}',(sign*.020,0,.039),(.030,.106,.050),stone)
    for sign in (-1,1):
        box(stage,root+f'/end{sign+1}',(0,sign*.052,.039),(.010,.008,.050),stone)
    stage.GetRootLayer().Save()


if __name__=='__main__':
    for build in (build_heater,build_pan,build_steak,build_spatula,build_holder): build()
    app.close()
