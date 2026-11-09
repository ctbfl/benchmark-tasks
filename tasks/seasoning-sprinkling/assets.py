"""Original dual-opening spice bottle, rigid seasoning grains, plate and steak."""
import math
import numpy as np
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt
from config import HERE, GRAIN_RADIUS, GRAIN_MASS_G, HINGE_Z, STEAK_AXES, STEAK_HEIGHT, grain_positions

def material(stage, root, name, rgb, metal=0., roughness=.5):
    mat = UsdShade.Material.Define(stage, root+'/Looks/'+name)
    shader = UsdShade.Shader.Define(stage, str(mat.GetPath())+'/Shader')
    shader.CreateIdAttr('UsdPreviewSurface')
    for key,typ,value in [('diffuseColor',Sdf.ValueTypeNames.Color3f,Gf.Vec3f(*rgb)),
                          ('roughness',Sdf.ValueTypeNames.Float,roughness),
                          ('metallic',Sdf.ValueTypeNames.Float,metal)]:
        shader.CreateInput(key,typ).Set(value)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
    physics = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    physics.CreateStaticFrictionAttr(.65)
    physics.CreateDynamicFrictionAttr(.5)
    physics.CreateRestitutionAttr(0.)
    return mat


def finish(shape, mat, collision=True):
    binding = UsdShade.MaterialBindingAPI.Apply(shape.GetPrim())
    binding.Bind(mat)
    binding.Bind(mat,materialPurpose='physics')
    if collision:
        UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
        cfg = PhysxSchema.PhysxCollisionAPI.Apply(shape.GetPrim())
        cfg.CreateContactOffsetAttr(.0003)
        cfg.CreateRestOffsetAttr(0.)
    return shape


def box(stage,path,pos,size,mat,collision=True):
    shape=UsdGeom.Cube.Define(stage,path)
    shape.CreateSizeAttr(1)
    shape.AddTranslateOp().Set(Gf.Vec3d(*pos))
    shape.AddScaleOp().Set(Gf.Vec3f(*size))
    return finish(shape,mat,collision)


def mesh(stage,path,vertices,faces,mat,collision=True):
    shape=UsdGeom.Mesh.Define(stage,path)
    shape.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(vertices,dtype=np.float32)))
    shape.CreateFaceVertexCountsAttr([len(f) for f in faces])
    shape.CreateFaceVertexIndicesAttr([i for f in faces for i in f])
    shape.CreateSubdivisionSchemeAttr('none')
    if collision:
        UsdPhysics.MeshCollisionAPI.Apply(shape.GetPrim()).CreateApproximationAttr('convexHull')
    return finish(shape,mat,collision)


def prism(stage,path,points,z,t,mat):
    n=len(points)
    vertices=[(x,y,z-t/2) for x,y in points]+[(x,y,z+t/2) for x,y in points]
    faces=[list(reversed(range(n))),list(range(n,2*n))]+[[i,(i+1)%n,(i+1)%n+n,i+n] for i in range(n)]
    return mesh(stage,path,vertices,faces,mat)


def rigid(stage,path,mass,pos=(0,0,0)):
    prim=UsdGeom.Xform.Define(stage,path).GetPrim()
    UsdGeom.Xformable(prim).AddTranslateOp().Set(Gf.Vec3d(*pos))
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(mass)
    PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
    return prim


def new_stage(name):
    stage = Usd.Stage.CreateNew(str(HERE/'assets'/f'{name}.usda'))
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdGeom.SetStageUpAxis(stage, 'Z')
    root = '/Asset'
    prim = UsdGeom.Xform.Define(stage, root).GetPrim()
    stage.SetDefaultPrim(prim)
    return stage, root, prim


def perforated_deck(stage, root, mat):
    from shapely.geometry import Polygon
    from shapely.ops import triangulate
    outer = [(-.022,-.030),(.022,-.030),(.022,.030),(-.022,.030)]
    holes = [[(-.017,.006),(-.017,.025),(.017,.025),(.017,.006)]]
    for x in (-.010, .010):
        for y in (-.009, -.020):
            holes.append([(x+.003*math.cos(a), y+.003*math.sin(a))
                          for a in np.linspace(0, 2*math.pi, 17)[:-1]])
    region = Polygon(outer, holes)
    pieces = [p for p in triangulate(region) if region.covers(p.representative_point())]
    assert abs(sum(p.area for p in pieces)-region.area) < 1e-10
    for i, p in enumerate(pieces):
        prism(stage, root+f'/deck{i}',list(p.exterior.coords)[:-1],.108,.004,mat)


def textured_material(stage,root,name,filename):
    mat=material(stage,root,name,(.5,.5,.5),roughness=.75)
    shader=UsdShade.Shader(stage.GetPrimAtPath(str(mat.GetPath())+'/Shader'))
    tex=UsdShade.Shader.Define(stage,str(mat.GetPath())+'/texture')
    tex.CreateIdAttr('UsdUVTexture')
    tex.CreateInput('file',Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(filename))
    tex.CreateInput('sourceColorSpace',Sdf.ValueTypeNames.Token).Set('sRGB')
    uv=UsdShade.Shader.Define(stage,str(mat.GetPath())+'/uv')
    uv.CreateIdAttr('UsdPrimvarReader_float2')
    uv.CreateInput('varname',Sdf.ValueTypeNames.Token).Set('st')
    uv.CreateOutput('result',Sdf.ValueTypeNames.Float2)
    tex.CreateInput('st',Sdf.ValueTypeNames.Float2).ConnectToSource(uv.ConnectableAPI(),'result')
    tex.CreateOutput('rgb',Sdf.ValueTypeNames.Float3)
    shader.GetInput('diffuseColor').ConnectToSource(tex.ConnectableAPI(),'rgb')
    return mat


def label_texture():
    from PIL import Image,ImageDraw,ImageFont
    image=Image.new('RGB',(512,512),(235,220,174))
    draw=ImageDraw.Draw(image)
    draw.rectangle((15,15,497,497),outline=(48,65,36),width=8)
    draw.rectangle((15,38,497,141),fill=(48,65,36))
    draw.text((256,86),'PEPPER SALT',anchor='mm',font=ImageFont.load_default(size=58),fill=(249,240,208))
    draw.text((256,190),'SEASONING',anchor='mm',font=ImageFont.load_default(size=43),fill=(48,65,36))
    rng=np.random.default_rng(1)
    for _ in range(110):
        x,y=rng.integers((125,245),(385,380));r=int(rng.integers(3,8))
        draw.ellipse((int(x-r),int(y-r),int(x+r),int(y+r)),fill=(99,68,29))
    draw.text((256,438),'NET 10 g',anchor='mm',font=ImageFont.load_default(size=35),fill=(48,65,36))
    image.save(HERE/'assets/label.png')


def bottle():
    stage, root, prim = new_stage('bottle')
    UsdPhysics.ArticulationRootAPI.Apply(prim)
    PhysxSchema.PhysxArticulationAPI.Apply(prim).CreateEnabledSelfCollisionsAttr(True)
    amber = material(stage, root, 'bottle',(.44,.29,.12),roughness=.38)
    red = material(stage, root, 'cap',(.53,.025,.018),roughness=.35)
    label_texture()
    cream = textured_material(stage,root,'label','label.png')
    dark = material(stage, root, 'ink',(.095,.035,.012),roughness=.8)
    rigid(stage,root+'/body',.085)
    box(stage,root+'/body/base',(0,0,.002),(.044,.060,.004),amber)
    for side in (-1,1):
        box(stage,root+f'/body/xwall{side+1}',(side*.0205,0,.054),(.003,.060,.104),amber)
        box(stage,root+f'/body/ywall{side+1}',(0,side*.0285,.054),(.038,.003,.104),amber)
        vertices=[(side*.02205,side*y,z) for y,z in [(-.029,.028),(.029,.028),(.029,.088),(-.029,.088)]]
        face=mesh(stage,root+f'/body/label{side+1}',vertices,[[0,1,2,3]],cream,False)
        UsdGeom.PrimvarsAPI(face).CreatePrimvar('st',Sdf.ValueTypeNames.TexCoord2fArray,'vertex').Set([(0,0),(1,0),(1,1),(0,1)])
    perforated_deck(stage,root+'/body',red)
    box(stage,root+'/body/hinge_bar',(0,0,.112),(.044,.005,.004),red)
    # Identical closed covers hide which side contains the perforations.
    for side, name in ((1,'pour'),(-1,'sprinkle')):
        path = root+'/'+name+'_lid'
        pivot = (0, side*.0025, HINGE_Z)
        rigid(stage,path,.004,pivot)
        box(stage,path+'/cover',(0,side*.0145,0),(.046,.028,.003),red)
        box(stage,path+'/tab',(0,side*.035,0),(.018,.014,.004),red)
        # Molded sealing plugs occupy the actual apertures when the cover is shut.
        if side>0:
            box(stage,path+'/seal',(0,.013,-.0035),(.033,.018,.006),red)
        else:
            for i,(x,y) in enumerate((x,y) for x in (-.010,.010) for y in (-.009,-.020)):
                plug=UsdGeom.Cylinder.Define(stage,path+f'/seal{i}')
                plug.CreateRadiusAttr(.0025)
                plug.CreateHeightAttr(.006)
                plug.AddTranslateOp().Set(Gf.Vec3d(x,y+.0025,-.0035))
                finish(plug,red)

        for i in range(3):
            box(stage,path+f'/rib{i}',(0,side*(.031+i*.0025),.0022),(.013,.0008,.0007),red,False)
        joint = UsdPhysics.RevoluteJoint.Define(stage,root+'/'+name+'_hinge')
        joint.CreateBody0Rel().SetTargets([root+'/body'])
        joint.CreateBody1Rel().SetTargets([path])
        joint.CreateLocalPos0Attr(Gf.Vec3f(*pivot))
        joint.CreateAxisAttr('X')
        joint.CreateLowerLimitAttr(0. if side>0 else -120.)
        joint.CreateUpperLimitAttr(120. if side>0 else 0.)
        joint.CreateCollisionEnabledAttr(False)
        axis = joint.GetPrim()
        axis.ApplyAPI('PhysxJointAxisAPI','angular')
        for key,value in [('staticFrictionEffort',.080),('dynamicFrictionEffort',.060),('viscousFrictionCoefficient',.00002)]:
            axis.CreateAttribute('physxJointAxis:angular:'+key,Sdf.ValueTypeNames.Float).Set(value)
    stage.GetRootLayer().Save()


def grains():
    from scipy.spatial import ConvexHull
    from scipy.spatial.transform import Rotation
    stage, root, _ = new_stage('grains')
    colors = [(.80,.68,.44),(.62,.43,.20),(.24,.13,.055),(.91,.83,.61)]
    mats = [material(stage,root,f'grain{i}',c,roughness=.95) for i,c in enumerate(colors)]
    phi=(1+math.sqrt(5))/2
    vertices=np.array([(0,a,b*phi) for a in (-1,1) for b in (-1,1)]+
                      [(a,b*phi,0) for a in (-1,1) for b in (-1,1)]+
                      [(b*phi,0,a) for a in (-1,1) for b in (-1,1)],float)
    vertices/=np.linalg.norm(vertices[0])
    faces=[]
    for face in ConvexHull(vertices).simplices:
        a,b,c=vertices[face]
        faces.append(face.tolist() if np.dot(np.cross(b-a,c-a),a)>0 else face[::-1].tolist())
    rotations=Rotation.random(len(grain_positions()),random_state=np.random.default_rng(7)).as_quat()
    for i,p in enumerate(grain_positions()):
        path=root+f'/grain_{i:04d}'
        prim=rigid(stage,path,GRAIN_MASS_G/1000,p)
        q=rotations[i]
        UsdGeom.Xformable(prim).AddOrientOp().Set(Gf.Quatf(float(q[3]),Gf.Vec3f(*q[:3])))
        PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateEnableCCDAttr(True)
        shape=mesh(stage,path+'/grain',vertices,faces,mats[i%4])
        shape.AddScaleOp().Set(Gf.Vec3f(GRAIN_RADIUS))
        PhysxSchema.PhysxCollisionAPI(shape.GetPrim()).CreateContactOffsetAttr(.0001)
    stage.GetRootLayer().Save()


def plate():
    stage,root,_=new_stage('plate')
    ceramic=material(stage,root,'ceramic',(.89,.91,.88),roughness=.22)
    rim=material(stage,root,'rim',(.025,.09,.065),roughness=.35)
    rigid(stage,root+'/body',.48)
    circle=[(.107*math.cos(a),.107*math.sin(a)) for a in np.linspace(0,2*math.pi,97)[:-1]]
    prism(stage,root+'/body/base',circle,.003,.006,ceramic)
    for i in range(96):
        angles=[i*2*math.pi/96,(i+1)*2*math.pi/96]
        profile=[(.106,.001),(.138,.010),(.138,.015),(.107,.006)]
        vertices=[(r*math.cos(a),r*math.sin(a),z) for a in angles for r,z in profile]
        faces=[[3,2,1,0],[4,5,6,7],[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7]]
        mesh(stage,root+f'/body/rim{i}',vertices,faces,ceramic)
    stage.GetRootLayer().Save()


def steak():
    from PIL import Image, ImageDraw, ImageFilter
    rng=np.random.default_rng(4)
    coarse=Image.fromarray(rng.integers(0,255,(32,32),dtype=np.uint8)).resize((512,512),Image.Resampling.BICUBIC)
    medium=Image.fromarray(rng.integers(0,255,(100,100),dtype=np.uint8)).resize((512,512),Image.Resampling.BICUBIC)
    noise=(np.asarray(coarse,dtype=float)-128)*.22+(np.asarray(medium,dtype=float)-128)*.12
    yy,xx=np.mgrid[:512,:512]/511
    radius=np.sqrt((2*xx-1)**2+(2*yy-1)**2)
    color=np.array([111.,55.,31.])+noise[...,None]*np.array([1.,.66,.40])
    fat=(radius>.83+np.asarray(coarse)/255*.08)&(xx>.46)&(yy>.16)
    color[fat]=np.array([166.,123.,77.])+noise[fat,None]*.6
    sear=xx*6.5+yy*4.1+.025*np.sin(yy*35)+.02*noise/20
    band=np.abs(sear-np.round(sear))<(.055+.015*np.sin(xx*48))
    color[band]*=np.array([.53,.52,.48])
    color[radius>.96]*=.72
    texture=Image.fromarray(np.clip(color,0,255).astype(np.uint8))
    draw=ImageDraw.Draw(texture)
    for _ in range(140):
        x,y=rng.integers(40,472,2)
        if ((x-256)/245)**2+((y-256)/245)**2>.8:continue
        length=int(rng.integers(4,18))
        draw.line((int(x),int(y),int(x+length),int(y+length//3)),fill=(149,99,63),width=1)
    texture.filter(ImageFilter.GaussianBlur(.5)).save(HERE/'assets/steak.png')
    stage,root,_=new_stage('steak')
    meat=textured_material(stage,root,'roasted_beef','steak.png')
    rigid(stage,root+'/body',.24)
    n=96
    a,b=STEAK_AXES
    contour=[(a*math.cos(t),b*math.sin(t)) for t in np.linspace(0,2*math.pi,n,endpoint=False)]
    shape=prism(stage,root+'/body/meat',contour,STEAK_HEIGHT/2,STEAK_HEIGHT,meat)
    points=shape.GetPointsAttr().Get()
    st=[Gf.Vec2f(float(x)/(2*a)+.5,float(y)/(2*b)+.5) for x,y,z in points]
    UsdGeom.PrimvarsAPI(shape).CreatePrimvar('st',Sdf.ValueTypeNames.TexCoord2fArray,'vertex').Set(st)
    stage.GetRootLayer().Save()


def build():
    (HERE/'assets').mkdir(exist_ok=True)
    bottle(); grains(); plate(); steak()


if __name__=='__main__':
    build()
