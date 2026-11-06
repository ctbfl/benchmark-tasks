"""Original perforated divider and contact-operated six-ring binder assets."""
import math
import numpy as np
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt
from config import HERE, HOLES_X, HOLE_Y, HOLE_R, CARD_T, WIRE_R, PIVOT_Y, PIVOT_Z, ring_points


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


def perforated_page(stage,path,mat,t=CARD_T,center=(0,0,0),tab=True):
    """A solid page with six polygonal holes; each collision piece is convex."""
    from shapely.geometry import Polygon
    from shapely.ops import triangulate
    x0,y0,z0=center
    box(stage,path+'/body',(x0,y0-.010,z0),(.210,.128,t),mat)
    if tab:
        box(stage,path+'/tab',(x0+.060,y0-.080,z0),(.035,.014,t),mat)
    bounds=[-.105,*((HOLES_X[1:]+HOLES_X[:-1])/2),.105]
    for j,x in enumerate(HOLES_X):
        circle=[(x+HOLE_R*math.cos(a),HOLE_Y+HOLE_R*math.sin(a)) for a in np.linspace(0,2*math.pi,25)[:-1]]
        region=Polygon([(bounds[j],.054),(bounds[j+1],.054),(bounds[j+1],.074),(bounds[j],.074)], [circle])
        pieces=[p for p in triangulate(region) if region.covers(p.representative_point())]
        assert abs(sum(p.area for p in pieces)-region.area)<1e-10
        for i,p in enumerate(pieces):
            pts=[(a+x0,b+y0) for a,b in list(p.exterior.coords)[:-1]]
            prism(stage,f'{path}/hole{j}_part{i}',pts,z0,t,mat)


def rigid(stage,path,mass,pos=(0,0,0)):
    prim=UsdGeom.Xform.Define(stage,path).GetPrim()
    UsdGeom.Xformable(prim).AddTranslateOp().Set(Gf.Vec3d(*pos))
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(mass)
    PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
    return prim


def capsule(stage,path,a,b,mat):
    from scipy.spatial.transform import Rotation
    a,b=np.asarray(a),np.asarray(b)
    axis=b-a
    axis/=np.linalg.norm(axis)
    cross=np.cross([0.,0.,1.],axis)
    rot=Rotation.from_rotvec(cross/np.linalg.norm(cross)*math.acos(axis[2]))
    q=rot.as_quat()
    shape=UsdGeom.Capsule.Define(stage,path)
    shape.CreateRadiusAttr(WIRE_R)
    shape.CreateHeightAttr(float(np.linalg.norm(b-a)))
    shape.AddTranslateOp().Set(Gf.Vec3d(*((a+b)/2)))
    shape.AddOrientOp().Set(Gf.Quatf(float(q[3]),Gf.Vec3f(*q[:3])))
    return finish(shape,mat)


def build():
    for name in ('binder','card','block'):
        root='/'+name.capitalize()
        stage=Usd.Stage.CreateNew(str(HERE/'assets'/f'{name}.usda'))
        UsdGeom.SetStageMetersPerUnit(stage,1.)
        UsdGeom.SetStageUpAxis(stage,'Z')
        prim=UsdGeom.Xform.Define(stage,root).GetPrim()
        stage.SetDefaultPrim(prim)
        cover=material(stage,root,'cover',(.055,.12,.105),roughness=.7)
        paper=material(stage,root,'ivory',(.87,.84,.72),roughness=.85)
        silver=material(stage,root,'steel',(.65,.68,.71),metal=.85,roughness=.24)
        red=material(stage,root,'red',(.62,.018,.027),roughness=.48)
        wood=material(stage,root,'wood',(.37,.20,.085),roughness=.7)
        if name=='block':
            rigid(stage,root+'/body',.65)
            box(stage,root+'/body/wood',(0,0,.05),(.10,.10,.10),wood)
            # Fine original grain marks on the wood surface.
            grain=material(stage,root,'grain',(.25,.12,.045),roughness=.8)
            for i,y in enumerate(np.linspace(-.046,.046,18)):
                box(stage,root+f'/body/grain{i}',(0,y,.10003),(.10,.00035,.00005),grain,False)
        elif name=='card':
            rigid(stage,root+'/body',.035)
            perforated_page(stage,root+'/body',red)
        else:
            UsdPhysics.ArticulationRootAPI.Apply(prim)
            PhysxSchema.PhysxArticulationAPI.Apply(prim).CreateEnabledSelfCollisionsAttr(True)
            rigid(stage,root+'/base',.45)
            box(stage,root+'/base/cover',(0,0,.0015),(.240,.342,.003),cover)
            box(stage,root+'/base/spine',(0,0,.006),(.228,.025,.006),silver)
            for side in (-1,1):
                # Paper stacks are fixed to the already-open notebook, as requested.
                path=root+('/base/right_pages' if side<0 else '/base/left_pages')
                perforated_page(stage,path,paper,t=.004,center=(0,-.081,.006),tab=False)
                if side>0:
                    xform=UsdGeom.Xform.Define(stage,path)
                    xform.AddRotateZOp().Set(180.)
                for i in range(5):
                    box(stage,path+f'/layer{i}',(0,-.091,.004+i*.0008),(.210,.128,.00012),paper,False)
            for side,label in ((1,'left'),(-1,'right')):
                body=root+'/'+label+'_rail'
                pivot=(0,side*PIVOT_Y,PIVOT_Z)
                rigid(stage,body,.035,pivot)
                box(stage,body+'/rail',(0,0,0),(.17,.004,.004),silver)
                pts=ring_points(side)
                for j,x in enumerate(HOLES_X):
                    for i,(a,b) in enumerate(zip(pts[:-1],pts[1:])):
                        capsule(stage,f'{body}/ring{j}_{i}',a+[x,0,0],b+[x,0,0],silver)
                joint=UsdPhysics.RevoluteJoint.Define(stage,root+'/'+label+'_hinge')
                joint.CreateBody0Rel().SetTargets([root+'/base'])
                joint.CreateBody1Rel().SetTargets([body])
                joint.CreateLocalPos0Attr(Gf.Vec3f(*pivot))
                joint.CreateAxisAttr('X')
                joint.CreateLowerLimitAttr(-36. if side>0 else 0.)
                joint.CreateUpperLimitAttr(0. if side>0 else 36.)
                joint.CreateCollisionEnabledAttr(False)
                drive=UsdPhysics.DriveAPI.Apply(joint.GetPrim(),'angular')
                drive.CreateStiffnessAttr(.30*math.pi/180)
                drive.CreateDampingAttr(.012*math.pi/180)
                drive.CreateMaxForceAttr(.30)
                drive.CreateTargetPositionAttr(0.)
            chrome=material(stage,root,'lever_chrome',(.88,.90,.93),metal=.95,roughness=.16)
            shadow=material(stage,root,'lever_recess',(.018,.023,.028),metal=.45,roughness=.5)
            box(stage,root+'/base/lever_recess',(-.111,0,.010),(.031,.028,.001),shadow,False)
            rigid(stage,root+'/button',.025,(-.111,0,.016))
            outline=[(-.014,-.009),(-.011,-.012),(.011,-.012),(.014,-.009),
                     (.014,.009),(.011,.012),(-.011,.012),(-.014,.009)]
            prism(stage,root+'/button/paddle',outline,0,.004,chrome)
            for i,x in enumerate((-.007,-.003,.001)):
                box(stage,root+f'/button/grip_groove{i}',(x,0,.00203),(.0007,.017,.00006),shadow,False)
            box(stage,root+'/button/hinge_fold',(.0105,0,.0025),(.002,.021,.001),chrome,False)
            joint=UsdPhysics.PrismaticJoint.Define(stage,root+'/press')
            joint.CreateBody0Rel().SetTargets([root+'/base'])
            joint.CreateBody1Rel().SetTargets([root+'/button'])
            joint.CreateLocalPos0Attr(Gf.Vec3f(-.111,0,.016))
            joint.CreateAxisAttr('Z')
            joint.CreateLowerLimitAttr(-.0045)
            joint.CreateUpperLimitAttr(0.)
            joint.CreateCollisionEnabledAttr(False)
            drive=UsdPhysics.DriveAPI.Apply(joint.GetPrim(),'linear')
            drive.CreateStiffnessAttr(1200.)
            drive.CreateDampingAttr(8.)
            drive.CreateTargetPositionAttr(0.)
        stage.GetRootLayer().Save()


if __name__=='__main__':
    build()
