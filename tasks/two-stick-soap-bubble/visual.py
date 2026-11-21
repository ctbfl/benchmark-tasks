"""Translucent surfaces for the measured wet film and detached bubble."""
import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt

from film import sphere_radius
from geometry import ROOT, STRANDS
from simulation import strands


class Surface:
    def __init__(self, stage, name, rings, segments, close_front):
        path = ROOT + "/" + name
        self.mesh = UsdGeom.Mesh.Define(stage, path)
        self.mesh.CreateSubdivisionSchemeAttr("none")
        self.mesh.CreateDoubleSidedAttr(True)
        faces = []
        if close_front:
            faces.append(list(range(segments-1, -1, -1)))
        for ring in range(rings-1):
            a, b = ring*segments, (ring+1)*segments
            for i in range(segments):
                j = (i+1)%segments
                faces.append([a+i, a+j, b+j, b+i])
        faces.append(list(range((rings-1)*segments, rings*segments)))
        self.mesh.CreateFaceVertexCountsAttr(Vt.IntArray([len(f) for f in faces]))
        self.mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([idx for f in faces for idx in f]))
        self.colors = self.mesh.CreateDisplayColorPrimvar("vertex")
        self.opacity = self.mesh.CreateDisplayOpacityPrimvar("vertex")
        look = UsdShade.Material.Define(stage, path + "/FilmPaint")
        sh = UsdShade.Shader.Define(stage, path + "/FilmPaint/Surface")
        sh.CreateIdAttr("UsdPreviewSurface")
        for name, src, node_type, shape in [
            ("diffuseColor", "displayColor", "UsdPrimvarReader_float3", Sdf.ValueTypeNames.Float3),
            ("opacity", "displayOpacity", "UsdPrimvarReader_float", Sdf.ValueTypeNames.Float),
        ]:
            reader = UsdShade.Shader.Define(stage, path + "/FilmPaint/" + src)
            reader.CreateIdAttr(node_type)
            reader.CreateInput("varname", Sdf.ValueTypeNames.String).Set(src)
            reader.CreateOutput("result", shape)
            sh.CreateInput(name, shape).ConnectToSource(reader.ConnectableAPI(), "result")
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(.16)
        sh.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(1.25)
        sh.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.55))
        sh.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(1)
        look.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(self.mesh.GetPrim()).Bind(look)

    def update(self, points=None, vectors=None):
        if points is None:
            self.mesh.MakeInvisible()
            return
        self.mesh.MakeVisible()
        points = np.ascontiguousarray(points.reshape(-1, 3), dtype=np.float32)
        vectors = vectors.reshape(-1, 3)
        self.mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points))
        self.mesh.GetExtentAttr().Set([Gf.Vec3f(*map(float, p)) for p in (points.min(0), points.max(0))])
        # A few bands of interference tint with a broad glint from the outdoor sky.
        phase = 10*vectors[:,2] + 2.2*vectors[:,1] + 5*points[:,2]
        glint = np.maximum(0., vectors @ np.array([.39, -.41, .825]))**12
        band = np.cos(phase[:,None] + np.array([0., 2.1, 4.1]))*.5+.5
        color = np.clip(.26 + .60*band + .32*glint[:,None], 0., 1.)
        alpha = .13 + .065*np.cos(2*phase)**2 + .33*glint
        self.colors.Set(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(color, dtype=np.float32)))
        self.opacity.Set(Vt.FloatArray.FromNumpy(np.ascontiguousarray(alpha, dtype=np.float32)))


class FilmMesh:
    def __init__(self, stage):
        segments = sum(n for _, n in STRANDS) + 2
        self.attached = Surface(stage, "InflatingFilm", 17, segments, False)
        self.detached = Surface(stage, "DetachedBubble", 31, 64, True)
        lat, lon = np.meshgrid(np.linspace(.005,np.pi-.005,31), np.linspace(0,2*np.pi,64,endpoint=False),
                               indexing="ij")
        self.sphere = np.stack((np.sin(lat)*np.cos(lon), np.sin(lat)*np.sin(lon), np.cos(lat)), axis=-1)

    def update(self, poses, visual):
        center = np.array(visual[:3])
        normal = np.array(visual[3:6])
        volume, area, wet = visual[6:9]
        if wet and area > 0.:
            upper, lower = strands(poses)
            edge = np.concatenate((upper, lower[-2:0:-1]))
            if volume < .00007:
                t = np.linspace(1., .005, 17)[:,None,None]
                p = center+t*(edge-center)+.010*(1-t*t)*normal
                n = np.broadcast_to(normal, p.shape)
            else:
                r = sphere_radius(volume)
                rel = edge-center
                in_plane = rel - np.outer(rel@normal, normal)
                unit = in_plane/np.maximum(np.linalg.norm(in_plane,axis=1,keepdims=True), 1e-6)
                theta = np.linspace(.42,np.pi-.008,14)[:,None,None]
                n_sphere = -np.cos(theta)*normal + np.sin(theta)*unit
                shell = center+(r+.028)*normal + r*n_sphere
                entry = np.linspace(0., 1., 4, endpoint=True)[:-1,None,None]
                throat = edge*(1-entry) + shell[0]*entry
                p = np.concatenate((throat, shell))
                n = np.concatenate((np.broadcast_to(unit,throat.shape), n_sphere))
            self.attached.update(p,n)
        else:
            self.attached.update()
        radius = visual[12]
        if radius > 0:
            self.detached.update(np.asarray(visual[9:12]) + radius*self.sphere, self.sphere)
        else:
            self.detached.update()
