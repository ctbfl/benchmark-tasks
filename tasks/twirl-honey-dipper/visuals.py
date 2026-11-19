"""Draw the measured film, pendant necks, airborne drops and surface deposits."""
import math

import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt

from honey import ANGLE, BREAK_G, CELLS, DENSITY
from layout import GROOVES, GROOVE_RADIUS, GROOVE_WIDTH


class HoneyVisual:
    def __init__(self, stage):
        self.stage = stage
        self.root = "/World/Honey"
        UsdGeom.Xform.Define(stage, self.root)
        self.amber = UsdShade.Material.Define(stage, self.root+"/Amber")
        shader = UsdShade.Shader.Define(stage, self.root+"/Amber/Surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set((.47, .132, .006))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(.17)
        shader.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(1.48)
        self.amber.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        self.films = [self.mesh(f"Film{i}", CELLS, 3) for i in range(len(GROOVES))]
        self.necks = [self.mesh(f"Neck{i}", 12, 4) for i in range(len(GROOVES))]
        self.drops = [self.blob(f"Drop{i}") for i in range(24)]
        self.deposit_count = 0

    def mesh(self, name, sides, rings):
        mesh = UsdGeom.Mesh.Define(self.stage, self.root+"/"+name)
        faces = [(j*sides+i, j*sides+(i+1)%sides, (j+1)*sides+(i+1)%sides, (j+1)*sides+i)
                 for j in range(rings-1) for i in range(sides)]
        mesh.CreateFaceVertexCountsAttr([4]*len(faces))
        mesh.CreateFaceVertexIndicesAttr([i for face in faces for i in face])
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateDoubleSidedAttr(True)
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(self.amber)
        return mesh

    def blob(self, name):
        sphere = UsdGeom.Sphere.Define(self.stage, self.root+"/"+name)
        sphere.CreateRadiusAttr(1.)
        sphere.MakeInvisible()
        UsdShade.MaterialBindingAPI.Apply(sphere.GetPrim()).Bind(self.amber)
        return sphere, sphere.AddTranslateOp(), sphere.AddScaleOp()

    @staticmethod
    def set_points(mesh, xyz):
        xyz = np.ascontiguousarray(xyz.reshape(-1, 3), dtype=np.float32)
        mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(xyz))
        mesh.GetExtentAttr().Set([Gf.Vec3f(*xyz.min(axis=0).tolist()),
                                   Gf.Vec3f(*xyz.max(axis=0).tolist())])

    @staticmethod
    def set_blob(blob, pos, shape):
        sphere, translate, scale = blob
        sphere.MakeVisible()
        translate.Set(Gf.Vec3d(*map(float, pos)))
        scale.Set(Gf.Vec3d(*map(float, shape)))

    def update(self, honey, pose):
        axis, down = honey.axis, honey.down
        tangent = np.cross(axis, down)
        radial = np.cos(ANGLE)[:, None]*down + np.sin(ANGLE)[:, None]*tangent
        neck_angle = np.arange(12)*math.tau/12
        neck_circle = np.c_[np.cos(neck_angle), np.sin(neck_angle), np.zeros(12)]
        for i, (film, neck) in enumerate(zip(self.films, self.necks)):
            centre = pose[:3]+GROOVES[i]*axis
            h = honey.film[i]
            points = [centre + axis*width + radial*(GROOVE_RADIUS+h[:, None]*factor)
                      for width, factor in ((-GROOVE_WIDTH/2, .50), (0, 1.), (GROOVE_WIDTH/2, .50))]
            self.set_points(film, np.asarray(points))

            mass = honey.pendant[i]
            if mass < .004:
                neck.MakeInvisible()
                continue
            neck.MakeVisible()
            underside = centre + down*(GROOVE_RADIUS+h[0]*.85)
            size = mass/BREAK_G
            length = .002 + .011*size
            base = np.array([underside, underside, underside, underside]) + (
                np.array([0, .35, .83, 1.])[:, None]*[0., 0., -length])
            radii = np.array([.002, .00095, .0017, 0]) * size**.35
            self.set_points(neck, base[:, None, :] + radii[:, None, None]*neck_circle)

        assert len(honey.flights) <= len(self.drops)
        for i, blob in enumerate(self.drops):
            if i >= len(honey.flights):
                blob[0].MakeInvisible()
                continue
            p, _, mass, _, _ = honey.flights[i]
            r = (3*mass/(4000*math.pi*DENSITY))**(1/3)
            self.set_blob(blob, p, (r*.79, r*.79, r*1.61))

        for index, drop in enumerate(honey.landings[self.deposit_count:], start=self.deposit_count):
            if drop.surface == "jar":
                continue
            r = math.sqrt(drop.mass_g/(1000*DENSITY*math.pi*.0012))
            p = np.asarray(drop.p)+[0, 0, .00025]
            self.set_blob(self.blob(f"Deposit{index}"), p, (r, r*.83, .0011))
        self.deposit_count = len(honey.landings)
