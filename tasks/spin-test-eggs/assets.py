"""Generate the task's authored geometry. Coordinates are metres."""
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
LONG = .029
SHORT = .021
MASS = .060
INERTIA = MASS / 5 * np.array([2 * SHORT**2, LONG**2 + SHORT**2, LONG**2 + SHORT**2])
BOARD_Z = .008
# Low-friction board approximation; not calibrated to measured egg spin-down.
GLASS_FRICTION = .0037
TRAY_FLOOR = .006
TRAY_HALF = np.array([.118, .082])
TRAYS = {"raw": (.36, .246), "boiled": (.36, -.246)}

HEADER = '''#usda 1.0
(
    defaultPrim = "Asset"
    metersPerUnit = 1
    upAxis = "Z"
)
'''
COLLISION = '''
    float physxCollision:contactOffset = 0.001
    float physxCollision:restOffset = 0
    float physxCollision:torsionalPatchRadius = 0
    float physxCollision:minTorsionalPatchRadius = 0
'''
SPIN_COLLISION = COLLISION.replace("contactOffset = 0.001", "contactOffset = 0.00025")
SHAPE_APIS = '"PhysicsCollisionAPI", "PhysxCollisionAPI", "MaterialBindingAPI"'
MATERIAL = '''
    def Material "Surface" (
        prepend apiSchemas = ["PhysicsMaterialAPI", "PhysxMaterialAPI"]
    ) {{
        float physics:staticFriction = {sf}
        float physics:dynamicFriction = {df}
        float physics:restitution = 0
        token physxMaterial:frictionCombineMode = "{mode}"
    }}
'''


def cube(name, center, size, color, collide=True, collision=COLLISION):
    half = tuple(v/2 for v in size)
    apis = f'(prepend apiSchemas = [{SHAPE_APIS}])' if collide else ""
    physics = collision + '    rel material:binding:physics = </Asset/Surface>' if collide else ""
    return f'''
    def Cube "{name}" {apis} {{
        double size = 2
        double3 xformOp:translate = {tuple(center)}
        double3 xformOp:scale = {half}
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        color3f[] primvars:displayColor = [{tuple(color)}]
        {physics}
    }}
'''


def egg():
    points = [[-LONG, 0., 0.], [LONG, 0., 0.]]
    for latitude in np.linspace(0, np.pi, 21)[1:-1]:
        u = np.cos(latitude)
        width = SHORT * np.sin(latitude) * (1 - .075*u**3)
        for theta in np.linspace(0, 2*np.pi, 40, endpoint=False):
            points.append([LONG*u, width*np.cos(theta), width*np.sin(theta)])
    points = np.array(points)
    hull = ConvexHull(points)
    triangles = []
    for face, plane in zip(hull.simplices, hull.equations):
        a, b, c = points[face]
        if np.dot(np.cross(b-a, c-a), plane[:3]) < 0:
            face = face[[0, 2, 1]]
        triangles.extend(face.tolist())
    coordinates = ", ".join(str(tuple(map(float, p))) for p in points.round(7))
    indices = ", ".join(map(str, triangles))
    counts = ", ".join(["3"] * len(hull.simplices))
    speckles = np.array([[.021, .003, .014], [.010, .014, .014], [.014, -.010, .016],
                         [-.006, -.006, .020], [-.014, .013, .013], [-.018, -.014, .009],
                         [.022, .011, -.005], [0., .019, -.009], [-.009, -.021, .001]])
    colors = []
    for face in hull.simplices:
        p = points[face].mean(axis=0)
        tone = .97 + .025*np.cos(p[0]*130) + .018*np.sin(np.dot(p, [714, 401, 236]))
        if np.min(np.linalg.norm(speckles-p, axis=1)) < .004:
            tone *= .58
        colors.append(tuple(round(float(c*tone), 4) for c in (.87, .677, .50)))
    return HEADER + f'''
def Xform "Asset" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxRigidBodyAPI"]) {{
    float physics:mass = {MASS}
    float3 physics:centerOfMass = (0, 0, 0)
    float3 physics:diagonalInertia = {tuple(map(float, INERTIA))}
    float physxRigidBody:angularDamping = 0.18
    float physxRigidBody:linearDamping = 0.06
    int physxRigidBody:solverPositionIterationCount = 12
    int physxRigidBody:solverVelocityIterationCount = 4
    {MATERIAL.format(sf=.72, df=.60, mode="average")}
    def Mesh "Shell" (
        prepend apiSchemas = [{SHAPE_APIS}, "PhysicsMeshCollisionAPI"]
    ) {{
        uniform token physics:approximation = "convexHull"
        rel material:binding:physics = </Asset/Surface>
        {SPIN_COLLISION}
        point3f[] points = [{coordinates}]
        int[] faceVertexCounts = [{counts}]
        int[] faceVertexIndices = [{indices}]
        uniform token subdivisionScheme = "none"
        color3f[] primvars:displayColor = {str(colors)} (
            interpolation = "uniform"
        )
    }}
}}
'''


GLYPHS = {
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "W": ("10001", "10001", "10001", "10101", "10101", "10101", "01010"),
    "H": ("10001", "10001", "10001", "11111", "10001", "10001", "10001"),
    "D": ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
    "B": ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "I": ("11111", "00100", "00100", "00100", "00100", "00100", "11111"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "-": ("00000", "00000", "00000", "01110", "00000", "00000", "00000"),
}


def label(text):
    # Top camera faces +X. Letter tops are farther away and reading proceeds toward -Y.
    points = []
    size = .00215
    for i, character in enumerate(text):
        for row, line in enumerate(GLYPHS[character]):
            for col, opaque in enumerate(line):
                if opaque == "1":
                    x = -.095 + (3-row)*size
                    y = (3*len(text)-i*6-col-.5)*size
                    for dx, dy in ((-1, -1), (-1, 1), (1, 1), (1, -1)):
                        points.append((round(x+dx*.5*size, 6), round(y+dy*.5*size, 6), .00612))
    return f'''
    def Mesh "Label" {{
        point3f[] points = {points}
        int[] faceVertexCounts = {[4]*(len(points)//4)}
        int[] faceVertexIndices = {list(range(len(points)))}
        uniform token subdivisionScheme = "none"
        color3f[] primvars:displayColor = [(0.97, 0.97, 0.94)]
    }}
'''


def static(shapes, sf, df):
    return HEADER + 'def Xform "Asset" {\n' + MATERIAL.format(sf=sf, df=df, mode="min") + shapes + '}\n'


def basket(text, color):
    width, depth = (.246, .174)
    parts = cube("Floor", (0., 0., .003), (width, depth, .006), color)
    for i, sign in enumerate((-1, 1)):
        parts += cube(f"End{i}", (sign*.1205, 0., .0105), (.005, depth, .013), color)
        parts += cube(f"Side{i}", (0., sign*.0845, .0105), (.236, .005, .013), color)
        parts += cube(f"EndRim{i}", (sign*.1205, 0., .034), (.005, depth, .004), color)
        parts += cube(f"SideRim{i}", (0., sign*.0845, .034), (.236, .005, .004), color)
        for j, y in enumerate(np.linspace(-.078, .078, 9)):
            parts += cube(f"EndSlat{i}_{j}", (sign*.1205, float(y), .0245), (.005, .006, .015), color)
        for j, x in enumerate(np.linspace(-.115, .115, 13)):
            parts += cube(f"SideSlat{i}_{j}", (float(x), sign*.0845, .0245), (.006, .005, .015), color)
    return static(parts + label(text), .64, .58)


def main():
    ASSETS.mkdir(exist_ok=True)
    (ASSETS / "egg.usda").write_text(egg())
    (ASSETS / "board.usda").write_text(static(
        cube("GlassTop", (0., 0., BOARD_Z/2), (.28, .31, BOARD_Z), (.13, .19, .20),
             collision=SPIN_COLLISION),
        GLASS_FRICTION, GLASS_FRICTION))
    (ASSETS / "raw.usda").write_text(basket("RAW", (.14, .32, .43)))
    (ASSETS / "boiled.usda").write_text(basket("HARD-BOILED", (.37, .23, .13)))


if __name__ == "__main__":
    main()
