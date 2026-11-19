"""Generate original grooved maple dipper, ceramic vessels and toast USD."""
from math import cos, pi, sin
from pathlib import Path

from layout import DIPPER, GROOVES, GROOVE_RADIUS, JAR, JAR_INNER, JAR_RISER, PLATE_R, STEM_RADIUS, TOAST_OUTLINE

OUT = Path(__file__).parent / "assets"
N = 64


def xyz(values):
    return "(" + ", ".join(f"{v:.7g}" for v in values) + ")"


def mesh(name, rings, color, roughness=.42):
    n = len(rings[0])
    faces = [
        (j*n+i, j*n+(i+1)%n, (j+1)*n+(i+1)%n, (j+1)*n+i)
        for j in range(len(rings)-1) for i in range(n)
    ]
    return f'''def Mesh "{name}" {{
        point3f[] points = [{", ".join(xyz(p) for ring in rings for p in ring)}]
        int[] faceVertexCounts = [{", ".join("4" for _ in faces)}]
        int[] faceVertexIndices = [{", ".join(str(i) for f in faces for i in f)}]
        uniform token subdivisionScheme = "none"
        uniform bool doubleSided = 1
        rel material:binding = </Asset/{name}Material>
    }}
    def Material "{name}Material" {{
        token outputs:surface.connect = </Asset/{name}Material/Surface.outputs:surface>
        def Shader "Surface" {{
            uniform token info:id = "UsdPreviewSurface"
            color3f inputs:diffuseColor = {xyz(color)}
            float inputs:roughness = {roughness}
            token outputs:surface
        }}
    }}'''


def rings(profile, axis="Z"):
    def ring(level, r):
        plane = [(r*cos(i*2*pi/N), r*sin(i*2*pi/N)) for i in range(N)]
        return [(level, u, v) if axis == "X" else (u, v, level) for u, v in plane]
    return [ring(level, r) for level, r in profile]


def collider(name, kind, p, shape, rotation=0):
    scale = f"double3 xformOp:scale = {xyz(shape)}" if kind == "Cube" else (
        f'uniform token axis = "{shape[2]}"\n        double radius = {shape[0]}\n        double height = {shape[1]}'
    )
    size = "double size = 1" if kind == "Cube" else ""
    ops = '"xformOp:translate", "xformOp:rotateZ"' + (', "xformOp:scale"' if kind == "Cube" else "")
    return f'''def {kind} "{name}" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "PhysxCollisionAPI"]
    ) {{
        token visibility = "invisible"
        double3 xformOp:translate = {xyz(p)}
        double xformOp:rotateZ = {rotation}
        {scale}
        {size}
        uniform token[] xformOpOrder = [{ops}]
        float physxCollision:contactOffset = 0.0007
        float physxCollision:restOffset = 0
        rel material:binding:physics = </Asset/Contact>
    }}'''


def save(name, parts, mass=None):
    dynamics = (
        '(prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxRigidBodyAPI"])'
        if mass else ""
    )
    physics = f"""float physics:mass = {mass}
    bool physxRigidBody:enableCCD = 1
    int physxRigidBody:solverPositionIterationCount = 12
    int physxRigidBody:solverVelocityIterationCount = 4""" if mass else ""
    (OUT / f"{name}.usda").write_text(f'''#usda 1.0
(defaultPrim = "Asset"
 upAxis = "Z"
 metersPerUnit = 1)
def Xform "Asset" {dynamics} {{
    {physics}
    def Material "Contact" (prepend apiSchemas = ["PhysicsMaterialAPI"]) {{
        float physics:staticFriction = 0.85
        float physics:dynamicFriction = 0.72
    }}
    {chr(10).join(parts)}
}}
''')


def main():
    OUT.mkdir(exist_ok=True)
    wood = (.60, .31, .112)
    handle = [(-.012, 0), (-.012, .0062), (-.010, STEM_RADIUS), (.041, STEM_RADIUS),
              (.068, .006), (.088, .006), (.095, GROOVE_RADIUS), (.138, GROOVE_RADIUS),
              (.142, .005), (.143, 0)]
    ribs = [.097, .1065, .1155, .1245, .134]
    parts = [mesh("Handle", rings(handle, "X"), wood),
             collider("Grip", "Cylinder", (.015, 0, 0), (STEM_RADIUS, .054, "X")),
             collider("Neck", "Cylinder", (.0645, 0, 0), (.0064, .045, "X")),
             collider("Core", "Cylinder", (.1165, 0, 0), (GROOVE_RADIUS, .043, "X"))]
    for i, x in enumerate(ribs):
        radius = (.013, .015, .016, .0155, .013)[i]
        width = .0032
        parts.extend([mesh(f"Rib{i}", rings([(x-width/2, GROOVE_RADIUS), (x-width/2, radius-.0007),
                    (x-width/2+.0007, radius), (x+width/2-.0007, radius),
                    (x+width/2, radius-.0007), (x+width/2, GROOVE_RADIUS)], "X"), wood),
                    collider(f"Collar{i}", "Cylinder", (x, 0, 0), (radius, width, "X"))])
    # Dark end-grain dot makes accumulated roll visible in the close-up.
    parts.append(mesh("RollMark", [[(-.01215, y+.003, z) for _, y, z in ring]
                       for ring in rings([(0, 0), (0, .00155)], "X")], (.14, .048, .014)))
    save("dipper", parts, .016)

    pot_r = .053
    z_profile = [(0, 0), (0, .042), (.003, .049), (.013, pot_r), (.066, pot_r),
                 (.076, .051), (.078, .049), (.078, .045), (.074, JAR_INNER),
                 (.065, JAR_INNER), (.061, .038), (.061, 0)]
    pot = rings(z_profile)
    for i, ring in enumerate(pot):
        if z_profile[i][0] > .069:
            pot[i] = [(x, y, z - .007*max(0, 1 - abs(j-N/2)/3)) for j, (x, y, z) in enumerate(ring)]
    parts = [mesh("Glaze", pot, (.095, .25, .27), .24),
             mesh("HoneyPool", rings([(.0687, 0), (.069, .0426), (.070, .0431)]), (.70, .28, .025), .15),
             collider("Foot", "Cylinder", (0, 0, .0305), (.043, .061, "Z"))]
    for i in range(32):
        a = i*2*pi/32
        height = .068 if abs(i-16) <= 2 else .078
        r = (pot_r+JAR_INNER)/2
        parts.append(collider(f"Wall{i}", "Cube", (r*cos(a), r*sin(a), height/2),
                             (pot_r-JAR_INNER, .011, height), i*360/32))
    # Two scalloped supports are the dry handle rest on the ceramic pot.
    for i, x in enumerate((-.079, -.049)):
        parts.extend([
            mesh(f"Rest{i}", rings([(.002, 0), (.002, .017), (.006, .018), (.0705, .009),
                                    (.0715, .006), (.0715, 0)]), (.095, .25, .27)),
            collider(f"RestFoot{i}", "Cylinder", (x, 0, .03575), (.009, .0715, "Z")),
        ])
        # Use sloped cheeks to centre the cylinder without constraining the lift.
        parts[-2] = parts[-2].replace('point3f[] points', f'double3 xformOp:translate = ({x}, 0, 0)\n        uniform token[] xformOpOrder = ["xformOp:translate"]\n        point3f[] points')
        for sign in (-1, 1):
            y = sign*.0105
            parts.extend([mesh(f"Cheek{i}{sign+1}", [[(x+u,y+v,z) for u,v,z in r]
                          for r in rings([(.067, .006), (.074, .005), (.077, .003), (.077, 0)])],
                          (.095, .25, .27)),
                          collider(f"Guide{i}{sign+1}", "Cube", (x, y, .073),
                                   (.009, .004, .008))])
    save("honey_pot", parts)
    save("pot_base", [mesh("Base", rings([(0, 0), (0, .047), (.005, .053), (JAR_RISER, .049), (JAR_RISER, 0)]), (.095, .25, .27)), collider("Foot", "Cylinder", (0, 0, JAR_RISER/2), (.050, JAR_RISER, "Z"))])

    plate = rings([(0, 0), (0, .050), (.002, .065), (.006, PLATE_R), (.011, PLATE_R),
                   (.012, .073), (.006, .058), (.006, 0)])
    save("plate", [mesh("Stoneware", plate, (.77, .83, .81), .23),
                   collider("Base", "Cylinder", (0, 0, .003), (.070, .006, "Z"))])

    def bread_ring(z, scale):
        return [(x*scale,y*scale,z) for x,y in TOAST_OUTLINE]
    crust = [bread_ring(.006, .94), bread_ring(.008, .99), bread_ring(.018, 1), bread_ring(.021, .95)]
    crumb = [bread_ring(.021, .95), bread_ring(.02105, .87), bread_ring(.021, 0)]
    parts = [mesh("Crust", crust, (.46, .17, .034), .78),
             mesh("Crumb", crumb, (.75, .47, .20), .91),
             collider("Slice", "Cube", (.005, 0, .0135), (.115, .086, .015))]
    # Scattered small baked pores are original geometry, deterministically placed.
    import random
    rng = random.Random(15)
    for i in range(85):
        x = rng.uniform(-.046, .052)
        y = rng.uniform(-.036, .036)
        r = rng.uniform(.00035, .0016)
        parts.append(mesh(f"Pore{i}", [[(x+u,y+v,z) for u,v,z in ring]
                          for ring in rings([(.02110, 0), (.02110, r)])],
                          (.50, .266, .076), .92))
    save("toast", parts)


if __name__ == "__main__":
    main()
