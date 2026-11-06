"""Reusable, physically pressed tabletop timer and calculator USD assets."""

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation, localcontext
import json
import math
import os
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade


HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets" / "electronics"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
PRESS_THRESHOLD = -0.0016
RELEASE_THRESHOLD = -0.0007
TRAVEL = 0.003
TIMER_KEYS = ("M", "S", "START")
CALCULATOR_LAYOUT = (
    ("%", "7", "8", "9", "/"),
    ("SQRT", "4", "5", "6", "*"),
    ("CE", "1", "2", "3", "-"),
    ("C", "0", ".", "+/-", "+"),
    ("=",),
)
KEY_NAMES = {
    "+/-": "sign", "/": "divide", "*": "multiply", "-": "minus",
    "+": "plus", "=": "equals", ".": "dot", "%": "percent", "SQRT": "sqrt",
}
SEGMENTS = {
    "0": "abcdef", "1": "bc", "2": "abdeg", "3": "abcdg",
    "4": "bcfg", "5": "acdfg", "6": "acdefg", "7": "abc",
    "8": "abcdefg", "9": "abcdfg", "-": "g", " ": "",
    "E": "adefg", "r": "eg", "o": "cdeg",
}


def pressed_buttons(positions, previous=(), *, press_threshold=PRESS_THRESHOLD,
                    release_threshold=RELEASE_THRESHOLD):
    """Convert measured key-joint positions in metres to a hysteretic pressed set."""
    previous = set(previous)
    return {key for key, value in positions.items()
            if value <= (release_threshold if key in previous else press_threshold)}


@dataclass
class TimerState:
    remaining_seconds: float = 15.0
    running: bool = True
    _pressed: set = field(default_factory=set, repr=False)

    @property
    def display_seconds(self):
        return max(0, math.ceil(self.remaining_seconds - 1e-9))

    @property
    def display(self):
        value = self.display_seconds
        return f"{value // 60:02d}:{value % 60:02d}"

    def advance(self, dt):
        if not math.isfinite(dt) or dt < 0:
            raise ValueError("Timer dt must be finite and nonnegative")
        if self.running:
            self.remaining_seconds = max(0.0, self.remaining_seconds - dt)
            if self.remaining_seconds <= 1e-9:
                self.remaining_seconds = 0.0
                self.running = False

    def update_buttons(self, pressed_set):
        pressed = set(pressed_set)
        if not pressed <= set(TIMER_KEYS):
            raise ValueError(f"Unknown timer key: {pressed}")
        rising = pressed - self._pressed
        chord = {"M", "S"} <= pressed
        previous_chord = {"M", "S"} <= self._pressed
        self._pressed = pressed
        if chord:
            if not previous_chord:
                self.remaining_seconds, self.running = 0.0, False
                return ["RESET"]
            return []
        events = []
        for key in TIMER_KEYS:
            if key not in rising:
                continue
            value = self.display_seconds
            minutes, seconds = divmod(value, 60)
            if key == "M":
                self.remaining_seconds = float(((minutes + 1) % 60) * 60 + seconds)
            elif key == "S":
                self.remaining_seconds = float(minutes * 60 + (seconds + 1) % 60)
            else:
                self.running = not self.running if self.remaining_seconds > 0 else False
            events.append(key)
        return events


@dataclass
class CalculatorState:
    entry: str = "0"
    accumulator: Decimal | None = None
    operator: str | None = None
    replace_entry: bool = False
    last_operator: str | None = None
    last_operand: Decimal | None = None
    last_submission: Decimal | None = None
    submission_count: int = 0
    _pressed: set = field(default_factory=set, repr=False)

    @property
    def display(self):
        return self.entry

    def _format(self, value):
        if not value.is_finite():
            return "Error"
        text = format(value, "f")
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        if len(text.replace(".", "")) > 8:
            if abs(value) >= Decimal("100000000"):
                return "Error"
            whole = len(str(abs(int(value)))) + int(value < 0)
            places = max(0, 8 - whole)
            text = f"{value:.{places}f}".rstrip("0").rstrip(".") if places else str(round(value))
        return "0" if text in ("", "-0") else text

    def _calculate(self, left, right, operator):
        with localcontext() as context:
            context.prec = 16
            if operator == "+":
                return left + right
            if operator == "-":
                return left - right
            if operator == "*":
                return left * right
            if operator == "/":
                return left / right
        raise ValueError(operator)

    def _clear(self):
        self.entry, self.accumulator, self.operator = "0", None, None
        self.replace_entry = False
        self.last_operator, self.last_operand, self.last_submission = None, None, None

    def _key(self, key):
        if key == "C":
            self._clear()
            return
        if key == "CE":
            self.entry, self.replace_entry, self.last_submission = "0", False, None
            return
        if self.entry == "Error":
            if key.isdigit() or key == ".":
                self._clear()
            else:
                return
        if key.isdigit():
            self.last_submission = None
            if self.replace_entry:
                self.entry, self.replace_entry = "0", False
            for digit in key:
                if len(self.entry.replace(".", "").replace("-", "")) < 8 or self.entry == "0":
                    self.entry = digit if self.entry == "0" else self.entry + digit
            return
        if key == ".":
            self.last_submission = None
            if self.replace_entry:
                self.entry, self.replace_entry = "0", False
            if "." not in self.entry:
                self.entry += "."
            return
        if key == "+/-":
            self.last_submission = None
            if Decimal(self.entry) != 0:
                self.entry = self.entry[1:] if self.entry.startswith("-") else "-" + self.entry
            return
        current = Decimal(self.entry)
        try:
            if key in ("%", "SQRT"):
                current = current / 100 if key == "%" else current.sqrt()
                self.entry, self.last_submission = self._format(current), None
            elif key in ("+", "-", "*", "/"):
                self.last_submission = None
                if self.operator is not None and not self.replace_entry:
                    current = self._calculate(self.accumulator, current, self.operator)
                    self.entry = self._format(current)
                self.accumulator, self.operator, self.replace_entry = current, key, True
                self.last_operator, self.last_operand = None, None
            elif key == "=":
                if self.operator is not None:
                    self.last_operand, self.last_operator = current, self.operator
                    current = self._calculate(self.accumulator, current, self.operator)
                elif self.last_operator is not None:
                    current = self._calculate(current, self.last_operand, self.last_operator)
                self.entry = self._format(current)
                self.last_submission = current if self.entry != "Error" else None
                self.submission_count += 1
                self.accumulator, self.operator, self.replace_entry = current, None, True
            else:
                raise ValueError(key)
        except (ArithmeticError, InvalidOperation):
            self.entry, self.operator, self.last_submission = "Error", None, None
            self.replace_entry = True

    def update_buttons(self, pressed_set):
        pressed = set(pressed_set)
        keys = tuple(key for row in CALCULATOR_LAYOUT for key in row)
        if not pressed <= set(keys):
            raise ValueError(f"Unknown calculator key: {pressed}")
        rising = pressed - self._pressed
        self._pressed = pressed
        events = []
        for key in keys:
            if key in rising:
                self._key(key)
                events.append(key)
        return events


def _material(stage, path, color, roughness=0.45, metallic=0.0):
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _bind(prim, material):
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)


def _body(stage, path, mass, position=(0, 0, 0), rotation=None):
    body = UsdGeom.Xform.Define(stage, path)
    body.AddTranslateOp().Set(Gf.Vec3d(*position))
    if rotation is not None:
        body.AddOrientOp().Set(Gf.Quatf(rotation))
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr(mass)
    body.GetPrim().AddAppliedSchema("PhysxRigidBodyAPI")
    body.GetPrim().CreateAttribute("physxRigidBody:enableCCD", Sdf.ValueTypeNames.Bool).Set(True)
    return body


def _collider(prim):
    UsdPhysics.CollisionAPI.Apply(prim)
    if prim.IsA(UsdGeom.Mesh):
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexHull")
    prim.AddAppliedSchema("PhysxCollisionAPI")
    prim.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(0.0003)
    prim.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(0.0)


def _transform(shape, position, rotation=None):
    shape.AddTranslateOp().Set(Gf.Vec3d(*position))
    if rotation is not None:
        shape.AddOrientOp().Set(Gf.Quatf(rotation))


def _rounded_box(stage, path, size, position, radius, material, rotation=None, collision=False):
    width, height, depth = size
    bevel = min(depth * 0.20, radius * 0.45, 0.0015)
    rings = []
    for z, inset in ((-depth / 2, bevel), (-depth / 2 + bevel, 0),
                     (depth / 2 - bevel, 0), (depth / 2, bevel)):
        hw, hh, r = width / 2 - inset, height / 2 - inset, radius - inset
        ring = []
        for cx, cy, start in ((hw - r, hh - r, 0), (-hw + r, hh - r, 90),
                              (-hw + r, -hh + r, 180), (hw - r, -hh + r, 270)):
            for index in range(7):
                angle = math.radians(start + index * 90 / 6)
                ring.append((cx + r * math.cos(angle), cy + r * math.sin(angle), z))
        rings.append(ring)
    count = len(rings[0])
    points = [point for ring in rings for point in ring]
    faces = [list(reversed(range(count)))]
    for level in range(3):
        for index in range(count):
            nxt = (index + 1) % count
            faces.append([level * count + index, level * count + nxt,
                          (level + 1) * count + nxt, (level + 1) * count + index])
    faces.append(list(range(3 * count, 4 * count)))
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([len(face) for face in faces])
    mesh.CreateFaceVertexIndicesAttr([i for face in faces for i in face])
    mesh.CreateSubdivisionSchemeAttr("none")
    _transform(mesh, position, rotation)
    _bind(mesh.GetPrim(), material)
    if collision:
        _collider(mesh.GetPrim())
    return mesh


def _box(stage, path, size, position, material, rotation=None, collision=False):
    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    _transform(box, position, rotation)
    box.AddScaleOp().Set(Gf.Vec3f(*size))
    _bind(box.GetPrim(), material)
    if collision:
        _collider(box.GetPrim())
    return box


def _frame(angle_deg, origin):
    angle = math.radians(angle_deg)
    u = Gf.Vec3d(0, -1, 0)
    v = Gf.Vec3d(math.cos(angle), 0, math.sin(angle))
    n = Gf.Vec3d(-math.sin(angle), 0, math.cos(angle))
    rotation = Gf.Matrix3d(*u, *v, *n).ExtractRotation().GetQuat()

    def point(x, y, z=0):
        return Gf.Vec3d(*origin) + u * x + v * y + n * z
    return point, rotation, tuple(n)


def _solid_label(stage, path, text, size, position, rotation=None, color=(.905, .930, .947)):
    """Opaque font-outline triangles for printed instrument lettering."""
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath
    from shapely import constrained_delaunay_triangles
    from shapely.geometry import GeometryCollection, Polygon

    outline = TextPath((0, 0), text, size=1, prop=FontProperties(fname=FONT))
    bounds = outline.get_extents()
    scale = min(size[0] / bounds.width, size[1] / bounds.height)
    center = ((bounds.x0 + bounds.x1) / 2, (bounds.y0 + bounds.y1) / 2)
    geometry = GeometryCollection()
    for ring in outline.to_polygons():
        geometry = geometry.symmetric_difference(Polygon(ring))
    triangles = list(constrained_delaunay_triangles(geometry).geoms)
    if not triangles:
        raise ValueError(f"No font geometry for {text!r}")
    points = []
    for triangle in triangles:
        ring = list(triangle.exterior.coords)[:3]
        cross = ((ring[1][0] - ring[0][0]) * (ring[2][1] - ring[0][1])
                 - (ring[1][1] - ring[0][1]) * (ring[2][0] - ring[0][0]))
        if cross < 0:
            ring.reverse()
        points.extend(((x - center[0]) * scale, (y - center[1]) * scale, 0) for x, y in ring)
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([3] * len(triangles))
    mesh.CreateFaceVertexIndicesAttr(list(range(len(points))))
    mesh.CreateNormalsAttr([Gf.Vec3f(0, 0, 1)] * len(triangles))
    mesh.SetNormalsInterpolation("uniform")
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    _transform(mesh, position, rotation)
    material = _material(stage, path + "_material", color, .55)
    _bind(mesh.GetPrim(), material)


def _root(stage, path):
    root = UsdGeom.Xform.Define(stage, path)
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    root.GetPrim().AddAppliedSchema("PhysxArticulationAPI")
    root.GetPrim().CreateAttribute("physxArticulation:enabledSelfCollisions", Sdf.ValueTypeNames.Bool).Set(True)
    return root


def _button(stage, root, key, center, rotation, size, material):
    name = "key_" + KEY_NAMES.get(key, key)
    path = root + "/" + name
    _body(stage, path, .006, center, rotation)
    _rounded_box(stage, path + "/cap", (*size, .005), (0, 0, 0), .003,
                 material, collision=True)
    joint = UsdPhysics.PrismaticJoint.Define(stage, root + "/joints/" + name)
    joint.CreateBody0Rel().SetTargets([root + "/body"])
    joint.CreateBody1Rel().SetTargets([path])
    joint.CreateAxisAttr("Z")
    joint.CreateLocalPos0Attr(Gf.Vec3f(center))
    joint.CreateLocalRot0Attr(Gf.Quatf(rotation))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0))
    joint.CreateLocalRot1Attr(Gf.Quatf(1))
    joint.CreateLowerLimitAttr(-TRAVEL)
    joint.CreateUpperLimitAttr(0)
    joint.CreateCollisionEnabledAttr(False)
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(350.0)
    drive.CreateDampingAttr(.5)
    drive.CreateMaxForceAttr(10.0)
    drive.CreateTargetPositionAttr(0.0)
    drive.CreateTargetVelocityAttr(0.0)
    return {"joint": name, "body": name, "center": list(center),
            "rotation_wxyz": [rotation.GetReal(), *rotation.GetImaginary()],
            "travel_m": TRAVEL, "press_threshold_m": PRESS_THRESHOLD,
            "release_threshold_m": RELEASE_THRESHOLD}


def _digit(stage, path, width, height, point, center, rotation, ink):
    t = width * .22
    h = height / 2
    positions = {
        "a": (0, h - t / 2, width - t, t),
        "g": (0, 0, width - t, t),
        "d": (0, -h + t / 2, width - t, t),
        "f": (-width / 2 + t / 2, h / 2, t, h - t),
        "b": (width / 2 - t / 2, h / 2, t, h - t),
        "e": (-width / 2 + t / 2, -h / 2, t, h - t),
        "c": (width / 2 - t / 2, -h / 2, t, h - t),
    }
    for segment, (x, y, w, hh) in positions.items():
        position = point(center[0] + x, center[1] + y, center[2])
        if w > hh:
            outline = [(-w/2, 0), (-w/2+t/2, -hh/2), (w/2-t/2, -hh/2),
                       (w/2, 0), (w/2-t/2, hh/2), (-w/2+t/2, hh/2)]
        else:
            outline = [(-w/2, -hh/2+t/2), (0, -hh/2), (w/2, -hh/2+t/2),
                       (w/2, hh/2-t/2), (0, hh/2), (-w/2, hh/2-t/2)]
        mesh = UsdGeom.Mesh.Define(stage, path + "/" + segment)
        mesh.CreatePointsAttr([(px, py, 0) for px, py in outline])
        mesh.CreateFaceVertexCountsAttr([6])
        mesh.CreateFaceVertexIndicesAttr(list(range(6)))
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateDoubleSidedAttr(True)
        _transform(mesh, position, rotation)
        _bind(mesh.GetPrim(), ink)
    dot = UsdGeom.Sphere.Define(stage, path + "/dot")
    dot.CreateRadiusAttr(t * .58)
    _transform(dot, point(center[0] + width * .59, center[1] - h + t, center[2]), rotation)
    _bind(dot.GetPrim(), ink)


def _show_digit(stage, path, character, decimal=False):
    segments = SEGMENTS[character]
    for segment in "abcdefg":
        UsdGeom.Imageable(stage.GetPrimAtPath(path + "/" + segment)).GetVisibilityAttr().Set(
            "inherited" if segment in segments else "invisible")
    UsdGeom.Imageable(stage.GetPrimAtPath(path + "/dot")).GetVisibilityAttr().Set(
        "inherited" if decimal else "invisible")


def build_timer(stage, path):
    """White rounded desk timer with labelled blue keys and an open rear stand."""
    _root(stage, path)
    point, rotation, normal = _frame(65, (0, 0, .080))
    white = _material(stage, path + "/Looks/white", (.82, .84, .83), .32)
    seam = _material(stage, path + "/Looks/seam", (.45, .48, .47), .55)
    blue = _material(stage, path + "/Looks/blue", (.23, .59, .73), .32)
    dark = _material(stage, path + "/Looks/dark", (.018, .025, .030), .46)
    lcd = _material(stage, path + "/Looks/lcd", (.51, .57, .53), .52)
    ink = _material(stage, path + "/Looks/ink", (.006, .012, .008), .58)
    _body(stage, path + "/body", .24)
    _rounded_box(stage, path + "/body/housing", (.140, .145, .023), point(0, 0, -.0115),
                 .023, white, rotation)
    _rounded_box(stage, path + "/body/rear_seam", (.138, .143, .001), point(0, 0, -.019),
                 .022, seam, rotation)
    _rounded_box(stage, path + "/body/collider", (.132, .137, .016), point(0, 0, -.015),
                 .019, white, rotation, collision=True).GetVisibilityAttr().Set("invisible")
    for side in (-1, 1):
        _rounded_box(stage, path + f"/body/front_foot_{side + 1}", (.023, .016, .008),
                     (-.015, side * .044, .004), .004, white, collision=True)
        start = point(side * .050, .028, -.024)
        end = Gf.Vec3d(.083, -side * .050, .006)
        vector = end - start
        orient = Gf.Rotation(Gf.Vec3d(0, 0, 1), vector.GetNormalized()).GetQuat()
        _rounded_box(stage, path + f"/body/stand_side_{side + 1}",
                     (.009, .010, vector.GetLength()), (start + end) / 2,
                     .004, white, orient, collision=True)
        cylinder = UsdGeom.Cylinder.Define(stage, path + f"/body/stand_hinge_{side + 1}")
        cylinder.CreateRadiusAttr(.005)
        cylinder.CreateHeightAttr(.009)
        cylinder.CreateAxisAttr("Y")
        _transform(cylinder, start)
        _bind(cylinder.GetPrim(), seam)
    _rounded_box(stage, path + "/body/stand_crossbar", (.012, .111, .008),
                 (.083, 0, .004), .004, white, collision=True)
    _rounded_box(stage, path + "/body/screen_bezel", (.124, .069, .0016), point(0, .028, .0006),
                 .012, seam, rotation)
    _rounded_box(stage, path + "/body/screen", (.116, .061, .0008), point(0, .028, .0018),
                 .009, lcd, rotation)
    for i, x in enumerate((-.043, -.018, .018, .043)):
        _digit(stage, path + f"/body/display/digit_{i}", .022, .049,
               point, (x, .028, .0024), rotation, ink)
    for i, y in enumerate((.017, .039)):
        _box(stage, path + f"/body/display/colon_{i}", (.0033, .0040, .00015),
             point(0, y, .0024), ink, rotation)
    buttons = {}
    for key, x, width in (("START", -.043, .040), ("M", .007, .028), ("S", .046, .028)):
        buttons[key] = _button(stage, path, key, point(x, -.025, .004), rotation,
                               (width, .021), blue)
    text_color = (.012, .016, .017)
    for name, text, x, y, w, h in (
        ("start", "START", -.039, -.0435, .026, .006),
        ("pause", "PAUSE", -.039, -.0520, .026, .006),
        ("minutes", "M", .007, -.0440, .011, .009),
        ("seconds", "S", .046, -.0440, .011, .009),
        ("clear", "CLEAR", .026, -.0610, .031, .006),
    ):
        _solid_label(stage, path + "/body/label_" + name, text, (w, h),
                     point(x, y, .00035), rotation, text_color)
    _box(stage, path + "/body/start_divider", (.026, .00045, .0001),
         point(-.039, -.0479, .0003), ink, rotation)
    for name, x, y, w, h in (
        ("left", -.009, -.0518, .0006, .014),
        ("left_bottom", -.004, -.0585, .010, .0006),
        ("right", .061, -.0518, .0006, .014),
        ("right_bottom", .056, -.0585, .010, .0006),
    ):
        _box(stage, path + "/body/clear_bracket_" + name, (w, h, .0001),
             point(x, y, .0003), ink, rotation)
    _solid_label(stage, path + "/body/play_symbol", "▶", (.007, .009),
                 point(-.060, -.0475, .00035), rotation, text_color)
    update_timer_display(stage, path, TimerState())
    return {"root": path, "body": "body", "buttons": buttons, "face_normal": normal,
            "size_m": [.122, .140, .146], "initial_display": "00:15", "initial_running": True,
            "travel_m": TRAVEL, "press_threshold_m": PRESS_THRESHOLD,
            "release_threshold_m": RELEASE_THRESHOLD, "button_order_left_to_right": ["START", "M", "S"],
            "stand": "Open rear folding-stand geometry, rigid in this task"}


def build_calculator(stage, path):
    """Reference the calculator explicitly requested from the tape-measure task."""
    source = ASSETS / "calculator_model.usda"
    if not source.is_file():
        raise FileNotFoundError(source)
    reference = os.path.relpath(source, Path(stage.GetRootLayer().realPath).parent)
    root = stage.DefinePrim(path, "Xform")
    root.GetReferences().AddReference(reference, "/calculator")
    buttons = {}
    for row, keys in enumerate(CALCULATOR_LAYOUT):
        for col, key in enumerate(keys):
            name = KEY_NAMES.get(key, key)
            center = ((col - 2) * .019, .008 - row * .016, .011)
            if key == "+":
                center = (.038, -.046, .011)
            elif key == "=":
                center = (0., -.057, .011)
            buttons[key] = {"joint": "press_" + name, "body": "key_" + name,
                            "center": list(center), "rotation_wxyz": [1, 0, 0, 0],
                            "travel_m": .0015, "press_threshold_m": -.0008,
                            "release_threshold_m": -.00025}
    return {"root": path, "body": "base", "buttons": buttons, "face_normal": [0, 0, 1],
            "size_m": [.1016, .1397, .021], "initial_display": "0", "travel_m": .0015,
            "press_threshold_m": -.0008, "release_threshold_m": -.00025,
            "scene_rotation_z_deg": -90, "source": "tape-measure/assets/calculator.usda"}


def update_timer_display(stage, path, state):
    for index, character in enumerate(state.display.replace(":", "")):
        _show_digit(stage, path + f"/body/display/digit_{index}", character)


def update_calculator_display(stage, path, state):
    characters = []
    for character in state.display:
        if character == ".":
            characters[-1][1] = True
        else:
            characters.append([character, False])
    if len(characters) > 8:
        characters = [[c, False] for c in "Error"]
    characters = [[" ", False]] * (8 - len(characters)) + characters
    for index, (character, decimal) in enumerate(characters):
        for segment in "abcdefg":
            prim = stage.GetPrimAtPath(path + f"/base/display/digit_{index}_{segment}")
            UsdGeom.Imageable(prim).GetVisibilityAttr().Set(
                "inherited" if segment in SEGMENTS[character] else "invisible")
        dot = stage.GetPrimAtPath(path + f"/base/display/digit_{index}_point")
        UsdGeom.Imageable(dot).GetVisibilityAttr().Set("inherited" if decimal else "invisible")


def export_assets():
    ASSETS.mkdir(parents=True, exist_ok=True)
    metadata = {}
    for name, build in (("timer", build_timer), ("calculator", build_calculator)):
        stage = Usd.Stage.CreateNew(str(ASSETS / (name + ".usda")))
        UsdGeom.SetStageUpAxis(stage, "Z")
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        path = "/" + name.title()
        metadata[name] = build(stage, path)
        stage.SetDefaultPrim(stage.GetPrimAtPath(path))
        stage.GetRootLayer().Save()
    (ASSETS / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


if __name__ == "__main__":
    export_assets()
