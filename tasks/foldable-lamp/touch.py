"""Contact-only on/off switch. Instantiate after reset; call update after each step."""

from omni.physx import get_physx_simulation_interface
from omni.physx.bindings._physx import ContactEventType
from pxr import Gf, PhysicsSchemaTools, UsdLux, UsdShade


class TouchSwitch:
    def __init__(self, stage, root="/Lamp"):
        self.root = root
        self.pad = root+"/base/touch_pad"
        self.state = stage.GetPrimAtPath(root).GetAttribute("lamp:on")
        self.light = UsdLux.RectLight(stage.GetPrimAtPath(root+"/head/Light")).GetIntensityAttr()
        self.emission = UsdShade.Shader(stage.GetPrimAtPath(root+"/Looks/Diffuser/Surface")).GetInput("emissiveColor")
        assert self.state and self.light and self.emission
        self.contacts = set()
        self.armed = True
        self.released = 0.
        self.time = 0.
        self.events = []
        self.subscription = get_physx_simulation_interface().subscribe_contact_report_events(self._contact)

    def _contact(self, headers, data):
        for h in headers:
            a, b = (str(PhysicsSchemaTools.intToSdfPath(p)) for p in (h.collider0, h.collider1))
            if self.pad not in (a, b):
                continue
            other = b if a == self.pad else a
            if other.startswith(self.root+"/"):
                continue
            if h.type == ContactEventType.CONTACT_LOST:
                self.contacts.discard(other)
            else:
                self.contacts.add(other)

    def update(self, dt):
        self.time += dt
        if self.contacts:
            self.released = 0.
            if self.armed:
                on = not self.state.Get()
                self.state.Set(on)
                self.light.Set(50000. if on else 0.)
                self.emission.Set(Gf.Vec3f(5., 4.55, 3.85) if on else Gf.Vec3f(0))
                self.events.append({"time_s": self.time, "on": on, "colliders": sorted(self.contacts)})
                self.armed = False
        else:
            self.released += dt
            if self.released >= .08:
                self.armed = True

    def close(self):
        self.subscription = None
