"""Force-fixture demonstration of unfolding, changing sockets and relocating.

This is a mechanical test, not a robot policy or a scripted pose animation.
"""

import json
import subprocess
import numpy as np
from scipy.spatial.transform import Rotation

import model
from runtime import Mechanism, simulation
from validate import seated


def smooth(t):
    return t*t*(3-2*t)


def setup_poses(width, theta, phi):
    poses = model.configuration(width, gear=0, angle=theta)
    for side in ("L", "R"):
        p, _ = poses["prop_"+side]
        poses["prop_"+side] = (p, model.ry(phi))
    return poses


class Fixture:
    """Finite external forces/torques on named bodies; all contacts stay active."""

    def __init__(self, holder):
        self.holder = holder
        self.max_force = 0.0
        self.max_torque = 0.0

    def follow(self, targets):
        h = self.holder
        state = h.state()
        forces, torques = {}, {}
        for name, (target_p, target_r) in targets.items():
            p, r, v = state[name]
            error = Rotation.from_matrix(target_r @ r.T).as_rotvec()
            if name.startswith("prop"):
                f = np.zeros(3)
                t = .8*error - .012*v[3:] - np.cross(r@h.com[name], [0,0,-h.mass[name]*9.81])
                t_limit = .05
            else:
                f = 800*(target_p + target_r@h.com[name] - p - r@h.com[name]) - 12*v[:3]
                f += [0,0,h.mass[name]*9.81]
                t = 1.5*error - .04*v[3:]
                t_limit = .2
            f *= min(1.0, 8.0 / max(np.linalg.norm(f), 1e-12))
            t *= min(1.0, t_limit / max(np.linalg.norm(t), 1e-12))
            forces[name], torques[name] = f, t
            self.max_force = max(self.max_force, float(np.linalg.norm(f)))
            self.max_torque = max(self.max_torque, float(np.linalg.norm(t)))
        h.wrench(forces, torques)


def run(render=True):
    sim = simulation()
    model.build(sim.stage, "/World/Holder")
    mark = model.material(sim.stage,"/World/materials/goal",(.12,.42,.28),(.5,.4))
    for i,(size,pos) in enumerate([
        ((.33,.002,.001),(.105,.19,.0005)), ((.33,.002,.001),(.105,.41,.0005)),
        ((.002,.22,.001),(-.06,.30,.0005)), ((.002,.22,.001),(.27,.30,.0005))]):
        model.shape(sim.stage,f"/World/Goal/edge_{i}",model.block(size,pos),mark,False)
    folder = model.ROOT/"outputs/demo"
    folder.mkdir(exist_ok=True)
    if render:
        import omni.replicator.core as rep
        import cv2
        from isaaclab_physx.renderers.isaac_rtx_renderer_utils import ensure_isaac_rtx_render_update
        camera=rep.create.camera(position=(.53,-.65,.50),look_at=(.10,.13,.055),focal_length=30,clipping_range=(.01,10))
        product=rep.create.render_product(camera,(960,720))
        rgb=rep.AnnotatorRegistry.get_annotator("rgb")
        rgb.attach(product)
        (folder/"videos").mkdir(exist_ok=True)
        video=subprocess.Popen(["ffmpeg","-hide_banner","-loglevel","error","-y","-f","rawvideo","-pix_fmt","rgb24",
            "-s","960x720","-r","30","-i","-","-an","-c:v","libx264","-crf","20","-pix_fmt","yuv420p",
            "-movflags","+faststart",str(folder/"videos/mechanism.mp4")],stdin=subprocess.PIPE)
    sim.reset()
    h=Mechanism(sim,"/World/Holder")
    fixture=Fixture(h)
    report={"kind":"external force fixture; no robot policy", "physics_hz":round(1/sim.cfg.dt), "phases":[]}
    tick=0

    def phase(label, seconds, controller=None):
        nonlocal tick
        for i in range(round(seconds/sim.cfg.dt)):
            if controller:
                controller(smooth(i/max(1,round(seconds/sim.cfg.dt)-1)))
            sim.step(render=False)
            if render and tick%round(1/(30*sim.cfg.dt))==0:
                sim.render()
                ensure_isaac_rtx_render_update(force=True)
                pixels=rgb.get_data()
                if pixels.size:
                    frame=np.ascontiguousarray(pixels[...,:3]).copy()
                    cv2.rectangle(frame,(0,0),(960,88),(24,28,34),-1)
                    cv2.putText(frame,"PHYSX | PASSIVE LAPTOP STAND",(24,31),cv2.FONT_HERSHEY_SIMPLEX,.72,(235,235,235),1,cv2.LINE_AA)
                    cv2.putText(frame,label+"  /  external force fixture",(24,65),cv2.FONT_HERSHEY_SIMPLEX,.55,(185,215,205),1,cv2.LINE_AA)
                    video.stdin.write(frame.tobytes())
            tick+=1
        m=h.measure()
        report["phases"].append({"name":label,"time":tick*sim.cfg.dt,"state":m})
        print(label, json.dumps(m), flush=True)
        if render:
            cv2.imwrite(str(folder/(label.replace(" ","_")+".png")), cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
        return m

    names=("base_L","base_R","deck_L","deck_R","prop_L","prop_R")
    def guide(width,theta,phi):
        poses=setup_poses(width,theta,phi)
        fixture.follow({n:poses[n] for n in names})

    try:
        phase("01 Folded",1)
        phase("02 Spread",3,lambda t:guide(model.CLOSED_WIDTH+t*(model.OPEN_WIDTH-model.CLOSED_WIDTH),0,0))
        high=np.deg2rad(52)
        phase("03 Lift decks",2,lambda t:guide(model.OPEN_WIDTH,t*high,-t*high))
        last_theta=high
        last_phi=-high
        for gear in (2,4):
            theta=model.gear_angle(gear)
            prop_r=model.configuration(gear=gear)["prop_L"][1]
            phi=Rotation.from_matrix(prop_r).as_rotvec()[1]
            if last_theta != high:
                phase("06 Withdraw pins",2,lambda t:guide(model.OPEN_WIDTH,last_theta+t*(high-last_theta),last_phi))
            phase(f"Select gear {gear}",2,lambda t:guide(model.OPEN_WIDTH,high,last_phi+t*(phi-last_phi)))
            phase(f"Seat gear {gear}",2,lambda t:guide(model.OPEN_WIDTH,high+t*(theta-high),phi))
            m=phase(f"Release gear {gear}",2)
            assert seated(m,gear), f"Gear {gear} not seated"
            last_theta,last_phi=theta,phi
        base={n:(p.copy(),r.copy()) for n,(p,r,_) in h.state().items() if n.startswith("base")}
        def transport(delta):
            fixture.follow({n:(p+delta,r) for n,(p,r) in base.items()})
        phase("09 Lift by bases",2,lambda t:transport(np.array([0,0,.07*t])))
        phase("10 Relocate",3,lambda t:transport(np.array([0,.30*t,.07])))
        phase("11 Lower to target",2,lambda t:transport(np.array([0,.30,.07*(1-t)])))
        m=phase("12 Released in target",3)
        assert seated(m,4), "Gear lost during transport"
        assert m["low"][0]>-.06 and m["high"][0]<.27 and m["low"][1]>.19 and m["high"][1]<.41
        assert m["max_speed"]<.01 and m["max_spin"]<.1, "Not settled"
        report["passed"]=True
    finally:
        report["max_fixture_force_N"]=fixture.max_force
        report["max_fixture_torque_Nm"]=fixture.max_torque
        (folder/"mechanics.json").write_text(json.dumps(report,indent=2)+"\n")
        if render:
            video.stdin.close()
            assert video.wait(timeout=30)==0
        sim.stop()
