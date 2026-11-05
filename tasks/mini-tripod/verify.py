"""Asset-level physics validation. Temporary fixtures replace robot actuation.

Run with --render to record the same experiment. No dynamic poses/joint states
are overwritten after reset. The only prescribed motion is a kinematic probe.
"""

from pathlib import Path
import json
import math
import sys
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parent
DT = 1 / 480
RELEASE = 12.
DURATION = 22.
PHASES = ((.5,"Folded; resting on table"), (2.,"Test fixture: lift by hub"),
          (4.,"Test fixture: open each leg"), (6.,"Test fixture: fold legs"),
          (8.,"Test fixture: open again"), (12.,"Test fixture: place at target"),
          (22.,"All fixture forces removed; passive support"))


def rotate(q, v):
    cross = 2 * np.cross(q[..., :3], v)
    return v + q[..., 3:4] * cross + np.cross(q[..., :3], cross)


def smooth(t, start, end):
    u = np.clip((t-start)/(end-start), 0, 1)
    return u*u*(3-2*u)


def run(render, out):
    import carb
    import omni.physics.tensors as tensors
    import omni.usd
    import trimesh
    from isaaclab.sim import SimulationCfg, SimulationContext
    from isaaclab.sim.utils import get_current_stage_id
    from isaaclab_physx.physics import PhysxCfg
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics
    from model import ASSETS, MASS, OPEN_DEG, export_assets, material, mesh_prim, rigid

    paths, heights = export_assets()
    physics_cfg = PhysxCfg(solver_type=1, enable_external_forces_every_iteration=True)
    sim = SimulationContext(SimulationCfg(dt=DT, device="cpu", use_fabric=False,
        physics=physics_cfg,
        log_dir=str(out/"logs"), render_interval=32))
    carb.settings.get_settings().set_bool("/physics/disableContactProcessing", False)
    stage = omni.usd.get_context().get_stage()

    def instance(name, variant, position, roll=0):
        prim = stage.DefinePrim("/"+name, "Xform")
        prim.GetReferences().AddReference(str(paths[variant]))
        x = UsdGeom.Xformable(prim)
        x.AddTranslateOp().Set(Gf.Vec3d(*position))
        x.AddOrientOp().Set(Gf.Quatf(Gf.Rotation(Gf.Vec3d(1,0,0),roll).GetQuat()))
        if render and name != "Tripod":
            UsdGeom.Imageable(prim).MakeInvisible()
        return prim

    instance("Tripod", "folded", [-.10,0,0])
    instance("DropOpen", "open", [-.35,1,.005], 1)
    instance("DropOpenTilt", "open", [0,1,.005], -1)
    instance("DropSide", "folded", [.35,1,.13], 90)
    instance("ProbeTripod", "folded", [.65,1,.14])
    hub_z = heights["folded"] + .14
    fixture = UsdPhysics.FixedJoint.Define(stage, "/ProbeFixture")
    fixture.CreateBody1Rel().SetTargets(["/ProbeTripod/hub"])
    fixture.CreateLocalPos0Attr(Gf.Vec3f(.65,1,hub_z))
    fixture.CreateLocalPos1Attr(Gf.Vec3f(0))
    probe = rigid(stage, "/Probe", [.65,1,hub_z-.055], Gf.Quatd(1), .1)
    UsdPhysics.RigidBodyAPI(probe.GetPrim()).CreateKinematicEnabledAttr(True)
    if render:
        UsdGeom.Imageable(probe.GetPrim()).MakeInvisible()
    PhysxSchema.PhysxRigidBodyAPI(probe.GetPrim()).CreateEnableCCDAttr(False)
    for path in ["/Probe", *[f"/Tripod/leg_{i}" for i in range(3)]]:
        PhysxSchema.PhysxContactReportAPI.Apply(stage.GetPrimAtPath(path)).CreateThresholdAttr(0.)
    table_mat = material(stage,"/TableMaterial",(.19,.23,.28),roughness=.85,friction=.8)
    probe_mat = material(stage,"/ProbeMaterial",(.7,.25,.12))
    mesh_prim(stage,"/Probe/Shape",trimesh.creation.box([.003,.005,.014]),probe_mat)
    table = trimesh.creation.box([2.3,2.3,.04])
    table.apply_translation([.1,.55,-.02])
    mesh_prim(stage,"/Table",table,table_mat)
    target_mat = material(stage,"/TargetMaterial",(.90,.55,.12),roughness=.9)
    ring = trimesh.creation.annulus(.102,.104,.00015,sections=96)
    ring.apply_translation([.10,0,.00015])
    mesh_prim(stage,"/Target",ring,target_mat,False)
    PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath("/physicsScene")).CreateEnableCCDAttr(True)
    recorder = None
    if render:
        from preview import Preview
        recorder = Preview(stage,out)
    stage.GetRootLayer().Export(str(out/"scene_initial.usda"))

    try:
        sim.reset()
        physics = tensors.create_simulation_view("numpy",stage_id=get_current_stage_id())
        physics.set_subspace_roots("/")
        main = physics.create_articulation_view("/Tripod")
        drops = {n:physics.create_articulation_view("/"+n) for n in ("DropOpen","DropOpenTilt","DropSide")}
        pushed = physics.create_articulation_view("/ProbeTripod")
        probe_view = physics.create_rigid_body_view("/Probe")
        contact = physics.create_rigid_contact_view("/Tripod/leg_*",["/Table"])
        probe_contact = physics.create_rigid_contact_view("/Probe",["/ProbeTripod/leg_0"])
        names = main.shared_metatype
        assert main.count == 1 and main.max_links == 4 and main.max_dofs == 3
        assert not names.fixed_base
        link = {n:i for i,n in enumerate(names.link_names)}
        dof = {n:i for i,n in enumerate(names.dof_names)}
        hinges = np.array([dof[f"fold_{i}"] for i in range(3)])
        print("Links:", names.link_names, "DOFs:", names.dof_names, flush=True)
        print("Masses:",main.get_masses().tolist(), "height:",heights,flush=True)
        print("Friction:",main.get_dof_friction_properties().tolist(),flush=True)
        vertices = dict(np.load(ASSETS/"collision_vertices.npz"))
        anchors = []
        for name in names.dof_names:
            j = UsdPhysics.Joint(stage.GetPrimAtPath("/Tripod/Joints/"+name))
            a,b = (link[getattr(j,f"GetBody{k}Rel")().GetTargets()[0].name] for k in (0,1))
            p0,p1 = (np.array(getattr(j,f"GetLocalPos{k}Attr")().Get()) for k in (0,1))
            anchors.append((a,b,p0,p1))
        indices = np.array([0],np.uint32)
        forces = np.zeros((1,4,3),np.float32)
        torques = np.zeros_like(forces)
        efforts = np.zeros((1,3),np.float32)
        history, poses_log, velocities_log, q_log, contacts_log = [],[],[],[],[]
        drop_log = {n:[] for n in drops}
        max_anchor_error = 0.
        max_probe_force = 0.
        max_probe_angle = 0.
        max_other_probe_angle = 0.
        min_surface_z = math.inf
        initial_side = drops["DropSide"].get_link_transforms()[0].copy()
        initial_side_v = drops["DropSide"].get_link_velocities()[0].copy()
        freefall_error = None
        prior_phase = -1
        snapshots = {round(t/DT):name for t,name in
            ((0,"folded.png"),(3.8,"opened.png"),(5.8,"refolded.png"),(12.8,"released.png"))}
        friction_before = main.get_dof_friction_properties().copy()
        if render:
            for _ in range(12):
                sim.render()

        for step in range(round(DURATION/DT)):
            t = step*DT
            phase = next(i for i,(end,_) in enumerate(PHASES) if t < end)
            changed = phase != prior_phase
            if changed:
                print(f"{t:.2f}s: {PHASES[phase][1]}",flush=True)
                prior_phase = phase
            p = main.get_link_transforms()[0].copy()
            v = main.get_link_velocities()[0].copy()
            q = main.get_dof_positions()[0].copy()
            qv = main.get_dof_velocities()[0].copy()
            assert np.isfinite(p).all() and np.isfinite(q).all(), "Nonfinite articulation state"
            assert np.linalg.norm(p[link["hub"],:3]) < 2, "Articulation escaped test workspace"
            forces.fill(0); torques.fill(0); efforts.fill(0)
            if .5 <= t < RELEASE:
                target_pos = np.array([-.10,0,heights["folded"]])
                target_pos[2] += (.245-heights["folded"])*smooth(t,.5,1.8)
                if t >= 8:
                    s = smooth(t,8,11.8)
                    target_pos = (1-s)*target_pos + s*np.array([.10,0,heights["open"]+.0001])
                hub = link["hub"]
                forces[0,hub] = np.clip(350*(target_pos-p[hub,:3])-10*v[hub,:3]+[0,0,MASS*9.81],-12,12)
                error = -2*p[hub,3:6]*np.sign(p[hub,6])
                torques[0,hub] = np.clip(np.array([.15,.15,.06])*error-
                    np.array([.009,.009,.0007])*v[hub,3:],-.05,.05)
                target = np.zeros(3)
                for i,h in enumerate(hinges):
                    target[h] = math.radians(OPEN_DEG+4)*smooth(t,2+i*.35,2.9+i*.35)
                if 4 <= t < 6:
                    s=smooth(t,4,5.8)
                    target[hinges] = (1-s)*target[hinges]-s*math.radians(4)
                elif t >= 6:
                    target[hinges] *= smooth(t,6,7.8)
                efforts[0,hinges] = np.clip(4*(target[hinges]-q[hinges])-.03*qv[hinges],-.30,.30)
                main.apply_forces_and_torques_at_position(forces,torques,None,indices,is_global=True)
                main.set_dof_actuation_forces(efforts,indices)
            elif step == round(RELEASE/DT):
                main.apply_forces_and_torques_at_position(forces,torques,None,indices,is_global=True)
                main.set_dof_actuation_forces(efforts,indices)
            # Push one sleeve by actual contact, then withdraw the test probe.
            if t <= 4.:
                radius = .027*smooth(t,1,2.8)*(1-smooth(t,3,3.8))
                probe_view.set_kinematic_targets(np.array([[.65,1-radius,hub_z-.055,0,0,0,1]],np.float32),indices)
            sim.step(render=False)
            if abs(t+DT-.08) < DT/2:
                elapsed = (step+1)*DT
                actual = drops["DropSide"].get_link_transforms()[0][:,:3]
                expected = initial_side[:,:3]+initial_side_v[:,:3]*elapsed+[0,0,-.5*9.81*elapsed**2]
                freefall_error = float(np.max(np.abs(actual-expected)))
            if step % 8 == 0:
                probe_force = np.linalg.norm(probe_contact.get_contact_force_matrix(DT))
                max_probe_force = max(max_probe_force,float(probe_force))
                pq = pushed.get_dof_positions()[0]
                max_probe_angle = max(max_probe_angle,float(np.rad2deg(pq[hinges[0]])))
                max_other_probe_angle = max(max_other_probe_angle,float(np.max(np.abs(np.rad2deg(pq[hinges[1:]])))))
                history.append([t,phase,float(np.linalg.norm(forces)+np.linalg.norm(torques)+np.linalg.norm(efforts))])
                poses_log.append(p); velocities_log.append(v); q_log.append(q)
                contacts_log.append(contact.get_contact_force_matrix(DT).copy())
                for n,view in drops.items():
                    dp = view.get_link_transforms()[0].copy()
                    dv = view.get_link_velocities()[0].copy()
                    dq = view.get_dof_positions()[0].copy()
                    assert np.isfinite(dp).all() and np.isfinite(dq).all()
                    drop_log[n].append((dp,dv,dq))
                for a,b,p0,p1 in anchors:
                    residual = p[b,:3]+rotate(p[b,3:],p1)-p[a,:3]-rotate(p[a,3:],p0)
                    max_anchor_error = max(max_anchor_error,float(np.linalg.norm(residual)))
                if t >= RELEASE:
                    for name,points in vertices.items():
                        pose = p[link[name]]
                        min_surface_z = min(min_surface_z,float((rotate(pose[3:],points)+pose[:3])[:,2].min()))
            if render and step % 32 == 0:
                sim.render()
                filename = snapshots.get(step)
                recorder.capture(PHASES[phase][1],t,np.rad2deg(q[hinges]),filename)

        h,p,v,q,cf = map(np.asarray,(history,poses_log,velocities_log,q_log,contacts_log))
        settled = h[:,0] >= DURATION-3
        released = h[:,0] >= RELEASE
        hub = link["hub"]
        tilt = np.rad2deg(np.arccos(np.clip(rotate(p[:,hub,3:],np.array([0,0,1]))[:,2],-1,1)))
        foot_forces = np.linalg.norm(cf.reshape(len(h),3,3),axis=2)
        awake_contact = released & (foot_forces.sum(axis=1) > .1)
        translation_drift = float(np.linalg.norm(np.ptp(p[settled,hub,:3],axis=0)))
        metrics = {"physics_dt_s":DT,"solver":"TGS","total_mass_kg":float(main.get_masses().sum()),
            "hinge_friction_static_dynamic_viscous":friction_before.tolist(),
            "hinge_armature_kg_m2":main.get_dof_armatures().tolist(),"fixture_torque_limit_Nm":.30,
            "placement_test_initial_height_m":.005,"placement_test_initial_tilt_degrees":1.,
            "heights_to_screw_tip_m":{n:z+.044 for n,z in heights.items()},
            "final_hinge_degrees":np.rad2deg(q[-1,hinges]).tolist(),
            "leg_structure":"Fixed short legs; no translational degrees of freedom",
            "final_hub_position_m":p[-1,hub,:3].tolist(),
            "final_hub_tilt_degrees":float(tilt[-1]),
            "target_xy_error_m":float(np.linalg.norm(p[-1,hub,:2]-[.10,0])),
            "drift_last_3s_m":translation_drift,
            "max_joint_anchor_error_m":max_anchor_error,
            "lowest_collision_surface_after_release_m":min_surface_z,
            "freefall_max_position_error_m":freefall_error,
            "probe_max_contact_force_N":max_probe_force,"probe_opening_degrees":max_probe_angle,
            "probe_other_leg_motion_degrees":max_other_probe_angle,
            "release_time_s":RELEASE,"unassisted_duration_s":DURATION-RELEASE,
            "peak_released_foot_contact_force_N":foot_forces[released].max(axis=0).tolist(),
            "mean_awake_support_N":float(foot_forces[awake_contact].sum(axis=1).mean()) if awake_contact.any() else 0.,
            "limitations":"Reference-derived geometry; internal dimensions, mass distribution and friction unmeasured. Fixed head. Test fixtures are not a robot policy."}
        def near(time, indices_, value, tolerance):
            samples = q[np.abs(h[:,0]-time)<.15][:,indices_]
            return bool(np.max(np.abs(samples-value)) < tolerance)
        checks = {
            "free_base_4_links_3_dofs":main.max_links==4 and main.max_dofs==3 and not names.fixed_base,
            "no_prismatic_joints":not any(prim.IsA(UsdPhysics.PrismaticJoint) for prim in stage.Traverse()),
            "mass_matches_reference":abs(metrics["total_mass_kg"]-MASS)<.0001,
            "no_joint_drives":bool(np.all(main.get_dof_stiffnesses()==0) and np.all(main.get_dof_dampings()==0)),
            "friction_unchanged":bool(np.array_equal(friction_before,main.get_dof_friction_properties())),
            "starts_folded":near(.25,hinges,0,math.radians(2)),
            "first_opening":near(3.9,hinges,math.radians(OPEN_DEG),math.radians(2)),
            "legs_refold":near(5.9,hinges,0,math.radians(2)),
            "second_opening":near(7.9,hinges,math.radians(OPEN_DEG),math.radians(2)),
            "all_fixture_forces_removed":bool(np.all(h[released,2]==0) and np.all(main.get_dof_actuation_forces()==0)),
            "three_feet_make_contact":bool(np.all(foot_forces[released].max(axis=0)>.03)),
            "settles_without_fixture":translation_drift<.001 and float(np.linalg.norm(v[settled,:,:3],axis=2).max())<.01,
            "legs_remain_fully_open":bool(np.max(np.abs(q[released][:,hinges]-math.radians(OPEN_DEG)))<math.radians(.5)),
            "stays_upright":bool(tilt[released].max()<3),
            "target_position_within_5mm":metrics["target_xy_error_m"]<.005,
            "no_significant_table_penetration":min_surface_z>-.0008,
            "joint_anchors_connected":max_anchor_error<.0005,
            "freefall_follows_gravity":freefall_error is not None and freefall_error<.002,
            "external_contact_opens_leg":max_probe_force>.02 and 25<max_probe_angle<OPEN_DEG+1,
            "legs_move_independently":max_other_probe_angle<3,
        }
        metrics["drop_tests"] = {}
        for n,samples in drop_log.items():
            dp,dv,dq = (np.array([row[k] for row in samples]) for k in range(3))
            drift = float(np.linalg.norm(np.ptp(dp[settled,hub,:3],axis=0)))
            speed = float(np.linalg.norm(dv[settled,:,:3],axis=2).max())
            low = min(float((rotate(dp[-1,link[name],3:],points)+dp[-1,link[name],:3])[:,2].min())
                      for name,points in vertices.items())
            up = rotate(dp[-1,hub,3:],np.array([0,0,1]))[2]
            metrics["drop_tests"][n] = {"drift_m":drift,"max_final_speed_m_s":speed,
                                      "lowest_surface_m":low,"upright_cosine":float(up),
                                      "final_joints":dq[-1].tolist()}
            np.savez_compressed(out/(n+".npz"),time=h[:,0],poses=dp,velocities=dv,joint_positions=dq)
            checks[n+"_settles"] = drift<.001 and speed<.01 and -.0008<low<.001
            if n != "DropSide":
                checks[n+"_upright"] = bool(up>math.cos(math.radians(3)))
        if render:
            checks["rendered_image_changes"] = recorder.max_change>1
            sim.render()
            recorder.capture("Released for 10 seconds; passive support",DURATION,
                             np.rad2deg(q[-1,hinges]),"final.png")
            recorder.look_at((.10,-.52,.27),(.10,0,.10))
            for _ in range(12):
                sim.render()
            recorder.capture("Detail: three hinged fixed-length legs",DURATION,
                             np.rad2deg(q[-1,hinges]),"detail.png",video=False)
        metrics["checks"] = checks
        (out/"metrics.json").write_text(json.dumps(metrics,indent=2)+"\n")
        np.savez_compressed(out/"trajectory.npz",time=h[:,0],phase=h[:,1],fixture_effort_norm=h[:,2],
            poses=p,velocities=v,joint_positions=q,foot_contact_forces=cf,link_names=names.link_names,dof_names=names.dof_names)
        stage.GetRootLayer().Export(str(out/"scene_final.usda"))
        print(json.dumps(metrics,indent=2),flush=True)
        assert all(checks.values()), f"Failed checks: {[k for k,v in checks.items() if not v]}"
    finally:
        if recorder:
            recorder.close()
        sim.stop()


if __name__ == "__main__":
    assert sys.argv[1:] in ([],["--render"]), "Usage: python verify.py [--render]"
    render = sys.argv[1:] == ["--render"]
    out = ROOT/"output"/("asset-demo" if render else "validation")
    (out/"logs").mkdir(parents=True,exist_ok=True)
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True,enable_cameras=render,kit_args=
        "--/plugins/carb.tasking.plugin/threadCount=8 --/plugins/omni.tbb.globalcontrol/maxThreadCount=8 "
        f"--/log/file={out}/logs/kit.log").app
    try:
        run(render,out)
    except Exception:
        traceback.print_exc()
        app.close(exit_code=1)
    else:
        app.close()
