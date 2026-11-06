"""Run a local check, expert development or fixed-action demo."""
import json
import sys
import traceback
from datetime import datetime, timezone

from circuit import CONFIG, HERE


def main():
    assert len(sys.argv) == 2 and sys.argv[1] in ("check", "expert", "demo"), "Usage: python run.py check|expert|demo"
    mode = sys.argv[1]
    out = HERE/"runs"/(mode+"-"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    (out/"logs").mkdir(parents=True)
    (out/"frames").mkdir()
    print("RUN_DIRECTORY="+str(out), flush=True)
    (out/"config.json").write_text(json.dumps(CONFIG, indent=2)+"\n")
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, enable_cameras=mode=="check", kit_args=
        "--/plugins/carb.tasking.plugin/threadCount=8 --/plugins/omni.tbb.globalcontrol/maxThreadCount=8 "
        f"--/log/file={out}/logs/kit.log").app
    experiment, code = None, 0
    try:
        from assets import build
        from runtime import Experiment
        if not (HERE/"assets/rheostat.usda").exists():
            build()
        experiment = Experiment(out, cameras=mode=="check", screenshot=mode=="check")
        if mode == "check":
            import numpy as np
            from PIL import Image
            from runtime import write_json
            images = experiment.capture()
            for name, pixels in images.items():
                assert pixels.shape == (360,640,3) and pixels.std() > 10
                Image.fromarray(pixels).save(out/"frames"/(name+".png"))
            camera = experiment.base.scene["screenshot"]
            camera.update(1/20, force_recompute=True)
            top = experiment.base.scene["top_cam"]
            k = camera.data.intrinsic_matrices.torch[0].cpu().numpy()
            top_k = top.data.intrinsic_matrices.torch[0].cpu().numpy()
            assert np.allclose(k[:2],3*top_k[:2],atol=1e-3)
            assert np.allclose(camera.data.pos_w.torch,top.data.pos_w.torch)
            assert camera.cfg.offset.rot==top.cfg.offset.rot
            write_json(out/"calibration.json",dict(top_intrinsics=top_k.tolist(),
                screenshot_intrinsics=k.tolist(),position=camera.data.pos_w.torch[0].cpu().tolist(),
                rotation=list(camera.cfg.offset.rot),screenshot_resolution=[1920,1080]))
            pixels = camera.data.output["rgb"].torch[0,...,:3].cpu().numpy()
            assert pixels.shape == (1080,1920,3)
            Image.fromarray(pixels).save(out/"frames/screenshot.png")
            initial = images["left_cam"]
            target = experiment.ready.copy()
            target[0] += .12
            for _ in range(20):
                experiment.step(target)
            images = experiment.capture()
            change = float(abs(images["left_cam"].astype(float)-initial).mean())
            assert change > 1 and experiment.score.ticks == 240
            reference = CONFIG["robot_reference"]
            native = {}
            for side,robot in zip(("left","right"),experiment.robots):
                native[side] = {}
                for name,cfg in robot.cfg.actuators.items():
                    actual = dict(effort=cfg.joint_effort_limit,velocity=cfg.joint_velocity_limit,
                                  stiffness=cfg.stiffness,damping=cfg.damping)
                    for key,value in actual.items():
                        assert abs(value-reference[name][key]) < 1e-10
                    native[side][name] = actual
            write_json(out/"robot-parameters.json",native)
            write_json(out/"check.json", dict(passed=True, api_calls=0, wrist_image_change=change,
                initial_voltage_V=1., initial_slider_m=experiment.rows[0][46],
                free_base=not experiment.rheostat.is_fixed_base, passive=True))
        elif mode == "expert":
            from expert import execute
            execute(experiment)
        else:
            import numpy as np
            for action in np.load(HERE/"demo_actions.npy", allow_pickle=False):
                experiment.step(action)
            assert experiment.score.result()["success"], experiment.score.result()
        print("RESULT="+json.dumps(experiment.score.result()), flush=True)
    except BaseException:
        code = 1
        traceback.print_exc()
    finally:
        if experiment is not None:
            experiment.save()
            experiment.env.close()
        app.close(exit_code=code)


if __name__ == "__main__":
    main()
