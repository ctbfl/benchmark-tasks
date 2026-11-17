"""Render measured physical trajectories; this does not execute or score a policy."""
import json
from pathlib import Path
import sys
import traceback

import numpy as np
from config import DT, HERE

FPS = 30
VIEWS = (('left_cam', 'Left Wrist Camera'), ('top_cam', 'Top Camera'), ('right_cam', 'Right Wrist Camera'))
EYE, TARGET = (.80, -.60, 1.32), (.30, .07, .86)


def render(out):
    import carb
    import carb.eventdispatcher
    import imageio_ffmpeg
    import omni.timeline
    from PIL import Image, ImageDraw, ImageFont
    from isaaclab_physx.renderers.isaac_rtx_renderer_utils import ensure_isaac_rtx_render_update
    from runtime import Experiment, CURTAIN, write_json
    from scene import make_env
    from score import score
    from types import SimpleNamespace
    from assets import display_cloth

    score(out)
    for name in ('scene.py', 'config.py', 'dream-defaults.json', 'assets/rod.usda', 'assets/stand.usda', 'assets/curtain.usda'):
        assert (HERE/name).read_bytes() == (out/'source'/name).read_bytes(), f'Source mismatch: {name}'
    states = np.load(out/'states.npy', allow_pickle=False, mmap_mode='r')
    schema = json.loads((out/'schema.json').read_text())
    mesh_states = np.load(out/'cloth.npz', allow_pickle=False)
    cloth_positions, cloth_faces = mesh_states['positions'], mesh_states['faces']
    fields = schema['fields']
    stride = round(1/(DT*FPS))
    assert abs(stride*DT-1/FPS) < 1e-12
    rows = states[stride::stride]
    assert abs(len(rows)/FPS-states[-1, 0]) < 1e-8
    assert np.allclose(mesh_states['times'][1:], rows[:, 0])
    for name in ('videos', 'frames'):
        (out/name).mkdir(exist_ok=True)
    from config import CONFIG
    from pxr import UsdGeom, Gf
    from scipy.spatial.transform import Rotation
    from yam_arena import YAM_URDF, station
    from yam_arena.robot import asset_link_paths
    from yam_arena.urdf import forward_kinematics
    # Offline drawing only: author recorded poses directly, without advancing physics.
    CONFIG['simulation']['use_fabric'] = False
    env = make_env(out, cameras=True)
    env.reset()
    base = env.unwrapped
    observer = Experiment.add_observer(SimpleNamespace(base=base), EYE, TARGET, (1920, 1080))
    writer = imageio_ffmpeg.write_frames(str(out/'videos/original.mp4'), (1920, 1440), fps=FPS,
        codec='libx264', pix_fmt_out='yuv420p', macro_block_size=2,
        output_params=['-an', '-profile:v', 'high', '-crf', '18', '-preset', 'fast',
                       '-threads', '4', '-movflags', '+faststart'])
    writer.send(None)
    font, small = ImageFont.load_default(size=32), ImageFont.load_default(size=24)
    settings = carb.settings.get_settings()
    paths = asset_link_paths()
    replay_ops = {}
    names = [CURTAIN+'/'+name for name in schema['rings']]+['/World/envs/env_0/'+n+'/body' for n in ('rod','stand')]
    names += ['/World/envs/env_0/'+side+'Arm'+path for side in ('Left','Right') for path in paths.values()]
    for path in names:
        xform = UsdGeom.Xformable(base.sim.stage.GetPrimAtPath(path))
        xform.ClearXformOpOrder()
        xform.SetResetXformStack(True)
        replay_ops[path] = xform.AddTransformOp(opSuffix='replay')
    expected_visual = {}
    def visual_pose(path, pose):
        m = np.eye(4)
        m[:3,:3] = Rotation.from_quat(pose[3:]).as_matrix().T
        m[3,:3] = pose[:3]
        replay_ops[path].Set(Gf.Matrix4d(*m.ravel()))
        expected_visual[path] = np.asarray(pose[:3])
    max_error, metadata = 0., {}
    try:
        for index, row in enumerate(rows):
            get = lambda name: np.asarray(row[slice(*fields[name])], dtype=np.float32)
            omni.timeline.get_timeline_interface().set_current_time(float(row[0]))
            q = get('robot_q')
            expected = get('rings').reshape(-1, 7)
            points = cloth_positions[index+1]
            display_cloth(base.sim.stage, CURTAIN, points, cloth_faces)
            for side_index, side in enumerate(('left', 'right')):
                robot = base.scene[side+'_robot']
                fk = forward_kinematics(YAM_URDF, dict(zip(robot.joint_names, q[side_index*8:side_index*8+8])))
                for name, path in paths.items():
                    transform = fk[name].copy()
                    transform[:3,3] += station.BASES[side]
                    full_path = '/World/envs/env_0/'+side.title()+'Arm'+path
                    replay_ops[full_path].Set(Gf.Matrix4d(*transform.T.ravel()))
                    expected_visual[full_path] = transform[:3,3]
            for name in ('rod', 'stand'):
                visual_pose('/World/envs/env_0/'+name+'/body', get(name))
            for name, pose in zip(schema['rings'], expected):
                visual_pose(CURTAIN+'/'+name, pose)
            cache = UsdGeom.XformCache()
            actual = np.array([cache.GetLocalToWorldTransform(base.sim.stage.GetPrimAtPath(path)).ExtractTranslation() for path in expected_visual])
            error = float(np.linalg.norm(actual-np.array(list(expected_visual.values())), axis=1).max())
            max_error = max(max_error, error)
            assert error < .0001, (index, error)
            settings.set_bool('/rtx-transient/post/dlss/forceParamReset', True)
            carb.eventdispatcher.get_eventdispatcher().dispatch_event(event_name='omni.rtx.ForceResetAccumulation')
            for _ in range(4 if index == 0 else 1):
                base.sim.render()
                ensure_isaac_rtx_render_update(force=True)
            pictures = {'observer': np.ascontiguousarray(observer.get_data()[..., :3])}
            assert pictures['observer'].shape == (1080, 1920, 3)
            for name, _ in VIEWS:
                camera = base.scene[name]
                camera.update(1/FPS, force_recompute=True)
                pictures[name] = camera.data.output['rgb'].torch[0, ..., :3].cpu().numpy().copy()
                assert pictures[name].shape == (360, 640, 3)
                if index == 0:
                    metadata[name] = dict(resolution=[640, 360], intrinsics=camera.data.intrinsic_matrices.torch[0].cpu().tolist())
            frame = Image.new('RGB', (1920, 1440))
            frame.paste(Image.fromarray(pictures['observer']))
            for i, (name, _) in enumerate(VIEWS):
                frame.paste(Image.fromarray(pictures[name]), (i*640, 1080))
            draw = ImageDraw.Draw(frame)
            labels = [(0, 0, 'External Observer View')]+[(i*640, 1080, title) for i, (_, title) in enumerate(VIEWS)]
            for x, y, title in labels:
                draw.rectangle((x, y, x+int(font.getlength(title))+31, y+(84 if y == 0 else 52)-1), fill='black')
                draw.text((x+16, y+10), title, font=font, fill='white')
                if y == 0:
                    draw.text((x+16, y+50), 'Not a robot camera', font=small, fill=(175, 215, 240))
            if index % (15*FPS) == 0 or index == len(rows)-1:
                frame.save(out/'frames'/f'video-{index:05d}.png')
            writer.send(np.asarray(frame))
            if index % 150 == 0:
                print(f'RENDER {index}/{len(rows)}', flush=True)
        write_json(out/'render.json', dict(frames=len(rows), fps=FPS, resolution=[1920, 1440],
            duration_s=len(rows)/FPS, max_link_position_error_m=max_error, cameras=metadata,
            observer=dict(eye=EYE, target=TARGET, fixed=True), source='states.npy',
            playback_speed=1., visual_poses='Recorded rigid poses and native URDF FK of measured joints; no physics steps.'))
    finally:
        writer.close()
        env.close()


if __name__ == '__main__':
    assert len(sys.argv) == 2, 'Usage: python render.py RUN_DIRECTORY'
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, enable_cameras=True,
        kit_args='--/plugins/carb.tasking.plugin/threadCount=8 --/plugins/omni.tbb.globalcontrol/maxThreadCount=8').app
    code = 0
    try:
        render(Path(sys.argv[1]).resolve())
    except BaseException:
        traceback.print_exc()
        code = 1
    finally:
        app.close(exit_code=code)
