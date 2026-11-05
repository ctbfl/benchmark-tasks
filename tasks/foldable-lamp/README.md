# Unfold a desk lamp to illuminate a book

Position, unfold and switch on a passive desk lamp beside a closed book. See [task.json](task.json) for the fixed setup and numeric scoring criteria.

Install the [YAM Arena environment and calibrated workstation](https://github.com/kairosiann/yam-isaac-lab-arena/tree/bd62771e60320433893bd1d2df0fc79b5ddcd6d1). Tested with Python 3.12, Isaac Sim 6.1, Isaac Lab 17.0.2, IsaacLab Arena 0.3.0 and YAM Arena 0.1.0. An NVIDIA GPU with RTX support is required. With that environment active:

```bash
python -m pip install inspect-robots==0.60.0 inspect-robots-dream==0.1.0 inspect-robots-yam==0.37.0 dreamlabs-client==1.0.14 numpy==2.3.1 scipy==1.17.0 trimesh==4.11.1 pillow==12.3.0 imageio-ffmpeg==0.6.0
```

Run from this directory. Configure Dream SDK credentials before the model trial.

```bash
python dream_eval.py --check  # Load scene and check cameras/actions without model calls.
python dream_eval.py          # Run and score a Dream-1 trial.
python expert.py              # Run the scripted demonstration and its validation.
```

The model runner prints its directory under `runs/dream-eval-<timestamp>/`. It scores automatically using `success_at_end`; `metrics.json` records all checks. Read its binary score with:

```bash
python -c 'import json,sys; m=json.load(open(sys.argv[1])); assert m["exit_code"] == 0; print(m["score"])' runs/dream-eval-<timestamp>/metrics.json
```

The demo writes `runs/demo-<timestamp>/result.json` and fails if its validation fails. Videos are saved in each run's `videos/` folder. The benchmark score uses only base containment, shade height, shade tilt and lamp-on state within 120 s. The expert retains motion execution checks; recorded contact, displacement and drift measurements are diagnostics.

Seed: 0; fixed layout, no jitter. Limit: 120 simulated seconds at 20 Hz, including 32 initial history steps; physics: 480 Hz. The recorded model trial took 11 min 41 s wall time including API waits and rendering, scoring 0. The demo took 82.7 simulated seconds at 30 Hz and scored 1; allow several minutes wall time for recording. Hardware and API latency affect wall time.

The recorded Dream prompt requested a vertical upper support and a level shade. Its saved trajectory was regraded under the final 20 cm / 15 degree rubric, still scoring 0. The current runner states these thresholds explicitly. No new model trial was run for this regrade.

Assets are included; no LIBERO checkout is needed at runtime. See [asset provenance](assets/README.md). Lamp switching requires `touch.py`; the USD alone does not process touches.
