# Unfold a laptop stand

Dual YAM arms unfold a passive laptop stand, seat both support rods in its highest slots, and release it inside the red rectangle. See `task.json` for the instruction, setup and numeric scoring rules.

Use Linux, an NVIDIA RTX GPU and FFmpeg. Install [yam-arena at bd62771](https://github.com/kairosiann/yam-isaac-lab-arena/tree/bd62771e60320433893bd1d2df0fc79b5ddcd6d1) using its `setup.sh`, then activate `source <ARENA_DIR>/yam.env`. Tested with Python 3.12, Isaac Sim 6.1.0.0, Isaac Lab package 17.0.2 and Arena 0.3.0 (commit `c7edf3a9d5d8276565208f72422f8f8a0806d6cf`). `requirements.txt` records the additional tested packages; install these in that environment. The stand USD is self-contained; robot assets and calibration come from yam-arena.

Run from this directory:

```bash
python dream_eval.py --check       # Scene, RGB, action and state checks; no API calls
python run.py expert              # Scripted demonstration; scored during execution
python dream_eval.py              # One Dream-1 trial; requires configured Dream API key
```

Dream uses the client's `~/.dreamlabs/credentials` configuration. Both runs use seed 0 and a fixed initial layout, with a limit of 2,400 steps at 20 Hz. Dream observes the calibrated top and wrist cameras (native 640x360, model input 320x176) and 14 measured joint positions. Actions are 14 joint/gripper targets. The observer camera provides video for human review. The expert uses measured mechanism state for planning and applies only robot joint targets.

`task.Score` evaluates measured geometry, contact, velocity and the one-second hold on every control step. Results and per-step measurements are saved under `outputs/`. Read the recorded scores and verify the archive with:

```bash
python -c 'import json; print(json.load(open("outputs/expert/headless/result.json"))["final"]["score"])'
python -m json.tool outputs/dream-highest-<timestamp>/metrics.json
python verify_dream.py outputs/dream-highest-<timestamp>  # Check archived actions, images and states
```

The Dream `metrics.json` reports the task's 0/0.25/0.5/0.75/1 score; Inspect Robots' `success_at_end` separately reports binary success. The submitted Dream trial scored 0 after 120 simulated seconds (493 seconds wall time on RTX 5090); the submitted robot demo scored 1 after 105.05 simulated seconds. A fresh headless expert run also scored 1 and took 164 seconds wall time; initialization and API latency vary.

Additional offline checks and video generation:

```bash
python run.py validate            # 34 physics checks, including all seven slots
python run.py exercise            # Force-fixture mechanism test
python check_scoring.py           # 23 scoring cases using that measured fixture
python run.py camera_ready        # Robot demo: 1080p observer + three 640x360 views
python verify_camera_ready.py     # Timing, calibration and complete video decode
python replay_dream.py outputs/dream-highest-<timestamp>  # Render archived states without API calls
```

Video uses real-time RTX at 30 FPS. Generated data, logs and media stay in `outputs/` and are excluded from version control. The model uses estimated rigid-body material and joint parameters; calibration against a physical product remains pending.
