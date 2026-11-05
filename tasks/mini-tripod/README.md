# Mini tripod

Requires Linux, an NVIDIA RTX GPU, and the [YAM Isaac Lab-Arena environment](https://github.com/kairosiann/yam-isaac-lab-arena/tree/bd62771e60320433893bd1d2df0fc79b5ddcd6d1). Tested versions and source commits are in `environment.json`. Activate that environment before running; no additional assets or textures need downloading for the tripod.

```bash
source /path/to/IsaacLab-Arena/yam.env
cd tasks/mini-tripod
python dream_eval.py --check  # Scene, cameras, action and contact check; no API calls
python verify.py              # Asset physics checks; fixtures are separate from the task
python expert.py              # Successful robot demo; no model API
python score.py demo output/demo
python dream_eval.py          # One Dream-1 trial; uses ~/.dreamlabs/credentials
python score.py dream output/dream-eval-YYYYMMDDTHHMMSSZ
python replay.py             # Render the demo trajectory
python replay.py --dream output/dream-eval-YYYYMMDDTHHMMSSZ
```

Use the run directory printed by `dream_eval.py` in the scoring and replay commands. Outputs are created automatically under `output/`; videos are under each run's `videos/`. Replay uses saved physical states and makes no API calls.

Fixed layout, seed 0, no randomization. Dream runs at 20 Hz with a 120 s simulation limit; its measured wall time was about 303 s on one RTX 5090. The demo takes 82.833 simulation seconds at 30 Hz. Physics runs at 480 Hz. The three robot cameras render at 640x360; the SDK sends 320x176 JPEGs to Dream. The external observer is used only for video.

Scoring runs automatically during evaluation. `score.py` independently rechecks the saved final five seconds; `demo` reconstructs collision geometry from recorded joints and root poses. The recorded runs score Dream 0/1 and demo 1/1. These are single fixed-layout trials.
