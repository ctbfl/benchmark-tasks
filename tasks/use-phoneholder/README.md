# Unfold a phone holder and dock a phone

Unfold a freely resting phone holder and place a phone on it. Success requires a released phone supported by the holder, screen tilt of at least 45 degrees, phone bottom more than 3 cm above the table, and holder/phone yaw errors of at most 30 degrees relative to the red square. See [task.json](task.json) for the setup and scoring criteria.

Use the [YAM Arena setup](https://github.com/kairosiann/yam-isaac-lab-arena/tree/bd62771e60320433893bd1d2df0fc79b5ddcd6d1) to install the simulator, calibrated workstation and robot assets. Tested versions: Python 3.12, Isaac Sim 6.1, Isaac Lab 17.0.2, IsaacLab Arena 0.3.0 and YAM Arena 0.1.0. A supported NVIDIA GPU is required.

With that environment active, install the task's Python dependencies:

```bash
python -m pip install inspect-robots==0.60.0 inspect-robots-dream==0.1.0 inspect-robots-yam==0.37.0 dreamlabs-client==1.0.14 numpy==2.3.1 scipy==1.17.0 pillow==12.3.0 imageio-ffmpeg==0.6.0
```

Run the following commands from `tasks/use-phoneholder/`. Configure Dream SDK credentials before running the model trial.

```bash
python dream_eval.py --check  # Check scene, cameras and controls without model inference.
python dream_eval.py          # Run and score one Dream-1 trial.
python expert.py              # Run and validate the scripted demonstration.
```

Both runners score automatically. The Dream runner prints its output directory; `eval/` contains the Inspect Robots `success_at_end` score and `metrics.json` contains the final checks and measurements. To read the binary score, replace `<timestamp>` below with the timestamp of the completed trial:

```bash
python -c 'import json,sys; m=json.load(open(sys.argv[1])); assert m["exit_code"] == 0; print(int(m["success"]))' 'output/dream-eval-<timestamp>/metrics.json'
```

The scripted demonstration writes its checks and measurements to `output/demo/result.json` and exits with an error if validation fails. Each run saves its video in its output directory's `videos/` folder.

The trial limit is 180 simulated seconds at 20 Hz, including 32 initial history steps. Physics runs at 480 Hz. Seed: 0; fixed layout without object jitter. Allow approximately 15 minutes wall time for a model trial with recording, depending on hardware and API latency. The scripted demonstration takes 125.233 simulated seconds at 30 Hz.

The holder is a custom passive articulation. The iPhone visual model is by Ibrahim.Bhl, licensed under CC BY 4.0; see [attribution](assets/iphone17pro/ATTRIBUTION.txt) and [source metadata](assets/iphone17pro/source.json).
