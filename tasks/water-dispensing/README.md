# Dispense 150 g of tea using a digital scale

Run the commands below from `tasks/water-dispensing/`, with the configured IsaacLab-Arena Python environment activated.

## Environment

Python 3.12; Isaac Sim 6.1; Isaac Lab 17.0.2; IsaacLab Arena 0.3.0; YAM Arena 0.1.0; an NVIDIA GPU supported by this environment. Use the [YAM Arena setup instructions](https://github.com/kairosiann/yam-isaac-lab-arena/tree/bd62771e60320433893bd1d2df0fc79b5ddcd6d1) for the calibrated workstation. Policy packages: `inspect-robots==0.60.0`, `inspect-robots-dream==0.1.0`, `inspect-robots-yam==0.37.0`, `dreamlabs-client==1.0.14`. Configure Dream SDK credentials before a model trial.

## Run and score a trial

```bash
python dream_eval.py --check  # Optional local adapter check; no model API call.
python dream_eval.py          # One Dream-1 trial, including automatic scoring.
python yam_demo.py            # Scripted feasibility demo; no model API call.
```

The trial uses `dreamlabs/dream-1-beta`. Inspect Robots computes `success_at_end` automatically. Results are written to `runs/dream-150g-<timestamp>/eval/`, with final physical quantities in `metrics.json`.

Score 1 requires 140–160 g inclusive, a closed faucet, all airborne tea landed, at least 0.3 s of settled zero flow, and no overflow. Otherwise the score is 0. Full task details are in [task.json](task.json).

## Runtime and seed

- Limit: 120 simulated seconds, or 2400 control steps at 20 Hz, including 32 initial warmup steps. End early on success.
- Measured model-trial runtime: approximately 469 seconds of wall-clock time, including inference and rendering; timing depends on the workstation and API latency.
- Physics and liquid transport: 240 Hz.
- Seed: 0. One fixed starting layout, without object jitter.
- Reset: the adapter restores both arms and the faucet, empties and tares the cup, refills the reservoir, and clears airborne tea, overflow and elapsed time.

Assets are procedural; the wood texture is [Poly Haven Wood Table 001](https://polyhaven.com/a/wood_table_001), CC0. The scripted demo uses direct simulated scale feedback; Dream uses three 320×176 RGB inputs and joint state. Uploaded media: `M-E47VNE`.
