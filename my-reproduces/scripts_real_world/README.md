# Real-World Reproduction Scripts

This folder contains standalone scripts to apply permABC on real-world datasets.

## Scripts

- run_weather_real_world_reproduce.py (recommended)
- run_sir_real_world_reproduce.py (SIR/COVID variant)

## Weather workflow

The weather script uses my-reproduces/data/df_weather_clean.csv and runs the same
permABC benchmarking logic at multiple scales:

- national
- regional
- provincial

It builds a Gaussian hierarchical model and compares:

- permABC-SMC
- ABC-SMC
- ABC-SMC (Gibbs Hb)
- permABC-SMC (Gibbs Hb)
- permABC-SMC-OS
- permABC-SMC-UM

## Quick run (weather)

```bash
python my-reproduces/scripts_real_world/run_weather_real_world_reproduce.py \
  --seed 42 \
  --feature day.maxtemp_c \
  --scales regional provincial \
  --methods "permABC-SMC" "ABC-SMC" "permABC-SMC-OS" \
  --n_particles 500
```

## Full run (weather, all methods/scales)

```bash
python my-reproduces/scripts_real_world/run_weather_real_world_reproduce.py --seed 42
```

## Output

Default directory:

- my-reproduces/results/weather_real_world_inference

Saved files:

- inference_weather_<scale>_<feature>_<method_tag>_seed_<seed>.pkl
- comparison_weather_<feature>_seed_<seed>.csv
