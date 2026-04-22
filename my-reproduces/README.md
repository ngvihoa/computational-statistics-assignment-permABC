# Reproduce Stability and Accuracy Simulations

This folder contains independent reproduction scripts (no core package edits required).

## Structure

- `run_stability_accuracy.py`: CLI entrypoint
- `run_method_comparisons.py`: so sánh nhiều phương pháp ABC trên Gaussian toy model
- `repro_runner.py`: benchmark orchestration and algorithm execution
- `repro_models_gk.py`: G-and-K synthetic model
- `repro_models_lotka.py`: Lotka-Volterra synthetic model
- `repro_metrics.py`: accuracy/stability metrics and aggregation
- `repro_io.py`: CSV/JSON output writer

Supported models:
- `gk`: hierarchical G-and-K model
- `lotka`: hierarchical Lotka-Volterra ecosystem model

Supported algorithms:
- `perm`: permutation-enhanced ABC-SMC (`perm_abc_smc`)
- `smc`: standard ABC-SMC without permutation (`abc_smc`)

## Quick run

```bash
PYENV_VERSION=permabc pyenv exec python my-reproduces/run_stability_accuracy.py \
  --model gk --methods perm smc --seeds 0 1 2 --K 6 --n-obs 20 --n-particles 250 --n-iter-max 4
```

```bash
PYENV_VERSION=permabc pyenv exec python my-reproduces/run_stability_accuracy.py \
  --model lotka --methods perm smc --seeds 0 1 2 --K 6 --n-obs 30 --n-particles 250 --n-iter-max 4
```

```bash
# 1) So sánh: ABC-Vanilla, permABC-Vanilla, ABC-SMC, ABC-PMC, permABC-SMC
PYENV_VERSION=permabc pyenv exec python my-reproduces/run_method_comparisons.py \
  --task all --seeds 0 1 2 --K 8 --n-obs 40 --n-particles 1000 --n-iter-max 10

# 2) So sánh riêng độ ổn định: permABC-SMC vs ABC-Gibbs
PYENV_VERSION=permabc pyenv exec python my-reproduces/run_method_comparisons.py \
  --task gibbs --seeds 0 1 2 --K 8 --n-obs 40 --n-particles 1000 --n-iter-max 10
```

## Output

Results are written under `my-reproduces/results/`:
- `*_per_seed.csv`: per-seed metrics
- `*_summary.json`: aggregated mean/std metrics by method

Main metrics:
- `mae_global`: MAE on global parameters
- `mae_local_mean`: MAE on mean local parameter
- `runtime_sec`: total runtime per run
- `n_sim_total`: total simulations used
