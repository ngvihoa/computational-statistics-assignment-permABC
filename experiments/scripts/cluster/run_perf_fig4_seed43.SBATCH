#!/bin/bash
#SBATCH --nodes=1
#SBATCH -c 20
#SBATCH --job-name="perf_fig4_s43"
#SBATCH --output=%x.%J.out
#SBATCH --error=%x.%J.out
#SBATCH --mail-user=luciano@ceremade.dauphine.fr
#SBATCH --mail-type=BEGIN,END,FAIL

# Fig 4 data (paper): K=20, outliers=0, no-osum, seed=43.
# Runs vanilla + SMC + PMC (no OS/UM) -> 5 methods:
#   ABC-Vanilla, permABC-Vanilla, ABC-SMC, ABC-PMC, permABC-SMC.
# ABC-PMC is the long pole (~many hours). No --time cap set so
# it uses the partition default.

cd ~/permABC
source .venv/bin/activate

export PYTHONUNBUFFERED=1

python3 experiments/scripts/run_performance_comparison.py \
    --K 20 --K_outliers 0 --no-osum --seed 43 \
    --N_points 1000000 --N_particles 1000

echo "Fig 4 seed=43 benchmark complete."
