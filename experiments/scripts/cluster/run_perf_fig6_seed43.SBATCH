#!/bin/bash
#SBATCH --nodes=1
#SBATCH -c 20
#SBATCH --job-name="perf_fig6_s43"
#SBATCH --output=%x.%J.out
#SBATCH --error=%x.%J.out
#SBATCH --mail-user=luciano@ceremade.dauphine.fr
#SBATCH --mail-type=BEGIN,END,FAIL

# Fig 6 data (paper): K=20, outliers=4, osum=True, seed=43.
# Runs vanilla + SMC + PMC + OS/UM -> 7 methods:
#   ABC-Vanilla, permABC-Vanilla, ABC-SMC, ABC-PMC,
#   permABC-SMC, permABC-SMC-OS, permABC-SMC-UM.
# ABC-PMC is the long pole (~11h at seed=42 run). No --time cap set
# so it uses the partition default.

cd ~/permABC
source .venv/bin/activate

export PYTHONUNBUFFERED=1

python3 experiments/scripts/run_performance_comparison.py \
    --K 20 --K_outliers 4 --osum --seed 43 \
    --N_points 1000000 --N_particles 1000

echo "Fig 6 seed=43 benchmark complete."
