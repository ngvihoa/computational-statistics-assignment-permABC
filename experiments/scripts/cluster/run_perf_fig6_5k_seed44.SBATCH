#!/bin/bash
#SBATCH --nodes=1
#SBATCH -c 20
#SBATCH --job-name="fig6_5k_s44"
#SBATCH --output=%x.%J.out
#SBATCH --error=%x.%J.out
#SBATCH --mail-user=luciano@ceremade.dauphine.fr
#SBATCH --mail-type=BEGIN,END,FAIL

# Fig 6 data: K=20, outliers=4, osum=True, seed=44, N_particles=5000.
# Methods: ABC-Vanilla, permABC-Vanilla, ABC-SMC, ABC-PMC,
#          permABC-SMC, permABC-SMC-OS, permABC-SMC-UM.

cd ~/permABC
source .venv/bin/activate

export PYTHONUNBUFFERED=1

python3 experiments/scripts/run_performance_comparison.py \
    --K 20 --K_outliers 4 --osum --seed 44 \
    --N_points 1000000 --N_particles 5000

echo "Fig 6 seed=44 N=5000 benchmark complete."
