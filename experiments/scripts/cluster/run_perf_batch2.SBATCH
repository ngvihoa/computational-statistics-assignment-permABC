#!/bin/bash
#SBATCH --nodes=1
#SBATCH -c 20
#SBATCH --job-name="perf_b2"
#SBATCH --output=%x.%J.out
#SBATCH --error=%x.%J.out
#SBATCH --mail-user=luciano@ceremade.dauphine.fr
#SBATCH --mail-type=END,FAIL

cd ~/permABC
source .venv/bin/activate

for pair in "5 1" "5 4" "10 1" "10 4" "15 1" "15 4" "20 1" "20 4" "30 1" "30 4" "50 1" "50 4"; do
    set -- $pair
    echo "=== K=$1, K_outliers=$2 ==="
    python3 experiments/scripts/run_performance_comparison.py \
        --K $1 --K_outliers $2 --no-osum --seed 42 \
        --N_points 1000000 --N_particles 1000
done

echo "Batch 2 complete."
