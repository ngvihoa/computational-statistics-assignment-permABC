#!/bin/bash
#SBATCH --nodes=1
#SBATCH -c 20
#SBATCH --job-name="perf_b1"
#SBATCH --output=%x.%J.out
#SBATCH --error=%x.%J.out
#SBATCH --mail-user=luciano@ceremade.dauphine.fr
#SBATCH --mail-type=END,FAIL

cd ~/permABC
source .venv/bin/activate

for pair in "5 0" "5 3" "10 0" "10 3" "15 0" "15 3" "20 0" "20 3" "30 0" "30 3" "50 0" "50 3"; do
    set -- $pair
    echo "=== K=$1, K_outliers=$2 ==="
    python3 experiments/scripts/run_performance_comparison.py \
        --K $1 --K_outliers $2 --no-osum --seed 42 \
        --N_points 1000000 --N_particles 1000
done

echo "Batch 1 complete."
