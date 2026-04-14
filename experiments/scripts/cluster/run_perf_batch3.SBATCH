#!/bin/bash
#SBATCH --nodes=1
#SBATCH -c 20
#SBATCH --job-name="perf_b3"
#SBATCH --output=%x.%J.out
#SBATCH --error=%x.%J.out
#SBATCH --mail-user=luciano@ceremade.dauphine.fr
#SBATCH --mail-type=END,FAIL

cd ~/permABC
source .venv/bin/activate

for pair in "2 0" "2 1" "5 2" "10 2" "10 5" "15 2" "15 5" "20 2" "20 5" "30 2" "30 5" "50 2" "50 5"; do
    set -- $pair
    echo "=== K=$1, K_outliers=$2 ==="
    python3 experiments/scripts/run_performance_comparison.py \
        --K $1 --K_outliers $2 --no-osum --seed 42 \
        --N_points 1000000 --N_particles 1000
done

echo "Batch 3 complete."
