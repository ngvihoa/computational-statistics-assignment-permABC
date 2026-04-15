#!/bin/bash
#SBATCH --nodes=1
#SBATCH -c 20
#SBATCH --job-name="osum_b3"
#SBATCH --output=%x.%J.out
#SBATCH --error=%x.%J.out
#SBATCH --mail-user=luciano@ceremade.dauphine.fr
#SBATCH --mail-type=END,FAIL

cd ~/permABC
source .venv/bin/activate

for pair in "30 0" "30 1" "30 2" "30 3" "50 0" "50 1" "50 2" "50 3"; do
    set -- $pair
    echo "=== K=$1, K_outliers=$2 (OSUM) ==="
    python3 experiments/scripts/run_performance_comparison.py \
        --K $1 --K_outliers $2 --osum --seed 42 \
        --N_points 1000000 --N_particles 1000
done

echo "OSUM Batch 3 complete."
