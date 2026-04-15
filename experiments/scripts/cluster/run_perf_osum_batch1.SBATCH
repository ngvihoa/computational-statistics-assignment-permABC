#!/bin/bash
#SBATCH --nodes=1
#SBATCH -c 20
#SBATCH --job-name="osum_b1"
#SBATCH --output=%x.%J.out
#SBATCH --error=%x.%J.out
#SBATCH --mail-user=luciano@ceremade.dauphine.fr
#SBATCH --mail-type=END,FAIL

cd ~/permABC
source .venv/bin/activate

for pair in "5 0" "5 1" "5 2" "5 3" "10 0" "10 1" "10 2" "10 3"; do
    set -- $pair
    echo "=== K=$1, K_outliers=$2 (OSUM) ==="
    python3 experiments/scripts/run_performance_comparison.py \
        --K $1 --K_outliers $2 --osum --seed 42 \
        --N_points 1000000 --N_particles 1000
done

echo "OSUM Batch 1 complete."
