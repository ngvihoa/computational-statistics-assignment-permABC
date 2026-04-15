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

export PYTHONUNBUFFERED=1

for pair in "30 0" "30 1" "30 2" "30 3" "30 4" "30 5" "50 0" "50 1" "50 2" "50 3" "50 4" "50 5"; do
    set -- $pair
    echo "=== K=$1, K_outliers=$2 (PMC+OSUM) ==="
    python3 experiments/scripts/run_performance_comparison.py \
        --K $1 --K_outliers $2 --methods pmc osum --seed 42 \
        --N_points 1000000 --N_particles 1000
done

echo "PMC+OSUM Batch 3 complete."
