#!/bin/bash
#SBATCH --nodes=1
#SBATCH -c 20
#SBATCH --job-name="pmc_b2"
#SBATCH --output=%x.%J.out
#SBATCH --error=%x.%J.out
#SBATCH --mail-user=luciano@ceremade.dauphine.fr
#SBATCH --mail-type=END,FAIL

cd ~/permABC
source .venv/bin/activate

export PYTHONUNBUFFERED=1

for pair in "15 0" "15 1" "15 2" "15 3" "15 4" "15 5" "20 0" "20 1" "20 2" "20 3" "20 4" "20 5"; do
    set -- $pair
    echo "=== K=$1, K_outliers=$2 (PMC only) ==="
    python3 experiments/scripts/run_performance_comparison.py \
        --K $1 --K_outliers $2 --methods pmc --seed 42 \
        --N_points 1000000 --N_particles 1000
done

echo "PMC Batch 2 complete."
