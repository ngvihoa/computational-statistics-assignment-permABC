#!/bin/bash
#SBATCH --nodes=1
#SBATCH -c 20
#SBATCH --job-name="pmc_b1"
#SBATCH --output=%x.%J.out
#SBATCH --error=%x.%J.out
#SBATCH --mail-user=luciano@ceremade.dauphine.fr
#SBATCH --mail-type=END,FAIL

cd ~/permABC
source .venv/bin/activate

export PYTHONUNBUFFERED=1

for pair in "2 0" "2 1" "5 0" "5 1" "5 2" "5 3" "5 4" "10 0" "10 1" "10 2" "10 3" "10 4" "10 5"; do
    set -- $pair
    echo "=== K=$1, K_outliers=$2 (PMC only) ==="
    python3 experiments/scripts/run_performance_comparison.py \
        --K $1 --K_outliers $2 --methods pmc --seed 42 \
        --N_points 1000000 --N_particles 1000
done

echo "PMC Batch 1 complete."
