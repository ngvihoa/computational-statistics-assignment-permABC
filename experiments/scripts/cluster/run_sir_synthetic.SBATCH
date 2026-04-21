#!/bin/bash
#SBATCH --nodes=1
#SBATCH -c 20
#SBATCH --job-name="sir_synth"
#SBATCH --output=%x.%J.out
#SBATCH --error=%x.%J.out
#SBATCH --mail-user=luciano@ceremade.dauphine.fr
#SBATCH --mail-type=BEGIN,END,FAIL

cd ~/permABC
source .venv/bin/activate

echo "=== SIR Synthetic Validation ==="
echo "Start: $(date)"

python3 experiments/scripts/run_sir_synthetic_validation.py \
    --seed 42 --K 10 --n_obs 120 --n_particles 1000 \
    --sigmas 0.01 0.05 0.10

echo "End: $(date)"
