#!/bin/bash
#SBATCH --nodes=1
#SBATCH -c 20
#SBATCH --job-name="test_dept"
#SBATCH --output=%x.%J.out
#SBATCH --error=%x.%J.out
#SBATCH --mail-user=luciano@ceremade.dauphine.fr
#SBATCH --mail-type=BEGIN,END,FAIL

cd ~/permABC
source .venv/bin/activate
export PYTHONUNBUFFERED=1

echo "=== Quick SIR test: departmental K=94, N=200 ==="
echo "Date: $(date)"
echo "Host: $(hostname)"

python3 experiments/scripts/run_sir_real_world.py \
    --seed 42 --scales departmental \
    --methods "permABC-SMC" \
    --n_particles 200 --final_iteration 10 \
    --sigma 0.05

echo "Done: $(date)"
