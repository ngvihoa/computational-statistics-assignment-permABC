#!/bin/bash
#SBATCH --nodes=1
#SBATCH -c 20
#SBATCH --job-name="sir_s001"
#SBATCH --output=%x.%J.out
#SBATCH --error=%x.%J.out
#SBATCH --mail-user=luciano@ceremade.dauphine.fr
#SBATCH --mail-type=BEGIN,END,FAIL

cd ~/permABC
source .venv/bin/activate
export PYTHONUNBUFFERED=1

SEED=42
N=500
FI=100
S=departmental

echo "=== SIR sigma=0.01 (N=$N, K=94) ==="
echo "Date: $(date), Host: $(hostname)"

for M in "permABC-SMC" "ABC-SMC"; do
    echo "--- $M ---"
    python3 experiments/scripts/run_sir_real_world.py \
        --seed $SEED --scales $S --methods "$M" \
        --n_particles $N --final_iteration $FI --sigma 0.01
done

for H in 5 10 15; do
    for M in "ABC-SMC (Gibbs ${H}b)" "permABC-SMC (Gibbs ${H}b)"; do
        echo "--- $M ---"
        python3 experiments/scripts/run_sir_real_world.py \
            --seed $SEED --scales $S --methods "$M" \
            --n_particles $N --final_iteration $FI --num_gibbs_blocks $H --sigma 0.01
    done
done

echo "Done: $(date)"
