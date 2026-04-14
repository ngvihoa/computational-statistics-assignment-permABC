#!/bin/bash
#SBATCH --nodes=1
#SBATCH -c 20
#SBATCH --job-name="sir_sens"
#SBATCH --output=%x.%J.out
#SBATCH --error=%x.%J.out
#SBATCH --mail-user=luciano@ceremade.dauphine.fr
#SBATCH --mail-type=BEGIN,END,FAIL

cd ~/permABC
source .venv/bin/activate

SEED=42
SCALE="departmental"
N_PARTICLES=1000
FINAL_ITER=100

echo "=== SIR Sensitivity Analysis ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "N_particles=$N_PARTICLES, Final_iter=$FINAL_ITER, Scale=$SCALE"
echo ""

for SIGMA in 0.01 0.05 0.10; do
    echo ""
    echo "=========================================="
    echo "  SIGMA = $SIGMA"
    echo "=========================================="

    # H=0: permABC-SMC (no Gibbs) + ABC-SMC baseline
    echo "--- permABC-SMC + ABC-SMC (sigma=$SIGMA) ---"
    python3 experiments/scripts/run_sir_real_world.py \
        --seed $SEED --scales $SCALE \
        --methods "permABC-SMC" "ABC-SMC" \
        --n_particles $N_PARTICLES --final_iteration $FINAL_ITER \
        --sigma $SIGMA

    # H=5: Gibbs with 5 blocks
    echo "--- Gibbs H=5 (sigma=$SIGMA) ---"
    python3 experiments/scripts/run_sir_real_world.py \
        --seed $SEED --scales $SCALE \
        --methods "ABC-SMC (Gibbs 5b)" "permABC-SMC (Gibbs 5b)" \
        --n_particles $N_PARTICLES --final_iteration $FINAL_ITER \
        --num_gibbs_blocks 5 --sigma $SIGMA

    # H=10: Gibbs with 10 blocks
    echo "--- Gibbs H=10 (sigma=$SIGMA) ---"
    python3 experiments/scripts/run_sir_real_world.py \
        --seed $SEED --scales $SCALE \
        --methods "ABC-SMC (Gibbs 10b)" "permABC-SMC (Gibbs 10b)" \
        --n_particles $N_PARTICLES --final_iteration $FINAL_ITER \
        --num_gibbs_blocks 10 --sigma $SIGMA

    # H=15: Gibbs with 15 blocks
    echo "--- Gibbs H=15 (sigma=$SIGMA) ---"
    python3 experiments/scripts/run_sir_real_world.py \
        --seed $SEED --scales $SCALE \
        --methods "ABC-SMC (Gibbs 15b)" "permABC-SMC (Gibbs 15b)" \
        --n_particles $N_PARTICLES --final_iteration $FINAL_ITER \
        --num_gibbs_blocks 15 --sigma $SIGMA

done

echo ""
echo "=== All runs complete: $(date) ==="
