#!/bin/bash
#SBATCH --nodes=1
#SBATCH -c 20
#SBATCH --job-name="permABC_sir"
#SBATCH --output=%x.%J.out
#SBATCH --error=%x.%J.out
#SBATCH --mail-user=luciano@ceremade.dauphine.fr
#SBATCH --mail-type=BEGIN,END,FAIL

cd ${SLURM_SUBMIT_DIR}
cd ~/permABC

source .venv/bin/activate

python3 experiments/scripts/run_sir_real_world.py \
    --seed 42 --scales regional \
    --methods "permABC-SMC" "ABC-SMC (Gibbs 3b)" "permABC-SMC (Gibbs 3b)" "ABC-Gibbs" "permABC-SMC-OS" "permABC-SMC-UM" \
    --n_particles 1000 --num_gibbs_blocks 3 --final_iteration 100
