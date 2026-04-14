#!/bin/bash
#SBATCH --nodes=1
#SBATCH -c 20
#SBATCH --job-name="permABC_fig9"
#SBATCH --output=%x.%J.out
#SBATCH --error=%x.%J.out
#SBATCH --mail-user=luciano@ceremade.dauphine.fr
#SBATCH --mail-type=BEGIN,END,FAIL

cd ${SLURM_SUBMIT_DIR}
cd ~/permABC

source .venv/bin/activate

python3 experiments/scripts/figures/fig9_assignment_comparison.py \
    --K 20 --seed 42 --N_particles 1000
