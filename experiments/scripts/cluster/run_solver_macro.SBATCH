#!/bin/bash
#SBATCH --nodes=1
#SBATCH -c 20
#SBATCH --job-name="permABC_solver_macro"
#SBATCH --output=%x.%J.out
#SBATCH --error=%x.%J.out
#SBATCH --mail-user=luciano@ceremade.dauphine.fr
#SBATCH --mail-type=BEGIN,END,FAIL

cd ${SLURM_SUBMIT_DIR}
cd ~/permABC

source .venv/bin/activate

# Macro only: full SMC comparison with cascade stats
# K=5,10,20 x 9 methods
python3 -u experiments/scripts/run_solver_final_benchmark.py \
    --level macro \
    --K 5 10 20 \
    --n-particles 1000 \
    --seed 42 \
    --nsim-factor 1000000
