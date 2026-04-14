#!/bin/bash
#SBATCH --nodes=1
#SBATCH -c 20
#SBATCH --job-name="permABC_solver_bench"
#SBATCH --output=%x.%J.out
#SBATCH --error=%x.%J.out
#SBATCH --mail-user=luciano@ceremade.dauphine.fr
#SBATCH --mail-type=BEGIN,END,FAIL

cd ${SLURM_SUBMIT_DIR}
cd ~/permABC

source .venv/bin/activate

# Run both micro and macro benchmarks
# Micro: K=5,10,20,50 x n_obs=2,10,50 x 5 repeats
# Macro: K=5,10,20 x 9 methods (full SMC)
python3 experiments/scripts/run_solver_final_benchmark.py \
    --level both \
    --K 5 10 20 50 \
    --n-obs 2 10 50 \
    --n-particles 1000 \
    --n-particles-micro 500 \
    --n-repeats-micro 5 \
    --seed 42 \
    --nsim-factor 1000000
