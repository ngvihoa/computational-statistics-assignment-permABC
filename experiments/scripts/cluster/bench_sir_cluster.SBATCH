#!/bin/bash
#SBATCH --nodes=1
#SBATCH -c 20
#SBATCH --job-name="bench_sir"
#SBATCH --output=%x.%J.out
#SBATCH --error=%x.%J.out
#SBATCH --mail-user=luciano@ceremade.dauphine.fr
#SBATCH --mail-type=BEGIN,END,FAIL

cd ~/permABC
source .venv/bin/activate

echo "=== SIR JAX vs Numba Benchmark (cluster, 20 cores) ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "CPUs: $(nproc)"
echo ""

python3 experiments/scripts/bench_sir_jax_vs_numba.py
