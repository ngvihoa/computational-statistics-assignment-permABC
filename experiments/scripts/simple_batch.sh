#!/bin/bash

# Script to launch experiments overnight
# K=20, K_outliers=[4,0,2], seeds=[42,0], osum=true, plot=both

echo "Starting ABC performance experiments"
echo "Start time: $(date)"
echo "=================================="

# Configuration - MODIFY HERE TO ADJUST SPEED
K=20
OSUM="--osum"
PLOT="--plot both"
OUTPUT_DIR="experiments"

# CHOOSE YOUR CONFIGURATION:
# Option 1: Ultra-fast (3-4h total)
# N_POINTS="300000"
# N_PARTICLES="500"

# Option 2: Fast (6-8h total) 
# N_POINTS="500000"
# N_PARTICLES="800"

# Option 3: Standard (10-12h total)
N_POINTS="1000000" 
N_PARTICLES="1000"

echo "Chosen configuration:"
echo "- N_points: $N_POINTS"
echo "- N_particles: $N_PARTICLES"
echo "- Estimated total: see table above"
echo ""

# Function to run an experiment
run_experiment() {
    local k_outliers=$1
    local seed=$2
    local exp_name="K${K}_outliers${k_outliers}_seed${seed}"
    
    echo ""
    echo "Running: $exp_name"
    echo "Time: $(date)"
    echo "-----------------------------------"
    
    python run_performance_comparison.py \
        --K $K \
        --K_outliers $k_outliers \
        --seed $seed \
        --N_points $N_POINTS \
        --N_particles $N_PARTICLES \
        $OSUM \
        $PLOT \
        --output-dir $OUTPUT_DIR
    
    if [ $? -eq 0 ]; then
        echo "Success: $exp_name"
    else
        echo "Failure: $exp_name"
    fi
    
    echo "Finished at: $(date)"
}

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled by user"
    exit 1
fi

# Experiments with seed=42 (in order: 4, 0, 2 outliers)
echo "=== SEED 42 ==="
run_experiment 4 42  # Hardest case first
run_experiment 0 42  # Case without outliers  
run_experiment 2 42  # Intermediate case

# Experiments with seed=0 (same order: 4, 0, 2 outliers)
echo "=== SEED 0 ==="
run_experiment 4 0   # Hardest case first
run_experiment 0 0   # Case without outliers
run_experiment 2 0   # Intermediate case
