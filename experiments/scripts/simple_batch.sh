#!/bin/bash

# Script pour lancer les expériences de cette nuit
# K=20, K_outliers=[4,0,2], seeds=[42,0], osum=true, plot=both

echo "🚀 Démarrage des expériences de performance ABC"
echo "Heure de début: $(date)"
echo "=================================="

# Configuration - MODIFIEZ ICI POUR AJUSTER LA VITESSE
K=20
OSUM="--osum"
PLOT="--plot both"
OUTPUT_DIR="experiments"

# CHOISISSEZ VOTRE CONFIGURATION:
# Option 1: Ultra-rapide (3-4h total)
# N_POINTS="300000"
# N_PARTICLES="500"

# Option 2: Rapide (6-8h total) 
# N_POINTS="500000"
# N_PARTICLES="800"

# Option 3: Standard (10-12h total)
N_POINTS="1000000" 
N_PARTICLES="1000"

echo "Configuration choisie:"
echo "- N_points: $N_POINTS"
echo "- N_particles: $N_PARTICLES"
echo "- Estimation totale: voir tableau ci-dessus"
echo ""

# Fonction pour lancer une expérience
run_experiment() {
    local k_outliers=$1
    local seed=$2
    local exp_name="K${K}_outliers${k_outliers}_seed${seed}"
    
    echo ""
    echo "🔬 Lancement: $exp_name"
    echo "Heure: $(date)"
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
        echo "✅ Succès: $exp_name"
    else
        echo "❌ Échec: $exp_name"
    fi
    
    echo "Terminé à: $(date)"
}

# Lancement des 6 expériences dans l'ordre demandé
echo "Plan d'exécution:"
echo "1. K=20, K_outliers=4, seed=42  (le plus dur d'abord)"
echo "2. K=20, K_outliers=0, seed=42  (cas standard)"
echo "3. K=20, K_outliers=2, seed=42  (cas intermédiaire)"
echo "4. K=20, K_outliers=4, seed=0   (le plus dur, seed différent)"
echo "5. K=20, K_outliers=0, seed=0   (cas standard, seed différent)"
echo "6. K=20, K_outliers=2, seed=0   (cas intermédiaire, seed différent)"
echo ""

read -p "Continuer avec cette configuration? [y/N]: " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Annulé par l'utilisateur"
    exit 1
fi

# Expériences avec seed=42 (dans l'ordre: 4, 0, 2 outliers)
echo "=== SEED 42 ==="
run_experiment 4 42  # Le plus difficile d'abord
run_experiment 0 42  # Cas sans outliers  
run_experiment 2 42  # Cas intermédiaire

# Expériences avec seed=0 (même ordre: 4, 0, 2 outliers)
echo "=== SEED 0 ==="
run_experiment 4 0   # Le plus difficile d'abord
run_experiment 0 0   # Cas sans outliers
run_experiment 2 0   # Cas intermédiaire

echo ""
echo "🎉 TOUTES LES EXPÉRIENCES TERMINÉES"
echo "Heure de fin: $(date)"
echo "=================================="
echo "Résultats disponibles dans:"
echo "- experiments/results/*.pkl (données complètes)"
echo "- experiments/figures/fig6/ (graphiques)"