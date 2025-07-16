#!/bin/bash
#
# Script pour recréer toutes les figures du projet permABC.
#
# Lance chaque script de figure avec des paramètres par défaut (seed=42).
# Assurez-vous d'avoir installé les dépendances nécessaires
# et d'avoir installé le package en mode éditable (`pip install -e .`).

# --- Configuration ---
SEED=42
BASE_DIR="experiments/scripts/figures"

# --- Lancement des scripts ---

echo "--- Génération de la Figure 2: Comparaison a posteriori de l'over-sampling ---"
python "${BASE_DIR}/fig2_over_sampling_posterior.py" --seed $SEED

echo -e "\n--- Génération de la Figure 3: Comparaison a posteriori permABC vs Vanilla ---"
python "${BASE_DIR}/fig3_posterior_comparison_perm_vanilla.py" --seed $SEED

echo -e "\n--- Génération de la Figure 4: Comparaison des performances (sans OSUM) ---"
# Ce script appelle run_performance_comparison.py avec --no-osum
python "${BASE_DIR}/fig4_performance_comparison_without_osum.py" --seed $SEED --K 20 --K_outliers 0

echo -e "\n--- Génération de la Figure 5: Comparaison permABC-SMC vs Gibbs ---"
python "${BASE_DIR}/fig5_posterior_comparison_perm_smc_vs_gibbs.py" --seed $SEED --K 15

echo -e "\n--- Génération de la Figure 6: Comparaison des performances (avec OSUM) ---"
# Ce script appelle run_performance_comparison.py avec --osum
python "${BASE_DIR}/fig6_performance_comparison_with_osum.py" --seed $SEED --K 20 --K_outliers 4

echo -e "\n--- Génération de la Figure 7: Comparaison a posteriori SIR ---"
# Note : Ce script suppose que les données d'inférence existent.
python "${BASE_DIR}/fig7_posterior_comparison_sir_real_world.py" --seed $SEED 
python "${BASE_DIR}/fig7_posterior_comparison_sir_real_world.py" --seed $SEED --regions


echo -e "\n--- Génération de la Figure 8: Cartes des paramètres SIR ---"
python "${BASE_DIR}/fig8_map_local_sir_real_world.py" --seed $SEED
python "${BASE_DIR}/fig8_map_local_sir_real_world.py" --seed $SEED --regions


# --- Figures SIR (actuellement en commentaire) ---
# Décommentez ces sections une fois que les scripts seront prêts.

# echo -e "\n--- Génération de la Figure 7: Comparaison a posteriori SIR ---"
# # Note : Ce script suppose que les données d'inférence existent.
# # Il les générera si elles sont manquantes en appelant run_sir_real_world.py.
# python "${BASE_DIR}/fig7_posterior_comparison_sir_real_world.py" --seed $SEED
# python "${BASE_DIR}/fig7_posterior_comparison_sir_real_world.py" --seed $SEED --regions

# echo -e "\n--- Génération de la Figure 8: Cartes des paramètres SIR ---"
# # Génère la carte des départements
# python "${BASE_DIR}/fig8_map_local_sir_real_world.py" --seed $SEED
# # Génère la carte des régions
# python "${BASE_dir}/fig8_map_local_sir_real_world.py" --seed $SEED --regions

echo -e "\n✅ Toutes les figures ont été générées."