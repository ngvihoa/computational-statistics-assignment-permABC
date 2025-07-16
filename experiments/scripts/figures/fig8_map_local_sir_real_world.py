#!/usr/bin/env python3
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import argparse
from pathlib import Path

try:
    import geopandas as gpd
except ImportError:
    print("❌ ERREUR: geopandas est nécessaire. Installez-le avec : pip install geopandas")
    sys.exit(1)

def preprocess_results(inference_data: dict, level: str) -> dict:
    """
    Loads metadata from CSV files and combines it with the raw inference results.
    """
    print("Preprocessing results: loading and attaching metadata...")
    project_root = Path(__file__).resolve().parents[3]
    data_dir = project_root / "experiments" / "data"
    
    # Load department and region lists/names from CSV files
    df_dep_hosp = pd.read_csv(data_dir / "data-dep.csv", sep=";")
    df_dep_hosp = df_dep_hosp[df_dep_hosp["sexe"] == 0]
    df_dep_hosp = df_dep_hosp[~df_dep_hosp['dep'].isin(["971", "972", "973", "974", "976", "978", "2A", "2B"])]
    dep_list = sorted(list(df_dep_hosp["dep"].unique()))

    df_pop = pd.read_csv(data_dir / "donnees_departements.csv", sep=";")
    dep_name = {row["CODDEP"]: row["DEP"] for _, row in df_pop.iterrows()}
    
    reg_list = sorted(list(df_pop[df_pop['CODDEP'].isin(dep_list)]['CODREG'].unique()))
    reg_name = {row["CODREG"]: row["REG"] for _, row in df_pop.iterrows()}

    # Create the metadata dictionary
    metadata = {
        'dep_list': dep_list,
        'reg_list': reg_list,
        'dep_name': dep_name,
        'reg_name': reg_name,
    }

    # Return a new, complete dictionary
    return {
        'result': inference_data,
        'metadata': metadata
    }
    
    
def load_inference(results_dir: str, seed: int, level: str) -> dict:
    file_path = Path(results_dir) / f"inference_sir_{level}_seed_{seed}.pkl"
    print(f"Loading lightweight data from: {file_path}")
    if not file_path.exists():
        raise FileNotFoundError(f"Fichier allégé non trouvé. Lancez d'abord le script 'lighten_sir_results.py'.")
    with open(file_path, "rb") as f:
        return pickle.load(f)

def get_department_list_and_names() -> tuple:
    """Loads CSV data to get the ordered list of departments and their names."""
    project_root = Path(__file__).resolve().parents[3]
    data_dir = project_root / "experiments" / "data"
    
    df_dep_hosp = pd.read_csv(data_dir / "data-dep.csv", sep=";")
    df_dep_hosp = df_dep_hosp[df_dep_hosp["sexe"] == 0]
    df_dep_hosp = df_dep_hosp[~df_dep_hosp['dep'].isin(["971", "972", "973", "974", "976", "978", "2A", "2B"])]
    dep_list = sorted(list(df_dep_hosp["dep"].unique()))

    df_pop = pd.read_csv(data_dir / "donnees_departements.csv", sep=";")
    dep_name = {row["CODDEP"]: row["DEP"] for _, row in df_pop.iterrows()}
    
    return dep_list, dep_name

def prepare_map_data(processed_data: dict, geo_df: pd.DataFrame, level: str) -> pd.DataFrame:
    """Merges processed inference results with geographic data."""
    # This function now expects the complete data structure
    metadata = processed_data['metadata']
    inference_result = processed_data['result']
    
    if level == 'departmental':
        id_list = metadata['dep_list']
    else: # regional
        id_list = metadata['reg_list']

    gamma_samples = inference_result['Thetas_final'].loc[:, :, 2]
    gamma_means = np.mean(gamma_samples, axis=0)

    if len(id_list) != len(gamma_means):
        raise ValueError("Inconsistency: Metadata list length does not match inference results length.")

    results_df = pd.DataFrame({'code': [str(code) for code in id_list], 'gamma_mean': gamma_means})
    merged_df = geo_df.merge(results_df, on='code')
    print(f"✅ Fusion réussie de {len(merged_df)} {level}s.")
    return merged_df


def create_map_plot(map_data: pd.DataFrame):
    """Creates the choropleth map."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    map_data.plot(column='gamma_mean', cmap='viridis', linewidth=0.8, ax=ax, edgecolor='black', legend=False)
    cbar = plt.colorbar(ax.collections[0], ax=ax, fraction=0.03, pad=0.04)
    # cbar.set_label("Mean Estimated $\\nu$", fontsize=12)
    ax.set_axis_off()
    # ax.set_title('Spatial Distribution of $\\nu$ Parameter', fontsize=16)
    return fig

def main():
    parser = argparse.ArgumentParser(description="Generate map of local SIR parameters")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--results-dir', type=str, default="experiments/results/sir_real_world_inference")
    parser.add_argument('--figure-dir', type=str, default="figures/fig8")
    parser.add_argument('--regions', action='store_true', help='Plot regional map')
    args = parser.parse_args()

    try:
        level = "regional" if args.regions else "departmental"
        
        # 1. Charger les résultats bruts de l'inférence
        raw_inference_data = load_inference(args.results_dir, args.seed, level)
        
        # 2. Pré-traiter les résultats pour ajouter les métadonnées
        processed_data = preprocess_results(raw_inference_data, level)

        # 3. Charger les données géographiques
        if level == 'departmental':
            url = "https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/departements-version-simplifiee.geojson"
        else:
            url = "https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/regions-version-simplifiee.geojson"
        geo_df = gpd.read_file(url)
            
        # 4. Préparer les données finales pour la carte
        map_data = prepare_map_data(processed_data, geo_df, level)
        
        # 5. Créer et sauvegarder le graphique
        fig = create_map_plot(map_data)
        plt.tight_layout()
        os.makedirs(args.figure_dir, exist_ok=True)
        fig_path = Path(args.figure_dir) / f"fig8_map_{level}_seed_{args.seed}.pdf"
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        
        print(f"\nFigure 8 ({level}) sauvegardée : {fig_path}")
        plt.show()
        plt.close(fig)

    except Exception as e:
        print(f"\n❌ ERREUR : {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
if __name__ == "__main__":
    main()