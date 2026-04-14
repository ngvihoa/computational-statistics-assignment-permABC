#!/usr/bin/env python3
"""
Figure 8: Choropleth map of local SIR parameters (gamma/nu) by department.

Usage:
    python fig8_map_local_sir_real_world.py
    python fig8_map_local_sir_real_world.py --seed 42 --regions
"""

import sys
import argparse
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

try:
    import geopandas as gpd
except ImportError:
    print("ERROR: geopandas is required. Install it with: pip install geopandas")
    sys.exit(1)

# Shared plot config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plot_config import setup_matplotlib, save_figure, find_project_root

setup_matplotlib()

_PROJECT_ROOT = find_project_root()

# ── Paths ────────────────────────────────────────────────────────────────────

RESULTS_DIR = _PROJECT_ROOT / "experiments" / "results" / "sir_real_world_inference"
DATA_DIR = _PROJECT_ROOT / "experiments" / "data"
FIGURES_DIR = _PROJECT_ROOT / "experiments" / "figures" / "fig8"


def _fig_path(level, seed):
    return FIGURES_DIR / f"fig8_map_{level}_seed_{seed}.pdf"


# ── Data loading ─────────────────────────────────────────────────────────────

def load_inference(seed, level):
    file_path = RESULTS_DIR / f"inference_sir_{level}_seed_{seed}.pkl"
    print(f"Loading data from: {file_path}")
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, "rb") as f:
        return pickle.load(f)


def load_metadata():
    df_dep_hosp = pd.read_csv(DATA_DIR / "data-dep.csv", sep=";")
    df_dep_hosp = df_dep_hosp[df_dep_hosp["sexe"] == 0]
    excluded = {"971", "972", "973", "974", "976", "978", "2A", "2B"}
    df_dep_hosp = df_dep_hosp[~df_dep_hosp['dep'].isin(excluded)]
    dep_list = sorted(list(df_dep_hosp["dep"].unique()))

    df_pop = pd.read_csv(DATA_DIR / "donnees_departements.csv", sep=";")
    dep_name = {row["CODDEP"]: row["DEP"] for _, row in df_pop.iterrows()}
    reg_list = sorted(list(df_pop[df_pop['CODDEP'].isin(dep_list)]['CODREG'].unique()))
    reg_name = {row["CODREG"]: row["REG"] for _, row in df_pop.iterrows()}

    return {
        'dep_list': dep_list,
        'reg_list': reg_list,
        'dep_name': dep_name,
        'reg_name': reg_name,
    }


def prepare_map_data(inference_data, metadata, geo_df, level):
    if level == 'departmental':
        id_list = metadata['dep_list']
    else:
        id_list = metadata['reg_list']

    gamma_samples = inference_data['Thetas_final'].loc[:, :, 2]
    gamma_means = np.mean(gamma_samples, axis=0)

    if len(id_list) != len(gamma_means):
        raise ValueError(
            f"Metadata list length ({len(id_list)}) != inference results ({len(gamma_means)})"
        )

    results_df = pd.DataFrame({
        'code': [str(code) for code in id_list],
        'gamma_mean': gamma_means,
    })
    merged = geo_df.merge(results_df, on='code')
    print(f"Merged {len(merged)} {level}s.")
    return merged


# ── Plotting ─────────────────────────────────────────────────────────────────

def create_map_plot(map_data):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    map_data.plot(column='gamma_mean', cmap='viridis', linewidth=0.8,
                  ax=ax, edgecolor='black', legend=False)
    plt.colorbar(ax.collections[0], ax=ax, fraction=0.03, pad=0.04)
    ax.set_axis_off()
    return fig


# ── CLI ──────────────────────────────────────────────────────────────────────

_GEOJSON_URLS = {
    'departmental': "https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/departements-version-simplifiee.geojson",
    'regional': "https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/regions-version-simplifiee.geojson",
}


def main():
    parser = argparse.ArgumentParser(description="Figure 8: Map of local SIR parameters")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--regions', action='store_true', help='Plot regional map')
    args = parser.parse_args()

    level = "regional" if args.regions else "departmental"
    print(f"Figure 8: Map of {level} SIR parameters (seed={args.seed})")

    try:
        inference_data = load_inference(args.seed, level)
        metadata = load_metadata()
        geo_df = gpd.read_file(_GEOJSON_URLS[level])
        map_data = prepare_map_data(inference_data, metadata, geo_df, level)

        fig = create_map_plot(map_data)
        fig.tight_layout()
        fig_path = _fig_path(level, args.seed)
        save_figure(fig, fig_path)
        print(f"Figure saved to: {fig_path}")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
