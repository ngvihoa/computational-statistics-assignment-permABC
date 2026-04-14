"""
Shared plotting configuration for permABC experiment figures.

Centralises method colours, markers, linestyles, helper functions,
and matplotlib defaults so that every figure script uses a single source.
"""

import os
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ═══════════════════════════════════════════════════════════════════════════════
# ABC Method styles
# ═══════════════════════════════════════════════════════════════════════════════

METHOD_COLORS = {
    "ABC-Vanilla":      "#d62728",   # red
    "permABC-Vanilla":  "#2ca02c",   # green
    "ABC-SMC":          "#ff7f0e",   # orange
    "ABC-PMC":          "#ffbb78",   # light orange
    "ABC-Gibbs":        "#8c564b",   # brown
    "permABC-SMC":      "#1f77b4",   # blue
    "permABC-SMC-OS":   "#e377c2",   # pink
    "permABC-SMC-UM":   "#9467bd",   # purple
}

METHOD_MARKERS = {
    "ABC-Vanilla": "s",  "permABC-Vanilla": "s",
    "ABC-SMC": "o",      "ABC-PMC": "o",
    "ABC-Gibbs": "v",    "permABC-SMC": "o",
    "permABC-SMC-OS": "^", "permABC-SMC-UM": "^",
}

METHOD_LINESTYLES = {
    "ABC-Vanilla": "-",  "permABC-Vanilla": "-",
    "ABC-SMC": "--",     "ABC-PMC": "--",
    "ABC-Gibbs": "-.",   "permABC-SMC": "--",
    "permABC-SMC-OS": "--", "permABC-SMC-UM": "--",
}

METHOD_ORDER = [
    "ABC-Vanilla", "permABC-Vanilla",
    "ABC-SMC", "ABC-PMC", "ABC-Gibbs",
    "permABC-SMC", "permABC-SMC-OS", "permABC-SMC-UM",
]

METHODS_EXCLUDE_NO_OSUM = {"permABC-SMC-OS", "permABC-SMC-UM"}


# ═══════════════════════════════════════════════════════════════════════════════
# Assignment method styles (fig9)
# ═══════════════════════════════════════════════════════════════════════════════

ASSIGNMENT_COLORS = {
    "LSA":              "#1f77b4",
    "Hilbert":          "#ff7f0e",
    "Hilbert+Swap":     "#2ca02c",
    "Swap":             "#d62728",
    "Smart Swap":       "#9467bd",
    "Smart Hilbert":    "#8c564b",
    "Smart H+S":        "#e377c2",
    "Sinkhorn":         "#7f7f7f",
    "Sinkhorn+Swap":    "#bcbd22",
}

ASSIGNMENT_MARKERS = {
    "LSA": "o",          "Hilbert": "s",
    "Hilbert+Swap": "D", "Swap": "^",
    "Smart Swap": "v",   "Smart Hilbert": "<",
    "Smart H+S": ">",    "Sinkhorn": "P",
    "Sinkhorn+Swap": "X",
}

ASSIGNMENT_LINESTYLES = {
    "LSA": "-",           "Hilbert": "--",
    "Hilbert+Swap": "-.", "Swap": ":",
    "Smart Swap": "-",    "Smart Hilbert": "--",
    "Smart H+S": "-.",    "Sinkhorn": ":",
    "Sinkhorn+Swap": "-",
}


# ═══════════════════════════════════════════════════════════════════════════════
# Matplotlib setup
# ═══════════════════════════════════════════════════════════════════════════════

def setup_matplotlib(fontsize=12):
    """Apply consistent matplotlib rcParams for paper figures."""
    plt.rcParams.update({
        "font.size": fontsize,
        "axes.labelsize": fontsize,
        "axes.titlesize": fontsize + 2,
        "legend.fontsize": fontsize - 2,
        "xtick.labelsize": fontsize - 1,
        "ytick.labelsize": fontsize - 1,
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


# ═══════════════════════════════════════════════════════════════════════════════
# Common helpers
# ═══════════════════════════════════════════════════════════════════════════════

def save_figure(fig, path, dpi=300, close=True):
    """Save figure to *path*, creating parent directories as needed."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    if close:
        plt.close(fig)


def records_from_pkl(pkl_path):
    """Load summary records from a benchmark pickle file.

    Returns (records, raw_data) where *records* is a list[dict].
    """
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    df = data.get("summary_df")
    if df is None:
        raise ValueError(f"No summary_df in {pkl_path}")
    if hasattr(df, "to_dict"):
        return df.to_dict(orient="records"), data
    return list(df), data


def detect_joint_key(records):
    """Return 'score_joint' if present in records, else 'kl_joint' (legacy)."""
    if any("score_joint" in r for r in records):
        return "score_joint"
    return "kl_joint"


def joint_ylabel(ykey):
    """Return a LaTeX ylabel string for the joint score metric."""
    if ykey == "score_joint":
        return r"$-\mathbb{E}_q[\log p^*(\theta \mid y)]$"
    return r"$\mathrm{KL}_{\mathrm{joint}}$ (legacy)"


# Labels for extended diagnostic metrics
METRIC_LABELS = {
    "kl_sigma2":     r"$\mathrm{KL}(\hat{q} \| p)_{\sigma^2}$",
    "kl_mu_avg":     r"$\overline{\mathrm{KL}}(\hat{q} \| p)_{\mu_k}$",
    "w2_sigma2":     r"$W_2(\hat{q}, p)_{\sigma^2}$",
    "w2_mu_avg":     r"$\overline{W}_2(\hat{q}, p)_{\mu_k}$",
    "score_joint":   r"$-\mathbb{E}_q[\log p^*(\theta \mid y)]$",
    "sw2_joint":     r"$\mathrm{SW}_2(\hat{q}, p)_{\mathrm{joint}}$",
}


def metric_ylabel(ykey):
    """Return a LaTeX ylabel for any supported diagnostic metric."""
    return METRIC_LABELS.get(ykey, ykey)


def extract_series(records, method, xkey, ykey, positive_y=True):
    """Extract sorted (xs, ys) arrays for a given method from records.

    Filters out non-finite values and optionally non-positive y values.
    Returns sorted arrays by x.
    """
    xs, ys = [], []
    for r in records:
        if r.get("method") != method:
            continue
        x, y = r.get(xkey), r.get(ykey)
        if x is None or y is None:
            continue
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        if x <= 0:
            continue
        if positive_y and y <= 0:
            continue
        xs.append(float(x))
        ys.append(float(y))
    xs, ys = np.asarray(xs), np.asarray(ys)
    if xs.size == 0:
        return xs, ys
    order = np.argsort(xs)
    return xs[order], ys[order]


def plot_method_panel(ax, records, methods, xkey, ykey, xlabel, ylabel,
                      log_y=True, colors=None, markers=None, linestyles=None,
                      markersize=5, linewidth=1.5):
    """Plot a single panel with one line per method.

    Parameters
    ----------
    colors, markers, linestyles : dict, optional
        Override default METHOD_* dicts.
    """
    _c = colors or METHOD_COLORS
    _m = markers or METHOD_MARKERS
    _ls = linestyles or METHOD_LINESTYLES

    for m in methods:
        xs, ys = extract_series(records, m, xkey, ykey, positive_y=log_y)
        if xs.size == 0:
            continue
        ax.plot(
            xs, ys, label=m,
            color=_c.get(m, "gray"),
            marker=_m.get(m, "o"),
            linestyle=_ls.get(m, "-"),
            markersize=markersize,
            linewidth=linewidth,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)


# ═══════════════════════════════════════════════════════════════════════════════
# Project root utility
# ═══════════════════════════════════════════════════════════════════════════════

def find_project_root(start=None):
    """Walk up from *start* until pyproject.toml is found."""
    p = Path(start or __file__).resolve()
    for parent in [p] + list(p.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    return p.parents[2]  # fallback
