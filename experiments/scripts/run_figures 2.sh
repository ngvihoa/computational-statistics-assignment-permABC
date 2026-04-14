#!/bin/bash
# Run paper figures 4bis, 6/6bis, 9, and 10.
#
# Usage:
#   bash experiments/scripts/run_figures.sh              # all figures, default params
#   bash experiments/scripts/run_figures.sh --fig 4 6    # only fig4bis and fig6
#   bash experiments/scripts/run_figures.sh --K 10       # override K
#
# Environment: requires the permabc pyenv virtualenv.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FIGURES_DIR="${SCRIPT_DIR}/figures"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ── Defaults ──────────────────────────────────────────────────────────────────
K=20
K_OUTLIERS=4
SEED=42
FIGS="4 6 9 10"

# ── Parse CLI args ────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --fig)   shift; FIGS=""; while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do FIGS="$FIGS $1"; shift; done ;;
        --K)     shift; K="$1"; shift ;;
        --K_outliers) shift; K_OUTLIERS="$1"; shift ;;
        --seed)  shift; SEED="$1"; shift ;;
        *)       echo "Unknown arg: $1"; exit 1 ;;
    esac
done

PYTHON="python"
if command -v pyenv &>/dev/null; then
    PYTHON="PYENV_VERSION=permabc pyenv exec python"
fi

run() {
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "  $1"
    echo "════════════════════════════════════════════════════════════════"
    eval "$2"
}

cd "$PROJECT_ROOT"

for fig in $FIGS; do
    case "$fig" in
        4|4bis)
            run "Figure 4bis: Performance comparison (no OSUM)" \
                "$PYTHON ${FIGURES_DIR}/fig4bis_performance_comparison_without_osum.py --K $K --K_outliers 0 --seed $SEED"
            ;;
        6)
            run "Figure 6: Performance comparison (with OSUM)" \
                "$PYTHON ${FIGURES_DIR}/fig6_performance_comparison_with_osum.py --K $K --K_outliers $K_OUTLIERS --seed $SEED"
            run "Figure 6bis: Performance comparison (with OSUM, panels)" \
                "$PYTHON ${FIGURES_DIR}/fig6bis_performance_comparison_with_osum.py --K $K --K_outliers $K_OUTLIERS --seed $SEED"
            ;;
        9)
            run "Figure 9: Assignment method comparison" \
                "$PYTHON ${FIGURES_DIR}/fig9_assignment_comparison.py --K 5 10 20 --seed $SEED"
            ;;
        10)
            run "Figure 10: Outlier influence" \
                "$PYTHON ${FIGURES_DIR}/fig10_outlier_influence.py --K $K --seed $SEED"
            ;;
        *)
            echo "WARNING: Unknown figure number: $fig (available: 4, 6, 9, 10)"
            ;;
    esac
done

echo ""
echo "Done. Figures saved in experiments/figures/"
