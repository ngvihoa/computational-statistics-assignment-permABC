#!/bin/bash
# Compile cgal_hilbert.cpp → cgal_hilbert.so
# Requires: CGAL via brew (brew install cgal), pybind11 (pip install pybind11)
#
# Usage:
#   bash build_cgal.sh              # Python actif dans PATH (auto-détecte archi)
#   bash build_cgal.sh permabc      # pyenv 'permabc'  — arm64, Python 3.10
#   bash build_cgal.sh anaconda     # Anaconda base    — x86_64, Python 3.9

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── Sélection du Python ──────────────────────────────────────────────────
ENV=${1:-"auto"}
if [[ "$ENV" == "permabc" ]]; then
    PYTHON=/Users/antoineluciano/.pyenv/versions/permabc/bin/python
elif [[ "$ENV" == "anaconda" ]]; then
    PYTHON=/Users/antoineluciano/opt/anaconda3/bin/python
else
    PYTHON=$(which python3)
fi

ARCH=$($PYTHON -c 'import platform; print(platform.machine())')
echo "Python         : $PYTHON"
echo "Architecture   : $ARCH"

PYBIND11_INC=$($PYTHON -c "import pybind11; print(pybind11.get_include())")
PYTHON_INC=$($PYTHON -c "import sysconfig; print(sysconfig.get_path('include'))")
PYTHON_EXT=$($PYTHON -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

echo "Output         : cgal_hilbert${PYTHON_EXT}"

# ── Flags selon architecture ─────────────────────────────────────────────
if [[ "$ARCH" == "x86_64" ]]; then
    ARCH_FLAGS="-arch x86_64"
    COMPILE_CMD="arch -x86_64 clang++"
else
    ARCH_FLAGS="-arch arm64"
    COMPILE_CMD="clang++"
fi

$COMPILE_CMD -O3 -std=c++17 -shared -fPIC \
    $ARCH_FLAGS \
    -I"${PYBIND11_INC}" \
    -I"${PYTHON_INC}" \
    -I/opt/homebrew/include \
    -undefined dynamic_lookup \
    "${SCRIPT_DIR}/cgal_hilbert.cpp" \
    -o "${SCRIPT_DIR}/cgal_hilbert${PYTHON_EXT}"

echo "Build OK → ${SCRIPT_DIR}/cgal_hilbert${PYTHON_EXT}"
$PYTHON -c "import sys; sys.path.insert(0,'${SCRIPT_DIR}'); import cgal_hilbert; print('Import test OK')"
