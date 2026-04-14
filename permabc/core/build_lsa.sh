#!/usr/bin/env bash
# Compile rectangular_lsap_custom.cpp → librectangular_lsap_custom.so
#
# Usage:
#   bash build_lsa.sh          # auto-detect architecture
#   bash build_lsa.sh x86_64   # force x86_64 (Anaconda Python 3.9)
#   bash build_lsa.sh arm64    # force arm64  (pyenv permabc Python 3.10)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ARCH=${1:-"auto"}

if [[ "$ARCH" == "auto" ]]; then
    ARCH=$(uname -m)
fi

echo "Architecture : $ARCH"

if [[ "$ARCH" == "x86_64" ]]; then
    CXX="arch -x86_64 g++"
else
    CXX="g++"
fi

$CXX -O3 -std=c++17 -fPIC -shared \
    -arch "$ARCH" \
    "${SCRIPT_DIR}/rectangular_lsap_custom.cpp" \
    -o "${SCRIPT_DIR}/librectangular_lsap_custom.so"

echo "Built → ${SCRIPT_DIR}/librectangular_lsap_custom.so"

# Quick sanity check
python3 -c "
import sys, numpy as np
sys.path.insert(0, '${SCRIPT_DIR}')
from lsa_ctypes import linear_sum_assignment

C = np.array([[4.,1.,3.],[2.,0.,5.],[3.,2.,2.]])
r, c, info = linear_sum_assignment(C, return_info=True)
assert abs(C[r, c].sum() - 5.0) < 1e-9, f'Wrong cost {C[r,c].sum()}'
print('Sanity check OK  cost=', C[r,c].sum(), '  assignment=', list(c), '  info=', info)
"
