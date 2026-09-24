#!/bin/bash
# The triangular engine (triangular-decision-mesh/core, unmodified) for the browser: its sources,
# the portable dense algebra in place of numeric.cpp, and one-thread stand-ins for OpenMP and the
# OpenBLAS thread call. The engine's main() is renamed and called through entry.cpp's dm_run(seed, env);
# static/mesh/fit-worker.js writes the design to the virtual filesystem and reads the dumps back.
# Usage: EIGEN_DIR=/path/holding/Eigen tools/trimesh/build_wasm.sh /path/to/triangular-decision-mesh/core
set -e
SRC=${1:?path to triangular-decision-mesh/core}
OUT=${2:-$(dirname "$0")/../../static/mesh}
NAME=${3:-trimesh}
mkdir -p $OUT
TMP=$(mktemp -d)
COMMON="-std=c++17 -O3 -ffp-contract=off -fwasm-exceptions -msimd128 -msse2 -Wno-unknown-pragmas -include $(dirname "$0")/compat/tochars_shim.h -I$SRC -I$(dirname "$0")/compat -I${EIGEN_DIR:?directory holding Eigen/ and wasm_compat/}"
em++ $COMMON -Dmain=dm_engine_main -c $SRC/main.cpp -o $TMP/main.o
FILES=$(cd $SRC && ls estimator/*.cpp fit/*.cpp mesh/*.cpp | grep -v 'numeric.cpp')
em++ -std=c++17 -O3 -ffp-contract=off -fwasm-exceptions -msimd128 -msse2 -Wno-unknown-pragmas -include $(dirname "$0")/compat/tochars_shim.h \
  -I$SRC -I$(dirname "$0")/compat -I${EIGEN_DIR:?directory holding Eigen/ and wasm_compat/} \
  $(for f in $FILES; do echo $SRC/$f; done) $(dirname "$0")/numeric_portable.cpp $TMP/main.o $(dirname "$0")/entry.cpp \
  -sMODULARIZE=1 -sEXPORT_NAME=DecisionMeshEngine -sINVOKE_RUN=0 -sEXIT_RUNTIME=0 \
  -sALLOW_MEMORY_GROWTH=1 -sMAXIMUM_MEMORY=2GB -sFORCE_FILESYSTEM=1 \
  -sEXPORTED_FUNCTIONS=_dm_run -sEXPORTED_RUNTIME_METHODS=ccall,FS,ENV -sENVIRONMENT=web,worker,node -sSINGLE_FILE=1 \
  -o $OUT/$NAME.js
ls -la $OUT/$NAME.js
echo done
