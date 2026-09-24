# The decision-mesh engines in the browser

`build_wasm.sh` compiles either engine's `core/` to one self-contained script (the wasm is inlined):

    EIGEN_DIR=/path/holding/Eigen tools/trimesh/build_wasm.sh ../triangular-decision-mesh/core static/mesh trimesh
    EIGEN_DIR=/path/holding/Eigen tools/trimesh/build_wasm.sh ../rectangular-decision-mesh/core static/mesh rectmesh

Needs Emscripten (3.1 works) and Eigen 3.4. The engines' sources are used unchanged; what is added:

- `numeric_portable.cpp`: the functions of `estimator/numeric.cpp` on Eigen instead of LAPACKE/OpenBLAS
  (the rectangular engine's `transpose_factor` / `cho_solve_transposed` included).
- `compat/`: one-thread `omp.h`, a `cblas.h` declaring the OpenBLAS thread call, an `mmintrin.h` stub
  for Eigen under wasm, and `tochars_shim.h` for the floating-point `std::to_chars` that Emscripten's
  libc++ lacks (general format at a precision, which is printf `%.*g`).
- `entry.cpp`: `dm_run(seed, env)`, which sets the environment (`DMESH_DATA`, `DMESH_DUMP`, ...) and calls
  the engine's own `main`, renamed at compile time.

`static/mesh/fit-worker.js` writes the design to the virtual filesystem, calls `dm_run`, and reads the
dumps back. Checked against the native builds: SMM 202606 and Ginnie (triangular) and SMM 202606
(rectangular) give the same mesh to 1e-11 and the same held-out deviance.
