// A single-threaded OpenMP for builds without it (the WebAssembly build): the pragmas are ignored
// and the two queries the engine makes answer as for one thread.
#pragma once
inline int omp_get_thread_num() { return 0; }
inline int omp_get_num_threads() { return 1; }
inline int omp_get_max_threads() { return 1; }
