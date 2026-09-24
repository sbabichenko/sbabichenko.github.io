// Only what the engine uses from cblas.h outside numeric.cpp, for builds without OpenBLAS.
#pragma once
extern "C" void openblas_set_num_threads(int);
