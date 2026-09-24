// The dense algebra of numeric.cpp without LAPACK or BLAS, for builds that have neither (the
// WebAssembly build for the browser). Same functions, same contracts; the factor that cho_factor
// leaves is private to cho_solve and cho_inverse, so it is stored the way Eigen wants it (the
// lower Cholesky factor, row-major) rather than LAPACK's 'U'. Results agree with the LAPACK
// build to rounding, not bit for bit; the pinned reference is for the native build.
#include "estimator/numeric.h"

#include <algorithm>
#include <cmath>

#include <Eigen/Dense>

namespace estimator {

namespace {
using RowMat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using ColMat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
using MapR = Eigen::Map<RowMat>;
using CMapR = Eigen::Map<const RowMat>;
using CMapC = Eigen::Map<const ColMat>;
using Vec = Eigen::Matrix<double, Eigen::Dynamic, 1>;
using MapV = Eigen::Map<Vec>;
using CMapV = Eigen::Map<const Vec>;
}  // namespace

double norm2(const std::vector<double>& x) { return std::sqrt(ddot(x.size(), x.data(), x.data())); }

double expit(double x) { return 1.0 / (1.0 + std::exp(-x)); }

double max_abs(const std::vector<double>& x) {
    double m = 0.;
    for (double v : x) m = std::max(m, std::fabs(v));
    return m;
}

double ddot(std::size_t n, const double* x, const double* y) {
    double s = 0.;
    for (std::size_t i = 0; i < n; ++i) s += x[i] * y[i];
    return s;
}

void dgemv(std::size_t p, const double* A, const double* x, double* y) {
    const Eigen::Index n = static_cast<Eigen::Index>(p);
    MapV(y, n).noalias() = CMapR(A, n, n) * CMapV(x, n);
}

void dgemv_t(std::size_t p, const double* A, const double* x, double* y) {
    const Eigen::Index n = static_cast<Eigen::Index>(p);
    MapV(y, n).noalias() = CMapR(A, n, n).transpose() * CMapV(x, n);
}

void dgemm_tn(std::size_t p, const double* At, const double* B, double* C) {
    const Eigen::Index n = static_cast<Eigen::Index>(p);
    MapR(C, n, n).noalias() = CMapR(At, n, n).transpose() * CMapR(B, n, n);
}

void dgemm_nt(std::size_t p, const double* A, const double* Bt, double* C) {
    const Eigen::Index n = static_cast<Eigen::Index>(p);
    MapR(C, n, n).noalias() = CMapR(A, n, n) * CMapR(Bt, n, n).transpose();
}

void dgemv_fortran(std::size_t p, const double* At, const double* x, double* y) { dgemv_t(p, At, x, y); }

bool cho_factor(std::size_t p, double* A) {
    // LAPACK's 'U' reads the upper triangle of the row-major matrix; Eigen's LLT reads the lower
    // triangle of what it is given, so it is given the transpose (the same buffer, column-major).
    const Eigen::Index n = static_cast<Eigen::Index>(p);
    Eigen::LLT<ColMat> llt(CMapC(A, n, n));
    if (llt.info() != Eigen::Success) return false;
    MapR(A, n, n) = llt.matrixL();          // lower factor, row-major, zeros above
    return true;
}

void cho_solve(std::size_t p, const double* factor, double* b) {
    const Eigen::Index n = static_cast<Eigen::Index>(p);
    const auto L = CMapR(factor, n, n).triangularView<Eigen::Lower>();
    MapV x(b, n);
    L.solveInPlace(x);
    L.transpose().solveInPlace(x);
}

// The rectangular engine's pair: LAPACK wants the factor transposed once for many solves. Here
// the factor is already in the form cho_solve reads, so the "transpose" is a copy and the solve is
// cho_solve. Declared weak so the triangular engine, whose numeric.h lacks them, links unchanged.
__attribute__((weak)) std::vector<double> transpose_factor(std::size_t p, const double* factor) {
    return std::vector<double>(factor, factor + p * p);
}
__attribute__((weak)) void cho_solve_transposed(std::size_t p, const double* factor_t, double* b) {
    cho_solve(p, factor_t, b);
}

void cho_inverse(std::size_t p, const double* factor, double* inverse) {
    const Eigen::Index n = static_cast<Eigen::Index>(p);
    RowMat X = RowMat::Identity(n, n);
    const auto L = CMapR(factor, n, n).triangularView<Eigen::Lower>();
    L.solveInPlace(X);
    L.transpose().solveInPlace(X);
    MapR(inverse, n, n) = X;
}

bool solve(std::size_t p, const double* A, double* b) {
    // dgesv: LU with partial pivoting; it reports failure on an exactly zero pivot
    const Eigen::Index n = static_cast<Eigen::Index>(p);
    Eigen::PartialPivLU<RowMat> lu(CMapR(A, n, n));
    const auto& U = lu.matrixLU();
    for (Eigen::Index i = 0; i < n; ++i) if (U(i, i) == 0.) return false;
    MapV x(b, n);
    x = lu.solve(Vec(x));
    return true;
}

double min_eigenvalue(std::size_t p, const double* A) {
    // numpy: dsyevd 'L' on the Fortran-ordered array, i.e. the row-major upper triangle; the
    // column-major view of the buffer is the transpose, whose lower triangle is that triangle
    if (p == 0) return 0.;
    const Eigen::Index n = static_cast<Eigen::Index>(p);
    Eigen::SelfAdjointEigenSolver<ColMat> es(CMapC(A, n, n), Eigen::EigenvaluesOnly);
    return es.eigenvalues()(0);
}

}  // namespace estimator

// engine.cpp pins OpenBLAS to one thread; without OpenBLAS there is nothing to pin
extern "C" void openblas_set_num_threads(int) {}
