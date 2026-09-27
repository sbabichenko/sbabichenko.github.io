import numpy as np, noisestate as ns
def cost_eigen(res, agent, n):
    """(The finite engine's Kernel.at takes (time, shock time), not the shock's age.)
    Eigenvalues (and vectors) of a player's realised total cost C = g'Mg over [0, T], g the standardized shock
    increments on n time cells per channel: C is a sum of independent weighted chi-squares."""
    c = res.compiled; T = res.model.horizon.T; dt = T / n; nW = c.nW
    tk = (np.arange(n) + 0.5) * dt
    atoms, Q, _ = c.loss[agent]
    A = {}
    for nm in {a for a, _ in atoms}:
        Kf = res.kernel(nm); M = np.zeros((n, n * nW))
        for k in range(n):
            js = np.arange(k + 1)
            vals = np.atleast_2d(Kf.at(np.full(k + 1, tk[k]), tk[js]))                  # (k+1, nW): at(t, shock time)
            M[k, (js[:, None] * nW + np.arange(nW)[None, :]).ravel()] = (vals * np.sqrt(dt)).ravel()
        A[nm] = M
    Mc = np.zeros((n * nW, n * nW))
    for i, (a1, _) in enumerate(atoms):
        for j, (a2, _) in enumerate(atoms):
            if Q[i, j]:
                Mc += 0.5 * Q[i, j] * dt * (A[a1].T @ A[a2])
    Mc = 0.5 * (Mc + Mc.T)
    lam, V = np.linalg.eigh(Mc)
    return lam, V, A, tk
