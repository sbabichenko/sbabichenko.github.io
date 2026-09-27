#!/usr/bin/env python3
"""static/noisestate/risk/noisestate.json for /noisestate/risk/: player 1's noise-state in Chapter 1's tracking game,
risk-neutral and risk-adjusted (Chapter 1's appendix, thm:risk_sensitive_appendix).

On n time cells per shock channel the shocks are g ~ N(0, I) (standardized increments).  At the equilibrium of the
noisestate example ch1_two_player_finite, player 1's signal increments are linear in g, y = H g, and its realised
cost is quadratic, C = g'Mg (cara_eig.cost_eigen; no linear part: the example has no means).  Its noise-state at t is
the conditional mean of the whole shock path given y up to t, m = H'(HH')^-1 y, with covariance C_t = I - H'(HH')^-1 H
(the future coordinates untouched); the risk-adjusted noise-state is m_theta = (I - theta C_t K)^-1 m with K = 2M,
valid while I - theta C_t^1/2 K C_t^1/2 is positive definite.  Check: the state's kernel applied to m reproduces
player 1's own estimate of the state.
    ~/Projects/noisestate/.venv/bin/python tools/risk/compute_risk.py
"""
import json, sys
from pathlib import Path
import numpy as np
import noisestate as ns
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cara_eig import cost_eigen

OUT = Path(__file__).resolve().parents[2] / "static" / "noisestate" / "risk" / "noisestate.json"
N, T_HALF, SEED = 100, 0.5, 11

res = ns.solve(ns.load(ns.example("ch1_two_player_finite"))).require_converged()
c = res.compiled; T = res.model.horizon.T; dt = T / N; nW = c.nW; p1 = dict(res.model.params)["p1"]
lam, V, A, tk = cost_eigen(res, "player1", N)
M = V @ np.diag(lam) @ V.T
K = 2.0 * M
ch = c.channels; i1 = ch.index("w1")
k_now = int(round(T_HALF / dt))                                  # cells observed: those before t
H = np.sqrt(p1 * dt) * A["X"][:k_now].copy()                    # dy1_k / sqrt(dt) = sqrt(p1 dt) X(t_k) + g_{w1,k}
for k in range(k_now):
    H[k, k * nW + i1] += 1.0
G = H @ H.T
kk = k_now - 1
for seed in range(SEED, SEED + 500):                            # a history in which player 1 has seen the state pushed up
    g = np.random.default_rng(seed).standard_normal(N * nW)
    m = H.T @ np.linalg.solve(G, H @ g)                         # the noise-state: E[g | y up to t]
    if 0.45 < A["X"][kk] @ m < 0.8 and 0.4 < A["X"][kk] @ g < 1.0:
        break
y = H @ g
Ct = np.eye(N * nW) - H.T @ np.linalg.solve(G, H)
w, U = np.linalg.eigh(Ct); Ct_half = U @ np.diag(np.sqrt(np.clip(w, 0, None))) @ U.T
theta_star = 1.0 / np.linalg.eigvalsh(Ct_half @ K @ Ct_half).max()

# check: player 1's estimate of the state at t from its own filter (noisestate) against the kernel applied to m
Ke = res.estimate("player1", "X")                              # player 1's own filter, as a kernel on the shocks
js = np.arange(kk + 1)
xhat_filter = float((np.atleast_2d(Ke.at(np.full(kk + 1, tk[kk]), tk[js])) * np.sqrt(dt)).ravel() @
                    g[(js[:, None] * nW + np.arange(nW)[None, :]).ravel()])
print(f"theta*_t = {theta_star:.3f}; E[X | y] at t: from the noise-state {float(A['X'][kk] @ m):+.4f}, from player 1's "
      f"filter {xhat_filter:+.4f} (true X {float(A['X'][kk] @ g):+.4f})")

def paths(vec):
    """Cumulative shock paths per channel, W(u) = sum over cells up to u of the increment sqrt(dt) g."""
    return {chn: np.concatenate([[0.0], np.cumsum(vec[j::nW]) * np.sqrt(dt)]).round(5).tolist() for j, chn in enumerate(ch)}

# the judgment of a deviation: player 1 pushes a little harder against the state now (a spike of -1 in D1 at t, held
# after: the frozen continuation of Chapter 1's first-order condition).  The change in its cost is linear in the shocks,
# dC = h'g: the state moves by the spike's response (noisestate, deviation_response) and the spike itself costs
# 2 r D1(t).  Judged with the belief, h'm (zero at the equilibrium: the plan is optimal); judged cost-weighted, h'm_theta.
r1 = dict(res.model.params)["r1"]
t_now = tk[kk]
resp = res.deviation_response("player1", ["X"], continuation="frozen")
later = np.arange(kk, N)
dX = -resp.over(tk[later], np.full(later.size, t_now))[:, 0]           # the state's response to a push of -1
h = 2.0 * dt * (dX[:, None] * A["X"][later]).sum(axis=0) + 2.0 * r1 * (-1.0) * A["D1"][kk]
thetas = np.linspace(0.0, 0.97 * theta_star, 41)
ms = [np.linalg.solve(np.eye(N * nW) - th * Ct @ K, m) for th in thetas]
push = [float(h @ v) for v in ms]
print(f"a push against the state now changes player 1's cost by {push[0]:+.5f} judged with its belief (0 at the "
      f"equilibrium), {push[20]:+.4f} cost-weighted at theta {thetas[20]:.2f}; scale: |h| |m| = {np.linalg.norm(h) * np.linalg.norm(m):.3f}")
state = lambda vec: np.concatenate([[0.0], A["X"] @ vec]).round(5).tolist()     # the state along a shock path
out = {"T": T, "t": T_HALF, "cells": N, "channels": ch, "theta_star": float(theta_star), "seed": seed,
       "labels": {"w0": "the state's shocks", "w1": "player 1's signal noise", "w2": "player 2's signal noise"},
       "true": paths(g), "true_state": state(g), "thetas": thetas.round(5).tolist(),
       "tilted": [paths(v) for v in ms], "state": [state(v) for v in ms], "push": push}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(out, separators=(",", ":")))
print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB)")
