#!/usr/bin/env python3
"""static/noisestate/risk/equilibria.json for /noisestate/risk/: Chapter 1's tracking game with both players risk
averse (the same theta), solved by noisestate's finite engine (risk_aversion on each agent; the engine climbs in theta
by itself when it has to).  Checked in the package against a brute-force discrete-time reference
(extras/leqg_reference.py, extrapolated in the step size) to about 1e-7 on the costs.  Continued up theta in steps of
0.1 until the engine reports the breakdown.  Needs the development version (branch `cara`):
    ~/Projects/noisestate-cara/.venv/bin/python tools/risk/compute_equilibria.py
"""
import json, re
from pathlib import Path
import numpy as np
import noisestate as ns

OUT = Path(__file__).resolve().parents[2] / "static" / "noisestate" / "risk" / "equilibria.json"
EX = Path.home() / "Projects" / "noisestate-cara" / "examples" / "ch1_two_player_finite.yaml"
n, s0 = 100, 0.2
tk = np.arange(n) / n                                   # the page draws cell k at time k / n
base = ns.load(EX).to_dict()
rows, breakdown = [], None
for th in np.round(np.arange(0.0, 3.51, 0.1), 3):
    m = json.loads(json.dumps(base))
    for a in m["agents"].values():
        a["risk_aversion"] = float(th)
    try:
        res = ns.solve(m)
    except ns.RiskBreakdown as e:
        print(f"theta {th}: breakdown ({e})")
        breakdown = float(re.search(r"breakdown \(E exp\(theta C\) infinite\) at risk_aversion ([0-9.]+)", str(e)).group(1)); break
    after = tk > s0                                    # the state shock at s0: its life from then on
    X = np.zeros(n); D1 = np.zeros(n)
    X[after] = np.asarray(res.kernel("X", "w0").at(tk[after], np.full(after.sum(), s0))).ravel()
    D1[after] = np.asarray(res.kernel("D1", "w0").at(tk[after], np.full(after.sum(), s0))).ravel()
    r = res.risk.get("player1") or {"entropic": res.costs["player1"], "expected": res.costs["player1"], "theta_lambda_max": 0.0}   # theta 0: no risk entry
    rows.append({"theta": float(th), "entropic": r["entropic"], "expected": r["expected"],
                 "X": X.round(5).tolist(), "D1": D1.round(5).tolist()})
    print(f"theta {th}: entropic {r['entropic']:.6f} expected {r['expected']:.6f} theta*lambda_max {r['theta_lambda_max']:.3f}", flush=True)
OUT.write_text(json.dumps({"n": n, "T": 1.0, "shock_at": s0, "breakdown": breakdown, "noisestate": ns.__version__ + " (cara)", "rows": rows},
                          separators=(",", ":")))
print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB, {len(rows)} values of theta)")
