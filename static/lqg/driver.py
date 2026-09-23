"""Driver for the in-browser noise-state explorer.

Runs under Pyodide in a Web Worker (see worker.js).  Every request builds the model from the
parameters the page sends, solves it with noisestate, and returns plain lists for plotting.
Nothing is precomputed.
"""
import json
import math
import time

import numpy as np
import noisestate as ns
from noisestate.results import Kernel

PRESETS = {}


def preset(name):
    def deco(fn):
        PRESETS[name] = fn
        return fn
    return deco


# ---------------------------------------------------------------------------------------------
# Model builders.  Parameter names and conventions follow the noisestate example files.

@preset("ch1")
def ch1_model(p):
    """Chapter 1: two-player tracking game with targets on [0, T]."""
    return {
        "name": "ch1_tracking_with_targets",
        "params": {k: float(p[k]) for k in ("p1", "p2", "r1", "r2", "b1", "b2", "sigma")},
        "channels": ["w0", "w1", "w2"],
        "states": {"X": {"drift": {"D1": 1.0, "D2": 1.0}, "noise": {"w0": "sigma"}}},
        "agents": {
            "player1": {
                "controls": ["D1"],
                "signals": {"y1": {"drift": {"X": "sqrt(p1)"}, "noise": {"w1": 1.0}}},
                "loss": [[1.0, "X", "X"], ["-2*b1", "X"], ["r1", "D1", "D1"]],
            },
            "player2": {
                "controls": ["D2"],
                "signals": {"y2": {"drift": {"X": "sqrt(p2)"}, "noise": {"w2": 1.0}}},
                "loss": [[1.0, "X", "X"], ["-2*b2", "X"], ["r2", "D2", "D2"]],
            },
        },
        "horizon": {"kind": "finite", "T": float(p["T"])},
        "numerics": {"nodes": int(p.get("nodes", 10))},
    }


@preset("ch3")
def ch3_model(p):
    """Chapter 3: stationary two-player tracking game, average cost."""
    return {
        "name": "ch3_stationary_tracking",
        "params": {k: float(p[k]) for k in ("p1", "p2", "r1", "r2")},
        "channels": ["w0", "w1", "w2"],
        "states": {"X": {"drift": {"D1": 1.0, "D2": 1.0}, "noise": {"w0": 1.0}}},
        "agents": {
            "player1": {
                "controls": ["D1"],
                "signals": {"y1": {"drift": {"X": "sqrt(p1)"}, "noise": {"w1": 1.0}}},
                "loss": [[0.5, "X", "X"], ["0.5*r1", "D1", "D1"]],
            },
            "player2": {
                "controls": ["D2"],
                "signals": {"y2": {"drift": {"X": "sqrt(p2)"}, "noise": {"w2": 1.0}}},
                "loss": [[0.5, "X", "X"], ["0.5*r2", "D2", "D2"]],
            },
        },
        "horizon": {"kind": "stationary", "discount": 0.0, "window": float(p.get("window", 8.0))},
        "numerics": {"nodes": int(p.get("nodes", 32))},
    }


@preset("ch4")
def ch4_model(p):
    """Chapter 4: stationary Kyle-Back market with one informed trader."""
    return {
        "name": "ch4_kyle_back",
        "params": {k: float(p[k]) for k in ("eps", "rho", "gamma1", "sigma_V", "sigma_Z")},
        "channels": ["wV", "wZ", "w1"],
        "states": {"V": {"drift": {}, "noise": {"wV": "sigma_V"}}},
        "agents": {
            "market_maker": {
                "controls": ["P"],
                "myopic": True,
                "signals": {"flow": {"drift": {"D1": 1.0}, "noise": {"wZ": "sigma_Z"}}},
                "loss": [[1.0, "P", "P"], [-2.0, "P", "V"]],
            },
            "trader1": {
                "controls": ["D1"],
                "signals": {
                    "y1": {"drift": {"V": "gamma1", "P": "-gamma1"}, "noise": {"w1": 1.0}},
                    "flow": {"drift": {}, "noise": {"wZ": "sigma_Z"}},
                },
                "loss": [[-1.0, "D1", "V"], [1.0, "D1", "P"], ["eps", "D1", "D1"]],
            },
        },
        "horizon": {"kind": "stationary", "discount": "rho", "window": float(p.get("window", 8.0))},
        "numerics": {"nodes": int(p.get("nodes", 16))},
    }


# ---------------------------------------------------------------------------------------------
# Result extraction.

def _clean(a):
    """ndarray -> list with NaN/inf replaced by None (JSON has no NaN)."""
    out = []
    for v in np.asarray(a, dtype=float).ravel():
        out.append(float(v) if math.isfinite(v) else None)
    return out


def _diagnostics(res):
    rows = []
    for r in res.diagnostics.rows:
        rows.append({
            "name": r.get("name"),
            "ok": bool(r.get("ok")),
            "severity": r.get("severity"),
            "value": r.get("value") if isinstance(r.get("value"), (int, float)) else None,
            "threshold": r.get("threshold") if isinstance(r.get("threshold"), (int, float)) else None,
            "flag": r.get("flag") if not r.get("ok") else "",
            "meaning": r.get("meaning", ""),
        })
    return rows


def _finite_payload(res, model, T):
    names = list(model.state_names) + list(model.control_names)
    chans = list(model.channels)
    dates = [T * f for f in (0.25, 0.5, 0.75, 1.0)]
    kernels = {}
    for nm in names:
        kernels[nm] = {}
        for ch in chans:
            k = res.kernel(nm, ch)
            curves = []
            for t in dates:
                s = np.linspace(0.0, t, 61)
                curves.append({"t": t, "s": _clean(s), "v": _clean(k.at(np.full_like(s, t), s))})
            kernels[nm][ch] = curves
    tt = np.linspace(0.0, T, 121)
    means = {nm: _clean(res.mean(nm, tt)) for nm in names} if res.has_means else {}
    foc = {}
    t_mid = 0.5 * T
    s = np.linspace(0.0, t_mid, 61)
    for agent, ctrls in res.foc.items():
        for ctrl, parts in ctrls.items():
            foc[ctrl] = {"agent": agent, "t": t_mid, "x": _clean(s), "channels": {}}
            for j, ch in enumerate(chans):
                foc[ctrl]["channels"][ch] = {
                    part: _clean(Kernel.of(np.asarray(parts[part])[:, j], res, part, ch).at(np.full_like(s, t_mid), s))
                    for part in ("physical", "wedge")
                }
    return {"kind": "finite", "dates": dates, "kernels": kernels, "mean_t": _clean(tt), "means": means, "foc": foc}


def _stationary_payload(res, model, L):
    names = list(model.state_names) + list(model.control_names)
    chans = list(model.channels)
    age = np.linspace(0.0, L, 161)
    kernels = {nm: {ch: _clean(res.kernel(nm, ch).at(age)) for ch in chans} for nm in names}
    foc = {}
    for agent, ctrls in res.foc.items():
        for ctrl, parts in ctrls.items():
            foc[ctrl] = {"agent": agent, "x": _clean(age), "channels": {}}
            for j, ch in enumerate(chans):
                foc[ctrl]["channels"][ch] = {
                    part: _clean(Kernel.of(np.asarray(parts[part])[:, j], res, part, ch).at(age))
                    for part in ("physical", "wedge")
                }
    return {"kind": "stationary", "age": _clean(age), "kernels": kernels, "foc": foc}


def solve(preset_name, params_json):
    """Entry point called from the worker. Returns a JSON string."""
    t0 = time.time()
    p = json.loads(params_json)
    model = ns.Model.from_dict(PRESETS[preset_name](p))
    res = ns.solve(model)
    h = model.to_dict(numeric=True)["horizon"]
    if h["kind"] == "finite":
        body = _finite_payload(res, model, float(h["T"]))
    else:
        body = _stationary_payload(res, model, float(h["window"]))
    body.update({
        "preset": preset_name,
        "params": p,
        "costs": {k: float(v) for k, v in res.costs.items()},
        "cost_parts": {a: {k: float(v) for k, v in d.items()} for a, d in (res.cost_parts or {}).items()},
        "cost_kind": str(res.cost_kind),
        "converged": bool(res.converged),
        "residual": float(res.residual),
        "evaluations": int(res.evaluations),
        "solve_seconds": float(res.seconds),
        "total_seconds": time.time() - t0,
        "diagnostics": _diagnostics(res),
        "flags": list(res.diagnostics.flags),
        "describe": str(model.describe()),
        "channels": list(model.channels),
        "names": list(model.state_names) + list(model.control_names),
        "version": getattr(ns, "__version__", ""),
    })
    return json.dumps(body)
