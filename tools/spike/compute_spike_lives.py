#!/usr/bin/env python3
"""The life of a spike, chapter by chapter: static/dissertation/spike/lives.json for the /dissertation/spike/ page.

One player makes a single instantaneous deviation (a unit spike in its control) and every other quantity responds;
the chapters differ only in what happens next.  Each life is noisestate's deviation response (res.deviation_response):
a frozen spike (the deviator does not follow it up), the others reacting as the chapter's model says.  Precomputed:
the page reads the JSON and never solves.  Run with the noisestate environment:
    ~/Projects/noisestate/.venv/bin/python tools/spike/compute_spike_lives.py
"""
import json, sys, time
from pathlib import Path
import numpy as np
import noisestate as ns

NS = Path(ns.__file__).resolve().parents[1]
sys.path[:0] = [str(NS / "examples"), str(NS / "tests")]
OUT = Path(__file__).resolve().parents[2] / "static" / "dissertation" / "spike" / "lives.json"
lives = []
CACHE = Path(__file__).resolve().parent / ".cache"
CACHE.mkdir(exist_ok=True)


def cached(key, fn):
    """One life per chapter, cached so a rerun only recomputes what was deleted from tools/spike/.cache."""
    f = CACHE / f"{key}.json"
    if f.exists():
        lives.append(json.loads(f.read_text()))
        return
    t0 = time.time()
    fn()
    f.write_text(json.dumps(lives[-1]))
    print(f"{key}: {time.time() - t0:.0f} s", flush=True)


DU = 0.01          # the page's time step, in time since the spike
SHOW = 3.0         # the page shows the first three time units of every life
STARTS = 24        # spike times per finite game: its life depends on how much of the game is left


def curves_at(resp, u, s=None):
    v = resp.over(u) if s is None else resp.over(s + u, np.full_like(u, s))
    return [v[:, i].round(4).tolist() for i in range(v.shape[1])]


# milestones in a life, read off its own curves; the page lights each one as the spike's age passes it
def peak(label, j):
    """Where curve j is largest in size, if that is inside the life (not at its start or its cut-off end)."""
    def f(u, ys):
        k = int(np.argmax(np.abs(ys[j])))
        return [{"at": round(float(u[k]), 3), "label": label}] if 0 < k < len(u) - 3 else []
    return f


def first_below(label, j, level):
    def f(u, ys):
        k = np.flatnonzero(np.asarray(ys[j]) <= level)
        return [{"at": round(float(u[k[0]]), 3), "label": label}] if len(k) else []
    return f


def halved(label, j):
    """Where curve j has lost half of its starting size."""
    def f(u, ys):
        y = np.abs(np.asarray(ys[j]))
        k = np.flatnonzero(y <= 0.5 * y[0])
        return [{"at": round(float(u[k[0]]), 3), "label": label}] if len(k) and y[0] > 0 else []
    return f


def crossing(label, j, i):
    """Where curve j first overtakes curve i."""
    def f(u, ys):
        k = np.flatnonzero(np.asarray(ys[j]) >= np.asarray(ys[i]))
        return [{"at": round(float(u[k[0]]), 3), "label": label}] if len(k) else []
    return f


def at(label, age):
    return lambda u, ys: [{"at": age, "label": label}]


def events(fns, u, ys):
    return sorted((m for f in fns for m in f(u, ys)), key=lambda m: m["at"])


def finite(key, title, story, model, origin, curves, marks=(), milestones=(), reference=None):
    """A finite game: the life of a spike depends on when it happens, so store one per spike time s. reference:
    (model, quantity, label), a curve from a second model on the same horizon, drawn as a grey reference."""
    res = ns.solve(model).require_converged()
    T = res.model.horizon.T
    resp = res.deviation_response(origin, [q for q, _ in curves])
    ref = ns.solve(reference[0]).require_converged().deviation_response(origin, [reference[1]]) if reference else None
    spikes = []
    for s in np.linspace(0.02 * T, 0.94 * T, STARTS):
        u = np.arange(0.0, min(SHOW, T - s) + 1e-9, DU)
        ys = curves_at(resp, u, s)
        ms = events(milestones, u, ys)
        if ref is not None:
            ys = ys + curves_at(ref, u, s)
        spikes.append({"s": round(float(s), 4), "y": ys, "marks": ms})
    labels = [lab for _, lab in curves] + ([reference[2]] if reference else [])
    lives.append({"key": key, "title": title, "story": story, "kind": "finite", "T": T, "du": DU, "marks": list(marks),
                  "labels": labels, "spikes": spikes})


def stationary(key, title, story, curves, control=None, milestones=()):
    """A stationary game: one life, the same whenever the spike happens. curves: (result, deviator, quantity, label).
    Past a model's window its kernels are zero by construction, not by physics, so the life ends there."""
    end = min([SHOW] + [r.model.horizon.window for r, *_ in curves])
    u = np.arange(0.0, end + 1e-9, DU)
    ys = [curves_at(r.deviation_response(o, [q], control=control), u)[0] for r, o, q, _ in curves]
    lives.append({"key": key, "title": title, "story": story, "kind": "stationary", "du": DU, "marks": events(milestones, u, ys),
                  "labels": [lab for *_, lab in curves], "y": ys})


def both(res, origin, curves):
    return [(res, origin, q, lab) for q, lab in curves]


def split(res, origin, control, quantities, off, u):
    """Permanent and temporary impact, as Chapter 4's Deviation impact section splits them.  The total: the deviation
    response under the blip convention (the origin, knowing its spike, then trades on through its own response to it).
    The permanent part: the market maker's own update to that same order path, the spike and the origin's continuation,
    with the `off` agents' strategies switched off (their trading against it is the temporary part).  Uses the result's
    closed-loop assembly (res.compiled, res.maps) for the market maker's response to a unit order."""
    c = res.compiled
    ctrls = list(next(a for a in res.model.agents if a.name == origin).controls)
    I = c.grid.interp(u)
    dev = res.deviation_response(origin, list(quantities) + ctrls, control=control).over(u)
    total, cont = dev[:, :len(quantities)], dev[:, len(quantities):]          # cont: the origin's continuation
    quiet = {k: (np.zeros_like(v) if k in off else v) for k, v in res.maps.items()}
    Z = c.closed_loop(quiet, excluded=origin, impulse_controls=ctrls)[:, c.nW:]
    unit = {o: np.stack([I @ Z[c.block(q), k] for q in quantities], axis=1) for k, o in enumerate(ctrls)}   # (len u, nq)
    du = u[1] - u[0]
    perm = unit[control].copy()
    for k, o in enumerate(ctrls):                                           # + the unit response convolved with the
        for m in range(1, len(u)):                                          #   continuation, trapezoid in the seed's age
            w = np.full(m + 1, du); w[0] = w[-1] = du / 2
            perm[m] += (w[:, None] * unit[o][m::-1] * cont[:m + 1, k][:, None]).sum(axis=0)
    return [total[:, i].round(4).tolist() for i in range(len(quantities))], [perm[:, i].round(4).tolist() for i in range(len(quantities))]


def longer(path, window, nodes):
    """The example on a longer window (the shipped Ch3 example stops at 3, where the response is still 0.2)."""
    d = ns.load(path).to_dict()
    d["horizon"]["window"] = window
    d.setdefault("numerics", {})["nodes"] = nodes
    return ns.Model.from_dict(d)


# Chapter 1: a finite horizon, so the spike's aftermath is cut off at T
cached("ch1", lambda: finite("ch1", "A finite horizon", "Player 1 kicks the state, and since it knows it did, it starts pulling the "
           "state back at once. Player 2 sees the spike only through its noisy signal and leans against it too. Near the "
           "end of the game there is little left to gain, so a late spike is left to stand.",
           ns.load(ns.example("ch1_two_player_finite")), "player1", [("X", "the state"), ("D2", "player 2's response")],
           milestones=[peak("2 pushes back hardest", 1)]))

# Chapter 2: a public signal seen with different delays (tools/spike/models/ch2_delayed_public.yaml). Player 2 sees
# the public signal late; grey, its response if it saw the public signal at once
M2 = str(Path(__file__).resolve().parent / "models" / "ch2_delayed_public.yaml")
def ch2():
    lag = ns.load(M2).to_dict()["params"]["Delta"]
    on_time = ns.load(M2).to_dict(); on_time["params"]["Delta"] = 0.0
    finite("delay", "A late public signal", "Both players watch a public signal as well as their own. Player 1 kicked the state and "
           "pulls it back itself. Player 2 sees the public signal late, so until then it pushes back on its private "
           "signal alone, less than it would with the public news on time (gray).",
           ns.load(M2), "player1", [("X", "the state"), ("D2", "player 2's response")],
           marks=[{"at": lag, "label": "2 sees it in public"}], milestones=[peak("2 pushes back hardest", 1)],
           reference=(ns.Model.from_dict(on_time), "D2", "player 2's response, public news on time"))
cached("delay", ch2)

# Chapter 3: no clock; the life is a fixed shape in the spike's age
cached("ch3", lambda: stationary("ch3", "No end", "The stationary game has no deadline: a spike's life is the same shape whenever it "
               "happens. Player 1 undoes most of its own spike, player 2 helps, and the state returns all the way home.",
               both(ns.solve(longer(ns.example("ch3_two_player"), 6.0, 48)), "player1", [("X", "the state"), ("D2", "player 2's response")]),
               milestones=[peak("2 pushes back hardest", 1), first_below("half the spike is gone", 0, 0.5)]))

# Chapter 4: a Kyle-Back market with two informed traders (tests/test_ch4.py's two-trader model, checked there against
# the dissertation's grid solver). Dashed: the permanent part of the price impact, the market maker's own update
# with trader 2 switched off; the gap to the price is the temporary part, trader 2's trading against the spike.
from test_ch4 import two_trader
U = np.arange(0.0, SHOW + 1e-9, DU)


def ch4():
    two = ns.solve(ns.Model.from_dict(two_trader(ns.load(ns.example("ch4_kyle_back")).to_dict()))).require_converged()
    (P, D2), (Pp, _) = split(two, "trader1", "D1", ["P", "D2"], {"trader2"}, U)
    ys = [P, D2]
    lives.append({"key": "ch4", "title": "Two insiders", "kind": "stationary", "du": DU, "labels": ["the price", "trader 2's order"],
                  "story": "The spike is an extra order from trader 1, who then sells part of it back. The market "
                           "maker cannot tell any of it from informed trading, and its own update to trader 1's orders is the "
                           "dashed line. Trader 2's signal says the asset is now overpriced, so it sells too, and between them "
                           "the price comes all the way back: the impact is temporary.",
                  "y": ys, "perm": [Pp, None],
                  "marks": events([at("trader 2 sells at once", 0.0), first_below("half the jump is undone", 0, 0.5)], U, ys)})
cached("ch4", ch4)


# Chapter 4's two-asset market, specialist version (tools/spike/models/kb_two_assets.yaml). Two kinds of spike: trader 1's
# order in the stock it knows, and its order in the stock trader 2 knows
def ch4b():
    d = ns.load(str(Path(__file__).resolve().parent / "models" / "kb_two_assets.yaml")).to_dict()
    d["horizon"]["window"] = 16.0
    d.setdefault("numerics", {})["nodes"] = 64
    res = ns.solve(ns.Model.from_dict(d)).require_converged()
    variants = []
    for ctrl, name, fns in (("D11", "in stock 1", [at("the spike spills into stock 2", 0.0), halved("half the spillover is undone", 1)]),
                            ("D12", "in stock 2", [at("trader 2 sells at once", 0.0), halved("half of stock 2's jump is undone", 1),
                                                  halved("half of stock 1's spillover is undone", 0)])):
        ys, perm = split(res, "trader1", ctrl, ["P1", "P2"], {"trader2"}, U)
        variants.append({"name": name, "y": ys, "perm": perm, "marks": events(fns, U, ys)})
    lives.append({"key": "ch4b", "title": "Two stocks", "kind": "stationary", "du": DU, "labels": ["stock 1's price", "stock 2's price"],
                  "story": "Two stocks whose values move together; trader 1 knows stock 1, trader 2 knows stock 2. The same "
                           "extra order from trader 1 has two lives. In stock 1, which nobody else knows, only trader 1 takes it "
                           "back, slowly (dashed: the market maker's update to trader 1's orders), and trader 2 trades away the "
                           "spillover into stock 2. In stock 2, trader 2 trades against it at once, in both stocks, and the "
                           "impact is gone much sooner than trader 1 alone would take it.",
                  "variants": variants, "y": variants[0]["y"], "perm": variants[0]["perm"], "marks": variants[0]["marks"]})
cached("ch4b", ch4b)

# Chapter 5: a cycle of firms; the spike travels around it one lag at a time
from make_ch5_cycle_market import build
cached("ch5", lambda: stationary("ch5", "A network", "Firm 0 places an unusual order with its supplier, firm 2, who sees the order "
               "book almost exactly and moves its price at once. Firm 1, the next firm around the cycle, sees firm 0's orders "
               "only through a much noisier signal, and its price moves later and less: the spike travels around the "
               "network, fading as it goes.",
               both(ns.solve(build(N=3)), "firm0", [("P2", "firm 2's price"), ("P1", "firm 1's price")]), control="o0",
               milestones=[at("firm 2 moves at once", 0.0), peak("firm 1 feels it most", 1)]))

# Chapter 6: who is watching. The same market maker's spike, seen by a privy trader (the transparent market: the trader
# sees the order flow and monitors the market maker; window 10 x 30, the inventory at age 3 within 0.3% of window 16's)
# and by a naive one (the opaque market: it sees the quote, filtered as a level row, and not the flow; continued from
# the competitive market up the inventory weight, window 12 x 36, within 0.2% of window 14 x 42).  Under the blip
# convention the market maker, which knows its spike, then manages the inventory it left, in both.
from test_monitoring import market
def ch6():
    d = market(0.1, nodes=30).to_dict()
    d["horizon"]["window"] = 10.0
    privy = ns.solve(ns.Model.from_dict(d)).require_converged()
    naive = None
    for g in np.round(np.arange(0.0, 0.1001, 0.01), 3):
        o = market(float(g), transparent=False, nodes=36).to_dict()
        o["agents"]["trader"]["signals"].pop("flow"); o["agents"]["trader"]["signals"]["quote"] = {"level": "P"}
        o["horizon"]["window"] = 12.0
        naive = ns.solve(ns.Model.from_dict(o), **({"start_from": naive.maps} if naive is not None else {})).require_converged()
    # two traders who both see only the quote, one privy to the market maker and one naive (on the path they are the
    # same trader), continued up the inventory weight; eps 0.2 so a privy block is again 2.5
    def two(g):
        tr = lambda j, privy: {"controls": f"D{j}", **({"monitors": "mm"} if privy else {}),
                               "observes": {f"y{j}": f"(V - P) dt + dwY{j}", "quote": {"level": "P", "filter": True}},
                               "loss": f"-V D{j} + P D{j} + eps D{j}^2"}
        return ns.Model.from_dict({
            "name": "two", "params": {"eps": 0.2, "gamma": g, "rho": 0.5, "sigma_Z": 1.0},
            "shocks": ["wV", "wZ", "wYA", "wYB"], "states": {"V": "dwV", "Q": "-DA dt - DB dt - sigma_Z dwZ"},
            "agents": {"mm": {"controls": "P", "observes": {"flow": "DA dt + DB dt + sigma_Z dwZ"},
                              "loss": "V DA + V DB - P DA - P DB + gamma Q^2"},
                       "A": tr("A", True), "B": tr("B", False)},
            "horizon": {"window": 10.0, "discount": "rho"}, "numerics": {"nodes": 30}})
    mixed = None
    for g in np.round(np.arange(0.0, 0.1001, 0.02), 3):
        mixed = ns.solve(two(float(g)), **({"start_from": mixed.maps} if mixed is not None else {})).require_converged()
    stationary("ch6", "Who is watching",
               "The market maker posts a quote that is too high for its plan. A trader privy to it knows the quote is a "
               "deviation, not news, and sells a block of 2.5 into it at once. A naive trader, who sees the quote but not "
               "the order flow, reads it as noise traders buying and sells less than a third as much. Either way the market "
               "maker, knowing what it did, then quotes low to work off the inventory. Facing one of each, it takes both "
               "blocks at once, the privy trader's whole and the naive one's partial.",
               [(privy, "mm", "Q", "market maker's inventory, facing a privy trader"), (naive, "mm", "Q", "market maker's inventory, facing a naive trader"),
                (mixed, "mm", "Q", "facing both, one privy and one naive")],
               milestones=[at("the trader sells at once", 0.0)])
cached("ch6", ch6)

MODELS = {
    "ch1": "Two players steer one state X, which moves with the sum of their actions plus noise. Each sees X only "
           "through its own noisy signal, and each pays for X being away from zero and for its own effort. The game "
           "ends at T = 1.",
    "delay": "The same two-player game, plus a public signal on X that both players watch: player 1 sees it at once, "
             "player 2 a quarter of a time unit late. Each player's own signal arrives at once. The game ends at T = 1.",
    "ch3": "The same game with no end date: the players pay per unit of time, forever. Player 2's own signal is more "
           "precise than player 1's.",
    "ch4": "A Kyle-Back market: one asset of unknown value, two informed traders with their own noisy signals on it, "
           "noise traders, and a competitive market maker who sets the price to the expected value given the total "
           "order flow. The traders pay a cost for trading fast and discount the future.",
    "ch4b": "Two stocks whose values are correlated (0.6); the noise traders in the two stocks are independent. "
            "Trader 1's signal is about stock 1 only, trader 2's about stock 2 only; both trade both stocks and see "
            "both stocks' order flow. One competitive market maker prices both.",
    "ch5": "Three firms in a cycle. Each buys its input from the firm before it, with delivery taking half a time "
           "unit, and sells to households. Each sets a price and an order, and sees its sales, its supplier's price, "
           "its customer's orders and, noisily, its supplier's other orders.",
    "ch6": "A market maker who sets its quote strategically and pays for holding inventory, and one informed trader "
           "with its own signal on the asset's value. The privy trader also sees the order flow, so a quote off the "
           "market maker's plan is recognised as one. The naive trader sees the quote but not the flow, and treats the "
           "quote as information about the flow.",
}
for life in lives:
    life["model"] = MODELS[life["key"]]

# "When someone is watching": the privy trader of Chapter 6's transparent market after the market maker's quote spike,
# expecting the market maker to play on (blip) or to hold still (frozen): its order rate D and the inventory Q, both
# from deviation_response on the same equilibrium.  The block itself (2.5) is the jump in Q at age 0.
def watch():
    d = market(0.1, nodes=30).to_dict(); d["horizon"]["window"] = 10.0
    r = ns.solve(ns.Model.from_dict(d)).require_converged()
    ages = np.linspace(0.0, SHOW, 121)
    out = {"ages": ages.round(4).tolist()}
    for c in ("blip", "frozen"):
        v = r.deviation_response("mm", ["D", "Q"], continuation=c).over(ages)
        out[c] = {"D": v[:, 0].round(5).tolist(), "Q": v[:, 1].round(5).tolist()}
    return out
_wf = CACHE / "watch.json"
if not _wf.exists():
    _wf.write_text(json.dumps(watch()))
WATCH = json.loads(_wf.read_text())

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps({"noisestate": ns.__version__, "show": SHOW, "lives": lives, "watch": WATCH}, separators=(",", ":")))
print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB, {len(lives)} lives)")


# ---------------------------------------------------------------- two continuations, one equilibrium (Chapter 6's lemma)
# The Chapter 3 game of the rows above.  (1) The life of player 1's kick under the two continuations: frozen (player 1
# holds its control after the kick) and blip (it plays on, knowing it kicked: its own response D^{1<-1}).  (2) Player 1's
# expected cost along a change of its strategy (a map on its signal, something it can do), followed either way, and
# counted as if nobody reacted: J(e) - J(0) = slope e + curvature e^2, from noisestate's expected_cost.
def lemma():
    d = ns.load(ns.example("ch3_two_player")).to_dict()
    d["horizon"]["window"] = 6.0; d.setdefault("numerics", {})["nodes"] = 48
    res = ns.solve(ns.Model.from_dict(d)).require_converged()
    S, c, agent = res._solver(), res.compiled, res.model.agents[0]
    N, own = c.N, c.block("D1")
    lives_ = {k: res.deviation_response("player1", ["X", "D1", "D2"], continuation=k).over(U) for k in ("frozen", "blip")}
    Zp, R0 = S._spikes(c, res.maps, agent)
    ytil, yinst = S._passive_rows(agent, S._passive_world(agent, res.maps, Zp, R0))
    Gk = S._row_operator(agent, ytil, yinst)
    a = c.grid.nodes
    dc = np.stack([Gk[k] @ np.tile(np.exp(-0.5 * a), len(agent.signals)) for k in range(c.nW)], axis=1)
    def quad(W):
        delta = np.zeros_like(res.world)
        for p in range(len(c.prim)):
            blk = slice(p * N, (p + 1) * N)
            delta[blk] = c.grid.conv_op(W[blk]) @ dc
        delta[own] += dc
        J = lambda e: S.expected_cost(agent, res.world + e * delta)
        h = 1e-2
        return (J(h) - J(-h)) / (2 * h), (J(h) + J(-h) - 2 * J(0.0)) / (2 * h * h)
    W = {"frozen": res._seed_world("player1", 0, "frozen"), "blip": res._seed_world("player1", 0, "blip")}
    bowls = {k: quad(v) for k, v in W.items()}
    bowls["alone"] = quad(np.zeros_like(W["blip"]))
    out = {"du": DU, "u": None,
           "lives": {k: {"X": v[:, 0].round(4).tolist(), "D1": v[:, 1].round(4).tolist(), "D2": v[:, 2].round(4).tolist()}
                     for k, v in lives_.items()},
           "bowls": {k: {"slope": float(s_), "curvature": float(q_)} for k, (s_, q_) in bowls.items()}}
    print("lemma bowls:", {k: (f"{s_:+.1e}", f"{q_:.4f}") for k, (s_, q_) in bowls.items()}, flush=True)
    (OUT.parent / "lemma.json").write_text(json.dumps(out, separators=(",", ":")))
    print(f"wrote {OUT.parent / 'lemma.json'}")

lemma()
