// /noisestate/risk: what CARA (entropic) risk aversion does, in two places where it is exact.
// Section 1: the risk-adjusted noise-state of Chapter 1's appendix (thm:risk_sensitive_appendix), W^theta =
// (I - theta C K)^-1 W: an exact two-shock picture, and player 1's noise-state in the tracking game (risk/noisestate.json).
// Section 2: the lone risk-sensitive regulator dX = D dt + dW, cost X^2 + 0.1 D^2: gain sqrt(10)/sqrt(1 - 0.2 theta),
// breakdown at theta* = 5; two paths driven by the same shocks.
(function () {
  "use strict";
  const NS = "http://www.w3.org/2000/svg";
  const me = document.currentScript;
  if (!me) return;
  // data fetches say so when they fail, instead of leaving blank figures and dead controls
  const getJSON = (u) => fetch(u).then((r) => { if (!r.ok) throw new Error(`${u}: ${r.status}`); return r.json(); });
  const failed = (ids) => (e) => {
    console.error(e);
    for (const id of ids) {
      const f = document.getElementById(id);
      if (!f || f.previousElementSibling?.classList.contains("loadfail")) continue;
      const p = document.createElement("p"); p.className = "loadfail"; p.textContent = "The data for this figure could not be loaded; try reloading the page.";
      f.before(p);
    }
  };

  const el = (tag, attrs, parent) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); if (parent) parent.appendChild(e); return e; };
  function mulberry32(a) { return function () { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
  function gauss(r) { let u = 0; while (u === 0) u = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r()); }
  const legend = (id, items) => {
    const box = document.getElementById(id); box.innerHTML = "";
    return items.map(([color, label, dash]) => { const s = document.createElement("span"); s.innerHTML = (dash ? `<i class="${dash === "dot" ? "dot" : "dash"}" style="border-top-color:${color}"></i>` : `<i style="background:${color}"></i>`) + `${label} <b></b>`; box.appendChild(s); return s.querySelector("b"); });
  };
  // Anything reweighted by e^{theta C} is drawn dashed blue, unfilled, with hollow centers: it is where a derivative is
  // evaluated, not something the player believes, so it never shares the belief's solid-past, dashed-forecast style.
  // The dash carries that meaning, not the color: later sections draw the risk-averse equilibrium itself in solid blue.
  const RW = { stroke: "var(--accent)", "stroke-dasharray": "6 4", "stroke-width": 1.7, fill: "none" };
  const hollow = (cx, cy, svg, r = 5) => el("circle", { r, cx, cy, fill: "var(--paper)", stroke: "var(--accent)", "stroke-width": 2.2 }, svg);
  const size = (svg, H) => { const W = Math.max(240, svg.clientWidth || 500); svg.setAttribute("viewBox", `0 0 ${W} ${H}`); svg.setAttribute("height", H); svg.innerHTML = ""; return W; };

  // ------------------------------------------------------------------ judging a deviation: one shock
  // Belief w ~ N(mu, S^2), cost C = a w^2 / 2 + b w; the weight e^{theta C} makes the belief N(m_theta, S^2 / (1 - theta a S^2))
  // with m_theta = (mu + theta S^2 b) / (1 - theta a S^2), affine in mu; theta* = 1 / (a S^2).  The change in cost from a
  // small deviation is a line g(w) = G0 + G1 w; its averages are summed over a fine grid of w, not taken from the formula.
  (function () {
    const S = 0.8, A = 1.0, B = 0.5, G0 = 0.3, G1 = 0.8, TH = 1 / (A * S * S);
    const sth = document.getElementById("theta-1d"), smu = document.getElementById("mu-1d");
    if (!sth || !smu) return;
    const rth = document.getElementById("theta-1d-read"), rmu = document.getElementById("mu-1d-read");
    const wl = legend("why-legend", [["var(--ink-3)", "belief"], ["var(--warm)", "weight e<sup><i>&theta;C</i></sup>,&nbsp;log scale", true], ["var(--accent)", "belief &times; weight (not a belief)", true]]);
    const al = legend("affine-legend", [["var(--ink-3)", "no risk aversion"], ["var(--accent)", "this \u03b8"]]);
    const center = (mu, th) => (mu + th * S * S * B) / (1 - th * A * S * S);
    function draw() {
      const th = (sth.value / 100) * TH, mu = smu.value / 100;
      rth.innerHTML = `${(sth.value / 100).toFixed(2)} &nbsp;(<i>&theta;</i> = ${th.toFixed(2)})`;
      rmu.textContent = `belief centered at ${mu.toFixed(2)}`;
      // the densities on a grid, and the averages of the line by summation
      // the grid follows both curves out to many sd, so neither is cut off (the drawing then shows the blow-up near theta*)
      const sdT = S / Math.sqrt(1 - th * A * S * S), cT = center(mu, th);
      const lo = Math.min(-3, mu - 4 * S), hi = Math.max(5, cT + 4 * sdT), NG = Math.ceil((hi - lo) / 0.004), dw = (hi - lo) / NG;
      const ws = Array.from({ length: NG + 1 }, (_, k) => lo + k * dw);
      const bel = ws.map((w) => Math.exp(-0.5 * ((w - mu) / S) ** 2));
      const lt = ws.map((w) => -0.5 * ((w - mu) / S) ** 2 + th * (0.5 * A * w * w + B * w)), ltm = Math.max(...lt);
      const tw = lt.map((v) => Math.exp(v - ltm));                 // belief x weight in logs: e^{theta C} overflows far out
      const norm = (y) => { const z = y.reduce((a, b) => a + b, 0) * dw; return y.map((v) => v / z); };
      const pb = norm(bel), pt = norm(tw);
      const avg = (p) => p.reduce((acc, v, k) => acc + v * (G0 + G1 * ws[k]), 0) * dw;
      const mean = (p) => p.reduce((acc, v, k) => acc + v * ws[k], 0) * dw;
      const mb = mean(pb), mt = mean(pt);
      let svg = document.getElementById("why"), H = 330, W = size(svg, H);
      const TOP = 200, X = (w) => ((w - lo) / (hi - lo)) * W;
      const ymax = Math.max(...pb, ...pt) * 1.1, Yd = (v) => 8 + (1 - v / ymax) * (TOP - 28);
      el("line", { class: "axis", x1: 0, x2: W, y1: Yd(0), y2: Yd(0) }, svg);
      const path = (ys, f) => ys.map((v, k) => (k ? "L" : "M") + X(ws[k]).toFixed(1) + "," + f(v).toFixed(1)).join("");
      if (th > 0) {                                   // the weight on a log scale: log e^{theta C} = theta C, a parabola
        const lw = ws.map((w) => th * (0.5 * A * w * w + B * w)), lmin = Math.min(...lw), lmax = Math.max(...lw);
        el("path", { class: "curve", d: path(lw, (v) => Yd(((v - lmin) / (lmax - lmin)) * ymax * 0.95)), stroke: "var(--warm)", "stroke-width": 1.3, "stroke-dasharray": "1.5 3" }, svg);
      }
      el("path", { class: "curve", d: path(pb, Yd), stroke: "var(--ink-3)" }, svg);
      el("path", { class: "curve", d: path(pt, Yd), ...RW }, svg);
      // the line below, and each curve's center dropped onto it
      const g = (w) => G0 + G1 * w, glo = g(lo), ghi = g(hi), Yg = (v) => TOP + 6 + ((ghi - v) / (ghi - glo)) * (H - TOP - 26);
      el("line", { class: "zero", x1: 0, x2: W, y1: Yg(0), y2: Yg(0) }, svg);
      el("line", { class: "curve", x1: X(lo), x2: X(hi), y1: Yg(glo), y2: Yg(ghi), stroke: "var(--ink)", "stroke-width": 1.6 }, svg);
      for (const [c, col] of [[mb, "var(--ink-3)"], [mt, "var(--accent)"]]) {
        el("line", { class: "mark", x1: X(c), x2: X(c), y1: Yd(0), y2: Yg(g(c)), stroke: col, "stroke-dasharray": "3 3", "stroke-width": 1.2 }, svg);
        if (col === "var(--accent)") hollow(X(c), Yg(g(c)), svg, 4.5); else el("circle", { class: "dot", r: 4.5, cx: X(c), cy: Yg(g(c)), fill: col }, svg);
      }
      if (th > 0.02 * TH) el("text", { x: X(mt) - 9, y: Yg(g(mt)) - 8, "text-anchor": "end", style: "fill: var(--accent)" }, svg).textContent = "judged here";   // the line rises, so up-left is clear
      el("text", { x: 4, y: TOP + 18 }, svg).textContent = "change in cost from a small deviation";
      const step = hi - lo > 40 ? 20 : hi - lo > 16 ? 5 : 2;
      for (let w = Math.ceil((lo + 0.5) / step) * step; w < hi - 0.3; w += step) el("text", { x: X(w), y: H - 4, "text-anchor": "middle" }, svg).textContent = w;
      wl[0].textContent = `average ${avg(pb).toFixed(3)} = line at ${mb.toFixed(2)}: ${g(mb).toFixed(3)}`;
      wl[1].textContent = "";
      wl[2].textContent = `average ${avg(pt).toFixed(3)} = line at ${mt.toFixed(2)}: ${g(mt).toFixed(3)}`;
      // the affine map
      svg = document.getElementById("affine"); W = size(svg, H);
      const m0 = -1.5, m1 = 1.5, clo = Math.min(-2, center(m0, th) - 0.5), chi = Math.max(7, center(m1, th) + 0.5);
      const Xm = (v) => ((v - m0) / (m1 - m0)) * W, Yc = (v) => 8 + ((chi - Math.max(clo, Math.min(chi, v))) / (chi - clo)) * (H - 30);
      el("line", { class: "zero", x1: 0, x2: W, y1: Yc(0), y2: Yc(0) }, svg);
      el("line", { class: "zero", x1: Xm(0), x2: Xm(0), y1: 8, y2: H - 22 }, svg);
      el("line", { class: "curve", x1: Xm(m0), x2: Xm(m1), y1: Yc(m0), y2: Yc(m1), stroke: "var(--ink-3)", "stroke-dasharray": "5 4" }, svg);
      el("line", { class: "curve", x1: Xm(m0), x2: Xm(m1), y1: Yc(center(m0, th)), y2: Yc(center(m1, th)), stroke: "var(--accent)" }, svg);
      el("circle", { class: "dot", r: 5, cx: Xm(mu), cy: Yc(mu), fill: "var(--ink-3)" }, svg);
      hollow(Xm(mu), Yc(center(mu, th)), svg);
      el("text", { x: W - 4, y: H - 6, "text-anchor": "end" }, svg).textContent = "the belief's center \u2192";
      el("text", { x: 4, y: 20 }, svg).textContent = "where a deviation is judged";
      const slope = 1 / (1 - th * A * S * S), shift = th * S * S * B / (1 - th * A * S * S);
      al[0].textContent = "the belief itself";
      al[1].textContent = `${slope.toFixed(2)} \u00d7 belief + ${shift.toFixed(2)}`;
    }
    sth.addEventListener("input", draw); smu.addEventListener("input", draw);
    new ResizeObserver(draw).observe(document.getElementById("why"));
    draw();
  })();

  // ------------------------------------------------------------------ a past and a future shock; the tracking game
  // The two-shock picture is exact: a (a shock that happened, seen as y = a + e, e ~ N(0, 1), y = 1.2) and b (one to
  // come), prior N(0, I), cost C = (a + b)^2 / 2.  Posterior N(m, S), S = diag(1/2, 1), m = (0.6, 0); tilted by e^{theta C}:
  // precision S^-1 - theta K, K = [[1, 1], [1, 1]], mean S_theta S^-1 m; theta* = 2/3.  The real panels read
  // risk/noisestate.json (tools/risk/compute_risk.py): player 1's noise-state at t = 1/2 on a theta grid to 0.97 theta*.
  getJSON(me.dataset.ns).then((d) => {
    const slider = document.getElementById("theta"), read = document.getElementById("theta-read");
    const frac = () => d.thetas[+slider.value] / d.theta_star;
    const tl = legend("toy-legend", [["var(--ink-3)", "belief"], ["var(--accent)", "reweighted by cost (not a belief)", true]]);
    const stoy = document.getElementById("theta-toy"), rtoy = document.getElementById("theta-toy-read");
    const pl = legend("push-legend", [["var(--ink-3)", "judged with the belief", true], ["var(--accent)", "judged cost-weighted"]]);
    const sl = legend("state-legend", [["var(--ink-3)", "belief: estimate, then forecast"], ["var(--accent)", "reweighted by cost (not a forecast)", true], ["var(--ink)", "what happened"]]);

    function toy(f) {
      const svg = document.getElementById("toy"), H = 300, W = size(svg, H), th = f * (2 / 3);
      const P = [[2 - th, -th], [-th, 1 - th]], det = P[0][0] * P[1][1] - P[0][1] * P[1][0];
      const Ct = [[P[1][1] / det, -P[0][1] / det], [-P[1][0] / det, P[0][0] / det]];
      const rhs = [2 * 0.6, 0];                                      // S^-1 m
      const mt = [Ct[0][0] * rhs[0] + Ct[0][1] * rhs[1], Ct[1][0] * rhs[0] + Ct[1][1] * rhs[1]];
      const lo = -2, hi = Math.max(3, mt[0] + Math.sqrt(Ct[0][0]) + 0.3, mt[1] + Math.sqrt(Ct[1][1]) + 0.3);   // the box holds the ellipse
      const S = Math.min(W, H - 20), off = (W - S) / 2;
      const X = (a) => off + ((a - lo) / (hi - lo)) * S, Y = (b) => H - 20 - ((b - lo) / (hi - lo)) * (H - 20);
      const box = el("clipPath", { id: "toy-box" }, el("defs", {}, svg)); el("rect", { x: off, y: 0, width: S, height: H - 20 }, box);
      const lines = el("g", { "clip-path": "url(#toy-box)" }, svg);
      // the costly direction: level lines of a + b
      for (let c = -4; c <= 6; ++c) el("line", { class: "zero", x1: X(lo), y1: Y(c - lo), x2: X(c - lo), y2: Y(lo), opacity: 0.35 }, lines);
      el("rect", { x: off, y: 0, width: S, height: H - 20, fill: "none", stroke: "var(--rule)" }, svg);
      el("line", { class: "axis", x1: X(lo), x2: X(hi), y1: Y(0), y2: Y(0) }, svg);
      el("line", { class: "axis", x1: X(0), x2: X(0), y1: Y(lo), y2: Y(hi) }, svg);
      el("text", { x: X(hi) - 4, y: H - 5, "text-anchor": "end" }, svg).textContent = "the past shock";
      el("text", { x: X(lo) + 4, y: 12 }, svg).textContent = "the next shock (up)";
      el("text", { x: X(hi) - 4, y: 12, "text-anchor": "end" }, svg).textContent = "costly: up and to the right";
      const gauss2 = (m, C, color, rw, cls) => {                       // the one-standard-deviation ellipse and its center
        const tr = C[0][0] + C[1][1], det = C[0][0] * C[1][1] - C[0][1] * C[0][1], disc = Math.sqrt(Math.max(0, tr * tr / 4 - det));
        const l1 = tr / 2 + disc, l2 = tr / 2 - disc, ang = Math.abs(C[0][1]) > 1e-12 ? Math.atan2(l1 - C[0][0], C[0][1]) : (C[0][0] >= C[1][1] ? 0 : Math.PI / 2);
        let dpath = "";
        for (let k = 0; k <= 72; ++k) {
          const t = (2 * Math.PI * k) / 72, u = Math.sqrt(l1) * Math.cos(t), v = Math.sqrt(Math.max(l2, 0)) * Math.sin(t);
          const a = m[0] + u * Math.cos(ang) - v * Math.sin(ang), b = m[1] + u * Math.sin(ang) + v * Math.cos(ang);
          dpath += (k ? "L" : "M") + X(a).toFixed(1) + "," + Y(b).toFixed(1);
        }
        if (rw) { el("path", { d: dpath, ...RW }, svg); hollow(X(m[0]), Y(m[1]), svg); return; }
        el("path", { d: dpath, fill: color, "fill-opacity": 0.08, stroke: color, "stroke-width": 1.8, class: cls }, svg);
        el("circle", { class: "dot", r: 5, cx: X(m[0]), cy: Y(m[1]), fill: color }, svg);
      };
      gauss2([0.6, 0], [[0.5, 0], [0, 1]], "var(--ink-3)");
      gauss2(mt, Ct, "var(--accent)", true);
      tl[0].textContent = "center (0.60, 0.00)";
      tl[1].textContent = `center (${mt[0].toFixed(2)}, ${mt[1].toFixed(2)})`;
    }

    function state(i) {
      const svg = document.getElementById("state"), H = 300, W = size(svg, H), PADB = 20, PADT = 8;
      const n = d.cells, tt = (k) => (k / n) * d.T, now = Math.round((d.t / d.T) * n);
      const S0 = d.state[0], S1 = d.state[i], TR = d.true_state;
      const hi = Math.max(1.2, ...S0, ...S1, ...TR.slice(0, now + 1)) * 1.05, lo = Math.min(-0.4, ...S0, ...S1, ...TR.slice(0, now + 1)) * 1.1;
      const X = (u) => (u / d.T) * W, Y = (v) => PADT + ((hi - v) / (hi - lo)) * (H - PADT - PADB);
      const tick = hi - lo > 8 ? 5 : hi - lo > 3 ? 1 : 0.5;                 // a few y ticks, since the scale moves with theta
      for (let v = Math.ceil(lo / tick) * tick; v <= hi; v += tick) if (Math.abs(v) > 1e-9) el("text", { x: W - 2, y: Y(v) - 3, "text-anchor": "end", opacity: 0.7 }, svg).textContent = +v.toFixed(1);
      el("line", { class: "zero", x1: 0, x2: W, y1: Y(0), y2: Y(0) }, svg);
      el("line", { class: "axis", x1: 0, x2: W, y1: H - PADB, y2: H - PADB }, svg);
      el("rect", { x: X(d.t), y: PADT, width: W - X(d.t), height: H - PADT - PADB, fill: "var(--ink)", opacity: 0.04 }, svg);
      el("text", { x: X(d.t) + 6, y: PADT + 12 }, svg).textContent = "after now";
      const seg = (y, a, b) => { let s = ""; for (let k = a; k <= b; ++k) s += (k === a ? "M" : "L") + X(tt(k)).toFixed(1) + "," + Y(y[k]).toFixed(1); return s; };
      el("path", { class: "curve", d: seg(TR, 0, now), stroke: "var(--ink)", "stroke-width": 1, opacity: 0.55 }, svg);
      el("path", { class: "curve", d: seg(S0, 0, now), stroke: "var(--ink-3)" }, svg);
      el("path", { class: "curve", d: seg(S0, now, n), stroke: "var(--ink-3)", "stroke-dasharray": "2 3" }, svg);
      el("path", { class: "curve", d: seg(S1, 0, n), ...RW }, svg);       // one style throughout: not an estimate, not a forecast
      for (const u of [0, d.t, d.T]) el("text", { x: X(u), y: H - 5, "text-anchor": u === 0 ? "start" : u === d.T ? "end" : "middle" }, svg).textContent = u === d.t ? "now" : u;
      sl[0].textContent = `now ${S0[now].toFixed(2)}, at the end ${S0[n].toFixed(2)}`;
      sl[1].textContent = `now ${S1[now].toFixed(2)}, at the end ${S1[n].toFixed(2)}`;
      sl[2].textContent = `now ${TR[now].toFixed(2)}`;
    }

    function shocks(i) {
      const svg = document.getElementById("shocks"), W0 = Math.max(240, svg.clientWidth || 500), narrow = W0 < 560;
      const chs = d.channels, per = narrow ? 150 : 190, H = narrow ? per * chs.length : per;
      const W = size(svg, H), PADB = 18, PADT = 18;
      const n = d.cells, now = Math.round((d.t / d.T) * n), w = narrow ? W : (W - 24) / chs.length;
      chs.forEach((ch, c) => {
        const x0 = narrow ? 0 : c * (w + 12), y0 = narrow ? c * per : 0;
        const A0 = d.tilted[0][ch], A1 = d.tilted[i][ch], TR = d.true[ch];
        const all = [...A0, ...A1, ...TR];
        const hi = Math.max(0.3, ...all) * 1.1, lo = Math.min(-0.3, ...all) * 1.1;
        const X = (k) => x0 + (k / n) * w, Y = (v) => y0 + PADT + ((hi - v) / (hi - lo)) * (per - PADT - PADB);
        el("rect", { x: X(now), y: y0 + PADT, width: X(n) - X(now), height: per - PADT - PADB, fill: "var(--ink)", opacity: 0.04 }, svg);
        el("line", { class: "zero", x1: X(0), x2: X(n), y1: Y(0), y2: Y(0) }, svg);
        const p = (y) => y.map((v, k) => (k ? "L" : "M") + X(k).toFixed(1) + "," + Y(v).toFixed(1)).join("");
        el("path", { class: "curve", d: p(TR), stroke: "var(--ink)", "stroke-width": 1, opacity: 0.45 }, svg);
        el("path", { class: "curve", d: p(A0), stroke: "var(--ink-3)" }, svg);
        el("path", { class: "curve", d: p(A1), ...RW }, svg);
        el("text", { x: x0, y: y0 + 12 }, svg).textContent = d.labels[ch] || ch;
        el("text", { x: X(now) + 4, y: y0 + per - 5 }, svg).textContent = "now";
      });
    }

    function draw() {
      const i = +slider.value, f = frac();
      read.innerHTML = `${f.toFixed(2)} &nbsp;(<i>&theta;</i> = ${d.thetas[i].toFixed(2)}, <i>&theta;</i>* = ${d.theta_star.toFixed(2)})`;
      state(i); shocks(i); push(i);
    }
    const drawToy = () => { const f = stoy.value / 100; rtoy.innerHTML = `${f.toFixed(2)}`; toy(f); };
    function push(i) {
      const svg = document.getElementById("push"), H = 180, W = size(svg, H), PADB = 20, PADT = 10;
      const fr = d.thetas.map((t) => t / d.theta_star), P = d.push;
      const lo = Math.min(...P.filter((_, k) => fr[k] <= 0.85)) * 1.25, hi = Math.max(0.1, -0.25 * lo);
      const X = (f) => (f / 1.0) * W, Y = (v) => PADT + ((hi - Math.max(lo, v)) / (hi - lo)) * (H - PADT - PADB);
      el("line", { class: "zero", x1: 0, x2: W, y1: Y(0), y2: Y(0) }, svg);
      el("line", { class: "axis", x1: 0, x2: W, y1: H - PADB, y2: H - PADB }, svg);
      el("line", { class: "curve", x1: X(0), x2: X(fr[fr.length - 1]), y1: Y(P[0]), y2: Y(P[0]), stroke: "var(--ink-3)", "stroke-dasharray": "5 4" }, svg);
      const kend = P.findIndex((v) => v < lo), last = kend > 0 ? kend : P.length;       // stop where it leaves the plot
      el("path", { class: "curve", d: P.slice(0, last).map((v, k) => (k ? "L" : "M") + X(fr[k]).toFixed(1) + "," + Y(v).toFixed(1)).join(""), stroke: "var(--accent)" }, svg);
      el("circle", { class: "dot", r: 5, cx: X(fr[i]), cy: Y(P[i]), fill: "var(--accent)" }, svg);
      for (const f of [0, 0.25, 0.5, 0.75, 1]) el("text", { x: X(f), y: H - 5, "text-anchor": f === 0 ? "start" : f === 1 ? "end" : "middle" }, svg).textContent = f === 1 ? "\u03b8*" : f === 0 ? "\u03b8/\u03b8* = 0" : f;
      el("text", { x: 4, y: H - PADB - 8 }, svg).textContent = W < 520 ? "change in cost from a harder push" : "change in cost from pushing harder now (below zero: it pays)";

      pl[0].textContent = P[0].toFixed(3); pl[1].textContent = `${P[i].toFixed(3)}` + (kend > 0 ? `, falling to ${P[P.length - 1].toFixed(1)} near \u03b8*` : "");
    }
    slider.addEventListener("input", draw); stoy.addEventListener("input", drawToy);
    new ResizeObserver(() => { draw(); drawToy(); }).observe(document.getElementById("state"));
    draw(); drawToy();
  }).catch(failed(["toy", "state", "push", "shocks"]));

  // ------------------------------------------------------------------ the equilibrium (risk/equilibria.json)
  getJSON(me.dataset.eq).then((d) => {
    const slider = document.getElementById("theta-eq"), read = document.getElementById("theta-eq-read");
    if (!slider) return;
    slider.max = d.rows.length - 1;
    const ll = legend("eq-life-legend", [["var(--ink-3)", "risk-neutral"], ["var(--accent)", "the state"], ["var(--warm)", "player 1's push"]]);
    const cl = legend("eq-costs-legend", [["var(--accent)", "entropic"], ["var(--ink)", "expected"]]);
    const emin = d.rows.reduce((a, r) => (r.expected < a.expected ? r : a), d.rows[0]);
    function draw() {
      const row = d.rows[+slider.value], base = d.rows[0];
      read.innerHTML = `${row.theta.toFixed(1)}`;
      let svg = document.getElementById("eq-life"), H = 220, W = size(svg, H), PADB = 20, PADT = 8;
      const n = d.n, k0 = Math.round(d.shock_at * n);
      const all = d.rows.flatMap((r) => [...r.X.slice(k0), ...r.D1.slice(k0)]), lo = Math.min(...all) - 0.08, hi = Math.max(...all) + 0.08;   // every row fits
      const X = (k) => (k / n) * W, Y = (v) => PADT + ((hi - v) / (hi - lo)) * (H - PADT - PADB);
      el("line", { class: "zero", x1: 0, x2: W, y1: Y(0), y2: Y(0) }, svg);
      el("line", { class: "axis", x1: 0, x2: W, y1: H - PADB, y2: H - PADB }, svg);
      const p = (y) => y.slice(k0).map((v, j) => (j ? "L" : "M") + X(k0 + j).toFixed(1) + "," + Y(v).toFixed(1)).join("");
      el("path", { class: "curve", d: p(base.X), stroke: "var(--ink-3)", "stroke-dasharray": "5 4" }, svg);
      el("path", { class: "curve", d: p(base.D1), stroke: "var(--ink-3)", "stroke-dasharray": "5 4" }, svg);
      el("path", { class: "curve", d: p(row.X), stroke: "var(--accent)" }, svg);
      el("path", { class: "curve", d: p(row.D1), stroke: "var(--warm)" }, svg);
      for (const u of [0, 0.2, 0.5, 1]) el("text", { x: X(u * n), y: H - 5, "text-anchor": u === 0 ? "start" : u === 1 ? "end" : "middle" }, svg).textContent = u === 0.2 ? "shock" : u;
      const at = Math.round(0.6 * n);
      ll[0].textContent = ""; ll[1].textContent = `at t = 0.6: ${row.X[at].toFixed(2)} (${base.X[at].toFixed(2)})`; ll[2].textContent = `${row.D1[at].toFixed(2)} (${base.D1[at].toFixed(2)})`;
      svg = document.getElementById("eq-costs"); W = size(svg, H);
      const bk = d.breakdown || d.rows[d.rows.length - 1].theta, cmax = Math.max(...d.rows.map((r) => r.entropic)) * 1.04, cmin = 0.35;
      const Xc = (t) => (t / bk) * W, Yc = (v) => PADT + ((cmax - v) / (cmax - cmin)) * (H - PADT - PADB);
      el("line", { class: "axis", x1: 0, x2: W, y1: H - PADB, y2: H - PADB }, svg);
      el("line", { class: "zero", x1: Xc(bk), x2: Xc(bk), y1: PADT, y2: H - PADB }, svg);
      const cp = (key) => d.rows.map((r, j) => (j ? "L" : "M") + Xc(r.theta).toFixed(1) + "," + Yc(r[key]).toFixed(1)).join("");
      el("path", { class: "curve", d: cp("entropic"), stroke: "var(--accent)" }, svg);
      el("path", { class: "curve", d: cp("expected"), stroke: "var(--ink)" }, svg);
      el("circle", { class: "dot", r: 5, cx: Xc(row.theta), cy: Yc(row.entropic), fill: "var(--accent)" }, svg);
      el("circle", { class: "dot", r: 5, cx: Xc(row.theta), cy: Yc(row.expected), fill: "var(--ink)" }, svg);
      for (const t of [0, 1, 2]) el("text", { x: Xc(t), y: H - 5, "text-anchor": t === 0 ? "start" : "middle" }, svg).textContent = t;
      el("text", { x: Xc(bk), y: H - 5, "text-anchor": "end" }, svg).textContent = `breakdown, ${bk.toFixed(2)}`;
      // inset: the expected cost alone, zoomed, where its dip can be seen
      const iw = Math.min(170, W * 0.42), ih = 78, ix = 8, iy = PADT + 4;
      const sub = d.rows.filter((r) => r.theta <= 1.8), elo = Math.min(...sub.map((r) => r.expected)), ehi = Math.max(...sub.map((r) => r.expected));
      const pad = (ehi - elo) * 0.25, a0 = elo - pad, a1 = ehi + pad;
      const Xi = (t) => ix + (t / 1.8) * iw, Yi = (v) => iy + ((a1 - v) / (a1 - a0)) * ih;
      el("rect", { x: ix - 4, y: iy - 4, width: iw + 8, height: ih + 22, fill: "var(--paper)", stroke: "var(--rule)" }, svg);
      el("path", { class: "curve", d: sub.map((r, j) => (j ? "L" : "M") + Xi(r.theta).toFixed(1) + "," + Yi(r.expected).toFixed(1)).join(""), stroke: "var(--ink)", "stroke-width": 1.6 }, svg);
      el("line", { class: "zero", x1: ix, x2: ix + iw, y1: Yi(base.expected), y2: Yi(base.expected) }, svg);
      el("line", { class: "zero", x1: Xi(emin.theta), x2: Xi(emin.theta), y1: Yi(emin.expected), y2: iy + ih }, svg);
      el("text", { x: Xi(emin.theta) + 7, y: Yi(emin.expected) + 14, "text-anchor": "start" }, svg).textContent = `lowest, \u03b8 \u2248 ${emin.theta.toFixed(1)}`;
      if (row.theta <= 1.8) el("circle", { class: "dot", r: 3.5, cx: Xi(row.theta), cy: Yi(row.expected), fill: "var(--ink)" }, svg);
      el("text", { x: ix, y: iy + ih + 14 }, svg).textContent = "expected cost, zoomed (dotted: risk-neutral)";
      cl[0].textContent = row.entropic.toFixed(3); cl[1].textContent = `${row.expected.toFixed(3)} (risk-neutral ${base.expected.toFixed(3)})`;
    }
    slider.addEventListener("input", draw);
    new ResizeObserver(draw).observe(document.getElementById("eq-life"));
    draw();
  }).catch(failed(["eq-life", "eq-costs"]));

  // ------------------------------------------------------------------ section 2: the regulator
  (function () {
    const R = 0.1, SIG = 1, K0 = Math.sqrt(1 / R), TH_STAR = 1 / (2 * R * SIG * SIG);
    const slider = document.getElementById("theta2"), read = document.getElementById("theta2-read");
    if (!slider) return;
    const gain = (th) => K0 / Math.sqrt(1 - 2 * th * R * SIG * SIG);
    const rng = mulberry32(7), DT = 0.01, TEND = 6, n = Math.round(TEND / DT);
    const dW = Array.from({ length: n }, () => gauss(rng) * Math.sqrt(DT));
    const path = (k) => { const x = [0]; for (let i = 0; i < n; ++i) x.push(x[i] * Math.exp(-k * DT) + SIG * dW[i]); return x; };
    const base = path(K0);
    const pl = legend("paths-legend", [["var(--ink-3)", "risk-neutral"], ["var(--accent)", "risk-averse"]]);

    function draw() {
      const th = (slider.value / 1000) * TH_STAR, k = gain(th);
      read.innerHTML = `${th.toFixed(2)} &nbsp;(gain ${k.toFixed(2)})`;
      let svg = document.getElementById("gain"), H = 190, W = size(svg, H), PADB = 20, PADT = 8;
      const kmax = 16, X = (t) => (t / TH_STAR) * W, Y = (v) => PADT + (1 - v / kmax) * (H - PADT - PADB);
      el("line", { class: "axis", x1: 0, x2: W, y1: H - PADB, y2: H - PADB }, svg);
      el("line", { class: "zero", x1: X(TH_STAR), x2: X(TH_STAR), y1: PADT, y2: H - PADB }, svg);
      let s = "";
      for (let i = 0; i <= 200; ++i) { const t = (i / 200) * TH_STAR * 0.998; const v = Math.min(kmax * 1.2, gain(t)); s += (i ? "L" : "M") + X(t).toFixed(1) + "," + Y(v).toFixed(1); }
      const clip = el("clipPath", { id: "gain-clip" }, el("defs", {}, svg)); el("rect", { x: 0, y: PADT, width: W, height: H - PADT - PADB }, clip);
      el("path", { class: "curve", d: s, stroke: "var(--accent)", "clip-path": "url(#gain-clip)" }, svg);
      el("circle", { class: "dot", r: 5, cx: X(th), cy: Y(Math.min(kmax, k)), fill: "var(--accent)" }, svg);
      for (const t of [0, 1, 2, 3, 4, 5]) el("text", { x: X(t), y: H - 5, "text-anchor": t === 0 ? "start" : t === 5 ? "end" : "middle" }, svg).textContent = t === 5 ? "θ* = 5" : t;
      for (const v of [5, 10, 15]) { el("line", { class: "zero", x1: 0, x2: W, y1: Y(v), y2: Y(v), opacity: 0.35 }, svg); el("text", { x: 2, y: Y(v) - 3 }, svg).textContent = v; }
      el("text", { x: 34, y: PADT + 10 }, svg).textContent = "gain";
      svg = document.getElementById("paths"); W = size(svg, H);
      const xk = path(k), lim = 1.05 * Math.max(...base.map(Math.abs), 2 * SIG / Math.sqrt(2 * K0));
      const Xt = (i) => (i / n) * W, Yx = (v) => PADT + ((lim - v) / (2 * lim)) * (H - PADT - PADB);
      for (const [kk, col] of [[K0, "var(--ink-3)"], [k, "var(--accent)"]]) {                 // the stationary +-2 sd band
        const sd = SIG / Math.sqrt(2 * kk);
        el("rect", { x: 0, y: Yx(2 * sd), width: W, height: Yx(-2 * sd) - Yx(2 * sd), fill: col, opacity: 0.08 }, svg);
      }
      el("line", { class: "zero", x1: 0, x2: W, y1: Yx(0), y2: Yx(0) }, svg);
      const p = (x) => x.map((v, i) => (i ? "L" : "M") + Xt(i).toFixed(1) + "," + Yx(v).toFixed(1)).join("");
      el("path", { class: "curve", d: p(base), stroke: "var(--ink-3)", "stroke-width": 1 }, svg);
      el("path", { class: "curve", d: p(xk), stroke: "var(--accent)", "stroke-width": 1.2 }, svg);
      for (let t = 0; t <= TEND; t += 2) el("text", { x: Xt(t / DT), y: H - 5, "text-anchor": t === 0 ? "start" : t === TEND ? "end" : "middle" }, svg).textContent = t;
      pl[0].textContent = "spread " + (SIG / Math.sqrt(2 * K0)).toFixed(2); pl[1].textContent = "spread " + (SIG / Math.sqrt(2 * k)).toFixed(2);
    }
    slider.addEventListener("input", draw);
    new ResizeObserver(draw).observe(document.getElementById("gain"));
    draw();
  })();

  // ------------------------------------------------------------------ weight then look, or look then weight
  // Two shocks, x seen and y to come, prior standard normal with correlation RHO, cost C = (x + y)^2 / 2, so e^{theta C}
  // keeps the prior Gaussian while theta < theta* = 1 / (2 (1 + RHO)).  Route 1 (the solver): weight the plane, then
  // read it along the line x = x0.  Route 2 (the appendix): read the line first (the belief about y given x0, which
  // moves with x0 through the correlation), then weight it by e^{theta C(x0, y)}.  By Bayes' rule both end on the same
  // curve; each is computed its own way and normalized on a wide grid.  The shock to come runs across in every panel.
  (function () {
    const sth = document.getElementById("theta-eq2"), sx = document.getElementById("seen-eq2");
    if (!sth || !sx) return;
    const rth = document.getElementById("theta-eq2-read"), rx = document.getElementById("seen-eq2-read");
    const RHO = 0.6, TS = 1 / (2 * (1 + RHO)), R = 4, SZ = 150, NH = 44;
    const lg = legend("routes-legend", [["var(--ink-3)", "the player\u2019s belief"], ["var(--accent)", "weight, then look", true], ["var(--warm)", "look, then weight", "dot"]]);
    const panel = (id) => { const svg = document.getElementById(id); svg.setAttribute("viewBox", `0 0 ${SZ} ${SZ}`); svg.setAttribute("width", SZ); svg.setAttribute("height", SZ); svg.innerHTML = ""; return svg; };
    const Xp = (v) => ((v + R) / (2 * R)) * SZ, Yp = (v) => ((R - v) / (2 * R)) * SZ;
    const ticks = (svg, B, up) => {                                   // -2, 0, 2 on the to-come axis (and up the seen one)
      for (const v of [-2, 0, 2]) {
        el("text", { x: Xp(v), y: B - 3, "text-anchor": "middle", opacity: 0.75 }, svg).textContent = v;
        if (up && v) el("text", { x: 3, y: Yp(v) + 3, opacity: 0.75 }, svg).textContent = v;
      }
    };
    const ring = (svg, cx, cy, col) => el("circle", { cx, cy, r: 5, fill: "var(--paper)", stroke: col, "stroke-width": 2 }, svg);
    function plane(id, dens, col, x0) {                                // a density over both shocks, shaded: to come across, seen up
      const svg = panel(id), c = (2 * R) / NH;
      let mx = 0; const v = [];
      for (let a = 0; a < NH; ++a) for (let b = 0; b < NH; ++b) { const x = -R + (a + 0.5) * c, y = -R + (b + 0.5) * c, d = dens(x, y); v.push([x, y, d]); mx = Math.max(mx, d); }
      for (const [x, y, d] of v) if (d > 0.01 * mx) el("rect", { x: Xp(y - c / 2), y: Yp(x + c / 2), width: SZ / NH + 0.3, height: SZ / NH + 0.3, fill: col, "fill-opacity": (0.85 * d / mx).toFixed(3) }, svg);
      el("rect", { x: 0.5, y: 0.5, width: SZ - 1, height: SZ - 1, fill: "none", stroke: "var(--rule)" }, svg);
      el("line", { class: "zero", x1: 0, x2: SZ, y1: Yp(0), y2: Yp(0) }, svg);
      el("line", { class: "zero", x1: Xp(0), x2: Xp(0), y1: 0, y2: SZ }, svg);
      el("line", { x1: 0, x2: SZ, y1: Yp(x0), y2: Yp(x0), stroke: "var(--ink)", "stroke-width": 1.8 }, svg);
      el("text", { x: 14, y: 12 }, svg).textContent = "seen \u2191";
      el("text", { x: SZ - 4, y: SZ - 16, "text-anchor": "end" }, svg).textContent = "to come \u2192";
      ticks(svg, SZ, true);
    }
    const ys = Array.from({ length: 1601 }, (_, k) => -12 + (k / 1600) * 26), dy = 26 / 1600;
    const norm = (f) => { const z = f.reduce((a, b) => a + b, 0) * dy; return f.map((q) => q / z); };
    const mean = (f) => f.reduce((a, q, k) => a + q * ys[k], 0) * dy;
    function profile(id, curves, top) {                               // curves over the shock to come, on the squares' axis
      const svg = panel(id), B = SZ - 16;
      el("rect", { x: 0.5, y: 0.5, width: SZ - 1, height: SZ - 1, fill: "none", stroke: "var(--rule)" }, svg);
      el("line", { class: "axis", x1: 0, x2: SZ, y1: B, y2: B }, svg);
      el("text", { x: SZ - 4, y: 12, "text-anchor": "end" }, svg).textContent = "to come \u2192";
      ticks(svg, SZ, false);
      const vis = ys.map((y, k) => k).filter((k) => Math.abs(ys[k]) <= R);
      for (const [f, attrs] of curves) el("path", { d: vis.map((k, n) => (n ? "L" : "M") + Xp(ys[k]).toFixed(1) + "," + (B - (f[k] / top) * (B - 18)).toFixed(1)).join(""), fill: "none", ...attrs }, svg);
      return { mark: (m, col) => ring(svg, Xp(m), B, col), dot: (m) => el("circle", { cx: Xp(m), cy: B, r: 4, fill: "var(--ink-3)" }, svg) };
    }
    function draw() {
      const th = (sth.value / 100) * TS, x0 = sx.value / 100;
      rth.innerHTML = `${(sth.value / 100).toFixed(2)} &nbsp;(<i>&theta;</i> = ${th.toFixed(3)})`;
      rx.textContent = x0.toFixed(2);
      const prior = (x, y) => Math.exp(-0.5 * (x * x - 2 * RHO * x * y + y * y) / (1 - RHO * RHO));
      const weighted = (x, y) => prior(x, y) * Math.exp(0.5 * th * (x + y) ** 2);
      plane("r1a", prior, "var(--ink-3)", x0); plane("r2a", prior, "var(--ink-3)", x0);
      plane("r1b", weighted, "var(--accent)", x0);
      const belief = norm(ys.map((y) => prior(x0, y)));                                    // the belief about y, given x0
      const start = norm(ys.map((y) => weighted(x0, y)));                                  // route 1: the weighted plane along the line
      const now = norm(belief.map((q, k) => q * Math.exp(0.5 * th * (x0 + ys[k]) ** 2))); // route 2: the belief, then weighted
      const top = Math.max(...belief, ...start) * 1.05;
      const faint = { stroke: "var(--ink-3)", "stroke-width": 1.2, opacity: 0.5 };
      const DOT = { stroke: "var(--warm)", "stroke-width": 2.8, "stroke-dasharray": "0.5 5", "stroke-linecap": "round" };
      const c1 = profile("r1c", [[belief, faint], [start, RW]], top); c1.mark(mean(start), "var(--accent)");
      profile("r2b", [[belief, { stroke: "var(--ink-3)", "stroke-width": 1.8 }]], top).dot(mean(belief));
      const c2 = profile("r2c", [[belief, faint], [start, { ...RW, "stroke-width": 1.4 }], [now, DOT]], top); c2.mark(mean(now), "var(--warm)");
      const c3 = (v) => (Math.abs(v) < 5e-4 ? 0 : v).toFixed(3);
      lg[0].textContent = `center ${c3(mean(belief))}`; lg[1].textContent = `center ${c3(mean(start))}`; lg[2].textContent = `center ${c3(mean(now))}`;
    }
    sth.addEventListener("input", draw); sx.addEventListener("input", draw);
    draw();
  })();
})();
