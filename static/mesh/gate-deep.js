// /gate/deep: the decision mesh one level below /gate/how. The page draws coin flips, fits them with both engines in
// fit-worker.js (candidate trace on), and builds each scene from the two runs. Sources, as cited below:
//   tri  = ~/triangular-decision-mesh/core   (fit/refine.cpp, fit/gate.cpp, estimator/*.cpp, mesh/*.cpp, docs/MODEL.md)
//   rect = ~/rectangular-decision-mesh/core  (fit/refine.cpp, fit/gate.cpp, docs/WRITEUP_2026-09-23.md)
// Live from the runs: basis2d (rect), score, lindsey, select, overlap, variance, stop, rect, and the numbers under
// refit and bias. Computed on the page with the engine's rule from the run's starting mesh: bisect. Sketches, and
// labelled so on the stage: basis1d, the vector picture in refit, the top of bias, rescore.
(function () {
  "use strict";
  const root = document.getElementById("gatedeep");
  const svg = document.getElementById("stage");
  const steps = [...document.querySelectorAll(".step")];
  if (!root || !svg) return;
  const NS = "http://www.w3.org/2000/svg";
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const $ = (id) => document.getElementById(id);

  // ------------------------------------------------------------------ helpers (as in gate-how.js)
  const el = (tag, attrs, parent) => { const n = document.createElementNS(NS, tag); for (const [k, v] of Object.entries(attrs || {})) n.setAttribute(k, v); if (parent) parent.appendChild(n); return n; };
  const clamp = (x, a = 0, b = 1) => Math.max(a, Math.min(b, x));
  const ease = (t) => (t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2);
  const seg = (t, a, b) => ease(clamp((t - a) / (b - a)));
  const lin = (t, a, b) => clamp((t - a) / (b - a));
  const fade = (n, t) => { n.style.opacity = clamp(t); };
  const text = (g, x, y, s, cls, anchor = "middle") => { const t = el("text", { x, y, class: cls || "", "text-anchor": anchor }, g); t.textContent = s; return t; };
  function mulberry32(a) { return function () { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
  function gauss(r) { let u = 0; while (u === 0) u = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r()); }
  function stroke(g, d, cls, width = 1.6) {
    const a = el("path", { d, class: "pencil " + (cls || ""), "stroke-width": width }, g);
    const len = a.getTotalLength() || 1;
    a.style.strokeDasharray = `${len} ${len}`; a.style.strokeDashoffset = len;
    return { a, set(t) { a.style.strokeDashoffset = len * (1 - clamp(t)); } };
  }
  const line = (g, x1, y1, x2, y2, cls, w = 1) => el("line", { x1, y1, x2, y2, class: "pencil " + (cls || ""), "stroke-width": w }, g);
  const poly = (pts) => "M" + pts.map((p) => p[0].toFixed(1) + "," + p[1].toFixed(1)).join(" L");
  const fmt = (v, d = 2) => (Math.abs(v) < 0.5 * Math.pow(10, -d) ? (0).toFixed(d) : v.toFixed(d)).replace("-", "−");
  const pct = (v) => Math.round(100 * v) + "%";
  const phi = (z) => Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI);
  const expit = (t) => 1 / (1 + Math.exp(-t));
  const k6 = (x, y) => (+x).toFixed(6) + "," + (+y).toFixed(6);
  // erfc, Numerical Recipes' erfcc (relative error below 1.2e-7): for the two-sided normal p-values of BH
  function erfc(x) {
    const z = Math.abs(x), t = 1 / (1 + 0.5 * z);
    const r = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
    return x >= 0 ? r : 2 - r;
  }
  // Gaussian elimination with partial pivoting, the same as tri fit/gate.cpp:10-26 (solve_small)
  function solve(A, b) {
    const n = b.length; A = A.map((r) => r.slice()); b = b.slice();
    for (let c = 0; c < n; ++c) {
      let piv = c;
      for (let r = c + 1; r < n; ++r) if (Math.abs(A[r][c]) > Math.abs(A[piv][c])) piv = r;
      [A[c], A[piv]] = [A[piv], A[c]]; [b[c], b[piv]] = [b[piv], b[c]];
      for (let r = c + 1; r < n; ++r) { const f = A[r][c] / A[c][c]; for (let j = c; j < n; ++j) A[r][j] -= f * A[c][j]; b[r] -= f * b[c]; }
    }
    const out = new Array(n).fill(0);
    for (let r = n - 1; r >= 0; --r) { let s = b[r]; for (let j = r + 1; j < n; ++j) s -= A[r][j] * out[j]; out[r] = s / A[r][r]; }
    return out;
  }

  // ------------------------------------------------------------------ the data (the same odds and flips as /gate/how)
  const BASE = -1, SITES = 6000, FLIPS = 20;
  let POOL_SD = 0.25;
  const TRUTHS = {
    hills: (x, y) => BASE + 1.8 * Math.exp(-((x - 0.3) ** 2 + (y - 0.7) ** 2) / 0.02) - 1.5 * Math.exp(-((x - 0.7) ** 2 + (y - 0.3) ** 2) / 0.03),
    peaks: (x, y) => BASE + 1.6 * Math.exp(-((x - 0.25) ** 2 + (y - 0.3) ** 2) / 0.006) + 1.1 * Math.exp(-((x - 0.7) ** 2 + (y - 0.7) ** 2) / 0.008) - 1.4 * Math.exp(-((x - 0.72) ** 2 + (y - 0.22) ** 2) / 0.01),
    ring: (x, y) => BASE + 1.4 * Math.exp(-((Math.hypot(x - 0.5, y - 0.5) - 0.28) ** 2) / 0.004),
  };
  function makeData(truth, seed) {
    const r = mulberry32(seed * 7919 + 17), f = TRUTHS[truth];
    const x = new Float64Array(SITES), y = new Float64Array(SITES), n = new Int32Array(SITES), k = new Int32Array(SITES);
    const rows = ["wala,wac,n,k"];
    for (let i = 0; i < SITES; ++i) {
      x[i] = r(); y[i] = r();
      n[i] = Math.max(1, Math.round(FLIPS * (0.5 + r())));
      const p = expit(f(x[i], y[i]) + POOL_SD * gauss(r));
      let kk = 0; for (let j = 0; j < n[i]; ++j) if (r() < p) ++kk;
      k[i] = kk;
      rows.push(x[i].toFixed(5) + "," + y[i].toFixed(5) + "," + n[i] + "," + k[i]);
    }
    return { x, y, n, k, csv: rows.join("\n") + "\n" };
  }

  // ------------------------------------------------------------------ colour: log-odds against the background
  function rgbOf(css) {
    const c = document.createElement("canvas").getContext("2d"); c.fillStyle = "#000"; c.fillStyle = css;
    const s = c.fillStyle; if (s[0] === "#") return [1, 3, 5].map((i) => parseInt(s.slice(i, i + 2), 16));
    return (s.match(/[\d.]+/g) || [0, 0, 0]).slice(0, 3).map(Number);
  }
  let INK = null;
  function inks() {
    const cs = getComputedStyle(root);
    INK = { acc: rgbOf(cs.getPropertyValue("--accent").trim() || "#2743d6"), warm: rgbOf(cs.getPropertyValue("--warm").trim() || "#c0532b"), paper: rgbOf(cs.getPropertyValue("--sheet").trim() || cs.getPropertyValue("--paper").trim() || "#fff") };
  }
  function ramp(v) {
    const t = clamp(v / 2.2, -1, 1), to = t > 0 ? INK.acc : INK.warm, a = Math.pow(Math.abs(t), 0.8);
    return INK.paper.map((p, i) => Math.round(p + (to[i] - p) * a));
  }

  // ------------------------------------------------------------------ the square on the stage
  const SQ = { x: 80, y: 70, s: 440 };
  const sx = (x) => SQ.x + x * SQ.s, sy = (y) => SQ.y + (1 - y) * SQ.s;
  function frame(g, label) {
    el("rect", { x: SQ.x, y: SQ.y, width: SQ.s, height: SQ.s, class: "frame" }, g);
    if (label) text(g, SQ.x, SQ.y - 12, label, "mono", "start");
  }
  let clipN = 0;
  function clipSquare(g) {
    const id = "deepclip" + ++clipN, d = el("defs", {}, g), c = el("clipPath", { id }, d);
    el("rect", { x: SQ.x, y: SQ.y, width: SQ.s, height: SQ.s }, c);
    return el("g", { "clip-path": `url(#${id})` }, g);
  }

  // ------------------------------------------------------------------ Lindsey's method, ported line by line from
  // tri fit/gate.cpp:106-254 (lindsey_lfdr) so the page can draw its pieces. Scale box: tri 6 (gate.cpp:231), rect 3
  // (rect fit/gate.cpp:236). The damped retry (gate.cpp:28-91) is not ported: a round needing it reports failure here.
  function lindsey(zIn, s0max) {
    const NB = 72, DEG = 6, lo = -9.5, hi = 9.5, d = (hi - lo) / NB;                     // gate.cpp:107, :113
    const z = zIn.map((v) => clamp(v, -9, 9)), M = z.length;                               // :112
    const counts = new Array(NB).fill(0), mids = [];
    for (let b = 0; b < NB; ++b) mids.push(lo + (b + 0.5) * d);
    for (const v of z) {                                                                    // linear binning, :116-125
      const t = (v - lo) / d - 0.5, b0 = clamp(Math.floor(t), 0, NB - 2), w = clamp(t - b0, 0, 1);
      counts[b0] += 1 - w; counts[b0 + 1] += w;
    }
    let mm = 0, ms = 0;                                                                     // :127-132
    for (const m of mids) mm += m;
    mm /= NB;
    for (const m of mids) ms += (m - mm) ** 2;
    ms = Math.sqrt(ms / (NB - 1));
    const B = mids.map((m) => { const t = (m - mm) / ms, r = []; let p = 1; for (let k = 0; k <= DEG; ++k) { r.push(p); p *= t; } return r; });   // :133-137
    let beta = new Array(DEG + 1).fill(0), converged = false;
    for (let it = 0; it < 100; ++it) {                                                      // Poisson IRLS, :140-174
      const A = [...Array(DEG + 1)].map(() => new Array(DEG + 1).fill(0)), rhs = new Array(DEG + 1).fill(0);
      for (let b = 0; b < NB; ++b) {
        let eta = 0; for (let k = 0; k <= DEG; ++k) eta += B[b][k] * beta[k];
        eta = clamp(eta, -30, 30);
        const mu = Math.exp(eta), zw = eta + (counts[b] - mu) / Math.max(mu, 1e-10);
        for (let k = 0; k <= DEG; ++k) { rhs[k] += B[b][k] * mu * zw; for (let l = 0; l <= DEG; ++l) A[k][l] += B[b][k] * mu * B[b][l]; }
      }
      for (let k = 0; k <= DEG; ++k) A[k][k] += 1e-8;
      const nb = solve(A, rhs);
      let maxdiff = 0;
      for (let k = 0; k <= DEG; ++k) { if (!isFinite(nb[k])) return null; maxdiff = Math.max(maxdiff, Math.abs(nb[k] - beta[k])); }
      beta = nb;
      if (maxdiff < 1e-9) { converged = true; break; }
    }
    if (!converged) return null;
    const dens = (v) => { const t = (clamp(v, -9, 9) - mm) / ms; let eta = 0, p = 1; for (let k = 0; k <= DEG; ++k) { eta += p * beta[k]; p *= t; } return Math.exp(clamp(eta, -30, 30)) / (M * d); };
    const fbins = mids.map((_, b) => { let eta = 0; for (let k = 0; k <= DEG; ++k) eta += B[b][k] * beta[k]; return Math.exp(clamp(eta, -30, 30)) / (M * d); });   // :176-195
    // central matching: a quadratic in z fitted to log fbins over |mid| <= 2, :197-209
    const A3 = [[0, 0, 0], [0, 0, 0], [0, 0, 0]], b3 = [0, 0, 0];
    mids.forEach((m, b) => {
      if (Math.abs(m) > 2) return;
      const lx = Math.log(Math.max(fbins[b], 1e-300)), xs = [1, m, m * m];
      for (let k = 0; k < 3; ++k) { b3[k] += xs[k] * lx; for (let l = 0; l < 3; ++l) A3[k][l] += xs[k] * xs[l]; }
    });
    const [a0, a1, a2] = solve(A3, b3);
    if (!(a2 < -1e-10)) return null;                                                        // invalid_central_fit, :211-214
    const s0 = clamp(Math.sqrt(-1 / (2 * a2)), 0.35, s0max);                               // :231-238
    const d0 = clamp(a1 * s0 * s0, -0.5, 0.5);
    const pi0 = clamp(Math.exp(a0 + (d0 * d0) / (2 * s0 * s0)) * Math.sqrt(2 * Math.PI) * s0, 0.05, 1);
    return { mids, counts, fbins, dens, a0, a1, a2, d0, s0, pi0, d, M };
  }

  // ------------------------------------------------------------------ the scenes, built from the two runs
  let scenes = {}, active = null, prog = 0;
  const group = (name) => { const g = el("g", { "data-scene": name }, svg); g.style.opacity = 0; g.style.transition = "opacity 0.6s"; return g; };
  const setV = (k, s) => document.querySelectorAll(`[data-v="${k}"]`).forEach((n) => { n.textContent = s; });

  function build(data, T, R) {
    svg.textContent = ""; scenes = {}; clipN = 0;
    inks();
    const DT = T.detail, DR = R.detail;
    const r0 = DT.cands.filter((c) => c.round === 0), cal0 = DT.calib.find((c) => c.round === 0) || DT.calib[0];
    // A tri location can hold a coarse and a fine node (the two-node hierarchy), so admission is read off every
    // vertex at the candidate's location, not the last one the worker's trace keyed.
    const admAt = new Set(DT.vertices.filter((v) => v.admitted).map((v) => v.round + "|" + k6(v.x, v.y)));
    const tAdm = (c) => admAt.has(c.round + "|" + k6(c.x, c.y));
    const hT = new Map(T.verts.map((v) => [k6(v.x, v.y), v.h]));
    // the training sites: every site whose coordinates are not among the held-out rows (run_obs.csv, is_train = 0)
    const held = new Set();
    if (T.heldout && T.heldout.rows) for (let i = 0; i < T.heldout.rows.x.length; ++i) held.add(T.heldout.rows.x[i].toFixed(5) + "," + T.heldout.rows.y[i].toFixed(5));
    const train = [];
    for (let i = 0; i < data.x.length; ++i) if (!held.has(data.x[i].toFixed(5) + "," + data.y[i].toFixed(5))) train.push(i);

    // 1. newest-vertex bisection ----------------------------------------------------------------------------------
    // The engine's rule on a small copy of its geometry. Root: the square cut along bottom-left to top-right, which is
    // both faces' refinement edge (tri mesh/mesh.cpp:52-69). A face's refinement edge is its hypotenuse, edges[0];
    // splitting any other edge first activates the hypotenuse's midpoint (mesh/face.cpp:62-67), and an edge splits the
    // faces on both its sides (mesh/edge.cpp:49-59). Each child's refinement edge is a short edge of its parent
    // (mesh/edge.cpp:126-129). Pre-refinement activates every inactive midpoint, sorted by x then y, until there are at
    // least 48 faces (fit/refine.cpp:37, :261-279).
    scenes.bisect = (() => {
      const g = group("bisect");
      const mid = (p, q) => [(p[0] + q[0]) / 2, (p[1] + q[1]) / 2];
      const ek = (p, q) => { const a = k6(p[0], p[1]), b = k6(q[0], q[1]); return a < b ? a + "|" + b : b + "|" + a; };
      let leaves = [{ a: [0, 1], b: [0, 0], c: [1, 1] }, { a: [1, 0], b: [0, 0], c: [1, 1] }];
      const verts = new Map();
      [[0, 0], [1, 0], [0, 1], [1, 1]].forEach((p) => verts.set(k6(p[0], p[1]), { p, kind: "pre" }));
      const hasEdge = (f, K) => ek(f.b, f.c) === K || ek(f.a, f.b) === K || ek(f.a, f.c) === K;
      const path = () => leaves.map((f) => `M${sx(f.a[0]).toFixed(1)},${sy(f.a[1]).toFixed(1)}L${sx(f.b[0]).toFixed(1)},${sy(f.b[1]).toFixed(1)}L${sx(f.c[0]).toFixed(1)},${sy(f.c[1]).toFixed(1)}Z`).join("");
      const events = [];
      function activate(p, q, kind) {
        const m = mid(p, q), mk = k6(m[0], m[1]);
        if (verts.has(mk)) return;
        const K = ek(p, q);
        for (;;) {
          const f = leaves.find((h) => hasEdge(h, K));
          if (!f) break;
          if (ek(f.b, f.c) === K) leaves.splice(leaves.indexOf(f), 1, { a: m, b: f.a, c: f.b }, { a: m, b: f.c, c: f.a });
          else activate(f.b, f.c, kind === "pre" ? "pre" : "forced");
        }
        verts.set(mk, { p: m, kind });
        if (kind !== "pre") events.push({ d: path(), p: m, kind });
      }
      const pre = [path()];
      while (leaves.length < 48) {
        const E = new Map();
        for (const f of leaves) for (const [p, q] of [[f.b, f.c], [f.a, f.b], [f.a, f.c]]) { const m = mid(p, q), mk = k6(m[0], m[1]); if (!verts.has(mk)) E.set(mk, [p, q, m]); }
        [...E.values()].sort((u, v) => u[2][0] - v[2][0] || u[2][1] - v[2][1]).forEach(([p, q]) => activate(p, q, "pre"));
        pre.push(path());
      }
      const faces0 = leaves.length;
      // five requested splits near the run's deepest admitted vertex: each asks for the midpoint of the short edge of the
      // face containing that point whose midpoint is nearest to it
      const deep = DT.vertices.filter((v) => v.active && v.admitted).sort((a, b) => b.depth - a.depth)[0] || { x: 0.3, y: 0.3 };
      const Q = [clamp(deep.x + 0.004, 0.001, 0.999), clamp(deep.y + 0.0013, 0.001, 0.999)];
      const inside = (f, q) => {
        const d = (p1, p2, p3) => (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1]);
        const d1 = d(q, f.a, f.b), d2 = d(q, f.b, f.c), d3 = d(q, f.c, f.a);
        return !((d1 < -1e-12 || d2 < -1e-12 || d3 < -1e-12) && (d1 > 1e-12 || d2 > 1e-12 || d3 > 1e-12));
      };
      for (let r = 0; r < 5; ++r) {
        const f = leaves.find((h) => inside(h, Q));
        if (!f) break;
        const legs = [[f.a, f.b], [f.a, f.c]].sort((u, v) => Math.hypot(...mid(...u).map((c, i) => c - Q[i])) - Math.hypot(...mid(...v).map((c, i) => c - Q[i])));
        activate(legs[0][0], legs[0][1], "req");
      }
      const forced = events.filter((e) => e.kind === "forced").length;
      frame(g);
      const lab = text(g, SQ.x, SQ.y - 12, "", "mono", "start");
      // after pre-refinement the view enlarges about the requested point, so the small faces stay legible
      const ZF = 4, Qs = [sx(Q[0]), sy(Q[1])], cen = [SQ.x + SQ.s / 2, SQ.y + SQ.s / 2];
      const zg = el("g", {}, clipSquare(g));
      const mesh = el("path", { d: pre[0], class: "meshdark", "vector-effect": "non-scaling-stroke" }, zg);
      const dotR = (e) => (e.kind === "req" ? 4.2 : 3.4);
      const dots = events.map((e) => el("circle", { cx: sx(e.p[0]), cy: sy(e.p[1]), r: dotR(e), class: e.kind === "req" ? "pos" : "neg" }, zg));
      const ring = el("circle", { r: 9, class: "ring", "vector-effect": "non-scaling-stroke" }, zg);
      const leg = el("g", {}, g);
      el("circle", { cx: SQ.x + 6, cy: SQ.y + SQ.s + 22, r: 4.2, class: "pos" }, leg); text(leg, SQ.x + 16, SQ.y + SQ.s + 26, "requested", "tiny", "start");
      el("circle", { cx: SQ.x + 126, cy: SQ.y + SQ.s + 22, r: 3.4, class: "neg" }, leg); text(leg, SQ.x + 136, SQ.y + SQ.s + 26, "added by the closure", "tiny", "start");
      setV("faces0", String(T.rounds.length ? T.rounds[0].faces : faces0));
      setV("forced", String(forced));
      if (T.rounds.length && T.rounds[0].faces !== faces0) console.warn("gate-deep: page pre-refinement", faces0, "faces, engine", T.rounds[0].faces);
      return { g, update(t) {
        let e = 0;
        const u = seg(t, 0.32, 0.42), z = 1 + (ZF - 1) * u, c = [Qs[0] + (cen[0] - Qs[0]) * u, Qs[1] + (cen[1] - Qs[1]) * u];
        zg.setAttribute("transform", `translate(${c[0].toFixed(2)},${c[1].toFixed(2)}) scale(${z.toFixed(4)}) translate(${-Qs[0]},${-Qs[1]})`);
        dots.forEach((d, i) => d.setAttribute("r", dotR(events[i]) / z)); ring.setAttribute("r", 9 / z);
        if (t < 0.32) {
          const i = Math.min(pre.length - 1, Math.floor(lin(t, 0.02, 0.3) * (pre.length - 1) + 1e-9));
          mesh.setAttribute("d", pre[i]);
          const nf = [2, 8, 32, 128][i] || faces0;
          lab.textContent = i === pre.length - 1 ? `the starting mesh: ${faces0} faces` : `pre-refinement, pass ${i}: ${nf} faces`;
        } else {
          e = Math.min(events.length, Math.floor(lin(t, 0.34, 0.92) * events.length + 1e-9));
          mesh.setAttribute("d", e ? events[e - 1].d : pre[pre.length - 1]);
          lab.textContent = `enlarged ${ZF}×: five requested splits, ${e} of ${events.length} new vertices`;
        }
        dots.forEach((d, i) => fade(d, i < e ? 1 : 0));
        if (e) { ring.setAttribute("cx", sx(events[e - 1].p[0])); ring.setAttribute("cy", sy(events[e - 1].p[1])); }
        fade(ring, e ? 1 : 0); fade(leg, seg(t, 0.32, 0.4));
      } };
    })();

    // 2. the hierarchical basis in one dimension (a sketch) -----------------------------------------------------------
    // A constrained vertex's height is its parents' average (tri mesh/vertex.cpp:154, :399-405); a free one adds its
    // surplus (docs/MODEL.md:53-55; rect WRITEUP_2026-09-23.md:29-31).
    scenes.basis1d = (() => {
      const g = group("basis1d");
      const gf = (u) => 0.45 * Math.sin(3.4 * u + 0.4) + 0.5 * Math.exp(-(((u - 0.63) / 0.08) ** 2)) - 0.1;
      const X = (u) => 80 + u * 440, Y = (v) => 235 - v * 140, YB = (v) => 470 - v * 140, LMAX = 4;
      const interp = (L, u) => { const n = 2 ** L, j = Math.min(n - 1, Math.floor(u * n)), a = j / n, b = (j + 1) / n; return gf(a) + ((u - a) / (b - a)) * (gf(b) - gf(a)); };
      const curve = (fn) => { const p = []; for (let i = 0; i <= 220; ++i) { const u = i / 220; p.push([X(u), Y(fn(u))]); } return poly(p); };
      text(g, 70, 64, "a one-dimensional sketch", "tiny", "start");
      line(g, 80, Y(0), 520, Y(0), "soft", 1);
      el("path", { d: curve(gf), class: "pencil soft", "stroke-width": 1.4, fill: "none" }, g);
      const prev = el("path", { class: "pencil accent", "stroke-width": 1.4, fill: "none", "stroke-dasharray": "4 4" }, g);
      const cur = el("path", { class: "pencil accent", "stroke-width": 2.2, fill: "none" }, g);
      const topLab = text(g, 70, 90, "", "mono", "start");
      line(g, 80, YB(0), 520, YB(0), "", 1);
      const botLab = text(g, 70, 345, "", "mono", "start");
      const levels = [];
      for (let L = 0; L <= LMAX; ++L) {
        const q = el("g", {}, g), n = 2 ** L;
        if (L === 0) {
          // the two ends: the only vertices without parents; their tents are the two straight ramps
          [[0, gf(0)], [1, gf(1)]].forEach(([u, v]) => {
            el("path", { d: poly([[X(u), YB(0)], [X(u), YB(v)], [X(1 - u), YB(0)]]) + "Z", class: v > 0 ? "shade" : "shadew" }, q);
            el("circle", { cx: X(u), cy: Y(v), r: 4.5, class: "ink" }, q);
          });
        } else {
          // each level on its own scale: the surpluses shrink level by level, and on the top panel's scale the deep ones
          // would be flat lines; the label says how much each level is enlarged
          let up = 0, dn = 0;
          for (let j = 0; j < n / 2; ++j) { const u = (2 * j + 1) / n, h = 1 / n, s = gf(u) - (gf(u - h) + gf(u + h)) / 2; up = Math.max(up, s); dn = Math.max(dn, -s); }
          // room: 105 px above the zero line (to the panel's label), 72 below (to the caption)
          const zoom = Math.min(40, up > 0 ? 105 / (140 * up) : 40, dn > 0 ? 72 / (140 * dn) : 40), ZB = (v) => YB(v * Math.max(1, zoom));
          q.dataset.zoom = zoom >= 1.5 ? `, enlarged ${zoom.toFixed(0)}×` : "";
          for (let j = 0; j < n / 2; ++j) {
            const u = (2 * j + 1) / n, h = 1 / n, avg = (gf(u - h) + gf(u + h)) / 2, s = gf(u) - avg;
            el("path", { d: poly([[X(u - h), ZB(0)], [X(u), ZB(s)], [X(u + h), ZB(0)]]) + "Z", class: s > 0 ? "shade" : "shadew" }, q);
            line(q, X(u - h), ZB(0), X(u), ZB(s), s > 0 ? "accent" : "warm", 1.2); line(q, X(u), ZB(s), X(u + h), ZB(0), s > 0 ? "accent" : "warm", 1.2);
            if (L <= 3) {
              el("circle", { cx: X(u), cy: Y(avg), r: 3.2, class: "ring" }, q);
              line(q, X(u), Y(avg), X(u), Y(gf(u)), s > 0 ? "accent" : "warm", 2);
              el("circle", { cx: X(u), cy: Y(gf(u)), r: 4, class: s > 0 ? "pos" : "neg" }, q);
            }
          }
          if (L === 1) {
            const u = 0.5, avg = (gf(0) + gf(1)) / 2;
            text(q, X(u) + 10, Y(avg) + 16, "parents' average", "tiny", "start");
            text(q, X(u) + 10, (Y(avg) + Y(gf(u))) / 2 + 4, "surplus", "label " + (gf(u) > avg ? "acc" : "warmt"), "start");
          }
        }
        levels.push(q);
      }
      const circ = text(g, 300, 560, "open circles: the parents' average; the stroke from it to the dot: the surplus", "tiny");
      return { g, update(t) {
        const L = Math.min(LMAX, Math.floor(lin(t, 0.04, 0.9) * (LMAX + 1)));
        levels.forEach((q, i) => fade(q, i === L ? 1 : 0));
        fade(circ, L >= 1 && L <= 3 ? 1 : 0);   // the circles are drawn on levels 1 to 3 only
        cur.setAttribute("d", curve((u) => interp(L, u)));
        prev.setAttribute("d", L ? curve((u) => interp(L - 1, u)) : ""); fade(prev, L ? 0.8 : 0);
        topLab.textContent = L === 0 ? "the two ends, joined by a line" : `the sum of tents through level ${L} (dashed: through level ${L - 1})`;
        botLab.textContent = L === 0 ? "the two end ramps" : `the level-${L} tents, each scaled by its surplus${levels[L].dataset.zoom || ""}`;
      } };
    })();

    // 3. the same sum in two dimensions, from the rectangle run ---------------------------------------------------------
    // Rect heights: parents' average plus surplus, no stored constraints (rect WRITEUP_2026-09-23.md:29-35). The page
    // rebuilds every vertex height from run_hier_vertices.csv (parent0, parent1, surplus) keeping surpluses to depth d,
    // and interpolates bilinearly in each leaf cell of run_mesh.csv (corner heights h00 h10 h01 h11).
    scenes.basis2d = (() => {
      const g = group("basis2d");
      const V = DR.vertices.filter((v) => v.active), byId = new Map(DR.vertices.map((v) => [v.id, v])), byXY = new Map(V.map((v) => [k6(v.x, v.y), v]));
      const dmax = Math.max(0, ...V.filter((v) => v.free && Math.abs(v.surplus) > 0).map((v) => v.depth));
      const heights = (d) => {
        const memo = new Map();
        const h = (v) => {
          if (memo.has(v.id)) return memo.get(v.id);
          const a = byId.get(v.p0), b = byId.get(v.p1);
          const r = (a && b ? (h(a) + h(b)) / 2 : 0) + (v.depth <= d && isFinite(v.surplus) ? v.surplus : 0);
          memo.set(v.id, r); return r;
        };
        return h;
      };
      const cells = [];
      for (let o = 0; o < R.tri.length; o += R.stride) cells.push(Array.from(R.tri.subarray(o, o + 8)));
      // the rebuilt full-depth heights against the engine's cell corners
      const hFull = heights(99);
      let worst = 0;
      for (const c of cells) [[c[0], c[1], c[4]], [c[2], c[1], c[5]], [c[0], c[3], c[6]], [c[2], c[3], c[7]]].forEach(([x, y, h]) => { const v = byXY.get(k6(x, y)); if (v) worst = Math.max(worst, Math.abs(hFull(v) - h)); });
      setV("rebuild", worst < 1e-12 ? "1e-12" : worst.toExponential(0).replace("-", "−"));
      const N = 150, imgs = [];
      for (let d = 0; d <= dmax; ++d) {
        const hd = heights(d), cv = document.createElement("canvas"); cv.width = cv.height = N;
        const ctx = cv.getContext("2d"), img = ctx.createImageData(N, N);
        for (const c of cells) {
          const H = [[c[0], c[1]], [c[2], c[1]], [c[0], c[3]], [c[2], c[3]]].map(([x, y], i) => { const v = byXY.get(k6(x, y)); return v ? hd(v) : c[4 + i]; });
          const i0 = Math.round(c[0] * N), i1 = Math.round(c[2] * N), j0 = Math.round(c[1] * N), j1 = Math.round(c[3] * N);
          for (let j = j0; j < j1; ++j) for (let i = i0; i < i1; ++i) {
            const u = ((i + 0.5) / N - c[0]) / (c[2] - c[0]), w = ((j + 0.5) / N - c[1]) / (c[3] - c[1]);
            const h = (H[0] * (1 - u) + H[1] * u) * (1 - w) + (H[2] * (1 - u) + H[3] * u) * w;
            const [r, gg, b] = ramp(R.baseline + h - BASE), o = 4 * ((N - 1 - j) * N + i);
            img.data[o] = r; img.data[o + 1] = gg; img.data[o + 2] = b; img.data[o + 3] = 255;
          }
        }
        ctx.putImageData(img, 0, 0);
        imgs.push(cv.toDataURL());
      }
      const im = el("image", { x: SQ.x, y: SQ.y, width: SQ.s, height: SQ.s, href: imgs[0], preserveAspectRatio: "none" }, g);
      frame(g);
      let md = "";
      for (const c of cells) md += `M${sx(c[0])},${sy(c[1])}H${sx(c[2])}V${sy(c[3])}H${sx(c[0])}Z`;
      el("path", { d: md, class: "meshline" }, g);
      const F = V.filter((v) => v.free && Math.abs(v.surplus) > 1e-12), smax = Math.max(1e-9, ...F.map((v) => Math.abs(v.surplus)));
      const dots = F.map((v) => ({ d: v.depth, n: el("circle", { cx: sx(v.x), cy: sy(v.y), r: 2 + 6 * Math.sqrt(Math.abs(v.surplus) / smax), class: v.surplus > 0 ? "pos" : "neg", stroke: "var(--sheet)", "stroke-width": 1 }, g) }));
      const lab = text(g, SQ.x, SQ.y - 12, "", "mono", "start");
      text(g, SQ.x, SQ.y + SQ.s + 24, "the rectangle run; dots: surpluses, sized by their magnitude; faint: its final cells", "tiny", "start");
      return { g, update(t) {
        const d = Math.min(dmax, Math.floor(lin(t, 0.04, 0.9) * (dmax + 1)));
        im.setAttribute("href", imgs[d]);
        let k = 0; dots.forEach((q) => { const on = q.d <= d; fade(q.n, on ? 1 : 0); if (on) ++k; });
        lab.textContent = `surpluses of depth 0 to ${d}: ${k} of ${F.length}`;
      } };
    })();

    // the candidate the next three scenes follow: among round 0's candidates at the centres of the starting squares
    // (the midpoints of the 128-face mesh's diagonals, inactive at round 0), the largest |z|
    const sqC = r0.filter((c) => c.seg && Math.abs(Math.abs(c.seg[2] - c.seg[0]) - 0.125) < 1e-9 && Math.abs(Math.abs(c.seg[3] - c.seg[1]) - 0.125) < 1e-9)
      .sort((a, b) => Math.abs(b.z) - Math.abs(a.z))[0] || r0.filter((c) => c.seg).sort((a, b) => Math.abs(b.z) - Math.abs(a.z))[0];
    const C = (() => {
      const c = sqC, p0 = [c.seg[0], c.seg[1]], p1 = [c.seg[2], c.seg[3]], m = [(p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2];
      const e = [p1[0] - m[0], p1[1] - m[1]], f = [-e[1], e[0]], a = [m[0] + f[0], m[1] + f[1]], a2 = [m[0] - f[0], m[1] - f[1]];
      const x0 = Math.min(p0[0], p1[0], a[0], a2[0]), y0 = Math.min(p0[1], p1[1], a[1], a2[1]), s = Math.max(p0[0], p1[0], a[0], a2[0]) - x0;
      const L = (q) => { const h = hT.get(k6(q[0], q[1])); return T.baseline + (h === undefined ? 0 : h); };
      const corners = [p0, p1, a, a2], Lc = corners.map(L);
      const ee = e[0] * e[0] + e[1] * e[1];
      // the pyramid over the two triangles that share the split edge: 1 at m, 0 at the four corners
      const tent = (q) => { const dx = q[0] - m[0], dy = q[1] - m[1]; return Math.max(0, 1 - Math.abs((dx * e[0] + dy * e[1]) / ee) - Math.abs((dx * f[0] + dy * f[1]) / ee)); };
      // barycentric weights on (p0, p1, a) or (p0, p1, a2): the corner tents of the unsplit square
      const bary = (q) => {
        const side = (q[0] - m[0]) * f[0] + (q[1] - m[1]) * f[1] >= 0 ? 2 : 3, A = p0, B = p1, Cc = corners[side];
        const det = (B[0] - A[0]) * (Cc[1] - A[1]) - (Cc[0] - A[0]) * (B[1] - A[1]);
        const l1 = ((q[0] - A[0]) * (Cc[1] - A[1]) - (Cc[0] - A[0]) * (q[1] - A[1])) / det, l2 = ((B[0] - A[0]) * (q[1] - A[1]) - (q[0] - A[0]) * (B[1] - A[1])) / det;
        const w = [0, 0, 0, 0]; w[0] = 1 - l1 - l2; w[1] = l1; w[side] = l2; return w;
      };
      const sites = [];
      for (const i of train) {
        const q = [data.x[i], data.y[i]];
        if (q[0] < x0 || q[0] > x0 + s || q[1] < y0 || q[1] > y0 + s) continue;
        const w = bary(q), logit = w.reduce((acc, wi, j) => acc + wi * Lc[j], 0), p = expit(logit), n = data.n[i];
        sites.push({ q, w, t: tent(q), n, k: data.k[i], p, info: n * p * (1 - p), r: data.k[i] - n * p });
      }
      let S = 0, I = 0; for (const u of sites) { S += u.t * u.r; I += u.t * u.t * u.info; }
      // the part of the tent the four corner tents can make, least squares weighted by np(1 - p) on these sites
      const G = [...Array(4)].map(() => [0, 0, 0, 0]), b = [0, 0, 0, 0];
      for (const u of sites) for (let j = 0; j < 4; ++j) { b[j] += u.info * u.w[j] * u.t; for (let l = 0; l < 4; ++l) G[j][l] += u.info * u.w[j] * u.w[l]; }
      for (let j = 0; j < 4; ++j) G[j][j] += 1e-12;
      const co = solve(G, b), absorbed = co.reduce((acc, v, j) => acc + v * b[j], 0);
      return { c, p0, p1, m, a, a2, x0, y0, s, sites, S, I, absorbed, under: sites.filter((u) => u.t > 0).length };
    })();

    // 4. where the score comes from ---------------------------------------------------------------------------------
    // Engine: per-row score s = (dA/deta) z and weight j = expected information (tri estimator/engine.cpp:305;
    // docs/MODEL.md:61-71); the candidate's raw score is x's (estimator/score.cpp:49); fewer than 10 supporting rows
    // leave it unscored (estimator/engine.cpp:603-605, mesh/vertex.h:11). At v = 0 these reduce to k - np and np(1 - p).
    scenes.score = (() => {
      const g = group("score");
      const Z = { x: 130, y: 96, s: 340 }, zx = (x) => Z.x + ((x - C.x0) / C.s) * Z.s, zy = (y) => Z.y + (1 - (y - C.y0) / C.s) * Z.s;
      text(g, Z.x, Z.y - 14, `one candidate, round 0: the square at (${fmt(C.x0, 3)}, ${fmt(C.y0, 3)})`, "mono", "start");
      el("rect", { x: Z.x, y: Z.y, width: Z.s, height: Z.s, class: "frame" }, g);
      const tents = el("g", {}, g);
      line(tents, zx(C.a[0]), zy(C.a[1]), zx(C.a2[0]), zy(C.a2[1]), "soft", 1);
      el("rect", { x: Z.x + Z.s / 4, y: Z.y + Z.s / 4, width: Z.s / 2, height: Z.s / 2, class: "pencil soft", "stroke-dasharray": "3 4", fill: "none" }, tents);
      text(tents, Z.x + Z.s / 4 + 4, Z.y + Z.s / 4 + 13, "tent = ½", "tiny", "start");
      line(g, zx(C.p0[0]), zy(C.p0[1]), zx(C.p1[0]), zy(C.p1[1]), "accent", 1.6);
      const TW = Math.max(1e-9, ...C.sites.map((u) => u.t * u.info));
      const dots = C.sites.filter((u) => u.t > 0).sort((a, b) => a.t * a.info - b.t * b.info)
        .map((u) => el("circle", { cx: zx(u.q[0]), cy: zy(u.q[1]), r: 1.5 + 8.5 * Math.sqrt((u.t * u.info) / TW), class: u.r > 0 ? "pos" : "neg", "fill-opacity": 0.75 }, g));
      el("circle", { cx: zx(C.m[0]), cy: zy(C.m[1]), r: 5, class: "ring" }, g);
      [C.p0, C.p1].forEach((p) => el("circle", { cx: zx(p[0]), cy: zy(p[1]), r: 4, class: "ink" }, g));
      // where the square sits
      const ins = el("g", {}, g), IN = { x: 490, y: 96, s: 80 };
      el("rect", { x: IN.x, y: IN.y, width: IN.s, height: IN.s, class: "frame" }, ins);
      el("rect", { x: IN.x + C.x0 * IN.s, y: IN.y + (1 - C.y0 - C.s) * IN.s, width: C.s * IN.s, height: C.s * IN.s, class: "pos" }, ins);
      text(ins, IN.x, IN.y + IN.s + 14, "the unit square", "tiny", "start");
      const nums = el("g", {}, g), y0 = Z.y + Z.s + 30;
      text(nums, Z.x, y0, `${C.under} training sites; dot area: tent × np(1 − p); blue: k > np`, "tiny", "start");
      text(nums, Z.x, y0 + 26, `S = Σ tent · (k − np) = ${fmt(C.S)}`, "num", "start");
      text(nums, Z.x, y0 + 46, `Σ tent² · np(1 − p) = ${fmt(C.I)}`, "num", "start");
      text(nums, Z.x, y0 + 70, `the engine's round-0 z for this candidate: ${fmt(C.c.z)}`, "tiny", "start");
      return { g, update(t) {
        fade(tents, seg(t, 0, 0.15));
        const n = Math.floor(seg(t, 0.05, 0.55) * dots.length);
        dots.forEach((d, i) => fade(d, i < n ? 1 : 0));
        fade(ins, seg(t, 0, 0.15)); fade(nums, seg(t, 0.55, 0.7));
      } };
    })();

    // 5. the first-order refit: the projection off the parents' span (a sketch, with numbers from the run) ------------
    // J = P + X'diag(j)X, c = X'diag(j)x, a = J^-1 c, h = x'diag(j)x - c'a (docs/MODEL.md:131-137;
    // estimator/score.cpp:36-41).
    scenes.refit = (() => {
      const g = group("refit");
      const sk = el("g", {}, g);
      text(sk, 70, 64, "a sketch: the candidate's column and the span of the columns already in the fit", "tiny", "start");
      const O = [150, 262], Xp = [392, 96], Fp = [392, 236];
      el("path", { d: poly([[80, 282], [420, 282], [520, 214], [180, 214]]) + "Z", class: "shadeink" }, sk);
      el("path", { d: poly([[80, 282], [420, 282], [520, 214], [180, 214]]) + "Z", class: "pencil soft", "stroke-width": 1, fill: "none" }, sk);
      text(sk, 96, 304, "the columns already in the fit: every surface it can make now", "tiny", "start");
      const vx = stroke(sk, poly([O, Xp]), "", 2), vxa = stroke(sk, poly([O, Fp]), "warm", 2), vr = stroke(sk, poly([Fp, Xp]), "accent", 2.6);
      vxa.a.setAttribute("stroke-dasharray", "6 4");
      const marks = el("g", {}, sk);
      el("path", { d: poly([[Fp[0] - 12, Fp[1]], [Fp[0] - 12, Fp[1] - 12], [Fp[0], Fp[1] - 12]]), class: "pencil", "stroke-width": 1, fill: "none" }, marks);
      el("circle", { cx: O[0], cy: O[1], r: 3, class: "ink" }, marks);
      const l1 = text(sk, (O[0] + Xp[0]) / 2 - 16, (O[1] + Xp[1]) / 2 - 6, "x, the candidate's tent", "label", "end");
      const l2 = text(sk, (O[0] + Fp[0]) / 2 + 20, (O[1] + Fp[1]) / 2 + 22, "Xa, with a = J⁻¹c", "label warmt", "middle");
      const l3 = text(sk, Xp[0] + 10, (Fp[1] + Xp[1]) / 2, "x − Xa", "label acc", "start");
      const l4 = text(sk, Xp[0] + 10, (Fp[1] + Xp[1]) / 2 + 18, "h = |x − Xa|²", "tiny", "start");
      // the numbers: the square from the last step
      const B = el("g", {}, g), bx = 80, bw = 440, by = 390;
      text(B, bx, by - 34, "the square from the last step, weights np(1 − p), its sites only", "mono", "start");
      const rows = [
        ["the tent's weighted squared length", C.I, "shadeink", ""],
        ["what the four corner tents can make", C.absorbed, "shadew", "warmt"],
        ["left for the candidate", C.I - C.absorbed, "shade", "acc"],
      ];
      const bars = rows.map(([lab, v, cls, tc], i) => {
        const q = el("g", {}, B), y = by + i * 52;
        el("rect", { x: bx, y, width: Math.max(1, (bw * v) / Math.max(C.I, 1e-9)), height: 22, class: cls }, q);
        text(q, bx + 6, y + 15, `${lab}: ${fmt(v)}${i ? ` (${pct(v / Math.max(C.I, 1e-9))})` : ""}`, "tiny " + tc, "start");
        return q;
      });
      setV("absorb", pct(C.absorbed / Math.max(C.I, 1e-9)));
      return { g, update(t) {
        vx.set(seg(t, 0.02, 0.2)); fade(l1, seg(t, 0.12, 0.22)); vxa.set(seg(t, 0.22, 0.38)); fade(l2, seg(t, 0.32, 0.4));
        vr.set(seg(t, 0.4, 0.52)); fade(l3, seg(t, 0.48, 0.56)); fade(l4, seg(t, 0.5, 0.58)); fade(marks, seg(t, 0.4, 0.5));
        bars.forEach((b, i) => fade(b, seg(t, 0.6 + i * 0.08, 0.7 + i * 0.08)));
      } };
    })();

    // 6. the shrinkage-bias correction and the z (a sketch, then the engine's numbers) -----------------------------------
    // z = (S - c'J^-1 P theta) / sqrt(V_cand) and sd = sqrt(V_cand) / h (tri estimator/engine.cpp:628-629, the bias
    // c'J^-1 P theta from estimator/score.cpp:22-24 and :50); V_cand = x'diag(V)x - 2a'X'diag(V)x + a'Omega a
    // (score.cpp:42-48, docs/MODEL.md:140-145); the proposed step S / (h + lambda) (engine.cpp:620; MODEL.md:150-154).
    // The sketch: sites on a line whose truth is straight; two coefficients, the left end free and the right end's
    // surplus under a prior of precision lambda; a candidate tent at the middle. Everything is computed here.
    scenes.bias = (() => {
      const g = group("bias");
      const n = 40, lam = 6, U = [...Array(n)].map((_, i) => (i + 0.5) / n), yT = U.map((u) => 0.15 + 0.9 * u);
      const Xc = U.map((u) => [1 - u, u]), tent = U.map((u) => 1 - Math.abs(2 * u - 1));
      const J = [[0, 0], [0, lam]], Xy = [0, 0], c = [0, 0];
      U.forEach((_, i) => { for (let a = 0; a < 2; ++a) { Xy[a] += Xc[i][a] * yT[i]; c[a] += Xc[i][a] * tent[i]; for (let b = 0; b < 2; ++b) J[a][b] += Xc[i][a] * Xc[i][b]; } });
      const th = solve(J, Xy), fit = U.map((_, i) => Xc[i][0] * th[0] + Xc[i][1] * th[1]), res = yT.map((y, i) => y - fit[i]);
      const S = tent.reduce((s, t, i) => s + t * res[i], 0), Jc = solve(J, c), bias = Jc[1] * lam * th[1];
      const X = (u) => 90 + u * 420, Y = (v) => 320 - v * 200, YT = (v) => 410 - v * 50;
      const sk = el("g", {}, g);
      text(sk, 70, 64, "a sketch: the truth is a straight line, so no new vertex is needed", "tiny", "start");
      line(sk, 80, 86, 104, 86, "soft", 1.4); text(sk, 110, 90, "the truth", "tiny", "start");
      line(sk, 200, 86, 224, 86, "", 2); text(sk, 230, 90, "the fit, with the right end's surplus pulled toward zero", "tiny", "start");
      el("path", { d: poly(U.map((u, i) => [X(u), YT(tent[i])])) + ` L${X(1)},${YT(0)} L${X(0)},${YT(0)} Z`, class: "shade" }, sk);
      line(sk, 90, YT(0), 510, YT(0), "soft", 1);
      text(sk, X(0.5), YT(1) - 6, "the candidate's tent", "tiny acc", "middle");
      el("path", { d: poly([[X(0), Y(0.15)], [X(1), Y(1.05)]]), class: "pencil soft", "stroke-width": 1.4, fill: "none" }, sk);
      el("path", { d: poly([[X(0), Y(th[0])], [X(1), Y(th[1])]]), class: "pencil", "stroke-width": 2, fill: "none" }, sk);
      const resid = U.map((u, i) => line(sk, X(u), Y(fit[i]), X(u), Y(yT[i]), res[i] > 0 ? "accent" : "warm", 2));
      U.forEach((u, i) => el("circle", { cx: X(u), cy: Y(yT[i]), r: 2.6, class: "ink" }, sk));
      const sn = text(sk, 300, 438, `score ${fmt(S)}; its shrinkage part by the engine's formula ${fmt(bias)}; left ${fmt(S - bias)}`, "tiny");
      const fo = el("g", {}, g);
      text(fo, 300, 486, "z = (S − c′J⁻¹Pθ) / √V", "formula");
      text(fo, 80, 518, `the square from the last two steps, round 0: z = ${fmt(C.c.z)}`, "num", "start");
      text(fo, 80, 540, `proposed surplus S / (h + λ) = ${fmt(C.c.beta)} logits`, "num", "start");
      text(fo, 80, 562, `its standard deviation √V / h = ${fmt(C.c.sd)}`, "num", "start");
      return { g, update(t) {
        resid.forEach((r, i) => fade(r, seg(t, 0.1 + (0.3 * i) / n, 0.14 + (0.3 * i) / n)));
        fade(sn, seg(t, 0.4, 0.5)); fade(fo, seg(t, 0.55, 0.7));
      } };
    })();

    // 7. Lindsey's method on round 0's scores -----------------------------------------------------------------------------
    // Engine: fit/gate.cpp:106-254 via fit/refine.cpp:443-447 (50 or more candidates); its result is the calibration
    // line, fit/refine.cpp:380-398.
    const LN = lindsey(r0.map((c) => c.z), 6);
    scenes.lindsey = (() => {
      const g = group("lindsey");
      if (!LN) { text(g, 300, 300, "the null could not be refitted on this page for this round", "mono"); return { g, update() {} }; }
      const W = Math.min(9, Math.max(4, Math.ceil(Math.max(...r0.map((c) => Math.abs(c.z)))))), M = LN.M, d = LN.d;
      const A = { x: 80, y: 90, w: 440, h: 220 }, X = (z) => A.x + ((z + W) / (2 * W)) * A.w;
      const peak = 1.08 * Math.max(...LN.counts, LN.dens(0) * M * d, (LN.pi0 * phi(0) * M * d) / LN.s0);
      const Y = (v) => A.y + A.h - (v / peak) * A.h;
      text(g, A.x, A.y - 18, `round 0: ${M} scores in 72 bins`, "mono", "start");
      line(g, A.x, A.y + A.h, A.x + A.w, A.y + A.h, "", 1);
      for (const z of [-8, -4, -2, 0, 2, 4, 8]) if (Math.abs(z) <= W) text(g, X(z), A.y + A.h + 15, fmt(z, 0), "tiny");
      const bars = LN.mids.map((m, b) => Math.abs(m) <= W && LN.counts[b] > 0 ? el("rect", { x: X(m - d / 2) + 0.5, y: Y(LN.counts[b]), width: Math.max(1, (A.w * d) / (2 * W) - 1), height: A.y + A.h - Y(LN.counts[b]), class: "ink", opacity: 0.22 }, g) : null).filter(Boolean);
      const cur = (fn) => { const p = []; for (let z = -W; z <= W + 1e-9; z += W / 120) p.push([X(z), Y(fn(z))]); return poly(p); };
      const fh = stroke(g, cur((z) => LN.dens(z) * M * d), "", 1.8);
      const tb = el("path", { d: cur((z) => phi(z) * M * d), class: "pencil soft", "stroke-width": 1.3, fill: "none", "stroke-dasharray": "4 4" }, g);
      const nl = stroke(g, cur((z) => (LN.pi0 * phi((z - LN.d0) / LN.s0) * M * d) / LN.s0), "accent", 2.2);
      const leg = el("g", {}, g);
      line(leg, 330, 104, 356, 104, "", 1.8); text(leg, 362, 108, "fitted density", "tiny", "start");
      line(leg, 330, 122, 356, 122, "accent", 2.2); text(leg, 362, 126, "π₀ × null", "tiny", "start");
      line(leg, 330, 140, 356, 140, "soft", 1.3).setAttribute("stroke-dasharray", "4 4"); text(leg, 362, 144, "N(0, 1)", "tiny", "start");
      // the log-density near zero and the central parabola
      const Bx = { x: 80, y: 372, w: 440, h: 150 }, WB = 3.5, XB = (z) => Bx.x + ((z + WB) / (2 * WB)) * Bx.w;
      const lf = LN.mids.map((m, b) => [m, Math.log(LN.fbins[b])]).filter(([m]) => Math.abs(m) <= WB);
      const par = (z) => LN.a0 + LN.a1 * z + LN.a2 * z * z;
      const lmin = Math.min(...lf.map((q) => q[1]), par(-WB), par(WB)), lmax = Math.max(...lf.map((q) => q[1]), par(0)) + 0.1;
      const YB = (v) => Bx.y + Bx.h - ((v - lmin) / (lmax - lmin)) * Bx.h;
      const low = el("g", {}, g);
      text(low, Bx.x, Bx.y - 14, "log density near zero", "mono", "start");
      el("rect", { x: XB(-2), y: Bx.y, width: XB(2) - XB(-2), height: Bx.h, class: "shade" }, low);
      text(low, XB(0), Bx.y + Bx.h - 6, "|z| ≤ 2", "tiny acc", "middle");
      line(low, Bx.x, Bx.y + Bx.h, Bx.x + Bx.w, Bx.y + Bx.h, "", 1);
      for (const z of [-3, -2, -1, 0, 1, 2, 3]) text(low, XB(z), Bx.y + Bx.h + 15, fmt(z, 0), "tiny");
      lf.forEach(([m, v]) => el("circle", { cx: XB(m), cy: YB(v), r: 2.6, class: "ink" }, low));
      const pp = []; for (let z = -WB; z <= WB + 1e-9; z += 0.05) pp.push([XB(z), YB(par(z))]);
      const pa = stroke(low, poly(pp), "warm", 2);
      text(low, Bx.x + Bx.w, Bx.y - 14, "the parabola fitted over |z| ≤ 2", "tiny warmt", "end");
      const cap = text(g, 300, 580, `on this page: centre ${fmt(LN.d0, 3)}, spread ${fmt(LN.s0, 3)}, π₀ ${fmt(LN.pi0, 3)}`, "tiny");
      setV("lnull", `centre ${fmt(LN.d0, 3)}, spread ${fmt(LN.s0, 3)}, π₀ ${fmt(LN.pi0, 3)}`);
      setV("enull", cal0 ? `${fmt(cal0.nullMean, 3)}, ${fmt(cal0.nullSd, 3)}, ${fmt(cal0.pi0, 3)}` : "not reported");
      return { g, update(t) {
        const n = Math.floor(seg(t, 0, 0.25) * bars.length); bars.forEach((b, i) => fade(b, i < n ? 0.22 : 0));
        fh.set(seg(t, 0.2, 0.4)); fade(leg, seg(t, 0.2, 0.3)); fade(tb, seg(t, 0.25, 0.35));
        fade(low, seg(t, 0.4, 0.5)); pa.set(seg(t, 0.45, 0.62)); nl.set(seg(t, 0.62, 0.8)); fade(cap, seg(t, 0.7, 0.8));
      } };
    })();

    // 8. lfdr running mean, BH and |z| > 1.96 on the same scores --------------------------------------------------------
    // lfdr = pi0 f0(z) / f(z) (tri fit/gate.cpp:245-252); sorted by lfdr, the longest prefix whose mean is at most
    // q = 0.10 (fit/refine.cpp:41, :470-480). BH on two-sided N(0, 1) p-values at q (fit/refine.cpp:424-440), used by
    // the engine below 50 candidates or when the central fit is invalid (:443-465).
    scenes.select = (() => {
      const g = group("select");
      const M = r0.length, byL = r0.map((c, i) => ({ c, i })).sort((a, b) => a.c.lfdr - b.c.lfdr || a.i - b.i);
      let cum = 0, kk = 0;
      for (let i = 0; i < M; ++i) { cum += byL[i].c.lfdr; if (cum / (i + 1) <= 0.1) kk = i + 1; else break; }
      const selL = new Set(byL.slice(0, kk).map((q) => q.c));
      const pv = r0.map((c, i) => ({ c, i, p: erfc(Math.abs(c.z) / Math.SQRT2) })).sort((a, b) => a.p - b.p || a.i - b.i);
      let kbh = 0; pv.forEach((q, i) => { if (q.p <= (0.1 * (i + 1)) / M) kbh = i + 1; });
      const selB = new Set(pv.slice(0, kbh).map((q) => q.c)), selN = new Set(r0.filter((c) => Math.abs(c.z) > 1.96));
      const engineK = r0.filter((c) => c.selected).length;
      if (engineK !== kk) console.warn("gate-deep: page running-mean prefix", kk, "engine", engineK);
      const W = Math.min(12, Math.max(4, Math.ceil(Math.max(...r0.map((c) => Math.abs(c.z)))))), X = (z) => 80 + ((clamp(z, -W, W) + W) / (2 * W)) * 440;
      const rows = [["lfdr, running mean at most 10%", engineK, (c) => c.selected], ["Benjamini–Hochberg at 10%, null N(0, 1)", kbh, (c) => selB.has(c)], ["|z| > 1.96", selN.size, (c) => selN.has(c)]];
      const panels = rows.map(([lab, k, sel], j) => {
        const q = el("g", {}, g), y = 150 + j * 130;
        text(q, 80, y - 34, `${lab}: ${k} of ${M}`, "mono", "start");
        line(q, 80, y + 12, 520, y + 12, "soft", 1);
        const ord = [...r0].sort((a, b) => (sel(a) ? 1 : 0) - (sel(b) ? 1 : 0));
        ord.forEach((c) => { const on = sel(c); el("line", { x1: X(c.z), x2: X(c.z), y1: y - 14, y2: y + 10, class: "pencil " + (on ? (c.z > 0 ? "accent" : "warm") : "soft"), "stroke-width": on ? 1.6 : 1, opacity: on ? 0.95 : 0.5 }, q); });
        if (j === 2) [-1.96, 1.96].forEach((z) => { line(q, X(z), y - 22, X(z), y + 18, "", 1).setAttribute("stroke-dasharray", "3 3"); });
        for (const z of [-W, -4, 0, 4, W]) text(q, X(z), y + 28, fmt(z, 0), "tiny");
        return q;
      });
      const cap = text(g, 300, 552, "one tick per round-0 candidate at its z; coloured: taken by the rule", "tiny");
      setV("nLfdr", String(engineK)); setV("nBH", String(kbh)); setV("nNaive", String(selN.size));
      return { g, update(t) { panels.forEach((p, i) => fade(p, seg(t, 0.05 + i * 0.2, 0.2 + i * 0.2))); fade(cap, seg(t, 0.5, 0.65)); } };
    })();

    // 9. admission with re-scoring (a sketch) ---------------------------------------------------------------------------
    // Engine: selected candidates are admitted best first; each is re-profiled at the live state and committed at its
    // live optimum (tri fit/refine.cpp:296-352, candidate_gain at :318). Here: residuals with one bump, two overlapping
    // tents of the same level, A admitted first at S_A / (h_A + lambda), then B scored again. All computed here.
    scenes.rescore = (() => {
      const g = group("rescore");
      const n = 64, lam = 0.5, U = [...Array(n)].map((_, i) => (i + 0.5) / n), res = U.map((u) => 0.8 * Math.exp(-(((u - 0.47) / 0.09) ** 2)));
      const hat = (c, hw) => U.map((u) => Math.max(0, 1 - Math.abs(u - c) / hw)), tA = hat(0.4375, 0.125), tB = hat(0.5625, 0.125);
      const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
      const SA = dot(tA, res), SB = dot(tB, res), bA = SA / (dot(tA, tA) + lam), res2 = res.map((r, i) => r - bA * tA[i]), SB2 = dot(tB, res2);
      const X = (u) => 80 + u * 440, Y = (v) => 300 - v * 220, YT = (v) => 360 - v * 60;
      text(g, 70, 64, "a sketch: one bump in the residuals, two overlapping tents", "tiny", "start");
      line(g, 80, Y(0), 520, Y(0), "soft", 1);
      const mk = (t, cls) => el("path", { d: poly(U.map((u, i) => [X(u), YT(t[i])])) + ` L${X(1)},${YT(0)} L${X(0)},${YT(0)} Z`, class: cls }, g);
      mk(tA, "shade"); mk(tB, "shadew");
      text(g, X(0.4375), YT(1) - 6, "A", "label acc"); text(g, X(0.5625), YT(1) - 6, "B", "label warmt");
      line(g, 80, YT(0), 520, YT(0), "", 1);
      const dots = U.map((u, i) => el("circle", { cx: X(u), cy: Y(res[i]), r: 2.8, class: "ink" }, g));
      const fitA = el("path", { d: poly(U.map((u, i) => [X(u), Y(bA * tA[i])])), class: "pencil accent", "stroke-width": 2, fill: "none" }, g);
      const fl = text(g, X(0.3), Y(bA) - 10, "A admitted", "tiny acc", "end");
      text(g, X(0), Y(0.9), "residuals", "tiny", "start");
      // the two scores as bars
      const B0 = { x: 80, y: 430, w: 400 }, sc = B0.w / Math.max(SA, SB, 1e-9);
      const bars = el("g", {}, g);
      text(bars, B0.x, B0.y - 12, "scores", "mono", "start");
      el("rect", { x: B0.x + 20, y: B0.y, width: SA * sc, height: 20, class: "shade" }, bars); text(bars, B0.x, B0.y + 15, "A", "label acc", "start");
      text(bars, B0.x + 26 + SA * sc, B0.y + 15, fmt(SA), "tiny", "start");
      const bB = el("rect", { x: B0.x + 20, y: B0.y + 34, width: SB * sc, height: 20, class: "shadew" }, bars); text(bars, B0.x, B0.y + 49, "B", "label warmt", "start");
      const bBt = text(bars, B0.x + 26 + SB * sc, B0.y + 49, fmt(SB), "tiny", "start");
      setV("skB", pct(SB2 / Math.max(SB, 1e-9)));
      return { g, update(t) {
        const u = seg(t, 0.4, 0.65);
        dots.forEach((d, i) => d.setAttribute("cy", Y(res[i] - u * bA * tA[i])));
        fade(fitA, seg(t, 0.3, 0.4) * (1 - seg(t, 0.65, 0.75))); fade(fl, seg(t, 0.3, 0.4) * (1 - seg(t, 0.65, 0.75)));
        const sb = SB + u * (SB2 - SB);
        bB.setAttribute("width", Math.max(0, sb * sc)); bBt.setAttribute("x", B0.x + 26 + Math.max(0, sb * sc)); bBt.textContent = u < 1 ? fmt(sb) : `${fmt(SB2)}, scored again after A`;
        fade(bars, seg(t, 0.05, 0.2));
      } };
    })();

    // 10. round 0's selections in the run, their tents and their overlaps -------------------------------------------------
    // Order: lfdr ascending, ties by index (tri fit/refine.cpp:470-473). The census: fit/refine.cpp:577-578.
    scenes.overlap = (() => {
      const g = group("overlap");
      const sel = r0.map((c, i) => ({ c, i })).filter((q) => q.c.selected && q.c.seg).sort((a, b) => a.c.lfdr - b.c.lfdr || a.i - b.i).map((q) => q.c);
      frame(g, `round 0: ${sel.length} selected, in admission order`);
      const cg = clipSquare(g);
      const dia = sel.map((c) => {
        const p0 = [c.seg[0], c.seg[1]], p1 = [c.seg[2], c.seg[3]], m = [(p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2], e = [p1[0] - m[0], p1[1] - m[1]], f = [-e[1], e[0]];
        return { c, m, e, f, pts: [p0, [m[0] + f[0], m[1] + f[1]], p1, [m[0] - f[0], m[1] - f[1]]] };
      });
      // overlapping pairs: tents sharing any point strictly inside both, on a 240 x 240 grid
      const G = 240, cover = new Map(), pairs = new Set();
      dia.forEach((d, j) => {
        const ee = d.e[0] ** 2 + d.e[1] ** 2, xs = d.pts.map((p) => p[0]), ys = d.pts.map((p) => p[1]);
        for (let a = Math.max(0, Math.floor(Math.min(...xs) * G)); a < Math.min(G, Math.ceil(Math.max(...xs) * G)); ++a)
          for (let b = Math.max(0, Math.floor(Math.min(...ys) * G)); b < Math.min(G, Math.ceil(Math.max(...ys) * G)); ++b) {
            const dx = (a + 0.5) / G - d.m[0], dy = (b + 0.5) / G - d.m[1];
            if (Math.abs((dx * d.e[0] + dy * d.e[1]) / ee) + Math.abs((dx * d.f[0] + dy * d.f[1]) / ee) >= 1 - 1e-9) continue;
            const key = a * G + b, at = cover.get(key) || [];
            for (const o of at) pairs.add(o + "," + j);
            at.push(j); cover.set(key, at);
          }
      });
      const shapes = dia.map((d) => {
        const q = el("g", {}, cg), P = d.pts.map((p) => [sx(p[0]), sy(p[1])]);
        el("path", { d: poly(P) + "Z", class: d.c.z > 0 ? "shade" : "shadew" }, q);
        el("path", { d: poly(P) + "Z", class: "pencil " + (d.c.z > 0 ? "accent" : "warm"), "stroke-width": 1.2, fill: "none" }, q);
        return q;
      });
      const marks = dia.map((d, i) => {
        const q = el("g", {}, g), X = sx(d.m[0]), Y = sy(d.m[1]);
        if (tAdm(d.c)) el("circle", { cx: X, cy: Y, r: 3, class: "ink" }, q);
        else { line(q, X - 5, Y - 5, X + 5, Y + 5, "", 2); line(q, X - 5, Y + 5, X + 5, Y - 5, "", 2); }
        text(q, X > 560 ? X - 5 : X + 5, Y - 5, String(i + 1), "tiny", X > 560 ? "end" : "start");
        return q;
      });
      const na = sel.filter(tAdm).length;
      const cap = text(g, SQ.x, SQ.y + SQ.s + 24, `${na} of ${sel.length} admitted in round 0 (dots)${na < sel.length ? `; crosses: not admitted` : ""}`, "tiny", "start");
      setV("sel0", String(sel.length));
      setV("ovl", pairs.size ? `${pairs.size} pairs of these tents overlap.` : "None of these tents overlap.");
      setV("reprof", DT.census ? String(DT.census.reprofiled) : "some");
      setV("rejected", DT.census ? String(DT.census.rejected) : "none");
      return { g, update(t) {
        const n = Math.floor(seg(t, 0.05, 0.75) * dia.length);
        shapes.forEach((s, i) => fade(s, i < n ? 1 : 0)); marks.forEach((m, i) => fade(m, i < n ? 1 : 0)); fade(cap, seg(t, 0.75, 0.85));
      } };
    })();

    // 11. the coin variance by round, and the depth variances with the admitted surpluses ------------------------------------
    // Pool variance: re-estimated with the refit at each round start (tri fit/refine.cpp:407; docs/MODEL.md:100-113).
    // Depth variance by EM: mean of surplus^2 + posterior variance over a depth's admitted vertices, at least 3, floored at
    // 0.005^2, nearest-depth borrowing, default 1 (estimator/engine.cpp:38-52; mesh/mesh.cpp:118-173, :25-29;
    // mesh/mesh.h:48-49). Error bars: sigma_pooled, 1 / (h + lambda) at admission (fit/refine.cpp:329).
    scenes.variance = (() => {
      const g = group("variance");
      const pv = T.rounds.map((r) => r.poolVariance).concat([T.poolVariance]), truth = POOL_SD * POOL_SD;
      const A = { x: 90, y: 100, w: 420, h: 150 }, vmax = Math.max(...pv, truth, 1e-6) * 1.12;
      const AX = (i) => A.x + (i / Math.max(1, pv.length - 1)) * A.w, AY = (v) => A.y + A.h - (v / vmax) * A.h;
      const a = el("g", {}, g);
      text(a, A.x - 10, A.y - 24, "coin variance v, by round", "mono", "start");
      line(a, A.x, A.y + A.h, A.x + A.w, A.y + A.h, "", 1);
      const tl = line(a, A.x, AY(truth), A.x + A.w, AY(truth), "warm", 1.4); tl.setAttribute("stroke-dasharray", "5 4");
      text(a, A.x + A.w, AY(truth) - 6, `the true ${POOL_SD.toFixed(2)}² = ${fmt(truth, 3)}`, "tiny warmt", "end");
      const pl = stroke(a, poly(pv.map((v, i) => [AX(i), AY(v)])), "accent", 2.2);
      const pd = pv.map((v, i) => el("circle", { cx: AX(i), cy: AY(v), r: 3.6, class: "pos" }, a));
      pv.forEach((v, i) => text(a, AX(i), A.y + A.h + 15, i < pv.length - 1 ? String(T.rounds[i].round) : "final", "tiny"));
      text(a, A.x - 6, AY(pv[0]) + 4, fmt(pv[0], 3), "tiny", "end");
      text(a, A.x + A.w + 6, AY(pv[pv.length - 1]) + 4, fmt(pv[pv.length - 1], 3), "tiny", "start");
      const V = DT.vertices.filter((v) => v.active && v.free && v.admitted && v.lambda > 0);
      const depths = [...new Set(V.map((v) => v.depth))].sort((p, q) => p - q);
      const Bx = { x: 90, y: 340, w: 420, h: 190 };
      const smax = Math.max(0.3, ...V.map((v) => Math.abs(v.surplus) + (v.postVar < 1e10 ? Math.sqrt(v.postVar) : 0)), ...V.map((v) => 1 / Math.sqrt(v.lambda))) * 1.08;
      const BX = (i) => Bx.x + ((i + 0.5) / Math.max(1, depths.length)) * Bx.w, BY = (s) => Bx.y + Bx.h / 2 - (s / smax) * (Bx.h / 2);
      const b = el("g", {}, g);
      text(b, Bx.x - 10, Bx.y - 24, "admitted surpluses by depth; shaded: ± τ at admission", "mono", "start");
      line(b, Bx.x, BY(0), Bx.x + Bx.w, BY(0), "soft", 1);
      const bands = [], dots = [];
      depths.forEach((d, i) => {
        const at = V.filter((v) => v.depth === d), lam = at.map((v) => v.lambda).sort((p, q) => p - q)[at.length >> 1], tau = 1 / Math.sqrt(lam);
        bands.push(el("rect", { x: BX(i) - 26, y: BY(tau), width: 52, height: BY(-tau) - BY(tau), class: "shade" }, b));
        at.forEach((v, j) => {
          const q = el("g", {}, b), X = BX(i) + ((j % 7) - 3) * 6;
          if (v.postVar < 1e10) line(q, X, BY(v.surplus - Math.sqrt(v.postVar)), X, BY(v.surplus + Math.sqrt(v.postVar)), v.surplus > 0 ? "accent" : "warm", 1);
          el("circle", { cx: X, cy: BY(v.surplus), r: 3, class: v.surplus > 0 ? "pos" : "neg" }, q);
          dots.push(q);
        });
        text(b, BX(i), Bx.y + Bx.h + 14, `depth ${d}`, "tiny");
        text(b, BX(i), Bx.y + Bx.h + 27, `τ ${fmt(tau)}`, "tiny");
        if (at.length < 3) text(b, BX(i), Bx.y + Bx.h + 40, "borrowed", "tiny warmt");
      });
      const cap = text(g, 300, 590, "bars: ± one posterior standard deviation at admission", "tiny");
      setV("truev", `${POOL_SD.toFixed(2)}² = ${fmt(truth, 3)}`);
      return { g, update(t) {
        fade(a, seg(t, 0, 0.1)); pl.set(seg(t, 0.05, 0.4)); pd.forEach((d, i) => fade(d, seg(t, 0.05 + (0.35 * i) / pd.length, 0.1 + (0.35 * i) / pd.length)));
        fade(b, seg(t, 0.4, 0.5)); dots.forEach((d) => fade(d, seg(t, 0.45, 0.6))); bands.forEach((d) => fade(d, seg(t, 0.6, 0.72))); fade(cap, seg(t, 0.6, 0.72));
      } };
    })();

    // 12. stopping, retirement and the final refit ----------------------------------------------------------------------
    // Face budget max(600, N / 24) with N every row (tri fit/refine.cpp:39-40, :520, :556); stop checks at :406 (faces),
    // :417 (nothing scoreable), :482-486 (three rounds selecting nothing); refits at :407, :485, :492, :559; retirement
    // :157-230, :566; final refit :568 (docs/MODEL.md:115-118).
    scenes.stop = (() => {
      const g = group("stop");
      const cap = Math.max(600, Math.floor(data.x.length / 24));
      const rs = T.rounds.map((r) => {
        const c = DT.cands.filter((q) => q.round === r.round), cl = DT.calib.find((q) => q.round === r.round);
        return { round: r.round, faces: r.faces, M: cl ? cl.M : c.length, sel: c.filter((q) => q.selected).length, adm: c.filter(tAdm).length };
      });
      let st = 0; rs.forEach((r) => { st = r.sel === 0 ? st + 1 : 0; r.stalls = st; });
      const nF = T.tri.length / T.stride;
      const cols = rs.length, x0 = 200, cw = Math.min(64, 300 / Math.max(1, cols));
      const A = { y: 90, h: 150 }, FY = (f) => A.y + A.h - (f / (cap * 1.05)) * A.h;
      const top = el("g", {}, g);
      text(top, 70, A.y - 20, `faces at the start of each round, and the budget of ${cap}`, "mono", "start");
      line(top, x0 - 10, A.y + A.h, x0 + cols * cw + 60, A.y + A.h, "", 1);
      const cl = line(top, x0 - 10, FY(cap), x0 + cols * cw + 60, FY(cap), "warm", 1.4); cl.setAttribute("stroke-dasharray", "6 4");
      text(top, x0 + cols * cw + 60, FY(cap) - 6, `budget ${cap}`, "tiny warmt", "end");
      rs.forEach((r, i) => { el("rect", { x: x0 + i * cw + 6, y: FY(r.faces), width: cw - 12, height: A.y + A.h - FY(r.faces), class: "shade" }, top); text(top, x0 + i * cw + cw / 2, FY(r.faces) - 5, String(r.faces), "tiny"); });
      el("rect", { x: x0 + cols * cw + 6, y: FY(nF), width: 36, height: A.y + A.h - FY(nF), class: "shadeink" }, top);
      text(top, x0 + cols * cw + 24, FY(nF) - 5, String(nF), "tiny");
      text(top, x0 + cols * cw + 24, A.y + A.h + 15, "end", "tiny");
      const tab = el("g", {}, g), ty = 290;
      [["round", "round"], ["scored", "M"], ["selected", "sel"], ["admitted", "adm"], ["no selection, in a row", "stalls"]].forEach(([lab, k], j) => {
        const y = ty + j * 24;
        text(tab, x0 - 16, y, lab, "tiny", "end");
        rs.forEach((r, i) => text(tab, x0 + i * cw + cw / 2, y, String(r[k]), "tiny" + (k === "stalls" && r[k] === 3 ? " warmt" : k === "adm" && r[k] ? " acc" : "")));
      });
      const retired = (() => { for (const l of T.log || []) { const m = /\[retirement\] retired (\d+) coefficients/.exec(l); if (m) return +m[1]; } return null; })();
      const flow = el("g", {}, g), fy = 470, boxes = ["refit", `retire: ${retired === null ? "?" : retired}`, "final refit"];
      boxes.forEach((s, i) => {
        const bx = 90 + i * 150;
        el("rect", { x: bx, y: fy, width: 118, height: 34, rx: 4, class: "frame" }, flow);
        text(flow, bx + 59, fy + 22, s, "tiny");
        if (i < boxes.length - 1) line(flow, bx + 122, fy + 17, bx + 146, fy + 17, "", 1.2);
      });
      text(flow, 90, fy + 64, `final coin variance ${fmt(T.poolVariance, 3)}; held-out mean deviance ${T.heldout ? T.heldout.deviance.toFixed(3) : "?"} per site`, "tiny", "start");
      setV("cap", String(cap)); setV("retired", retired === null ? "none reported" : String(retired)); setV("nr", String(rs.length));
      return { g, update(t) { fade(top, seg(t, 0, 0.2)); fade(tab, seg(t, 0.2, 0.4)); fade(flow, seg(t, 0.5, 0.7)); } };
    })();

    // 13. the rectangle engine: bilinear cells, either axis, hanging vertices ---------------------------------------------
    // 8 x 8 start, corners and midlines free (rect fit/refine.cpp:37-39; WRITEUP_2026-09-23.md:84-86); hanging vertices take
    // their edge's line (WRITEUP :29-35); null scale cap 3 (rect fit/gate.cpp:236); a stall is a round admitting nothing
    // (rect fit/refine.cpp:269-277).
    scenes.rect = (() => {
      const g = group("rect");
      const cells = [];
      for (let o = 0; o < R.tri.length; o += R.stride) cells.push(Array.from(R.tri.subarray(o, o + 4)));
      frame(g, `the rectangle run: ${cells.length} cells`);
      let md = "";
      for (const c of cells) md += `M${sx(c[0])},${sy(c[1])}H${sx(c[2])}V${sy(c[3])}H${sx(c[0])}Z`;
      const mesh = el("path", { d: md, class: "meshdark" }, g);
      const corners = new Map();
      for (const c of cells) [[c[0], c[1]], [c[2], c[1]], [c[0], c[3]], [c[2], c[3]]].forEach(([x, y]) => corners.set(k6(x, y), [x, y]));
      const eps = 1e-9, hang = [];
      for (const [x, y] of corners.values()) {
        if (cells.some((c) => (Math.abs(x - c[0]) < eps || Math.abs(x - c[2]) < eps) && y > c[1] + eps && y < c[3] - eps
          || (Math.abs(y - c[1]) < eps || Math.abs(y - c[3]) < eps) && x > c[0] + eps && x < c[2] - eps)) hang.push([x, y]);
      }
      const adm = DR.cands.filter((c) => c.admitted && c.seg);
      const horiz = adm.filter((c) => Math.abs(c.seg[1] - c.seg[3]) < eps).length, vert = adm.length - horiz;
      const ad = el("g", {}, g);
      adm.forEach((c) => el("circle", { cx: sx(c.x), cy: sy(c.y), r: 3, class: "pos" }, ad));
      const hg = el("g", {}, g);
      hang.forEach(([x, y]) => el("circle", { cx: sx(x), cy: sy(y), r: 4, class: "neg" }, hg));
      const leg = el("g", {}, g);
      el("circle", { cx: SQ.x + 6, cy: SQ.y + SQ.s + 22, r: 3, class: "pos" }, leg); text(leg, SQ.x + 16, SQ.y + SQ.s + 26, "admitted vertices", "tiny", "start");
      el("circle", { cx: SQ.x + 156, cy: SQ.y + SQ.s + 22, r: 4, class: "neg" }, leg); text(leg, SQ.x + 166, SQ.y + SQ.s + 26, `hanging vertices: ${hang.length}`, "tiny", "start");
      setV("hang", `In this run ${hang.length} of its ${corners.size} vertices hang. Of the ${adm.length} vertices it admitted, ${horiz} split a horizontal segment and ${vert} a vertical one.`);
      // each with its standard error over the held-out sites, and whether the difference is more than the noise
      const hs = (run) => {
        const h = run.heldout; if (!h) return null;
        if (!h.rows || !h.rows.n.length) return { mean: h.deviance, se: null };
        const R_ = h.rows, dv = [];
        for (let i = 0; i < R_.n.length; ++i) {
          const n = R_.n[i], k = R_.k[i], p = Math.min(1 - 1e-6, Math.max(1e-6, R_.p[i]));
          dv.push((k > 0 ? 2 * k * Math.log(k / (n * p)) : 0) + (n - k > 0 ? 2 * (n - k) * Math.log((n - k) / (n * (1 - p))) : 0));
        }
        const m = dv.reduce((s, x) => s + x, 0) / dv.length, v = dv.reduce((s, x) => s + (x - m) ** 2, 0) / (dv.length - 1);
        return { mean: m, se: Math.sqrt(v / dv.length) };
      };
      const hr = hs(R), ht = hs(T), show = (h) => (h ? h.mean.toFixed(3) + (h.se != null ? ` ± ${h.se.toFixed(3)}` : "") : "?");
      setV("devR", show(hr)); setV("devT", show(ht));
      const noise = hr && ht && hr.se != null && ht.se != null ? 2 * Math.hypot(hr.se, ht.se) : null;
      setV("devV", noise == null ? "" : Math.abs(hr.mean - ht.mean) < noise
        ? ", a difference well inside the noise"
        : `, a difference of ${Math.abs(hr.mean - ht.mean).toFixed(3)}, more than twice its standard error`);
      return { g, update(t) { fade(mesh, seg(t, 0, 0.2)); fade(ad, seg(t, 0.2, 0.4)); fade(hg, seg(t, 0.45, 0.6)); fade(leg, seg(t, 0.45, 0.6)); } };
    })();

    active = null; measure();
  }

  // ------------------------------------------------------------------ the runs: triangles, then rectangles, same flips
  let worker = null, seq = 0, want = null, data = null, tri = null, runs = null;
  function status(s, busy) { const n = $("howstatus"); n.textContent = s; n.classList.toggle("busy", !!busy); }
  function fitNow() {
    const truth = $("truth").value, seed = +(root.dataset.seed || 1);
    POOL_SD = +$("coinsd").value;
    data = makeData(truth, seed); data.truth = truth;
    if (!worker) { worker = new Worker(root.dataset.worker); worker.onmessage = onMsg; worker.onerror = () => status("the estimator failed to load"); }
    want = ++seq; tri = null;
    status("Fitting…", true);
    if (!svg.firstChild || svg.querySelector(".stagebusy")) { svg.textContent = ""; text(svg, 300, 300, "fitting…", "stagebusy"); }
    else svg.style.opacity = 0.35;
    worker.postMessage({ id: want, engine: "tri", design: data.csv, seed: 7, detail: true });
  }
  function onMsg(ev) {
    const m = ev.data;
    if (m.type === "ready" || m.id !== want) return;
    if (m.type === "error") { svg.style.opacity = 1; status("the fit failed: " + m.message); return; }
    if (m.engine === "tri") { tri = m; worker.postMessage({ id: want, engine: "rect", design: data.csv, seed: 7, detail: true }); return; }
    svg.style.opacity = 1;
    runs = { tri, rect: m };
    status(`both estimators · ${((tri.ms + m.ms) / 1000).toFixed(1)} s in your browser`);
    build(data, tri, m);
  }
  const sdLabel = () => { $("coinsdval").textContent = (+$("coinsd").value).toFixed(2); };
  $("coinsd").addEventListener("input", sdLabel);
  $("coinsd").addEventListener("change", fitNow);
  $("truth").addEventListener("change", fitNow);
  $("again").addEventListener("click", () => { root.dataset.seed = (+(root.dataset.seed || 1) % 97) + 1; fitNow(); });
  new MutationObserver(() => { if (runs) build(data, runs.tri, runs.rect); }).observe(document.documentElement, { attributes: true, attributeFilter: ["class"] });

  // ------------------------------------------------------------------ scroll to scene, and the progress line
  const bar = document.getElementById("progress");
  function measure() {
    const vh = window.innerHeight;
    let best = null, bestD = Infinity;
    for (const s of steps) { const r = s.getBoundingClientRect(), d = Math.abs(r.top + r.height / 2 - (window.readLine ? window.readLine() : vh * 0.55)); if (d < bestD) { bestD = d; best = s; } }
    if (bar) { const h = document.documentElement.scrollHeight - vh; bar.style.width = (h > 0 ? (100 * window.scrollY) / h : 0) + "%"; }
    if (!best) return;
    const r = best.getBoundingClientRect();
    prog = clamp((vh * 0.85 - r.top) / (r.height * 0.9));
    if (best !== active) {
      active = best;
      for (const s of steps) s.classList.toggle("on", s === best);
      for (const [name, sc] of Object.entries(scenes)) sc.g.style.opacity = name === best.dataset.scene ? 1 : 0;
    }
  }
  window.addEventListener("scroll", measure, { passive: true });
  window.addEventListener("resize", measure);
  (function frameLoop(now) {
    const story = document.getElementById("story").getBoundingClientRect();
    if (active && story.top < window.innerHeight && story.bottom > 0) { const sc = scenes[active.dataset.scene]; if (sc) sc.update(reduced ? 1 : prog, now); }
    requestAnimationFrame(frameLoop);
  })(performance.now());
  fitNow();
  measure();
})();
