// /decision-mesh, "How the Gate Decides": the decision-mesh gate told in steps. The section narrates the live fit
// above it (gate.js): the same coins, settings and engine, from the same worker, which returns the candidate trace
// with each fit on the coins. Every scene is built from that run: each round's scored candidates with the segment each would split, the gate's empirical null, the lfdr
// cutoff, the admissions, the pool variance by round, the admitted surpluses by depth, and the final surface. Only
// the "corrections" scene is a sketch (a one-dimensional toy computed here), and its caption says so.
(function () {
  "use strict";
  const root = document.getElementById("gatehow");
  const svg = document.getElementById("stage");
  const steps = root ? [...root.querySelectorAll(".step")] : [];
  if (!root || !svg) return;
  const NS = "http://www.w3.org/2000/svg";
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const $ = (id) => document.getElementById(id);

  // ------------------------------------------------------------------ helpers
  const el = (tag, attrs, parent) => { const n = document.createElementNS(NS, tag); for (const [k, v] of Object.entries(attrs || {})) n.setAttribute(k, v); if (parent) parent.appendChild(n); return n; };
  const clamp = (x, a = 0, b = 1) => Math.max(a, Math.min(b, x));
  const ease = (t) => (t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2);
  const seg = (t, a, b) => ease(clamp((t - a) / (b - a)));
  const fade = (n, t) => { n.style.opacity = clamp(t); };
  const text = (g, x, y, s, cls, anchor = "middle") => { const t = el("text", { x, y, class: cls || "", "text-anchor": anchor }, g); t.textContent = s; return t; };
  function mulberry32(a) { return function () { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
  function gauss(r) { let u = 0; while (u === 0) u = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r()); }
  function pencil(pts, seed, wob = 0.6) {
    const r = mulberry32(seed);
    let d = `M${pts[0][0].toFixed(1)},${pts[0][1].toFixed(1)}`;
    for (let i = 1; i < pts.length; ++i) { const [x0, y0] = pts[i - 1], [x1, y1] = pts[i]; d += ` Q${((x0 + x1) / 2 + (r() - 0.5) * wob * 2).toFixed(1)},${((y0 + y1) / 2 + (r() - 0.5) * wob * 2).toFixed(1)} ${x1.toFixed(1)},${y1.toFixed(1)}`; }
    return d;
  }
  function stroke(g, d, cls, width = 1.6) {
    const a = el("path", { d, class: "pencil " + (cls || ""), "stroke-width": width }, g);
    const len = a.getTotalLength() || 1;
    a.style.strokeDasharray = `${len} ${len}`; a.style.strokeDashoffset = len;
    return { a, set(t) { a.style.strokeDashoffset = len * (1 - clamp(t)); } };
  }
  const line = (g, x1, y1, x2, y2, cls, w = 1) => el("line", { x1, y1, x2, y2, class: "pencil " + (cls || ""), "stroke-width": w }, g);
  const fmt = (v, d = 2) => (Math.abs(v) < 0.005 && d <= 2 ? "0" : v.toFixed(d)).replace("-", "−");
  const pct = (v) => Math.round(100 * v) + "%";
  const phi = (z) => Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI);
  // the normal cdf (Numerical Recipes' erfcc, relative error below 1.2e-7), for the counts under each null
  function Phi(x) {
    const z = Math.abs(x) / Math.SQRT2, t = 1 / (1 + 0.5 * z);
    const r = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
    return x >= 0 ? 1 - r / 2 : r / 2;
  }
  // The smooth density the null is matched to: Lindsey's Poisson fit of a degree-6 polynomial to 72 linearly binned
  // counts, the same port as gate-deep.js (tri fit/gate.cpp:106-195; the rect engine's is identical up to its scale
  // box). Returns the density, or null where the engine would need its damped retry.
  function lindseyDensity(zIn) {
    const NB = 72, DEG = 6, lo = -9.5, hi = 9.5, d = (hi - lo) / NB;
    const z = zIn.map((v) => clamp(v, -9, 9)), M = z.length, counts = new Array(NB).fill(0), mids = [];
    for (let b = 0; b < NB; ++b) mids.push(lo + (b + 0.5) * d);
    for (const v of z) { const t = (v - lo) / d - 0.5, b0 = clamp(Math.floor(t), 0, NB - 2), w = clamp(t - b0, 0, 1); counts[b0] += 1 - w; counts[b0 + 1] += w; }
    let mm = 0, ms = 0;
    for (const m of mids) mm += m;
    mm /= NB;
    for (const m of mids) ms += (m - mm) ** 2;
    ms = Math.sqrt(ms / (NB - 1));
    const B = mids.map((m) => { const t = (m - mm) / ms, r = []; let p = 1; for (let k = 0; k <= DEG; ++k) { r.push(p); p *= t; } return r; });
    const solve = (A, b) => {
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
    };
    let beta = new Array(DEG + 1).fill(0), converged = false;
    for (let it = 0; it < 100 && !converged; ++it) {
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
      beta = nb; converged = maxdiff < 1e-9;
    }
    if (!converged) return null;
    return (v) => { const t = (clamp(v, -9, 9) - mm) / ms; let eta = 0, p = 1; for (let k = 0; k <= DEG; ++k) { eta += p * beta[k]; p *= t; } return Math.exp(clamp(eta, -30, 30)) / (M * d); };
  }

  // The mesh's growth, replayed from the trace, for the admissions scene. The fit returns only its final cells; the
  // mesh it starts from is the uniform one with rounds[0].faces cells, and every selected candidate splits the
  // cells whose edge it is the midpoint of (a turned-back one is split in too, and stays a vertex that isn't free).
  // For right triangles a split first splits any neighbor whose longest edge is not that edge (the closure), and
  // split() returns each new line with its depth in that chain. The caller checks the replay ends on the fit's cells.
  function meshReplay(fit, G) {
    const Q = 4096, I = (v) => Math.round(v * Q), key = (p) => p[0] + "," + p[1];
    const same = (a, b) => a[0] === b[0] && a[1] === b[1];
    if (fit.engine === "rect") {
      let cells = [];
      for (let i = 0; i < G; ++i) for (let j = 0; j < G; ++j) cells.push([I(i / G), I(j / G), I((i + 1) / G), I((j + 1) / G)]);
      const initial = [];
      for (let i = 0; i <= G; ++i) { initial.push([i / G, 0, i / G, 1]); initial.push([0, i / G, 1, i / G]); }
      return {
        initial,
        split(x, y) {
          const X = I(x), Y = I(y), out = [], next = [];
          for (const c of cells) {
            const [x0, y0, x1, y1] = c, mx = (x0 + x1) / 2, my = (y0 + y1) / 2;
            if ((X === x0 || X === x1) && Y === my) { next.push([x0, y0, x1, my], [x0, my, x1, y1]); out.push([[x0 / Q, my / Q, x1 / Q, my / Q], 0]); }
            else if ((Y === y0 || Y === y1) && X === mx) { next.push([x0, y0, mx, y1], [mx, y0, x1, y1]); out.push([[mx / Q, y0 / Q, mx / Q, y1 / Q], 0]); }
            else next.push(c);
          }
          cells = next;
          return out;
        },
        cells: () => cells.map((c) => c.map((v) => v / Q)),
      };
    }
    // right triangles: { r: the right-angle corner, p, q: the longest edge }
    let tris = [];
    const orient = new Map();   // cell -> true when its diagonal runs (i, j)-(i+1, j+1)
    {
      const t = fit.tri, K = fit.stride, n = t.length / K, h = 1 / G, eps = 1e-9;
      for (let k = 0; k < n; ++k) {
        const P = [0, 3, 6].map((o) => [t[K * k + o], t[K * k + o + 1]]);
        const xs = P.map((p) => p[0]), ys = P.map((p) => p[1]);
        const x0 = Math.min(...xs), y0 = Math.min(...ys);
        if (Math.abs(Math.max(...xs) - x0 - h) > eps || Math.abs(Math.max(...ys) - y0 - h) > eps) continue;
        const i = Math.round(x0 / h), j = Math.round(y0 / h);
        if (Math.abs(i * h - x0) > eps || Math.abs(j * h - y0) > eps) continue;
        // the corner of the cell not in the triangle says which diagonal it has
        const has = (x, y) => P.some((p) => Math.abs(p[0] - x) < eps && Math.abs(p[1] - y) < eps);
        const main = has(x0, y0) && has(x0 + h, y0 + h);
        orient.set(i + "," + j, main);
      }
      const par = [null, null];
      for (const [k, m] of orient) { const [i, j] = k.split(",").map(Number), s = (i + j) & 1; if (par[s] === null) par[s] = m; else if (par[s] !== m) return null; }
      if (par[0] === null && par[1] === null) return null;
      if (par[0] === null) par[0] = !par[1];
      if (par[1] === null) par[1] = !par[0];
      for (let i = 0; i < G; ++i) for (let j = 0; j < G; ++j) {
        const a = [I(i / G), I(j / G)], b = [I((i + 1) / G), I(j / G)], c = [I((i + 1) / G), I((j + 1) / G)], d = [I(i / G), I((j + 1) / G)];
        if (par[(i + j) & 1]) tris.push({ r: b, p: a, q: c }, { r: d, p: a, q: c });
        else tris.push({ r: a, p: b, q: d }, { r: c, p: b, q: d });
      }
    }
    const initial = [], seen = new Set();
    for (const T of tris) for (const [u, v] of [[T.r, T.p], [T.r, T.q], [T.p, T.q]]) {
      const k = [key(u), key(v)].sort().join("|"); if (seen.has(k)) continue; seen.add(k);
      initial.push([u[0] / Q, u[1] / Q, v[0] / Q, v[1] / Q]);
    }
    const hasEdge = (T, u, v) => { const S = [T.r, T.p, T.q]; return S.some((s) => same(s, u)) && S.some((s) => same(s, v)); };
    const isHyp = (T, u, v) => (same(T.p, u) && same(T.q, v)) || (same(T.p, v) && same(T.q, u));
    function bisect(u, v, level, out) {
      for (let guard = 0; guard < 64; ++guard) {
        const bad = tris.find((T) => hasEdge(T, u, v) && !isHyp(T, u, v));
        if (!bad) break;
        bisect(bad.p, bad.q, level + 1, out);
      }
      const m = [(u[0] + v[0]) / 2, (u[1] + v[1]) / 2], next = [];
      for (const T of tris) {
        if (!isHyp(T, u, v)) { next.push(T); continue; }
        next.push({ r: m, p: T.p, q: T.r }, { r: m, p: T.q, q: T.r });
        out.push([[m[0] / Q, m[1] / Q, T.r[0] / Q, T.r[1] / Q], level]);
      }
      tris = next;
    }
    return {
      initial,
      split(x, y) {
        const m = [I(x), I(y)];
        if (tris.some((T) => same(T.r, m) || same(T.p, m) || same(T.q, m))) return [];
        // the edge m is the midpoint of
        for (const T of tris) for (const [u, v] of [[T.r, T.p], [T.r, T.q], [T.p, T.q]]) {
          if (u[0] + v[0] === 2 * m[0] && u[1] + v[1] === 2 * m[1]) { const out = []; bisect(u, v, 0, out); return out; }
        }
        return null;
      },
      cells: () => tris.map((T) => [T.r, T.p, T.q].map((p) => [p[0] / Q, p[1] / Q])),
    };
  }

  // ------------------------------------------------------------------ the data: the live fit's run (gate.js)
  const BASE = -1;           // the coins' background log-odds, as in gate.js
  const expit = (t) => 1 / (1 + Math.exp(-t));

  // ------------------------------------------------------------------ color: log-odds against the background
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
  function ramp(v) {   // v: log-odds minus the background, clipped at ±2.2
    const t = clamp(v / 2.2, -1, 1), to = t > 0 ? INK.acc : INK.warm, a = Math.pow(Math.abs(t), 0.8);
    // the ink at coverage a (as alpha), not blended into a flat paper colour: over the textured paper it reads the
    // same, and the grain shows through
    return [to[0], to[1], to[2], Math.round(255 * a)];
  }
  function raster(fn, N) {   // an image of fn(x, y) on an N × N grid, y up
    const c = document.createElement("canvas"); c.width = c.height = N;
    const ctx = c.getContext("2d"), img = ctx.createImageData(N, N);
    for (let j = 0; j < N; ++j) for (let i = 0; i < N; ++i) {
      const [r, g, b, al] = ramp(fn((i + 0.5) / N, 1 - (j + 0.5) / N)), o = 4 * (j * N + i);
      img.data[o] = r; img.data[o + 1] = g; img.data[o + 2] = b; img.data[o + 3] = al;
    }
    ctx.putImageData(img, 0, 0);
    return c.toDataURL();
  }

  // ------------------------------------------------------------------ the square on the stage
  const SQ = { x: 80, y: 70, s: 440 };
  const sx = (x) => SQ.x + x * SQ.s, sy = (y) => SQ.y + (1 - y) * SQ.s;
  function frame(g, label) {
    el("rect", { x: SQ.x, y: SQ.y, width: SQ.s, height: SQ.s, class: "frame" }, g);
    if (label) text(g, SQ.x, SQ.y - 12, label, "mono", "start");
  }
  function meshLines(g, fit, cls = "meshline") {
    const t = fit.tri, K = fit.stride, n = t.length / K;
    let d = "";
    for (let i = 0; i < n; ++i) {
      const o = K * i;
      if (fit.engine === "rect") d += `M${sx(t[o])},${sy(t[o + 1])}H${sx(t[o + 2])}V${sy(t[o + 3])}H${sx(t[o])}Z`;
      else d += `M${sx(t[o])},${sy(t[o + 1])}L${sx(t[o + 3])},${sy(t[o + 4])}L${sx(t[o + 6])},${sy(t[o + 7])}Z`;
    }
    return el("path", { d, class: cls }, g);
  }

  // ------------------------------------------------------------------ the scenes, built from one run
  let scenes = {}, active = null, prog = 0, run = null;
  const group = (name) => { const g = el("g", { "data-scene": name }, svg); g.style.opacity = 0; g.style.transition = "opacity 0.6s"; return g; };

  function build(data, fit) {
    svg.textContent = ""; scenes = {};
    inks();
    const D = fit.detail, cands = D.cands, rounds = [...new Set(cands.map((c) => c.round))].sort((a, b) => a - b);
    // A triangle-mesh location can hold a coarse and a fine vertex (the two-node hierarchy), and the worker's trace keys
    // one vertex per location, so a candidate's own flag can miss its admission. Read it off every vertex there instead.
    {
      const k6 = (x, y) => x.toFixed(6) + "," + y.toFixed(6);
      const admAt = new Set(D.vertices.filter((v) => v.admitted).map((v) => v.round + "|" + k6(v.x, v.y)));
      cands.forEach((c) => { if (c.selected) c.admitted = admAt.has(c.round + "|" + k6(c.x, c.y)); });
    }
    const byRound = (r) => cands.filter((c) => c.round === r);
    const r0 = byRound(0), cal = D.calib, cal0 = cal.find((c) => c.round === 0) || cal[0];
    const zmax = Math.max(4, ...r0.map((c) => Math.abs(c.z)));
    // the fitted log-odds at any point: the baseline plus the surface, bilinear between the dump's grid centers
    const SG = D.surface, SN = Math.round(Math.sqrt(SG.f.length));
    const fitLogit = (x, y) => {
      const gx = clamp(x * SN - 0.5, 0, SN - 1.001), gy = clamp(y * SN - 0.5, 0, SN - 1.001), i = Math.floor(gx), j = Math.floor(gy), u = gx - i, v = gy - j;
      const F = (a, b) => SG.f[a * SN + b];   // gx slowest
      return fit.baseline + (F(i, j) * (1 - u) + F(i + 1, j) * u) * (1 - v) + (F(i, j + 1) * (1 - u) + F(i + 1, j + 1) * u) * v;
    };
    // standardized residuals: a binomial's spread is fixed by its mean, so (k − n p) / √(n p (1 − p)) has sd 1 if
    // the coins are binomial at odds p, whatever p and n are
    const resid = (logit) => Array.from(data.x, (_, i) => { const p = expit(logit(data.x[i], data.y[i])), n = data.n[i]; return (data.k[i] - n * p) / Math.sqrt(n * p * (1 - p)); });
    const zFit = resid(fitLogit), zFlat = resid(() => fit.baseline);
    const disp = (z) => z.reduce((a, b) => a + b * b, 0) / z.length;
    const outside = zFit.filter((z) => Math.abs(z) > 1.96).length / zFit.length;
    // what a pool variance v alone predicts for that mean square: 1 + (n − 1) p (1 − p) v per site, to first order
    const predicted = Array.from(data.x, (_, i) => { const p = expit(fitLogit(data.x[i], data.y[i])); return 1 + (data.n[i] - 1) * p * (1 - p) * fit.poolVariance; }).reduce((a, b) => a + b, 0) / data.x.length;

    // 1. the sites ----------------------------------------------------------------------------------------------
    scenes.sites = (() => {
      const g = group("sites");
      const c = document.createElement("canvas"), N = 440; c.width = c.height = N * 2;
      const ctx = c.getContext("2d");
      for (let i = 0; i < data.x.length; ++i) {
        const p = (data.k[i] + 0.5) / (data.n[i] + 1), v = Math.log(p / (1 - p)) - BASE;
        const [r, gg, b, al] = ramp(v); ctx.fillStyle = `rgba(${r},${gg},${b},${al / 255})`;
        ctx.beginPath(); ctx.arc(data.x[i] * 2 * N, (1 - data.y[i]) * 2 * N, 3.2, 0, 2 * Math.PI); ctx.fill();
      }
      const img = el("image", { x: SQ.x, y: SQ.y, width: SQ.s, height: SQ.s, href: c.toDataURL() }, g);
      frame(g, `${data.x.length.toLocaleString("en-US")} sites, each dot one site's share of heads`);
      const truth = el("image", { x: SQ.x, y: SQ.y, width: SQ.s, height: SQ.s, href: raster((x, y) => data.tf(x, y) - BASE, 110), opacity: 0 }, g);
      g.insertBefore(truth, img);                            // the true odds come in beneath the dots, not over them
      const tl = text(g, SQ.x + SQ.s, SQ.y + SQ.s + 26, "the odds the coins really have", "mono", "end");
      const leg = el("g", {}, g);
      [["more heads than usual", "pos"], ["fewer", "neg"]].forEach(([s, cls], i) => { el("circle", { cx: SQ.x + 6 + i * 170, cy: SQ.y + SQ.s + 22, r: 5, class: cls }, leg); text(leg, SQ.x + 16 + i * 170, SQ.y + SQ.s + 26, s, "tiny", "start"); });
      return { g, update(t) { fade(img, seg(t, 0, 0.15)); truth.style.opacity = 0.9 * seg(t, 0.55, 0.8); fade(tl, seg(t, 0.6, 0.8)); fade(leg, seg(t, 0.1, 0.25) * (1 - seg(t, 0.55, 0.7))); } };
    })();

    // 1b. the funnel: how far a binomial may stray is known from its mean ---------------------------------------
    scenes.funnel = (() => {
      const g = group("funnel");
      const B = { x: 80, y: 110, w: 440, h: 360 }, nmin = Math.min(...data.n), nmax = Math.max(...data.n), ymax = 1.1;
      const X = (n) => B.x + ((n - nmin) / (nmax - nmin + 1)) * B.w, Y = (v) => B.y + B.h / 2 - (v / ymax) * (B.h / 2);
      text(g, B.x, B.y - 30, "share of heads minus the fitted odds, per unit of a coin's spread", "mono", "start");
      line(g, B.x, B.y + B.h, B.x + B.w, B.y + B.h, "", 1); line(g, B.x, Y(0), B.x + B.w, Y(0), "soft", 1);
      const c = document.createElement("canvas"); c.width = 2 * B.w; c.height = 2 * B.h;
      const ctx = c.getContext("2d"), r = mulberry32(9);
      zFit.forEach((z, i) => {
        const n = data.n[i], v = z / Math.sqrt(n), o = Math.abs(z) > 1.96;
        const col = o ? (z > 0 ? INK.acc : INK.warm) : [128, 128, 128];
        ctx.fillStyle = `rgba(${col[0]},${col[1]},${col[2]},${o ? 0.75 : 0.25})`;
        ctx.beginPath(); ctx.arc(2 * (X(n + (r() - 0.5) * 0.8) - B.x), 2 * (Y(clamp(v, -ymax, ymax)) - B.y), 2.4, 0, 2 * Math.PI); ctx.fill();
      });
      const img = el("image", { x: B.x, y: B.y, width: B.w, height: B.h, href: c.toDataURL() }, g);
      const band = [], lo = [];
      for (let n = nmin; n <= nmax; ++n) { band.push([X(n), Y(1.96 / Math.sqrt(n))]); lo.push([X(n), Y(-1.96 / Math.sqrt(n))]); }
      const up = stroke(g, "M" + band.map((q) => q.join(",")).join(" L"), "warm", 2), dn = stroke(g, "M" + lo.map((q) => q.join(",")).join(" L"), "warm", 2);
      text(g, B.x, B.y + B.h + 18, `${nmin} flips`, "tiny", "start"); text(g, B.x + B.w, B.y + B.h + 18, `${nmax} flips`, "tiny", "end");
      const lab = text(g, B.x + 6, B.y + B.h - 10, "a binomial stays between the lines 95% of the time", "label warmt halo", "start");
      const cap = text(g, 300, B.y + B.h + 50, `outside: ${pct(outside)} of sites, not 5%; mean squared residual ${disp(zFit).toFixed(2)}, not 1`, "mono");
      return { g, update(t) { fade(img, seg(t, 0, 0.2)); up.set(seg(t, 0.25, 0.5)); dn.set(seg(t, 0.25, 0.5)); fade(lab, seg(t, 0.45, 0.6)); fade(cap, seg(t, 0.6, 0.75)); } };
    })();

    // 1b'. a wrong mean shows up as variance, whichever way the mean is missed ----------------------------------
    // A site's miss is its true log-odds less the fitted: the fitted surface missing the true surface, plus the
    // coin's own effect, the part of its odds the two axes don't describe. Both are known here, since the page drew them.
    scenes.wrongmean = (() => {
      const g = group("wrongmean");
      const tf = data.tf;
      const zTrue = resid(tf);
      // per site: the whole miss, the squared residual, and what the miss predicts: 1 + n p(1 − p) miss², to first order
      const sites = (logit, z) => Array.from(data.x, (_, i) => {
        const L = logit(data.x[i], data.y[i]), p = expit(L), m = tf(data.x[i], data.y[i]) + data.u[i] - L;
        return { m: Math.abs(m), z2: z[i] * z[i], pred: 1 + data.n[i] * p * (1 - p) * m * m };
      });
      const S0 = sites(() => fit.baseline, zFlat), S1 = sites(fitLogit, zFit), S2 = sites(tf, zTrue);
      const top = [...S0].map((q) => q.m).sort((a, b) => a - b)[Math.floor(0.995 * S0.length)], K = 14, bw = top / K;
      const bin = (arr) => {
        const out = [];
        for (let b = 0; b < K; ++b) {
          const s = arr.filter((q) => q.m >= b * bw && (q.m < (b + 1) * bw || (b === K - 1 && q.m <= top)));
          if (s.length < 25) continue;
          const mean = (f) => s.reduce((t, q) => t + f(q), 0) / s.length;
          out.push({ m: mean((q) => q.m), z2: mean((q) => q.z2), pred: mean((q) => q.pred), n: s.length });
        }
        return out;
      };
      const flat = bin(S0), fin = bin(S1), tru = bin(S2);

      // one site, its miss taken apart
      const pick = (() => { let best = 0, bs = -1; data.x.forEach((_, i) => { const sm = tf(data.x[i], data.y[i]) - fit.baseline, am = data.u[i]; const sc = Math.min(Math.abs(sm), 1.2) + 2 * Math.min(Math.abs(am), 0.4) - (Math.sign(sm) !== Math.sign(am) ? 9 : 0); if (sc > bs) { bs = sc; best = i; } }); return best; })();
      const L0 = fit.baseline, L1 = tf(data.x[pick], data.y[pick]), L2 = L1 + data.u[pick];
      const lo = Math.min(L0, L1, L2) - 0.15, hi = Math.max(L0, L1, L2) + 0.15, AX = (v) => 110 + ((v - lo) / (hi - lo)) * 380, ay = 150;
      const top1 = el("g", {}, g);
      text(top1, 70, 82, "one site's error, in log-odds", "mono", "start");
      line(top1, 90, ay, 510, ay, "soft", 1);
      const tick = (v, lab, cls, dy) => { line(top1, AX(v), ay - 8, AX(v), ay + 8, cls, 2); text(top1, AX(v), ay + dy, lab, "tiny", "middle"); };
      tick(L0, "fitted (flat)", "", 26); tick(L1, "the true surface", "warm", -16); tick(L2, "this coin", "accent", 26);
      const brace = (a, b, y, lab, cls) => { const q = el("g", {}, top1); line(q, AX(a), y, AX(b), y, cls, 2.4); text(q, (AX(a) + AX(b)) / 2, y + 16, lab, "tiny " + (cls === "accent" ? "acc" : cls === "warm" ? "warmt" : ""), "middle"); return q; };
      const b1 = brace(L0, L1, ay + 44, "surface error", "warm"), b2 = brace(L1, L2, ay + 44, "axes error", "accent"), b3 = brace(L0, L2, ay + 80, "total error", "");

      // the chart
      const B = { x: 90, y: 318, w: 430, h: 210 }, mmax = top * 1.05, ymax = Math.max(2, ...flat.map((q) => Math.max(q.z2, q.pred))) * 1.08;
      const X = (m) => B.x + (m / mmax) * B.w, Y = (y) => B.y + B.h - ((y - 0.5) / (ymax - 0.5)) * B.h;
      const ch = el("g", {}, g);
      text(ch, B.x - 10, B.y - 16, "mean squared residual, by the size of the total error", "mono", "start");
      line(ch, B.x, B.y + B.h, B.x + B.w, B.y + B.h, "", 1); line(ch, B.x, B.y, B.x, B.y + B.h, "", 1);
      text(ch, B.x, B.y + B.h + 16, "0", "tiny", "start"); text(ch, B.x + B.w, B.y + B.h + 16, `error ${fmt(mmax, 1)}`, "tiny", "end");
      for (const y of [1, 2, 3, 4, 5, 6, 8].filter((y) => y < ymax)) text(ch, B.x - 8, Y(y) + 4, String(y), "tiny", "end");
      line(ch, B.x, Y(1), B.x + B.w, Y(1), "soft", 1.2).setAttribute("stroke-dasharray", "4 4");
      text(ch, B.x + B.w, Y(1) - 6, "1: no error", "tiny", "end");
      const theory = stroke(ch, "M" + [...flat].sort((a, b) => a.m - b.m).map((q) => `${X(q.m).toFixed(1)},${Y(q.pred).toFixed(1)}`).join(" L"), "soft", 1.6);
      const rad = (q) => 2 + Math.sqrt(q.n) / 7;
      const dots = (arr, cls) => arr.map((q) => el("circle", { cx: X(q.m), cy: Y(q.z2), r: rad(q), class: cls, "fill-opacity": 0.8 }, ch));
      const d0 = dots(flat, "neg"), dF = dots(fin, "pos");
      const dT = tru.map((q) => el("circle", { cx: X(q.m), cy: Y(q.z2), r: rad(q) + 1.5, class: "pencil", "stroke-width": 1.6, fill: "none" }, ch));
      const leg = el("g", {}, ch);
      [["against a flat surface", "neg"], ["against the final fit", "pos"], ["against the true surface", "ring"]].forEach(([s2, c], i) => { el("circle", c === "ring" ? { cx: B.x + 14, cy: B.y + 6 + i * 17, r: 5, class: "pencil", "stroke-width": 1.6, fill: "none" } : { cx: B.x + 14, cy: B.y + 6 + i * 17, r: 5, class: c }, leg); text(leg, B.x + 26, B.y + 10 + i * 17, s2, "tiny", "start"); });
      line(leg, B.x + 6, B.y + 57, B.x + 22, B.y + 57, "soft", 1.6); text(leg, B.x + 26, B.y + 61, "1 + n p(1 − p) · error²", "tiny", "start");
      const cap = text(g, 300, 586, "the true surface is still off by what the axes don't capture", "mono");
      return { g, update(t) {
        fade(top1, seg(t, 0, 0.08)); fade(b1, seg(t, 0.06, 0.14)); fade(b2, seg(t, 0.12, 0.2)); fade(b3, seg(t, 0.18, 0.26));
        fade(ch, seg(t, 0.25, 0.32)); theory.set(seg(t, 0.3, 0.45));
        const n = Math.floor(seg(t, 0.32, 0.55) * d0.length); d0.forEach((d, i) => fade(d, i < n ? 1 : 0));
        dF.forEach((d) => fade(d, seg(t, 0.55, 0.65))); dT.forEach((d) => fade(d, seg(t, 0.68, 0.78))); fade(leg, seg(t, 0.3, 0.4)); fade(cap, seg(t, 0.75, 0.88));
      } };
    })();

    // 1c. two kinds of excess: shared by neighbors, or each site's own ------------------------------------------
    scenes.coherent = (() => {
      const g = group("coherent");
      const w = 250, gap = 20, x0 = 300 - w - gap / 2, x1 = 300 + gap / 2, y0 = 150;
      const map = (z) => {
        const c = document.createElement("canvas"), N = 2 * w; c.width = c.height = N;
        const ctx = c.getContext("2d");
        z.forEach((v, i) => { const [r, gg, b, al] = ramp(clamp(v, -3, 3) * (2.2 / 3)); ctx.fillStyle = `rgba(${r},${gg},${b},${al / 255})`; ctx.beginPath(); ctx.arc(data.x[i] * N, (1 - data.y[i]) * N, 3.4, 0, 2 * Math.PI); ctx.fill(); });
        return c.toDataURL();
      };
      const A = el("g", {}, g), Bg = el("g", {}, g);
      el("image", { x: x0, y: y0, width: w, height: w, href: map(zFlat) }, A); el("rect", { x: x0, y: y0, width: w, height: w, class: "frame" }, A);
      text(A, x0, y0 - 12, "against a flat surface", "mono", "start");
      text(A, x0 + w / 2, y0 + w + 24, "coherent: neighbors share it", "label");
      text(A, x0 + w / 2, y0 + w + 42, `mean squared residual ${disp(zFlat).toFixed(2)}`, "tiny");
      el("image", { x: x1, y: y0, width: w, height: w, href: map(zFit) }, Bg); el("rect", { x: x1, y: y0, width: w, height: w, class: "frame" }, Bg);
      text(Bg, x1, y0 - 12, "against the final fit", "mono", "start");
      text(Bg, x1 + w / 2, y0 + w + 24, "incoherent: each site's own", "label");
      text(Bg, x1 + w / 2, y0 + w + 42, `mean squared residual ${disp(zFit).toFixed(2)}`, "tiny");
      text(Bg, x1 + w / 2, y0 + w + 56, `the coin variance alone predicts ${predicted.toFixed(2)}`, "tiny");
      const cap = text(g, 300, y0 + w + 96, "each dot: (heads − n p) / √(n p (1 − p)), blue above, orange below", "tiny");
      return { g, update(t) { fade(A, seg(t, 0, 0.2)); fade(Bg, seg(t, 0.35, 0.55)); fade(cap, seg(t, 0.55, 0.7)); } };
    })();

    // 2. the coarse mesh: round 0's candidate segments are the edges the fit starts with ---------------------------
    scenes.coarse = (() => {
      const g = group("coarse");
      frame(g, "round 0: the segments a candidate could split");
      const segs = r0.filter((c) => c.seg);
      const order = segs.map((c, i) => [c.seg[0] + c.seg[1] + c.seg[2] + c.seg[3], i]).sort((a, b) => a[0] - b[0]);
      const lines = order.map(([, i]) => { const s = segs[i].seg; return line(g, sx(s[0]), sy(s[1]), sx(s[2]), sy(s[3]), "", 1); });
      const mids = segs.map((c) => el("circle", { cx: sx(c.x), cy: sy(c.y), r: 2.6, class: "sheet pencil", "stroke-width": 1 }, g));
      const free = D.vertices.filter((v) => v.free && !v.admitted);
      const dots = free.map((v) => el("circle", { cx: sx(v.x), cy: sy(v.y), r: 5, class: "ink" }, g));
      const cap = text(g, 300, SQ.y + SQ.s + 28, `${free.length} free to begin with (black); ${segs.length} candidates (hollow)`, "mono");
      return { g, update(t) {
        const n = Math.floor(seg(t, 0, 0.45) * lines.length);
        lines.forEach((l, i) => fade(l, i < n ? 0.55 : 0));
        dots.forEach((d) => fade(d, seg(t, 0.35, 0.5)));
        mids.forEach((m) => fade(m, seg(t, 0.5, 0.7))); fade(cap, seg(t, 0.55, 0.75));
      } };
    })();

    // 3. surplus: one segment in profile --------------------------------------------------------------------------
    scenes.surplus = (() => {
      const g = group("surplus");
      const X0 = 110, X1 = 490, XM = 300, base = 430, h0 = 190, h1 = 270, hm = (h0 + h1) / 2, s = 120;
      line(g, 70, base, 530, base, "soft", 1);
      const par = [[X0, h0, "parent"], [X1, h1, "parent"]].map(([x, h, l]) => { const q = el("g", {}, g); line(q, x, base, x, base - h, "soft", 1); el("circle", { cx: x, cy: base - h, r: 7, class: "ink" }, q); text(q, x, base + 22, l, "mono"); return q; });
      const chord = stroke(g, pencil([[X0, base - h0], [X1, base - h1]], 3, 0.4), "", 1.6);
      const mid = el("circle", { cx: XM, cy: base - hm, r: 7, class: "sheet pencil", "stroke-width": 1.6 }, g);
      const ml = text(g, XM + 14, base - hm + 26, "not free: the average of its parents", "mono", "start");
      const tent = el("path", { class: "shade" }, g);
      const tentl = el("path", { class: "pencil accent", "stroke-width": 2.2, fill: "none" }, g);
      const arrow = el("g", {}, g);
      const al = line(arrow, XM, base - hm, XM, base - hm, "accent", 1.6);
      const at = text(arrow, XM - 14, base - hm - s / 2, "surplus", "label acc", "end");
      const free = el("circle", { cx: XM, cy: base - hm, r: 7, class: "pos" }, g);
      const cap = text(g, 300, 520, "the tent: how the surface moves when the surplus does", "mono");
      return { g, update(t) {
        par.forEach((p) => fade(p, seg(t, 0, 0.1))); chord.set(seg(t, 0.05, 0.25));
        fade(mid, seg(t, 0.2, 0.3)); fade(ml, seg(t, 0.22, 0.32) * (1 - seg(t, 0.45, 0.55)));
        const u = seg(t, 0.45, 0.75), ym = base - hm - s * u;
        free.setAttribute("cy", ym); fade(free, seg(t, 0.45, 0.5));
        al.setAttribute("y2", ym); fade(arrow, seg(t, 0.5, 0.6)); at.setAttribute("y", (base - hm + ym) / 2 + 5);
        const d = `M${X0},${base - h0} L${XM},${ym} L${X1},${base - h1}`;
        tentl.setAttribute("d", d); tent.setAttribute("d", d + " Z"); fade(tentl, seg(t, 0.5, 0.65)); fade(tent, seg(t, 0.55, 0.7));
        fade(cap, seg(t, 0.7, 0.85));
      } };
    })();

    // 4. the scores of round 0 on the square ------------------------------------------------------------------------
    scenes.score = (() => {
      const g = group("score");
      frame(g, `round 0: ${r0.length} candidates, sized by |z|`);
      const segs = el("g", {}, g);
      r0.forEach((c) => { if (c.seg) line(segs, sx(c.seg[0]), sy(c.seg[1]), sx(c.seg[2]), sy(c.seg[3]), "soft", 0.8); });
      const sorted = [...r0].sort((a, b) => Math.abs(a.z) - Math.abs(b.z));
      const dots = sorted.map((c) => el("circle", { cx: sx(c.x), cy: sy(c.y), r: 1.5 + 11 * Math.sqrt(Math.min(Math.abs(c.z), 12) / 12), class: c.z > 0 ? "pos" : "neg", "fill-opacity": 0.3 + 0.6 * Math.min(1, Math.abs(c.z) / 4) }, g));
      const best = sorted[sorted.length - 1];
      const lab = text(g, sx(best.x) + (best.x > 0.6 ? -18 : 18), sy(best.y) - 16, `z = ${fmt(best.z, 1)}`, "label halo " + (best.z > 0 ? "acc" : "warmt"), best.x > 0.6 ? "end" : "start");
      return { g, update(t) {
        fade(segs, seg(t, 0, 0.15));
        const n = Math.floor(seg(t, 0.05, 0.6) * dots.length);
        dots.forEach((d, i) => fade(d, i < n ? 1 : 0));
        fade(lab, seg(t, 0.6, 0.75));
      } };
    })();

    // 5. the corrections, sketched in one dimension ---------------------------------------------------------------
    scenes.corrections = (() => {
      const g = group("corrections");
      const r = mulberry32(5), n = 36, U = [], R = [], E = [];
      for (let i = 0; i < n; ++i) { const u = (i + 0.5) / n; U.push(u); E.push(0.18 + 0.1 * Math.sin(7 * u)); R.push(0.55 * (u - 0.5) + 0.6 * Math.max(0, 1 - Math.abs(2 * u - 1)) - 0.25 + E[i] + 0.22 * gauss(r)); }
      const hat = U.map((u) => 1 - Math.abs(2 * u - 1));
      // the tent's projection on its parents' columns (1 − u and u), by least squares on these sites
      const a11 = U.reduce((s, u) => s + (1 - u) ** 2, 0), a12 = U.reduce((s, u) => s + (1 - u) * u, 0), a22 = U.reduce((s, u) => s + u * u, 0);
      const b1 = U.reduce((s, u, i) => s + (1 - u) * hat[i], 0), b2 = U.reduce((s, u, i) => s + u * hat[i], 0), det = a11 * a22 - a12 * a12;
      const c1 = (a22 * b1 - a12 * b2) / det, c2 = (a11 * b2 - a12 * b1) / det;
      const proj = U.map((u) => c1 * (1 - u) + c2 * u), rem = hat.map((h, i) => h - proj[i]);
      const X = (u) => 90 + u * 420;
      // panel a: residuals and their expected values
      const A = el("g", {}, g), Ay = 170, As = 70;
      text(A, 70, 70, "1 · center each residual on its own expected value", "mono", "start");
      line(A, 80, Ay, 520, Ay, "soft", 1);
      const dotsA = U.map((u, i) => el("circle", { cx: X(u), cy: Ay - As * R[i], r: 3.2, class: "ink" }, A));
      const ticks = U.map((u, i) => line(A, X(u) - 5, Ay - As * E[i], X(u) + 5, Ay - As * E[i], "warm", 1.6));
      // panel b: the tent less what the parents absorb
      const B = el("g", {}, g), By = 390, Bs = 110;
      text(B, 70, 262, "2 · take out what the parents could absorb", "mono", "start");
      line(B, 80, By, 520, By, "soft", 1);
      const P = (arr) => "M" + U.map((u, i) => `${X(u).toFixed(1)},${(By - Bs * arr[i]).toFixed(1)}`).join(" L");
      const hatL = el("path", { d: P(hat), class: "pencil", "stroke-width": 1.6, fill: "none" }, B);
      const projL = el("path", { d: P(proj), class: "pencil warm", "stroke-width": 1.6, fill: "none", "stroke-dasharray": "5 4" }, B);
      const remA = el("path", { d: P(rem) + ` L${X(U[n - 1])},${By} L${X(U[0])},${By} Z`, class: "shade" }, B);
      const remL = el("path", { d: P(rem), class: "pencil accent", "stroke-width": 2, fill: "none" }, B);
      const lb = [text(B, X(0.5) + 10, By - Bs * hat[n >> 1] - 4, "the tent", "tiny", "start"), text(B, X(0.8), By - Bs * proj[n - 2] - 8, "the parents' share", "tiny", "start"), text(B, X(0.5), By + 20, "what only the candidate can do", "tiny")];
      // panel c: the result
      const C = el("g", {}, g);
      text(C, 70, 462, "3 · remove the shrinkage bias, divide by the standard deviation", "mono", "start");
      const zl = text(C, 300, 506, "z  =  (centered score on the remainder − shrinkage bias) / sd", "label");
      const cap = text(g, 300, 570, "a one-dimensional sketch, not the run", "tiny");
      return { g, update(t) {
        const u = seg(t, 0.12, 0.32);
        dotsA.forEach((d, i) => { d.setAttribute("cy", Ay - As * (R[i] - u * E[i])); fade(d, seg(t, 0, 0.1)); });
        ticks.forEach((k, i) => { fade(k, seg(t, 0.04, 0.12) * (1 - u)); });
        fade(hatL, seg(t, 0.35, 0.45)); fade(projL, seg(t, 0.45, 0.55)); fade(remA, seg(t, 0.55, 0.65)); fade(remL, seg(t, 0.55, 0.65));
        lb.forEach((l, i) => fade(l, seg(t, 0.38 + i * 0.1, 0.48 + i * 0.1)));
        fade(C, seg(t, 0.7, 0.85)); fade(zl, seg(t, 0.75, 0.9)); fade(cap, seg(t, 0.2, 0.3));
      } };
    })();

    // histogram of a round's scores, with the textbook null and the round's own
    function histogram(g, zs, c, box, opts = {}) {
      const { x, y, w, h } = box, lo = -Math.min(zmax, 12), hi = Math.min(zmax, 12), bw = opts.bw || 0.5;
      const nb = Math.ceil((hi - lo) / bw), cnt = new Array(nb).fill(0), sel = new Array(nb).fill(0);
      zs.forEach((q) => { const b = clamp(Math.floor((clamp(q.z, lo, hi - 1e-9) - lo) / bw), 0, nb - 1); cnt[b]++; if (q.selected) sel[b]++; });
      const M = zs.length, peak = 1.05 * Math.max(...cnt, M * bw * phi(0), c ? (c.pi0 * M * bw * phi(0)) / c.nullSd : 0, 1);
      const X = (z) => x + ((z - lo) / (hi - lo)) * w, Y = (v) => y + h - (v / peak) * h;
      const bars = cnt.map((k, b) => el("rect", { x: X(lo + b * bw) + 0.5, y: Y(k), width: Math.max(1, (w * bw) / (hi - lo) - 1), height: y + h - Y(k), class: "ink", opacity: 0.22 }, g));
      const sbars = sel.map((k, b) => k ? el("rect", { x: X(lo + b * bw) + 0.5, y: Y(k), width: Math.max(1, (w * bw) / (hi - lo) - 1), height: y + h - Y(k), class: lo + b * bw > 0 ? "pos" : "neg" }, g) : null).filter(Boolean);
      line(g, x, y + h, x + w, y + h, "", 1);
      const curve = (m, s, scale) => { const p = []; for (let z = lo; z <= hi + 1e-9; z += (hi - lo) / 160) p.push([X(z), Y(scale * M * bw * phi((z - m) / s) / s)]); return "M" + p.map((q) => q[0].toFixed(1) + "," + q[1].toFixed(1)).join(" L"); };
      const textbook = el("path", { d: curve(0, 1, 1), class: "pencil soft", "stroke-width": 1.4, fill: "none", "stroke-dasharray": "4 4" }, g);
      const own = c ? el("path", { d: curve(c.nullMean, c.nullSd, c.pi0), class: "pencil accent", "stroke-width": 2, fill: "none" }, g) : null;
      if (!opts.bare) for (const z of [-8, -4, 0, 4, 8]) if (z >= lo && z <= hi) text(g, X(z), y + h + 16, fmt(z, 0), "tiny");
      return { bars, sbars, textbook, own, X, Y, M, bw, lo, hi };
    }

    // 6. the family: round 0's scores and its null ---------------------------------------------------------------
    scenes.family = (() => {
      const g = group("family");
      text(g, 70, 90, `round 0: ${r0.length} scores`, "mono", "start");
      const H = histogram(g, r0, cal0, { x: 70, y: 120, w: 460, h: 300 });
      H.sbars.forEach((b) => b.remove());
      // The null is matched to the center of a smooth fit to the whole histogram, not to the bars. A degree-6 curve
      // across z from -9.5 to 9.5 cannot follow a sharp peak, so when the bars peak sharply the null comes out wider
      // than they are. Drawn here so that gap reads as the method's, and counted so the reader can check which null
      // carries the center: the scores within |z| <= 2 against what each drawn curve puts there.
      const dens = lindseyDensity(r0.map((c) => c.z));
      const smooth = dens ? el("path", { d: (() => { const p = []; for (let z = H.lo; z <= H.hi + 1e-9; z += (H.hi - H.lo) / 240) p.push([H.X(z), H.Y(dens(z) * H.M * H.bw)]); return "M" + p.map((q) => q[0].toFixed(1) + "," + q[1].toFixed(1)).join(" L"); })(), class: "pencil", "stroke-width": 1.4, fill: "none" }, g) : null;
      const l1 = el("g", {}, g), l2 = el("g", {}, g), l3 = el("g", {}, g), l4 = el("g", {}, g);
      line(l1, 80, 470, 110, 470, "soft", 1.4).setAttribute("stroke-dasharray", "4 4"); text(l1, 118, 474, "the textbook null, N(0, 1)", "tiny", "start");
      line(l2, 80, 492, 110, 492, "accent", 2); text(l2, 118, 496, `this round's null: center ${fmt(cal0.nullMean)}, spread ${fmt(cal0.nullSd)}, share ${pct(cal0.pi0)}`, "tiny", "start");
      if (smooth) { line(l3, 80, 514, 110, 514, "", 1.4); text(l3, 118, 518, "the smooth fit its center is matched to", "tiny", "start"); }
      const inC = r0.filter((c) => Math.abs(c.z) <= 2).length;
      const ownC = cal0.pi0 * r0.length * (Phi((2 - cal0.nullMean) / cal0.nullSd) - Phi((-2 - cal0.nullMean) / cal0.nullSd));
      const tbC = r0.length * (Phi(2) - Phi(-2));
      text(l4, 80, 544, `within |z| ≤ 2: ${inC} scores; this round's null holds ${Math.round(ownC)}, N(0, 1) ${Math.round(tbC)}`, "tiny", "start");
      if (smooth) text(l4, 80, 562, "the null is wider than the sharp peak: many null scores really are this spread out", "tiny", "start");
      return { g, update(t) {
        const n = Math.floor(seg(t, 0, 0.35) * H.bars.length);
        H.bars.forEach((b, i) => fade(b, i < n ? 0.22 : 0));
        fade(H.textbook, seg(t, 0.35, 0.45)); fade(l1, seg(t, 0.35, 0.45));
        if (smooth) fade(smooth, seg(t, 0.45, 0.55)); fade(l3, seg(t, 0.45, 0.55));
        if (H.own) fade(H.own, seg(t, 0.5, 0.65)); fade(l2, seg(t, 0.5, 0.65));
        fade(l4, seg(t, 0.65, 0.8));
      } };
    })();

    // 7. lfdr, best first, and the running mean -------------------------------------------------------------------
    // The cutoff q sweeps slowly while the scene is on screen (reduced motion: the live fit's q, still), and the run it
    // takes moves with it, in the drawing and in the text. This is the gate's rule applied to round 0's scores at each
    // q, not a refit: at the live fit's q it takes what the fit selected, and that q stays marked.
    const qLive = data.q, byL = [...r0].sort((a, b) => a.lfdr - b.lfdr || Math.abs(b.z) - Math.abs(a.z));
    const runOf = (q) => { let cum = 0, k = 0; byL.forEach((c, i) => { cum += c.lfdr; if (cum / (i + 1) <= q) k = i + 1; }); return k; };
    const QLO = 0.03, QHI = 0.4, QPER = 14;   // the sweep: 3% to 40% and back, evenly in log q, every 14 s
    const u0 = clamp(Math.log(qLive / QLO) / Math.log(QHI / QLO)), ph0 = Math.acos(1 - 2 * u0);
    const qAt = (s) => QLO * Math.pow(QHI / QLO, (1 - Math.cos(ph0 + (2 * Math.PI * s) / QPER)) / 2);
    scenes.lfdr = (() => {
      const g = group("lfdr");
      const K = Math.min(byL.length, Math.max(40, Math.min(110, runOf(QHI) + 12))), x0 = 70, w = 460, y0 = 130, h = 330, bw = w / K;
      text(g, x0, 100, `round 0, best ${K} of ${byL.length} candidates by lfdr`, "mono", "start");
      const pre = el("g", {}, g), shade = el("rect", { x: x0, y: y0 - 6, width: 0, height: h + 6, class: "shade" }, pre), runT = text(pre, x0 + 10, y0 + 12, "", "label acc", "start");
      line(g, x0, y0 + h, x0 + w, y0 + h, "", 1);
      for (const v of [0, 0.5, 1]) text(g, x0 - 8, y0 + h - v * h + 4, pct(v), "tiny", "end");
      // each bar is marked real or noise by one fixed draw, noise with the chance its lfdr gives: what a calibrated
      // lfdr means, so the run's share of noise sits near q (an illustration; the draws are not the run's truth)
      const rd = mulberry32(71), real = byL.slice(0, K).map((c) => rd() >= c.lfdr);
      const bars = byL.slice(0, K).map((c, i) => el("rect", { x: x0 + i * bw + 0.5, y: y0 + h - c.lfdr * h, width: Math.max(1, bw - 1), height: Math.max(0.5, c.lfdr * h) }, g));
      let cum = 0; const run = byL.slice(0, K).map((c, i) => { cum += c.lfdr; return [x0 + (i + 0.5) * bw, y0 + h - (cum / (i + 1)) * h]; });
      const runL = stroke(g, "M" + run.map((p) => p[0].toFixed(1) + "," + p[1].toFixed(1)).join(" L"), "", 2);
      const mine = el("g", {}, g);         // the live fit's q, for reference
      line(mine, x0, y0 + h - qLive * h, x0 + w, y0 + h - qLive * h, "soft", 1.2).setAttribute("stroke-dasharray", "2 4");
      text(mine, x0 + 6, y0 + h - qLive * h - 5, `your q ${pct(qLive)}`, "tiny", "start");
      const cut = el("g", {}, g);
      const cutL = line(cut, x0, 0, x0 + w, 0, "warm", 1.4); cutL.setAttribute("stroke-dasharray", "6 4");
      const cutT = text(cut, x0 + w, 0, "", "label warmt halo", "end");
      const cap = text(g, 300, y0 + h + 40, reduced ? "bars: each candidate's lfdr; blue real, gray noise, drawn by that chance" : "bars: each candidate's lfdr; blue real, gray noise, drawn by that chance; q sweeps", "mono");
      let lastK = -1, lastQ = "";
      const show = (q) => {
        const Y = y0 + h - q * h, qs = pct(q);
        cutL.setAttribute("y1", Y); cutL.setAttribute("y2", Y); cutT.setAttribute("y", Y - 8);
        if (qs !== lastQ) { lastQ = qs; cutT.textContent = `q = ${qs}`; setV("q", qs); }
        const k = runOf(q);
        if (k === lastK) return;
        lastK = k; setV("prefix0", String(k));
        bars.forEach((b, i) => { const on = i < k; b.setAttribute("class", real[i] ? "pos" : "noise"); b.setAttribute("opacity", on ? 0.9 : 0.25); });
        const realIn = real.slice(0, Math.min(k, K)).filter(Boolean).length;
        shade.setAttribute("width", Math.min(k, K) * bw);
        runT.setAttribute("x", x0 + Math.min(k, K) * bw + 6);
        runT.textContent = k ? `the run: ${k}, ${k - realIn} noise` : "no run: nothing selected";
        runT.setAttribute("text-anchor", k > 0.7 * K ? "end" : "start");
        if (k > 0.7 * K) runT.setAttribute("x", x0 + Math.min(k, K) * bw - 6);
      };
      return { g, live: !reduced, leave() { lastK = -1; lastQ = ""; show(qLive); }, update(t, s) {
        const n = Math.floor(seg(t, 0, 0.4) * bars.length);
        bars.forEach((b, i) => { b.style.opacity = i < n ? "" : 0; });
        runL.set(seg(t, 0.35, 0.65)); fade(cut, seg(t, 0.3, 0.4)); fade(mine, seg(t, 0.3, 0.4)); fade(pre, seg(t, 0.65, 0.8)); fade(cap, seg(t, 0.1, 0.25));
        show(s == null ? qLive : qAt(s));
        return s != null;
      } };
    })();

    // 8. admission: round 0's selections, admitted or turned back -----------------------------------------------------
    // Time-driven while it is on screen: the selections are taken best first, each lighting the segment it splits,
    // then the new lines grow from its midpoint (the closure's a beat later, one step of the chain at a time), then
    // its dot, or a cross if re-scoring turned it back. The first few go slowly; the rest at about four a second.
    scenes.admit = (() => {
      const g = group("admit");
      const sel = r0.filter((c) => c.selected).sort((a, b) => a.lfdr - b.lfdr || Math.abs(b.z) - Math.abs(a.z));
      el("rect", { x: SQ.x, y: SQ.y, width: SQ.s, height: SQ.s, class: "frame" }, g);
      const head = text(g, SQ.x, SQ.y - 12, "", "mono", "start");
      // the replay, checked against the fit's own cells: every round's selections, in order
      const G0 = Math.round(Math.sqrt(fit.rounds[0].faces / (fit.engine === "rect" ? 1 : 2)));
      let rp = null;
      try {
        const chk = meshReplay(fit, G0);
        if (chk) {
          let ok = true;
          for (const c of [...cands].filter((c) => c.selected).sort((a, b) => a.round - b.round)) if (chk.split(c.x, c.y) === null) ok = false;
          const K = fit.stride, n = fit.tri.length / K, k5 = (v) => v.toFixed(5);
          const norm = (pts) => pts.map((p) => p.map(k5).join(",")).sort().join("|");
          const fin = new Set();
          for (let i = 0; i < n; ++i) fin.add(fit.engine === "rect" ? Array.from(fit.tri.slice(K * i, K * i + 4), k5).join(",") : norm([0, 3, 6].map((o) => [fit.tri[K * i + o], fit.tri[K * i + o + 1]])));
          const mine = chk.cells().map((c) => (fit.engine === "rect" ? c.map(k5).join(",") : norm(c)));
          if (ok && mine.length === n && mine.every((m) => fin.has(m))) rp = meshReplay(fit, G0);
        }
      } catch (e) { rp = null; }
      if (rp) el("path", { d: rp.initial.map((s) => `M${sx(s[0])},${sy(s[1])}L${sx(s[2])},${sy(s[3])}`).join(""), class: "meshline" }, g);
      else meshLines(g, fit);                    // no replay: the mesh the fit ends with, as before
      D.vertices.filter((v) => v.free && !v.admitted).forEach((v) => el("circle", { cx: sx(v.x), cy: sy(v.y), r: 4, class: "ink" }, g));
      // the schedule: step i starts at S[i] seconds and lasts d[i]
      const N = sel.length, d = [], S = [];
      const slow = Math.min(N, 8);
      for (let i = 0; i < slow; ++i) d.push(1.1 * Math.pow(0.88, i));
      const used = d.reduce((a, b) => a + b, 0), rest = clamp((16 - used) / Math.max(1, N - slow), 0.18, 0.45);
      for (let i = slow; i < N; ++i) d.push(rest);
      d.reduce((a, b, i) => { S[i] = a; return a + b; }, 0.4);
      const END = N ? S[N - 1] + d[N - 1] : 0.4;
      const hl = [], splits = [], marks = [];
      const fresh = el("g", {}, g);
      sel.forEach((c, i) => {
        hl.push(c.seg ? line(fresh, sx(c.seg[0]), sy(c.seg[1]), sx(c.seg[2]), sy(c.seg[3]), c.z > 0 ? "posS" : "negS", 2.6) : null);
        const out = rp ? rp.split(c.x, c.y) || [] : [];
        for (const [s, lev] of out) {
          const L = line(fresh, sx(s[0]), sy(s[1]), sx(s[0]), sy(s[1]), lev ? "" : "accent", 1.8);
          splits.push({ L, x0: sx(s[0]), y0: sy(s[1]), x1: sx(s[2]), y1: sy(s[3]), at: S[i] + d[i] * (0.3 + 0.25 * lev), grow: Math.min(0.35, 0.45 * d[i] + 0.1), state: -1 });
        }
        const q = el("g", {}, g);
        if (c.admitted) el("circle", { cx: sx(c.x), cy: sy(c.y), r: 4.5, class: c.z > 0 ? "pos" : "neg" }, q);
        else { const X = sx(c.x), Y = sy(c.y); line(q, X - 5, Y - 5, X + 5, Y + 5, "", 2); line(q, X - 5, Y + 5, X + 5, Y - 5, "", 2); }
        marks.push(q);
      });
      const na = sel.filter((c) => c.admitted).length, closure = rp && fit.engine !== "rect";
      const cap = el("g", {}, g);
      text(cap, 300, SQ.y + SQ.s + 24, !N ? "round 0 selected nothing" : na === N ? `all ${na} admitted (dots)` : `${na} admitted (dots), ${N - na} turned back on re-scoring (crosses)`, "tiny");
      if (rp && N) text(cap, 300, SQ.y + SQ.s + 40, closure ? "blue: the splits; black: neighbors split so no vertex sits mid-edge" : "blue: the splits", "tiny");
      let shown = -1;
      return { g, live: true, update(t, s) {
        const now = s == null ? Infinity : s;
        let k = 0;
        for (let i = 0; i < N; ++i) {
          const a = now - S[i];
          if (a >= d[i] * 0.55) k = i + 1;
          if (hl[i]) { hl[i].style.opacity = a < 0 ? 0 : a < d[i] ? 0.95 : Math.max(0, 0.95 - (a - d[i]) / 0.9); }
          marks[i].style.opacity = a >= d[i] * 0.55 ? 1 : 0;
        }
        for (const q of splits) {
          const a = now - q.at, st = a < 0 ? 0 : a < q.grow ? 1 : a < q.grow + 1.2 ? 2 : 3;
          if (st === q.state && st !== 1 && st !== 2) continue;
          q.state = st;
          if (st === 0) { q.L.style.opacity = 0; continue; }
          const f = st === 1 ? ease(a / q.grow) : 1;
          q.L.setAttribute("x2", q.x0 + (q.x1 - q.x0) * f); q.L.setAttribute("y2", q.y0 + (q.y1 - q.y0) * f);
          if (st === 3) { q.L.setAttribute("class", "meshnew"); q.L.style.opacity = ""; continue; }
          q.L.style.opacity = st === 2 ? 1 - 0.6 * (a - q.grow) / 1.2 : 1;
        }
        if (k !== shown) { shown = k; head.textContent = k < N ? `round 0: ${N} selected, best first · ${k} done` : `round 0: ${N} selected`; }
        fade(cap, (now - END) / 0.8);
        return now < END + 1.6;
      } };
    })();

    // 9. the rounds, side by side ------------------------------------------------------------------------------------
    scenes.rounds = (() => {
      const g = group("rounds");
      const R = rounds.slice(0, 6), cols = R.length > 3 ? 3 : R.length, rows = Math.ceil(R.length / cols);
      const cw = 460 / cols, ch = rows > 1 ? 190 : 300, panels = [];
      R.forEach((r, i) => {
        const q = el("g", {}, g), cx = 70 + (i % cols) * cw, cy = 110 + Math.floor(i / cols) * (ch + 50);
        const zs = byRound(r), c = cal.find((u) => u.round === r);
        text(q, cx, cy - 12, `round ${r}`, "mono", "start");
        histogram(q, zs, c, { x: cx + 4, y: cy, w: cw - 18, h: ch - 74 }, { bare: true, bw: 1 });
        const sel = zs.filter((u) => u.selected).length, adm = zs.filter((u) => u.admitted).length;
        text(q, cx, cy + ch - 48, `${zs.length} scored`, "tiny", "start");
        text(q, cx, cy + ch - 34, `${sel} selected`, "tiny", "start");
        text(q, cx, cy + ch - 20, `${adm} admitted`, "tiny " + (adm ? "acc" : ""), "start");
        if (c) text(q, cx, cy + ch - 6, `null spread ${fmt(c.nullSd)}`, "tiny", "start");
        panels.push(q);
      });
      const cap = text(g, 300, 588, "bars: the scores; dashed: N(0, 1); blue: the round's own null", "tiny");
      return { g, update(t) { panels.forEach((p, i) => fade(p, seg(t, i * 0.1, i * 0.1 + 0.15))); fade(cap, seg(t, 0.5, 0.65)); } };
    })();

    // 10. the two variances ------------------------------------------------------------------------------------
    scenes.variance = (() => {
      const g = group("variance");
      // pool variance by round
      const pv = fit.rounds.map((r) => r.poolVariance).concat([fit.poolVariance]), truth = data.sd * data.sd;
      const A = { x: 80, y: 110, w: 200, h: 230 }, vmax = Math.max(...pv, truth) * 1.1;
      const AX = (i) => A.x + (i / Math.max(1, pv.length - 1)) * A.w, AY = (v) => A.y + A.h - (v / vmax) * A.h;
      const a = el("g", {}, g);
      text(a, A.x - 10, A.y - 30, "coin variance, by round", "mono", "start");
      line(a, A.x, A.y + A.h, A.x + A.w, A.y + A.h, "", 1); line(a, A.x, A.y, A.x, A.y + A.h, "", 1);
      const tl = line(a, A.x, AY(truth), A.x + A.w, AY(truth), "warm", 1.4); tl.setAttribute("stroke-dasharray", "5 4");
      text(a, A.x + A.w, A.y + A.h - 12, `dashed: the true ${data.sd.toFixed(2)}²`, "tiny", "end");
      const pl = stroke(a, "M" + pv.map((v, i) => `${AX(i).toFixed(1)},${AY(v).toFixed(1)}`).join(" L"), "accent", 2.2);
      const pd = pv.map((v, i) => el("circle", { cx: AX(i), cy: AY(v), r: 4, class: "pos" }, a));
      text(a, A.x, A.y + A.h + 16, "start", "tiny", "start"); text(a, A.x + A.w, A.y + A.h + 16, "final", "tiny", "end");
      text(a, A.x - 6, AY(pv[0]) + 4, fmt(pv[0], 3), "tiny", "end"); text(a, A.x + A.w + 6, AY(pv[pv.length - 1]) + 4, fmt(pv[pv.length - 1], 3), "tiny", "start");
      // admitted surpluses by depth, with the prior's ±1 sd
      const V = D.vertices.filter((v) => v.free && v.admitted && v.lambda > 0), depths = [...new Set(V.map((v) => v.depth))].sort((p, q) => p - q);
      const B = { x: 350, y: 110, w: 190, h: 400 }, smax = Math.max(0.5, ...V.map((v) => Math.abs(v.surplus)), ...V.map((v) => 1 / Math.sqrt(v.lambda))) * 1.1;
      const BX = (i) => B.x + ((i + 0.5) / Math.max(1, depths.length)) * B.w, BY = (s) => B.y + B.h / 2 - (s / smax) * (B.h / 2);
      const b = el("g", {}, g);
      text(b, B.x - 10, B.y - 30, "admitted surpluses, by depth", "mono", "start");
      line(b, B.x, BY(0), B.x + B.w, BY(0), "soft", 1);
      const bands = [], dots = [];
      depths.forEach((d, i) => {
        const at = V.filter((v) => v.depth === d), tau = 1 / Math.sqrt(at.map((v) => v.lambda).sort((p, q) => p - q)[at.length >> 1]);
        bands.push(el("rect", { x: BX(i) - 14, y: BY(tau), width: 28, height: BY(-tau) - BY(tau), class: "shade" }, b));
        at.forEach((v, j) => dots.push(el("circle", { cx: BX(i) + ((j % 5) - 2) * 4, cy: BY(v.surplus), r: 3.4, class: v.surplus > 0 ? "pos" : "neg" }, b)));
        text(b, BX(i), B.y + B.h + 16, String(d), "tiny");
      });
      text(b, B.x + B.w / 2, B.y + B.h + 32, "depth", "tiny");
      const cap = text(g, 300, 580, "shaded: the prior's ± one standard deviation at each depth, fitted by EM", "tiny");
      return { g, update(t) {
        fade(a, seg(t, 0, 0.1)); pl.set(seg(t, 0.05, 0.4)); pd.forEach((d, i) => fade(d, seg(t, 0.05 + (0.35 * i) / pd.length, 0.1 + (0.35 * i) / pd.length)));
        fade(b, seg(t, 0.35, 0.45)); dots.forEach((d) => fade(d, seg(t, 0.45, 0.6))); bands.forEach((d) => fade(d, seg(t, 0.6, 0.75))); fade(cap, seg(t, 0.65, 0.8));
      } };
    })();

    // 11. the fit, against the odds -------------------------------------------------------------------------------
    scenes.done = (() => {
      const g = group("done");
      const surf = (x, y) => fitLogit(x, y) - BASE;
      const w = 250, gap = 20, x0 = 300 - w - gap / 2, x1 = 300 + gap / 2, y0 = 150;
      el("image", { x: x0, y: y0, width: w, height: w, href: raster((x, y) => data.tf(x, y) - BASE, 100) }, g);
      el("rect", { x: x0, y: y0, width: w, height: w, class: "frame" }, g);
      const fitImg = el("image", { x: x1, y: y0, width: w, height: w, href: raster(surf, 100) }, g);
      el("rect", { x: x1, y: y0, width: w, height: w, class: "frame" }, g);
      const saved = { ...SQ }; Object.assign(SQ, { x: x1, y: y0, s: w });
      const ml = meshLines(g, fit, "meshline");
      const dots = D.vertices.filter((v) => v.admitted).map((v) => el("circle", { cx: sx(v.x), cy: sy(v.y), r: 2.4, class: "ink" }, g));
      Object.assign(SQ, saved);
      text(g, x0, y0 - 12, "the odds", "mono", "start"); text(g, x1, y0 - 12, "the fit, with its mesh", "mono", "start");
      const cap = text(g, 300, y0 + w + 34, `${D.vertices.filter((v) => v.admitted).length} vertices admitted in ${rounds.length} rounds`, "mono");
      return { g, update(t) { fade(fitImg, seg(t, 0.05, 0.25)); fade(ml, seg(t, 0.25, 0.45)); dots.forEach((d) => fade(d, seg(t, 0.35, 0.5))); fade(cap, seg(t, 0.4, 0.55)); } };
    })();

    // the numbers in the text
    const setV = (k, s) => root.querySelectorAll(`[data-v="${k}"]`).forEach((n) => { n.textContent = s; });
    setV("M0", String(r0.length));
    setV("outside", pct(outside));
    setV("dispFlat", disp(zFlat).toFixed(2)); setV("dispFit", disp(zFit).toFixed(2));
    setV("unscoreable", D.census ? String(D.census.unscoreable) : "some");
    setV("nullMean", fmt(cal0.nullMean)); setV("nullSd", fmt(cal0.nullSd)); setV("pi0", pct(cal0.pi0));
    setV("capnote", fit.engine === "rect" && cal0.nullSd >= 2.999 ? " The spread is at its cap of 3. Without a cap, a round where most candidates carry signal can read the signal as a wide null and admit nothing." : fit.engine === "tri" && cal0.nullSd >= 5.999 ? " The spread is at its cap of 6." : "");
    setV("prefix0", String(r0.filter((c) => c.selected).length));
    setV("naive0", String(r0.filter((c) => Math.abs(c.z) > 1.96).length));
    const sel0 = r0.filter((c) => c.selected), adm0 = sel0.filter((c) => c.admitted).length;
    setV("admit0", !sel0.length ? "Round 0 selected nothing." : adm0 === sel0.length ? `Round 0 selected ${sel0.length}, and all ${adm0} were admitted.`
      : `Round 0 selected ${sel0.length}: ${adm0} admitted, ${sel0.length - adm0} turned back on re-scoring.`);
    setV("nrounds", String(rounds.length));
    // the run's settings, in the text
    const words = { 3000: "Three thousand", 6000: "Six thousand", 12000: "Twelve thousand", 5: "five", 10: "ten", 20: "twenty", 40: "forty", 80: "eighty", 160: "a hundred and sixty" };
    setV("sitesWord", words[data.x.length] || data.x.length.toLocaleString("en-US"));
    setV("flipsWord", words[data.flips] || String(data.flips));
    setV("sites", data.x.length.toLocaleString("en-US")); setV("flips", String(data.flips));
    setV("nlo", String(Math.min(...data.n))); setV("nhi", String(Math.max(...data.n)));
    setV("q", pct(data.q));
    // with its standard error, and the floor: the true surface scored on the same sites (paired gap)
    const H = (() => {
      const h = fit.heldout; if (!h || !h.rows || !h.rows.n.length) return null;
      const R = h.rows, m = R.n.length, f = data.tf, lg = (t) => 1 / (1 + Math.exp(-t));
      const dev = (k, n, p) => { p = Math.min(1 - 1e-6, Math.max(1e-6, p)); return (k > 0 ? 2 * k * Math.log(k / (n * p)) : 0) + (n - k > 0 ? 2 * (n - k) * Math.log((n - k) / (n * (1 - p))) : 0); };
      const a = [], b = [];
      for (let i = 0; i < m; ++i) { a.push(dev(R.k[i], R.n[i], R.p[i])); b.push(dev(R.k[i], R.n[i], lg(f(R.x[i], R.y[i])))); }
      const mean = (v) => v.reduce((s, x) => s + x, 0) / v.length, se = (v, mu) => Math.sqrt(v.reduce((s, x) => s + (x - mu) ** 2, 0) / (v.length - 1) / v.length);
      const d = a.map((x, i) => x - b[i]);
      return { fit: mean(a), fitSe: se(a, mean(a)), floor: mean(b), floorSe: se(b, mean(b)), gap: mean(d), gapSe: se(d, mean(d)), m };
    })();
    setV("heldout", H ? `mean deviance ${H.fit.toFixed(3)} ± ${H.fitSe.toFixed(3)} per site over ${H.m.toLocaleString()} sites the fit never saw; `
      + `the true surface scores ${H.floor.toFixed(3)} ± ${H.floorSe.toFixed(3)} on them, so the fit is ${H.gap.toFixed(3)} ± ${H.gapSe.toFixed(3)} above the floor`
      : fit.heldout ? `mean deviance ${fit.heldout.deviance.toFixed(3)} per site over ${fit.heldout.pools.toLocaleString()} sites the fit never saw` : "no held-out sites");
    active = null; measure();
  }

  // ------------------------------------------------------------------ the run: the live fit's, from gate.js
  let data = null;
  function status(s, busy) { const n = $("gh-status"); n.textContent = s; n.classList.toggle("busy", !!busy); }
  function show(r) {
    if (!r || !r.fit || !r.fit.detail) return false;
    run = r.fit;
    data = { ...r.data, tf: r.tf, sd: r.sd, q: r.q, flips: r.flips };
    svg.style.opacity = 1;
    status(`${run.engine === "rect" ? "rectangles" : "right triangles"} · ${(run.ms / 1000).toFixed(1)} s in your browser`);
    root.dataset.engine = run.engine;
    build(data, run);
    return true;
  }
  const G = window.gateDemo;
  document.addEventListener("gatefitstart", (ev) => {
    if (ev.detail.user) return;                       // a visitor's file: the story stays on the last coins
    status("Fitting…", true);
    if (svg.firstChild && !svg.querySelector(".stagebusy")) svg.style.opacity = 0.35;
  });
  document.addEventListener("gatefit", (ev) => {
    const d = ev.detail;
    if (d.error) { svg.style.opacity = 1; status("the fit failed: " + d.error); return; }
    if (d.user) { svg.style.opacity = 1; status(run ? "the story follows the coins: back to them above to refit it" : "the story follows the coins, not a file"); return; }
    show(d);
  });
  new MutationObserver(() => { if (run) build(data, run); }).observe(document.documentElement, { attributes: true, attributeFilter: ["class"] });

  // ------------------------------------------------------------------ scroll to scene (the page's own reading line
  // is data-auto, drawn by whimsy.js)
  // A step's progress runs from 0 as it becomes the active step (its top at the reading line) to 1 once 70% of it has
  // passed the line, so its drawing builds while the paragraph is read, on the way down as well as up.
  function measure() {
    const vh = window.innerHeight, line = window.readLine ? window.readLine() : vh * 0.55;
    let best = null, bestD = Infinity;
    for (const s of steps) { const r = s.getBoundingClientRect(), d = Math.abs(r.top + r.height / 2 - line); if (d < bestD) { bestD = d; best = s; } }
    if (!best) return;
    const r = best.getBoundingClientRect();
    prog = clamp((line - r.top) / (r.height * 0.7));
    if (best !== active) {
      const was = active && scenes[active.dataset.scene];
      if (was && was.leave) was.leave();
      // reached from above, a drawing builds from nothing; from below, or rebuilt in place (active reset), it is there
      shown = active && steps.indexOf(best) > steps.indexOf(active) ? 0 : prog; lastDraw = null;
      active = best; clock = 0; lastNow = null;     // a scene that moves by itself starts over each time it is reached
      for (const s of steps) s.classList.toggle("on", s === best);
      for (const [name, sc] of Object.entries(scenes)) sc.g.style.opacity = name === best.dataset.scene ? 1 : 0;
    }
    queueDraw();
  }
  window.addEventListener("scroll", measure, { passive: true });
  window.addEventListener("resize", measure);
  // A scene is redrawn only when the reading position or the scene changes, except the two that move by themselves
  // (live: the admissions and the sweeping q), which run on their own clock, counted only while the story is on
  // screen, and stop asking for frames when they are finished or scrolled away. With reduced motion every scene is
  // drawn once, finished (the sweep at the live fit's q), and left alone until another takes its place.
  const storyEl = document.getElementById("story");
  // The drawing follows the reading position no faster than PACE a second, so a flick of the wheel is caught up
  // over a moment rather than skipped.
  const PACE = 0.9;
  let drawnScene = null, drawnProg = -1, queued = false, clock = 0, lastNow = null, shown = 0, lastDraw = null;
  function redraw(now) {
    queued = false;
    const sc = active && scenes[active.dataset.scene];
    if (!sc) return;
    const dt = lastDraw === null ? 0 : Math.min(0.05, (now - lastDraw) / 1000);
    shown = reduced ? 1 : Math.abs(prog - shown) <= PACE * dt ? prog : shown + Math.sign(prog - shown) * PACE * dt;
    const chasing = !reduced && shown !== prog;
    lastDraw = chasing ? now : null;
    if (sc.live && !reduced) {
      const r = storyEl.getBoundingClientRect();
      if (r.top >= window.innerHeight || r.bottom <= 0) { lastNow = null; lastDraw = null; return; }   // paused; a scroll back resumes it
      if (lastNow !== null) clock += Math.min(100, Math.max(0, now - lastNow));
      drawnScene = sc; drawnProg = -1;
      if (sc.update(shown, clock / 1000) || chasing) { lastNow = now; queued = true; requestAnimationFrame(redraw); }
      else lastNow = null;
      return;
    }
    if (chasing) queueDraw();
    if (sc === drawnScene && shown === drawnProg) return;
    drawnScene = sc; drawnProg = shown;
    sc.update(shown, null);
  }
  function queueDraw() { if (!queued) { queued = true; requestAnimationFrame(redraw); } }
  if (!(G && G.run && show(G.run))) {
    if (G && G.user) status("the story follows the coins, not a file");
    else { status("Fitting…", true); text(svg, 300, 300, "fitting…", "stagebusy"); }
  }
  measure();
  // On a phone the drawing's labels are set larger (the page's stylesheet). A label that would then run past the
  // drawing's edge, 600 units wide, is shrunk back until it fits, measured from where it is anchored.
  const phone = window.matchMedia("(max-width: 820px)");
  let fitQueued = false;
  function fitLabels() {
    fitQueued = false;
    const mode = phone.matches ? "phone" : "wide";
    for (const t of svg.querySelectorAll("text")) {
      const key = mode + t.textContent;
      if (t.fitKey === key) continue;           // measured already, for this text at this width
      t.fitKey = key; t.style.fontSize = "";
      if (mode !== "phone" || !t.textContent) continue;
      const b = t.getBBox(), x = +t.getAttribute("x") || 0, a = t.getAttribute("text-anchor") || "start";
      const room = a === "middle" ? 2 * Math.min(x, 600 - x) : a === "end" ? x : 600 - x;
      if (b.width > room && room > 0) t.style.fontSize = (parseFloat(getComputedStyle(t).fontSize) * room / b.width).toFixed(2) + "px";
    }
  }
  const queueFit = () => { if (!fitQueued) { fitQueued = true; requestAnimationFrame(fitLabels); } };
  new MutationObserver(queueFit).observe(svg, { childList: true, subtree: true, characterData: true });
  phone.addEventListener("change", queueFit);
})();
