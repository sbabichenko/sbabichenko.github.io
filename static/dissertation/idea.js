// /dissertation/idea: from beliefs about beliefs to the noise-state. Everything drawn is computed here, from one small
// problem in discrete time: a state X_k = ρ X_{k-1} + w_k, and two players, player i seeing y^i_k = X_k + σ_i e^i_k.
// A player's noise-state is the exact Gaussian estimate of the shock path w given their signals so far; their
// forecast of X is the kernel of X run against it; player 1's forecast of player 2's forecast is the kernel of
// player 2's forecast run against player 1's noise-state. The page checks those against direct computations.
(function () {
  "use strict";
  const NS = "http://www.w3.org/2000/svg";
  const svg = document.getElementById("stage");
  const steps = [...document.querySelectorAll(".step")];
  if (!svg || !steps.length) return;
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // ------------------------------------------------------------------ the model
  const N = 48, RHO = 0.92, SIG = [1.2, 2.0];
  function mulberry32(a) { return function () { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
  function gauss(r) { let u = 0; while (u === 0) u = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r()); }
  const rng = mulberry32(20260924);
  const w = Array.from({ length: N }, () => gauss(rng));
  const e = [Array.from({ length: N }, () => gauss(rng)), Array.from({ length: N }, () => gauss(rng))];
  const K = (k, j) => (j <= k ? Math.pow(RHO, k - j) : 0);          // the kernel of X: L(t, s) on the grid
  const X = Array.from({ length: N }, (_, k) => { let s = 0; for (let j = 0; j <= k; ++j) s += K(k, j) * w[j]; return s; });
  const Y = [0, 1].map((i) => X.map((x, k) => x + SIG[i] * e[i][k]));

  // small dense solves: S z = b for symmetric positive definite S (Cholesky)
  function cholSolve(S, b) {
    const n = b.length, L = S.map((r) => r.slice());
    for (let i = 0; i < n; ++i) for (let j = 0; j <= i; ++j) {
      let s = L[i][j]; for (let k = 0; k < j; ++k) s -= L[i][k] * L[j][k];
      L[i][j] = i === j ? Math.sqrt(s) : s / L[j][j];
    }
    const z = b.slice();
    for (let i = 0; i < n; ++i) { for (let k = 0; k < i; ++k) z[i] -= L[i][k] * z[k]; z[i] /= L[i][i]; }
    for (let i = n - 1; i >= 0; --i) { for (let k = i + 1; k < n; ++k) z[i] -= L[k][i] * z[k]; z[i] /= L[i][i]; }
    return z;
  }
  // G[i][k]: the map from player i's signals y_0..y_k to their estimate of the shocks w_0..w_{N-1} (zero past k)
  // ŵ = K_{0:k}^T S^{-1} y_{0:k},  S = K_{0:k} K_{0:k}^T + σ² I
  const G = [0, 1].map((i) => {
    const out = [];
    for (let k = 0; k < N; ++k) {
      const S = [];
      for (let a = 0; a <= k; ++a) { S.push([]); for (let b = 0; b <= k; ++b) { let s = 0; for (let j = 0; j <= Math.min(a, b); ++j) s += K(a, j) * K(b, j); S[a].push(s + (a === b ? SIG[i] * SIG[i] : 0)); } }
      // columns of S^{-1}: G_k[u][m] = Σ_a K(a,u) (S^{-1})_{a m}
      const Sinv = [];
      for (let m = 0; m <= k; ++m) { const unit = Array(k + 1).fill(0); unit[m] = 1; Sinv.push(cholSolve(S, unit)); }
      const Gk = [];
      for (let u = 0; u < N; ++u) { const row = Array(k + 1).fill(0); if (u <= k) for (let m = 0; m <= k; ++m) { let s = 0; for (let a = u; a <= k; ++a) s += K(a, u) * Sinv[m][a]; row[m] = s; } Gk.push(row); }
      out.push(Gk);
    }
    return out;
  });
  // the noise-states: Ŵ[i][k][u] = E[w_u | y^i_0..y^i_k]
  const What = [0, 1].map((i) => G[i].map((Gk, k) => Gk.map((row) => row.reduce((s, g, m) => s + g * Y[i][m], 0))));
  // forecasts from the kernel: X̂^i_k = Σ_u L(k,u) Ŵ^i_k(u)
  const Xhat = [0, 1].map((i) => X.map((_, k) => { let s = 0; for (let u = 0; u <= k; ++u) s += K(k, u) * What[i][k][u]; return s; }));
  // the same forecasts from a Kalman filter, as the check
  const kalman = (i) => { const out = []; let xp = 0, Pp = 1; for (let k = 0; k < N; ++k) { const g = Pp / (Pp + SIG[i] * SIG[i]); const xh = xp + g * (Y[i][k] - xp); out.push(xh); const P = (1 - g) * Pp; xp = RHO * xh; Pp = RHO * RHO * P + 1; } return out; };
  const checkFilter = Math.max(...[0, 1].map((i) => Math.max(...kalman(i).map((v, k) => Math.abs(v - Xhat[i][k])))));
  // player 2's forecast as a kernel on the shocks: A2[k][u] = Σ_{a≤k} (Σ_v L(k,v) G2_k[v][a]) L(a,u), plus a part on e²
  const A2 = X.map((_, k) => {
    const c = Array(k + 1).fill(0);                              // c_a = Σ_v L(k,v) G2_k[v][a]
    for (let v = 0; v <= k; ++v) for (let a = 0; a <= k; ++a) c[a] += K(k, v) * G[1][k][v][a];
    return Array.from({ length: N }, (_, u) => { let s = 0; for (let a = u; a <= k; ++a) s += c[a] * K(a, u); return s; });
  });
  // player 1's forecast of player 2's forecast: the kernel A2 run against player 1's noise-state
  const X12 = X.map((_, k) => A2[k].reduce((s, a, u) => s + a * What[0][k][u], 0));
  // checked by conditioning X̂²_k on player 1's signals directly: Cov(X̂²_k, y¹) S1^{-1} y¹, Cov = A2_k K^T
  const checkTower = Math.max(...X.map((_, k) => {
    const S = [], cov = [];
    for (let a = 0; a <= k; ++a) { S.push([]); for (let b = 0; b <= k; ++b) { let s = 0; for (let j = 0; j <= Math.min(a, b); ++j) s += K(a, j) * K(b, j); S[a].push(s + (a === b ? SIG[0] * SIG[0] : 0)); }
      let c = 0; for (let u = 0; u <= a; ++u) c += A2[k][u] * K(a, u); cov.push(c); }
    const z = cholSolve(S, Y[0].slice(0, k + 1));
    return Math.abs(cov.reduce((s, c, a) => s + c * z[a], 0) - X12[k]);
  }));
  window.ideaChecks = { filter: checkFilter, tower: checkTower };

  // ------------------------------------------------------------------ drawing helpers
  const el = (tag, attrs, parent) => { const n = document.createElementNS(NS, tag); for (const [k, v] of Object.entries(attrs || {})) n.setAttribute(k, v); if (parent) parent.appendChild(n); return n; };
  const clamp = (x, a = 0, b = 1) => Math.max(a, Math.min(b, x));
  const ease = (t) => (t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2);
  const seg = (t, a, b) => ease(clamp((t - a) / (b - a)));
  const fade = (n, t) => { n.style.opacity = clamp(t); };
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
  function text(g, x, y, s, cls, anchor = "middle") { const t = el("text", { x, y, class: cls || "", "text-anchor": anchor }, g); t.textContent = s; return t; }
  const TX = (k) => 60 + (490 * k) / (N - 1);
  function formula(key, x, y, width, parent, anchor = "start") {
    const src = window.IDEA_FX && window.IDEA_FX[key];
    if (!src) return el("g", {}, parent);
    const doc = new DOMParser().parseFromString(src, "image/svg+xml").documentElement;
    const n = document.importNode(doc, true);
    const vb = (n.getAttribute("viewBox") || "0 0 1 1").split(" ").map(Number), h = (width * vb[3]) / vb[2];
    n.setAttribute("width", width); n.setAttribute("height", h);
    n.setAttribute("x", anchor === "middle" ? x - width / 2 : x); n.setAttribute("y", y - h / 2);
    n.removeAttribute("style");
    n.style.color = "var(--ink)";
    parent.appendChild(n);
    return n;
  }
  function bars(g, vals, base, scale, cls, width = 6) {
    return vals.map((v, k) => el("rect", { x: TX(k) - width / 2, y: v >= 0 ? base - v * scale : base, width, height: Math.abs(v) * scale, class: cls }, g));
  }
  const scenes = {};
  const group = (name) => { const g = el("g", { "data-scene": name }, svg); g.style.opacity = 0; g.style.transition = "opacity 0.6s"; return g; };

  // ------------------------------------------------------------------ 0. a few researchers and a telescope
  function person(g, x, y, cls) {
    const p = el("g", { transform: `translate(${x} ${y})` }, g);
    el("circle", { cx: 0, cy: -13, r: 7, class: cls }, p);
    el("path", { d: "M-11,12 Q-11,-3 0,-3 Q11,-3 11,12 Z", class: cls }, p);
    return p;
  }
  // a head in profile, facing right (or left with flip), with a circle where its model of the world lives
  const HEAD = "M-24,60 C-24,48 -30,40 -38,30 C-50,14 -52,-12 -44,-32 C-34,-56 -8,-66 16,-60 C38,-54 50,-36 48,-12 L47,-4 L57,12 C59,15 56,17 52,17 L51,23 L47,26 L50,30 C50,34 47,36 45,37 C46,45 40,49 30,47 L22,46 L22,60";
  function head(g, x, y, s, flip, name) {
    const h = el("g", { transform: `translate(${x} ${y}) scale(${flip ? -s : s} ${s})` }, g);
    el("path", { d: HEAD + " Z", class: "headfill" }, h);
    el("path", { d: HEAD, class: "pencil", "stroke-width": 1.8 / s, fill: "none" }, h);
    const brain = el("circle", { cx: 2, cy: -18, r: 30, class: "pencil soft", "stroke-width": 1.2 / s, fill: "none" }, h);
    if (name) { const t = text(g, x, y + 60 * s + 24, name, "label"); return { h, t, brain }; }
    return { h, brain };
  }

  // ------------------------------------------------------------------ 00. one person knows: the world goes on
  scenes.doctor = (() => {
    const g = group("doctor");
    const crowd = [];
    for (let i = 0; i < 8; ++i) for (let j = 0; j < 4; ++j) {
      const x = 85 + i * 62, y = 280 + j * 74, k = i === 3 && j === 0;
      const p = person(g, x, y, "fillacc"); p.style.fill = k ? "var(--accent)" : "currentColor";
      crowd.push({ p, x, y, k, ph: (i * 7 + j * 3) % 5 });
    }
    // the one who knows: a calendar over their head, the months crossed off
    const me = crowd.find((c) => c.k);
    const cal = el("g", { transform: `translate(${me.x} ${me.y - 100})` }, g);
    el("rect", { x: -54, y: -30, width: 108, height: 62, rx: 6, class: "box pencil", "stroke-width": 1.4 }, cal);
    el("path", { d: `M${me.x - 6 - me.x},32 L0,62 L8,32`, class: "box pencil", "stroke-width": 1.4 }, cal);
    const months = [];
    for (let m = 0; m < 6; ++m) {
      const cx = -36 + m * 14.4;
      el("rect", { x: cx - 5, y: -6, width: 10, height: 12, class: "pencil soft", "stroke-width": 1, fill: "none" }, cal);
      months.push(el("path", { d: `M${cx - 6},-7 L${cx + 6},7`, class: "pencil accent", "stroke-width": 1.8 }, cal));
    }
    text(cal, 0, -12, "six months", "mono");
    const cap = text(g, 300, 585, "one person knows; the world goes on", "mono");
    return { g, update(t, now) {
      crowd.forEach((c) => {
        fade(c.p, seg(t, 0, 0.12) * (c.k ? 1 : 0.4));
        // everyone else keeps walking about their business
        const dx = c.k ? 0 : Math.sin(now / 1100 + c.ph) * 6, dy = c.k ? 0 : Math.abs(Math.sin(now / 280 + c.ph)) * -2;
        c.p.setAttribute("transform", `translate(${c.x + dx} ${c.y + dy})`);
      });
      fade(cal, seg(t, 0.15, 0.3));
      months.forEach((m, i) => fade(m, seg(t, 0.3 + i * 0.07, 0.36 + i * 0.07)));
      fade(cap, seg(t, 0.7, 0.85));
    } };
  })();

  // ------------------------------------------------------------------ 00b. everyone knows: pandemonium
  scenes.sun = (() => {
    const g = group("sun");
    const sun = el("g", { transform: "translate(300 150)" }, g);
    const halo = el("circle", { r: 118, class: "pencil warm", "stroke-width": 1, fill: "none", opacity: 0.35 }, sun);
    const star = (n, r0, r1) => { let d = ""; for (let k = 0; k < 2 * n; ++k) { const a = (k * Math.PI) / n, r = k % 2 ? r0 : r1; d += (k ? "L" : "M") + (r * Math.cos(a)).toFixed(1) + "," + (r * Math.sin(a)).toFixed(1); } return d + "Z"; };
    const outer = el("path", { d: star(15, 52, 88), class: "fillwarm pencil warm", "stroke-width": 1.4 }, sun);
    const inner = el("path", { d: star(12, 36, 60), class: "sunmid" }, sun);
    el("circle", { r: 26, class: "suncore" }, sun);
    const r = mulberry32(11), crowd = [];
    for (let i = 0; i < 9; ++i) for (let j = 0; j < 4; ++j) {
      const x = 70 + i * 57, y = 330 + j * 60;
      const p = person(g, x, y, "fillacc"); p.style.fill = "var(--accent)";
      crowd.push({ p, x, y, jx: (r() - 0.5) * 40, jy: (r() - 0.5) * 30, ja: (r() - 0.5) * 70, ph: r() * 6 });
    }
    const cap = text(g, 300, 585, "everyone knows; everyone's plans change at once", "mono");
    return { g, update(t, now) {
      fade(sun, seg(t, 0, 0.15));
      const pulse = 1 + 0.05 * Math.sin(now / 160) + 0.25 * seg(t, 0.3, 0.7);
      outer.setAttribute("transform", `rotate(${now / 90 % 360}) scale(${pulse})`);
      inner.setAttribute("transform", `rotate(${-now / 60 % 360}) scale(${pulse})`);
      halo.setAttribute("r", 118 * pulse);
      const chaos = seg(t, 0.25, 0.75);
      crowd.forEach((c) => {
        fade(c.p, seg(t, 0.05, 0.2));
        const wob = Math.sin(now / 200 + c.ph);
        c.p.setAttribute("transform", `translate(${c.x + chaos * (c.jx + wob * 6)} ${c.y + chaos * c.jy}) rotate(${chaos * (c.ja + wob * 12)})`);
      });
      fade(cap, seg(t, 0.7, 0.85));
    } };
  })();

  scenes.telescope = (() => {
    const g = group("telescope");
    const sun = el("g", { transform: "translate(470 110)" }, g);
    for (let a = 0; a < 8; ++a) { const t = (a * Math.PI) / 4; el("line", { x1: 30 * Math.cos(t), y1: 30 * Math.sin(t), x2: 44 * Math.cos(t), y2: 44 * Math.sin(t), class: "pencil warm", "stroke-width": 2 }, sun); }
    el("circle", { r: 24, class: "fillwarm" }, sun);
    const scope = el("g", { transform: "translate(150 190)" }, g);
    el("path", { d: "M0,0 L-14,40 M0,0 L14,40 M0,0 L2,42", class: "pencil", "stroke-width": 1.8 }, scope);
    el("path", { d: "M-6,4 L74,-44 L82,-32 L2,16 Z", class: "box pencil", "stroke-width": 1.8 }, scope);
    const sight = el("line", { x1: 230, y1: 150, x2: 440, y2: 118, class: "pencil soft", "stroke-width": 1, "stroke-dasharray": "3 5" }, g);
    const crowd = [], know = new Set(["3,1", "2,1", "4,1", "3,2", "3,0"]);
    for (let i = 0; i < 8; ++i) for (let j = 0; j < 3; ++j) {
      const x = 85 + i * 62, y = 330 + j * 72, k = know.has(`${i},${j}`);
      const p = person(g, x, y, "fillacc"); p.style.fill = k ? "var(--accent)" : "currentColor";
      crowd.push({ p, x, y, k, d: Math.hypot(i - 3, (j - 1) * 1.15) });
    }
    const rings = [0, 1, 2].map(() => el("circle", { cx: 271, cy: 402, class: "pencil accent", "stroke-width": 1.2 }, g));
    const cap = text(g, 300, 585, "five know; the rest find out from what they do", "mono");
    return { g, update(t, now) {
      fade(sun, seg(t, 0, 0.1)); fade(scope, seg(t, 0.02, 0.12)); fade(sight, seg(t, 0.08, 0.18) * 0.8);
      const spread = seg(t, 0.35, 0.95) * 2.4;          // word spreads, but slowly: the far edge never hears
      crowd.forEach((c) => {
        if (c.k) { fade(c.p, seg(t, 0.1, 0.2)); return; }
        const learned = clamp(spread - c.d + 1);
        c.p.style.opacity = seg(t, 0.05, 0.15) * (0.28 + 0.5 * learned);
        c.p.style.fill = learned > 0.5 ? "var(--accent)" : "currentColor";
      });
      rings.forEach((r, i) => { const ph = ((now / 2400 + i / 3) % 1); r.setAttribute("r", 30 + ph * 200); fade(r, seg(t, 0.3, 0.4) * (1 - ph) * 0.5); });
      fade(cap, seg(t, 0.6, 0.75));
    } };
  })();

  // ------------------------------------------------------------------ 0b. many hands on the state
  scenes.hands = (() => {
    const g = group("hands");
    const l1 = el("g", {}, g), l2 = el("g", {}, g), l3 = el("g", {}, g);
    formula("hands1", 300, 170, 470, l1, "middle");
    text(l2, 300, 260, "each applies the single-controller rule", "mono");
    formula("hands2", 300, 300, 190, l2, "middle");
    formula("hands3", 300, 410, 490, l3, "middle");
    const ul = [];
    for (const [x, w2] of [[300 - 490 / 2 + 205, 70], [300 - 490 / 2 + 300, 70]]) ul.push(stroke(g, pencil([[x - w2 / 2, 440], [x + w2 / 2, 441]], x, 1), "warm", 2.2));
    const cap = text(g, 300, 500, "the state now moves with everyone's estimates", "label warmt");
    return { g, update(t) {
      fade(l1, seg(t, 0, 0.15)); fade(l2, seg(t, 0.2, 0.35)); fade(l3, seg(t, 0.4, 0.55));
      ul.forEach((u) => u.set(seg(t, 0.6, 0.75))); fade(cap, seg(t, 0.7, 0.85));
    } };
  })();

  // ------------------------------------------------------------------ 1. the tower
  scenes.tower = (() => {
    const g = group("tower");
    const p1 = el("g", {}, g), p2 = el("g", {}, g);
    head(p1, 110, 500, 0.62, false, "Alice (1)"); head(p2, 490, 500, 0.62, true, "Bob (2)");
    const levels = ["t1", "t2", "t3", "t4"].map((k, i) => {
      const lg = el("g", {}, g), y = 430 - i * 92, w = 150 + i * 62, x = 300 + (i % 2 ? 30 : -30);
      el("rect", { x: x - w / 2 - 18, y: y - 32, width: w + 36, height: 64, rx: 30, class: "box pencil", "stroke-width": 1.4 }, lg);
      formula(k, x, y, w, lg, "middle");
      // the thought dots from player 1
      for (let d = 0; d < 3; ++d) el("circle", { cx: 120 + (x - w / 2 - 120) * (0.25 + d * 0.22), cy: 455 - (455 - y - 32) * (0.25 + d * 0.22), r: 3 + d, class: "pencil soft", "stroke-width": 1.2 }, lg);
      return lg;
    });
    const counts = ["1", "n", "n²", "n³"].map((c, i) => text(g, 28, 436 - i * 92, c, "label acc", "start"));
    const dots = text(g, 300, 58, "…", "label"); dots.style.fontSize = "34px";
    const cap = text(g, 300, 585, "forecasting the forecasts of others, without end", "mono");
    return { g, update(t, now) {
      fade(p1, seg(t, 0, 0.1)); fade(p2, seg(t, 0, 0.1));
      levels.forEach((l, i) => { fade(l, seg(t, 0.08 + i * 0.14, 0.2 + i * 0.14)); l.setAttribute("transform", `translate(0 ${Math.sin(now / 900 + i) * 3})`); });
      counts.forEach((c, i) => fade(c, seg(t, 0.14 + i * 0.14, 0.24 + i * 0.14)));
      fade(dots, seg(t, 0.66, 0.8) * (0.6 + 0.4 * Math.sin(now / 300))); fade(cap, seg(t, 0.75, 0.9));
    } };
  })();

  // ------------------------------------------------------------------ 1b. beliefs about randomness
  scenes.randomness = (() => {
    const g = group("randomness");
    const heads = [[160, 215, "Alice", false], [440, 215, "Bob", true]].map(([x, y, l, f], i) => {
      const h = el("g", {}, g); head(h, x, y, 1, f, l);
      // inside each head, a small model of the other one, which the noise-state lets them throw away
      const bx = x + (f ? -2 : 2), by = y - 18, mini = el("g", {}, g);
      head(mini, bx, by + 2, 0.3, !f);
      const cross = [stroke(g, pencil([[bx - 18, by - 18], [bx + 18, by + 18]], 90 + i, 0.8), "warm", 2.2), stroke(g, pencil([[bx + 18, by - 18], [bx - 18, by + 18]], 94 + i, 0.8), "warm", 2.2)];
      return { h, x, y, mini, cross };
    });
    // the interface between them: all either can see of the other
    const iface = el("g", {}, g);
    el("line", { x1: 300, y1: 130, x2: 300, y2: 300, class: "pencil soft", "stroke-width": 1.2, "stroke-dasharray": "4 5" }, iface);
    text(iface, 300, 120, "interface", "mono");
    // the nested models, each struck out
    const models = [["Alice's model of Bob", 150, 110], ["Bob's model of Alice", 450, 110], ["Alice's model of Bob's model of Alice", 150, 60], ["Bob's model of Alice's model of Bob", 450, 60]].map(([s2, x, y], i) => {
      const m = el("g", {}, g); const tt = text(m, x, y, s2, "label"); tt.style.fontSize = "14px";
      const w2 = Math.min(270, s2.length * 6.4);
      const strike = stroke(m, pencil([[x - w2 / 2 - 4, y - 3], [x + w2 / 2 + 4, y - 7]], 70 + i, 1), "warm", 2);
      return { m, strike };
    });
    const oval = el("g", {}, g);
    el("ellipse", { cx: 300, cy: 440, rx: 170, ry: 48, class: "pencil warm", "stroke-width": 2 }, oval);
    el("ellipse", { cx: 300, cy: 440, rx: 170, ry: 48, class: "fillwarm", opacity: 0.1 }, oval);
    text(oval, 300, 446, "sources of randomness", "label warmt");
    // a few shocks jiggling inside it
    const r = mulberry32(3), kicks = [];
    for (let k = 0; k < 18; ++k) kicks.push({ x: 160 + k * 16.5, h: gauss(r) * 16, el: el("line", { x1: 160 + k * 16.5, x2: 160 + k * 16.5, y1: 470, y2: 470, class: "pencil warm", "stroke-width": 2 }, oval) });
    const arrows = heads.map((h, i) => stroke(g, pencil([[300 + (i ? 60 : -60), 392], [h.x + (i ? -8 : 8), 330 - 40 - 10 + 60], [h.x + (i ? -2 : 2), h.y + 12]], 80 + i, 2), "warm", 1.8));
    const cap = text(g, 300, 560, "both estimate the same thing: the shocks", "mono");
    return { g, update(t, now) {
      heads.forEach((h) => fade(h.h, seg(t, 0, 0.1))); fade(iface, seg(t, 0.02, 0.12) * (1 - 0.6 * seg(t, 0.45, 0.6)));
      heads.forEach((h, i) => { fade(h.mini, seg(t, 0.04, 0.14) * (1 - 0.6 * seg(t, 0.4, 0.55))); h.cross.forEach((c) => c.set(seg(t, 0.3 + i * 0.05, 0.4 + i * 0.05))); });
      models.forEach((m, i) => { fade(m.m, seg(t, 0.02 + i * 0.04, 0.12 + i * 0.04) * (1 - 0.55 * seg(t, 0.45, 0.6))); m.strike.set(seg(t, 0.25 + i * 0.05, 0.35 + i * 0.05)); });
      fade(oval, seg(t, 0.45, 0.6));
      kicks.forEach((k, i) => { const hh = k.h * (0.6 + 0.4 * Math.sin(now / 300 + i)); k.el.setAttribute("y2", 470 - Math.abs(hh)); k.el.setAttribute("y1", 470); });
      arrows.forEach((a) => a.set(seg(t, 0.6, 0.8))); fade(cap, seg(t, 0.8, 0.92));
    } };
  })();

  // ------------------------------------------------------------------ 2. shocks and the state they move
  scenes.shocks = (() => {
    const g = group("shocks");
    el("line", { x1: 50, y1: 470, x2: 560, y2: 470, class: "pencil soft", "stroke-width": 1 }, g);
    el("line", { x1: 50, y1: 240, x2: 560, y2: 240, class: "pencil soft", "stroke-width": 1 }, g);
    const b = bars(g, w, 470, 22, "fillwarm");
    const xp = stroke(g, pencil(X.map((v, k) => [TX(k), 240 - v * 22]), 3), "", 2);
    const lw = text(g, 50, 540, "W: the primitive shocks", "label warmt", "start");
    const lx = text(g, 50, 110, "X: what they push around", "label", "start");
    return { g, update(t) {
      b.forEach((r, k) => fade(r, seg(t, 0.02 + (k / N) * 0.5, 0.06 + (k / N) * 0.5) * 0.8));
      xp.set(seg(t, 0.08, 0.62)); fade(lw, seg(t, 0.05, 0.2)); fade(lx, seg(t, 0.3, 0.45));
    } };
  })();

  // ------------------------------------------------------------------ 3. one shock and its impulse response
  scenes.irf = (() => {
    const g = group("irf"), s = 10;
    el("line", { x1: 50, y1: 470, x2: 560, y2: 470, class: "pencil soft", "stroke-width": 1 }, g);
    el("line", { x1: 50, y1: 360, x2: 560, y2: 360, class: "pencil soft", "stroke-width": 1 }, g);
    const kick = el("rect", { x: TX(s) - 3, y: 470 - 60, width: 6, height: 60, class: "fillwarm" }, g);
    const pts = []; for (let k = s; k < N; ++k) pts.push([TX(k), 360 - 180 * K(k, s)]);
    const curve = stroke(g, pencil(pts, 5, 0.3), "accent", 2.6);
    const sl = text(g, TX(s), 500, "s", "label warmt");
    const probe = el("g", {}, g), tk = 30;
    el("line", { x1: TX(tk), y1: 360, x2: TX(tk), y2: 360 - 180 * K(tk, s), class: "pencil accent", "stroke-width": 1.2, "stroke-dasharray": "3 4" }, probe);
    text(probe, TX(tk), 385, "t", "label acc");
    text(probe, TX(tk) + 10, 360 - 180 * K(tk, s) - 12, "L(t, s)", "label acc", "start");
    const cap = text(g, 300, 570, "the impulse response: what is left of the shock at s by time t", "mono");
    return { g, update(t) { fade(kick, seg(t, 0, 0.1)); fade(sl, seg(t, 0, 0.1)); curve.set(seg(t, 0.1, 0.55)); fade(probe, seg(t, 0.5, 0.65)); fade(cap, seg(t, 0.6, 0.75)); } };
  })();

  // ------------------------------------------------------------------ 4. responses add up to the path
  scenes.superpose = (() => {
    const g = group("superpose");
    el("line", { x1: 50, y1: 500, x2: 560, y2: 500, class: "pencil soft", "stroke-width": 1 }, g);
    el("line", { x1: 50, y1: 260, x2: 560, y2: 260, class: "pencil soft", "stroke-width": 1 }, g);
    const b = bars(g, w, 500, 16, "fillwarm");
    const copies = w.map((v, j) => {
      const pts = []; for (let k = j; k < N; ++k) pts.push([TX(k), 260 - 22 * v * K(k, j)]);
      return el("path", { d: pencil(pts.length > 1 ? pts : [pts[0], pts[0]], 40 + j, 0.2), class: "pencil", "stroke-width": 1 }, g);
    });
    const truth = el("path", { d: pencil(X.map((v, k) => [TX(k), 260 - 22 * v]), 9, 0.1), class: "pencil soft", "stroke-width": 1.2, "stroke-dasharray": "4 5" }, g);
    const sum = el("path", { class: "pencil accent", "stroke-width": 2.6 }, g);
    const cap = text(g, 300, 570, "each shock's response, scaled; their sum is the path", "mono");
    return { g, update(t) {
      const J = Math.floor(seg(t, 0.05, 0.85) * N);
      b.forEach((r, k) => fade(r, k < J ? 0.85 : 0.15));
      copies.forEach((c, j) => { c.style.opacity = j < J ? (j === J - 1 ? 0.7 : 0.16) : 0; });
      const pts = X.map((_, k) => { let s = 0; for (let j = 0; j < Math.min(J, k + 1); ++j) s += K(k, j) * w[j]; return [TX(k), 260 - 22 * s]; });
      sum.setAttribute("d", pencil(pts, 11, 0.1)); fade(sum, J > 0 ? 1 : 0);
      fade(truth, seg(t, 0.02, 0.1) * 0.8); fade(cap, seg(t, 0.4, 0.55));
    } };
  })();

  // ------------------------------------------------------------------ 5. the noise-state: estimates of the shocks, revised
  scenes.noisestate = (() => {
    const g = group("noisestate");
    el("line", { x1: 50, y1: 200, x2: 560, y2: 200, class: "pencil soft", "stroke-width": 1 }, g);
    el("line", { x1: 50, y1: 440, x2: 560, y2: 440, class: "pencil soft", "stroke-width": 1 }, g);
    const sig = Y[0].map((v, k) => el("circle", { cx: TX(k), cy: 200 - v * 13, r: 2.8, class: "fillacc" }, g));
    sig.forEach((c) => { c.style.fill = "currentColor"; });
    const xs = el("path", { d: pencil(X.map((v, k) => [TX(k), 200 - v * 13]), 13, 0.1), class: "pencil soft", "stroke-width": 1.2 }, g);
    const trueBars = bars(g, w, 440, 26, "", 7); trueBars.forEach((r) => { r.style.fill = "currentColor"; r.style.opacity = 0.14; });
    const est = w.map((_, k) => el("rect", { x: TX(k) - 2.5, width: 5, class: "fillacc" }, g));
    const now = el("line", { y1: 90, y2: 520, class: "pencil warm", "stroke-width": 1.4, "stroke-dasharray": "3 4" }, g);
    const nowl = text(g, 0, 82, "now, t", "label warmt");
    text(g, 50, 110, "player 1's signal", "mono", "start");
    const lab = text(g, 50, 560, "Ŵ¹ₜ(u): player 1's estimate, at t, of the shock at u", "label acc", "start");
    text(g, 50, 585, "grey: the true shocks, which nobody sees", "mono", "start");
    return { g, update(t) {
      const k = 4 + Math.round(seg(t, 0.05, 0.9) * (N - 5));
      sig.forEach((c, j) => fade(c, j <= k ? 0.75 : 0));
      xs.style.opacity = 0.35;
      const e1 = What[0][k];
      est.forEach((r, u) => { const v = u <= k ? e1[u] : 0; r.setAttribute("y", v >= 0 ? 440 - v * 26 : 440); r.setAttribute("height", Math.abs(v) * 26); fade(r, u <= k ? 1 : 0); });
      now.setAttribute("x1", TX(k)); now.setAttribute("x2", TX(k)); nowl.setAttribute("x", TX(k));
      fade(lab, seg(t, 0.1, 0.25));
    } };
  })();

  // ------------------------------------------------------------------ 6. forecasting: the same kernel against the estimate
  scenes.samekernel = (() => {
    const g = group("samekernel");
    el("line", { x1: 50, y1: 300, x2: 560, y2: 300, class: "pencil soft", "stroke-width": 1 }, g);
    const sig = Y[0].map((v, k) => el("circle", { cx: TX(k), cy: 300 - v * 16, r: 2.6 }, g));
    sig.forEach((c) => { c.style.fill = "currentColor"; c.style.opacity = 0.35; });
    const xs = stroke(g, pencil(X.map((v, k) => [TX(k), 300 - v * 16]), 21, 0.1), "", 1.6);
    const xh = stroke(g, pencil(Xhat[0].map((v, k) => [TX(k), 300 - v * 16]), 22, 0.1), "accent", 2.6);
    const l1 = text(g, 50, 110, "X, the state", "label", "start");
    const l2 = text(g, 50, 135, "X̂¹, the kernel of X run against Ŵ¹", "label acc", "start");
    const l3 = text(g, 50, 160, "dots: player 1's signal", "mono", "start");
    const chk = text(g, 300, 560, "", "mono");
    return { g, update(t) {
      xs.set(seg(t, 0, 0.35)); xh.set(seg(t, 0.25, 0.75)); fade(l1, seg(t, 0, 0.15)); fade(l2, seg(t, 0.25, 0.4)); fade(l3, seg(t, 0, 0.15));
      chk.textContent = `same as a Kalman filter computed directly: largest gap ${checkFilter.toExponential(0)}`; fade(chk, seg(t, 0.7, 0.85));
    } };
  })();

  // ------------------------------------------------------------------ 7. the tower collapses into a composition
  scenes.collapse = (() => {
    const g = group("collapse");
    const levels = ["t1", "t2", "t3", "t4"].map((k, i) => { const lg = el("g", {}, g); formula(k, 300, 70 + i * 46, 120 + i * 40, lg, "middle"); return lg; });
    // the kernel of player 2's forecast on the grid of dates: row t, column u
    const hm = el("g", {}, g), H0 = 60, S = 180 / N;
    let mx = 0; for (const r of A2) for (const v of r) mx = Math.max(mx, Math.abs(v));
    for (let k = 0; k < N; ++k) for (let u = 0; u <= k; ++u) {
      const v = A2[k][u] / mx;
      const c = el("rect", { x: 90 + u * S, y: H0 + k * S, width: S + 0.3, height: S + 0.3, class: "fillacc" }, hm);
      c.style.opacity = Math.min(1, Math.abs(v) * 1.3);
    }
    el("rect", { x: 90, y: H0, width: 180, height: 180, class: "pencil soft", "stroke-width": 1 }, hm);
    text(hm, 180, H0 + 205, "player 2's forecast, as a kernel", "mono");
    text(hm, 180, H0 + 222, "row: date t · column: shock u", "mono");
    const arrow = el("g", {}, g);
    el("path", { d: "M290,150 L345,150 M336,142 L346,150 L336,158", class: "pencil", "stroke-width": 1.6 }, arrow);
    const w1 = el("g", {}, g);
    for (let u = 0; u < N; ++u) { const v = What[0][N - 1][u]; el("rect", { x: 370 + u * (180 / N), y: v >= 0 ? 150 - v * 20 : 150, width: 180 / N - 0.6, height: Math.abs(v) * 20, class: "fillacc" }, w1); }
    text(w1, 460, 245, "run against player 1's noise-state", "mono");
    el("line", { x1: 50, y1: 450, x2: 560, y2: 450, class: "pencil soft", "stroke-width": 1 }, g);
    const x2 = stroke(g, pencil(Xhat[1].map((v, k) => [TX(k), 450 - v * 16]), 31, 0.1), "warm", 2);
    const x12 = stroke(g, pencil(X12.map((v, k) => [TX(k), 450 - v * 16]), 32, 0.1), "accent", 2.6);
    const l2 = text(g, 50, 330, "X̂²: what player 2 forecasts", "label warmt", "start");
    const l12 = text(g, 50, 352, "𝔼¹[X̂²]: what player 1 thinks player 2 forecasts", "label acc", "start");
    const chk = text(g, 300, 585, "", "mono");
    return { g, update(t) {
      levels.forEach((l, i) => { const k = seg(t, 0.05 + i * 0.03, 0.3 + i * 0.03); fade(l, (1 - k) * seg(t, 0, 0.05)); l.setAttribute("transform", `translate(${-120 * k} ${(60 - i * 46) * k}) scale(${1 - 0.6 * k})`); l.style.transformOrigin = "300px 150px"; });
      fade(hm, seg(t, 0.25, 0.4)); fade(arrow, seg(t, 0.38, 0.45)); fade(w1, seg(t, 0.42, 0.52));
      x2.set(seg(t, 0.5, 0.75)); x12.set(seg(t, 0.62, 0.9)); fade(l2, seg(t, 0.5, 0.6)); fade(l12, seg(t, 0.62, 0.72));
      chk.textContent = `same as conditioning on player 1's signals directly: largest gap ${checkTower.toExponential(0)}`; fade(chk, seg(t, 0.85, 0.95));
    } };
  })();

  // ------------------------------------------------------------------ 8. a fixed point in kernels
  scenes.fixedpoint = (() => {
    const g = group("fixedpoint"), M = 30, T = (k) => 70 + (470 * k) / (M - 1);
    el("line", { x1: 60, y1: 470, x2: 560, y2: 470, class: "pencil soft", "stroke-width": 1 }, g);
    // the filter's gain from a poor first guess, iterated through its Riccati map; each iterate is a kernel:
    // the response of the forecast to a unit shock at age 0
    const kernelsFrom = (P0) => {
      const out = []; let Pp = P0;
      for (let n = 0; n < 6; ++n) {
        const gain = Pp / (Pp + SIG[0] * SIG[0]), h = [];
        let hv = 0; for (let k = 0; k < M; ++k) { hv = (1 - gain) * RHO * hv + gain * Math.pow(RHO, k); h.push(hv); }
        out.push(h);
        const P = (1 - gain) * Pp; Pp = RHO * RHO * P + 1;
      }
      return out;
    };
    // two first guesses, one too timid and one too trusting of the signal, iterated through the same map
    const low = kernelsFrom(0.004), high = kernelsFrom(40);
    const draw = (h, n, cls, seed) => el("path", { d: pencil(h.map((v, k) => [T(k), 470 - v * 320]), seed + n, 0.1), class: "pencil " + cls, "stroke-width": 1.4 }, g);
    const pl = low.map((h, n) => draw(h, n, "warm", 60)), ph = high.map((h, n) => draw(h, n, "", 80));
    const fin = el("path", { d: pencil(low[5].map((v, k) => [T(k), 470 - v * 320]), 99, 0.1), class: "pencil accent", "stroke-width": 3 }, g);
    const lab = text(g, 300, 520, "the forecast's response to a shock, by age", "mono");
    const it = text(g, 540, 110, "", "label acc", "end");
    const l1 = text(g, 540, 135, "from a guess too trusting of the signal", "mono", "end");
    const l2 = text(g, 540, 155, "and one too timid", "mono warmt", "end"); l2.style.fill = "var(--warm)";
    const cap = text(g, 300, 560, "both land on the same kernel (a filter's here, standing in for a game's)", "mono");
    return { g, update(t) {
      const n = Math.min(5, Math.floor(seg(t, 0.05, 0.75) * 6));
      [pl, ph].forEach((arr) => arr.forEach((p, i) => { p.style.opacity = i > n ? 0 : i === n ? 0.9 : 0.12 + 0.35 * (i / Math.max(1, n)); }));
      fade(fin, seg(t, 0.75, 0.85)); it.textContent = `iteration ${n + 1}`;
      fade(lab, seg(t, 0, 0.1)); fade(l1, seg(t, 0, 0.1)); fade(l2, seg(t, 0, 0.1)); fade(cap, seg(t, 0.8, 0.95));
    } };
  })();

  // ------------------------------------------------------------------ scroll to scene, and the progress line
  const bar = document.getElementById("progress");
  let active = null, prog = 0;
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
  measure();
  let last = performance.now();
  (function frame(now) {
    const dt = Math.min(0.05, (now - last) / 1000); last = now;
    const story = document.getElementById("story").getBoundingClientRect();
    if (active && story.top < window.innerHeight && story.bottom > 0) { const sc = scenes[active.dataset.scene]; if (sc) sc.update(reduced ? 1 : prog, now, dt); }
    requestAnimationFrame(frame);
  })(performance.now());
})();
