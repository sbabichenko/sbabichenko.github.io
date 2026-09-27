// /gate/how: the decision-mesh gate told in steps. The page draws coin flips from the chosen odds, sends them to
// fit-worker.js with the candidate trace on, and builds every scene from what comes back: each round's scored
// candidates with the segment each would split, the gate's empirical null, the lfdr cutoff, the admissions, the
// pool variance by round, the admitted surpluses by depth, and the final surface. Only the "corrections" scene is
// a sketch (a one-dimensional toy computed here), and its caption says so.
(function () {
  "use strict";
  const root = document.getElementById("gatehow");
  const svg = document.getElementById("stage");
  const steps = [...document.querySelectorAll(".step")];
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

  // ------------------------------------------------------------------ the data (the same odds as /gate)
  const BASE = -1, SITES = 6000, FLIPS = 20;
  let POOL_SD = 0.25;   // the coin effects' sd on the log-odds, from the slider
  const TRUTHS = {
    hills: (x, y) => BASE + 1.8 * Math.exp(-((x - 0.3) ** 2 + (y - 0.7) ** 2) / 0.02) - 1.5 * Math.exp(-((x - 0.7) ** 2 + (y - 0.3) ** 2) / 0.03),
    peaks: (x, y) => BASE + 1.6 * Math.exp(-((x - 0.25) ** 2 + (y - 0.3) ** 2) / 0.006) + 1.1 * Math.exp(-((x - 0.7) ** 2 + (y - 0.7) ** 2) / 0.008) - 1.4 * Math.exp(-((x - 0.72) ** 2 + (y - 0.22) ** 2) / 0.01),
    ring: (x, y) => BASE + 1.4 * Math.exp(-((Math.hypot(x - 0.5, y - 0.5) - 0.28) ** 2) / 0.004),
  };
  const expit = (t) => 1 / (1 + Math.exp(-t));
  function makeData(truth, seed) {
    const r = mulberry32(seed * 7919 + 17), f = TRUTHS[truth];
    const x = new Float64Array(SITES), y = new Float64Array(SITES), n = new Int32Array(SITES), k = new Int32Array(SITES), u = new Float64Array(SITES);
    const rows = ["wala,wac,n,k"];
    for (let i = 0; i < SITES; ++i) {
      x[i] = r(); y[i] = r();
      n[i] = Math.max(1, Math.round(FLIPS * (0.5 + r())));
      u[i] = POOL_SD * gauss(r);                         // the coin's own effect: what the two axes don't describe
      const p = expit(f(x[i], y[i]) + u[i]);
      let kk = 0; for (let j = 0; j < n[i]; ++j) if (r() < p) ++kk;
      k[i] = kk;
      rows.push(x[i].toFixed(5) + "," + y[i].toFixed(5) + "," + n[i] + "," + k[i]);
    }
    return { x, y, n, k, u, csv: rows.join("\n") + "\n" };
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
  function ramp(v) {   // v: log-odds minus the background, clipped at ±2.2
    const t = clamp(v / 2.2, -1, 1), to = t > 0 ? INK.acc : INK.warm, a = Math.pow(Math.abs(t), 0.8);
    return INK.paper.map((p, i) => Math.round(p + (to[i] - p) * a));
  }
  function raster(fn, N) {   // an image of fn(x, y) on an N × N grid, y up
    const c = document.createElement("canvas"); c.width = c.height = N;
    const ctx = c.getContext("2d"), img = ctx.createImageData(N, N);
    for (let j = 0; j < N; ++j) for (let i = 0; i < N; ++i) {
      const [r, g, b] = ramp(fn((i + 0.5) / N, 1 - (j + 0.5) / N)), o = 4 * (j * N + i);
      img.data[o] = r; img.data[o + 1] = g; img.data[o + 2] = b; img.data[o + 3] = 255;
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
    // the fitted log-odds at any point: the baseline plus the surface, bilinear between the dump's grid centres
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
        const [r, gg, b] = ramp(v); ctx.fillStyle = `rgb(${r},${gg},${b})`;
        ctx.beginPath(); ctx.arc(data.x[i] * 2 * N, (1 - data.y[i]) * 2 * N, 3.2, 0, 2 * Math.PI); ctx.fill();
      }
      const img = el("image", { x: SQ.x, y: SQ.y, width: SQ.s, height: SQ.s, href: c.toDataURL() }, g);
      frame(g, "6,000 sites, each dot one site's share of heads");
      const truth = el("image", { x: SQ.x, y: SQ.y, width: SQ.s, height: SQ.s, href: raster((x, y) => TRUTHS[data.truth](x, y) - BASE, 110), opacity: 0 }, g);
      g.insertBefore(truth, img);                            // the true odds come in beneath the dots, not over them
      const tl = text(g, SQ.x + SQ.s, SQ.y + SQ.s + 26, "the odds the coins really have", "mono", "end");
      const leg = el("g", {}, g);
      [["more heads than usual", "pos"], ["fewer", "neg"]].forEach(([s, cls], i) => { el("circle", { cx: SQ.x + 6 + i * 170, cy: SQ.y + SQ.s + 22, r: 5, class: cls }, leg); text(leg, SQ.x + 16 + i * 170, SQ.y + SQ.s + 26, s, "tiny", "start"); });
      return { g, update(t) { fade(img, seg(t, 0, 0.15)); truth.style.opacity = 0.9 * seg(t, 0.55, 0.8); fade(tl, seg(t, 0.6, 0.8)); fade(leg, seg(t, 0.1, 0.25) * (1 - seg(t, 0.55, 0.7))); } };
    })();

    // 1b. the funnel: how far a binomial may stray is known from its mean ---------------------------------------
    scenes.funnel = (() => {
      const g = group("funnel");
      const B = { x: 80, y: 110, w: 440, h: 360 }, nmax = Math.max(...data.n), ymax = 1.1;
      const X = (n) => B.x + ((n - 0.5 * FLIPS) / (nmax - 0.5 * FLIPS + 1)) * B.w, Y = (v) => B.y + B.h / 2 - (v / ymax) * (B.h / 2);
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
      for (let n = Math.ceil(0.5 * FLIPS); n <= nmax; ++n) { band.push([X(n), Y(1.96 / Math.sqrt(n))]); lo.push([X(n), Y(-1.96 / Math.sqrt(n))]); }
      const up = stroke(g, "M" + band.map((q) => q.join(",")).join(" L"), "warm", 2), dn = stroke(g, "M" + lo.map((q) => q.join(",")).join(" L"), "warm", 2);
      text(g, B.x, B.y + B.h + 18, `${Math.ceil(0.5 * FLIPS)} flips`, "tiny", "start"); text(g, B.x + B.w, B.y + B.h + 18, `${nmax} flips`, "tiny", "end");
      const lab = text(g, B.x + 6, B.y + B.h - 10, "a binomial stays between the lines 95% of the time", "label warmt halo", "start");
      const cap = text(g, 300, B.y + B.h + 50, `outside: ${pct(outside)} of sites, not 5%; mean squared residual ${disp(zFit).toFixed(2)}, not 1`, "mono");
      return { g, update(t) { fade(img, seg(t, 0, 0.2)); up.set(seg(t, 0.25, 0.5)); dn.set(seg(t, 0.25, 0.5)); fade(lab, seg(t, 0.45, 0.6)); fade(cap, seg(t, 0.6, 0.75)); } };
    })();

    // 1b'. a wrong mean shows up as variance, whichever way the mean is missed ----------------------------------
    // A site's miss is its true log-odds less the fitted: the fitted surface missing the true surface, plus the
    // coin's own effect, the part of its odds the two axes don't describe. Both are known here, since the page drew them.
    scenes.wrongmean = (() => {
      const g = group("wrongmean");
      const tf = TRUTHS[data.truth];
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

    // 1c. two kinds of excess: shared by neighbours, or each site's own ------------------------------------------
    scenes.coherent = (() => {
      const g = group("coherent");
      const w = 250, gap = 20, x0 = 300 - w - gap / 2, x1 = 300 + gap / 2, y0 = 150;
      const map = (z) => {
        const c = document.createElement("canvas"), N = 2 * w; c.width = c.height = N;
        const ctx = c.getContext("2d");
        z.forEach((v, i) => { const [r, gg, b] = ramp(clamp(v, -3, 3) * (2.2 / 3)); ctx.fillStyle = `rgb(${r},${gg},${b})`; ctx.beginPath(); ctx.arc(data.x[i] * N, (1 - data.y[i]) * N, 3.4, 0, 2 * Math.PI); ctx.fill(); });
        return c.toDataURL();
      };
      const A = el("g", {}, g), Bg = el("g", {}, g);
      el("image", { x: x0, y: y0, width: w, height: w, href: map(zFlat) }, A); el("rect", { x: x0, y: y0, width: w, height: w, class: "frame" }, A);
      text(A, x0, y0 - 12, "against a flat surface", "mono", "start");
      text(A, x0 + w / 2, y0 + w + 24, "coherent: neighbours share it", "label");
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
      text(A, 70, 70, "1 · centre each residual on its own expected value", "mono", "start");
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
      const zl = text(C, 300, 506, "z  =  (centred score on the remainder − shrinkage bias) / sd", "label");
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
      return { bars, sbars, textbook, own, X, Y };
    }

    // 6. the family: round 0's scores and its null ---------------------------------------------------------------
    scenes.family = (() => {
      const g = group("family");
      text(g, 70, 90, `round 0: ${r0.length} scores`, "mono", "start");
      const H = histogram(g, r0, cal0, { x: 70, y: 120, w: 460, h: 300 });
      H.sbars.forEach((b) => b.remove());
      const l1 = el("g", {}, g), l2 = el("g", {}, g);
      line(l1, 80, 470, 110, 470, "soft", 1.4).setAttribute("stroke-dasharray", "4 4"); text(l1, 118, 474, "the textbook null, N(0, 1)", "tiny", "start");
      line(l2, 80, 492, 110, 492, "accent", 2); text(l2, 118, 496, `this round's null: centre ${fmt(cal0.nullMean)}, spread ${fmt(cal0.nullSd)}, share ${pct(cal0.pi0)}`, "tiny", "start");
      return { g, update(t) {
        const n = Math.floor(seg(t, 0, 0.35) * H.bars.length);
        H.bars.forEach((b, i) => fade(b, i < n ? 0.22 : 0));
        fade(H.textbook, seg(t, 0.35, 0.45)); fade(l1, seg(t, 0.35, 0.45));
        if (H.own) fade(H.own, seg(t, 0.5, 0.65)); fade(l2, seg(t, 0.5, 0.65));
      } };
    })();

    // 7. lfdr, best first, and the running mean -------------------------------------------------------------------
    scenes.lfdr = (() => {
      const g = group("lfdr");
      const byL = [...r0].sort((a, b) => a.lfdr - b.lfdr || Math.abs(b.z) - Math.abs(a.z));
      const K = Math.min(byL.length, Math.max(40, Math.min(90, cal0.prefix * 2 + 20))), x0 = 70, w = 460, y0 = 130, h = 330, bw = w / K;
      text(g, x0, 100, `round 0, best ${K} of ${byL.length} candidates by lfdr`, "mono", "start");
      line(g, x0, y0 + h, x0 + w, y0 + h, "", 1);
      for (const v of [0, 0.5, 1]) text(g, x0 - 8, y0 + h - v * h + 4, pct(v), "tiny", "end");
      const bars = byL.slice(0, K).map((c, i) => el("rect", { x: x0 + i * bw + 0.5, y: y0 + h - c.lfdr * h, width: Math.max(1, bw - 1), height: Math.max(0.5, c.lfdr * h), class: c.selected ? (c.z > 0 ? "pos" : "neg") : "ink", opacity: c.selected ? 0.85 : 0.2 }, g));
      let cum = 0; const run = byL.slice(0, K).map((c, i) => { cum += c.lfdr; return [x0 + (i + 0.5) * bw, y0 + h - (cum / (i + 1)) * h]; });
      const runL = stroke(g, "M" + run.map((p) => p[0].toFixed(1) + "," + p[1].toFixed(1)).join(" L"), "", 2);
      const cut = el("g", {}, g);
      line(cut, x0, y0 + h - 0.1 * h, x0 + w, y0 + h - 0.1 * h, "warm", 1.4).setAttribute("stroke-dasharray", "6 4");
      text(cut, x0 + w, y0 + h - 0.1 * h - 8, "10%", "label warmt", "end");
      const pre = el("g", {}, g), k = byL.filter((c) => c.selected).length;
      if (k) { el("rect", { x: x0, y: y0 - 6, width: k * bw, height: h + 6, class: "shade" }, pre); text(pre, x0 + k * bw + 6, y0 + 12, `the run: ${k}`, "label acc", "start"); }
      else text(pre, x0 + 10, y0 + 12, "no run: nothing selected", "label", "start");
      const cap = text(g, 300, y0 + h + 40, "bars: each candidate's lfdr; line: the running mean", "mono");
      return { g, update(t) {
        const n = Math.floor(seg(t, 0, 0.4) * bars.length);
        bars.forEach((b, i) => { b.style.opacity = i < n ? "" : 0; });
        runL.set(seg(t, 0.35, 0.65)); fade(cut, seg(t, 0.3, 0.4)); fade(pre, seg(t, 0.65, 0.8)); fade(cap, seg(t, 0.1, 0.25));
      } };
    })();

    // 8. admission: round 0's selections, admitted or turned back -----------------------------------------------------
    scenes.admit = (() => {
      const g = group("admit");
      const sel = r0.filter((c) => c.selected);
      frame(g, `round 0: ${sel.length} selected`);
      const after = meshLines(g, fit); after.style.opacity = 0;
      const segs = sel.map((c) => c.seg ? line(g, sx(c.seg[0]), sy(c.seg[1]), sx(c.seg[2]), sy(c.seg[3]), c.z > 0 ? "posS" : "negS", 1.8) : null);
      const marks = sel.map((c) => {
        const q = el("g", {}, g);
        if (c.admitted) el("circle", { cx: sx(c.x), cy: sy(c.y), r: 6, class: c.z > 0 ? "pos" : "neg" }, q);
        else { const X = sx(c.x), Y = sy(c.y); line(q, X - 6, Y - 6, X + 6, Y + 6, "", 2); line(q, X - 6, Y + 6, X + 6, Y - 6, "", 2); }
        return q;
      });
      const na = sel.filter((c) => c.admitted).length;
      const cap = text(g, 300, SQ.y + SQ.s + 28, na === sel.length ? `all ${na} admitted (dots); faint: the mesh the fit ends with` : `${na} admitted (dots), ${sel.length - na} turned back on re-scoring (crosses); faint: the mesh the fit ends with`, "tiny");
      return { g, update(t) {
        const n = Math.floor(seg(t, 0.05, 0.7) * sel.length);
        segs.forEach((s, i) => s && fade(s, i < n ? 0.9 : 0)); marks.forEach((m, i) => fade(m, i < n ? 1 : 0));
        fade(after, seg(t, 0.7, 0.85)); fade(cap, seg(t, 0.7, 0.85));
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
      const pv = fit.rounds.map((r) => r.poolVariance).concat([fit.poolVariance]), truth = POOL_SD * POOL_SD;
      const A = { x: 80, y: 110, w: 200, h: 230 }, vmax = Math.max(...pv, truth) * 1.1;
      const AX = (i) => A.x + (i / Math.max(1, pv.length - 1)) * A.w, AY = (v) => A.y + A.h - (v / vmax) * A.h;
      const a = el("g", {}, g);
      text(a, A.x - 10, A.y - 30, "coin variance, by round", "mono", "start");
      line(a, A.x, A.y + A.h, A.x + A.w, A.y + A.h, "", 1); line(a, A.x, A.y, A.x, A.y + A.h, "", 1);
      const tl = line(a, A.x, AY(truth), A.x + A.w, AY(truth), "warm", 1.4); tl.setAttribute("stroke-dasharray", "5 4");
      text(a, A.x + A.w, A.y + A.h - 12, `dashed: the true ${POOL_SD.toFixed(2)}²`, "tiny", "end");
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
      el("image", { x: x0, y: y0, width: w, height: w, href: raster((x, y) => TRUTHS[data.truth](x, y) - BASE, 100) }, g);
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
    const setV = (k, s) => document.querySelectorAll(`[data-v="${k}"]`).forEach((n) => { n.textContent = s; });
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
    // with its standard error, and the floor: the true surface scored on the same sites (paired gap)
    const H = (() => {
      const h = fit.heldout; if (!h || !h.rows || !h.rows.n.length) return null;
      const R = h.rows, m = R.n.length, f = TRUTHS[data.truth], lg = (t) => 1 / (1 + Math.exp(-t));
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

  // ------------------------------------------------------------------ the run
  let worker = null, seq = 0, want = null, data = null;
  function status(s, busy) { const n = $("howstatus"); n.textContent = s; n.classList.toggle("busy", !!busy); }
  function fitNow() {
    const engine = $("engine").value, truth = $("truth").value, seed = +(root.dataset.seed || 1);
    POOL_SD = +$("coinsd").value;
    root.dataset.engine = engine;
    data = makeData(truth, seed); data.truth = truth;
    if (!worker) { worker = new Worker(root.dataset.worker); worker.onmessage = onMsg; worker.onerror = () => status("the estimator failed to load"); }
    want = ++seq;
    status("Fitting…", true);
    if (!svg.firstChild || svg.querySelector(".stagebusy")) { svg.textContent = ""; text(svg, 300, 300, "fitting…", "stagebusy"); }
    else svg.style.opacity = 0.35;
    worker.postMessage({ id: want, engine, design: data.csv, seed: 7, detail: true });
  }
  function onMsg(ev) {
    const m = ev.data;
    if (m.type === "ready") return;
    if (m.id !== want) return;
    svg.style.opacity = 1;
    if (m.type === "error") { status("the fit failed: " + m.message); return; }
    run = m;
    status(`${m.engine === "rect" ? "rectangles" : "right triangles"} · ${(m.ms / 1000).toFixed(1)} s in your browser`);
    build(data, m);
  }
  const sdLabel = () => { $("coinsdval").textContent = (+$("coinsd").value).toFixed(2); };
  $("coinsd").addEventListener("input", sdLabel);
  $("coinsd").addEventListener("change", fitNow);
  $("engine").addEventListener("change", fitNow);
  $("truth").addEventListener("change", fitNow);
  $("again").addEventListener("click", () => { root.dataset.seed = (+(root.dataset.seed || 1) % 97) + 1; fitNow(); });
  new MutationObserver(() => { if (run) build(data, run); }).observe(document.documentElement, { attributes: true, attributeFilter: ["class"] });

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
