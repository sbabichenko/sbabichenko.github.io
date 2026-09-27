// /gate: coin flips on a square, fitted by the triangular decision-mesh estimator itself (fit-worker.js runs
// triangular-decision-mesh/core compiled to WebAssembly). This file makes the data, sends it to the worker,
// and draws what comes back: the truth, the data, the fitted mesh, and the gate's account of every round.
// A visitor can also fit a CSV of their own (see "your own data" below); it is read here and never leaves the page.
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const isDark = () => document.documentElement.classList.contains("dark");
  const expit = (t) => 1 / (1 + Math.exp(-t));

  // ------------------------------------------------------------------ truths, on the log-odds scale
  const BASE = -1;           // log-odds of the background coin: expit(-1) = 27%
  // each site's coin carries its own effect on the log-odds, normal with this sd (the slider); the estimator fits
  // their variance alongside the surface (its "pool variance")
  const coinSd = () => +$("coinsd").value;
  const gateQ = () => +$("q").value;        // the gate's false discovery rate (the engines' DMESH_Q; 0.10 by default)
  const G = 64;              // the drawing grid
  const drawing = new Float32Array(G * G).fill(BASE);
  const TRUTHS = {
    hills: { name: "Two hills", f: (x, y) => BASE + 1.8 * Math.exp(-((x - 0.3) ** 2 + (y - 0.7) ** 2) / 0.02) - 1.5 * Math.exp(-((x - 0.7) ** 2 + (y - 0.3) ** 2) / 0.03) },
    island: { hidden: true, name: "An island", f: (x, y) => (Math.hypot(x - 0.55, y - 0.5) < 0.27 ? BASE + 1.6 : BASE - 0.6) },
    fault: { hidden: true, name: "A fault line", f: (x, y) => BASE - 0.2 + 2 * Math.tanh(25 * (x - 0.35 - 0.3 * y)) },
    peaks: { name: "Three peaks", f: (x, y) => BASE + 1.6 * Math.exp(-((x - 0.25) ** 2 + (y - 0.3) ** 2) / 0.006) + 1.1 * Math.exp(-((x - 0.7) ** 2 + (y - 0.7) ** 2) / 0.008) - 1.4 * Math.exp(-((x - 0.72) ** 2 + (y - 0.22) ** 2) / 0.01) },
    ring: { name: "A ring", f: (x, y) => BASE + 1.4 * Math.exp(-((Math.hypot(x - 0.5, y - 0.5) - 0.28) ** 2) / 0.004) },
    nothing: { name: "Nothing at all", f: () => BASE },
    drawing: { name: "Your drawing", f: (x, y) => sampleDrawing(x, y) },
  };
  function sampleDrawing(x, y) {
    const gx = Math.min(G - 1.001, Math.max(0, x * (G - 1))), gy = Math.min(G - 1.001, Math.max(0, y * (G - 1)));
    const i = Math.floor(gx), j = Math.floor(gy), u = gx - i, v = gy - j;
    const a = drawing[j * G + i], b = drawing[j * G + i + 1], c = drawing[(j + 1) * G + i], d = drawing[(j + 1) * G + i + 1];
    return (a * (1 - u) + b * u) * (1 - v) + (c * (1 - u) + d * u) * v;
  }
  function brush(x, y, amount, radius) {
    for (let j = 0; j < G; ++j) for (let i = 0; i < G; ++i) {
      const dx = i / (G - 1) - x, dy = j / (G - 1) - y, w = Math.exp(-(dx * dx + dy * dy) / (2 * radius * radius));
      if (w < 1e-3) continue;
      drawing[j * G + i] = Math.max(BASE - 2.5, Math.min(BASE + 2.2, drawing[j * G + i] + amount * w));
    }
  }
  // a face to start from, so the first fit of "your drawing" has something to find
  (function smile() {
    brush(0.35, 0.68, 2.0, 0.06); brush(0.65, 0.68, 2.0, 0.06);
    for (let t = 0; t <= 1; t += 0.05) { const a = Math.PI * (1.15 + 0.7 * t); brush(0.5 + 0.26 * Math.cos(a), 0.5 + 0.26 * Math.sin(a), 0.55, 0.035); }
  })();

  // ------------------------------------------------------------------ data
  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function gauss(r) { let u = 0; while (u === 0) u = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r()); }
  function binomial(r, n, p) { let k = 0; for (let i = 0; i < n; ++i) if (r() < p) ++k; return k; }

  function makeData(truth, sites, flips, seed, sd) {
    const r = mulberry32(seed * 7919 + 17), f = TRUTHS[truth].f;
    const x = new Float64Array(sites), y = new Float64Array(sites), n = new Int32Array(sites), k = new Int32Array(sites);
    const rows = ["wala,wac,n,k"];
    for (let i = 0; i < sites; ++i) {
      x[i] = r(); y[i] = r();
      n[i] = Math.max(1, Math.round(flips * (0.5 + r())));
      k[i] = binomial(r, n[i], expit(f(x[i], y[i]) + sd * gauss(r)));
      rows.push(x[i].toFixed(5) + "," + y[i].toFixed(5) + "," + n[i] + "," + k[i]);
    }
    return { x, y, n, k, csv: rows.join("\n") + "\n" };
  }

  // ------------------------------------------------------------------ colour and rasters
  const D = 200, VMAX = 2.2;
  let CENTRE = BASE;         // the log-odds at the middle of the colour scale: the coins' background, or a file's overall rate
  function ramp() {
    const hex = (h) => [1, 3, 5].map((q) => parseInt(h.slice(q, q + 2), 16));
    const s = isDark()
      ? [hex("#2b6cf0"), hex("#5b8ff9"), hex("#2a2b30"), hex("#f0845c"), hex("#f5c451")]
      : [hex("#1d4ed8"), hex("#6d9cf5"), hex("#f7f6ee"), hex("#ef7a55"), hex("#b91c1c")];
    const lut = new Uint8ClampedArray(256 * 3);
    for (let q = 0; q < 256; ++q) {
      const t = (q / 255) * 4, a = Math.min(3, Math.floor(t)), u = t - a;
      for (let c = 0; c < 3; ++c) lut[3 * q + c] = s[a][c] + (s[a + 1][c] - s[a][c]) * u;
    }
    return lut;
  }
  let LUT = ramp();
  const colour = (logit) => {
    const q = Math.max(0, Math.min(255, Math.round((((logit - CENTRE) / VMAX) * 0.5 + 0.5) * 255)));
    return [LUT[3 * q], LUT[3 * q + 1], LUT[3 * q + 2]];
  };
  const off = document.createElement("canvas"); off.width = off.height = D;
  const offctx = off.getContext("2d"), img = offctx.createImageData(D, D);
  // grids are row-major with row 0 at the top (y = 1)
  function truthGrid(truth) {
    const f = TRUTHS[truth].f, g = new Float32Array(D * D);
    for (let r = 0; r < D; ++r) for (let c = 0; c < D; ++c) g[r * D + c] = f((c + 0.5) / D, 1 - (r + 0.5) / D);
    return g;
  }
  // the fitted surface: each triangle interpolates its corners' heights, plus the engine's baseline
  function fitGrid(fit) {
    const g = new Float32Array(D * D).fill(NaN), t = fit.tri;
    const [x0, x1, y0, y1] = fit.bounds;
    const X = (u) => x0 + u * (x1 - x0), Y = (u) => y0 + u * (y1 - y0);
    if (fit.engine === "rect") {
      // each cell is bilinear in its four corner heights
      for (let i = 0; i < t.length; i += 8) {
        const ax = X(t[i]), ay = Y(t[i + 1]), bx = X(t[i + 2]), by = Y(t[i + 3]);
        const c0 = Math.max(0, Math.floor(ax * D)), c1 = Math.min(D - 1, Math.ceil(bx * D));
        const r0 = Math.max(0, Math.floor((1 - by) * D)), r1 = Math.min(D - 1, Math.ceil((1 - ay) * D));
        for (let r = r0; r <= r1; ++r) for (let c = c0; c <= c1; ++c) {
          const px = (c + 0.5) / D, py = 1 - (r + 0.5) / D;
          if (px < ax || px > bx || py < ay || py > by) continue;
          const u = (px - ax) / (bx - ax), v = (py - ay) / (by - ay);
          g[r * D + c] = fit.baseline + (1 - u) * (1 - v) * t[i + 4] + u * (1 - v) * t[i + 5] + (1 - u) * v * t[i + 6] + u * v * t[i + 7];
        }
      }
    } else for (let i = 0; i < t.length; i += 9) {
      const ax = X(t[i]), ay = Y(t[i + 1]), ah = t[i + 2], bx = X(t[i + 3]), by = Y(t[i + 4]), bh = t[i + 5], cx = X(t[i + 6]), cy = Y(t[i + 7]), ch = t[i + 8];
      const det = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy);
      if (Math.abs(det) < 1e-15) continue;
      const c0 = Math.max(0, Math.floor(Math.min(ax, bx, cx) * D - 1)), c1 = Math.min(D - 1, Math.ceil(Math.max(ax, bx, cx) * D));
      const r0 = Math.max(0, Math.floor((1 - Math.max(ay, by, cy)) * D - 1)), r1 = Math.min(D - 1, Math.ceil((1 - Math.min(ay, by, cy)) * D));
      for (let r = r0; r <= r1; ++r) for (let c = c0; c <= c1; ++c) {
        const px = (c + 0.5) / D, py = 1 - (r + 0.5) / D;
        const l1 = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / det;
        const l2 = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / det;
        const l3 = 1 - l1 - l2;
        if (l1 < -1e-9 || l2 < -1e-9 || l3 < -1e-9) continue;
        g[r * D + c] = fit.baseline + l1 * ah + l2 * bh + l3 * ch;
      }
    }
    // the mesh spans the data's bounding box; a pixel just outside it takes its nearest neighbour in the row
    for (let r = 0; r < D; ++r) {
      let last = NaN;
      for (let c = 0; c < D; ++c) { const k = r * D + c; if (g[k] === g[k]) last = g[k]; else if (last === last) g[k] = last; }
      for (let c = D - 1; c >= 0; --c) { const k = r * D + c; if (g[k] === g[k]) last = g[k]; else g[k] = last; }
    }
    return g;
  }
  function fitCanvas(c) {
    const r = c.getBoundingClientRect(), dpr = Math.min(2, window.devicePixelRatio || 1);
    const w = Math.max(1, Math.round(r.width * dpr)), h = Math.max(1, Math.round(r.height * dpr));
    if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
  }
  function paint(canvas, g) {
    fitCanvas(canvas);
    const d = img.data, empty = isDark() ? [31, 32, 34] : [255, 255, 240];
    for (let k = 0; k < D * D; ++k) {
      const v = g[k], rgb = v === v ? colour(v) : empty;
      d[4 * k] = rgb[0]; d[4 * k + 1] = rgb[1]; d[4 * k + 2] = rgb[2]; d[4 * k + 3] = 255;
    }
    offctx.putImageData(img, 0, 0);
    const ctx = canvas.getContext("2d");
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(off, 0, 0, canvas.width, canvas.height);
    return ctx;
  }

  // ------------------------------------------------------------------ state
  const S = { data: null, fit: null, truthG: null, fitG: null, busy: false, queued: false, id: 0, ready: false, user: null };
  window.gateDemo = S;
  const opts = () => ({ engine: $("engine").value, truth: $("truth").value, sites: +$("sites").value, flips: +$("flips").value, seed: +$("seed").value, sd: coinSd(), q: gateQ() });

  function drawTruth() {
    if (S.user) return;                                  // a file has no truth to draw
    const ctx = paint($("cvtruth"), S.truthG);
    if ($("truth").value === "drawing") {
      const W = ctx.canvas.width;
      ctx.fillStyle = isDark() ? "rgba(255,255,255,0.7)" : "rgba(20,20,30,0.6)";
      ctx.font = `${Math.round(W / 26)}px ui-sans-serif, system-ui, sans-serif`;
      ctx.fillText("draw here", W * 0.04, W * 0.07);
    }
  }
  function drawData() {
    const c = $("cvdata"); fitCanvas(c);
    const ctx = c.getContext("2d"), W = c.width;
    ctx.fillStyle = isDark() ? "#1f2022" : "#fffff0"; ctx.fillRect(0, 0, W, W);
    if (!S.data) return;
    const { x, y, n, k } = S.data, rad = Math.max(1.2, W / Math.sqrt(x.length) * 0.42);
    for (let i = 0; i < x.length; ++i) {
      const rgb = colour(Math.log((k[i] + 0.5) / (n[i] - k[i] + 0.5)));
      ctx.fillStyle = `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
      ctx.fillRect(x[i] * W - rad, (1 - y[i]) * W - rad, 2 * rad, 2 * rad);
    }
  }
  const ROUND_COLOURS = () => (isDark() ? ["#8fb0ff", "#f5c451", "#f07ab8", "#7fe0c0"] : ["#1f3fd0", "#d97706", "#c0267a", "#0f8a6a"]);
  function drawFit() {
    const c = $("cvfit");
    if (!S.fitG) { fitCanvas(c); const ctx = c.getContext("2d"); ctx.fillStyle = isDark() ? "#1f2022" : "#fffff0"; ctx.fillRect(0, 0, c.width, c.height); return; }
    const ctx = paint(c, S.fitG), W = c.width, fit = S.fit;
    const [x0, x1, y0, y1] = fit.bounds;
    const X = (u) => (x0 + u * (x1 - x0)) * W, Y = (u) => (1 - (y0 + u * (y1 - y0))) * W;
    if ($("showedges").checked) {
      ctx.strokeStyle = isDark() ? "rgba(255,255,255,0.28)" : "rgba(20,20,30,0.34)";
      ctx.lineWidth = Math.max(1, W / 420);
      ctx.beginPath();
      const t = fit.tri;
      if (fit.engine === "rect") for (let i = 0; i < t.length; i += 8) {
        const ax = X(t[i]), ay = Y(t[i + 1]), bx = X(t[i + 2]), by = Y(t[i + 3]);
        ctx.rect(Math.min(ax, bx), Math.min(ay, by), Math.abs(bx - ax), Math.abs(by - ay));
      } else for (let i = 0; i < t.length; i += 9) {
        ctx.moveTo(X(t[i]), Y(t[i + 1])); ctx.lineTo(X(t[i + 3]), Y(t[i + 4])); ctx.lineTo(X(t[i + 6]), Y(t[i + 7])); ctx.closePath();
      }
      ctx.stroke();
    }
    if ($("showadmit").checked) {
      const cols = ROUND_COLOURS(), r = Math.max(2.5, W / 110);
      for (const v of fit.verts) {
        if (!v.admitted) continue;
        ctx.fillStyle = cols[Math.min(cols.length - 1, Math.max(0, v.round))];
        ctx.beginPath(); ctx.arc(X(v.x), Y(v.y), r, 0, 2 * Math.PI); ctx.fill();
        ctx.lineWidth = Math.max(1, W / 400); ctx.strokeStyle = isDark() ? "#111" : "#fff"; ctx.stroke();
      }
    }
  }
  function drawAll() { drawTruth(); drawData(); drawFit(); }

  function drawScale() {
    const stops = [];
    for (let q = 0; q <= 8; ++q) { const t = CENTRE + ((q / 8) * 2 - 1) * VMAX, rgb = colour(t); stops.push(`rgb(${rgb[0]},${rgb[1]},${rgb[2]})`); }
    const pct = (t) => Math.round(100 * expit(t)) + "%";
    $("scale").innerHTML = `${pct(CENTRE - VMAX)} <i style="background:linear-gradient(90deg,${stops.join(",")})"></i> ${pct(CENTRE + VMAX)} chance of ${S.user ? "success" : "heads"}`;
  }

  // ------------------------------------------------------------------ results
  const fmt = (v, d = 3) => (v === undefined || v === null || v !== v ? "–" : (+v).toFixed(d));
  function rmse(a, b) {
    let s = 0, n = 0;
    for (let k = 0; k < a.length; ++k) if (a[k] === a[k] && b[k] === b[k]) { s += (a[k] - b[k]) ** 2; ++n; }
    return Math.sqrt(s / n);
  }
  // Held-out deviance with its standard error, and the floor: the same sites scored by the true surface (the mesh
  // exactly equal to the hidden odds), which no surface can beat on average. The gap is paired, site by site.
  function heldStats(h) {
    if (!h || !h.rows || !h.rows.n.length) return null;
    const f = TRUTHS[opts().truth].f, R = h.rows, m = R.n.length;
    const dev = (k, n, p) => { p = Math.min(1 - 1e-6, Math.max(1e-6, p)); return (k > 0 ? 2 * k * Math.log(k / (n * p)) : 0) + (n - k > 0 ? 2 * (n - k) * Math.log((n - k) / (n * (1 - p))) : 0); };
    const a = new Float64Array(m), b = new Float64Array(m);
    for (let i = 0; i < m; ++i) { a[i] = dev(R.k[i], R.n[i], R.p[i]); b[i] = dev(R.k[i], R.n[i], expit(f(R.x[i], R.y[i]))); }
    const mean = (v) => v.reduce((s, x) => s + x, 0) / v.length;
    const se = (v, mu) => Math.sqrt(v.reduce((s, x) => s + (x - mu) ** 2, 0) / (v.length - 1) / v.length);
    const d = a.map((x, i) => x - b[i]), ma = mean(a), mb = mean(b), md = mean(d);
    return { fit: ma, fitSe: se(a, ma), floor: mb, floorSe: se(b, mb), gap: md, gapSe: se(d, md), m };
  }
  // a file's held-out sites, scored by the fit and by a flat fit at the file's overall rate (paired, site by site)
  function ownHeld(h, rate) {
    if (!h || !h.rows || !h.rows.n.length) return null;
    const R = h.rows, m = R.n.length;
    const dev = (k, n, p) => { p = Math.min(1 - 1e-6, Math.max(1e-6, p)); return (k > 0 ? 2 * k * Math.log(k / (n * p)) : 0) + (n - k > 0 ? 2 * (n - k) * Math.log((n - k) / (n * (1 - p))) : 0); };
    const a = new Float64Array(m), b = new Float64Array(m);
    for (let i = 0; i < m; ++i) { a[i] = dev(R.k[i], R.n[i], R.p[i]); b[i] = dev(R.k[i], R.n[i], rate); }
    const mean = (v) => v.reduce((s, x) => s + x, 0) / v.length;
    const se = (v, mu) => Math.sqrt(v.reduce((s, x) => s + (x - mu) ** 2, 0) / (v.length - 1) / v.length);
    const d = b.map((x, i) => x - a[i]), ma = mean(a), mb = mean(b), md = mean(d);
    return { fit: ma, fitSe: se(a, ma), flat: mb, flatSe: se(b, mb), gain: md, gainSe: se(d, md), m };
  }
  function ownCards(f, admitted) {
    const u = S.user, H = ownHeld(f.heldout, u.rate), rounds = f.rounds;
    return [
      ["Vertices admitted", admitted, `over ${rounds.filter((r) => r.admitted > 0).length} round${rounds.filter((r) => r.admitted > 0).length === 1 ? "" : "s"}; ${f.tri.length / f.stride} ${f.engine === "rect" ? "rectangles" : "triangles"}`],
      ["Sites", u.sites.toLocaleString(), u.grid ? `from ${u.rows.toLocaleString()} rows, grouped on a ${u.grid} &times; ${u.grid} grid` : `from ${u.rows.toLocaleString()} rows, each with its own trials`],
      ["Site variance", fmt(f.poolVariance, 4), "the variance of the sites' own effects on the log-odds, fitted with the surface"],
      H ? ["Held-out deviance", `${fmt(H.fit, 3)} <small>&plusmn; ${fmt(H.fitSe, 3)}</small>`,
        `per site, on ${H.m.toLocaleString()} sites never fitted. A flat fit at the overall rate scores ${fmt(H.flat, 3)} &plusmn; ${fmt(H.flatSe, 3)} there; `
        + `the mesh is ${fmt(Math.abs(H.gain), 3)} &plusmn; ${fmt(H.gainSe, 3)} ${H.gain >= 0 ? "below" : "above"} it.`]
        : ["Held-out deviance", f.heldout ? fmt(f.heldout.deviance, 3) : "–", f.heldout ? `per site, on ${f.heldout.pools.toLocaleString()} sites never fitted` : ""],
    ];
  }
  function report() {
    const f = S.fit, rounds = f.rounds, admitted = rounds.reduce((a, r) => a + r.admitted, 0);
    const flat = new Float32Array(D * D).fill(f.baseline);
    const err = S.user ? NaN : rmse(S.fitG, S.truthG), errFlat = S.user ? NaN : rmse(flat, S.truthG);
    $("cards").innerHTML = (S.user ? ownCards(f, admitted) : [
      ["Vertices admitted", admitted, `over ${rounds.filter((r) => r.admitted > 0).length} round${rounds.filter((r) => r.admitted > 0).length === 1 ? "" : "s"}; ${f.tri.length / f.stride} ${f.engine === "rect" ? "rectangles" : "triangles"}`],
      ["Error against the truth", fmt(err, 3), `log-odds RMSE; a flat fit: ${fmt(errFlat, 3)}`],
      ["Coin variance", fmt(f.poolVariance, 4), `the coin effects' variance; truly ${fmt(coinSd() ** 2, 4)}`],
      (() => {
        const H = heldStats(f.heldout);
        if (!H) return ["Held-out deviance", f.heldout ? fmt(f.heldout.deviance, 3) : "–", f.heldout ? `per site, on ${f.heldout.pools.toLocaleString()} sites never fitted` : ""];
        return ["Held-out deviance", `${fmt(H.fit, 3)} <small>&plusmn; ${fmt(H.fitSe, 3)}</small>`,
          `per site, on ${H.m.toLocaleString()} sites never fitted. The true surface scores ${fmt(H.floor, 3)} &plusmn; ${fmt(H.floorSe, 3)} there, `
          + `the floor for any surface; the fit is ${fmt(H.gap, 3)} &plusmn; ${fmt(H.gapSe, 3)} above it.`];
      })(),
    ]).map(([k, v, d]) => `<div class="card"><div class="k">${k}</div><div class="v">${v}</div><div class="d">${d}</div></div>`).join("");

    const cap = f.engine === "rect" ? 3 : 6;             // each engine's upper bound on the null's spread
    const rule = (r) => (r.method === "lindsey" ? "empirical null" : r.method === "BH-fallback" ? "theoretical null (BH)" : r.method === "defer-invalid-null" ? "deferred" : r.method || "–");
    const cols = ROUND_COLOURS();
    $("rounds").innerHTML = `<table class="diag"><thead><tr><th>Round</th><th>Scored</th><th>Null centre</th><th>Null spread</th><th>&pi;<sub>0</sub></th><th>Rule</th><th>Admitted</th><th>${S.user ? "Site" : "Coin"} variance</th></tr></thead><tbody>` +
      rounds.map((r) => `<tr><td class="mono"><i class="dot" style="background:${cols[Math.min(cols.length - 1, r.round)]}"></i>${r.round}</td><td class="mono">${r.candidates}</td><td class="mono">${fmt(r.nullMean, 2)}</td><td class="mono">${fmt(r.nullSd, 2)}${r.nullSd >= cap - 1e-3 ? " <span class='cap'>cap</span>" : ""}</td><td class="mono">${fmt(r.pi0, 2)}</td><td>${rule(r)}</td><td class="mono"><b>${r.admitted}</b></td><td class="mono">${fmt(r.poolVariance, 4)}</td></tr>`).join("") +
      `</tbody></table>`;

    // what happened, in the gate's own terms
    const r0 = rounds[0], last = rounds[rounds.length - 1];
    let say = admitted
      ? `Admitted ${admitted} vertices, then stopped: the last round scored ${last.candidates} candidates and none cleared the gate.`
      : `Admitted nothing: the fit is the starting mesh.`;
    if (r0 && !admitted && r0.method === "lindsey" && r0.nullSd > 2.5) say += ` In round 0 the empirical null came out ${fmt(r0.nullSd, 1)} times as wide as the textbook one${r0.nullSd >= cap - 1e-3 ? " (its cap)" : ""}, so it took in the scores that stood out and nothing cleared the gate.`;
    else if (r0 && r0.method === "BH-fallback") say += " In round 0 the scores had no central peak to fit a null to, so the gate used the theoretical null.";
    else if (!admitted && !S.user && $("truth").value === "nothing") say += " There was nothing to find.";
    $("statustext").textContent = say + ` ${Math.round(f.ms)} ms.`;
  }

  // ------------------------------------------------------------------ the worker
  let worker;
  function setChip(kind, text) { const c = $("chip"); c.className = "chip " + kind; c.textContent = text; }
  function startWorker() {
    worker = new Worker($("gate").dataset.worker);
    worker.onmessage = (ev) => {
      const m = ev.data;
      if (m.type === "ready") { S.ready = true; run(); return; }
      if (m.type === "error") {
        S.busy = false; setChip("bad", "Error");
        let text = m.message;
        if (S.user && (m.log || []).some((l) => /unidentified/.test(l)))
          text = "The estimator could not separate the sites' own effects from the noise: most sites have too few trials. Give it sites with more trials each, or plain 0/1 rows, which this page groups into sites.";
        $("statustext").textContent = text;
        if (S.user) ownSay(text, true);
        return;
      }
      if (m.id !== S.id) { S.busy = false; if (S.queued) run(); return; }
      S.busy = false;
      S.fit = m; S.fitG = fitGrid(m);
      if (window.siteTally) window.siteTally("fit", 1, S.user ? `a decision mesh on ${S.user.rows.toLocaleString()} rows of your own data` : `a decision mesh on ${opts().sites.toLocaleString()} sites of coin flips`);
      $("results").classList.remove("stale");
      setChip("ok", "Fitted"); report(); drawFit();
      if (S.queued) run();
    };
    worker.onerror = (e) => { setChip("bad", "Error"); $("statustext").textContent = "The engine could not start in this browser: " + (e.message || e); };
  }
  function run() {
    if (!S.ready) return;
    if (S.busy) { S.queued = true; return; }
    S.queued = false;
    const o = opts();
    if (S.user) {                                        // a visitor's file: its sites as they are, no truth, no hash
      S.data = S.user.data; S.truthG = null;
      drawData();
      $("results").classList.add("stale");
      S.busy = true; S.id += 1;
      setChip("busy", "Fitting");
      $("statustext").textContent = `The ${o.engine === "rect" ? "rectangular" : "right-triangle"} estimator is fitting ${S.user.sites.toLocaleString()} sites from ${S.user.name}…`;
      worker.postMessage({ id: S.id, engine: o.engine, design: S.user.data.csv, seed: 7, q: o.q });
      return;
    }
    S.truthG = truthGrid(o.truth);
    S.data = makeData(o.truth, o.sites, o.flips, o.seed, o.sd);
    drawTruth(); drawData();
    $("results").classList.add("stale");
    S.busy = true; S.id += 1;
    setChip("busy", "Fitting");
    $("statustext").textContent = `The ${o.engine === "rect" ? "rectangular" : "right-triangle"} estimator is fitting ${o.sites.toLocaleString()} sites…`;
    worker.postMessage({ id: S.id, engine: o.engine, design: S.data.csv, seed: 7, q: o.q });
    writeHash();
  }
  let timer = 0;
  const soon = () => { clearTimeout(timer); timer = setTimeout(run, 180); };

  // ------------------------------------------------------------------ controls
  function readHash() {
    const h = new URLSearchParams(location.hash.slice(1));
    for (const k of ["engine", "truth", "sites", "flips"]) if (h.get(k) && [...$(k).options].some((o) => o.value === h.get(k))) $(k).value = h.get(k);
    if (h.get("seed")) $("seed").value = Math.max(1, parseInt(h.get("seed"), 10) || 1);
    if (h.get("sd") && isFinite(+h.get("sd"))) $("coinsd").value = Math.max(0, Math.min(1, +h.get("sd")));
    if (h.get("q") && isFinite(+h.get("q"))) $("q").value = Math.max(0.01, Math.min(0.3, +h.get("q")));
  }
  function writeHash() { const o = opts(); history.replaceState(null, "", `#engine=${o.engine}&truth=${o.truth}&sites=${o.sites}&flips=${o.flips}&sd=${o.sd}&q=${o.q}&seed=${o.seed}`); }
  function syncTools() { $("drawtools").hidden = $("truth").value !== "drawing"; }

  // the island and the fault line are reachable by link only (#truth=island): on them the first round's
  // scores are mostly signal, and the empirical null can widen until it absorbs them (see the notes to Sam)
  const wanted = new URLSearchParams(location.hash.slice(1)).get("truth");
  for (const k of ["hills", "peaks", "ring", "drawing", "nothing", "island", "fault"]) {
    if (TRUTHS[k].hidden && k !== wanted) continue;
    const o = document.createElement("option"); o.value = k; o.textContent = TRUTHS[k].name; $("truth").appendChild(o);
  }
  readHash(); syncTools();
  $("truth").addEventListener("change", () => { syncTools(); soon(); });
  $("sites").addEventListener("change", soon);
  $("engine").addEventListener("change", () => { $("statustext").textContent = "Loading the other estimator…"; soon(); });
  $("flips").addEventListener("change", soon);
  const sdLabel = () => { $("coinsdval").textContent = coinSd().toFixed(2); };
  $("coinsd").addEventListener("input", sdLabel); $("coinsd").addEventListener("change", soon); sdLabel();
  const qLabel = () => { $("qval").textContent = gateQ().toFixed(2); };
  $("q").addEventListener("input", qLabel); $("q").addEventListener("change", soon); qLabel();
  // a link on the page (or the back button) that changes the settings runs them
  window.addEventListener("hashchange", () => { if (S.user) { S.user = null; setOwnView(false); ownSay(""); } readHash(); syncTools(); sdLabel(); qLabel(); run(); });
  // hold to shake: the coins keep being flipped while the button is held, the flips redrawn each time, and the
  // mesh is fitted once to wherever they land when it is let go
  let holdT = 0, shaking = false, swallow = false;
  const shake = () => {
    if (!shaking) return;
    $("seed").value = +$("seed").value + 1;
    const o = opts();
    S.data = makeData(o.truth, o.sites, o.flips, o.seed, o.sd);
    drawData();
    const c = $("newdata").querySelector(".mini-coin");
    if (c) { c.classList.remove("spin"); void c.offsetWidth; c.classList.add("spin"); }
    setTimeout(shake, 170);
  };
  const letGo = () => {
    clearTimeout(holdT);
    if (!shaking) return;
    shaking = false; swallow = true;
    run();
  };
  $("newdata").addEventListener("pointerdown", () => { holdT = setTimeout(() => { shaking = true; shake(); }, 350); });
  ["pointerup", "pointerleave", "pointercancel"].forEach((ev) => $("newdata").addEventListener(ev, letGo));
  $("newdata").addEventListener("click", () => {
    if (swallow) { swallow = false; return; }            // the click that ends a hold: already refitted
    $("seed").value = +$("seed").value + 1; run();
  });
  $("showedges").addEventListener("change", drawFit);
  $("showadmit").addEventListener("change", drawFit);
  $("clear").addEventListener("click", () => { drawing.fill(BASE); S.truthG = truthGrid("drawing"); drawTruth(); soon(); });

  // painting on the truth: drag to raise the odds, with "lower" (or shift) to lower them
  const cv = $("cvtruth");
  let painting = false;
  function stroke(ev) {
    const r = cv.getBoundingClientRect(), x = (ev.clientX - r.left) / r.width, y = 1 - (ev.clientY - r.top) / r.height;
    const lower = $("lower").checked !== ev.shiftKey;
    brush(x, y, lower ? -0.35 : 0.35, 0.05);
    S.truthG = truthGrid("drawing"); drawTruth();
  }
  cv.addEventListener("pointerdown", (ev) => {
    if ($("truth").value !== "drawing") return;
    painting = true; cv.setPointerCapture(ev.pointerId); ev.preventDefault(); stroke(ev);
  });
  cv.addEventListener("pointermove", (ev) => { if (painting) stroke(ev); });
  const end = () => { if (painting) { painting = false; run(); } };
  cv.addEventListener("pointerup", end); cv.addEventListener("pointercancel", end);

  new MutationObserver(() => { LUT = ramp(); drawScale(); drawAll(); }).observe(document.documentElement, { attributes: true, attributeFilter: ["class"] });
  let rz = 0; window.addEventListener("resize", () => { clearTimeout(rz); rz = setTimeout(drawAll, 120); });

  // ------------------------------------------------------------------ your own data
  // A CSV read in this browser: two position columns and a 0/1 outcome per row (one trial each), or successes and
  // trials per row. Positions are rescaled to the unit square, like the coin sites. Plain 0/1 rows are grouped into
  // sites on a grid (about eight rows to a cell, at most 80 cells a side): the estimator fits each site's own effect
  // on the log-odds alongside the surface, and a site of a single trial cannot show one (the engine stops, unidentified).
  const MAX_ROWS = 200000, MIN_ROWS = 200, MAX_MB = 25;
  let table = null;                                     // { name, head, cols: one array of strings per column }
  const esc = (t) => String(t).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  function ownSay(text, bad) { const m = $("ownmsg"); m.textContent = text; m.classList.toggle("bad", !!bad); }
  const isNum = (v) => v !== "" && v !== undefined && isFinite(+v);
  function parseTable(text) {
    const lines = text.replace(/^﻿/, "").split(/\r\n|\n|\r/).filter((l) => l.trim() !== "");
    if (lines.length < 2) return { error: "The file needs a header row and rows of data under it." };
    const delim = [",", ";", "\t"].map((d) => [d, lines[0].split(d).length]).sort((a, b) => b[1] - a[1])[0][0];
    const unq = (f) => { f = f.trim(); return f.length > 1 && f[0] === '"' && f[f.length - 1] === '"' ? f.slice(1, -1).replace(/""/g, '"').trim() : f; };
    const head = lines[0].split(delim).map(unq);
    if (head.length < 3) return { error: "The file needs at least three columns: two for the position and one for the outcome." };
    if (lines.length - 1 > MAX_ROWS) return { error: `The file has ${(lines.length - 1).toLocaleString()} rows; this page takes at most ${MAX_ROWS.toLocaleString()}.` };
    const cols = head.map(() => new Array(lines.length - 1));
    for (let i = 1; i < lines.length; ++i) { const f = lines[i].split(delim); for (let j = 0; j < head.length; ++j) cols[j][i - 1] = f[j] === undefined ? "" : unq(f[j]); }
    return { head: head.map((h, j) => h || `column ${j + 1}`), cols };
  }
  // a first guess at each column's role, from its name, then from its values
  function guess(t) {
    const H = t.head.map((h) => h.toLowerCase()), n = t.cols[0].length;
    const numeric = t.cols.map((c) => { let ok = 0; const m = Math.min(n, 500); for (let i = 0; i < m; ++i) if (isNum(c[i])) ++ok; return ok >= 0.9 * m; });
    const binary = t.cols.map((c) => { const m = Math.min(n, 2000); for (let i = 0; i < m; ++i) if (c[i] !== "" && c[i] !== "0" && c[i] !== "1") return false; return true; });
    const find = (names, ok = () => true) => { for (const nm of names) { const j = H.indexOf(nm); if (j >= 0 && ok(j)) return j; } return -1; };
    const nn = find(["n", "trials", "flips", "count", "tries"]);
    let k = find(["outcome", "success", "successes", "event", "hit", "heads", "result", "label", "target", "k"], (j) => j !== nn);
    if (k < 0) k = binary.findIndex((b, j) => b && j !== nn);
    let x = find(["x", "lon", "longitude", "across"], (j) => j !== k && j !== nn);
    let y = find(["y", "lat", "latitude", "up"], (j) => j !== k && j !== nn && j !== x);
    const free = (j) => numeric[j] && j !== k && j !== nn && j !== x && j !== y;
    if (x < 0) x = t.head.findIndex((_, j) => free(j));
    if (y < 0) y = t.head.findIndex((_, j) => free(j));
    return { x, y, k, n: nn };
  }
  function fillSelect(id, head, pick, none) {
    $(id).innerHTML = (none ? `<option value="-1">${none}</option>` : "") + head.map((h, j) => `<option value="${j}">${esc(h)}</option>`).join("");
    $(id).value = String(pick >= 0 ? pick : none ? -1 : 0);
  }
  async function readFile(file) {
    if (!file) return;
    if (file.size > MAX_MB * 1e6) { ownSay(`The file is ${(file.size / 1e6).toFixed(1)} MB; this page reads files up to ${MAX_MB} MB.`, true); return; }
    ownSay(`Reading ${file.name}…`);
    let text;
    try { text = await file.text(); } catch (e) { ownSay("The file could not be read.", true); return; }
    const t = parseTable(text);
    if (t.error) { table = null; $("colpick").hidden = true; ownSay(t.error, true); return; }
    table = { name: file.name, head: t.head, cols: t.cols };
    const g = guess(table);
    fillSelect("colx", table.head, g.x); fillSelect("coly", table.head, g.y); fillSelect("colk", table.head, g.k);
    fillSelect("coln", table.head, g.n, "none: each row is one trial");
    $("colpick").hidden = false;
    ownSay(`${table.cols[0].length.toLocaleString()} rows and ${table.head.length} columns in ${file.name}. Check the columns, then fit.`);
  }
  // the chosen columns, checked, as sites on the unit square
  function build() {
    const jx = +$("colx").value, jy = +$("coly").value, jk = +$("colk").value, jn = +$("coln").value, H = table.head;
    if (new Set([jx, jy, jk].concat(jn >= 0 ? [jn] : [])).size !== (jn >= 0 ? 4 : 3)) return { error: "Choose a different column for each role." };
    const X = table.cols[jx], Y = table.cols[jy], Kc = table.cols[jk], Nc = jn >= 0 ? table.cols[jn] : null, m = X.length;
    const xs = [], ys = [], ks = [], ns = [];
    let skipped = 0, firstBad = null;
    for (let i = 0; i < m; ++i) {
      const cells = [[X[i], H[jx]], [Y[i], H[jy]], [Kc[i], H[jk]]].concat(Nc ? [[Nc[i], H[jn]]] : []);
      const bad = cells.find(([v]) => !isNum(v));
      if (bad) { ++skipped; if (!firstBad) firstBad = { row: i + 2, v: bad[0], col: bad[1] }; continue; }
      const k = +Kc[i], n = Nc ? +Nc[i] : 1;
      if (!Nc && k !== 0 && k !== 1) return { error: `The outcome column, ${H[jk]}, must hold 0 or 1; row ${i + 2} has “${Kc[i]}”. If it counts successes, choose its trials column too.` };
      if (Nc && (!(n >= 1) || n !== Math.round(n) || k !== Math.round(k) || k < 0 || k > n)) return { error: `Row ${i + 2} has ${Kc[i]} successes in ${Nc[i]} trials. Trials must be a whole number of at least 1, and successes a whole number from 0 up to the trials.` };
      xs.push(+X[i]); ys.push(+Y[i]); ks.push(k); ns.push(n);
    }
    if (firstBad && skipped > 0.05 * m) return { error: `Row ${firstBad.row}: “${firstBad.v}” in column ${firstBad.col} is not a number, and ${skipped.toLocaleString()} rows are like it.` };
    const rows = xs.length;
    if (rows < MIN_ROWS) return { error: `Only ${rows.toLocaleString()} usable rows; the estimator needs at least ${MIN_ROWS}.` };
    const range = (a) => { let lo = Infinity, hi = -Infinity; for (const v of a) { if (v < lo) lo = v; if (v > hi) hi = v; } return [lo, hi]; };
    const xr = range(xs), yr = range(ys);
    for (const [r, j] of [[xr, jx], [yr, jy]]) if (r[0] === r[1]) return { error: `Column ${H[j]} holds a single value; the position needs two columns that vary.` };
    let K = 0, N = 0;
    for (let i = 0; i < rows; ++i) { K += ks[i]; N += ns[i]; }
    if (K === 0 || K === N) return { error: `Every outcome is ${K === 0 ? "a failure" : "a success"}, so there is nothing to fit.` };
    const ux = (v) => (v - xr[0]) / (xr[1] - xr[0]), uy = (v) => (v - yr[0]) / (yr[1] - yr[0]);
    let sx, sy, sn, sk, grid = 0;
    if (Nc) { sx = xs.map(ux); sy = ys.map(uy); sn = ns; sk = ks; }
    else {
      grid = Math.max(8, Math.min(80, Math.floor(Math.sqrt(rows / 8))));
      const cells = new Map();
      for (let i = 0; i < rows; ++i) {
        const u = ux(xs[i]), v = uy(ys[i]), c = Math.min(grid - 1, Math.floor(u * grid)) * grid + Math.min(grid - 1, Math.floor(v * grid));
        let q = cells.get(c);
        if (!q) cells.set(c, (q = { u: 0, v: 0, n: 0, k: 0 }));
        q.u += u; q.v += v; q.n += 1; q.k += ks[i];
      }
      sx = []; sy = []; sn = []; sk = [];
      for (const q of cells.values()) { sx.push(q.u / q.n); sy.push(q.v / q.n); sn.push(q.n); sk.push(q.k); }
    }
    const lines = ["wala,wac,n,k"];
    for (let i = 0; i < sx.length; ++i) lines.push(sx[i].toFixed(6) + "," + sy[i].toFixed(6) + "," + sn[i] + "," + sk[i]);
    return {
      name: table.name, xName: H[jx], yName: H[jy], xr, yr, rows, skipped, grid, sites: sx.length, rate: K / N,
      data: { x: Float64Array.from(sx), y: Float64Array.from(sy), n: Int32Array.from(sn), k: Int32Array.from(sk), csv: lines.join("\n") + "\n" },
    };
  }
  // the page shown for a file: no truth panel and no coin controls, the axes named, the colours centred on its rate
  const short = (v) => { const a = Math.abs(v); return a >= 1000 ? Math.round(v).toLocaleString() : a !== 0 && a < 1e-3 ? v.toExponential(2) : String(+v.toPrecision(4)); };
  const coinCaption = $("cvdata").closest("figure").querySelector("figcaption").innerHTML;
  function setOwnView(on) {
    const u = S.user;
    $("cvtruth").closest("figure").hidden = on;
    document.querySelector(".explorer.gate .stage").classList.toggle("two", on);
    for (const id of ["truth", "sites", "flips", "coinsd"]) $(id).closest(".ctl").hidden = on;
    $("drawtools").hidden = on || $("truth").value !== "drawing";
    $("newdata").hidden = on;
    document.querySelector(".explorer.gate .tries").hidden = on;
    $("ownback").hidden = !on;
    $("coinnote").hidden = on; $("sitenote").hidden = !on;
    $("cvdata").closest("figure").querySelector("figcaption").innerHTML = on ? `<b>Your data</b> <span class="muted">each site's share of successes</span>` : coinCaption;
    document.querySelectorAll(".explorer.gate .cv .ax").forEach((e) => e.remove());
    if (on) for (const id of ["cvdata", "cvfit"])
      $(id).parentElement.insertAdjacentHTML("beforeend", `<div class="ax x">${esc(u.xName)}: ${short(u.xr[0])} to ${short(u.xr[1])}</div><div class="ax y">${esc(u.yName)}: ${short(u.yr[0])} to ${short(u.yr[1])}</div>`);
    CENTRE = on ? Math.log(u.rate / (1 - u.rate)) : BASE;
    LUT = ramp(); drawScale();
  }
  $("file").addEventListener("change", () => readFile($("file").files[0]));
  const drop = $("drop");
  ["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (ev) => { ev.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((t) => drop.addEventListener(t, () => drop.classList.remove("over")));
  drop.addEventListener("drop", (ev) => { ev.preventDefault(); ev.stopPropagation(); readFile(ev.dataTransfer.files[0]); });
  // a file let go anywhere else on the page is read too, rather than the browser leaving the page to open it
  const carriesFile = (ev) => ev.dataTransfer && [...(ev.dataTransfer.types || [])].includes("Files");
  window.addEventListener("dragover", (ev) => { if (carriesFile(ev)) { ev.preventDefault(); drop.classList.add("over"); } });
  window.addEventListener("dragleave", (ev) => { if (!ev.relatedTarget) drop.classList.remove("over"); });
  window.addEventListener("drop", (ev) => {
    if (!carriesFile(ev)) return;
    ev.preventDefault(); drop.classList.remove("over");
    if (ev.dataTransfer.files[0]) { readFile(ev.dataTransfer.files[0]); drop.scrollIntoView({ behavior: "smooth", block: "center" }); }
  });
  $("ownfit").addEventListener("click", () => {
    if (!table) return;
    const u = build();
    if (u.error) { ownSay(u.error, true); return; }
    S.user = u; setOwnView(true);
    ownSay(`${u.rows.toLocaleString()} rows${u.skipped ? ` (${u.skipped.toLocaleString()} skipped for a missing or non-numeric value)` : ""}`
      + (u.grid ? `, grouped into ${u.sites.toLocaleString()} sites on a ${u.grid} × ${u.grid} grid, since the estimator fits each site's own effect and a single trial cannot show one.` : ", each row a site."));
    $("statusbar").scrollIntoView({ behavior: "smooth", block: "start" });
    run();
  });
  $("ownback").addEventListener("click", () => { S.user = null; setOwnView(false); ownSay(""); S.fit = null; S.fitG = null; run(); });

  drawScale();
  S.truthG = truthGrid($("truth").value);
  S.data = makeData($("truth").value, +$("sites").value, +$("flips").value, +$("seed").value, coinSd());
  drawAll();
  setChip("busy", "Loading");
  $("statustext").textContent = "Loading the estimator (about 370 kB)…";
  if (typeof Worker === "undefined" || typeof WebAssembly === "undefined") {
    setChip("bad", "Unsupported"); $("statustext").textContent = "This browser has no WebAssembly workers, which the estimator needs.";
  } else startWorker();
})();
