// /gate: coin flips on a square, fitted by the triangular decision-mesh estimator itself (fit-worker.js runs
// triangular-decision-mesh/core compiled to WebAssembly). This file makes the data, sends it to the worker,
// and draws what comes back: the truth, the data, the fitted mesh, and the gate's account of every round.
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
    const q = Math.max(0, Math.min(255, Math.round((((logit - BASE) / VMAX) * 0.5 + 0.5) * 255)));
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
  const S = { data: null, fit: null, truthG: null, fitG: null, busy: false, queued: false, id: 0, ready: false };
  window.gateDemo = S;
  const opts = () => ({ engine: $("engine").value, truth: $("truth").value, sites: +$("sites").value, flips: +$("flips").value, seed: +$("seed").value, sd: coinSd() });

  function drawTruth() {
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
    for (let q = 0; q <= 8; ++q) { const t = BASE + ((q / 8) * 2 - 1) * VMAX, rgb = colour(t); stops.push(`rgb(${rgb[0]},${rgb[1]},${rgb[2]})`); }
    const pct = (t) => Math.round(100 * expit(t)) + "%";
    $("scale").innerHTML = `${pct(BASE - VMAX)} <i style="background:linear-gradient(90deg,${stops.join(",")})"></i> ${pct(BASE + VMAX)} chance of heads`;
  }

  // ------------------------------------------------------------------ results
  const fmt = (v, d = 3) => (v === undefined || v === null || v !== v ? "–" : (+v).toFixed(d));
  function rmse(a, b) {
    let s = 0, n = 0;
    for (let k = 0; k < a.length; ++k) if (a[k] === a[k] && b[k] === b[k]) { s += (a[k] - b[k]) ** 2; ++n; }
    return Math.sqrt(s / n);
  }
  function report() {
    const f = S.fit, rounds = f.rounds, admitted = rounds.reduce((a, r) => a + r.admitted, 0);
    const flat = new Float32Array(D * D).fill(f.baseline);
    const err = rmse(S.fitG, S.truthG), errFlat = rmse(flat, S.truthG);
    $("cards").innerHTML = [
      ["Vertices admitted", admitted, `over ${rounds.filter((r) => r.admitted > 0).length} round${rounds.filter((r) => r.admitted > 0).length === 1 ? "" : "s"}; ${f.tri.length / f.stride} ${f.engine === "rect" ? "rectangles" : "triangles"}`],
      ["Error against the truth", fmt(err, 3), `log-odds RMSE; a flat fit: ${fmt(errFlat, 3)}`],
      ["Coin variance", fmt(f.poolVariance, 4), `the coin effects' variance; truly ${fmt(coinSd() ** 2, 4)}`],
      ["Held-out deviance", f.heldout ? fmt(f.heldout.deviance, 3) : "–", f.heldout ? `per site, on ${f.heldout.pools.toLocaleString()} sites never fitted` : ""],
    ].map(([k, v, d]) => `<div class="card"><div class="k">${k}</div><div class="v">${v}</div><div class="d">${d}</div></div>`).join("");

    const cap = f.engine === "rect" ? 3 : 6;             // each engine's upper bound on the null's spread
    const rule = (r) => (r.method === "lindsey" ? "empirical null" : r.method === "BH-fallback" ? "theoretical null (BH)" : r.method === "defer-invalid-null" ? "deferred" : r.method || "–");
    const cols = ROUND_COLOURS();
    $("rounds").innerHTML = `<table class="diag"><thead><tr><th>Round</th><th>Scored</th><th>Null centre</th><th>Null spread</th><th>&pi;<sub>0</sub></th><th>Rule</th><th>Admitted</th><th>Coin variance</th></tr></thead><tbody>` +
      rounds.map((r) => `<tr><td class="mono"><i class="dot" style="background:${cols[Math.min(cols.length - 1, r.round)]}"></i>${r.round}</td><td class="mono">${r.candidates}</td><td class="mono">${fmt(r.nullMean, 2)}</td><td class="mono">${fmt(r.nullSd, 2)}${r.nullSd >= cap - 1e-3 ? " <span class='cap'>cap</span>" : ""}</td><td class="mono">${fmt(r.pi0, 2)}</td><td>${rule(r)}</td><td class="mono"><b>${r.admitted}</b></td><td class="mono">${fmt(r.poolVariance, 4)}</td></tr>`).join("") +
      `</tbody></table>`;

    // what happened, in the gate's own terms
    const r0 = rounds[0], last = rounds[rounds.length - 1];
    let say = admitted
      ? `Admitted ${admitted} vertices, then stopped: the last round scored ${last.candidates} candidates and none cleared the gate.`
      : `Admitted nothing: the fit is the starting mesh.`;
    if (r0 && !admitted && r0.method === "lindsey" && r0.nullSd > 2.5) say += ` In round 0 the empirical null came out ${fmt(r0.nullSd, 1)} times as wide as the textbook one${r0.nullSd >= cap - 1e-3 ? " (its cap)" : ""}, so it took in the scores that stood out and nothing cleared the gate.`;
    else if (r0 && r0.method === "BH-fallback") say += " In round 0 the scores had no central peak to fit a null to, so the gate used the theoretical null.";
    else if (!admitted && $("truth").value === "nothing") say += " There was nothing to find.";
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
      if (m.type === "error") { S.busy = false; setChip("bad", "Error"); $("statustext").textContent = m.message; console.error(m.log); return; }
      if (m.id !== S.id) { S.busy = false; if (S.queued) run(); return; }
      S.busy = false;
      S.fit = m; S.fitG = fitGrid(m);
      if (window.siteTally) window.siteTally("fit", 1, `a decision mesh on ${opts().sites.toLocaleString()} sites of coin flips`);
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
    S.truthG = truthGrid(o.truth);
    S.data = makeData(o.truth, o.sites, o.flips, o.seed, o.sd);
    drawTruth(); drawData();
    $("results").classList.add("stale");
    S.busy = true; S.id += 1;
    setChip("busy", "Fitting");
    $("statustext").textContent = `The ${o.engine === "rect" ? "rectangular" : "right-triangle"} estimator is fitting ${o.sites.toLocaleString()} sites…`;
    worker.postMessage({ id: S.id, engine: o.engine, design: S.data.csv, seed: 7 });
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
  }
  function writeHash() { const o = opts(); history.replaceState(null, "", `#engine=${o.engine}&truth=${o.truth}&sites=${o.sites}&flips=${o.flips}&sd=${o.sd}&seed=${o.seed}`); }
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
  // a link on the page (or the back button) that changes the settings runs them
  window.addEventListener("hashchange", () => { readHash(); syncTools(); sdLabel(); run(); });
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
