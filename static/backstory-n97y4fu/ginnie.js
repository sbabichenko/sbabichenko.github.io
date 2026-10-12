// /backstory, "Ginnie Mae": the estimator on the data it was built for. One month of Ginnie Mae's public pool
// disclosure (30-year pools of at least 25 loans; a pool's terminations are the month's fall in its loan count), fitted
// in this page by the site's own WebAssembly engines (mesh/fit-worker.js), on the engine's half-and-half split.
// Three drawings, one claim each:
//   the pools   the raw data are too noisy to read a surface off: most pools show no terminations or one;
//   the fit     the mesh is fine only where the pools support it (young loans, the common coupons), coarse elsewhere;
//   the cut     along one coupon, the fitted rate against loan age passes through the pools' pooled rates.
window.Ginnie = (function () {
  "use strict";
  const root = document.getElementById("ginnie");
  if (!root) return { shown() {} };
  const $ = (id) => document.getElementById(id);
  const isDark = () => document.documentElement.classList.contains("dark");
  const expit = (t) => 1 / (1 + Math.exp(-t));
  const MONTHS = { 202002: "February 2020", 202606: "June 2026" };
  // The months of public pool data this screen can fit, as the files ginnie/<pools>_<month>.csv (age, coupon, loans,
  // terminations). The first whose file exists is fitted at the start; the others are a click away under the map.
  //   thirty_202606  Ginnie Mae 30-year pools only, June 2026: built by backstory/tools/ginnie_month.py once the two raw
  //                  disclosure files are on disk; absent until then.
  //   fnma_202002    Fannie Mae 30-year fixed (prefix CL), resecuritisations (seller SCR) left out.
  //   all_202606     Ginnie Mae, every term, June 2026 (the mesh repository's smm_design_202606.csv; no term column).
  //   all_202002     Ginnie Mae 30-year pools, February 2020 (the disclosure panel).
  const SETS = [["thirty", 202606, "Ginnie Mae, June 2026", "30-year", "Ginnie Mae 2026"], ["fnma", 202002, "Fannie Mae, February 2020", "30-year", "Fannie Mae 2020"],
                ["all", 202606, "Ginnie Mae, June 2026", "all terms", "Ginnie Mae 2026"], ["all", 202002, "Ginnie Mae, February 2020", "30-year", "Ginnie Mae 2020"]];
  const VMAX = 1.5;                         // the colours span this many logits either side of the month's overall rate
  const BAND = 0.125;                       // the cut: pools within this of the coupon
  // ...pooled by age: two months at a time while the rate moves fast (the first three years), six months after
  const EDGES = (() => { const e = []; for (let a = 0; a < 36; a += 2) e.push(a); for (let a = 36; a <= 366; a += 6) e.push(a); return e; })();
  const binOf = (age) => (age < 36 ? Math.floor(age / 2) : Math.min(EDGES.length - 2, 18 + Math.floor((age - 36) / 6)));
  const M = { l: 46, r: 10, t: 10, b: 34 }; // plot margins, css px

  const S = { data: null, fit: null, grid: null, coupon: null, hover: null, id: 0, busy: false, shownOnce: false };
  const cache = new Map();

  // ------------------------------------------------------------------ data: wala, wac, loans, terminations per pool
  async function load(name) {
    if (cache.has(name)) return cache.get(name);
    const text = await (await fetch(root.dataset.base.replace(/\/?$/, "/") + name + ".csv")).text();
    const lines = text.trim().split("\n"), N = lines.length;
    const age = new Float32Array(N), wac = new Float32Array(N), n = new Float64Array(N), k = new Float64Array(N);
    let loans = 0, terms = 0;
    for (let i = 0; i < N; ++i) { const f = lines[i].split(","); age[i] = +f[0]; wac[i] = +f[1]; n[i] = +f[2]; k[i] = +f[3]; loans += n[i]; terms += k[i]; }
    // the coupon the cut starts at: the eighth of a point with the most loans
    const by = new Map();
    for (let i = 0; i < N; ++i) { const c = Math.round(wac[i] * 8) / 8; by.set(c, (by.get(c) || 0) + n[i]); }
    const busiest = [...by.entries()].sort((a, b) => b[1] - a[1])[0][0];
    const d = { name, text, N, age, wac, n, k, loans, terms, rate: terms / loans, busiest };
    cache.set(name, d);
    return d;
  }

  // ------------------------------------------------------------------ the engine
  const worker = new Worker(root.dataset.worker);
  let ready = false, pending = null;
  worker.onmessage = (ev) => {
    const d = ev.data;
    if (d.type === "ready") { ready = true; if (pending) { worker.postMessage(pending); pending = null; } return; }
    if (d.id !== S.id) return;
    S.busy = false;
    if (d.type !== "fit") { status("bad", "Failed", d.message || "the engine stopped"); return; }
    S.fit = d; S.grid = null; S.tri9 = null;
    report(); drawAll();
    keep(d);
  };
  // A fit takes seconds, so the map is not left empty meanwhile: the same fit from last time is drawn at once and the
  // live one replaces it when it lands (same data, same seed, so the same surface). "Last time" is this browser's own
  // last run (localStorage), or else the copy shipped beside the data (ginnie/<name>.fit.json, made by
  // backstory/tools/prefit.js). Only the default settings are kept; the held-out rows are left out for size.
  const eng = () => ($("gn-engine").value === "rect" ? ".rect" : ""), fitKey = () => `gn-fit:${$("gn-pools").value}_${$("gn-month").value}${eng()}`, isDefault = () => $("gn-q").value === "0.4";
  function slim(d) { return { type: "fit", engine: d.engine, tri: Array.from(d.tri), stride: d.stride, verts: d.verts, rounds: d.rounds, baseline: d.baseline, bounds: Array.from(d.bounds), poolVariance: d.poolVariance, ms: d.ms }; }
  function keep(d) { if (!isDefault()) return; try { const j = JSON.stringify(slim(d)); S.lastSlim = j; localStorage.setItem(fitKey(), j); } catch (e) {} }
  async function kept(name) {
    if (!isDefault()) return null;
    try { const j = localStorage.getItem(fitKey()); if (j) return JSON.parse(j); } catch (e) {}
    if (files.includes(name + eng() + ".fit")) try { const r = await fetch(root.dataset.base.replace(/\/?$/, "/") + name + eng() + ".fit.json"); if (r.ok) return await r.json(); } catch (e) {}
    return null;
  }
  window.GinnieFit = () => S.lastSlim;   // for backstory/tools/prefit.js (which saves <name>.fit.json and <name>.rect.fit.json)
  function status(kind, chip, text) { const c = $("gn-chip"); c.className = "chip " + kind; c.textContent = chip; $("gn-statustext").textContent = text; }

  async function run() {
    const name = $("gn-pools").value + "_" + $("gn-month").value;
    status("busy", "Reading", MONTHS[$("gn-month").value] + "…");
    const d = await load(name);
    if (S.data !== d) { S.data = d; if (S.coupon == null) S.coupon = d.busiest; }
    const id = S.id + 1, old = await kept(name);
    if (id !== S.id + 1) return;                    // another run started while this one was reading
    S.fit = old; S.grid = null; S.busy = true; d.occ = null;
    if (old) report(); else root.classList.add("stale");
    status("busy", "Fitting", `${d.N.toLocaleString()} pools, ${d.loans.toLocaleString()} loans…`);
    drawAll();
    const msg = { id: ++S.id, design: d.text, seed: 1, q: +$("gn-q").value, engine: $("gn-engine").value };
    if (ready) worker.postMessage(msg); else pending = msg;
  }

  // ------------------------------------------------------------------ the surface
  // the fitted height (logits from the baseline) at a point of the normalised square: the cell or triangle holding it
  function heightAt(u, v) {
    const f = S.fit, t = f.engine !== "rect" && S.tri9 ? S.tri9 : f.tri;   // the surface at the slider's depth, or the whole fit
    if (f.engine === "rect") {
      for (let i = 0; i < t.length; i += 8) {
        if (u < t[i] - 1e-9 || u > t[i + 2] + 1e-9 || v < t[i + 1] - 1e-9 || v > t[i + 3] + 1e-9) continue;
        const a = (u - t[i]) / (t[i + 2] - t[i]), b = (v - t[i + 1]) / (t[i + 3] - t[i + 1]);
        return (1 - a) * (1 - b) * t[i + 4] + a * (1 - b) * t[i + 5] + (1 - a) * b * t[i + 6] + a * b * t[i + 7];
      }
      return NaN;
    }
    for (let i = 0; i < t.length; i += 9) {
      const ax = t[i], ay = t[i + 1], bx = t[i + 3], by = t[i + 4], cx = t[i + 6], cy = t[i + 7];
      const det = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy);
      if (Math.abs(det) < 1e-15) continue;
      const l1 = ((by - cy) * (u - cx) + (cx - bx) * (v - cy)) / det, l2 = ((cy - ay) * (u - cx) + (ax - cx) * (v - cy)) / det, l3 = 1 - l1 - l2;
      if (l1 < -1e-7 || l2 < -1e-7 || l3 < -1e-7) continue;
      return l1 * t[i + 2] + l2 * t[i + 5] + l3 * t[i + 8];
    }
    return NaN;
  }
  const clamp01 = (x) => Math.max(0, Math.min(1, x));
  // the fitted monthly rate at an age and coupon: the average rate of pools there (the pool effect is mean-neutral)
  function rateAt(age, wac) {
    const [x0, x1, y0, y1] = S.fit.bounds;
    const h = heightAt(clamp01((age - x0) / (x1 - x0)), clamp01((wac - y0) / (y1 - y0)));
    return h === h ? expit(S.fit.baseline + h) : NaN;
  }
  const D = 220;
  function grid() {
    if (S.grid) return S.grid;
    const g = new Float32Array(D * D);
    for (let r = 0; r < D; ++r) for (let c = 0; c < D; ++c) g[r * D + c] = heightAt((c + 0.5) / D, 1 - (r + 0.5) / D);
    return (S.grid = g);
  }

  // ------------------------------------------------------------------ colours: slow (blue) to fast (orange), the
  // month's overall rate in the paper's own tone
  function rgb(dl) {
    const dark = isDark(), lo = dark ? [110, 150, 255] : [31, 72, 200], mid = dark ? [34, 35, 38] : [250, 247, 238], hi = dark ? [255, 140, 80] : [205, 66, 20];
    const s = Math.max(-1, Math.min(1, dl / VMAX)), a = s < 0 ? lo : hi, w = Math.pow(Math.abs(s), 0.8);
    return [0, 1, 2].map((j) => Math.round(mid[j] + (a[j] - mid[j]) * w));
  }
  const css = (c) => `rgb(${c[0]},${c[1]},${c[2]})`;
  const ink = () => (isDark() ? "#ececef" : "#1b1b1f"), faint = () => (isDark() ? "#8a8a93" : "#6b6b74"), rule = () => (isDark() ? "rgba(255,255,255,0.14)" : "rgba(0,0,0,0.13)");

  // ------------------------------------------------------------------ drawing
  function canvas(id) {
    const c = $(id), r = c.getBoundingClientRect(), dpr = Math.min(2, window.devicePixelRatio || 1);
    const w = Math.max(1, Math.round(r.width * dpr)), h = Math.max(1, Math.round(r.height * dpr));
    if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
    const ctx = c.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, r.width, r.height);
    return { c, ctx, W: r.width, H: r.height, pw: r.width - M.l - M.r, ph: r.height - M.t - M.b };
  }
  function bounds() {
    if (S.fit) return S.fit.bounds;
    const d = S.data; let x1 = 0, y0 = Infinity, y1 = -Infinity;
    for (let i = 0; i < d.N; ++i) { if (d.age[i] > x1) x1 = d.age[i]; if (d.wac[i] < y0) y0 = d.wac[i]; if (d.wac[i] > y1) y1 = d.wac[i]; }
    return [0, x1, y0, y1];
  }
  function ticks(lo, hi, step) { const out = []; for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(6)); return out; }
  // age across, coupon up
  function mapAxes(p, b) {
    const { ctx } = p, [x0, x1, y0, y1] = b;
    const X = (a) => M.l + ((a - x0) / (x1 - x0)) * p.pw, Y = (w) => M.t + (1 - (w - y0) / (y1 - y0)) * p.ph;
    ctx.font = "11px ui-sans-serif, system-ui, sans-serif"; ctx.fillStyle = faint(); ctx.strokeStyle = rule(); ctx.lineWidth = 1;
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    for (const a of ticks(x0, x1, 60)) ctx.fillText(a, X(a), M.t + p.ph + 5);
    ctx.fillText("loan age, months", M.l + p.pw / 2, M.t + p.ph + 19);
    ctx.textAlign = "right"; ctx.textBaseline = "middle";
    for (const w of ticks(y0, y1, 1)) ctx.fillText(w + "%", M.l - 6, Y(w));
    ctx.save(); ctx.translate(11, M.t + p.ph / 2); ctx.rotate(-Math.PI / 2); ctx.textAlign = "center"; ctx.fillText("coupon", 0, 0); ctx.restore();
    ctx.strokeRect(M.l + 0.5, M.t + 0.5, p.pw - 1, p.ph - 1);
    return { X, Y };
  }
  // the coupon of the cut, marked on both maps
  function cutMark(p, ax) {
    if (S.coupon == null || !$("gn-cvcut").offsetParent) return;   // no line when the cut drawing is not on screen
    const { ctx } = p, y = ax.Y(S.coupon);
    if (y < M.t || y > M.t + p.ph) return;
    ctx.save(); ctx.strokeStyle = ink(); ctx.lineWidth = 1; ctx.setLineDash([5, 4]);
    ctx.beginPath(); ctx.moveTo(M.l, y); ctx.lineTo(M.l + p.pw, y); ctx.stroke(); ctx.restore();
  }

  function drawPools() {
    const d = S.data; if (!d) return;
    const p = canvas("gn-cvpools"), b = bounds(), ax = mapAxes(p, b), { ctx } = p;
    const base = Math.log(d.rate / (1 - d.rate));
    ctx.save(); ctx.beginPath(); ctx.rect(M.l, M.t, p.pw, p.ph); ctx.clip();
    ctx.globalAlpha = 0.75;
    for (let i = 0; i < d.N; ++i) {
      ctx.fillStyle = css(rgb(Math.log((d.k[i] + 0.5) / (d.n[i] - d.k[i] + 0.5)) - base));
      const r = Math.max(0.9, Math.min(5, Math.sqrt(d.n[i]) / 9));
      ctx.fillRect(ax.X(d.age[i]) - r, ax.Y(d.wac[i]) - r, 2 * r, 2 * r);
    }
    ctx.restore();
    cutMark(p, ax);
  }

  // where the month has no pools the surface is only the mesh carried outward: those parts are drawn faded
  const OC = 44;
  function occupied() {
    const d = S.data; if (d.occ && d.occB === S.fit.bounds.join()) return d.occ;
    const [x0, x1, y0, y1] = S.fit.bounds, raw = new Uint8Array(OC * OC), occ = new Float32Array(OC * OC);
    for (let i = 0; i < d.N; ++i) {
      const c = Math.min(OC - 1, Math.max(0, Math.floor(((d.age[i] - x0) / (x1 - x0)) * OC))), r = Math.min(OC - 1, Math.max(0, Math.floor((1 - (d.wac[i] - y0) / (y1 - y0)) * OC)));
      raw[r * OC + c] = 1;
    }
    // softened: each cell takes the share of its 3 x 3 neighbourhood that holds pools, twice over
    let cur = Float32Array.from(raw);
    for (let pass = 0; pass < 2; ++pass) {
      const nxt = new Float32Array(OC * OC);
      for (let r = 0; r < OC; ++r) for (let c = 0; c < OC; ++c) {
        let sum = 0;
        for (let dr = -1; dr <= 1; ++dr) for (let dc = -1; dc <= 1; ++dc) { const rr = r + dr, cc = c + dc; if (rr >= 0 && rr < OC && cc >= 0 && cc < OC) sum += cur[rr * OC + cc]; }
        nxt[r * OC + c] = sum / 9;
      }
      cur = nxt;
    }
    for (let k = 0; k < OC * OC; ++k) occ[k] = Math.min(1, cur[k] * 6);
    d.occ = occ; d.occB = S.fit.bounds.join();
    return occ;
  }
  const off = document.createElement("canvas"); off.width = off.height = D;
  // Where the data is: the picture cut into small cells, the loans in each. Drawn over the map as a faint stipple, one
  // dot per cell that holds pools, its AREA in proportion to the cell's loans, so the eye reads the loans where they are
  // and a cell with a handful is a speck. (The few fullest cells are capped at the 99th percentile, or they alone would show.)
  const GC = 120, GR = 64;
  function density() {
    const d = S.data, key = bounds().join(); if (d.den && d.denB === key) return d.den;
    const [x0, x1, y0, y1] = bounds(), g = new Float64Array(GC * GR); let top = 0;
    for (let i = 0; i < d.N; ++i) {
      const c = Math.min(GC - 1, Math.max(0, Math.floor(((d.age[i] - x0) / (x1 - x0)) * GC))), r = Math.min(GR - 1, Math.max(0, Math.floor((1 - (d.wac[i] - y0) / (y1 - y0)) * GR)));
      g[r * GC + c] += d.n[i];
    }
    const full = Array.from(g).filter((v) => v > 0).sort((a, b) => a - b); top = full[Math.floor(0.99 * (full.length - 1))] || 1;
    d.den = { g, top }; d.denB = key;
    return d.den;
  }
  function drawDensity(p) {
    if (!$("gn-where").classList.contains("on")) return;
    const { ctx } = p, { g, top } = density(), cw = p.pw / GC, ch = p.ph / GR;
    ctx.fillStyle = isDark() ? "rgba(255,255,255,0.3)" : "rgba(15,15,25,0.26)"; ctx.beginPath();
    for (let r = 0; r < GR; ++r) for (let c = 0; c < GC; ++c) {
      const v = g[r * GC + c]; if (!v) continue;
      const rad = 0.52 * Math.min(cw, ch) * Math.sqrt(Math.min(1, v / top)), x = M.l + (c + 0.5) * cw, y = M.t + (r + 0.5) * ch;
      ctx.moveTo(x + rad, y); ctx.arc(x, y, rad, 0, 2 * Math.PI);
    }
    ctx.fill();
  }
  function drawFit() {
    const p = canvas("gn-cvfit"), b = bounds(), ax = mapAxes(p, b), { ctx } = p;
    S.fitPlot = { p, b };
    if (!S.fit || p.pw <= 0 || p.ph <= 0) return;   // nothing to draw into while the map is off screen
    // the depth the slider is at (backstory/rounds.js): the mesh down to that depth, the surface it gives, and the vertices
    // admitted at that depth or coarser. A new fit puts the slider at the end, the whole mesh.
    const stops = window.MeshRounds.depths(S.fit), sr = $("gn-depth");
    if (sr._fit !== S.fit) { const end = +sr.value >= +sr.max; sr._fit = S.fit; sr.max = stops.length - 1; if (end || +sr.value > +sr.max) sr.value = sr.max; }
    const stop = stops[Math.min(stops.length - 1, +sr.value)];
    if (S.tri9 !== stop.tri) { S.tri9 = stop.tri; S.grid = null; }
    const g = grid(), octx = off.getContext("2d"), img = octx.createImageData(D, D);
    const occ = occupied(), m0 = rgb(0);
    for (let k = 0; k < D * D; ++k) {
      const c = rgb(g[k] === g[k] ? g[k] : 0), r = Math.floor(k / D), col = k % D;
      // the softened occupancy, read between its cells
      const fr = Math.max(0, Math.min(OC - 1.001, ((r + 0.5) * OC) / D - 0.5)), fc = Math.max(0, Math.min(OC - 1.001, ((col + 0.5) * OC) / D - 0.5));
      const r0 = Math.floor(fr), c0 = Math.floor(fc), tr = fr - r0, tc = fc - c0;
      const o = (1 - tr) * ((1 - tc) * occ[r0 * OC + c0] + tc * occ[r0 * OC + c0 + 1]) + tr * ((1 - tc) * occ[(r0 + 1) * OC + c0] + tc * occ[(r0 + 1) * OC + c0 + 1]);
      const w = 0.22 + 0.78 * o;
      for (let j = 0; j < 3; ++j) img.data[4 * k + j] = Math.round(m0[j] + (c[j] - m0[j]) * w);
      img.data[4 * k + 3] = 255;
    }
    octx.putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = true; ctx.drawImage(off, M.l, M.t, p.pw, p.ph);
    drawDensity(p);
    const X = (u) => M.l + u * p.pw, Y = (v) => M.t + (1 - v) * p.ph, t = S.fit.tri;
    const e = stop.lines;
    if ($("gn-edges").checked) {
      ctx.strokeStyle = isDark() ? "rgba(255,255,255,0.3)" : "rgba(20,20,30,0.36)"; ctx.lineWidth = 0.8; ctx.beginPath();
      if (e) for (let i = 0; i < e.length; i += 6) { ctx.moveTo(X(e[i]), Y(e[i + 1])); ctx.lineTo(X(e[i + 2]), Y(e[i + 3])); ctx.lineTo(X(e[i + 4]), Y(e[i + 5])); ctx.closePath(); }
      else if (S.fit.engine === "rect") for (let i = 0; i < t.length; i += 8) ctx.rect(X(t[i]), Y(t[i + 3]), X(t[i + 2]) - X(t[i]), Y(t[i + 1]) - Y(t[i + 3]));
      else for (let i = 0; i < t.length; i += 9) { ctx.moveTo(X(t[i]), Y(t[i + 1])); ctx.lineTo(X(t[i + 3]), Y(t[i + 4])); ctx.lineTo(X(t[i + 6]), Y(t[i + 7])); ctx.closePath(); }
      ctx.stroke();
    }
    if ($("gn-admit").checked) for (const v of S.fit.verts) {
      if (!v.admitted || window.MeshRounds.level(v.x, v.y) > stop.depth) continue;
      ctx.beginPath(); ctx.arc(X(v.x), Y(v.y), Math.max(1.5, 5 - 0.5 * window.MeshRounds.level(v.x, v.y)), 0, 2 * Math.PI);   // larger when coarser
      ctx.fillStyle = isDark() ? "#f5c451" : "#18181b"; ctx.fill(); ctx.lineWidth = 1; ctx.strokeStyle = isDark() ? "#111" : "#fff"; ctx.stroke();
    }
    cutMark(p, ax);
    if (S.hover) { ctx.beginPath(); ctx.arc(ax.X(S.hover.age), ax.Y(S.hover.wac), 4.5, 0, 2 * Math.PI); ctx.strokeStyle = ink(); ctx.lineWidth = 1.5; ctx.stroke(); }
  }

  // the cut: along one coupon, the fitted monthly rate against loan age, over the pools near that coupon pooled by age
  function drawCut() {
    const d = S.data; if (!d) return;
    const p = canvas("gn-cvcut"), { ctx } = p, [x0, x1] = bounds(), c = S.coupon;
    $("gn-cutcap").textContent = `coupon ${c.toFixed(3)}%, pools within ${BAND}`;
    const nb = EDGES.length - 1, bn = new Float64Array(nb), bk = new Float64Array(nb);
    for (let i = 0; i < d.N; ++i) if (Math.abs(d.wac[i] - c) <= BAND) { const j = binOf(d.age[i]); bn[j] += d.n[i]; bk[j] += d.k[i]; }
    const curve = [];
    if (S.fit) for (let a = x0; a <= x1; a += (x1 - x0) / 240) curve.push([a, 100 * rateAt(a, c)]);
    let top = 0.5;
    for (const [, r] of curve) if (r > top) top = r;
    for (let j = 0; j < nb; ++j) if (bn[j] >= 1000 && (100 * bk[j]) / bn[j] > top) top = (100 * bk[j]) / bn[j];
    const step = top > 6 ? 2 : top > 3 ? 1 : 0.5; top = Math.ceil((top * 1.08) / step) * step;
    const X = (a) => M.l + ((a - x0) / (x1 - x0)) * p.pw, Y = (r) => M.t + (1 - r / top) * p.ph;
    ctx.font = "11px ui-sans-serif, system-ui, sans-serif"; ctx.fillStyle = faint(); ctx.strokeStyle = rule(); ctx.lineWidth = 1;
    ctx.textAlign = "right"; ctx.textBaseline = "middle";
    for (const r of ticks(0, top, step)) { ctx.fillText(r + "%", M.l - 6, Y(r)); ctx.beginPath(); ctx.moveTo(M.l, Y(r) + 0.5); ctx.lineTo(M.l + p.pw, Y(r) + 0.5); ctx.stroke(); }
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    for (const a of ticks(x0, x1, 60)) ctx.fillText(a, X(a), M.t + p.ph + 5);
    ctx.fillText("loan age, months", M.l + p.pw / 2, M.t + p.ph + 19);
    ctx.save(); ctx.translate(11, M.t + p.ph / 2); ctx.rotate(-Math.PI / 2); ctx.fillText("terminations a month", 0, -6); ctx.restore();
    ctx.save(); ctx.beginPath(); ctx.rect(M.l, M.t - 6, p.pw, p.ph + 6); ctx.clip();
    // the pools' own rates: one dot per span of age, its area the loans behind it
    let most = 0; for (let j = 0; j < nb; ++j) if (bn[j] > most) most = bn[j];
    ctx.fillStyle = isDark() ? "rgba(236,236,239,0.5)" : "rgba(27,27,31,0.42)";
    for (let j = 0; j < nb; ++j) {
      if (!bn[j]) continue;
      ctx.beginPath(); ctx.arc(X((EDGES[j] + EDGES[j + 1]) / 2), Y(Math.min(top * 1.2, (100 * bk[j]) / bn[j])), Math.max(2, 9 * Math.sqrt(bn[j] / most)), 0, 2 * Math.PI); ctx.fill();
    }
    if (curve.length) {
      ctx.strokeStyle = isDark() ? "#8fb0ff" : "#1f3fd0"; ctx.lineWidth = 2.2; ctx.lineJoin = "round"; ctx.beginPath();
      curve.forEach(([a, r], i) => (i ? ctx.lineTo(X(a), Y(r)) : ctx.moveTo(X(a), Y(r)))); ctx.stroke();
    }
    if (S.hover && S.fit) { const r = 100 * rateAt(S.hover.age, c); ctx.beginPath(); ctx.arc(X(S.hover.age), Y(r), 4.5, 0, 2 * Math.PI); ctx.strokeStyle = ink(); ctx.lineWidth = 1.5; ctx.stroke(); }
    ctx.restore();
  }

  function drawScale() {
    const d = S.data; if (!d) return;
    const base = Math.log(d.rate / (1 - d.rate)), stops = [];
    for (let q = 0; q <= 8; ++q) stops.push(css(rgb(((q / 8) * 2 - 1) * VMAX)));
    const pct = (t) => (100 * expit(t)).toFixed(1) + "%";
    $("gn-scale").innerHTML = `${pct(base - VMAX)} <i style="background:linear-gradient(90deg,${stops.join(",")})"></i> ${pct(base + VMAX)} of loans terminating in the month`;
  }
  function drawAll() { if (!S.data || root.offsetParent === null) return; drawPools(); drawFit(); drawCut(); drawScale(); readout(); }

  // ------------------------------------------------------------------ what the fit found
  const dev = (k, n, p) => 2 * ((k > 0 ? k * Math.log(k / (n * p)) : 0) + (n - k > 0 ? (n - k) * Math.log((n - k) / (n * (1 - p))) : 0));
  function report() {
    const f = S.fit, d = S.data, admitted = f.verts.filter((v) => v.admitted).length, cells = f.tri.length / f.stride;
    const rounds = f.rounds.filter((r) => r.admitted > 0).length;
    // the pools never fitted, scored by the surface and by one flat rate (the month's overall rate), pool by pool
    let held = "–", heldNote = "";
    const h = f.heldout && f.heldout.rows;
    if (h) {
      let a = 0, b = 0; const N = h.n.length;
      for (let i = 0; i < N; ++i) { a += dev(h.k[i], h.n[i], h.p[i]); b += dev(h.k[i], h.n[i], d.rate); }
      held = (a / N).toFixed(3);
      heldNote = `per pool, on the ${N.toLocaleString()} pools never fitted; one flat rate scores ${(b / N).toFixed(3)}`;
    }
    const cards = [
      ["Pools", d.N.toLocaleString(), `${d.loans.toLocaleString()} loans, ${d.terms.toLocaleString()} terminations: ${(100 * d.rate).toFixed(2)}% in the month`],
      ["Vertices admitted", admitted, `in ${rounds} round${rounds === 1 ? "" : "s"}; ${cells} ${f.engine === "rect" ? "rectangles" : "triangles"}`],
      ["Pool effect, sd", Math.sqrt(f.poolVariance).toFixed(2), "what age and coupon leave unexplained, pool by pool, on the log-odds"],
      ["Held-out deviance", held, heldNote],
    ];
    $("gn-cards").innerHTML = cards.map(([k, v, t]) => `<div class="card"><div class="k">${k}</div><div class="v">${v}</div><div class="d">${t}</div></div>`).join("");
    root.classList.remove("stale");
    status("ok", "Fitted", `Admitted ${admitted} vertices, then stopped. ${Math.round(f.ms).toLocaleString()} ms in this browser.`);
  }
  function readout() {
    const el = $("gn-read");
    const bar = $("gn-scale").querySelector("i");   // the colour scale carries a tick at the height under the cursor
    if (!S.hover || !S.fit) { el.textContent = ""; if (bar) bar.style.removeProperty("--at"); return; }
    const r = rateAt(S.hover.age, S.hover.wac);
    if (bar && r === r) { const base = Math.log(S.data.rate / (1 - S.data.rate)); bar.style.setProperty("--at", (100 * Math.max(0, Math.min(1, ((Math.log(r / (1 - r)) - base) / VMAX) * 0.5 + 0.5))).toFixed(1) + "%"); }
    el.textContent = `Age ${Math.round(S.hover.age)} months, coupon ${S.hover.wac.toFixed(2)}%: ${(100 * r).toFixed(2)}% of loans terminate in the month, ${(100 * (1 - Math.pow(1 - r, 12))).toFixed(1)}% in a year at that pace.`;
  }

  // ------------------------------------------------------------------ controls
  const qLabel = () => { $("gn-qval").textContent = (+$("gn-q").value).toFixed(2); };
  for (const id of ["gn-month", "gn-pools", "gn-engine"]) $(id).addEventListener("change", run);
  $("gn-q").addEventListener("input", qLabel);
  $("gn-q").addEventListener("change", run);
  for (const id of ["gn-edges", "gn-admit"]) $(id).addEventListener("change", drawFit);
  $("gn-depth").addEventListener("input", () => { drawFit(); readout(); });
  function at(ev) {
    const fp = S.fitPlot; if (!fp) return null;
    const r = fp.p.c.getBoundingClientRect(), [x0, x1, y0, y1] = fp.b;
    const u = (ev.clientX - r.left - M.l) / fp.p.pw, v = 1 - (ev.clientY - r.top - M.t) / fp.p.ph;
    return u < 0 || u > 1 || v < 0 || v > 1 ? null : { age: x0 + u * (x1 - x0), wac: y0 + v * (y1 - y0) };
  }
  const cvfit = $("gn-cvfit");
  let dragging = false;
  const setCut = (h) => { if (h) { S.coupon = Math.round(h.wac * 8) / 8; } };
  cvfit.addEventListener("pointermove", (ev) => { S.hover = at(ev); if (dragging) { setCut(S.hover); drawPools(); } drawFit(); drawCut(); readout(); });
  cvfit.addEventListener("pointerleave", () => { S.hover = null; dragging = false; drawFit(); drawCut(); readout(); });
  cvfit.addEventListener("pointerdown", (ev) => { dragging = true; setCut(at(ev)); drawPools(); drawFit(); drawCut(); });
  // a click on an admitted vertex asks for its star (the screen one level down); the page listens for "gn-vertex".
  // A vertex's address is its dyadic one, as the star inspector writes it: L4 (1, 2) is the point (1/16, 2/16).
  function addressOf(v) { for (let L = 0; L <= 14; ++L) { const a = v.x * 2 ** L, b = v.y * 2 ** L; if (Math.abs(a - Math.round(a)) < 1e-6 && Math.abs(b - Math.round(b)) < 1e-6) return `L${L} (${Math.round(a)}, ${Math.round(b)})`; } return null; }
  cvfit.addEventListener("click", (ev) => {
    const fp = S.fitPlot; if (!fp || !S.fit) return;
    const r = fp.p.c.getBoundingClientRect(), px = ev.clientX - r.left, py = ev.clientY - r.top; let best = null, bd = 13 * 13;
    for (const v of S.fit.verts) { if (!v.admitted) continue; const d = (M.l + v.x * fp.p.pw - px) ** 2 + (M.t + (1 - v.y) * fp.p.ph - py) ** 2; if (d < bd) { bd = d; best = v; } }
    if (best) window.dispatchEvent(new CustomEvent("gn-vertex", { detail: addressOf(best) }));
  });
  window.addEventListener("pointerup", () => { dragging = false; });
  let rz = 0; window.addEventListener("resize", () => { clearTimeout(rz); rz = setTimeout(drawAll, 120); });
  new MutationObserver(drawAll).observe(document.documentElement, { attributes: true, attributeFilter: ["class"] });

  // ------------------------------------------------------------------ which month of which agency
  let setsReady = null, files = [];   // files: ginnie/index.json, written by build.sh: the data files and saved fits that exist
  async function sets() {
    try { files = await (await fetch(root.dataset.base.replace(/\/?$/, "/") + "index.json")).json(); } catch (e) {}
    const have = SETS.filter((st) => files.includes(st[0] + "_" + st[1]));
    const pick = (st) => {
      $("gn-pools").value = st[0]; $("gn-month").value = String(st[1]); $("gn-title").textContent = st[2];
      $("gn-pools").dispatchEvent(new Event("change-set"));
      for (const b of $("gn-sets").children) b.classList.toggle("on", b.dataset.k === st[0] + st[1]);
    };
    $("gn-sets").innerHTML = have.map((st) => `<button data-k="${st[0]}${st[1]}" title="${st[2]}, ${st[3]}">${st[4]} <i>${st[3]}</i></button>`).join("");
    [...$("gn-sets").children].forEach((b, i) => b.addEventListener("click", () => { pick(have[i]); run(); }));
    if (have.length) pick(have[0]);
    $("gn-where").addEventListener("click", () => { $("gn-where").classList.toggle("on"); drawFit(); });
    // the same pools through the rectangular engine: cells cut across one axis at a time, where the triangles bisect
    $("gn-rect").addEventListener("click", () => { const on = $("gn-rect").classList.toggle("on"); $("gn-engine").value = on ? "rect" : "tri"; run(); });
  }

  qLabel();
  // the first fit starts when the tab is first on screen (a hidden canvas has no size to draw into)
  function shown() { if (!S.shownOnce) { S.shownOnce = true; (setsReady = sets()).then(run); } else drawAll(); }
  return { shown };
})();
