// The Decision Mesh demo: noisy data from a chosen surface, a decision mesh (decision-mesh.js) and a best-first
// regression tree grown to the same number of pieces, drawn side by side with the truth, and their error curves.
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const root = document.querySelector(".explorer");
  const css = (name) => getComputedStyle(root).getPropertyValue(name).trim();
  const isDark = () => document.documentElement.classList.contains("dark");

  // ------------------------------------------------------------------ data
  const SURFACES = {
    cliff: { name: "Diagonal cliff", f: (x, y) => 2 * Math.tanh(1.6 * (x - y + 0.6 * Math.sin(0.8 * (x + y)))) },
    bump: { name: "Two hills", f: (x, y) => 2.6 * Math.exp(-((x - 1.4) ** 2 + (y - 1) ** 2) / 1.8) - 2 * Math.exp(-((x + 1.6) ** 2 + (y + 1.2) ** 2) / 1.2) },
    saddle: { name: "Saddle", f: (x, y) => (x * x - y * y) / 6 },
    disk: { name: "Disk (a jump)", f: (x, y) => (x * x + y * y < 5 ? 1.5 : -1) },
    ripples: { name: "Ripples (the notebook's)", f: (x, y) => 2 * Math.cos(5 * x) * Math.cos(2 * y) },
    waves: { name: "Slow waves", f: (x, y) => 1.6 * Math.sin(1.1 * x) * Math.cos(0.8 * y) + 0.3 * x },
  };
  const LO = -4, HI = 4;

  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function gauss(rng) { let u = 0; while (u === 0) u = rng(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng()); }

  // ------------------------------------------------------------------ best-first regression tree (CART)
  const MIN_LEAF = 5;
  class Tree {
    constructor(X, Y) {
      this.X = X; this.Y = Y;
      const n = Y.length, ids = new Int32Array(n);
      for (let i = 0; i < n; ++i) ids[i] = i;
      this.leaves = [this.node(ids, LO, HI, LO, HI)];
      this.cuts = [];
    }
    node(ids, x0, x1, y0, y1) {
      let s = 0, q = 0;
      for (const i of ids) { s += this.Y[i]; q += this.Y[i] * this.Y[i]; }
      const n = ids.length, nd = { ids, x0, x1, y0, y1, mean: n ? s / n : 0, sse: n ? q - (s * s) / n : 0, split: null };
      nd.split = this.bestSplit(nd);
      return nd;
    }
    bestSplit(nd) {
      const ids = nd.ids, n = ids.length;
      if (n < 2 * MIN_LEAF) return null;
      let best = null;
      for (let f = 0; f < 2; ++f) {
        const order = Array.from(ids).sort((a, b) => this.X[2 * a + f] - this.X[2 * b + f]);
        let sl = 0, ql = 0, st = 0, qt = 0;
        for (const i of order) { st += this.Y[i]; qt += this.Y[i] * this.Y[i]; }
        for (let k = 0; k < n - MIN_LEAF; ++k) {
          const y = this.Y[order[k]];
          sl += y; ql += y * y;
          const nl = k + 1;
          if (nl < MIN_LEAF) continue;
          const a = this.X[2 * order[k] + f], b = this.X[2 * order[k + 1] + f];
          if (a === b) continue;
          const nr = n - nl, sr = st - sl, qr = qt - ql;
          const sse = ql - (sl * sl) / nl + qr - (sr * sr) / nr;
          const gain = nd.sse - sse;
          if (!best || gain > best.gain) best = { f, thr: (a + b) / 2, gain };
        }
      }
      return best && best.gain > 1e-12 ? best : null;
    }
    // split the leaf with the largest gain; false when no leaf can split
    grow() {
      let bi = -1, bg = 0;
      for (let i = 0; i < this.leaves.length; ++i) { const s = this.leaves[i].split; if (s && s.gain > bg) { bg = s.gain; bi = i; } }
      if (bi < 0) return false;
      const nd = this.leaves[bi], { f, thr } = nd.split;
      const L = [], R = [];
      for (const i of nd.ids) (this.X[2 * i + f] <= thr ? L : R).push(i);
      const a = f === 0 ? this.node(Int32Array.from(L), nd.x0, thr, nd.y0, nd.y1) : this.node(Int32Array.from(L), nd.x0, nd.x1, nd.y0, thr);
      const b = f === 0 ? this.node(Int32Array.from(R), thr, nd.x1, nd.y0, nd.y1) : this.node(Int32Array.from(R), nd.x0, nd.x1, thr, nd.y1);
      this.leaves.splice(bi, 1, a, b);
      this.cuts.push(f === 0 ? [thr, nd.y0, thr, nd.y1] : [nd.x0, thr, nd.x1, thr]);
      return true;
    }
  }

  // ------------------------------------------------------------------ rasters
  const D = 300;                                  // value grid (display and error), D x D over the square
  const px = (x) => ((x - LO) / (HI - LO)) * D;   // data to grid coordinates (y up)
  const py = (y) => ((HI - y) / (HI - LO)) * D;
  const cellX = (j) => LO + ((j + 0.5) / D) * (HI - LO);
  const cellY = (i) => HI - ((i + 0.5) / D) * (HI - LO);

  function truthGrid(f) {
    const g = new Float32Array(D * D);
    for (let i = 0; i < D; ++i) for (let j = 0; j < D; ++j) g[i * D + j] = f(cellX(j), cellY(i));
    return g;
  }
  function meshGrid(mesh, g) {
    g.fill(NaN);
    for (const f of mesh.activeFaces) {
      const [a, b, c] = f.vertices;
      const ax = px(a.x), ay = py(a.y), bx = px(b.x), by = py(b.y), cx = px(c.x), cy = py(c.y);
      const den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy);
      if (Math.abs(den) < 1e-12) continue;
      const i0 = Math.max(0, Math.floor(Math.min(ay, by, cy))), i1 = Math.min(D - 1, Math.ceil(Math.max(ay, by, cy)));
      const j0 = Math.max(0, Math.floor(Math.min(ax, bx, cx))), j1 = Math.min(D - 1, Math.ceil(Math.max(ax, bx, cx)));
      for (let i = i0; i <= i1; ++i) {
        const yy = i + 0.5;
        for (let j = j0; j <= j1; ++j) {
          const xx = j + 0.5;
          const l0 = ((by - cy) * (xx - cx) + (cx - bx) * (yy - cy)) / den;
          const l1 = ((cy - ay) * (xx - cx) + (ax - cx) * (yy - cy)) / den;
          const l2 = 1 - l0 - l1;
          if (l0 < -1e-9 || l1 < -1e-9 || l2 < -1e-9) continue;
          g[i * D + j] = l0 * a.height + l1 * b.height + l2 * c.height;
        }
      }
    }
    return g;
  }
  function treeGrid(tree, g) {
    for (const L of tree.leaves) {
      const j0 = Math.max(0, Math.round(px(L.x0))), j1 = Math.min(D, Math.round(px(L.x1)));
      const i0 = Math.max(0, Math.round(py(L.y1))), i1 = Math.min(D, Math.round(py(L.y0)));
      for (let i = i0; i < i1; ++i) g.fill(L.mean, i * D + j0, i * D + j1);
    }
    return g;
  }
  function rmse(g, t) {
    let s = 0, n = 0;
    for (let k = 0; k < g.length; ++k) { const v = g[k]; if (v === v) { s += (v - t[k]) ** 2; ++n; } }
    return n ? Math.sqrt(s / n) : NaN;
  }

  // colour: a diverging ramp through the page's background tone
  function ramp() {
    const hex = (h) => [1, 3, 5].map((k) => parseInt(h.slice(k, k + 2), 16));
    const stops = isDark()
      ? [hex("#2b6cf0"), hex("#5b8ff9"), hex("#2a2b30"), hex("#f0845c"), hex("#f5c451")]
      : [hex("#1d4ed8"), hex("#6d9cf5"), hex("#f7f6ee"), hex("#ef7a55"), hex("#b91c1c")];
    const lut = new Uint8ClampedArray(256 * 3);
    for (let k = 0; k < 256; ++k) {
      const t = (k / 255) * 4, s = Math.min(3, Math.floor(t)), u = t - s;
      for (let c = 0; c < 3; ++c) lut[3 * k + c] = stops[s][c] + (stops[s + 1][c] - stops[s][c]) * u;
    }
    return lut;
  }
  let LUT = ramp();

  const off = document.createElement("canvas"); off.width = off.height = D;
  const offctx = off.getContext("2d");
  const img = offctx.createImageData(D, D);
  function paint(canvas, g, vmax, smooth, overlay) {
    const d = img.data;
    const empty = isDark() ? [31, 32, 34] : [255, 255, 240];
    for (let k = 0; k < D * D; ++k) {
      const v = g[k];
      if (v !== v) { d[4 * k] = empty[0]; d[4 * k + 1] = empty[1]; d[4 * k + 2] = empty[2]; d[4 * k + 3] = 255; continue; }
      const q = Math.max(0, Math.min(255, Math.round(((v / vmax) * 0.5 + 0.5) * 255)));
      d[4 * k] = LUT[3 * q]; d[4 * k + 1] = LUT[3 * q + 1]; d[4 * k + 2] = LUT[3 * q + 2]; d[4 * k + 3] = 255;
    }
    offctx.putImageData(img, 0, 0);
    const ctx = canvas.getContext("2d"), W = canvas.width;
    ctx.imageSmoothingEnabled = smooth;
    ctx.drawImage(off, 0, 0, W, W);
    if (overlay) overlay(ctx, W / D);
  }

  // ------------------------------------------------------------------ state
  const S = { running: false, done: false, history: [] };
  window.meshDemo = S;
  let mesh, tree, truth, gMesh = new Float32Array(D * D), gTree = new Float32Array(D * D), vmax = 1;
  let X, Y, sigma;

  function readHash() {
    const h = new URLSearchParams(location.hash.slice(1));
    const s = h.get("surface");
    $("surface").value = s && SURFACES[s] ? s : "cliff";
  }
  function writeHash() { history.replaceState(null, "", "#surface=" + $("surface").value); }

  let seed = 7;
  function reset(newSeed) {
    if (newSeed) seed = (seed * 1103515245 + 12345) >>> 0;
    const rng = mulberry32(seed), n = +$("npts").value, f = SURFACES[$("surface").value].f;
    sigma = +$("noise").value;
    X = new Float64Array(2 * n); Y = new Float64Array(n);
    for (let i = 0; i < n; ++i) {
      const x = LO + (HI - LO) * rng(), y = LO + (HI - LO) * rng();
      X[2 * i] = x; X[2 * i + 1] = y; Y[i] = f(x, y) + sigma * gauss(rng);
    }
    DM.reset();
    mesh = new DM.DecisionMesh(X, Y, { maxAspectRatio: +$("aspect").value, minPoints: +$("minpts").value, refresh: $("refresh").checked, rng: mulberry32(seed + 1) });
    tree = new Tree(X, Y);
    truth = truthGrid(f);
    vmax = 0; for (const v of truth) vmax = Math.max(vmax, Math.abs(v)); vmax = vmax || 1;
    S.history = []; S.done = false;
    syncTree(); record();
    drawTruth(); draw();
    setStatus(S.running ? "busy" : "idle", S.running ? "Growing" : "Ready", S.running ? "" : "Press play, or step once.");
  }

  function syncTree() { while (tree.leaves.length < mesh.activeFaces.size && tree.grow()); }
  function record() {
    meshGrid(mesh, gMesh); treeGrid(tree, gTree);
    S.history.push({ pieces: mesh.activeFaces.size, leaves: tree.leaves.length, m: rmse(gMesh, truth), t: rmse(gTree, truth) });
  }

  // ------------------------------------------------------------------ drawing
  function fitCanvas(c) {
    const r = c.getBoundingClientRect(), dpr = Math.min(2, window.devicePixelRatio || 1);
    const w = Math.max(1, Math.round(r.width * dpr)), h = Math.max(1, Math.round(r.height * dpr));
    if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
  }
  function lineColor() { return isDark() ? "rgba(255,255,255,0.32)" : "rgba(20,20,30,0.4)"; }
  function dots(ctx, s) {
    if (!$("showpts").checked) return;
    const n = Y.length, step = Math.max(1, Math.floor(n / 3000));
    ctx.fillStyle = isDark() ? "rgba(255,255,255,0.5)" : "rgba(0,0,0,0.45)";
    for (let i = 0; i < n; i += step) ctx.fillRect(px(X[2 * i]) * s - 0.9, py(X[2 * i + 1]) * s - 0.9, 1.8, 1.8);
  }
  function drawTruth() {
    const c = $("cvtruth"); fitCanvas(c);
    paint(c, truth, vmax, true, (ctx, s) => dots(ctx, s));
  }
  function draw() {
    const cm = $("cvmesh"), ct = $("cvtree"); fitCanvas(cm); fitCanvas(ct);
    const edges = $("showedges").checked;
    paint(cm, gMesh, vmax, true, (ctx, s) => {
      if (edges) {
        ctx.strokeStyle = lineColor(); ctx.lineWidth = Math.max(0.5, s * 0.5); ctx.beginPath();
        for (const e of mesh.activeEdges) { ctx.moveTo(px(e.v0.x) * s, py(e.v0.y) * s); ctx.lineTo(px(e.v1.x) * s, py(e.v1.y) * s); }
        ctx.stroke();
      }
      dots(ctx, s);
    });
    paint(ct, gTree, vmax, false, (ctx, s) => {
      if (edges) {
        ctx.strokeStyle = lineColor(); ctx.lineWidth = Math.max(0.5, s * 0.5); ctx.beginPath();
        for (const [x0, y0, x1, y1] of tree.cuts) { ctx.moveTo(px(x0) * s, py(y0) * s); ctx.lineTo(px(x1) * s, py(y1) * s); }
        ctx.stroke();
      }
      dots(ctx, s);
    });
    const h = S.history[S.history.length - 1];
    $("capmesh").textContent = `${h.pieces.toLocaleString()} triangles · RMSE ${h.m.toFixed(3)}`;
    $("captree").textContent = `${h.leaves.toLocaleString()} boxes · RMSE ${h.t.toFixed(3)}`;
    drawCurve();
    drawScale();
  }
  function drawScale() {
    const stops = []; for (let k = 0; k <= 8; ++k) { const q = Math.round((k / 8) * 255); stops.push(`rgb(${LUT[3 * q]},${LUT[3 * q + 1]},${LUT[3 * q + 2]})`); }
    $("scale").innerHTML = `${(-vmax).toFixed(1)} <i style="background:linear-gradient(90deg,${stops.join(",")})"></i> ${vmax.toFixed(1)}`;
  }
  function drawCurve() {
    const c = $("cvcurve"); fitCanvas(c);
    const ctx = c.getContext("2d"), W = c.width, H = c.height, dpr = W / c.getBoundingClientRect().width || 1;
    ctx.clearRect(0, 0, W, H);
    const L = 44 * dpr, R = 12 * dpr, T = 12 * dpr, B = 26 * dpr;
    const hist = S.history, maxP = Math.max(20, ...hist.map((h) => h.pieces));
    let maxY = sigma * 1.05; for (const h of hist) maxY = Math.max(maxY, h.m, h.t); maxY = maxY * 1.05 || 1;
    const X_ = (p) => L + ((W - L - R) * Math.log(p)) / Math.log(maxP);
    const Y_ = (v) => T + (H - T - B) * (1 - v / maxY);
    ctx.font = `${11 * dpr}px ui-monospace, Menlo, monospace`; ctx.fillStyle = css("--faint"); ctx.strokeStyle = css("--grid"); ctx.lineWidth = dpr;
    for (let k = 0; k <= 4; ++k) {
      const v = (maxY * k) / 4, y = Y_(v);
      ctx.beginPath(); ctx.moveTo(L, y); ctx.lineTo(W - R, y); ctx.stroke();
      ctx.textAlign = "right"; ctx.fillText(v.toFixed(2), L - 6 * dpr, y + 4 * dpr);
    }
    ctx.textAlign = "center";
    for (const p of [2, 10, 100, 1000, 10000]) if (p <= maxP) ctx.fillText(p.toLocaleString(), X_(p), H - 8 * dpr);
    ctx.textAlign = "left"; ctx.fillText("pieces (log)", L + 4 * dpr, H - 8 * dpr - 12 * dpr);
    if (sigma > 0) {
      ctx.setLineDash([5 * dpr, 4 * dpr]); ctx.strokeStyle = css("--faint");
      ctx.beginPath(); ctx.moveTo(L, Y_(sigma)); ctx.lineTo(W - R, Y_(sigma)); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillText("noise σ", W - R - 60 * dpr, Y_(sigma) - 5 * dpr);
    }
    const line = (key, col, pk) => {
      ctx.strokeStyle = col; ctx.lineWidth = 2 * dpr; ctx.beginPath();
      hist.forEach((h, i) => { const x = X_(Math.max(1, h[pk])), y = Y_(h[key]); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
      ctx.stroke();
    };
    line("t", css("--c4"), "leaves");
    line("m", css("--accent"), "pieces");
    ctx.fillStyle = css("--accent"); ctx.fillText("mesh", L + 8 * dpr, T + 12 * dpr);
    ctx.fillStyle = css("--c4"); ctx.fillText("tree", L + 56 * dpr, T + 12 * dpr);
  }

  // ------------------------------------------------------------------ loop
  function setStatus(kind, chip, text) {
    const c = $("chip"); c.className = "chip " + kind; c.textContent = chip;
    if (text !== undefined) $("statustext").textContent = text;
  }
  const MAX_PIECES = () => Math.min(3000, Math.floor(Y.length / 10));
  let stepMs = 0;
  function advance(k, budget) {
    const t0 = performance.now();
    let did = 0;
    const eps = +$("eps").value;
    for (let i = 0; i < k; ++i) {
      if (mesh.activeFaces.size >= MAX_PIECES()) { S.done = true; break; }
      const r = mesh.step(eps);
      ++did;
      if (r === "none") { S.done = true; break; }
      if (performance.now() - t0 > budget) break;
    }
    if (did) stepMs = 0.8 * stepMs + 0.2 * ((performance.now() - t0) / did);
    syncTree(); record(); draw();
    const h = S.history[S.history.length - 1];
    const lead = h.m < h.t ? `the mesh is ${(100 * (1 - h.m / h.t)).toFixed(0)}% closer to the truth` : `the tree is ${(100 * (1 - h.t / h.m)).toFixed(0)}% closer to the truth`;
    const txt = `Step ${mesh.steps.toLocaleString()}: ${h.pieces.toLocaleString()} triangles on ${mesh.activeVertices().toLocaleString()} vertices; at the same number of pieces ${lead}.`;
    $("speedfact").textContent = `about ${stepMs < 1 ? stepMs.toFixed(2) : stepMs.toFixed(1)} ms a step here`;
    if (S.done) { S.running = false; $("playbtn").textContent = "Play"; setStatus("ok", "Done", txt + " Stopped at ten points per triangle, on average."); }
    else setStatus(S.running ? "busy" : "idle", S.running ? "Growing" : "Paused", txt);
  }
  function loop() {
    if (!S.running) return;
    advance(+$("speed").value, 40);
    if (S.running) requestAnimationFrame(loop);
  }

  // ------------------------------------------------------------------ wiring
  $("surface").innerHTML = Object.entries(SURFACES).map(([k, s]) => `<option value="${k}">${s.name}</option>`).join("");
  readHash();
  const show = () => {
    $("noiseval").textContent = (+$("noise").value).toFixed(2);
    $("epsval").textContent = (+$("eps").value).toFixed(2);
    $("aspectval").textContent = (+$("aspect").value).toFixed(1);
    $("speedval").textContent = $("speed").value + ($("speed").value === "1" ? " step" : " steps");
    $("minptsval").textContent = $("minpts").value;
  };
  show();
  for (const id of ["noise", "eps", "aspect", "speed", "minpts"]) $(id).addEventListener("input", show);
  for (const id of ["surface", "npts"]) $(id).addEventListener("change", () => { writeHash(); reset(false); });
  for (const id of ["noise", "aspect", "minpts"]) $(id).addEventListener("change", () => reset(false));
  $("resetbtn").onclick = () => reset(true);
  $("stepbtn").onclick = () => { S.running = false; $("playbtn").textContent = "Play"; if (S.done) reset(false); advance(1, 1e9); };
  $("playbtn").onclick = () => {
    if (S.running) { S.running = false; $("playbtn").textContent = "Play"; advance(0, 0); return; }
    if (S.done) reset(false);
    S.running = true; $("playbtn").textContent = "Pause"; setStatus("busy", "Growing"); requestAnimationFrame(loop);
  };
  $("refresh").onchange = () => reset(false);
  $("showedges").onchange = draw; $("showpts").onchange = () => { drawTruth(); draw(); };
  window.addEventListener("hashchange", () => { readHash(); reset(false); });
  let rt; window.addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(() => { drawTruth(); draw(); }, 100); });
  const retheme = () => setTimeout(() => { LUT = ramp(); drawTruth(); draw(); }, 30);
  document.body.addEventListener("set-theme", retheme);
  new MutationObserver(retheme).observe(document.documentElement, { attributes: true, attributeFilter: ["class"] });

  reset(false);
  writeHash();
})();
