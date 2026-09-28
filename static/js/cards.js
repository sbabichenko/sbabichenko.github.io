// The two project cards on the home page, made touchable. Nothing heavy runs until someone touches a card.
//
// Decision Mesh: the picture at rest is the card image (two faces: the odds drawn on the left, the fitted mesh on
// the right). Scribbling on it raises the odds under the pointer, as "your drawing" does on /decision-mesh, and
// the left half is repainted from the drawing. When the stroke ends, a few thousand coins are flipped at the drawn
// odds and shown on the right half; the triangular estimator (static/mesh/fit-worker.js and trimesh.wasm, loaded on
// the first touch or when a mouse comes over the picture) fits them, and its surface and mesh replace the coins.
// The drawing starts from the same face as the image, with the image's color ramp, mesh lines and crop, so a
// scribble makes your own version of the card. "clear" puts the image itself back.
// On a touch screen a quick touch that moves is a scroll; drawing starts once a finger has rested on the picture for
// a fifth of a second (a tap drops a single dab of odds), so the card never takes the page's scrolling.
//
// Noise-State Calculus: the two-player tracking game with opposite targets, drawn in the pencil style of
// /dissertation/wedge. Dragging across the card sets how clearly both players see each other (precision p from
// 0.01 to 1000, log scale); the curves were solved ahead of time by the explorer's solver
// (tools/cards/make_card_curves.js writes static/js/card-noisestate.json) and are interpolated in log p here.
(function () {
  "use strict";
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const isDark = () => document.documentElement.classList.contains("dark");
  const clamp = (x, a = 0, b = 1) => Math.max(a, Math.min(b, x));
  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  const themeWatchers = [];
  new MutationObserver(() => themeWatchers.forEach((f) => f())).observe(document.documentElement, { attributes: true, attributeFilter: ["class"] });

  // ================================================================== Decision Mesh
  (function meshCard() {
    const shot = document.getElementById("card-mesh");
    if (!shot) return;
    const cv = shot.querySelector("canvas"), ctx = cv.getContext("2d");
    const clearBtn = shot.querySelector(".clear"), status = shot.querySelector(".status");
    const imgs = [...shot.querySelectorAll("img")];

    // the image's composition, in its own 960 x 480 pixels: two squares of side 532, the drawing at (-53, -52)
    // and the fit at (480, -52), each cropped by the frame (measured from the image itself)
    const CW = 960, CH = 480, SQ = 532, LX = -53, RX = 480, TOP = -52;
    cv.width = CW; cv.height = CH;
    const BASE = -1, VMAX = 2.2, G = 64, D = 160;
    const SITES = 3000, FLIPS = 20, SD = 0.25;
    const drawing = new Float32Array(G * G);
    function brush(x, y, amount, radius) {
      for (let j = 0; j < G; ++j) for (let i = 0; i < G; ++i) {
        const dx = i / (G - 1) - x, dy = j / (G - 1) - y, w = Math.exp(-(dx * dx + dy * dy) / (2 * radius * radius));
        if (w < 1e-3) continue;
        drawing[j * G + i] = Math.max(BASE - 2.5, Math.min(BASE + 2.2, drawing[j * G + i] + amount * w));
      }
    }
    function smile() {       // the face in the image
      drawing.fill(BASE);
      brush(0.35, 0.68, 2.0, 0.06); brush(0.65, 0.68, 2.0, 0.06);
      for (let t = 0; t <= 1; t += 0.05) { const a = Math.PI * (1.15 + 0.7 * t); brush(0.5 + 0.26 * Math.cos(a), 0.5 + 0.26 * Math.sin(a), 0.55, 0.035); }
    }
    smile();
    function sample(x, y, g) {
      const gx = Math.min(G - 1.001, Math.max(0, x * (G - 1))), gy = Math.min(G - 1.001, Math.max(0, y * (G - 1)));
      const i = Math.floor(gx), j = Math.floor(gy), u = gx - i, v = gy - j;
      return (g[j * G + i] * (1 - u) + g[j * G + i + 1] * u) * (1 - v) + (g[(j + 1) * G + i] * (1 - u) + g[(j + 1) * G + i + 1] * u) * v;
    }

    // the image's color ramp: high odds warm, low odds blue, the background coin at the paper
    let LUT;
    function ramp() {
      const hex = (h) => [1, 3, 5].map((q) => parseInt(h.slice(q, q + 2), 16));
      const s = (isDark() ? ["#2b6cf0", "#5b8ff9", "#2a2b30", "#f0845c", "#f5c451"] : ["#1d4ed8", "#6d9cf5", "#f7f6ee", "#ef7a55", "#b91c1c"]).map(hex);
      LUT = new Uint8ClampedArray(256 * 3);
      for (let q = 0; q < 256; ++q) {
        const t = (q / 255) * 4, a = Math.min(3, Math.floor(t)), u = t - a;
        for (let c = 0; c < 3; ++c) LUT[3 * q + c] = s[a][c] + (s[a + 1][c] - s[a][c]) * u;
      }
    }
    ramp();
    const color = (l) => { const q = 3 * Math.max(0, Math.min(255, Math.round((((l - BASE) / VMAX) * 0.5 + 0.5) * 255))); return [LUT[q], LUT[q + 1], LUT[q + 2]]; };
    const off = document.createElement("canvas"); off.width = off.height = D;
    const offctx = off.getContext("2d"), raster = offctx.createImageData(D, D);
    function paintSquare(x0, f) {                  // f(u, v) on the unit square, v up
      const d = raster.data;
      for (let r = 0; r < D; ++r) for (let c = 0; c < D; ++c) {
        const rgb = color(f((c + 0.5) / D, 1 - (r + 0.5) / D)), k = 4 * (r * D + c);
        d[k] = rgb[0]; d[k + 1] = rgb[1]; d[k + 2] = rgb[2]; d[k + 3] = 255;
      }
      offctx.putImageData(raster, 0, 0);
      ctx.save(); ctx.beginPath(); ctx.rect(x0 < RX ? 0 : RX, 0, x0 < RX ? RX : CW - RX, CH); ctx.clip();
      ctx.imageSmoothingEnabled = true; ctx.drawImage(off, x0, TOP, SQ, SQ); ctx.restore();
    }
    const paper = () => (isDark() ? "#2a2b30" : "#f7f6ee");

    // ---------------------------------------------------------------- state
    const S = { live: false, data: null, fit: null, fitG: null, id: 0, worker: null, ready: false, queued: null, shownAt: 0, timer: 0 };
    const currentImg = () => imgs.find((im) => im.offsetParent !== null) || imgs[0];
    function drawLeft() { paintSquare(LX, (u, v) => sample(u, v, drawing)); }
    function drawRight() {
      if (S.fitG) { drawFit(); return; }
      if (S.data) { drawCoins(); return; }
      const im = currentImg();       // until something has been fitted, the right half is the image's own
      if (im && im.complete && im.naturalWidth) ctx.drawImage(im, im.naturalWidth / 2, 0, im.naturalWidth / 2, im.naturalHeight, RX, 0, CW - RX, CH);
      else { ctx.fillStyle = paper(); ctx.fillRect(RX, 0, CW - RX, CH); }
    }
    function drawCoins() {
      ctx.save(); ctx.beginPath(); ctx.rect(RX, 0, CW - RX, CH); ctx.clip();
      ctx.fillStyle = paper(); ctx.fillRect(RX, 0, CW - RX, CH);
      const { x, y, n, k } = S.data, rad = (SQ / Math.sqrt(x.length)) * 0.42;
      for (let i = 0; i < x.length; ++i) {
        const rgb = color(Math.log((k[i] + 0.5) / (n[i] - k[i] + 0.5)));
        ctx.fillStyle = `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
        ctx.fillRect(RX + x[i] * SQ - rad, TOP + (1 - y[i]) * SQ - rad, 2 * rad, 2 * rad);
      }
      ctx.restore();
    }
    // the fitted surface on a grid: each triangle interpolates its corners' heights, plus the engine's baseline
    function fitGrid(fit) {
      const g = new Float32Array(D * D).fill(NaN), t = fit.tri, [x0, x1, y0, y1] = fit.bounds;
      const X = (u) => x0 + u * (x1 - x0), Y = (u) => y0 + u * (y1 - y0);
      for (let i = 0; i < t.length; i += 9) {
        const ax = X(t[i]), ay = Y(t[i + 1]), ah = t[i + 2], bx = X(t[i + 3]), by = Y(t[i + 4]), bh = t[i + 5], cx = X(t[i + 6]), cy = Y(t[i + 7]), ch = t[i + 8];
        const det = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy);
        if (Math.abs(det) < 1e-15) continue;
        const c0 = Math.max(0, Math.floor(Math.min(ax, bx, cx) * D - 1)), c1 = Math.min(D - 1, Math.ceil(Math.max(ax, bx, cx) * D));
        const r0 = Math.max(0, Math.floor((1 - Math.max(ay, by, cy)) * D - 1)), r1 = Math.min(D - 1, Math.ceil((1 - Math.min(ay, by, cy)) * D));
        for (let r = r0; r <= r1; ++r) for (let c = c0; c <= c1; ++c) {
          const px = (c + 0.5) / D, py = 1 - (r + 0.5) / D;
          const l1 = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / det, l2 = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / det, l3 = 1 - l1 - l2;
          if (l1 < -1e-9 || l2 < -1e-9 || l3 < -1e-9) continue;
          g[r * D + c] = fit.baseline + l1 * ah + l2 * bh + l3 * ch;
        }
      }
      for (let r = 0; r < D; ++r) {           // just outside the data's box: the nearest value in the row
        let last = NaN;
        for (let c = 0; c < D; ++c) { const k = r * D + c; if (g[k] === g[k]) last = g[k]; else if (last === last) g[k] = last; }
        for (let c = D - 1; c >= 0; --c) { const k = r * D + c; if (g[k] === g[k]) last = g[k]; else g[k] = last; }
      }
      return g;
    }
    function drawFit() {
      const g = S.fitG, fit = S.fit;
      paintSquare(RX, (u, v) => { const c = Math.min(D - 1, Math.floor(u * D)), r = Math.min(D - 1, Math.floor((1 - v) * D)), h = g[r * D + c]; return h === h ? h : BASE; });
      const [x0, x1, y0, y1] = fit.bounds;
      const X = (u) => RX + (x0 + u * (x1 - x0)) * SQ, Y = (u) => TOP + (1 - (y0 + u * (y1 - y0))) * SQ;
      ctx.save(); ctx.beginPath(); ctx.rect(RX, 0, CW - RX, CH); ctx.clip();
      // the mesh lines and the admitted points, as on /decision-mesh (and in the image) at this size
      ctx.strokeStyle = isDark() ? "rgba(255,255,255,0.28)" : "rgba(20,20,30,0.34)";
      ctx.lineWidth = SQ / 420;
      ctx.beginPath();
      const t = fit.tri;
      for (let i = 0; i < t.length; i += 9) { ctx.moveTo(X(t[i]), Y(t[i + 1])); ctx.lineTo(X(t[i + 3]), Y(t[i + 4])); ctx.lineTo(X(t[i + 6]), Y(t[i + 7])); ctx.closePath(); }
      ctx.stroke();
      const cols = isDark() ? ["#8fb0ff", "#f5c451", "#f07ab8", "#7fe0c0"] : ["#1f3fd0", "#d97706", "#c0267a", "#0f8a6a"];
      const rad = SQ / 110, lw = SQ / 400;
      for (const v of fit.verts) {
        if (!v.admitted) continue;
        ctx.beginPath(); ctx.arc(X(v.x), Y(v.y), rad, 0, 2 * Math.PI);
        ctx.fillStyle = cols[Math.min(cols.length - 1, Math.max(0, v.round))]; ctx.fill();
        ctx.lineWidth = lw; ctx.strokeStyle = isDark() ? "#111" : "#fff"; ctx.stroke();
      }
      ctx.restore();
    }
    function drawAll() { if (!S.live) return; ramp(); drawRight(); drawLeft(); }
    themeWatchers.push(drawAll);

    // ---------------------------------------------------------------- coins and the estimator
    function gauss(r) { let u = 0; while (u === 0) u = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r()); }
    function makeData() {
      const r = mulberry32(20260927), snap = drawing.slice();
      const x = new Float64Array(SITES), y = new Float64Array(SITES), n = new Int32Array(SITES), k = new Int32Array(SITES), rows = [];
      for (let i = 0; i < SITES; ++i) {
        x[i] = r(); y[i] = r(); n[i] = Math.max(1, Math.round(FLIPS * (0.5 + r())));
        const p = 1 / (1 + Math.exp(-(sample(x[i], y[i], snap) + SD * gauss(r))));
        let h = 0; for (let f = 0; f < n[i]; ++f) if (r() < p) ++h;
        k[i] = h; rows.push(x[i].toFixed(5) + "," + y[i].toFixed(5) + "," + n[i] + "," + h);
      }
      return { x, y, n, k, csv: rows.join("\n") + "\n" };
    }
    function say(s) { status.textContent = s; }
    function warm() {                   // start the estimator loading: on hover or the first touch
      if (S.worker) return;
      try { S.worker = new Worker(shot.dataset.worker); } catch (e) { say("this browser cannot run the estimator"); return; }
      S.worker.onmessage = (ev) => {
        const m = ev.data;
        if (m.type === "ready") { S.ready = true; if (S.queued) { const q = S.queued; S.queued = null; S.worker.postMessage(q); } return; }
        if (m.id !== S.id) return;              // a stroke or a clear came since
        if (m.type === "error") { say("the estimator stopped: " + (m.message || "error")); return; }
        const show = () => {
          if (m.id !== S.id) return;
          S.fit = m; S.fitG = fitGrid(m);
          drawRight();
          const admitted = m.verts.filter((v) => v.admitted).length;
          say(`${SITES.toLocaleString()} coins, ${admitted} point${admitted === 1 ? "" : "s"} admitted` + (shot.clientWidth < 420 ? "" : `, ${Math.round(m.ms)} ms`));
          if (window.siteTally) window.siteTally("fit", 1, "a decision mesh on the home page's card");
        };
        // the coins stay up for a moment, so the step from noisy flips to the fitted odds can be seen
        clearTimeout(S.timer); S.timer = setTimeout(show, Math.max(0, S.shownAt + 450 - performance.now()));
      };
      S.worker.onerror = () => say("the estimator could not be loaded");
    }
    function fit() {
      S.data = makeData(); S.fit = null; S.fitG = null; S.id += 1;
      drawRight(); S.shownAt = performance.now();
      const msg = { id: S.id, engine: "tri", design: S.data.csv, seed: 7, q: 0.1 };
      warm();
      if (S.ready) S.worker.postMessage(msg); else S.queued = msg;
      say(S.ready ? `fitting ${SITES.toLocaleString()} coins…` : "loading the estimator…");
    }
    function goLive() {
      if (S.live) return;
      S.live = true; shot.classList.add("live");
      drawAll();
    }
    function clear() {
      S.live = false; S.id += 1; S.queued = null; S.data = null; S.fit = null; S.fitG = null; clearTimeout(S.timer);
      smile(); shot.classList.remove("live", "inking"); say("");
    }
    clearBtn.addEventListener("click", (ev) => { ev.stopPropagation(); clear(); });
    clearBtn.addEventListener("pointerdown", (ev) => ev.stopPropagation());
    clearBtn.addEventListener("touchstart", (ev) => ev.stopPropagation(), { passive: true });

    // ---------------------------------------------------------------- the pointer, onto the drawing
    // the canvas fills the frame the way the image does (object-fit: cover), so a point on screen maps to the
    // image's pixels, and from either half to the same unit square
    function toUnit(clientX, clientY) {
      const r = cv.getBoundingClientRect(), s = Math.max(r.width / CW, r.height / CH);
      const cx = (clientX - r.left - (r.width - CW * s) / 2) / s, cy = (clientY - r.top - (r.height - CH * s) / 2) / s;
      return [(cx - (cx < RX ? LX : RX)) / SQ, 1 - (cy - TOP) / SQ];
    }
    const INK = 0.22;                     // log-odds added under the pen per move event
    let inking = false, lower = false;
    function dab(clientX, clientY, amount) {
      const [x, y] = toUnit(clientX, clientY);
      brush(x, y, lower ? -amount : amount, 0.045);
      drawLeft();
    }
    function begin(clientX, clientY) { goLive(); inking = true; shot.classList.add("inking"); dab(clientX, clientY, INK); }
    function end() { if (!inking) return; inking = false; shot.classList.remove("inking"); fit(); }

    // mouse and pen: draw at once
    shot.addEventListener("pointerenter", (ev) => { if (ev.pointerType !== "touch") warm(); });
    shot.addEventListener("pointerdown", (ev) => {
      if (ev.pointerType === "touch" || ev.button !== 0) return;
      ev.preventDefault(); lower = ev.shiftKey; shot.setPointerCapture(ev.pointerId); begin(ev.clientX, ev.clientY);
    });
    shot.addEventListener("pointermove", (ev) => { if (ev.pointerType !== "touch" && inking) { lower = ev.shiftKey; dab(ev.clientX, ev.clientY, INK); } });
    shot.addEventListener("pointerup", (ev) => { if (ev.pointerType !== "touch") end(); });
    shot.addEventListener("pointercancel", (ev) => { if (ev.pointerType !== "touch") end(); });
    shot.addEventListener("dragstart", (ev) => ev.preventDefault());
    shot.addEventListener("contextmenu", (ev) => { if (inking || hold) ev.preventDefault(); });

    // touch: a finger that moves at once is scrolling the page and is left alone; one that rests for HOLD ms draws,
    // and from then on its moves are the drawing's (touchmove's default, the scroll, is prevented); a short tap
    // drops one dab
    const HOLD = 200, SLOP = 9;
    let hold = 0, t0 = null, moved = false;
    shot.addEventListener("touchstart", (ev) => {
      warm();
      if (ev.touches.length !== 1) { clearTimeout(hold); hold = 0; return; }
      const p = ev.touches[0]; t0 = { x: p.clientX, y: p.clientY, at: performance.now() }; moved = false; lower = false;
      clearTimeout(hold);
      hold = setTimeout(() => { hold = 0; if (!moved) begin(t0.x, t0.y); }, HOLD);
    }, { passive: true });
    shot.addEventListener("touchmove", (ev) => {
      const p = ev.touches[0];
      if (inking) { ev.preventDefault(); dab(p.clientX, p.clientY, INK); return; }
      if (t0 && Math.hypot(p.clientX - t0.x, p.clientY - t0.y) > SLOP) { moved = true; clearTimeout(hold); hold = 0; }
    }, { passive: false });
    shot.addEventListener("touchend", (ev) => {
      if (inking) { ev.preventDefault(); end(); return; }
      if (hold && !moved && t0) {                // a tap: one dab of odds, then the fit
        clearTimeout(hold); hold = 0; ev.preventDefault();
        begin(t0.x, t0.y); dab(t0.x, t0.y, 0.6); end();
      }
      t0 = null;
    }, { passive: false });
    shot.addEventListener("touchcancel", () => { clearTimeout(hold); hold = 0; t0 = null; end(); });
  })();

  // ================================================================== Noise-State Calculus
  (function tugCard() {
    const shot = document.getElementById("card-tug");
    if (!shot) return;
    const NS = "http://www.w3.org/2000/svg";
    const svg = shot.querySelector("svg");
    const el = (tag, attrs, parent) => { const n = document.createElementNS(NS, tag); for (const [k, v] of Object.entries(attrs || {})) n.setAttribute(k, v); if (parent) parent.appendChild(n); return n; };
    const text = (g, x, y, s, cls, anchor = "start") => { const t = el("text", { x, y, class: cls || "", "text-anchor": anchor }, g); t.textContent = s; return t; };
    function pencil(pts, seed, wob = 0.6) {
      const r = mulberry32(seed);
      let d = `M${pts[0][0].toFixed(1)},${pts[0][1].toFixed(1)}`;
      for (let i = 1; i < pts.length; ++i) { const [x0, y0] = pts[i - 1], [x1, y1] = pts[i]; d += ` Q${((x0 + x1) / 2 + (r() - 0.5) * wob * 2).toFixed(1)},${((y0 + y1) / 2 + (r() - 0.5) * wob * 2).toFixed(1)} ${x1.toFixed(1)},${y1.toFixed(1)}`; }
      return d;
    }
    const LO = -2, HI = 3, START = Math.log10(9);
    let lp = START, C = null, W = 0, H = 0, parts = null, touched = false;

    function curvesAt(q) {
      const L = C.logp; let k = 1; while (k < L.length - 1 && L[k] < q) k++;
      const w = clamp((q - L[k - 1]) / (L[k] - L[k - 1]));
      const mix = (A) => A[k - 1].map((a, i) => a + (A[k][i] - a) * w);
      return { D1: mix(C.D1), D2: mix(C.D2), X: mix(C.X) };
    }
    const fmtP = (p) => (p >= 10 ? Math.round(p).toString() : p >= 1 ? p.toFixed(1).replace(/\.0$/, "") : p.toPrecision(1));

    // the fixed parts are laid out for the frame's size in CSS pixels, so the text stays its size on any card
    function layout() {
      const r = svg.getBoundingClientRect();
      W = Math.max(200, r.width); H = Math.max(100, r.height);
      svg.setAttribute("viewBox", `0 0 ${W.toFixed(1)} ${H.toFixed(1)}`);
      svg.textContent = "";
      const small = W < 420;
      const m = { l: small ? 14 : 22, r: small ? 12 : 20, t: small ? 28 : 36, b: small ? 34 : 42 };
      const X0 = m.l, PW = W - m.l - m.r, Y0 = m.t + (H - m.t - m.b) / 2, S = (H - m.t - m.b) / 2 / 10.6;
      const sx = (t) => X0 + PW * t, sy = (v) => Y0 - S * v;
      const fixed = el("g", {}, svg);
      for (const sg of [1, -1]) {
        const pts = []; for (let k = 0; k <= 20; ++k) pts.push([sx(k / 20), sy((sg * (1 - k / 20)) / C.r)]);
        el("path", { d: pencil(pts, sg > 0 ? 81 : 82, 0.2), class: "pencil soft", "stroke-width": 1.4, "stroke-dasharray": "2 5" }, fixed);
      }
      text(fixed, sx(1), sy(10) - 4, small ? "dotted: if nobody watched" : "dotted: if nobody were watching, (T − t)/r", "mono", "end");
      text(fixed, sx(0.03), Y0 - 6, "the state stays at 0", "label");
      const d1 = el("path", { class: "pencil accent", "stroke-width": 2.4 }, svg);
      const d2 = el("path", { class: "pencil warm", "stroke-width": 2.4 }, svg);
      const xs = el("path", { class: "pencil", "stroke-width": 1.8 }, svg);
      // each player's label on the outside of its curve, over the late stretch where the curve is low: clear of the
      // dotted line there too, which bounds every curve
      const lx = sx(0.985), lt = small ? 0.62 : 0.55;
      text(fixed, lx, sy((1 - lt) / C.r) - 6, "player 1 pushes up", "label acc", "end");
      text(fixed, lx, sy(-(1 - lt) / C.r) + (small ? 13 : 15), "player 2 pushes down", "label warmt", "end");
      const pv = text(svg, m.l, small ? 17 : 21, "", "mono pv");
      // the dial along the foot: where p sits between blurry and sharp
      const foot = H - (small ? 9 : 12);
      const ta = m.l + 34, tb = W - m.r - 34, ty = foot - (small ? 16 : 19);
      text(svg, m.l, ty + 3.5, "0.01", "mono");
      text(svg, W - m.r, ty + 3.5, "1000", "mono", "end");
      el("path", { d: pencil([[ta, ty], [(ta + tb) / 2, ty], [tb, ty]], 86, 0.3), class: "pencil soft", "stroke-width": 1.2 }, svg);
      text(svg, W / 2, foot, "drag: how clearly they see each other", "mono cap", "middle");
      const tick = el("circle", { r: 4, class: "fillacc" }, svg);
      parts = { sx, sy, d1, d2, xs, pv, tick, ta, tb, ty, m };
      draw();
    }
    function draw() {
      if (!parts || !C) return;
      const { sx, sy, d1, d2, xs, pv, tick, ta, tb, ty } = parts, q = curvesAt(lp), t = C.t;
      d1.setAttribute("d", pencil(t.map((u, i) => [sx(u), sy(q.D1[i])]), 83, 0.15));
      d2.setAttribute("d", pencil(t.map((u, i) => [sx(u), sy(q.D2[i])]), 84, 0.15));
      xs.setAttribute("d", pencil(t.map((u, i) => [sx(u), sy(q.X[i])]), 85, 0.15));
      const p = Math.pow(10, lp);
      pv.textContent = `p = ${fmtP(p)} \u00b7 ${W < 420 ? "" : "player 1's "}first push ${q.D1[0].toFixed(2)}`;
      tick.setAttribute("cx", (ta + (tb - ta) * (lp - LO) / (HI - LO)).toFixed(1)); tick.setAttribute("cy", ty.toFixed(1));
      shot.setAttribute("aria-valuenow", p.toPrecision(3));
      shot.setAttribute("aria-valuetext", `precision ${fmtP(p)}: player 1's first push ${q.D1[0].toFixed(2)}`);
    }
    const set = (q) => { lp = clamp(q, LO, HI); draw(); };

    // idle: one slow sweep, blurry to sharp and back to the start, when the card first comes into view
    let sweeping = 0;
    function sweep() {
      if (reduced || touched) return;
      const t0 = performance.now(), dur = 5200;
      const path = (u) => {              // 9 -> 0.01 -> 1000 -> 9, eased
        const e = (s) => (s < 0.5 ? 2 * s * s : 1 - Math.pow(-2 * s + 2, 2) / 2);
        if (u < 0.25) return START + (LO - START) * e(u / 0.25);
        if (u < 0.75) return LO + (HI - LO) * e((u - 0.25) / 0.5);
        return HI + (START - HI) * e((u - 0.75) / 0.25);
      };
      const step = (now) => {
        if (touched) { sweeping = 0; return; }
        const u = (now - t0) / dur;
        set(path(Math.min(1, u)));
        sweeping = u < 1 ? requestAnimationFrame(step) : 0;
      };
      sweeping = requestAnimationFrame(step);
    }
    const stop = () => { touched = true; if (sweeping) cancelAnimationFrame(sweeping); sweeping = 0; };

    // load the curves when the card comes near the screen (like a lazy image), draw, and sweep once in view
    let loaded = false, seen = false;
    function load() {
      if (loaded) return; loaded = true;
      fetch(shot.dataset.curves).then((r) => r.json()).then((j) => { C = j; layout(); if (seen) sweep(); })
        .catch(() => { shot.classList.add("failed"); });
    }
    if ("IntersectionObserver" in window) {
      new IntersectionObserver((es, o) => { if (es.some((e) => e.isIntersecting)) { load(); o.disconnect(); } }, { rootMargin: "400px" }).observe(shot);
      new IntersectionObserver((es, o) => {
        if (es.some((e) => e.isIntersecting && e.intersectionRatio > 0.6)) { seen = true; o.disconnect(); if (C) setTimeout(sweep, 500); }
      }, { threshold: [0.6] }).observe(shot);
    } else load();
    if ("ResizeObserver" in window) { let w0 = 0, h0 = 0; new ResizeObserver(() => { const r = svg.getBoundingClientRect(); if (C && (Math.abs(r.width - w0) > 0.5 || Math.abs(r.height - h0) > 0.5)) { w0 = r.width; h0 = r.height; layout(); } }).observe(shot); }

    // drag across: the pointer's place along the card is log p. A touch that starts vertical is a scroll
    // (touch-action: pan-y hands it to the page), and a press alone does not move p until it moves sideways or lets go
    let drag = null;
    const fromX = (clientX) => { const r = svg.getBoundingClientRect(), a = parts.ta * r.width / W, b = parts.tb * r.width / W; return LO + (HI - LO) * clamp((clientX - r.left - a) / (b - a)); };
    shot.addEventListener("pointerdown", (ev) => {
      if (!parts || ev.button > 0) return;
      stop(); drag = { x: ev.clientX, id: ev.pointerId, on: ev.pointerType !== "touch" };
      if (drag.on) { shot.setPointerCapture(ev.pointerId); set(fromX(ev.clientX)); }
    });
    shot.addEventListener("pointermove", (ev) => {
      if (!drag || ev.pointerId !== drag.id) return;
      if (!drag.on && Math.abs(ev.clientX - drag.x) > 4) { drag.on = true; try { shot.setPointerCapture(ev.pointerId); } catch (e) { /* gone */ } }
      if (drag.on) set(fromX(ev.clientX));
    });
    shot.addEventListener("pointerup", (ev) => { if (drag && !drag.on) set(fromX(ev.clientX)); drag = null; });
    shot.addEventListener("pointercancel", () => { drag = null; });
    shot.addEventListener("keydown", (ev) => {
      const k = ev.key, d = { ArrowRight: 0.1, ArrowUp: 0.1, ArrowLeft: -0.1, ArrowDown: -0.1, PageUp: 0.5, PageDown: -0.5 }[k];
      if (d !== undefined) { stop(); set(lp + d); ev.preventDefault(); }
      else if (k === "Home") { stop(); set(LO); ev.preventDefault(); }
      else if (k === "End") { stop(); set(HI); ev.preventDefault(); }
    });
  })();
})();
