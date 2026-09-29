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
// Noise-State Calculus: the picture is baked (images/card-nsheads-*) and its hover is CSS (home.css and
// partials/card-nsheads.html). A touch screen has no hover, so a tap on the picture toggles the same state.
(function () {
  "use strict";
  const isDark = () => document.documentElement.classList.contains("dark");
  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  const themeWatchers = [];
  let wasDark = document.documentElement.classList.contains("dark");   // only a change of theme, not every class on <html> (heroink toggles one as the pointer crosses the mesh)
  new MutationObserver(() => { const d = document.documentElement.classList.contains("dark"); if (d === wasDark) return; wasDark = d; themeWatchers.forEach((f) => f()); })
    .observe(document.documentElement, { attributes: true, attributeFilter: ["class"] });

  // ================================================================== Decision Mesh
  (function meshCard() {
    const shot = document.getElementById("card-mesh");
    if (!shot) return;
    const cv = shot.querySelector("canvas"), ctx = cv.getContext("2d");
    const clearBtn = shot.querySelector(".clear"), status = shot.querySelector(".status");
    const imgs = [...shot.querySelectorAll("img")];

    // the image's composition, in its own 960 x 480 pixels: two squares of side 532, the drawing and the fit, each
    // centred in its own half (at x -26 and 454, y -52) and cropped the same way by it, so a point of the drawing and
    // the same point of the fit sit at the same place in their halves. MID splits the halves; each paints only its own.
    const CW = 960, CH = 480, SQ = 532, MID = CW / 2, LX = (MID - SQ) / 2, RX = MID + (MID - SQ) / 2, TOP = -52;
    cv.width = CW; cv.height = CH;
    const BASE = -1, VMAX = 2.2, G = 64, D = 160;
    const SITES = 8000, FLIPS = 30, SD = 0.25;   // enough coins that the fit follows a scribble closely
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
      const s = (isDark() ? ["#2b6cf0", "#5b8ff9", "#2c2c2c", "#f0845c", "#f5c451"] : ["#1d4ed8", "#6d9cf5", "#ffffff", "#f29a58", "#d9591c"]).map(hex);   // the middle is the paper: pure white (multiplied) and neutral 44 (which the dark filter takes to black), so an untouched coin shows the paper and nothing else
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
      ctx.save(); ctx.beginPath(); const left = x0 === LX; ctx.rect(left ? 0 : MID, 0, left ? MID : CW - MID, CH); ctx.clip();   // each square only in its own half
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
      if (S.blank) { ctx.fillStyle = paper(); ctx.fillRect(MID, 0, CW - MID, CH); return; }   // cleared: nothing drawn yet
      const im = currentImg();       // until something has been fitted, the right half is the image's own
      if (im && im.complete && im.naturalWidth) ctx.drawImage(im, im.naturalWidth / 2, 0, im.naturalWidth / 2, im.naturalHeight, MID, 0, CW - MID, CH);
      else { ctx.fillStyle = paper(); ctx.fillRect(MID, 0, CW - MID, CH); }
    }
    function drawCoins() {
      ctx.save(); ctx.beginPath(); ctx.rect(MID, 0, CW - MID, CH); ctx.clip();
      ctx.fillStyle = paper(); ctx.fillRect(MID, 0, CW - MID, CH);
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
      ctx.save(); ctx.beginPath(); ctx.rect(MID, 0, CW - MID, CH); ctx.clip();
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
        k[i] = h;
      }
      // the estimator gets the coins pooled into a BIN x BIN grid of cells (flips and heads summed): the fit's cost
      // grows with its rows, and on a card the pooled cells lose nothing visible, at about half the time
      const BIN = 40, bn = new Int32Array(BIN * BIN), bk = new Int32Array(BIN * BIN);
      for (let i = 0; i < SITES; ++i) { const c = Math.min(BIN - 1, Math.floor(y[i] * BIN)) * BIN + Math.min(BIN - 1, Math.floor(x[i] * BIN)); bn[c] += n[i]; bk[c] += k[i]; }
      for (let c = 0; c < BIN * BIN; ++c) if (bn[c]) rows.push((((c % BIN) + 0.5) / BIN).toFixed(5) + "," + ((Math.floor(c / BIN) + 0.5) / BIN).toFixed(5) + "," + bn[c] + "," + bk[c]);
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
      const msg = { id: S.id, engine: "tri", design: S.data.csv, seed: 7, q: 0.3, split: false };   // every coin fits; a looser gate
      warm();
      if (S.ready) S.worker.postMessage(msg); else S.queued = msg;
      say(S.ready ? `fitting ${SITES.toLocaleString()} coins…` : "loading the estimator…");
    }
    function goLive() {
      if (S.live) return;
      S.live = true; shot.classList.add("live");
      drawAll();
    }
    // clear wipes the card, face and all, to a blank sheet of odds to draw on; the picture comes back on the next visit
    function clear() {
      S.id += 1; S.queued = null; S.data = null; S.fit = null; S.fitG = null; clearTimeout(S.timer);
      drawing.fill(BASE); S.blank = true; shot.classList.remove("inking");
      S.live = false; goLive(); say("");
    }
    clearBtn.addEventListener("click", (ev) => { ev.stopPropagation(); clear(); });
    clearBtn.addEventListener("pointerdown", (ev) => ev.stopPropagation());
    clearBtn.addEventListener("touchstart", (ev) => ev.stopPropagation(), { passive: true });

    // ---------------------------------------------------------------- the pointer, onto the drawing
    // the canvas fills the frame the way the image does (object-fit: cover), so a point on screen maps to the
    // image's pixels, and from either half to the same unit square
    function toUnit(clientX, clientY) {
      const r = shot.getBoundingClientRect(), s = Math.max(r.width / CW, r.height / CH);   // the frame, which the canvas fills (the canvas is hidden until the first touch)
      const cx = (clientX - r.left - (r.width - CW * s) / 2) / s, cy = (clientY - r.top - (r.height - CH * s) / 2) / s;
      return [(cx - (cx < MID ? LX : RX)) / SQ, 1 - (cy - TOP) / SQ];
    }
    const INK = 0.22;                     // log-odds added under the pen per move event
    let inking = false, lower = false;
    // a touch that lands off the drawn square (the crop's margins, the strip between the halves) draws nothing
    const inside = (x, y) => x >= -0.02 && x <= 1.02 && y >= -0.02 && y <= 1.02;
    function dab(clientX, clientY, amount) {
      const [x, y] = toUnit(clientX, clientY);
      if (!inside(x, y)) return;
      brush(x, y, lower ? -amount : amount, 0.045);
      drawLeft();
    }
    function begin(clientX, clientY) { goLive(); inking = true; shot.classList.add("inking"); dab(clientX, clientY, INK); }
    function end() { if (!inking) return; inking = false; shot.classList.remove("inking"); fit(); }

    // mouse and pen: draw at once
    shot.addEventListener("pointerenter", (ev) => { if (ev.pointerType !== "touch") warm(); });
    // the drawing affordance is CSS (home.css); a touch screen, which has no hover, gets the sketch frames once,
    // briefly, when the card first comes into view
    if (window.matchMedia("(hover: none) and (pointer: coarse)").matches && "IntersectionObserver" in window) {
      const io = new IntersectionObserver((es) => { if (es.some((e) => e.isIntersecting)) { io.disconnect();
        shot.classList.add("peek"); setTimeout(() => shot.classList.remove("peek"), 1800); } }, { threshold: 0.6 });
      io.observe(shot);
    }
    shot.addEventListener("pointerdown", (ev) => {
      if (ev.pointerType === "touch" || ev.button !== 0) return;
      if (!inside(...toUnit(ev.clientX, ev.clientY))) return;
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
      hold = setTimeout(() => { hold = 0; if (!moved && t0 && inside(...toUnit(t0.x, t0.y))) begin(t0.x, t0.y); }, HOLD);
    }, { passive: true });
    shot.addEventListener("touchmove", (ev) => {
      const p = ev.touches[0];
      if (inking) { ev.preventDefault(); dab(p.clientX, p.clientY, INK); return; }
      if (t0 && Math.hypot(p.clientX - t0.x, p.clientY - t0.y) > SLOP) { moved = true; clearTimeout(hold); hold = 0; }
    }, { passive: false });
    shot.addEventListener("touchend", (ev) => {
      if (inking) { ev.preventDefault(); end(); return; }
      if (hold && !moved && t0 && inside(...toUnit(t0.x, t0.y))) {   // a tap on the square: one dab of odds, then the fit
        clearTimeout(hold); hold = 0; ev.preventDefault();
        begin(t0.x, t0.y); dab(t0.x, t0.y, 0.6); end();
      }
      t0 = null;
    }, { passive: false });
    shot.addEventListener("touchcancel", () => { clearTimeout(hold); hold = 0; t0 = null; end(); });
  })();

  // ================================================================== the cards' frames
  // Each card's border, drawn by hand (home.css), in one svg per card, laid out once for the card's size (and again
  // when it changes); CSS fades it (the Noise-State card's hover fades its frame as it did its border). Each is the
  // project's own:
  //   Noise-State: one random walk around the rounded rectangle, wandering a few pixels off it, closing where it began;
  //   Decision Mesh: a thin band of the mesh's own cells, isosceles right triangles, coarse where nothing asks for
  //     detail and bisected (newest-vertex bisection, the estimator's own refinement) into smaller ones near the
  //     corners and at a few places along the edges, as the estimator refines only where the flips ask for it.
  // Both keep to the outer 10px of the card. The Decision Mesh picture is set in past that (--mat); the Noise-State
  // picture runs out to the walk and is clipped by it.
  (function cardFrames() {
    const NS = "http://www.w3.org/2000/svg";
    const mk = (tag, attrs, parent) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); parent && parent.appendChild(e); return e; };
    const f1 = (v) => (Math.round(v * 10) / 10).toString();
    const gauss = (r) => Math.sqrt(-2 * Math.log(1 - r())) * Math.cos(2 * Math.PI * r());
    // the rounded rectangle inset from the card's box: its length P and, at distance d clockwise from the top-left
    // end of the top side, the point and the outward normal
    function roundRect(W, H, inset, R) {
      const x0 = inset, y0 = inset, w = W - 2 * inset, h = H - 2 * inset, sw = w - 2 * R, sh = h - 2 * R, arc = (Math.PI / 2) * R;
      const segs = [["l", sw, x0 + R, y0, 1, 0], ["a", arc, x0 + w - R, y0 + R, -Math.PI / 2], ["l", sh, x0 + w, y0 + R, 0, 1], ["a", arc, x0 + w - R, y0 + h - R, 0],
        ["l", sw, x0 + w - R, y0 + h, -1, 0], ["a", arc, x0 + R, y0 + h - R, Math.PI / 2], ["l", sh, x0, y0 + h - R, 0, -1], ["a", arc, x0 + R, y0 + R, Math.PI]];
      const P = 2 * (sw + sh) + 4 * arc;
      return { P, at(d) {
        d = ((d % P) + P) % P;
        for (const [kind, len, ax, ay, p, qq] of segs) {
          if (d <= len) { if (kind === "l") return [ax + p * d, ay + qq * d, qq, -p]; const t = p + d / R, c = Math.cos(t), s = Math.sin(t); return [ax + R * c, ay + R * s, c, s]; }
          d -= len;
        }
        return [x0 + R, y0, 0, -1];
      } };
    }
    // Noise-State, path: a random walk off the rounded rectangle, pulled back toward it (an Ornstein-Uhlenbeck path,
    // Brownian at small scales, wandering about sd px over stretches of about ell px), its end drift taken out so it
    // closes where it began; the pencil runs a few px past its start, as a hand closing a loop does
    function walk(W, H, inset, R, sd, ell, seed) {
      const r = mulberry32(seed), rr = roundRect(W, H, inset, R), step = 2, N = Math.ceil(rr.P / step), th = step / ell, sig = sd * Math.sqrt(2 * th);
      const S = [sd * gauss(r)];
      for (let k = 1; k <= N; ++k) S.push(S[k - 1] * (1 - th) + sig * gauss(r));
      const B0 = S.map((s, k) => s - (k / N) * (S[N] - S[0])), cap = 2.6 * sd, d0 = rr.P * (0.08 + 0.2 * r());
      const B = B0.map((_, k) => { let a = 0; for (let j = -3; j <= 3; ++j) a += B0[(k + j + N) % N]; return a / 7; });   // the pencil's own width: no kinks finer than it
      const pts = [];
      for (let k = 0; k <= N + 3; ++k) { const [x, y, nx, ny] = rr.at(d0 + (k * rr.P) / N), o = Math.max(-cap, Math.min(cap, B[k % N]));
        pts.push([x + nx * o, y + ny * o]); }
      return pts;                                   // N + 4 points: the loop, then the few past its start
    }
    const polyline = (pts, dx = 0, dy = 0, k = 1, cx = 0, cy = 0) =>
      "M" + pts.map(([x, y]) => f1(cx + (x - dx - cx) / k) + "," + f1(cy + (y - dy - cy) / k)).join(" L");
    // Decision Mesh, mesh: the band between the card's edge (inset o) and b further in. Along each side, isosceles
    // right triangles on alternate edges of the band (hypotenuse 2b, the right angle on the far edge); in each corner a
    // b x b square cut on its diagonal. Each triangle is bisected, from its right angle to the middle of its hypotenuse,
    // as many times as its nearness to a corner or to one of a few seeded places along the edges asks; the edges on
    // the card's edge go to one path, those on the band's inner edge to another, the rest to a third
    function meshBand(W, H, o, b, levels, reach, nspots, seed) {
      const r = mulberry32(seed), tris = [];
      const X0 = o, Y0 = o, X1 = W - o, Y1 = H - o;
      const side = (P0, P1, n0) => {           // P0 -> P1 along the outer edge; n0 the inward normal
        const L = Math.hypot(P1[0] - P0[0], P1[1] - P0[1]), ux = (P1[0] - P0[0]) / L, uy = (P1[1] - P0[1]) / L, Ls = L - 2 * b;
        const m = Math.max(2, 2 * Math.round(Ls / (2 * b))), s = Ls / m;
        const pt = (t, inner) => [P0[0] + ux * (b + t) + (inner ? n0[0] * b : 0), P0[1] + uy * (b + t) + (inner ? n0[1] * b : 0)];
        tris.push([pt(0, true), pt(0, false), pt(s, true)]);                         // [right angle, hypotenuse ends]
        for (let k = 1; k < m; ++k) tris.push([pt(k * s, k % 2 === 1), pt((k - 1) * s, k % 2 === 0), pt((k + 1) * s, k % 2 === 0)]);
        tris.push([pt(Ls, true), pt(Ls, false), pt(Ls - s, true)]);
        // the corner square at P1's end: outer corner P1, inner corner, and the two points where the sides' bands meet it
        const Oc = P1, Ic = [P1[0] - ux * b + n0[0] * b, P1[1] - uy * b + n0[1] * b], A = [P1[0] - ux * b, P1[1] - uy * b], Bp = [P1[0] + n0[0] * b, P1[1] + n0[1] * b];
        tris.push([A, Oc, Ic], [Bp, Oc, Ic]);
      };
      side([X0, Y0], [X1, Y0], [0, 1]); side([X1, Y0], [X1, Y1], [-1, 0]); side([X1, Y1], [X0, Y1], [0, -1]); side([X0, Y1], [X0, Y0], [1, 0]);
      const corners = [[X0, Y0], [X1, Y0], [X1, Y1], [X0, Y1]];
      // a few places along the edges where the flips ask for detail: each a point of the band, how deep it refines
      // there (1 or 2 levels) and how far that reaches; kept clear of the corners, which refine anyway
      const Pm = 2 * (X1 - X0 + Y1 - Y0), onBand = (d) => { const w = X1 - X0, h = Y1 - Y0; d = ((d % Pm) + Pm) % Pm;
        if (d < w) return [X0 + d, Y0]; d -= w; if (d < h) return [X1, Y0 + d]; d -= h; if (d < w) return [X1 - d, Y1]; d -= w; return [X0, Y1 - d]; };
      const spots = [], c0 = r() * Pm;
      for (let k = 0; k < nspots; ++k) {
        let p; for (let t = 0; t < 40; ++t) { p = onBand(c0 + Pm * (k + 0.2 + 0.6 * r()) / nspots);
          if (Math.min(...corners.map(([x, y]) => Math.hypot(p[0] - x, p[1] - y))) > reach * 0.9) break; }
        spots.push([p[0], p[1], 1 + (r() < 0.5 ? 1 : 0), b * (3 + 3 * r())]);
      }
      const segs = new Map(), key = (p) => p[0].toFixed(2) + "," + p[1].toFixed(2);
      const add = (p, q2) => { const a = key(p), c = key(q2), k = a < c ? a + "|" + c : c + "|" + a; if (!segs.has(k)) segs.set(k, [p, q2]); };
      const leaf = (T, lev) => {
        if (lev > 0) { const [A, B, C] = T, M = [(B[0] + C[0]) / 2, (B[1] + C[1]) / 2]; leaf([M, A, B], lev - 1); leaf([M, C, A], lev - 1); return; }
        add(T[0], T[1]); add(T[1], T[2]); add(T[2], T[0]);
      };
      for (const T of tris) {
        const cx = (T[0][0] + T[1][0] + T[2][0]) / 3, cy = (T[0][1] + T[1][1] + T[2][1]) / 3;
        const d = Math.min(...corners.map(([x, y]) => Math.hypot(cx - x, cy - y)));
        let lev = Math.floor(levels * (1 - d / reach) + 0.5 + (r() - 0.5) * 0.9);
        for (const [sx, sy, sl, sr] of spots) lev = Math.max(lev, Math.floor(Math.min(sl, levels) * (1 - Math.hypot(cx - sx, cy - sy) / sr) + 0.5 + (r() - 0.5) * 0.6));
        lev = Math.max(0, Math.min(levels, lev));
        leaf(T, lev);
      }
      // a segment along one line x = c or y = c
      const along = (p, q2, xs, ys) => xs.some((c) => Math.abs(p[0] - c) < 0.01 && Math.abs(q2[0] - c) < 0.01) || ys.some((c) => Math.abs(p[1] - c) < 0.01 && Math.abs(q2[1] - c) < 0.01);
      let outer = "", cells = "", inner = "";
      for (const [p, q2] of segs.values()) {
        const s = `M${f1(p[0])},${f1(p[1])} L${f1(q2[0])},${f1(q2[1])}`;
        if (along(p, q2, [X0, X1], [Y0, Y1])) outer += s;
        else if (along(p, q2, [X0 + b, X1 - b], [Y0 + b, Y1 - b])) inner += s; else cells += s;
      }
      return { outer, cells, inner };
    }
    const cards = [...document.querySelectorAll(".home .demos article.demo")];
    cards.forEach((card) => {
      const ns = card.classList.contains("nsheads-card"), dm = !!card.querySelector("#card-mesh");
      if (!ns && !dm) return;
      const svg = mk("svg", { class: "cframe", "aria-hidden": "true", focusable: "false" });
      card.insertBefore(svg, card.firstChild);
      const shot = ns ? card.querySelector(".shot.nsheads") : null;
      const P = ns ? { path: mk("path", { class: "path" }, svg) }
        : { cells: mk("path", { class: "mesh cell" }, svg), inner: mk("path", { class: "mesh inner" }, svg), edge: mk("path", { class: "mesh edge" }, svg) };
      function lay() {
        const W = card.clientWidth, H = card.clientHeight;
        if (!W || !H) return;
        svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
        const small = W < 520;
        if (ns) {
          const pts = walk(W, H, 4, 10, small ? 1.2 : 1.6, small ? 80 : 110, 307), loop = pts.slice(0, -4);
          P.path.setAttribute("d", polyline(pts));
          card.nsLoop = loop; card.dispatchEvent(new Event("nsloop"));      // for the hover's morph into the rims (headsCard)
          // the picture runs edge to edge, out to the walk itself: its images and overlay are clipped to the loop (in
          // their own box, the picture's), and, since they zoom by 1.04 on hover about their centre, to the loop shrunk
          // by 1.04 about that centre while zoomed, so on screen the edge stays on the line (home.css eases between them)
          if (shot) { const dx = shot.offsetLeft, dy = shot.offsetTop, cx = shot.offsetWidth / 2, cy = shot.offsetHeight / 2;
            shot.style.setProperty("--nsclip", `path("${polyline(loop, dx, dy)} Z")`);
            shot.style.setProperty("--nsclipz", `path("${polyline(loop, dx, dy, 1.04, cx, cy)} Z")`); }
        }
        else {   // the finest cells only where there are pixels for them
          const b = small ? 7 : 8, m = meshBand(W, H, 1.5, b, (window.devicePixelRatio || 1) >= 1.5 ? 3 : 2, 12 * b, small ? 3 : 5, 401);
          P.cells.setAttribute("d", m.cells); P.inner.setAttribute("d", m.inner); P.edge.setAttribute("d", m.outer);
        }
      }
      lay();
      if ("ResizeObserver" in window) new ResizeObserver(lay).observe(card);
    });
  })();

  // ================================================================== Noise-State Calculus
  (function headsCard() {
    const shot = document.querySelector(".home .shot.nsheads"), card = shot && shot.closest(".demo"), rim = card && card.querySelector(".nsheads-rim");
    if (!shot) return;
    // a finger or a pen has no hover: a tap on the picture toggles the same state. Decided by the tap itself, not by
    // what the browser says about the device, which some phones (and "desktop site" modes) get wrong
    // A tap or a press-and-hold both count (a long press fires no click, so the release decides); a finger that moves
    // is scrolling the page and counts as neither
    let down = null;
    shot.addEventListener("pointerdown", (ev) => { down = ev.pointerType === "mouse" ? null : { x: ev.clientX, y: ev.clientY }; });
    shot.addEventListener("pointermove", (ev) => { if (down && Math.hypot(ev.clientX - down.x, ev.clientY - down.y) > 10) down = null; });
    shot.addEventListener("pointercancel", () => { down = null; });
    // a tapped card goes back to rest on its own: after a while untouched, or once it has scrolled away
    let rest = 0;
    const settle = () => { clearTimeout(rest); if (card.classList.contains("on")) rest = setTimeout(() => { card.classList.remove("on"); sync(); }, 15000); };
    shot.addEventListener("pointerup", () => { if (down) { card.classList.toggle("on"); sync(); settle(); } down = null; });
    if ("IntersectionObserver" in window) new IntersectionObserver(([e]) => { if (!e.isIntersecting) { card.classList.remove("on"); sync(); clearTimeout(rest); } }).observe(card);
    // The hover's way in and its way back (home.css) are CSS transitions, so either one turns back from wherever the
    // other got to. Only the timing needs help: each way has beats that wait for the one before (the rims wait for the
    // trails, the split for the rims), and when the other way was cut short those beats have nothing to wait for. So,
    // as the state flips, this works out how far the interrupted way had got, as the matching moment of the new way
    // (EQ: a moment of the way in and the moment of the way back that shows about the same picture), and sets it as
    // --nsin or --nsout, which the CSS subtracts from that way's delays. Set in the same task as the flip, so the
    // transitions it starts already see it.
    // the beats now: way in, clouds fade 0-1.15, bubbles 0.15 / 0.4 / 0.8 / 1.15, frame into rims 0.5-1.45, split 1.8-2.76;
    // way back, world 0-0.94, outer bubbles 0.59 / 0.8, rims into frame 1.25-2.1, inner bubbles 1.1 / 1.4, clouds
    // 1.03-2.2. While the third bubble is up, the matching moment on the way back keeps the frame line waiting until it
    // has gone (so no bubble ever lies across the line, cards.js morph)
    const EQ = [[0, 2.2], [0.4, 1.6], [0.8, 0.85], [1.45, 0.7], [1.8, 0.587], [2.76, 0]];   // [seconds into the way in, seconds into the way back]
    const along = (x, a, b) => {                  // piecewise linear through EQ, from column a to column b (clamped)
      const P = EQ.slice().sort((p, q) => p[a] - q[a]);
      if (x <= P[0][a]) return P[0][b];
      for (let j = 1; j < P.length; ++j) if (x <= P[j][a]) { const u = (x - P[j - 1][a]) / (P[j][a] - P[j - 1][a]); return P[j - 1][b] + u * (P[j][b] - P[j - 1][b]); }
      return P[P.length - 1][b];
    };
    const canHover = window.matchMedia("(hover: hover)");
    let hovering = false, lit = false, since = -1e9, head = 0;
    const engagedNow = () => { let f = false; try { f = card.matches(":has(:focus-visible)"); } catch (e) {} return (canHover.matches && hovering) || card.classList.contains("on") || f; };
    function sync() {
      const now = engagedNow(); if (now === lit) return;
      const got = (performance.now() - since) / 1000 + head;       // how far the way that is ending had got
      lit = now; since = performance.now();
      head = now ? along(got, 1, 0) : along(got, 0, 1);
      card.style.setProperty(now ? "--nsin" : "--nsout", head.toFixed(3) + "s");
      if (morph) morph(now, head);
    }
    let morph = null;
    card.addEventListener("pointerenter", (ev) => { if (ev.pointerType !== "touch") { hovering = true; sync(); } });
    card.addEventListener("pointerleave", (ev) => { if (ev.pointerType !== "touch") { hovering = false; sync(); } });
    for (const type of ["focusin", "focusout"]) card.addEventListener(type, () => { sync(); setTimeout(sync); });
    new MutationObserver(sync).observe(card, { attributes: true, attributeFilter: ["class"] });
    shot.addEventListener("contextmenu", (ev) => ev.preventDefault());   // no save-image menu on a long press
    thinking();
    if (!rim) return;
    // the two thought-cloud rims, laid out once for the card's size (and again when it changes size): scallops along a
    // rounded rectangle just outside the card, one per player, a few pixels apart and with their own hand wobble, and
    // the bubble trails that lead up to them from the heads. CSS fades them in and out; nothing here runs per frame.
    const PAD = 12;
    function scallops(W, H, inset, step, seed) {
      const r = mulberry32(seed), x0 = inset, y0 = inset, w = W - 2 * inset, h = H - 2 * inset, R = 16;
      const straight = [w - 2 * R, h - 2 * R], arc = (Math.PI / 2) * R, P = 2 * (straight[0] + straight[1]) + 4 * arc;
      const at = (d) => {                     // the point at distance d along the rounded rectangle, clockwise from its top-left
        d = ((d % P) + P) % P;
        const segs = [["l", straight[0], x0 + R, y0, 1, 0], ["a", arc, x0 + w - R, y0 + R, -Math.PI / 2], ["l", straight[1], x0 + w, y0 + R, 0, 1], ["a", arc, x0 + w - R, y0 + h - R, 0],
          ["l", straight[0], x0 + w - R, y0 + h, -1, 0], ["a", arc, x0 + R, y0 + h - R, Math.PI / 2], ["l", straight[1], x0, y0 + h - R, 0, -1], ["a", arc, x0 + R, y0 + R, Math.PI]];
        for (const [kind, len, ax, ay, p, q] of segs) {
          if (d <= len) return kind === "l" ? [ax + p * d, ay + q * d] : [ax + R * Math.cos(p + d / R), ay + R * Math.sin(p + d / R)];
          d -= len;
        }
        return [x0 + R, y0];
      };
      const n = Math.max(12, Math.round(P / step)), off = r() * P / n;
      const pts = Array.from({ length: n }, (_, k) => at(off + (k * P) / n));
      let d = `M${pts[0][0].toFixed(1)},${pts[0][1].toFixed(1)}`;
      const line = [];                          // the same scallops as points, for the morph
      for (let k = 1; k <= n; ++k) {
        const [x, y] = pts[k % n], [px, py] = pts[k - 1], c = Math.hypot(x - px, y - py), rr = c * (0.56 + 0.1 * r());
        d += ` A${rr.toFixed(1)},${rr.toFixed(1)} 0 0 1 ${x.toFixed(1)},${y.toFixed(1)}`;   // clockwise arcs bulge outward
        const hh = Math.sqrt(Math.max(0, rr * rr - (c / 2) * (c / 2))), mx = (x + px) / 2, my = (y + py) / 2;
        const cx = mx - ((y - py) / c) * hh, cy = my + ((x - px) / c) * hh;           // the centre, inside the rectangle
        let a0 = Math.atan2(py - cy, px - cx), a1 = Math.atan2(y - cy, x - cx); if (a1 < a0) a1 += 2 * Math.PI;
        for (let q = 0; q < 10; ++q) { const a = a0 + ((a1 - a0) * q) / 10; line.push([cx + rr * Math.cos(a), cy + rr * Math.sin(a)]); }
      }
      scallops.last = line;
      return d + " Z";
    }
    function layout() {
      const W = card.offsetWidth + 2 * PAD, H = card.offsetHeight + 2 * PAD;
      if (!W || !H) return;
      rim.setAttribute("viewBox", `0 0 ${W} ${H}`);
      const [blue, warm] = rim.querySelectorAll(".scallop");
      blue.setAttribute("d", scallops(W, H, 7, 30, 71)); rims[0] = scallops.last;
      warm.setAttribute("d", scallops(W, H, 3, 34, 73)); rims[1] = scallops.last;
      shapes = null;
      // a trail of four bubbles from each head up to that player's own rim, growing as it rises and leaning a little
      // toward the middle: it starts just off the forehead (a point of the picture, 480 x 240 units, drawn by the same
      // prototype as the images) and its last bubble sits just inside the inner (blue) rim, clear of it, so the warm trail never
      // crosses the blue rim on its way to its own. The picture is cropped to the
      // card ("slice") and zooms by 1.04 on hover, so the start is mapped through both.
      const sw = shot.offsetWidth, sh = shot.offsetHeight, k = Math.max(sw / 480, sh / 240), Z = 1.04;
      const pic = ([u, v]) => [PAD + shot.offsetLeft + sw / 2 + Z * (u - 240) * k, PAD + shot.offsetTop + sh / 2 + Z * (v - 120) * k];
      const R = [2, 2.9, 3.8, 4.8].map((r) => r * Math.min(1, Math.max(0.8, k)));
      [["acc", [128, 45], 7], ["warm", [352, 45], 3]].forEach(([c, at, inset]) => {
        const [x0, y0] = pic(at), y1 = 7 + 3.6 + R[3], lean = (at[0] < 240 ? 1 : -1) * 0.32 * (y0 - y1);
        rim.querySelectorAll(".tail." + c).forEach((b, j) => {
          const t = [0, 0.3, 0.62, 1][j], x = x0 + lean * t * t, y = y0 + (y1 - y0) * t;   // a gentle curve, steeper at the start
          b.setAttribute("cx", x.toFixed(1)); b.setAttribute("cy", y.toFixed(1)); b.setAttribute("r", R[j].toFixed(1));
        });
      });
    }
    const rims = [null, null]; let shapes = null;
    layout();
    if ("ResizeObserver" in window) new ResizeObserver(layout).observe(card);

    // ---------------------------------------------------------------- the frame becomes the two rims
    // On the way in (0.5 to 1.45 s), as the trails rise, the card's own frame line (the random walk, cardFrames)
    // becomes two lines that move out to where the two rims are, taking on their scallops and going from ink to the
    // two players' colours; on the way back (1.25 to 2.1 s) they come in again, lose their scallops and meet in the one
    // ink line. No bubble ever lies across a line: the last two bubbles (0.8 and 1.15 s in) come only once the lines
    // have moved out past them, every trail ends inside the inner (blue) rim, and on the way back the lines wait
    // outside until those two bubbles have shrunk away (they go first, 0.59 and 0.8 s). The frame and each rim are resampled to the same number of points at equal steps of length, each from
    // its top-left-most point and clockwise, and the points move straight between the two; a frame is drawn only while
    // the lines move. Either way turns back from wherever the other got to.
    const M = 720, reduce = window.matchMedia("(prefers-reduced-motion: reduce)");
    function resample(loop) {
      const n = loop.length, L = [0];
      let s0 = 0; for (let k = 1; k < n; ++k) if (loop[k][0] + loop[k][1] < loop[s0][0] + loop[s0][1]) s0 = k;
      const P = Array.from({ length: n + 1 }, (_, k) => loop[(s0 + k) % n]);
      for (let k = 1; k <= n; ++k) L.push(L[k - 1] + Math.hypot(P[k][0] - P[k - 1][0], P[k][1] - P[k - 1][1]));
      const out = []; let j = 0;
      for (let m = 0; m < M; ++m) { const d = (L[n] * m) / M; while (L[j + 1] < d) ++j; const u = (d - L[j]) / Math.max(1e-9, L[j + 1] - L[j]);
        out.push([P[j][0] + (P[j + 1][0] - P[j][0]) * u, P[j][1] + (P[j + 1][1] - P[j][1]) * u]); }
      return out;
    }
    const lines = [mk("acc"), mk("warm")];
    function mk(c) { const e = document.createElementNS("http://www.w3.org/2000/svg", "path"); e.setAttribute("class", "morph " + c); rim.appendChild(e); return e; }
    let u = 0, goal = 0, raf = 0, wait = 0, last = 0, dur = 0.8;
    const ease = (x) => x * x * (3 - 2 * x);
    function draw() {
      if (!shapes) {
        if (!card.nsLoop || !rims[0] || !rims[1]) return false;
        const F = resample(card.nsLoop.map(([x, y]) => [x + PAD, y + PAD]));          // the frame, in the rims' box
        shapes = { F, R: rims.map(resample) };
      }
      const e = ease(u), frame = card.querySelector(".cframe .path");
      card.classList.toggle("morphing", u > 0);
      if (u <= 0) { lines.forEach((l) => l.removeAttribute("d")); return true; }
      const fo = frame ? +getComputedStyle(frame).opacity || 0.6 : 0.6, a0 = 1 - Math.sqrt(1 - fo);   // two copies over each other look like the one line
      const fw = frame ? parseFloat(getComputedStyle(frame).strokeWidth) || 1.25 : 1.25;
      const rw = parseFloat(getComputedStyle(rim.querySelector(".scallop")).strokeWidth) || 1.7;
      lines.forEach((l, i) => {
        const R = shapes.R[i], F = shapes.F;
        l.setAttribute("d", "M" + F.map(([x, y], k) => (x + (R[k][0] - x) * e).toFixed(1) + "," + (y + (R[k][1] - y) * e).toFixed(1)).join(" L") + " Z");
        l.style.stroke = `color-mix(in srgb, var(${i ? "--warm" : "--accent"}) ${(100 * e).toFixed(1)}%, var(--ink))`;
        l.style.strokeWidth = (fw + (rw - fw) * e).toFixed(2);
        l.style.opacity = (a0 + (1 - a0) * e).toFixed(3);
      });
      return true;
    }
    function tick(now) {
      const dt = Math.min(0.05, (now - last) / 1000); last = now;
      u = goal > u ? Math.min(goal, u + dt / dur) : Math.max(goal, u - dt / dur);
      draw();
      raf = u === goal ? 0 : requestAnimationFrame(tick);
    }
    morph = (on, head) => {                     // on: the way in; head: how far along the new way the picture already is
      clearTimeout(wait); cancelAnimationFrame(raf); raf = 0;
      goal = on ? 1 : 0; dur = on ? 0.95 : 0.85;
      if (reduce.matches) { u = goal; draw(); return; }
      const delay = Math.max(0, (on ? 0.5 : 1.25) - head);
      wait = setTimeout(() => { last = performance.now(); raf = requestAnimationFrame(tick); }, delay * 1000);
    };
    card.addEventListener("nsloop", () => { shapes = null; if (u > 0) draw(); });
    card.classList.add("nsmorph");

    // ---------------------------------------------------------------- the card thinks
    // Every 3 to 5 s a new shock lands at the right end of the row, the row moves over by one (the oldest leaves at
    // the left), and each head's bars move with it and take the new signal in: E[w_u | y_i,0..t] for the 14 latest
    // shocks, in each head from its own noisy signal (which carries the other's action). The states are the
    // generator's own model run on past the picture (js/card-nsheads-think.json, from scratch nsheads think/seq.py);
    // the first is the picture itself. It goes on during the hover too: each shock is one group (its ink tick and the
    // blue and warm ticks it splits into on hover, which always sit at the current estimates in the two heads), and
    // every bar and every shock keeps its element as it moves, so a step is one transition of transforms for all of
    // them (no per-frame script). It waits while the card is off screen or the tab is hidden, and does not run at all
    // under reduced motion.
    function thinking() {
      const ov = shot.querySelector(".ov"), think = ov && ov.querySelector(".think");
      if (!think || !shot.dataset.think || window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
      const K = 19, kept = (g) => [2, 8, 12].includes(g % 15);        // which shocks stay as ink on hover (by identity)
      const heads = [...think.querySelectorAll(".tk")].map((g) => ({ x0: +g.dataset.x0, sl: +g.dataset.sl, y: +g.dataset.y,
        i: g.classList.contains("h1") ? 1 : 0, slots: [...g.querySelectorAll("rect")] }));
      const sg = think.querySelector(".shocks"), row = { x0: +sg.dataset.x0, sl: +sg.dataset.sl, y: +sg.dataset.y,
        slots: [...sg.querySelectorAll(".sh")].map((g) => ({ g, ink: g.querySelector(".ink"), pb: g.querySelector(".pb"), pw: g.querySelector(".pw") })) };
      let D = null, s = 0, timer = 0, seen = false, loading = false;
      const scale = (v) => (v >= 0 ? 1 : -1) * Math.max(Math.abs(v) * K, 1);
      const place = (r, x, y, v) => { r.style.transform = `translate(${x.toFixed(3)}px, ${y.toFixed(3)}px) scale(1, ${v === null ? 0 : scale(v).toFixed(3)})`; };
      const sy = (el, v) => { el.style.transform = `scale(1, ${v === null ? 0 : scale(v).toFixed(3)})`; };
      // one shock's group at slot u of state t: its ink tick at the truth, its pair at the two heads' estimates (the
      // pair's --k takes it back to the truth where the hover's split starts)
      function shock(o, u, t, x) {
        const gi = t + u, w = D.w[gi], ms = [D.m[t][0][u], D.m[t][1][u]];
        o.g.style.transform = `translate(${x.toFixed(3)}px, ${row.y}px)`; o.g.style.setProperty("--d", (0.02 * Math.max(0, u)).toFixed(2) + "s");
        sy(o.ink, w);
        [o.pb, o.pw].forEach((p, i) => { sy(p, ms[i]); p.firstChild.style.setProperty("--k", ((w * K) / scale(ms[i])).toFixed(3)); });
      }
      function identity(o, gi) {             // set once, when a shock lands: kept or not, and how far its pair steps aside
        const k = kept(gi); o.g.classList.toggle("kept", k);
        o.pb.firstChild.style.setProperty("--sx", (k ? -4.9 : -3) + "px"); o.pw.firstChild.style.setProperty("--sx", (k ? 4.9 : 3) + "px");
      }
      function step() {
        const wrap = s + 1 >= D.m.length, t = wrap ? 0 : s + 1;
        if (wrap) {                           // the end of the run (about half an hour): back to the picture, in place
          think.classList.add("snap");
          heads.forEach((R) => { const inc = R.slots[14]; inc.style.opacity = "0"; place(inc, R.x0 + R.sl * 13.5, R.y, null); });
          const inc = row.slots[14]; inc.g.style.opacity = "0"; inc.g.style.transform = `translate(${(row.x0 + row.sl * 13.5).toFixed(3)}px, ${row.y}px)`;
          inc.g.classList.add("in"); sy(inc.ink, null); sy(inc.pb, null); sy(inc.pw, null); identity(inc, t + 13);
          think.getBoundingClientRect(); think.classList.remove("snap"); inc.g.classList.remove("in"); think.getBoundingClientRect();
          s = 0;
          heads.forEach((R) => R.slots.forEach((r, u) => { if (u < 14) { r.style.opacity = ""; place(r, R.x0 + R.sl * (u + 0.5), R.y, D.m[0][R.i][u]); } }));
          row.slots.forEach((o, u) => { if (u < 14) { o.g.style.opacity = ""; identity(o, u); shock(o, u, 0, row.x0 + row.sl * (u + 0.5)); } });
          return;
        }
        // One step is one slide for everything, in every row at once. First the newcomers (the elements the last step's
        // leavers were, faded out by now) are parked unseen one slot beyond the right end, already at the heights they
        // land with; only they skip their transitions for that (.in), so nothing else is interrupted
        const incs = heads.map((R) => R.slots[14]), inc = row.slots[14];
        heads.forEach((R, j) => { const b = incs[j]; b.classList.add("in"); b.style.opacity = "0"; place(b, R.x0 + R.sl * 14.5, R.y, D.m[t][R.i][13]); });
        inc.g.classList.add("in"); inc.g.style.opacity = "0"; identity(inc, t + 13); shock(inc, 13, t, row.x0 + row.sl * 14.5);
        think.getBoundingClientRect(); incs.forEach((b) => b.classList.remove("in")); inc.g.classList.remove("in"); think.getBoundingClientRect();
        s = t;
        // then every row moves over by one slot together, each bar and each shock keeping its element, its height
        // easing to the revised estimate; the oldest slides on past the left end as it fades out, and the newcomer
        // slides into the last slot as it fades in (the same duration and easing for all, home.css)
        heads.forEach((R) => {
          const [out, ...stay] = R.slots.slice(0, 14), inb = R.slots[14];
          place(out, R.x0 - R.sl * 0.5, R.y, D.m[s - 1][R.i][0]); out.style.opacity = "0";
          stay.forEach((r, u) => place(r, R.x0 + R.sl * (u + 0.5), R.y, D.m[s][R.i][u]));
          inb.style.opacity = ""; place(inb, R.x0 + R.sl * 13.5, R.y, D.m[s][R.i][13]);
          R.slots = [...stay, inb, out];
        });
        const [out, ...stay] = row.slots.slice(0, 14);
        out.g.style.transform = `translate(${(row.x0 - row.sl * 0.5).toFixed(3)}px, ${row.y}px)`; out.g.style.opacity = "0";
        stay.forEach((o, u) => shock(o, u, s, row.x0 + row.sl * (u + 0.5)));
        inc.g.style.opacity = ""; shock(inc, 13, s, row.x0 + row.sl * 13.5);
        row.slots = [...stay, inc, out];
      }
      // It takes turns with the mesh behind the name (heroink.js announces the mesh's phases): while that canvas is on
      // screen, a shock lands only in the mesh's still hold, one or two per hold, each done sliding before the mesh
      // fades out, and the card stays put while the mesh grows or fades. Off screen, on a page without the mesh, or
      // while the card is hovered, tapped or focused, it keeps its own time.
      const ink = document.getElementById("heroink");
      let meshOn = false, turn = 0;                        // turn: the shocks landed in this hold so far
      // a touch screen keeps :hover on whatever was tapped last, so hover only counts where there is a real pointer
      const engaged = () => {
        if (canHover.matches && card.matches(":hover")) return true;
        try { return card.matches(".on, :has(:focus-visible)"); } catch (e) { return card.matches(".on"); }
      };
      const taking = () => meshOn && !!window.heroinkPhase && !engaged();
      if (ink && "IntersectionObserver" in window) new IntersectionObserver((es) => { meshOn = es[es.length - 1].isIntersecting; if (D) loop(); }).observe(ink);
      document.addEventListener("heroink:phase", (ev) => { if (!ev.detail.still) turn = 0; if (D) loop(); });
      for (const type of ["pointerenter", "pointerleave", "focusin", "focusout"]) card.addEventListener(type, () => { if (D) loop(); });
      new MutationObserver(() => { if (D) loop(); }).observe(card, { attributes: true, attributeFilter: ["class"] });
      // While the card is hovered, tapped or focused, once the entrance has landed and held still a moment to be read
      // (4.5 s in), the row drifts on as a conveyor: steps back to back, each as long as the interval and linear
      // (.flow), so every bar and shock glides left at one steady speed, one slot every FLOW s, the newest fading in at
      // the right as the oldest fades out at the left. When the hover ends mid-glide, that glide eases out over what
      // is left of it (.settle) instead of stopping dead, and the calm stepping takes over again.
      const FLOW = 2.0, READY = 4.5;
      let flowing = false, onSince = 0, wasOn = false, stepEnd = 0;
      const els = () => [...think.querySelectorAll(".tk rect, .shocks .sh, .shocks .sh > g")];
      function settle() {
        const left = stepEnd - performance.now();
        think.classList.remove("flow"); flowing = false;
        if (left < 60) return;
        const E = els(), fin = E.map((e) => [e.style.transform, e.style.opacity]), now = E.map((e) => { const cs = getComputedStyle(e); return [cs.transform, cs.opacity]; });
        think.classList.add("snap"); E.forEach((e, k) => { e.style.transform = now[k][0] === "none" ? "" : now[k][0]; e.style.opacity = now[k][1]; });
        think.getBoundingClientRect(); think.classList.remove("snap");
        const dur = Math.min(2 * left, 700);             // brief: once the hover ends the drift should stop, not coast on
        think.style.setProperty("--settle", (dur / 1000).toFixed(3) + "s"); think.classList.add("settle"); think.getBoundingClientRect();
        E.forEach((e, k) => { e.style.transform = fin[k][0]; e.style.opacity = fin[k][1]; });
        setTimeout(() => think.classList.remove("settle"), dur + 50);
      }
      function flowStep() {
        if (!engaged()) { settle(); return loop(); }
        if (seen && !document.hidden) { step(); stepEnd = performance.now() + FLOW * 1000; }
        timer = setTimeout(flowStep, FLOW * 1000);
      }
      function loop() {
        const on = engaged();
        if (on && !wasOn) onSince = performance.now();
        wasOn = on;
        if (on && flowing) return;                       // the conveyor keeps its own time
        clearTimeout(timer);
        if (!on && flowing) settle();
        if (on) {
          const head = parseFloat(getComputedStyle(card).getPropertyValue("--nsin")) || 0;
          timer = setTimeout(() => { if (!engaged()) return loop(); flowing = true; think.classList.add("flow"); flowStep(); },
            Math.max(0, onSince + (READY - head) * 1000 - performance.now()));
          return;
        }
        if (taking()) {
          const m = window.heroinkPhase;
          if (!m.still) return;                            // the mesh is moving: wait for its next hold
          const wait = turn ? 2000 + 1000 * Math.random() : 300;
          if (performance.now() + wait + 1300 > m.ends) return;   // it would still be sliding when the fade begins
          timer = setTimeout(() => { if (!taking()) return loop(); if (seen && !document.hidden) { step(); ++turn; } loop(); }, wait);
          return;
        }
        timer = setTimeout(() => { if (taking()) return loop(); if (engaged()) return loop(); if (seen && !document.hidden) step(); loop(); }, 3000 + 2000 * Math.random());
      }
      function load() {
        if (D || loading) return; loading = true;
        fetch(shot.dataset.think).then((r) => r.json()).then((d) => { D = d; loop(); }).catch(() => {});
      }
      if ("IntersectionObserver" in window) new IntersectionObserver((es) => { seen = es.some((e) => e.isIntersecting); if (seen) load(); }).observe(card);
      else { seen = true; load(); }
      document.addEventListener("visibilitychange", () => { if (!document.hidden && D) loop(); });
    }
  })();
})();
