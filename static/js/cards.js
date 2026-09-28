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
      const s = (isDark() ? ["#2b6cf0", "#5b8ff9", "#2a2b30", "#f0845c", "#f5c451"] : ["#1d4ed8", "#6d9cf5", "#f7f6ee", "#f29a58", "#d9591c"]).map(hex);
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
    shot.addEventListener("pointerup", () => { if (down) card.classList.toggle("on"); down = null; });
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
      for (let k = 1; k <= n; ++k) {
        const [x, y] = pts[k % n], [px, py] = pts[k - 1], c = Math.hypot(x - px, y - py), rr = c * (0.56 + 0.1 * r());
        d += ` A${rr.toFixed(1)},${rr.toFixed(1)} 0 0 1 ${x.toFixed(1)},${y.toFixed(1)}`;   // clockwise arcs bulge outward
      }
      return d + " Z";
    }
    function layout() {
      const W = card.offsetWidth + 2 * PAD, H = card.offsetHeight + 2 * PAD;
      if (!W || !H) return;
      rim.setAttribute("viewBox", `0 0 ${W} ${H}`);
      const [blue, warm] = rim.querySelectorAll(".scallop");
      blue.setAttribute("d", scallops(W, H, 7, 30, 71));
      warm.setAttribute("d", scallops(W, H, 3, 34, 73));
      // a trail of four bubbles from each head up to that player's own rim, growing as it rises and leaning a little
      // toward the middle: it starts just off the forehead (a point of the picture, 480 x 240 units, drawn by the same
      // prototype as the images) and its last bubble sits against the rim's inner edge. The picture is cropped to the
      // card ("slice") and zooms by 1.04 on hover, so the start is mapped through both.
      const sw = shot.offsetWidth, sh = shot.offsetHeight, k = Math.max(sw / 480, sh / 240), Z = 1.04;
      const pic = ([u, v]) => [PAD + shot.offsetLeft + sw / 2 + Z * (u - 240) * k, PAD + shot.offsetTop + sh / 2 + Z * (v - 120) * k];
      const R = [2, 2.9, 3.8, 4.8].map((r) => r * Math.min(1, Math.max(0.8, k)));
      [["acc", [128, 45], 7], ["warm", [352, 45], 3]].forEach(([c, at, inset]) => {
        const [x0, y0] = pic(at), y1 = inset + R[3] + 1.5, lean = (at[0] < 240 ? 1 : -1) * 0.32 * (y0 - y1);
        rim.querySelectorAll(".tail." + c).forEach((b, j) => {
          const t = [0, 0.3, 0.62, 1][j], x = x0 + lean * t * t, y = y0 + (y1 - y0) * t;   // a gentle curve, steeper at the start
          b.setAttribute("cx", x.toFixed(1)); b.setAttribute("cy", y.toFixed(1)); b.setAttribute("r", R[j].toFixed(1));
        });
      });
    }
    layout();
    if ("ResizeObserver" in window) new ResizeObserver(layout).observe(card);

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
        // first, in every row at once, the next bar and the next shock wait hidden and flat where they will land (no
        // transition for that; turning transitions off is for the whole layer, so it happens before anything moves)
        think.classList.add("snap");
        heads.forEach((R) => { const inc = R.slots[14]; inc.style.opacity = "0"; place(inc, R.x0 + R.sl * 13.5, R.y, null); });
        const inc = row.slots[14]; inc.g.style.opacity = "0"; inc.g.style.transform = `translate(${(row.x0 + row.sl * 13.5).toFixed(3)}px, ${row.y}px)`;
        sy(inc.ink, null); sy(inc.pb, null); sy(inc.pw, null); identity(inc, t + 13);
        think.getBoundingClientRect(); think.classList.remove("snap"); think.getBoundingClientRect();
        if (wrap) {                           // the end of the run (about half an hour): back to the picture, in place
          s = 0;
          heads.forEach((R) => R.slots.forEach((r, u) => { if (u < 14) { r.style.opacity = ""; place(r, R.x0 + R.sl * (u + 0.5), R.y, D.m[0][R.i][u]); } }));
          row.slots.forEach((o, u) => { if (u < 14) { o.g.style.opacity = ""; identity(o, u); shock(o, u, 0, row.x0 + row.sl * (u + 0.5)); } });
          return;
        }
        s = t;
        // then every row moves over by one slot together, each bar and each shock keeping its element, its height
        // easing to the revised estimate; the oldest slides out at the left and fades, the new one grows in
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
      function loop() {
        clearTimeout(timer);
        timer = setTimeout(() => { if (seen && !document.hidden) step(); loop(); }, 3000 + 2000 * Math.random());
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
