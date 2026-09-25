// The ink behind the name: a decision mesh drawing itself on a smooth field, edges only, faint.
// Same engine as the /mesh page (static/mesh/decision-mesh.js); here it is ornament, so it runs
// slowly, brightens each new cut for a moment, and starts over on a new surface when it fills in.
//
// Two toys on top. The edges near the pointer brighten, like a lens. A click or tap anywhere in the
// hero drops a sharp bump into the field at that spot and grows the mesh again from coarse, so it
// crowds its cuts around wherever you poked it.
//
// On the 404 page (data-mode="nothing") there is no field at all: the starting mesh sits there and
// its candidate cuts light up one at a time and are turned down, since there is nothing to find.
(function () {
  "use strict";
  const cv = document.getElementById("heroink");
  if (!cv || typeof DM === "undefined") return;
  const hero = cv.closest(".hero") || cv.parentElement;
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const nothing = cv.dataset.mode === "nothing";
  const ctx = cv.getContext("2d");
  const LO = -4, HI = 4, N = 6000, TARGET = 820, PREFILL = 340;

  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function gauss(r) { let u = 0; while (u === 0) u = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r()); }

  // smooth fields, each with a different grain, so no two visits draw the same figure
  const FIELDS = [
    (x, y) => 2.2 * Math.tanh(1.3 * (x - 0.7 * y)),
    (x, y) => 2.4 * Math.exp(-((x - 1.5) ** 2 + (y - 1) ** 2) / 2.2) - 2.1 * Math.exp(-((x + 1.5) ** 2 + (y + 1.3) ** 2) / 1.6),
    (x, y) => (x * x - y * y) / 5,
    (x, y) => 1.7 * Math.sin(0.9 * x) * Math.cos(0.7 * y) + 0.3 * x,
    (x, y) => 2.0 * Math.tanh(1.1 * (2.2 - Math.hypot(x, y))),
  ];

  let mesh, seed = (Date.now() / 1000) | 0, drawn = new Map(), phase = "grow", fade = 1, rng, field, pokes = [];
  // a new field on a timer; the same field plus the bumps poked into it on a click
  function start(keepField) {
    seed = (seed * 1103515245 + 12345) >>> 0;
    rng = mulberry32(seed);
    if (!keepField) { field = FIELDS[seed % FIELDS.length]; pokes = []; }
    const f = (x, y) => {
      // once poked, the field is turned down so the bumps are what the mesh has to chase
      let v = nothing ? 0 : field(x, y) * (pokes.length ? 0.3 : 1);
      for (const p of pokes) v += p.h * Math.exp(-((x - p.x) ** 2 + (y - p.y) ** 2) / 0.3);
      return v;
    };
    const X = new Float64Array(2 * N), Y = new Float64Array(N);
    for (let i = 0; i < N; ++i) {
      const x = LO + (HI - LO) * rng(), y = LO + (HI - LO) * rng();
      X[2 * i] = x; X[2 * i + 1] = y; Y[i] = f(x, y) + 0.45 * gauss(rng);
    }
    if (nothing) { mesh = startingMesh(); drawn = new Map(); phase = "grow"; fade = 1; return; }
    DM.reset();
    mesh = new DM.DecisionMesh(X, Y, { maxAspectRatio: 5, minPoints: 4, refresh: true, rng: mulberry32(seed + 1) });
    const prefill = keepField ? 40 : PREFILL;       // after a poke, grow from coarse so the cuts are seen to gather
    if (!nothing) for (let i = 0; i < prefill * 3 && mesh.activeFaces.size < prefill; ++i) if (mesh.step(0.08) === "none") break;
    drawn = new Map();
    phase = "grow"; fade = 1;
  }

  // the engine's starting mesh: the square cut along its diagonal and bisected evenly, 128 right triangles
  function startingMesh() {
    const k = 8, h = (HI - LO) / k, edges = [], V = (i, j) => ({ x: LO + i * h, y: LO + j * h });
    for (let i = 0; i <= k; ++i) for (let j = 0; j <= k; ++j) {
      if (i < k) edges.push({ v0: V(i, j), v1: V(i + 1, j) });
      if (j < k) edges.push({ v0: V(i, j), v1: V(i, j + 1) });
      if (i < k && j < k) edges.push((i + j) % 2 ? { v0: V(i, j), v1: V(i + 1, j + 1) } : { v0: V(i + 1, j), v1: V(i, j + 1) });
    }
    return { activeEdges: edges };
  }

  function fit() {
    const r = cv.getBoundingClientRect(), dpr = Math.min(2, window.devicePixelRatio || 1);
    const w = Math.max(1, Math.round(r.width * dpr)), h = Math.max(1, Math.round(r.height * dpr));
    if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
    return dpr;
  }

  // the square is drawn taller than wide and pushed right, so the name sits over its quiet corner;
  // it is drawn far larger than the hero, so only an interior patch shows and no boundary reads as a frame
  // On a screen wider than the column the canvas runs to both edges of the window (home.css), and so does the
  // mesh: it is drawn wide enough to leave off both sides, with the quiet patch kept under the name.
  function column() {
    const c = cv.getBoundingClientRect(), h = hero.getBoundingClientRect(), k = cv.width / Math.max(1, c.width);
    return { c0: (h.left - c.left) * k, c1: (h.right - c.left) * k, wide: c.width > h.width + 40 };
  }
  function geometry() {
    const W = cv.width, H = cv.height, col = column();
    const side = col.wide ? Math.max(H * 1.9, W * 1.08) : H * 1.9;
    const ox = col.wide ? W - side * 0.96 : W - side * 0.92, oy = (H - side) / 2;
    return { W, H, side, ox, oy, col,
      px: (x) => ox + ((x - LO) / (HI - LO)) * side, py: (y) => oy + ((HI - y) / (HI - LO)) * side,
      ux: (X) => LO + ((X - ox) / side) * (HI - LO), uy: (Y) => HI - ((Y - oy) / side) * (HI - LO) };
  }

  let pointer = null, lensAt = 0;                   // canvas pixels, and when the pointer last moved
  function draw(now) {
    const dpr = fit(), G = geometry(), { W, H, px, py } = G;
    ctx.clearRect(0, 0, W, H);
    const dark = document.documentElement.classList.contains("dark");
    const base = dark ? "255,255,255" : "20,22,40";
    const glow = dark ? "150,180,255" : "31,63,208";
    ctx.lineWidth = Math.max(0.6, 0.8 * dpr);
    ctx.lineCap = "round";
    // the vignette is per edge, not a CSS mask: a clipped mask cuts lines off square, this fades them
    const { col } = G, cw = col.c1 - col.c0;
    const fx = col.wide ? col.c0 + cw * 0.84 : W * 0.84, fy = H * 0.5, R = 0.95 * Math.max(W * 0.55, H);
    // wide: faint under the name, full strength everywhere else, out to both edges of the window
    const nameX = col.c0 + cw * 0.24, quietR = cw * 0.62;
    const lensR = 150 * dpr, lensOn = pointer && now - lensAt < 2500 ? 1 - Math.max(0, (now - lensAt - 1500) / 1000) : 0;
    for (const e of mesh.activeEdges) {
      let t = drawn.get(e);
      if (t === undefined) { t = now; drawn.set(e, t); }
      const x0 = px(e.v0.x), y0 = py(e.v0.y), x1 = px(e.v1.x), y1 = py(e.v1.y);
      const mx = (x0 + x1) / 2, my = (y0 + y1) / 2;
      const d = col.wide
        ? Math.max(0, 1 - Math.hypot(mx - nameX, (my - fy) * 1.4) / quietR) * 0.75
        : Math.hypot(mx - fx, (my - fy) * 0.85) / R;
      // the lens reaches past the vignette, so the quiet corner under the name wakes up too
      let lens = 0;
      if (lensOn) { const q = Math.hypot(mx - pointer.x, my - pointer.y) / lensR; if (q < 1) lens = lensOn * (1 - q) * (1 - q); }
      if (d >= 1 && lens === 0) continue;
      // and a fade into the top and bottom of the band, so no line stops on the canvas edge
      const band = Math.min(1, Math.min(my, H - my) / (0.3 * H));
      const v = Math.max(0, (1 - Math.min(1, d)) ** 2) * band * band;
      const age = (now - t) / 1400;                       // a new cut glows, then settles
      const fresh = age < 1 ? 1 - age : 0;
      const a = (dark ? 0.38 : 0.34) * fade * v + (dark ? 0.5 : 0.42) * lens * band;
      ctx.strokeStyle = fresh > 0.02 || lens > 0.05
        ? `rgba(${glow},${a + 0.55 * Math.max(fresh * v, lens * 0.6) * (dark ? 0.32 : 0.28)})`
        : `rgba(${base},${a})`;
      ctx.beginPath();
      ctx.moveTo(x0, y0);
      ctx.lineTo(x1, y1);
      ctx.stroke();
    }
    if (nothing) drawRejections(now, G, dark);
    if (ripple && now - ripple.t < 1100) {             // where the poke landed
      const age = (now - ripple.t) / 1100;
      ctx.strokeStyle = `rgba(${glow},${0.55 * (1 - age)})`;
      ctx.lineWidth = 1.5 * dpr;
      ctx.beginPath(); ctx.arc(ripple.x, ripple.y, (8 + 70 * age) * dpr, 0, 2 * Math.PI); ctx.stroke();
    }
  }
  let ripple = null;

  // 404: candidate midpoints light up and are turned down, one after another
  const rejections = [];
  function drawRejections(now, G, dark) {
    const edges = [...mesh.activeEdges];
    if (!rejections.length || now - rejections[rejections.length - 1].t > 300) {
      const e = edges[Math.floor(Math.random() * edges.length)];
      if (e) rejections.push({ x: (e.v0.x + e.v1.x) / 2, y: (e.v0.y + e.v1.y) / 2, t: now });
    }
    while (rejections.length && now - rejections[0].t > 1800) rejections.shift();
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    for (const r of rejections) {
      const age = (now - r.t) / 1800, X = G.px(r.x), Y = G.py(r.y);
      if (X < 0 || X > G.W || Y < 0 || Y > G.H) continue;
      const a = age < 0.25 ? age / 0.25 : 1 - (age - 0.25) / 0.75, s = 6 * dpr;
      ctx.strokeStyle = dark ? `rgba(245,196,81,${0.8 * a})` : `rgba(185,28,28,${0.7 * a})`;
      ctx.lineWidth = 1.4 * dpr;
      ctx.beginPath(); ctx.arc(X, Y, s * (1 + 0.6 * age), 0, 2 * Math.PI); ctx.stroke();
      if (age > 0.3) { ctx.beginPath(); ctx.moveTo(X - s * 0.6, Y - s * 0.6); ctx.lineTo(X + s * 0.6, Y + s * 0.6); ctx.moveTo(X + s * 0.6, Y - s * 0.6); ctx.lineTo(X - s * 0.6, Y + s * 0.6); ctx.stroke(); }
    }
  }

  if (reduced) {                                          // one still figure, no motion
    start();
    if (!nothing) for (let i = 0; i < 700 && mesh.activeFaces.size < TARGET; ++i) if (mesh.step(0.08) === "none") break;
    const once = () => draw(performance.now() - 1e6);
    once(); window.addEventListener("resize", once);
    new MutationObserver(once).observe(document.documentElement, { attributes: true, attributeFilter: ["class"] });
    return;
  }

  // the toys: listen on the hero, not the canvas, which sits under the text
  const toCanvas = (ev) => {
    const r = cv.getBoundingClientRect(), dpr = cv.width / Math.max(1, r.width);
    return { x: (ev.clientX - r.left) * dpr, y: (ev.clientY - r.top) * dpr };
  };
  hero.addEventListener("pointermove", (ev) => { if (ev.pointerType === "mouse") { pointer = toCanvas(ev); lensAt = performance.now(); } });
  hero.addEventListener("pointerleave", () => { pointer = null; });
  if (!nothing) hero.addEventListener("click", (ev) => {
    if (ev.target.closest("a, button, input, select, textarea")) return;
    const p = toCanvas(ev), G = geometry(), x = G.ux(p.x), y = G.uy(p.y);
    if (x < LO || x > HI || y < LO || y > HI) return;
    pokes.push({ x, y, h: pokes.length % 2 ? -3.2 : 3.2 });
    if (pokes.length > 4) pokes.shift();
    pointer = p; lensAt = performance.now(); ripple = { x: p.x, y: p.y, t: performance.now() };
    start(true);
  });

  // unfolding the phone draws a ridge down the middle of the field, where the fold was, for the mesh to find
  if (!nothing) window.addEventListener("sitefold", (ev) => {
    if (ev.detail.kind !== "unfold") return;
    const G = geometry(), r = cv.getBoundingClientRect(), dpr = cv.width / Math.max(1, r.width), cx = G.ux((innerWidth / 2 - r.left) * dpr);
    if (!(cx > LO && cx < HI)) return;
    pokes.length = 0;
    for (const y of [-2.4, -0.8, 0.8, 2.4]) pokes.push({ x: cx, y, h: 3 });
    start(true);
  });

  let acc = 0, settled = 0;
  function frame(now) {
    // about thirty frames a second, only while the figure is on screen and something is moving
    if (!document.hidden && now - settled > 30 && cv.getBoundingClientRect().bottom > 0) {
      const dt = Math.min(120, settled ? now - settled : 32);
      settled = now;
      const lensy = pointer && now - lensAt < 2600;
      if (nothing) {
        draw(now);
      } else if (phase === "grow") {
        acc += dt;
        const every = pokes.length ? 45 : 120;            // a poked mesh grows faster, to be seen reacting
        while (acc > every) {
          acc -= every;
          if (mesh.activeFaces.size >= TARGET || mesh.step(0.08) === "none") { phase = "hold"; acc = 0; break; }
        }
        draw(now);
      } else if (phase === "hold") {
        acc += dt;
        if (acc > (pokes.length ? 12000 : 6500)) { phase = "out"; acc = 0; }
        // once the last cut's glow has faded there is nothing new to draw, unless the lens is moving
        if (acc < 1600 || lensy) draw(now);
      } else {
        acc += dt;
        fade = Math.max(0, 1 - acc / 1600);
        if (fade <= 0) start();
        draw(now);
      }
    }
    requestAnimationFrame(frame);
  }
  start();
  requestAnimationFrame(frame);
})();
