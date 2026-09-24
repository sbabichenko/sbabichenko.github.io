// The ink behind the name: a decision mesh drawing itself on a smooth field, edges only, faint.
// Same engine as the /mesh page (static/mesh/decision-mesh.js); here it is ornament, so it runs
// slowly, brightens each new cut for a moment, and starts over on a new surface when it fills in.
(function () {
  "use strict";
  const cv = document.getElementById("heroink");
  if (!cv || typeof DM === "undefined") return;
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
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

  let mesh, seed = (Date.now() / 1000) | 0, drawn = new Map(), phase = "grow", fade = 1, rng;
  function start() {
    seed = (seed * 1103515245 + 12345) >>> 0;
    rng = mulberry32(seed);
    const f = FIELDS[seed % FIELDS.length];
    const X = new Float64Array(2 * N), Y = new Float64Array(N);
    for (let i = 0; i < N; ++i) {
      const x = LO + (HI - LO) * rng(), y = LO + (HI - LO) * rng();
      X[2 * i] = x; X[2 * i + 1] = y; Y[i] = f(x, y) + 0.45 * gauss(rng);
    }
    DM.reset();
    mesh = new DM.DecisionMesh(X, Y, { maxAspectRatio: 5, minPoints: 4, refresh: true, rng: mulberry32(seed + 1) });
    for (let i = 0; i < PREFILL * 3 && mesh.activeFaces.size < PREFILL; ++i) if (mesh.step(0.08) === "none") break;
    drawn = new Map();
    phase = "grow"; fade = 1;
  }

  function fit() {
    const r = cv.getBoundingClientRect(), dpr = Math.min(2, window.devicePixelRatio || 1);
    const w = Math.max(1, Math.round(r.width * dpr)), h = Math.max(1, Math.round(r.height * dpr));
    if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
    return dpr;
  }

  // the square is drawn taller than wide and pushed right, so the name sits over its quiet corner
  function draw(now) {
    const dpr = fit(), W = cv.width, H = cv.height;
    ctx.clearRect(0, 0, W, H);
    // drawn far larger than the hero, so only an interior patch shows and no boundary reads as a frame
    const side = H * 1.9;
    const ox = W - side * 0.92, oy = (H - side) / 2;
    const px = (x) => ox + ((x - LO) / (HI - LO)) * side;
    const py = (y) => oy + ((HI - y) / (HI - LO)) * side;
    const dark = document.documentElement.classList.contains("dark");
    const base = dark ? "255,255,255" : "20,22,40";
    const glow = dark ? "150,180,255" : "31,63,208";
    ctx.lineWidth = Math.max(0.6, 0.8 * dpr);
    ctx.lineCap = "round";
    // the vignette is per edge, not a CSS mask: a clipped mask cuts lines off square, this fades them
    const fx = W * 0.84, fy = H * 0.5, R = 0.95 * Math.max(W * 0.55, H);
    for (const e of mesh.activeEdges) {
      let t = drawn.get(e);
      if (t === undefined) { t = now; drawn.set(e, t); }
      const x0 = px(e.v0.x), y0 = py(e.v0.y), x1 = px(e.v1.x), y1 = py(e.v1.y);
      const mx = (x0 + x1) / 2, my = (y0 + y1) / 2;
      const d = Math.hypot(mx - fx, (my - fy) * 0.85) / R;
      if (d >= 1) continue;
      // and a fade into the top and bottom of the band, so no line stops on the canvas edge
      const band = Math.min(1, Math.min(my, H - my) / (0.3 * H));
      const v = (1 - d) * (1 - d) * band * band;
      const age = (now - t) / 1400;                       // a new cut glows, then settles
      const fresh = age < 1 ? 1 - age : 0;
      const a = (dark ? 0.38 : 0.34) * fade * v;
      ctx.strokeStyle = fresh > 0.02
        ? `rgba(${glow},${a + 0.55 * fresh * v * (dark ? 0.32 : 0.28)})`
        : `rgba(${base},${a})`;
      ctx.beginPath();
      ctx.moveTo(x0, y0);
      ctx.lineTo(x1, y1);
      ctx.stroke();
    }
  }

  if (reduced) {                                          // one still figure, no motion
    start();
    for (let i = 0; i < 700 && mesh.activeFaces.size < TARGET; ++i) if (mesh.step(0.08) === "none") break;
    const once = () => draw(performance.now() - 1e6);
    once(); window.addEventListener("resize", once);
    new MutationObserver(once).observe(document.documentElement, { attributes: true, attributeFilter: ["class"] });
    return;
  }

  let acc = 0, settled = 0;
  function frame(now) {
    // about thirty frames a second, only while the figure is on screen and something is moving
    if (!document.hidden && now - settled > 30 && cv.getBoundingClientRect().bottom > 0) {
      const dt = Math.min(120, settled ? now - settled : 32);
      settled = now;
      if (phase === "grow") {
        acc += dt;
        while (acc > 120) {                               // about eight cuts a second
          acc -= 120;
          if (mesh.activeFaces.size >= TARGET || mesh.step(0.08) === "none") { phase = "hold"; acc = 0; break; }
        }
      } else if (phase === "hold") {
        acc += dt;
        if (acc > 6500) { phase = "out"; acc = 0; }
      } else {
        acc += dt;
        fade = Math.max(0, 1 - acc / 1600);
        if (fade <= 0) start();
      }
      // once the last cut's glow has faded there is nothing new to draw until the figure moves again
      if (phase !== "hold" || acc < 1600) draw(now);
    }
    requestAnimationFrame(frame);
  }
  start();
  requestAnimationFrame(frame);
})();
