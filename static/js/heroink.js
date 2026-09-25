// The ink behind the name: a decision mesh drawing itself on a smooth field, edges only, faint.
// Same engine as the /mesh page (static/mesh/decision-mesh.js); here it is ornament, so it runs
// slowly, brightens each new cut for a moment, and starts over on a new surface when it fills in.
//
// Toys on top. The edges near the pointer brighten, like a lens. A click or tap raises a sharp bump in the data
// under the mesh on screen and adds a cloud of samples around it; the mesh then refines only where the bump is,
// by the engine's own steps, so it stays one mesh (every split cuts the triangles on both sides of its edge) and
// nothing else in the picture moves. A mouse drag across empty background (not across text, which keeps its
// selection) raises a ridge along the path; the mesh takes it up from the start of the path to its end, and a
// dotted trace of the path stays. Pressing and holding keeps collecting samples under the pointer, so the mesh
// gets finer there the longer you hold (moving while holding paints). Cuts made by pokes keep a good shape (no
// sliver past 4 to 1) and stop at about 14 pixels, and shorter edges are drawn lighter, so a poked patch reads as
// finer texture rather than a tangle. A poked picture lasts a minute; a double-click starts a new one.
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

  let mesh, seed = (Date.now() / 1000) | 0, drawn = new Map(), phase = "grow", fade = 1, rng, field, target = TARGET;
  // after a poke: where it landed (a disk, or a tube along a drag that is uncovered from its start), the steps left
  // to spend there, and the bumps raised so far (new samples are drawn from the field with all of them added)
  let pxPerUnit = 150;                              // screen pixels per unit of the field, set on every draw
  let region = null, budget = 0, spare = 0, pokes = 0, lastPoke = 0, pokeId = 1, bumps = [];
  // squared distance from (x, y) to a polyline
  function ridgeD2(pts, x, y) {
    let best = Infinity;
    for (let i = 1; i < pts.length; ++i) {
      const a = pts[i - 1], b = pts[i], dx = b.x - a.x, dy = b.y - a.y, L = dx * dx + dy * dy;
      const u = L > 0 ? Math.min(1, Math.max(0, ((x - a.x) * dx + (y - a.y) * dy) / L)) : 0, ex = a.x + u * dx - x, ey = a.y + u * dy - y;
      best = Math.min(best, ex * ex + ey * ey);
    }
    return best;
  }
  const d2to = (R, x, y) => (R.pts ? ridgeD2(R.pts, x, y) : (x - R.x) ** 2 + (y - R.y) ** 2);
  const valueAt = (x, y) => { let v = field(x, y); for (const b of bumps) v += b.h * Math.exp(-d2to(b.R, x, y) / b.w); return v; };
  function hit(x, y, R) {
    if (!R.pts) return { in: (x - R.x) ** 2 + (y - R.y) ** 2 < R.r * R.r, idx: 0 };
    let best = Infinity, at = 0; const P = R.pts;
    for (let i = 1; i < P.length; ++i) {
      const a = P[i - 1], b = P[i], dx = b.x - a.x, dy = b.y - a.y, L = dx * dx + dy * dy;
      const u = L > 0 ? Math.min(1, Math.max(0, ((x - a.x) * dx + (y - a.y) * dy) / L)) : 0, ex = a.x + u * dx - x, ey = a.y + u * dy - y;
      if (ex * ex + ey * ey < best) { best = ex * ex + ey * ey; at = i - 1 + u; }
    }
    return { in: best < R.r * R.r, idx: at / Math.max(1, P.length - 1) };
  }
  const front = (now) => (region && region.pts ? Math.min(1, (now - region.t0) / region.dur) : 1);

  // a new surface: a fresh field, fresh samples, the mesh grown from its start
  function start() {
    seed = (seed * 1103515245 + 12345) >>> 0;
    rng = mulberry32(seed);
    field = FIELDS[seed % FIELDS.length]; target = TARGET;
    region = null; budget = 0; pokes = 0; bumps = []; trace = null;
    const X = new Float64Array(2 * N), Y = new Float64Array(N);
    for (let i = 0; i < N; ++i) {
      const x = LO + (HI - LO) * rng(), y = LO + (HI - LO) * rng();
      X[2 * i] = x; X[2 * i + 1] = y; Y[i] = (nothing ? 0 : field(x, y)) + 0.45 * gauss(rng);
    }
    if (nothing) { mesh = startingMesh(); drawn = new Map(); phase = "grow"; fade = 1; return; }
    DM.reset();
    mesh = new DM.DecisionMesh(X, Y, { maxAspectRatio: 4, minPoints: 4, refresh: true, rng: mulberry32(seed + 1) });
    for (let i = 0; i < PREFILL * 3 && mesh.activeFaces.size < PREFILL; ++i) if (mesh.step(0.08) === "none") break;
    drawn = new Map();
    phase = "grow"; fade = 1;
  }

  // ---- pokes change the data under the mesh on screen
  function inFace(f, x, y) {
    const [a, b, c] = f.vertices;
    const d1 = (x - b.x) * (a.y - b.y) - (a.x - b.x) * (y - b.y), d2 = (x - c.x) * (b.y - c.y) - (b.x - c.x) * (y - c.y), d3 = (x - a.x) * (c.y - a.y) - (c.x - a.x) * (y - a.y);
    return !((d1 < -1e-12 || d2 < -1e-12 || d3 < -1e-12) && (d1 > 1e-12 || d2 > 1e-12 || d3 > 1e-12));
  }
  const cat = (a, extra) => { const o = new Int32Array(a.length + extra.length); o.set(a); o.set(extra, a.length); return o; };
  // new samples: each goes into the active triangle that holds it and into the two halves that triangle has ready
  // for each of its possible splits (where the engine keeps its points); returns the triangles that gained any
  function addData(pts) {
    const n0 = mesh.n, n1 = n0 + pts.length, X = new Float64Array(2 * n1), Y = new Float64Array(n1);
    X.set(mesh.X); Y.set(mesh.Y);
    // find each sample's triangle through a coarse grid of the square (each triangle listed in the cells its box
    // covers), not by trying every triangle
    const G = 24, cell = (HI - LO) / G, grid = new Map(), adds = new Map();
    const cix = (v) => Math.min(G - 1, Math.max(0, Math.floor((v - LO) / cell)));
    let bx0 = HI, bx1 = LO, by0 = HI, by1 = LO;
    for (const [x, y] of pts) { bx0 = Math.min(bx0, x); bx1 = Math.max(bx1, x); by0 = Math.min(by0, y); by1 = Math.max(by1, y); }
    for (const f of mesh.activeFaces) {
      const xs = f.vertices.map((v) => v.x), ys = f.vertices.map((v) => v.y);
      const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
      if (x1 < bx0 || x0 > bx1 || y1 < by0 || y0 > by1) continue;          // nowhere near the new samples
      for (let a = cix(x0); a <= cix(x1); ++a) for (let b = cix(y0); b <= cix(y1); ++b) { const k = a * G + b; if (!grid.has(k)) grid.set(k, []); grid.get(k).push(f); }
    }
    pts.forEach(([x, y, v], k) => {
      const i = n0 + k; X[2 * i] = x; X[2 * i + 1] = y; Y[i] = v;
      const f = (grid.get(cix(x) * G + cix(y)) || []).find((g) => inFace(g, x, y));
      if (f) { if (!adds.has(f)) adds.set(f, []); adds.get(f).push(i); }
    });
    mesh.X = X; mesh.Y = Y; mesh.n = n1;
    for (const [f, list] of adds) {
      f.idx = cat(f.idx, list); f._W = null;
      f.edges.forEach((e, j) => {
        const sd = f.subdiv[j];
        if (!sd.e) return;
        const up = [], dn = [];
        for (const i of list) (X[2 * i] * sd.e.nx + X[2 * i + 1] * sd.e.ny >= sd.e.intercept ? up : dn).push(i);
        sd["+"].idx = cat(sd["+"].idx, up); sd["+"]._W = null;
        sd["-"].idx = cat(sd["-"].idx, dn); sd["-"]._W = null;
        // a split turned down for want of points may have them now
        if (e.disqualifying.has(f) && Math.max(sd["+"].aspectRatio(), sd["-"].aspectRatio()) < mesh.maxAspectRatio &&
            Math.min(sd["+"].idx.length, sd["-"].idx.length) >= mesh.minPoints) {
          e.disqualifying.delete(f);
          if (!e.disqualifying.size) e.midpoint.disqualified = false;
        }
      });
    }
    return adds.keys();
  }
  // the engine reads the data afresh whenever it scores a split or a refit: rescore everything the change touches
  function rescore(faces) {
    const vs = new Set();
    for (const f of faces) {
      for (const v of f.vertices) vs.add(v);
      for (const e of f.edges) if (e.midpoint && !e.midpoint.active) vs.add(e.midpoint);
    }
    for (const v of vs) v.updateInfo();
  }
  // a bump (R a disk) or a ridge (R a path): raise the data already there, add samples round it, and spend the
  // next steps inside R only
  function poke(R, h, w, count, sd) {
    const r = mulberry32(++pokeId * 7919), X = mesh.X, Y = mesh.Y, touched = new Set();
    // the bump is raised out to where it is still a fair fraction of the noise (3 w), and only triangles whose box
    // comes that close are looked at
    const reach = Math.sqrt(3 * w), P = R.pts || [R];
    const bx0 = Math.min(...P.map((q) => q.x)) - reach, bx1 = Math.max(...P.map((q) => q.x)) + reach;
    const by0 = Math.min(...P.map((q) => q.y)) - reach, by1 = Math.max(...P.map((q) => q.y)) + reach;
    for (const f of mesh.activeFaces) {
      const [a, b, c] = f.vertices;
      if (Math.max(a.x, b.x, c.x) < bx0 || Math.min(a.x, b.x, c.x) > bx1 || Math.max(a.y, b.y, c.y) < by0 || Math.min(a.y, b.y, c.y) > by1) continue;
      let any = false;
      for (const i of f.idx) { const d2 = d2to(R, X[2 * i], X[2 * i + 1]); if (d2 < 3 * w) { Y[i] += h * Math.exp(-d2 / w); any = true; } }
      if (any) touched.add(f);
    }
    bumps.push({ R, h, w });
    const pts = [];
    for (let j = 0; j < count; ++j) {
      let cx = R.x, cy = R.y;
      if (R.pts) { const k = Math.floor(r() * (R.pts.length - 1)), a = R.pts[k], b = R.pts[k + 1], u = r(); cx = a.x + u * (b.x - a.x); cy = a.y + u * (b.y - a.y); }
      const x = Math.min(HI, Math.max(LO, cx + sd * gauss(r))), y = Math.min(HI, Math.max(LO, cy + sd * gauss(r)));
      pts.push([x, y, valueAt(x, y) + 0.45 * gauss(r)]);
    }
    for (const f of addData(pts)) touched.add(f);
    rescore(touched);
    // the budget counts cuts (refits, which move no edge, draw on a separate allowance)
    region = R; budget = R.pts ? 180 : 70; spare = 3000; ++pokes; lastPoke = performance.now();
    phase = "grow"; fade = 1;
    // the first cuts land at once, so the click is answered on the spot (a drag is taken up along its path instead)
    if (!R.pts) for (let k = 0, guard = 0; k < 12 && guard < 200; ++guard) { const did = localStep(lastPoke); if (!did) break; if (did === "split") { ++k; --budget; } }
  }
  // pressing and holding: more samples under the pointer (no new bump), and a few more cuts there, again and again
  function collect(x, y) {
    if (mesh.n > 40000) return;
    const r = mulberry32(++pokeId * 7919), pts = [];
    for (let j = 0; j < 140; ++j) {
      const px_ = Math.min(HI, Math.max(LO, x + 0.3 * gauss(r))), py_ = Math.min(HI, Math.max(LO, y + 0.3 * gauss(r)));
      pts.push([px_, py_, valueAt(px_, py_) + 0.45 * gauss(r)]);
    }
    rescore(addData(pts));
    region = { x, y, r: 0.55 }; budget = Math.max(budget, 0) + 10; spare = 3000; lastPoke = performance.now();
    phase = "grow"; fade = 1;
  }
  // a cut is taken only if no new triangle is thinner than 4 to 1 and no new edge is shorter than about 14 pixels
  function shapely(v) {
    if (v.active) return true;
    for (const f of v.getSimFaces()) {
      if (!f) continue;
      if (f.aspectRatio() > 4) return false;
      for (const e of f.edges) if (e.length * pxPerUnit < 14) return false;
    }
    return true;
  }
  // one step of the engine, taken from the best candidates inside the poked region (the part uncovered so far)
  function localStep(now) {
    let best = null, bp = -1e-12;
    for (const [v, p] of mesh.heap) if (p < bp) { const m = hit(v.x, v.y, region); if (m.in && m.idx <= front(now) && shapely(v)) { bp = p; best = v; } }
    if (!best) return null;
    if (best.active) { best.updateHeight(); return "refit"; }
    best.activate(); return "split";
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
    // wide: faint behind the text block (the column's left 55%), full strength everywhere else, out to both edges
    // of the window; the edges of the quiet block are soft, so lines fade in rather than stop
    const ramp = (v) => Math.min(1, Math.max(0, v));
    const quietAt = (x) => x < col.c0 ? ramp(1 - (col.c0 - x) / (0.07 * cw)) : ramp(1 - (x - (col.c0 + 0.55 * cw)) / (0.14 * cw));
    pxPerUnit = G.side / (HI - LO) / dpr;
    // the formula in the hero keeps a quiet patch behind it, like the text block
    const mo = document.querySelector(".hero .motif"), cr = cv.getBoundingClientRect();
    const mr = mo ? (() => { const b = mo.getBoundingClientRect(); return { l: (b.left - cr.left - 24) * dpr, r: (b.right - cr.left + 24) * dpr, t: (b.top - cr.top - 16) * dpr, b: (b.bottom - cr.top + 16) * dpr }; })() : null;
    const lensR = 150 * dpr, lensOn = pointer && now - lensAt < 2500 ? 1 - Math.max(0, (now - lensAt - 1500) / 1000) : 0;
    const edge = (ax, ay, bx, by, t) => {
      const x0 = px(ax), y0 = py(ay), x1 = px(bx), y1 = py(by);
      const mx = (x0 + x1) / 2, my = (y0 + y1) / 2;
      const d = col.wide ? quietAt(mx) * 0.78 : Math.hypot(mx - fx, (my - fy) * 0.85) / R;
      // the lens reaches past the vignette, so the quiet corner under the name wakes up too
      let lens = 0;
      if (lensOn) { const q = Math.hypot(mx - pointer.x, my - pointer.y) / lensR; if (q < 1) lens = lensOn * (1 - q) * (1 - q); }
      if (d >= 1 && lens === 0) return;
      // and a fade into the top and bottom of the band, so no line stops on the canvas edge
      const band = Math.min(1, Math.min(my, H - my) / (0.3 * H));
      let v = Math.max(0, (1 - Math.min(1, d)) ** 2) * band * band;
      if (mr && mx > mr.l && mx < mr.r && my > mr.t && my < mr.b) v *= 0.3;
      // shorter edges are drawn lighter, so a finely cut patch keeps the tone of the rest (as pencil gets lighter
      // where it is dense) instead of going dark
      const len = Math.hypot(x1 - x0, y1 - y0) / dpr;
      v *= Math.min(1, 0.4 + len / 50);                   // full strength from about 30 px up: only fine cuts lighten
      const age = (now - t) / 2000;                       // a new cut glows, then settles
      const fresh = age < 1 ? 1 - age : 0;
      const a = (dark ? 0.38 : 0.44) * fade * v + (dark ? 0.5 : 0.42) * lens * band;
      ctx.strokeStyle = fresh > 0.02 || lens > 0.05
        ? `rgba(${glow},${a + 0.55 * Math.max(fresh * v, lens * 0.6) * (dark ? 0.32 : 0.28)})`
        : `rgba(${base},${a})`;
      ctx.beginPath();
      ctx.moveTo(x0, y0);
      ctx.lineTo(x1, y1);
      ctx.stroke();
    };
    for (const e of mesh.activeEdges) {
      let t = drawn.get(e);
      if (t === undefined) { t = now; drawn.set(e, t); }
      edge(e.v0.x, e.v0.y, e.v1.x, e.v1.y, t);
    }
    // the ridge being drawn; after the release it dims to a faint trace that stays while the picture does
    if (stroke && !stroke.holding && stroke.pts.length > 1) {
      ctx.strokeStyle = `rgba(${glow},0.45)`; ctx.lineWidth = 1.5 * dpr;
      ctx.beginPath(); stroke.pts.forEach((q, i) => (i ? ctx.lineTo(q.cx, q.cy) : ctx.moveTo(q.cx, q.cy))); ctx.stroke();
    }
    for (const tr of trace || []) {
      // solid as drawn, then dotted, so the path reads as yours and not as one more edge of the mesh
      const k = Math.min(1, (now - tr.t) / 1200), al = (0.45 - 0.05 * k) * fade;
      ctx.strokeStyle = `rgba(${glow},${al})`; ctx.lineWidth = (1.5 + 0.5 * k) * dpr;
      ctx.setLineDash(k < 1 ? [] : [0.1, 5 * dpr]);
      ctx.beginPath(); tr.pts.forEach((q, i) => (i ? ctx.lineTo(px(q.x), py(q.y)) : ctx.moveTo(px(q.x), py(q.y)))); ctx.stroke();
      ctx.setLineDash([]);
    }
    if (nothing) drawRejections(now, G, dark);
    if (stroke && stroke.holding && holdRing) {        // a hold: a small ring breathing under the pointer
      const k = 0.5 + 0.5 * Math.sin((now - holdRing.t) / 90);
      ctx.strokeStyle = `rgba(${glow},${0.35 + 0.2 * k})`; ctx.lineWidth = 1.3 * dpr;
      ctx.beginPath(); ctx.arc(holdRing.x, holdRing.y, (10 + 4 * k) * dpr, 0, 2 * Math.PI); ctx.stroke();
    }
    if (ripple && now - ripple.t < 1100) {             // where the poke landed
      const age = (now - ripple.t) / 1100;
      ctx.strokeStyle = `rgba(${glow},${0.55 * (1 - age)})`;
      ctx.lineWidth = 1.5 * dpr;
      ctx.beginPath(); ctx.arc(ripple.x, ripple.y, (8 + 70 * age) * dpr, 0, 2 * Math.PI); ctx.stroke();
    }
  }
  let ripple = null, stroke = null, trace = null, holdRing = null;   // trace: the ridges drawn into this picture

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
  // the mesh runs past the column to the window's edges, so it listens wherever it is drawn, not only on the hero
  const overMesh = (ev) => {
    const r = cv.getBoundingClientRect();
    if (ev.clientX < r.left || ev.clientX > r.right || ev.clientY < r.top || ev.clientY > r.bottom) return false;
    return !(ev.target.closest && ev.target.closest("a, button, input, select, textarea, header, nav, .motif"));
  };
  document.addEventListener("pointermove", (ev) => {
    const on = overMesh(ev);
    document.documentElement.classList.toggle("over-mesh", on && !nothing);
    if (on && ev.pointerType === "mouse") { pointer = toCanvas(ev); lensAt = performance.now(); }
    else if (!on) pointer = null;
  }, { passive: true });
  document.addEventListener("pointerleave", () => { pointer = null; });
  let dragged = false;
  if (!nothing) document.addEventListener("click", (ev) => {
    if (dragged) { dragged = false; return; }
    if (!overMesh(ev)) return;
    const p = toCanvas(ev), G = geometry(), x = G.ux(p.x), y = G.uy(p.y);
    if (x < LO || x > HI || y < LO || y > HI) return;
    const now = performance.now();
    pointer = p; lensAt = now; ripple = { x: p.x, y: p.y, t: now };
    poke({ x, y, r: 1.05 }, pokes % 2 ? -3.2 : 3.2, 0.3, 750, 0.55);
  });
  // is the pointer on a line of text (not merely inside a text block's box, which runs the column's width)?
  function overText(ev) {
    const el = ev.target.closest && ev.target.closest("p, h1, h2, h3, li, .katex");
    if (!el) return false;
    const r = document.createRange(); r.selectNodeContents(el);
    for (const b of r.getClientRects()) if (ev.clientX >= b.left - 4 && ev.clientX <= b.right + 4 && ev.clientY >= b.top - 2 && ev.clientY <= b.bottom + 2) return true;
    return false;
  }
  // a mouse drag across empty background lays a ridge along the path (text keeps its own drag, selection)
  if (!nothing) {
    document.addEventListener("pointerdown", (ev) => {
      if (ev.pointerType !== "mouse" || ev.button !== 0 || !overMesh(ev) || overText(ev)) return;
      ev.preventDefault();                                 // a press on the background starts no text selection
      const p = toCanvas(ev), G = geometry();
      stroke = { pts: [{ cx: p.x, cy: p.y, x: G.ux(p.x), y: G.uy(p.y) }], len: 0, at: performance.now(), hold: null };
      // held still for a moment, the press becomes a hold: samples collect under the pointer every quarter second
      const s0 = stroke;
      setTimeout(() => {
        if (stroke !== s0 || s0.len > 10) return;
        const tick = () => {
          if (stroke !== s0) return;
          const q = s0.pts[s0.pts.length - 1];
          if (q.x > LO && q.x < HI && q.y > LO && q.y < HI) { collect(q.x, q.y); holdRing = { x: q.cx, y: q.cy, t: performance.now() }; }
          s0.hold = setTimeout(tick, 250);
        };
        s0.holding = true; ++pokes; tick();
      }, 350);
    });
    document.addEventListener("pointermove", (ev) => {
      if (!stroke) return;
      const p = toCanvas(ev), G = geometry(), last = stroke.pts[stroke.pts.length - 1], d = Math.hypot(p.x - last.cx, p.y - last.cy);
      if (d < 4) return;
      stroke.len += d; stroke.pts.push({ cx: p.x, cy: p.y, x: G.ux(p.x), y: G.uy(p.y) });
      if (stroke.holding) { stroke.pts = [stroke.pts[0], stroke.pts[stroke.pts.length - 1]]; stroke.len = 0; return; }   // holding and moving paints
      if (stroke.len > 12) { ev.preventDefault(); window.getSelection && window.getSelection().removeAllRanges(); }
    });
    document.addEventListener("pointerup", () => {
      if (!stroke) return;
      const s = stroke; stroke = null;
      if (s.hold) clearTimeout(s.hold);
      if (s.holding) { dragged = true; setTimeout(() => { dragged = false; }, 0); return; }   // a hold is not also a click
      if (s.len < 24) return;                              // a click, handled above
      dragged = true; setTimeout(() => { dragged = false; }, 0);
      // one ridge along the path, thinned to a point every 0.25 units of the field
      const pts = [];
      for (const q of s.pts) {
        if (q.x < LO || q.x > HI || q.y < LO || q.y > HI) continue;
        const last = pts[pts.length - 1];
        if (!last || Math.hypot(q.x - last.x, q.y - last.y) >= 0.25) pts.push({ x: q.x, y: q.y });
      }
      if (pts.length < 2) return;
      const path = pts.slice(0, 80), now = performance.now();
      (trace = trace || []).push({ pts: path, t: now });
      ripple = null;
      poke({ pts: path, r: 0.55, t0: now + 150, dur: Math.min(3600, 1000 + 110 * path.length) }, 2.8, 0.12, 1400, 0.35);
    });
  }

  // a double-click (off the text) lets the picture go and starts a new surface
  if (!nothing) document.addEventListener("dblclick", (ev) => {
    if (!overMesh(ev) || overText(ev)) return;
    phase = "out"; acc = 0;
  });

  // unfolding the phone draws a ridge down the middle of the field, where the fold was, for the mesh to find
  if (!nothing) window.addEventListener("sitefold", (ev) => {
    if (ev.detail.kind !== "unfold") return;
    const G = geometry(), r = cv.getBoundingClientRect(), dpr = cv.width / Math.max(1, r.width), cx = G.ux((innerWidth / 2 - r.left) * dpr);
    if (!(cx > LO && cx < HI)) return;
    const path = [-3.6, -2.4, -1.2, 0, 1.2, 2.4, 3.6].map((y) => ({ x: cx, y }));
    poke({ pts: path, r: 0.55, t0: performance.now(), dur: 1600 }, 3, 0.12, 1400, 0.35);
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
        // before any poke the whole mesh grows, slowly; after one, only the poked region does, and faster
        const every = region ? 30 : 120;
        while (acc > every) {
          acc -= every;
          if (region) {
            let k = region.pts ? 2 : 1;
            // refits move no edge and cost little: take them as they come (at most 40 a frame), count only the cuts
            let refits = 40;
            while (k > 0 && budget > 0) {
              const did = localStep(now);
              if (did === "split") { --budget; --k; }
              else if (did === "refit") { if (--spare <= 0) budget = 0; if (--refits <= 0) break; }
              else { if (front(now) >= 1) budget = 0; break; }
            }
            if (budget <= 0) { phase = "hold"; acc = 0; break; }
          } else if (mesh.activeFaces.size >= target || mesh.step(0.08) === "none") { phase = "hold"; acc = 0; break; }
        }
        draw(now);
      } else if (phase === "hold") {
        acc += dt;
        if (pokes ? now - lastPoke > 60000 : acc > 6500) { phase = "out"; acc = 0; }
        // once the last cut's glow has faded there is nothing new to draw, unless the lens is moving
        if (acc < 2200 || lensy || stroke || (trace && trace.some((tr) => now - tr.t < 1300))) draw(now);
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
