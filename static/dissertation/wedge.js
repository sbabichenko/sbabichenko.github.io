// /dissertation/wedge: the information wedge, in steps. Four diagrams, then two drawings solved live by the
// explorer's solver (static/noisestate/worker.js) on the Chapter 1 tracking game: player 1's first-order condition split into
// its physical part and the wedge; and the tug of war, where both players' precision p sets how hard they push.
(function () {
  "use strict";
  const NS = "http://www.w3.org/2000/svg";
  const svg = document.getElementById("stage");
  const steps = [...document.querySelectorAll(".step")];
  const page = document.getElementById("wedgepage");
  if (!svg || !steps.length || !page) return;
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const el = (tag, attrs, parent) => { const n = document.createElementNS(NS, tag); for (const [k, v] of Object.entries(attrs || {})) n.setAttribute(k, v); if (parent) parent.appendChild(n); return n; };
  function mulberry32(a) { return function () { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
  const clamp = (x, a = 0, b = 1) => Math.max(a, Math.min(b, x));
  const ease = (t) => (t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2);
  const seg = (t, a, b) => ease(clamp((t - a) / (b - a)));
  const fade = (n, t) => { n.style.opacity = clamp(t); };
  function pencil(pts, seed, wob = 0.6) {
    const r = mulberry32(seed);
    let d = `M${pts[0][0].toFixed(1)},${pts[0][1].toFixed(1)}`;
    for (let i = 1; i < pts.length; ++i) { const [x0, y0] = pts[i - 1], [x1, y1] = pts[i]; d += ` Q${((x0 + x1) / 2 + (r() - 0.5) * wob * 2).toFixed(1)},${((y0 + y1) / 2 + (r() - 0.5) * wob * 2).toFixed(1)} ${x1.toFixed(1)},${y1.toFixed(1)}`; }
    return d;
  }
  function stroke(g, d, cls, width = 1.6) {
    const a = el("path", { d, class: "pencil " + (cls || ""), "stroke-width": width }, g);
    const len = a.getTotalLength() || 1;
    a.style.strokeDasharray = `${len} ${len}`; a.style.strokeDashoffset = len;
    return { a, set(t) { a.style.strokeDashoffset = len * (1 - clamp(t)); } };
  }
  function text(g, x, y, s, cls, anchor = "middle") { const t = el("text", { x, y, class: cls || "", "text-anchor": anchor }, g); t.textContent = s; return t; }
  function box(g, x, y, label, w) { const b = el("g", {}, g); const W = w || label.length * 9 + 28; el("rect", { x: x - W / 2, y: y - 19, width: W, height: 38, rx: 6, class: "box pencil", "stroke-width": 1.6 }, b); text(b, x, y + 6, label); return b; }
  function arrowHead(g, x, y, ang, cls) { return el("path", { d: `M${x - 10 * Math.cos(ang - 0.45)},${y - 10 * Math.sin(ang - 0.45)} L${x},${y} L${x - 10 * Math.cos(ang + 0.45)},${y - 10 * Math.sin(ang + 0.45)}`, class: "pencil " + (cls || ""), "stroke-width": 1.8 }, g); }
  const scenes = {};
  const group = (name) => { const g = el("g", { "data-scene": name }, svg); g.style.opacity = 0; g.style.transition = "opacity 0.6s"; return g; };

  // ------------------------------------------------------------------ the solver, run in the background
  const MODEL = {"name":"ch1_tracking_with_targets","params":{"p1":9,"p2":9,"r1":0.1,"r2":0.1,"b1":1,"b2":-1,"sigma":1,"T":1},"channels":["w0","w1","w2"],"states":{"X":{"drift":{"D1":1,"D2":1},"noise":{"w0":"sigma"}}},"agents":{"player1":{"controls":["D1"],"signals":{"y1":{"drift":{"X":"sqrt(p1)"},"noise":{"w1":1}}},"loss":[[1,"X","X"],["-2*b1","X"],["r1","D1","D1"]]},"player2":{"controls":["D2"],"signals":{"y2":{"drift":{"X":"sqrt(p2)"},"noise":{"w2":1}}},"loss":[[1,"X","X"],["-2*b2","X"],["r2","D2","D2"]]}},"horizon":{"kind":"finite","T":"T"},"numerics":{"nodes":12}};
  const model = (params) => { const m = JSON.parse(JSON.stringify(MODEL)); Object.assign(m.params, params); return m; };
  // the tug of war: both players at precision p. The default, p = 9, goes first: it also feeds the split drawing.
  const TUG = [9, 0.01, 1000, 1, 100, 3, 30, 0.3, 0.1];
  const R = { base: null, tug: {} };
  const status = document.getElementById("solvestate");
  const jobs = TUG.map((p) => ({ p, params: { p1: p, p2: p } }));
  let done = 0, t0 = performance.now();
  const maxAbs = (a) => a.reduce((m, v) => Math.max(m, Math.abs(v)), 0);
  function onResult(job, res) {
    done++;
    if (res.ok) {
      const f = res.samples.foc.D1, ch = Object.values(f.channels);
      if (job.p === 9) R.base = f;
      // the wedge's share of player 1's first-order condition: its largest size over the physical part's, across shocks
      const wedge = Math.max(...ch.map((d) => maxAbs(d.wedge))) / (Math.max(...ch.map((d) => maxAbs(d.physical))) || 1);
      R.tug[job.p] = { t: res.samples.mean_t, D1: res.samples.means.D1, D2: res.samples.means.D2, X: res.samples.means.X,
        J1: res.costs.player1, J2: res.costs.player2, wedge };
    }
    for (const w of waits) w.textContent = `solving… ${done} of ${jobs.length}`;
    if (status) {
      status.textContent = done < jobs.length ? `Solving: ${done} of ${jobs.length} equilibria…` : `Solved ${jobs.length} equilibria in your browser, in ${((performance.now() - t0) / 1000).toFixed(1)} s.`;
      status.classList.toggle("ok", done === jobs.length);
    }
    if (window.siteTally) window.siteTally("solve", 1, done === jobs.length ? `${jobs.length} equilibria for the information wedge` : "");
  }
  try {
    const worker = new Worker(page.dataset.worker);
    let i = 0;
    const next = () => { if (i < jobs.length) worker.postMessage({ type: "solve", id: i, model: model(jobs[i].params), request: {} }); };
    worker.onmessage = (ev) => {
      const m = ev.data;
      if (m.type === "ready") { t0 = performance.now(); next(); }
      else if (m.type === "result") { let res; try { res = JSON.parse(m.result); } catch (e) { res = { ok: false }; } onResult(jobs[m.id], res); i++; next(); }
      else if (m.type === "fatal" && status) status.textContent = "The solver could not start in this browser: " + m.message;
    };
  } catch (e) { if (status) status.textContent = "This browser cannot run the solver."; }

  // ------------------------------------------------------------------ 1. alone: estimate, then act
  scenes.alone = (() => {
    const g = group("alone");
    const me = el("g", {}, g); el("circle", { cx: 130, cy: 300, r: 34, class: "box pencil", "stroke-width": 1.8 }, me); text(me, 130, 307, "you");
    const fog = el("g", {}, g), r = mulberry32(4);
    for (let k = 0; k < 26; ++k) el("circle", { cx: 300 + (r() - 0.5) * 90, cy: 300 + (r() - 0.5) * 150, r: 12 + r() * 16, class: "fillacc", opacity: 0.05 }, fog);
    const st = box(g, 470, 300, "state x");
    const est = box(g, 300, 180, "estimate x̂");
    const sees = stroke(g, pencil([[440, 280], [330, 200]], 1, 1), "soft", 1.4);
    const uses = stroke(g, pencil([[270, 190], [150, 272]], 2, 1), "accent", 1.8);
    const acts = stroke(g, pencil([[165, 312], [300, 360], [435, 318]], 3, 1), "", 2);
    const h = arrowHead(g, 435, 318, -0.3, "");
    const l1 = text(g, 400, 222, "learn", "mono"), l2 = text(g, 150, 150, "act on x̂ as if it were x", "mono acc"), l3 = text(g, 300, 395, "push", "mono");
    const l4 = text(g, 300, 428, "what you don't know about x", "mono acc");
    const cap = text(g, 300, 480, "learning and acting, separately", "label");
    return { g, update(t) {
      fade(me, seg(t, 0, 0.1)); fade(st, seg(t, 0, 0.1)); fade(fog, seg(t, 0.05, 0.2)); fade(l4, seg(t, 0.1, 0.2));
      sees.set(seg(t, 0.15, 0.3)); fade(l1, seg(t, 0.2, 0.3)); fade(est, seg(t, 0.25, 0.35));
      uses.set(seg(t, 0.35, 0.5)); fade(l2, seg(t, 0.4, 0.5)); acts.set(seg(t, 0.5, 0.65)); fade(h, seg(t, 0.62, 0.66)); fade(l3, seg(t, 0.55, 0.65));
      fade(cap, seg(t, 0.7, 0.85));
    } };
  })();

  // ------------------------------------------------------------------ 2. your push is also a signal: a pulse round the loop
  scenes.signal = (() => {
    const g = group("signal");
    const P = { p1: [110, 300], X: [300, 120], y2: [490, 300], b2: [300, 480] };
    const n1 = box(g, P.p1[0], P.p1[1], "player 1"), nx = box(g, P.X[0], P.X[1], "state X"), ny = box(g, P.y2[0], P.y2[1], "2's signal"), nb = box(g, P.b2[0], P.b2[1], "2's forecast");
    const legs = [["p1", "X"], ["X", "y2"], ["y2", "b2"], ["b2", "X"]];
    const paths = legs.map(([a, b], k) => {
      const [x0, y0] = P[a], [x1, y1] = P[b];
      const mx = (x0 + x1) / 2 + (k === 3 ? -40 : 0), my = (y0 + y1) / 2 + (k === 3 ? 0 : 0);
      const d = k === 3 ? pencil([[x0 - 10, y0 - 22], [mx + 10, (y0 + y1) / 2 + 30], [x1 - 20, y1 + 22]], 10 + k, 1) : pencil([[x0 + (x1 > x0 ? 40 : -40), y0 + (y1 > y0 ? 20 : -20)], [x1 - (x1 > x0 ? 40 : -40), y1 - (y1 > y0 ? 22 : -22)]], 10 + k, 1);
      return stroke(g, d, k === 3 ? "warm" : "", 1.8);
    });
    const labs = [text(g, 170, 190, "you push", "mono"), text(g, 440, 190, "they see it", "mono"), text(g, 440, 410, "they revise", "mono"), text(g, 284, 322, "they push back", "mono warmt", "start")];
    const pulse = el("circle", { r: 8, class: "fillacc" }, g);
    const cap = text(g, 300, 570, "your action does something and says something", "label");
    return { g, update(t, now) {
      [n1, nx, ny, nb].forEach((n, i) => fade(n, seg(t, i * 0.05, 0.1 + i * 0.05)));
      paths.forEach((p, i) => p.set(seg(t, 0.15 + i * 0.12, 0.27 + i * 0.12))); labs.forEach((l, i) => fade(l, seg(t, 0.2 + i * 0.12, 0.3 + i * 0.12)));
      const on = t > 0.65 && !reduced;
      fade(pulse, on ? 1 : 0);
      if (on) { const u = (now / 3200) % 1, k = Math.floor(u * 4), v = u * 4 - k, p = paths[k].a, L = p.getTotalLength(), q = p.getPointAtLength(v * L); pulse.setAttribute("cx", q.x); pulse.setAttribute("cy", q.y); }
      fade(cap, seg(t, 0.7, 0.85));
    } };
  })();

  // ------------------------------------------------------------------ 3. the shadow price of the state
  scenes.stateprice = (() => {
    // top: the expected future cost against the state, a valley. The slope where the player stands is the shadow price H;
    // a push moves the state a little and the cost falls by about the slope times the push.
    // bottom: the saving |H| x push against the effort cost (1/2) G push^2; the best push is where the gap is widest,
    // D = -H / G, the formula beside the drawing (the player acts on its estimate of H)
    const g = group("stateprice");
    const C = (x) => 300 - (0.0022 * (x - 360) ** 2 + 20), dC = (x) => 0.0044 * (x - 360);   // screen y; cost slope
    const axes = el("g", {}, g);
    el("line", { x1: 80, y1: 300, x2: 520, y2: 300, class: "pencil soft", "stroke-width": 1 }, axes);
    el("line", { x1: 80, y1: 300, x2: 80, y2: 92, class: "pencil soft", "stroke-width": 1 }, axes);
    text(axes, 520, 318, "state", "mono", "end"); text(axes, 88, 88, "expected future cost", "mono", "start");
    const pts = []; for (let x = 84; x <= 520; x += 8) pts.push([x, C(x)]);
    const curve = stroke(g, pencil(pts, 5, 0.3), "", 2);
    const x0 = 170, x1 = 230, y0 = C(x0), y1 = C(x1), m = -dC(x0);          // screen slope of the tangent
    const tan = el("line", { x1: x0 - 70, y1: y0 - 70 * m, x2: x0 + 70, y2: y0 + 70 * m, class: "pencil accent", "stroke-width": 2 }, g);
    const lab = text(g, x0 - 58, y0 - 70 * m - 26, "slope: the shadow price H", "label acc", "start");
    const push = el("g", {}, g);
    const pa = stroke(push, pencil([[x0, y0 - 30], [x1, y0 - 30]], 7, 0.3), "warm", 2.2);
    const ph = arrowHead(push, x1, y0 - 30, 0, "warm");
    const pl = text(push, (x0 + x1) / 2, y0 - 40, "a push", "label warmt");
    const drop = el("g", {}, g);
    el("line", { x1: x1 + 14, y1: y0, x2: x1 + 14, y2: y1, class: "pencil", "stroke-width": 1.6, "stroke-dasharray": "3 3" }, drop);
    el("line", { x1: x0 + 6, y1: y0, x2: x1 + 20, y2: y0, class: "pencil soft", "stroke-width": 1, "stroke-dasharray": "2 4" }, drop);
    text(drop, x1 + 22, (y0 + y1) / 2 + 5, "the future cost falls", "label", "start");
    const ball = el("circle", { r: 10, class: "fillacc" }, g);
    // bottom panel: push size u in [0, 400] px; saving u/2, effort u^2/800, widest gap at u = 200
    const B = el("g", {}, g), X = (u) => 100 + u, Y = (v) => 560 - v;
    el("line", { x1: X(0), y1: Y(0), x2: X(410), y2: Y(0), class: "pencil soft", "stroke-width": 1 }, B);
    text(B, X(410), Y(0) + 18, "push", "mono", "end");
    const save = stroke(B, pencil([[X(0), Y(0)], [X(400), Y(200)]], 11, 0.2), "accent", 2);
    const eff = []; for (let u = 0; u <= 400; u += 10) eff.push([X(u), Y(u * u / 800)]);
    const effort = stroke(B, pencil(eff, 12, 0.2), "warm", 2);
    const bl = [text(B, X(250), Y(125) - 12, "saving: |H| × push", "label acc", "end"), text(B, X(395), Y(196) + 30, "effort: ½ G × push²", "label warmt", "end")];
    const best = el("g", {}, B);
    el("line", { x1: X(200), y1: Y(0), x2: X(200), y2: Y(100), class: "pencil", "stroke-width": 1.4, "stroke-dasharray": "3 4" }, best);
    el("circle", { cx: X(200), cy: Y(0), r: 4.5, class: "fillacc" }, best);
    text(best, X(200), Y(0) + 20, "best push", "label");
    return { g, update(t) {
      fade(axes, seg(t, 0, 0.1)); curve.set(seg(t, 0, 0.22));
      fade(tan, seg(t, 0.2, 0.3)); fade(lab, seg(t, 0.24, 0.32));
      pa.set(seg(t, 0.32, 0.42)); fade(ph, seg(t, 0.4, 0.43)); fade(pl, seg(t, 0.34, 0.42));
      const u = seg(t, 0.42, 0.52), bx = x0 + (x1 - x0) * u;
      ball.setAttribute("cx", bx); ball.setAttribute("cy", C(bx) - 10); fade(ball, seg(t, 0.14, 0.2));
      fade(drop, seg(t, 0.5, 0.58));
      fade(B, seg(t, 0.58, 0.64)); save.set(seg(t, 0.6, 0.72)); effort.set(seg(t, 0.64, 0.76));
      bl.forEach((l, k) => fade(l, seg(t, 0.68 + 0.04 * k, 0.76 + 0.04 * k))); fade(best, seg(t, 0.8, 0.9));
    } };
  })();

  // ------------------------------------------------------------------ 4. the backward equation, and the wedge in it
  scenes.wedge = (() => {
    // what the shadow price prices, as two routes out of one push. Across the top, the physical part: the push moves the
    // state, the state moves your cost. Below, the information wedge: the push shows up in player 2's signal, player 2's
    // noise-state moves, player 2 acts on it, and that moves the state again. Pulses run both routes; the backward
    // equation sits underneath with the wedge's term tinted.
    const g = group("wedge");
    const top = el("g", {}, g), TY = 112;
    const nPush = box(top, 100, TY, "your push"), nX = box(top, 300, TY, "state X"), nCost = box(top, 500, TY, "your cost");
    const phys = [stroke(g, pencil([[152, TY], [252, TY]], 31, 0.4), "", 2), stroke(g, pencil([[348, TY], [446, TY]], 32, 0.4), "", 2)];
    const physH = [arrowHead(g, 252, TY, 0, ""), arrowHead(g, 446, TY, 0, "")];
    const physL = text(g, 300, 72, "physical part", "label");
    // the loop through player 2, on an ellipse under the state
    const CX = 300, CY = 236, RX = 158, RY = 110, E = (a) => [CX + RX * Math.cos(a), CY + RY * Math.sin(a)];
    const at = { X: -Math.PI / 2, sig: 0, ns: Math.PI / 2, act: Math.PI };
    const loopN = el("g", {}, g);
    box(loopN, ...E(at.sig), "2's signal"); box(loopN, ...E(at.ns), "2's noise-state"); box(loopN, ...E(at.act), "2's action");
    // each leg leaves and reaches a box clear of it; the noise-state box is the widest, so its gaps are larger
    const legs = [["X", "sig", 0.42, 0.42], ["sig", "ns", 0.42, 0.74], ["ns", "act", 0.74, 0.42], ["act", "X", 0.42, 0.42]].map(([a, b, g0, g1], k) => {
      let a0 = at[a] + g0, a1 = at[b] - g1; if (a1 < a0) a1 += 2 * Math.PI;
      const pts = []; for (let q = 0; q <= 16; ++q) pts.push(E(a0 + (a1 - a0) * q / 16));
      const s_ = stroke(g, pencil(pts, 40 + k, 0.6), "accent", 2);
      const [ex, ey] = pts[16], [px, py] = pts[14];
      return { s: s_, h: arrowHead(g, ex, ey, Math.atan2(ey - py, ex - px), "accent"), a0, a1 };
    });
    g.appendChild(loopN);                                           // the boxes over the lines
    const wl = text(g, CX, CY + RY + 42, "information wedge", "label acc");
    // pulses: black along the top, blue round the loop
    const pb = el("circle", { r: 5, class: "fillacc" }, g); pb.style.fill = "currentColor";
    const pw = el("circle", { r: 5.5, class: "fillacc" }, g);
    // the equation, small, underneath
    const f = el("g", {}, g);
    const put = (key, x, y, W) => {
      const src = window.WEDGE_FX && window.WEDGE_FX[key]; if (!src) return null;
      const doc = new DOMParser().parseFromString(src, "image/svg+xml").documentElement, n = document.importNode(doc, true);
      const vb = n.getAttribute("viewBox").split(" ").map(Number), H = (W * vb[3]) / vb[2];
      n.setAttribute("width", W); n.setAttribute("height", H); n.setAttribute("x", x); n.setAttribute("y", y - H / 2); n.removeAttribute("style"); n.style.color = "var(--ink)";
      f.appendChild(n); return { x, y, W, H };
    };
    const eqL = text(g, 300, 422, "the backward equation for the shadow price", "mono");
    put("backward1", 50, 462, 500);
    const b2 = put("backward2", 80, 548, 440);
    const hl = b2 ? el("rect", { x: b2.x - 12, y: b2.y - b2.H / 2 - 8, width: b2.W + 24, height: b2.H + 16, rx: 10, class: "fillacc" }, g) : null;
    if (hl) g.insertBefore(hl, f);
    return { g, update(t, now) {
      fade(top, seg(t, 0, 0.1));
      phys.forEach((p, k) => p.set(seg(t, 0.06 + 0.06 * k, 0.16 + 0.06 * k))); physH.forEach((h, k) => fade(h, seg(t, 0.15 + 0.06 * k, 0.18 + 0.06 * k)));
      fade(physL, seg(t, 0.14, 0.24)); fade(loopN, seg(t, 0.24, 0.34));
      legs.forEach((l, k) => { l.s.set(seg(t, 0.3 + 0.06 * k, 0.4 + 0.06 * k)); fade(l.h, seg(t, 0.39 + 0.06 * k, 0.42 + 0.06 * k)); });
      fade(wl, seg(t, 0.52, 0.6));
      // the black pulse crosses the top in 2.2 s; the blue one goes round the loop in 4.4 s, pausing at each box
      const u = (now / 2200) % 1, xb = u < 0.5 ? 152 + (252 - 152) * (u / 0.5) : 348 + (446 - 348) * ((u - 0.5) / 0.5);
      pb.setAttribute("cx", xb); pb.setAttribute("cy", TY); fade(pb, seg(t, 0.2, 0.26) * (reduced ? 0 : 1));
      const v = (now / 4400) % 1, k = Math.floor(v * 4), w = Math.min(1, (v * 4 - k) / 0.8), L = legs[k], [wx, wy] = E(L.a0 + (L.a1 - L.a0) * w);
      pw.setAttribute("cx", wx); pw.setAttribute("cy", wy); fade(pw, seg(t, 0.55, 0.62) * (reduced ? 0 : 1));
      fade(eqL, seg(t, 0.64, 0.74)); fade(f, seg(t, 0.66, 0.8)); if (hl) fade(hl, seg(t, 0.8, 0.9) * 0.14);
    } };
  })();

  // ------------------------------------------------------------------ plotting helpers for the solved drawings
  function axes(g, x0, y0, W, H, xlab) {
    el("line", { x1: x0, y1: y0, x2: x0 + W, y2: y0, class: "pencil soft", "stroke-width": 1 }, g);
    if (xlab) text(g, x0 + W / 2, y0 + H + 34, xlab, "mono");
  }
  const waits = [];
  const waiting = (g) => { const t = text(g, 300, 300, "solving…", "mono"); waits.push(t); return t; };

  // ------------------------------------------------------------------ 5. the split, solved
  scenes.split = (() => {
    const g = group("split"), live = el("g", {}, g), wait = waiting(g);
    let drawn = false;
    const NAMES = { w0: "the common shock", w1: "player 1's signal noise", w2: "player 2's signal noise" };
    function draw() {
      const f = R.base; live.innerHTML = "";
      const chans = Object.keys(f.channels);
      // each shock's row on its own scale (the percentage beside it compares the two parts within the row), with its
      // zero line drawn out past the curves and marked, so a wedge lying along zero reads as small, not as missing
      chans.forEach((c, k) => {
        const mx = Math.max(maxAbs(f.channels[c].physical), maxAbs(f.channels[c].wedge)) || 1;
        const top = 90 + k * 160, H = 110, x0 = 60, W = 490, mid = top + H / 2, s = f.s, sx = (v) => x0 + (W * v) / s[s.length - 1], sy = (v) => mid - (v / mx) * (H / 2);
        el("line", { x1: x0 - 14, y1: mid, x2: x0 + W, y2: mid, class: "pencil soft", "stroke-width": 1, "stroke-dasharray": "3 4" }, live);
        text(live, x0 - 20, mid + 4, "0", "mono", "end");
        // labels sit just above the panel: the curves are scaled to fill it, so the largest reaches its top
        text(live, x0, top - 8, NAMES[c] || c, "mono", "start");
        const pm = maxAbs(f.channels[c].physical), wm = maxAbs(f.channels[c].wedge);
        if (pm > 1e-9) text(live, x0 + W, top - 8, `wedge up to ${Math.round((100 * wm) / pm)}% of the physical part`, "mono acc", "end");
        el("path", { d: pencil(s.map((v, i) => [sx(v), sy(f.channels[c].physical[i])]), 20 + k, 0.1), class: "pencil", "stroke-width": 2 }, live);
        el("path", { d: pencil(s.map((v, i) => [sx(v), sy(f.channels[c].wedge[i])]), 30 + k, 0.1), class: "pencil accent", "stroke-width": 2.6 }, live);
      });
      // the legend names each line by a sample of it, not by a color (the physical part is white on the dark theme)
      el("line", { x1: 60, y1: 555, x2: 84, y2: 555, class: "pencil", "stroke-width": 2 }, live);
      text(live, 92, 560, "physical part", "label", "start");
      el("line", { x1: 250, y1: 555, x2: 274, y2: 555, class: "pencil accent", "stroke-width": 2.6 }, live);
      text(live, 282, 560, "the information wedge", "label acc", "start");
      text(live, 300, 588, "by the time s of the shock, at t = 1/2; each row on its own scale", "mono");
      drawn = true;
    }
    return { g, update(t) { if (R.base && !drawn) draw(); fade(wait, R.base ? 0 : 1); fade(live, seg(t, 0.05, 0.25)); } };
  })();

  // ------------------------------------------------------------------ 6. the tug of war: both see more, both push less
  scenes.tug = (() => {
    // the two mean pushes over the game, player 1 up and player 2 down, with the mean state flat at zero between them and
    // the no-information limit (T - t)/r dotted. The slider sets both players' precision p on a log scale; between two
    // solved values of p the curves and numbers are interpolated in log p.
    const g = group("tug"), live = el("g", {}, g), wait = waiting(g);
    const input = document.getElementById("tug-p"), val = document.getElementById("tug-val"), read = document.getElementById("tug-read");
    const X0 = 80, W = 440, Y0 = 285, S = 17.5, r = MODEL.params.r1;      // 17.5 px per unit of push; the open-loop start 10 sits 175 px out
    const sx = (t) => X0 + W * t, sy = (v) => Y0 - S * v;
    const grid = TUG.slice().sort((a, b) => a - b);
    // the fixed parts: axes, the dotted no-information limit on both sides, labels
    const fixed = el("g", {}, live);
    el("line", { x1: X0, y1: sy(11), x2: X0, y2: sy(-11), class: "pencil soft", "stroke-width": 1 }, fixed);
    text(fixed, X0 - 10, sy(10) + 4, "10", "mono", "end"); text(fixed, X0 - 10, sy(-10) + 4, "−10", "mono", "end"); text(fixed, X0 - 10, Y0 + 4, "0", "mono", "end");
    text(fixed, X0, sy(-11) + 22, "t = 0", "mono", "start"); text(fixed, X0 + W, sy(-11) + 22, "t = 1", "mono", "end");
    for (const sg of [1, -1]) {
      const pts = []; for (let k = 0; k <= 20; ++k) pts.push([sx(k / 20), sy(sg * (1 - k / 20) / r)]);
      el("path", { d: pencil(pts, sg > 0 ? 81 : 82, 0.2), class: "pencil soft", "stroke-width": 1.6, "stroke-dasharray": "2 5" }, fixed);
    }
    text(fixed, sx(0.02), sy(10) - 12, "if nobody were watching: (T − t)/r", "mono", "start");
    const d1 = el("path", { class: "pencil accent", "stroke-width": 2.6 }, live);
    const d2 = el("path", { class: "pencil warm", "stroke-width": 2.6 }, live);
    const xs = el("path", { class: "pencil", "stroke-width": 2.2 }, live);
    const l1 = text(live, 0, 0, "player 1 pushes up", "label acc", "start"), l2 = text(live, 0, 0, "player 2 pushes down", "label warmt", "start");
    text(live, sx(0.6), Y0 - 9, "the state stays at 0", "label");
    const pl = text(live, 300, 36, "", "label");
    const r1 = text(live, 300, 536, "", "label"), r2 = text(live, 300, 561, "", "label"), r3 = text(live, 300, 586, "", "label acc");
    let drawn = null;
    const lerp = (a, b, w) => a + (b - a) * w;
    function at(lp) {
      // the two solved neighbours of log10 p, and the weight between them
      const L = grid.map(Math.log10);
      let k = 1; while (k < grid.length - 1 && L[k] < lp) k++;
      const a = R.tug[grid[k - 1]], b = R.tug[grid[k]], w = clamp((lp - L[k - 1]) / (L[k] - L[k - 1]));
      if (!a || !b) return null;
      const mix = (u, v) => u.map((x, i) => lerp(x, v[i], w));
      return { t: a.t, D1: mix(a.D1, b.D1), D2: mix(a.D2, b.D2), X: mix(a.X, b.X), J1: lerp(a.J1, b.J1, w), J2: lerp(a.J2, b.J2, w), wedge: lerp(a.wedge, b.wedge, w) };
    }
    const fmtP = (p) => (p >= 10 ? Math.round(p).toString() : p >= 1 ? p.toFixed(1).replace(/\.0$/, "") : p.toPrecision(1));
    function draw() {
      // the slider snaps to a solved p when it is near one, so the grid's own numbers are the ones read at it
      let lp = input ? Number(input.value) : Math.log10(9);
      for (const p of grid) if (Math.abs(Math.log10(p) - lp) < 0.04) lp = Math.log10(p);
      const q = at(lp), key = lp + ":" + done;
      if (key === drawn) return;
      const p = Math.pow(10, lp);
      if (val) val.textContent = "p = " + fmtP(p);
      if (!q) { fade(live, 0); fade(wait, 1); return; }
      drawn = key; fade(wait, 0);
      d1.setAttribute("d", pencil(q.t.map((t, i) => [sx(t), sy(q.D1[i])]), 83, 0.15));
      d2.setAttribute("d", pencil(q.t.map((t, i) => [sx(t), sy(q.D2[i])]), 84, 0.15));
      xs.setAttribute("d", pencil(q.t.map((t, i) => [sx(t), sy(q.X[i])]), 85, 0.15));
      // each player's label sits between its curve and the state, clear of the curve over the label's width
      const span = q.t.map((t, i) => i).filter((i) => q.t[i] <= 0.34);
      l1.setAttribute("x", sx(0.02)); l1.setAttribute("y", Math.max(...span.map((i) => sy(q.D1[i]))) + 20);
      l2.setAttribute("x", sx(0.02)); l2.setAttribute("y", Math.min(...span.map((i) => sy(q.D2[i]))) - 9);
      pl.textContent = `both players see with precision p = ${fmtP(p)}`;
      r1.textContent = `player 1's first push: ${q.D1[0].toFixed(2)}`;
      r2.textContent = `each player's cost: ${q.J1.toFixed(2)} and ${q.J2.toFixed(2)}`;
      r3.textContent = `the wedge: ${(100 * q.wedge).toFixed(1)}% of player 1's first-order condition`;
      if (read) read.textContent = `At p = ${fmtP(p)}: player 1's first push ${q.D1[0].toFixed(2)}, each player's cost ${q.J1.toFixed(2)}, the wedge ${(100 * q.wedge).toFixed(1)}% of player 1's first-order condition.`;
    }
    if (input) input.addEventListener("input", () => { drawn = null; draw(); });
    return { g, update(t) {
      if (input && input.disabled && done === jobs.length) input.disabled = false;
      draw(); if (drawn) fade(live, seg(t, 0.05, 0.25));
    } };
  })();

  // ------------------------------------------------------------------ scroll to scene, and the progress line
  const bar = document.getElementById("progress");
  let active = null, prog = 0;
  function measure() {
    const vh = window.innerHeight;
    let best = null, bestD = Infinity;
    for (const s of steps) { const r = s.getBoundingClientRect(), d = Math.abs(r.top + r.height / 2 - (window.readLine ? window.readLine() : vh * 0.55)); if (d < bestD) { bestD = d; best = s; } }
    if (bar) { const h = document.documentElement.scrollHeight - vh; bar.style.width = (h > 0 ? (100 * window.scrollY) / h : 0) + "%"; }
    if (!best) return;
    const r = best.getBoundingClientRect();
    prog = clamp((vh * 0.85 - r.top) / (r.height * 0.9));
    if (best !== active) {
      active = best;
      for (const s of steps) s.classList.toggle("on", s === best);
      for (const [name, sc] of Object.entries(scenes)) sc.g.style.opacity = name === best.dataset.scene ? 1 : 0;
    }
  }
  window.addEventListener("scroll", measure, { passive: true });
  window.addEventListener("resize", measure);
  measure();
  let still = null;
  (function frame(now) {
    const story = document.getElementById("story").getBoundingClientRect();
    // reduced motion: each scene is drawn once, finished and still, when it becomes active (and again as solves land)
    const key = active && active.dataset.scene + done;
    if (active && story.top < window.innerHeight && story.bottom > 0 && !(reduced && still === key)) { const sc = scenes[active.dataset.scene]; if (reduced) still = key; if (sc) sc.update(reduced ? 1 : prog, reduced ? 0 : now); }
    requestAnimationFrame(frame);
  })(performance.now());
})();
