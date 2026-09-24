// /dissertation/wedge: the information wedge, told in steps. Four diagrams, then three drawings solved live by the
// explorer's solver (static/lqg/worker.js) on the Chapter 1 tracking game: player 1's first-order condition split into
// its physical part and the wedge; the wedge shrinking as player 2's signal is blurred; and information starvation.
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
  const SWEEP = [100, 30, 9, 3, 1, 0.3, 0.1, 0.03, 0.01];
  const STARVE = [0.02, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.98];
  const R = { base: null, sweep: [], starve: [], full: null };
  const status = document.getElementById("solvestate");
  const jobs = [{ kind: "base", params: {} }]
    .concat(SWEEP.map((p2) => ({ kind: "sweep", p2, params: { p2 } })))
    .concat([{ kind: "full", params: { p1: 1e4, p2: 1e4, r1: 0.1, r2: 0.5 } }])
    .concat(STARVE.map((f) => ({ kind: "starve", f, params: { p1: 10 * f, p2: 10 * (1 - f), r1: 0.1, r2: 0.5 } })));
  let done = 0, t0 = performance.now();
  const maxAbs = (a) => a.reduce((m, v) => Math.max(m, Math.abs(v)), 0);
  function onResult(job, res) {
    done++;
    if (!res.ok) return;
    if (job.kind === "base") R.base = res.samples.foc.D1;
    else if (job.kind === "sweep") {
      const f = res.samples.foc.D1;
      R.sweep.push({ p2: job.p2, wedge: Object.values(f.channels).reduce((m, d) => Math.max(m, maxAbs(d.wedge)), 0), foc: f });
      R.sweep.sort((a, b) => b.p2 - a.p2);
    } else if (job.kind === "full") R.full = res.costs.player1 + res.costs.player2 + 2;
    else { R.starve.push({ f: job.f, J: res.costs.player1 + res.costs.player2 + 2 }); R.starve.sort((a, b) => a.f - b.f); }
    if (status) {
      status.textContent = done < jobs.length ? `Solving: ${done} of ${jobs.length} equilibria…` : `Solved ${jobs.length} equilibria in your browser, in ${((performance.now() - t0) / 1000).toFixed(1)} s.`;
      status.classList.toggle("ok", done === jobs.length);
    }
    if (window.siteTally) window.siteTally("solve");
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
    const cap = text(g, 300, 480, "learning and acting, separately", "label");
    return { g, update(t) {
      fade(me, seg(t, 0, 0.1)); fade(st, seg(t, 0, 0.1)); fade(fog, seg(t, 0.05, 0.2));
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
    const labs = [text(g, 170, 190, "you push", "mono"), text(g, 440, 190, "they see it", "mono"), text(g, 440, 410, "they revise", "mono"), text(g, 220, 330, "they push back", "mono warmt")];
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
    const g = group("stateprice");
    const V = (x) => 330 - 150 * Math.exp(-((x - 330) ** 2) / 26000) + 0.0009 * (x - 330) ** 2;
    const pts = []; for (let x = 70; x <= 530; x += 8) pts.push([x, V(x)]);
    const curve = stroke(g, pencil(pts, 5, 0.3), "", 2);
    const x0 = 210, slope = (V(x0 + 1) - V(x0 - 1)) / 2;
    const tan = el("line", { x1: x0 - 90, y1: V(x0) - 90 * slope, x2: x0 + 90, y2: V(x0) + 90 * slope, class: "pencil accent", "stroke-width": 2 }, g);
    const ball = el("circle", { cx: x0, cy: V(x0) - 11, r: 11, class: "fillacc" }, g);
    const nudge = stroke(g, pencil([[x0 + 14, V(x0) - 34], [x0 + 60, V(x0 + 46) - 34]], 7, 0.6), "warm", 2);
    const lab = text(g, x0 + 100, V(x0) + 100 * slope - 10, "slope: the shadow price H", "label acc", "start");
    const lv = text(g, 300, 520, "the value of the future, as a function of the state", "mono");
    return { g, update(t) {
      curve.set(seg(t, 0, 0.35)); fade(lv, seg(t, 0.2, 0.35)); fade(ball, seg(t, 0.3, 0.4));
      fade(tan, seg(t, 0.45, 0.6)); fade(lab, seg(t, 0.5, 0.65)); nudge.set(seg(t, 0.6, 0.8));
    } };
  })();

  // ------------------------------------------------------------------ 4. the backward equation, and the wedge in it
  scenes.wedge = (() => {
    const g = group("wedge");
    const f = el("g", {}, g);
    const put = (key, x, y, W) => {
      const src = window.WEDGE_FX && window.WEDGE_FX[key]; if (!src) return null;
      const doc = new DOMParser().parseFromString(src, "image/svg+xml").documentElement, n = document.importNode(doc, true);
      const vb = n.getAttribute("viewBox").split(" ").map(Number), H = (W * vb[3]) / vb[2];
      n.setAttribute("width", W); n.setAttribute("height", H); n.setAttribute("x", x); n.setAttribute("y", y - H / 2); n.removeAttribute("style"); n.style.color = "var(--ink)";
      f.appendChild(n); return { x, y, W, H };
    };
    put("backward1", 40, 250, 470);
    const b2 = put("backward2", 150, 370, 330);
    const hl = b2 ? el("rect", { x: b2.x - 14, y: b2.y - b2.H / 2 - 10, width: b2.W + 28, height: b2.H + 20, rx: 10, class: "fillacc" }, g) : null;
    if (hl) g.insertBefore(hl, f);
    const note = text(g, 300, 170, "the shadow price of the state, run backward in time", "mono");
    const w = text(g, 315, 470, "the information wedge", "label acc");
    return { g, update(t) { fade(note, seg(t, 0, 0.15)); fade(f, seg(t, 0.05, 0.25)); if (hl) fade(hl, seg(t, 0.45, 0.6) * 0.12); fade(w, seg(t, 0.5, 0.65)); } };
  })();

  // ------------------------------------------------------------------ plotting helpers for the solved drawings
  function axes(g, x0, y0, W, H, xlab) {
    el("line", { x1: x0, y1: y0, x2: x0 + W, y2: y0, class: "pencil soft", "stroke-width": 1 }, g);
    if (xlab) text(g, x0 + W / 2, y0 + H + 34, xlab, "mono");
  }
  const waiting = (g) => { const t = text(g, 300, 300, "solving…", "mono"); return t; };

  // ------------------------------------------------------------------ 5. the split, solved
  scenes.split = (() => {
    const g = group("split"), live = el("g", {}, g), wait = waiting(g);
    let drawn = false;
    const NAMES = { w0: "the common shock", w1: "player 1's signal noise", w2: "player 2's signal noise" };
    function draw() {
      const f = R.base; live.innerHTML = "";
      const chans = Object.keys(f.channels);
      let mx = 0; for (const c of chans) mx = Math.max(mx, maxAbs(f.channels[c].physical), maxAbs(f.channels[c].wedge));
      chans.forEach((c, k) => {
        const top = 90 + k * 160, H = 110, x0 = 60, W = 490, mid = top + H / 2, s = f.s, sx = (v) => x0 + (W * v) / s[s.length - 1], sy = (v) => mid - (v / mx) * (H / 2);
        el("line", { x1: x0, y1: mid, x2: x0 + W, y2: mid, class: "pencil soft", "stroke-width": 1 }, live);
        text(live, x0, top + 4, NAMES[c] || c, "mono", "start");
        const pm = maxAbs(f.channels[c].physical), wm = maxAbs(f.channels[c].wedge);
        if (pm > 1e-9) text(live, x0 + W, top + 4, `wedge up to ${Math.round((100 * wm) / pm)}% of the physical part`, "mono acc", "end");
        el("path", { d: pencil(s.map((v, i) => [sx(v), sy(f.channels[c].physical[i])]), 20 + k, 0.1), class: "pencil", "stroke-width": 2 }, live);
        el("path", { d: pencil(s.map((v, i) => [sx(v), sy(f.channels[c].wedge[i])]), 30 + k, 0.1), class: "pencil accent", "stroke-width": 2.6 }, live);
      });
      text(live, 60, 560, "black: physical part", "label", "start");
      text(live, 250, 560, "blue: the information wedge", "label acc", "start");
      text(live, 300, 588, "by the time s of the shock, at t = 1/2", "mono");
      drawn = true;
    }
    return { g, update(t) { if (R.base && !drawn) draw(); fade(wait, R.base ? 0 : 1); fade(live, seg(t, 0.05, 0.25)); } };
  })();

  // ------------------------------------------------------------------ 6. cut the loop: the wedge shrinks with player 2's precision
  scenes.cut = (() => {
    const g = group("cut"), live = el("g", {}, g), wait = waiting(g);
    let n = 0;
    const X = (p) => 80 + (440 * (Math.log10(100) - Math.log10(p))) / (Math.log10(100) - Math.log10(0.01));
    function draw() {
      live.innerHTML = "";
      const mx = Math.max(...R.sweep.map((q) => q.wedge)) || 1, Y = (v) => 470 - (v / mx) * 300;
      el("line", { x1: 70, y1: 470, x2: 530, y2: 470, class: "pencil soft", "stroke-width": 1 }, live);
      const pts = R.sweep.map((q) => [X(q.p2), Y(q.wedge)]);
      if (pts.length > 1) live.appendChild(el("path", { d: pencil(pts, 50, 0.2), class: "pencil accent", "stroke-width": 2.6 }));
      R.sweep.forEach((q) => el("circle", { cx: X(q.p2), cy: Y(q.wedge), r: 4.5, class: "fillacc" }, live));
      for (const p of [100, 1, 0.01]) text(live, X(p), 495, String(p), "mono");
      text(live, 300, 530, "player 2's signal precision p₂  →  blurred", "mono");
      text(live, 70, 140, "size of player 1's wedge", "label acc", "start");
      const last = R.sweep[R.sweep.length - 1];
      if (last && last.p2 <= 0.01) {
        text(live, X(0.01) - 6, Y(last.wedge) - 18, `${last.wedge.toExponential(0)}`, "mono acc", "end");
        text(live, 300, 575, "blind player 2: no wedge, and separation holds again", "label");
      }
      n = R.sweep.length;
    }
    const marker = el("line", { y1: 150, y2: 470, class: "pencil warm", "stroke-width": 1.4, "stroke-dasharray": "3 5" }, g);
    return { g, update(t) {
      if (R.sweep.length !== n) draw();
      fade(wait, R.sweep.length ? 0 : 1); fade(live, seg(t, 0.02, 0.2));
      const lp = 2 - 4 * seg(t, 0.2, 0.85), x = X(Math.pow(10, lp));
      marker.setAttribute("x1", x); marker.setAttribute("x2", x); fade(marker, R.sweep.length > 1 ? seg(t, 0.15, 0.25) : 0);
    } };
  })();

  // ------------------------------------------------------------------ 7. information starvation
  scenes.starve = (() => {
    const g = group("starve"), live = el("g", {}, g), wait = waiting(g);
    let n = 0;
    const X = (f) => 80 + 440 * f;
    function draw() {
      live.innerHTML = "";
      const Js = R.starve.map((q) => q.J).concat(R.full ? [R.full] : []);
      const lo = Math.min(...Js) - 0.1, hi = Math.max(...Js) + 0.1, Y = (v) => 470 - ((v - lo) / (hi - lo)) * 300;
      el("line", { x1: 70, y1: 470, x2: 530, y2: 470, class: "pencil soft", "stroke-width": 1 }, live);
      const pts = R.starve.map((q) => [X(q.f), Y(q.J)]);
      if (pts.length > 1) live.appendChild(el("path", { d: pencil(pts, 60, 0.2), class: "pencil warm", "stroke-width": 2.6 }));
      R.starve.forEach((q) => el("circle", { cx: X(q.f), cy: Y(q.J), r: 4.5, class: "fillwarm" }, live));
      if (R.full) {
        const y = Y(R.full);
        el("line", { x1: 80, y1: y, x2: 520, y2: y, class: "pencil soft", "stroke-width": 1.2, "stroke-dasharray": "5 5" }, live);
        text(live, 520, y - 8, "if both saw everything", "mono", "end");
      }
      const even = R.starve.find((q) => q.f === 0.5);
      if (even) { el("circle", { cx: X(0.5), cy: Y(even.J), r: 9, class: "pencil", "stroke-width": 1.4 }, live); text(live, X(0.5), Y(even.J) - 18, "split evenly", "mono"); }
      for (const f of [0, 0.5, 1]) text(live, X(f), 495, f === 0 ? "all to player 2" : f === 1 ? "all to player 1" : "half", "mono");
      text(live, 70, 140, "the players' total cost", "label warmt", "start");
      text(live, 300, 540, "share of the precision budget given to player 1", "mono");
      n = R.starve.length;
    }
    return { g, update(t) { if (R.starve.length !== n) draw(); fade(wait, R.starve.length ? 0 : 1); fade(live, seg(t, 0.05, 0.25)); } };
  })();

  // ------------------------------------------------------------------ scroll to scene, and the progress line
  const bar = document.getElementById("progress");
  let active = null, prog = 0;
  function measure() {
    const vh = window.innerHeight;
    let best = null, bestD = Infinity;
    for (const s of steps) { const r = s.getBoundingClientRect(), d = Math.abs(r.top + r.height / 2 - vh * 0.55); if (d < bestD) { bestD = d; best = s; } }
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
  (function frame(now) {
    const story = document.getElementById("story").getBoundingClientRect();
    if (active && story.top < window.innerHeight && story.bottom > 0) { const sc = scenes[active.dataset.scene]; if (sc) sc.update(reduced ? 1 : prog, now); }
    requestAnimationFrame(frame);
  })(performance.now());
})();
