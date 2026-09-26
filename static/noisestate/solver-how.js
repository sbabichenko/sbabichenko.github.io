// /noisestate/how: the noise-state solver's algorithm, in steps. The drawings marked "solved in your browser" are read
// off solves run by the explorer's solver (static/noisestate/worker.js) when the page opens: the Chapter 1 tracking game
// from a zero start with Anderson and with plain damped iteration, from the coarse start, and with refinement and
// stability; the Chapter 3 stationary game at windows 2, 4 and 8 and at window 2 with discounting; the explorer's
// regime-change preset. The drawings marked "diagram" are drawn by hand.
(function () {
  "use strict";
  const NS = "http://www.w3.org/2000/svg";
  const svg = document.getElementById("stage");
  const steps = [...document.querySelectorAll(".step")];
  const page = document.getElementById("solverpage");
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
  const poly = (pts) => "M" + pts.map((p) => p[0].toFixed(1) + "," + p[1].toFixed(1)).join(" L");
  function stroke(g, d, cls, width = 1.6) {
    const a = el("path", { d, class: "pencil " + (cls || ""), "stroke-width": width }, g);
    const len = a.getTotalLength() || 1;
    a.style.strokeDasharray = `${len} ${len}`; a.style.strokeDashoffset = len;
    return { a, set(t) { a.style.strokeDashoffset = len * (1 - clamp(t)); } };
  }
  function text(g, x, y, s, cls, anchor = "middle") { const t = el("text", { x, y, class: cls || "", "text-anchor": anchor }, g); t.textContent = s; return t; }
  function box(g, x, y, label, w, cls) { const b = el("g", {}, g); const W = w || label.length * 9 + 28; el("rect", { x: x - W / 2, y: y - 19, width: W, height: 38, rx: 6, class: "box pencil " + (cls || ""), "stroke-width": 1.6 }, b); text(b, x, y + 6, label); return b; }
  function arrowHead(g, x, y, ang, cls) { return el("path", { d: `M${x - 10 * Math.cos(ang - 0.45)},${y - 10 * Math.sin(ang - 0.45)} L${x},${y} L${x - 10 * Math.cos(ang + 0.45)},${y - 10 * Math.sin(ang + 0.45)}`, class: "pencil " + (cls || ""), "stroke-width": 1.8 }, g); }
  const SUP = { "-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };
  function sci(x, d = 1) {
    if (x === 0) return "0";
    const neg = x < 0; x = Math.abs(x);
    let e = Math.floor(Math.log10(x)), m = x / Math.pow(10, e);
    if (+m.toFixed(d) >= 10) { m /= 10; e += 1; }
    const s = e >= -2 && e <= 2 ? String(+x.toPrecision(d + 1)) : `${m.toFixed(d)} × 10${String(e).split("").map((c) => SUP[c]).join("")}`;
    return (neg ? "−" : "") + s;
  }
  const scenes = {};
  const group = (name) => { const g = el("g", { "data-scene": name }, svg); g.style.opacity = 0; g.style.transition = "opacity 0.6s"; return g; };
  const corner = (g, s) => text(g, 590, 20, s, "mono", "end");
  const waiting = (g) => text(g, 300, 300, "solving…", "mono");

  // ------------------------------------------------------------------ the solves, run in the background
  const CH1 = {"name":"ch1_tracking_with_targets","params":{"p1":9,"p2":9,"r1":0.1,"r2":0.1,"b1":1,"b2":-1,"sigma":1,"T":1},"channels":["w0","w1","w2"],"states":{"X":{"drift":{"D1":1,"D2":1},"noise":{"w0":"sigma"}}},"agents":{"player1":{"controls":["D1"],"signals":{"y1":{"drift":{"X":"sqrt(p1)"},"noise":{"w1":1}}},"loss":[[1,"X","X"],["-2*b1","X"],["r1","D1","D1"]]},"player2":{"controls":["D2"],"signals":{"y2":{"drift":{"X":"sqrt(p2)"},"noise":{"w2":1}}},"loss":[[1,"X","X"],["-2*b2","X"],["r2","D2","D2"]]}},"horizon":{"kind":"finite","T":"T"},"numerics":{"nodes":12}};
  const CH3 = {"name":"ch3_stationary_tracking","params":{"p1":3,"p2":10,"r1":1,"r2":1},"channels":["w0","w1","w2"],"states":{"X":{"drift":{"D1":1,"D2":1},"noise":{"w0":1}}},"agents":{"player1":{"controls":["D1"],"signals":{"y1":{"drift":{"X":"sqrt(p1)"},"noise":{"w1":1}}},"loss":[[0.5,"X","X"],["0.5*r1","D1","D1"]]},"player2":{"controls":["D2"],"signals":{"y2":{"drift":{"X":"sqrt(p2)"},"noise":{"w2":1}}},"loss":[[0.5,"X","X"],["0.5*r2","D2","D2"]]}},"horizon":{"kind":"stationary","discount":0,"window":8},"numerics":{"nodes":32}};
  const TRAN_AGENTS = {"player1":{"controls":["D1"],"signals":{"y1":{"drift":{"X":"sqrt(p1)"},"noise":{"w1":1}}},"loss":[[0.5,"X","X"],["0.5*r1","D1","D1"]]},"player2":{"controls":["D2"],"signals":{"y2":{"drift":{"X":"sqrt(p2)"},"noise":{"w2":1}}},"loss":[[0.5,"X","X"],["0.5*r2","D2","D2"]]}};
  const TRAN_STATES = {"X":{"drift":{"X":"-a","D1":1,"D2":1},"noise":{"w0":1}}};
  const TRAN = {"name":"regime_change","params":{"p1":6,"p2":3,"r1":1,"r2":1,"a":1,"T":6},"channels":["w0","w1","w2"],"states":TRAN_STATES,"agents":TRAN_AGENTS,
    "horizon":{"kind":"transition","T":"T","past":{"model":{"name":"before","params":{"p1":1,"p2":3,"r1":1,"r2":1,"a":1},"channels":["w0","w1","w2"],"states":TRAN_STATES,"agents":TRAN_AGENTS,"horizon":{"kind":"stationary","window":3},"numerics":{"nodes":12}}},"continuation":"stationary"},
    "numerics":{"nodes":8,"continuation_nodes":12}};
  const copy = (m) => JSON.parse(JSON.stringify(m));
  const plain = copy(CH1); plain.numerics.settings = { anderson_m: 0 };
  const stat = (window, discount) => { const m = copy(CH3); m.horizon.window = window; m.horizon.discount = discount; return m; };
  const jobs = [
    { kind: "base", model: CH1, request: { start_policy: "zero" } },
    { kind: "plain", model: plain, request: { start_policy: "zero" } },
    { kind: "coarse", model: CH1, request: { start_policy: "coarse" } },
    { kind: "checks", model: CH1, request: { start_policy: "coarse", refine: true, stability: true } },
    { kind: "window", L: 2, rho: 0, model: stat(2, 0), request: { start_policy: "coarse" } },
    { kind: "window", L: 4, rho: 0, model: stat(4, 0), request: { start_policy: "coarse" } },
    { kind: "window", L: 8, rho: 0, model: stat(8, 0), request: { start_policy: "coarse" } },
    { kind: "window", L: 2, rho: 1, model: stat(2, 1), request: { start_policy: "coarse" } },
    { kind: "window", L: 2, rho: 3, model: stat(2, 3), request: { start_policy: "coarse" } },
    { kind: "transition", model: TRAN, request: {} },
  ];
  const R = { base: null, trace: {}, checks: null, windows: [], transition: null, version: 0 };
  const status = document.getElementById("solvestate");
  let done = 0, t0 = performance.now();
  const prog = jobs.map(() => []);
  function onResult(k, res) {
    done++;
    const job = jobs[k];
    if (res.ok) {
      if (job.kind === "base") { R.base = res; R.trace.anderson = prog[k]; }
      else if (job.kind === "plain") R.trace.plain = prog[k];
      else if (job.kind === "coarse") R.trace.coarse = prog[k];
      else if (job.kind === "checks") R.checks = res;
      else if (job.kind === "window") { R.windows.push({ L: job.L, rho: job.rho, res }); }
      else if (job.kind === "transition") R.transition = res;
      R.version++;
    }
    if (status) {
      status.textContent = done < jobs.length ? `Solving: ${done} of ${jobs.length} solves…` : `Ran ${jobs.length} solves in your browser, in ${((performance.now() - t0) / 1000).toFixed(1)} s.`;
      status.classList.toggle("ok", done === jobs.length);
    }
    if (window.siteTally) window.siteTally("solve", 1, done === jobs.length ? `${jobs.length} solves for the solver's inner workings` : "");
  }
  try {
    const worker = new Worker(page.dataset.worker);
    let i = 0;
    const next = () => { if (i < jobs.length) worker.postMessage({ type: "solve", id: i, model: jobs[i].model, request: jobs[i].request }); };
    worker.onmessage = (ev) => {
      const m = ev.data;
      if (m.type === "ready") { t0 = performance.now(); next(); }
      else if (m.type === "progress") { if (m.id === i && prog[i]) prog[i].push([m.evaluation, m.residual]); }
      else if (m.type === "result") { let res; try { res = JSON.parse(m.result); } catch (e) { res = { ok: false }; } onResult(m.id, res); i++; next(); }
      else if (m.type === "fatal" && status) status.textContent = "The solver could not start in this browser: " + m.message;
    };
  } catch (e) { if (status) status.textContent = "This browser cannot run the solver."; }

  // a scene whose drawing needs solved data: build() once the data it reads has arrived
  function live(name, ready, build, anim) {
    const g = group(name);
    corner(g, "solved in your browser");
    const wait = waiting(g);
    const body = el("g", {}, g);
    let built = false;
    return { g, update(t, now) {
      if (!built && ready()) { build(body); built = true; }
      fade(wait, built ? 0 : 1);
      if (built && anim) anim(t, now);
      fade(body, built ? seg(t, 0.02, 0.2) : 0);
    } };
  }
  const nodeKernel = (res, ctl, ch) => res.kernels[ctl][ch];
  const peak = (a) => a.reduce((m, v) => Math.max(m, Math.abs(v)), 0) || 1;
  function dotField(g, res, vals, X, Y, rmax, scale) {
    const n = res.nodes.t.length, m = scale || peak(vals);
    for (let k = 0; k < n; ++k) {
      const t = res.nodes.t[k], s = t - res.nodes.a[k], v = vals[k], r = 1.2 + rmax * Math.sqrt(Math.min(1, Math.abs(v) / m));
      el("circle", { cx: X(t).toFixed(1), cy: Y(s).toFixed(1), r: r.toFixed(2), class: v < 0 ? "fillacc" : "fillwarm", opacity: Math.abs(v) < 1e-9 * m ? 0.25 : 0.8 }, g);
    }
  }
  function triangleOutline(g, X, Y, seed) {
    stroke(g, pencil([[X(0), Y(0)], [X(1), Y(0)], [X(1), Y(1)], [X(0), Y(0)]], seed, 0.4), "soft", 1.3).set(1);
  }

  // ------------------------------------------------------------------ 1. kernels on the causal triangle
  scenes.triangle = live("triangle", () => R.base, (g) => {
    const res = R.base, X = (t) => 110 + 210 * t, Y = (s) => 250 - 210 * s;
    triangleOutline(g, X, Y, 3);
    text(g, X(0.5), Y(0) + 26, "date t", "label"); text(g, X(1) + 14, Y(0.5), "shock date s", "label", "start");
    text(g, X(0.625), Y(0.18) + 5, "s ≤ t", "mono");
    const sl = res.samples.kernels.D1.w0, ops = [0.35, 0.55, 0.75, 1];
    sl.forEach((q, j) => { el("line", { x1: X(q.t), x2: X(q.t), y1: Y(0), y2: Y(q.t), class: "pencil accent", "stroke-width": 2.2, opacity: ops[j] }, g); });
    const lo = Math.min(0, ...sl.flatMap((q) => q.v)), hi = Math.max(0, ...sl.flatMap((q) => q.v)), pad = 0.08 * (hi - lo || 1);
    const PX = (s) => 110 + 420 * s, PY = (v) => 545 - 190 * (v - (lo - pad)) / (hi - lo + 2 * pad);
    el("line", { x1: PX(0), x2: PX(1), y1: PY(0), y2: PY(0), class: "pencil soft", "stroke-width": 1 }, g);
    el("line", { x1: PX(0), x2: PX(0), y1: PY(lo - pad), y2: PY(hi + pad), class: "pencil soft", "stroke-width": 1 }, g);
    text(g, PX(0) - 8, PY(0) + 5, "0", "mono", "end");
    text(g, PX(0.5), 578, "shock date s", "label");
    { const e = el("text", { x: PX(0), y: PY(hi + pad) - 10, class: "mono", "text-anchor": "start" }, g);
      [["D¹"], ["W,t", 1], ["(s), state shock w⁰"]].forEach(([str, low]) => { const sp = el("tspan", low ? { "baseline-shift": "sub", "font-size": "75%" } : {}, e); sp.textContent = str; }); }
    sl.forEach((q, j) => {
      el("path", { d: poly(q.s.map((s, i) => [PX(s), PY(q.v[i])])), class: "pencil accent", "stroke-width": 2, opacity: ops[j] }, g);
      const e = q.s.length - 1;
      text(g, PX(q.s[e]) + 6, PY(q.v[e]) + (j === 3 ? -8 : 16), `t = ${q.t}`, "mono", j === 3 ? "end" : "start");
    });
  });

  // ------------------------------------------------------------------ 1b. the strategy on estimates, and the same action on the shocks
  // Both read off the Chapter 1 solve at date t = 0.5 (res.samples.foc is sampled there). Remark 1.13: D_t = -E[H_t | F_t] / (2 r1), so D_t(u) = -H_t(u) / (2 r1). The first-order condition of
  // player 1 is E[phi_t | F_t] = 0 with phi_t = 2 r1 D_t + C_t (2 r1 is the Hessian of r1 D1^2, C the continuation), so
  // D_t = E[-C_t / (2 r1) | F_t]: the rule on the noise-state with kernel D_t(u) = D_W,t(u) - phi_t(u) / (2 r1). Checked
  // against the package: player 1's projection of phi is 1e-5 of phi off the diagonal and shrinks with the nodes on it.
  scenes.estimates = live("estimates", () => R.base && R.base.samples.foc && R.base.samples.foc.D1, (g) => {
    const res = R.base, f = res.samples.foc.D1, t = f.t, r1 = CH1.params.r1, chs = ["w0", "w1", "w2"];
    const interp = (xs, ys, x) => { let k = 1; while (k < xs.length - 1 && xs[k] < x) ++k; const w = (x - xs[k - 1]) / (xs[k] - xs[k - 1] || 1); return ys[k - 1] + w * (ys[k] - ys[k - 1]); };
    const nb = 11, ss = [...Array(nb).keys()].map((k) => t * k / (nb - 1));
    const DW = {}, DE = {};
    chs.forEach((c) => {
      const q = res.samples.kernels.D1[c].find((z) => Math.abs(z.t - t) < 1e-9);
      DW[c] = ss.map((x) => interp(q.s, q.v, x));
      DE[c] = ss.map((x, k) => DW[c][k] - interp(f.s, f.channels[c].foc, x) / (2 * r1));
    });
    const top = Math.max(...chs.flatMap((c) => DE[c].concat(DW[c]).map(Math.abs)));
    const px = 118 / top, X = (k) => 100 + k * 44, bw = 11, off = { w0: -12, w1: 0, w2: 12 };
    const cls = { w0: "fillacc", w1: "fillwarm", w2: "" };
    function sub(x, y, parts, c, anchor) {
      const e = el("text", { x, y, class: c || "", "text-anchor": anchor || "start" }, g);
      parts.forEach(([str, low]) => { const sp = el("tspan", low ? { "baseline-shift": "sub", "font-size": "75%" } : {}, e); sp.textContent = str; });
      return e;
    }
    function panel(vals, y0, head, lab) {
      sub(70, y0 - 88, head, "mono");
      el("line", { x1: 72, x2: 560, y1: y0, y2: y0, class: "pencil soft", "stroke-width": 1 }, g);
      text(g, 64, y0 + 4, "0", "mono", "end");
      chs.forEach((c) => vals[c].forEach((v, k) => {
        const h = v * px, r = el("rect", { x: X(k) + off[c] - bw / 2, width: bw, y: Math.min(y0, y0 - h), height: Math.abs(h), class: cls[c] }, g);
        if (!cls[c]) { r.setAttribute("fill", "currentColor"); r.setAttribute("opacity", 0.4); }
      }));
      [0, 5, 10].forEach((k) => text(g, X(k), y0 + 138, ss[k].toFixed(2).replace(/0$/, ""), "mono"));
      text(g, 315, y0 + 162, lab, "label");
    }
    panel(DE, 138, [["strategy kernel D¹"], ["t", 1], ["(u): weights on the estimates Ŵ¹"], ["t", 1], ["(u)"]], "date u of the shock");
    panel(DW, 418, [["the same action by shock, D¹"], ["W,t", 1], ["(s)"]], "date s of the shock");
    const key = [["w0", "state shock w⁰", "mono acc"], ["w1", "player 1's noise w¹", "mono warmt"], ["w2", "player 2's noise w²", "mono"]];
    key.forEach(([c, lab, k], i) => text(g, 560, 352 + 16 * i, lab, k, "end"));
  });

  // ------------------------------------------------------------------ 2. nodes, and the representation error
  scenes.nodes = live("nodes", () => R.base, (g) => {
    const res = R.base, X = (t) => 150 + 390 * t, Y = (s) => 520 - 390 * s;
    triangleOutline(g, X, Y, 5);
    dotField(g, res, nodeKernel(res, "D1", "w0"), X, Y, 7);
    text(g, X(0.5), Y(0) + 28, "date t", "label"); text(g, X(1) + 12, Y(0.55), "s", "label", "start");
    // the square the triangle is mapped from, with the same nodes at (t, a / t)
    const S0x = 40, S0y = 60, SW = 150, sx = (u) => S0x + SW * u, sy = (t) => S0y + SW * (1 - t);
    el("rect", { x: S0x, y: S0y, width: SW, height: SW, class: "pencil soft", "stroke-width": 1.2 }, g);
    for (let k = 0; k < res.nodes.t.length; ++k) { const t = res.nodes.t[k], u = t > 0 ? res.nodes.a[k] / t : (k % 12) / 11; el("circle", { cx: sx(u).toFixed(1), cy: sy(t).toFixed(1), r: 1.8, fill: "currentColor", opacity: 0.6 }, g); }
    text(g, S0x + SW / 2, S0y - 10, "the same nodes on a square", "mono");
    stroke(g, pencil([[S0x + SW + 12, S0y + SW * 0.7], [X(0.55) - 40, Y(0.4)]], 7, 1), "soft", 1.3).set(1);
    arrowHead(g, X(0.55) - 40, Y(0.4), Math.atan2(Y(0.4) - (S0y + SW * 0.7), X(0.55) - 40 - (S0x + SW + 12)), "soft");
    const n = Math.round(Math.sqrt(res.nodes.t.length));
    text(g, 300, 568, `${n} × ${n} = ${res.nodes.t.length} nodes, shaded by D¹ on w⁰`, "label");
    const rep = Math.max(...Object.values(res.representation_error));
    text(g, 300, 592, `representation error ${sci(rep)}`, "label acc");
  });

  // ------------------------------------------------------------------ 3. the fixed-point loop (diagram)
  scenes.loop = (() => {
    const g = group("loop");
    corner(g, "diagram");
    const S = box(g, 105, 300, "strategies D", 150), F = box(g, 490, 300, "filters and beliefs", 190);
    const top = stroke(g, pencil([[135, 270], [300, 170], [465, 270]], 11, 1.5), "", 1.8), ht = arrowHead(g, 465, 270, 0.62, "");
    const bot = stroke(g, pencil([[465, 330], [300, 430], [135, 330]], 12, 1.5), "accent", 1.8), hb = arrowHead(g, 135, 330, -2.52, "accent");
    const l1 = text(g, 300, 150, "actions enter others' observations", "mono");
    const l2 = text(g, 300, 462, "best response, linear in the noise-state", "mono acc");
    const c = text(g, 300, 306, "equilibrium: G(D) = D", "label");
    const it = text(g, 300, 540, "one pass around the loop is one evaluation of G", "mono");
    return { g, update(t) {
      fade(S, seg(t, 0, 0.1)); fade(F, seg(t, 0.05, 0.15));
      top.set(seg(t, 0.1, 0.35)); fade(ht, seg(t, 0.33, 0.36)); fade(l1, seg(t, 0.2, 0.35));
      bot.set(seg(t, 0.35, 0.6)); fade(hb, seg(t, 0.58, 0.61)); fade(l2, seg(t, 0.45, 0.6));
      fade(c, seg(t, 0.6, 0.75)); fade(it, seg(t, 0.7, 0.85));
    } };
  })();

  // ------------------------------------------------------------------ 4. one best response in the passive world (diagram)
  scenes.passive = (() => {
    const g = group("passive");
    corner(g, "diagram");
    const X = box(g, 300, 110, "state X", 120);
    const P2 = box(g, 490, 230, "player 2", 130);
    const Y1 = box(g, 110, 230, "player 1's signal", 170);
    const P1 = box(g, 110, 110, "player 1: off", 150, "soft");
    const a1 = stroke(g, pencil([[360, 120], [470, 210]], 21, 1), "", 1.6); const h1 = arrowHead(g, 470, 210, 0.7, "");
    const a2 = stroke(g, pencil([[440, 240], [330, 132]], 22, 1), "", 1.6); const h2 = arrowHead(g, 330, 132, -2.35, "");
    const a3 = stroke(g, pencil([[245, 122], [140, 210]], 23, 1), "", 1.6); const h3 = arrowHead(g, 140, 210, 2.44, "");
    const la = text(g, 470, 160, "signal, then action", "mono", "start");
    // the time axis: the passive world forward, the value of a push collected over [t, T]
    const ax = stroke(g, pencil([[70, 420], [530, 420]], 24, 0.4), "", 1.6), ha = arrowHead(g, 530, 420, 0, "");
    const t0l = text(g, 70, 448, "0", "mono"), tTl = text(g, 520, 448, "T", "mono"), ttl = text(g, 250, 448, "t", "mono");
    const fw = stroke(g, pencil([[80, 385], [500, 385]], 25, 0.6), "", 1.4), hf = arrowHead(g, 500, 385, 0, "");
    const lf = text(g, 290, 372, "passive world, forward in time", "mono");
    const br = stroke(g, pencil([[250, 470], [250, 482], [520, 482], [520, 470]], 26, 0.3), "accent", 1.6);
    const lb = text(g, 385, 506, "value of a push at t: its effects on [t, T]", "mono acc");
    const sol = box(g, 300, 560, "one linear solve for D¹", 230);
    return { g, update(t) {
      fade(X, seg(t, 0, 0.08)); fade(P2, seg(t, 0.03, 0.1)); fade(Y1, seg(t, 0.06, 0.13)); fade(P1, seg(t, 0.08, 0.16));
      a1.set(seg(t, 0.1, 0.22)); fade(h1, seg(t, 0.21, 0.23)); a2.set(seg(t, 0.18, 0.3)); fade(h2, seg(t, 0.29, 0.31));
      a3.set(seg(t, 0.25, 0.37)); fade(h3, seg(t, 0.36, 0.38)); fade(la, seg(t, 0.2, 0.3));
      ax.set(seg(t, 0.38, 0.5)); fade(ha, seg(t, 0.49, 0.51)); [t0l, tTl, ttl].forEach((n) => fade(n, seg(t, 0.45, 0.52)));
      fw.set(seg(t, 0.5, 0.65)); fade(hf, seg(t, 0.64, 0.66)); fade(lf, seg(t, 0.55, 0.65));
      br.set(seg(t, 0.65, 0.78)); fade(lb, seg(t, 0.7, 0.8)); fade(sol, seg(t, 0.8, 0.92));
    } };
  })();

  // ------------------------------------------------------------------ 5. the first-order condition, physical part and wedge, at every node
  scenes.split = live("split", () => R.base, (g) => {
    const res = R.base, f = res.foc_nodal.D1.w0;
    const panel = (x0, vals, label, cls) => {
      const X = (t) => x0 + 230 * t, Y = (s) => 330 - 230 * s;
      triangleOutline(g, X, Y, x0);
      dotField(g, res, vals, X, Y, 6.5);
      text(g, X(0.5), 70, label, "label " + cls);
      text(g, X(0.5), 360, `largest ${sci(peak(vals), 2)}`, "mono");
    };
    panel(40, f.physical, "physical part", "");
    panel(330, f.wedge, "wedge", "acc");
    text(g, 300, 388, "each on its own scale, date t across, shock date s up", "mono");
    // one date, both parts on a common scale
    const q = res.samples.foc.D1, c = q.channels.w0, all = c.physical.concat(c.wedge, [0]);
    const lo = Math.min(...all), hi = Math.max(...all), PX = (s) => 90 + 440 * s / q.t, PY = (v) => 560 - 130 * (v - lo) / (hi - lo || 1);
    el("line", { x1: PX(0), x2: PX(q.t), y1: PY(0), y2: PY(0), class: "pencil soft", "stroke-width": 1 }, g);
    el("path", { d: poly(q.s.map((s, i) => [PX(s), PY(c.physical[i])])), class: "pencil", "stroke-width": 2 }, g);
    el("path", { d: poly(q.s.map((s, i) => [PX(s), PY(c.wedge[i])])), class: "pencil accent", "stroke-width": 2 }, g);
    const e = q.s.length - 1;
    text(g, PX(q.t) + 6, PY(c.physical[e]) + 4, "physical", "mono", "start");
    text(g, PX(q.t) + 6, PY(c.wedge[e]) + 4, "wedge", "mono acc", "start");
    text(g, PX(0) - 6, PY(0) + 4, "0", "mono", "end");
    text(g, 90, 420, `at t = ${q.t}, on a common scale, against s`, "mono", "start");
  });

  // ------------------------------------------------------------------ 6 and 7. residual traces
  const LX = (e, emax) => 90 + 450 * e / emax, LY = (r) => 90 + 400 * (2 - Math.log10(Math.max(r, 1e-10))) / 11;
  function resAxes(g, emax) {
    for (const p of [0, -3, -6, -9]) { el("line", { x1: 90, x2: 540, y1: LY(Math.pow(10, p)), y2: LY(Math.pow(10, p)), class: "pencil soft", "stroke-width": 0.8 }, g); text(g, 82, LY(Math.pow(10, p)) + 4, p === 0 ? "1" : "10" + String(p).split("").map((c) => SUP[c]).join(""), "mono", "end"); }
    el("line", { x1: 90, x2: 540, y1: LY(1e-8), y2: LY(1e-8), class: "pencil accent", "stroke-width": 1, "stroke-dasharray": "4 5", opacity: 0.7 }, g);
    text(g, 96, LY(1e-8) - 6, "tolerance 10⁻⁸", "mono acc", "start");
    for (let e = 0; e <= emax; e += 5) text(g, LX(e, emax), 520, String(e), "mono");
    text(g, 315, 548, "evaluations of the best-response map G", "label");
    text(g, 90, 66, "residual", "mono", "start");
  }
  function trace(g, pts, emax, cls, width) { return stroke(g, poly(pts.map(([e, r]) => [LX(e, emax), LY(r)])), cls, width); }
  let andAnim = null;
  scenes.anderson = live("anderson", () => R.trace.anderson && R.trace.plain, (g) => {
    const a = R.trace.anderson, p = R.trace.plain, emax = 5 * Math.ceil(Math.max(a.length, p.length, 1) / 5);
    resAxes(g, emax);
    const tp = trace(g, p, emax, "", 1.8), ta = trace(g, a, emax, "accent", 2.2);
    const lp = text(g, 540, 110, `plain, halfway steps: ${p.length} evaluations`, "mono", "end");
    const la = text(g, 540, 130, `Anderson, memory 15: ${a.length} evaluations`, "mono acc", "end");
    andAnim = (t) => { tp.set(seg(t, 0.1, 0.6)); ta.set(seg(t, 0.1, 0.6)); fade(lp, seg(t, 0.55, 0.65)); fade(la, seg(t, 0.55, 0.65)); };
  }, (t) => andAnim && andAnim(t));

  let coarseAnim = null;
  scenes.coarse = live("coarse", () => R.trace.coarse && R.trace.anderson, (g) => {
    const c = R.trace.coarse, z = R.trace.anderson;
    let cut = c.findIndex((q, i) => i > 0 && q[0] <= c[i - 1][0]); if (cut < 0) cut = c.length;
    const lo = c.slice(0, cut), hi = c.slice(cut), emax = 5 * Math.ceil(Math.max(lo.length, hi.length, z.length, 1) / 5);
    resAxes(g, emax);
    const tl = trace(g, lo, emax, "warm", 2), th = hi.length ? trace(g, hi, emax, "accent", 2.4) : null;
    const nodes = R.base ? Math.round(Math.sqrt(R.base.nodes.t.length)) : 12;
    const ll = text(g, 540, 110, `${Math.round(nodes / 2)} nodes, from zero: ${lo.length} evaluations`, "mono warmt", "end");
    const lh = hi.length ? text(g, 540, 130, `${nodes} nodes, from the ${Math.round(nodes / 2)}-node answer: ${hi.length}`, "mono acc", "end") : null;
    const lz = text(g, 540, 150, `${nodes} nodes, from zero (previous drawing): ${z.length}`, "mono", "end");
    coarseAnim = (t) => { fade(lz, seg(t, 0.05, 0.2)); tl.set(seg(t, 0.1, 0.4)); fade(ll, seg(t, 0.15, 0.3));
      if (th) { th.set(seg(t, 0.4, 0.7)); fade(lh, seg(t, 0.45, 0.6)); } };
  }, (t) => coarseAnim && coarseAnim(t));

  // ------------------------------------------------------------------ 8. the checks
  const refineLine = document.getElementById("refineline");
  scenes.checks = live("checks", () => R.checks, (g) => {
    const res = R.checks, rows = [];
    for (const c of res.checks) {
      if (c.name === "converged") rows.push(["residual", c.value, "below " + sci(c.threshold), c.ok]);
      else if (c.name === "resolution") rows.push(["representation error", c.value, "below " + sci(c.threshold), c.ok]);
      else if (c.name.startsWith("second_order:")) rows.push([`curvature, ${c.name.split(":")[1].replace("player", "player ")}`, c.value, "above " + sci(c.threshold), c.ok]);
      else if (c.name === "refinement") {
        rows.push([`refinement: costs`, c.value.cost_change, "below " + sci(c.threshold.cost_change), c.value.cost_change <= c.threshold.cost_change]);
        rows.push([`refinement: kernels`, c.value.kernel_change, "below " + sci(c.threshold.kernel_change), c.value.kernel_change <= c.threshold.kernel_change]);
      } else if (c.name === "stability") rows.push(["spectral radius", c.value, "below 1", c.ok]);
    }
    text(g, 30, 90, "check", "mono", "start"); text(g, 330, 90, "measured", "mono", "end"); text(g, 470, 90, "threshold", "mono", "end"); text(g, 575, 90, "verdict", "mono", "end");
    rows.forEach(([name, v, th, ok], i) => {
      const y = 130 + 44 * i;
      el("line", { x1: 30, x2: 575, y1: y + 14, y2: y + 14, class: "pencil soft", "stroke-width": 0.6 }, g);
      text(g, 30, y, name, "label", "start");
      text(g, 330, y, sci(v), "label", "end");
      text(g, 470, y, th, "label", "end");
      text(g, 575, y, ok ? "passes" : "fails", "label " + (ok ? "acc" : "warmt"), "end");
    });
    const rf = res.refinement;
    text(g, 30, 130 + 44 * rows.length + 20, `refinement at ${rf.nodes} nodes, curvature as smallest over largest`, "mono", "start");
    if (refineLine && rf) refineLine.textContent = `Here, solving again at ${rf.nodes} nodes moves the costs by ${sci(rf.cost_change)} and the kernels by ${sci(rf.kernel_change)}, against tolerances of 10⁻⁶ and 10⁻⁵.`;
  });

  // ------------------------------------------------------------------ 9. the lag quadrant (diagram)
  scenes.quadrant = (() => {
    const g = group("quadrant");
    corner(g, "diagram");
    const X = (t) => 50 + 170 * t, Y = (s) => 300 - 170 * s;
    const tri = stroke(g, pencil([[X(0), Y(0)], [X(1), Y(0)], [X(1), Y(1)], [X(0), Y(0)]], 31, 0.4), "", 1.6);
    const lt = text(g, X(0.5), Y(0) + 26, "date t", "mono"), ls = text(g, X(0.62), Y(0.28), "s ≤ t", "mono");
    const l1 = text(g, X(0.5), 50, "finite horizon", "label");
    const arr = stroke(g, pencil([[245, 220], [320, 220]], 32, 0.4), "", 1.6), ha = arrowHead(g, 320, 220, 0, "");
    const QX = (a) => 360 + 200 * a, QY = (b) => 300 - 200 * b;
    const qa = stroke(g, pencil([[QX(0), QY(0)], [QX(1.1), QY(0)]], 33, 0.3), "", 1.6), qb = stroke(g, pencil([[QX(0), QY(0)], [QX(0), QY(1.1)]], 34, 0.3), "", 1.6);
    const sq = el("g", {}, g); el("rect", { x: QX(0), y: QY(0.75), width: QX(0.75) - QX(0), height: QY(0) - QY(0.75), class: "fillacc", opacity: 0.12 }, sq);
    const sqe = stroke(g, poly([[QX(0.75), QY(0)], [QX(0.75), QY(0.75)], [QX(0), QY(0.75)]]), "accent", 1.6);
    const la = text(g, QX(0.55), QY(0) + 44, "age of the shock", "mono"), lb = text(g, QX(0) + 8, QY(1.1) + 4, "age of the belief", "mono", "start");
    const lL = text(g, QX(0.75), QY(0) + 22, "L", "label acc"), lL2 = text(g, QX(0) - 14, QY(0.75) + 5, "L", "label acc", "end");
    const l2 = text(g, QX(0.5), 50, "stationary", "label");
    // an action kernel reads one age: the line [0, L]
    const AX = (a) => 120 + 360 * a;
    const act = stroke(g, pencil([[AX(0), 470], [AX(1), 470]], 35, 0.3), "accent", 2);
    const nodes = el("g", {}, g);
    for (let k = 0; k < 32; ++k) { const u = 0.5 - 0.5 * Math.cos(Math.PI * k / 31); el("circle", { cx: AX(u).toFixed(1), cy: 470, r: 2.6, class: "fillacc" }, nodes); }
    const l3 = text(g, AX(0), 500, "0", "mono"), l4 = text(g, AX(1), 500, "L", "mono acc"), l5 = text(g, 300, 540, "where an action kernel is stored: 32 Chebyshev nodes over the age of the shock, 0 to L", "mono");
    return { g, update(t) {
      tri.set(seg(t, 0, 0.15)); [lt, ls, l1].forEach((n) => fade(n, seg(t, 0.08, 0.18)));
      arr.set(seg(t, 0.18, 0.28)); fade(ha, seg(t, 0.27, 0.29));
      qa.set(seg(t, 0.28, 0.4)); qb.set(seg(t, 0.28, 0.4)); [la, lb, l2].forEach((n) => fade(n, seg(t, 0.35, 0.45)));
      fade(sq, seg(t, 0.45, 0.55)); sqe.set(seg(t, 0.45, 0.58)); fade(lL, seg(t, 0.5, 0.58)); fade(lL2, seg(t, 0.5, 0.58));
      act.set(seg(t, 0.6, 0.72)); fade(nodes, seg(t, 0.68, 0.8)); [l3, l4, l5].forEach((n) => fade(n, seg(t, 0.7, 0.8)));
    } };
  })();

  // ------------------------------------------------------------------ 10. window truncation
  scenes.window = live("window", () => R.windows.filter((w) => w.rho === 0).length === 3, (g) => {
    const W = R.windows.filter((w) => w.rho === 0).sort((a, b) => a.L - b.L), D = R.windows.filter((w) => w.rho > 0).sort((a, b) => a.rho - b.rho);
    const Lmax = W[W.length - 1].L, all = W.flatMap((w) => w.res.samples.kernels.D1.w0).concat([0]);
    const lo = Math.min(...all), hi = Math.max(...all), pad = 0.1 * (hi - lo || 1);
    const X = (a) => 80 + 470 * a / Lmax, Y = (v) => 400 - 290 * (v - (lo - pad)) / (hi - lo + 2 * pad);
    el("line", { x1: X(0), x2: X(Lmax), y1: Y(0), y2: Y(0), class: "pencil soft", "stroke-width": 1 }, g);
    text(g, X(0) - 6, Y(0) + 4, "0", "mono", "end");
    for (let a = 0; a <= Lmax; a += 2) text(g, X(a), 430, String(a), "mono");
    text(g, 315, 456, "age of the shock", "label");
    text(g, 80, 80, "player 1's action kernel on w⁰", "mono", "start");
    const cls = { 2: "warm", 4: "", 8: "accent" }, tcl = { 2: "warmt", 4: "", 8: "acc" };
    W.forEach((w, j) => {
      const a = w.res.samples.age, v = w.res.samples.kernels.D1.w0, L = w.L;
      el("path", { d: poly(a.map((x, i) => [X(x), Y(v[i])])), class: "pencil " + (cls[L] || ""), "stroke-width": 2 }, g);
      const tail = a.map((x, i) => [x, v[i]]).filter(([x]) => x >= 0.9 * L);
      el("path", { d: poly(tail.map(([x, y]) => [X(x), Y(y)])), class: "pencil " + (cls[L] || ""), "stroke-width": 6, opacity: 0.35 }, g);
      el("line", { x1: X(L), x2: X(L), y1: Y(v[v.length - 1]) - 8, y2: Y(v[v.length - 1]) + 8, class: "pencil " + (cls[L] || ""), "stroke-width": 1.4 }, g);
      text(g, 80, 486 + 20 * j, `L = ${L}: window check ${(100 * w.res.window_tail).toFixed(2)}%, player 1's cost ${w.res.costs.player1.toFixed(4)}`, "label " + (tcl[L] || ""), "start");
    });
    D.forEach((w, j) => text(g, 80, 486 + 20 * (W.length + j), `L = ${w.L}, ρ = ${w.rho}: window check ${(100 * w.res.window_tail).toFixed(2)}%`, "mono", "start"));
    text(g, 80, 486 + 20 * (W.length + D.length), "thick: the last tenth of each window", "mono", "start");
  });

  // ------------------------------------------------------------------ 11. a transition
  scenes.transition = live("transition", () => R.transition, (g) => {
    const res = R.transition, tr = res.transition, L = res.window, T = res.T;
    const times = tr.times, path = tr.loss_path.player1, old = tr.old_flows.player1, neu = tr.new_flows.player1;
    const all = path.concat([old, neu]), lo = Math.min(...all), hi = Math.max(...all), pad = 0.15 * (hi - lo || 1);
    const t0 = -L, t1 = T + L, X = (t) => 70 + 490 * (t - t0) / (t1 - t0), Y = (v) => 420 - 280 * (v - (lo - pad)) / (hi - lo + 2 * pad);
    el("rect", { x: X(t0), y: 110, width: X(0) - X(t0), height: 320, fill: "currentColor", opacity: 0.06 }, g);
    el("rect", { x: X(T), y: 110, width: X(t1) - X(T), height: 320, class: "fillacc", opacity: 0.08 }, g);
    text(g, (X(t0) + X(0)) / 2, 100, "old regime", "mono");
    text(g, (X(0) + X(T)) / 2, 100, "solved on [0, T]", "mono");
    text(g, (X(T) + X(t1)) / 2, 100, "buffer", "mono acc");
    el("line", { x1: X(t0), x2: X(0), y1: Y(old), y2: Y(old), class: "pencil", "stroke-width": 2 }, g);
    el("line", { x1: X(0), x2: X(t1), y1: Y(neu), y2: Y(neu), class: "pencil soft", "stroke-width": 1.2, "stroke-dasharray": "5 5" }, g);
    text(g, X(t1), Y(neu) + 18, "new stationary loss", "mono", "end");
    const dpath = poly(times.map((t, i) => [X(t), Y(path[i])]));
    const pl = stroke(g, dpath, "accent", 2.2);
    pl.set(1);
    for (const t of [t0, 0, T - L, T, t1]) text(g, X(t), 452, t === t0 ? `−${L}` : String(+t.toFixed(2)), "mono");
    text(g, 315, 478, "date", "label");
    text(g, 70, 74, "player 1's loss per unit time", "mono", "start");
    // the range of the settled check
    const by = 520;
    stroke(g, poly([[X(T - L), by - 10], [X(T - L), by], [X(T), by], [X(T), by - 10]]), "", 1.4).set(1);
    const ok = (res.checks.find((c) => c.name === "settled") || {}).ok;
    text(g, (X(T - L) + X(T)) / 2, by + 24, `settled ${sci(res.settled)}, threshold 10⁻⁴: ${ok ? "passes" : "fails"}`, "label " + (ok ? "acc" : "warmt"));
  });

  // ------------------------------------------------------------------ scroll to scene, and the progress line
  const bar = document.getElementById("progress");
  let active = null, prog2 = 0;
  function measure() {
    const vh = window.innerHeight;
    let best = null, bestD = Infinity;
    for (const s of steps) { const r = s.getBoundingClientRect(), d = Math.abs(r.top + r.height / 2 - (window.readLine ? window.readLine() : vh * 0.55)); if (d < bestD) { bestD = d; best = s; } }
    if (bar) { const h = document.documentElement.scrollHeight - vh; bar.style.width = (h > 0 ? (100 * window.scrollY) / h : 0) + "%"; }
    if (!best) return;
    const r = best.getBoundingClientRect();
    prog2 = clamp((vh * 0.85 - r.top) / (r.height * 0.9));
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
    if (active && story.top < window.innerHeight && story.bottom > 0) { const sc = scenes[active.dataset.scene]; if (sc) sc.update(reduced ? 1 : prog2, now); }
    requestAnimationFrame(frame);
  })(performance.now());
})();
