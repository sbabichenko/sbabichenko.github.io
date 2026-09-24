// /dissertation: the cover and the scroll-told opening. Everything is drawn here, in SVG, with a pencil that
// wobbles a little: the cover's faint lines are Brownian paths, the primitive shocks; the story's drawing
// changes with each paragraph (a pencil, a network of prices, the loop, its cuts, the noise-state, the wedge),
// and each drawing builds as its paragraph scrolls past.
(function () {
  "use strict";
  const NS = "http://www.w3.org/2000/svg";
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const el = (tag, attrs, parent) => {
    const e = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs || {})) e.setAttribute(k, v);
    if (parent) parent.appendChild(e);
    return e;
  };
  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function gauss(r) { let u = 0; while (u === 0) u = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r()); }
  const clamp = (x, a = 0, b = 1) => Math.max(a, Math.min(b, x));
  const ease = (t) => (t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2);
  const seg = (t, a, b) => ease(clamp((t - a) / (b - a)));   // 0 before a, 1 after b

  // a pencil line through points: each segment bowed a little, deterministically, so it reads as drawn
  function pencil(pts, seed, wob = 1.2) {
    const r = mulberry32(seed);
    let d = `M${pts[0][0].toFixed(1)},${pts[0][1].toFixed(1)}`;
    for (let i = 1; i < pts.length; ++i) {
      const [x0, y0] = pts[i - 1], [x1, y1] = pts[i];
      const mx = (x0 + x1) / 2 + (r() - 0.5) * wob * 2, my = (y0 + y1) / 2 + (r() - 0.5) * wob * 2;
      d += ` Q${mx.toFixed(1)},${my.toFixed(1)} ${x1.toFixed(1)},${y1.toFixed(1)}`;
    }
    return d;
  }
  // a line drawn twice, the second faint and a hair off, as a pencil goes over its own stroke
  function stroke(g, d, cls, w = 1.6) {
    const a = el("path", { d, class: "pencil " + (cls || ""), "stroke-width": w }, g);
    const b = el("path", { d, class: "pencil soft " + (cls || ""), "stroke-width": w * 0.6, transform: "translate(0.8,0.6)" }, g);
    const len = a.getTotalLength ? a.getTotalLength() : 1000;
    for (const p of [a, b]) { p.style.strokeDasharray = `${len} ${len}`; p.style.strokeDashoffset = len; }
    return { set(t) { const o = len * (1 - clamp(t)); a.style.strokeDashoffset = o; b.style.strokeDashoffset = o; }, a, b, len };
  }
  const circlePts = (cx, cy, r, a0, a1, n = 24) => { const p = []; for (let i = 0; i <= n; ++i) { const a = a0 + (a1 - a0) * i / n; p.push([cx + r * Math.cos(a), cy + r * Math.sin(a)]); } return p; };
  function text(g, x, y, s, cls, anchor = "middle") { const t = el("text", { x, y, class: cls || "", "text-anchor": anchor }, g); t.textContent = s; return t; }
  const fade = (node, t) => { node.style.opacity = clamp(t); };

  // ------------------------------------------------------------------ the cover: primitive shocks, drawn slowly
  const cover = document.querySelector(".cover");
  if (cover) {
    requestAnimationFrame(() => cover.classList.add("in"));
    const sv = document.getElementById("shocks");
    let paths = [], W = 0, H = 0, seed = 11;
    const walk = (r) => {
      const pts = [], n = 170;
      let y = H * (0.18 + 0.66 * r());
      for (let i = 0; i <= n; ++i) { pts.push([W * i / n, y]); y += gauss(r) * H * 0.011; y = clamp(y, H * 0.06, H * 0.96); }
      return pts;
    };
    function lay() {
      W = sv.clientWidth; H = sv.clientHeight;
      sv.setAttribute("viewBox", `0 0 ${W} ${H}`);
      sv.innerHTML = "";
      paths = [];
      const r = mulberry32(seed++);
      for (let k = 0; k < 7; ++k) {
        const acc = k === 2;
        const p = stroke(sv, pencil(walk(r), k + 3, 0.6), acc ? "accent" : "", acc ? 1.5 : 1.1);
        p.a.style.opacity = acc ? 0.45 : 0.26; p.b.style.opacity = acc ? 0.22 : 0.1;
        paths.push({ p, k, speed: 0.1 + 0.06 * r(), t: -r() * 0.5 - k * 0.12, acc });
      }
    }
    lay();
    // each path is drawn across the page, lingers, fades, and is drawn again as a new path: shocks keep arriving
    function renew(q) {
      q.p.a.remove(); q.p.b.remove();
      const r = mulberry32(seed++ * 97 + q.k);
      q.p = stroke(sv, pencil(walk(r), seed, 0.6), q.acc ? "accent" : "", q.acc ? 1.5 : 1.1);
      q.t = 0;
    }
    let last = performance.now();
    function tick(now) {
      const dt = Math.min(0.05, (now - last) / 1000); last = now;
      if (!reduced && cover.getBoundingClientRect().bottom > 0) for (const q of paths) {
        q.t += q.speed * dt; q.p.set(q.t);
        const f = q.t < 1.25 ? 1 : Math.max(0, 1 - (q.t - 1.25) / 0.3);
        q.p.a.style.opacity = (q.acc ? 0.45 : 0.26) * f; q.p.b.style.opacity = (q.acc ? 0.22 : 0.1) * f;
        if (q.t > 1.55) renew(q);
      }
      requestAnimationFrame(tick);
    }
    if (reduced) paths.forEach((q) => q.p.set(1)); else requestAnimationFrame(tick);
    let rt; window.addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(() => { lay(); if (reduced) paths.forEach((q) => q.p.set(1)); }, 200); });
  }

  // ------------------------------------------------------------------ the story
  const svg = document.getElementById("stage");
  const steps = [...document.querySelectorAll(".step")];
  if (!svg || !steps.length) return;

  const scenes = {};
  const group = (name) => { const g = el("g", { class: "scene", "data-scene": name }, svg); g.style.opacity = 0; g.style.transition = "opacity 0.6s"; return g; };

  // -- a pencil, made of parts from four corners of the world
  scenes.pencil = (() => {
    const g = group("pencil");
    const parts = [
      { name: "cedar", from: [-170, -160], lx: 150, ly: 150, build(pg) {
          return [stroke(pg, pencil([[170, 283], [440, 283]], 1), "", 2), stroke(pg, pencil([[170, 327], [440, 327]], 2), "", 2),
                  stroke(pg, pencil([[170, 305], [440, 305]], 3), "soft", 1)]; } },
      { name: "graphite", from: [170, -170], lx: 480, ly: 170, build(pg) {
          return [stroke(pg, pencil([[440, 283], [500, 305], [440, 327]], 4), "", 2), stroke(pg, pencil([[484, 299], [500, 305], [484, 311]], 5), "accent", 2.4)]; } },
      { name: "lacquer", from: [-160, 170], lx: 150, ly: 470, build(pg) {
          const f = el("rect", { x: 171, y: 284, width: 268, height: 42, class: "fillacc", opacity: 0.1 }, pg);
          return [{ set: (t) => fade(f, t * 0.16) }, stroke(pg, pencil([[176, 294], [434, 294]], 6), "accent soft", 1)]; } },
      { name: "rubber", from: [170, 170], lx: 470, ly: 470, build(pg) {
          return [stroke(pg, pencil([[170, 283], [140, 283], [140, 327], [170, 327]], 7), "", 2),
                  stroke(pg, pencil([[150, 283], [150, 327]], 8), "soft", 1), stroke(pg, pencil([[160, 283], [160, 327]], 9), "soft", 1),
                  stroke(pg, pencil([[140, 286], [116, 288], [112, 305], [116, 322], [140, 324]], 10), "warm", 2)]; } },
    ];
    for (const p of parts) {
      p.g = el("g", {}, g);
      p.lines = p.build(p.g);
      p.label = text(g, p.lx, p.ly, p.name, "label");
      p.lead = el("line", { x1: p.lx, y1: p.ly + (p.ly < 300 ? 8 : -20), x2: p.lx, y2: p.ly + (p.ly < 300 ? 8 : -20), class: "pencil soft", "stroke-dasharray": "3 4" }, g);
    }
    const tag = el("g", {}, g);
    stroke(tag, pencil([[300, 327], [310, 370]], 11), "soft", 1).set(1);
    el("rect", { x: 250, y: 370, width: 124, height: 34, rx: 4, class: "box pencil", "stroke-width": 1.4 }, tag);
    text(tag, 312, 392, "almost nothing", "mono");
    return {
      g,
      update(t) {
        parts.forEach((p, i) => {
          const k = seg(t, 0.05 + 0.1 * i, 0.4 + 0.1 * i);
          p.g.setAttribute("transform", `translate(${p.from[0] * (1 - k)},${p.from[1] * (1 - k)})`);
          p.lines.forEach((l) => l.set(seg(t, 0.02 + 0.1 * i, 0.3 + 0.1 * i)));
          fade(p.label, seg(t, 0.02 + 0.1 * i, 0.2 + 0.1 * i));
          const cx = (p.ly < 300 ? 305 : 305) + p.from[0] * (1 - k) * 0.4;
          p.lead.setAttribute("x2", p.lx + (cx - p.lx) * 0.35); p.lead.setAttribute("y2", p.ly + (305 - p.ly) * 0.45);
          fade(p.lead, seg(t, 0.2, 0.5) * 0.8);
        });
        tag.setAttribute("transform", `rotate(${Math.sin(performance.now() / 700) * 3} 310 370)`);
        fade(tag, seg(t, 0.72, 0.9));
      },
    };
  })();

  // -- prices: pulses crossing a network of people who each see their own corner
  scenes.prices = (() => {
    const g = group("prices"), r = mulberry32(5), nodes = [], edges = [];
    for (let gx = -4; gx <= 4; ++gx) for (let gy = -4; gy <= 4; ++gy) {
      const x = 300 + gx * 56 + (r() - 0.5) * 34, y = 300 + gy * 56 + (r() - 0.5) * 34;
      if (Math.hypot(x - 300, y - 300) < 240 && r() < 0.62) nodes.push({ x, y, hit: -1e9 });
    }
    const d2 = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
    const inTree = new Set([0]);                                  // Prim's tree: everyone is reachable
    while (inTree.size < nodes.length) {
      let best = null;
      for (const i of inTree) nodes.forEach((m, j) => { if (!inTree.has(j) && (!best || d2(nodes[i], m) < best[0])) best = [d2(nodes[i], m), i, j]; });
      edges.push([best[1], best[2]]); inTree.add(best[2]);
    }
    nodes.forEach((n, i) => {                                     // and a few shortcuts, so it is a network, not a tree
      const j = nodes.map((m, k) => [d2(n, m), k]).filter(([, k]) => k !== i).sort((a, b) => a[0] - b[0])[1][1];
      if (r() < 0.45 && !edges.some((e) => (e[0] === i && e[1] === j) || (e[0] === j && e[1] === i))) edges.push([i, j]);
    });
    const lines = edges.map(([i, j], k) => stroke(g, pencil([[nodes[i].x, nodes[i].y], [nodes[j].x, nodes[j].y]], 20 + k, 2), "soft", 1));
    for (const n of nodes) {
      n.ring = el("circle", { cx: n.x, cy: n.y, r: 24, class: "pencil soft", "stroke-dasharray": "2 4" }, g);
      n.dot = el("circle", { cx: n.x, cy: n.y, r: 3.6, class: "fillacc" }, g);
      n.dot.style.fill = "currentColor";
    }
    const adj = nodes.map(() => []);
    edges.forEach(([i, j]) => { adj[i].push(j); adj[j].push(i); });
    const pulses = el("g", {}, g), live = [];
    let lastWave = 0;
    const lab = text(g, 300, 578, "prices, passing through people who see only their own corner", "mono");
    return {
      g,
      update(t, now) {
        lines.forEach((l, k) => l.set(seg(t, 0.02 + (k % 10) * 0.02, 0.35 + (k % 10) * 0.02)));
        nodes.forEach((n, i) => { fade(n.ring, seg(t, 0, 0.2) * 0.8); fade(n.dot, seg(t, 0, 0.15)); });
        fade(lab, seg(t, 0.3, 0.5));
        if (t > 0.3 && now - lastWave > 1600) {                    // a price moves somewhere, and spreads
          lastWave = now;
          const src = Math.floor(Math.random() * nodes.length), seen = new Set([src]);
          let front = [src], depth = 0;
          while (front.length) {
            const next = [];
            for (const i of front) for (const j of adj[i]) if (!seen.has(j)) { seen.add(j); next.push(j); live.push({ i, j, t0: now + depth * 420 }); }
            front = next; depth++;
          }
          nodes[src].hit = now;
        }
        pulses.innerHTML = "";
        for (let k = live.length - 1; k >= 0; --k) {
          const q = live[k], u = (now - q.t0) / 420;
          if (u > 1) { nodes[q.j].hit = q.t0 + 420; live.splice(k, 1); continue; }
          if (u < 0) continue;
          const a = nodes[q.i], b = nodes[q.j];
          el("circle", { cx: a.x + (b.x - a.x) * u, cy: a.y + (b.y - a.y) * u, r: 3, class: "fillacc" }, pulses);
        }
        for (const n of nodes) {
          const h = clamp(1 - (now - n.hit) / 700);
          n.dot.setAttribute("r", 3.6 + 3 * h);
          n.dot.style.fill = h > 0.02 ? "var(--accent)" : "currentColor";
        }
      },
    };
  })();

  // -- the loop, drawn once and reused by the next three scenes
  const C = [300, 300], R = 178;
  const POS = { actions: Math.PI, state: -Math.PI / 2, observations: 0, beliefs: Math.PI / 2 };
  const at = (a, r = R) => [C[0] + r * Math.cos(a), C[1] + r * Math.sin(a)];
  function loopDrawing(g, seed) {
    const arcs = [["actions", "state"], ["state", "observations"], ["observations", "beliefs"], ["beliefs", "actions"]].map(([a, b], k) => {
      let a0 = POS[a] + 0.3, a1 = POS[b] - 0.3;
      if (a1 < a0) a1 += Math.PI * 2;
      const pts = circlePts(C[0], C[1], R, a0, a1, 18);
      const s = stroke(g, pencil(pts, seed + k, 1.4), "", 1.8);
      const [ex, ey] = pts[pts.length - 1], [px, py] = pts[pts.length - 3];
      const ang = Math.atan2(ey - py, ex - px);
      const head = el("path", { d: `M${ex - 11 * Math.cos(ang - 0.45)},${ey - 11 * Math.sin(ang - 0.45)} L${ex},${ey} L${ex - 11 * Math.cos(ang + 0.45)},${ey - 11 * Math.sin(ang + 0.45)}`, class: "pencil", "stroke-width": 1.8 }, g);
      return { s, head };
    });
    const [ax, ay] = at(POS.actions, R - 58), [ox, oy] = at(POS.observations, R - 78);
    const chord = stroke(g, pencil([[ax, ay], [ox, oy]], seed + 9, 1), "accent", 1.5);
    chord.a.style.strokeDasharray = "5 6"; chord.b.style.display = "none";
    const chordLab = text(g, 300, 284, "direct channel", "mono");
    const nodes = Object.entries(POS).map(([name, a]) => {
      const [x, y] = at(a), w = name.length * 9.2 + 26;
      const n = el("g", {}, g);
      el("rect", { x: x - w / 2, y: y - 19, width: w, height: 38, rx: 6, class: "box pencil", "stroke-width": 1.6 }, n);
      text(n, x, y + 6, name);
      return n;
    });
    return {
      set(t) {
        nodes.forEach((n, i) => fade(n, seg(t, i * 0.12, i * 0.12 + 0.15)));
        arcs.forEach((a, i) => { a.s.set(seg(t, 0.12 + i * 0.12, 0.3 + i * 0.12)); fade(a.head, seg(t, 0.28 + i * 0.12, 0.32 + i * 0.12)); });
        chord.a.style.strokeDashoffset = 0; fade(chord.a, seg(t, 0.72, 0.85)); fade(chordLab, seg(t, 0.78, 0.9));
      },
    };
  }
  const token = (g) => el("circle", { r: 7, class: "fillacc" }, g);
  const tokenAt = (tok, a, r = R) => { const [x, y] = at(a, r); tok.setAttribute("cx", x); tok.setAttribute("cy", y); };

  scenes.loop = (() => {
    const g = group("loop"), d = loopDrawing(g, 40), tok = token(g);
    let ang = Math.PI;
    return { g, update(t, now, dt) { d.set(t); fade(tok, seg(t, 0.62, 0.7)); ang += dt * 1.1; tokenAt(tok, ang); } };
  })();

  scenes.cuts = (() => {
    const g = group("cuts"), d = loopDrawing(g, 40), tok = token(g);
    d.set(1);
    const cuts = [
      { a: POS.beliefs, r: R, label: "the privacy of beliefs", lx: 300, ly: 548 },
      { chord: true, label: "what actions can teach", lx: 300, ly: 330 },
      { a: POS.actions, r: R, label: "the weight of one player", lx: 92, ly: 244 },
      { a: Math.PI / 4, r: R, label: "observations → beliefs", lx: 486, ly: 468 },
      { a: -3 * Math.PI / 4, r: R, label: "time itself", lx: 118, ly: 118 },
    ].map((c, k) => {
      const [x, y] = c.chord ? [300, 300] : at(c.a, c.r);
      const m = el("g", {}, g);
      el("path", { d: `M${x - 10},${y - 10} L${x + 10},${y + 10} M${x + 10},${y - 10} L${x - 10},${y + 10}`, class: "pencil warm", "stroke-width": 3 }, m);
      text(m, c.lx, c.ly, c.label, "label warmt");
      return m;
    });
    let ang = Math.PI;
    return {
      g,
      update(t, now, dt) {
        cuts.forEach((m, i) => fade(m, seg(t, 0.08 + i * 0.13, 0.2 + i * 0.13)));
        // the token runs until the first cut, then stops at it and shivers
        if (t < 0.1) ang += dt * 1.1;
        const stop = POS.actions + Math.PI * 2 * Math.ceil((ang - POS.actions) / (Math.PI * 2)) - 0.18;
        if (t >= 0.1) ang += Math.min(dt * 1.1, Math.max(0, stop - ang));
        tokenAt(tok, ang + (t >= 0.1 ? Math.sin(now / 45) * 0.006 : 0));
      },
    };
  })();

  scenes.intact = (() => {
    const g = group("intact"), d = loopDrawing(g, 40);
    d.set(1);
    const glow = el("circle", { cx: C[0], cy: C[1], r: R, class: "pencil accent", "stroke-width": 7 }, g);
    const trail = [];
    for (let i = 0; i < 9; ++i) trail.push(el("circle", { r: 7 - i * 0.6, class: "fillacc" }, g));
    const stitches = [POS.beliefs, POS.actions, Math.PI / 4, -3 * Math.PI / 4].map((a) => {
      const [x, y] = at(a); const s = el("g", {}, g);
      for (const o of [-5, 5]) el("line", { x1: x + o - 3, y1: y - 9, x2: x + o + 3, y2: y + 9, class: "pencil accent", "stroke-width": 1.6 }, s);
      return s;
    });
    const lab = text(g, 300, 590, "intact: a belief can be a price, and a price can be moved", "mono");
    let ang = 0;
    return {
      g,
      update(t, now, dt) {
        fade(glow, seg(t, 0.35, 0.7) * (0.14 + 0.06 * Math.sin(now / 500)));
        stitches.forEach((s, i) => fade(s, seg(t, 0.05 + i * 0.05, 0.2 + i * 0.05) * (1 - seg(t, 0.6, 0.85))));
        ang += dt * (1.1 + 1.6 * seg(t, 0.3, 0.8));
        trail.forEach((c, i) => { tokenAt(c, ang - i * 0.07); fade(c, (1 - i / 9) * seg(t, 0.2, 0.35)); });
        fade(lab, seg(t, 0.5, 0.7));
      },
    };
  })();

  // -- the noise-state: the shocks, and each player's estimate of them
  scenes.noise = (() => {
    const g = group("noise"), r = mulberry32(21), n = 220, W = [], E1 = [], E2 = [];
    let w = 0, e1 = 0, e2 = 0;
    for (let i = 0; i < n; ++i) {
      w += gauss(r) * 7; w *= 0.985; W.push(w);
      e1 += 0.16 * (w - e1); E1.push(e1);
      e2 += 0.06 * (w - e2) + gauss(r) * 0.6; E2.push(e2);
    }
    const X = (i) => 50 + 490 * i / (n - 1), Y = (v) => 300 - v * 2.6;
    el("line", { x1: 50, y1: 300, x2: 560, y2: 300, class: "pencil soft", "stroke-width": 1 }, g);
    const sW = stroke(g, pencil(W.map((v, i) => [X(i), Y(v)]), 30, 0.3), "", 1.5);
    const s1 = stroke(g, pencil(E1.map((v, i) => [X(i), Y(v)]), 31, 0.2), "accent", 2.2);
    const s2 = stroke(g, pencil(E2.map((v, i) => [X(i), Y(v)]), 32, 0.2), "warm", 2.2);
    const lW = text(g, 0, 0, "W", "", "start"), l1 = text(g, 0, 0, "Ŵ¹", "acc", "start"), l2 = text(g, 0, 0, "Ŵ²", "warmt", "start");
    const cap = text(g, 300, 560, "the primitive shocks W, and two players' estimates of them", "mono");
    return {
      g,
      update(t) {
        const k = seg(t, 0.02, 0.8);
        sW.set(k); s1.set(clamp(k - 0.02)); s2.set(clamp(k - 0.035));
        const i = Math.max(1, Math.round(k * (n - 1)));
        const place = (lab, arr, j, dy) => { lab.setAttribute("x", X(j) + 8); lab.setAttribute("y", Y(arr[j]) + dy); fade(lab, seg(t, 0.1, 0.2)); };
        place(lW, W, i, -8); place(l1, E1, Math.max(0, i - 4), 16); place(l2, E2, Math.max(0, i - 8), 30);
        fade(cap, seg(t, 0.4, 0.6));
      },
    };
  })();

  // -- the wedge: A's action moves B's belief, and the move has a price
  scenes.wedge = (() => {
    const g = group("wedge");
    const bell = (m) => { const p = []; for (let i = 0; i <= 60; ++i) { const x = 110 + 380 * i / 60; p.push([x, 420 - 170 * Math.exp(-((x - m) ** 2) / (2 * 48 ** 2))]); } return p; };
    el("line", { x1: 90, y1: 420, x2: 510, y2: 420, class: "pencil soft", "stroke-width": 1 }, g);
    const before = el("path", { d: pencil(bell(270), 50, 0.3), class: "pencil soft", "stroke-width": 1.6, "stroke-dasharray": "5 5" }, g);
    const area = el("path", { class: "fillacc", opacity: 0.14 }, g);
    const after = el("path", { class: "pencil accent", "stroke-width": 2.2 }, g);
    const A = el("g", {}, g), B = el("g", {}, g);
    el("circle", { cx: 120, cy: 150, r: 26, class: "box pencil", "stroke-width": 1.8 }, A); text(A, 120, 157, "A");
    el("circle", { cx: 480, cy: 150, r: 26, class: "box pencil", "stroke-width": 1.8 }, B); text(B, 480, 157, "B");
    text(B, 480, 205, "B's belief", "label");
    const push = stroke(g, pencil([[150, 160], [230, 205], [300, 238]], 51, 1.2), "warm", 2);
    const tip = el("path", { d: "M288,226 L300,238 L284,242", class: "pencil warm", "stroke-width": 2 }, g);
    const act = text(g, 190, 150, "A acts", "label warmt");
    const wl = text(g, 300, 470, "the wedge: what it is worth to A to move B's belief", "mono");
    const note = text(g, 300, 500, "cut the loop and it is zero", "mono");
    return {
      g,
      update(t, now) {
        fade(A, seg(t, 0, 0.1)); fade(B, seg(t, 0, 0.1)); fade(before, seg(t, 0.02, 0.15));
        push.set(seg(t, 0.15, 0.4)); fade(tip, seg(t, 0.38, 0.42)); fade(act, seg(t, 0.15, 0.3));
        const m = 270 + 70 * seg(t, 0.35, 0.7) + Math.sin(now / 900) * 3 * seg(t, 0.7, 0.8);
        const p0 = bell(270), p1 = bell(m);
        after.setAttribute("d", pencil(p1, 52, 0.2)); fade(after, seg(t, 0.3, 0.4));
        area.setAttribute("d", "M" + p1.map((q) => q.join(",")).join(" L") + " L" + p0.slice().reverse().map((q) => q.join(",")).join(" L") + "Z");
        fade(area, seg(t, 0.45, 0.65) * 0.16);
        fade(wl, seg(t, 0.55, 0.7)); fade(note, seg(t, 0.8, 0.95));
      },
    };
  })();

  // ------------------------------------------------------------------ scroll to scene
  let active = null, prog = 0;
  function measure() {
    const vh = window.innerHeight;
    let best = null, bestD = Infinity;
    for (const s of steps) {
      const r = s.getBoundingClientRect();
      const mid = r.top + r.height / 2, d = Math.abs(mid - vh * 0.55);
      if (d < bestD) { bestD = d; best = s; }
    }
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
  let last = performance.now();
  function frame(now) {
    const dt = Math.min(0.05, (now - last) / 1000); last = now;
    const story = document.getElementById("story").getBoundingClientRect();
    if (active && story.top < window.innerHeight && story.bottom > 0) {
      const sc = scenes[active.dataset.scene];
      if (sc) sc.update(reduced ? 1 : prog, now, reduced ? 0 : dt);
    }
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
})();
