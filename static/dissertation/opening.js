// /dissertation: the cover and the scroll-told opening. Everything is drawn here, in SVG, with a pencil that
// wobbles a little: the cover's faint lines are Brownian paths, the primitive shocks; the story's drawing
// changes with each paragraph (a corner, a pencil, Smith's exchange, Hayek's prices, a used car, the loop; then,
// after the detour map of detour.js, a surface, the noise-state, the wedge and the pencil again),
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
  const steps = [...document.querySelectorAll(".story .step")];
  if (!svg || !steps.length) return;
  // the story comes in two parts around the detour map, each with its own drawing
  const svgFor = (name) => { const st = document.querySelector(`.story .step[data-scene="${name}"]`); return (st && st.closest(".story").querySelector(".stage svg")) || svg; };

  const scenes = {};
  const group = (name) => { const g = el("g", { class: "scene", "data-scene": name }, svgFor(name)); g.style.opacity = 0; g.style.transition = "opacity 0.6s"; return g; };

  // -- a pencil, made of parts from four corners of the world. It comes back at the end of the story, whole,
  // writing: the stories of Hayek and I, Pencil put to paper.
  function makePencil(name, finale) {
    const g = group(name);
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
    if (finale) {
      // whole from the start, labels and tag put away; the pencil rides the end of a line it draws
      const body = el("g", {}, g);
      parts.forEach((p) => { body.appendChild(p.g); p.lines.forEach((l) => l.set(1)); p.label.remove(); p.lead.remove(); });
      tag.remove();
      const r = mulberry32(77), pts = [];
      let y = 430;
      for (let x = 60; x <= 430; x += 5) { pts.push([x, y]); y += gauss(r) * 5; y = clamp(y, 380, 480); }
      const line = stroke(g, pencil(pts, 78, 0.6), "accent", 1.8);
      return {
        g,
        update(t) {
          const q = seg(t, 0.05, 0.85);
          line.set(q);
          const L = line.a.getTotalLength(), pt = line.a.getPointAtLength(L * q);
          body.setAttribute("transform", `translate(${pt.x},${pt.y}) scale(0.42) rotate(140) translate(-500,-305)`);
        },
      };
    }
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
  }
  scenes.pencil = makePencil("pencil");
  if (document.querySelector('.story .step[data-scene="finale"]')) scenes.finale = makePencil("finale", true);



  // -- a person, drawn small: a head and shoulders
  function person(g, x, y, sc, cls) {
    const p = el("g", { transform: `translate(${x},${y}) scale(${sc || 1})` }, g);
    el("path", { d: "M-15,22 Q-15,6 0,6 Q15,6 15,22", class: "box pencil " + (cls || ""), "stroke-width": 1.6 }, p);
    el("circle", { cx: 0, cy: -6, r: 9, class: "box pencil " + (cls || ""), "stroke-width": 1.6 }, p);
    return p;
  }

  // -- each of us sees a corner, and watches the others
  scenes.corner = (() => {
    const g = group("corner"), r = mulberry32(3), folk = [];
    const spots = [[150, 150], [300, 110], [455, 160], [110, 300], [300, 300], [490, 305], [160, 455], [320, 480], [465, 440]];
    spots.forEach(([x, y], i) => {
      const j = [x + (r() - 0.5) * 24, y + (r() - 0.5) * 24];
      const ring = el("circle", { cx: j[0], cy: j[1] + 6, r: 44, class: "pencil soft", "stroke-dasharray": "2 5" }, g);
      const lit = el("circle", { cx: j[0], cy: j[1] + 6, r: 44, class: i === 4 ? "fillacc" : "" }, g);
      lit.style.fill = i === 4 ? "" : "currentColor";
      folk.push({ x: j[0], y: j[1], ring, lit, p: person(g, j[0], j[1], 1, i === 4 ? "accent" : "") });
    });
    const me = folk[4], looks = [0, 2, 3, 5, 7].map((k, n) => {
      const o = folk[k], dx = o.x - me.x, dy = o.y - me.y, L = Math.hypot(dx, dy);
      return stroke(g, pencil([[me.x + dx / L * 30, me.y + dy / L * 30], [o.x - dx / L * 30, o.y - dy / L * 30]], 30 + n, 1), "accent", 1.2);
    });
    looks.forEach((l) => { l.a.style.strokeDasharray = "4 5"; l.b.style.display = "none"; });
    const lab = text(g, 300, 580, "each sees a corner, and watches what the others do", "mono");
    return {
      g,
      update(t) {
        folk.forEach((f, i) => { fade(f.p, seg(t, 0.02 + i * 0.025, 0.12 + i * 0.025)); fade(f.ring, seg(t, 0.2, 0.35) * 0.9);
          fade(f.lit, seg(t, 0.22, 0.4) * (f === me ? 0.14 : 0.05)); });
        looks.forEach((l, k) => { l.a.style.strokeDashoffset = 0; fade(l.a, seg(t, 0.45 + k * 0.06, 0.55 + k * 0.06)); });
        fade(lab, seg(t, 0.6, 0.8));
      },
    };
  })();

  // -- Smith: exchange around a price that no one sets
  scenes.smith = (() => {
    const g = group("smith"), n = 6, folk = [], arcs = [];
    for (let i = 0; i < n; ++i) {
      const a = -Math.PI / 2 + i * 2 * Math.PI / n;
      folk.push(person(g, 300 + 185 * Math.cos(a), 300 + 185 * Math.sin(a), 1.25));
      const a0 = a + 0.36, a1 = a + 2 * Math.PI / n - 0.36;
      const pts = circlePts(300, 300, 198, a0, a1, 10);
      const s = stroke(g, pencil(pts, 60 + i, 1), "", 1.5);
      const [ex, ey] = pts[pts.length - 1], [px, py] = pts[pts.length - 3], ang = Math.atan2(ey - py, ex - px);
      const head = el("path", { d: `M${ex - 9 * Math.cos(ang - 0.45)},${ey - 9 * Math.sin(ang - 0.45)} L${ex},${ey} L${ex - 9 * Math.cos(ang + 0.45)},${ey - 9 * Math.sin(ang + 0.45)}`, class: "pencil", "stroke-width": 1.5 }, g);
      arcs.push({ s, head });
    }
    const tag = el("g", {}, g);
    stroke(tag, pencil([[300, 238], [300, 262]], 70, 0.4), "soft", 1).set(1);
    el("circle", { cx: 300, cy: 236, r: 3.5, class: "fillacc" }, tag);
    el("path", { d: "M252,262 L348,262 L348,330 L252,330 Z", class: "box pencil", "stroke-width": 1.6 }, tag);
    text(tag, 300, 304, "price");
    const lab = text(g, 300, 368, "set by no one", "mono");
    return {
      g,
      update(t, now) {
        folk.forEach((f, i) => fade(f, seg(t, 0.02 + i * 0.04, 0.14 + i * 0.04)));
        arcs.forEach((a, i) => { a.s.set(seg(t, 0.25 + i * 0.05, 0.4 + i * 0.05)); fade(a.head, seg(t, 0.38 + i * 0.05, 0.42 + i * 0.05)); });
        fade(tag, seg(t, 0.55, 0.7)); tag.setAttribute("transform", `rotate(${Math.sin(now / 800) * 4} 300 236)`);
        fade(lab, seg(t, 0.7, 0.85));
      },
    };
  })();

  // -- the used car: the seller knows more, and shapes what you learn
  scenes.car = (() => {
    const g = group("car");
    const body = stroke(g, pencil([[150, 430], [150, 392], [205, 388], [245, 345], [365, 345], [405, 388], [455, 395], [455, 430], [150, 430]], 80, 1.2), "", 2);
    const win = stroke(g, pencil([[255, 385], [285, 355], [345, 355], [375, 385], [255, 385]], 81, 0.8), "soft", 1.3);
    const wheels = [210, 395].map((x) => el("circle", { cx: x, cy: 432, r: 22, class: "box pencil", "stroke-width": 2 }, g));
    const seller = person(g, 95, 330, 1.5), buyer = person(g, 510, 330, 1.5);
    text(seller, 0, 48, "seller", "label"); text(buyer, 0, 48, "you", "label");
    const bub = el("g", {}, g);
    el("path", { d: "M60,250 Q60,215 125,215 Q195,215 195,250 Q195,282 125,282 L112,300 L108,282 Q60,282 60,250 Z", class: "box pencil", "stroke-width": 1.5 }, bub);
    text(bub, 127, 256, "runs great", "label");
    const bell = (m) => { const p = []; for (let i = 0; i <= 40; ++i) { const x = 400 + 190 * i / 40; p.push([x, 250 - 90 * Math.exp(-((x - m) ** 2) / (2 * 26 ** 2))]); } return p; };
    const axis = el("line", { x1: 395, y1: 250, x2: 595, y2: 250, class: "pencil soft", "stroke-width": 1 }, g);
    const old = el("path", { d: pencil(bell(470), 82, 0.2), class: "pencil soft", "stroke-width": 1.4, "stroke-dasharray": "4 4" }, g);
    const now_ = el("path", { class: "pencil accent", "stroke-width": 2 }, g);
    const bl = text(g, 495, 130, "what you believe it’s worth", "mono");
    const lab = text(g, 300, 540, "you pay what you believe; the seller shapes what you learn", "mono");
    return {
      g,
      update(t) {
        body.set(seg(t, 0.02, 0.25)); win.set(seg(t, 0.15, 0.3)); wheels.forEach((w) => fade(w, seg(t, 0.2, 0.3)));
        fade(seller, seg(t, 0.25, 0.35)); fade(buyer, seg(t, 0.3, 0.4));
        fade(axis, seg(t, 0.35, 0.45)); fade(old, seg(t, 0.35, 0.45)); fade(bl, seg(t, 0.38, 0.5));
        fade(bub, seg(t, 0.5, 0.6));
        now_.setAttribute("d", pencil(bell(470 + 55 * seg(t, 0.58, 0.85)), 83, 0.2)); fade(now_, seg(t, 0.55, 0.62));
        fade(lab, seg(t, 0.75, 0.9));
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
    const lab = text(g, 300, 578, "prices carry what the rest of the world needs them to know", "mono");
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

  // -- the loop
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
    const chordLab = text(g, 300, 284, "chord", "mono");
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

  // -- Art Moore's point: a PDE is harder because it can describe so much more
  scenes.moore = (() => {
    const g = group("moore"), N = 13, curves = [];
    const f = (x, k) => 60 * Math.sin(x / 55 + k * 0.45) * Math.exp(-((x - 200) ** 2) / 26000) + 18 * Math.sin(x / 23 - k * 0.8);
    for (let k = 0; k < N; ++k) {
      const pts = []; for (let x = 60; x <= 380; x += 8) pts.push([x + k * 11, 300 - f(x, k) - k * 9 + 60]);
      curves.push(stroke(g, pencil(pts, 90 + k, 0.5), k === 0 ? "accent" : "", k === 0 ? 2.2 : 1.1));
    }
    const l1 = text(g, 150, 470, "an ODE: one path", "mono"), l2 = text(g, 420, 170, "a PDE: a whole surface", "mono");
    return {
      g,
      update(t) {
        curves[0].set(seg(t, 0.02, 0.3)); fade(l1, seg(t, 0.2, 0.35));
        for (let k = 1; k < N; ++k) curves[k].set(seg(t, 0.3 + k * 0.03, 0.45 + k * 0.03));
        fade(l2, seg(t, 0.65, 0.8));
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
      const mid = r.top + r.height / 2, d = Math.abs(mid - (window.readLine ? window.readLine() : vh * 0.55));
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
    const story = active && active.closest(".story").getBoundingClientRect();
    if (active && story.top < window.innerHeight && story.bottom > 0) {
      const sc = scenes[active.dataset.scene];
      if (sc) sc.update(reduced ? 1 : prog, now, reduced ? 0 : dt);
    }
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
})();
