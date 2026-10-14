// Written by tools/diss_slides.py: the site's pencil drawings for the screens of this tab. Do not edit.
(function () {
  const mk = (w, h, cls) => { const s = document.createElementNS("http://www.w3.org/2000/svg", "svg"); s.setAttribute("viewBox", `0 0 ${w} ${h}`); s.setAttribute("class", cls + " di-bench"); s.setAttribute("width", w); s.setAttribute("height", h); s.setAttribute("aria-hidden", "true"); document.body.appendChild(s); return s; };
  window.DissStage = mk(600, 600, "di-sk"); window.DissMapStage = mk(1000, 560, "di-map"); window.DissScenes = {};   // drawn on a bench that is laid out, so lengths can be measured
})();
// ---- from the site's dissertation/opening.js
// /dissertation: the cover and the scroll-told opening. Everything is drawn here, in SVG, with a pencil that
// wobbles a little: the cover's faint lines are Brownian paths, the primitive shocks; the story's drawing
// changes with each paragraph (a corner, a pencil, Smith's exchange, Hayek's prices, a used car, the loop; then,
// after the detour map of detour.js, a surface, the noise-state, the wedge and the pencil again),
// and each drawing builds as its paragraph scrolls past.
(function () {
  "use strict";
  const { el, mulberry32, gauss, clamp, ease, seg, fade, text } = Sketch;   // static/js/sketch.js
  const reduced = false;
  // A one-shot motion (a part sliding home, a line being drawn): once the step's progress passes `at` it plays over
  // `dur` seconds whatever the scroll speed, and runs back if the reader scrolls above `at` again. With reduced
  // motion it jumps to where it is headed. step(t, dt) returns 0..1, not eased.
  const oneShot = (at, dur) => Sketch.oneShot(at, dur, reduced);

  // a pencil line through points: each segment bowed a little, deterministically, so it reads as drawn
  const pencil = (pts, seed, wob = 1.2) => Sketch.pencil(pts, seed, wob);
  // a line drawn twice, the second faint and a hair off, as a pencil goes over its own stroke
  const stroke = (g, d, cls, w = 1.6) => Sketch.stroke2(g, d, cls, w, "pencil", 0.8, 0.6);
  const circlePts = (cx, cy, r, a0, a1, n = 24) => { const p = []; for (let i = 0; i <= n; ++i) { const a = a0 + (a1 - a0) * i / n; p.push([cx + r * Math.cos(a), cy + r * Math.sin(a)]); } return p; };

  // ------------------------------------------------------------------ the cover: primitive shocks, drawn slowly
  const cover = null;
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
    let last = performance.now(), ticking = 0;
    function tick(now) {
      ticking = 0;
      const dt = Math.min(0.05, (now - last) / 1000); last = now;
      if (cover.getBoundingClientRect().bottom <= 0) return;      // scrolled past: asleep until the cover is back
      for (const q of paths) {
        q.t += q.speed * dt; q.p.set(q.t);
        const f = q.t < 1.25 ? 1 : Math.max(0, 1 - (q.t - 1.25) / 0.3);
        q.p.a.style.opacity = (q.acc ? 0.45 : 0.26) * f; q.p.b.style.opacity = (q.acc ? 0.22 : 0.1) * f;
        if (q.t > 1.55) renew(q);
      }
      ticking = requestAnimationFrame(tick);
    }
    if (reduced) paths.forEach((q) => q.p.set(1));
    else {
      ticking = requestAnimationFrame(tick);
      window.addEventListener("scroll", () => { if (!ticking && cover.getBoundingClientRect().bottom > 0) { last = performance.now(); ticking = requestAnimationFrame(tick); } }, { passive: true });
    }
    let rt; window.addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(() => { lay(); if (reduced) paths.forEach((q) => q.p.set(1)); }, 200); });
  }

  // ------------------------------------------------------------------ the story
  const svg = window.DissStage;
  // the story comes in two parts around the detour map, each with its own drawing
  const svgFor = () => svg;

  const scenes = {};
  const group = (name) => { const g = el("g", { class: "scene", "data-scene": name }, svgFor(name)); g.style.opacity = 0; g.style.transition = "opacity 0.6s"; return g; };

  // -- a pencil, made of parts from four corners of the world. It comes back at the end of the story, whole,
  // writing: the stories of Hayek and I, Pencil put to paper.
  function makePencil(name, finale) {
    const g = group(name);
    const parts = [
      { name: "cedar", who: "\u201cthe logger in Oregon\u201d", from: [-170, -160], lx: 150, ly: 150, tx: 236, ty: 283, build(pg) {
          return [stroke(pg, pencil([[170, 283], [440, 283]], 1), "", 2), stroke(pg, pencil([[170, 327], [440, 327]], 2), "", 2),
                  stroke(pg, pencil([[440, 283], [482, 301]], 4), "", 2), stroke(pg, pencil([[440, 327], [482, 309]], 12), "", 2)]; } },   // the body; its cone stops at the lead
      // the graphite is a rod with a point: it flies in from beyond the corner to line up with the pencil beyond its
      // tip, then slides in along the axis, the wood hiding what has gone in, until only the sharpened point shows
      { name: "graphite", who: "\u201cthe miner of graphite in Ceylon\u201d", from: [170, -170], lx: 480, ly: 170, tx: 490, ty: 301,
        path: (k) => (k < 0.55 ? [400 - 100 * (k / 0.55), -330 * (1 - k / 0.55)] : [300 * (1 - (k - 0.55) / 0.45), 0]),
        build(pg) {
          this.rod = el("path", { d: "M172,301 L482,301 L482,309 L172,309 Z", class: "fillink" }, pg);        // the core
          this.point = el("path", { d: "M482,301 L500,305 L482,309 Z", class: "fillink", opacity: 0 }, pg);   // the exposed lead
          this.edge = stroke(pg, pencil([[482, 301], [172, 301], [172, 309], [482, 309]], 3), "", 1.2);
          return [this.edge, this.tip = stroke(pg, pencil([[482, 301], [500, 305], [482, 309]], 5), "", 1.4)]; } },
      { name: "lacquer", who: "\u201cthe growers of castor beans\u201d", from: [160, 170], lx: 470, ly: 440, tx: 392, ty: 326, build(pg) {
          const f = el("rect", { x: 171, y: 284, width: 268, height: 42, class: "fillacc", opacity: 0.1 }, pg);
          return [{ set: (t) => fade(f, t * 0.16) }, stroke(pg, pencil([[176, 294], [434, 294]], 6), "accent soft", 1)]; } },
      { name: "rubber", who: "\u201crapeseed oil from the Dutch East Indies\u201d", from: [-170, 170], lx: 130, ly: 440, tx: 118, ty: 318, build(pg) {
          return [stroke(pg, pencil([[170, 283], [140, 283], [140, 327], [170, 327]], 7), "", 2),
                  stroke(pg, pencil([[150, 283], [150, 327]], 8), "soft", 1), stroke(pg, pencil([[160, 283], [160, 327]], 9), "soft", 1),
                  stroke(pg, pencil([[140, 286], [116, 288], [112, 305], [116, 322], [140, 324]], 10), "warm", 2)]; } },
    ];
    // the wood hides what has gone into it: the graphite is drawn only outside the pencil's body and cone
    const sheath = (into) => { const id = "wood-" + name, c = el("clipPath", { id, clipPathUnits: "userSpaceOnUse" }, into);
      el("path", { d: "M-2000,-2000 H3000 V3000 H-2000 Z M130,280 H440 L482,301 L482,309 L440,330 H130 Z", "clip-rule": "evenodd" }, c);
      return el("g", { "clip-path": `url(#${id})` }, into); };
    for (const p of parts) {
      p.g = el("g", {}, p.name === "graphite" ? sheath(g) : g);
      p.lines = p.build(p.g);
      // in Leonard Read's own words, who or what brings the part (I, Pencil): above the leader at the top, below it
      // at the bottom
      p.label = el("g", {}, g);
      text(p.label, Math.max(135, Math.min(465, p.lx)), p.ly + (p.ly < 300 ? -20 : 20), p.who, "quote");
      p.lead = el("line", { x1: p.lx, y1: p.ly + (p.ly < 300 ? 8 : -20), x2: p.lx, y2: p.ly + (p.ly < 300 ? 8 : -20), class: "pencil soft", "stroke-dasharray": "3 4" }, g);
    }
    const read = el("g", {}, g);                                     // Read's line, under the pencil
    text(read, 300, 530, "\u201cnot a single person on the face of this earth", "quote");
    text(read, 300, 548, "knows how to make me\u201d", "quote");
    text(read, 300, 570, "Leonard Read, I, Pencil (1958)", "src");
    if (finale) {
      // whole from the start, labels and Read's line put away; the pencil rides the end of a line it draws
      const body = el("g", {}, g);
      parts.forEach((p) => { body.appendChild(p.g); p.lines.forEach((l) => l.set(1)); p.label.remove(); p.lead.remove(); });
      parts.forEach((p) => { if (p.rod) { p.rod.setAttribute("opacity", 0); p.point.setAttribute("opacity", 0.7); p.edge.a.style.opacity = p.edge.b.style.opacity = 0; } });
      read.remove();
      const r = mulberry32(77), pts = [];
      let y = 430;
      for (let x = 60; x <= 430; x += 5) { pts.push([x, y]); y += gauss(r) * 5; y = clamp(y, 380, 480); }
      const line = stroke(g, pencil(pts, 78, 0.6), "accent", 1.8);
      const write = oneShot(0.08, 2.6);                              // the line is written at the pencil's pace, not the wheel's
      return {
        g, live: false, shots: [write],
        update(t, now, dt) {
          const q = ease(write.step(t, dt));
          line.set(q);
          const L = line.a.getTotalLength(), pt = line.a.getPointAtLength(L * q);
          body.setAttribute("transform", `translate(${pt.x},${pt.y}) scale(0.42) rotate(140) translate(-500,-305)`);
          return write.moving;
        },
      };
    }
    // the graphite's flight and slide home is one action, timed, so it is seen whole however fast the page is scrolled;
    // it starts once the wood has landed (0.4), so it lines up with the pencil where the pencil is
    const slide = oneShot(0.4, 1.7);
    return {
      g, live: false, shots: [slide],
      update(t, now, dt) {
        const kg = ease(slide.step(t, dt));
        parts.forEach((p, i) => {
          const k = p.path ? kg : seg(t, 0.05 + 0.1 * i, 0.4 + 0.1 * i), off = p.path ? p.path(k) : [p.from[0] * (1 - k), p.from[1] * (1 - k)];
          p.g.setAttribute("transform", `translate(${off[0]},${off[1]})`);
          if (p.rod) {                                                  // solid while it travels
            const seen = seg(k, 0, 0.04);                                // out of sight until it sets off
            p.rod.setAttribute("opacity", 0.7 * seen);
            p.point.setAttribute("opacity", 0.7 * seen);
            p.edge.a.style.opacity = p.edge.b.style.opacity = seen;
            p.tip.a.style.opacity = p.tip.b.style.opacity = seen;
          }
          p.lines.forEach((l) => l.set(seg(t, 0.02 + 0.1 * i, 0.3 + 0.1 * i)));
          fade(p.label, seg(t, 0.02 + 0.1 * i, 0.2 + 0.1 * i));
          // the leader runs from the label to its own part, following the part in, and stops just short of it
          const x1 = +p.lead.getAttribute("x1"), y1 = +p.lead.getAttribute("y1");
          const tx = p.tx + off[0], ty = p.ty + off[1], L = Math.hypot(tx - x1, ty - y1) || 1;
          p.lead.setAttribute("x2", tx - (tx - x1) / L * 7); p.lead.setAttribute("y2", ty - (ty - y1) / L * 7);
          fade(p.lead, seg(t, 0.2, 0.5) * 0.8);
        });
        fade(read, seg(t, 0.72, 0.9));
        return slide.moving;
      },
    };
  }
  scenes.pencil = makePencil("pencil");



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
    const when = []; when[4] = 0.02; [0, 2, 5, 7, 3].forEach((i, k) => { when[i] = 0.26 + k * 0.07; }); [1, 8, 6].forEach((i, k) => { when[i] = 0.62 + k * 0.02; });   // clockwise from the top left
    // (added for the screens) neighbours of neighbours: further out and fainter, each watched by one or two people a step nearer in
    const far = [], ties = [];
    { const rr = mulberry32(17), all = folk.map((f, i) => ({ x: f.x, y: f.y, gen: i === 4 ? 0 : [0, 2, 3, 5, 7].includes(i) ? 1 : -1, own: true }));
      for (let row = 0, y = -70; y <= 680; y += 125, ++row) for (let x = -360 + (row % 2) * 70; x <= 960; x += 140) {
        const px = x + (rr() - 0.5) * 50, py = y + (rr() - 0.5) * 40;
        if (folk.some((f) => Math.hypot(f.x - px, f.y - py) < 150)) continue;   // clear of the nine
        all.push({ x: px, y: py, gen: -1 });
      }
      for (let gen = 1; gen < 7; ++gen) {
        const from = all.filter((a) => a.gen === gen);
        for (const b of all.filter((a) => a.gen === -1)) {
          const near = from.map((a) => [Math.hypot(a.x - b.x, a.y - b.y), a]).filter(([d]) => d < 215).sort((p, q) => p[0] - q[0]).slice(0, 2);
          if (near.length) { b.gen = gen + 1; for (const [, a] of near) ties.push({ a, b }); }
        }
      }
      const LOOK = [0, 0, 0.42, 0.26, 0.17, 0.09, 0.05, 0.03], LINE = [0, 0, 0.45, 0.26, 0.14, 0.07, 0.04, 0.02];
      ties.forEach(({ a, b }, n) => {
        const dx = b.x - a.x, dy = b.y - a.y, L = Math.hypot(dx, dy), s = stroke(g, pencil([[a.x + dx / L * 30, a.y + dy / L * 30], [b.x - dx / L * 30, b.y - dy / L * 30]], 60 + n, 1), "", 1.1);
        s.a.style.strokeDasharray = "4 5"; s.b.style.display = "none"; far.push({ e: s.a, gen: b.gen, top: LINE[b.gen] });
      });
      for (const b of all) if (!b.own && b.gen > 1) {
        far.push({ e: el("circle", { cx: b.x, cy: b.y + 6, r: 44, class: "pencil soft", "stroke-dasharray": "2 5" }, g), gen: b.gen, top: LOOK[b.gen] * 0.9 });
        far.push({ e: person(g, b.x, b.y, 1, ""), gen: b.gen, top: LOOK[b.gen] });
      }
    }
    return {
      g,
      update(t) {
        // (changed for the screens) in the order of the story: the one in the middle and their corner, then each person they
        // watch as the line to them arrives, going round, then the rest of the nine with the first step further out
        folk.forEach((f, i) => { const w = when[i]; fade(f.p, seg(t, w, w + 0.08)); fade(f.ring, seg(t, w + 0.04, w + 0.12) * 0.9);
          fade(f.lit, seg(t, w + 0.06, w + 0.14) * (f === me ? 0.14 : 0.05)); });
        looks.forEach((l, k) => { const w = when[[0, 2, 3, 5, 7][k]]; l.a.style.strokeDashoffset = 0; fade(l.a, seg(t, w - 0.06, w + 0.02)); });
        far.forEach((f) => fade(f.e, seg(t, 0.62 + (f.gen - 2) * 0.08, 0.72 + (f.gen - 2) * 0.08) * f.top));   // one step further out at a time
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
    return {
      g,
      update(t, now) {
        folk.forEach((f, i) => fade(f, seg(t, 0.02 + i * 0.04, 0.14 + i * 0.04)));
        arcs.forEach((a, i) => { a.s.set(seg(t, 0.25 + i * 0.05, 0.4 + i * 0.05)); fade(a.head, seg(t, 0.38 + i * 0.05, 0.42 + i * 0.05)); });
        fade(tag, seg(t, 0.55, 0.7)); tag.setAttribute("transform", `rotate(${Math.sin(now / 800) * 4} 300 236)`);
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
    const think = el("g", {}, g);                                 // what you believe the car is worth, as a thought of yours
    el("path", { d: "M415,148 H570 Q602,148 602,180 V232 Q602,264 570,264 H415 Q383,264 383,232 V180 Q383,148 415,148 Z", class: "box pencil", "stroke-width": 1.5 }, think);
    el("circle", { cx: 507, cy: 278, r: 5, class: "box pencil", "stroke-width": 1.3 }, think);
    el("circle", { cx: 509, cy: 294, r: 3, class: "box pencil", "stroke-width": 1.3 }, think);
    const axis = el("line", { x1: 395, y1: 250, x2: 595, y2: 250, class: "pencil soft", "stroke-width": 1 }, g);
    const old = el("path", { d: pencil(bell(470), 82, 0.2), class: "pencil soft", "stroke-width": 1.4, "stroke-dasharray": "4 4" }, g);
    const now_ = el("path", { class: "pencil accent", "stroke-width": 2 }, g);
    return {
      g,
      update(t) {
        body.set(seg(t, 0.02, 0.25)); win.set(seg(t, 0.15, 0.3)); wheels.forEach((w) => fade(w, seg(t, 0.2, 0.3)));
        fade(seller, seg(t, 0.25, 0.35)); fade(buyer, seg(t, 0.3, 0.4));
        fade(think, seg(t, 0.35, 0.45)); fade(axis, seg(t, 0.35, 0.45)); fade(old, seg(t, 0.35, 0.45));
        fade(bub, seg(t, 0.5, 0.6));
        now_.setAttribute("d", pencil(bell(470 + 55 * seg(t, 0.58, 0.85)), 83, 0.2)); fade(now_, seg(t, 0.55, 0.62));
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
    // a price moves at one person and spreads, link by link, to everyone
    function wave(src, now) {
      lastWave = now;
      const seen = new Set([src]);
      let front = [src], depth = 0;
      while (front.length) {
        const next = [];
        for (const i of front) for (const j of adj[i]) if (!seen.has(j)) { seen.add(j); next.push(j); live.push({ i, j, t0: now + depth * 420 }); }
        front = next; depth++;
      }
      nodes[src].hit = now;
    }
    // unannounced: a click on a person starts a price change there. The targets are live only while this drawing shows.
    const hits = el("g", {}, g);
    hits.style.pointerEvents = "none";
    nodes.forEach((n, i) => {
      const h = el("circle", { cx: n.x, cy: n.y, r: 24, fill: "transparent" }, hits);
      h.style.cursor = "pointer";
      h.addEventListener("pointerdown", (ev) => { ev.preventDefault(); wave(i, performance.now()); });
    });
    // only the showing drawing is updated each frame, so the targets are also switched off here when it is hidden
    new MutationObserver(() => { if (+g.style.opacity < 0.5) hits.style.pointerEvents = "none"; }).observe(g, { attributes: true, attributeFilter: ["style"] });
    // Reduced motion draws this scene once, with no clock, so no wave would ever start. Instead one wave is drawn
    // stopped part-way: it left the leftmost person, the people it has reached are shaded by how many links back
    // the change began (strongest at the front), and a pulse sits halfway along each link it is crossing now.
    function frozenWave() {
      const src = nodes.reduce((b, n, i) => (n.x < nodes[b].x ? i : b), 0), hop = nodes.map(() => -1);
      hop[src] = 0;
      for (let front = [src]; front.length;) {
        const next = [];
        for (const i of front) for (const j of adj[i]) if (hop[j] < 0) { hop[j] = hop[i] + 1; next.push(j); }
        front = next;
      }
      const D = Math.max(2, Math.round(0.45 * Math.max(...hop)));
      pulses.innerHTML = "";
      nodes.forEach((n, i) => {
        if (hop[i] > D) { n.dot.setAttribute("r", 3.6); n.dot.style.fill = "currentColor"; return; }
        const u = hop[i] / D;
        n.dot.setAttribute("r", (i === src ? 6.6 : 3.6 + 2.4 * u).toFixed(1));
        n.dot.style.fill = "var(--accent)";
        n.dot.style.opacity = (0.35 + 0.65 * u).toFixed(2);
      });
      edges.forEach(([i, j]) => {
        const [a, b] = hop[i] === D && hop[j] === D + 1 ? [nodes[i], nodes[j]] : hop[j] === D && hop[i] === D + 1 ? [nodes[j], nodes[i]] : [];
        if (a) el("circle", { cx: (a.x + b.x) / 2, cy: (a.y + b.y) / 2, r: 3, class: "fillacc" }, pulses);
      });
    }
    return {
      g,
      update(t, now) {
        if (reduced) {
          lines.forEach((l) => l.set(1));
          nodes.forEach((n) => { fade(n.ring, 0.8); });
          hits.style.pointerEvents = "none";      // a click would start a wave that nothing then draws
          frozenWave();
          return;
        }
        lines.forEach((l, k) => l.set(seg(t, 0.02 + (k % 10) * 0.02, 0.35 + (k % 10) * 0.02)));
        nodes.forEach((n, i) => { fade(n.ring, seg(t, 0, 0.2) * 0.8); fade(n.dot, seg(t, 0, 0.15)); });
        hits.style.pointerEvents = t > 0.3 && +g.style.opacity > 0.5 ? "auto" : "none";
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
        return live.length > 0 || nodes.some((n) => now - n.hit < 700);   // (added for the screens) a wave is still on its way: keep drawing
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
        chord.a.style.strokeDashoffset = 0; fade(chord.a, seg(t, 0.72, 0.85));
      },
    };
  }
  const token = (g) => el("circle", { r: 7, class: "fillacc" }, g);
  const tokenAt = (tok, a, r = R) => { const [x, y] = at(a, r); tok.setAttribute("cx", x); tok.setAttribute("cy", y); };

  scenes.loop = (() => {
    const g = group("loop"), d = loopDrawing(g, 40), tok = token(g);
    // (changed for the screens) the dot goes round the loop once at the end of the run and comes to rest between two boxes
    return { g, update(t) { d.set(t); fade(tok, seg(t, 0.62, 0.7)); tokenAt(tok, Math.PI * 1.25 + seg(t, 0.66, 1) * 2 * Math.PI); } };
  })();

  // -- the models the field set out to build: the loop again, whole, chord and all
  scenes.attempt = (() => {
    const g = group("attempt"), d = loopDrawing(g, 41), tok = token(g);
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
    return {
      g,
      update(t) {
        curves[0].set(seg(t, 0.02, 0.3));
        for (let k = 1; k < N; ++k) curves[k].set(seg(t, 0.3 + k * 0.03, 0.45 + k * 0.03));
      },
    };
  })();

  // -- the noise-state: each player's estimate, at the present t, of every shock so far. Each new signal revises
  // the estimates of old shocks too, so the bars behind the "now" line keep moving as it advances. The numbers are
  // exact Gaussian conditioning: shocks w, a state x_k = sum_j phi^(k-j) w_j, and each player sees x plus their own
  // noise; W-hat_t(u) = E[w_u | player's signals up to t].
  scenes.noise = (() => {
    const g = group("noise"), r = mulberry32(21), N = 30, phi = 0.85, SIG = [0.55, 1.4];
    const w = Array.from({ length: N }, () => gauss(r));
    const x = w.map((_, k) => { let v = 0; for (let j = 0; j <= k; ++j) v += Math.pow(phi, k - j) * w[j]; return v; });
    const y = SIG.map((sd) => x.map((v) => v + sd * gauss(r)));
    const cxx = (m, l) => { let v = 0; for (let j = 0; j <= Math.min(m, l); ++j) v += Math.pow(phi, m - j) * Math.pow(phi, l - j); return v; };
    function solve(A, b) {                                           // Cholesky: A is symmetric positive definite
      const n = b.length, L = A.map((row) => row.slice());
      for (let i = 0; i < n; ++i) for (let j = 0; j <= i; ++j) {
        let s = L[i][j]; for (let k = 0; k < j; ++k) s -= L[i][k] * L[j][k];
        L[i][j] = i === j ? Math.sqrt(s) : s / L[j][j];
      }
      const z = b.slice();
      for (let i = 0; i < n; ++i) { for (let k = 0; k < i; ++k) z[i] -= L[i][k] * z[k]; z[i] /= L[i][i]; }
      for (let i = n - 1; i >= 0; --i) { for (let k = i + 1; k < n; ++k) z[i] -= L[k][i] * z[k]; z[i] /= L[i][i]; }
      return z;
    }
    // est[i][t][u]: player i's estimate, at t, of the shock at u (u <= t)
    const est = SIG.map((sd, i) => Array.from({ length: N }, (_, t) => {
      const A = Array.from({ length: t + 1 }, (_, m) => Array.from({ length: t + 1 }, (_, l) => cxx(m, l) + (m === l ? sd * sd : 0)));
      const a = solve(A, y[i].slice(0, t + 1));
      return Array.from({ length: t + 1 }, (_, u) => { let v = 0; for (let m = u; m <= t; ++m) v += Math.pow(phi, m - u) * a[m]; return v; });
    }));
    const X = (u) => 60 + (480 * (u + 0.5)) / N, Y0 = 330, S = 52, slot = 480 / N;
    el("line", { x1: 50, y1: Y0, x2: 560, y2: Y0, class: "pencil soft", "stroke-width": 1 }, g);
    const bar = (u, dx, wd, cls) => { const b = el("rect", { x: X(u) + dx, width: wd, class: cls }, g); return b; };
    const truth = w.map((v, u) => { const b = bar(u, -slot * 0.42, slot * 0.84, ""); b.style.fill = "currentColor"; b.style.opacity = 0.13;
      b.setAttribute("y", v >= 0 ? Y0 - v * S : Y0); b.setAttribute("height", Math.abs(v) * S); return b; });
    const b1 = w.map((_, u) => bar(u, -slot * 0.34, slot * 0.3, "fillacc")), b2 = w.map((_, u) => bar(u, slot * 0.04, slot * 0.3, "fillwarm"));
    const now = el("line", { y1: 120, y2: 500, class: "pencil warm", "stroke-width": 1.4, "stroke-dasharray": "3 4" }, g);
    const nowl = text(g, 0, 112, "now, t", "label warmt");
    // the key, one word per color: each player's estimates, and the shocks themselves in gray
    const key = text(g, 60, 150, "", "mono", "start");
    key.innerHTML = '<tspan style="fill:var(--accent)">player 1</tspan><tspan dx="16" style="fill:var(--warm)">player 2</tspan><tspan dx="16">the shocks</tspan>';
    const put = (b, v, on) => { b.setAttribute("y", v >= 0 ? Y0 - v * S : Y0); b.setAttribute("height", Math.abs(v) * S); b.style.opacity = on ? 1 : 0; };
    return {
      g,
      update(t) {
        const k = Math.min(N - 1, Math.max(2, Math.round(seg(t, 0.05, 0.9) * (N - 1))));
        truth.forEach((b, u) => { b.style.opacity = u <= k ? 0.13 : 0.05; });
        for (let u = 0; u < N; ++u) { put(b1[u], u <= k ? est[0][k][u] : 0, u <= k); put(b2[u], u <= k ? est[1][k][u] : 0, u <= k); }
        const xn = X(k) + slot / 2;
        now.setAttribute("x1", xn); now.setAttribute("x2", xn); nowl.setAttribute("x", xn);
        fade(key, seg(t, 0.05, 0.2));
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
    const push = stroke(g, pencil([[150, 160], [230, 205], [300, 238]], 51, 1.2), "warm", 2);
    const tip = el("path", { d: "M288,226 L300,238 L284,242", class: "pencil warm", "stroke-width": 2 }, g);
    // the shift is B's belief moving; the wedge is its price to A, a separate mark: a tag hung on A's push
    const tag = el("g", {}, g);
    el("path", { d: pencil([[141, 168], [84, 214], [84, 438]], 53, 0.4), class: "pencil warm", "stroke-width": 1, "stroke-dasharray": "2 4" }, tag);
    el("path", { d: "M84,448 L93,457 L93,478 L75,478 L75,457 Z", class: "pencil warm", "stroke-width": 1.6 }, tag);
    el("circle", { cx: 84, cy: 458, r: 2, class: "pencil warm", "stroke-width": 1.2 }, tag);
    const wl = text(g, 102, 470, "the wedge: what it is worth to A to move B's belief", "mono warmt", "start");
    return {
      g,
      update(t, now) {
        fade(A, seg(t, 0, 0.1)); fade(B, seg(t, 0, 0.1)); fade(before, seg(t, 0.02, 0.15));
        push.set(seg(t, 0.15, 0.4)); fade(tip, seg(t, 0.38, 0.42));
        const m = 270 + 70 * seg(t, 0.35, 0.7) + Math.sin(now / 900) * 3 * seg(t, 0.7, 0.8);
        const p0 = bell(270), p1 = bell(m);
        after.setAttribute("d", pencil(p1, 52, 0.2)); fade(after, seg(t, 0.3, 0.4));
        area.setAttribute("d", "M" + p1.map((q) => q.join(",")).join(" L") + " L" + p0.slice().reverse().map((q) => q.join(",")).join(" L") + "Z");
        fade(area, seg(t, 0.45, 0.65) * 0.16);
        fade(tag, seg(t, 0.6, 0.72)); fade(wl, seg(t, 0.62, 0.75));
      },
    };
  })();

  Object.assign(window.DissScenes, scenes);
})();
// ---- from the site's dissertation/idea.js
// /dissertation/idea: from beliefs about beliefs to the noise-state. Everything drawn is computed here, from one small
// problem in discrete time: a state X_k = ρ X_{k-1} + w_k, and two players, player i seeing y^i_k = X_k + σ_i e^i_k.
// A player's noise-state is the exact Gaussian estimate of the shock path w given their signals so far; their
// forecast of X is the kernel of X run against it; player 1's forecast of player 2's forecast is the kernel of
// player 2's forecast run against player 1's noise-state. The page checks those against direct computations.
(function () {
  "use strict";
  const { el, mulberry32, gauss, clamp, seg, fade, stroke, text } = Sketch;   // static/js/sketch.js
  const svg = window.DissStage;
  const reduced = false;

  // ------------------------------------------------------------------ the model
  const N = 48, RHO = 0.92, SIG = [1.2, 2.0];
  const rng = mulberry32(20260924);
  const w = Array.from({ length: N }, () => gauss(rng));
  const e = [Array.from({ length: N }, () => gauss(rng)), Array.from({ length: N }, () => gauss(rng))];
  const RP = Array.from({ length: N }, (_, d) => Math.pow(RHO, d));
  const K = (k, j) => (j <= k ? RP[k - j] : 0);          // the kernel of X: L(t, s) on the grid
  const X = Array.from({ length: N }, (_, k) => { let s = 0; for (let j = 0; j <= k; ++j) s += K(k, j) * w[j]; return s; });
  const Y = [0, 1].map((i) => X.map((x, k) => x + SIG[i] * e[i][k]));

  // small dense solves: S z = b for symmetric positive definite S (Cholesky)
  function cholFactor(S) {
    const n = S.length, L = S.map((r) => r.slice());
    for (let i = 0; i < n; ++i) for (let j = 0; j <= i; ++j) {
      let s = L[i][j]; for (let k = 0; k < j; ++k) s -= L[i][k] * L[j][k];
      L[i][j] = i === j ? Math.sqrt(s) : s / L[j][j];
    }
    return L;
  }
  function cholSolve(S, b, L = cholFactor(S)) {   // pass L to solve against a matrix already factored
    const n = b.length;
    const z = b.slice();
    for (let i = 0; i < n; ++i) { for (let k = 0; k < i; ++k) z[i] -= L[i][k] * z[k]; z[i] /= L[i][i]; }
    for (let i = n - 1; i >= 0; --i) { for (let k = i + 1; k < n; ++k) z[i] -= L[k][i] * z[k]; z[i] /= L[i][i]; }
    return z;
  }
  // G[i][k]: the map from player i's signals y_0..y_k to their estimate of the shocks w_0..w_{N-1} (zero past k)
  // ŵ = K_{0:k}^T S^{-1} y_{0:k},  S = K_{0:k} K_{0:k}^T + σ² I
  const G = [0, 1].map((i) => {
    const out = [];
    for (let k = 0; k < N; ++k) {
      const S = [];
      for (let a = 0; a <= k; ++a) { S.push([]); for (let b = 0; b <= k; ++b) { let s = 0; for (let j = 0; j <= Math.min(a, b); ++j) s += K(a, j) * K(b, j); S[a].push(s + (a === b ? SIG[i] * SIG[i] : 0)); } }
      // columns of S^{-1}: G_k[u][m] = Σ_a K(a,u) (S^{-1})_{a m}
      const Sinv = [];
      const L = cholFactor(S);
      for (let m = 0; m <= k; ++m) { const unit = Array(k + 1).fill(0); unit[m] = 1; Sinv.push(cholSolve(S, unit, L)); }
      const Gk = [];
      for (let u = 0; u < N; ++u) { const row = Array(k + 1).fill(0); if (u <= k) for (let m = 0; m <= k; ++m) { let s = 0; for (let a = u; a <= k; ++a) s += K(a, u) * Sinv[m][a]; row[m] = s; } Gk.push(row); }
      out.push(Gk);
    }
    return out;
  });
  // the noise-states: Ŵ[i][k][u] = E[w_u | y^i_0..y^i_k]
  const What = [0, 1].map((i) => G[i].map((Gk, k) => Gk.map((row) => row.reduce((s, g, m) => s + g * Y[i][m], 0))));
  // forecasts from the kernel: X̂^i_k = Σ_u L(k,u) Ŵ^i_k(u)
  const Xhat = [0, 1].map((i) => X.map((_, k) => { let s = 0; for (let u = 0; u <= k; ++u) s += K(k, u) * What[i][k][u]; return s; }));
  // the same forecasts from a Kalman filter, as the check
  const kalman = (i) => { const out = []; let xp = 0, Pp = 1; for (let k = 0; k < N; ++k) { const g = Pp / (Pp + SIG[i] * SIG[i]); const xh = xp + g * (Y[i][k] - xp); out.push(xh); const P = (1 - g) * Pp; xp = RHO * xh; Pp = RHO * RHO * P + 1; } return out; };
  const checkFilter = Math.max(...[0, 1].map((i) => Math.max(...kalman(i).map((v, k) => Math.abs(v - Xhat[i][k])))));
  // player 2's forecast as a kernel on the shocks: A2[k][u] = Σ_{a≤k} (Σ_v L(k,v) G2_k[v][a]) L(a,u), plus a part on e²
  const A2 = X.map((_, k) => {
    const c = Array(k + 1).fill(0);                              // c_a = Σ_v L(k,v) G2_k[v][a]
    for (let v = 0; v <= k; ++v) for (let a = 0; a <= k; ++a) c[a] += K(k, v) * G[1][k][v][a];
    return Array.from({ length: N }, (_, u) => { let s = 0; for (let a = u; a <= k; ++a) s += c[a] * K(a, u); return s; });
  });
  // player 1's forecast of player 2's forecast: the kernel A2 run against player 1's noise-state
  const X12 = X.map((_, k) => A2[k].reduce((s, a, u) => s + a * What[0][k][u], 0));
  // checked by conditioning X̂²_k on player 1's signals directly: Cov(X̂²_k, y¹) S1^{-1} y¹, Cov = A2_k K^T
  const checkTower = Math.max(...X.map((_, k) => {
    const S = [], cov = [];
    for (let a = 0; a <= k; ++a) { S.push([]); for (let b = 0; b <= k; ++b) { let s = 0; for (let j = 0; j <= Math.min(a, b); ++j) s += K(a, j) * K(b, j); S[a].push(s + (a === b ? SIG[0] * SIG[0] : 0)); }
      let c = 0; for (let u = 0; u <= a; ++u) c += A2[k][u] * K(a, u); cov.push(c); }
    const z = cholSolve(S, Y[0].slice(0, k + 1));
    return Math.abs(cov.reduce((s, c, a) => s + c * z[a], 0) - X12[k]);
  }));
  window.ideaChecks = { filter: checkFilter, tower: checkTower };

  // ------------------------------------------------------------------ drawing helpers
  // A one-shot motion (a head turning, a sun going up) plays forward once the progress passes `at` going down, and back
  // again if the reader scrolls above `at` while it is on screen. A drawing that leaves the screen is put straight back
  // to its start (jump(0) below).
  const oneShot = (at, dur) => Sketch.oneShot(at, dur, reduced);
  const pencil = (pts, seed, wob = 0.6) => Sketch.pencil(pts, seed, wob);
  const TX = (k) => 60 + (490 * k) / (N - 1);
  function formula(key, x, y, width, parent, anchor = "start") {
    const src = window.IDEA_FX && window.IDEA_FX[key];
    if (!src) return el("g", {}, parent);
    const doc = new DOMParser().parseFromString(src, "image/svg+xml").documentElement;
    const n = document.importNode(doc, true);
    const vb = (n.getAttribute("viewBox") || "0 0 1 1").split(" ").map(Number), h = (width * vb[3]) / vb[2];
    n.setAttribute("width", width); n.setAttribute("height", h);
    n.setAttribute("x", anchor === "middle" ? x - width / 2 : x); n.setAttribute("y", y - h / 2);
    n.removeAttribute("style");
    n.style.color = "var(--ink)";
    parent.appendChild(n);
    return n;
  }
  function bars(g, vals, base, scale, cls, width = 6) {
    return vals.map((v, k) => el("rect", { x: TX(k) - width / 2, y: v >= 0 ? base - v * scale : base, width, height: Math.abs(v) * scale, class: cls }, g));
  }
  const scenes = {};
  const group = (name) => { const g = el("g", { "data-scene": name }, svg); g.style.opacity = 0; g.style.transition = "opacity 0.6s"; return g; };

  // ------------------------------------------------------------------ 0. a few researchers and a telescope
  function person(g, x, y, cls) {
    const p = el("g", { transform: `translate(${x} ${y})` }, g);
    el("circle", { cx: 0, cy: -13, r: 7, class: cls }, p);
    el("path", { d: "M-11,12 Q-11,-3 0,-3 Q11,-3 11,12 Z", class: cls }, p);
    return p;
  }
  // a head in profile, facing right (or left with flip), with a circle where its model of the world lives
  const HEAD = "M-24,60 C-24,48 -30,40 -38,30 C-50,14 -52,-12 -44,-32 C-34,-56 -8,-66 16,-60 C38,-54 50,-36 48,-12 L47,-4 L57,12 C59,15 56,17 52,17 L51,23 L47,26 L50,30 C50,34 47,36 45,37 C46,45 40,49 30,47 L22,46 L22,60";
  function head(g, x, y, s, flip, name) {
    const h = el("g", { transform: `translate(${x} ${y}) scale(${flip ? -s : s} ${s})` }, g);
    el("path", { d: HEAD + " Z", class: "headfill" }, h);
    el("path", { d: HEAD, class: "pencil", "stroke-width": 1.8 / s, fill: "none" }, h);
    const brain = el("circle", { cx: 2, cy: -18, r: 30, class: "pencil soft", "stroke-width": 1.2 / s, fill: "none" }, h);
    if (name) { const t = text(g, x, y + 60 * s + 24, name, "label"); return { h, t, brain }; }
    return { h, brain };
  }

  // ------------------------------------------------------------------ 00. one person knows: the world goes on
  scenes.doctor = (() => {
    const g = group("doctor");
    const crowd = [];
    for (let i = 0; i < 8; ++i) for (let j = 0; j < 4; ++j) {
      const x = 85 + i * 62, y = 280 + j * 74, k = i === 3 && j === 0;
      const p = person(g, x, y, "fillacc"); p.style.fill = k ? "var(--accent)" : "currentColor";
      crowd.push({ p, x, y, k, ph: (i * 7 + j * 3) % 5 });
    }
    // the one who knows: a calendar over their head, the months crossed off
    const me = crowd.find((c) => c.k);
    const cal = el("g", { transform: `translate(${me.x} ${me.y - 100})` }, g);
    el("rect", { x: -54, y: -30, width: 108, height: 62, rx: 6, class: "box pencil", "stroke-width": 1.4 }, cal);
    el("path", { d: `M${me.x - 6 - me.x},32 L0,62 L8,32`, class: "box pencil", "stroke-width": 1.4 }, cal);
    const months = [];
    for (let m = 0; m < 6; ++m) {
      const cx = -36 + m * 14.4;
      el("rect", { x: cx - 5, y: -6, width: 10, height: 12, class: "pencil soft", "stroke-width": 1, fill: "none" }, cal);
      months.push(el("path", { d: `M${cx - 6},-7 L${cx + 6},7`, class: "pencil accent", "stroke-width": 1.8 }, cal));
    }
    // at the sixth month the person is gone: a small headstone where they stood, while everyone else carries on
    const stone = el("g", { transform: `translate(${me.x} ${me.y + 12})` }, g);
    el("path", { d: "M-9,0 L-9,-16 Q-9,-26 0,-26 Q9,-26 9,-16 L9,0 Z", class: "headfill" }, stone);
    el("path", { d: "M-9,0 L-9,-16 Q-9,-26 0,-26 Q9,-26 9,-16 L9,0", class: "pencil", "stroke-width": 1.6, fill: "none" }, stone);
    el("path", { d: "M0,-21 L0,-9 M-4,-17 L4,-17", class: "pencil soft", "stroke-width": 1.2 }, stone);
    el("path", { d: "M-15,0.5 L15,0.5", class: "pencil soft", "stroke-width": 1.2 }, stone);
    fade(stone, 0);
    const end = oneShot(0.74, 1.5);                        // after the last month is crossed off; timed, so it is seen whole
    return { g, shots: [end], update(t, now, dt) {
      const v = now ? end.step(t, dt) : 0, gone = seg(v, 0, 0.45), rise = seg(v, 0.3, 1);
      stone.setAttribute("transform", `translate(${me.x} ${me.y + 12 + 10 * (1 - rise)})`);
      fade(stone, rise);
      crowd.forEach((c) => {
        fade(c.p, seg(t, 0, 0.12) * (c.k ? 1 - gone : 0.4));
        // everyone else keeps walking about their business
        const dx = c.k ? 0 : Math.sin(now / 1100 + c.ph) * 6, dy = c.k ? 0 : Math.abs(Math.sin(now / 280 + c.ph)) * -2;
        c.p.setAttribute("transform", `translate(${c.x + dx} ${c.y + dy})`);
      });
      fade(cal, seg(t, 0.15, 0.3));
      months.forEach((m, i) => fade(m, seg(t, 0.3 + i * 0.07, 0.36 + i * 0.07)));
    } };
  })();

  // ------------------------------------------------------------------ 00b. everyone knows: pandemonium
  scenes.sun = (() => {
    const g = group("sun");
    const sun = el("g", { transform: "translate(300 150)" }, g);
    const halo = el("circle", { r: 118, class: "pencil warm", "stroke-width": 1, fill: "none", opacity: 0.35 }, sun);
    const star = (n, r0, r1) => { let d = ""; for (let k = 0; k < 2 * n; ++k) { const a = (k * Math.PI) / n, r = k % 2 ? r0 : r1; d += (k ? "L" : "M") + (r * Math.cos(a)).toFixed(1) + "," + (r * Math.sin(a)).toFixed(1); } return d + "Z"; };
    const outer = el("path", { d: star(15, 52, 88), class: "fillwarm pencil warm", "stroke-width": 1.4 }, sun);
    const inner = el("path", { d: star(12, 36, 60), class: "sunmid" }, sun);
    el("circle", { r: 26, class: "suncore" }, sun);
    const r = mulberry32(11), crowd = [];
    for (let i = 0; i < 9; ++i) for (let j = 0; j < 4; ++j) {
      const x = 70 + i * 57, y = 330 + j * 60;
      const p = person(g, x, y, "fillacc"); p.style.fill = "var(--accent)";
      crowd.push({ p, x, y, jx: (r() - 0.5) * 40, jy: (r() - 0.5) * 30, ja: (r() - 0.5) * 70, ph: r() * 6 });
    }
    const flash = el("circle", { cx: 300, cy: 150, r: 0, class: "suncore" }, g);
    const ring = el("circle", { cx: 300, cy: 150, r: 0, class: "pencil warm", "stroke-width": 3, fill: "none" }, g);
    const shards = [];
    for (let k = 0; k < 30; ++k) {
      const a = (k / 30) * 2 * Math.PI + (r() - 0.5) * 0.3, len = 8 + r() * 16, sp = 0.6 + r() * 0.8;
      shards.push({ a, sp, s: el("path", { d: `M0,0 L${len},0`, class: "pencil warm", "stroke-width": 2 + r() * 2, "stroke-linecap": "round" }, g) });
    }
    [flash, ring].forEach((n) => fade(n, 0)); shards.forEach((q) => fade(q.s, 0));
    const blast = oneShot(0.65, 1.6);                     // timed: seen whole however fast the page is scrolled
    return { g, shots: [blast], update(t, now, dt) {
      // it blows up as the reader moves on; with reduced motion it stays whole
      const boom = now ? 1 - Math.pow(1 - blast.step(t, dt), 2) : 0;
      fade(sun, seg(t, 0, 0.15) * (1 - seg(boom, 0.05, 0.3)));
      sun.style.transformOrigin = "300px 150px";
      fade(flash, boom > 0 ? Math.sin(Math.PI * seg(boom, 0, 0.5)) * 0.95 : 0); flash.setAttribute("r", 30 + 520 * seg(boom, 0, 0.5));
      fade(ring, boom > 0 ? 1 - boom : 0); ring.setAttribute("r", 60 + 700 * boom);
      shards.forEach((q) => {
        const d = 40 + 620 * q.sp * boom;
        q.s.setAttribute("transform", `translate(${300 + d * Math.cos(q.a)} ${150 + d * Math.sin(q.a)}) rotate(${(q.a * 180) / Math.PI})`);
        fade(q.s, boom > 0 ? (1 - boom) * 0.9 : 0);
      });
      const pulse = 1 + 0.05 * Math.sin(now / 160) + 0.25 * seg(t, 0.3, 0.7) + 0.6 * seg(boom, 0, 0.2);
      outer.setAttribute("transform", `rotate(${now / 90 % 360}) scale(${pulse})`);
      inner.setAttribute("transform", `rotate(${-now / 60 % 360}) scale(${pulse})`);
      halo.setAttribute("r", 118 * pulse);
      const chaos = seg(t, 0.25, 0.75);
      crowd.forEach((c) => {
        fade(c.p, seg(t, 0.05, 0.2));
        const wob = Math.sin(now / 200 + c.ph);
        const push = boom * boom * 60, ang = Math.atan2(c.y - 150, c.x - 300);
        c.p.setAttribute("transform", `translate(${c.x + chaos * (c.jx + wob * 6) + push * Math.cos(ang)} ${c.y + chaos * c.jy + push * Math.sin(ang)}) rotate(${chaos * (c.ja + wob * 12) + boom * 90 * Math.sign(c.jx)})`);
      });
    } };
  })();

  scenes.telescope = (() => {
    const g = group("telescope");
    const sun = el("g", { transform: "translate(470 110)" }, g);
    for (let a = 0; a < 8; ++a) { const t = (a * Math.PI) / 4; el("line", { x1: 30 * Math.cos(t), y1: 30 * Math.sin(t), x2: 44 * Math.cos(t), y2: 44 * Math.sin(t), class: "pencil warm", "stroke-width": 2 }, sun); }
    el("circle", { r: 24, class: "fillwarm" }, sun);
    const scope = el("g", { transform: "translate(150 190)" }, g);
    el("path", { d: "M0,0 L-14,40 M0,0 L14,40 M0,0 L2,42", class: "pencil", "stroke-width": 1.8 }, scope);
    el("path", { d: "M-6,4 L74,-44 L82,-32 L2,16 Z", class: "box pencil", "stroke-width": 1.8 }, scope);
    const sight = el("line", { x1: 230, y1: 150, x2: 440, y2: 118, class: "pencil soft", "stroke-width": 1, "stroke-dasharray": "3 5" }, g);
    const crowd = [], know = new Set(["3,1", "2,1", "4,1", "3,2", "3,0"]);
    for (let i = 0; i < 8; ++i) for (let j = 0; j < 3; ++j) {
      const x = 85 + i * 62, y = 330 + j * 72, k = know.has(`${i},${j}`);
      const p = person(g, x, y, "fillacc"); p.style.fill = k ? "var(--accent)" : "currentColor";
      // the five who know are ringed, so they stay apart from the people who later work it out from what they do
      if (k) el("circle", { cx: 0, cy: -2, r: 22, class: "pencil accent", "stroke-width": 1.4, fill: "none" }, p);
      crowd.push({ p, x, y, k, d: Math.hypot(i - 3, (j - 1) * 1.15) });
    }
    const rings = [0, 1, 2].map(() => el("circle", { cx: 271, cy: 402, class: "pencil accent", "stroke-width": 1.2 }, g));
    return { g, update(t, now) {
      fade(sun, seg(t, 0, 0.1)); fade(scope, seg(t, 0.02, 0.12)); fade(sight, seg(t, 0.08, 0.18) * 0.8);
      const spread = seg(t, 0.35, 0.95) * 2.4;          // word spreads, but slowly: the far edge never hears
      crowd.forEach((c) => {
        if (c.k) { fade(c.p, seg(t, 0.1, 0.2)); return; }
        const learned = clamp(spread - c.d + 1);
        c.p.style.opacity = seg(t, 0.05, 0.15) * (0.28 + 0.2 * learned);
        c.p.style.fill = learned > 0.5 ? "var(--accent)" : "currentColor";
      });
      rings.forEach((r, i) => { const ph = ((now / 2400 + i / 3) % 1); r.setAttribute("r", 30 + ph * 200); fade(r, seg(t, 0.3, 0.4) * (1 - ph) * 0.5); });
    } };
  })();

  // ------------------------------------------------------------------ 0b. many hands on the state
  scenes.hands = (() => {
    const g = group("hands");
    const l1 = el("g", {}, g), l2 = el("g", {}, g), l3 = el("g", {}, g);
    formula("hands1", 300, 170, 470, l1, "middle");
    formula("hands2", 300, 300, 190, l2, "middle");
    formula("hands3", 300, 410, 490, l3, "middle");
    const ul = [];
    for (const [x, w2] of [[300 - 490 / 2 + 205, 70], [300 - 490 / 2 + 300, 70]]) ul.push(stroke(g, pencil([[x - w2 / 2, 440], [x + w2 / 2, 441]], x, 1), "warm", 2.2));
    return { g, update(t) {
      fade(l1, seg(t, 0, 0.15)); fade(l2, seg(t, 0.2, 0.35)); fade(l3, seg(t, 0.4, 0.55));
      ul.forEach((u) => u.set(seg(t, 0.6, 0.75)));
    } };
  })();

  // ------------------------------------------------------------------ 1. the tower
  scenes.tower = (() => {
    const g = group("tower");
    const p1 = el("g", {}, g), p2 = el("g", {}, g);
    head(p1, 110, 500, 0.62, false, "Alice (1)"); head(p2, 490, 500, 0.62, true, "Bob (2)");
    const levels = ["t1", "t2", "t3", "t4"].map((k, i) => {
      const lg = el("g", {}, g), y = 430 - i * 92, x = 300 + (i % 2 ? 30 : -30);
      // the deeper formulas are tall for their width: shrink any that would not fit a pill 64 high, then size the pill to it
      let w = 150 + i * 62, f = formula(k, x, y, w, lg, "middle");
      const fh = +f.getAttribute("height");
      if (fh > 44) { w *= 44 / fh; f.remove(); f = formula(k, x, y, w, lg, "middle"); }
      const left = x - w / 2 - 18;
      lg.insertBefore(el("rect", { x: left, y: y - 32, width: w + 36, height: 64, rx: 30, class: "box pencil", "stroke-width": 1.4 }), f);
      // a short trail of thought bubbles off the pill's own left end, so no trail crosses another pill
      for (let d = 0; d < 3; ++d) el("circle", { cx: left - 34 + d * 11, cy: y + 26 - d * 9, r: 2 + d, class: "pencil soft", "stroke-width": 1.2 }, lg);
      return lg;
    });
    // how many beliefs of each order n players hold: n first-order, then each belief about another player (n - 1 choices)
    const counts = ["n", "n(n−1)", "n(n−1)²", "n(n−1)³"].map((c, i) => text(g, 28, 436 - i * 92, c, "label acc", "start"));
    const dots = text(g, 300, 58, "…", "label"); dots.style.fontSize = "34px";
    return { g, update(t, now) {
      fade(p1, seg(t, 0, 0.1)); fade(p2, seg(t, 0, 0.1));
      levels.forEach((l, i) => { fade(l, seg(t, 0.08 + i * 0.14, 0.2 + i * 0.14)); l.setAttribute("transform", `translate(0 ${Math.sin(now / 900 + i) * 3})`); });
      counts.forEach((c, i) => fade(c, seg(t, 0.14 + i * 0.14, 0.24 + i * 0.14)));
      fade(dots, seg(t, 0.66, 0.8) * (0.6 + 0.4 * Math.sin(now / 300)));
    } };
  })();

  // ------------------------------------------------------------------ 1b. beliefs about randomness
  scenes.randomness = (() => {
    const g = group("randomness");
    const heads = [[160, 215, "Alice", false], [440, 215, "Bob", true]].map(([x, y, l, f], i) => {
      const h = el("g", {}, g); head(h, x, y, 1, f, l);
      // inside each head, a small model of the other one, which the noise-state lets them throw away
      const bx = x + (f ? -2 : 2), by = y - 18, mini = el("g", {}, g);
      head(mini, bx, by + 2, 0.3, !f);
      const cross = [stroke(g, pencil([[bx - 18, by - 18], [bx + 18, by + 18]], 90 + i, 0.8), "warm", 2.2), stroke(g, pencil([[bx + 18, by - 18], [bx - 18, by + 18]], 94 + i, 0.8), "warm", 2.2)];
      return { h, x, y, mini, cross };
    });
    // the interface between them: all either can see of the other
    const iface = el("g", {}, g);
    el("line", { x1: 300, y1: 130, x2: 300, y2: 300, class: "pencil soft", "stroke-width": 1.2, "stroke-dasharray": "4 5" }, iface);
    // the nested models, each struck out
    const models = [["Alice's model of Bob", 150, 110], ["Bob's model of Alice", 450, 110], ["Alice's model of Bob's model of Alice", 150, 60], ["Bob's model of Alice's model of Bob", 450, 60]].map(([s2, x, y], i) => {
      const m = el("g", {}, g); const tt = text(m, x, y, s2, "label"); tt.style.fontSize = "14px";
      const w2 = Math.min(270, s2.length * 6.4);
      const strike = stroke(m, pencil([[x - w2 / 2 - 4, y - 3], [x + w2 / 2 + 4, y - 7]], 70 + i, 1), "warm", 2);
      return { m, strike };
    });
    const oval = el("g", {}, g);
    el("ellipse", { cx: 300, cy: 440, rx: 170, ry: 48, class: "pencil warm", "stroke-width": 2 }, oval);
    el("ellipse", { cx: 300, cy: 440, rx: 170, ry: 48, class: "fillwarm", opacity: 0.1 }, oval);
    // a few shocks jiggling inside it
    const r = mulberry32(3), kicks = [];
    for (let k = 0; k < 18; ++k) kicks.push({ x: 160 + k * 16.5, h: gauss(r) * 16, el: el("line", { x1: 160 + k * 16.5, x2: 160 + k * 16.5, y1: 470, y2: 470, class: "pencil warm", "stroke-width": 2 }, oval) });
    const arrows = heads.map((h, i) => stroke(g, pencil([[300 + (i ? 60 : -60), 392], [h.x + (i ? -8 : 8), 330 - 40 - 10 + 60], [h.x + (i ? -2 : 2), h.y + 12]], 80 + i, 2), "warm", 1.8));
    return { g, update(t, now) {
      heads.forEach((h) => fade(h.h, seg(t, 0, 0.1))); fade(iface, seg(t, 0.02, 0.12) * (1 - 0.6 * seg(t, 0.45, 0.6)));
      heads.forEach((h, i) => { fade(h.mini, seg(t, 0.04, 0.14) * (1 - 0.6 * seg(t, 0.4, 0.55))); h.cross.forEach((c) => c.set(seg(t, 0.3 + i * 0.05, 0.4 + i * 0.05))); });
      models.forEach((m, i) => { fade(m.m, seg(t, 0.02 + i * 0.04, 0.12 + i * 0.04) * (1 - 0.55 * seg(t, 0.45, 0.6))); m.strike.set(seg(t, 0.25 + i * 0.05, 0.35 + i * 0.05)); });
      fade(oval, seg(t, 0.45, 0.6));
      kicks.forEach((k, i) => { const hh = k.h * (0.6 + 0.4 * Math.sin(now / 300 + i)); k.el.setAttribute("y2", 470 - Math.abs(hh)); k.el.setAttribute("y1", 470); });
      arrows.forEach((a) => a.set(seg(t, 0.6, 0.8)));
    } };
  })();

  // ------------------------------------------------------------------ 2. shocks and the state they move
  scenes.shocks = (() => {
    const g = group("shocks");
    el("line", { x1: 50, y1: 470, x2: 560, y2: 470, class: "pencil soft", "stroke-width": 1 }, g);
    el("line", { x1: 50, y1: 240, x2: 560, y2: 240, class: "pencil soft", "stroke-width": 1 }, g);
    const b = bars(g, w, 470, 22, "fillwarm");
    const xp = stroke(g, pencil(X.map((v, k) => [TX(k), 240 - v * 22]), 3), "", 2);
    const lw = text(g, 50, 540, "W: the primitive shocks", "label warmt", "start");
    const lx = text(g, 50, 110, "X: the physical world", "label", "start");
    return { g, update(t) {
      b.forEach((r, k) => fade(r, seg(t, 0.02 + (k / N) * 0.5, 0.06 + (k / N) * 0.5) * 0.8));
      xp.set(seg(t, 0.08, 0.62)); fade(lw, seg(t, 0.05, 0.2)); fade(lx, seg(t, 0.3, 0.45));
    } };
  })();

  // ------------------------------------------------------------------ 3. one shock and its impulse response
  scenes.irf = (() => {
    const g = group("irf"), s = 10;
    el("line", { x1: 50, y1: 470, x2: 560, y2: 470, class: "pencil soft", "stroke-width": 1 }, g);
    el("line", { x1: 50, y1: 360, x2: 560, y2: 360, class: "pencil soft", "stroke-width": 1 }, g);
    const kick = el("rect", { x: TX(s) - 3, y: 470 - 60, width: 6, height: 60, class: "fillwarm" }, g);
    const pts = []; for (let k = s; k < N; ++k) pts.push([TX(k), 360 - 180 * K(k, s)]);
    const curve = stroke(g, pencil(pts, 5, 0.3), "accent", 2.6);
    const sl = text(g, TX(s), 500, "s", "label warmt");
    const probe = el("g", {}, g), tk = 30;
    el("line", { x1: TX(tk), y1: 360, x2: TX(tk), y2: 360 - 180 * K(tk, s), class: "pencil accent", "stroke-width": 1.2, "stroke-dasharray": "3 4" }, probe);
    text(probe, TX(tk), 385, "t", "label acc");
    text(probe, TX(tk) + 10, 360 - 180 * K(tk, s) - 12, "L(t, s)", "label acc", "start");
    return { g, update(t) { fade(kick, seg(t, 0, 0.1)); fade(sl, seg(t, 0, 0.1)); curve.set(seg(t, 0.1, 0.55)); fade(probe, seg(t, 0.5, 0.65)); } };
  })();

  // ------------------------------------------------------------------ 4. responses add up to the path
  scenes.superpose = (() => {
    // L_t = mean(t) + sum over s <= t of L(t, s) dW_s, at one date t that sweeps across with the scroll: the shocks; the
    // kernel's row at t (the weight on each shock, sliding with t); their products; and the path, which at t is the mean
    // plus the sum of the products. A dot product, one date at a time, with the mean shifting the level.
    const g = group("superpose");
    const MEAN = (k) => 2.2 * (1 - Math.exp(-k / 10));                  // an illustrative mean, drifting to a level
    const rows = { w: 128, L: 262, p: 342, out: 505 }, WS = 13, LS = 60, OS = 12;
    for (const y of [rows.w, rows.L, rows.p, rows.out]) el("line", { x1: 50, y1: y, x2: 560, y2: y, class: "pencil soft", "stroke-width": 1 }, g);
    text(g, 50, 96, "the shocks dW", "mono", "start");
    text(g, 50, 290, "shock × weight", "mono", "start");
    text(g, 50, 396, "Lₜ: the mean plus the sum of the products", "mono", "start");
    const shock = bars(g, w, rows.w, WS, "fillwarm", 5);
    const wt = w.map((_, j) => el("rect", { x: TX(j) - 2.5, width: 5, class: "fillacc" }, g));
    const pr = w.map((_, j) => { const r = el("rect", { x: TX(j) - 2.5, width: 5 }, g); r.style.fill = "currentColor"; return r; });
    const meanL = el("path", { d: pencil(X.map((_, k) => [TX(k), rows.out - OS * MEAN(k)]), 21, 0.1), class: "pencil", "stroke-width": 1.3, "stroke-dasharray": "5 5" }, g);
    const path = el("path", { class: "pencil accent", "stroke-width": 2.4 }, g);
    const sumSeg = el("line", { class: "pencil accent", "stroke-width": 3.2 }, g);
    const dot = el("circle", { r: 4.5, class: "fillacc" }, g);
    const now = el("line", { y1: 108, y2: 580, class: "pencil warm", "stroke-width": 1.4, "stroke-dasharray": "3 4" }, g);
    const nowT = text(g, 0, 100, "now, t", "label warmt");
    return { g, update(t) {
      const k = Math.min(N - 1, 3 + Math.floor(seg(t, 0.08, 0.92) * (N - 3))), x = TX(k), on = seg(t, 0.04, 0.12);
      shock.forEach((r, j) => fade(r, j <= k ? 0.9 : 0.15));
      w.forEach((v, j) => {
        const kk = K(k, j), h = LS * kk, pv = v * kk * WS;
        wt[j].setAttribute("y", rows.L - h); wt[j].setAttribute("height", h); fade(wt[j], j <= k ? on : 0);
        pr[j].setAttribute("y", pv >= 0 ? rows.p - pv : rows.p); pr[j].setAttribute("height", Math.abs(pv)); fade(pr[j], j <= k ? 0.7 * on : 0);
      });
      const val = (i) => rows.out - OS * (MEAN(i) + X[i]);
      path.setAttribute("d", pencil(X.slice(0, k + 1).map((_, i) => [TX(i), val(i)]), 11, 0.1));
      sumSeg.setAttribute("x1", x); sumSeg.setAttribute("x2", x); sumSeg.setAttribute("y1", rows.out - OS * MEAN(k)); sumSeg.setAttribute("y2", val(k));
      dot.setAttribute("cx", x); dot.setAttribute("cy", val(k));
      now.setAttribute("x1", x); now.setAttribute("x2", x); nowT.setAttribute("x", x);
      [path, sumSeg, dot, now, nowT].forEach((n) => fade(n, on));
      fade(meanL, seg(t, 0.02, 0.1) * 0.7);
    } };
  })();

  // ------------------------------------------------------------------ 5. the noise-state: estimates of the shocks, revised
  scenes.noisestate = (() => {
    const g = group("noisestate");
    el("line", { x1: 50, y1: 200, x2: 560, y2: 200, class: "pencil soft", "stroke-width": 1 }, g);
    el("line", { x1: 50, y1: 440, x2: 560, y2: 440, class: "pencil soft", "stroke-width": 1 }, g);
    const sig = Y[0].map((v, k) => el("circle", { cx: TX(k), cy: 200 - v * 13, r: 2.8, class: "fillacc" }, g));
    sig.forEach((c) => { c.style.fill = "currentColor"; });
    const xs = el("path", { d: pencil(X.map((v, k) => [TX(k), 200 - v * 13]), 13, 0.1), class: "pencil soft", "stroke-width": 1.2 }, g);
    const trueBars = bars(g, w, 440, 26, "", 7); trueBars.forEach((r) => { r.style.fill = "currentColor"; r.style.opacity = 0.14; });
    const est = w.map((_, k) => el("rect", { x: TX(k) - 2.5, width: 5, class: "fillacc" }, g));
    const now = el("line", { y1: 90, y2: 520, class: "pencil warm", "stroke-width": 1.4, "stroke-dasharray": "3 4" }, g);
    const nowl = text(g, 0, 82, "now, t", "label warmt");
    text(g, 50, 110, "player 1's signal", "mono", "start");
    const lab = text(g, 50, 560, "Ŵ¹ₜ(u): player 1's estimate, at t, of the shock at u", "label acc", "start");
    return { g, update(t) {
      const k = 4 + Math.round(seg(t, 0.05, 0.9) * (N - 5));
      sig.forEach((c, j) => fade(c, j <= k ? 0.75 : 0));
      xs.style.opacity = 0.35;
      const e1 = What[0][k];
      est.forEach((r, u) => { const v = u <= k ? e1[u] : 0; r.setAttribute("y", v >= 0 ? 440 - v * 26 : 440); r.setAttribute("height", Math.abs(v) * 26); fade(r, u <= k ? 1 : 0); });
      now.setAttribute("x1", TX(k)); now.setAttribute("x2", TX(k)); nowl.setAttribute("x", TX(k));
      fade(lab, seg(t, 0.1, 0.25));
    } };
  })();

  // ------------------------------------------------------------------ 6. forecasting: the same kernel against the estimate
  scenes.samekernel = (() => {
    const g = group("samekernel");
    el("line", { x1: 50, y1: 300, x2: 560, y2: 300, class: "pencil soft", "stroke-width": 1 }, g);
    const sig = Y[0].map((v, k) => el("circle", { cx: TX(k), cy: 300 - v * 16, r: 2.6 }, g));
    sig.forEach((c) => { c.style.fill = "currentColor"; c.style.opacity = 0.35; });
    const xs = stroke(g, pencil(X.map((v, k) => [TX(k), 300 - v * 16]), 21, 0.1), "", 1.6);
    const xh = stroke(g, pencil(Xhat[0].map((v, k) => [TX(k), 300 - v * 16]), 22, 0.1), "accent", 2.6);
    const l1 = text(g, 50, 110, "X, the state", "label", "start");
    const l2 = text(g, 50, 135, "X̂¹, the kernel of X run against Ŵ¹", "label acc", "start");
    const chk = text(g, 300, 560, "", "mono");
    return { g, update(t) {
      xs.set(seg(t, 0, 0.35)); xh.set(seg(t, 0.25, 0.75)); fade(l1, seg(t, 0, 0.15)); fade(l2, seg(t, 0.25, 0.4));
      chk.textContent = `same as a Kalman filter computed directly: largest gap ${checkFilter.toExponential(0)}`; fade(chk, seg(t, 0.7, 0.85));
    } };
  })();

  // ------------------------------------------------------------------ 7. the tower collapses into a composition
  scenes.collapse = (() => {
    const g = group("collapse");
    const levels = ["t1", "t2", "t3", "t4"].map((k, i) => { const lg = el("g", {}, g); formula(k, 300, 70 + i * 46, 120 + i * 40, lg, "middle"); return lg; });
    // the kernel of player 2's forecast on the grid of dates: row t, column u
    const hm = el("g", {}, g), H0 = 60, S = 180 / N;
    let mx = 0; for (const r of A2) for (const v of r) mx = Math.max(mx, Math.abs(v));
    for (let k = 0; k < N; ++k) for (let u = 0; u <= k; ++u) {
      const v = A2[k][u] / mx;
      const c = el("rect", { x: 90 + u * S, y: H0 + k * S, width: S + 0.3, height: S + 0.3, class: "fillacc" }, hm);
      c.style.opacity = Math.min(1, Math.abs(v) * 1.3);
    }
    el("rect", { x: 90, y: H0, width: 180, height: 180, class: "pencil soft", "stroke-width": 1 }, hm);
    text(hm, 180, H0 + 222, "row: date t · column: shock u", "mono");
    const arrow = el("g", {}, g);
    // the kernel times the noise-state gives the forecast path below: × between them, the arrow down to the path
    const times = text(arrow, 320, 159, "×", "label"); times.style.fontSize = "28px";
    el("path", { d: "M460,258 L460,306 M452,297 L460,307 L468,297", class: "pencil", "stroke-width": 1.6 }, arrow);
    const w1 = el("g", {}, g);
    for (let u = 0; u < N; ++u) { const v = What[0][N - 1][u]; el("rect", { x: 370 + u * (180 / N), y: v >= 0 ? 150 - v * 20 : 150, width: 180 / N - 0.6, height: Math.abs(v) * 20, class: "fillacc" }, w1); }
    el("line", { x1: 50, y1: 450, x2: 560, y2: 450, class: "pencil soft", "stroke-width": 1 }, g);
    const x2 = stroke(g, pencil(Xhat[1].map((v, k) => [TX(k), 450 - v * 16]), 31, 0.1), "warm", 2);
    const x12 = stroke(g, pencil(X12.map((v, k) => [TX(k), 450 - v * 16]), 32, 0.1), "accent", 2.6);
    const l2 = text(g, 50, 330, "X̂²: what player 2 forecasts", "label warmt", "start");
    const l12 = text(g, 50, 352, "𝔼¹[X̂²]: what player 1 thinks player 2 forecasts", "label acc", "start");
    const chk = text(g, 300, 585, "", "mono");
    return { g, update(t) {
      levels.forEach((l, i) => { const k = seg(t, 0.05 + i * 0.03, 0.3 + i * 0.03); fade(l, (1 - k) * seg(t, 0, 0.05)); l.setAttribute("transform", `translate(${-120 * k} ${(60 - i * 46) * k}) scale(${1 - 0.6 * k})`); l.style.transformOrigin = "300px 150px"; });
      fade(hm, seg(t, 0.25, 0.4)); fade(arrow, seg(t, 0.38, 0.45)); fade(w1, seg(t, 0.42, 0.52));
      x2.set(seg(t, 0.5, 0.75)); x12.set(seg(t, 0.62, 0.9)); fade(l2, seg(t, 0.5, 0.6)); fade(l12, seg(t, 0.62, 0.72));
      chk.textContent = `same as conditioning on player 1's signals directly: largest gap ${checkTower.toExponential(0)}`; fade(chk, seg(t, 0.85, 0.95));
    } };
  })();

  // ------------------------------------------------------------------ 8. a fixed point in kernels
  scenes.fixedpoint = (() => {
    const g = group("fixedpoint"), M = 30, T = (k) => 70 + (470 * k) / (M - 1);
    el("line", { x1: 60, y1: 470, x2: 560, y2: 470, class: "pencil soft", "stroke-width": 1 }, g);
    // the filter's gain from a poor first guess, iterated through its Riccati map; each iterate is a kernel:
    // the response of the forecast to a unit shock at age 0
    const kernelsFrom = (P0) => {
      const out = []; let Pp = P0;
      for (let n = 0; n < 6; ++n) {
        const gain = Pp / (Pp + SIG[0] * SIG[0]), h = [];
        let hv = 0; for (let k = 0; k < M; ++k) { hv = (1 - gain) * RHO * hv + gain * Math.pow(RHO, k); h.push(hv); }
        out.push(h);
        const P = (1 - gain) * Pp; Pp = RHO * RHO * P + 1;
      }
      return out;
    };
    // two first guesses, one trusting the signal too little and one too much, iterated through the same map
    const low = kernelsFrom(0.004), high = kernelsFrom(40);
    const draw = (h, n, cls, seed) => el("path", { d: pencil(h.map((v, k) => [T(k), 470 - v * 320]), seed + n, 0.1), class: "pencil " + cls, "stroke-width": 1.4 }, g);
    const pl = low.map((h, n) => draw(h, n, "warm", 60)), ph = high.map((h, n) => draw(h, n, "", 80));
    const fin = el("path", { d: pencil(low[5].map((v, k) => [T(k), 470 - v * 320]), 99, 0.1), class: "pencil accent", "stroke-width": 3 }, g);
    const it = text(g, 540, 110, "", "label acc", "end");
    return { g, update(t) {
      const n = Math.min(5, Math.floor(seg(t, 0.05, 0.75) * 6));
      [pl, ph].forEach((arr) => arr.forEach((p, i) => { p.style.opacity = i > n ? 0 : i === n ? 0.9 : 0.12 + 0.35 * (i / Math.max(1, n)); }));
      fade(fin, seg(t, 0.75, 0.85)); it.textContent = `iteration ${n + 1}`;
    } };
  })();

  Object.assign(window.DissScenes, scenes);
})();
// ---- from the site's dissertation/detour.js
// The detour map, under the dissertation's title: ninety years of economics going around one problem.
// A band runs across the map from 1936, where Keynes described it: people who learn from each other's actions.
// Each line of work comes toward the band in its year, bends, and runs alongside it from then on, so over the
// scroll the band gets outlined by the ways around it. In 2026 one line goes straight through, and each old line's
// direct route into the band appears. Time runs left to right on a wide screen, top to bottom on a phone.
(function () {
  "use strict";
  const { el, mulberry32, clamp, seg } = Sketch;   // static/js/sketch.js

  // a pencil line through points, each segment bowed a little (by up to half of wob either way) so it reads as drawn,
  // and drawn twice, as a pencil goes over its own stroke
  const pencil = (pts, seed, wob) => Sketch.pencil(pts, seed, wob / 2);
  const stroke = (g, d, cls, w) => Sketch.stroke2(g, d, cls, w, "pn", 0.7, 0.5);
  // a smooth curve through control points (Catmull-Rom), sampled
  function smooth(cp, n = 10) {
    const out = [];
    for (let i = 0; i < cp.length - 1; ++i) {
      const p0 = cp[Math.max(0, i - 1)], p1 = cp[i], p2 = cp[i + 1], p3 = cp[Math.min(cp.length - 1, i + 2)];
      for (let k = 0; k < n; ++k) {
        const t = k / n, t2 = t * t, t3 = t2 * t;
        const f = (a, b, c, d) => 0.5 * (2 * b + (-a + c) * t + (2 * a - 5 * b + 4 * c - d) * t2 + (-a + 3 * b - 3 * c + d) * t3);
        out.push([f(p0[0], p1[0], p2[0], p3[0]), f(p0[1], p1[1], p2[1], p3[1])]);
      }
    }
    out.push(cp[cp.length - 1]);
    return out;
  }

  // The lines of work, each with the way it went around. From the dissertation's introduction.
  const Y0 = 1936, Y1 = 2026;
  const ROADS = [   // side 0 above the band, 1 below; row and anchor place the label clear of the other roads
    { y: 1962, idea: "one shared goal", sub: "team theory", side: 0, row: 1, an: "middle" },
    { y: 1972, idea: "price takers", sub: "Lucas’s islands", side: 0, row: 0, an: "middle" },
    { y: 1980, idea: "competitive traders", sub: "rational expectations", side: 1, row: 0, an: "end" },
    { y: 1983, idea: "a lag reveals all", sub: "Townsend", side: 0, row: 1, an: "middle" },
    { y: 1985, idea: "one insider", sub: "Kyle", side: 1, row: 2, an: "middle", kyle: true },
    { y: 1996, idea: "identical insiders", sub: "symmetric markets", side: 0, row: 2, an: "middle" },
    { y: 2000, idea: "negligible agents", sub: "frequency domain", side: 1, row: 1, an: "end" },
    { y: 2002, idea: "one round", sub: "beauty contests", side: 0, row: 1, an: "start" },
    { y: 2007, idea: "a continuum of players", sub: "mean-field games", side: 1, row: 2, an: "middle" },
    { y: 2011, idea: "static information", sub: "information design", side: 0, row: 0, an: "start" },
    { y: 2013, idea: "a shared record", sub: "common information", side: 1, row: 0, an: "start" },
  ];
  const BAND = [0.44, 0.56];

  function draw(svg, opts) {
    const vertical = !!opts.vertical;
    svg.innerHTML = "";
    const W = vertical ? 500 : 1000, H = vertical ? 800 : 560;
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    // u: time, 0 at 1936 and 1 at 2026; v: across the map, the band in the middle
    const mu = vertical ? [70, 40] : [70, 36], vEdge = vertical ? 0.33 : 0.215;
    const P = (u, v) => vertical ? [v * W, mu[0] + u * (H - mu[0] - mu[1])] : [mu[0] + u * (W - mu[0] - mu[1]), v * H];
    const U = (year) => (year - Y0) / (Y1 - Y0);
    const g = el("g", {}, svg);
    const parts = [];   // { set(t) } with the scroll window it draws in

    // the band: faint hatching, two wobbly edges drawn from 1936 on, its name
    const band = el("g", { class: "band" }, g);
    const hatch = el("g", { class: "hatch" }, band);
    const r = mulberry32(7);
    for (let u = 0.004; u < 1; u += vertical ? 0.012 : 0.008) {
      const a = P(u, BAND[0] + 0.004), b = P(u + (vertical ? 0.02 : 0.014), BAND[1] - 0.004);
      el("path", { d: pencil([a, b], Math.floor(r() * 1e6), 1.5), class: "pn", "stroke-width": 0.7 }, hatch);
    }
    const edgeA = [], edgeB = [];
    for (let i = 0; i <= 60; ++i) { const u = i / 60; edgeA.push(P(u, BAND[0] + (r() - 0.5) * 0.004)); edgeB.push(P(u, BAND[1] + (r() - 0.5) * 0.004)); }
    const ew = 1.3, eA = stroke(band, pencil(edgeA, 3, 1), "", ew), eB = stroke(band, pencil(edgeB, 4, 1), "", ew);
    parts.push({ set: (t) => { eA.set(t); eB.set(t); hatch.style.opacity = 0.9 * t; }, w: [0.0, 0.07] });

    let bandLabel = null, unmapped = null;
    const [lx, ly] = P(vertical ? 0.2 : 0.5, 0.5);
    bandLabel = el("text", { x: lx, y: ly + 5, class: "bandname halo", "text-anchor": "middle",
      transform: vertical ? `rotate(90 ${lx} ${ly})` : "" }, g);
    bandLabel.textContent = "people who learn from each other’s actions";
    const [kx, ky] = P(0, BAND[0]);
    const flag = el("g", { class: "keynes" }, g);
    el("circle", { cx: kx, cy: ky + (vertical ? 0 : 0), r: 3.2, class: "dot" }, flag);
    const kt = el("text", { x: vertical ? kx + 34 : kx - 4, y: vertical ? ky - 26 : ky - 34, class: "lab", "text-anchor": vertical ? "start" : "start" }, flag);
    kt.innerHTML = `<tspan class="who">Keynes 1936</tspan><tspan x="${vertical ? kx + 34 : kx - 4}" dy="15">the beauty contest</tspan>`;
    stroke(flag, pencil([[kx, ky], vertical ? [kx + 30, ky - 30] : [kx + 2, ky - 30]], 9, 0.6), "soft2", 0.8).set(1);
    parts.push({ set: (t) => { flag.style.opacity = t; bandLabel.style.opacity = t; }, w: [0.01, 0.06] });
    // "unmapped" goes in the band clear of the band's name: measured, then placed in the longer free stretch beside it
    unmapped = el("text", { class: "mono halo", "text-anchor": "middle" }, g);
    unmapped.textContent = "unmapped";
    const along = (u) => { const [x, y] = P(u, 0.5); return vertical ? y : x; };
    const L = bandLabel.getComputedTextLength ? bandLabel.getComputedTextLength() : 0, uW = unmapped.getComputedTextLength ? unmapped.getComputedTextLength() : 60;
    const c = along(vertical ? 0.2 : 0.5), a0 = along(0), a1 = along(1), gap = 28;
    const before = [a0 + gap, c - L / 2 - gap], after = [c + L / 2 + gap, a1 - gap];
    const room = (s) => s[1] - s[0];
    const pick = room(after) >= room(before) ? after : before;
    const m = room(pick) >= uW ? (pick === after ? pick[0] + uW / 2 : pick[1] - uW / 2) : null;
    if (m === null) { unmapped.remove(); unmapped = null; }
    else {
      const [ux, uy] = vertical ? [P(0, 0.5)[0], m] : [m, P(0, 0.5)[1]];
      unmapped.setAttribute("x", ux); unmapped.setAttribute("y", uy + 4);
      if (vertical) unmapped.setAttribute("transform", `rotate(90 ${ux} ${uy})`);
    }

    // the roads: down from the map's edge in their year, a bend, then alongside the band to the present
    const lane = [0, 0];
    const doors = [];
    ROADS.forEach((rd, i) => {
      const u0 = U(rd.y), s = rd.side, sgn = s === 0 ? -1 : 1;
      const k = lane[s]++, off = vertical ? 0.016 + 0.015 * k : 0.022 + 0.021 * k;
      const vBand = s === 0 ? BAND[0] : BAND[1];
      const vRun = vBand + sgn * off, vStart = s === 0 ? vEdge : 1 - vEdge;
      const bend = 0.028;
      let cp;
      if (rd.kyle) {   // the closest approach: into the edge of the band and back out
        cp = [[u0, vStart], [u0, vRun + sgn * 0.05], [u0 + 0.004, vBand - sgn * 0.012], [u0 + 0.02, vBand - sgn * 0.032],
              [u0 + 0.038, vBand - sgn * 0.008], [u0 + 0.056, vRun], [u0 + 0.08, vRun], [0.97, vRun]];
      } else cp = [[u0, vStart], [u0, vRun + sgn * 0.05], [u0 + bend * 0.6, vRun + sgn * 0.004], [u0 + bend * 1.6, vRun], [0.97, vRun]];
      const pts = smooth(cp.map(([u, v]) => P(u, v)), 8);
      const line = stroke(g, pencil(pts, 20 + i, 1.2), "road", 1.4);
      // each road is drawn while its caption is up: one each through Kyle, then the rest together
      const t0 = i < 5 ? [0.08, 0.17, 0.26, 0.34, 0.42][i] : 0.5 + 0.022 * (i - 5), t1 = t0 + (i < 5 ? 0.06 : 0.04);
      parts.push({ set: line.set, w: [t0, t1] });
      doors.push({ from: P(u0 + bend * 1.6, vRun), to: P(u0 + bend * 1.6, 0.5), i });
      const [x, y] = P(u0, vStart);
      const lab = el("g", { class: "roadlab" }, g);
      let tx, ty, anchor;
      if (vertical) { tx = s === 0 ? x - 8 : x + 8; ty = y - 4; anchor = s === 0 ? "end" : "start"; }
      else {
        const nudge = rd.an === "end" ? 8 : rd.an === "start" ? -8 : 0;
        tx = x + nudge; anchor = rd.an;
        ty = s === 0 ? y - 26 - rd.row * 36 : y + 20 + rd.row * 36;
        if (rd.row) stroke(lab, pencil([[x, s === 0 ? ty + 20 : ty - 16], [x, y + (s === 0 ? -3 : 3)]], 40 + i, 0.3), "soft2 dash", 0.7).set(1);
      }
      const t = el("text", { x: tx, y: ty, class: "lab", "text-anchor": anchor }, lab);
      t.innerHTML = `<tspan class="who">${rd.idea}</tspan><tspan x="${tx}" dy="14">${rd.y} · ${rd.sub}</tspan>`;
      parts.push({ set: (q) => { lab.style.opacity = q; }, w: [t0, t0 + 0.04] });
    });

    // 2026: straight through
    const uT = 0.985;
    const thru = stroke(g, pencil(smooth([P(uT, vEdge - 0.02), P(uT, 0.5), P(uT, 1 - vEdge + 0.02)], 12), 99, 0.8), "acc", 2.4);
    parts.push({ set: thru.set, w: [0.8, 0.88] });
    const [x, y] = vertical ? P(uT, vEdge - 0.02) : P(uT, 1 - vEdge + 0.02);
    const tx = vertical ? x - 8 : x + 6;
    const t = el("text", { x: tx, y: vertical ? y - 4 : y + 56, class: "lab accl", "text-anchor": "end" }, g);
    t.innerHTML = `<tspan class="who">2026</tspan><tspan x="${tx}" dy="14">noise-state calculus</tspan>`;
    parts.push({ set: (q) => { t.style.opacity = q; }, w: [0.8, 0.83] });
    // and then each line's own way in
    doors.forEach((d) => {
      const n = 7, dg = el("g", { class: "door" }, g), bits = [];
      for (let k = 0; k < n; ++k) {
        const a = k / n, b = (k + 0.5) / n, L = (q) => [d.from[0] + (d.to[0] - d.from[0]) * q, d.from[1] + (d.to[1] - d.from[1]) * q];
        const e = el("path", { d: pencil([L(a), L(b)], 60 + d.i * 9 + k, 0.4), class: "pn acc", "stroke-width": 1.2 }, dg);
        e.style.opacity = 0; bits.push(e);
      }
      const t0 = 0.9 + 0.007 * d.i;
      parts.push({ set: (t) => bits.forEach((e, k) => { e.style.opacity = t * n > k ? 0.85 : 0; }), w: [t0, t0 + 0.03] });
    });
    parts.push({ set: (t) => { hatch.style.opacity = 0.9 - 0.55 * t; if (unmapped) unmapped.style.opacity = 1 - t; }, w: [0.9, 0.99] });
    return function (p) { for (const q of parts) q.set(seg(p, q.w[0], q.w[1])); };
  }

  window.DissDetour = draw;
})();
// ---- the screens' side: a drawing the site drives by scroll is run here by a play control (space), from its start to its end
(function () {
  "use strict";
  const scenes = window.DissScenes, NOW = 1;   // the drawings' own clock stands still, so nothing sways or drifts by itself
  const players = new Map();
  let map = null;
  for (const sec of document.querySelectorAll("#diss .or-slide")) {
    const svg = sec.querySelector("svg[data-scene]");
    if (!svg) continue;
    const input = sec.querySelector("input.di-scrub"), ctl = sec.querySelector(".di-play"), btn = sec.querySelector(".di-playbtn"), trial = sec.querySelector(".di-trial");
    const D = svg.dataset, dur = +D.dur || 0, pos0 = input ? +input.value / 1000 : 0;
    // TRIAL: a screen may carry a second way to be shown (data-alt-*), with a switch on it; "before" is the screen as written
    const ways = { before: { scene: D.scene, lo: +D.lo, hi: +D.hi, view: svg.getAttribute("viewBox"), start: pos0, stops: (D.stops || "").split(",").filter(Boolean).map(Number) } };
    if (trial) ways.after = { scene: D.altScene || D.scene, lo: +(D.altLo || D.lo), hi: +(D.altHi || D.hi), view: D.altView || ways.before.view, start: D.altStart ? +D.altStart : pos0, stops: (D.altStops || "").split(",").filter(Boolean).map(Number) };
    let W = ways.before, name = W.scene;
    let pos = pos0, playing = false, target = 1;   // pos: how far through the run, 0 to 1; target: where this leg of it ends
    const at = () => (input ? W.lo + (W.hi - W.lo) * pos : +D.at);
    let raf = 0, last = 0, sc = null, until = 0;   // until: a price wave set off by a click is drawn till then
    const mark = () => { if (ctl) ctl.dataset.state = playing ? "playing" : pos >= 1 ? "done" : "ready"; };
    function frame(now) {
      raf = 0;
      const dt = last ? Math.min(0.05, (now - last) / 1000) : 0; last = now;
      if (playing) {
        if (sec.hidden || !sec.offsetParent) playing = false;   // left behind: it waits where it was
        else { pos = Math.min(target, pos + dt / dur); input.value = Math.round(pos * 1000); if (pos >= target) playing = false; }
        mark();
      }
      let busy = playing || now < until;
      if (name === "detour") map(at());
      else busy = sc.update(at(), name === "prices" ? now : NOW, sc.shots ? dt : 0) || (sc.shots && sc.shots.some((s) => s.moving)) || busy;   // a timed motion set off on the way (the graphite sliding home) is played to its end
      if (busy) raf = requestAnimationFrame(frame); else last = 0;
    }
    const wake = () => { if (!raf) raf = requestAnimationFrame(frame); };
    const jump = () => { if (sc && sc.shots) sc.shots.forEach((s) => s.jump(at())); };
    function show() {
      if (name === "detour") {
        if (!map) { map = window.DissDetour(window.DissMapStage, { vertical: false }); while (window.DissMapStage.firstChild) svg.appendChild(window.DissMapStage.firstChild); }
      } else {
        if (sc && sc !== scenes[name] && sc.g.parentNode === svg) svg.removeChild(sc.g);
        sc = scenes[name];
        if (sc.g.parentNode !== svg) svg.appendChild(sc.g);   // a drawing shown on two screens goes to whichever is up
        sc.g.style.transition = "none"; sc.g.style.opacity = 1;
        jump();   // arrived at, not played to: already where it was left
      }
      last = 0; wake();
    }
    function toggle() {
      if (playing) playing = false;
      else {
        if (pos >= 1) { pos = W.start; input.value = Math.round(pos * 1000); jump(); }   // done: space plays it again from its start
        target = W.stops.find((s) => s > pos + 1e-6) || 1; playing = true; last = 0;   // a run with a stop in it plays to the stop and waits there
      }
      mark(); wake();
    }
    svg.addEventListener("pointerdown", () => { if (name === "prices") { until = performance.now() + 500; last = 0; wake(); } });   // after that the drawing keeps the frames coming while its wave is on its way
    if (input) {
      input.addEventListener("input", () => { pos = +input.value / 1000; playing = false; mark(); wake(); });   // a and d nudge it by hand, and stop the run
      btn.addEventListener("click", () => { toggle(); btn.blur(); });
      players.set(sec, toggle); mark();
    }
    if (trial) {
      const key = "trial-diss-" + D.scene;
      const set = (v) => {
        W = ways[v]; name = W.scene; playing = false; pos = W.start; input.value = Math.round(pos * 1000); svg.setAttribute("viewBox", W.view);
        try { localStorage.setItem(key, v); } catch (e) {}
        for (const b of trial.querySelectorAll("button")) b.classList.toggle("on", b.dataset.v === v);
        mark(); if (!sec.hidden) show();
      };
      let v = "after"; try { v = localStorage.getItem(key) || v; } catch (e) {} set(ways[v] ? v : "after");
      trial.addEventListener("click", (ev) => { const b = ev.target.closest("button"); if (b) { set(b.dataset.v); b.blur(); } });
    }
    new MutationObserver(() => { if (sec.hidden) { playing = false; mark(); } else show(); }).observe(sec, { attributes: true, attributeFilter: ["hidden"] });
    if (!sec.hidden) show();
  }
  // space, from the page's own script: true if the screen that is up has a drawing to run
  window.DissPlay = () => { const sec = document.querySelector("#diss .or-slide:not([hidden])"), t = sec && players.get(sec); if (t) t(); return !!t; };
})();
