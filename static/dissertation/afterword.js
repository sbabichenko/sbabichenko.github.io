// The afterword, illustrated: each paragraph beside a pencil drawing that builds as it scrolls past. An illusion whose
// lines are the same length; lineages that branch and stabilize; an anchoring experiment; papers whose promises
// stop at a wall; an orbit; a signal hidden in noise; the two simple questions.
(function () {
  "use strict";
  const NS = "http://www.w3.org/2000/svg";
  const svg = document.getElementById("stage");
  const steps = [...document.querySelectorAll(".step")];
  if (!svg || !steps.length) return;
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const el = (tag, attrs, parent) => { const e = document.createElementNS(NS, tag); for (const [k, v] of Object.entries(attrs || {})) e.setAttribute(k, v); if (parent) parent.appendChild(e); return e; };
  function mulberry32(a) { return function () { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
  function gauss(r) { let u = 0; while (u === 0) u = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r()); }
  const clamp = (x, a = 0, b = 1) => Math.max(a, Math.min(b, x));
  const ease = (t) => (t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2);
  const seg = (t, a, b) => ease(clamp((t - a) / (b - a)));
  const fade = (n, t) => { n.style.opacity = clamp(t); };
  function pencil(pts, seed, wob = 1.2) {
    const r = mulberry32(seed);
    let d = `M${pts[0][0].toFixed(1)},${pts[0][1].toFixed(1)}`;
    for (let i = 1; i < pts.length; ++i) {
      const [x0, y0] = pts[i - 1], [x1, y1] = pts[i];
      d += ` Q${((x0 + x1) / 2 + (r() - 0.5) * wob * 2).toFixed(1)},${((y0 + y1) / 2 + (r() - 0.5) * wob * 2).toFixed(1)} ${x1.toFixed(1)},${y1.toFixed(1)}`;
    }
    return d;
  }
  function stroke(g, d, cls, w = 1.6) {
    const a = el("path", { d, class: "pencil " + (cls || ""), "stroke-width": w }, g);
    const b = el("path", { d, class: "pencil soft " + (cls || ""), "stroke-width": w * 0.6, transform: "translate(0.8,0.6)" }, g);
    const len = a.getTotalLength();
    for (const p of [a, b]) { p.style.strokeDasharray = `${len} ${len}`; p.style.strokeDashoffset = len; }
    return { set(t) { const o = len * (1 - clamp(t)); a.style.strokeDashoffset = o; b.style.strokeDashoffset = o; }, a, b };
  }
  const line = (g, x1, y1, x2, y2, seed, cls, w) => stroke(g, pencil([[x1, y1], [x2, y2]], seed, 0.8), cls, w);
  function text(g, x, y, s, cls, anchor = "middle") { const t = el("text", { x, y, class: cls || "", "text-anchor": anchor }, g); t.textContent = s; return t; }
  const scenes = {};
  const group = (name) => { const g = el("g", { "data-scene": name }, svg); g.style.opacity = 0; g.style.transition = "opacity 0.6s"; return g; };

  // -- Müller-Lyer: the same length, told otherwise by the fins
  scenes.illusion = (() => {
    const g = group("illusion");
    const shafts = [line(g, 170, 220, 430, 220, 1, "", 2.4), line(g, 170, 380, 430, 380, 2, "", 2.4)];
    const fins = [];
    // the top pair's fins point outward and the bottom pair's inward: the same shaft, read as longer and shorter
    for (const [x, y, side, sd] of [[170, 220, -1, 3], [430, 220, 1, 5], [170, 380, -1, 7], [430, 380, 1, 9]]) {
      const reach = (y === 220 ? 1 : -1) * side * 38;
      fins.push(stroke(g, pencil([[x + reach, y - 34], [x, y], [x + reach, y + 34]], sd, 0.6), "", 2.2));
    }
    const guides = [line(g, 170, 170, 170, 430, 11, "accent", 1.2), line(g, 430, 170, 430, 430, 12, "accent", 1.2)];
    guides.forEach((q) => { q.a.setAttribute("stroke-dasharray", "4 6"); });
    return { g, update(t) {
      shafts.forEach((s, i) => s.set(seg(t, 0.02 + i * 0.06, 0.25 + i * 0.06)));
      fins.forEach((f, i) => f.set(seg(t, 0.18 + i * 0.06, 0.4 + i * 0.06)));
      guides.forEach((q) => { q.a.style.strokeDasharray = "4 6"; q.a.style.strokeDashoffset = 0; q.b.style.display = "none"; fade(q.a, seg(t, 0.6, 0.75)); });
    } };
  })();

  // -- lineages: branching from a root, most ending, a few carrying on, and the ground moving under them
  scenes.selection = (() => {
    const g = group("selection"), r = mulberry32(377), segs = [];
    const words = ["laws", "economies", "languages", "traditions", "fairness"];
    const tips = [];
    // a living lineage grows until it reaches the top; a dead one stops after a few generations. Each branch leans a
    // little back toward upright, so the living ones climb rather than sprawl. The seed is one whose tree stays on the
    // page and whose living tips spread wide enough for the five words.
    function grow(x, y, a, depth, alive, seed) {
      const L = 46 + r() * 20, nx = x + Math.cos(a) * L, ny = y - Math.sin(a) * L;
      segs.push({ s: stroke(g, pencil([[x, y], [nx, ny]], seed, 1.2), alive ? "" : "soft", alive ? 1.7 : 1.1), depth });
      if (alive ? ny < 190 || depth >= 11 : depth >= 6 || ny < 200) { if (alive) tips.push([nx, ny]); return; }
      const kids = r() < 0.75 ? 2 : 1;
      for (let k = 0; k < kids; ++k) {
        const na = a + 0.1 * (Math.PI / 2 - a) + (kids === 2 ? (k ? 0.36 : -0.36) : 0) + (r() - 0.5) * 0.3;
        // most branches die out; the ones that live are the ones still reaching the top
        grow(nx, ny, na, depth + 1, alive && (r() < 0.62 || depth < 2), seed * 7 + k + 1);
      }
    }
    grow(300, 560, Math.PI / 2, 0, true, 3);
    tips.sort((a, b) => a[0] - b[0]);
    const pick = tips.length <= 5 ? tips : [0, 0.25, 0.5, 0.75, 1].map((q) => tips[Math.round(q * (tips.length - 1))]);
    // each word above its tip, nudged sideways just enough that neighbors do not touch
    const labels = pick.map((p, i) => text(g, p[0], p[1] - 14, words[i] || "", "label acc"));
    const wd = labels.map((l, i) => { try { return l.getComputedTextLength() || 8 * words[i].length; } catch (e) { return 8 * words[i].length; } });
    const lx = pick.map((p) => p[0]);
    for (let i = 1; i < lx.length; ++i) lx[i] = Math.max(lx[i], lx[i - 1] + (wd[i - 1] + wd[i]) / 2 + 12);
    const shift = lx.reduce((a, v, i) => a + v - pick[i][0], 0) / lx.length;
    labels.forEach((l, i) => l.setAttribute("x", (lx[i] - shift).toFixed(1)));
    const band = el("path", { class: "pencil accent soft", "stroke-width": 1.4, "stroke-dasharray": "3 5" }, g);
    return { g, update(t, now) {
      segs.forEach((q) => q.s.set(seg(t, 0.02 + q.depth * 0.055, 0.12 + q.depth * 0.055)));
      labels.forEach((l, i) => fade(l, seg(t, 0.66 + i * 0.03, 0.76 + i * 0.03)));
      // the environment the tips answer to keeps moving
      const pts = []; for (let i = 0; i <= 30; ++i) { const x = 40 + 520 * i / 30; pts.push([x, 84 + 14 * Math.sin(i / 3 + now / 1400)]); }
      band.setAttribute("d", pencil(pts, 99, 0.2)); fade(band, seg(t, 0.72, 0.85));
    } };
  })();

  // -- anchoring: bids against the last two digits of a social security number
  scenes.anchor = (() => {
    const g = group("anchor"), r = mulberry32(7);
    const X = (d) => 90 + 440 * d / 99, Y = (b) => 500 - b * 3.4;
    const ax = [line(g, 80, 500, 540, 500, 1, "", 1.3), line(g, 80, 500, 80, 120, 2, "", 1.3)];
    const xl = text(g, 310, 540, "last two digits of the subject's social security number", "mono");
    const yl = el("text", { x: 50, y: 310, class: "mono", "text-anchor": "middle", transform: "rotate(-90 50 310)" }, g); yl.textContent = "bid";
    const pts = [];
    for (let i = 0; i < 44; ++i) {
      const d = Math.floor(r() * 100), b = 18 + 0.62 * d + gauss(r) * 13;
      pts.push(el("circle", { cx: X(d), cy: Y(clamp(b, 2, 108)), r: 4, class: "fillacc" }, g));
    }
    const fit = line(g, X(0), Y(18), X(99), Y(18 + 0.62 * 99), 5, "warm", 2.4);
    const note = text(g, 310, 575, "illustrative: the shape of the result, not the study's data", "mono");
    return { g, update(t) {
      ax.forEach((a) => a.set(seg(t, 0, 0.15))); fade(xl, seg(t, 0.05, 0.2)); fade(yl, seg(t, 0.05, 0.2));
      pts.forEach((p, i) => fade(p, seg(t, 0.12 + i * 0.01, 0.2 + i * 0.01) * 0.85));
      fit.set(seg(t, 0.62, 0.82)); fade(note, seg(t, 0.85, 1));
    } };
  })();

  // -- papers: introductions that promise, bodies that stop at a wall
  scenes.barrier = (() => {
    const g = group("barrier"), pages = [];
    for (let k = 0; k < 4; ++k) {
      const x = 70 + k * 118, y = 150 + (k % 2) * 26, pg = el("g", { transform: `rotate(${(k - 1.5) * 2} ${x + 50} ${y + 150})` }, g);
      el("rect", { x, y, width: 100, height: 300, rx: 3, class: "box pencil", "stroke-width": 1.2 }, pg);
      const intro = [], body = [];
      for (let i = 0; i < 5; ++i) intro.push(line(pg, x + 10, y + 22 + i * 12, x + 90 - (i === 4 ? 30 : 0), y + 22 + i * 12, k * 40 + i, "accent", 1.4));
      for (let i = 0; i < 14; ++i) body.push(line(pg, x + 10, y + 100 + i * 13, x + 90 - (i % 5 === 4 ? 25 : 0), y + 100 + i * 13, k * 40 + 10 + i, "soft", 1.2));
      pages.push({ intro, body });
    }
    const wall = line(g, 40, 236, 560, 236, 77, "warm", 3);
    return { g, update(t) {
      pages.forEach((p, k) => {
        p.intro.forEach((l, i) => l.set(seg(t, 0.02 + k * 0.06 + i * 0.02, 0.12 + k * 0.06 + i * 0.02)));
        // below the wall the lines only get so far before they stop
        p.body.forEach((l, i) => l.set(seg(t, 0.35 + k * 0.04, 0.6 + k * 0.04) * (0.25 + 0.2 * ((i * 7 + k * 3) % 5) / 4)));
      });
      wall.set(seg(t, 0.3, 0.45));
    } };
  })();

  // -- an orbit, and the law that writes it down
  scenes.orbit = (() => {
    const g = group("orbit");
    const a = 190, e = 0.55, b = a * Math.sqrt(1 - e * e), cx = 300, cy = 300, fx = cx - a * e;
    const pts = []; for (let i = 0; i <= 80; ++i) { const th = (i / 80) * Math.PI * 2; pts.push([cx + a * Math.cos(th), cy + b * Math.sin(th)]); }
    const orbit = stroke(g, pencil(pts, 5, 0.6), "", 1.6);
    const sun = el("circle", { cx: fx, cy, r: 13, class: "fillwarm" }, g);
    const planet = el("circle", { r: 7, class: "fillacc" }, g);
    const radius = el("line", { class: "pencil soft", "stroke-width": 1, "stroke-dasharray": "3 4" }, g);
    const law = text(g, 300, 540, "F = G m M / r²", "label");
    let M = 0;
    return { g, update(t, now, dt) {
      orbit.set(seg(t, 0.02, 0.3)); fade(sun, seg(t, 0.1, 0.2));
      // Kepler's equation, by a few Newton steps: faster near the sun
      M += dt * 1.3;
      let E = M; for (let k = 0; k < 6; ++k) E -= (E - e * Math.sin(E) - M) / (1 - e * Math.cos(E));
      const x = cx + a * Math.cos(E), y = cy + b * Math.sin(E);
      planet.setAttribute("cx", x); planet.setAttribute("cy", y); fade(planet, seg(t, 0.25, 0.35));
      radius.setAttribute("x1", fx); radius.setAttribute("y1", cy); radius.setAttribute("x2", x); radius.setAttribute("y2", y);
      fade(radius, seg(t, 0.35, 0.45) * 0.8);
      fade(law, seg(t, 0.5, 0.65));
    } };
  })();

  // -- a signal in noise: dots first, then the curve they were drawn from
  scenes.noise = (() => {
    const g = group("noise"), r = mulberry32(12);
    const f = (x) => 60 * Math.sin(x / 70) + 30 * Math.sin(x / 23 + 1);
    const dots = [];
    for (let i = 0; i < 160; ++i) {
      const x = 60 + r() * 480, y = 300 - f(x) + gauss(r) * 55;
      dots.push(el("circle", { cx: x, cy: y, r: 2.6, class: "fillacc" }, g));
      dots[i].style.fill = "currentColor";
    }
    const pts = []; for (let x = 60; x <= 540; x += 6) pts.push([x, 300 - f(x)]);
    const curve = stroke(g, pencil(pts, 3, 0.3), "warm", 2.6);
    return { g, update(t) {
      dots.forEach((d, i) => fade(d, seg(t, 0.02 + (i % 40) * 0.005, 0.1 + (i % 40) * 0.005) * (0.8 - 0.35 * seg(t, 0.6, 0.8))));
      curve.set(seg(t, 0.55, 0.85));
    } };
  })();

  // -- the two simple questions, written out
  scenes.questions = (() => {
    const g = group("questions");
    const make = (y, s, cls, seed) => {
      const clip = el("clipPath", { id: "clip" + seed }, g), rect = el("rect", { x: 0, y: y - 70, width: 0, height: 100 }, clip);
      const tt = el("text", { x: 300, y, "text-anchor": "middle", class: cls, "clip-path": `url(#clip${seed})` }, g);
      tt.textContent = s; tt.style.fontSize = "46px";
      const u = stroke(g, pencil([[140, y + 18], [300, y + 22], [460, y + 16]], seed, 3), cls === "acc" ? "accent" : "warm", 2);
      return { set(t) { rect.setAttribute("width", 600 * clamp(t)); u.set(seg(t, 0.85, 1)); } };
    };
    const a = make(250, "What is optimal?", "acc", 91), b = make(370, "What is a relic?", "warmt", 92);
    return { g, update(t) { a.set(seg(t, 0.05, 0.45)); b.set(seg(t, 0.4, 0.8)); } };
  })();

  // ------------------------------------------------------------------ scroll to scene
  // A step's progress runs from 0 as it becomes the active step (its top at the reading line) to 1 once 70% of it has
  // passed the line, so its drawing builds while the paragraph is read, on the way down as well as up. The drawing
  // follows that progress no faster than PACE a second: a flick of the wheel is caught up over a moment, not skipped.
  // Frames are asked for only while something moves.
  const PACE = 0.9;
  let active = null, prog = 0, shown = 0, current = null, raf = 0, last = 0, still = null, from = null;
  function measure() {
    const vh = window.innerHeight, line = window.readLine ? window.readLine() : vh * 0.55;
    let best = null, bestD = Infinity;
    for (const s of steps) { const r = s.getBoundingClientRect(), d = Math.abs(r.top + r.height / 2 - line); if (d < bestD) { bestD = d; best = s; } }
    if (!best) return;
    const r = best.getBoundingClientRect();
    prog = clamp((line - r.top) / (r.height * 0.7));
    if (best !== active) {
      active = best;
      for (const s of steps) s.classList.toggle("on", s === best);
    }
    wake();
  }
  function wake() { if (!raf) raf = requestAnimationFrame(frame); }
  window.addEventListener("scroll", measure, { passive: true });
  window.addEventListener("resize", measure);
  function frame(now) {
    raf = 0;
    const dt = last ? Math.min(0.05, (now - last) / 1000) : 0; last = now;
    const story = document.getElementById("story").getBoundingClientRect();
    if (!active || story.top >= window.innerHeight || story.bottom <= 0) { last = 0; return; }   // asleep until a scroll
    const want = active.dataset.scene;
    if (want !== current) {
      // reached from above, a drawing builds from nothing; from below, it is there already
      shown = !from || steps.indexOf(active) > steps.indexOf(from) ? 0 : prog; current = want; from = active;
      for (const [name, sc] of Object.entries(scenes)) sc.g.style.opacity = name === want ? 1 : 0;
    }
    const sc = scenes[want];
    if (!sc) { last = 0; return; }
    // reduced motion: each scene is drawn once, finished and still, when it becomes active
    if (reduced) { if (still !== active) { still = active; sc.update(1, 0, 0); } last = 0; return; }
    shown = Math.abs(prog - shown) <= PACE * dt ? prog : shown + Math.sign(prog - shown) * PACE * dt;
    const busy = sc.update(shown, now, dt);
    // a scene that moves by itself (its update reads the clock) keeps asking for frames while the story is on screen
    if (busy || shown !== prog || (sc.live !== undefined ? sc.live : sc.update.length > 1)) raf = requestAnimationFrame(frame);
    else last = 0;
  }
  measure();
})();
