// The pencil drawings' shared kit, for the scripts that draw in SVG (the dissertation's opening, detour map, stories,
// vignettes and afterword, the spike and risk pages, the solver's and the mesh's deep dives): elements, a seeded random
// stream, easing, and pencil lines that bow the same way for the same seed on every visit. Loaded before the page's
// own script, which takes what it needs from window.Sketch.
window.Sketch = (() => {
  "use strict";
  const NS = "http://www.w3.org/2000/svg";
  const el = (tag, attrs, parent) => { const n = document.createElementNS(NS, tag); for (const [k, v] of Object.entries(attrs || {})) n.setAttribute(k, v); if (parent) parent.appendChild(n); return n; };
  // a small seeded generator (mulberry32), so a drawing is the same on every visit; and a standard normal from it
  function mulberry32(a) { return function () { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
  function gauss(r) { let u = 0; while (u === 0) u = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r()); }
  const clamp = (x, a = 0, b = 1) => Math.max(a, Math.min(b, x));
  const ease = (t) => (t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2);
  const seg = (t, a, b) => ease(clamp((t - a) / (b - a)));   // 0 before a, 1 after b
  const fade = (n, t) => { n.style.opacity = clamp(t); };
  // A one-shot motion: once the step's progress passes `at` it plays over `dur` seconds whatever the scroll speed, and
  // runs back if the reader scrolls above `at` again. With reduced motion it jumps to where it is headed. step(t, dt)
  // returns 0..1, not eased.
  function oneShot(at, dur, reduced) {
    let v = 0, to = 0;
    return {
      step(t, dt) { to = t >= at ? 1 : 0; v = reduced ? to : v + clamp(to - v, -dt / dur, dt / dur); return v; },
      get moving() { return v !== to; },
      get playing() { return to === 1 && v < 1; },     // on its way forward: worth finishing before the drawing is put away
      jump(t) { to = v = t >= at ? 1 : 0; },           // reached from below: already where the progress puts it
    };
  }
  // a pencil line through points: each segment bowed by up to wob either way, deterministically, so it reads as drawn
  function pencil(pts, seed, wob) {
    const r = mulberry32(seed);
    let d = `M${pts[0][0].toFixed(1)},${pts[0][1].toFixed(1)}`;
    for (let i = 1; i < pts.length; ++i) { const [x0, y0] = pts[i - 1], [x1, y1] = pts[i]; d += ` Q${((x0 + x1) / 2 + (r() - 0.5) * wob * 2).toFixed(1)},${((y0 + y1) / 2 + (r() - 0.5) * wob * 2).toFixed(1)} ${x1.toFixed(1)},${y1.toFixed(1)}`; }
    return d;
  }
  // a path drawn on by the progress: set(t) uncovers it from 0 to 1
  function stroke(g, d, cls, width = 1.6) {
    const a = el("path", { d, class: "pencil " + (cls || ""), "stroke-width": width }, g);
    const len = a.getTotalLength() || 1;
    a.style.strokeDasharray = `${len} ${len}`; a.style.strokeDashoffset = len;
    return { a, set(t) { a.style.strokeDashoffset = len * (1 - clamp(t)); } };
  }
  // the same line drawn twice, the second faint and a hair off (dx, dy), as a pencil goes over its own stroke
  function stroke2(g, d, cls, w, base, dx, dy) {
    const a = el("path", { d, class: base + " " + (cls || ""), "stroke-width": w }, g);
    const b = el("path", { d, class: base + " soft " + (cls || ""), "stroke-width": w * 0.6, transform: `translate(${dx},${dy})` }, g);
    const len = a.getTotalLength();
    for (const p of [a, b]) { p.style.strokeDasharray = `${len} ${len}`; p.style.strokeDashoffset = len; }
    return { a, b, len, set(t) { const o = len * (1 - clamp(t)); a.style.strokeDashoffset = o; b.style.strokeDashoffset = o; } };
  }
  const line = (g, x1, y1, x2, y2, cls, w = 1) => el("line", { x1, y1, x2, y2, class: "pencil " + (cls || ""), "stroke-width": w }, g);
  const poly = (pts) => "M" + pts.map((p) => p[0].toFixed(1) + "," + p[1].toFixed(1)).join(" L");
  const text = (g, x, y, s, cls, anchor = "middle") => { const t = el("text", { x, y, class: cls || "", "text-anchor": anchor }, g); t.textContent = s; return t; };
  // The reading position in a scroll-told story: the step whose middle is nearest the reading line (the page's
  // window.readLine() where it sets one, else 55% down the window), and its progress, from 0 as its top reaches the
  // line to 1 once 70% of it has passed, so its drawing builds while the paragraph is read. null without steps.
  function reading(steps) {
    const vh = window.innerHeight, line = window.readLine ? window.readLine() : vh * 0.55;
    let best = null, bestD = Infinity;
    for (const s of steps) { const r = s.getBoundingClientRect(), d = Math.abs(r.top + r.height / 2 - line); if (d < bestD) { bestD = d; best = s; } }
    if (!best) return null;
    const r = best.getBoundingClientRect();
    return { step: best, prog: clamp((line - r.top) / (r.height * 0.7)) };
  }
  // a progress line's width: how far down the page the reader is
  const progressBar = (bar) => { if (bar) { const h = document.documentElement.scrollHeight - window.innerHeight; bar.style.width = (h > 0 ? (100 * window.scrollY) / h : 0) + "%"; } };
  // the drawing follows the reading progress no faster than pace a second: a flick of the wheel is caught up over a
  // moment, not skipped
  const follow = (shown, prog, dt, pace) => (Math.abs(prog - shown) <= pace * dt ? prog : shown + Math.sign(prog - shown) * pace * dt);
  // On a phone the drawings' labels are set larger (the page's stylesheet). A label that would then run past the
  // drawing's edge, 600 units wide, is shrunk back until it fits, measured from where it is anchored.
  function fitLabels(svg) {
    const phone = window.matchMedia("(max-width: 820px)");
    let fitQueued = false;
    function fit() {
      fitQueued = false;
      const mode = phone.matches ? "phone" : "wide";
      for (const t of svg.querySelectorAll("text")) {
        const key = mode + t.textContent;
        if (t.fitKey === key) continue;           // measured already, for this text at this width
        t.fitKey = key; t.style.fontSize = "";
        if (mode !== "phone" || !t.textContent) continue;
        const b = t.getBBox(), x = +t.getAttribute("x") || 0, a = t.getAttribute("text-anchor") || "start";
        const room = a === "middle" ? 2 * Math.min(x, 600 - x) : a === "end" ? x : 600 - x;
        if (b.width > room && room > 0) t.style.fontSize = (parseFloat(getComputedStyle(t).fontSize) * room / b.width).toFixed(2) + "px";
      }
    }
    const queueFit = () => { if (!fitQueued) { fitQueued = true; requestAnimationFrame(fit); } };
    new MutationObserver(queueFit).observe(svg, { childList: true, subtree: true, characterData: true });
    phone.addEventListener("change", queueFit);
  }
  return { NS, el, mulberry32, gauss, clamp, ease, seg, fade, oneShot, pencil, stroke, stroke2, line, poly, text, reading, progressBar, follow, fitLabels };
})();
