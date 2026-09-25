// The CV's margin: a random walk pinned at each entry (a Brownian bridge from one to the next), drawn in pencil
// as far down the page as you have read. In the margin where there is one; on a phone, in a slim gutter cut from
// the text's left edge.
(function () {
  "use strict";
  const art = document.querySelector("article");
  if (!art) return;
  const stops = [...art.querySelectorAll("p > strong:first-child")].filter((s) => !/:\s*$/.test(s.textContent));
  if (stops.length < 2) return;
  const NS = "http://www.w3.org/2000/svg";
  const el = (tag, attrs, parent) => { const n = document.createElementNS(NS, tag); for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v); if (parent) parent.appendChild(n); return n; };
  function gauss() { let u = 0; while (u === 0) u = Math.random(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * Math.random()); }
  let svg = null, path = null, dots = [], len = 0, pinLen = [], walker = null, walkAnim = 0;
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function build() {
    if (svg) svg.remove();
    art.classList.remove("cv-gutter");
    let a = art.getBoundingClientRect(), W, left;
    if (a.left >= 100) { W = 70; left = -W - 24; }                 // a real margin
    else if (a.left >= 56) { W = a.left - 24; left = -W - 12; }    // a narrow one (a small tablet, a phone unfolded)
    else { art.classList.add("cv-gutter"); W = 24; left = 0; a = art.getBoundingClientRect(); }   // none: make a gutter
    art.style.position = "relative";
    svg = el("svg", { class: "cvwalk", width: W, height: a.height, viewBox: `0 0 ${W} ${a.height}`, "aria-hidden": "true" }, art);
    Object.assign(svg.style, { position: "absolute", left: left + "px", top: 0, overflow: "visible", pointerEvents: "none" });
    const wide = W >= 70, amp = 3.2 * Math.min(1, W / 70) * (wide ? 1 : 1.4);
    const ys = stops.map((s) => s.getBoundingClientRect().top - a.top + 10), x0 = W / 2;
    let d = `M${x0},${ys[0]}`;
    for (let i = 1; i < ys.length; ++i) {
      // a Brownian bridge from x0 at ys[i-1] back to x0 at ys[i]
      const n = Math.max(8, Math.round((ys[i] - ys[i - 1]) / 6)), w = [0];
      for (let k = 1; k <= n; ++k) w.push(w[k - 1] + gauss());
      for (let k = 1; k <= n; ++k) {
        const b = w[k] - (k / n) * w[n], x = x0 + Math.max(-W / 2 + 3, Math.min(W / 2 - 3, b * amp));
        d += ` L${x.toFixed(1)},${(ys[i - 1] + ((ys[i] - ys[i - 1]) * k) / n).toFixed(1)}`;
      }
    }
    path = el("path", { d, fill: "none", stroke: "currentColor", "stroke-width": 1.2, "stroke-linejoin": "round", opacity: 0.45 }, svg);
    len = path.getTotalLength();
    path.style.strokeDasharray = `${len} ${len}`;
    dots = ys.map((y) => el("circle", { cx: x0, cy: y, r: wide ? 3.5 : 3, fill: "var(--accent, #1f3fd0)" }, svg));
    // where along the walk each pin sits (the walk only moves down, so its height finds the length)
    pinLen = ys.map((y) => { let lo = 0, hi = len; for (let k = 0; k < 30; ++k) { const m = (lo + hi) / 2; if (path.getPointAtLength(m).y < y) lo = m; else hi = m; } return hi; });
    walker = el("circle", { r: wide ? 2.6 : 2.2, fill: "currentColor", opacity: 0 }, svg);
    // the caption: centred over the walk in a wide margin, else under the last stop's pin, reading off to the right
    const cy = wide ? ys[0] - 14 : ys[ys.length - 1] + 16, cx = wide ? x0 : x0 - 3, anchor = wide ? "middle" : "start";
    const t = el("text", { x: cx, y: cy, "text-anchor": anchor, "font-size": 9, fill: "currentColor", opacity: 0.5, "font-family": "ui-monospace, Menlo, monospace" }, svg);
    t.textContent = "a random walk,"; const t2 = el("text", { x: cx, y: cy + 11, "text-anchor": anchor, "font-size": 9, fill: "currentColor", opacity: 0.5, "font-family": "ui-monospace, Menlo, monospace" }, svg);
    t2.textContent = "pinned at each stop";
    draw();
  }
  function draw() {
    if (!svg) return;
    const a = art.getBoundingClientRect();
    // drawn to 80% down the screen; at the bottom of the page, all the way (the last pin never gets that high)
    const atEnd = innerHeight + scrollY >= document.documentElement.scrollHeight - 4;
    const seen = reduced || atEnd ? a.height : Math.max(0, Math.min(a.height, innerHeight * 0.8 - a.top));
    let far = 0;
    // walk the path to the depth read so far
    let lo = 0, hi = len;
    for (let k = 0; k < 24; ++k) { const m = (lo + hi) / 2; if (path.getPointAtLength(m).y < seen) lo = m; else hi = m; }
    far = lo;
    path.style.strokeDashoffset = len - far;
    dots.forEach((c) => { c.style.opacity = +c.getAttribute("cy") <= seen ? 1 : 0.15; });
  }
  // pointing at an entry sends a dot down the walk from the pin before it to its own
  function walkTo(i) {
    if (!svg || !walker || i < 1 || reduced) return;
    cancelAnimationFrame(walkAnim);
    const a = pinLen[i - 1], b = pinLen[i], t0 = performance.now(), ms = Math.min(1400, 350 + (b - a) * 1.1);
    const step = (now) => {
      const q = Math.min(1, (now - t0) / ms), e = q < 0.5 ? 2 * q * q : 1 - Math.pow(-2 * q + 2, 2) / 2;
      const pt = path.getPointAtLength(a + (b - a) * e);
      walker.setAttribute("cx", pt.x); walker.setAttribute("cy", pt.y);
      walker.setAttribute("opacity", q < 0.9 ? 0.75 : 0.75 * (1 - q) * 10);
      if (q < 1) walkAnim = requestAnimationFrame(step);
    };
    walkAnim = requestAnimationFrame(step);
  }
  stops.forEach((s, i) => { const p = s.closest("p") || s; p.addEventListener("mouseenter", () => walkTo(i)); });
  let raf = 0;
  window.addEventListener("scroll", () => { cancelAnimationFrame(raf); raf = requestAnimationFrame(draw); }, { passive: true });
  let rz = 0; window.addEventListener("resize", () => { clearTimeout(rz); rz = setTimeout(build, 200); });
  if (document.readyState === "complete") build(); else window.addEventListener("load", build);
})();
