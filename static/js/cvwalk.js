// The CV's margin: a random walk pinned at each entry (a Brownian bridge from one to the next), drawn in pencil
// as far down the page as you have read. Only where there is a margin to draw in.
(function () {
  "use strict";
  const art = document.querySelector("article");
  if (!art) return;
  const stops = [...art.querySelectorAll("p > strong:first-child")].filter((s) => !/:\s*$/.test(s.textContent));
  if (stops.length < 2) return;
  const NS = "http://www.w3.org/2000/svg";
  const el = (tag, attrs, parent) => { const n = document.createElementNS(NS, tag); for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v); if (parent) parent.appendChild(n); return n; };
  function gauss() { let u = 0; while (u === 0) u = Math.random(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * Math.random()); }
  let svg = null, path = null, dots = [], len = 0;
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function build() {
    if (svg) svg.remove();
    const a = art.getBoundingClientRect(), W = 70;
    if (a.left < W + 30) { svg = null; return; }
    art.style.position = "relative";
    svg = el("svg", { class: "cvwalk", width: W, height: a.height, viewBox: `0 0 ${W} ${a.height}`, "aria-hidden": "true" }, art);
    Object.assign(svg.style, { position: "absolute", left: -W - 24 + "px", top: 0, overflow: "visible", pointerEvents: "none" });
    const ys = stops.map((s) => s.getBoundingClientRect().top - a.top + 10), x0 = W / 2;
    let d = `M${x0},${ys[0]}`;
    for (let i = 1; i < ys.length; ++i) {
      // a Brownian bridge from x0 at ys[i-1] back to x0 at ys[i]
      const n = Math.max(8, Math.round((ys[i] - ys[i - 1]) / 6)), w = [0];
      for (let k = 1; k <= n; ++k) w.push(w[k - 1] + gauss());
      for (let k = 1; k <= n; ++k) {
        const b = w[k] - (k / n) * w[n], x = x0 + Math.max(-W / 2 + 4, Math.min(W / 2 - 4, b * 3.2));
        d += ` L${x.toFixed(1)},${(ys[i - 1] + ((ys[i] - ys[i - 1]) * k) / n).toFixed(1)}`;
      }
    }
    path = el("path", { d, fill: "none", stroke: "currentColor", "stroke-width": 1.2, "stroke-linejoin": "round", opacity: 0.45 }, svg);
    len = path.getTotalLength();
    path.style.strokeDasharray = `${len} ${len}`;
    dots = ys.map((y) => el("circle", { cx: x0, cy: y, r: 3.5, fill: "var(--accent, #1f3fd0)" }, svg));
    const t = el("text", { x: x0, y: ys[0] - 14, "text-anchor": "middle", "font-size": 9, fill: "currentColor", opacity: 0.5, "font-family": "ui-monospace, Menlo, monospace" }, svg);
    t.textContent = "a random walk,"; const t2 = el("text", { x: x0, y: ys[0] - 3, "text-anchor": "middle", "font-size": 9, fill: "currentColor", opacity: 0.5, "font-family": "ui-monospace, Menlo, monospace" }, svg);
    t2.textContent = "pinned at each stop";
    draw();
  }
  function draw() {
    if (!svg) return;
    const a = art.getBoundingClientRect(), seen = reduced ? a.height : Math.max(0, Math.min(a.height, innerHeight * 0.8 - a.top));
    let far = 0;
    // walk the path to the depth read so far
    let lo = 0, hi = len;
    for (let k = 0; k < 24; ++k) { const m = (lo + hi) / 2; if (path.getPointAtLength(m).y < seen) lo = m; else hi = m; }
    far = lo;
    path.style.strokeDashoffset = len - far;
    dots.forEach((c) => { c.style.opacity = +c.getAttribute("cy") <= seen ? 1 : 0.15; });
  }
  let raf = 0;
  window.addEventListener("scroll", () => { cancelAnimationFrame(raf); raf = requestAnimationFrame(draw); }, { passive: true });
  let rz = 0; window.addEventListener("resize", () => { clearTimeout(rz); rz = setTimeout(build, 200); });
  if (document.readyState === "complete") build(); else window.addEventListener("load", build);
})();
