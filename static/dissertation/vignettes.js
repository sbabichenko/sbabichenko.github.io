// A pencil vignette at the head of each chapter page, drawn on as the page opens: the chapter's idea in one small
// picture. The drawings are plain SVG built here; which one is picked from the chapter number on the page's heading.
(function () {
  "use strict";
  const body = document.querySelector(".body");
  const h1 = body && body.querySelector("h1");
  if (!h1) return;
  const num = h1.getAttribute("data-num") || (/^introduction$/i.test(h1.textContent.trim()) ? "intro" : "");
  const NS = "http://www.w3.org/2000/svg";
  const el = (tag, attrs, parent) => { const n = document.createElementNS(NS, tag); for (const [k, v] of Object.entries(attrs || {})) n.setAttribute(k, v); if (parent) parent.appendChild(n); return n; };
  function mulberry32(a) { return function () { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
  function gauss(r) { let u = 0; while (u === 0) u = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r()); }
  function pencil(pts, seed, wob = 0.8) {
    const r = mulberry32(seed);
    let d = `M${pts[0][0].toFixed(1)},${pts[0][1].toFixed(1)}`;
    for (let i = 1; i < pts.length; ++i) { const [x0, y0] = pts[i - 1], [x1, y1] = pts[i]; d += ` Q${((x0 + x1) / 2 + (r() - 0.5) * wob * 2).toFixed(1)},${((y0 + y1) / 2 + (r() - 0.5) * wob * 2).toFixed(1)} ${x1.toFixed(1)},${y1.toFixed(1)}`; }
    return d;
  }
  const svg = el("svg", { viewBox: "0 0 600 150", class: "vignette", "aria-hidden": "true" });
  let order = 0;
  const line = (pts, cls, w = 1.6, seed) => { const p = el("path", { d: pencil(pts, seed || ++order * 7, 0.8), class: "v-line " + (cls || ""), "stroke-width": w }, svg); p.dataset.k = order++; return p; };
  const dot = (x, y, r, cls) => { const c = el("circle", { cx: x, cy: y, r, class: "v-dot " + (cls || "") }, svg); c.dataset.k = order++; return c; };
  const label = (x, y, s, cls, anchor = "middle") => { const t = el("text", { x, y, class: "v-text " + (cls || ""), "text-anchor": anchor }, svg); t.textContent = s; t.dataset.k = order++; return t; };
  const arrow = (x0, y0, x1, y1, cls, w) => { line([[x0, y0], [x1, y1]], cls, w); const a = Math.atan2(y1 - y0, x1 - x0); line([[x1 - 9 * Math.cos(a - 0.45), y1 - 9 * Math.sin(a - 0.45)], [x1, y1], [x1 - 9 * Math.cos(a + 0.45), y1 - 9 * Math.sin(a + 0.45)]], cls, w); };
  const r = mulberry32(7);

  const draw = {
    // the loop, and its chord
    intro() {
      const P = [[90, 75, "actions"], [300, 22, "state"], [510, 75, "observations"], [300, 128, "beliefs"]];
      P.forEach(([x, y, s]) => label(x, y + 5, s, ""));
      arrow(130, 58, 262, 28); arrow(338, 28, 452, 58); arrow(452, 94, 342, 124); arrow(258, 124, 132, 94);
      line([[150, 75], [440, 75]], "acc dash", 1.4);
      label(300, 68, "direct channel", "small acc");
    },
    // two players pull one state toward opposite targets, each through fog
    "1"() {
      line([[40, 30], [560, 30]], "soft dash", 1); line([[40, 120], [560, 120]], "soft dash", 1);
      label(566, 34, "b₁", "acc", "start"); label(566, 124, "b₂", "warm", "start");
      const pts = []; let y = 75; for (let x = 40; x <= 560; x += 10) { pts.push([x, y]); y += gauss(r) * 4 + 0.15 * (75 - y); y = Math.max(45, Math.min(105, y)); }
      line(pts, "", 2);
      arrow(200, 70, 200, 40, "acc", 1.8); arrow(380, 80, 380, 110, "warm", 1.8);
      label(300, 146, "two players push one state, each seeing it through their own noise", "small");
    },
    // a trade moves beliefs until the news arrives and explains it away
    "2"() {
      el("rect", { x: 170, y: 20, width: 230, height: 100, class: "v-fill acc" }, svg);
      line([[40, 110], [560, 110]], "", 1.4);
      const b = []; for (let x = 40; x <= 560; x += 8) { const u = x < 170 ? 0 : x < 400 ? 1 - Math.exp(-(x - 170) / 60) : Math.exp(-(x - 400) / 25); b.push([x, 100 - 60 * u]); }
      line(b, "acc", 2.2);
      arrow(170, 140, 170, 114, "warm", 1.6); label(170, 148, "trade", "small warm");
      arrow(400, 140, 400, 114, "", 1.6); label(400, 148, "news arrives", "small");
      label(285, 16, "disclosure window", "small acc");
    },
    // the causal triangle of dates becomes a quadrant of ages
    "3"() {
      line([[60, 130], [200, 130], [200, 20], [60, 130]], "", 1.8);
      // lines of constant age t - s, parallel to the diagonal
      for (let k = 1; k < 8; ++k) { const x0 = 60 + k * 17.5; line([[x0, 130], [200, 130 - ((200 - x0) * 110) / 140]], "soft", 0.8); }
      label(130, 146, "dates (t, s), s ≤ t", "small");
      arrow(240, 75, 330, 75, "", 1.6);
      line([[380, 20], [380, 130], [560, 130]], "", 1.8);
      for (let k = 1; k < 7; ++k) line([[380, 130 - k * 16], [560, 130 - k * 16]], "soft", 0.6);
      label(470, 146, "ages: the calendar drops out", "small");
    },
    // an informed trader's orders hide in the flow, and the price creeps toward value
    "4"() {
      line([[40, 30], [560, 30]], "warm dash", 1.2); label(566, 34, "V", "warm", "start");
      for (let k = 0; k < 40; ++k) { const x = 50 + k * 13, h = gauss(r) * 6 + (k % 3 === 0 ? 5 : 0); el("rect", { x: x - 3, y: h > 0 ? 118 - h : 118, width: 6, height: Math.abs(h), class: "v-bar " + (k % 3 === 0 ? "acc" : "") }, svg).dataset.k = order++; }
      const p = []; for (let k = 0; k <= 52; ++k) { const x = 40 + k * 10; p.push([x, 90 - 55 * (1 - Math.exp(-k / 20)) + gauss(r) * 2]); }
      line(p, "", 2.2);
      label(300, 148, "blue: the informed orders, hidden in the flow", "small acc");
    },
    // a ring of local markets, each firm reading demand off its customer
    "5"() {
      const n = 10, cx = 300, cy = 72, R = 58;
      const P = Array.from({ length: n }, (_, k) => [cx + R * 2.2 * Math.cos((2 * Math.PI * k) / n - Math.PI / 2), cy + R * Math.sin((2 * Math.PI * k) / n - Math.PI / 2)]);
      P.forEach(([x, y], k) => { const [x2, y2] = P[(k + 1) % n]; const f = 0.22; arrow(x + (x2 - x) * f, y + (y2 - y) * f, x + (x2 - x) * (1 - f), y + (y2 - y) * (1 - f), "soft", 1.2); });
      P.forEach(([x, y], k) => dot(x, y, 8, k === 0 ? "acc" : ""));
      label(300, 148, "each firm sells to the next; symmetry makes one of them enough", "small");
    },
    // one deviation, two observers: one sees a shock, the other sees who did it
    "6"() {
      arrow(250, 75, 350, 75, "warm", 2.2); label(300, 62, "a deviation", "small warm");
      const eye = (x, lab, cls) => { line([[x - 30, 75], [x, 55], [x + 30, 75], [x, 95], [x - 30, 75]], cls, 1.6); dot(x, 75, 6, cls); label(x, 124, lab, "small " + cls); };
      eye(110, "naive: a shock", ""); eye(490, "privy: player 1 did it", "acc");
      line([[150, 75], [240, 75]], "soft dash", 1); line([[360, 75], [450, 75]], "soft dash", 1);
    },
    // the wedge: a belief moved, and the gap priced
    "7"() {
      const bell = (m) => { const p = []; for (let x = 120; x <= 480; x += 8) p.push([x, 125 - 95 * Math.exp(-((x - m) ** 2) / (2 * 45 ** 2))]); return p; };
      line([[100, 125], [500, 125]], "soft", 1);
      line(bell(270), "soft dash", 1.4); line(bell(330), "acc", 2.2);
      arrow(275, 20, 325, 20, "warm", 1.8);
      label(300, 146, "the wedge: what moving someone's belief is worth", "small");
    },
  };
  if (!draw[num]) return;
  draw[num]();
  const wrap = document.createElement("div");
  wrap.className = "vignette-wrap";
  wrap.appendChild(svg);
  h1.parentNode.insertBefore(wrap, h1);

  // draw on: every line from nothing, in the order it was made
  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  const items = [...svg.querySelectorAll("[data-k]")].sort((a, b) => a.dataset.k - b.dataset.k);
  const step = 1400 / Math.max(1, items.length);
  items.forEach((n, i) => {
    const delay = 200 + i * step;
    if (n.tagName === "path") {
      const len = n.getTotalLength();
      if (n.classList.contains("dash")) { n.style.opacity = 0; n.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 500, delay, fill: "forwards" }); return; }
      n.style.strokeDasharray = `${len} ${len}`; n.style.strokeDashoffset = len;
      n.animate([{ strokeDashoffset: len }, { strokeDashoffset: 0 }], { duration: Math.min(1200, 300 + len * 1.5), delay, easing: "cubic-bezier(.4,.1,.3,1)", fill: "forwards" });
    } else {
      n.style.opacity = 0;
      n.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 500, delay, fill: "forwards" });
    }
  });
})();
