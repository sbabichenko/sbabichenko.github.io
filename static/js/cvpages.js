// The CV's two downloads drawn as the documents themselves: each sheet is the PDF's own page in miniature, one
// pencil line per line of text where it really sits (from the PDFs' text layout, LAYOUT below: x0, y, x1 in PDF
// points, and t a title, h a section heading, n a line). The full CV is its two pages, the resume its one, standing
// in a small printer. The first time the pointer comes by, the printer finishes printing them; the click that
// downloads folds the top page into a paper airplane that flies off, and the printer prints another copy.
(function () {
  "use strict";
  const LAYOUT = {"cv":[[[43,43,177,"t"],[43,63,401,"n"],[43,87,103,"h"],[43,113,513,"n"],[43,136,316,"n"],[43,158,568,"n"],[43,181,439,"n"],[43,206,367,"n"],[43,229,430,"n"],[43,257,95,"h"],[43,283,430,"n"],[43,306,342,"n"],[43,328,542,"n"],[43,351,547,"n"],[43,373,551,"n"],[43,396,546,"n"],[43,419,565,"n"],[43,442,228,"n"],[43,467,463,"n"],[43,490,504,"n"],[43,512,566,"n"],[43,535,435,"n"],[43,560,531,"n"],[43,583,568,"n"],[43,608,499,"n"],[43,631,554,"n"],[43,653,542,"n"],[43,676,123,"n"]],[[43,41,108,"h"],[43,67,541,"n"],[43,93,528,"n"],[43,115,131,"n"],[63,141,452,"n"],[63,166,242,"n"],[43,195,140,"h"],[43,221,313,"n"],[43,244,318,"n"],[43,266,539,"n"],[43,289,538,"n"],[43,311,509,"n"],[43,337,387,"n"],[43,359,543,"n"],[43,382,531,"n"],[43,405,549,"n"],[43,427,93,"n"],[43,456,75,"h"],[43,481,326,"n"],[43,507,551,"n"],[43,533,526,"n"],[43,556,374,"n"]]],"resume":[[[40,44,231,"t"],[40,65,193,"n"],[40,78,365,"n"],[40,95,557,"n"],[40,107,539,"n"],[40,119,447,"n"],[40,139,76,"n"],[40,156,79,"h"],[102,156,560,"n"],[102,168,269,"n"],[40,181,572,"n"],[40,194,388,"n"],[40,214,106,"n"],[40,233,351,"n"],[490,233,573,"n"],[40,249,430,"n"],[504,249,572,"n"],[52,263,568,"n"],[52,275,541,"n"],[52,287,100,"n"],[52,300,537,"n"],[52,312,566,"n"],[52,324,106,"n"],[52,338,568,"n"],[52,350,369,"n"],[40,366,419,"n"],[479,366,572,"n"],[52,379,572,"n"],[52,391,572,"n"],[52,403,381,"n"],[52,416,559,"n"],[52,428,561,"n"],[52,440,328,"n"],[52,454,560,"n"],[52,466,466,"n"],[40,487,95,"n"],[40,505,332,"n"],[52,518,552,"n"],[52,530,263,"n"],[52,544,562,"n"],[52,556,180,"n"],[40,572,397,"n"],[52,586,548,"n"],[52,598,555,"n"],[52,610,76,"n"],[40,626,358,"n"],[52,639,543,"n"],[52,651,560,"n"],[52,663,546,"n"],[40,683,106,"n"],[40,702,336,"n"],[549,702,572,"n"],[40,719,481,"n"],[549,719,572,"n"],[40,734,548,"n"],[40,746,302,"n"]]]};
  const NS = "http://www.w3.org/2000/svg", PW = 612, PH = 792, W = 30, H = W * PH / PW;   // a page on screen, px
  const SLOT = H + 2;                                                                    // the printer's mouth
  const el = (tag, attrs, parent) => { const n = document.createElementNS(NS, tag); for (const k in attrs) n.setAttribute(k, attrs[k]); if (parent) parent.appendChild(n); return n; };
  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
  let seed = 7;
  const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
  const wob = (a) => (rnd() - 0.5) * a;

  // one page: a hand-drawn edge with a folded corner, and its lines of text in pencil
  function page(lines, parent) {
    const g = el("g", { class: "cvp-page" }, parent), f = 6.5;
    el("path", { class: "cvp-sheet", d: `M${wob(.5)} ${wob(.5)} Q${W / 2} ${wob(.6)} ${W - f} ${wob(.4)} L${W + wob(.3)} ${f} Q${W + wob(.7)} ${H / 2} ${W + wob(.4)} ${H + wob(.4)} Q${W / 2} ${H + wob(.7)} ${wob(.4)} ${H + wob(.3)} Q${wob(.7)} ${H / 2} ${wob(.5)} ${wob(.5)}Z` }, g);
    el("path", { class: "cvp-fold", d: `M${W - f} 0 Q${W - f + wob(.4)} ${f / 2} ${W - f} ${f} Q${W - f / 2} ${f + wob(.4)} ${W} ${f}` }, g);
    const s = W / PW;
    for (const [x0, y, x1, kind] of lines) {
      const X0 = x0 * s, X1 = Math.min(x1 * s, y * s < f + 1 ? W - f - 1 : W - 1.6), Y = y * s + wob(.25);
      el("path", { class: "cvp-line cvp-" + kind, d: `M${X0} ${Y} Q${(X0 + X1) / 2} ${Y + wob(.5)} ${X1} ${Y + wob(.3)}` }, g);
    }
    return g;
  }

  // the printer under the pile: a low box with a slot, a paper rest behind it and a light
  function printer(parent) {
    const g = el("g", { class: "cvp-printer" }, parent), x0 = -5, x1 = W + 8, y0 = SLOT - 1.5, y1 = SLOT + 9;
    el("path", { class: "cvp-sheet", d: `M${x0} ${y0 + 2} Q${x0} ${y0} ${x0 + 2.5} ${y0} L${x1 - 2.5} ${y0 + wob(.3)} Q${x1} ${y0} ${x1} ${y0 + 2.5} L${x1 + wob(.3)} ${y1} Q${(x0 + x1) / 2} ${y1 + wob(.5)} ${x0} ${y1}Z` }, g);
    el("path", { class: "cvp-fold", d: `M${-1} ${SLOT + 1.2} Q${W / 2} ${SLOT + 1.5} ${W + 4} ${SLOT + 1.1}` }, g);
    el("circle", { class: "cvp-light", cx: x1 - 3.5, cy: y1 - 3, r: 0.9 }, g);
    return g;
  }

  // The throw. One sheet does it all: an HTML box laid exactly over the top page (so it can follow a CSS motion path)
  // holds a copy of that page, and its outline is folded into an airplane in two moves: the top corners in to the
  // middle, then the sides in and the wings down. Each outline has the same six points, so one shape eases into the
  // next (SMIL, so every browser animates it). The plane is no wider than the folded sheet, so nothing seems to
  // unfold; only once it is folded does it turn its nose onto the path, already moving, and fly.
  const h = H / 2, w = W / 2, pts = (p) => "M" + p.map((q) => q.join(" ")).join(" L") + "Z";
  const OUTLINE = [
    [[-w, -h], [0, -h], [w, -h], [w, h], [0, h], [-w, h]],                                                    // the sheet
    [[-w, -h * 0.25], [0, -h], [w, -h * 0.25], [w, h], [0, h], [-w, h]],                                      // top corners to the middle
    [[-w * 0.6, h * 0.75], [0, -h], [w * 0.6, h * 0.75], [w * 0.12, h * 0.55], [0, h * 0.8], [-w * 0.12, h * 0.55]],   // sides in, wings down
  ];
  const CREASE = [   // the lines the folds leave: the flaps' lower edges, then the wing roots
    [[0, -h], [0, -h], [0, -h], [0, -h]],
    [[-w, -h * 0.25], [0, -h * 0.25], [w, -h * 0.25], [0, -h * 0.25]],
    [[-w * 0.12, h * 0.55], [0, -h], [w * 0.12, h * 0.55], [0, -h]],
  ].map((p) => `M${p[0].join(" ")} L${p[1].join(" ")} M${p[2].join(" ")} L${p[3].join(" ")}`);
  const FOLD = { begin: 0.1, dur: 0.45 }, KEYS = "0;.45;1", SPLINES = ".45 0 .55 1;.45 0 .55 1";

  function flyer(a, top) {
    const box = document.createElement("span"); box.className = "cvp-flyer"; box.setAttribute("aria-hidden", "true");
    const s = el("svg", { width: W + 10, height: H + 10, viewBox: `${-w - 5} ${-h - 5} ${W + 10} ${H + 10}` });
    const turn = el("g", { class: "cvp-turn" }, s);
    const sheet = el("path", { class: "cvp-sheet", d: pts(OUTLINE[0]) }, turn);
    el("animate", { attributeName: "d", values: OUTLINE.map(pts).join(";"), keyTimes: KEYS, calcMode: "spline", keySplines: SPLINES, dur: FOLD.dur + "s", begin: "indefinite", fill: "freeze" }, sheet);
    const crease = el("path", { class: "cvp-fold", d: CREASE[0] }, turn);
    el("animate", { attributeName: "d", values: CREASE.join(";"), keyTimes: KEYS, calcMode: "spline", keySplines: SPLINES, dur: FOLD.dur + "s", begin: "indefinite", fill: "freeze" }, crease);
    // the page's own lines, which vanish into the first fold
    const words = el("g", { class: "cvp-words", transform: `translate(${-w} ${-h})` }, turn);
    for (const n of top.querySelectorAll(".cvp-line, .cvp-fold")) words.appendChild(n.cloneNode());
    box.appendChild(s); a.appendChild(box);
    const t = el("svg", { class: "cvp-trail", width: 1, height: 1, "aria-hidden": "true" });
    const id = "cvp-trailmask-" + Math.random().toString(36).slice(2, 8);
    const mk = el("mask", { id, maskUnits: "userSpaceOnUse", x: -50, y: -250, width: 650, height: 450 }, el("defs", {}, t));
    const reveal = el("path", { class: "cvp-reveal", pathLength: 100 }, mk), dash = el("path", { class: "cvp-dash", mask: `url(#${id})` }, t);
    a.appendChild(t);
    // at the click: start the flight where the top page lies now, upright, and set the folds going
    return function launch() {
      const r = a.getBoundingClientRect(), q = top.querySelector(".cvp-sheet").getBoundingClientRect();
      const x = q.left + q.width / 2 - r.left, y = q.top + q.height / 2 - r.top;
      const path = `M${x} ${y} C ${x + 40} ${y - 40}, ${x + 150} ${y - 62}, ${x + 330} ${y - 80}`;   // one climbing glide, below the title
      box.style.offsetPath = `path("${path}")`; reveal.setAttribute("d", path); dash.setAttribute("d", path);
      // the box turns with the path from the start (offset-rotate: auto); the sheet inside is turned back to upright,
      // and turns its nose (the top of the sheet) onto the path as the last fold closes
      box.style.setProperty("--a0", (-Math.atan2(-40, 40) * 180 / Math.PI) + "deg");   // the path's first heading
      for (const m of s.querySelectorAll("animate")) m.beginElementAt(FOLD.begin);
    };
  }

  function replay(a, cls, ms) { a.classList.remove(cls); void a.offsetWidth; a.classList.add(cls); clearTimeout(a["_" + cls]); a["_" + cls] = setTimeout(() => a.classList.remove(cls), ms); }

  for (const a of document.querySelectorAll(".cv-download")) {
    const pages = LAYOUT[/Resume_/.test(a.getAttribute("href")) ? "resume" : "cv"], n = pages.length;
    const svg = el("svg", { class: "cvp", width: W + 22, height: SLOT + 22, viewBox: `-8 -9 ${W + 22} ${SLOT + 22}`, "aria-hidden": "true" });
    printer(svg);
    const clipId = "cvp-slot-" + Math.random().toString(36).slice(2, 8);
    el("rect", { x: -20, y: -40, width: W + 60, height: SLOT + 40 }, el("clipPath", { id: clipId }, el("defs", {}, svg)));
    const out = el("g", { class: "cvp-out", "clip-path": `url(#${clipId})` }, svg);
    // the last page goes in first, so page 1 lies on top
    for (let i = n - 1; i >= 0; --i) {
      const g = page(pages[i], out); g.style.setProperty("--i", i); g.style.setProperty("--n", n); g.style.setProperty("--H", (H + 6) + "px");
      if (i === 0) g.classList.add("cvp-top");
    }
    const launch = flyer(a, out.querySelector(".cvp-top"));
    a.prepend(svg);
    a.classList.add("cvp-on");
    a.style.setProperty("--n", n);
    a.style.setProperty("--rest", (H * 0.55) + "px");
    // the print runs once, on the first visit; a throw asked for while it runs waits for it to finish
    const PRINT = n === 1 ? 400 : 420 + (n - 1) * 450 + 50, FLY = 1900;   // ms, as in custom.css
    let printed = false, queued = false;
    const done = () => a.classList.add("printed");
    const print = () => {
      if (printed) return;
      printed = true;
      if (reduced) return done();
      replay(a, "printing", PRINT); setTimeout(() => { done(); if (queued) { queued = false; fly(); } }, PRINT);
    };
    const fly = () => { if (a.classList.contains("flying")) return; launch(); replay(a, "flying", FLY); };
    a.addEventListener("pointerenter", print); a.addEventListener("focus", print);
    a.addEventListener("click", () => {
      if (reduced) return;
      if (a.classList.contains("printing")) queued = true; else fly();
    });
  }
})();
