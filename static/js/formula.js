// The noise-state definition in the hero, written on: each glyph's outline is traced in pencil, left to right,
// then filled in as ink. The SVG is MathJax's, made at build time (tools/make_formula.js).
(function () {
  "use strict";
  const motif = document.querySelector(".hero .motif");
  if (!motif) return;
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (reduced) return;
  const lines = [...motif.querySelectorAll(".def svg, .bp svg")];
  let delay = 900;                                             // after the name has settled
  lines.forEach((svg, li) => {
    const paths = [...svg.querySelectorAll("path")];
    // order the glyphs left to right by where they sit
    const xs = paths.map((p) => { try { const b = p.getBBox(); const m = p.getCTM(); return m ? m.e + b.x * m.a : b.x; } catch (e) { return 0; } });
    const order = paths.map((p, i) => i).sort((a, b) => xs[a] - xs[b]);
    const span = li === 0 ? 1900 : 1700, step = span / Math.max(1, paths.length);
    order.forEach((i, k) => {
      const p = paths[i];
      let len = 0;
      try { len = p.getTotalLength(); } catch (e) { len = 0; }
      if (!len) return;
      p.style.fillOpacity = 0;
      p.style.stroke = "currentColor";
      p.style.strokeWidth = li === 0 ? "14px" : "16px";
      p.style.strokeLinecap = "round";
      p.style.strokeDasharray = `${len} ${len}`;
      p.style.strokeDashoffset = len;
      const t0 = delay + k * step, dur = Math.min(900, 250 + len / 6);
      p.animate([{ strokeDashoffset: len }, { strokeDashoffset: 0 }], { duration: dur, delay: t0, easing: "cubic-bezier(.4,.1,.3,1)", fill: "forwards" });
      p.animate([{ fillOpacity: 0 }, { fillOpacity: 1 }], { duration: 700, delay: t0 + dur * 0.7, fill: "forwards" });
      p.animate([{ strokeOpacity: 1 }, { strokeOpacity: 0 }], { duration: 900, delay: t0 + dur, fill: "forwards" });
    });
    delay += span + 250;
  });
})();
