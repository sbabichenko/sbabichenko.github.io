// The detour map, under the dissertation's title: ninety years of economics going around one problem.
// A band runs across the map from 1936, where Keynes described it: people who learn from each other's actions.
// Each line of work comes toward the band in its year, bends, and runs alongside it from then on, so over the
// scroll the band gets outlined by the ways around it. In 2026 one line goes straight through, and each old line's
// direct route into the band appears. Time runs left to right on a wide screen, top to bottom on a phone.
// Any <svg data-detour="still"> elsewhere (the home page) gets the finished map, small and without labels.
(function () {
  "use strict";
  const NS = "http://www.w3.org/2000/svg";
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
  const clamp = (x, a = 0, b = 1) => Math.max(a, Math.min(b, x));
  const ease = (t) => (t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2);
  const seg = (t, a, b) => ease(clamp((t - a) / (b - a)));

  // a pencil line through points, each segment bowed a little so it reads as drawn
  function pencil(pts, seed, wob) {
    const r = mulberry32(seed);
    let d = `M${pts[0][0].toFixed(1)},${pts[0][1].toFixed(1)}`;
    for (let i = 1; i < pts.length; ++i) {
      const [x0, y0] = pts[i - 1], [x1, y1] = pts[i];
      d += ` Q${((x0 + x1) / 2 + (r() - 0.5) * wob).toFixed(1)},${((y0 + y1) / 2 + (r() - 0.5) * wob).toFixed(1)} ${x1.toFixed(1)},${y1.toFixed(1)}`;
    }
    return d;
  }
  function stroke(g, d, cls, w) {
    const a = el("path", { d, class: "pn " + (cls || ""), "stroke-width": w }, g);
    const b = el("path", { d, class: "pn soft " + (cls || ""), "stroke-width": w * 0.6, transform: "translate(0.7,0.5)" }, g);
    const len = a.getTotalLength ? a.getTotalLength() : 1000;
    for (const p of [a, b]) { p.style.strokeDasharray = `${len} ${len}`; p.style.strokeDashoffset = len; }
    return { a, b, set(t) { const o = len * (1 - clamp(t)); a.style.strokeDashoffset = o; b.style.strokeDashoffset = o; } };
  }
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
    { y: 1962, who: "Radner", what: "one shared goal", side: 0, row: 1, an: "middle" },
    { y: 1972, who: "Lucas", what: "price takers", side: 0, row: 0, an: "middle" },
    { y: 1980, who: "Grossman–Stiglitz", what: "competitive traders", side: 1, row: 0, an: "end" },
    { y: 1983, who: "Townsend", what: "a lag reveals all", side: 0, row: 1, an: "middle" },
    { y: 1985, who: "Kyle", what: "one insider", side: 1, row: 2, an: "middle", kyle: true },
    { y: 1996, who: "Foster–Viswanathan", what: "identical insiders", side: 0, row: 2, an: "middle" },
    { y: 2000, who: "Kasa", what: "negligible agents", side: 1, row: 1, an: "end" },
    { y: 2002, who: "Morris–Shin", what: "one round", side: 0, row: 1, an: "start" },
    { y: 2007, who: "Lasry–Lions", what: "continuum of players", side: 1, row: 2, an: "middle" },
    { y: 2011, who: "Kamenica–Gentzkow", what: "static information", side: 0, row: 0, an: "start" },
    { y: 2013, who: "Nayyar et al.", what: "a shared record", side: 1, row: 0, an: "start" },
  ];
  const BAND = [0.44, 0.56];

  function draw(svg, opts) {
    const still = !!opts.still, vertical = !!opts.vertical;
    svg.innerHTML = "";
    const W = vertical ? 500 : 1000, H = vertical ? 800 : 560;
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    // u: time, 0 at 1936 and 1 at 2026; v: across the map, the band in the middle
    const mu = vertical ? [70, 40] : [70, 36], vEdge = still ? 0.06 : vertical ? 0.33 : 0.215;
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
      el("path", { d: pencil([a, b], Math.floor(r() * 1e6), 1.5), class: "pn", "stroke-width": still ? 1.6 : 0.7 }, hatch);
    }
    const edgeA = [], edgeB = [];
    for (let i = 0; i <= 60; ++i) { const u = i / 60; edgeA.push(P(u, BAND[0] + (r() - 0.5) * 0.004)); edgeB.push(P(u, BAND[1] + (r() - 0.5) * 0.004)); }
    const ew = still ? 6 : 1.3, eA = stroke(band, pencil(edgeA, 3, 1), "", ew), eB = stroke(band, pencil(edgeB, 4, 1), "", ew);
    parts.push({ set: (t) => { eA.set(t); eB.set(t); hatch.style.opacity = 0.9 * t; }, w: [0.0, 0.12] });

    let bandLabel = null, unmapped = null;
    if (!still) {
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
      parts.push({ set: (t) => { flag.style.opacity = t; bandLabel.style.opacity = t; }, w: [0.02, 0.1] });
      unmapped = el("text", { x: P(vertical ? 0.55 : 0.28, 0.5)[0], y: P(vertical ? 0.55 : 0.28, 0.5)[1] + 4, class: "mono halo", "text-anchor": "middle",
        transform: vertical ? `rotate(90 ${P(0.55, 0.5)[0]} ${P(0.55, 0.5)[1]})` : "" }, g);
      unmapped.textContent = "unmapped";
    }

    // the roads: down from the map's edge in their year, a bend, then alongside the band to the present
    const lane = [0, 0];
    const doors = [];
    ROADS.forEach((rd, i) => {
      const u0 = U(rd.y), s = rd.side, sgn = s === 0 ? -1 : 1;
      const k = lane[s]++, off = still ? 0.034 + 0.033 * k : vertical ? 0.016 + 0.015 * k : 0.022 + 0.021 * k;
      const vBand = s === 0 ? BAND[0] : BAND[1];
      const vRun = vBand + sgn * off, vStart = s === 0 ? vEdge : 1 - vEdge;
      const bend = 0.028;
      let cp;
      if (rd.kyle) {   // the closest approach: into the edge of the band and back out
        cp = [[u0, vStart], [u0, vRun + sgn * 0.05], [u0 + 0.004, vBand - sgn * 0.012], [u0 + 0.02, vBand - sgn * 0.032],
              [u0 + 0.038, vBand - sgn * 0.008], [u0 + 0.056, vRun], [u0 + 0.08, vRun], [0.97, vRun]];
      } else cp = [[u0, vStart], [u0, vRun + sgn * 0.05], [u0 + bend * 0.6, vRun + sgn * 0.004], [u0 + bend * 1.6, vRun], [0.97, vRun]];
      const pts = smooth(cp.map(([u, v]) => P(u, v)), 8);
      const line = stroke(g, pencil(pts, 20 + i, still ? 0.6 : 1.2), "road", still ? 7 : 1.4);
      const t0 = 0.1 + 0.44 * (i / ROADS.length), t1 = t0 + 0.09;
      parts.push({ set: line.set, w: [t0, t1] });
      doors.push({ from: P(u0 + bend * 1.6, vRun), to: P(u0 + bend * 1.6, 0.5), i });
      if (!still) {
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
        t.innerHTML = vertical ? `<tspan class="who">${rd.who}</tspan><tspan x="${tx}" dy="14">${rd.y} · ${rd.what}</tspan>`
          : `<tspan class="who">${rd.who} ${rd.y}</tspan><tspan x="${tx}" dy="14">${rd.what}</tspan>`;
        parts.push({ set: (q) => { lab.style.opacity = q; }, w: [t0, t0 + 0.04] });
      }
    });

    // 2026: straight through
    const uT = 0.985;
    const thru = stroke(g, pencil(smooth([P(uT, still ? 0.04 : vEdge - 0.02), P(uT, 0.5), P(uT, still ? 0.96 : 1 - vEdge + 0.02)], 12), 99, 0.8), "acc", still ? 11 : 2.4);
    parts.push({ set: thru.set, w: [0.66, 0.8] });
    if (!still) {
      const [x, y] = vertical ? P(uT, vEdge - 0.02) : P(uT, 1 - vEdge + 0.02);
      const tx = vertical ? x - 8 : x + 6;
      const t = el("text", { x: tx, y: vertical ? y - 4 : y + 56, class: "lab accl", "text-anchor": "end" }, g);
      t.innerHTML = `<tspan class="who">2026</tspan><tspan x="${tx}" dy="14">this dissertation</tspan>`;
      parts.push({ set: (q) => { t.style.opacity = q; }, w: [0.66, 0.7] });
    }
    // and then each line's own way in
    doors.forEach((d) => {
      const n = 7, dg = el("g", { class: "door" }, g), bits = [];
      for (let k = 0; k < n; ++k) {
        const a = k / n, b = (k + 0.5) / n, L = (q) => [d.from[0] + (d.to[0] - d.from[0]) * q, d.from[1] + (d.to[1] - d.from[1]) * q];
        const e = el("path", { d: pencil([L(a), L(b)], 60 + d.i * 9 + k, 0.4), class: "pn acc", "stroke-width": 1.2 }, dg);
        e.style.opacity = 0; bits.push(e);
      }
      const t0 = 0.82 + 0.012 * d.i;
      parts.push({ set: (t) => bits.forEach((e, k) => { e.style.opacity = t * n > k ? 0.85 : 0; }), w: [t0, t0 + 0.06] });
    });
    parts.push({ set: (t) => { hatch.style.opacity = 0.9 - 0.55 * t; if (unmapped) unmapped.style.opacity = 1 - t; }, w: [0.84, 0.98] });

    return function (p) { for (const q of parts) q.set(seg(p, q.w[0], q.w[1])); };
  }

  // the stills
  document.querySelectorAll("svg[data-detour=still]").forEach((s) => draw(s, { still: true })(1));

  // the scrolled map
  const sec = document.getElementById("detour");
  if (!sec) return;
  const svg = sec.querySelector("svg"), caps = [...sec.querySelectorAll(".dt-cap")];
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  let render = null, vertical = null;
  function lay() {
    const v = innerWidth < 760 && innerHeight > innerWidth;
    if (v === vertical && render) return;
    vertical = v; render = draw(svg, { vertical: v }); tick();
  }
  function tick() {
    const r = sec.getBoundingClientRect(), span = r.height - innerHeight;
    const p = clamp(-r.top / Math.max(1, span));
    render(reduced ? Math.max(p, 0.001) : p);
    let on = 0;
    caps.forEach((c, i) => { if (p >= +c.dataset.at) on = i; });
    caps.forEach((c, i) => c.classList.toggle("on", i === on));
  }
  sec.classList.add("js");
  lay();
  addEventListener("scroll", tick, { passive: true });
  let rt; addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(() => { render = null; lay(); }, 150); });
})();
