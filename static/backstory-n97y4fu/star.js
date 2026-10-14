// /backstory, "Analyzing fit": one vertex of the real fit, its place in the network and its evidence.
// The arithmetic is the mesh repository's own star inspector (analysis/starcore.js, loaded unchanged, on the data
// analysis/star_export.py packs: stars.json, rebuilt by backstory/tools/stars.sh). Only the drawing is this page's.
//
//   map       loan age across, coupon up; the mesh, its vertices (a solid diamond: a free coefficient; a small open
//             circle: one held at its parents' average), and the pools under the chosen vertex, sized by their share
//             of its information and coloured by their residual (blue below the fit, red above).
//   star      the blue wash: the coefficient's own tent, where changing its surplus moves the surface when every other
//             surplus is held, darker where it moves it more: full at the vertex, falling to nothing at the edge of the
//             vertex's neighbourhood at its own scale. (The inspector's star stops at other free vertices instead.)
//   network   blue arrows run to the vertex from its two parents, the ends A and A' of the edge it bisected, and to them
//             from theirs, fading generation by generation up to the corners: whom it borrows from. Dashed blue goes
//             to the vertices that borrow from it. Pointing at another vertex previews its ancestry; a click chooses it.
//   two views the star, framed to hold the vertex's reach and its two parents; and, above the lineage, the whole map
//             with the same ancestry on it and the star's frame marked. Both take pointing and clicking. Nothing zooms.
//   scales    one vertex can be seen at more than one scale. That is a different move from going to another vertex, and
//             looks different: the chips beside the data set's name (keys w and s), and in the lineage a rail of
//             squares, one for each of its scales; other vertices are circles and are reached by clicking them.
//   lineage   the same hierarchy as a diagram (see lineage() below): finer down the page, the vertex's ancestors above it
//             and the vertices that borrow from it below; a click on any of them goes there.
//   numbers   its address and the score that admitted it against what it needed.
//   borrowing a bar split into the share of its own pools' say that the vertex keeps and the share it takes from its
//             parents (see shrink() below); and on the sections, the tent its pools alone ask for (dashed, with an open
//             circle and their uncertainty at its peak) over the tent it keeps (solid), with the pull between the peaks.
//   sections  along its parent edge (A to A') and across it (B to B'): zero is what its parents alone would give; the
//             blue line is the coefficient's own contribution. For each bin of pools: a short dark bar at what the fit
//             expects, a box and a whisker for the noise around it (binomial noise and the pool effect: two draws in
//             three, nineteen in twenty), and a dot for what was seen, larger for more loans, ringed red if outside.
window.Star = (function () {
  "use strict";
  const root = document.getElementById("st");
  if (!root) return { open() {}, shown() {} };
  const $ = (id) => document.getElementById(id), cv = $("st-cv"), f = (v) => v.toFixed(1);
  const SC = () => window.StarCore;
  const S = { runs: null, run: null, vid: null, hov: null, card: null, sf: null, nodal: null, order: null, secA: null, secB: null, req: 0, loading: null, want: null, verts: null };
  const isDark = () => document.documentElement.classList.contains("dark");
  const css = (name) => getComputedStyle(root).getPropertyValue(name).trim();

  // ------------------------------------------------------------------ data
  function load() {
    if (!S.loading) S.loading = fetch(root.dataset.src).then((r) => r.json()).then((D) => {
      SC().setQuadrature(D.gh.x, D.gh.w, D.gh.x8, D.gh.w8);
      S.runs = D.runs.map(SC().decodeRun);
      // the engine works coarse scale first (a vertex's coarse node is made late and has a late id): so does the arithmetic
      // These data sets carry no issuer type: the inspector then guesses "multi-issuer" from pool size and draws two marks
      // a bin. The fits here treat every pool alike (one pool effect), so the sections do too: one mark a bin.
      for (const run of S.runs) if (run.typeIsProxy) for (let i = 0; i < run.pools.n; i++) run.pools.flags[i] &= ~2;
      // The reach drawn is the coefficient's own tent: what changing its surplus moves when every other surplus is held,
      // which is how the fit is parametrised (see backstory/tools/star_nodes.py). The arithmetic stops a coefficient's
      // reach at other free vertices; with the free flags cleared it carries on through them. The true flags are free0.
      for (const run of S.runs) { const V = run.vertices; V.free0 = V.free_coefficient.slice(); V.free_coefficient.fill(0); }
      for (const run of S.runs) { const V = run.vertices; run.orderById = Array.from({ length: V.n }, (_, j) => j).sort((a, b) => V.scale[a] - V.scale[b] || V.id[a] - V.id[b]); }
    });
    return S.loading;
  }
  // One entry per place on the mesh. A place whose neighbourhood has been refined holds two nodes (backstory/tools/
  // star_nodes.py): a coarse role, the centre its coarser children shrink toward, and a fine role, its own coefficient at
  // the new scale. `roles` are the free coefficients there, coarse first; `geo` is the node the surface itself reads.
  function points(run) {
    const V = run.vertices, at = new Map();
    for (let j = 0; j < V.n; j++) {
      if (V.active[j] !== 1 || V.standin[j] === 1) continue;
      const key = V.x[j].toFixed(7) + "," + V.y[j].toFixed(7); let q = at.get(key);
      if (!q) at.set(key, q = { x: V.x[j], y: V.y[j], roles: [], geo: null, corner: false });
      if (V.virtual[j] !== 1) { q.geo = V.id[j]; q.corner = V.parent0[j] < 0 && V.coarse[j] < 0 || V.depth[j] === 0; }
      if (V.free0[j] === 1) q.roles.push(V.id[j]);
    }
    for (const q of at.values()) { q.roles.sort((a, b) => V.scale[run.byId.get(a)] - V.scale[run.byId.get(b)]); q.free = q.roles.length > 0; q.id = q.free ? q.roles[0] : q.geo; }
    return [...at.values()];
  }
  const placeOf = (id) => { const [x, y] = xy(S.run, id); return S.verts.find((q) => Math.abs(q.x - x) < 1e-7 && Math.abs(q.y - y) < 1e-7); };
  // The vertex a data set opens on: among the admitted coefficients inside the map (not on its edge) that are deep
  // enough to have a real ancestry, have others borrowing from them, and visibly borrow themselves (they keep between
  // 45% and 90% of what their own pools say), the one whose split bought the most. (The inspector opens on the best split outright; that one is usually on the map's edge, with half a star.)
  function pickInitial(run) {
    const V = run.vertices; let best = null, top = -1, any = null, anyTop = -1;
    const kids = new Map(); for (let j = 0; j < V.n; j++) if (V.active[j] === 1 && V.standin[j] !== 1) for (const p of new Set([V.parent0[j], V.parent1[j]])) if (p >= 0) kids.set(p, (kids.get(p) || 0) + 1);
    for (let j = 0; j < V.n; j++) {
      if (V.active[j] !== 1 || V.free0[j] !== 1 || V.parent0[j] < 0 || V.depth[j] === 0) continue;
      const g = V.admit_gain ? V.admit_gain[j] : NaN; if (!isFinite(g)) continue;
      if (g > anyTop) { anyTop = g; any = V.id[j]; }
      const inside = V.x[j] > 0.02 && V.x[j] < 0.98 && V.y[j] > 0.02 && V.y[j] < 0.98;
      const keep = V.current_xTx[j] > 0 ? 1 / (1 + V.lambda_v[j] / V.current_xTx[j]) : 1;   // the share of its own pools' say that it keeps
      if (inside && V.depth[j] >= 4 && (kids.get(V.id[j]) || 0) >= 2 && V.current_fit_points[j] >= 60 && keep < 0.9 && keep > 0.45 && g > top) { top = g; best = V.id[j]; }
    }
    return best ?? any ?? V.id[0];
  }
  function findAddress(run, text) {
    const a = SC().parseAddress(text); if (!a) return null;
    const x = a[1] / 2 ** a[0], y = a[2] / 2 ** a[0], p = S.verts.find((q) => Math.abs(q.x - x) < 1e-7 && Math.abs(q.y - y) < 1e-7);
    return p && p.free ? p.id : null;
  }
  const centres = (run, id) => { const V = run.vertices, j = run.byId.get(id); return V.coarse[j] >= 0 ? [V.coarse[j]] : [...new Set([V.parent0[j], V.parent1[j]])].filter((p) => p >= 0 && V.active[run.byId.get(p)] === 1); };
  function ancestry(run, vid) {   // id -> generations up
    const V = run.vertices, seen = new Map([[vid, 0]]); let gen = [vid], k = 0;
    while (gen.length && k < 16) { const next = []; ++k; for (const id of gen) for (const p of centres(run, id)) if (!seen.has(p)) { seen.set(p, k); next.push(p); } gen = next; }
    return seen;
  }
  const xy = (run, id) => { const j = run.byId.get(id); return [run.vertices.x[j], run.vertices.y[j]]; };

  // ------------------------------------------------------------------ choosing a vertex
  function select(vid) {
    const run = S.run, req = ++S.req;
    S.vid = vid; S.hov = null;
    S.card = SC().cardOf(run, vid, false);
    S.sf = SC().starFaces(run, vid); S.nodal = new Set(SC().nodalFaces(run, vid)); S.infl = SC().influence(run, vid);
    S.order = S.card.poolList.map((e, q) => ({ i: e.i, s: S.card.share[q] })).sort((a, b) => b.s - a.s);
    S.anc = ancestry(run, vid);
    const V = run.vertices; S.kids = [];
    for (let j = 0; j < V.n; j++) if (V.active[j] === 1 && V.standin[j] !== 1 && V.id[j] !== vid && centres(run, V.id[j]).includes(vid)) S.kids.push(V.id[j]);
    S.place = placeOf(vid);
    S.others = S.place.roles.filter((id) => id !== vid).map((id) => SC().starFaces(run, id));   // its reach at its other scales
    S.secA = S.secB = null;
    // the star's frame: what the slider closes in on
    let x0 = 1, x1 = 0, y0 = 1, y1 = 0; const F = run.faces;
    for (const s of [S.sf, ...S.others].flat()) for (const id of [F.v0[s.f], F.v1[s.f], F.v2[s.f]]) { const [x, y] = xy(run, id); x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y); }
    S.box = [x0, x1, y0, y1];
    cards(); shrink(); lineage(); draw(); sections();
    // the pool effect and the bands are the slow part: in short slices, so the screen keeps answering
    const vs = SC().starVarianceStart(run, vid);
    const fitVariance = () => {
      if (S.req !== req) return;
      const until = performance.now() + 20;
      while (performance.now() < until) if (!SC().starVarianceStep(vs)) return bands(SC().starVarianceFinish(vs));
      setTimeout(fitVariance, 0);
    };
    const bands = (s2) => {
      S.secA = SC().sectionData(run, vid, "A", false, s2, false); S.secB = SC().sectionData(run, vid, "B", false, s2, false);
      sections();
      const todo = []; for (const sec of [S.secA, S.secB]) for (const b of sec.bins) if (!b.zero && b.band) todo.push(b);
      let done = 0, running = null;
      const step = () => {
        if (S.req !== req) return;
        const until = performance.now() + 20;
        while (done < todo.length && performance.now() < until) {
          if (!running) running = SC().bandStart(todo[done].band);
          if (SC().bandStep(running)) continue;
          const r = SC().bandFinish(running), w = wide(running); todo[done].lo = r[0]; todo[done].hi = r[1]; todo[done].lo2 = w[0]; todo[done].hi2 = w[1]; running = null; done++;
        }
        sections();
        if (done < todo.length) setTimeout(step, 0);
      };
      step();
    };
    setTimeout(fitVariance, 10);
  }

  // The inspector's band is the middle two thirds of its simulated draws (bandFinish: the 16th to 84th percentile). The
  // same draws, put on the same scale the same way, also give the middle nineteen in twenty, drawn as the thin whisker.
  function wide(st) {
    if (st.draws.length < 50) return [NaN, NaN];
    const b = st.b, lg = st.draws.map((x) => { const pp = Math.min(Math.max((x + 0.5) / (b.pn + 1), 1e-12), 1 - 1e-12); return Math.log(pp / (1 - pp)); }).sort((p, q) => p - q);
    const at = (q) => { const i = (q / 100) * (lg.length - 1), lo = Math.floor(i), hi = Math.ceil(i); return lg[lo] + (lg[hi] - lg[lo]) * (i - lo); };
    return [at(2.5) - b.nullG, at(97.5) - b.nullG];
  }

  // ------------------------------------------------------------------ the numbers
  function cards() {
    const c = S.card, run = S.run, V = run.vertices, j = run.byId.get(c.id), free = V.free0[j] === 1, num = (v, d) => (isFinite(v) ? v.toFixed(d) : "\u2013");
    const roles = S.place ? S.place.roles : [];
    $("st-title").innerHTML = `<span>${run.label || run.key}</span>` + (roles.length > 1 ? `<span class="st-roles" title="w and s move this vertex to its coarser and finer scale">this vertex at ${roles.map((id) => `<button data-id="${id}"${id === c.id ? ' class="on"' : ""}>L${V.scale[run.byId.get(id)]}</button>`).join("")}</span>` : "");
    const rows = [
      ["Vertex", c.name, `${roles.length > 1 ? "seen at L" + V.scale[j] + ", one of its " + roles.length + " scales" : ""}${roles.length > 1 ? "; " : ""}${c.poolsFit.toLocaleString()} pools under it`],
      c.admit ? ["Score", num(Math.abs(c.admit.z), 1), `needed ${num(c.admit.c, 1)} to be admitted`] : ["Score", "\u2013", free ? "a starting coefficient, never scored" : "not admitted: it sits at its parents\u2019 average"],
    ];
    $("st-cards").innerHTML = rows.map(([k, v, d]) => `<div class="card"><div class="k">${k}</div><div class="v">${v}</div><div class="d">${d}</div></div>`).join("");
  }

  // ------------------------------------------------------------------ how the borrowing works
  // A vertex's value is a compromise between two things: what its own pools say, and what its parents give it (their
  // average). The engine's record of the vertex says how the compromise came out (run_hier_vertices.csv): its data-only
  // variance s2 = 1 / current_xTx, the prior precision at its scale lambda_v, and the surplus it kept, d = height - centre.
  // The share of its own pools' say that it keeps is 1 / (1 + lambda s2), and what the pools alone say is d (1 + lambda s2):
  // the normal-normal identity that also gives the engine's posterior variance. Shown two ways: a bar split into the two
  // shares, and, on the sections below, the tent the pools alone ask for (dashed) above the tent the vertex keeps (solid),
  // with the pull between their peaks.
  function shrink() {
    const el = $("st-split"), run = S.run, V = run.vertices, j = run.byId.get(S.vid), c = centres(run, S.vid); S.shr = null;
    if (!c.length || V.free0[j] !== 1 || !(V.current_xTx[j] > 0) || !(V.lambda_v[j] > 0)) { el.innerHTML = ""; return; }
    const d = V.height[j] - c.reduce((a, p) => a + V.height[run.byId.get(p)], 0) / c.length, s2 = 1 / V.current_xTx[j], lam = V.lambda_v[j], keep = 1 / (1 + lam * s2);
    S.shr = { d, own: d * (1 + lam * s2), se: Math.sqrt(s2), keep };
    // Most of the borrowing is all-or-nothing, decided by the gate: a vertex the gate did not admit takes its parents'
    // average outright, and one it admitted keeps nearly all of what its own pools say (the prior's spread at each scale
    // is fitted to the admitted vertices, and they were admitted for standing well clear of their noise). So the bar is
    // the whole mesh, and the line under it is this vertex.
    const places = S.verts.filter((q) => !q.corner), own = places.filter((q) => q.free).length, held = places.length - own;
    el.innerHTML = `<div class="bar"><i class="par" style="flex:${held}"></i><i class="own" style="flex:${own}"></i></div>`
      + `<div class="lab"><span><b>${held}</b> take their parents\u2019 average</span><span><b>${own}</b> keep their own</span></div>`
      + `<div class="one">this vertex keeps <b>${Math.round(100 * keep)}%</b> of what its own pools say</div>`;
  }

  // ------------------------------------------------------------------ the map
  const ML = 58, MB = 40, MT = 8, MR = 8;
  // Two views of the one map, both always on screen, so nothing has to be zoomed: the star, framed to hold the vertex's
  // reach and its two parents; and, in the corner farthest from the vertex, the whole map with the same ancestry drawn
  // on it and the star's frame marked. As on the real-data screen the map fills its frame.
  function frames(W, H) {
    const pw = W - ML - MR, ph = H - MT - MB, run = S.run; let [x0, x1, y0, y1] = S.box;
    for (const p of SC().parentsOf(run, S.vid)) if (p >= 0) { const [x, y] = xy(run, p); x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y); }
    const z = Math.max(1, Math.min(1 / Math.max(x1 - x0, 1e-3), 1 / Math.max(y1 - y0, 1e-3)) / 1.16), half = 0.5 / z;
    const cx = Math.max(half, Math.min(1 - half, (x0 + x1) / 2)), cy = Math.max(half, Math.min(1 - half, (y0 + y1) / 2));
    const main = { x: ML, y: MT, w: pw, h: ph, X: (x) => ML + pw * (0.5 + (x - cx) * z), Y: (y) => MT + ph * (0.5 - (y - cy) * z), box: [cx - half, cx + half, cy - half, cy + half] };
    return { main };
  }
  function tone(run, i) { const P = run.pools, n = P.loans[i], p = P.phat[i], I = P.info[i]; return I > 0 ? Math.max(-1, Math.min(1, (P.k[i] - n * p) / Math.sqrt(I) / 3)) : 0; }
  function ramp(t) {   // the page's blue-to-red, through a neutral that still shows on the paper
    const d = isDark(), lo = d ? [110, 150, 255] : [31, 72, 200], mid = d ? [150, 150, 158] : [150, 146, 138], hi = d ? [255, 140, 80] : [205, 66, 20], a = t < 0 ? lo : hi, w = Math.pow(Math.abs(t), 0.7);
    return `rgb(${[0, 1, 2].map((j) => Math.round(mid[j] + (a[j] - mid[j]) * w)).join(",")})`;
  }
  let pending = 0;
  const redraw = () => { if (!pending) pending = requestAnimationFrame(() => { pending = 0; draw(); }); };
  function draw() {
    if (!S.run || S.vid === null || root.offsetParent === null) return;
    const r = cv.getBoundingClientRect(), dpr = Math.min(2, window.devicePixelRatio || 1), W = r.width, H = r.height;
    if (W < 10 || H < 10) return;
    if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(H * dpr)) { cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); }
    const g = cv.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, W, H);
    const run = S.run, V = run.vertices, F = run.faces, P = run.pools, { main } = frames(W, H); S.vw = main;
    const ink = css("--text"), blue = css("--corr"), faint = css("--faint"), ground = css("--ground") || (isDark() ? "#1f2022" : "#fff");
    const rgbOf = (c) => { g.fillStyle = c; const h = g.fillStyle; return [1, 3, 5].map((k) => parseInt(h.slice(k, k + 2), 16)).join(","); }, bl = rgbOf(blue), al = (w) => 0.03 + 0.4 * Math.max(0, Math.min(1, w));
    const who = S.vid, anc = S.anc, soft = 1;   // pointing at another vertex rings it and changes nothing else

    // one view: `big` is the star's own, with everything; the small one keeps to the network
    const scene = (g, vw, big) => {
      const { X, Y } = vw, k = big ? 1 : 0.55, pt = (id) => { const [x, y] = xy(run, id); return [X(x), Y(y)]; };
      const out = (x, y, m) => x < vw.x - m || x > vw.x + vw.w + m || y < vw.y - m || y > vw.y + vw.h + m;
      g.save(); g.beginPath(); g.rect(vw.x, vw.y, vw.w, vw.h); g.clip();
      // The star: the coefficient's tent, shaded by its weight (1 at the vertex, falling linearly to 0).
      for (const s of S.sf) {
        const p = [pt(F.v0[s.f]), pt(F.v1[s.f]), pt(F.v2[s.f])], w = [s.w0, s.w1, s.w2];
        g.beginPath(); g.moveTo(p[0][0], p[0][1]); g.lineTo(p[1][0], p[1][1]); g.lineTo(p[2][0], p[2][1]); g.closePath();
        // the weight is linear on the face: a linear gradient along its slope draws it exactly
        const d1x = p[1][0] - p[0][0], d1y = p[1][1] - p[0][1], d2x = p[2][0] - p[0][0], d2y = p[2][1] - p[0][1], det = d1x * d2y - d2x * d1y;
        const gx = ((w[1] - w[0]) * d2y - (w[2] - w[0]) * d1y) / det, gy = ((w[2] - w[0]) * d1x - (w[1] - w[0]) * d2x) / det, gl = Math.hypot(gx, gy);
        if (!(gl > 1e-9)) g.fillStyle = `rgba(${bl},${al(w[0]).toFixed(3)})`;
        else {
          const ux = gx / gl, uy = gy / gl, sv = p.map((q) => q[0] * ux + q[1] * uy), i0 = sv.indexOf(Math.min(...sv)), i1 = sv.indexOf(Math.max(...sv));
          const gr = g.createLinearGradient(p[i0][0], p[i0][1], p[i0][0] + ux * (sv[i1] - sv[i0]), p[i0][1] + uy * (sv[i1] - sv[i0]));
          gr.addColorStop(0, `rgba(${bl},${al(w[i0]).toFixed(3)})`); gr.addColorStop(1, `rgba(${bl},${al(w[i1]).toFixed(3)})`); g.fillStyle = gr;
        }
        g.fill();
      }
      // The same place at its other scales: each role's tent is outlined (dashed), so the coarse and the fine reach of
      // one vertex are on the map together; the role chosen is the one shaded, and it is outlined solid.
      const edgeOf = (sf, dash, col, wd) => {
        g.beginPath();
        for (const s of sf) { const p = [pt(F.v0[s.f]), pt(F.v1[s.f]), pt(F.v2[s.f])], w = [s.w0, s.w1, s.w2];
          for (const [i, j] of [[0, 1], [1, 2], [2, 0]]) if (w[i] < 1e-9 && w[j] < 1e-9) { g.moveTo(p[i][0], p[i][1]); g.lineTo(p[j][0], p[j][1]); } }
        g.setLineDash(dash); g.strokeStyle = col; g.lineWidth = wd * (big ? 1 : 0.6); g.stroke(); g.setLineDash([]);
      };
      // the mesh
      g.strokeStyle = ink; g.globalAlpha = big ? 0.2 : 0.16; g.lineWidth = big ? 0.8 : 0.5; g.beginPath();
      for (let q = 0; q < F.n; q++) { const a = pt(F.v0[q]), b = pt(F.v1[q]), c = pt(F.v2[q]); g.moveTo(a[0], a[1]); g.lineTo(b[0], b[1]); g.lineTo(c[0], c[1]); g.closePath(); }
      g.stroke(); g.globalAlpha = 1;
      g.globalAlpha = 0.85; for (const o of S.others) edgeOf(o, [6, 5], ink, 1.7); g.globalAlpha = 1; edgeOf(S.sf, [], blue, 1.5);
      // the pools under the vertex: area by share of its information, colour by residual; the largest first, so the small show
      if (big) for (const e of S.order) {
        const x = X(P.sx[e.i]), y = Y(P.sy[e.i]); if (out(x, y, 30)) continue;
        const rad = 1.2 + 22 * Math.sqrt(Math.max(e.s, 0));
        g.beginPath(); g.arc(x, y, rad, 0, 6.2832); g.globalAlpha = 0.85; g.fillStyle = ramp(tone(run, e.i)); g.fill();
        if (rad > 3) { g.globalAlpha = 0.6; g.strokeStyle = ground; g.lineWidth = 0.8; g.stroke(); }
      }
      g.globalAlpha = 1;
      // whom it borrows from: parent -> child arrows up the ancestry of the chosen vertex (or the one pointed at)
      const arrow = (a, b, w, alpha, cut) => {
        const dx = b[0] - a[0], dy = b[1] - a[1], L = Math.hypot(dx, dy); if (L < cut + 4) return;
        const ux = dx / L, uy = dy / L, ex = b[0] - ux * cut, ey = b[1] - uy * cut, hs = (4 + 1.6 * w) * (big ? 1 : 0.8);
        g.globalAlpha = alpha; g.strokeStyle = blue; g.fillStyle = blue; g.lineWidth = w; g.lineCap = "round";
        g.beginPath(); g.moveTo(a[0], a[1]); g.lineTo(ex - ux * hs, ey - uy * hs); g.stroke();
        g.beginPath(); g.moveTo(ex, ey); g.lineTo(ex - ux * hs * 1.7 - uy * hs * 0.75, ey - uy * hs * 1.7 + ux * hs * 0.75); g.lineTo(ex - ux * hs * 1.7 + uy * hs * 0.75, ey - uy * hs * 1.7 - ux * hs * 0.75); g.closePath(); g.fill();
      };
      for (const [id, gen] of [...anc].sort((a, b) => b[1] - a[1])) for (const p of centres(run, id)) arrow(pt(p), pt(id), Math.max(1.1, 3.4 - 0.45 * gen) * (big ? 1 : 0.6), soft * Math.max(0.25, 1 - 0.14 * gen), (gen === 0 ? 9 : 6) * k);
      // who borrows from it
      if (who === S.vid && big) { g.globalAlpha = 0.9; g.strokeStyle = blue; g.lineWidth = 1.5; g.setLineDash([4, 4]); g.beginPath(); for (const id of S.kids) { const a = pt(S.vid), b = pt(id); g.moveTo(a[0], a[1]); g.lineTo(b[0], b[1]); } g.stroke(); g.setLineDash([]); }
      // the section across the parent edge, B to B'
      const j0 = run.byId.get(S.vid), [p0, p1] = SC().parentsOf(run, S.vid); let ends = null;
      if (p0 >= 0 && big) {
        const [ax, ay] = xy(run, p0), [bx, by] = xy(run, p1), vx = V.x[j0], vy = V.y[j0], hx = (bx - ax) / 2, hy = (by - ay) / 2;
        ends = { A: [X(ax), Y(ay)], A2: [X(bx), Y(by)], B: [X(vx + hy), Y(vy - hx)], B2: [X(vx - hy), Y(vy + hx)] };
        if (ends.B[0] > ends.B2[0] || (Math.abs(ends.B[0] - ends.B2[0]) < 1 && ends.B[1] < ends.B2[1])) [ends.B, ends.B2] = [ends.B2, ends.B];
        if (ends.A[0] > ends.A2[0] || (Math.abs(ends.A[0] - ends.A2[0]) < 1 && ends.A[1] < ends.A2[1])) [ends.A, ends.A2] = [ends.A2, ends.A];
        g.globalAlpha = 0.55; g.strokeStyle = ink; g.lineWidth = 1.2; g.setLineDash([5, 5]); g.beginPath(); g.moveTo(ends.B[0], ends.B[1]); g.lineTo(ends.B2[0], ends.B2[1]); g.stroke(); g.setLineDash([]);
      }
      g.globalAlpha = 1;
      // the vertices: a solid diamond is a free coefficient; a small open circle is held at its parents' average, and is a
      // pie when it carries part of the chosen coefficient
      for (const q of S.verts) {
        const x = X(q.x), y = Y(q.y); if (out(x, y, 8)) continue;
        const on = q.roles.some((id) => anc.has(id)) || anc.has(q.geo) || S.kids.includes(q.id) || q === S.place, rad = big ? 3.1 : 1.5;
        const carried = false;   // (the tent's shading already says how much of the coefficient each vertex carries)
        g.globalAlpha = on || carried ? 1 : 0.45; g.fillStyle = ink; g.strokeStyle = ink; g.lineWidth = 1; g.beginPath();
        if (q.free) { g.moveTo(x, y - rad * 1.25); g.lineTo(x + rad * 1.25, y); g.lineTo(x, y + rad * 1.25); g.lineTo(x - rad * 1.25, y); g.closePath(); g.fill(); }
        else if (carried) { g.arc(x, y, 3.4, 0, 6.2832); g.fillStyle = ground; g.fill(); g.strokeStyle = blue; g.lineWidth = 1.4; g.stroke(); g.beginPath(); g.moveTo(x, y); g.arc(x, y, 3.4, -Math.PI / 2, -Math.PI / 2 + 6.2832 * carried); g.closePath(); g.fillStyle = blue; g.fill(); }
        else if (big) { g.arc(x, y, 1.9, 0, 6.2832); g.stroke(); }
      }
      g.globalAlpha = 1;
      for (const [id, gen] of anc) if (id !== who) { const a = pt(id); g.strokeStyle = blue; g.globalAlpha = soft * Math.max(0.4, 1 - 0.1 * gen); g.lineWidth = 1.8 * (big ? 1 : 0.7); g.beginPath(); g.arc(a[0], a[1], 6.5 * k, 0, 6.2832); g.stroke(); }
      g.globalAlpha = 1;
      if (who === S.vid && big) for (const id of S.kids) { const a = pt(id); g.strokeStyle = blue; g.lineWidth = 1.8; g.beginPath(); g.arc(a[0], a[1], 6, 0, 6.2832); g.stroke(); }
      const me = pt(S.vid); g.strokeStyle = ink; g.lineWidth = big ? 2.6 : 1.8; g.beginPath(); g.arc(me[0], me[1], 9.5 * k, 0, 6.2832); g.stroke();
      if (S.hov !== null && S.hov !== S.vid) { const h = pt(S.hov); g.strokeStyle = ink; g.lineWidth = 2 * (big ? 1 : 0.8); g.setLineDash([3, 3]); g.beginPath(); g.arc(h[0], h[1], 10 * k, 0, 6.2832); g.stroke(); g.setLineDash([]); }
      // the ends of the two sections, named
      if (ends) {
        // Each name sits just past its end, on a small plate of the page's own tone so that no line or dot runs through it.
        // It is always inside the picture: an end at or beyond an edge has its name held at that edge, moved aside from the
        // end's own mark, and two names that would land on each other are parted.
        g.font = "600 14px ui-sans-serif, system-ui, sans-serif"; g.textAlign = "center"; g.textBaseline = "middle";
        const plate = isDark() ? "#26272a" : "#efe6cd", pad = 13, placed = [];
        const tag = (p, s, col) => {
          const dx = p[0] - me[0], dy = p[1] - me[1], L = Math.hypot(dx, dy) || 1, ux = dx / L, uy = dy / L;
          const hold = (x, y) => [Math.max(vw.x + pad, Math.min(vw.x + vw.w - pad, x)), Math.max(vw.y + pad, Math.min(vw.y + vw.h - pad, y))];
          let [x, y] = hold(p[0] + ux * 15, p[1] + uy * 15);
          if (Math.hypot(x - p[0], y - p[1]) < 12) {   // held on top of the end's mark: step aside, to whichever side has room
            const c1 = hold(p[0] - uy * 17 - ux * 4, p[1] + ux * 17 - uy * 4), c2 = hold(p[0] + uy * 17 - ux * 4, p[1] - ux * 17 - uy * 4);
            [x, y] = Math.hypot(c1[0] - p[0], c1[1] - p[1]) >= Math.hypot(c2[0] - p[0], c2[1] - p[1]) ? c1 : c2;
          }
          if (Math.hypot(x - me[0], y - me[1]) < 18) {   // held on top of the vertex itself (its end lies outside the map): beside it
            const c1 = hold(me[0] - uy * 22, me[1] + ux * 22), c2 = hold(me[0] + uy * 22, me[1] - ux * 22);
            [x, y] = Math.hypot(c1[0] - me[0], c1[1] - me[1]) >= Math.hypot(c2[0] - me[0], c2[1] - me[1]) ? c1 : c2;
          }
          for (const q of placed) if (Math.abs(q[0] - x) < 24 && Math.abs(q[1] - y) < 20) [x, y] = hold(x, y + (y >= q[1] ? 1 : -1) * (20 - Math.abs(q[1] - y)));
          placed.push([x, y]);
          const w = g.measureText(s).width + 10, h = 20;
          g.globalAlpha = 0.94; g.fillStyle = plate; g.beginPath(); g.roundRect(x - w / 2, y - h / 2, w, h, 6); g.fill();
          g.globalAlpha = 0.45; g.strokeStyle = col; g.lineWidth = 1; g.stroke();
          g.globalAlpha = 1; g.fillStyle = col; g.fillText(s, x, y + 0.5);
        };
        tag(ends.A, "A", blue); tag(ends.A2, "A\u2032", blue); tag(ends.B, "B", ink); tag(ends.B2, "B\u2032", ink);
      }
      g.restore();
    };

    scene(g, main, true);
    // the whole map, in its own small frame above the lineage: the same ancestry on it, and a box where the star's view is
    const mc = $("st-mini"), mr = mc.getBoundingClientRect();
    if (mr.width > 10) {
      if (mc.width !== Math.round(mr.width * dpr) || mc.height !== Math.round(mr.height * dpr)) { mc.width = Math.round(mr.width * dpr); mc.height = Math.round(mr.height * dpr); }
      const m = mc.getContext("2d"); m.setTransform(dpr, 0, 0, dpr, 0, 0); m.clearRect(0, 0, mr.width, mr.height);
      const iw = mr.width - 2, ih = mr.height - 2, mini = { x: 1, y: 1, w: iw, h: ih, X: (x) => 1 + iw * x, Y: (y) => 1 + ih * (1 - y) }; S.mini = mini;
      scene(m, mini, false);
      const bx = main.box; m.strokeStyle = ink; m.lineWidth = 1.5; m.strokeRect(mini.X(bx[0]), mini.Y(bx[3]), mini.X(bx[1]) - mini.X(bx[0]), mini.Y(bx[2]) - mini.Y(bx[3]));
      m.strokeStyle = css("--border") || "rgba(127,127,127,.3)"; m.lineWidth = 1; m.strokeRect(0.5, 0.5, mr.width - 1, mr.height - 1);
    }
    // the frame and its scales
    const { X, Y } = main;
    g.strokeStyle = css("--border") || "rgba(127,127,127,.3)"; g.lineWidth = 1; g.strokeRect(ML + 0.5, MT + 0.5, main.w - 1, main.h - 1);
    g.fillStyle = faint; g.font = "12px ui-sans-serif, system-ui, sans-serif";
    const m = run.model, age = (x) => m.wala_min + x * (m.wala_max - m.wala_min), cpn = (y) => m.wac_min + y * (m.wac_max - m.wac_min);
    const ticks = (lo, hi, n) => { const s0 = (hi - lo) / n, mag = Math.pow(10, Math.floor(Math.log10(s0))), q = s0 / mag, st = (q < 1.5 ? 1 : q < 3 ? 2 : q < 7 ? 5 : 10) * mag, t = []; for (let v = Math.ceil(lo / st) * st; v <= hi + 1e-9; v += st) t.push(v); return t; };
    g.textAlign = "center"; g.textBaseline = "top";
    for (const a of ticks(age(main.box[0]), age(main.box[1]), 6)) g.fillText(String(+a.toFixed(1)), X((a - m.wala_min) / (m.wala_max - m.wala_min)), MT + main.h + 5);
    g.textAlign = "right"; g.textBaseline = "middle";
    for (const c of ticks(cpn(main.box[2]), cpn(main.box[3]), 5)) g.fillText(+c.toFixed(2) + "%", ML - 6, Y((c - m.wac_min) / (m.wac_max - m.wac_min)));
    // the axes' names, as on the screen before this one
    g.textAlign = "center"; g.textBaseline = "top"; g.fillText("loan age, months", ML + main.w / 2, MT + main.h + 22);
    g.save(); g.translate(10, MT + main.h / 2); g.rotate(-Math.PI / 2); g.textBaseline = "middle"; g.fillText("coupon", 0, 0); g.restore();
  }

  // ------------------------------------------------------------------ the lineage: the hierarchy, drawn as one
  // Down the page is finer. The chosen vertex sits in its row; above it, row by row, everything its value is built
  // from: its two parents, theirs, up to the corners of the map (where a place holds two roles, the fine one hangs
  // straight under the coarse one). Below it, the vertices that borrow from it in turn. Each dot is one vertex's own
  // departure from what those above it give it: red above, blue below, larger for more. Pointing at a dot shows that
  // vertex on the map; a click goes to it, so the hierarchy can be walked up and down.
  function lineage() {
    const el = $("st-lin"), run = S.run, V = run.vertices, J = (id) => run.byId.get(id), anc = S.anc;
    const own = (id) => { const c = centres(run, id); return c.length ? V.height[J(id)] - c.reduce((a, p) => a + V.height[J(p)], 0) / c.length : null; };
    const node = (id, gen) => ({ id, row: V.scale[J(id)], gx: V.x[J(id)], gy: V.y[J(id)], own: own(id), gen });
    const nodes = [...anc.keys()].map((id) => node(id, anc.get(id)));
    // the same place seen at a finer scale: its other roles below this one, hung straight under it, not among the borrowers
    const finer = S.place.roles.filter((id) => V.scale[J(id)] > V.scale[J(S.vid)]); for (const id of finer) nodes.push(node(id, -1));
    const kids = S.kids.filter((id) => !finer.includes(id)).sort((a, b) => V.x[J(a)] - V.x[J(b)] || V.y[J(a)] - V.y[J(b)]), PER = 8, shown = kids.slice(0, 2 * PER);
    const levels = [...new Set(nodes.map((n) => n.row))].sort((a, b) => a - b), krows = Math.ceil(shown.length / PER);
    const L = 50, R = 284, T = 22, KH = 34, B = 534 - (krows ? 36 + KH * krows : 0), rowY = (i) => (levels.length > 1 ? T + ((B - T) * i) / (levels.length - 1) : (T + B) / 2);
    const x0 = Math.min(...nodes.map((n) => n.gx)), x1 = Math.max(...nodes.map((n) => n.gx)), top = Math.max(1e-9, ...nodes.map((n) => Math.abs(n.own || 0)));
    const place = (list, gap) => {   // across by where the vertex is on the map, then pushed apart to a minimum gap
      list.sort((a, b) => a.gx - b.gx || a.gy - b.gy);
      list.forEach((n) => { n.x = x1 > x0 ? L + ((n.gx - x0) / (x1 - x0)) * (R - L) : (L + R) / 2; });
      for (let i = 1; i < list.length; ++i) list[i].x = Math.max(list[i].x, list[i - 1].x + gap);
      const over = list.length ? list[list.length - 1].x - R : 0;
      if (over > 0) { list[list.length - 1].x = R; for (let i = list.length - 2; i >= 0; --i) list[i].x = Math.min(list[i].x, list[i + 1].x - gap); }
    };
    levels.forEach((lv, i) => { const row = nodes.filter((n) => n.row === lv); place(row, Math.min(34, (R - L) / Math.max(1, row.length - 1))); row.forEach((n) => { n.y = rowY(i); }); });
    const at = new Map(nodes.map((n) => [n.id, n])), me = at.get(S.vid), same = (a, b) => Math.abs(a.gx - b.gx) < 1e-9 && Math.abs(a.gy - b.gy) < 1e-9;
    // The chosen vertex at its other scales is the same place, not another vertex: all of them stand on one rail, straight
    // above and below it, drawn as squares; every other vertex is a circle.
    for (const n of nodes) { n.self = same(n, me); if (n.self) n.x = me.x; }
    const selves = nodes.filter((n) => n.self).sort((a, b) => a.y - b.y);
    const kn = shown.map((id, i) => ({ id, y: B + 46 + KH * Math.floor(i / PER), x: L + ((R - L) * ((i % PER) + 0.5)) / Math.min(PER, shown.length - PER * Math.floor(i / PER)) }));
    let h = levels.map((lv, i) => `<text class="or-ax" x="6" y="${f(rowY(i) + 4)}">L${lv}</text>`).join("");
    if (selves.length > 1) h += `<rect class="st-lrail" x="${f(me.x - 17)}" y="${f(selves[0].y - 19)}" width="34" height="${f(selves[selves.length - 1].y - selves[0].y + 38)}" rx="17"/>`;
    for (const n of nodes) for (const p of centres(run, n.id)) { const a = at.get(p); if (!a) continue;
      h += same(a, n) ? `<path class="st-lsame" d="M${f(a.x)} ${f(a.y)}L${f(n.x)} ${f(n.y)}"/>`   // one place, two scales
        : `<path class="st-ledge" style="opacity:${Math.max(0.3, 1 - 0.13 * Math.max(0, n.gen)).toFixed(2)};stroke-width:${Math.max(1.2, 3.2 - 0.4 * Math.max(0, n.gen)).toFixed(1)}" d="M${f(a.x)} ${f(a.y)}L${f(n.x)} ${f(n.y)}"/>`; }
    const low = finer.length ? at.get(finer[finer.length - 1]) : me;
    for (const c of kn) h += `<path class="st-lkid" d="M${f(me.x)} ${f(me.y)}L${f(c.x)} ${f(c.y)}"/>`;
    let hits = "";
    for (const n of nodes) {
      const r = n.own === null ? 5 : 4.5 + 9 * Math.sqrt(Math.abs(n.own) / top), cls = n.own === null ? "root" : n.own > 0 ? "up" : "dn";
      const q = Math.max(6, r * 0.9);   // a square of about the circle's area
      h += (n.self ? `<rect class="st-ln ${cls}" data-n="${n.id}" x="${f(n.x - q)}" y="${f(n.y - q)}" width="${f(2 * q)}" height="${f(2 * q)}" rx="2.5"/>` : `<circle class="st-ln ${cls}" data-n="${n.id}" cx="${f(n.x)}" cy="${f(n.y)}" r="${f(r)}"/>`)
        + (n.id === S.vid ? `<rect class="st-lme" x="${f(n.x - q - 5)}" y="${f(n.y - q - 5)}" width="${f(2 * q + 10)}" height="${f(2 * q + 10)}" rx="5"/>` : "");
      hits += `<circle class="st-hit" data-id="${n.id}" cx="${f(n.x)}" cy="${f(n.y)}" r="15"/>`;
    }
    for (const c of kn) { h += `<circle class="st-ln kid" data-n="${c.id}" cx="${f(c.x)}" cy="${f(c.y)}" r="7"/>`; hits += `<circle class="st-hit" data-id="${c.id}" cx="${f(c.x)}" cy="${f(c.y)}" r="14"/>`; }
    if (kids.length > shown.length) h += `<text class="or-ax" x="${R}" y="${f(B + 44 + KH * krows)}" text-anchor="end">and ${kids.length - shown.length} more</text>`;
    el.innerHTML = h + '<text class="st-lname" id="st-lname"></text>' + hits;
  }
  // Pointing at a dot names it and rings it, here and on the map, and changes nothing else; a click goes to it.
  { const el = $("st-lin"), idOf = (ev) => { const t = ev.target.closest && ev.target.closest(".st-hit"); return t ? +t.dataset.id : null; };
    const point = (id, t) => {
      for (const c of el.querySelectorAll(".st-ln.hot")) c.classList.remove("hot");
      const name = $("st-lname"); if (!name) return;
      if (id === null) { name.textContent = ""; return; }
      const dot = el.querySelector(`.st-ln[data-n="${id}"]`); if (dot) dot.classList.add("hot");
      const V = S.run.vertices, j = S.run.byId.get(id), a = SC().address(V.x[j], V.y[j]), x = +t.getAttribute("cx"), y = +t.getAttribute("cy");
      name.textContent = a ? `L${a[0]} (${a[1]}, ${a[2]})` : ""; name.setAttribute("x", x > 190 ? x - 16 : x + 16); name.setAttribute("y", y - 14); name.setAttribute("text-anchor", x > 190 ? "end" : "start");
    };
    el.addEventListener("pointermove", (ev) => { const id = idOf(ev); if (id !== S.hov) { S.hov = id; point(id, ev.target); redraw(); } });
    el.addEventListener("pointerleave", () => { if (S.hov !== null) { S.hov = null; point(null); redraw(); } });
    el.addEventListener("click", (ev) => { const id = idOf(ev); if (id === null) return; const q = placeOf(id); if (q) go(q); else if (id !== S.vid) select(id); }); }

  // ------------------------------------------------------------------ the two sections
  function sections() {
    const secs = [S.secA, S.secB];
    const most = Math.max(1, ...secs.flatMap((q) => (q ? q.bins.filter((b) => !b.zero).map((b) => b.loans) : [])));   // the dots' sizes share one scale
    let lo = 0, hi = 0, tlo = 0, thi = 0;
    for (const s of secs) {
      if (!s) continue;
      // a bin with next to no loans does not set the scale; it is drawn at the edge
      for (const b of s.bins) { if (b.zero || b.loans < 0.03 * most) continue; lo = Math.min(lo, b.y, isFinite(b.lo) ? b.lo : b.y); hi = Math.max(hi, b.y, isFinite(b.hi) ? b.hi : b.y); }
      for (const p of s.line) { lo = Math.min(lo, p[1]); hi = Math.max(hi, p[1]); }
      for (const p of s.tent) { tlo = Math.min(tlo, p[1]); thi = Math.max(thi, p[1]); }
    }
    if (S.shr) { tlo = Math.min(tlo, S.shr.own - S.shr.se); thi = Math.max(thi, S.shr.own + S.shr.se); }   // room for what the pools alone ask for
    const span = Math.max(hi - lo, 0.4);   // the coefficient's line may stretch the scale by half again, no more (as the inspector)
    lo = Math.max(Math.min(lo, tlo), lo - 0.8 * span); hi = Math.min(Math.max(hi, thi), hi + 0.8 * span);
    const pad = 0.1 * (hi - lo || 1); lo -= pad; hi += pad;
    const L = 46, R = 440 - 16, T = 10, B = 136, X = (t) => L + (t + 1.06) / 2.12 * (R - L), Y = (y) => B - (Math.max(lo, Math.min(hi, y)) - lo) / (hi - lo) * (B - T);
    secs.forEach((sec, q) => {
      const el = $(q ? "st-secB" : "st-secA"), ends = q ? ["B", "B′"] : ["A", "A′"];
      let s = `<path class="or-axis" d="M${L} ${B}H${R}"/><path class="nw-line" d="M${L} ${f(Y(0))}H${R}"/>`;
      const st = (hi - lo) / 3, mag = Math.pow(10, Math.floor(Math.log10(st))), nice = (st / mag < 1.5 ? 1 : st / mag < 3.5 ? 2 : st / mag < 7.5 ? 5 : 10) * mag;
      for (let v = Math.ceil(lo / nice) * nice; v <= hi; v += nice) s += `<text class="or-ax" x="${L - 7}" y="${f(Y(v) + 4)}" text-anchor="end">${Math.abs(v) < 1e-9 ? "0" : (v > 0 ? "+" : "−") + Math.abs(v).toFixed(nice < 1 ? 1 : 0)}</text>`;
      s += `<text class="st-end${q ? "" : " a"}" x="${f(X(-1))}" y="${B + 20}" text-anchor="middle">${ends[0]}</text><text class="st-end${q ? "" : " a"}" x="${f(X(1))}" y="${B + 20}" text-anchor="middle">${ends[1]}</text>`;
      if (sec) {
        // what its pools alone ask for: the same tent, as tall as their say (dashed), and the pull from its peak to the kept one
        const sh = S.shr, up = sh && Math.abs(sh.d) > 1e-9 ? sh.own / sh.d : 1;
        if (sh) s += `<path class="st-alone" d="M${sec.tent.map((p) => f(X(p[0])) + " " + f(Y(p[1] * up))).join("L")}"/>`;
        s += `<path class="st-tent" d="M${sec.tent.map((p) => f(X(p[0])) + " " + f(Y(p[1]))).join("L")}"/>`;
        if (sh) s += `<path class="or-bar" d="M${f(X(0))} ${f(Y(sh.own - sh.se))}V${f(Y(sh.own + sh.se))}"/><path class="or-pull" style="stroke-width:4" d="M${f(X(0))} ${f(Y(sh.own))}V${f(Y(sh.d))}"/>`
          + `<circle class="or-raw" cx="${f(X(0))}" cy="${f(Y(sh.own))}" r="6.5"/><circle class="or-shrunk" cx="${f(X(0))}" cy="${f(Y(sh.d))}" r="5"/>`;
        // Each bin of pools: what the fit expects there (the short dark bar), the noise around it (the box holds two
        // draws in three, the whisker nineteen in twenty), and what was seen (the dot, larger for more loans, ringed in
        // red when it falls outside the whisker). A small open circle on the axis is a bin with no terminations at all.
        const expect = new Map(sec.line.map((p) => [p[0].toFixed(4), p[1]]));
        for (const b of sec.bins) {
          const x = X(b.x);
          if (b.zero) { s += `<circle class="st-none" cx="${f(x)}" cy="${B - 5}" r="3"/>`; continue; }
          if (isFinite(b.lo2)) s += `<path class="st-whisk" d="M${f(x)} ${f(Y(b.lo2))}V${f(Y(b.hi2))}"/>`;
          if (isFinite(b.lo)) s += `<rect class="st-box" x="${f(x - 5)}" y="${f(Y(b.hi))}" width="10" height="${f(Math.max(1.5, Y(b.lo) - Y(b.hi)))}" rx="2.5"/>`;
          const e = expect.get(b.x.toFixed(4)); if (e !== undefined) s += `<path class="st-exp" d="M${f(x - 7)} ${f(Y(e))}h14"/>`;
        }
        for (const b of sec.bins) if (!b.zero) {
          const off = isFinite(b.lo2) && (b.y < b.lo2 || b.y > b.hi2);
          s += `<circle class="st-obs${off ? " off" : ""}" cx="${f(X(b.x))}" cy="${f(Y(b.y))}" r="${f((off ? 4 : 3.4) + 3 * Math.sqrt(b.loans / most))}"/>`;
        }
      }
      el.innerHTML = s;
    });
  }

  // ------------------------------------------------------------------ pointer: preview an ancestry, or choose a vertex
  function nearest(ev) {
    if (!S.vw) return null;
    const small = ev.currentTarget !== cv, r = ev.currentTarget.getBoundingClientRect(), px = ev.clientX - r.left, py = ev.clientY - r.top, vw = small ? S.mini : S.vw; if (!vw) return null;
    let best = null, bd = small ? 81 : 324;
    for (const q of S.verts) { if (q.corner || (small && !q.free)) continue; const d = (vw.X(q.x) - px) ** 2 + (vw.Y(q.y) - py) ** 2 - (q.free && !small ? 40 : 0); if (d < bd) { bd = d; best = q; } }
    return best;
  }
  cv.addEventListener("pointermove", (ev) => { const q = nearest(ev), id = q && q !== S.place ? q.id : null; cv.style.cursor = q ? "pointer" : "default"; if (id !== S.hov) { S.hov = id; redraw(); } });
  cv.addEventListener("pointerleave", () => { if (S.hov !== null) { S.hov = null; redraw(); } });
  // A click on a vertex, on the map, the small map or the lineage: another vertex opens at its coarsest scale; the one
  // already open goes one scale finer (and from its finest back to its coarsest), if it has more than one.
  function go(q) {
    if (q !== S.place) select(q.id);
    else if (q.roles.length > 1) select(q.roles[(q.roles.indexOf(S.vid) + 1) % q.roles.length]);
  }
  cv.addEventListener("click", (ev) => { const q = nearest(ev); if (q) go(q); });
  { const mc = $("st-mini");
    mc.addEventListener("pointermove", (ev) => { const q = nearest(ev), id = q && q !== S.place ? q.id : null; mc.style.cursor = q ? "pointer" : "default"; if (id !== S.hov) { S.hov = id; redraw(); } });
    mc.addEventListener("pointerleave", () => { if (S.hov !== null) { S.hov = null; redraw(); } });
    mc.addEventListener("click", (ev) => { const q = nearest(ev); if (q) go(q); }); }
  $("st-title").addEventListener("click", (ev) => { const b = ev.target.closest("button[data-id]"); if (b && +b.dataset.id !== S.vid) select(+b.dataset.id); });
  let rz = 0; window.addEventListener("resize", () => { clearTimeout(rz); rz = setTimeout(draw, 120); });
  new MutationObserver(() => setTimeout(draw, 40)).observe(document.documentElement, { attributes: true, attributeFilter: ["class"] });

  // ------------------------------------------------------------------ outside
  // open(key, address): the data set (the run's key, e.g. "fnma-202002"; an unknown one keeps the current) and,
  // optionally, the vertex to choose. shown(): the screen has come up; draw at its real size.
  function open(key, addr) {
    S.want = { key, addr: addr || (S.want && S.want.addr) || null };   // a later call without a vertex keeps one still waiting
    load().then(() => {
      const w = S.want; if (!w) return; S.want = null;
      const run = S.runs.find((r) => r.key === w.key) || S.run || S.runs[0], changed = run !== S.run;
      if (changed) { S.run = run; S.verts = points(run); }
      const vid = (w.addr && findAddress(run, w.addr)) ?? (changed || S.vid === null ? pickInitial(run) : S.vid);
      if (changed || vid !== S.vid) select(vid); else draw();
    });
  }
  // where(): where on the window the chosen vertex is drawn (for the tests in shots/)
  // role(d): the chosen vertex at its next coarser (-1) or finer (+1) scale, if it has one
  const role = (d) => { if (!S.place) return; const r = S.place.roles, i = r.indexOf(S.vid) + d; if (i >= 0 && i < r.length) select(r[i]); };
  return { open, role, shown: () => { if (S.run) draw(); }, where: () => { const r = cv.getBoundingClientRect(), [x, y] = xy(S.run, S.vid); return [r.left + S.vw.X(x), r.top + S.vw.Y(y)]; } };
})();
