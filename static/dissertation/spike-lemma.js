// /dissertation/spike, "Two ways to carry on, one equilibrium": Chapter 6's lemma on blip and frozen continuations,
// drawn from noisestate's numbers for the Chapter 3 game (tools/spike/compute_spike_lives.py -> spike/lemma.json).
// The builder: deviations add up, so a blip's life is the frozen life of the kick plus the frozen lives of the small
// later spikes its follow-up is made of (B = F + c * F, c the follow-up D^{1<-1}), and a frozen spike's life is a blip's
// minus blips canceling its follow-up, round after round (F = B + w * B, w = -c + c*c - ...); the sums are computed
// here from the stored lives and shown as they converge.  The bowls: player 1's extra cost against the size of a change
// to its strategy, J(e) - J(0) = slope e + curvature e^2, followed either way and counted as if nobody reacted.
(function () {
  "use strict";
  const NS = "http://www.w3.org/2000/svg";
  const me = document.currentScript;
  const svgC = document.getElementById("build-svg"), svgB = document.getElementById("lemma-bowls");
  if (!me || !svgC || !svgB) return;
  // data fetches say so when they fail, instead of leaving blank figures and dead controls
  const getJSON = (u) => fetch(u).then((r) => { if (!r.ok) throw new Error(`${u}: ${r.status}`); return r.json(); });
  const failed = (ids) => (e) => {
    console.error(e);
    for (const id of ids) {
      const f = document.getElementById(id);
      if (!f || f.previousElementSibling?.classList.contains("loadfail")) continue;
      const p = document.createElement("p"); p.className = "loadfail"; p.textContent = "The data for this figure could not be loaded; try reloading the page.";
      f.before(p);
    }
  };
  const el = (tag, attrs, parent) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); if (parent) parent.appendChild(e); return e; };
  const fmt = (v) => (Math.abs(v) < 0.0005 ? "0.000" : (v < 0 ? "−" : "+") + Math.abs(v).toFixed(3));
  const EMAX = 0.4;                                   // the bowls' range of the change's size

  getJSON(me.dataset.lemma).then((d) => {
    const eps = document.getElementById("lemma-eps");
    const COLORS = { frozen: "var(--ink-3)", blip: "var(--accent)", alone: "var(--ink)" };
    const NAMES = { frozen: "held still", blip: "playing on", alone: "as if nobody reacted" };

    function legend(id, items) {
      const box = document.getElementById(id); box.innerHTML = "";
      return items.map(([key, label]) => {
        const s = document.createElement("span");
        s.innerHTML = `<i style="background:${COLORS[key] || key}"></i>${label} <b></b>`;
        box.appendChild(s);
        return s.querySelector("b");
      });
    }

    // ---------------------------------------------------------------- the builder
    const du = d.du, F = d.lives.frozen.X, B = d.lives.blip.X, cont = d.lives.blip.D1, n = Math.min(F.length, B.length, cont.length);
    const SHOW = (n - 1) * du;
    function conv(a, b) {                              // (a * b)(age) = int_0^age a(s) b(age - s) ds, trapezoid
      const out = new Array(n).fill(0);
      for (let m = 1; m < n; ++m) {
        let acc = 0.5 * (a[0] * b[m] + a[m] * b[0]);
        for (let k = 1; k < m; ++k) acc += a[k] * b[m - k];
        out[m] = acc * du;
      }
      return out;
    }
    // a blip from frozen spikes: the follow-up as NSP spikes, at the centers of cells of width DT, each of its mass there
    const DT = 0.1, NSP = Math.round(SHOW / DT);
    const spikes = Array.from({ length: NSP }, (_, j) => {
      const lo = Math.round((j * DT) / du), hi = Math.round(((j + 1) * DT) / du);
      let m = 0; for (let k = lo; k < hi; ++k) m += cont[k] * du;
      return { t: (j + 0.5) * DT, mass: m, rate: cont[Math.min(n - 1, Math.round(((j + 0.5) * DT) / du))] };
    });
    const shifted = (y, t, scale) => { const k0 = Math.round(t / du); return Array.from({ length: n }, (_, m) => (m >= k0 ? scale * y[m - k0] : 0)); };
    const blipSums = [F.slice()];
    for (const sp of spikes) { const prev = blipSums[blipSums.length - 1], add = shifted(F, sp.t, sp.mass); blipSums.push(prev.map((v, m) => v + add[m])); }
    // a frozen spike from blips: rounds of corrections, w_k = sum_{i<=k} (-c)^{*i}
    const ROUNDS = 6, frozenSums = [B.slice()], weights = [new Array(n).fill(0)];
    let term = cont.map((v) => -v), w = new Array(n).fill(0);
    for (let r = 1; r <= ROUNDS; ++r) {
      w = w.map((v, m) => v + term[m]);
      const cw = conv(w, B);
      frozenSums.push(B.map((v, m) => v + cw[m])); weights.push(w.slice());
      term = conv(cont, term).map((v) => -v);
    }
    const gap = (a, b) => { let g = 0; for (let m = 0; m < n; ++m) g = Math.max(g, Math.abs(a[m] - b[m])); return g; };

    let mode = "blip", step = 0, playing = 0;
    const bBlip = document.getElementById("build-blip"), bFrozen = document.getElementById("build-frozen");
    const slider = document.getElementById("build-n"), count = document.getElementById("build-count"), what = document.getElementById("build-what");
    const caption = document.getElementById("build-caption"), play = document.getElementById("build-play");
    const CAPTIONS = {
      blip: "In Chapter 3\u2019s game, player 1\u2019s blip is its spike followed by its own follow-up, and the follow-up is just more small spikes, later "
          + "(the bars below). Each frozen spike moves the state along the frozen life, shifted to when it happens and scaled "
          + "by its size; add them up and you get the blip\u2019s life. The small steps come from spacing the spikes apart; they shrink as the spikes get finer.",
      frozen: "The other way round: a frozen spike is a blip minus blips that cancel its follow-up. Those corrections have "
          + "follow-ups of their own, canceled in the next round, and so on. The leftover shrinks every round, because a "
          + "follow-up only reaches forward in time, so the rounds settle on the frozen life."};

    function drawBuild() {
      const W = Math.max(260, svgC.clientWidth || 800), TOP = 180, BAND = 64, H = TOP + BAND + 26, PADT = 8;
      svgC.setAttribute("viewBox", `0 0 ${W} ${H}`); svgC.setAttribute("height", H); svgC.innerHTML = "";
      const X = (u) => (u / SHOW) * W, lo = -0.12, hi = 1.05;
      const Y = (v) => PADT + ((hi - v) / (hi - lo)) * (TOP - PADT);
      const path = (y) => { let s = ""; for (let k = 0; k < n; ++k) s += (k ? "L" : "M") + X(k * du).toFixed(1) + "," + Y(y[k]).toFixed(1); return s; };
      el("line", { class: "zero", x1: 0, x2: W, y1: Y(0), y2: Y(0) }, svgC);
      const target = mode === "blip" ? B : F, start = mode === "blip" ? F : B;
      const sum = mode === "blip" ? blipSums[step] : frozenSums[step];
      el("path", { class: "curve", d: path(start), stroke: mode === "blip" ? COLORS.frozen : COLORS.blip, opacity: 0.55 }, svgC);   // blip always blue, frozen always gray
      el("path", { class: "target", d: path(target), stroke: mode === "blip" ? COLORS.blip : COLORS.frozen }, svgC);
      if (mode === "blip" && step > 0) {                   // the latest frozen spike's own contribution, faint
        const sp = spikes[step - 1];
        el("path", { class: "piece", d: path(shifted(F, sp.t, sp.mass).map((v) => v)), stroke: "var(--warm)" }, svgC);
      }
      el("path", { class: "sum", d: path(sum), stroke: "var(--ink)" }, svgC);
      // the band below: the spikes (a blip from frozen spikes) or the correction blips' weights (a frozen spike from blips)
      const bz = TOP + BAND / 2 + 6, bs = (BAND / 2 - 4) / 0.8;
      el("line", { class: "axis", x1: 0, x2: W, y1: bz, y2: bz }, svgC);
      el("text", { x: 2, y: TOP + 14 }, svgC).textContent = mode === "blip" ? "player 1\u2019s follow-up, as spikes" : "correction blips, all rounds so far";
      el("line", { class: "comb", x1: 1.5, x2: 1.5, y1: bz, y2: bz - (BAND / 2 - 4) }, svgC);   // the kick (or the first blip)
      if (mode === "blip") {
        spikes.forEach((sp, j) => el("line", { class: "comb", x1: X(sp.t), x2: X(sp.t), y1: bz, y2: bz - bs * sp.rate, opacity: j < step ? 0.95 : 0.15 }, svgC));
      } else {
        const wk = weights[step];
        for (let t = 0.05; t < SHOW; t += DT) {
          const v = wk[Math.round(t / du)];
          if (Math.abs(v) > 1e-4) el("line", { class: "comb", x1: X(t), x2: X(t), y1: bz, y2: bz - bs * v }, svgC);
        }
      }
      for (let k = 0; k <= SHOW; ++k) el("text", { x: X(k), y: H - 4, "text-anchor": k ? (k === SHOW ? "end" : "middle") : "start" }, svgC).textContent = k;
      const box = document.getElementById("build-legend");
      box.innerHTML = mode === "blip"
        ? `<span><i style="background:var(--ink-3)"></i>the spike, held still</span><span><i style="background:var(--accent)"></i>the blip (target)</span><span><i style="background:var(--ink)"></i>the sum so far</span><span>gap to the blip <b>${gap(sum, target).toFixed(3)}</b></span>`
        : `<span><i style="background:var(--accent);opacity:.55"></i>the blip we start from</span><span><i style="background:var(--ink-3)"></i>the frozen spike (target)</span><span><i style="background:var(--ink)"></i>after ${step} round${step === 1 ? "" : "s"}</span><span>gap <b>${gap(sum, target).toFixed(3)}</b></span>`;
      caption.textContent = CAPTIONS[mode];
      count.textContent = step;
    }
    function setMode(m) {
      mode = m; step = 0;
      bBlip.setAttribute("aria-pressed", m === "blip"); bFrozen.setAttribute("aria-pressed", m === "frozen");
      slider.max = m === "blip" ? NSP : ROUNDS; slider.value = 0;
      what.textContent = m === "blip" ? "frozen spikes added" : "rounds of corrections";
      drawBuild();
    }
    bBlip.addEventListener("click", () => setMode("blip"));
    bFrozen.addEventListener("click", () => setMode("frozen"));
    slider.addEventListener("input", () => { clearInterval(playing); playing = 0; step = +slider.value; drawBuild(); });
    play.addEventListener("click", () => {
      clearInterval(playing); step = 0; slider.value = 0; drawBuild();
      const max = +slider.max, dt = mode === "blip" ? 160 : 700;
      playing = setInterval(() => { if (step >= max) { clearInterval(playing); playing = 0; return; } step += 1; slider.value = step; drawBuild(); }, dt);
    });

    let bowlItems = null;
    function drawBowls() {
      const W = Math.max(240, svgB.clientWidth || 480), H = 200, PADT = 10, PADB = 22;
      svgB.setAttribute("viewBox", `0 0 ${W} ${H}`); svgB.setAttribute("height", H); svgB.innerHTML = "";
      const lo = -0.25, hi = 0.8;
      const X = (e) => ((e + EMAX) / (2 * EMAX)) * W, Y = (v) => PADT + ((hi - v) / (hi - lo)) * (H - PADT - PADB);
      const clip = el("clipPath", { id: "lemma-clip" }, el("defs", {}, svgB));
      el("rect", { x: 0, y: PADT, width: W, height: H - PADT - PADB }, clip);
      el("line", { class: "zero", x1: 0, x2: W, y1: Y(0), y2: Y(0) }, svgB);
      el("line", { class: "zero", x1: X(0), x2: X(0), y1: PADT, y2: H - PADB }, svgB);
      el("line", { class: "axis", x1: 0, x2: W, y1: H - PADB, y2: H - PADB }, svgB);
      for (const e of [-EMAX, -EMAX / 2, 0, EMAX / 2, EMAX]) el("text", { x: X(e), y: H - 5, "text-anchor": e === -EMAX ? "start" : e === EMAX ? "end" : "middle" }, svgB).textContent = e.toFixed(1);
      const g = el("g", { "clip-path": "url(#lemma-clip)" }, svgB);
      const J = (k, e) => d.bowls[k].slope * e + d.bowls[k].curvature * e * e;
      for (const k of ["frozen", "blip", "alone"]) {
        let s = "";
        for (let i = 0; i <= 120; ++i) { const e = -EMAX + (2 * EMAX * i) / 120; s += (i ? "L" : "M") + X(e).toFixed(1) + "," + Y(J(k, e)).toFixed(1); }
        el("path", { class: "curve", d: s, stroke: COLORS[k] }, g);
        const t = EMAX * 0.55, sl = d.bowls[k].slope;           // the tangent at zero: its slope is the first-order condition
        el("line", { class: "tangent", x1: X(-t), x2: X(t), y1: Y(-sl * t), y2: Y(sl * t), stroke: COLORS[k] }, g);
      }
      const dots = ["frozen", "blip", "alone"].map((k) => el("circle", { class: "dot", r: 4.5, fill: COLORS[k] }, svgB));
      bowlItems = { dots, J, X, Y };
      const b = legend("lemma-bowls-legend", ["frozen", "blip", "alone"].map((k) => [k, NAMES[k]]));
      bowlItems.labels = b;
      update();
    }

    function update() {
      if (!bowlItems) return;
      const e = (eps.value / 1000) * EMAX;
      ["frozen", "blip", "alone"].forEach((k, i) => {
        const v = bowlItems.J(k, e);
        bowlItems.dots[i].setAttribute("cx", bowlItems.X(e));
        bowlItems.dots[i].setAttribute("cy", bowlItems.Y(Math.min(0.8, Math.max(-0.25, v))));
        bowlItems.dots[i].style.opacity = v > 0.8 ? 0.3 : 1;
        bowlItems.labels[i].textContent = `${fmt(v)} (slope at 0: ${Math.abs(d.bowls[k].slope) < 1e-6 ? "0" : d.bowls[k].slope.toFixed(2)})`;
      });
    }

    eps.addEventListener("input", update);
    const redraw = () => { drawBuild(); drawBowls(); };
    let width = 0;
    new ResizeObserver(() => { const nw = svgC.clientWidth; if (nw !== width) { width = nw; redraw(); } }).observe(svgC);
    setMode("blip");
    redraw();
  }).catch(failed(["build-svg", "lemma-bowls"]));
})();
