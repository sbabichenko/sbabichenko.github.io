// /dissertation/spike: one sudden move, followed through each chapter's model. The curves are precomputed deviation
// responses (tools/spike/compute_spike_lives.py -> spike/lives.json); this file only draws them, on one clock: the
// spike's age, the time since it happened.
//
// One spike (the default): a single spike, aged by the slider, by Play, or by dragging across any chart. Each row's
// milestones (read off its own curves by the script) light up as the age passes them; a milestone the game ends
// before is struck out. A finite game's response depends on when in the game the spike comes, which the second
// slider sets; a stationary game's does not.
// Many spikes: spikes keep arriving at random, in both directions, and every row lives through the same spike at
// once; older lives fade. A tap on a chart sends one.
(function () {
  "use strict";
  const { el, mulberry32 } = Sketch;   // static/js/sketch.js
  const host = document.getElementById("strips");
  const me = document.currentScript;
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
  if (!host || !me) return;
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const CHAPTER = { ch1: "Chapter 1", delay: "Chapter 2", ch3: "Chapter 3", ch4: "Chapter 4", ch4b: "Chapter 4", ch5: "Chapter 5", ch6: "Chapter 6" };
  const COLORS = ["var(--accent)", "var(--warm)", "var(--ink-3)"];
  const SPEED = 0.5;                  // age, in time units, per second
  const LINGER = 2.4;                 // seconds a finished life takes to fade (many spikes)
  const GAP = 2.6;                    // mean seconds between spikes (many spikes)
  const LEVEL = 13;                   // line height of the milestone labels

  const fmt = (v) => (Math.abs(v) < 0.005 ? "0.00" : (v < 0 ? "−" : "") + Math.abs(v).toFixed(2));
  const rng = mulberry32((Date.now() & 0xffff) ^ 0x5eed);
  let SHOW = 3;

  // one life of a row: its curves for a spike, where it ends, and its milestones
  function lifeOf(row, spike) {
    if (row.kind === "stationary") {
      const v = row.variants ? row.variants[row.vi || 0] : row;                // a row with two kinds of spike
      return { y: v.y, perm: v.perm || [], end: (v.y[0].length - 1) * row.du, s: null, marks: v.marks };
    }
    const sp = row.spikes[Math.min(row.spikes.length - 1, Math.floor(spike.when * row.spikes.length))];
    return { y: sp.y, perm: [], end: (sp.y[0].length - 1) * row.du, s: sp.s, marks: [...row.marks, ...sp.marks].sort((a, b) => a.at - b.at) };
  }
  const valueAt = (y, du, age) => { const k = Math.min(y.length - 1, Math.max(0, age / du)), i = Math.floor(k), f = k - i; return i + 1 < y.length ? y[i] + f * (y[i + 1] - y[i]) : y[i]; };

  function strip(row, handlers) {
    const art = document.createElement("article");
    art.className = "strip";
    art.innerHTML = `<div class="words"><h2><small>${CHAPTER[row.key] || ""}</small>${row.title}</h2><p>${row.story}</p>${row.model ? `<p class="model"><b>The model.</b> ${row.model}</p>` : ""}</div>
      <div class="plot">${row.variants ? `<div class="variants" role="group" aria-label="Which spike"><span>Trader 1&rsquo;s order</span>${row.variants.map((v, i) => `<button type="button" data-i="${i}" aria-pressed="${!i}">${v.name}</button>`).join("")}</div>` : ""}<svg role="img" aria-label="${row.title}: ${row.labels.join(", ")} after a spike"></svg><div class="legend"></div></div>`;
    host.appendChild(art);
    const svg = art.querySelector("svg"), legend = art.querySelector(".legend");
    art.querySelectorAll(".variants button").forEach((b) => b.addEventListener("click", () => {
      row.vi = +b.dataset.i;
      art.querySelectorAll(".variants button").forEach((o) => o.setAttribute("aria-pressed", o === b));
      handlers.redraw();
    }));

    // the chart is also a slider: press and drag to age the spike (or, with many spikes, a tap sends one)
    const ageFrom = (ev) => { const r = svg.getBoundingClientRect(), x = ((ev.clientX - r.left) / r.width) * W; return Math.max(0, Math.min(SHOW, ((x - PRE()) / (W - PRE())) * SHOW)); };
    svg.addEventListener("pointerdown", (ev) => { handlers.press(ageFrom(ev)); if (handlers.dragging()) svg.setPointerCapture(ev.pointerId); });
    svg.addEventListener("pointermove", (ev) => { if (svg.hasPointerCapture(ev.pointerId)) handlers.drag(ageFrom(ev)); });

    // vertical range: symmetric with many spikes (both signs); with one positive spike, only as low as the curves go
    const all = (row.kind === "finite" ? row.spikes.map((sp) => sp.y) : (row.variants || [row]).flatMap((v) => [v.y, (v.perm || []).filter(Boolean)])).flat(2);
    const top = 1.08 * Math.max(...all.map(Math.abs)), lowest = Math.min(0, ...all);
    let bottom = -top;

    const hasPerm = (row.variants || [row]).some((v) => (v.perm || []).some(Boolean));
    const items = row.labels.map((lab, j) => {
      const d = document.createElement("span");
      d.innerHTML = `<i style="background:${COLORS[j]}"></i>${lab} <b>0.00</b><em></em>`;
      legend.appendChild(d);
      return { b: d.querySelector("b"), em: d.querySelector("em") };
    });
    if (hasPerm) {
      const d = document.createElement("span");
      d.className = "key";
      d.innerHTML = `<i class="dash"></i>the market maker&rsquo;s part alone (trader 2 switched off)`;
      legend.appendChild(d);
    }
    const when = document.createElement("span");
    when.className = "when";
    if (row.kind === "finite") legend.appendChild(when);

    const hasMarks = row.marks.length || (row.spikes || []).some((sp) => sp.marks.length);
    let W = 0, H = 0, BAND = 0, layer = null;
    const PADB = 20;
    const PRE = () => Math.min(28, 0.05 * W);                  // a short stretch before the spike, so a jump shows as one
    const X = (u) => PRE() + (u / SHOW) * (W - PRE());
    const Y = (v) => BAND + 6 + ((top - v) / (top - bottom)) * (H - BAND - 6 - PADB);
    function frame(bothSigns) {
      bottom = bothSigns ? -top : Math.min(-0.12 * top, 1.08 * lowest);
      W = Math.max(200, svg.clientWidth || 600);
      BAND = hasMarks ? LEVEL * (W < 560 ? 3 : 2) + 4 : 0;
      H = (W < 480 ? 124 : 146) + BAND;
      svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
      svg.setAttribute("height", H);
      svg.innerHTML = "";
      el("line", { class: "zero", x1: 0, x2: W, y1: Y(0), y2: Y(0) }, svg);
      el("line", { class: "axis", x1: 0, x2: W, y1: H - PADB, y2: H - PADB }, svg);
      for (let k = 0; k <= SHOW; ++k) el("text", { x: X(k), y: H - 5, "text-anchor": k ? (k === SHOW ? "end" : "middle") : "start" }, svg).textContent = k;
      layer = el("g", {}, svg);
    }

    // milestones: a dashed line down the chart and a label in the band above, placed on the first level with room;
    // lit once the age has passed them, struck out if the game ends first
    function milestones(g, life, age) {
      const levels = Math.max(1, Math.round((BAND - 4) / LEVEL)), used = Array(levels).fill(-Infinity);
      for (const m of life.marks) {
        const x = X(m.at), never = life.s !== null && m.at > life.end + 1e-9, lit = !never && age >= m.at - 1e-9;
        const w = 6.1 * m.label.length + 6, anchorEnd = x + w > W, x0 = anchorEnd ? x - w : x;
        let lv = used.findIndex((r) => r < x0 - 4);
        if (lv < 0) lv = levels - 1;
        used[lv] = x0 + w;
        const cls = never ? "never" : lit ? "lit" : "", y = LEVEL * (lv + 1) - 2;
        el("line", { class: "mark " + cls, x1: x, x2: x, y1: y + 3, y2: H - PADB }, g);
        el("text", { class: "marklab " + cls, x: anchorEnd ? x - 3 : x + 3, y, "text-anchor": anchorEnd ? "end" : "start" }, g).textContent = m.label;
      }
    }

    // the lives: each { spike, age, alpha }, the newest last; with a cursor (one spike) the age line and the rest of
    // the life ahead, faint
    function draw(lives, cursor) {
      layer.innerHTML = "";
      lives.forEach((L, n) => {
        const life = lifeOf(row, L.spike), newest = n === lives.length - 1;
        const a = L.spike.sign * L.spike.size, age = Math.min(L.age, life.end);
        const lg = el("g", {}, layer);
        lg.style.opacity = L.alpha * (newest ? 1 : 0.45);
        if (life.s !== null && life.end < SHOW) {                  // this game ends before the page's clock does
          el("rect", { class: newest ? "wall" : "wall old", x: X(life.end), y: BAND + 2, width: W - X(life.end), height: H - BAND - 2 - PADB }, lg);
          if (newest && (cursor || L.age >= life.end - 1e-9)) el("text", { class: "walllab", x: X(life.end) + 6, y: H - PADB - 6 }, lg).textContent = "the game is over";
        }
        if (newest) milestones(lg, life, L.age);
        if (L.age < 0.18) {                                      // the spike itself: a flash at age 0
          const h = Math.max(10, (H - BAND - PADB) * 0.45 * L.spike.size);
          el("line", { class: "bolt", x1: X(0), x2: X(0), y1: Y(0), y2: Y(0) - L.spike.sign * h }, lg).style.opacity = 1 - L.age / 0.18;
        }
        if (newest && cursor) el("line", { class: "cursor", x1: X(L.age), x2: X(L.age), y1: BAND + 2, y2: H - PADB }, lg);
        life.y.forEach((c, j) => {
          const kmax = Math.floor(age / row.du + 1e-9), v = a * valueAt(c, row.du, age);
          let d = "";
          d = "M0," + Y(0).toFixed(1) + "L" + X(0).toFixed(1) + "," + Y(0).toFixed(1);   // zero before the spike, then the jump
          for (let k = 0; k <= kmax; ++k) d += "L" + X(k * row.du).toFixed(1) + "," + Y(a * c[k]).toFixed(1);
          d += "L" + X(age).toFixed(1) + "," + Y(v).toFixed(1);
          if (newest && cursor) {
            let rest = "M" + X(age).toFixed(1) + "," + Y(v).toFixed(1);
            for (let k = kmax + 1; k < c.length; ++k) rest += "L" + X(k * row.du).toFixed(1) + "," + Y(a * c[k]).toFixed(1);
            el("path", { class: "ahead", d: rest, stroke: COLORS[j] }, lg);
          }
          if (life.perm[j]) {                                    // the permanent part, dashed, in the same color
            let pd = "";
            for (let k = 0; k <= kmax; ++k) pd += (k ? "L" : "M") + X(k * row.du).toFixed(1) + "," + Y(a * life.perm[j][k]).toFixed(1);
            el("path", { class: "perm", d: pd, stroke: COLORS[j] }, lg);
          }
          el("path", { class: "curve" + (newest ? "" : " old"), d, stroke: COLORS[j] }, lg);
          if (newest) {
            el("circle", { class: "dot", r: 4, cx: X(age), cy: Y(v), fill: COLORS[j] }, lg);
            items[j].b.textContent = fmt(v);
            if (row.key === "ch6") el("text", { class: "jumplab", x: X(0) + 6, y: Y(a * c[0]) + (a * c[0] >= 0 ? -4 : 12), style: `fill: ${COLORS[j]}` }, lg).textContent = `jumps to ${fmt(a * c[0])}`;   // the size of the block is the point
            items[j].em.textContent = life.perm[j] ? ` (market maker alone ${fmt(a * valueAt(life.perm[j], row.du, age))})` : "";
          }
        });
        if (newest && life.s !== null) when.textContent = `this spike came at t = ${life.s.toFixed(2)}, in a game that ends at t = ${row.T}`;
      });
    }
    return { draw, frame };
  }

  // "When someone is watching": the privy trader's order rate after the market maker's quote spike, expecting the
  // market maker to play on (blip, blue) or to hold still (frozen, gray), from the same equilibrium
  function watchPanel(w) {
    const svg = document.getElementById("watch");
    if (!svg) return;
    const box = document.getElementById("watch-legend");
    box.innerHTML = `<span><i style="background:${COLORS[0]}"></i>expects it to play on (blip) <b></b></span><span><i style="background:var(--ink-3)"></i>expects it to hold the quote (frozen) <b></b></span>`;
    const bs = box.querySelectorAll("b");
    function draw() {
      const H = 200, W = Math.max(240, svg.clientWidth || 600), PADB = 20, PADT = 12, PRE = Math.min(28, 0.05 * W);
      svg.setAttribute("viewBox", `0 0 ${W} ${H}`); svg.setAttribute("height", H); svg.innerHTML = "";
      const all = [...w.blip.D, ...w.frozen.D], lo = Math.min(...all) * 1.08, hi = -0.14 * lo;   // headroom so buying back shows above zero
      const X = (u) => PRE + (u / SHOW) * (W - PRE), Y = (v) => PADT + ((hi - v) / (hi - lo)) * (H - PADT - PADB);
      el("line", { class: "zero", x1: 0, x2: W, y1: Y(0), y2: Y(0) }, svg);
      el("line", { class: "axis", x1: 0, x2: W, y1: H - PADB, y2: H - PADB }, svg);
      for (let k = 0; k <= SHOW; ++k) el("text", { x: X(k), y: H - 5, "text-anchor": k ? (k === SHOW ? "end" : "middle") : "start" }, svg).textContent = k;
      const p = (ys) => "M0," + Y(0).toFixed(1) + "L" + X(0).toFixed(1) + "," + Y(0).toFixed(1) + ys.map((v, k) => "L" + X(w.ages[k]).toFixed(1) + "," + Y(v).toFixed(1)).join("");
      el("path", { class: "curve", d: p(w.frozen.D), stroke: "var(--ink-3)" }, svg);
      el("path", { class: "curve", d: p(w.blip.D), stroke: COLORS[0] }, svg);
      el("text", { x: X(0) + 6, y: H - PADB - 8 }, svg).textContent = "the trader\u2019s order rate after its block";
      const n = w.ages.length - 1;
      bs[0].textContent = `${fmt(w.blip.D[n])} at age ${SHOW}; inventory ${fmt(w.blip.Q[n])}`;
      bs[1].textContent = `${fmt(w.frozen.D[n])} at age ${SHOW}; inventory ${fmt(w.frozen.Q[n])}`;
    }
    new ResizeObserver(draw).observe(svg);
    draw();
  }

  getJSON(me.dataset.lives).then((data) => {
    SHOW = data.show;
    if (data.watch) watchPanel(data.watch);
    const $ = (id) => document.getElementById(id);
    const bOne = $("mode-one"), bMany = $("mode-many"), oneControls = $("one-controls"), manyNote = $("many-note");
    const scrub = $("scrub"), whenIn = $("when"), play = $("play"), clock = $("clock");

    let mode = "one", seen = false, visible = false, raf = 0, last = 0;
    let age = 0, sweeping = false;                  // one spike
    let spikes = [], tsec = 0, next = 0;            // many spikes: { spike, born (seconds) }
    const newSpike = () => ({ sign: rng() < 0.5 ? -1 : 1, size: 0.5 + 0.5 * rng(), when: rng() });
    const send = () => { spikes.push({ spike: newSpike(), born: tsec }); next = tsec + GAP * (0.6 + rng()); wake(); };
    const setPlay = (on) => { sweeping = on; play.innerHTML = on ? "&#10074;&#10074; Pause" : "&#9654; Play"; };
    const handlers = {
      redraw: () => (mode === "one" ? drawOne() : drawMany()),
      press: (a) => { if (mode === "many") send(); else { setPlay(false); setAge(a); } },
      drag: (a) => { if (mode === "one") setAge(a); },
      dragging: () => mode === "one",
    };
    const rows = data.lives.map((row) => strip(row, handlers));

    const oneSpike = () => ({ sign: 1, size: 1, when: whenIn.value / 1000 });
    function drawOne() {
      clock.textContent = age.toFixed(2);
      scrub.value = Math.round((age / SHOW) * 1000);
      rows.forEach((r) => r.draw([{ spike: oneSpike(), age, alpha: 1 }], true));
    }
    function setAge(a) { age = a; drawOne(); }
    // a life that ends early (a finite game running out) holds at its end, so every row keeps the same spike on
    // screen for the same time, and all fade together once the page's clock has run out
    function drawMany() {
      spikes = spikes.filter((s) => tsec - s.born < SHOW / SPEED + LINGER);
      const lives = spikes.map((s) => {
        const done = tsec - s.born - SHOW / SPEED;
        return { spike: s.spike, age: (tsec - s.born) * SPEED, alpha: done > 0 ? Math.max(0, 1 - done / LINGER) : 1 };
      }).filter((L) => L.alpha > 0.01);
      rows.forEach((r) => r.draw(lives, false));
    }
    function tick(ts) {
      raf = 0;
      if (!visible) { last = 0; return; }
      const dt = last ? Math.min(0.1, (ts - last) / 1000) : 0;
      last = ts;
      if (mode === "many") {
        tsec += dt;
        if (tsec >= next) send();
        drawMany();
      } else if (sweeping) {
        setAge(Math.min(SHOW, age + dt * SPEED));
        if (age >= SHOW) setPlay(false);
      }
      if (visible && (mode === "many" || sweeping)) raf = requestAnimationFrame(tick); else last = 0;
    }
    function wake() { if (!raf && visible && (mode === "many" || sweeping)) raf = requestAnimationFrame(tick); }

    function setMode(m) {
      mode = m;
      bOne.setAttribute("aria-pressed", m === "one");
      bMany.setAttribute("aria-pressed", m === "many");
      oneControls.hidden = m !== "one";
      manyNote.hidden = m !== "many";
      $("howto").hidden = m !== "one";
      rows.forEach((r) => r.frame(m === "many"));
      if (m === "many") { setPlay(false); spikes = []; tsec = 0; next = 0.2; wake(); }
      else drawOne();
    }
    bOne.addEventListener("click", () => setMode("one"));
    bMany.addEventListener("click", () => setMode("many"));
    play.addEventListener("click", () => {
      if (sweeping) return setPlay(false);
      if (age >= SHOW - 1e-9) setAge(0);
      setPlay(true); wake();
    });
    scrub.addEventListener("input", () => { setPlay(false); setAge((scrub.value / 1000) * SHOW); });
    whenIn.addEventListener("input", drawOne);

    // redraw the axes at a new width; animate only while the rows are on screen and the tab is showing
    let width = host.clientWidth;
    new ResizeObserver(() => { if (host.clientWidth === width) return; width = host.clientWidth; rows.forEach((r) => r.frame(mode === "many")); mode === "one" ? drawOne() : drawMany(); }).observe(host);
    let first = true;
    function syncVisibility() {
      visible = seen && !document.hidden;
      last = 0;
      if (visible && first) { first = false; if (!reduced && mode === "one") { setAge(0); setPlay(true); } }
      if (!visible) { cancelAnimationFrame(raf); raf = 0; }
      else wake();
    }
    document.addEventListener("visibilitychange", syncVisibility);
    // the spike ages once by itself the first time the rows come into view (shown fully aged, and still, if motion
    // is reduced)
    new IntersectionObserver((es) => {
      seen = es.some((e) => e.isIntersecting);
      syncVisibility();
    }).observe(host);
    setMode("one");
    setAge(reduced ? SHOW : 0);
  }).catch(failed(["strips"]));
})();
