// Small things, on every page:
// - the theme toggle brings the new theme in as a mesh, triangles filling from the button (View Transitions);
// - type "flip" anywhere (outside a form field) and a coin is tossed in the corner, with this visit's running count;
// - type "noise" and every heading on the page takes a short random walk, then settles back;
// - type "forecast" and the page tries to guess each next key before you press it, and keeps score;
// - on a phone: shake it to toss the coin, and tap three times on a blank spot for the random walk;
// - leave the tab and its title notes that no new observations are coming in;
// - leave the page alone for a while and something gets doodled in an empty margin (three at most), the page's own
//   kind first: the loop on the dissertation, a pinned walk on the CV, a null over scores by the mesh;
// - a hello in the console, for anyone who opens it.
(function () {
  "use strict";
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // ---- ink for the theme toggle
  const btn = document.querySelector(".btn-dark");
  if (btn && document.startViewTransition && !reduced) {
    let passing = false;
    btn.addEventListener("click", (e) => {
      if (passing) return;
      e.stopImmediatePropagation(); e.preventDefault();
      const r = btn.getBoundingClientRect();
      const frames = meshFrames(r.left + r.width / 2, r.top + r.height / 2);
      document.documentElement.classList.add("inking");
      const t = document.startViewTransition(() => { passing = true; btn.click(); passing = false; });
      t.ready.then(() => document.documentElement.animate({ clipPath: frames },
        { duration: 520, easing: "linear", fill: "both", pseudoElement: "::view-transition-new(root)" })).catch(() => {});
      t.finished.finally(() => document.documentElement.classList.remove("inking"));
    }, true);
  }

  // the new theme arrives as a mesh: right triangles, alternating diagonals like the home page's, each growing
  // from its middle, the ones near the toggle first. One clip-path per frame, every triangle in every frame
  // (a triangle not yet started is a point), so the browser can play them in order.
  function meshFrames(ox, oy) {
    const W = innerWidth, H = innerHeight, s = Math.max(72, W / 16), nx = Math.ceil(W / s), ny = Math.ceil(H / s);
    const tris = [];
    for (let i = 0; i < nx; ++i) for (let j = 0; j < ny; ++j) {
      const x0 = i * s, y0 = j * s, x1 = x0 + s, y1 = y0 + s;
      const pair = (i + j) % 2 ? [[[x0, y0], [x1, y0], [x1, y1]], [[x0, y0], [x1, y1], [x0, y1]]]
        : [[[x0, y1], [x0, y0], [x1, y0]], [[x0, y1], [x1, y0], [x1, y1]]];
      for (const t of pair) tris.push(t);
    }
    const far = Math.hypot(Math.max(ox, W - ox), Math.max(oy, H - oy));
    const jit = (k) => { const x = Math.sin(k * 12.9898) * 43758.5453; return x - Math.floor(x); };
    const info = tris.map((t, k) => {
      const cx = (t[0][0] + t[1][0] + t[2][0]) / 3, cy = (t[0][1] + t[1][1] + t[2][1]) / 3;
      return { t, cx, cy, d: 0.62 * Math.hypot(cx - ox, cy - oy) / far + 0.12 * jit(k) };
    });
    const out = [], N = 16;
    for (let f = 0; f <= N; ++f) {
      const T = f / N;
      let d = "";
      for (const { t, cx, cy, d: at } of info) {
        const q = Math.min(1, Math.max(0, (T - at) / 0.26)), k = q === 1 ? 1.03 : (1 - Math.pow(1 - q, 3)) * 1.03;
        const p = t.map(([x, y]) => `${(cx + (x - cx) * k).toFixed(1)} ${(cy + (y - cy) * k).toFixed(1)}`);
        d += `M${p[0]}L${p[1]}L${p[2]}Z`;
      }
      out.push(`path('${d}')`);
    }
    return out;
  }

  // ---- the coin on a "Flip again" button tosses when pressed
  document.addEventListener("click", (e) => {
    const c = e.target.closest && e.target.closest("button") && e.target.closest("button").querySelector(".mini-coin");
    if (!c || reduced) return;
    c.classList.remove("spin"); void c.offsetWidth; c.classList.add("spin");
  });

  // ---- buttons that copy a command
  document.addEventListener("click", (e) => {
    const b = e.target.closest && e.target.closest("[data-copy]");
    if (!b || !navigator.clipboard) return;
    navigator.clipboard.writeText(b.dataset.copy).then(() => { b.classList.add("done"); setTimeout(() => b.classList.remove("done"), 1600); }, () => {});
  });

  // ---- words typed at the page
  let typed = "";
  const tail = (w) => typed.endsWith(w);
  document.addEventListener("keydown", (e) => {
    const t = e.target;
    if (e.metaKey || e.ctrlKey || e.altKey || !e.key || e.key.length !== 1) return;
    if (t && (t.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName))) return;
    const k = e.key.toLowerCase();
    if (fc) { e.preventDefault(); forecastKey(k); return; }
    typed = (typed + k).slice(-12);
    if (tail("flip")) { typed = ""; flip(); }
    else if (tail("noise")) { typed = ""; jiggle(); }
    else if (tail("forecast")) { typed = ""; forecast(); }
  });

  // ---- a coin, tossed
  let heads = 0, tails = 0, coin = null;
  function flip() {
    if (!coin) {
      coin = document.createElement("div"); coin.className = "whimsy-coin"; coin.setAttribute("aria-live", "polite");
      coin.innerHTML = '<div class="face"><span class="h">H</span><span class="t">T</span></div><p></p>';
      document.body.appendChild(coin);
    }
    const up = Math.random() < 0.5, face = coin.querySelector(".face");
    up ? ++heads : ++tails;
    coin.classList.remove("gone");
    face.style.animation = "none"; void face.offsetWidth;
    face.style.setProperty("--end", up ? "0deg" : "180deg");
    face.style.animation = reduced ? "none" : "";
    face.style.transform = reduced ? `rotateY(${up ? 0 : 180}deg)` : "";
    const n = heads + tails;
    coin.querySelector("p").textContent = `${up ? "Heads" : "Tails"}. ${heads} of ${n} this visit` + (n >= 5 ? `, ${(heads / n).toFixed(2)}` : "") + ".";
    clearTimeout(coin._t); coin._t = setTimeout(() => coin.classList.add("gone"), 4200);
  }

  // ---- the page forecasts your next key: an order-two Markov chain on what you type, falling back to order
  // one, then to letter frequencies of English. It says its guess only after you press the key.
  let fc = null;
  const EN = "etaoinshrdlcumwfgypbvkjxqz";
  function forecast() {
    if (fc) return;
    fc = { box: document.createElement("div"), hist: " ", n: 0, hit: 0, c2: {}, c1: {}, guess: "e", timer: 0 };
    fc.box.className = "whimsy-forecast"; fc.box.setAttribute("aria-live", "polite");
    fc.box.innerHTML = "<p class=\"q\">I have a guess for your next key. Type anything.</p><p class=\"s\"></p>";
    document.body.appendChild(fc.box);
    fc.timer = setTimeout(endForecast, 15000);
  }
  function best(o) { let b = null, m = 0; for (const c in o) if (o[c] > m) { m = o[c]; b = c; } return b; }
  function forecastKey(k) {
    clearTimeout(fc.timer);
    const hit = k === fc.guess; fc.n++; if (hit) fc.hit++;
    const two = fc.hist.slice(-2), one = fc.hist.slice(-1);
    (fc.c2[two] = fc.c2[two] || {})[k] = (fc.c2[two][k] || 0) + 1;
    (fc.c1[one] = fc.c1[one] || {})[k] = (fc.c1[one][k] || 0) + 1;
    fc.hist += k;
    const show = (c) => c === " " ? "space" : c;
    fc.box.querySelector(".q").textContent = hit ? `Called it: ${show(k)}.` : `I guessed ${show(fc.guess)}. You typed ${show(k)}.`;
    fc.box.querySelector(".s").textContent = `${fc.hit} of ${fc.n}`;
    const t2 = fc.hist.slice(-2), t1 = fc.hist.slice(-1);
    fc.guess = best(fc.c2[t2] || {}) || best(fc.c1[t1] || {}) || (t1 === " " ? "t" : EN[fc.n % 5]);
    if (fc.n >= 40) endForecast(); else fc.timer = setTimeout(endForecast, 8000);
  }
  function endForecast() {
    if (!fc) return;
    const { box, hit, n } = fc; fc = null;
    box.querySelector(".q").textContent = n ? `${hit} of ${n} keys forecast. Forecasting the forecasts of others is harder: that took a dissertation.` : "No keys, no forecasts.";
    box.querySelector(".s").textContent = "";
    setTimeout(() => { box.classList.add("gone"); setTimeout(() => box.remove(), 600); }, 4000);
  }

  // ---- on a phone: shake to toss the coin, tap three times on a blank spot for the random walk
  if (window.DeviceMotionEvent && typeof DeviceMotionEvent.requestPermission !== "function") {
    let last = 0, jolts = [];
    addEventListener("devicemotion", (e) => {
      const a = e.acceleration && e.acceleration.x != null ? e.acceleration : null;
      if (!a) return;
      const m = Math.hypot(a.x || 0, a.y || 0, a.z || 0), now = performance.now();
      if (m < 14) return;
      jolts = jolts.filter((t) => now - t < 700); jolts.push(now);
      if (jolts.length >= 3 && now - last > 2000) { last = now; jolts = []; flip(); }
    });
  }
  let taps = [];
  document.addEventListener("pointerup", (e) => {
    if (e.pointerType !== "touch") return;
    if (e.target.closest && e.target.closest("a, button, input, select, textarea, label, canvas, svg, [role=button], .katex, summary")) { taps = []; return; }
    const now = performance.now();
    taps = taps.filter((t) => now - t.t < 650 && Math.hypot(t.x - e.clientX, t.y - e.clientY) < 60);
    taps.push({ t: now, x: e.clientX, y: e.clientY });
    if (taps.length >= 3) { taps = []; jiggle(); }
  }, { passive: true });

  // ---- leave the tab and it notices
  let away = 0, title = document.title;
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) { title = document.title; away = performance.now(); document.title = "No new observations"; }
    else {
      if (document.title !== "No new observations") return;
      if (performance.now() - away < 4000) { document.title = title; return; }
      document.title = "Updating beliefs";
      setTimeout(() => { if (document.title === "Updating beliefs") document.title = title; }, 1400);
    }
  });

  // ---- every heading takes a short random walk
  function jiggle() {
    if (reduced) return;
    const hs = [...document.querySelectorAll("h1, h2, h3")].filter((h) => h.getBoundingClientRect().bottom > 0 && h.getBoundingClientRect().top < innerHeight);
    const walks = hs.map((h) => ({ h, x: 0, y: 0, a: 0, prev: h.style.transform }));
    const t0 = performance.now(), T = 2600;
    (function step(now) {
      const s = (now - t0) / T, damp = s < 0.7 ? 1 : Math.max(0, 1 - (s - 0.7) / 0.3);
      for (const w of walks) {
        w.x = 0.92 * w.x + (Math.random() - 0.5) * 3; w.y = 0.92 * w.y + (Math.random() - 0.5) * 2; w.a = 0.9 * w.a + (Math.random() - 0.5) * 0.6;
        w.h.style.transform = `translate(${(w.x * damp).toFixed(2)}px, ${(w.y * damp).toFixed(2)}px) rotate(${(w.a * damp).toFixed(2)}deg)`;
      }
      if (s < 1) requestAnimationFrame(step); else walks.forEach((w) => { w.h.style.transform = w.prev; });
    })(t0);
  }

  // ---- idle doodles in an empty margin
  const NS = "http://www.w3.org/2000/svg";
  let idle = 0, doodles = 0;
  const poke = () => { clearTimeout(idle); if (doodles < 3) idle = setTimeout(doodle, 50000); };
  ["pointermove", "keydown", "scroll", "touchstart"].forEach((ev) => window.addEventListener(ev, poke, { passive: true }));
  poke();
  function blank(x, y) {   // nothing but page under a 90 × 90 square at (x, y), in viewport coordinates
    for (let i = 0; i <= 2; ++i) for (let j = 0; j <= 2; ++j) {
      const e = document.elementFromPoint(x + i * 45, y + j * 45);
      if (!e || !/^(HTML|BODY|MAIN|SECTION|DIV)$/.test(e.tagName) || (e.textContent || "").trim().length && e.children.length === 0) return false;
      if (e.closest("a, button, canvas, svg, img, p, h1, h2, h3, li, pre, table, figure, input, select, textarea")) return false;
    }
    return true;
  }
  function doodle() {
    if (document.hidden || innerWidth < 1100) return;
    let spot = null;
    for (let k = 0; k < 40 && !spot; ++k) {
      const x = Math.random() < 0.5 ? 10 + Math.random() * 90 : innerWidth - 100 - Math.random() * 90, y = 80 + Math.random() * (innerHeight - 200);
      if (blank(x, y)) spot = [x, y];
    }
    if (!spot) { poke(); return; }
    const svg = document.createElementNS(NS, "svg");
    svg.setAttribute("viewBox", "0 0 90 90"); svg.setAttribute("class", "whimsy-doodle"); svg.setAttribute("aria-hidden", "true");
    Object.assign(svg.style, { left: spot[0] + scrollX + "px", top: spot[1] + scrollY + "px" });
    const path = (d, w = 1.2) => { const p = document.createElementNS(NS, "path"); p.setAttribute("d", d); p.setAttribute("stroke-width", w); svg.appendChild(p); return p; };
    const bell = (x, c = 45, h = 55, w = 250, base = 75) => (base - h * Math.exp(-((x - c) ** 2) / w)).toFixed(1);
    const dot = (x, y, r = 2.2) => { const c = document.createElementNS(NS, "circle"); c.setAttribute("cx", x); c.setAttribute("cy", y); c.setAttribute("r", r); svg.appendChild(c); return c; };
    const DRAW = {
      walk() {               // a random walk
        let x = 45, y = 45, d = `M${x},${y}`;
        for (let i = 0; i < 70; ++i) { x = Math.max(5, Math.min(85, x + (Math.random() - 0.5) * 16)); y = Math.max(5, Math.min(85, y + (Math.random() - 0.5) * 16)); d += ` L${x.toFixed(1)},${y.toFixed(1)}`; }
        path(d);
      },
      tail() {               // a bell curve, and a tail shaded by hand
        let d = "M5,75"; for (let x = 5; x <= 85; x += 2) d += ` L${x},${bell(x)}`;
        path("M5,75 L85,75", 0.8); path(d);
        let h = ""; for (let x = 66; x <= 84; x += 3) h += `M${x},75 L${x},${bell(x)} `; path(h, 0.7);
      },
      bisect() {             // a triangle, bisected a few times
        path("M10,80 L80,80 L10,10 Z M45,45 L10,80 M45,45 L45,80 M27.5,62.5 L45,80 M27.5,27.5 L10,45 M27.5,62.5 L10,45");
      },
      coin() {               // a coin, mid-air
        path("M45,20 m-14,0 a14,14 0 1,0 28,0 a14,14 0 1,0 -28,0"); path("M40,15 L40,25 M50,15 L50,25 M40,20 L50,20", 1.4);
        path("M36,42 Q45,47 54,42 M38,50 Q45,54 52,50", 0.8); path("M20,82 L70,82", 0.8);
      },
      loop() {               // actions, state, observations, beliefs, and back; the chord across
        const P = [[45, 14], [76, 45], [45, 76], [14, 45]];
        P.forEach(([x, y]) => path(`M${x + 5},${y} a5,5 0 1,0 -10,0 a5,5 0 1,0 10,0`, 1));
        path("M52,16 Q72,22 75,38 M74,52 Q70,72 52,75 M38,75 Q20,70 16,52 M16,38 Q20,20 38,15");
        path("M19,41 L70,49", 0.7).setAttribute("stroke-dasharray", "2 3");
      },
      impulse() {            // a shock, and what is left of it later
        path("M10,15 L10,78 L84,78", 0.8);
        let d = "M10,78 L12,24"; for (let x = 12; x <= 84; x += 2) d += ` L${x},${(78 - 54 * Math.exp(-(x - 12) / 14) * Math.cos((x - 12) / 9)).toFixed(1)}`;
        path(d);
      },
      triangle() {           // the causal triangle: a kernel lives below the diagonal
        path("M12,12 L12,80 L80,80 Z", 1);
        let h = ""; for (let k = 18; k < 80; k += 6) h += `M12,${k} L${k},${k} `; path(h, 0.6);
      },
      bridge() {             // a walk pinned at both ends
        const n = 44, w = [0]; for (let i = 1; i <= n; ++i) w.push(w[i - 1] + (Math.random() - 0.5) * 2);
        let d = ""; for (let i = 0; i <= n; ++i) { const b = w[i] - (i / n) * w[n]; d += `${i ? "L" : "M"}${(10 + i * 70 / n).toFixed(1)},${(45 + b * 5).toFixed(1)} `; }
        path(d); dot(10, 45); dot(80, 45);
      },
      discovery() {          // scores, the null over them, and the few that stand out
        const hs = [4, 9, 17, 27, 33, 27, 17, 9, 5, 3, 6, 11];
        let b = ""; hs.forEach((h, i) => { const x = 8 + i * 6.3; b += `M${x},78 L${x},${78 - h} L${x + 5},${78 - h} L${x + 5},78 `; }); path(b, 0.7);
        let d = "M8,78"; for (let x = 8; x <= 84; x += 2) d += ` L${x},${bell(x, 34, 36, 110, 78)}`; path(d, 1.1);
        let h = ""; for (let x = 72; x <= 83; x += 2.5) h += `M${x},78 L${x},${78 - (x > 77 ? 11 : 6)} `; path(h, 0.6);
      },
      shrink() {             // raw estimates, pulled toward the pooled mean
        path("M8,45 L82,45", 0.7).setAttribute("stroke-dasharray", "3 3");
        [[16, 14], [30, 72], [44, 24], [58, 64], [72, 20]].forEach(([x, y]) => {
          const to = 45 + (y - 45) * 0.45; dot(x, y, 2); path(`M${x},${y} L${x},${to.toFixed(1)}`, 0.8); dot(x, to, 1.4);
        });
      },
    };
    const here = location.pathname;
    const own = /\/dissertation\//.test(here) ? ["loop", "impulse", "triangle"] : /\/cv\//.test(here) ? ["bridge"]
      : /\/(gate|mesh)\//.test(here) ? ["discovery", "bisect"] : /\/noisestate\//.test(here) ? ["impulse", "triangle"]
      : /\/writing\//.test(here) ? ["shrink", "tail"] : [];
    const pool = [...own, ...["walk", "tail", "bisect", "coin"].filter((k) => !own.includes(k))];
    DRAW[pool[doodles % pool.length]]();
    document.body.appendChild(svg);
    ++doodles; poke();
  }

  // folding or unfolding moves the margins, so the doodles go
  window.addEventListener("sitefold", () => { document.querySelectorAll(".whimsy-doodle").forEach((d) => d.remove()); doodles = 0; poke(); });

  // ---- the reading-progress line, drawn for the page: a random walk on the dissertation and noise-state pages, a
  // strip of mesh triangles on the Decision Mesh pages. The pages still set the line's width; the drawing is full width underneath and the width uncovers it.
  (function progressLine() {
    const bar = document.getElementById("progress");
    if (!bar) return;
    const mode = /^\/gate(\/|$)/.test(location.pathname) ? "mesh" : "walk";
    bar.classList.add("bar-" + mode);
    const NS = "http://www.w3.org/2000/svg", H = 10;
    const mk = (tag, attrs, parent) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); if (parent) parent.appendChild(e); return e; };
    let seed = 0; for (const c of location.pathname) seed = (Math.imul(seed, 31) + c.charCodeAt(0)) | 0;
    const rng = (a => () => { a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; })(seed);
    const gauss = () => Math.sqrt(-2 * Math.log(1 - rng())) * Math.cos(2 * Math.PI * rng());
    let svg = null, ys = null, tip = null, cells = [], step = 2;
    function draw() {
      const W = Math.max(320, window.innerWidth);
      if (svg) svg.remove();
      svg = mk("svg", { class: "bar-art", width: W, height: H, viewBox: `0 0 ${W} ${H}`, "aria-hidden": "true" }, bar);
      cells = []; tip = null;
      if (mode === "walk") {
        // one random path per page (seeded by its address): Brownian at the scale of a few pixels, pulled gently
        // back to the middle (an Ornstein-Uhlenbeck path) so that it stays in the band
        const n = Math.ceil(W / step) + 1, lo = 1.6, hi = H - 1.6; let y = H / 2;
        ys = [y];
        for (let i = 1; i < n; ++i) { y += -0.04 * (y - H / 2) + 0.75 * gauss(); y = y < lo ? 2 * lo - y : y > hi ? 2 * hi - y : y; ys.push(y); }
        mk("path", { d: "M" + ys.map((y, i) => `${(i * step).toFixed(1)},${y.toFixed(2)}`).join(" L"), class: "bar-ink" }, svg);
        tip = mk("circle", { r: 2.2, class: "bar-dot" }, svg);
      } else if (mode === "mesh") {
        // an irregular strip, like the adaptive mesh: three rows of vertices spaced by a slowly varying density
        // (fine in some stretches, coarse in others), the middle row wandering, stitched into triangles row to row
        const dens = (x) => 0.55 + 0.45 * Math.sin(x / 83 + seed % 7) * Math.sin(x / 29 + 1.3);
        const row = (yfn) => { const pts = [[-4, yfn()]]; let x = -4; while (x < W + 16) { x += (4 + 13 * (0.5 + 0.5 * dens(x))) * (0.7 + 0.6 * rng()); pts.push([x, yfn()]); } return pts; };
        const top = row(() => 0.5 + 2.6 * rng()), mid = row(() => H / 2 + (rng() - 0.5) * 2.4), bot = row(() => H - 0.5 - 2.6 * rng());
        const stitch = (A, B) => {           // triangulate between two rows, always advancing the one that lags
          let i = 0, k = 0;
          while (i < A.length - 1 || k < B.length - 1) {
            const takeA = k >= B.length - 1 || (i < A.length - 1 && A[i + 1][0] <= B[k + 1][0]);
            const tri = takeA ? [A[i], A[i + 1], B[k]] : [A[i], B[k + 1], B[k]];
            if (takeA) ++i; else ++k;
            const p = mk("polygon", { points: tri.map((v) => v[0].toFixed(1) + "," + v[1].toFixed(1)).join(" "), class: "bar-tri" }, svg);
            cells.push({ p, x: Math.max(...tri.map((v) => v[0])) });
          }
        };
        stitch(top, mid); stitch(mid, bot);
      }
      place();
    }
    function place() {
      // the width the page asked for (some pages animate it, so the measured width lags behind)
      const pct = parseFloat(bar.style.width), x = isFinite(pct) ? pct / 100 * document.documentElement.clientWidth : bar.getBoundingClientRect().width;
      if (mode === "mesh") {
        for (const c of cells) { const lag = x - c.x; c.p.classList.toggle("on", lag >= 0); c.p.classList.toggle("fresh", lag >= 0 && lag < 22); }
        return;
      }
      if (!tip || !ys) return;
      const i = Math.min(ys.length - 1, Math.max(0, Math.round(x / step))), y = ys[i];
      tip.setAttribute("cx", Math.max(2.2, x - 2.2).toFixed(1)); tip.setAttribute("cy", y.toFixed(2)); tip.style.opacity = x > 3 ? 1 : 0;
    }
    draw();
    new MutationObserver(place).observe(bar, { attributes: true, attributeFilter: ["style"] });
    let rt = 0; window.addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(draw, 150); });
  })();

  // ---- hello, console
  try { console.log("%cHello. Everything on this site is computed in your browser.\nThe game solver is also a Python package: pip install noisestate", "font: 13px Georgia, serif; color: #1f3fd0"); } catch (e) {}
})();
