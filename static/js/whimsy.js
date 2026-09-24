// Small things, on every page:
// - the theme toggle spreads the new theme from the button like ink, where the browser can (View Transitions);
// - type "flip" anywhere (outside a form field) and a coin is tossed in the corner, with this visit's running count;
// - type "noise" and every heading on the page takes a short random walk, then settles back;
// - type "forecast" and the page tries to guess each next key before you press it, and keeps score;
// - on a phone: shake it to toss the coin, and tap three times on a blank spot for the random walk;
// - leave the tab and its title notes that no new observations are coming in;
// - leave the page alone for a while and something gets doodled in an empty margin (three at most);
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
      const r = btn.getBoundingClientRect(), root = document.documentElement.style;
      root.setProperty("--ink-x", r.left + r.width / 2 + "px"); root.setProperty("--ink-y", r.top + r.height / 2 + "px");
      document.documentElement.classList.add("inking");
      const t = document.startViewTransition(() => { passing = true; btn.click(); passing = false; });
      t.finished.finally(() => document.documentElement.classList.remove("inking"));
    }, true);
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
    const kind = doodles % 4;
    if (kind === 0) {        // a random walk
      let x = 45, y = 45, d = `M${x},${y}`;
      for (let i = 0; i < 70; ++i) { x = Math.max(5, Math.min(85, x + (Math.random() - 0.5) * 16)); y = Math.max(5, Math.min(85, y + (Math.random() - 0.5) * 16)); d += ` L${x.toFixed(1)},${y.toFixed(1)}`; }
      path(d);
    } else if (kind === 1) { // a bell curve, and a tail shaded by hand
      let d = "M5,75"; for (let x = 5; x <= 85; x += 2) d += ` L${x},${(75 - 55 * Math.exp(-((x - 45) ** 2) / 250)).toFixed(1)}`;
      path("M5,75 L85,75", 0.8); path(d);
      let h = ""; for (let x = 66; x <= 84; x += 3) h += `M${x},75 L${x},${(75 - 55 * Math.exp(-((x - 45) ** 2) / 250)).toFixed(1)} `; path(h, 0.7);
    } else if (kind === 2) { // a triangle, bisected a few times
      path("M10,80 L80,80 L10,10 Z M45,45 L10,80 M45,45 L45,80 M27.5,62.5 L45,80 M27.5,27.5 L10,45 M27.5,62.5 L10,45");
    } else {                 // a coin, mid-air
      path("M45,20 m-14,0 a14,14 0 1,0 28,0 a14,14 0 1,0 -28,0"); path("M40,15 L40,25 M50,15 L50,25 M40,20 L50,20", 1.4);
      path("M36,42 Q45,47 54,42 M38,50 Q45,54 52,50", 0.8); path("M20,82 L70,82", 0.8);
    }
    document.body.appendChild(svg);
    ++doodles; poke();
  }

  // folding or unfolding moves the margins, so the doodles go
  window.addEventListener("sitefold", () => { document.querySelectorAll(".whimsy-doodle").forEach((d) => d.remove()); doodles = 0; poke(); });

  // ---- hello, console
  try { console.log("%cHello. Everything on this site is computed in your browser.\nThe game solver is also a Python package: pip install noisestate", "font: 13px Georgia, serif; color: #1f3fd0"); } catch (e) {}
})();
