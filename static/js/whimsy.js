// Small things, on every page:
// - the theme toggle spreads the new theme from the button like ink, where the browser can (View Transitions);
// - type "flip" anywhere (outside a form field) and a coin is tossed in the corner, with this visit's running count;
// - type "noise" and every heading on the page takes a short random walk, then settles back.
// The colophon lists these, along with the ones on other pages.
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

  // ---- words typed at the page
  let typed = "";
  const tail = (w) => typed.endsWith(w);
  document.addEventListener("keydown", (e) => {
    const t = e.target;
    if (e.metaKey || e.ctrlKey || e.altKey || !e.key || e.key.length !== 1) return;
    if (t && (t.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName))) return;
    typed = (typed + e.key.toLowerCase()).slice(-12);
    if (tail("flip")) { typed = ""; flip(); }
    else if (tail("noise")) { typed = ""; jiggle(); }
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
})();
