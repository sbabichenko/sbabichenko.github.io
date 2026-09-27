// Foldable phones (Galaxy Z Fold, iPhone Duo and the like).
//
// Layout: where the browser reports two screen segments (the Viewport Segments API) the page gets a class and
// the hinge's position as CSS variables, so a layout can put one thing on each side of the fold:
//   html.fold-book   two segments side by side (the phone half-open like a book): --hinge-x, --hinge-w
//   html.fold-table  two segments one above the other (half-open like a laptop):   --hinge-y, --hinge-h
//   html.posture-folded  the Device Posture API says the phone is half-open
// ?fold=book or ?fold=table in the address fakes either, for checking layouts on an ordinary screen.
//
// Story pages read with window.readLine(): the height in the window that picks the active step. Normally 55%
// of the way down; when the drawing sits across the top of a narrow screen, the middle of what is left below
// it; on a laptop-shaped fold, the middle of the lower half.
//
// And a little fun: folding or unfolding the phone gives the home page's mesh a ridge along the fold to chase,
// and the footer keeps count.
(function () {
  "use strict";
  const root = document.documentElement;
  const fake = new URLSearchParams(location.search).get("fold");

  function segments() {
    const W = innerWidth, H = innerHeight;
    if (fake === "book") return [{ x: 0, y: 0, width: W / 2 - 10, height: H }, { x: W / 2 + 10, y: 0, width: W / 2 - 10, height: H }];
    if (fake === "table") return [{ x: 0, y: 0, width: W, height: H / 2 - 10 }, { x: 0, y: H / 2 + 10, width: W, height: H / 2 - 10 }];
    const v = window.viewport;
    if (v && v.segments && v.segments.length > 1) return [...v.segments];
    return null;
  }
  function apply() {
    const s = segments(), st = root.style;
    root.classList.remove("fold-book", "fold-table");
    if (s && s.length === 2) {
      const [a, b] = s;
      if (Math.abs(a.y - b.y) < 2) {
        root.classList.add("fold-book");
        st.setProperty("--hinge-x", a.x + a.width + "px"); st.setProperty("--hinge-w", Math.max(0, b.x - a.x - a.width) + "px");
        st.setProperty("--page-l", a.width + "px"); st.setProperty("--page-r", b.width + "px");
      } else {
        root.classList.add("fold-table");
        st.setProperty("--hinge-y", a.y + a.height + "px"); st.setProperty("--hinge-h", Math.max(0, b.y - a.y - a.height) + "px");
        st.setProperty("--page-t", a.height + "px");
      }
    }
    const posture = navigator.devicePosture && navigator.devicePosture.type;
    root.classList.toggle("posture-folded", posture === "folded" || fake === "table" || fake === "book");
  }
  apply();
  if (navigator.devicePosture && navigator.devicePosture.addEventListener) navigator.devicePosture.addEventListener("change", apply);

  window.readLine = function () {
    const vh = innerHeight;
    if (root.classList.contains("fold-table")) {
      const y = parseFloat(getComputedStyle(root).getPropertyValue("--hinge-y")) || vh / 2;
      return y + (vh - y) / 2;
    }
    const st = document.querySelector(".story .stage");
    if (st) {
      const r = st.getBoundingClientRect();
      if (r.width > innerWidth * 0.8 && r.bottom > 0 && r.bottom < vh * 0.8) return r.bottom + (vh - r.bottom) * 0.45;
    }
    return vh * 0.55;
  };

  // ---- folding and unfolding: a big change in the window's area that is not a rotation
  let last = { w: innerWidth, h: innerHeight }, timer = 0;
  addEventListener("resize", () => {
    apply();
    clearTimeout(timer);
    timer = setTimeout(() => {
      const w = innerWidth, h = innerHeight, ratio = (w * h) / (last.w * last.h);
      const rotated = Math.abs(w - last.h) < 40 && Math.abs(h - last.w) < 40;
      if (!rotated && (ratio > 1.45 || ratio < 0.69) && Math.min(w, last.w) < 1100) {
        const kind = ratio > 1 ? "unfold" : "fold";
        count(kind);
        window.dispatchEvent(new CustomEvent("sitefold", { detail: { kind } }));
      }
      last = { w, h };
    }, 250);
  });

  const FK = "sb-folds";
  const readFolds = () => { try { return JSON.parse(sessionStorage.getItem(FK) || "{}"); } catch (e) { return {}; } };
  function renderFolds() {
    const el = document.getElementById("foldtally"), n = readFolds().unfold || 0;
    if (el) el.textContent = n ? ` Unfolded ${n === 1 ? "once" : n === 2 ? "twice" : n + " times"}.` : "";
  }
  function count(kind) {
    const t = readFolds(); t[kind] = (t[kind] || 0) + 1;
    try { sessionStorage.setItem(FK, JSON.stringify(t)); } catch (e) {}
    renderFolds();
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", renderFolds); else renderFolds();

})();
