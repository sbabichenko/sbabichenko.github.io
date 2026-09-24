// A running count of what this visit has computed: games solved, meshes fitted, meshes grown. The
// demos call window.siteTally("solve" | "fit" | "grow"); the count lives in sessionStorage, so it
// follows you between pages and is gone when the tab closes. Nothing leaves the browser.
(function () {
  "use strict";
  const KEY = "sb-tally";
  const read = () => { try { return JSON.parse(sessionStorage.getItem(KEY)) || {}; } catch (e) { return {}; } };
  const write = (t) => { try { sessionStorage.setItem(KEY, JSON.stringify(t)); } catch (e) {} };
  const words = { solve: ["equilibrium solved", "equilibria solved"], fit: ["mesh fitted", "meshes fitted"], grow: ["mesh grown", "meshes grown"] };
  function render() {
    const el = document.getElementById("tally");
    if (!el) return;
    const t = read(), parts = [];
    for (const k of ["solve", "fit", "grow"]) if (t[k]) parts.push(`<b>${t[k].toLocaleString()}</b> ${words[k][t[k] === 1 ? 0 : 1]}`);
    el.innerHTML = parts.length ? `This visit, on your machine: ${parts.join(", ")}.` : "Nothing computed yet this visit. Your machine is idle, for now.";
  }
  window.siteTally = function (kind, n) {
    const t = read(); t[kind] = (t[kind] || 0) + (n || 1); write(t); render();
  };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", render); else render();
})();
