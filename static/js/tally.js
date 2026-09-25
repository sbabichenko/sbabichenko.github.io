// A running count of what this visit has computed: games solved, meshes fitted, meshes grown, and a note of the
// last thing, like a line in a lab notebook. The demos call window.siteTally("solve" | "fit" | "grow", n, note);
// the count lives in sessionStorage, so it follows you between pages and is gone when the tab closes. Nothing
// leaves the browser.
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
    const last = t.last ? ` Last: ${t.last.replace(/[<&]/g, "")}.` : "";
    el.innerHTML = parts.length ? `This visit, on your machine: ${parts.join(", ")}.${last}` : "Nothing computed yet this visit.";
  }
  window.siteTally = function (kind, n, note) {
    const t = read(); t[kind] = (t[kind] || 0) + (n || 1); if (note) t.last = note; write(t); render();
  };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", render); else render();
})();
