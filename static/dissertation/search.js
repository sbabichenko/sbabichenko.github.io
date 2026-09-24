// Search across the whole dissertation. The index (search.json, written by tools/dissertation/search_index.py)
// holds every heading, theorem, numbered equation and paragraph with the anchor to jump to; it is fetched the
// first time the box is used. Every word typed must appear; headings and theorems rank first. "/" focuses the box.
// On arrival, the words searched for are marked in the passage jumped to, for a few seconds.
(function () {
  "use strict";
  const input = document.getElementById("dsearch");
  const BASE = ((input && input.dataset.base) || "/dissertation/").replace(/\/?$/, "/");
  let index = null, loading = null, panel = null, hits = [], pick = -1;

  const esc = (s) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
  const norm = (s) => s.toLowerCase().normalize("NFKD").replace(/[̀-ͯ]/g, "").replace(/[’‘]/g, "'").replace(/[–—]/g, "-");
  function load() {
    if (index) return Promise.resolve(index);
    if (!loading) loading = fetch(BASE + "search.json").then((r) => r.json()).then((d) => {
      d.entries.forEach((e) => { e.n = norm((e.l || "") + " " + (e.x || "")); });
      return (index = d);
    });
    return loading;
  }
  function search(q) {
    const words = norm(q).split(/\s+/).filter(Boolean);
    if (!words.length) return [];
    const phrase = words.join(" "), out = [];
    for (const e of index.entries) {
      if (!words.every((w) => e.n.includes(w))) continue;
      // an equation is found by its number; its surrounding words already belong to the paragraph's own entry
      if (e.k === "e" && !words.some((w) => /\d/.test(w) && norm(e.l).includes(w))) continue;
      let score = { h: 40, t: 30, e: 20, p: 0 }[e.k] || 0;
      if (e.l) { const l = norm(e.l); if (l === phrase || l.endsWith(" " + phrase) || l.startsWith(phrase + " ")) score += 200; else if (l.includes(phrase)) score += 60; }
      if (e.n.includes(phrase)) score += 15;
      for (const w of words) score += Math.min(5, e.n.split(w).length - 1);
      out.push([score, e]);
    }
    return out.sort((a, b) => b[0] - a[0]).slice(0, 14).map((x) => x[1]);
  }
  function snippet(text, words) {
    const n = norm(text);
    let at = -1; for (const w of words) { const i = n.indexOf(w); if (i >= 0 && (at < 0 || i < at)) at = i; }
    const from = Math.max(0, at - 60), s = (from ? "…" : "") + text.slice(from, from + 190) + (from + 190 < text.length ? "…" : "");
    let h = esc(s);
    for (const w of words) if (w.length > 1) h = h.replace(new RegExp("(" + w.replace(/[.*+?^${}()|[\]\\]/g, "\\$&") + ")", "gi"), "<mark>$1</mark>");
    return h;
  }
  const hrefOf = (e) => BASE + e.s + "/" + (e.a ? "#" + encodeURIComponent(e.a).replace(/%3A/g, ":") : "");
  function place() {
    const r = input.getBoundingClientRect(), w = Math.min(520, innerWidth - 24);
    Object.assign(panel.style, { top: r.bottom + 6 + "px", left: Math.max(12, Math.min(r.left, innerWidth - w - 12)) + "px", width: w + "px" });
  }
  function show() {
    if (!panel) { panel = document.createElement("div"); panel.className = "dsearch-panel"; panel.setAttribute("role", "listbox"); document.body.appendChild(panel); }
    const q = input.value.trim(), words = norm(q).split(/\s+/).filter(Boolean);
    if (!q) { panel.hidden = true; return; }
    hits = search(q); pick = hits.length ? 0 : -1;
    panel.innerHTML = hits.length ? hits.map((e, i) => `<a class="hit${i === pick ? " on" : ""}" role="option" href="${hrefOf(e)}" data-i="${i}">
        <span class="where">${esc(index.pages[e.s] || e.s)}${e.l ? " · " + esc(e.l) : ""}</span>${e.x ? `<span class="what">${snippet(e.x, words)}</span>` : ""}</a>`).join("")
      : `<p class="none">Nothing matches every word of that. No one knows much.</p>`;
    panel.hidden = false; place();
  }
  function go(e) { try { sessionStorage.setItem("dsearch", input.value.trim()); } catch (x) {} location.href = hrefOf(e); if (panel) panel.hidden = true; }

  if (input) {
    input.addEventListener("focus", () => { load().then(() => input.value && show()); });
    input.addEventListener("input", () => load().then(show));
    input.addEventListener("keydown", (ev) => {
      if (!panel || panel.hidden) return;
      const links = [...panel.querySelectorAll(".hit")];
      if (ev.key === "ArrowDown" || ev.key === "ArrowUp") {
        ev.preventDefault(); pick = (pick + (ev.key === "ArrowDown" ? 1 : -1) + links.length) % Math.max(1, links.length);
        links.forEach((a, i) => a.classList.toggle("on", i === pick)); if (links[pick]) links[pick].scrollIntoView({ block: "nearest" });
      } else if (ev.key === "Enter" && hits[pick]) { ev.preventDefault(); go(hits[pick]); }
      else if (ev.key === "Escape") { panel.hidden = true; input.blur(); }
    });
    document.addEventListener("click", (ev) => {
      if (!panel || panel.hidden) return;
      const a = ev.target.closest && ev.target.closest(".dsearch-panel .hit");
      if (a) { ev.preventDefault(); go(hits[+a.dataset.i]); return; }
      if (ev.target !== input && !panel.contains(ev.target)) panel.hidden = true;
    });
    addEventListener("resize", () => panel && !panel.hidden && place());
    addEventListener("scroll", () => panel && !panel.hidden && place(), { passive: true });
    document.addEventListener("keydown", (ev) => {
      const t = ev.target;
      if (ev.key === "/" && !(t && (t.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName)))) { ev.preventDefault(); input.focus(); }
    });
  }

  // on arrival from a search: mark the words in the passage jumped to
  let q = ""; try { q = sessionStorage.getItem("dsearch") || ""; sessionStorage.removeItem("dsearch"); } catch (x) {}
  if (q && location.hash) {
    const target = document.getElementById(decodeURIComponent(location.hash.slice(1)));
    if (target) {
      const words = norm(q).split(/\s+/).filter((w) => w.length > 1), marks = [];
      const walker = document.createTreeWalker(target, NodeFilter.SHOW_TEXT, { acceptNode: (n) => n.parentElement.closest(".katex, mark") ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT });
      const nodes = []; while (walker.nextNode()) nodes.push(walker.currentNode);
      for (const node of nodes) {
        const n = norm(node.data); let i = -1, w = null;
        for (const x of words) { const j = n.indexOf(x); if (j >= 0 && (i < 0 || j < i)) { i = j; w = x; } }
        if (i < 0 || node.data.length !== n.length) continue;
        const mid = node.splitText(i); mid.splitText(w.length);
        const m = document.createElement("mark"); m.className = "dsearch-hit"; mid.parentNode.replaceChild(m, mid); m.appendChild(mid); marks.push(m);
        if (marks.length > 30) break;
      }
      setTimeout(() => marks.forEach((m) => m.classList.add("fade")), 3500);
    }
  }
})();
