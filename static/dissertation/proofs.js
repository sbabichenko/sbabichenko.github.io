// Proofs and what rests on what, on the dissertation's chapter pages.
//
// - Proofs fold to one line ("Proof · 3 paragraphs"); a tap opens one, and "show every proof" opens them all
//   (remembered per reader). A link or search hit that lands inside a folded proof opens it; paper prints all.
// - The proof's closing mark leads back up to the statement it proves.
// - Each result says which later results use it, and links to "just what it rests on" (/dissertation/path/).
// - An assumption can light up every result on the page that relies on it, directly or through others.
// - Following a reference out of a proof leaves a chip that brings you back to the line you left.
// The graph comes from deps.json (tools/dissertation/deps.py): only the references the text itself makes.
(function () {
  "use strict";
  const paper = document.getElementById("paper");
  const body = paper && paper.querySelector("article.body");
  if (!body) return;
  const BASE = (document.currentScript && document.currentScript.src.replace(/proofs\.js.*$/, "")) || "/dissertation/";
  const store = {
    get(k) { try { return localStorage.getItem(k); } catch (e) { return null; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* private mode: not remembered */ } },
    sget(k) { try { return JSON.parse(sessionStorage.getItem(k) || "null"); } catch (e) { return null; } },
    sset(k, v) { try { v === null ? sessionStorage.removeItem(k) : sessionStorage.setItem(k, JSON.stringify(v)); } catch (e) { /* no session storage */ } },
  };
  const el = (tag, cls, html) => { const n = document.createElement(tag); if (cls) n.className = cls; if (html != null) n.innerHTML = html; return n; };
  const labelOf = (thm) => { const s = thm && thm.querySelector("strong"); return s ? s.textContent.trim().replace(/\.$/, "") : ""; };
  const norm = (t) => t.replace(/\s+/g, " ").trim();
  // the proof's own opening words ("Proof.", "Proof sketch.", "Proof of Theorem 4.6.")
  const openingOf = (proof) => { const p = proof.querySelector("p"), em = p && p.firstElementChild; return em && em.tagName === "EM" && p.firstChild === em ? em : null; };
  // the result a proof proves: the one it names, else the one it follows
  function resultOf(proof) {
    const em = openingOf(proof), m = em && norm(em.textContent).match(/^Proof of (.+?)\.?$/);
    if (m) for (const t of body.querySelectorAll(".thm")) if (norm(labelOf(t)) === norm(m[1])) return t;
    const prev = proof.previousElementSibling;
    return prev && prev.classList.contains("thm") ? prev : null;
  }

  // ---- folding
  const proofs = [...body.querySelectorAll(".proof")].filter((p) => !p.parentElement.closest(".proof"));
  let allOpen = store.get("proofs-open") === "1";
  proofs.forEach((p, i) => {
    const n = p.querySelectorAll(":scope > p, :scope > .math, :scope > ol, :scope > ul").length || 1;
    const em = openingOf(p), own = em ? norm(em.textContent).replace(/\.$/, "") : "Proof";
    const of = /^Proof\.?$/.test(own) && resultOf(p) ? ` of ${labelOf(resultOf(p))}` : "";
    if (em) em.classList.add("proof-opening");             // the button says it now; shown again when printed
    const btn = el("button", "proof-toggle", `<span class="pt-word">${own}${of}</span>`
      + `<span class="pt-len"> · ${n} paragraph${n > 1 ? "s" : ""}</span><span class="pt-arrow" aria-hidden="true"></span>`);
    btn.type = "button";
    const inner = el("div", "proof-body");
    inner.id = "proof-body-" + i;
    while (p.firstChild) inner.appendChild(p.firstChild);
    p.append(btn, inner);
    p.classList.add("foldable");
    btn.setAttribute("aria-controls", inner.id);
    btn.addEventListener("click", () => setOpen(p, !p.classList.contains("open")));
    setOpen(p, allOpen);
  });
  function setOpen(p, on) {
    p.classList.toggle("open", on);
    p.querySelector(".proof-toggle").setAttribute("aria-expanded", on ? "true" : "false");
  }
  if (proofs.length) {
    const bar = el("div", "proofs-bar");
    const note = el("span", "pb-note"), sw = el("button", "proofs-all");
    sw.type = "button";
    const k = `${proofs.length} proof${proofs.length > 1 ? "s" : ""} in this chapter`;
    const paint = () => {
      note.textContent = `${k}, ${allOpen ? "all shown" : "each folded to one line"}.`;
      sw.textContent = allOpen ? "Fold them" : "Show every proof";
      sw.setAttribute("aria-pressed", allOpen ? "true" : "false");
    };
    sw.addEventListener("click", () => { allOpen = !allOpen; store.set("proofs-open", allOpen ? "1" : "0"); proofs.forEach((p) => setOpen(p, allOpen)); paint(); });
    paint();
    bar.append(note, sw);
    body.insertBefore(bar, body.firstElementChild && body.firstElementChild.nextElementSibling || body.firstChild);
  }
  // a link, a search hit or a back-chip that lands inside a folded proof opens it
  function reveal(target) {
    const p = target && target.closest && target.closest(".proof.foldable");
    if (p && !p.classList.contains("open")) setOpen(p, true);
  }
  const fromHash = () => { if (location.hash.length > 1) { const t = document.getElementById(decodeURIComponent(location.hash.slice(1))); if (t) { reveal(t); t.scrollIntoView(); } } };
  window.addEventListener("hashchange", fromHash);
  if (location.hash.length > 1) fromHash();
  document.addEventListener("dissertation:reveal", (e) => reveal(e.detail));   // search.js can announce a hit
  window.addEventListener("beforeprint", () => proofs.forEach((p) => p.classList.add("print-open")));
  window.addEventListener("afterprint", () => proofs.forEach((p) => p.classList.remove("print-open")));

  // ---- the closing mark leads back to the statement
  proofs.forEach((p) => {
    const thm = resultOf(p);
    if (!thm || !thm.id) return;
    const walker = document.createTreeWalker(p, NodeFilter.SHOW_TEXT);
    let last = null;
    for (let t = walker.nextNode(); t; t = walker.nextNode()) if (/[◻∎□]/.test(t.nodeValue) && !t.parentElement.closest(".katex")) last = t;
    if (!last) return;
    const i = Math.max(last.nodeValue.lastIndexOf("◻"), last.nodeValue.lastIndexOf("∎"), last.nodeValue.lastIndexOf("□"));
    const after = last.splitText(i);
    after.splitText(1);
    const a = el("a", "qed", after.nodeValue);
    a.href = "#" + thm.id;
    a.title = "Back to " + labelOf(thm);
    a.setAttribute("aria-label", "End of proof: back to " + labelOf(thm));
    after.replaceWith(a);
  });

  // ---- the way back: following a reference out of a result or proof leaves a chip
  const here = location.pathname;
  body.addEventListener("click", (ev) => {
    const a = ev.target.closest("a.xref[href]");
    if (!a) return;
    const box = a.closest(".proof, .thm");
    if (!box) return;
    const thm = box.classList.contains("thm") ? box : resultOf(box);
    const what = box.classList.contains("proof") ? `the proof of ${labelOf(thm)}` : labelOf(thm);
    store.sset("way-back", { path: here, y: window.scrollY, what, t: Date.now() });
    setTimeout(showChip, 60);                            // a reference on this page: show the chip after the jump
  });
  function showChip() {
    const w = store.sget("way-back");
    const old = document.querySelector(".wayback");
    if (old) old.remove();
    if (!w || Date.now() - w.t > 30 * 60 * 1000) return;
    if (w.path === here && Math.abs(window.scrollY - w.y) < 200) return;       // still where it was left
    const chip = el("div", "wayback");
    const go = el("a", "wb-go", `&#8617; Back to ${w.what}`);
    go.href = w.path;
    const x = el("button", "wb-x", "&times;");
    x.type = "button"; x.setAttribute("aria-label", "Dismiss");
    go.addEventListener("click", (ev) => {
      store.sset("way-back", null);
      if (w.path === here) { ev.preventDefault(); window.scrollTo({ top: w.y, behavior: "smooth" }); chip.remove(); }
      else store.sset("way-back-restore", { path: w.path, y: w.y });
    });
    x.addEventListener("click", () => { store.sset("way-back", null); chip.remove(); });
    chip.append(go, x);
    (paper.closest(".thesis") || document.body).appendChild(chip);
  }
  const restore = store.sget("way-back-restore");
  if (restore && restore.path === here) {
    store.sset("way-back-restore", null);
    const y = restore.y;
    const land = () => { window.scrollTo(0, y); const t = document.elementFromPoint(window.innerWidth / 2, 120); reveal(t); window.scrollTo(0, y); };
    if (document.readyState === "complete") land(); else window.addEventListener("load", land);
  } else showChip();

  // ---- the graph: "used in", "what it rests on", assumption tracing
  fetch(BASE + "deps.json").then((r) => r.json()).then(({ nodes, edges }) => {
    const byId = new Map(nodes.map((n) => [n.id, n]));
    const users = new Map(), uses = new Map();
    for (const [a, b] of edges) {
      if (!users.has(b)) users.set(b, []);
      users.get(b).push(a);
      if (!uses.has(a)) uses.set(a, []);
      uses.get(a).push(b);
    }
    const closure = (id, next) => { const seen = new Set(), st = [id]; while (st.length) for (const y of next.get(st.pop()) || []) if (!seen.has(y)) { seen.add(y); st.push(y); } return seen; };
    const href = (n) => (n.page === pageSlug ? "" : `${BASE}${n.page}/`) + "#" + n.id;
    const pageSlug = (here.match(/\/dissertation\/([\w-]+)\/?$/) || [])[1];
    for (const thm of body.querySelectorAll(".thm[id]")) {
      const n = byId.get(thm.id);
      if (!n) continue;
      const line = el("div", "thm-links");
      const u = (users.get(n.id) || []).map((id) => byId.get(id)).filter((x) => x && x.kind !== "equation");
      if (u.length)
        line.append(el("span", "tl-used", "Used in " + u.map((x) => `<a class="xref" href="${href(x)}">${x.label}</a>`).join(", ")));
      const below = closure(n.id, uses);
      if (below.size >= 2) {
        const a = el("a", "tl-path", "Just what it rests on &rarr;");
        a.href = `${BASE}path/#${n.id}`;
        a.title = `${below.size} results and equations, in reading order`;
        line.append(a);
      }
      if (n.kind === "assumption") {
        const relies = closure(n.id, users);
        if (relies.size) {
          const b = el("button", "tl-trace", `Light up what relies on it (${relies.size})`);
          b.type = "button";
          b.addEventListener("click", () => trace(n, relies, b));
          line.append(b);
        }
      }
      if (line.childNodes.length) thm.appendChild(line);
    }
    let tracing = null;
    function trace(n, relies, btn) {
      const off = tracing === n.id;
      body.classList.toggle("tracing", !off);
      body.querySelectorAll(".relies, .traced").forEach((x) => x.classList.remove("relies", "traced"));
      document.querySelectorAll(".tl-trace[aria-pressed]").forEach((x) => x.removeAttribute("aria-pressed"));
      const note = document.querySelector(".trace-note");
      if (note) note.remove();
      tracing = off ? null : n.id;
      if (off) return;
      btn.setAttribute("aria-pressed", "true");
      document.getElementById(n.id).classList.add("traced");
      let here_ = 0;
      const elsewhere = new Map();
      for (const id of relies) {
        const m = byId.get(id), t = document.getElementById(id);
        if (t && t.closest("article.body") === body) {
          (t.classList.contains("thm") ? t : t.closest("p, .thm, li") || t).classList.add("relies");
          const pf = t.nextElementSibling;
          if (pf && pf.classList.contains("proof")) pf.classList.add("relies");
          here_++;
        } else if (m) elsewhere.set(m.chapter, (elsewhere.get(m.chapter) || 0) + 1);
      }
      const bits = [...elsewhere].map(([c, k]) => `${k} in ${c}`);
      const msg = `${n.label}: ${here_} on this page rel${here_ === 1 ? "ies" : "y"} on it`
        + (bits.length ? `, and ${bits.join(", ")}` : "") + ".";
      const bar = el("div", "trace-note", `<span>${msg}</span> <a href="${BASE}map/#focus=${encodeURIComponent(n.id)}">See it on the map</a> <button type="button">Clear</button>`);
      bar.querySelector("button").addEventListener("click", () => trace(n, relies, btn));
      (paper.closest(".thesis") || document.body).appendChild(bar);
    }
  }).catch(() => { /* no graph: the pages read as before */ });
})();
