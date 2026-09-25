// Proofs and what rests on what, on the dissertation's chapter pages.
//
// - Proofs fold to one line ("Proof · 3 paragraphs"); a tap opens one, and "show every proof" opens them all
//   (remembered per reader). A link or search hit that lands inside a folded proof opens it; paper prints all.
// - The proof's closing mark leads back up to the statement it proves.
// - Each result says which later results use it, and links to "just what it rests on" (/dissertation/path/).
// - An assumption can light up every result on the page that relies on it, directly or through others.
// - Following a reference out of a proof leaves a chip that brings you back to the line you left.
// - A few results link to the page that computes them live; much-cited equations say how often they are cited.
// - A proof with several displays can be stepped through, one display and its sentence at a time.
// - On a screen wide enough, proofs can sit beside the text, level with what they prove.
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
  // results with a page that computes them (checked by hand; a chapter's game is linked from one result only)
  const ROOT = BASE.replace(/dissertation\/$/, "");
  const LIVE = {
    "thm:noise_state_filter_compact": [`${BASE}idea/`, "Drawn as you read: The Noise-State, Explained"],
    "thm:wedge-adjoints": [`${BASE}wedge/`, "Solved live: The Price of Changing Someone’s Mind"],
    "cor:nsl-br": [`${ROOT}noisestate/#game=ch1`, "Checked on every solve: the tracking game in the explorer"],
    "prop:stationary_verification": [`${ROOT}noisestate/#game=ch3`, "This chapter’s game, solved in the explorer"],
    "thm:stationary-best-response": [`${ROOT}noisestate/#game=ch4`, "This chapter’s market, solved in the explorer"],
    "cor:graph_unique_equilibrium": [`${ROOT}noisestate/#game=ch5`, "This chapter’s game, solved in the explorer"],
    "prop:gain_reduction": [`${ROOT}noisestate/#game=ch6`, "This chapter’s game, solved in the explorer"],
  };
  fetch(BASE + "deps.json").then((r) => r.json()).then(({ nodes, edges, cites }) => {
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
      if (LIVE[n.id]) {
        const a = el("a", "tl-live", `${LIVE[n.id][1]} &rarr;`);
        a.href = LIVE[n.id][0];
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
    // equations the text keeps coming back to
    for (const [eid, k] of Object.entries(cites || {})) {
      const eq = document.getElementById(eid);
      if (!eq || k < 4 || eq.closest("article.body") !== body) continue;
      const note = el("span", "eqcite", `cited ${k} times`);
      note.title = "How often the dissertation refers to this equation";
      eq.after(note);
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

  // ---- step through a proof: each display and the sentence that leads to it, the rest held back
  proofs.forEach((p) => {
    const inner = p.querySelector(".proof-body");
    const shows = [...inner.querySelectorAll(".math.display")].filter((m) => !m.parentElement.closest(".math.display"));
    if (shows.length < 2) return;
    const bar = el("div", "proof-steps");
    const go = el("button", "ps-start", `Step through, ${shows.length} displays`);
    go.type = "button";
    bar.append(go);
    inner.prepend(bar);
    const blocks = [...inner.children].filter((c) => c !== bar);
    const blockOf = (m) => blocks.find((b) => b.contains(m));
    let at = -1;
    function paint() {
      const on = at >= 0;
      p.classList.toggle("stepping", on);
      shows.forEach((m, i) => m.classList.toggle("step-now", on && i === at));
      const cut = on ? blocks.indexOf(blockOf(shows[at])) : -1;
      blocks.forEach((b, i) => b.classList.toggle("step-later", on && i > cut));
      if (on) {
        // inside the block that holds this display, what comes after it waits too
        const inBlock = [...blocks[cut].querySelectorAll(".math.display")], k = inBlock.indexOf(shows[at]);
        inBlock.forEach((m, i) => m.classList.toggle("step-later", i > k));
        bar.innerHTML = "";
        const prev = el("button", "", "&lsaquo; Back"), next = el("button", "", at < shows.length - 1 ? "Next &rsaquo;" : "Show all"), n = el("span", "ps-n", `${at + 1} of ${shows.length}`);
        prev.type = next.type = "button";
        prev.disabled = at === 0;
        prev.addEventListener("click", () => { at--; paint(); });
        next.addEventListener("click", () => { at = at < shows.length - 1 ? at + 1 : -1; paint(); });
        bar.append(prev, n, next);
        shows[at].scrollIntoView({ block: "center", behavior: "smooth" });
      } else {
        inner.querySelectorAll(".step-later").forEach((x) => x.classList.remove("step-later"));
        bar.innerHTML = ""; bar.append(go);
      }
    }
    go.addEventListener("click", () => { setOpen(p, true); at = 0; paint(); });
  });

  // ---- proofs beside the text, on a screen with room for a third column
  const wideEnough = window.matchMedia("(min-width: 1500px)");
  const bar = body.querySelector(".proofs-bar");
  if (bar && proofs.length) {
    const side = el("button", "proofs-side");
    side.type = "button";
    bar.append(side);
    let beside = store.get("proofs-beside") === "1";
    const thesis = paper.closest(".thesis");
    function place() {
      if (!thesis.classList.contains("beside")) return;
      let floor = 0;
      for (const p of proofs) {
        const thm = resultOf(p);
        const want = thm ? thm.offsetTop : p.offsetTop;
        const top = Math.max(want, floor);
        p.style.top = top + "px";
        floor = top + p.offsetHeight + 16;
      }
      body.style.minHeight = floor + "px";
    }
    function apply() {
      const on = beside && wideEnough.matches;
      thesis.classList.toggle("beside", on);
      side.hidden = !wideEnough.matches;
      side.textContent = on ? "Proofs in the text" : "Proofs beside the text";
      proofs.forEach((p) => { p.classList.toggle("side", on); if (on) setOpen(p, true); else p.style.top = ""; });
      if (!on) body.style.minHeight = "";
      place();
    }
    side.addEventListener("click", () => { beside = !beside; store.set("proofs-beside", beside ? "1" : "0"); apply(); });
    wideEnough.addEventListener ? wideEnough.addEventListener("change", apply) : wideEnough.addListener(apply);
    window.addEventListener("resize", () => requestAnimationFrame(place));
    window.addEventListener("load", place);
    body.addEventListener("click", () => requestAnimationFrame(place));
    apply();
  }
})();
