// The dissertation as a graph of its results (deps.json, from tools/dissertation/deps.py).
//   map:  every result and cited equation by chapter, a line from each to what it cites; tap one to light up
//         what it rests on and what builds on it. #focus=<id> opens with one selected.
//   path: /dissertation/path/#<id>: that result with only what it rests on, fetched from the chapters and set
//         out in reading order, its own statement and proof last.
(function () {
  "use strict";
  const root = document.getElementById("graph");
  const out = document.getElementById("graph-out");
  if (!root || !out) return;
  const BASE = (root.dataset.base || "/dissertation/").replace(/\/?$/, "/");
  const el = (tag, cls, html) => { const n = document.createElement(tag); if (cls) n.className = cls; if (html != null) n.innerHTML = html; return n; };
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
  const SHORT = { assumption: "Asm", definition: "Def", lemma: "Lem", proposition: "Prop", theorem: "Thm", corollary: "Cor", conjecture: "Conj" };
  const short = (n) => n.kind === "equation" ? n.label.replace(/^Equation\s*/, "") : `${SHORT[n.kind] || n.kind} ${n.label.split(" ").pop()}`;

  fetch(BASE + "deps.json").then((r) => r.json()).then(({ nodes, edges, lean }) => {
    const byId = new Map(nodes.map((n, i) => [n.id, Object.assign(n, { i })]));
    const uses = new Map(), users = new Map();
    for (const [a, b] of edges) {
      (uses.get(a) || uses.set(a, []).get(a)).push(b);
      (users.get(b) || users.set(b, []).get(b)).push(a);
    }
    const closure = (id, next) => { const seen = new Set(), st = [id]; while (st.length) for (const y of next.get(st.pop()) || []) if (!seen.has(y)) { seen.add(y); st.push(y); } return seen; };
    const tools = { byId, uses, users, closure };
    if (root.dataset.mode === "path") path(nodes, tools); else { map(nodes, edges, tools); leanOn(lean || []); }
  }).catch(() => { out.innerHTML = '<p class="graph-wait">The graph could not be loaded.</p>'; });

  // ------------------------------------------------------------------ the map
  function map(nodes, edges, { byId, uses, users, closure }) {
    out.innerHTML = "";
    const controls = el("div", "gctl");
    const eqs = el("label", "", '<input type="checkbox" checked> cited equations');
    const legend = el("span", "glegend", '<i class="k-def"></i>definitions and assumptions <i class="k-res"></i>results <i class="k-eq"></i>equations'
      + ' &nbsp; <i class="k-anc"></i>what it rests on <i class="k-desc"></i>what builds on it');
    controls.append(eqs, legend);
    const wrap = el("div", "gmap");
    const cols = el("div", "gcols");
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("class", "gedges");
    wrap.append(svg, cols);
    const panel = el("div", "gpanel", '<p class="gp-hint">Tap a result.</p>');
    out.append(controls, wrap, panel);

    const pills = new Map(), pages = new Map();
    const page = (slug) => {
      if (!pages.has(slug)) pages.set(slug, fetch(`${BASE}${slug}/`).then((r) => r.text()).then((t) => new DOMParser().parseFromString(t, "text/html")));
      return pages.get(slug);
    };
    let chapter = null, col = null;
    for (const n of nodes) {
      if (n.chapter !== chapter) {
        chapter = n.chapter;
        col = el("div", "gcol", `<div class="gch">${esc(chapter)}</div>`);
        cols.appendChild(col);
      }
      const kindClass = n.kind === "equation" ? "k-eq" : (n.kind === "definition" || n.kind === "assumption") ? "k-def" : "k-res";
      const b = el("button", `gnode ${kindClass}`, esc(short(n)));
      b.type = "button";
      b.dataset.id = n.id;
      b.title = n.label + (n.name ? ` (${n.name})` : "");
      if (!(uses.get(n.id) || users.get(n.id))) b.classList.add("lonely");
      b.addEventListener("click", () => select(n.id));
      col.appendChild(b);
      pills.set(n.id, b);
    }
    let selected = null;
    function draw() {
      const box = cols.getBoundingClientRect();
      svg.setAttribute("width", cols.scrollWidth);
      svg.setAttribute("height", cols.scrollHeight);
      svg.innerHTML = "";
      const anc = selected ? closure(selected, uses) : null, desc = selected ? closure(selected, users) : null;
      for (const [a, b] of edges) {
        const pa = pills.get(a), pb = pills.get(b);
        if (!pa || !pb || pa.offsetParent === null || pb.offsetParent === null) continue;
        const ra = pa.getBoundingClientRect(), rb = pb.getBoundingClientRect();
        const ya = ra.top + ra.height / 2 - box.top, yb = rb.top + rb.height / 2 - box.top;
        let d;
        if (Math.abs(ra.left - rb.left) < 4) {                          // same chapter: an arc out to the right
          const x = ra.right - box.left, bulge = 14 + Math.min(60, Math.abs(ya - yb) * 0.25);
          d = `M${x},${ya} C${x + bulge},${ya} ${x + bulge},${yb} ${x},${yb}`;
        } else {                                                          // from what is cited (left) to what cites it
          const [l, r, yl, yr] = rb.left < ra.left ? [rb, ra, yb, ya] : [ra, rb, ya, yb];
          const x1 = l.right - box.left, x2 = r.left - box.left, mx = (x1 + x2) / 2;
          d = `M${x1},${yl} C${mx},${yl} ${mx},${yr} ${x2},${yr}`;
        }
        const p = document.createElementNS("http://www.w3.org/2000/svg", "path");
        p.setAttribute("d", d);
        let cls = "ge";
        if (selected) {
          if ((a === selected || anc.has(a)) && anc.has(b)) cls += " ge-anc";
          else if (desc.has(a) && (b === selected || desc.has(b))) cls += " ge-desc";
          else cls += " ge-dim";
        }
        p.setAttribute("class", cls);
        svg.appendChild(p);
      }
    }
    function select(id) {
      selected = selected === id ? null : id;
      const anc = selected ? closure(selected, uses) : new Set(), desc = selected ? closure(selected, users) : new Set();
      for (const [nid, b] of pills) {
        b.classList.toggle("sel", nid === selected);
        b.classList.toggle("anc", anc.has(nid));
        b.classList.toggle("desc", desc.has(nid));
        b.classList.toggle("dim", !!selected && nid !== selected && !anc.has(nid) && !desc.has(nid));
      }
      cols.classList.toggle("focused", !!selected);
      draw();
      if (!selected) { panel.innerHTML = '<p class="gp-hint">Tap a result.</p>'; history.replaceState(null, "", location.pathname); return; }
      history.replaceState(null, "", "#focus=" + encodeURIComponent(selected));
      const n = byId.get(selected);
      const count = (s, one, many) => `${s.size} ${s.size === 1 ? one : many}`;
      panel.innerHTML = `<div class="gp-head"><b>${esc(n.label)}</b>${n.name ? ` <span>(${esc(n.name)})</span>` : ""} <em>${esc(n.chapter)}</em></div>`
        + `<div class="gp-stmt"><p class="gp-text">${esc(n.text)}${n.text.length >= 220 ? "…" : ""}</p></div>`
        + `<p class="gp-meta">Rests on ${count(anc, "item", "items")}; ${count(desc, "later result builds", "later results build")} on it.</p>`
        + `<p class="gp-links"><a href="${BASE}${n.page}/#${encodeURIComponent(n.id)}">Open it in ${esc(n.chapter)} &rarr;</a>`
        + (anc.size ? ` <a href="${BASE}path/#${encodeURIComponent(n.id)}">Just what it rests on &rarr;</a>` : "") + "</p>";
      // the statement itself, typeset, from its chapter (the plain text above stands in until it arrives)
      const want = selected;
      page(n.page).then((d) => {
        const src = d.getElementById(n.id);
        if (!src || selected !== want) return;
        const piece = (n.kind === "equation" ? src.closest("p") || src : src).cloneNode(true);
        piece.querySelectorAll('a[href^="#"]').forEach((a) => { a.href = `${BASE}${n.page}/${a.getAttribute("href")}`; });
        piece.querySelectorAll("[id]").forEach((e) => e.removeAttribute("id"));
        const box = panel.querySelector(".gp-stmt");
        box.innerHTML = "";
        box.appendChild(piece);
        if (n.kind !== "equation") {                  // the statement carries its own label and name now
          panel.querySelector(".gp-head").remove();
          panel.querySelector(".gp-meta").prepend(`${n.chapter}. `);
        }
      }).catch(() => { /* keep the plain text */ });
    }
    eqs.querySelector("input").addEventListener("change", (e) => {
      cols.classList.toggle("no-eq", !e.target.checked);
      draw();
    });
    window.addEventListener("resize", () => requestAnimationFrame(draw));
    const m = location.hash.match(/^#focus=(.+)$/);
    requestAnimationFrame(() => {
      draw();
      if (m && byId.has(decodeURIComponent(m[1]))) {
        select(decodeURIComponent(m[1]));
        const b = pills.get(decodeURIComponent(m[1]));
        b.scrollIntoView({ block: "center", inline: "center" });
      }
    });
  }

  // ------------------------------------------------------------------ the equations the dissertation leans on
  function leanOn(lean) {
    if (!lean.length) return;
    const sec = el("section", "lean");
    sec.appendChild(el("h2", "", "The Equations It Comes Back To"));
    sec.appendChild(el("p", "lean-lede", "The equations the text refers to most often, counting every reference in every chapter."));
    const ol = el("ol", "lean-list");
    for (const e of lean) {
      const li = el("li", "", `<a href="${BASE}${e.page}/#${encodeURIComponent(e.id)}"><b>${esc(e.label.replace(/^Equation\s*/, ""))}</b></a>`
        + ` <span class="lean-n">cited ${e.cited} times, ${esc(e.chapter)}</span><span class="lean-t">${esc(e.text)}</span>`);
      ol.appendChild(li);
    }
    sec.appendChild(ol);
    out.appendChild(sec);
  }

  // ------------------------------------------------------------------ one result and just what it rests on
  function path(nodes, { byId, uses, closure }) {
    const pages = new Map();
    const page = (slug) => {
      if (!pages.has(slug)) pages.set(slug, fetch(`${BASE}${slug}/`).then((r) => r.text()).then((t) => new DOMParser().parseFromString(t, "text/html")));
      return pages.get(slug);
    };
    async function render() {
      const id = decodeURIComponent(location.hash.slice(1));
      const n = byId.get(id);
      if (!n) {
        out.innerHTML = `<p>Pick a result on <a href="${BASE}map/">the map</a>, or follow &ldquo;Just what it rests on&rdquo; from any result in a chapter.</p>`;
        return;
      }
      const need = [...closure(id, uses)].map((x) => byId.get(x)).filter(Boolean).sort((a, b) => a.i - b.i);
      document.title = `Just What ${n.label} Rests On`;
      root.querySelector("h1").textContent = `Just What ${n.label} Rests On`;
      out.innerHTML = `<p class="gp-lede">${esc(n.label)}${n.name ? ` (${esc(n.name)})` : ""} rests on ${need.length} `
        + `${need.length === 1 ? "item" : "items"}: the definitions, assumptions, results and equations its statement and proof cite, `
        + `and what those cite in turn. Here they are in reading order, then ${esc(n.label)} itself. `
        + `<a href="${BASE}map/#focus=${encodeURIComponent(id)}">See it on the map</a>.</p><p class="graph-wait">Gathering them&hellip;</p>`;
      const docs = await Promise.all([...new Set([...need, n].map((x) => x.page))].map((s) => page(s).then((d) => [s, d])));
      const doc = new Map(docs);
      const list = el("div", "path-list");
      let chapter = null;
      for (const x of [...need, n]) {
        if (x.chapter !== chapter) { chapter = x.chapter; list.appendChild(el("h2", "path-ch", esc(chapter))); }
        const src = doc.get(x.page).getElementById(x.id);
        if (!src) continue;
        const item = el("section", "path-item" + (x === n ? " path-target" : ""));
        let piece;
        if (x.kind === "equation") {
          piece = (src.closest("p") || src).cloneNode(true);
        } else {
          piece = el("div");
          piece.appendChild(src.cloneNode(true));
          const pf = src.nextElementSibling;
          if (pf && pf.classList.contains("proof")) {
            const d = el("details", "path-proof");
            if (x === n) d.open = true;
            d.appendChild(el("summary", "", "Proof"));
            d.appendChild(pf.cloneNode(true));
            piece.appendChild(d);
          }
        }
        // references inside point at their own chapter
        piece.querySelectorAll('a[href^="#"]').forEach((a) => { a.href = `${BASE}${x.page}/${a.getAttribute("href")}`; });
        piece.querySelectorAll("[id]").forEach((e) => e.removeAttribute("id"));
        item.appendChild(piece);
        item.appendChild(el("a", "path-src", `in context, ${esc(x.chapter)} &rarr;`)).href = `${BASE}${x.page}/#${encodeURIComponent(x.id)}`;
        list.appendChild(item);
      }
      out.querySelector(".graph-wait").replaceWith(list);
    }
    window.addEventListener("hashchange", render);
    render();
  }
})();
