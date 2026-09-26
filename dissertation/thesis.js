// The dissertation pages: a reading-progress line, the rail following the section you are in, and previews:
// hover (or focus) a reference such as "Theorem 4.2", "(3.14)" or a citation and the thing it points to
// appears beside it, fetched from its own chapter when it lives on another page.
(function () {
  "use strict";
  const paper = document.getElementById("paper");
  if (!paper) return;

  // ---- progress through this page's text, counted in characters rather than pixels: sections off screen are
  // not laid out (content-visibility in thesis.css) and grow to their real height as you reach them, so a
  // fraction of the page's height would slide backwards. The section under the middle of the screen is laid
  // out, so its own share is read from its height; every block above it counts in full. The blocks are the
  // chapter's top-level pieces (paragraphs, displays, figures and sections), each worth at least a short
  // paragraph so that a figure counts.
  const bar = document.getElementById("progress");
  const secs = [...paper.querySelectorAll(".body > section.level1")].flatMap((s) => [...s.children]);
  const weight = secs.map((s) => Math.max(200, s.textContent.length)), total = weight.reduce((a, w) => a + w, 0);
  let lastY = -1, lastF = 0;
  const onScroll = () => {
    const vh = window.innerHeight, doc = document.documentElement;
    let f;
    if (!total) { const r = paper.getBoundingClientRect(), h = r.height - vh; f = h > 0 ? -r.top / h : 1; }
    else if (window.scrollY + vh >= doc.scrollHeight - 2) f = 1;
    else {
      const line = vh / 2;
      let done = 0;
      for (let i = 0; i < secs.length; ++i) {
        const r = secs[i].getBoundingClientRect();
        if (r.bottom <= line) { done += weight[i]; continue; }
        if (r.top < line && r.height > 0) done += weight[i] * (line - r.top) / r.height;
        break;
      }
      f = done / total;
    }
    // the section being read can still settle as it is laid out; scrolling down never moves the line back
    f = Math.min(1, Math.max(0, f));
    if (window.scrollY > lastY && f < lastF) f = lastF;
    lastY = window.scrollY; lastF = f;
    bar.style.width = f * 100 + "%";
  };
  window.addEventListener("scroll", onScroll, { passive: true });
  window.addEventListener("resize", onScroll);
  onScroll();

  // ---- the end of the page draws itself when it comes into view
  const fin = document.querySelector(".fin");
  if (fin && "IntersectionObserver" in window) {
    const fo = new IntersectionObserver((es) => { if (es.some((e) => e.isIntersecting)) { fin.classList.add("drawn"); fo.disconnect(); } });
    fo.observe(fin);
  }

  // ---- under a figure's caption: how it was computed (solver, grid, script), and, where the explorer has the game,
  // a link that solves it there at the figure's parameters (data/dissertation/figure_notes.json)
  const notesEl = document.getElementById("fignotes");
  if (notesEl) {
    let notes = {};
    try { notes = JSON.parse(notesEl.textContent); } catch (e) { /* no notes */ }
    for (const [id, n] of Object.entries(notes)) {
      const fig = document.getElementById(id), cap = fig && fig.querySelector(":scope > figcaption");
      if (!cap || fig.querySelector(":scope > .figsrc")) continue;
      const p = document.createElement("p");
      p.className = "figsrc";
      if (n.note) p.append(n.note);
      if (n.code) {
        const c = document.createElement("a");
        c.href = n.code.url; c.textContent = n.code.label + " →";
        if (n.note) p.append(" ");
        p.append(c);
      }
      if (n.inspect) {
        const b = document.createElement("button");
        b.type = "button"; b.className = "inspect"; b.textContent = "Inspect the code";
        b.setAttribute("aria-expanded", "false");
        b.addEventListener("click", () => inspect(fig, p, n.inspect, b));
        if (n.note) p.append(" ");
        p.append(b);
      }
      if (n.explorer) {
        const a = document.createElement("a");
        a.href = notesEl.dataset.explorer + "#" + n.explorer.hash;
        a.textContent = n.explorer.label + " →";
        if (n.note || n.code || n.inspect) p.append(n.code || n.inspect ? " · " : " ");
        p.append(a);
      }
      cap.after(p);
    }
  }

  // ---- a figure's Inspect panel: the code that draws it and the model it solves, line by line
  // (static/dissertation/figcode/, copied there by tools/figures/render.sh from the files it ran)
  const FIGCODE = "/dissertation/figcode/";
  const escHtml = (s) => s.replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
  const PY_KW = new Set("import from as def return for in if elif else while with and or not None True False lambda class try except raise yield".split(" "));
  // a small highlighter: comments, strings, numbers and keywords (Python), keys (YAML); enough to read by, not a parser
  function highlight(src, lang) {
    const out = [];
    const re = lang === "py"
      ? /("""[\s\S]*?"""|r?"(?:[^"\\\n]|\\.)*"|r?'(?:[^'\\\n]|\\.)*')|(#[^\n]*)|(\b\d+(?:\.\d+)?(?:e-?\d+)?\b)|([A-Za-z_]\w*)|([\s\S])/g
      : /(#[^\n]*)|("(?:[^"\\\n]|\\.)*"|'[^'\n]*')|(^[ \t-]*[\w.]+(?=:))|(\b-?\d+(?:\.\d+)?\b)|([\s\S])/gm;
    let m;
    while ((m = re.exec(src))) {
      if (lang === "py") {
        if (m[1]) out.push(`<span class="s">${escHtml(m[1])}</span>`);
        else if (m[2]) out.push(`<span class="c">${escHtml(m[2])}</span>`);
        else if (m[3]) out.push(`<span class="n">${m[3]}</span>`);
        else if (m[4]) out.push(PY_KW.has(m[4]) ? `<span class="k">${m[4]}</span>` : m[4]);
        else out.push(escHtml(m[5]));
      } else {
        if (m[1]) out.push(`<span class="c">${escHtml(m[1])}</span>`);
        else if (m[2]) out.push(`<span class="s">${escHtml(m[2])}</span>`);
        else if (m[3]) out.push(`<span class="k">${escHtml(m[3])}</span>`);
        else if (m[4]) out.push(`<span class="n">${m[4]}</span>`);
        else out.push(escHtml(m[5]));
      }
    }
    // one row per line, numbered by the stylesheet; a highlighted span never crosses a line except a docstring,
    // which is split and re-opened on each line
    const lines = out.join("").split("\n");
    let open = null;
    return lines.map((ln) => {
      let h = (open ? `<span class="${open}">` : "") + ln;
      const opens = (h.match(/<span class="(\w)">/g) || []).length, closes = (h.match(/<\/span>/g) || []).length;
      if (opens > closes) { open = h.match(/<span class="(\w)">(?![\s\S]*<span)/)?.[1] || open || "s"; h += "</span>"; } else open = null;
      return `<span class="ln">${h || " "}</span>`;
    }).join("");
  }
  function inspect(fig, after, files, btn) {
    let panel = fig.querySelector(":scope > .inspector");
    if (panel) { const hide = !panel.hidden; panel.hidden = hide; btn.setAttribute("aria-expanded", String(!hide)); btn.textContent = hide ? "Inspect the code" : "Hide the code"; return; }
    panel = document.createElement("div");
    panel.className = "inspector";
    const tabs = [["code", "Code", files.code, "py"], ["model", "Model", files.model, "yaml"], ["run", "Run it", null, null]];
    panel.innerHTML = `<div class="tabs" role="tablist">${tabs.map(([k, l], i) => `<button type="button" role="tab" data-k="${k}" aria-selected="${i === 0}">${l}</button>`).join("")}</div>`
      + tabs.map(([k, , f], i) => `<div class="pane" data-k="${k}"${i ? " hidden" : ""}>${f ? `<div class="bar"><span class="file">${escHtml(f)}</span><button type="button" class="copy">Copy</button><a href="${FIGCODE}${f}" download>Download</a></div><pre class="code"><code>Loading…</code></pre>` : ""}</div>`).join("");
    after.after(panel);
    const run = panel.querySelector('.pane[data-k="run"]');
    run.innerHTML = `<p>Put <a href="${FIGCODE}${files.code}" download>${escHtml(files.code)}</a> and <a href="${FIGCODE}${files.model}" download>${escHtml(files.model)}</a> in one folder, then:</p>
      <pre class="code"><code><span class="ln">pip install "noisestate&gt;=2" matplotlib</span><span class="ln">python ${escHtml(files.code)}</span></code></pre>
      <p>It solves the game and writes the figure as a PDF next to the script. Change the numbers in the model to see how the figure moves.</p>`;
    panel.querySelectorAll('[role="tab"]').forEach((t) => t.addEventListener("click", () => {
      panel.querySelectorAll('[role="tab"]').forEach((u) => u.setAttribute("aria-selected", String(u === t)));
      panel.querySelectorAll(".pane").forEach((p) => { p.hidden = p.dataset.k !== t.dataset.k; });
    }));
    for (const [k, , f, lang] of tabs) {
      if (!f) continue;
      const pane = panel.querySelector(`.pane[data-k="${k}"]`), code = pane.querySelector("code");
      fetch(FIGCODE + f).then((r) => (r.ok ? r.text() : Promise.reject(r.status))).then((src) => {
        code.innerHTML = highlight(src.replace(/\n$/, ""), lang);
        pane.querySelector(".copy").addEventListener("click", (ev) => {
          navigator.clipboard.writeText(src).then(() => { ev.target.textContent = "Copied"; setTimeout(() => { ev.target.textContent = "Copy"; }, 1400); }, () => {});
        });
      }).catch(() => { code.textContent = "The file could not be loaded."; });
    }
    btn.setAttribute("aria-expanded", "true"); btn.textContent = "Hide the code";
  }

  // ---- the rail: the section on screen is marked; on narrow screens the contents fold under the title
  const links = [...document.querySelectorAll(".rail .secs a[data-sec]")];
  const targets = links.map((a) => document.getElementById(a.dataset.sec)).filter(Boolean);
  if (targets.length && "IntersectionObserver" in window) {
    const on = new Set();
    const mark = () => {
      let best = null;
      for (const t of targets) if (on.has(t) && (!best || t.getBoundingClientRect().top < best.getBoundingClientRect().top)) best = t;
      if (!best) return;
      for (const a of links) a.classList.toggle("on", a.dataset.sec === best.id);
      // sections scrolled past get a pencil tick, and keep it
      const at = links.findIndex((a) => a.dataset.sec === best.id);
      links.forEach((a, i) => { if (i < at) a.classList.add("read"); });
    };
    const io = new IntersectionObserver((es) => { for (const e of es) (e.isIntersecting ? on.add(e.target) : on.delete(e.target)); mark(); },
      { rootMargin: "-80px 0px -55% 0px" });
    targets.forEach((t) => io.observe(t));
  }
  const rail = document.querySelector(".rail"), book = rail && rail.querySelector(".book");
  if (book) book.addEventListener("click", (ev) => {
    if (window.matchMedia("(max-width: 1060px)").matches) { ev.preventDefault(); rail.classList.toggle("open"); }
  });

  // ---- phones: a figure too wide to read at this width opens full screen, to pan and pinch. Figures with a
  // phone layout of their own (a <source> for narrow screens) are readable in place and are left alone.
  const phone = window.matchMedia("(max-width: 600px)");
  const zoomable = (fig) => !fig.querySelector("source[media]");
  const mark = () => paper.querySelectorAll(".body figure").forEach((f) => {
    if (!f.parentElement.closest("figure")) f.classList.toggle("zoomable", phone.matches && zoomable(f));
  });
  mark();
  phone.addEventListener ? phone.addEventListener("change", mark) : phone.addListener(mark);
  let zoom = null;
  function closeZoom(fromHistory) {
    if (!zoom) return;
    zoom.remove(); zoom = null;
    document.documentElement.style.overflow = "";
    if (!fromHistory && history.state && history.state.figzoom) history.back();
  }
  function openZoom(img, fig) {
    const num = fig.querySelector(".fignum");
    zoom = document.createElement("div");
    zoom.className = "figzoom";
    zoom.setAttribute("role", "dialog"); zoom.setAttribute("aria-modal", "true");
    zoom.innerHTML = `<div class="fz-bar"><span>${num ? num.textContent.trim().replace(/\.$/, "") + " · " : ""}drag to pan, pinch to zoom</span>`
      + `<button type="button" aria-label="Close">&times;</button></div><div class="fz-scroll"></div>`;
    const big = document.createElement("img");
    big.src = img.currentSrc || img.src; big.alt = "";
    if (!fig.classList.contains("webfig")) big.className = "print";     // print conversions need their white ground
    zoom.querySelector(".fz-scroll").appendChild(big);
    zoom.querySelector("button").addEventListener("click", () => closeZoom(false));
    (paper.closest(".thesis") || document.body).appendChild(zoom);   // inside .thesis, where the colours are defined
    document.documentElement.style.overflow = "hidden";
    history.pushState({ figzoom: true }, "");                           // the back button closes it
    const sc = zoom.querySelector(".fz-scroll");
    big.addEventListener("load", () => { sc.scrollLeft = 0; });
  }
  paper.addEventListener("click", (ev) => {
    if (!phone.matches || ev.target.closest("a")) return;
    const img = ev.target.closest(".body figure img");
    const fig = img && img.closest("figure");
    if (!fig || !zoomable(fig) || fig.closest(".peek")) return;
    openZoom(img, fig);
  });
  window.addEventListener("popstate", () => closeZoom(true));
  window.addEventListener("keydown", (ev) => { if (ev.key === "Escape") closeZoom(false); });

  // ---- previews
  const peek = document.getElementById("peek");
  if (!peek || window.matchMedia("(hover: none)").matches) return;
  const pages = new Map();                               // url -> Promise<Document>
  const getDoc = (url) => {
    if (!pages.has(url)) pages.set(url, fetch(url).then((r) => r.text()).then((t) => new DOMParser().parseFromString(t, "text/html")));
    return pages.get(url);
  };
  function excerpt(el) {
    if (!el) return null;
    // an equation: its display, and the sentence it sits in if short
    if (el.matches("span.math")) return el.cloneNode(true);
    if (el.matches(".thm, figure, li, table")) return el.cloneNode(true);
    if (el.matches("section")) {
      const box = document.createElement("div"), h = el.querySelector("h1, h2, h3, h4"), p = el.querySelector("p");
      if (h) box.appendChild(h.cloneNode(true));
      if (p) box.appendChild(p.cloneNode(true));
      return box;
    }
    const wrap = el.closest("p, .thm, figure") || el;
    return wrap.cloneNode(true);
  }
  let timer = 0, current = null;
  function hide() { clearTimeout(timer); peek.hidden = true; current = null; }
  async function show(a) {
    const url = new URL(a.href, location.href);
    const id = decodeURIComponent(url.hash.slice(1));
    if (!id) return;
    const here = url.pathname === location.pathname;
    let doc = document;
    if (!here) { try { doc = await getDoc(url.pathname); } catch (e) { return; } }
    if (current !== a) return;
    const el = doc.getElementById(id);
    const ex = excerpt(el);
    if (!ex) return;
    ex.removeAttribute("id");
    ex.querySelectorAll("[id]").forEach((n) => n.removeAttribute("id"));
    peek.innerHTML = "";
    const where = document.createElement("div");
    where.className = "where";
    const h1 = doc.querySelector(".body h1");
    where.textContent = here ? "On this page" : (h1 ? [...h1.childNodes].map((n) => n.textContent).join(" ").replace(/\s+/g, " ").trim() : "Dissertation").slice(0, 80);
    if (url.pathname.includes("/references/")) where.textContent = "Reference";
    peek.append(where, ex);
    peek.hidden = false;
    const r = a.getBoundingClientRect(), w = peek.offsetWidth, h = peek.offsetHeight;
    let x = window.scrollX + r.left + r.width / 2 - w / 2;
    x = Math.max(window.scrollX + 12, Math.min(x, window.scrollX + document.documentElement.clientWidth - w - 12));
    const below = r.bottom + h + 16 < window.innerHeight;
    const y = below ? window.scrollY + r.bottom + 8 : window.scrollY + r.top - h - 8;
    peek.style.left = x + "px"; peek.style.top = y + "px";
  }
  const isRef = (a) => a && (a.classList.contains("xref") || /\/references\/#ref-/.test(a.getAttribute("href") || ""));
  paper.addEventListener("mouseover", (ev) => {
    const a = ev.target.closest("a");
    if (!isRef(a) || a === current) return;
    current = a; clearTimeout(timer);
    timer = setTimeout(() => show(a), 180);
  });
  paper.addEventListener("mouseout", (ev) => {
    const a = ev.target.closest("a");
    if (!isRef(a)) return;
    if (ev.relatedTarget && (peek.contains(ev.relatedTarget) || a.contains(ev.relatedTarget))) return;
    timer = setTimeout(hide, 220);
  });
  peek.addEventListener("mouseleave", () => { timer = setTimeout(hide, 220); });
  peek.addEventListener("mouseenter", () => clearTimeout(timer));
  paper.addEventListener("focusin", (ev) => { const a = ev.target.closest("a"); if (isRef(a)) { current = a; show(a); } });
  paper.addEventListener("focusout", hide);
  window.addEventListener("keydown", (ev) => { if (ev.key === "Escape") hide(); });

})();
