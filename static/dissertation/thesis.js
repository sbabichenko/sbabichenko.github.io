// The dissertation pages: a reading-progress line, the rail following the section you are in, and previews:
// hover (or focus) a reference such as "Theorem 4.2", "(3.14)" or a citation and the thing it points to
// appears beside it, fetched from its own chapter when it lives on another page.
(function () {
  "use strict";
  const paper = document.getElementById("paper");
  if (!paper) return;

  // ---- progress through this page's text
  const bar = document.getElementById("progress");
  const onScroll = () => {
    const r = paper.getBoundingClientRect(), h = r.height - window.innerHeight;
    bar.style.width = (h > 0 ? Math.min(1, Math.max(0, -r.top / h)) * 100 : 100) + "%";
  };
  window.addEventListener("scroll", onScroll, { passive: true });
  onScroll();

  // ---- the end of the page draws itself when it comes into view
  const fin = document.querySelector(".fin");
  if (fin && "IntersectionObserver" in window) {
    const fo = new IntersectionObserver((es) => { if (es.some((e) => e.isIntersecting)) { fin.classList.add("drawn"); fo.disconnect(); } });
    fo.observe(fin);
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
