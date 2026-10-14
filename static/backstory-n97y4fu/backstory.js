// /backstory: the bar. Tabs, jumps within the Decision Mesh page, the gate story's steps as dots, framed pages, keys.
//   1 2 3        the three tabs
//   ← →          the previous / next main screen (Where it started, Dissertation)
//   ↓ ↑          down to the screen under this one, where there is one, and back up
//   space, page keys, home, end   also step the screens (a clicker sends the page keys); on the race
//                screen, space plays and pauses instead and escape closes an enlarged drawing
//                and on a Dissertation screen whose drawing runs, space plays and pauses it (again from the start once it is done)
//   f            full screen
//   g            a board of every screen's title in this tab; click one to go there
//   t            light or dark
//   a d          on a screen with exactly one slider: move it left and right
//   w s          on "Analyzing fit": the chosen vertex at its coarser / finer scale
(function () {
  "use strict";
  const $ = (s, r) => (r || document).querySelector(s), $$ = (s, r) => [...(r || document).querySelectorAll(s)];
  const TABS = ["origin", "diss", "work"].filter((t) => document.getElementById("pane-" + t));   // a build may leave the last one out; the site's Decision Mesh page (#pane-mesh) stays hidden: it only lends its parts
  const bar = $("#iv-bar"), native = $("#mesh-native");
  const frames = { mesh: $("#frame-mesh") };
  const scrollAt = { origin: 0, mesh: 0, diss: 0, work: 0 };
  let tab = null, meshSeen = false;

  // ---- tabs: a hidden tab keeps its state (the fit, the framed page) and its scroll position
  function show(name) {
    if (!TABS.includes(name) || name === tab) return;
    if (tab) scrollAt[tab] = window.scrollY;
    tab = name;
    lendRace(name === "origin" && !!raceSlot && O.slides[O.oi].contains(raceSlot));
    lendGate(name === "origin" && !!gateSlot && O.slides[O.oi].contains(gateSlot));
    for (const d of Object.values(decks)) d.showBoard(false);
    for (const t of TABS) $("#pane-" + t).hidden = t !== name;
    for (const b of $$(".iv-tabs button")) b.setAttribute("aria-selected", b.dataset.tab === name);
    for (const j of $$(".iv-jumps")) j.hidden = j.dataset.for !== name;
    document.body.dataset.tab = name;
    if (decks[name]) decks[name].shown();
    if (name === "work" && window.Flow) window.Flow.shown();
    if (name === "origin" && window.Ginnie) window.Ginnie.shown();
    // the mesh page drew into hidden canvases at load: its own resize handler redraws them at their real size
    if (name === "mesh" && !meshSeen) { meshSeen = true; window.dispatchEvent(new Event("resize")); }
    window.scrollTo(0, scrollAt[name]);
    try { sessionStorage.setItem("iv-tab", name); } catch (e) {}
  }
  for (const b of $$(".iv-tabs button")) b.addEventListener("click", () => show(b.dataset.tab));

  // ---- a page of the site in a frame, without its own header and footer (same origin, so the frame can be styled)
  function frame(which, url) {
    const f = frames[which];
    f.hidden = false;
    if (which === "mesh") native.hidden = true;
    if (f.dataset.url !== url) { f.dataset.url = url; f.src = url; }
    for (const a of $$(`.iv-jumps[data-for="${which}"] a`)) a.classList.toggle("on", a.dataset.frame === url);
    window.scrollTo(0, 0);
  }
  for (const f of $$("iframe.iv-frame")) f.addEventListener("load", () => {
    try {
      const d = f.contentDocument, st = d.createElement("style");
      // (a framed page is being talked over: its idle doodles stay away and its animations run once)
      st.textContent = "header.blur-header, header.header, footer.site-foot { display: none !important; } html { scroll-padding-top: 0 !important; } .whimsy-doodle { display: none !important; } *, *::before, *::after { animation-iteration-count: 1 !important; }";
      d.head.appendChild(st);
      d.documentElement.classList.toggle("dark", document.documentElement.classList.contains("dark"));
    } catch (e) {}
  });

  // ---- jumps within the Decision Mesh page
  const barH = () => bar.offsetHeight;
  function jump(sel) {
    if (tab === "mesh" && native.hidden) { native.hidden = false; frames.mesh.hidden = true; for (const a of $$('.iv-jumps[data-for="mesh"] a')) a.classList.remove("on"); }
    const el = $(sel);
    if (el) window.scrollTo({ top: el.getBoundingClientRect().top + window.scrollY - barH() - 12 });
  }
  for (const a of $$(".iv-jumps a[data-view]")) a.addEventListener("click", () => window.Flow && window.Flow.setView(a.dataset.view));
  for (const a of $$(".iv-jumps a:not([data-view]), a[data-goto]")) a.addEventListener("click", () => {
    if (a.dataset.goto) { show(a.dataset.goto); requestAnimationFrame(() => jump(a.dataset.jump)); }
    else if (a.dataset.jump) jump(a.dataset.jump);
    else frame(a.closest(".iv-jumps").dataset.for, a.dataset.frame);
  });

  // ---- the gate story's steps as dots: click one to land on that step with its drawing fully built (the story reads
  // its progress off the scroll: a step is complete once 70% of it has passed the line 55% down the window)
  const steps = $$("#story .step"), dots = $("#iv-dots");
  const NAMES = { sites: "the data", funnel: "variance from the mean", wrongmean: "error in the mean", coherent: "two kinds of overdispersion",
    coarse: "the starting mesh", surplus: "surplus and tent", score: "scoring candidates", corrections: "three corrections", family: "the null",
    lfdr: "local false discovery rate", admit: "admission", rounds: "rounds", variance: "the two variances", done: "the fit" };
  function toStep(i) {
    const s = steps[Math.max(0, Math.min(steps.length - 1, i))];
    if (!s) return;
    if (native.hidden) jump("#how");
    const r = s.getBoundingClientRect();
    window.scrollTo({ top: window.scrollY + r.top - (window.innerHeight * 0.55 - r.height * 0.75) });
  }
  steps.forEach((s, i) => {
    const b = document.createElement("button");
    b.textContent = i + 1; b.title = NAMES[s.dataset.scene] || s.dataset.scene;
    b.addEventListener("click", () => toStep(i));
    dots.appendChild(b);
  });
  const current = () => {
    const line = window.innerHeight * 0.55;
    let best = -1, bd = Infinity;
    steps.forEach((s, i) => { const r = s.getBoundingClientRect(), d = Math.abs(r.top + r.height / 2 - line); if (d < bd) { bd = d; best = i; } });
    const story = $("#story").getBoundingClientRect();
    return story.top < line && story.bottom > line ? best : -1;
  };
  function mark() {
    if (tab !== "mesh" || native.hidden) return;
    const c = current();
    [...dots.children].forEach((b, i) => b.classList.toggle("on", i === c));
  }
  window.addEventListener("scroll", mark, { passive: true });

  // ---- the live race of geometries belongs to the Decision Mesh tab (it is the site's own "Why right triangles").
  // The first tab borrows it for one screen: the whole element moves into that screen's slot and back again, so the one
  // running race, with its state, is what both places show.
  const race = $("#race"), raceSlot = $("#or-race-slot"), raceHome = document.createComment("race");
  if (race) race.before(raceHome);
  function lendRace(out) {
    if (!race || !raceSlot || (race.parentNode === raceSlot) === out) return;
    if (!out) solo(null);
    if (out) raceSlot.appendChild(race); else raceHome.after(race);
    window.dispatchEvent(new Event("resize"));   // its canvases redraw at the size they now have
  }
  // The coin demo (the site's live fit on coins with known odds) is lent the same way, to the screen under "The gate".
  // It drew into canvases with no size while out of sight, so it is asked to redraw when it arrives.
  const gate = $("#gate"), gateSlot = $("#or-gate-slot"), gateHome = document.createComment("gate");
  if (gate) gate.before(gateHome);
  function lendGate(out) {
    if (!gate || !gateSlot || (gate.parentNode === gateSlot) === out) return;
    if (out) gateSlot.appendChild(gate); else gateHome.after(gate);
    window.dispatchEvent(new Event("resize"));
  }
  // its things to try: the demo reads its settings from the address's hash, as on the site
  for (const b of $$(".or-ctl.tries button")) b.addEventListener("click", () => { location.hash = b.dataset.try; b.blur(); });

  // For presenting it: a choice of the surface under the noise, a faster pace, a chart of each fit's error as the fits
  // grow (the race's own script draws it once a canvas with this id exists), and any drawing enlarged on a click.
  const raceStage = race && $(".stage", race);
  function solo(fig) {
    if (!raceStage) return;
    for (const g of $$("figure", raceStage)) g.classList.toggle("solo", g === fig);
    raceStage.classList.toggle("has-solo", !!fig);
    window.dispatchEvent(new Event("resize"));
  }
  if (raceStage) {
    const fig = document.createElement("figure"); fig.className = "curvefig";
    fig.innerHTML = '<div class="cv"><canvas id="race-cvcurve"></canvas></div><figcaption><b>Error as the fits grow</b></figcaption>';
    raceStage.appendChild(fig);
    const sel = document.createElement("select"), fast = document.createElement("button");
    sel.id = "race-surface"; sel.title = "The surface under the noise"; fast.className = "secondary"; fast.textContent = "Faster"; fast.title = "Six steps at a time";
    $("#race-statusbar .buttons").prepend(sel, fast);
    const fill = setInterval(() => {   // the race's script arrives when its drawings first come into view
      const D = window.meshDemo; if (!D || !D.surfaces) return;
      clearInterval(fill);
      sel.innerHTML = Object.entries(D.surfaces).map(([k, n]) => `<option value="${k}">${n}</option>`).join(""); sel.value = D.get("surface");
    }, 300);
    sel.addEventListener("change", () => {   // a new surface waits for Play, like the first
      window.meshDemo.set("surface", sel.value); sel.blur();
      window.dispatchEvent(new Event("resize"));
    });
    fast.addEventListener("click", () => { const on = fast.classList.toggle("on"); fast.textContent = on ? "Slower" : "Faster"; if (window.meshDemo) window.meshDemo.set("speed", on ? 6 : 1); });
    raceStage.addEventListener("click", (ev) => {
      const f = ev.target.closest("figure");
      if (!f || f.classList.contains("boardfig") || race.parentNode !== raceSlot) return;
      solo(f.classList.contains("solo") ? null : f);
    });
  }

  // ---- a screen drawn in stages: its slider shows the one group whose data-stage matches, fading between them
  const stageOf = (sec) => { const sl = $("input.or-stage", sec); if (sl) for (const g of $$("[data-stage]", sec)) g.classList.toggle("on", g.dataset.stage === sl.value); };
  for (const sl of $$("input.or-stage")) {
    const sec = sl.closest(".or-slide");
    sl.addEventListener("input", () => stageOf(sec)); sl.addEventListener("change", () => sl.blur());
    stageOf(sec);
  }

  // ---- a deck: a tab told one drawing per screen. Left and right walk the main screens; some have more underneath,
  // reached with the down arrow and left with up, for going further only when the conversation asks for it. A "view" is
  // a screen, or a deeper layout of the same screen (data-modes on its section; the section then carries data-mode).
  // The heavy drawings are not in the page: each is fetched once, the current screen's first, then its neighbours, then
  // the rest while nothing else is happening, so the page opens at once and every screen is there when stepped to.
  const idle = window.requestIdleCallback || ((f) => setTimeout(f, 60));
  const tip = document.createElement("div"); tip.className = "iv-tip"; tip.hidden = true; document.body.appendChild(tip);
  function Deck(name, key, after) {
    const root = $("#pane-" + name), odots = $(`.iv-jumps[data-for="${name}"] .iv-toc`);
    const slides = $$(".or-slide", root), views = [], fetched = new Map(), obtn = [];
    slides.forEach((sec, si) => {
      views.push({ si, sec, mode: 0, name: sec.dataset.name, depth: +sec.dataset.depth || 0 });
      (sec.dataset.modes || "").split("|").filter(Boolean).forEach((nm, k) => views.push({ si, sec, mode: k + 1, name: nm, depth: (+sec.dataset.depth || 0) + k + 1 }));
    });
    const mains = views.map((v, i) => i).filter((i) => views[i].depth === 0);
    const mainOf = (i) => { while (views[i].depth) --i; return i; };
    const D = { name, slides, views, mains, mainOf, oi: 0, vi: 0 };   // oi: the section on screen; vi: the view
    function fill(i) {
      const sec = slides[i], svg = sec && $("svg[data-src]", sec);
      if (!svg) return Promise.resolve();
      if (!fetched.has(i)) fetched.set(i, fetch(svg.dataset.src).then((r) => r.text()).then((t) => { svg.innerHTML = t; stageOf(sec); }).catch(() => { fetched.delete(i); }));
      return fetched.get(i);
    }
    function fillRest(i) { if (i < slides.length) fill(i).then(() => idle(() => fillRest(i + 1))); }
    let filling = false;
    D.shown = () => { if (!filling) { filling = true; setTimeout(() => fillRest(0), 300); } };
    const cue = document.createElement("span"); cue.className = "or-cue";
    D.toSlide = function (i) {
      const vi = D.vi = Math.max(0, Math.min(views.length - 1, i)), v = views[vi], oi = D.oi = v.si;
      fill(oi); fill(oi + 1); fill(oi - 1);
      slides.forEach((s, k) => { s.hidden = k !== oi; });
      if (v.sec.dataset.modes) v.sec.dataset.mode = v.mode;
      const fr = $("iframe[data-url]", v.sec); if (fr && !fr.getAttribute("src")) fr.src = fr.dataset.url;   // a page of the site, loaded when first shown
      obtn.forEach((b, k) => { b.classList.toggle("on", k === vi); b.classList.toggle("in", views[k].depth > 0 && mainOf(k) === mainOf(vi)); });
      for (const g of odots.children) g.classList.toggle("on", g.dataset.group === slides[oi].dataset.group);
      // beside the title: what lies one level down, or the way back up
      const below = views[vi + 1] && views[vi + 1].depth > v.depth ? views[vi + 1] : null;
      cue.innerHTML = (v.depth ? `<span>↑ ${views[vi - 1].name === v.name ? "" : views[vi - 1].name}</span>` : "") + (below ? `<span>↓ ${below.name}</span>` : "");
      $(".or-head h2", v.sec).after(cue);
      try { sessionStorage.setItem(key, vi); } catch (e) {}
      if (after) after(v);
    };
    D.across = (d) => { const m = mains.indexOf(mainOf(D.vi)) + d; if (m >= 0 && m < mains.length) D.toSlide(mains[m]); };
    D.deeper = (d) => { const n = views[D.vi + d]; if (d > 0 ? n && n.depth > views[D.vi].depth : views[D.vi].depth > 0) D.toSlide(D.vi + d); };
    cue.addEventListener("click", (ev) => { if (ev.target.textContent.startsWith("↑")) D.deeper(-1); else D.deeper(1); });
    // the bar's table of contents: each chapter's name, then a numbered dot for each main screen and a small dot for each
    // screen under it. A chapter's name goes to its first screen; pointing at a dot names its screen at once.
    views.forEach((v, i) => {
      let g = odots.lastElementChild; const group = v.sec.dataset.group;
      if (!g || g.dataset.group !== group) {
        g = document.createElement("span"); g.className = "grp"; g.dataset.group = group;
        const a = document.createElement("a"); a.textContent = group; a.addEventListener("click", () => D.toSlide(i));
        g.appendChild(a); odots.appendChild(g);
      }
      const b = document.createElement("button");
      if (v.depth) b.className = "sub"; else b.textContent = mains.indexOf(i) + 1;
      b.dataset.name = v.name; b.setAttribute("aria-label", v.name);
      b.addEventListener("click", () => { D.toSlide(i); tip.hidden = true; });
      b.addEventListener("mouseenter", () => { const r = b.getBoundingClientRect(); tip.textContent = (v.depth ? "↓ " : "") + v.name; tip.hidden = false; tip.style.left = Math.max(8, Math.min(window.innerWidth - tip.offsetWidth - 8, r.left + r.width / 2 - tip.offsetWidth / 2)) + "px"; });
      b.addEventListener("mouseleave", () => { tip.hidden = true; });
      g.appendChild(b); obtn.push(b);
    });
    // g: every screen's title on one board, to jump straight to one when a question comes
    const board = D.board = document.createElement("div"); board.className = "or-board"; board.hidden = true; board.dataset.deck = name;
    { let h = "", g = null, cols = 0;
      views.forEach((v, i) => { const grp = v.sec.dataset.group; if (grp !== g) { h += (g === null ? "" : "</div>") + `<div class="col"><h3>${grp}</h3>`; g = grp; ++cols; } h += `<button data-i="${i}"${v.depth ? ' class="sub"' : ""}><b>${v.depth ? "↓" : mains.indexOf(i) + 1}</b>${v.name}</button>`; });
      board.innerHTML = h + "</div>"; board.style.setProperty("--cols", cols); }
    document.body.appendChild(board);
    D.showBoard = (on) => { board.hidden = !on; if (on) for (const b of $$("button", board)) b.classList.toggle("on", +b.dataset.i === D.vi); };
    board.addEventListener("click", (ev) => { const b = ev.target.closest("button"); if (b) { show(name); D.toSlide(+b.dataset.i); } D.showBoard(false); });
    return D;
  }

  // ---- Where it started. Some of its screens borrow live parts of the site's Decision Mesh page, and "Analyzing fit"
  // shows one vertex of the real fit (star.js): it follows the data set chosen on the real-data screen, and a vertex
  // clicked on that screen's map opens here.
  const decks = {};
  const starKey = () => $("#gn-pools").value + "-" + $("#gn-month").value;
  const O = decks.origin = Deck("origin", "iv-slide", (v) => {
    lendRace(tab === "origin" && !!raceSlot && v.sec.contains(raceSlot));
    lendGate(tab === "origin" && !!gateSlot && v.sec.contains(gateSlot));
    if (O.vi === starView && window.Star) { window.Star.open(starKey()); requestAnimationFrame(() => window.Star.shown()); }
    if (tab === "origin" && window.Ginnie) window.Ginnie.shown();   // the live fit redraws at its real size when its screen comes up
  });
  const starView = O.views.findIndex((v) => v.sec.querySelector("#st"));
  window.addEventListener("gn-vertex", (ev) => { if (starView < 0 || !window.Star) return; window.Star.open(starKey(), ev.detail); O.toSlide(starView); });
  // ---- Dissertation: the same kind of deck; every slider on it shows one of several ready-drawn stages
  if ($("#pane-diss .or-slide")) decks.diss = Deck("diss", "iv-slide-diss");
  for (const [k, d] of Object.entries(decks)) { let at = 0; try { at = +sessionStorage.getItem(k === "origin" ? "iv-slide" : "iv-slide-" + k) || 0; } catch (e) {} d.toSlide(at); }
  window.addEventListener("load", () => O.shown());

  // the opening screen: its slider carries every pool to its cohort's average pool and back
  { const m = $("#co-merge"), g = $(".or-pools"); if (m && g) m.addEventListener("input", () => g.style.setProperty("--p", m.value)); }

  // The Decision Mesh tab's live fit no longer colours its dots by round (this page's copy of gate.js): say what they are
  { const cap = $$("#mesh-native figcaption span").find((e) => /admitted, by round/.test(e.textContent)); if (cap) cap.textContent = "dots: admitted vertices"; }

  // a slider lets go of the keyboard as soon as the pointer does, so the arrows go back to stepping screens
  document.addEventListener("pointerup", (ev) => { if (ev.target && ev.target.matches && ev.target.matches('input[type="range"]')) ev.target.blur(); });

  // ---- theme: the site's own button does the work (it is in the page, hidden); frames follow
  $("#iv-theme").addEventListener("click", () => {
    const b = $(".btn-dark"); if (b) b.click();
    setTimeout(() => { for (const f of $$("iframe.iv-frame")) try { f.contentDocument.documentElement.classList.toggle("dark", document.documentElement.classList.contains("dark")); } catch (e) {} }, 50);
  });

  // ---- keys (never while typing in a control)
  window.addEventListener("keydown", (ev) => {
    if (ev.metaKey || ev.ctrlKey || ev.altKey) return;
    const t = ev.target;
    if (t && (t.isContentEditable || /^(INPUT|SELECT|TEXTAREA)$/.test(t.tagName))) return;
    if (ev.key >= "1" && ev.key <= "3") { show(TABS[+ev.key - 1]); return; }
    if (ev.key === "t") { $("#iv-theme").click(); return; }
    const D = decks[tab];
    if (ev.key === "g" && D) { D.showBoard(D.board.hidden); return; }
    if (D && !D.board.hidden) { if (ev.key === "Escape") D.showBoard(false); return; }
    if (ev.key === "f") { if (document.fullscreenElement) document.exitFullscreen(); else document.documentElement.requestFullscreen().catch(() => {}); return; }
    if (tab === "work" && (ev.key === "ArrowRight" || ev.key === "ArrowLeft") && window.Flow) { window.Flow.setView(ev.key === "ArrowRight" ? 1 : -1); ev.preventDefault(); return; }
    if (D) {
      if (tab === "origin" && (ev.key === "w" || ev.key === "s") && O.vi === starView && window.Star) { window.Star.role(ev.key === "w" ? -1 : 1); return; }   // one vertex, coarser or finer
      if (ev.key === "a" || ev.key === "d") {   // on a screen with one slider, a and d move it (about a twenty-fifth of its run a press)
        const sl = $$(".or-slide:not([hidden]) input[type=range]").filter((x) => x.offsetParent);
        if (sl.length === 1) {
          const x = sl[0], st = +x.step || 1, by = Math.max(st, Math.round((+x.max - +x.min) / 25 / st) * st);
          x.value = Math.max(+x.min, Math.min(+x.max, +x.value + (ev.key === "d" ? by : -by))); x.dispatchEvent(new Event("input", { bubbles: true }));
        }
        return;
      }
      const onRace = tab === "origin" && race && race.parentNode === raceSlot;
      if (onRace && ev.key === " ") { $("#race-playbtn").click(); ev.preventDefault(); return; }             // space plays and pauses the race
      if (onRace && ev.key === "Escape") { solo(null); return; }
      if (tab === "diss" && ev.key === " " && window.DissPlay && window.DissPlay()) { ev.preventDefault(); return; }   // space plays and pauses a drawing that runs
      const step = { ArrowRight: 1, PageDown: 1, " ": 1, ArrowLeft: -1, PageUp: -1 }[ev.key];             // a clicker sends the page keys
      if (step) { D.across(step); ev.preventDefault(); return; }
      if (ev.key === "ArrowDown" || ev.key === "ArrowUp") { D.deeper(ev.key === "ArrowDown" ? 1 : -1); ev.preventDefault(); return; }
      if (ev.key === "Home" || ev.key === "End") { D.toSlide(ev.key === "Home" ? 0 : D.mains[D.mains.length - 1]); ev.preventDefault(); return; }
    }
    if (tab === "mesh" && (ev.key === "ArrowRight" || ev.key === "ArrowLeft")) {
      const c = current(), fwd = ev.key === "ArrowRight";
      toStep(c < 0 ? (fwd ? 0 : steps.length - 1) : c + (fwd ? 1 : -1));
      ev.preventDefault();
    }
  });

  let saved = null;
  try { saved = sessionStorage.getItem("iv-tab"); } catch (e) {}
  show(TABS.includes(saved) ? saved : "origin");
})();
