// A bookmark ribbon for the dissertation. Chapter pages remember the section being read (in this browser only);
// the contents, on the chapter pages' rail and on the dissertation's front page, hang a ribbon on that chapter,
// and the ribbon's link goes straight back to the section.
(function () {
  "use strict";
  const KEY = "sb-bookmark";
  const read = () => { try { return JSON.parse(localStorage.getItem(KEY) || "null"); } catch (e) { return null; } };
  const here = location.pathname.replace(/\/?$/, "/");

  // on a chapter page: the section whose heading last passed the upper third of the window
  const secs = [...document.querySelectorAll(".rail .secs a[data-sec]")].map((a) => document.getElementById(a.dataset.sec)).filter(Boolean);
  if (document.getElementById("paper")) {
    let t = 0;
    const save = () => {
      if (scrollY < 400) return;
      let sec = null;
      for (const h of secs) if (h.getBoundingClientRect().top < innerHeight / 3) sec = h.id;
      try { localStorage.setItem(KEY, JSON.stringify({ path: here, sec })); } catch (e) { /* storage off: no ribbon */ }
      mark();
    };
    addEventListener("scroll", () => { clearTimeout(t); t = setTimeout(save, 400); }, { passive: true });
  }

  function mark() {
    const b = read();
    document.querySelectorAll(".ribbon").forEach((el) => el.classList.remove("ribbon"));
    if (!b) return;
    const entries = document.querySelectorAll(".rail .toc > li > a, .contents .cards a");
    for (const a of entries) {
      const u = new URL(a.href, location.href);
      if (u.pathname.replace(/\/?$/, "/") !== b.path) continue;
      a.classList.add("ribbon");
      a.title = "Where you stopped reading";
      a.href = b.path + (b.sec ? "#" + b.sec : "");
    }
  }
  mark();
})();
