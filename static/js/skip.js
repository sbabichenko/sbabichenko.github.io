// A way past the scroll-told parts. A page marks a long drawn section with
//   <a class="skip-pill" href="#target" data-from="#first" data-until="#last">Skip to …</a>
// and while the reader is inside that stretch, with more than about a screen of it left, the pill floats at the
// bottom of the window. Pressing it jumps straight to the target, without animating through the drawings.
(function () {
  "use strict";
  document.querySelectorAll(".skip-pill").forEach((pill) => {
    const from = document.querySelector(pill.dataset.from), until = document.querySelector(pill.dataset.until || pill.dataset.from);
    const target = document.querySelector(pill.getAttribute("href"));
    if (!from || !until || !target) return;
    document.body.appendChild(pill);   // out of any sticky or transformed parent, so it stays fixed to the window
    function check() {
      const vh = innerHeight, a = from.getBoundingClientRect(), b = until.getBoundingClientRect();
      pill.classList.toggle("on", a.top < vh * 0.6 && b.bottom > vh * 2.1);
    }
    pill.addEventListener("click", (e) => {
      e.preventDefault();
      if (!target.hasAttribute("tabindex")) target.setAttribute("tabindex", "-1");
      target.scrollIntoView({ behavior: "instant", block: "start" });
      target.focus({ preventScroll: true });
      history.replaceState(null, "", pill.getAttribute("href"));
      check();
    });
    addEventListener("scroll", check, { passive: true });
    addEventListener("resize", check);
    check();
  });
})();
