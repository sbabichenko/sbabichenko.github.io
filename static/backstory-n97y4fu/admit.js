// /backstory, "The gate": which candidates a round admits, with the false discovery level q on a slider.
// The data come from backstory/tools/origin_slides.py (simulated scores, with the gate's own treatment already applied):
// each candidate's histogram bin, its local false discovery rate, whether it was really a signal, and its score, sorted
// best first. The rule is the estimator's: the longest run, best first, whose average rate stays at or below q.
//   grey bars   every candidate, by score;   dark curve   the noise, read off the centre of the scores
//   blue        admitted, and real;          amber        admitted, and noise (knowable here only because it is simulated)
(function () {
  "use strict";
  const el = document.getElementById("gt-data"), sl = document.getElementById("gt-q");
  if (!el || !sl) return;
  const D = JSON.parse(el.textContent), f = (v) => v.toFixed(1), $ = (id) => document.getElementById(id);
  function draw() {
    const q = +sl.value; let cum = 0, k = 0;
    D.c.forEach((c, i) => { cum += c[1]; if (cum / (i + 1) <= q) k = i + 1; });
    const good = new Map(), bad = new Map(); let nb = 0, lo = Infinity, hi = -Infinity;   // lo, hi: the admitted scores nearest the centre
    for (let i = 0; i < k; ++i) {
      const [b, , real, z] = D.c[i];
      (real ? good : bad).set(b, ((real ? good : bad).get(b) || 0) + 1); if (!real) ++nb;
      if (z > 0) lo = Math.min(lo, z); else hi = Math.max(hi, z);
    }
    let g = "";
    for (const b of new Set([...good.keys(), ...bad.keys()])) {
      const x = D.x0 + D.bw * b + 1, w = D.bw - 2, ng = good.get(b) || 0, nn = bad.get(b) || 0;
      if (ng) g += `<rect class="or-adm" x="${f(x)}" y="${f(D.base - D.unit * ng)}" width="${f(w)}" height="${f(D.unit * ng)}"/>`;
      if (nn) g += `<rect class="or-adm false" x="${f(x)}" y="${f(D.base - D.unit * (ng + nn))}" width="${f(w)}" height="${f(D.unit * nn)}"/>`;
    }
    $("gt-adm").innerHTML = g;
    const X = (z) => D.x0 + D.bw * ((z - D.lo) / D.step);
    $("gt-cut").innerHTML = [hi, lo].filter(isFinite).map((z) => `<path class="or-cut" d="M${f(X(z))} ${D.lb + 6}V${D.base - 250}"/>`).join("");
    $("gt-qv").textContent = "q " + q.toFixed(2);
    $("gt-read").textContent = k ? `${k} admitted, ${nb} false positive${nb === 1 ? "" : "s"}` : "none admitted";
  }
  sl.addEventListener("input", draw);
  draw();
})();
