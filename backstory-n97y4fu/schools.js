// /backstory, "Eight schools" (Rubin 1981): partial pooling with the spread between schools on a slider.
//   top     each school's measured effect with its standard error (amber, the noise), and where the estimate lands after
//           pooling (blue);
//   bottom  the distribution of school effects that spread implies (purple, the analogue of the pool effect), with the
//           pooled estimates sitting under it.
// At zero spread every school gets the common average; as the spread grows each school keeps its own number.
(function () {
  "use strict";
  const top = document.getElementById("sch-top"), bot = document.getElementById("sch-bot"), sl = document.getElementById("sch-tau");
  if (!top || !sl) return;
  const Y = [28, 8, -3, 7, -1, 1, 18, 12], SE = [15, 10, 16, 11, 9, 11, 10, 18];
  const f = (v) => v.toFixed(1);
  const yTop = (v) => 150 - 3.1 * (v - 8);                       // effect, vertical, top drawing
  const xBot = (v) => 500 + 9.5 * (v - 8), BASE = 520, PEAK = 150; // effect, horizontal, bottom drawing
  function draw() {
    const tau = +sl.value, t2 = tau * tau;
    let a = 0, b = 0;
    Y.forEach((y, i) => { const w = 1 / (SE[i] * SE[i] + t2); a += w * y; b += w; });
    const mu = a / b, sh = Y.map((y, i) => mu + (t2 / (t2 + SE[i] * SE[i])) * (y - mu));
    let g = `<path class="or-mu" d="M70 ${f(yTop(mu))}H930"/>`;
    Y.forEach((y, i) => {
      const x = 140 + i * 103, e = SE[i];
      g += `<path class="or-bar" d="M${x} ${f(yTop(y - e))}V${f(yTop(y + e))}M${x - 9} ${f(yTop(y - e))}h18M${x - 9} ${f(yTop(y + e))}h18"/>` +
        `<path class="or-pull" d="M${x} ${f(yTop(y))}V${f(yTop(sh[i]))}"/><circle class="or-raw" cx="${x}" cy="${f(yTop(y))}" r="9"/><circle class="or-shrunk" cx="${x}" cy="${f(yTop(sh[i]))}" r="9"/>`;
    });
    top.innerHTML = g;
    // the bell is drawn at one height whatever its width, so only its width changes with the slider
    let h = `<path class="or-axis" d="M150 ${BASE}H850"/>`;
    if (tau < 0.15) h += `<path class="or-bell line" d="M${f(xBot(mu))} ${BASE}V${BASE - PEAK}"/>`;
    else {
      let d = `M150 ${BASE}`;
      for (let x = 150; x <= 850; x += 2) { const v = 8 + (x - 500) / 9.5, z = (v - mu) / tau; d += `L${x} ${f(BASE - PEAK * Math.exp(-0.5 * z * z))}`; }
      h += `<path class="or-bell" d="${d}L850 ${BASE}Z"/>`;
    }
    h += sh.map((v) => `<circle class="or-shrunk" cx="${f(xBot(v))}" cy="${BASE}" r="6"/>`).join("");
    bot.innerHTML = h;
  }
  sl.addEventListener("input", draw);
  sl.addEventListener("change", () => sl.blur());   // hand the arrow keys back to the screens
  draw();
})();
