// The eight schools, pooled by the post's own formula: each school's estimate is pulled toward the shared mean
// by w_j = sigma_j^2 / (sigma_j^2 + tau^2), the mean itself weighted by 1 / (sigma_j^2 + tau^2). Drag tau: at 0
// every school gets the fully pooled 7.68; as it grows, each keeps more of its own y_j.
(function () {
  "use strict";
  const box = document.getElementById("pooltoy");
  if (!box) return;
  const Y = [28, 8, -3, 7, -1, 1, 18, 12], S = [15, 10, 16, 11, 9, 11, 10, 18];
  const NS = "http://www.w3.org/2000/svg";
  const el = (tag, attrs, parent) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); if (parent) parent.appendChild(e); return e; };
  const W = 600, H = 260, L = 44, R = 16, T = 14, B = 34, lo = -22, hi = 48;
  const x = (j) => L + (j + 0.5) * (W - L - R) / 8, y = (v) => T + (hi - v) / (hi - lo) * (H - T - B);
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, class: "pt-svg", role: "img", "aria-label": "Eight school estimates pulled toward their shared mean" }, box);
  for (let v = -20; v <= 40; v += 20) {
    el("line", { x1: L, x2: W - R, y1: y(v), y2: y(v), class: "pt-grid" }, svg);
    el("text", { x: L - 8, y: y(v) + 4, class: "pt-tick", "text-anchor": "end" }, svg).textContent = v;
  }
  const mean = el("line", { x1: L, x2: W - R, class: "pt-mean" }, svg);
  const meanLab = el("text", { x: L + 6, class: "pt-meanlab", "text-anchor": "start" }, svg);
  const pts = Y.map((yj, j) => {
    el("line", { x1: x(j), x2: x(j), y1: y(yj - S[j]), y2: y(yj + S[j]), class: "pt-bar" }, svg);
    el("circle", { cx: x(j), cy: y(yj), r: 4.5, class: "pt-raw" }, svg);
    const pull = el("line", { x1: x(j), x2: x(j), y1: y(yj), class: "pt-pull" }, svg);
    const dot = el("circle", { cx: x(j), r: 5, class: "pt-dot" }, svg);
    el("text", { x: x(j), y: H - 12, class: "pt-tick", "text-anchor": "middle" }, svg).textContent = j + 1;
    return { pull, dot };
  });
  const row = document.createElement("label");
  row.className = "pt-row";
  row.innerHTML = '<span class="pt-tau">&tau; = <b>10.0</b></span><input type="range" min="0" max="30" step="0.5" value="10" aria-label="tau, how much the schools truly differ"><span class="pt-note"></span>';
  box.appendChild(row);
  const input = row.querySelector("input"), tauOut = row.querySelector("b"), note = row.querySelector(".pt-note");
  function update() {
    const tau = +input.value, t2 = tau * tau;
    let num = 0, den = 0;
    for (let j = 0; j < 8; ++j) { const p = 1 / (S[j] * S[j] + t2); num += Y[j] * p; den += p; }
    const mu = num / den;
    mean.setAttribute("y1", y(mu)); mean.setAttribute("y2", y(mu));
    meanLab.setAttribute("y", y(mu) - 5); meanLab.textContent = `shared mean ${(Math.floor(mu * 100) / 100).toFixed(2)}`;   // truncated, as the post prints it (7.68)
    pts.forEach(({ pull, dot }, j) => {
      const w = S[j] * S[j] / (S[j] * S[j] + t2), th = (1 - w) * Y[j] + w * mu;
      dot.setAttribute("cy", y(th)); pull.setAttribute("y2", y(th));
    });
    tauOut.textContent = tau.toFixed(1);
    note.textContent = tau === 0 ? "full pooling: one estimate for every school" : tau >= 30 ? "close to no pooling" : "";
  }
  input.addEventListener("input", update);
  update();
})();
