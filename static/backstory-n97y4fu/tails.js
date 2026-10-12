// /backstory, "Binomial noise": the chance of each count of terminations in a pool, live. Colours are the equation's.
//   dark line   the normal with the plain binomial's mean and variance: what a calibration that assumes normal noise expects;
//   amber       the binomial: its bars on the left, and on the right the upper tail it has beyond the normal's;
//   purple      the same pool once each pool's own rate is spread around the average (tau > 0, normal on the log-odds,
//               the average rate held where the slider puts it): outlined bars on the left, a further tail on the right;
//   grey        left, the normal's weight below zero, where no count can be.
// Left is a plain scale, right a log scale. Counts are placed by standard deviations from the mean, so the normal keeps
// one shape whatever the sliders say.
(function () {
  "use strict";
  const lin = document.getElementById("tl-lin"), log = document.getElementById("tl-log");
  if (!lin) return;
  const $ = (id) => document.getElementById(id), f = (v) => v.toFixed(1);
  const LG = [76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5];
  function lgamma(x) { let y = x, t = x + 5.5, s = 1.000000000190015; t -= (x + 0.5) * Math.log(t); for (let j = 0; j < 6; ++j) s += LG[j] / ++y; return -t + Math.log(2.5066282746310005 * s / x); }
  const expit = (t) => 1 / (1 + Math.exp(-t));
  // the pool effect: 41 points of a standard normal, scaled by tau on the log-odds
  const TH = [], TW = []; { let sw = 0; for (let i = -20; i <= 20; ++i) { const x = i * 0.2; TH.push(x); TW.push(Math.exp(-0.5 * x * x)); sw += TW[TW.length - 1]; } for (let i = 0; i < TW.length; ++i) TW[i] /= sw; }
  function rates(rate, tau) {
    if (tau < 1e-6) return [[rate, 1]];
    let lo = -14, hi = 6, mid = 0;
    for (let it = 0; it < 50; ++it) { mid = (lo + hi) / 2; let m = 0; for (let i = 0; i < TH.length; ++i) m += TW[i] * expit(mid + tau * TH[i]); if (m < rate) lo = mid; else hi = mid; }
    return TH.map((x, i) => [expit(mid + tau * x), TW[i]]);
  }
  function pmf(k, n, ps) {   // k may be any real number from 0 to n: the curves are drawn through the gamma function
    const c = lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1); let s = 0;
    for (const [p, w] of ps) s += w * Math.exp(c + k * Math.log(p) + (n - k) * Math.log(1 - p));
    return s;
  }
  const ZLO = -4.6, ZHI = 8.2, TOP = 44, BASE = 470, LLO = -7;
  const A = { x: 60, w: 400 }, B = { x: 585, w: 375 };
  function draw() {
    const rate = Math.pow(10, +$("tl-rate").value), n = Math.round(Math.pow(10, +$("tl-n").value)), tau = +$("tl-tau").value;
    $("tl-rate-v").textContent = (100 * rate).toFixed(rate < 0.1 ? 1 : 0) + "%"; $("tl-n-v").textContent = n.toLocaleString(); $("tl-tau-v").textContent = tau.toFixed(2);
    const mu = n * rate, sd = Math.sqrt(n * rate * (1 - rate)), plain = [[rate, 1]], pooled = tau > 1e-6 ? rates(rate, tau) : null;
    const k0 = Math.max(0, Math.ceil(mu + ZLO * sd)), k1 = Math.min(n, Math.floor(mu + ZHI * sd)), ks = [];
    for (let k = k0; k <= k1; ++k) ks.push(k);
    const pb = ks.map((k) => pmf(k, n, plain)), pu = pooled ? ks.map((k) => pmf(k, n, pooled)) : null;
    const norm = (k) => Math.exp(-0.5 * ((k - mu) / sd) ** 2) / (sd * 2.5066282746310002);
    const X = (P, k) => P.x + P.w * ((k - mu) / sd - ZLO) / (ZHI - ZLO), step = A.w / ((ZHI - ZLO) * sd);
    const zs = []; for (let i = 0; i <= 220; ++i) zs.push(mu + sd * (ZLO + (ZHI - ZLO) * i / 220));
    const kc = zs.filter((k) => k >= 0 && k <= n);   // where a count can be, for the smooth curves
    // ---- plain scale
    const top = Math.max(...pb, ...(pu || [0]), norm(mu)), YL = (q) => BASE - (BASE - TOP) * q / top;
    let g = "";
    const neg = zs.filter((k) => k <= -0.5);
    if (neg.length > 1) g += `<polygon class="or-below" points="${neg.map((k) => f(X(A, k)) + "," + f(YL(norm(k)))).join(" ")} ${f(X(A, -0.5))},${BASE} ${f(X(A, neg[0]))},${BASE}"/>`;
    const bw = Math.max(1.2, Math.min(26, step * 0.72));
    const bars = (ps, cls) => ks.map((k, i) => `<rect class="${cls}" x="${f(X(A, k) - bw / 2)}" y="${f(YL(ps[i]))}" width="${f(bw)}" height="${f(BASE - YL(ps[i]))}"/>`).join("");
    g += bars(pb, "or-pmf") + (pu ? bars(pu, "or-pmfu") : "");
    g += `<path class="or-normal" d="M${zs.map((k) => f(X(A, k)) + " " + f(YL(norm(k)))).join("L")}"/><path class="or-axis" d="M${A.x} ${BASE}H${A.x + A.w}"/>`;
    if (mu + ZLO * sd < -0.5) g += `<path class="or-zero" d="M${f(X(A, -0.5))} ${BASE + 6}V${TOP}"/>`;
    lin.innerHTML = g;
    // ---- log scale
    const YG = (q) => BASE - (BASE - TOP) * (Math.max(Math.log10(Math.max(q, 1e-300)), LLO - 3) - LLO) / (0 - LLO);
    let h = `<clipPath id="tl-clip"><rect x="${B.x}" y="${TOP - 8}" width="${B.w}" height="${BASE - TOP + 8}"/></clipPath>`;
    for (const [e, t] of [[-2, "1 in 100"], [-4, "1 in 10,000"], [-6, "1 in a million"]]) h += `<path class="or-grid" d="M${B.x} ${f(YG(10 ** e))}H${B.x + B.w}"/><text class="or-ax" x="${B.x - 10}" y="${f(YG(10 ** e) + 5)}" text-anchor="end">${t}</text>`;
    h += `<g clip-path="url(#tl-clip)">`;
    const cb = kc.map((k) => pmf(k, n, plain)), cu = pooled ? kc.map((k) => pmf(k, n, pooled)) : null;
    const wedge = (upper, lower, cls) => {   // the part of the upper tail (above the mean) where `upper` sits above `lower`
      const idx = kc.map((k, i) => i).filter((i) => kc[i] > mu && upper[i] > lower[i]);
      return idx.length > 1 ? `<polygon class="or-wedge ${cls}" points="${idx.map((i) => f(X(B, kc[i])) + "," + f(YG(upper[i]))).join(" ")} ${idx.slice().reverse().map((i) => f(X(B, kc[i])) + "," + f(YG(lower[i]))).join(" ")}"/>` : "";
    };
    const cn = kc.map(norm);
    h += wedge(cb, cn, "noise") + (cu ? wedge(cu, cb, "pool") : "");
    h += `<path class="or-normal" d="M${zs.map((k) => f(X(B, k)) + " " + f(YG(norm(k)))).join("L")}"/>`;
    const line = (c, cls) => `<path class="or-tail ${cls}" d="M${kc.map((k, i) => f(X(B, k)) + " " + f(YG(c[i]))).join("L")}"/>`;
    const dots = (ps, cls) => (ks.length <= 60 ? ks.map((k, i) => `<circle class="or-pt ${cls}" cx="${f(X(B, k))}" cy="${f(YG(ps[i]))}" r="${ks.length > 30 ? 3.2 : 4.8}"/>`).join("") : "");
    h += line(cb, "noise") + (cu ? line(cu, "pool") : "") + dots(pb, "noise") + (pu ? dots(pu, "pool") : "");
    h += `</g><path class="or-axis" d="M${B.x} ${BASE}H${B.x + B.w}"/>`;
    log.innerHTML = h;
  }
  // each slider is a symbol of the equation at the top right: sigma(.) the rate, n the loans, tau the pool effect. Pointing
  // at a slider, or moving it, lights that symbol there.
  for (const lab of document.querySelectorAll(".or-ctl.multi label[data-sym]")) {
    const lit = (on) => { for (const e of document.querySelectorAll(`#tl-eq [data-sym="${lab.dataset.sym}"]`)) e.classList.toggle("lit", on); lab.classList.toggle("lit", on); };
    lab.addEventListener("pointerenter", () => lit(true)); lab.addEventListener("pointerleave", () => lit(false));
    lab.addEventListener("focusin", () => lit(true)); lab.addEventListener("focusout", () => lit(false));
  }
  for (const id of ["tl-rate", "tl-n", "tl-tau"]) { $(id).addEventListener("input", draw); $(id).addEventListener("change", () => $(id).blur()); }
  draw();
})();
