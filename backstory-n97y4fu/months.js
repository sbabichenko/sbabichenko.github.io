// /backstory, "Finding model uncertainty": the question of the second project. Uncertainty here is the surface's, not a
// pool's: cohort by cohort, where is the model off, and how random is that surface from one month to the next?
// The big map is one month: each cohort (a cell) coloured by how far it came in from the model, red above and blue below.
// The strip under it is twelve months of the same map, and the slider (a and d, or a click on a month) picks one. Beside
// the map, two small ones gather the months up to the one shown: their average (where the model is off) and how much
// each cohort moves from month to month (how random the surface is). Nothing moves by itself.
// The terrain is a stand-in, as on the other screens: the smoothed map plays the model and the real ground (slides/
// spanish_peaks.u16) what happens on average; each month adds a smooth random surface of its own, larger on the high
// ground, and a little noise per cohort. All of it is drawn here from a fixed seed.
(function () {
  "use strict";
  const box = document.getElementById("mu"); if (!box) return;
  const inner = document.getElementById("mu-in"), cv = document.getElementById("mu-map"), ctx = cv.getContext("2d");
  const cAvg = document.getElementById("mu-avg"), cSd = document.getElementById("mu-sd"), cStrip = document.getElementById("mu-strip");
  const sl = document.getElementById("mu-t"), read = document.getElementById("mu-read");
  const SW = +box.dataset.w, SH = +box.dataset.h, GX = 30, GY = 15, T = 12, GAP = 22, SIDE = 0.3, THUMB = 16, FULL = 280, SPREAD = 150;
  let truth = null, model = null, miss = null, lines = null, W = 0, H = 0, dpr = 1, builtFor = "", cw = 0;
  const isDark = () => document.documentElement.classList.contains("dark");
  const css = (name) => getComputedStyle(box).getPropertyValue(name).trim();
  const clamp = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);
  function mulberry32(a) { return () => { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
  const at = (a, u, v) => {   // bilinear, u and v in [0, 1]
    const fx = u * (SW - 1), fy = v * (SH - 1), i = Math.min(SW - 2, Math.floor(fx)), j = Math.min(SH - 2, Math.floor(fy)), s = fx - i, t = fy - j, k = j * SW + i;
    return (a[k] * (1 - s) + a[k + 1] * s) * (1 - t) + (a[k + SW] * (1 - s) + a[k + SW + 1] * s) * t;
  };

  // the model: three passes of a running mean each way, which is as good as the Gaussian the other screens' drawings use
  function smooth(a) {
    let cur = a; const R = 26;
    for (let pass = 0; pass < 3; ++pass) {
      const tmp = new Float32Array(SW * SH), out = new Float32Array(SW * SH);
      for (let y = 0; y < SH; ++y) for (let x = 0; x < SW; ++x) { let s = 0; for (let k = -R; k <= R; ++k) s += cur[y * SW + Math.max(0, Math.min(SW - 1, x + k))]; tmp[y * SW + x] = s / (2 * R + 1); }
      for (let y = 0; y < SH; ++y) for (let x = 0; x < SW; ++x) { let s = 0; for (let k = -R; k <= R; ++k) s += tmp[Math.max(0, Math.min(SH - 1, y + k)) * SW + x]; out[y * SW + x] = s / (2 * R + 1); }
      cur = out;
    }
    return cur;
  }

  // every cohort's miss in every month: what stays (the real ground less the model), that month's own surface, and noise
  function months() {
    const rnd = mulberry32(20201013), gauss = () => Math.sqrt(-2 * Math.log(1 - rnd())) * Math.cos(2 * Math.PI * rnd());
    const stay = new Float32Array(GX * GY), swing = new Float32Array(GX * GY), noise = new Float32Array(GX * GY);
    for (let j = 0; j < GY; ++j) for (let i = 0; i < GX; ++i) {
      let s = 0, z = 0, n = 0;
      for (let b = 0; b < 6; ++b) for (let a = 0; a < 6; ++a) { const u = (i + (a + 0.5) / 6) / GX, v = (j + (b + 0.5) / 6) / GY; s += at(truth, u, v) - at(model, u, v); z += at(truth, u, v); ++n; }
      stay[j * GX + i] = s / n; swing[j * GX + i] = 40 + 130 * clamp((z / n - 2850) / 1000); noise[j * GX + i] = 26 / Math.sqrt(Math.exp(0.7 * gauss()));
    }
    const KX = 8, KY = 5; miss = [];
    for (let t = 0; t < T; ++t) {
      const knots = Float32Array.from({ length: KX * KY }, gauss), m = new Float32Array(GX * GY);
      for (let j = 0; j < GY; ++j) for (let i = 0; i < GX; ++i) {
        const fx = ((i + 0.5) / GX) * (KX - 1), fy = ((j + 0.5) / GY) * (KY - 1), a = Math.min(KX - 2, Math.floor(fx)), b = Math.min(KY - 2, Math.floor(fy));
        const s = fx - a, q = fy - b, sm = (x) => x * x * (3 - 2 * x), ss = sm(s), qq = sm(q), k = b * KX + a;
        const r = (knots[k] * (1 - ss) + knots[k + 1] * ss) * (1 - qq) + (knots[k + KX] * (1 - ss) + knots[k + KX + 1] * ss) * qq;
        m[j * GX + i] = stay[j * GX + i] + swing[j * GX + i] * r + noise[j * GX + i] * gauss();
      }
      miss.push(m);
    }
  }

  function size(c, w, h) { c.style.width = w + "px"; c.style.height = h + "px"; c.width = Math.round(w * dpr); c.height = Math.round(h * dpr); }
  function build() {
    const r = box.getBoundingClientRect(), asp = (SW - 1) / (SH - 1);
    cw = Math.floor(Math.min((r.width - GAP) / (1 + SIDE), ((r.height - 48) / (1 + 1 / T)) * asp));
    const ch = Math.round(cw / asp), side = Math.round(cw * SIDE);
    if (cw < 60 || !truth) return false;
    dpr = Math.min(2, window.devicePixelRatio || 1);
    const key = cw + "x" + ch + "@" + dpr + isDark();
    if (key === builtFor) return true;
    builtFor = key;
    inner.style.gridTemplateColumns = cw + "px " + side + "px";
    size(cv, cw, ch); W = cv.width; H = cv.height;
    size(cAvg, side, Math.round(side / asp)); size(cSd, side, Math.round(side / asp)); size(cStrip, cw, Math.round((cw / T - 6) / asp) + 6);
    sl.style.width = (cw * (T - 1)) / T + THUMB + "px"; sl.style.marginLeft = cw / (2 * T) - THUMB / 2 + "px";
    // the model's own smooth contours, to lie faintly over the month: this is still the map
    const BW = Math.min(W, 900), BH = Math.round((BW * H) / W), z = new Float32Array(BW * BH);
    for (let y = 0; y < BH; ++y) for (let x = 0; x < BW; ++x) z[y * BW + x] = at(model, x / (BW - 1), y / (BH - 1));
    lines = document.createElement("canvas"); lines.width = BW; lines.height = BH;
    const lx = lines.getContext("2d"), im = lx.createImageData(BW, BH), d = im.data; lx.fillStyle = css("--text"); const h = lx.fillStyle, ink = [1, 3, 5].map((k) => parseInt(h.slice(k, k + 2), 16));
    for (let y = 0; y < BH; ++y) for (let x = 0; x < BW; ++x) {
      const i = y * BW + x, g = Math.hypot((z[y * BW + Math.min(BW - 1, x + 1)] - z[y * BW + Math.max(0, x - 1)]) / 2, (z[Math.min(BH - 1, y + 1) * BW + x] - z[Math.max(0, y - 1) * BW + x]) / 2) + 1e-3;
      const q = z[i] / 100, dm = Math.abs(q - Math.round(q)) * 100;
      d[4 * i] = ink[0]; d[4 * i + 1] = ink[1]; d[4 * i + 2] = ink[2]; d[4 * i + 3] = Math.round(255 * 0.2 * clamp(0.9 - dm / g));
    }
    lx.putImageData(im, 0, 0);
    return true;
  }

  // one map of cells: red above the model and blue below, or (one colour) how much a cell moves
  function cells(c, x0, y0, w, h, val, hi, lo, full) {
    const cx = Math.ceil(w / GX), cy = Math.ceil(h / GY);
    for (let j = 0; j < GY; ++j) for (let i = 0; i < GX; ++i) {
      const v = val[j * GX + i]; c.globalAlpha = 0.92 * Math.pow(clamp(Math.abs(v) / full), 0.85); c.fillStyle = v > 0 ? hi : lo;
      c.fillRect(Math.floor(x0 + (i * w) / GX), Math.floor(y0 + (j * h) / GY), cx, cy);
    }
    c.globalAlpha = 1;
  }

  function draw() {
    if (!lines) return;
    const t = +sl.value, hi = css("--hi"), lo = css("--lo"), ink = css("--text");
    ctx.clearRect(0, 0, W, H); cells(ctx, 0, 0, W, H, miss[t - 1], hi, lo, FULL); ctx.imageSmoothingEnabled = true; ctx.drawImage(lines, 0, 0, W, H);
    // the months so far, gathered: their average, and how far each cohort moves around it
    const avg = new Float32Array(GX * GY), sd = new Float32Array(GX * GY);
    for (let k = 0; k < GX * GY; ++k) { let s = 0, q = 0; for (let m = 0; m < t; ++m) s += miss[m][k]; s /= t; for (let m = 0; m < t; ++m) q += (miss[m][k] - s) ** 2; avg[k] = s; sd[k] = t > 1 ? Math.sqrt(q / (t - 1)) : 0; }
    const a = cAvg.getContext("2d"), s = cSd.getContext("2d");
    a.clearRect(0, 0, cAvg.width, cAvg.height); cells(a, 0, 0, cAvg.width, cAvg.height, avg, hi, lo, FULL);
    s.clearRect(0, 0, cSd.width, cSd.height); cells(s, 0, 0, cSd.width, cSd.height, sd, ink, ink, SPREAD);
    // the strip: every month; the ones not gathered yet are faint, the one shown is ringed
    const st = cStrip.getContext("2d"), slot = cStrip.width / T, pad = 3 * dpr;
    st.clearRect(0, 0, cStrip.width, cStrip.height);
    for (let m = 0; m < T; ++m) {
      const x = m * slot + pad, w = slot - 2 * pad, h = cStrip.height - 2 * pad;
      st.save(); st.globalAlpha = m < t ? 1 : 0.4; st.beginPath(); st.rect(x, pad, w, h); st.clip();
      for (let j = 0; j < GY; ++j) for (let i = 0; i < GX; ++i) { const v = miss[m][j * GX + i]; st.globalAlpha = (m < t ? 1 : 0.4) * 0.92 * Math.pow(clamp(Math.abs(v) / FULL), 0.85); st.fillStyle = v > 0 ? hi : lo; st.fillRect(Math.floor(x + (i * w) / GX), Math.floor(pad + (j * h) / GY), Math.ceil(w / GX), Math.ceil(h / GY)); }
      st.restore();
      st.strokeStyle = ink; st.globalAlpha = m === t - 1 ? 0.95 : 0.22; st.lineWidth = (m === t - 1 ? 2 : 1) * dpr; st.strokeRect(x, pad, w, h); st.globalAlpha = 1;
    }
    read.textContent = `month ${t} of ${T}`;
  }

  const start = () => { if (build()) draw(); };
  sl.addEventListener("input", draw);
  cStrip.addEventListener("click", (ev) => { const r = cStrip.getBoundingClientRect(); sl.value = Math.max(1, Math.min(T, 1 + Math.floor(((ev.clientX - r.left) / r.width) * T))); sl.dispatchEvent(new Event("input", { bubbles: true })); });
  new IntersectionObserver((es) => { if (es.some((e) => e.isIntersecting)) start(); }).observe(box);
  let rz = 0; window.addEventListener("resize", () => { clearTimeout(rz); rz = setTimeout(() => { if (box.offsetParent) start(); }, 150); });
  new MutationObserver(() => setTimeout(() => { if (box.offsetParent) start(); }, 40)).observe(document.documentElement, { attributes: true, attributeFilter: ["class"] });
  fetch(box.dataset.src).then((r) => r.arrayBuffer()).then((buf) => {
    const u = new Uint16Array(buf); truth = new Float32Array(u.length); for (let i = 0; i < u.length; ++i) truth[i] = 2000 + u[i] / 10;
    model = smooth(truth); months(); if (box.offsetParent) start();
  });
})();
