// /backstory, "A topographical map": the map, alive. The claim of the screen is that grouping by height gathers places
// that are far apart: one band of height rings both peaks. The band's height is on the slider under the map (a and d
// move it), and a click on the map puts the band at the height clicked, lighting every other place at that height.
// Nothing moves by itself. (The next screen answers it: point at the tree and one region lights.)
// Drawn per pixel from the elevation grid itself (slides/spanish_peaks.u16: the Spanish Peaks, Colorado, decimetres above
// 2,000 m; public Terrain Tiles): tint by height, a little relief shading, contours every 50 m, heavier every 250 m.
(function () {
  "use strict";
  const box = document.getElementById("tp"); if (!box) return;
  const cv = document.getElementById("tp-cv"), ctx = cv.getContext("2d");
  const SW = +box.dataset.w, SH = +box.dataset.h, HALF = 75;   // the band is 150 m of height
  let src = null, base = null, band = null, bctx = null, bimg = null, zs = null, gs = null, BW = 0, BH = 0, W = 0, H = 0;
  let builtFor = "";
  const isDark = () => document.documentElement.classList.contains("dark");
  const rgb = (name) => { ctx.fillStyle = getComputedStyle(box).getPropertyValue(name).trim(); const h = ctx.fillStyle; return [1, 3, 5].map((k) => parseInt(h.slice(k, k + 2), 16)); };
  const clamp = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);

  function resample(a, sw, sh, w, h) {   // Catmull-Rom, across then down, so the contours come out as curves
    const cr = (p0, p1, p2, p3, t) => p1 + 0.5 * t * (p2 - p0 + t * (2 * p0 - 5 * p1 + 4 * p2 - p3 + t * (3 * (p1 - p2) + p3 - p0)));
    const tmp = new Float32Array(w * sh), out = new Float32Array(w * h);
    for (let x = 0; x < w; ++x) {
      const fx = (x / (w - 1)) * (sw - 1), i = Math.min(sw - 2, Math.floor(fx)), t = fx - i, i0 = Math.max(0, i - 1), i3 = Math.min(sw - 1, i + 2);
      for (let y = 0; y < sh; ++y) { const r = y * sw; tmp[y * w + x] = cr(a[r + i0], a[r + i], a[r + i + 1], a[r + i3], t); }
    }
    for (let y = 0; y < h; ++y) {
      const fy = (y / (h - 1)) * (sh - 1), i = Math.min(sh - 2, Math.floor(fy)), t = fy - i, r0 = Math.max(0, i - 1) * w, r1 = i * w, r2 = (i + 1) * w, r3 = Math.min(sh - 1, i + 2) * w;
      for (let x = 0; x < w; ++x) out[y * w + x] = cr(tmp[r0 + x], tmp[r1 + x], tmp[r2 + x], tmp[r3 + x], t);
    }
    return out;
  }

  function build() {
    const r = box.getBoundingClientRect(), asp = (SW - 1) / (SH - 1), cw = Math.floor(Math.min(r.width, r.height * asp)), ch = Math.round(cw / asp);
    if (cw < 60 || !src) return false;
    const dpr = Math.min(2, window.devicePixelRatio || 1), key = cw + "x" + ch + "@" + dpr + isDark();
    if (key === builtFor) return true;
    builtFor = key;
    cv.style.width = cw + "px"; cv.style.height = ch + "px"; W = cv.width = Math.round(cw * dpr); H = cv.height = Math.round(ch * dpr);
    const z = resample(src, SW, SH, W, H), ink = rgb("--text"), dark = isDark(), mpp = ((SW - 1) * 30.4) / W;
    base = document.createElement("canvas"); base.width = W; base.height = H;
    const bx = base.getContext("2d"), im = bx.createImageData(W, H), d = im.data;
    for (let y = 0; y < H; ++y) for (let x = 0; x < W; ++x) {
      const i = y * W + x, zc = z[i], gx = (z[y * W + Math.min(W - 1, x + 1)] - z[y * W + Math.max(0, x - 1)]) / 2, gy = (z[Math.min(H - 1, y + 1) * W + x] - z[Math.max(0, y - 1) * W + x]) / 2, g = Math.hypot(gx, gy) + 1e-3;
      // tint by height; then relief, lit from the upper left: in the light theme the far slopes darken, in the dark one the near slopes glow
      let a = 0.2 * clamp((zc - 2500) / 1600);
      const s = (-(gx + gy) * 0.7071) / mpp;   // > 0 where the ground rises away from the light
      a += 0.15 * clamp((dark ? -s : s) / 0.7);
      // contours: how far, in pixels, to the nearest multiple of the interval
      const line = (c, half, op) => { const q = zc / c, dm = Math.abs(q - Math.round(q)) * c; return op * clamp(half + 0.5 - dm / g); };
      const l = Math.max(line(50, 0.3 * dpr, 0.4), line(250, 0.65 * dpr, 0.62));
      a += l * (1 - a);
      d[4 * i] = ink[0]; d[4 * i + 1] = ink[1]; d[4 * i + 2] = ink[2]; d[4 * i + 3] = Math.round(255 * a);
    }
    bx.putImageData(im, 0, 0);
    // the band's own, coarser grid: heights, and how fast they change, for a soft edge
    BW = Math.min(W, 960); BH = Math.round((BW * H) / W);
    zs = new Float32Array(BW * BH); gs = new Float32Array(BW * BH);
    for (let y = 0; y < BH; ++y) for (let x = 0; x < BW; ++x) zs[y * BW + x] = z[Math.min(H - 1, Math.round((y * H) / BH)) * W + Math.min(W - 1, Math.round((x * W) / BW))];
    for (let y = 0; y < BH; ++y) for (let x = 0; x < BW; ++x) gs[y * BW + x] = Math.hypot((zs[y * BW + Math.min(BW - 1, x + 1)] - zs[y * BW + Math.max(0, x - 1)]) / 2, (zs[Math.min(BH - 1, y + 1) * BW + x] - zs[Math.max(0, y - 1) * BW + x]) / 2) + 0.5;
    band = document.createElement("canvas"); band.width = BW; band.height = BH; bctx = band.getContext("2d"); bimg = bctx.createImageData(BW, BH);
    const acc = rgb("--accent"), bd = bimg.data;
    for (let i = 0; i < BW * BH; ++i) { bd[4 * i] = acc[0]; bd[4 * i + 1] = acc[1]; bd[4 * i + 2] = acc[2]; }
    return true;
  }

  // The band is drawn when its height changes and at no other time: nothing on this screen moves by itself.
  const sl = document.getElementById("tp-level");
  function draw() {
    if (!base) return;
    const level = +sl.value, bd = bimg.data;
    for (let i = 0, n = BW * BH; i < n; ++i) { const a = 0.5 + (HALF - Math.abs(zs[i] - level)) / gs[i]; bd[4 * i + 3] = a <= 0 ? 0 : a >= 1 ? 153 : (153 * a) | 0; }
    bctx.putImageData(bimg, 0, 0);
    ctx.clearRect(0, 0, W, H); ctx.drawImage(base, 0, 0); ctx.imageSmoothingEnabled = true; ctx.drawImage(band, 0, 0, W, H);
  }
  const start = () => { if (build()) draw(); };
  sl.addEventListener("input", draw);
  // a click on the map puts the band at the height clicked
  cv.addEventListener("click", (ev) => {
    if (!zs) return;
    const r = cv.getBoundingClientRect(), x = Math.max(0, Math.min(BW - 1, Math.round(((ev.clientX - r.left) / r.width) * BW))), y = Math.max(0, Math.min(BH - 1, Math.round(((ev.clientY - r.top) / r.height) * BH)));
    sl.value = zs[y * BW + x]; draw();
  });
  new IntersectionObserver((es) => { if (es.some((e) => e.isIntersecting)) start(); }).observe(box);
  let rz = 0; window.addEventListener("resize", () => { clearTimeout(rz); rz = setTimeout(() => { if (box.offsetParent) start(); }, 150); });
  new MutationObserver(() => setTimeout(() => { if (box.offsetParent) start(); }, 40)).observe(document.documentElement, { attributes: true, attributeFilter: ["class"] });
  fetch(box.dataset.src).then((r) => r.arrayBuffer()).then((buf) => { const u = new Uint16Array(buf); src = new Float32Array(u.length); for (let i = 0; i < u.length; ++i) src[i] = 2000 + u[i] / 10; if (box.offsetParent) start(); });
})();
