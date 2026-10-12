// /backstory, "Past two dimensions": the rectangular mesh with a third input. The same Fannie Mae month as the real-data
// screen, fitted by the rectangular engine on loan age, coupon AND average loan balance (third.json, made natively by
// backstory/tools/third_input.sh). A box splits across one axis at a time, whichever the gate chooses, so the same rule
// works with any number of inputs; this is the three-input fit, seen one slice of loan balance at a time.
//   map      the fitted surface over loan age and coupon at the slider's loan balance, with the boxes that slice cuts;
//            faint dots are the pools with balances near it (area by loans), and the surface is faded where there are
//            none, so the mesh's guesses in empty corners are not taken for findings.
//   box      the three inputs, with the slice's place in it.
//   numbers  held-out deviance of the two-input and the three-input fits: same pools, same split, same engine.
(function () {
  "use strict";
  const root = document.getElementById("th"); if (!root) return;
  const $ = (id) => document.getElementById(id), cv = $("th-cv"), sl = $("th-z");
  const isDark = () => document.documentElement.classList.contains("dark");
  const VMAX = 1.5, M = { l: 46, r: 10, t: 10, b: 34 };
  let D = null;
  function rgb(dl) {   // the real-data screen's colours
    const dark = isDark(), lo = dark ? [110, 150, 255] : [31, 72, 200], mid = dark ? [34, 35, 38] : [250, 247, 238], hi = dark ? [255, 140, 80] : [205, 66, 20];
    const s = Math.max(-1, Math.min(1, dl / VMAX)), a = s < 0 ? lo : hi, w = Math.pow(Math.abs(s), 0.8);
    return [0, 1, 2].map((j) => Math.round(mid[j] + (a[j] - mid[j]) * w));
  }
  const money = (v) => "$" + (Math.round(v / 1000)).toLocaleString() + "k";
  function draw() {
    if (!D || root.offsetParent === null) return;
    const r = cv.getBoundingClientRect(), dpr = Math.min(2, window.devicePixelRatio || 1), W = r.width, H = r.height; if (W < 20 || H < 20) return;
    if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(H * dpr)) { cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); }
    const g = cv.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, W, H);
    const pw = W - M.l - M.r, ph = H - M.t - M.b, z = +sl.value, C = D.cells, N = C.length / 14;
    // the slice, cell by cell: each box the slice passes through gives a rectangle whose four corners are read off at z
    const RW = Math.max(2, Math.round(pw / 2)), RH = Math.max(2, Math.round(ph / 2)), off = document.createElement("canvas"); off.width = RW; off.height = RH;
    const oc = off.getContext("2d"), im = oc.createImageData(RW, RH), d = im.data, boxes = [];
    // Where no pools have a balance near this one the surface is the mesh's guess, and is drawn faded (as on the real-data
    // screen): the share of each small cell's neighbourhood that holds pools, in this slab of balances or the next ones.
    const O = D.occ, slab = Math.min(O.slabs - 1, Math.floor(z * O.slabs)); let near = new Float32Array(O.rows * O.cols);
    for (let ss = Math.max(0, slab - 1); ss <= Math.min(O.slabs - 1, slab + 1); ++ss) for (let k = 0; k < near.length; ++k) if (O.loans[ss * near.length + k] > 0) near[k] = 1;
    for (let pass = 0; pass < 2; ++pass) { const nx = new Float32Array(near.length);
      for (let rr = 0; rr < O.rows; ++rr) for (let cc = 0; cc < O.cols; ++cc) { let sum = 0; for (let a = -1; a <= 1; ++a) for (let b = -1; b <= 1; ++b) { const r2 = rr + a, c2 = cc + b; if (r2 >= 0 && r2 < O.rows && c2 >= 0 && c2 < O.cols) sum += near[r2 * O.cols + c2]; } nx[rr * O.cols + cc] = sum / 9; }
      near = nx; }
    const m0 = rgb(0), seen = (u, v) => {   // read between the small cells, so the fade has no steps
      const fx = Math.max(0, Math.min(O.cols - 1.001, u * O.cols - 0.5)), fy = Math.max(0, Math.min(O.rows - 1.001, v * O.rows - 0.5)), c = Math.floor(fx), r = Math.floor(fy), a = fx - c, b = fy - r;
      const o = (1 - b) * ((1 - a) * near[r * O.cols + c] + a * near[r * O.cols + c + 1]) + b * ((1 - a) * near[(r + 1) * O.cols + c] + a * near[(r + 1) * O.cols + c + 1]);
      return 0.2 + 0.8 * Math.min(1, 5 * o); };
    for (let i = 0; i < N; ++i) {
      const o = 14 * i, z0 = C[o + 2], z1 = C[o + 5]; if (z < z0 || z > z1 || (z === z1 && z1 < 1)) continue;
      const t = z1 > z0 ? (z - z0) / (z1 - z0) : 0, x0 = C[o], y0 = C[o + 1], x1 = C[o + 3], y1 = C[o + 4];
      const h00 = C[o + 6] * (1 - t) + C[o + 10] * t, h10 = C[o + 7] * (1 - t) + C[o + 11] * t, h01 = C[o + 8] * (1 - t) + C[o + 12] * t, h11 = C[o + 9] * (1 - t) + C[o + 13] * t;
      boxes.push([x0, y0, x1, y1]);
      const c0 = Math.max(0, Math.floor(x0 * RW)), c1 = Math.min(RW, Math.ceil(x1 * RW)), r0 = Math.max(0, Math.floor((1 - y1) * RH)), r1 = Math.min(RH, Math.ceil((1 - y0) * RH));
      for (let rr = r0; rr < r1; ++rr) { const v = Math.max(0, Math.min(1, (1 - (rr + 0.5) / RH - y0) / (y1 - y0)));
        for (let cc = c0; cc < c1; ++cc) { const u = Math.max(0, Math.min(1, ((cc + 0.5) / RW - x0) / (x1 - x0))), q = rgb((h00 * (1 - u) + h10 * u) * (1 - v) + (h01 * (1 - u) + h11 * u) * v), k = 4 * (rr * RW + cc), w = seen((cc + 0.5) / RW, 1 - (rr + 0.5) / RH);
          d[k] = m0[0] + (q[0] - m0[0]) * w; d[k + 1] = m0[1] + (q[1] - m0[1]) * w; d[k + 2] = m0[2] + (q[2] - m0[2]) * w; d[k + 3] = 255; } }
    }
    oc.putImageData(im, 0, 0); g.imageSmoothingEnabled = true; g.drawImage(off, M.l, M.t, pw, ph);
    const X = (u) => M.l + u * pw, Y = (v) => M.t + (1 - v) * ph, ink = isDark() ? "255,255,255" : "20,20,30";
    // the pools with balances near this one
    const cw = pw / O.cols, ch = ph / O.rows, base = slab * O.rows * O.cols;
    g.fillStyle = `rgba(${ink},${isDark() ? 0.32 : 0.28})`; g.beginPath();
    for (let rr = 0; rr < O.rows; ++rr) for (let cc = 0; cc < O.cols; ++cc) { const n = O.loans[base + rr * O.cols + cc]; if (!n) continue;
      const rad = 0.5 * Math.min(cw, ch) * Math.sqrt(Math.min(1, n / D.occTop)), x = M.l + (cc + 0.5) * cw, y = M.t + ph - (rr + 0.5) * ch; g.moveTo(x + rad, y); g.arc(x, y, rad, 0, 6.2832); }
    g.fill();
    g.strokeStyle = `rgba(${ink},0.36)`; g.lineWidth = 0.8; g.beginPath(); for (const [x0, y0, x1, y1] of boxes) g.rect(X(x0), Y(y1), X(x1) - X(x0), Y(y0) - Y(y1)); g.stroke();
    // axes, as on the real-data screen
    const [a0, a1, w0, w1] = D.bounds, faint = getComputedStyle(root).getPropertyValue("--faint").trim();
    g.font = "11px ui-sans-serif, system-ui, sans-serif"; g.fillStyle = faint; g.textAlign = "center"; g.textBaseline = "top";
    for (let a = 0; a <= a1; a += 60) g.fillText(a, X((a - a0) / (a1 - a0)), M.t + ph + 5);
    g.fillText("loan age, months", M.l + pw / 2, M.t + ph + 19); g.textAlign = "right"; g.textBaseline = "middle";
    for (let w = Math.ceil(w0); w <= w1; ++w) g.fillText(w + "%", M.l - 6, Y((w - w0) / (w1 - w0)));
    g.save(); g.translate(11, M.t + ph / 2); g.rotate(-Math.PI / 2); g.textAlign = "center"; g.fillText("coupon", 0, 0); g.restore();
    g.strokeStyle = `rgba(${ink},0.25)`; g.lineWidth = 1; g.strokeRect(M.l + 0.5, M.t + 0.5, pw - 1, ph - 1);
    // the slider's balance, and the slice's place in the box of three inputs
    $("th-bal").textContent = money(Math.exp(D.balance[0] + z * (D.balance[1] - D.balance[0])));
    const bx = 34, by = 150, bw = 150, bh = 96, dx = 86, dy = 64, P = (x, y, zz) => `${(bx + x * bw + zz * dx).toFixed(1)} ${(by - y * bh - zz * dy).toFixed(1)}`;
    const face = (zz) => `M${P(0, 0, zz)}L${P(1, 0, zz)}L${P(1, 1, zz)}L${P(0, 1, zz)}Z`;
    $("th-box").innerHTML = `<path class="th-edge" d="${face(1)}M${P(0, 0, 0)}L${P(0, 0, 1)}M${P(1, 0, 0)}L${P(1, 0, 1)}M${P(1, 1, 0)}L${P(1, 1, 1)}M${P(0, 1, 0)}L${P(0, 1, 1)}"/>`
      + `<path class="th-slice" d="${face(z)}"/><path class="th-edge front" d="${face(0)}"/>`
      + `<text class="or-ax" x="${bx + bw / 2}" y="${by + 17}" text-anchor="middle">loan age</text><text class="or-ax" transform="translate(${bx - 9} ${by - bh / 2}) rotate(-90)" text-anchor="middle">coupon</text>`
      + `<text class="or-ax" transform="translate(${bx + bw + dx / 2 + 12} ${by - dy / 2 + 12}) rotate(${(-Math.atan2(dy, dx) * 180 / Math.PI).toFixed(1)})" text-anchor="middle">loan balance</text>`;
  }
  fetch(root.dataset.src).then((r) => r.json()).then((J) => {
    D = J; const full = D.occ.loans.filter((v) => v > 0).sort((a, b) => a - b); D.occTop = full[Math.floor(0.99 * (full.length - 1))] || 1;
    sl.value = ((D.median_balance - D.balance[0]) / (D.balance[1] - D.balance[0])).toFixed(2);
    $("th-cards").innerHTML = [["Loan age and coupon", D.heldout.two.toFixed(3), `${D.cells2d} rectangles`], ["With loan balance", D.heldout.three.toFixed(3), `${D.cells.length / 14} boxes`]]
      .map(([k, v, t]) => `<div class="card"><div class="k">${k}</div><div class="v">${v}</div><div class="d">${t}</div></div>`).join("") + '<div class="th-note">held-out deviance per pool (lower is better)</div>';
    // a box is cut across one axis at a time, whichever the gate picks: the same rule with any number of inputs
    const cube = (ox, axis) => { const w = 52, h = 36, dx = 24, dy = 18, by = 62, P = (x, y, z) => `${(ox + x * w + z * dx).toFixed(1)} ${(by - y * h - z * dy).toFixed(1)}`;
      const face = (z) => `M${P(0, 0, z)}L${P(1, 0, z)}L${P(1, 1, z)}L${P(0, 1, z)}Z`;
      const cut = axis === 0 ? `M${P(.5, 0, 0)}L${P(.5, 1, 0)}L${P(.5, 1, 1)}L${P(.5, 0, 1)}Z` : axis === 1 ? `M${P(0, .5, 0)}L${P(1, .5, 0)}L${P(1, .5, 1)}L${P(0, .5, 1)}Z` : face(0.5);
      return `<path class="th-edge" d="${face(1)}M${P(0, 0, 0)}L${P(0, 0, 1)}M${P(1, 0, 0)}L${P(1, 0, 1)}M${P(1, 1, 0)}L${P(1, 1, 1)}M${P(0, 1, 0)}L${P(0, 1, 1)}"/><path class="th-cut" d="${cut}"/><path class="th-edge front" d="${face(0)}"/>`
        + `<text class="or-ax" x="${ox + 38}" y="${by + 22}" text-anchor="middle">${["age", "coupon", "balance"][axis]}</text>`; };
    $("th-cuts").innerHTML = cube(8, 0) + cube(108, 1) + cube(208, 2);
    draw();
  });
  sl.addEventListener("input", draw);
  new IntersectionObserver((es) => { if (es.some((e) => e.isIntersecting)) draw(); }).observe(root);
  let rz = 0; window.addEventListener("resize", () => { clearTimeout(rz); rz = setTimeout(draw, 120); });
  new MutationObserver(() => setTimeout(draw, 40)).observe(document.documentElement, { attributes: true, attributeFilter: ["class"] });
})();
