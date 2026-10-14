'use strict';
// The star page's arithmetic, ported from analysis/starplot.py so the browser can recompute a
// vertex's star, card and sections on a click. Loadable in node (module.exports) for the test
// that compares it with the Python, and in the page (globalThis.StarCore).
(function (root) {

// ---- decoding ---------------------------------------------------------------------------
function decodeB64(b64, Type) {
  let bytes;
  if (typeof atob === 'function') {
    const bin = atob(b64);
    bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  } else {
    bytes = new Uint8Array(Buffer.from(b64, 'base64'));   // a fresh, aligned buffer
  }
  return new Type(bytes.buffer, 0, bytes.byteLength / Type.BYTES_PER_ELEMENT);
}

function decodeRun(raw) {
  const P = raw.pools, V = raw.vertices, F = raw.faces, FP = raw.facePools;
  const run = {
    key: raw.key, label: raw.label, model: raw.model, typeIsProxy: raw.typeIsProxy, combined: raw.combined,
    pools: {
      n: P.n,
      wala: decodeB64(P.wala, Float32Array), wac: decodeB64(P.wac, Float32Array),
      loans: decodeB64(P.loans, Int32Array), k: decodeB64(P.k, Int32Array),
      phat: decodeB64(P.phat, Float32Array), ghat: decodeB64(P.ghat, Float32Array),
      base: decodeB64(P.base, Float32Array), info: decodeB64(P.info, Float32Array),
      pid: decodeB64(P.pid, Int32Array), flags: decodeB64(P.flags, Uint8Array),
    },
    vertices: {n: V.n},
    faces: {n: F.n, v0: decodeB64(F.v0, Int32Array), v1: decodeB64(F.v1, Int32Array), v2: decodeB64(F.v2, Int32Array)},
    facePools: {
      start: decodeB64(FP.start, Int32Array), pool: decodeB64(FP.pool, Int32Array),
      l0: decodeB64(FP.l0, Float64Array), l1: decodeB64(FP.l1, Float64Array),
    },
  };
  for (const name of V.columns) run.vertices[name] = decodeB64(V[name], Float64Array);
  run.pools.ids = P.ids || Array.from(run.pools.pid, String);
  run.pools.weight = P.weight ? decodeB64(P.weight, Float64Array) : run.pools.info;
  run.pools.usesMesh = P.usesMesh ? decodeB64(P.usesMesh, Uint8Array) : new Uint8Array(P.n).fill(1);
  run.pools.components = {};
  for (const name of Object.keys(P.components || {})) run.pools.components[name] = decodeB64(P.components[name], Float64Array);
  // the unit-square coordinates of every pool, and index by vertex id
  const m = run.model, P2 = run.pools;
  P2.sx = new Float64Array(P2.n); P2.sy = new Float64Array(P2.n);
  for (let i = 0; i < P2.n; i++) {
    P2.sx[i] = (P2.wala[i] - m.wala_min) / (m.wala_max - m.wala_min);
    P2.sy[i] = (P2.wac[i] - m.wac_min) / (m.wac_max - m.wac_min);
  }
  run.byId = new Map();
  for (let j = 0; j < run.vertices.n; j++) run.byId.set(run.vertices.id[j], j);
  run.orderById = Array.from({length: run.vertices.n}, (_, j) => j)
      .sort((a, b) => run.vertices.id[a] - run.vertices.id[b]);
  run.H = m.height_scale;
  return run;
}

const isTrain = (run, i) => (run.pools.flags[i] & 1) !== 0;
const isMulti = (run, i) => (run.pools.flags[i] & 2) !== 0;
const inFit   = (run, i) => (run.pools.flags[i] & 4) !== 0;

// ---- the dyadic address (docs/VERTEX_ADDRESS.txt) ---------------------------------------
function address(x, y, tol) {
  // the dumps print ten significant digits, so x * 2^L carries an error up to 5e-11 * 2^L,
  // which passes 1e-9 from depth 11; the tolerance scales with the level and is capped well
  // below the half-integer a finer vertex would show (see analysis/address.py)
  for (let L = 0; L <= 30; L++) {
    const s = Math.pow(2, L);
    const t = tol === undefined ? Math.min(Math.max(s * 2e-10, 1e-9), 1e-3) : tol;
    if (Math.abs(x * s - Math.round(x * s)) < t && Math.abs(y * s - Math.round(y * s)) < t)
      return [L, Math.round(x * s), Math.round(y * s)];
  }
  return null;
}
const addressName = (a) => a ? ('L' + a[0] + ' (' + a[1] + ', ' + a[2] + ')') : '?';
function parseAddress(text) {
  const m = /^\s*L(\d+)\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)\s*$/.exec(text);
  return m ? [+m[1], +m[2], +m[3]] : null;
}
const vertexName = (run, vid) => {
  const j = run.byId.get(vid);
  return j === undefined ? '?' : addressName(address(run.vertices.x[j], run.vertices.y[j]));
};

// ---- geometry ----------------------------------------------------------------------------
// A free coefficient's weight in every active vertex's height: 1 at itself, and half of what its
// parents carry at each conformity descendant. Ids increase from parent to child, so one pass.
function influence(run, vid) {
  const V = run.vertices, w = new Map([[vid, 1]]);
  for (const j of run.orderById) {
    const id = V.id[j];
    if (id === vid || V.free_coefficient[j] === 1 || V.active[j] !== 1) continue;
    const c = (w.get(V.parent0[j]) || 0) + (w.get(V.parent1[j]) || 0);
    if (c > 0) w.set(id, 0.5 * c);
  }
  return w;
}

function starFaces(run, vid) {
  const w = influence(run, vid), F = run.faces, out = [];
  for (let f = 0; f < F.n; f++) {
    const w0 = w.get(F.v0[f]) || 0, w1 = w.get(F.v1[f]) || 0, w2 = w.get(F.v2[f]) || 0;
    if (w0 !== 0 || w1 !== 0 || w2 !== 0) out.push({f: f, w0: w0, w1: w1, w2: w2});
  }
  return out;
}

function nodalFaces(run, vid) {
  const F = run.faces, out = [];
  for (let f = 0; f < F.n; f++)
    if (F.v0[f] === vid || F.v1[f] === vid || F.v2[f] === vid) out.push(f);
  return out;
}

// Pools inside the star with the coefficient's weight at each (its nodal hat continued at half
// weight through every conformity descendant). A pool on a shared edge takes its first face.
function starPools(run, vid) {
  const sf = starFaces(run, vid), FP = run.facePools, seen = new Set(), out = [];
  for (let s = 0; s < sf.length; s++) {
    const {f, w0, w1, w2} = sf[s];
    for (let t = FP.start[f]; t < FP.start[f + 1]; t++) {
      const i = FP.pool[t];
      if (seen.has(i)) continue;
      seen.add(i);                      // the first star face containing it wins, whatever its hat
      const l0 = FP.l0[t], l1 = FP.l1[t], l2 = 1 - l0 - l1;
      const hat = w0 * l0 + w1 * l1 + w2 * l2;
      if (hat > 1e-9) out.push({i: i, hat: hat});
    }
  }
  return out;
}

// ---- the card ----------------------------------------------------------------------------
function parentsOf(run, vid) {
  const j = run.byId.get(vid), V = run.vertices;
  return [V.parent0[j], V.parent1[j]];
}
function heightOf(run, vid) { return run.vertices.height[run.byId.get(vid)]; }

function cardOf(run, vid, showHeldOut) {
  const V = run.vertices, P = run.pools, H = run.H, j = run.byId.get(vid);
  const all = starPools(run, vid);
  const pools = showHeldOut ? all : all.filter(e => isTrain(run, e.i));
  const [p0, p1] = parentsOf(run, vid);
  const nullH = p0 >= 0 ? (heightOf(run, p0) + heightOf(run, p1)) / 2 : V.height[j];
  let xTx = 0;
  for (const e of pools) if (inFit(run, e.i)) xTx += P.weight[e.i] * e.hat * e.hat;
  let maxShare = -1, maxIdx = -1, sumSq = 0, nFit = 0;
  const share = pools.map(e => {
    const w = inFit(run, e.i) ? P.weight[e.i] : 0;
    const s = xTx > 0 ? w * e.hat * e.hat / xTx : 0;
    sumSq += s * s;
    if (inFit(run, e.i)) nFit++;
    if (s > maxShare) { maxShare = s; maxIdx = e.i; }
    return s;
  });
  // excess over binomial noise per unit information, on the training pools: against the baseline
  // alone (T) and against the fitted surface (Xm), so T - Xm is what the surface explains
  const excess = (rate) => {
    let num = 0, den = 0;
    for (const e of pools) {
      if (!isTrain(run, e.i)) continue;
      const i = e.i, n = P.loans[i], p = rate(i), I = n * p * (1 - p);
      if (!(I > 0)) continue;
      num += Math.pow(P.k[i] - n * p, 2) / I - 1;
      den += I;
    }
    return den > 0 ? num / den : NaN;
  };
  const T = excess(i => P.usesMesh[i] ? 1 / (1 + Math.exp(-P.base[i])) : P.phat[i]), Xm = excess(i => P.phat[i]);
  const byType = [];
  for (const multi of [false, true]) {
    let np_ = 0, nl = 0, ke = 0, fitw = 0, wsum = 0;
    for (const e of pools) {
      const i = e.i;
      if (!isTrain(run, i) || isMulti(run, i) !== multi) continue;
      np_++; nl += P.loans[i]; ke += P.k[i]; fitw += P.phat[i] * P.loans[i]; wsum += P.loans[i];
    }
    if (np_) byType.push({multi: multi, pools: np_, loans: nl, obs: 100 * ke / Math.max(nl, 1), fit: 100 * fitw / Math.max(wsum, 1)});
  }
  const surplus = (V.height[j] - nullH) / H;
  return {
    id: vid, name: vertexName(run, vid), depth: V.depth[j] | 0, free: V.free_coefficient[j] === 1,
    parents: [p0, p1], parentNames: p0 >= 0 ? [vertexName(run, p0), vertexName(run, p1)] : null,
    pools: pools.length, poolsFit: nFit, poolsHeldOut: pools.filter(e => !isTrain(run, e.i)).length,
    xTx: xTx, keffShare: sumSq > 0 ? 1 / sumSq : NaN,
    maxShare: maxShare, maxSharePid: maxIdx >= 0 ? P.pid[maxIdx] : -1,
    maxShareLoans: maxIdx >= 0 ? P.loans[maxIdx] : 0,
    surplus: surplus, odds: Math.exp(surplus), nullH: nullH, T: T, S: T - Xm, R: Xm,
    byType: byType, share: share, poolList: pools,
    admit: V.admit_round[j] >= 0 ? {
      round: V.admit_round[j] | 0, z: V.admit_z[j], c: V.admit_c[j],
      gain: V.admit_gain === undefined ? NaN : V.admit_gain[j],
      root: V.admit_signed_root === undefined ? V.admit_beta[j] : V.admit_signed_root[j],
    } : null,
  };
}

// ---- the pool effect ----------------------------------------------------------------------
let GHx = null, GHw = null, GH8x = null, GH8w = null;
function setQuadrature(x, w, x8, w8) {
  GHx = x; GHw = w;
  GH8x = x8 || x; GH8w = w8 || w;      // the coarser set integrates the vertex's shared posterior
}

// The shift c per pool such that a normal effect N(-c, s2) leaves the pool's expected rate at
// expit(eta): the surface is fitted to counts, so an uncentred effect would raise the mean.
function nodes(s2) {              // the quadrature at this variance, normalised, built once
  const G = GHx.length, u = new Float64Array(G), w = new Float64Array(G);
  const sd = Math.sqrt(2 * s2), rt = Math.sqrt(Math.PI);
  for (let g = 0; g < G; g++) { u[g] = sd * GHx[g]; w[g] = GHw[g] / rt; }
  return {G: G, u: u, w: w};
}

function centredShift(eta, s2, q) {
  const c = new Float64Array(eta.length);
  if (!(s2 > 0)) return c;
  const {G, u, w} = q || nodes(s2);
  for (let i = 0; i < eta.length; i++) {
    const e = eta[i], target = 1 / (1 + Math.exp(-e));
    let ci = s2 / 2;
    for (let it = 0; it < 10; it++) {     // Newton; the small-variance value s2/2 starts it close
      let mean = 0, slope = 0;
      for (let g = 0; g < G; g++) {
        const p = 1 / (1 + Math.exp(-(e - ci + u[g])));
        mean += p * w[g]; slope += p * (1 - p) * w[g];
      }
      const d = mean - target;
      if (!(slope > 0)) break;
      ci += d / slope;
      if (Math.abs(d) < 1e-13) break;
    }
    c[i] = ci;
  }
  return c;
}

const VARIANCE_SAMPLE = 1200;     // pools beyond this add precision the band cannot show

// The variance of a normal, mean-preserving pool effect, by maximum marginal likelihood.
// Above VARIANCE_SAMPLE pools it is fitted on a deterministic stride through them: the
// estimate's relative error is about sqrt(2/m), so 1200 pools fix the standard deviation to
// about 2%, which moves a band by less than its line width, and the cost stays bounded.
function poolVariance(n, k, eta) {
  const st = varianceStart(n, k, eta);
  if (!st) return 0;
  while (varianceStep(st)) { /* keep going */ }
  return varianceFinish(st);
}

// The same fit, a few golden-section steps at a time, so a star of thousands of pools never
// blocks a frame. varianceStep returns true while there is more to do.
const VARIANCE_ITERATIONS = 42;

function varianceStart(n, k, eta) {
  if (n.length > VARIANCE_SAMPLE) {
    const stride = Math.ceil(n.length / VARIANCE_SAMPLE), nn = [], kk = [], ee = [];
    for (let i = 0; i < n.length; i += stride) { nn.push(n[i]); kk.push(k[i]); ee.push(eta[i]); }
    n = nn; k = kk; eta = Float64Array.from(ee);
  }
  if (n.length < 3) return null;
  const ll = new Float64Array(GHx.length);
  const nll = (logS2) => {
    const s2 = Math.exp(logS2), q = nodes(s2), shift = centredShift(eta, s2, q);
    let total = 0;
    for (let i = 0; i < n.length; i++) {
      const c = eta[i] - shift[i];
      let mx = -Infinity;
      for (let g = 0; g < q.G; g++) {
        const e = c + q.u[g];
        ll[g] = k[i] * e - n[i] * Math.log1p(Math.exp(e));
        if (ll[g] > mx) mx = ll[g];
      }
      let acc = 0;
      for (let g = 0; g < q.G; g++) acc += Math.exp(ll[g] - mx) * q.w[g];
      total += mx + Math.log(acc);
    }
    return -total;
  };
  const g = (Math.sqrt(5) - 1) / 2, lo = Math.log(1e-4), hi = Math.log(9);
  const c1 = hi - g * (hi - lo), d1 = lo + g * (hi - lo);
  return {nll: nll, g: g, a: lo, b: hi, c: c1, d: d1, fc: nll(c1), fd: nll(d1), it: 0};
}

function varianceStep(st, iterations) {
  const n = iterations === undefined ? 6 : iterations;
  for (let q = 0; q < n && st.it < VARIANCE_ITERATIONS; q++, st.it++) {
    if (st.fc < st.fd) { st.b = st.d; st.d = st.c; st.fd = st.fc; st.c = st.b - st.g * (st.b - st.a); st.fc = st.nll(st.c); }
    else { st.a = st.c; st.c = st.d; st.fc = st.fd; st.d = st.a + st.g * (st.b - st.a); st.fd = st.nll(st.d); }
  }
  return st.it < VARIANCE_ITERATIONS;
}

const varianceFinish = (st) => Math.exp((st.a + st.b) / 2);

// ---- random numbers ------------------------------------------------------------------------
function rng32(seed) {           // mulberry32; the bands are Monte Carlo, so the stream need not
  let t = seed >>> 0;            // match numpy's, only its distribution
  return function () {
    t += 0x6D2B79F5;
    let r = t;
    r = Math.imul(r ^ (r >>> 15), r | 1);
    r ^= r + Math.imul(r ^ (r >>> 7), r | 61);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}
function gauss(rand) {
  let u = 0, v = 0;
  while (u === 0) u = rand();
  while (v === 0) v = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
// Inversion below a mean of 30, where the counts are rare; the normal approximation above it,
// where n is in the thousands and the approximation is tight.
function binomial(rand, n, p) {
  if (!(p > 0) || n <= 0) return 0;
  if (p >= 1) return n;
  const mean = n * p;
  if (mean <= 30) {
    const q = 1 - p, ratio = p / q;
    let f = Math.pow(q, n), s = f, u = rand(), i = 0;
    while (u > s && i < n) { i++; f *= (n - i + 1) / i * ratio; s += f; }
    return i;
  }
  const x = Math.round(mean + Math.sqrt(mean * (1 - p)) * gauss(rand));
  return x < 0 ? 0 : (x > n ? n : x);
}
// erfc to about 1e-7, and the normal CDF from it
function erfc(x) {
  const z = Math.abs(x), t = 2 / (2 + z);
  const y = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 +
    t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
  return x >= 0 ? y : 2 - y;
}
const normalCdf = (z) => 0.5 * erfc(-z / Math.SQRT2);

// The quantile of a weighted mixture of normals, by bisection on its CDF. The bin's total is
// normal given the vertex's shared draw, so over that draw it is a mixture: when the vertex's
// posterior is wide the mixture is skewed and taking mean +/- sd would be wrong.
function mixtureQuantile(prob, weight, mu, sd) {
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < mu.length; i++) { lo = Math.min(lo, mu[i] - 8 * sd[i]); hi = Math.max(hi, mu[i] + 8 * sd[i]); }
  for (let it = 0; it < 60; it++) {
    const mid = 0.5 * (lo + hi);
    let f = 0;
    for (let i = 0; i < mu.length; i++) f += weight[i] * normalCdf((mid - mu[i]) / sd[i]);
    if (f < prob) lo = mid; else hi = mid;
  }
  return 0.5 * (lo + hi);
}

function percentile(sorted, q) {      // numpy's linear interpolation
  const idx = (q / 100) * (sorted.length - 1), lo = Math.floor(idx), hi = Math.ceil(idx);
  return lo === hi ? sorted[lo] : sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo);
}

// ---- a section through the star --------------------------------------------------------------
const STRIP = 0.35, BINS = 8, DRAWS = 1500;

// The bin's 68% band: the total under the fitted surface with one mean-preserving pool effect
// per pool and the vertex's shared posterior, conditioned on an event as the display is.
// A bin of thousands of pools costs half a second at DRAWS draws, so the work is resumable:
// the page takes it a slice at a time and never blocks a frame. bandStep returns true when
// there is more to do.
function bandStart(b) {
  return {b: b, rand: rng32(b.seed), draws: [], done: 0,
          chunk: Math.max(40, Math.round(120000 / Math.max(b.nn.length, 1)))};
}
function bandStep(st) {
  const b = st.b, rand = st.rand, end = Math.min(DRAWS, st.done + st.chunk);
  for (; st.done < end; st.done++) {
    const uv = gauss(rand) * b.sdV;
    let tot = 0;
    for (let q = 0; q < b.nn.length; q++) {
      const eta = b.etaFit[q] - b.shift[q] + uv * b.hats[q] + gauss(rand) * b.sd;
      tot += binomial(rand, b.nn[q], 1 / (1 + Math.exp(-eta)));
    }
    if (tot >= 1) st.draws.push(tot + rand() - 0.5);
  }
  return st.done < DRAWS;
}
function bandFinish(st) {
  if (st.draws.length < 50) return [NaN, NaN];
  const b = st.b, lg = st.draws.map(x => {
    const pp = Math.min(Math.max((x + 0.5) / (b.pn + 1), 1e-12), 1 - 1e-12);
    return Math.log(pp / (1 - pp));
  }).sort((p, q) => p - q);
  return [percentile(lg, 16) - b.nullG, percentile(lg, 84) - b.nullG];
}
function bandOf(b) {                       // the whole band at once, for the tests
  const st = bandStart(b);
  while (bandStep(st)) { /* keep going */ }
  return bandFinish(st);
}

function sectionData(run, vid, which, showHeldOut, s2type, withBands) {
  withBands = withBands !== false;
  const V = run.vertices, P = run.pools, H = run.H, j = run.byId.get(vid);
  const [p0, p1] = parentsOf(run, vid);
  const vx = V.x[j], vy = V.y[j];
  let dir;
  if (p0 >= 0) {
    const a = run.byId.get(p0), b = run.byId.get(p1);
    dir = [V.x[b] - V.x[a], V.y[b] - V.y[a]];
  } else dir = [1, 0];
  let perp = [-dir[1], dir[0]];
  if (dir[0] < -1e-12 || (Math.abs(dir[0]) <= 1e-12 && dir[1] < 0)) dir = [-dir[0], -dir[1]];
  if (perp[1] < -1e-12 || (Math.abs(perp[1]) <= 1e-12 && perp[0] < 0)) perp = [-perp[0], -perp[1]];
  const vec = which === 'A' ? dir : perp;
  const norm = Math.hypot(vec[0], vec[1]), d = [vec[0] / norm, vec[1] / norm], half = norm / 2;

  const all = starPools(run, vid);
  const kept = (showHeldOut ? all : all.filter(e => isTrain(run, e.i))).filter(e => {
    const off = ((P.sx[e.i] - vx) * (-d[1]) + (P.sy[e.i] - vy) * d[0]) / half;
    return Math.abs(off) <= STRIP;
  });
  const t = kept.map(e => ((P.sx[e.i] - vx) * d[0] + (P.sy[e.i] - vy) * d[1]) / half);
  const nullH = p0 >= 0 ? (heightOf(run, p0) + heightOf(run, p1)) / 2 : V.height[j];
  const surplusUnits = V.height[j] - nullH;
  const sdV = V.sigma_pooled[j] < 1e100 ? Math.sqrt(V.sigma_pooled[j]) / H : 0;

  // the exact tent on the line: the surplus times the coefficient's weight along the section
  const sf = starFaces(run, vid), tent = [];
  for (let s = 0; s <= 200; s++) {
    const tt = -1 + 2 * s / 200, px = vx + tt * half * d[0], py = vy + tt * half * d[1];
    let hat = 0;
    for (const face of sf) {
      const l = baryAt(run, face.f, px, py);
      if (l && l[0] >= -1e-9 && l[1] >= -1e-9 && l[2] >= -1e-9) {
        hat = face.w0 * l[0] + face.w1 * l[1] + face.w2 * l[2]; break;
      }
    }
    tent.push([tt, surplusUnits * hat / H]);
  }

  const edges = []; for (let b = 0; b <= BINS; b++) edges.push(-1 + 2 * b / BINS);
  const width = edges[1] - edges[0];
  const bins = [], line = [];
  for (let b = 0; b < BINS; b++) {
    const a_ = edges[b], b_ = edges[b + 1];
    const m = [];
    for (let q = 0; q < kept.length; q++)
      if (t[q] >= a_ && (b_ < 1 ? t[q] < b_ : t[q] <= b_)) m.push(q);
    if (!m.length) continue;
    for (const multi of [false, true]) {
      const g = m.filter(q => isMulti(run, kept[q].i) === multi);
      if (!g.length) continue;
      let pk = 0, pn = 0;
      for (const q of g) { pk += P.k[kept[q].i]; pn += P.loans[kept[q].i]; }
      const x_ = (a_ + b_) / 2 + (multi ? 0.12 : -0.12) * width;
      // the null on the observation's scale: the log-odds of the summed expected counts
      let expNull = 0;
      const etaFit = [], etaNull = [], nn = [], hats = [];
      for (const q of g) {
        const i = kept[q].i;
        const predictionHat = P.usesMesh[i] ? kept[q].hat : 0;
        const fNull = P.ghat[i] - surplusUnits * predictionHat;
        const eN = P.base[i] + fNull / H;
        etaNull.push(eN); etaFit.push(P.base[i] + P.ghat[i] / H);
        nn.push(P.loans[i]); hats.push(predictionHat);
        expNull += P.loans[i] * (1 / (1 + Math.exp(-eN)));
      }
      const nullG = Math.log(expNull / (pn - expNull));
      const expFit = etaFit.reduce((sum, eta, q) => sum + nn[q] / (1 + Math.exp(-eta)), 0);
      line.push([x_, Math.log(expFit / (pn - expFit)) - nullG]);
      if (pk === 0) { bins.push({x: x_, zero: true, loans: pn, multi: multi}); continue; }
      const obs = Math.log((pk + 0.5) / (pn - pk + 0.5));
      const s2 = multi ? s2type[1] : s2type[0];
      const shift = centredShift(Float64Array.from(etaFit), s2), sd = Math.sqrt(s2);
      // The band is Monte Carlo and is the expensive part of a star, so it is separated from
      // the rest: sectionData leaves it out when asked, and the page fills it in bin by bin so
      // a large star never freezes the page. Everything the band needs is kept on the bin.
      const inputs = {nn: nn, etaFit: etaFit, hats: hats, shift: shift, sd: sd, sdV: sdV,
                      pn: pn, nullG: nullG, seed: Math.trunc(1e6 * (a_ + 2)) + (multi ? 1 : 0)};
      let lo = NaN, hi = NaN;
      if (withBands) { const r = bandOf(inputs); lo = r[0]; hi = r[1]; }
      bins.push({x: x_, zero: false, multi: multi, y: obs - nullG, lo: lo, hi: hi,
                 loans: pn, events: pk, pools: g.length, band: inputs});
    }
  }
  const endKind = (ex, ey) => {
    for (let q = 0; q < V.n; q++)
      if (V.active[q] === 1 && Math.hypot(V.x[q] - ex, V.y[q] - ey) < 1e-6)
        return V.free_coefficient[q] === 1 ? 'free' : 'conformity';
    return 'no vertex';
  };
  return {
    which: which, tent: tent, line: line, bins: bins,
    ends: [endKind(vx - d[0] * half, vy - d[1] * half), endKind(vx + d[0] * half, vy + d[1] * half)],
    dir: d, half: half, s2type: s2type,
  };
}

function baryAt(run, f, px, py) {
  const F = run.faces, V = run.vertices;
  const a = run.byId.get(F.v0[f]), b = run.byId.get(F.v1[f]), c = run.byId.get(F.v2[f]);
  if (a === undefined || b === undefined || c === undefined) return null;
  const x0 = V.x[a], y0 = V.y[a], x1 = V.x[b], y1 = V.y[b], x2 = V.x[c], y2 = V.y[c];
  const det = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2);
  if (det === 0) return null;
  const l0 = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / det;
  const l1 = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / det;
  return [l0, l1, 1 - l0 - l1];
}

// The pool-effect variance behind a star's bands, fitted per issue type on its training pools.
// starVarianceStart / Step / Finish is the same fit in slices; starPoolVariance runs it whole.
function starVarianceJobs(run, vid) {
  const P = run.pools, H = run.H, pools = starPools(run, vid);
  const grab = (multi) => {
    const n = [], k = [], eta = [];
    for (const e of pools) {
      if (!isTrain(run, e.i)) continue;
      if (multi !== null && isMulti(run, e.i) !== multi) continue;
      n.push(P.loans[e.i]); k.push(P.k[e.i]); eta.push(P.base[e.i] + P.ghat[e.i] / H);
    }
    return [n, k, Float64Array.from(eta)];
  };
  return [grab(null), grab(false), grab(true)];
}

function starVarianceStart(run, vid) {
  const jobs = starVarianceJobs(run, vid);
  return {jobs: jobs, i: 0, st: null, done: [0, 0, 0]};
}

function starVarianceStep(s) {
  while (s.i < 3) {
    if (!s.st) {
      s.st = varianceStart(s.jobs[s.i][0], s.jobs[s.i][1], s.jobs[s.i][2]);
      if (!s.st) { s.done[s.i] = null; s.i++; continue; }
    }
    if (varianceStep(s.st)) return true;
    s.done[s.i] = varianceFinish(s.st);
    s.st = null; s.i++;
    return s.i < 3;
  }
  return false;
}

function starVarianceFinish(s) {
  const all = s.done[0] === null ? 0 : s.done[0];
  return [s.done[1] === null ? all : s.done[1], s.done[2] === null ? all : s.done[2]];
}

function starPoolVariance(run, vid) {
  const s = starVarianceStart(run, vid);
  while (starVarianceStep(s)) { /* keep going */ }
  return starVarianceFinish(s);
}

// Open the free coefficient contributing the largest conditional change at this pool.
function poolVertex(run, i) {
  const P = run.pools, V = run.vertices, F = run.faces;
  let corners = null, l = null;
  for (let f = 0; f < F.n; f++) {
    const b = baryAt(run, f, P.sx[i], P.sy[i]);
    if (b && b.every(x => x >= -1e-7)) { corners = [F.v0[f], F.v1[f], F.v2[f]]; l = b; break; }
  }
  if (!corners) return null;
  let best = corners[0], magnitude = -1;
  for (let j = 0; j < V.n; j++) {
    if (V.active[j] !== 1 || V.free_coefficient[j] !== 1 || V.parent0[j] < 0) continue;
    const id = V.id[j], w = influence(run, id);
    const hat = corners.reduce((s, v, q) => s + l[q] * (w.get(v) || 0), 0);
    if (hat <= 1e-9) continue;
    const delta = V.height[j] - (heightOf(run, V.parent0[j]) + heightOf(run, V.parent1[j])) / 2;
    const score = Math.abs(delta * hat);
    if (score > magnitude) { magnitude = score; best = id; }
  }
  return best;
}

function poolDeviance(n, k, p) {
  if (!(p > 0 && p < 1)) return NaN;
  return 2 * ((k > 0 ? k * Math.log(k / (n * p)) : 0) + (k < n ? (n - k) * Math.log((n - k) / (n * (1 - p))) : 0));
}

const api = {
  decodeRun: decodeRun, setQuadrature: setQuadrature, address: address, addressName: addressName,
  parseAddress: parseAddress, vertexName: vertexName, influence: influence, starFaces: starFaces,
  nodalFaces: nodalFaces, starPools: starPools, cardOf: cardOf, poolVariance: poolVariance,
  centredShift: centredShift, starPoolVariance: starPoolVariance, sectionData: sectionData,
  starVarianceStart: starVarianceStart, starVarianceStep: starVarianceStep, starVarianceFinish: starVarianceFinish,
  baryAt: baryAt, isTrain: isTrain, isMulti: isMulti, inFit: inFit, parentsOf: parentsOf,
  bandOf: bandOf, bandStart: bandStart, bandStep: bandStep, bandFinish: bandFinish,
  heightOf: heightOf, STRIP: STRIP, BINS: BINS, DRAWS: DRAWS,
  poolVertex: poolVertex, poolDeviance: poolDeviance,
};
if (typeof module !== 'undefined' && module.exports) module.exports = api;
root.StarCore = api;

})(typeof globalThis !== 'undefined' ? globalThis : this);
