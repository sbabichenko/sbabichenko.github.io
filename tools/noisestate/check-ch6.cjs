// Independent SciPy fixtures cover both markets, slider-box corners and gamma=0.
// Run: node tools/noisestate/check-ch6.cjs
const assert = require('node:assert/strict');
const M = require('../../static/noisestate/ch6-markov.js');
const G = require('../../static/noisestate/gaussian-paths.js');
const reference = require('./ch6-reference.json');
let worst = 0;
function close(a, b) {
  if (Array.isArray(b)) { assert.equal(a.length, b.length); b.forEach((v, i) => close(a[i], v)); return; }
  assert(Number.isFinite(a));
  const error = Math.abs(a - b) / Math.max(1, Math.abs(b));
  worst = Math.max(worst, error); assert(error < 1e-8, `${a} differs from ${b}`);
}
for (const c of reference.cases) {
  const e = M.solve(c.market, c.params), samples = M.sample(e, reference.ages);
  assert(e.residual < 1e-9); close(e.root, c.root);
  for (const [a, cost] of Object.entries(c.costs)) close(e.costs[a], cost);
  for (const [name, values] of Object.entries(c.kernels))
    close(values.map((_, i) => ['wV', 'wZ', 'wY'].map(w => samples.kernels[name][w][i])), values);
  for (const [name, values] of Object.entries(c.deviation)) close(samples.deviation[name], values);
  const paths = M.paths(e), covariance = factor => factor.map(row => factor.map(other => row.reduce((s, x, j) => s+x*other[j], 0)));
  close(paths.transition, c.paths.transition);
  close(covariance(paths.noise), c.paths.noise_covariance);
  close(covariance(paths.initial), c.paths.initial_covariance);
  const moments = G.simulate(paths, 0, () => { throw Error('No draws requested'); });
  for (const [t, values] of Object.entries(c.paths.variances))
    close(['V','Q','P','D'].map(name => moments[name].sd[Math.round(+t/paths.h)]**2), values);
  if (c.params.gamma > 0) assert(e.delta > 0 && e.beta > 0 && e.pq < 0);
}
for (const key of ['eps', 'rho', 'sigma_Z']) {
  assert.throws(() => M.solve('transparent', { eps: .2, gamma: .1, rho: .5, sigma_Z: 1, [key]: 0 }));
  assert.throws(() => M.solve('opaque', { eps: .2, gamma: .1, rho: .5, sigma_Z: 1, [key]: NaN }));
}
assert.throws(() => M.solve('transparent', { eps: .2, gamma: -.1, rho: .5, sigma_Z: 1 }));
assert.throws(() => M.payload({ name: 'arbitrary game', params: { eps: .2, gamma: .1, rho: .5, sigma_Z: 1 } }));
// The old absolute Newton stopping threshold rounded tiny positive penalties
// to zero inventory reversion. These are independent stationary-cost limits,
// including the discontinuity from gamma exactly zero (random-walk inventory).
const beta0 = (Math.sqrt(21) - 1) / 4;
const deltaLimits = { transparent: 10 / 7, opaque: 4 * beta0 / (1 + 2.4 * beta0) };
for (const market of ['transparent', 'opaque']) {
  for (const gamma of [1e-4, 1e-8, 1e-12, 1e-16]) {
    const e = M.solve(market, { eps: .2, gamma, rho: .5, sigma_Z: 1 });
    assert(e.delta > 0 && e.pq < 0 && e.residual < 1e-12);
    assert(Math.abs(e.delta / gamma / deltaLimits[market] - 1) < 6 * gamma + 1e-12);
    assert(Math.abs(e.accounts.inventory * 2 * deltaLimits[market] - 1) < 6 * gamma + 1e-12);
  }
  const zero = M.solve(market, { eps: .2, gamma: 0, rho: .5, sigma_Z: 1 });
  assert.equal(zero.delta, 0); assert.equal(zero.accounts.inventory, 0);
}
const propagation = require('./ch6-kernel-reference.json');
for (const c of propagation.cases) {
  const samples = M.sample(c.equilibrium, c.ages);
  for (const [name, values] of Object.entries(c.kernels))
    close(values.map((_, i) => ['wV', 'wZ', 'wY'].map(w => samples.kernels[name][w][i])), values);
  for (const [name, values] of Object.entries(c.deviation)) close(samples.deviation[name], values);
}
console.log(`${reference.cases.length} equilibrium and ${propagation.cases.length} propagation reference cases passed; worst relative difference ${worst.toExponential(2)}`);
