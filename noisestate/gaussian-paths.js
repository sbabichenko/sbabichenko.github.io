/* Gaussian state paths, O(cells) storage/work for a fixed state dimension.
 * The caller supplies a seeded independent standard-normal generator per draw.
 * No truncated shock history, browser state, or model-specific equations here.
 */
(function (root) {
  "use strict";
  const dot = (a, b) => a.reduce((s, x, i) => s + x*b[i], 0);
  const mv = (a, v) => a.map(row => dot(row, v));
  const transpose = a => a[0].map((_, j) => a.map(row => row[j]));
  const mul = (a, b) => a.map(row => b[0].map((_, j) => row.reduce((s, x, k) => s+x*b[k][j], 0)));
  function simulate(P, draws, normalForDraw) {
    const names = Object.keys(P.outputs), n = P.transition.length;
    const t = Array.from({length: P.cells+1}, (_, i) => i*P.h);
    const out = Object.fromEntries(names.map(name => [name, {t, draws: [], mean: t.map(() => 0), sd: []}]));
    let covariance = mul(P.initial, transpose(P.initial));
    const noise = mul(P.noise, transpose(P.noise)), AT = transpose(P.transition);
    for (let j = 0; j <= P.cells; j++) {
      for (const name of names) {
        const c = P.outputs[name];
        out[name].sd.push(Math.sqrt(Math.max(0, dot(c, mv(covariance, c)))));
      }
      if (j < P.cells) covariance = mul(mul(P.transition, covariance), AT).map((row, i) => row.map((x, k) => x+noise[i][k]));
    }
    for (let d = 0; d < draws; d++) {
      const normal = normalForDraw(d), gaussian = () => Array.from({length: n}, normal);
      let state = mv(P.initial, gaussian());
      const values = Object.fromEntries(names.map(name => [name, []]));
      for (let j = 0; j <= P.cells; j++) {
        for (const name of names) values[name].push(dot(P.outputs[name], state));
        if (j < P.cells) {
          const shock = mv(P.noise, gaussian());
          state = mv(P.transition, state).map((x, i) => x+shock[i]);
        }
      }
      for (const name of names) out[name].draws.push(values[name]);
    }
    return out;
  }
  const api = {simulate};
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.GaussianPaths = api;
})(typeof self !== "undefined" ? self : globalThis);
