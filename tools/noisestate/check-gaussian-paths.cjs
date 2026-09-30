const assert = require('node:assert/strict');
const {simulate} = require('../../static/noisestate/gaussian-paths.js');
const {solve, paths} = require('../../static/noisestate/ch6-markov.js');
// Exact deterministic recursion: confirms the initial draw and each innovation
// enter at the intended date, and coupled outputs share the same state draw.
const fixture = {h: .5, cells: 2, transition: [[.5,0],[1,1]], noise: [[1,0],[0,2]], initial: [[2,0],[0,3]], outputs: {x:[1,0],y:[0,1],sum:[1,1]}};
let sequence = 0;
const result = simulate(fixture, 1, () => () => ++sequence);
assert.deepEqual(result.x.draws[0], [2,4,7]);
assert.deepEqual(result.y.draws[0], [6,16,32]);
assert.deepEqual(result.sum.draws[0], [8,20,39]);
assert.deepEqual(result.x.t, [0,.5,1]);
assert.equal(result.x.sd[0], 2); assert.equal(result.y.sd[0],3);
// Independent seeded normal draws: finite samples should reproduce the checked
// covariance recursion in both stationary inventory and the gamma=0 random walk.
let seed = 823716;
function uniform() { seed = (Math.imul(1664525,seed)+1013904223)>>>0; return (seed+.5)/4294967296; }
function normal() { return Math.sqrt(-2*Math.log(uniform()))*Math.cos(2*Math.PI*uniform()); }
for (const gamma of [0,.1]) {
  const eq = solve('transparent', {eps:.2,gamma,rho:.5,sigma_Z:1});
  const spec = paths(eq,.1,10), sampled = simulate(spec,5000,()=>normal);
  for (const [name, s] of Object.entries(sampled)) for (const j of [0,10]) {
    const mean=s.draws.reduce((a,v)=>a+v[j],0)/s.draws.length;
    const variance=s.draws.reduce((a,v)=>a+(v[j]-mean)**2,0)/s.draws.length;
    if (!s.sd[j]) { assert.equal(variance,0); continue; }
    assert(Math.abs(mean)<.06*s.sd[j], `${name} mean at ${j}`);
    assert(Math.abs(variance/s.sd[j]**2-1)<.08, `${name} variance at ${j}`);
  }
}
console.log('Gaussian path recursion and sampled moments passed');
