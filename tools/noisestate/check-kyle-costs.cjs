// The full pricing loss adds only a term independent of the maker's control.
// Verify unchanged strategies, a stable cost, and the formerly under-resolved cases.
// node tools/noisestate/check-kyle-costs.cjs [path/to/noisestate-mt.js]
const assert = require('node:assert/strict'), fs = require('node:fs'), path = require('node:path'), vm = require('node:vm');
const root = path.resolve(__dirname, '../..');
const src = fs.readFileSync(path.join(root, 'static/noisestate/explorer.js'), 'utf8');
const presets = vm.runInNewContext(src.slice(src.indexOf('const PRESETS = {'), src.indexOf('\nconst EXAMPLES')) + '\nPRESETS;');
const yaml = require(path.join(root, 'static/noisestate/js-yaml.min.js'));
(async () => {
  const entry = path.resolve(process.argv[2] || path.join(root, 'static/noisestate/noisestate.js'));
  const mod = await require(entry)({wasmBinary: fs.readFileSync(entry.replace(/\.js$/, '.wasm'))});
  const solve = mod.cwrap('ns_solve', 'number', ['string', 'string']), free = mod.cwrap('ns_free', null, ['number']);
  function run(params, {window = 12, nodes = 32, full = true} = {}) {
    const model = yaml.load(presets.ch4.yaml);
    Object.assign(model.params, params); model.horizon.window = window; model.numerics.nodes = nodes;
    if (!full) model.agents.market_maker.loss = model.agents.market_maker.loss.filter(t => !(t[1] === 'V' && t[2] === 'V'));
    const ptr = solve(JSON.stringify(model), JSON.stringify({start_policy: 'coarse', stability: false, threads: entry.includes('-mt.js') ? 4 : 1}));
    try { return JSON.parse(mod.UTF8ToString(ptr)); } finally { free(ptr); }
  }
  for (const params of [{}, {eps: .05}, {sigma_Z: .2}]) {
    const r = run(params), old = run(params, {full: false}), longer = run(params, {window: 18}), finer = run(params, {nodes: 40});
    for (const q of [r, longer, finer]) {
      assert(q.ok && q.converged, JSON.stringify({params, nodes:q.numerics, checks:q.checks.filter(d=>d.ok===false)}));
      assert.deepEqual(q.checks.filter(d => d.ok === false), []);
      assert(q.costs.market_maker > 0);
      assert(Math.abs(q.costs.market_maker - r.costs.market_maker) < 1e-6);
      assert(Math.abs(q.costs.trader1 - r.costs.trader1) < 1e-6);
    }
    assert.deepEqual(r.kernels, old.kernels, 'adding V squared changed an equilibrium strategy');
    assert.equal(r.costs.trader1, old.costs.trader1);
    assert(Math.abs(r.costs.market_maker - old.costs.market_maker - 12) < 1e-10);
    console.log(JSON.stringify({params, makerCost: r.costs.market_maker, windowChange: longer.costs.market_maker-r.costs.market_maker, gridChange: finer.costs.market_maker-r.costs.market_maker}));
  }
  console.log('Kyle–Back cost and resolution checks passed');
  process.exit(0); // the threaded Emscripten build owns an idle worker pool
})().catch(e => { console.error(e); process.exit(1); });
