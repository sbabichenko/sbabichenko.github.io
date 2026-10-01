// Browser regression: real solver output, final SVG geometry, compared-model
// failures, retained starts and gaps in a sweep. Exported for a shared harness.
const assert = require('node:assert/strict');
async function checkResultValidity(browser, base) {
  const page = await browser.newPage({serviceWorkers: 'block'}), errors = [];
  page.on('pageerror', e => errors.push(String(e)));
  await page.addInitScript(() => { window.coi = {shouldRegister: () => false}; });
  await page.route('https://**', r => r.abort());
  const settled = async () => {
    await page.waitForFunction(() => typeof lastResult !== 'undefined' && lastResult && !inFlight && !pending);
    await page.waitForFunction(() => document.querySelectorAll('#results .js-plotly-plot').length > 0 &&
      [...document.querySelectorAll('#results .js-plotly-plot')].every(g => !plotWork.has(g)));
  };
  try {
    await page.goto(base + '/noisestate/#game=tr'); await settled();
    // A valid data array was not sufficient: the old SVG ended at time 1 on a
    // 0..3 axis. Test the rendered endpoint in pixels, including band boundaries.
    for (const T of [3, 12, 3]) {
      await page.evaluate(T => { values.tr.T = T; renderAll(); requestSolve(0); }, T);
      await page.waitForFunction(T => lastResult.T === T && !inFlight, T); await settled();
      const endpoints = await page.evaluate(() => [...document.querySelectorAll('#pgrid .js-plotly-plot')].flatMap(g =>
        [...g.querySelectorAll('.scatterlayer .trace.scatter')].flatMap((trace, i) => {
          const line = trace.querySelector('.js-line'); if (!line) return [];
          const p = line.getPointAtLength(line.getTotalLength()), t = g.data[i];
          return [{x: p.x, expectedX: g._fullLayout.xaxis.l2p(t.x.at(-1)), y: p.y, expectedY: g._fullLayout.yaxis.l2p(t.y.at(-1))}];
        })));
      assert(endpoints.length >= 12);
      for (const p of endpoints) { assert(Math.abs(p.x-p.expectedX) < .1, JSON.stringify(p)); assert(Math.abs(p.y-p.expectedY) < .1, JSON.stringify(p)); }
    }
    await page.evaluate(() => { game = 'ch4'; renderAll(); requestSolve(0); });
    await page.waitForFunction(() => lastResult.name === 'ch4_kyle_back' && !inFlight); await settled();
    assert.equal(await page.locator('#chip').innerText(), 'Solved');
    assert.match(await page.locator('#results .cards').first().innerText(), /2\.1165/);
    // Slow signals legitimately need a longer window. Keep the useful curves,
    // label their truncated costs and offer the actual window remedy.
    await page.evaluate(() => { values.ch4.gamma1 = .1; renderAll(); requestSolve(0); });
    await page.waitForFunction(() => lastResult.model.params.gamma1 === .1 && !inFlight); await settled();
    assert.match(await page.locator('#results .cards').first().innerText(), /truncated at lag/);
    assert.match(await page.locator('#statusfix').innerText(), /window/);
    await page.evaluate(() => { game = 'ch6'; opts.reference = false; renderAll(); requestSolve(0); });
    await page.waitForFunction(() => lastResult.name.includes('ch6') && lastResult.engine !== 'ch6-markov' && !inFlight); await settled();
    assert.match(await page.locator('#statustext').innerText(), /opaque/i);
    assert.match(await page.locator('#results').innerText(), /Diagnostics: Opaque/);
    await page.evaluate(() => { window.goodResult = structuredClone(lastResult); });
    // Reproduce the reported slider order: default -> failed low epsilon -> high.
    await page.evaluate(() => { values.ch6.eps = .1; renderAll(); requestSolve(0); });
    await page.waitForFunction(() => !inFlight && lastResult.model?.params.eps === .1);
    assert.equal(await page.locator('#results .js-plotly-plot').count(), 0);
    assert.equal(await page.locator('#results .cards').count(), 0);
    assert.equal(await page.locator('#use-reference').count(), 1);
    await page.evaluate(() => { values.ch6.eps = 1; renderAll(); requestSolve(0); });
    await page.waitForFunction(() => !inFlight && lastResult.model?.params.eps === 1);
    console.log('High epsilon after warm start:', await page.evaluate(() => ({problem: displayProblem(lastResult), peak: Math.max(...lastResult.compare.samples.kernels.P.wZ.map(Math.abs))})));
    if (await page.evaluate(() => !!displayProblem(lastResult))) {
      assert.equal(await page.locator('#results .js-plotly-plot').count(), 0);
      await page.locator('#solve-cold').click();
      await page.waitForFunction(() => !inFlight && lastResult.model?.params.eps === 1 && !displayProblem(lastResult));
    }
    await settled();
    const peak = await page.evaluate(() => Math.max(...lastResult.compare.samples.kernels.P.wZ.map(Math.abs)));
    assert(peak < 2, `opaque price response ${peak}`);
    // Only the opaque result is invalid. It must control the status and the
    // plotted comparison, and must not poison its saved warm start.
    const injected = await page.evaluate(() => {
      const r = structuredClone(goodResult), warm = lastStartCompare[game];
      r.compare.checks.find(d => d.name === 'window').value = .5;
      r.compare.start = {invalidMarker: true};
      const id = ++reqId; inFlight = {id, game, key: JSON.stringify([currentModel(), currentRequest()])};
      onSolved({id, result: JSON.stringify(r), wall: .1});
      return {warmUnchanged: lastStartCompare[game] === warm, issue: displayProblem(r)};
    });
    assert(injected.warmUnchanged); assert.match(injected.issue, /Opaque.*50\.0%/);
    assert.equal(await page.locator('#chip').innerText(), 'Result not reliable');
    assert.equal(await page.locator('#results .js-plotly-plot').count(), 0);
    await page.locator('#use-reference').click();
    await page.waitForFunction(() => lastResult.engine === 'ch6-markov' && !inFlight); await settled();
    assert.equal(await page.locator('#chip').innerText(), 'Equilibrium');
    assert(!new URL(page.url()).hash.includes('method=solver'));
    // Drive the real sweep receiver with good/bad/good responses. The missing
    // middle result is an explicit null in both markets, never bridged by a line.
    const sweepData = await page.evaluate(() => {
      const r = structuredClone(lastResult), bad = structuredClone(r);
      bad.compare.checks.push({name: 'resolution', ok: false, value: .01, threshold: 1e-6});
      const jobs = [0, 1, 2].map(() => ({model: currentModel(), request: {compare: {}}}));
      sweep = {game, key: 'eps', label: 'eps', xs: [.1, .2, 1], i: 0, id: 7000, costs: {}, naive: {}, values: {...values[game]}, jobs, running: true, failed: 0, t0: performance.now(), base: sweepBase('eps')};
      const saved = nextSweep; nextSweep = () => {};
      try { for (const q of [r, bad, r]) onSweepResult({id: sweep.id+sweep.i, result: JSON.stringify(q)}); }
      finally { nextSweep = saved; sweep.running = false; sweep.wall = 0; }
      return {costs: sweep.costs, compared: sweep.naive, failed: sweep.failed};
    });
    assert.equal(sweepData.failed, 1);
    for (const series of [...Object.values(sweepData.costs), ...Object.values(sweepData.compared)]) {
      assert.equal(series.length, 3); assert.equal(series[1][1], null); assert(Number.isFinite(series[0][1]) && Number.isFinite(series[2][1]));
    }
    assert.deepEqual(errors, []);
    return {animatedHorizonEndpoints: true, kyleCosts: true, truncationLabel: true, comparisonDiagnostics: true,
      failedCurvesWithheld: true, sliderRecovery: true, rejectedWarmStart: true, referenceRecovery: true, sweepGaps: true};
  } finally { await page.close(); }
}
module.exports = {checkResultValidity};

if (require.main === module) {
  const {chromium} = require('playwright');
  (async () => {
    const browser = await chromium.launch({headless: true});
    try { console.log(JSON.stringify(await checkResultValidity(browser, process.argv[2] || 'http://127.0.0.1:8788'))); }
    finally { await browser.close(); }
  })().catch(e => { console.error(e); process.exitCode = 1; });
}
