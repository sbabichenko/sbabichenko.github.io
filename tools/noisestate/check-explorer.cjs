/* Browser regression for Plotly disposal and stale async rendering.
 * With a local site server running:
 *   NODE_PATH=/path/to/playwright/node_modules node tools/noisestate/check-explorer.cjs http://127.0.0.1:8788
 * The exported function also runs in a caller's existing headless browser.
 */
const assert = require('node:assert/strict');
async function checkExplorerRetention(browser, base) {
  // The COI service worker can fetch scripts outside Playwright's routing.
  // Disable it in this timing test so the lazy-load gate is deterministic.
  const page = await browser.newPage({viewport: {width: 1100, height: 900}, reducedMotion: 'reduce', serviceWorkers: 'block'});
  const errors = [];
  page.on('pageerror', e => errors.push(String(e)));
  await page.addInitScript(() => { window.coi = {shouldRegister: () => false}; });
  await page.route('https://**', route => route.abort());
  let release;
  const gate = new Promise(resolve => { release = resolve; });
  await page.route('**/plotly-*.js*', async route => { await gate; await route.continue(); });
  try {
    await page.goto(base + '/noisestate/#game=ch6');
    await page.waitForFunction(() => typeof lastResult !== 'undefined' && lastResult && !inFlight);
    assert.equal(await page.evaluate(() => !!window.Plotly), false);
    // These queued plots must not attach handlers or data after their nodes disappear.
    await page.evaluate(() => { for (let i = 0; i < 4; i++) renderResults(lastResult); });
    release();
    await page.waitForFunction(() => document.querySelectorAll('#results .js-plotly-plot').length >= 5);
    const cdp = await page.context().newCDPSession(page), rows = [];
    for (let i = 0; i < 16; i++) {
      await page.evaluate(() => { renderResults(lastResult); plotSnapshot = null; });
      await page.evaluate(() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r))));
      await cdp.send('HeapProfiler.collectGarbage');
      const dom = await cdp.send('Memory.getDOMCounters');
      const heap = await cdp.send('Runtime.getHeapUsage');
      const listeners = await cdp.send('Runtime.evaluate', {
        expression: '(getEventListeners(window).resize||[]).length', includeCommandLineAPI: true, returnByValue: true,
      });
      rows.push({iteration: i, ...dom, heap: heap.usedSize, resize: listeners.result.value});
    }
    const first = rows[3], last = rows.at(-1);
    assert.equal(last.resize, first.resize, 'responsive chart handlers leaked');
    assert(last.nodes <= first.nodes + 50, 'detached chart nodes leaked');
    assert(last.jsEventListeners <= first.jsEventListeners + 2, 'chart event handlers leaked');
    // Exercise the animation promise chain being disposed before it settles.
    await page.emulateMedia({reducedMotion: 'no-preference'});
    await page.evaluate(async () => {
      prevResult = lastResult;
      for (let i = 0; i < 8; i++) {
        renderResults(lastResult);
        await new Promise(r => requestAnimationFrame(r));
      }
      await new Promise(r => setTimeout(r, 800));
    });
    await cdp.send('HeapProfiler.collectGarbage');
    const after = await cdp.send('Runtime.evaluate', {
      expression: '(getEventListeners(window).resize||[]).length', includeCommandLineAPI: true, returnByValue: true,
    });
    assert.equal(after.result.value, last.resize, 'cancelled animations retained charts');
    assert.deepEqual(errors, []);
    return rows;
  } finally { release(); await page.close(); }
}
async function checkExplorerCancellation(browser, base) {
  const page = await browser.newPage({serviceWorkers: 'block', reducedMotion: 'reduce'});
  await page.addInitScript(() => {
    window.coi = {shouldRegister: () => false};
    const NativeWorker = window.Worker;
    window.workerStarts = 0;
    window.Worker = class extends NativeWorker {
      constructor(...args) { super(...args); window.workerStarts++; }
    };
  });
  await page.route('https://**', route => route.abort());
  const workerURL = '**/noisestate/worker.js*';
  await page.route(workerURL, route => route.abort());
  try {
    await page.goto(base + '/noisestate/#game=ch6');
    await page.waitForFunction(() => document.querySelector('#chip').textContent === 'Failed');
    assert.equal(await page.locator('#solvebtn').isEnabled(), true);
    await page.unroute(workerURL);
    await page.locator('#solvebtn').click();
    await page.waitForFunction(() => lastResult && !inFlight);
    // Hold one request in the real page so Stop is deterministic even on fast hardware.
    const before = await page.evaluate(() => {
      worker.postMessage = m => { window.heldRequest = m; };
      return workerStarts;
    });
    await page.locator('#solvebtn').click();
    await page.waitForFunction(() => !!inFlight);
    const stoppedID = await page.evaluate(() => inFlight.id);
    await page.locator('#stopbtn').click();
    await page.waitForTimeout(250);
    assert.equal(await page.evaluate(() => workerStarts), before, 'Stop restarted the worker');
    assert.equal(await page.evaluate(() => worker === null && !workerReady && !inFlight && !pending), true);
    assert.equal(await page.locator('#chip').innerText(), 'Stopped');
    await page.evaluate(id => {
      const newer = {id: id+100}; inFlight = newer;
      onSolved({id, result: 'stale payload must never be parsed'});
      if (inFlight !== newer) throw Error('stale result replaced the active request');
      inFlight = null;
    }, stoppedID);
    await page.locator('#solvebtn').click();
    await page.waitForFunction(id => reqId > id && lastResult && !inFlight && workerReady, stoppedID);
    assert.equal(await page.evaluate(() => workerStarts), before+1);
    // A failed lazy solver import must also be retryable in the same worker.
    const solverURL = '**/noisestate/noisestate.js*';
    await page.route(solverURL, route => route.abort());
    await page.locator('#ch6-view').selectOption('solver');
    await page.waitForFunction(() => document.querySelector('#chip').textContent === 'Error');
    await page.unroute(solverURL);
    await page.locator('#solvebtn').click();
    await page.waitForFunction(() => lastResult && lastResult.engine !== 'ch6-markov' && !inFlight);
    assert((await page.locator('.cards').first().innerText()).includes('1.4043'));
    // Failure of the old tab must still service the new tab's queued request.
    await page.evaluate(() => { worker.postMessage = m => { window.heldRequest = m; }; requestSolve(0); });
    await page.waitForFunction(() => !!inFlight);
    const failedID = await page.evaluate(() => { const id=inFlight.id; game='ch3'; renderAll(); requestSolve(0); return id; });
    await page.waitForFunction(() => pending);
    await page.evaluate(id => onSolved({id, result: JSON.stringify({ok: false, error: 'old tab failure'})}), failedID);
    assert.equal(await page.evaluate(id => inFlight && inFlight.id > id && inFlight.game === 'ch3', failedID), true);
    await page.evaluate(() => onSolved({id: inFlight.id, result: 'unreadable result'}));
    assert.equal(await page.locator('#chip').innerText(), 'Failed');
    assert.equal(await page.evaluate(() => worker === null && !inFlight), true);
    const custom = await page.evaluate(async () => {
      const NativeWorker = window.Worker;
      let created;
      window.Worker = class { constructor() { created=this; this.sent=[]; } postMessage(m) { this.sent.push(m); } terminate() {} };
      try {
        game='custom'; renderAll(); startWorker(); pending=true; customReady();
        created.onmessage({data: {type: 'ready'}});
        await new Promise(r => setTimeout(r, 25));
        return {sent: created.sent.length, pending, active: !!inFlight};
      } finally { window.Worker=NativeWorker; }
    });
    assert.deepEqual(custom, {sent: 0, pending: false, active: false});
    return {workerStartupRetry: true, stopStaysIdle: true, staleResultIgnored: true, lazyImportRetry: true,
      queuedTabSurvivesFailure: true, malformedResultRetryable: true, customWaitsForSolve: true};
  } finally { await page.close(); }
}
async function checkSweepRecovery(browser, base) {
  const page = await browser.newPage({serviceWorkers: 'block', reducedMotion: 'reduce'});
  await page.addInitScript(() => { window.coi = {shouldRegister: () => false}; });
  await page.route('https://**', route => route.abort());
  try {
    await page.goto(base + '/noisestate/#game=ch6');
    await page.waitForFunction(() => lastResult && !inFlight);
    await page.evaluate(() => { SWEEP_POINTS = 3; });
    const workerURL = '**/noisestate/worker.js*';
    await page.route(workerURL, route => route.abort());
    await page.locator('#sweepbtn').click();
    await page.waitForFunction(() => sweep && !sweep.running && sweep.error);
    assert.equal(await page.evaluate(() => sweepWorker === null && sweepReady === null), true);
    await page.unroute(workerURL);
    await page.locator('#sweepbtn').click();
    await page.waitForFunction(() => sweep && !sweep.running && sweep.i === 3);
    assert.equal(await page.evaluate(() => sweep.failed), 0);
    // A worker can fail after ready has already resolved. That must stop the UI too.
    await page.evaluate(() => { sweepWorker.postMessage = () => {}; });
    await page.locator('#sweepbtn').click();
    await page.waitForFunction(() => sweep.running);
    await page.evaluate(() => sweepWorker.onerror({message: 'simulated runtime failure'}));
    assert.equal(await page.evaluate(() => !sweep.running && sweepWorker === null), true);
    assert((await page.locator('#sweepnote').innerText()).includes('simulated runtime failure'));
    // Cancel during startup, then start again before the old await continuation runs.
    const result = await page.evaluate(async () => {
      const NativeWorker = window.Worker, created = [];
      window.Worker = class {
        constructor() { created.push(this); this.sent = []; }
        terminate() { this.terminated = true; }
        postMessage(m) { this.sent.push(m); }
      };
      try {
        const old = runSweep(); stopSweep();
        const current = runSweep();
        created[0].onmessage({data: {type: 'ready'}});
        created[1].onmessage({data: {type: 'ready'}});
        await Promise.all([old, current]);
        const heldEps = values.ch6.eps;
        values.ch6.eps = .9; game = 'ch3';
        sweep.i = 1; nextSweep();
        const message = created[1].sent.at(-1);
        const frozen = message.model.name === 'ch6_transparent_market' && message.model.params.eps === heldEps
          && message.request.method === 'ch6-markov' && !sweep.jobs[1].request.start;
        game = 'ch6'; values.ch6.eps = heldEps;
        const answer = {oldTerminated: created[0].terminated, oldSent: created[0].sent.length,
          currentSent: created[1].sent.length, currentRunning: sweep.running, frozen};
        stopSweep(); return answer;
      } finally { window.Worker = NativeWorker; }
    });
    assert.deepEqual(result, {oldTerminated: true, oldSent: 0, currentSent: 2, currentRunning: true, frozen: true});
    return {startupRetry: true, runtimeFailureStops: true, startupCancellation: true, frozenParameters: true};
  } finally { await page.close(); }
}
module.exports = {checkExplorerRetention, checkExplorerCancellation, checkSweepRecovery};
if (require.main === module) {
  const {chromium} = require('playwright');
  (async () => {
    const browser = await chromium.launch({headless: true});
    try {
      const base = process.argv[2] || 'http://127.0.0.1:8788';
      console.log(JSON.stringify(await checkExplorerRetention(browser, base), null, 2));
      console.log(JSON.stringify(await checkExplorerCancellation(browser, base)));
      console.log(JSON.stringify(await checkSweepRecovery(browser, base)));
    }
    finally { await browser.close(); }
  })().catch(e => {console.error(e); process.exitCode = 1;});
}
