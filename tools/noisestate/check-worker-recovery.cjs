// Exercise the real worker protocol with a native-runtime trap, then retry normally.
const assert = require('node:assert/strict');
async function checkWorkerRuntimeRecovery(browser, base) {
  const page = await browser.newPage({serviceWorkers: 'block', reducedMotion: 'reduce'});
  const errors = [];
  page.on('pageerror', error => errors.push(String(error)));
  await page.addInitScript(() => {
    window.coi = {shouldRegister: () => false};
    const NativeWorker = window.Worker;
    window.workerStarts = 0; window.workerTerminations = 0;
    window.Worker = class extends NativeWorker {
      constructor(...args) { super(...args); window.workerStarts++; }
      terminate() { window.workerTerminations++; return super.terminate(); }
    };
  });
  await page.route('https://**', route => route.abort());
  const solverURL = '**/noisestate/noisestate.js*';
  await page.route(solverURL, route => route.fulfill({contentType: 'application/javascript', body: `
    self.NoiseState = async () => ({
      cwrap(name) {
        if (name === 'ns_version') return () => 1;
        if (name === 'ns_free') return () => {};
        return () => { throw new WebAssembly.RuntimeError('unreachable runtime fixture'); };
      },
      UTF8ToString() { return 'runtime fixture'; }
    });
  `}));
  try {
    await page.goto(base + '/noisestate/#game=ch3');
    await page.waitForFunction(() => document.querySelector('#chip').textContent === 'Failed');
    const failed = await page.evaluate(() => ({
      released: worker === null && !workerReady && !inFlight && !pending,
      starts: workerStarts, terminations: workerTerminations,
      message: document.querySelector('#statustext').textContent,
    }));
    assert.equal(failed.released, true, 'native trap retained the failed runtime');
    assert.equal(failed.starts, 1);
    assert.equal(failed.terminations, 1);
    assert.match(failed.message, /solver stopped/i);
    assert.match(failed.message, /unreachable runtime fixture/);
    assert.equal(await page.locator('#solvebtn').isEnabled(), true);
    await page.unroute(solverURL);
    await page.locator('#solvebtn').click();
    await page.waitForFunction(() => lastResult && lastResult.ok && !inFlight, null, {timeout: 60000});
    assert.equal(await page.evaluate(() => workerStarts), 2, 'retry did not create a fresh runtime');
    assert.equal(await page.evaluate(() => lastResult.converged), true);
    assert.deepEqual(errors, []);
    return {nativeTrapReleasesWorker: true, retryCreatesFreshWorker: true, realSolveAfterRetry: true};
  } finally { await page.close(); }
}
module.exports = {checkWorkerRuntimeRecovery};
