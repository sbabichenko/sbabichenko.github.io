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
module.exports = {checkExplorerRetention};
if (require.main === module) {
  const {chromium} = require('playwright');
  (async () => {
    const browser = await chromium.launch({headless: true});
    try { console.log(JSON.stringify(await checkExplorerRetention(browser, process.argv[2] || 'http://127.0.0.1:8788'), null, 2)); }
    finally { await browser.close(); }
  })().catch(e => {console.error(e); process.exitCode = 1;});
}
