/* Homepage motion should resume after a failed fetch and sleep while unseen. */
const assert = require('node:assert/strict');
async function checkCardLifecycle(browser, base) {
  const page = await browser.newPage({viewport: {width: 1100, height: 650}, serviceWorkers: 'block'});
  const errors = [];
  page.on('pageerror', e => errors.push(String(e)));
  await page.addInitScript(() => {
    const later = window.setTimeout, cancel = window.clearTimeout;
    window.cardTimers = new Set();
    window.setTimeout = (fn, delay, ...args) => {
      const card = delay >= 1500 && new Error().stack.includes('/js/cards.js');
      let id = later(() => { cardTimers.delete(id); fn(...args); }, delay);
      if (card) cardTimers.add(id);
      return id;
    };
    window.clearTimeout = id => { cardTimers.delete(id); return cancel(id); };
    window.cardHidden = false;
    Object.defineProperty(document, 'hidden', {get: () => window.cardHidden});
  });
  let requests = 0;
  await page.route('**/card-nsheads-think.json*', route => {
    requests++;
    return requests === 1 ? route.fulfill({status: 503, body: 'temporary failure'}) : route.continue();
  });
  await page.route('https://**', route => route.abort());
  try {
    await page.goto(base + '/');
    const card = page.locator('.shot.nsheads');
    await card.scrollIntoViewIfNeeded();
    await page.waitForFunction(() => performance.getEntriesByType('resource').some(r => r.name.includes('card-nsheads-think.json')));
    assert.equal(requests, 1);
    // Move completely away, then return: the failed data request must be retryable.
    await page.evaluate(() => {
      const spacer = document.createElement('div'); spacer.id = 'qa-spacer'; spacer.style.height = '2000px'; document.body.append(spacer);
      window.scrollTo(0, document.body.scrollHeight);
    });
    await page.waitForTimeout(150);
    await card.scrollIntoViewIfNeeded();
    await card.hover();
    await page.waitForFunction(() => cardTimers.size > 0);
    assert.equal(requests, 2);
    await page.waitForFunction(() => document.querySelector('.think.flow'));
    const bars = () => page.locator('.think .tk rect').evaluateAll(es => es.map(e => e.style.transform));
    const before = await bars();
    await page.waitForTimeout(2300);
    assert.notDeepEqual(await bars(), before, 'visible card did not advance');
    await page.evaluate(() => { cardHidden = true; document.dispatchEvent(new Event('visibilitychange')); });
    assert.equal(await page.evaluate(() => cardTimers.size), 0, 'hidden card kept a timer');
    await page.evaluate(() => { cardHidden = false; document.dispatchEvent(new Event('visibilitychange')); });
    await page.waitForFunction(() => cardTimers.size > 0);
    await page.mouse.move(0, 0);
    await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
    await page.waitForTimeout(150);
    assert.equal(await page.evaluate(() => cardTimers.size), 0, 'offscreen card kept a timer');
    await card.scrollIntoViewIfNeeded();
    await card.hover();
    await page.waitForFunction(() => cardTimers.size > 0);
    assert.deepEqual(errors, []);
    return {failedLoadRetry: true, visibleMotion: true, hiddenIdle: true, offscreenIdle: true, resumes: true};
  } finally { await page.close(); }
}
module.exports = {checkCardLifecycle};

async function checkMeshCardLifecycle(browser, base) {
  const page = await browser.newPage({viewport: {width: 1100, height: 700}, serviceWorkers: 'block', reducedMotion: 'reduce'});
  const errors = [];
  page.on('pageerror', e => errors.push(String(e)));
  await page.addInitScript(() => {
    window.meshWorkers = [];
    window.Worker = class {
      constructor() { this.sent = []; meshWorkers.push(this); setTimeout(() => this.onmessage?.({data: {type: 'ready'}}), 0); }
      postMessage(m) { this.sent.push(m); }
      terminate() { this.terminated = true; }
    };
    window.meshAnswer = (w, id) => w.onmessage({data: {type: 'fit', id, ms: 1, baseline: -1, bounds: [0,1,0,1], tri: [0,0,0,1,0,0,0,1,0], verts: []}});
  });
  await page.route('https://**', route => route.abort());
  try {
    await page.goto(base + '/');
    const card = page.locator('#card-mesh');
    await card.scrollIntoViewIfNeeded();
    await card.hover();
    await page.waitForFunction(() => meshWorkers.length > 0);
    await page.evaluate(() => meshWorkers.at(-1).onmessage({data: {type: 'error', message: 'simulated startup failure'}}));
    assert((await card.locator('.status').innerText()).includes('simulated startup failure'));
    assert(await page.evaluate(() => meshWorkers.at(-1).terminated));
    const dab = () => card.click({position: {x: 80, y: 80}});
    await dab();
    await page.waitForFunction(() => meshWorkers.at(-1).sent.length === 1);
    await dab(); await dab();
    assert.equal(await page.evaluate(() => meshWorkers.at(-1).sent.length), 1, 'queued every superseded drawing');
    await page.evaluate(() => { const w = meshWorkers.at(-1); meshAnswer(w, w.sent[0].id); });
    assert.deepEqual(await page.evaluate(() => meshWorkers.at(-1).sent.map(m => m.id)), [1,3]);
    await page.evaluate(() => { const w = meshWorkers.at(-1); meshAnswer(w, w.sent[1].id); });
    await page.waitForFunction(() => document.querySelector('#card-mesh .status').textContent.includes('point'));
    await card.locator('.clear').click();
    assert(await page.evaluate(() => meshWorkers.at(-1).terminated));
    await dab();
    await page.waitForFunction(() => meshWorkers.at(-1).sent.length === 1);
    const status = await card.locator('.status').innerText();
    await page.evaluate(() => { const old = meshWorkers.at(-2); meshAnswer(old, old.sent.at(-1).id); });
    assert.equal(await card.locator('.status').innerText(), status, 'old worker changed the new request');
    await page.evaluate(() => meshWorkers.at(-1).onerror({message: 'runtime failure'}));
    assert(await page.evaluate(() => meshWorkers.at(-1).terminated));
    await dab();
    await page.waitForFunction(() => meshWorkers.at(-1).sent.length === 1);
    await page.evaluate(() => { const w = meshWorkers.at(-1); meshAnswer(w, w.sent[0].id); });
    await page.evaluate(() => {
      const spacer = document.createElement('div'); spacer.style.height = '2000px'; document.body.append(spacer);
      window.scrollTo(0, document.body.scrollHeight);
    });
    await page.waitForFunction(() => meshWorkers.at(-1).terminated);
    assert.deepEqual(errors, []);
  } finally { await page.close(); }

  const real = await browser.newPage({viewport: {width: 1100, height: 700}, serviceWorkers: 'block', reducedMotion: 'reduce'});
  const realErrors = [];
  real.on('pageerror', e => realErrors.push(String(e)));
  await real.route('https://**', route => route.abort());
  try {
    await real.goto(base + '/');
    const card = real.locator('#card-mesh');
    await card.scrollIntoViewIfNeeded();
    await card.click({position: {x: 80, y: 80}});
    await real.waitForFunction(() => document.querySelector('#card-mesh .status').textContent.includes('point'), null, {timeout: 120000});
    const status = await card.locator('.status').innerText();
    assert(real.workers().length > 0);
    await card.locator('.clear').click();
    await real.waitForTimeout(150);
    assert.equal(real.workers().length, 0, 'cleared card kept its WASM worker');
    assert.deepEqual(realErrors, []);
    return {startupRetry: true, latestDrawingOnly: true, clearReleasesWorker: true, staleWorkerIgnored: true, runtimeRetry: true, offscreenReleasesWorker: true, realFit: status};
  } finally { await real.close(); }
}
module.exports.checkMeshCardLifecycle = checkMeshCardLifecycle;
