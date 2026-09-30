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
