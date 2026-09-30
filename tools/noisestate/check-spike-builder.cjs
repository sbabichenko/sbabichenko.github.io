/* Switching or hiding a finite drawing must cancel its old playback timer. */
const assert = require('node:assert/strict');

async function checkSpikeBuilder(browser, base) {
  const page = await browser.newPage({viewport: {width: 1200, height: 800}, serviceWorkers: 'block'});
  const errors = [];
  page.on('pageerror', e => errors.push(String(e)));
  await page.addInitScript(() => {
    window.builderHidden = false;
    Object.defineProperty(document, 'hidden', {get: () => builderHidden});
    const interval = window.setInterval, cancel = window.clearInterval;
    window.builderTimers = new Set();
    window.setInterval = (fn, delay, ...args) => {
      const tracked = new Error().stack.includes('/dissertation/spike-lemma.js');
      const id = interval(fn, tracked ? 30 : delay, ...args);
      if (tracked) builderTimers.add(id);
      return id;
    };
    window.clearInterval = id => { builderTimers.delete(id); cancel(id); };
  });
  await page.route('https://**', route => route.abort());
  try {
    await page.goto(base + '/dissertation/spike/');
    await page.waitForFunction(() => document.querySelector('#build-caption')?.textContent.length > 30);
    const play = page.locator('#build-play');
    await page.locator('#build-svg').scrollIntoViewIfNeeded();
    await page.waitForTimeout(100);
    await play.click();
    await page.waitForFunction(() => +document.querySelector('#build-count').textContent > 0);
    await page.locator('#build-frozen').click();
    assert.equal(await page.evaluate(() => builderTimers.size), 0, 'mode switch retained the old timer');
    await page.waitForTimeout(300);
    assert.equal(await page.locator('#build-count').textContent(), '0');
    await play.click();
    await page.waitForFunction(() => +document.querySelector('#build-count').textContent === 6 && builderTimers.size === 0);

    await page.locator('#build-blip').click();
    await play.click();
    await page.waitForFunction(() => builderTimers.size === 1);
    await page.evaluate(() => { builderHidden = true; document.dispatchEvent(new Event('visibilitychange')); });
    assert.equal(await page.evaluate(() => builderTimers.size), 0, 'hidden drawing kept playing');
    const stopped = await page.locator('#build-count').textContent();
    await page.waitForTimeout(100);
    assert.equal(await page.locator('#build-count').textContent(), stopped);
    await page.evaluate(() => { builderHidden = false; document.dispatchEvent(new Event('visibilitychange')); });
    await play.click();
    await page.waitForFunction(() => builderTimers.size === 1);
    await page.evaluate(() => window.scrollTo(0, 0));
    await page.waitForFunction(() => builderTimers.size === 0);

    await play.scrollIntoViewIfNeeded();
    await page.locator('#build-svg').scrollIntoViewIfNeeded();
    await page.waitForTimeout(100);
    await play.click();
    await page.locator('#build-n').evaluate(el => { el.value = '3'; el.dispatchEvent(new Event('input')); });
    assert.equal(await page.evaluate(() => builderTimers.size), 0, 'scrubbing retained playback');
    assert.equal(await page.locator('#build-count').textContent(), '3');
    assert.deepEqual(errors, []);
    return {modeSwitch: true, completion: true, hidden: true, offscreen: true, scrub: true};
  } finally { await page.close(); }
}
module.exports = {checkSpikeBuilder};
