/* Returning to the tab must not restart an offscreen spike animation. */
const assert = require('node:assert/strict');
async function checkSpikeLifecycle(browser, base) {
  const page = await browser.newPage({viewport: {width: 1100, height: 650}, serviceWorkers: 'block'});
  const errors = [];
  page.on('pageerror', e => errors.push(String(e)));
  await page.addInitScript(() => {
    const request = window.requestAnimationFrame, cancel = window.cancelAnimationFrame;
    window.spikeFrames = new Set();
    window.spikeFrameCount = 0;
    window.requestAnimationFrame = fn => {
      const spike = new Error().stack.includes('/dissertation/spike.js');
      let id = request(ts => { spikeFrames.delete(id); if (spike) spikeFrameCount++; fn(ts); });
      if (spike) spikeFrames.add(id);
      return id;
    };
    window.cancelAnimationFrame = id => { spikeFrames.delete(id); cancel(id); };
    window.spikeHidden = false;
    Object.defineProperty(document, 'hidden', {get: () => spikeHidden});
  });
  await page.route('https://**', route => route.abort());
  try {
    await page.goto(base + '/dissertation/spike/');
    await page.waitForSelector('#strips .strip svg');
    await page.locator('#mode-many').click();
    await page.locator('#strips .strip').first().scrollIntoViewIfNeeded();
    await page.waitForFunction(() => spikeFrameCount > 5 && spikeFrames.size > 0);
    await page.evaluate(() => { spikeHidden = true; document.dispatchEvent(new Event('visibilitychange')); });
    assert.equal(await page.evaluate(() => spikeFrames.size), 0);
    const paused = await page.evaluate(() => spikeFrameCount);
    await page.waitForTimeout(150);
    assert.equal(await page.evaluate(() => spikeFrameCount), paused, 'hidden animation advanced');
    await page.evaluate(() => { spikeHidden = false; document.dispatchEvent(new Event('visibilitychange')); });
    await page.waitForFunction(n => spikeFrameCount > n, paused);
    await page.evaluate(() => {
      const spacer = document.createElement('div'); spacer.style.height = '2000px'; document.body.append(spacer);
      window.scrollTo(0, document.body.scrollHeight);
    });
    await page.waitForTimeout(150);
    assert.equal(await page.evaluate(() => spikeFrames.size), 0);
    await page.evaluate(() => {
      spikeHidden = true; document.dispatchEvent(new Event('visibilitychange'));
      spikeHidden = false; document.dispatchEvent(new Event('visibilitychange'));
    });
    assert.equal(await page.evaluate(() => spikeFrames.size), 0, 'returning to tab restarted offscreen animation');
    await page.locator('#strips .strip').first().scrollIntoViewIfNeeded();
    await page.waitForFunction(() => spikeFrames.size > 0);
    assert.deepEqual(errors, []);
    return {visibleMotion: true, hiddenIdle: true, offscreenIdleAfterTabReturn: true, resumes: true};
  } finally { await page.close(); }
}
module.exports = {checkSpikeLifecycle};
