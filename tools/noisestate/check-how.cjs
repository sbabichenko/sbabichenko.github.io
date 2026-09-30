/* The explanatory page should sleep while waiting for data and release its solver. */
const assert = require('node:assert/strict');
async function checkHowLifecycle(browser, base, story = 'how') {
  const route = story === 'wedge' ? '/dissertation/wedge/' : '/noisestate/how/';
  const count = story === 'wedge' ? 9 : 11;
  const waiting = await browser.newPage({serviceWorkers: 'block', reducedMotion: 'reduce'});
  await waiting.addInitScript(() => {
    window.coi = {shouldRegister: () => false};
    const frame = window.requestAnimationFrame;
    window.howFrames = 0;
    window.requestAnimationFrame = fn => { window.howFrames++; return frame.call(window, fn); };
    window.Worker = class {
      constructor() { window.howWorker = this; setTimeout(() => this.onmessage({data: {type: 'ready'}}), 0); }
      postMessage(m) { this.lastRequest = m; }
      terminate() { this.terminated = true; }
    };
  });
  await waiting.route('https://**', route => route.abort());
  try {
    await waiting.goto(base + route);
    await waiting.locator('.step').first().scrollIntoViewIfNeeded();
    await waiting.waitForFunction(() => howWorker.lastRequest);
    await waiting.waitForTimeout(400);
    const before = await waiting.evaluate(() => howFrames);
    await waiting.waitForTimeout(500);
    const after = await waiting.evaluate(() => howFrames);
    assert(after - before <= 2, `waiting drawing kept polling: ${after-before} frames`);
    await waiting.evaluate(() => howWorker.onmessage({data: {type: 'fatal', message: 'simulated load failure'}}));
    assert.equal(await waiting.evaluate(() => howWorker.terminated), true);
    assert((await waiting.locator('#solvestate').innerText()).includes('simulated load failure'));
  } finally { await waiting.close(); }

  const page = await browser.newPage({serviceWorkers: 'block', reducedMotion: 'reduce'});
  const errors = [];
  page.on('pageerror', e => errors.push(String(e)));
  await page.addInitScript(() => {
    window.coi = {shouldRegister: () => false};
    const NativeWorker = window.Worker;
    window.howTerminations = 0;
    window.Worker = class extends NativeWorker {
      terminate() { window.howTerminations++; return super.terminate(); }
    };
  });
  await page.route('https://**', route => route.abort());
  try {
    await page.goto(base + route);
    await page.waitForFunction(() => howTerminations === 1, null, {timeout: 120000});
    const status = await page.locator('#solvestate').innerText();
    assert(status.startsWith(`Ran ${count} solves`), status);
    assert(!status.includes('failed'), status);
    await page.locator(story === 'wedge' ? '.step[data-scene=split]' : '.step').first().scrollIntoViewIfNeeded();
    await page.waitForTimeout(150);
    assert.equal(page.workers().length, 0, 'the completed solver worker remained alive');
    const drawn = story === 'wedge' ? '#stage [data-scene=split] path[d]' : '#stage circle';
    assert(await page.locator(drawn).count() > 0, 'solved drawing did not appear');
    assert.deepEqual(errors, []);
    return {idleWaiting: true, failureReleasesWorker: true, completedWorkerReleased: true, status};
  } finally { await page.close(); }
}
module.exports = {checkHowLifecycle};
