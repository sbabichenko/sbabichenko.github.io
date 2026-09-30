/* A cancelled press must not keep collecting samples or flipping coins. */
const assert = require('node:assert/strict');

async function checkHoldLifecycle(browser, base) {
  const page = await browser.newPage({viewport: {width: 1200, height: 800}, serviceWorkers: 'block'});
  const errors = [];
  page.on('pageerror', e => errors.push(String(e)));
  await page.addInitScript(() => {
    window.holdHidden = false;
    Object.defineProperty(document, 'hidden', {get: () => holdHidden});
    const later = window.setTimeout, cancel = window.clearTimeout;
    window.heroHoldTimers = new Set(); window.heroHoldTicks = 0;
    window.setTimeout = (fn, delay, ...args) => {
      const tracked = (delay === 350 || delay === 250) && new Error().stack.includes('/js/heroink.js');
      let id = later(() => {
        heroHoldTimers.delete(id);
        if (tracked) heroHoldTicks++;
        fn(...args);
      }, delay);
      if (tracked) heroHoldTimers.add(id);
      return id;
    };
    window.clearTimeout = id => { heroHoldTimers.delete(id); cancel(id); };
    window.holdFits = 0;
    window.Worker = class {
      constructor() { later(() => this.onmessage?.({data: {type: 'ready'}}), 10); }
      postMessage(m) {
        holdFits++;
        later(() => this.onmessage?.({data: {type: 'error', id: m.id, message: 'fixture'}}), 10);
      }
      terminate() {}
    };
  });
  await page.route('https://**', route => route.abort());
  try {
    await page.goto(base + '/');
    await page.waitForFunction(() => document.querySelector('#heroink')?.width > 0 && window.heroinkPhase);
    const pressHero = () => page.evaluate(() => {
      const r = document.querySelector('#heroink').getBoundingClientRect();
      document.body.dispatchEvent(new PointerEvent('pointerdown', {
        bubbles: true, pointerType: 'mouse', button: 0, pointerId: 1,
        clientX: r.left + r.width / 2, clientY: r.top + r.height / 2
      }));
    });
    await pressHero();
    assert.equal(await page.evaluate(() => heroHoldTimers.size), 1);
    await page.evaluate(() => document.dispatchEvent(new PointerEvent('pointercancel')));
    assert.equal(await page.evaluate(() => heroHoldTimers.size), 0);
    const cancelled = await page.evaluate(() => heroHoldTicks);
    await page.waitForTimeout(400);
    assert.equal(await page.evaluate(() => heroHoldTicks), cancelled);
    for (const cause of ['blur', 'hidden', 'pointerup']) {
      await page.evaluate(() => { holdHidden = false; document.dispatchEvent(new Event('visibilitychange')); });
      const before = await page.evaluate(() => heroHoldTicks);
      await pressHero();
      await page.waitForFunction(n => heroHoldTicks >= n + 2, before);
      await page.evaluate(cause => {
        if (cause === 'blur') window.dispatchEvent(new Event('blur'));
        else if (cause === 'hidden') { holdHidden = true; document.dispatchEvent(new Event('visibilitychange')); }
        else document.dispatchEvent(new PointerEvent('pointerup'));
      }, cause);
      assert.equal(await page.evaluate(() => heroHoldTimers.size), 0, cause);
      const stopped = await page.evaluate(() => heroHoldTicks);
      await page.waitForTimeout(300);
      assert.equal(await page.evaluate(() => heroHoldTicks), stopped, cause);
    }

    await page.goto(base + '/decisionmesh/');
    await page.waitForFunction(() => holdFits > 0);
    const seed = () => page.locator('#seed').inputValue().then(Number);
    const press = () => page.locator('#newdata').dispatchEvent('pointerdown', {button: 0, pointerType: 'mouse'});
    for (const cause of ['blur', 'hidden', 'pointercancel', 'pointerleave']) {
      await page.evaluate(() => { holdHidden = false; document.dispatchEvent(new Event('visibilitychange')); });
      const before = await seed();
      await press();
      await page.waitForFunction(n => +document.querySelector('#seed').value >= n + 2, before);
      await page.evaluate(cause => {
        if (cause === 'blur') window.dispatchEvent(new Event('blur'));
        else if (cause === 'hidden') { holdHidden = true; document.dispatchEvent(new Event('visibilitychange')); }
        else document.querySelector('#newdata').dispatchEvent(new PointerEvent(cause));
      }, cause);
      const stopped = await seed();
      await page.waitForTimeout(400);
      assert.equal(await seed(), stopped, cause + ' kept flipping');
      await page.evaluate(() => { holdHidden = false; document.dispatchEvent(new Event('visibilitychange')); });
      await page.locator('#newdata').dispatchEvent('click');
      assert.equal(await seed(), stopped + 1, cause + ' swallowed the next click');
    }
    const before = await seed();
    await press();
    await page.waitForFunction(n => +document.querySelector('#seed').value > n, before);
    await page.locator('#newdata').dispatchEvent('pointerup');
    const released = await seed();
    await page.locator('#newdata').dispatchEvent('click');
    assert.equal(await seed(), released, 'release click flipped again');
    assert.deepEqual(errors, []);
    return {pendingCancelled: true, activeCancelled: true, hiddenIdle: true, nextClickWorks: true, normalRelease: true};
  } finally { await page.close(); }
}
module.exports = {checkHoldLifecycle};
