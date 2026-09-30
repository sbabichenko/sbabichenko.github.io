/* Worker failure paths use fixtures; a real fit checks the unchanged protocol. */
const assert = require('node:assert/strict');

async function checkMeshWorkerRecovery(browser, base) {
  const page = await browser.newPage({serviceWorkers: 'block'});
  const errors = [];
  page.on('pageerror', e => errors.push(String(e)));
  await page.route('https://**', r => r.abort());
  await page.addInitScript(() => {
    window.meshWorkers = []; window.failConstruction = true;
    window.Worker = class {
      constructor() {
        if (failConstruction) { failConstruction = false; throw Error('construction fixture'); }
        this.jobs = []; this.dead = false; meshWorkers.push(this);
      }
      postMessage(m) { this.jobs.push(m); }
      terminate() { this.dead = true; }
      emit(data) { this.onmessage?.({data}); }
      fail() { this.onerror?.({message: 'runtime fixture', preventDefault() {}}); }
    };
  });
  try {
    await page.goto(base + '/decisionmesh/#sites=3000&flips=5');
    await page.waitForFunction(() => document.querySelector('#chip').textContent === 'Error');
    await page.locator('#newdata').click();
    assert.equal(await page.evaluate(() => meshWorkers.length), 1);
    await page.evaluate(() => meshWorkers[0].emit({type: 'error', fatal: true, message: 'startup fixture'}));
    assert(await page.evaluate(() => meshWorkers[0].dead));
    await page.locator('#newdata').click();
    await page.evaluate(() => meshWorkers[1].emit({type: 'ready'}));
    assert.equal(await page.evaluate(() => meshWorkers[1].jobs.length), 1);
    await page.evaluate(() => meshWorkers[1].emit({id: -1, type: 'error', fatal: true, message: 'stale fixture'}));
    assert.equal(await page.locator('#chip').innerText(), 'Fitting');
    await page.evaluate(() => meshWorkers[1].fail());
    assert(await page.evaluate(() => meshWorkers[1].dead));
    await page.locator('#newdata').click();
    await page.evaluate(() => meshWorkers[2].emit({type: 'ready'}));
    const current = await page.evaluate(() => meshWorkers[2].jobs[0].id);
    await page.evaluate(() => {
      meshWorkers[1].emit({type: 'ready'});
      meshWorkers[1].fail();
    });
    assert.equal(await page.locator('#chip').innerText(), 'Fitting');
    await page.locator('#newdata').click();
    await page.locator('#newdata').click();
    assert.equal(await page.evaluate(() => meshWorkers[2].jobs.length), 1);
    await page.evaluate(id => meshWorkers[2].emit({id, type: 'error', message: 'fit fixture'}), current);
    assert.equal(await page.evaluate(() => meshWorkers[2].jobs.length), 2, 'queued changes were stranded');
    assert.equal(await page.locator('#chip').innerText(), 'Fitting');
    await page.evaluate(() => {
      const w = meshWorkers[2];
      w.emit({id: w.jobs[1].id, type: 'error', fatal: true, message: 'parse fixture'});
    });
    assert(await page.evaluate(() => meshWorkers[2].dead));
    await page.locator('#newdata').click();
    await page.evaluate(() => meshWorkers[3].emit({type: 'ready'}));
    assert.equal(await page.evaluate(() => meshWorkers[3].jobs.length), 1);
    assert.deepEqual(errors, []);
  } finally { await page.close(); }

  const actual = await browser.newPage({serviceWorkers: 'block'});
  const actualErrors = [];
  actual.on('pageerror', e => actualErrors.push(String(e)));
  await actual.route('https://**', r => r.abort());
  await actual.addInitScript(() => {
    window.meshHidden = false; window.realMeshWorkers = [];
    Object.defineProperty(document, 'hidden', {get: () => meshHidden});
    const Native = window.Worker;
    window.Worker = function(...args) {
      const worker = new Native(...args), record = {dead: false};
      realMeshWorkers.push(record);
      const terminate = worker.terminate.bind(worker);
      worker.terminate = () => { record.dead = true; terminate(); };
      return worker;
    };
  });
  try {
    await actual.goto(base + '/decisionmesh/#sites=3000&flips=5');
    await actual.waitForFunction(() => document.querySelector('#chip').textContent === 'Fitted', null, {timeout: 60000});
    const displayed = await actual.locator('#results').innerHTML();
    await actual.evaluate(() => { meshHidden = true; document.dispatchEvent(new Event('visibilitychange')); });
    assert.equal(await actual.evaluate(() => realMeshWorkers.filter(w => !w.dead).length), 0);
    assert.equal(await actual.locator('#results').innerHTML(), displayed);
    await actual.evaluate(() => { meshHidden = false; document.dispatchEvent(new Event('visibilitychange')); });
    assert.equal(await actual.evaluate(() => realMeshWorkers.filter(w => !w.dead).length), 0);
    await actual.locator('#newdata').click();
    await actual.waitForFunction(() => document.querySelector('#chip').textContent === 'Fitting');
    await actual.evaluate(() => { meshHidden = true; document.dispatchEvent(new Event('visibilitychange')); });
    assert.equal(await actual.evaluate(() => realMeshWorkers.filter(w => !w.dead).length), 1, 'active fit was cancelled');
    await actual.waitForFunction(() => document.querySelector('#chip').textContent === 'Fitted', null, {timeout: 60000});
    assert.equal(await actual.evaluate(() => realMeshWorkers.filter(w => !w.dead).length), 0);
    await actual.evaluate(() => { meshHidden = false; document.dispatchEvent(new Event('visibilitychange')); });
    await actual.locator('#engine').selectOption('rect');
    await actual.waitForFunction(() => document.querySelector('#chip').textContent === 'Fitting');
    await actual.waitForFunction(() => document.querySelector('#chip').textContent === 'Fitted', null, {timeout: 60000});
    // Exercise the actual worker's outer error handler, not only the page's
    // response to a fabricated error message.
    await actual.route('**/mesh/trimesh.js', r => r.fulfill({contentType: 'application/javascript', body:
      'self.DecisionMeshEngine=async()=>({FS:{mkdir(){},readdir(){return []},writeFile(){},readFile(){return "bad"}},ccall(){return 0}});'}));
    const parseFailure = await actual.evaluate(() => new Promise((resolve, reject) => {
      const w = new Worker('/mesh/fit-worker.js?parse-fixture');
      const timeout = setTimeout(() => { w.terminate(); reject(Error('parse failure was not reported')); }, 5000);
      w.onmessage = ({data}) => {
        if (data.type === 'ready') w.postMessage({id: 42, design: '0,0,1,1\n', seed: 7});
        else { clearTimeout(timeout); w.terminate(); resolve(data); }
      };
    }));
    assert.equal(parseFailure.type, 'error'); assert.equal(parseFailure.fatal, true); assert.equal(parseFailure.id, 42);
    assert.deepEqual(actualErrors, []);
    return {constructionRetry: true, startupRetry: true, runtimeRetry: true,
      staleIgnored: true, queuedErrorRecovery: true, fatalRetry: true, parseFailureReported: true,
      hiddenIdleRelease: true, hiddenActiveCompletes: true, realTriAndRect: true};
  } finally { await actual.close(); }
}
module.exports = {checkMeshWorkerRecovery};
