// A thrown native runtime error invalidates that worker; model errors do not.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname, '../../static/noisestate/worker.js'), 'utf8');

async function worker(sequence, {lazy = false, decodeError = false} = {}) {
  const messages = [], calls = {loads: 0, frees: 0, closed: 0};
  let current = '';
  const context = {
    URL, Promise, performance: {now: () => 0}, navigator: {hardwareConcurrency: 2},
    location: {href: 'https://local.invalid/noisestate/worker.js' + (lazy ? '?lazy=1' : '')},
    crossOriginIsolated: false,
    postMessage: message => messages.push(message), close: () => calls.closed++,
    importScripts: () => {},
    NoiseState: async () => {
      calls.loads++;
      return {
        cwrap(name) {
          if (name === 'ns_version') return () => 1;
          if (name === 'ns_free') return () => calls.frees++;
          assert.equal(name, 'ns_solve');
          return () => {
            const next = sequence.shift();
            if (next instanceof Error) throw next;
            current = next;
            return 2;
          };
        },
        UTF8ToString(pointer) {
          if (pointer === 1) return 'test solver';
          if (decodeError) throw new Error('response conversion failed');
          return current;
        },
      };
    },
    Ch6Markov: {payload() {
      const next = sequence.shift();
      if (next instanceof Error) throw next;
      return next;
    }},
  };
  context.self = context;
  vm.createContext(context);
  vm.runInContext(source, context);
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(messages[0].type, 'ready');
  return {messages, calls, solve: (id = 1) => context.onmessage({data: {
    type: 'solve', id, model: {}, request: lazy ? {method: 'ch6-markov'} : {},
  }})};
}

(async () => {
  const trapped = await worker([new Error('unreachable')]);
  await trapped.solve();
  assert.equal(trapped.messages.at(-1).type, 'fatal', 'native exceptions must discard the worker');
  assert.equal(trapped.messages.at(-1).phase, 'solve');
  assert.equal(trapped.calls.closed, 1);
  assert.equal(trapped.messages.filter(m => m.type === 'result').length, 0);

  const malformed = await worker(['']);
  await malformed.solve();
  assert.equal(malformed.messages.at(-1).type, 'fatal', 'an empty native response cannot retain the runtime');
  assert.equal(malformed.calls.frees, 1);

  const decoding = await worker(['unused'], {decodeError: true});
  await decoding.solve();
  assert.equal(decoding.messages.at(-1).type, 'fatal');
  assert.equal(decoding.calls.frees, 1, 'allocated response must be freed even if conversion throws');

  const normal = await worker([JSON.stringify({ok: false, error: 'invalid model'}), JSON.stringify({ok: true})]);
  await normal.solve(); await normal.solve(2);
  assert.equal(normal.calls.loads, 1, 'normal model errors do not reload a healthy runtime');
  assert.equal(normal.calls.closed, 0);
  assert.equal(normal.calls.frees, 2);
  assert.equal(JSON.parse(normal.messages.at(-1).result).ok, true);

  const reference = await worker([new Error('invalid reference parameters'), {ok: true}], {lazy: true});
  await reference.solve(); await reference.solve(2);
  assert.equal(reference.calls.loads, 0, 'reference validation never loads WASM');
  assert.equal(reference.calls.closed, 0);
  assert.equal(JSON.parse(reference.messages.at(-1).result).ok, true);
  console.log('Worker runtime failure, empty response, cleanup, and healthy retry checks passed.');
})().catch(error => { console.error(error); process.exitCode = 1; });
