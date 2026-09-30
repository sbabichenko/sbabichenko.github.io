// Background solver worker. Chapter 6's reference can load on its own; general
// models load the native solver compiled to WebAssembly, with threads when available.
const MT = self.crossOriginIsolated === true && typeof SharedArrayBuffer !== "undefined";
const THREADS = MT ? Math.max(1, Math.min(4, navigator.hardwareConcurrency || 2)) : 1;
const SOLVER_REV = "65ca18a";
const asset = (name) => new URL(name + "?v=" + SOLVER_REV, self.location.href).href;
const entry = asset(MT ? "noisestate-mt.js" : "noisestate.js");
let runtime = null;
function loadSolver() {
  if (!runtime) runtime = Promise.resolve().then(() => {
    importScripts(entry);
    return NoiseState({ locateFile: (name) => asset(name), ...(MT ? { mainScriptUrlOrBlob: entry } : {}) });
  }).then((mod) => {
    const ns_solve = mod.cwrap("ns_solve", "number", ["string", "string"]);
    const ns_free = mod.cwrap("ns_free", null, ["number"]);
    const ns_version = mod.cwrap("ns_version", "number", []);
    return { version: mod.UTF8ToString(ns_version()), solve(model, request) {
      const p = ns_solve(model, request);
      try { return mod.UTF8ToString(p); } finally { ns_free(p); }
    } };
  }).catch(err => { runtime = null; throw err; });
  return runtime;
}
const lazy = new URL(self.location.href).searchParams.get("lazy") === "1";
const ready = lazy ? Promise.resolve({ version: "Chapter 6 finite-state reference" }) : loadSolver();
ready.then(info => postMessage({ type: "ready", version: info.version, threads: THREADS }))
  .catch(err => postMessage({ type: "fatal", message: String(err && err.message ? err.message : err) }));

let current = null;
self.nsProgress = (evaluation, residual, newton) => {
  if (current !== null) postMessage({ type: "progress", id: current, evaluation, residual, phase: newton ? "newton" : "anderson" });
};
onmessage = async ({ data: msg }) => {
  if (msg.type !== "solve") return;
  const t0 = performance.now();
  let out;
  try {
    await ready;
    const request = { ...(msg.request || {}), threads: THREADS };
    if (request.method === "ch6-markov") {
      if (!self.Ch6Markov) importScripts(new URL("ch6-markov.js?v=20260930-2", self.location.href).href);
      out = JSON.stringify(Ch6Markov.payload(msg.model, request));
    } else {
      const solver = await loadSolver();
      current = msg.id;
      out = solver.solve(JSON.stringify(msg.model), JSON.stringify(request));
    }
  } catch (err) {
    out = JSON.stringify({ ok: false, error: "the solver stopped: " + String(err && err.message ? err.message : err) });
  } finally { current = null; }
  postMessage({ type: "result", id: msg.id, result: out, wall: (performance.now() - t0) / 1000 });
};
