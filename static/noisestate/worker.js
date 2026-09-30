// Web Worker: runs the C++ port of noisestate compiled to WebAssembly, so a solve never blocks the page.  When the page
// is cross-origin isolated (coi-serviceworker.js sets the headers GitHub Pages cannot), the threaded build runs the
// agents' best responses in parallel; otherwise the single-threaded one.  Results are identical either way.
// Messages in: {type: "solve", id, model, request}; out: "ready", "progress", "result", "fatal".
const MT = self.crossOriginIsolated === true && typeof SharedArrayBuffer !== "undefined";
const THREADS = MT ? Math.max(1, Math.min(4, navigator.hardwareConcurrency || 2)) : 1;
// One revision identifies the JS, WASM and pthread entry point together.
const SOLVER_REV = "03ed584";
const asset = (name) => new URL(name + "?v=" + SOLVER_REV, self.location.href).href;
const entry = asset(MT ? "noisestate-mt.js" : "noisestate.js");
importScripts(entry);

let solve = null;
// the threaded build's pool workers load the main script by URL (a module inside a worker cannot find its own)
const ready = NoiseState({ locateFile: (name) => asset(name), ...(MT ? { mainScriptUrlOrBlob: entry } : {}) }).then((mod) => {
  const ns_solve = mod.cwrap("ns_solve", "number", ["string", "string"]);
  const ns_free = mod.cwrap("ns_free", null, ["number"]);
  const ns_version = mod.cwrap("ns_version", "number", []);
  solve = (model, request) => {
    const p = ns_solve(model, request);
    try { return mod.UTF8ToString(p); }
    finally { ns_free(p); }
  };
  postMessage({ type: "ready", version: mod.UTF8ToString(ns_version()), threads: THREADS });
}).catch((err) => postMessage({ type: "fatal", message: String(err && err.message ? err.message : err) }));

let current = null;
self.nsProgress = (evaluation, residual, newton) => {
  if (current !== null) postMessage({ type: "progress", id: current, evaluation, residual, phase: newton ? "newton" : "anderson" });
};

onmessage = async (ev) => {
  const msg = ev.data;
  if (msg.type !== "solve") return;
  await ready;
  current = msg.id;
  const t0 = performance.now();
  let out;
  try {
    out = solve(JSON.stringify(msg.model), JSON.stringify({ ...(msg.request || {}), threads: THREADS }));
  } catch (err) {
    out = JSON.stringify({ ok: false, error: "the solver stopped: " + String(err && err.message ? err.message : err) });
  }
  current = null;
  postMessage({ type: "result", id: msg.id, result: out, wall: (performance.now() - t0) / 1000 });
};
