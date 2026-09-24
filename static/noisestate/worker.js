// Web Worker: runs the C++ port of noisestate compiled to WebAssembly, so a solve never blocks the page.  When the page
// is cross-origin isolated (coi-serviceworker.js sets the headers GitHub Pages cannot), the threaded build runs the
// agents' best responses in parallel; otherwise the single-threaded one.  Results are identical either way.
// Messages in: {type: "solve", id, model, request}; out: "ready", "progress", "result", "fatal".
const MT = self.crossOriginIsolated === true && typeof SharedArrayBuffer !== "undefined";
const THREADS = MT ? Math.max(1, Math.min(4, navigator.hardwareConcurrency || 2)) : 1;
importScripts(MT ? "noisestate-mt.js" : "noisestate.js");

let solve = null;
// the threaded build's pool workers load the main script by URL (a module inside a worker cannot find its own)
const ready = NoiseState(MT ? { mainScriptUrlOrBlob: new URL("noisestate-mt.js", self.location.href).href } : {}).then((mod) => {
  const ns_solve = mod.cwrap("ns_solve", "number", ["string", "string"]);
  const ns_free = mod.cwrap("ns_free", null, ["number"]);
  const ns_version = mod.cwrap("ns_version", "number", []);
  solve = (model, request) => {
    const p = ns_solve(model, request);
    const out = mod.UTF8ToString(p);
    ns_free(p);
    return out;
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
