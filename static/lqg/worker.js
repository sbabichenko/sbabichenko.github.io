// Web Worker: runs the C++ port of noisestate compiled to WebAssembly (noisestate.js), so a solve never
// blocks the page.  Messages in: {type: "solve", id, model, request}; out: "ready", "progress", "result".
importScripts("noisestate.js");

let solve = null;
const ready = NoiseState().then((mod) => {
  const ns_solve = mod.cwrap("ns_solve", "number", ["string", "string"]);
  const ns_free = mod.cwrap("ns_free", null, ["number"]);
  const ns_version = mod.cwrap("ns_version", "number", []);
  solve = (model, request) => {
    const p = ns_solve(model, request);
    const out = mod.UTF8ToString(p);
    ns_free(p);
    return out;
  };
  postMessage({ type: "ready", version: mod.UTF8ToString(ns_version()) });
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
    out = solve(JSON.stringify(msg.model), JSON.stringify(msg.request || {}));
  } catch (err) {
    out = JSON.stringify({ ok: false, error: "the solver stopped: " + String(err && err.message ? err.message : err) });
  }
  current = null;
  postMessage({ type: "result", id: msg.id, result: out, wall: (performance.now() - t0) / 1000 });
};
