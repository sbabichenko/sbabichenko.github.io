// Runs the triangular decision-mesh estimator (triangular-decision-mesh/core, compiled to WebAssembly as
// trimesh.js) off the page's thread. A message carries a design CSV and a seed; the engine reads the CSV
// from its virtual filesystem, fits with the held-out split on, and the dumps it writes come back parsed.
"use strict";
importScripts("trimesh.js");

let engine = null, log = [];
const ready = DecisionMeshEngine({ print: (s) => log.push(s), printErr: (s) => log.push(s) })
  .then((m) => { engine = m; m.FS.mkdir("/w"); postMessage({ type: "ready" }); })
  .catch((e) => postMessage({ type: "error", message: "engine failed to load: " + e }));

function csv(text) {
  const lines = text.trim().split("\n"), head = lines[0].split(",");
  const cols = head.map(() => []);
  for (let i = 1; i < lines.length; ++i) {
    const f = lines[i].split(",");
    for (let j = 0; j < head.length; ++j) cols[j].push(f[j]);
  }
  const out = {};
  head.forEach((h, j) => { out[h] = cols[j]; });
  return out;
}
const num = (a) => Float64Array.from(a, Number);

onmessage = async (ev) => {
  await ready;
  if (!engine) return;
  const { id, design, seed } = ev.data;
  log = [];
  const FS = engine.FS;
  for (const f of FS.readdir("/w")) if (f.startsWith("run")) FS.unlink("/w/" + f);
  FS.writeFile("/w/design.csv", design);
  const t0 = performance.now();
  let rc;
  try {
    rc = engine.ccall("dm_run", "number", ["number", "string"],
      [seed, "DMESH_DATA=/w/design.csv\nDMESH_DUMP=/w/run\nDMESH_SPLIT=1"]);
  } catch (e) {
    postMessage({ id, type: "error", message: String(e), log });
    return;
  }
  const ms = performance.now() - t0;
  if (rc !== 0) { postMessage({ id, type: "error", message: "engine exited with " + rc, log }); return; }
  const read = (f) => FS.readFile("/w/" + f, { encoding: "utf8" });

  const m = csv(read("run_mesh.csv"));
  const tri = new Float64Array(m.x0.length * 9);
  const keys = ["x0", "y0", "h0", "x1", "y1", "h1", "x2", "y2", "h2"];
  for (let i = 0; i < m.x0.length; ++i) for (let k = 0; k < 9; ++k) tri[9 * i + k] = +m[keys[k]][i];

  const v = csv(read("run_hier_vertices.csv"));
  const verts = [];
  for (let i = 0; i < v.id.length; ++i) {
    if (v.active[i] !== "1") continue;
    verts.push({ x: +v.x[i], y: +v.y[i], h: +v.height[i], admitted: v.gate_admitted[i] === "1", round: +v.admit_round[i] });
  }

  const rounds = JSON.parse(read("run_centered_rounds.json")).map((r) => ({
    round: r.round, candidates: r.candidates, admitted: r.admitted, faces: r.faces_before,
    poolVariance: r.pool_variance, nullMean: r.null_mean, nullSd: r.null_sd, coefficients: r.coefficients,
  }));
  // the gate's own account of each round: [calibration] round R ... emp-null mean/sd/pi0 a/b/c method M ...
  for (const line of log) {
    const g = /^\[calibration\] round (\d+) .*emp-null mean\/sd\/pi0 ([^/]+)\/([^/]+)\/(\S+) method (\S+) admitted-prefix (\d+) lindsey-status (\S+)/.exec(line);
    if (!g) continue;
    const r = rounds.find((q) => q.round === +g[1]);
    if (r) Object.assign(r, { pi0: +g[4], method: g[5], status: g[7] });
  }
  const model = JSON.parse(read("run_model.json"));
  const fin = JSON.parse(read("run_final_fit.json"));
  const held = log.map((s) => /^HELDOUT: mean deviance\/pool ([\d.]+) \| X2\/info ([\d.]+) \| (\d+) pools/.exec(s)).find(Boolean);

  postMessage({
    id, type: "fit", ms, tri, verts, rounds,
    baseline: model.baseline_logit, bounds: [model.wala_min, model.wala_max, model.wac_min, model.wac_max],
    poolVariance: fin.pool_variance, coefficients: fin.coefficients,
    heldout: held ? { deviance: +held[1], x2: +held[2], pools: +held[3] } : null,
    log: log.slice(0, 400),
  }, [tri.buffer]);
};
