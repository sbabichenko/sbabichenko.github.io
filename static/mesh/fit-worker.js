// Runs the decision-mesh estimators off the page's thread: triangular-decision-mesh/core (trimesh.js, right
// triangles) and rectangular-decision-mesh/core (rectmesh.js, rectangles), each compiled to WebAssembly
// unchanged. A message names the engine and carries a design CSV and a seed; the engine reads the CSV from
// its virtual filesystem, fits with the held-out split on, and the dumps it writes come back parsed.
"use strict";
let log = [];
const engines = {};
// both builds export the same factory name, so each is loaded in turn and its factory kept
function load(name) {
  if (engines[name]) return engines[name];
  self.DecisionMeshEngine = undefined;
  importScripts(name === "rect" ? "rectmesh.js" : "trimesh.js");
  const factory = self.DecisionMeshEngine;
  engines[name] = factory({ print: (s) => log.push(s), printErr: (s) => log.push(s) })
    .then((m) => { m.FS.mkdir("/w"); return m; });
  return engines[name];
}
load("tri").then(() => postMessage({ type: "ready" }))
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
  const { id, design, seed } = ev.data, kind = ev.data.engine === "rect" ? "rect" : "tri";
  let engine;
  try { engine = await load(kind); } catch (e) { postMessage({ id, type: "error", message: "engine failed to load: " + e }); return; }
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

  // triangles: x0 y0 h0 x1 y1 h1 x2 y2 h2; rectangles: x0 y0 x1 y1 h00 h10 h01 h11 (corner heights)
  const m = csv(read("run_mesh.csv"));
  const keys = kind === "rect" ? ["x0", "y0", "x1", "y1", "h00", "h10", "h01", "h11"] : ["x0", "y0", "h0", "x1", "y1", "h1", "x2", "y2", "h2"];
  const K = keys.length, tri = new Float64Array(m.x0.length * K);
  for (let i = 0; i < m.x0.length; ++i) for (let k = 0; k < K; ++k) tri[K * i + k] = +m[keys[k]][i];

  const v = csv(read("run_hier_vertices.csv"));
  const verts = [];
  for (let i = 0; i < v.id.length; ++i) {
    if (v.active && v.active[i] !== "1") continue;
    verts.push({ x: +v.x[i], y: +v.y[i], h: +v.height[i], admitted: v.gate_admitted[i] === "1", round: +v.admit_round[i] });
  }

  const rounds = JSON.parse(read("run_centered_rounds.json")).map((r) => ({
    round: r.round, candidates: r.candidates, admitted: r.admitted, faces: r.faces_before ?? r.cells_before,
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
    id, type: "fit", engine: kind, ms, tri, stride: K, verts, rounds,
    baseline: model.baseline_logit, bounds: [model.wala_min, model.wala_max, model.wac_min, model.wac_max],
    poolVariance: fin.pool_variance, coefficients: fin.coefficients,
    heldout: held ? { deviance: +held[1], x2: +held[2], pools: +held[3] } : null,
    log: log.slice(0, 400),
  }, [tri.buffer]);
};
