// Web Worker: loads Pyodide, installs noisestate, and solves on request.
// Solving here keeps the page responsive while Python runs.
const PYODIDE_VERSION = "0.29.5";
const NOISESTATE_VERSION = "1.0.1";
const PYODIDE_URL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;

let solveFn = null;

function status(step, detail) {
  postMessage({ type: "status", step, detail: detail || "" });
}

async function init() {
  const t0 = performance.now();
  try {
    status("runtime");
    importScripts(PYODIDE_URL + "pyodide.js");
    const pyodide = await loadPyodide({ indexURL: PYODIDE_URL });
    status("packages");
    await pyodide.loadPackage(["numpy", "scipy", "pyyaml", "micropip"]);
    status("noisestate");
    const micropip = pyodide.pyimport("micropip");
    await micropip.install(`noisestate==${NOISESTATE_VERSION}`);
    const src = await (await fetch(new URL("driver.py", self.location.href))).text();
    pyodide.FS.writeFile("/home/pyodide/driver.py", src);
    pyodide.runPython("import sys; sys.path.insert(0, '/home/pyodide'); import driver");
    solveFn = pyodide.pyimport("driver").solve;
    postMessage({ type: "ready", seconds: (performance.now() - t0) / 1000, version: NOISESTATE_VERSION });
  } catch (err) {
    postMessage({ type: "fatal", message: String(err && err.message ? err.message : err) });
  }
}

const ready = init();

onmessage = async (ev) => {
  const msg = ev.data;
  if (msg.type !== "solve") return;
  await ready;
  if (!solveFn) return;
  try {
    const out = solveFn(msg.preset, JSON.stringify(msg.params));
    postMessage({ type: "result", id: msg.id, result: out });
  } catch (err) {
    postMessage({ type: "error", id: msg.id, message: String(err && err.message ? err.message : err) });
  }
};
