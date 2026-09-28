// Precomputes the home page's Noise-State card: the two-player tracking game with opposite targets (b1 = 1,
// b2 = -1, r = 0.1, T = 1), solved by the explorer's own wasm solver (static/noisestate/noisestate.wasm) at
// log-spaced precisions p1 = p2 = p. Writes the mean pushes D1, D2 and the mean state X on a coarse time grid
// to static/js/card-noisestate.json, which static/js/cards.js interpolates between p values.
//   node tools/cards/make_card_curves.js
"use strict";
const fs = require("fs"), path = require("path");
const root = path.join(__dirname, "..", "..");
const NoiseState = require(path.join(root, "static/noisestate/noisestate.js"));
const MODEL = {"name":"card","params":{"p1":9,"p2":9,"r1":0.1,"r2":0.1,"b1":1,"b2":-1,"sigma":1,"T":1},"channels":["w0","w1","w2"],"states":{"X":{"drift":{"D1":1,"D2":1},"noise":{"w0":"sigma"}}},"agents":{"player1":{"controls":["D1"],"signals":{"y1":{"drift":{"X":"sqrt(p1)"},"noise":{"w1":1}}},"loss":[[1,"X","X"],["-2*b1","X"],["r1","D1","D1"]]},"player2":{"controls":["D2"],"signals":{"y2":{"drift":{"X":"sqrt(p2)"},"noise":{"w2":1}}},"loss":[[1,"X","X"],["-2*b2","X"],["r2","D2","D2"]]}},"horizon":{"kind":"finite","T":"T"},"numerics":{"nodes":12}};
const NP = 25, NT = 21;                      // 25 precisions from 0.01 to 1000, 21 times on [0, 1]
NoiseState({ wasmBinary: fs.readFileSync(path.join(root, "static/noisestate/noisestate.wasm")) }).then((mod) => {
  const solve = mod.cwrap("ns_solve", "number", ["string", "string"]), free = mod.cwrap("ns_free", null, ["number"]);
  const run = (params) => { const m = JSON.parse(JSON.stringify(MODEL)); Object.assign(m.params, params);
    const q = solve(JSON.stringify(m), "{}"), o = JSON.parse(mod.UTF8ToString(q)); free(q); return o; };
  // linear interpolation of a sampled path onto the coarse grid
  const at = (ts, ys, t) => { let i = 1; while (i < ts.length - 1 && ts[i] < t) ++i;
    const w = (t - ts[i - 1]) / (ts[i] - ts[i - 1] || 1); return ys[i - 1] + w * (ys[i] - ys[i - 1]); };
  const r3 = (v) => Math.round(v * 1000) / 1000;
  const out = { note: "two-player tracking game, b1 = 1, b2 = -1, r = 0.1, T = 1; p1 = p2 = p; made by tools/cards/make_card_curves.js",
    r: 0.1, T: 1, logp: [], t: [], D1: [], D2: [], X: [] };
  for (let j = 0; j < NT; ++j) out.t.push(r3(j / (NT - 1)));
  for (let i = 0; i < NP; ++i) {
    const lp = -2 + 5 * i / (NP - 1), p = Math.pow(10, lp), o = run({ p1: p, p2: p });
    if (!o.ok || !o.converged) throw new Error("solve failed at p = " + p + ": " + o.message);
    const ts = o.samples.mean_t, M = o.samples.means;
    out.logp.push(r3(lp));
    for (const k of ["D1", "D2", "X"]) out[k].push(out.t.map((t) => r3(at(ts, M[k], t))));
    console.log(`p ${p.toPrecision(3)}\tD1(0) ${M.D1[0].toFixed(3)}  D2(0) ${M.D2[0].toFixed(3)}  max|X| ${Math.max(...M.X.map(Math.abs)).toExponential(1)}`);
  }
  const check = run({ p1: 9, p2: 9 });
  console.log("check p = 9: D1(0) =", check.samples.means.D1[0].toFixed(3));
  const file = path.join(root, "static/js/card-noisestate.json");
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, JSON.stringify(out));
  console.log("wrote", file, fs.statSync(file).size, "bytes");
});
