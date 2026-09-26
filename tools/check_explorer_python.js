// Checks the explorer's Python (static/noisestate/explorer.js, PYTHON): for every game, at its defaults and at
// random slider settings, the page's own model file and its Python are taken from the served site, the Python is
// run with noisestate, and the model it builds must be the model file's (both through Model.from_dict().to_dict(),
// the comparison noisestate's tests/test_expr.py makes). Needs playwright, python3 that imports noisestate (pip install
// noisestate, or PYTHONPATH=path/to/noisestate), and the site
// served locally:
//     node tools/check_explorer_python.js [http://127.0.0.1:8770] [cases per game]
// Build with the local address as the base URL (zola build --base-url http://127.0.0.1:8770 --output-dir ...):
// without it the page loads explorer.js from the live site, and the check silently tests the published file.
const { chromium } = require("playwright");
const { execFileSync } = require("child_process");

const BASE = (process.argv[2] || "http://127.0.0.1:8770").replace(/\/$/, "");
const PER_GAME = +(process.argv[3] || 4);

const CHECK = String.raw`
import copy, json, sys
import noisestate as ns

def canon(d):
    d = copy.deepcopy(d)
    for a in d.get("agents", {}).values():
        if a.get("myopic") is False: del a["myopic"]
        for r in (a.get("signals") or {}).values():
            if r.get("delay") == 0: del r["delay"]
    h = d.get("horizon", {})
    if h.get("kind") == "transition" and "model" in h.get("past", {}):
        h["past"]["model"] = canon(h["past"]["model"])
    return ns.Model.from_dict(d).to_dict()

bad = 0
for case in json.load(sys.stdin):
    want = canon(case["model"])
    if case["model"]["horizon"].get("settle") is not None:          # the march: the code solves to a fixed T
        want["horizon"].pop("settle", None); want["horizon"]["T"] = 6.0
    scope = {}
    exec(case["code"].replace("eq = game.solve()", ""), scope)
    got = scope["game"].to_dict()
    ok = got == want
    bad += not ok
    print(("ok  " if ok else "DIFF"), case["label"])
    if not ok:
        for k in sorted(set(got) | set(want)):
            if got.get(k) != want.get(k): print("   ", k, "\n      python:", str(got.get(k))[:400], "\n      yaml:  ", str(want.get(k))[:400])
sys.exit(1 if bad else 0)
`;

(async () => {
  const b = await chromium.launch();
  const ctx = await b.newContext({ serviceWorkers: "block" });
  const p = await ctx.newPage();
  await p.goto(BASE + "/noisestate/", { waitUntil: "load" });
  await p.waitForFunction(() => typeof PYTHON === "object" && typeof currentModel === "function");
  const cases = await p.evaluate((perGame) => {
    let seed = 7;
    const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
    const out = [];
    for (const g of Object.keys(PYTHON)) {
      for (let c = 0; c < perGame; ++c) {
        game = g;
        const def = PRESETS[g], m = presetModel(g);
        values[g] = { nodes: m.numerics.nodes };
        if (def.grid) values[g][def.grid.key] = def.grid.options[c % def.grid.options.length][0];
        for (const s of def.sliders) {
          const v0 = sliderParams(m, s)[0][s.param || s.key];
          if (c === 0) { values[g][s.key] = v0; continue; }
          const u = rnd();
          values[g][s.key] = s.log ? +(s.min * Math.pow(s.max / s.min, u)).toPrecision(3) : +(s.min + (s.max - s.min) * u).toFixed(2);
        }
        if (def.nodes && c > 0) values[g].nodes = def.nodes.options[c % def.nodes.options.length][0];
        opts.march = !!def.march && c === 3;
        const d = currentModel();
        out.push({ label: `${g} case ${c}${opts.march ? " (march)" : ""}`, model: d, code: PYTHON[g](d) });
      }
    }
    opts.march = false;
    return out;
  }, PER_GAME);
  await b.close();
  try {
    process.stdout.write(execFileSync("python3", ["-c", CHECK], { input: JSON.stringify(cases) }).toString());
    console.log(`all ${cases.length} cases build the model file's model`);
  } catch (e) {
    process.stdout.write((e.stdout || "").toString()); process.stderr.write((e.stderr || "").toString());
    process.exit(1);
  }
})();
