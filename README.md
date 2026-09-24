Code for my personal website.

Most code is adapted from my friend [Daniel's website](https://dtnaylor.com) who in term adapted it from their friend [Tanay's website](https://github.com/TanayB11/me). Based on the [Zola](https://www.getzola.org/) SSG and [Kita](https://www.getzola.org/themes/kita/) theme.
## Interactive pages

Two pages run code in the browser. Both are ordinary Zola pages (the site's header, footer and theme) whose
assets live under `static/`, so `zola serve` shows them like any other page.

- `/lqg`: the noise-state game explorer. `content/lqg.md` and `templates/explorer.html`; the page's code and style
  are `static/lqg/explorer.js` and `static/lqg/explorer.css` (every rule scoped to `.explorer`). The solver is the C++
  port of noisestate compiled to WebAssembly: `noisestate.js` (single-threaded, wasm inlined) and
  `noisestate-mt.{js,wasm,worker.js}` (threaded), rebuilt in the port's repository with `make wasm wasm-mt` and copied
  here. `worker.js` picks the threaded build when the page is cross-origin isolated, which
  `coi-serviceworker.js` (MIT, scoped to `/lqg/`) arranges on GitHub Pages; the page reloads once on a first visit.
- `/mesh`: the Decision Mesh demo. `content/mesh.md` and `templates/mesh.html`; `static/mesh/decision-mesh.js` is a
  port of `DecisionMesh.py` (its squared error matches the Python's to 1e-12), `static/mesh/mesh-demo.js` draws it.
  It reuses the explorer's stylesheet.

Presets in the explorer are model files in `PRESETS` at the top of `explorer.js`, each with its sliders; a new tab is
a new entry there.
