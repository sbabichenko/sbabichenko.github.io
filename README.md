Code for my personal website.

Most code is adapted from my friend [Daniel's website](https://dtnaylor.com) who in term adapted it from their friend [Tanay's website](https://github.com/TanayB11/me). Based on the [Zola](https://www.getzola.org/) SSG and [Kita](https://www.getzola.org/themes/kita/) theme.
## Interactive pages

Two pages run code in the browser. Both are ordinary Zola pages (the site's header, footer and theme) whose
assets live under `static/`, so `zola serve` shows them like any other page.

- `/noisestate`: the noise-state game explorer. `content/noisestate.md` and `templates/explorer.html`; the page's code
  and style are `static/noisestate/explorer.js` and `static/noisestate/explorer.css` (every rule scoped to
  `.explorer`). The solver is the C++ port of noisestate compiled to WebAssembly: `noisestate.{js,wasm}`
  (single-threaded) and `noisestate-mt.{js,wasm,worker.js}` (threaded), rebuilt in the port's repository and copied
  here. `worker.js` picks the threaded build when the page is cross-origin isolated, which `coi-serviceworker.js`
  (MIT) arranges on GitHub Pages; the page reloads once on a first visit. The old `/lqg/` address redirects here
  (`static/lqg/index.html`).
- `/decision-mesh`: the Decision Mesh. `content/decision-mesh.md` and `templates/decision-mesh.html`: the live fit
  (`static/mesh/gate.js`, the WebAssembly estimators in `static/mesh/fit-worker.js`), the illustrated gate
  (`static/mesh/gate-how.js`, `gate-how.css`) and the geometry race (`static/mesh/decision-mesh.js`, a port of
  `DecisionMesh.py` whose squared error matches the Python's to 1e-12, drawn by `static/mesh/mesh-demo.js`). The story
  and the race load their scripts when scrolled near. `/decision-mesh/detail` is `content/decision-mesh-detail.md`
  and `templates/gate-deep.html`. The old `/gate/`, `/gate/how/`, `/mesh/` and `/gate/deep/` are aliases.
  It reuses the explorer's stylesheet.

Presets in the explorer are model files in `PRESETS` at the top of `explorer.js`, each with its sliders; a new tab is
a new entry there.

`node tools/check_explorer_python.js` runs the explorer's Python tab through noisestate and checks that it builds each
preset's model.  It needs the site built and served locally with the local address as the base URL
(`zola build --base-url http://127.0.0.1:8770 --output-dir <dir>`, then serve `<dir>` on that port).  A build with the
default base URL loads `explorer.js` from the live site, so the check would silently test the published file.

## The social card

`static/images/og.png` (the `og:image`) is rendered, not drawn: `tools/make_og.js` runs the mesh
engine in a headless browser and screenshots the result. Re-run it with the site served locally:

    node tools/make_og.js          # needs playwright and the site on http://127.0.0.1:8770
