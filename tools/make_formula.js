// Typesets the noise-state definition as plain SVG paths (MathJax, fonts converted to outlines), for the home
// page's hero, where static/js/formula.js writes it on stroke by stroke. Run after editing the TeX below:
//     node tools/make_formula.js         (needs `npm install` in tools/dissertation)
const path = require("path"), fs = require("fs");
const MJ = path.join(__dirname, "dissertation", "node_modules", "mathjax-full", "js");
const { mathjax } = require(path.join(MJ, "mathjax.js"));
const { TeX } = require(path.join(MJ, "input/tex.js"));
require(path.join(MJ, "input/tex/AllPackages.js"));
const { SVG } = require(path.join(MJ, "output/svg.js"));
const { liteAdaptor } = require(path.join(MJ, "adaptors/liteAdaptor.js"));
const { RegisterHTMLHandler } = require(path.join(MJ, "handlers/html.js"));

const adaptor = liteAdaptor();
RegisterHTMLHandler(adaptor);
const doc = mathjax.document("", { InputJax: new TeX({ packages: ["base", "ams"] }), OutputJax: new SVG({ fontCache: "none" }) });

const LINES = {
  // the definition: player i's estimate, at time t, of the shock path up to u
  def: String.raw`\widehat{W}^{i}_t(u) = \mathbb{E}\left[\,W_u \mid \mathcal{F}^{i}_t\,\right]`,
  // and how filtering writes it: a singular part plus a deterministic kernel, the blueprint
  // the idea page (templates/idea.html): a linear process, and a player's forecast of it
  linear: String.raw`L_t = \bar L(t) + \int_0^t L(t,s)\, dW_s`,
  cond: String.raw`\widehat L^{i}_t = \bar L(t) + \int_0^t L(t,u)\, d_u\widehat W^{i}_t(u)`,
  hands1: String.raw`\Delta X_{t+1} = A X_t + a^1_t + a^2_t + a^3_t + \cdots + \text{noise}_{t+1}`,
  hands2: String.raw`a^i_t = K^i\,\widehat X^{i}_t + k^i`,
  hands3: String.raw`\Delta X_{t+1} = A X_t + K^1\widehat X^{1}_t + K^2\widehat X^{2}_t + \cdots + \text{noise}_{t+1}`,
  // the wedge page (templates/wedge.html)
  sep: String.raw`a^\star_t = K\,x_t + k_t \;\longrightarrow\; a^\star_t = K\,\widehat x_t + k_t`,
  backward1: String.raw`-\frac{d}{dt}H^{X,i}_t = \underbrace{G^{XX,i}X_t + G^{X,i}}_{\text{immediate value}} + \underbrace{A^{\top} H^{X,i}_t}_{\text{propagated value}}`,
  backward2: String.raw`+\;\underbrace{\sum_{k\neq i}\Gamma^{k\top}\!\int_0^t \widetilde X^{k}(t,z)\, H^{k,i}(t,z)\,dz}_{\text{manipulation price}}`,
  control: String.raw`D^i_t = -\big(G^{DD,i}\big)^{-1}\,\widehat H^{X,i}_t`,
  t1: String.raw`\mathbb{E}^1[X]`,
  t2: String.raw`\mathbb{E}^1\big[\mathbb{E}^2[X]\big]`,
  t3: String.raw`\mathbb{E}^1\Big[\mathbb{E}^2\big[\mathbb{E}^1[X]\big]\Big]`,
  t4: String.raw`\mathbb{E}^1\bigg[\mathbb{E}^2\Big[\mathbb{E}^1\big[\mathbb{E}^2[X]\big]\Big]\bigg]`,
  blueprint: String.raw`d_u \widehat{W}^{i}_t(u) = \Pi^i\, dW_u + \Big(\int_0^t F^{i}_t(u,s)\, dW_s\Big)\, du`,
};
const out = {};
for (const [k, tex] of Object.entries(LINES)) {
  const node = doc.convert(tex, { display: true, em: 16, ex: 8, containerWidth: 1200 });
  let svg = adaptor.innerHTML(node);
  // plain paths only: MathJax marks glyphs with data attributes and nests <use>-free groups when fontCache is none
  svg = svg.replace(/ data-[a-z-]+="[^"]*"/g, "").replace(/<title>.*?<\/title>/g, "");
  out[k] = { tex, svg };
}
const dest = path.join(__dirname, "..", "data", "formula.json");
fs.writeFileSync(dest, JSON.stringify(out, null, 1));
console.log("wrote", dest, Object.fromEntries(Object.entries(out).map(([k, v]) => [k, v.svg.length])));
