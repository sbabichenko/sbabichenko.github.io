// Typesets the asides' formulas (asides.py) the way prerender.js does the pages': KaTeX MathML, mathvariant letters
// mapped for Chrome.  Reads [{tex, display}] on stdin, writes the typeset spans as JSON.
const path = require("path");
const katex = require(path.join(__dirname, "node_modules", "katex"));
const fixVariant = require(path.join(__dirname, "mathvariant.js"));
let buf = "";
process.stdin.on("data", (d) => (buf += d)).on("end", () => {
  const out = JSON.parse(buf).map(({ tex, display }) =>
    `<span class="math ${display ? "display" : "inline"}">` +
    fixVariant(katex.renderToString(tex, { displayMode: display, throwOnError: true, strict: false, output: "mathml" })) + "</span>");
  process.stdout.write(JSON.stringify(out));
});
