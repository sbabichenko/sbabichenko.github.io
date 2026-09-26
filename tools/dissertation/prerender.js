// Typesets every formula in data/dissertation/*.html with KaTeX at build time, in place, as native MathML (KaTeX's
// "mathml" output, with mathvariant letters mapped to Unicode math alphanumerics by mathvariant.js, since Chrome renders
// only those), so the pages arrive with their mathematics already set: about a quarter of the HTML and half the DOM of
// KaTeX's HTML output. The font is a Latin Modern Math subset (subset_math_font.py, run after this). Formulas KaTeX
// rejects are listed and left as TeX. Usage: node tools/dissertation/prerender.js
// (needs `npm install` in tools/dissertation, which fetches the pinned katex)
const fs = require("fs"), path = require("path");
const katex = require(path.join(__dirname, "node_modules", "katex"));
const fixVariant = require(path.join(__dirname, "mathvariant.js"));
const dir = path.join(__dirname, "../../data/dissertation");
const unescape = (s) => s.replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&quot;/g, '"').replace(/&#39;/g, "'").replace(/&amp;/g, "&");
let n = 0, bad = 0;
for (const f of fs.readdirSync(dir).filter((f) => f.endsWith(".html"))) {
  const file = path.join(dir, f);
  let src = fs.readFileSync(file, "utf8");
  src = src.replace(/<span class="math (inline|display)"( id="[^"]*")?>([\s\S]*?)<\/span>/g, (all, mode, id, body) => {
    const tex = unescape(body).replace(/^\\[([]/, "").replace(/\\[)\]]$/, "");
    n++;
    try {
      // KaTeX's MathML drops the numbers of a multi-row numbered display (align*/gather* with a \tag per row: its number
      // cells come out empty), so those keep KaTeX's HTML, whose stylesheet the pages still load
      const rowTags = /\\tag\b/.test(tex) && /\\begin\{(align|gather)\*?\}/.test(tex);
      const out = rowTags
        ? katex.renderToString(tex, { displayMode: true, throwOnError: true, strict: false, output: "html" })
        : fixVariant(katex.renderToString(tex, { displayMode: mode === "display", throwOnError: true, strict: false, output: "mathml" }));
      return `<span class="math ${mode}"${id || ""}>${out}</span>`;
    } catch (e) {
      bad++;
      console.error(f, e.message.slice(0, 120));
      return all;
    }
  });
  fs.writeFileSync(file, src);
}
console.log(`typeset ${n} formulas, ${bad} left as TeX`);
