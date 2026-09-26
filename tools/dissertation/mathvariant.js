// Map MathML mathvariant (script, bold, double-struck, fraktur, sans-serif, monospace, bold-italic, italic) to Unicode
// mathematical alphanumerics, which MathML Core renders (Chrome ignores mathvariant other than "normal").
const base = { "bold": [0x1D400, 0x1D41A, 0x1D7CE], "italic": [0x1D434, 0x1D44E, null], "bold-italic": [0x1D468, 0x1D482, null],
  "script": [0x1D49C, 0x1D4B6, null], "bold-script": [0x1D4D0, 0x1D4EA, null], "fraktur": [0x1D504, 0x1D51E, null],
  "double-struck": [0x1D538, 0x1D552, 0x1D7D8], "bold-fraktur": [0x1D56C, 0x1D586, null], "sans-serif": [0x1D5A0, 0x1D5BA, 0x1D7E2],
  "bold-sans-serif": [0x1D5D4, 0x1D5EE, 0x1D7EC], "sans-serif-italic": [0x1D608, 0x1D622, null], "monospace": [0x1D670, 0x1D68A, 0x1D7F6] };
const holes = { "italic": { h: 0x210E }, "script": { B: 0x212C, E: 0x2130, F: 0x2131, H: 0x210B, I: 0x2110, L: 0x2112, M: 0x2133, R: 0x211B, e: 0x212F, g: 0x210A, o: 0x2134 },
  "fraktur": { C: 0x212D, H: 0x210C, I: 0x2111, R: 0x211C, Z: 0x2128 }, "double-struck": { C: 0x2102, H: 0x210D, N: 0x2115, P: 0x2119, Q: 0x211A, R: 0x211D, Z: 0x2124 } };
const bold_greek = 0x1D6A8;
function mapChar(v, c) {
  const b = base[v]; if (!b) return null;
  if (holes[v] && holes[v][c]) return String.fromCodePoint(holes[v][c]);
  const code = c.codePointAt(0);
  if (c >= "A" && c <= "Z") return String.fromCodePoint(b[0] + code - 65);
  if (c >= "a" && c <= "z") return String.fromCodePoint(b[1] + code - 97);
  if (c >= "0" && c <= "9" && b[2]) return String.fromCodePoint(b[2] + code - 48);
  if (v === "bold" && code >= 0x391 && code <= 0x3A9) return String.fromCodePoint(bold_greek + code - 0x391);
  if (v === "bold" && code >= 0x3B1 && code <= 0x3C9) return String.fromCodePoint(bold_greek + 26 + code - 0x3B1);
  return null;
}
module.exports = function fix(mathml) {
  return mathml.replace(/<(mi|mn|mo|mtext)([^>]*?) mathvariant="([a-z-]+)"([^>]*)>([^<]*)<\/\1>/g, (all, tag, a1, v, a2, text) => {
    if (v === "normal") return all;
    let ok = true; const out = [...text].map((c) => { const r = mapChar(v, c); if (r === null && /[A-Za-z0-9]/.test(c)) ok = false; return r ?? c; }).join("");
    if (!ok) return all;
    const attrs = tag === "mi" && [...out].length === 1 ? ' mathvariant="normal"' : "";   // stop the single-letter auto-italic
    return `<${tag}${a1}${a2}${attrs}>${out}</${tag}>`;
  });
};
