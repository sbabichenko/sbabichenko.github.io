#!/usr/bin/env python3
"""Subset Latin Modern Math to the characters the pages' MathML uses, keeping its OpenType MATH table (the stretchy
delimiters and radicals need it), as static/dissertation/lm-math.woff2.  Run after prerender.js (build.py does).
Needs fontTools with brotli, and the font from TeX Live (fonts-lmodern / texlive): LM_MATH overrides its path."""
import glob, html, os, re, subprocess, sys, tempfile
from pathlib import Path

SITE = Path(__file__).resolve().parents[2]
FONT = os.environ.get("LM_MATH", "/usr/share/texmf/fonts/opentype/public/lm-math/latinmodern-math.otf")
chars = set()
for f in glob.glob(str(SITE / "data" / "dissertation" / "*.html")):
    for m in re.findall(r"<math[\s\S]*?</math>", Path(f).read_text()):
        m = re.sub(r"<annotation[\s\S]*?</annotation>", "", m)
        for t in re.findall(r">([^<]+)<", m):
            chars.update(html.unescape(t))
chars.discard("\n")
with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
    fh.write("".join(sorted(chars)))
out = SITE / "static" / "dissertation" / "lm-math.woff2"
subprocess.run([sys.executable, "-m", "fontTools.subset", FONT, f"--text-file={fh.name}", "--layout-features=*",
                "--flavor=woff2", f"--output-file={out}"], check=True)
print(f"{len(chars)} characters -> {out} ({out.stat().st_size // 1024} KB)")
