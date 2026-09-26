#!/usr/bin/env python3
"""Where the dissertation pages on the site and the LaTeX have drifted apart.

    python3 tools/dissertation/drift.py [/path/to/dissertation] [--built DIR] [--keep] [--only chapter-1,...] [--out FILE]

The site's pages (data/dissertation/*.html) have been edited after they were built, and the LaTeX has been edited
too, so neither is a copy of the other. This builds the LaTeX (default ~/dissertation, after a full pdflatex build,
as for build.py) into a scratch copy of the site, never into the site itself, and compares the two page by page:
every paragraph, heading, caption and list item, as plain text. Formulas are compared by the glyphs KaTeX drew for
them (the pages carry no TeX), so a changed formula shows up as a changed paragraph. The blocks of a page are
aligned in order, and a block that moved but kept most of its words is reported as changed rather than as one
removed and one added.

The report lists, per page: blocks only on the site (edits to carry back into the LaTeX), blocks only in the LaTeX,
and blocks that differ, with a word-level diff ([-site-] {+LaTeX+}). Needs what build.py needs, pandoc 3 on PATH
(or in $PANDOC_DIR) and rsync. --built DIR compares against a scratch copy built earlier (its data/dissertation)
instead of building again; --keep leaves the scratch copy in place and prints where it is.
"""
import difflib
import html
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from bs4 import BeautifulSoup

SITE = Path(__file__).resolve().parents[2]
ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
FLAGS = [a for a in sys.argv[1:] if a.startswith("--")]


def flag(name):
    """The value of --name=VALUE or --name VALUE, or None."""
    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a == f"--{name}" and i + 1 < len(argv):
            ARGS.remove(argv[i + 1]) if argv[i + 1] in ARGS else None
            return argv[i + 1]
        if a.startswith(f"--{name}="):
            return a.split("=", 1)[1]
    return None


BUILT, ONLY, OUT = flag("built"), flag("only"), flag("out")
KEEP = "--keep" in FLAGS
SRC = Path(ARGS[0] if ARGS else "~/dissertation").expanduser().resolve()


# ------------------------------------------------------------------ building the LaTeX into a scratch copy
def build_scratch():
    """Copy the site (without .git and public/) to a temporary folder and run build.py there; returns the folder."""
    env = dict(os.environ)
    if os.environ.get("PANDOC_DIR"):
        env["PATH"] = os.environ["PANDOC_DIR"] + os.pathsep + env["PATH"]
    if not shutil.which("pandoc", path=env["PATH"]):
        sys.exit("drift.py: pandoc 3 is not on PATH; put it there or set PANDOC_DIR to the folder that holds it")
    work = Path(tempfile.mkdtemp(prefix="dissertation-drift-"))
    subprocess.run(["rsync", "-a", "--exclude", ".git", "--exclude", "public", f"{SITE}/", f"{work}/"], check=True)
    print(f"building {SRC} into {work} ...", file=sys.stderr)
    r = subprocess.run([sys.executable, str(work / "tools/dissertation/build.py"), str(SRC), "--force"],
                       env=env, capture_output=True, text=True)
    if r.returncode:
        sys.stderr.write(r.stdout[-3000:] + r.stderr[-3000:])
        sys.exit("drift.py: the build failed (above)")
    return work


# ------------------------------------------------------------------ a page as a list of text blocks
BLOCKS = ["p", "h1", "h2", "h3", "h4", "h5", "figcaption", "li", "td", "th", "dt", "dd"]


def formula(node):
    """A formula as the glyphs KaTeX drew, without its spacing: the same TeX gives the same string."""
    t = node.get_text("")
    t = re.sub(r"[​‌‍⁠﻿\s]+", "", t)
    return f"⟦{t}⟧"


def blocks(path):
    """[(heading, text)]: every block element that holds text directly, in page order, with the heading it sits under."""
    soup = BeautifulSoup(path.read_text(), "html.parser")
    for tag in soup.find_all(["script", "style", "svg", "picture", "img", "button"]):
        tag.decompose()
    for m in soup.select("span.katex-display, span.katex"):
        if m.find_parent(class_="katex"):
            continue
        m.replace_with(formula(m))
    out, heading = [], ""
    for el in soup.find_all(BLOCKS):
        # a block inside another counted block (a paragraph in a list item) is read with its parent, not twice
        if el.find_parent(BLOCKS):
            continue
        text = re.sub(r"\s+", " ", html.unescape(el.get_text(" "))).strip()
        # a hyphen at a line break in the LaTeX comes out as "finite- dimensional"; a suspended one ("first- and") stays
        text = re.sub(r"(\w)- (?!(?:and|or|to|through|as|nor)\b)([a-z])", r"\1-\2", text)
        if not text:
            continue
        if el.name in ("h1", "h2", "h3", "h4", "h5"):
            heading = text
        out.append((heading, text))
    return out


# ------------------------------------------------------------------ comparing
def worddiff(a, b, context=6):
    """The words that differ, with a little context: [-site words-] {+LaTeX words+}."""
    wa, wb, parts = a.split(), b.split(), []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, wa, wb, autojunk=False).get_opcodes():
        if op == "equal":
            seg = wa[i1:i2]
            parts.append(" ".join(seg) if len(seg) <= 2 * context else " ".join(seg[:context]) + " ... " + " ".join(seg[-context:]))
            continue
        if i2 > i1:
            parts.append("[-" + " ".join(wa[i1:i2]) + "-]")
        if j2 > j1:
            parts.append("{+" + " ".join(wb[j1:j2]) + "+}")
    return " ".join(parts)


def compare(site, tex):
    """Align two block lists; returns (only_site, only_tex, changed) as lists of (heading, text) / (heading, a, b)."""
    ka = [t for _, t in site]
    kb = [t for _, t in tex]
    only_site, only_tex, changed = [], [], []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, ka, kb, autojunk=False).get_opcodes():
        if op == "equal":
            continue
        A, B = list(range(i1, i2)), list(range(j1, j2))
        # pair the blocks of a replaced run by word similarity, greedily, best pairs first
        pairs = []
        for i in A:
            for j in B:
                # word by word, or letter by letter for a block of a few words (a displayed formula is one "word")
                short = min(len(ka[i].split()), len(kb[j].split())) <= 3
                x, y = (ka[i], kb[j]) if short else (ka[i].split(), kb[j].split())
                r = difflib.SequenceMatcher(None, x, y, autojunk=False).ratio()
                if r >= 0.5:
                    pairs.append((r, i, j))
        used_a, used_b = set(), set()
        for r, i, j in sorted(pairs, reverse=True):
            if i in used_a or j in used_b:
                continue
            used_a.add(i); used_b.add(j)
            changed.append((site[i][0] or tex[j][0], ka[i], kb[j]))
        only_site += [site[i] for i in A if i not in used_a]
        only_tex += [tex[j] for j in B if j not in used_b]
    return only_site, only_tex, changed


def clip(s, n=240):
    return s if len(s) <= n else s[:n] + " ..."


def main():
    work = Path(BUILT).expanduser().resolve() if BUILT else build_scratch()
    fresh = work / "data" / "dissertation"
    live = SITE / "data" / "dissertation"
    pages = sorted(p.name for p in live.glob("*.html"))
    if ONLY:
        pages = [p for p in pages if p[:-5] in ONLY.split(",")]
    lines, totals = [], [0, 0, 0]
    for name in pages:
        if not (fresh / name).exists():
            lines.append(f"\n## {name[:-5]}: not in the LaTeX build")
            continue
        s, t = blocks(live / name), blocks(fresh / name)
        only_s, only_t, ch = compare(s, t)
        totals[0] += len(only_s); totals[1] += len(only_t); totals[2] += len(ch)
        if not (only_s or only_t or ch):
            lines.append(f"\n## {name[:-5]}: the same ({len(s)} blocks)")
            continue
        lines.append(f"\n## {name[:-5]}: {len(ch)} changed, {len(only_s)} only on the site, {len(only_t)} only in the LaTeX")
        for h, a, b in ch:
            lines.append(f"\n  changed, under \"{clip(h, 70)}\":\n    {worddiff(a, b)}")
        for h, a in only_s:
            lines.append(f"\n  only on the site, under \"{clip(h, 70)}\":\n    {clip(a)}")
        for h, b in only_t:
            lines.append(f"\n  only in the LaTeX, under \"{clip(h, 70)}\":\n    {clip(b)}")
    head = (f"# Drift between the site's dissertation pages and {SRC}\n"
            f"# in all: {totals[2]} changed blocks, {totals[0]} only on the site, {totals[1]} only in the LaTeX\n"
            f"# [-...-] is the site's wording, {{+...+}} the LaTeX's")
    report = head + "\n" + "\n".join(lines) + "\n"
    if OUT:
        Path(OUT).write_text(report)
        print(f"wrote {OUT}", file=sys.stderr)
    else:
        print(report)
    if BUILT is None:
        if KEEP:
            print(f"scratch copy kept at {work}", file=sys.stderr)
        else:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
