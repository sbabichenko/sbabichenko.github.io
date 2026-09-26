#!/usr/bin/env python3
"""The dissertation's search index: static/dissertation/search.json, read by static/dissertation/search.js.

One entry per heading, per theorem-like block, per figure caption and per paragraph or list item of every page in
data/dissertation (the fragments build.py writes), each with the anchor a hit should jump to: the block's own id
if it has one, else the nearest section, theorem or figure above it. Math is left out of the text; each numbered
equation gets an entry of its own ("Equation (3.14)") with the words around it, and a paragraph with math in it
shows "…" where the math was.
build.py runs this at the end; it can also be run alone after the fragments change:

    python3 tools/dissertation/search_index.py
"""
import json
import re
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, Tag

SITE = Path(__file__).resolve().parents[2]
DATA = SITE / "data" / "dissertation"
OUT = SITE / "static" / "dissertation" / "search.json"
BLOCKS = {"p", "li", "figcaption", "h1", "h2", "h3", "h4", "dt", "dd", "blockquote"}


def text_of(node):
    """The words of a node, with each formula replaced by a single ellipsis and equation tags kept as (n.m)."""
    out = []
    for d in node.descendants:
        if isinstance(d, Tag) and "katex" in (d.get("class") or []):
            continue
        if isinstance(d, NavigableString):
            if any("katex" in (p.get("class") or []) for p in d.parents if isinstance(p, Tag)):
                continue
            out.append(str(d))
        elif isinstance(d, Tag) and "math" in (d.get("class") or []):
            out.append(" … ")
    t = re.sub(r"\s+", " ", "".join(out)).strip()
    return re.sub(r"(?:\s*…\s*)+", " … ", t).strip()


def main():
    manifest = json.loads((DATA / "manifest.json").read_text())
    entries = []
    for page in manifest:
        slug = page["slug"]
        soup = BeautifulSoup((DATA / f"{slug}.html").read_text(), "html.parser")
        anchor = ""
        for el in soup.find_all(True):
            classes = el.get("class") or []
            if el.get("id") and (el.name in ("section", "figure") or "thm" in classes):
                anchor = el["id"]
            if el.name == "span" and "display" in classes and el.get("id"):
                tag = el.select_one(".katex-html .tag")
                if tag is None:                          # MathML: the number is the last cell's text, "(1.2.3)"
                    cells = [m for m in el.select("mtd > mtext") if re.fullmatch(r"\([\w.\s\u200b]+\)", m.get_text())]
                    tag = cells[-1] if cells else None
                if tag:
                    num = re.sub(r"[\s\u200b]+", "", tag.get_text())
                    block = el.find_parent(BLOCKS) or el.parent
                    entries.append({"s": slug, "a": el["id"], "k": "e", "l": "Equation " + num, "x": text_of(block)[:160]})
                continue
            if "thm" in classes:
                head = el.find("strong")
                if head:
                    title = text_of(el.find("p") or el)[:160]
                    entries.append({"s": slug, "a": el["id"], "k": "t", "l": head.get_text(" ", strip=True), "x": title})
                continue
            if el.name not in BLOCKS or el.find_parent(["li", "figcaption", "blockquote"]) is not None and el.name == "p":
                continue
            if el.find_parent(class_="thm") is not None:
                # a theorem's paragraphs are searched too, under the theorem's own anchor
                pass
            t = text_of(el)
            if not t or len(t) < 3:
                continue
            if el.name in ("h1", "h2", "h3", "h4"):
                sec = el.find_parent("section")
                num = el.get("data-num", "")
                label = re.sub(r"^\s*" + re.escape(num) + r"\s*", "", t) if num else t
                if el.name == "h1" and num:
                    label = page["title"]
                    num = ""
                entries.append({"s": slug, "a": (sec.get("id") if sec else "") or anchor, "k": "h", "l": (num + " " + label).strip()})
            else:
                entries.append({"s": slug, "a": anchor, "k": "p", "x": t})
    pages = {p["slug"]: (p.get("label") or p["title"]) for p in manifest}
    OUT.write_text(json.dumps({"pages": pages, "entries": entries}, ensure_ascii=False, separators=(",", ":")))
    print(f"search index: {len(entries)} entries, {OUT.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
