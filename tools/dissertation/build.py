#!/usr/bin/env python3
"""The dissertation, as web pages.

    python3 tools/dissertation/build.py /path/to/dissertation

The argument is the dissertation's source folder after a full pdflatex build (so that the .aux, .toc and .bbl
are current). The script

  1. reads the numbers LaTeX assigned (labels, section numbers, bibliography order) from the .aux, .toc and .bbl,
  2. rewrites the sources for the web: citations become numbered links, TikZ pictures become the SVGs that
     `tikz external` wrote, PDF figures become images,
  3. converts the whole document with pandoc, once, so its own theorem and footnote bookkeeping sees everything,
  4. splits the result into one page per chapter, and puts LaTeX's numbers back on every heading, theorem,
     equation, figure and cross-reference (links across pages included),
  5. writes content/dissertation/*.md (front matter only) and the page bodies as HTML fragments that the
     templates include, plus the figures under static/dissertation/.

Every formula is then typeset with KaTeX at build time (prerender.js), which also lists any it cannot typeset.
Needs pandoc 3, pdftocairo (poppler), Pillow, BeautifulSoup and `npm install` in this folder.
"""
import html
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString

ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
FORCE = "--force" in sys.argv
SRC = Path(ARGS[0] if ARGS else ".").resolve()
SITE = Path(__file__).resolve().parents[2]
# which LaTeX the pages on the site were built from (written after each build, checked before the next)
STAMP = SITE / "data" / "dissertation" / "source.json"
CONTENT = SITE / "content" / "dissertation"
STATIC = SITE / "static" / "dissertation"
WORK = Path("/tmp/dissertation-web")
BASE = "/dissertation"

# ------------------------------------------------------------------ brace-aware readers
def group(s, i):
    """The {...} group starting at s[i] == '{'; returns (content, index after the group)."""
    assert s[i] == "{", s[i:i + 20]
    depth, j = 0, i
    while True:
        c = s[j]
        if c == "\\":
            j += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
        j += 1


def strip_tex(s):
    s = re.sub(r"\\(relax|protect|nobreakspace)\b\s*", "", s)
    s = s.replace("~", " ")
    return s.strip()


# ------------------------------------------------------------------ what LaTeX numbered
def read_aux(path):
    labels = {}
    s = path.read_text(errors="replace")
    for m in re.finditer(r"\\newlabel\{", s):
        key, j = group(s, m.end() - 1)
        if s[j] != "{":
            continue
        body, _ = group(s, j)
        if not body.startswith("{"):
            continue
        num, k = group(body, 0)
        page, _ = group(body, k) if k < len(body) and body[k] == "{" else ("", k)
        labels[key] = {"num": strip_tex(num), "page": strip_tex(page)}
    return labels


def read_toc(path):
    toc = []
    s = path.read_text(errors="replace")
    for m in re.finditer(r"\\contentsline\s*\{(\w+)\}", s):
        level = m.group(1)
        title, _ = group(s, m.end() + (1 if s[m.end()] != "{" else 0) - (0))
        num = ""
        n = re.match(r"\s*\\numberline\s*", title)
        if n:
            num, j = group(title, n.end())
            title = title[j:]
        toc.append({"level": level, "num": strip_tex(num), "title": title.strip()})
    return toc


def read_bbl(path):
    s = path.read_text(errors="replace")
    items = []
    parts = re.split(r"\\bibitem\[", s)[1:]
    for p in parts:
        # [label]{key} body
        depth, j = 1, 0
        while depth:
            if p[j] == "[":
                depth += 1
            elif p[j] == "]":
                depth -= 1
            j += 1
        label = p[:j - 1]
        key, k = group(p, j)
        body = p[k:].split("\\end{thebibliography}")[0].strip()
        m = re.match(r"(.*?)\((.*?)\)(.*)", label, re.S)
        short, year, long_ = (m.group(1), m.group(2), m.group(3)) if m else (label, "", label)
        items.append({"key": key.strip(), "short": short.replace("~", " ").replace("{", "").replace("}", "").strip(),
                      "year": year.strip(), "body": body})
    for n, it in enumerate(items, 1):
        it["n"] = n
    return items


# ------------------------------------------------------------------ the sources, rewritten for the web
def cite_link(key):
    return f"\\href{{{BASE}/references/\\#ref-{key}}}"


def compress(nums):
    """natbib's sort&compress: 1, 2, 3, 7 -> 1-3, 7"""
    nums = sorted(set(nums))
    out, i = [], 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1][0] == nums[j][0] + 1:
            j += 1
        if j - i >= 2:
            out.append(f"{nums[i][1]}--{nums[j][1]}")
        else:
            out.extend(n[1] for n in nums[i:j + 1])
        i = j + 1
    return ", ".join(out)


def rewrite_cites(s, bib):
    pat = re.compile(r"\\(citep|citet|cite|citealp|citeauthor|citeyear|citealt)\*?\s*((?:\[[^\]]*\]){0,2})\s*\{([^}]*)\}")

    def rep(m):
        kind, opts, keys = m.group(1), m.group(2), [k.strip() for k in m.group(3).split(",") if k.strip()]
        o = re.findall(r"\[([^\]]*)\]", opts)
        pre, post = ("", o[0]) if len(o) == 1 else (o[0], o[1]) if len(o) == 2 else ("", "")
        known = [k for k in keys if k in bib]
        missing = [k for k in keys if k not in bib]
        if missing:
            print("  unknown citation", missing, file=sys.stderr)
        links = [(bib[k]["n"], cite_link(k) + "{" + str(bib[k]["n"]) + "}") for k in known]
        if kind == "citeauthor":
            return ", ".join(bib[k]["short"] for k in known)
        if kind == "citeyear":
            return ", ".join(bib[k]["year"] for k in known)
        if kind in ("citet", "citealt"):
            parts = [bib[k]["short"] + "~[" + cite_link(k) + "{" + str(bib[k]["n"]) + "}" + (", " + post if post and i == len(known) - 1 else "") + "]"
                     for i, k in enumerate(known)]
            return ", ".join(parts)
        inner = compress(links)
        if pre:
            inner = pre + " " + inner
        if post:
            inner = inner + ", " + post
        return "[" + inner + "]" if kind != "citealp" else inner

    return pat.sub(rep, s)


TIKZ = {"n": 0}


def unwrap(s, cmd, skip):
    """\\cmd{a}{b}{body} -> body, for commands pandoc would drop along with their body"""
    out, i = [], 0
    while True:
        j = s.find("\\" + cmd, i)
        if j < 0:
            out.append(s[i:])
            return "".join(out)
        out.append(s[i:j])
        k = j + len(cmd) + 1
        for _ in range(skip):
            while s[k].isspace():
                k += 1
            _, k = group(s, k)
        while s[k].isspace():
            k += 1
        body, k = group(s, k)
        out.append(body)
        i = k


def rewrite(s, bib):
    s = rewrite_cites(s, bib)
    s = unwrap(s, "resizebox", 2)
    s = re.sub(r"\\looseness\s*=\s*-?\d+\s*", "", s)
    s = re.sub(r"\\[Nn]eedspace\{[^}]*\}", "", s)
    s = re.sub(r"\\setstretch\{[^}]*\}", "", s)
    # print-only spacing (the abstract's): pandoc would keep "=plus minus 1.5" as text
    s = re.sub(r"\\(?:x?spaceskip)\s*=[^\n]*", "", s)
    s = re.sub(r"\\tolerance\s*=\s*\d+", "", s)
    # the vita's tables fix a column width for print (@{}p{0.87\textwidth}r@{}); pandoc reads that spec as text, so
    # give it a plain two-column table
    s = re.sub(r"\\begin\{tabular\}\{@\{\}p\{[\d.]+\\textwidth\}r@\{\}\}", r"\\begin{tabular}{lr}", s)
    for env in ("compactmath", "singlespace", "doublespace", "Large"):
        s = re.sub(r"\\begin\{" + env + r"\}|\\end\{" + env + r"\}", "", s)

    def tikz(m):
        n = TIKZ["n"]
        TIKZ["n"] += 1
        return f"\\includegraphics{{{BASE}/media/tikz-{n}.svg}}"

    s = re.sub(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", tikz, s, flags=re.S)
    s = re.sub(r"\\includegraphics(\[[^\]]*\])?\{figs/(?:ch5/)?([^}]*?)\.(pdf|png)\}",
               lambda m: f"\\includegraphics{{{BASE}/media/{m.group(2)}.webp}}", s)
    return s


def web_master(bib):
    master = (SRC / "combined_dissertation.tex").read_text()
    pre, body = master.split("\\begin{document}")
    body = body.split("\\end{document}")[0]
    pre = re.sub(r"\\bmdefine\\(\w+)\{\\(\w+)\}", r"\\newcommand{\\\1}{\\boldsymbol{\\\2}}", pre)
    body = body.replace("\\input{chapters/dissertation_titlepages.tex}", "")
    body = re.sub(r"\\tableofcontents|\\pagenumbering\{\w+\}|\\setcounter\{page\}\{\d+\}", "", body)
    body = re.sub(r"\\bibliographystyle\{[^}]*\}|\\bibliography\{[^}]*\}", "", body)

    def inp(m):
        f = SRC / m.group(1)
        if not f.suffix:
            f = f.with_suffix(".tex")
        t = f.read_text()
        # a line break forced inside a heading ("Control stationarity\newline and ...") is a space on the web
        t = re.sub(r"(\\(?:sub)*section\*?(?:\[[^\]]*\])?\{[^{}]*?)\\newline\s*", r"\1 ", t)
        if f.name == "dissertation_abstract.tex":        # a centred title block in print; a chapter here
            t = "\\chapter*{Abstract}\n" + t.split("\\end{singlespace}\\end{center}", 1)[1]
        if f.name in ("dissertation_copyright.tex", "dissertation_dedication.tex"):
            return ""                                    # the page template carries these
        return rewrite(t, bib)

    body = re.sub(r"\\input\{([^}]*)\}", inp, body)
    return pre + "\\begin{document}\n" + body + "\n\\end{document}\n"


# ------------------------------------------------------------------ figures
def media():
    out = STATIC / "media"
    out.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    for pdf in sorted((SRC / "figs").glob("**/*.pdf")):
        if "_pre" in str(pdf):
            continue
        png = WORK / "fig" / pdf.stem
        png.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["pdftocairo", "-png", "-r", "220", "-singlefile", str(pdf), str(png)], check=True)
        im = Image.open(str(png) + ".png").convert("RGB")
        if im.width > 1800:
            im = im.resize((1800, round(im.height * 1800 / im.width)), Image.LANCZOS)
        im.save(out / (pdf.stem + ".webp"), quality=90, method=6)
    for n, pdf in enumerate(sorted((SRC / "tikzext").glob("*-figure*.pdf"), key=lambda p: int(re.search(r"figure(\d+)", p.name).group(1)))):
        subprocess.run(["pdftocairo", "-svg", str(pdf), str(out / f"tikz-{n}.svg")], check=True)


# ------------------------------------------------------------------ the pages
# which of pandoc's top-level sections go on which page, by the start of their titles
PAGES = [
    ("front", "Front matter", ["Abstract", "Acknowledgements", "Vita", "Notation"]),
    ("introduction", "Introduction", ["Introduction"]),
    ("chapter-1", None, ["Baseline"]),
    ("chapter-2", None, ["Delayed"]),
    ("chapter-3", None, ["Stationary Infinite"]),
    ("chapter-4", None, ["The Information Wedge in a Stationary"]),
    ("chapter-5", None, ["Local Strategic"]),
    ("chapter-6", None, ["Monitored"]),
    ("chapter-7", None, ["Conclusion"]),
    ("afterword", "Afterword", ["Afterword"]),
    ("appendix-a", None, ["Additional Finite"]),
]
THEOREMS = {"theorem", "thm", "lem", "lemma", "prop", "proposition", "cor", "corollary", "assum", "assumption",
            "conj", "defn", "definition", "exmp", "prob", "rem", "remark", "quest"}
NAMES = {"thm": "Theorem", "theorem": "Theorem", "lem": "Lemma", "lemma": "Lemma", "prop": "Proposition",
         "proposition": "Proposition", "cor": "Corollary", "corollary": "Corollary", "assum": "Assumption",
         "assumption": "Assumption", "conj": "Conjecture", "defn": "Definition", "definition": "Definition",
         "exmp": "Example", "prob": "Problem", "rem": "Remark", "remark": "Remark", "quest": "Question"}


def norm(t):
    t = re.sub(r"\\\(.*?\\\)|\$.*?\$", "", t)
    t = re.sub(r"\\[a-zA-Z]+", "", t)
    return re.sub(r"[^a-z0-9]", "", t.lower())


def fix_math(soup, labels, stats):
    for sp in soup.select("span.math"):
        tex = sp.get_text()
        keys = re.findall(r"\\label\{([^}]*)\}", tex)
        if not keys:
            continue
        for k in keys:
            num = labels.get(k, {}).get("num")
            if "display" in sp["class"] and num:
                tex = tex.replace(f"\\label{{{k}}}", f"\\tag{{{num}}}", 1)
                stats["eq"] += 1
            else:
                tex = tex.replace(f"\\label{{{k}}}", "", 1)
        body = tex[2:-2]
        # a reference inside a formula: KaTeX has no \eqref, so it becomes the printed number
        body = re.sub(r"\\eqref\{([^}]*)\}", lambda m: "\\text{(" + labels.get(m.group(1), {}).get("num", "?") + ")}", body)
        body = re.sub(r"\\ref\{([^}]*)\}", lambda m: "\\text{" + labels.get(m.group(1), {}).get("num", "?") + "}", body)
        if "\\tag" in body:
            b = body.strip()
            if b.startswith("\\begin{aligned}") and b.endswith("\\end{aligned}"):
                body = "\\begin{align*}" + b[len("\\begin{aligned}"):-len("\\end{aligned}")] + "\\end{align*}"
            body = body.replace("\\nonumber", "\\notag")
        body = re.sub(r"\\begin\{multline\*?\}", r"\\begin{gather*}", body)
        body = re.sub(r"\\end\{multline\*?\}", r"\\end{gather*}", body)
        sp.string = tex[:2] + body + tex[-2:]
        sp["id"] = keys[0]
        for k in keys[1:]:
            a = soup.new_tag("span", id=k)
            sp.insert_before(a)
    for sp in soup.select("span.math"):                # multline without a label; references inside formulas
        t = sp.get_text()
        t2 = re.sub(r"\\eqref\{([^}]*)\}", lambda m: "\\text{(" + labels.get(m.group(1), {}).get("num", "?") + ")}", t)
        t2 = re.sub(r"\\ref\{([^}]*)\}", lambda m: "\\text{" + labels.get(m.group(1), {}).get("num", "?") + "}", t2)
        if t2 != t:
            sp.string = t = t2
        if "multline" in t:
            sp.string = re.sub(r"\\(begin|end)\{multline\*?\}", r"\\\1{gather*}", t)


def fix_theorems(soup, labels, stats):
    # one counter shared by every theorem-like environment, reset per chapter, as \newtheorem{..}[theorem]
    # sets it up: a labelled result takes LaTeX's number, an unlabelled one the next after it
    chap, n = "", 0
    for el in soup.find_all(["h1", "div"]):
        if el.name == "h1":
            chap, n = el.get("data-num", ""), 0
            continue
        d = el
        cls = [c for c in d.get("class", []) if c in THEOREMS]
        if not cls:
            continue
        num = labels.get(d.get("id", ""), {}).get("num")
        if num:
            chap, n = num.rsplit(".", 1)[0], int(re.sub(r"\D", "", num.rsplit(".", 1)[1]) or 0)
        elif chap:
            n += 1
            d["data-seq"] = f"{chap}.{n}"
    for d in soup.find_all("div"):
        cls = [c for c in d.get("class", []) if c in THEOREMS]
        if not cls:
            continue
        env = cls[0]
        d["class"] = ["thm", "thm-" + NAMES[env].lower()]
        strong = d.find("strong")
        num = labels.get(d.get("id", ""), {}).get("num") or d.get("data-seq")
        d.attrs.pop("data-seq", None)
        if strong:
            m = re.match(r"(\w+)\s+([\w.]+)(.*)", strong.get_text())
            if m and num:
                if m.group(2) != num:
                    stats["thm_renum"] += 1
                strong.string = f"{NAMES[env]} {num}"
            stats["thm"] += 1


def fix_figures(soup, labels):
    for f in soup.find_all("figure"):
        num = labels.get(f.get("id", ""), {}).get("num")
        cap = f.find("figcaption")
        if cap and num:
            b = soup.new_tag("span", attrs={"class": "fignum"})
            b.string = f"Figure {num}. "
            cap.insert(0, b)
        for e in f.find_all("embed"):
            e.name = "img"
            e["alt"] = cap.get_text(" ", strip=True)[:140] if cap else ""
        for im in f.find_all("img"):
            im["loading"] = "lazy"
            im.attrs.pop("style", None)
            im.attrs.pop("width", None)
    for t in soup.find_all("table"):
        cap = t.find("caption")
        tid = t.get("id") or (t.parent.get("id") if t.parent and t.parent.name == "div" else None)
        num = labels.get(tid or "", {}).get("num")
        if cap and num:
            b = soup.new_tag("span", attrs={"class": "fignum"})
            b.string = f"Table {num}. "
            cap.insert(0, b)


def fix_tables(soup):
    for t in soup.find_all("table"):
        for cg in t.find_all("colgroup"):
            cg.decompose()
        t.attrs.pop("style", None)
        w = soup.new_tag("div", attrs={"class": "tablewrap"})
        t.wrap(w)


def fix_headings(soup, toc, stats):
    lv = {"h1": "chapter", "h2": "section", "h3": "subsection", "h4": "subsubsection"}
    ptr = 0
    for h in soup.find_all(["h1", "h2", "h3", "h4"]):
        title = norm(h.get_text())
        want = lv[h.name]
        for j in range(ptr, min(len(toc), ptr + 40)):
            e = toc[j]
            if e["level"] == want and norm(e["title"]) == title:
                ptr = j + 1
                if e["num"]:
                    n = soup.new_tag("span", attrs={"class": "secnum"})
                    n.string = ("Chapter " if want == "chapter" and e["num"].isdigit() else
                                "Appendix " if want == "chapter" else "") + e["num"]
                    h.insert(0, n)
                    h["data-num"] = e["num"]
                    stats["heads"] += 1
                break


# pages told as a story (templates/thesis-story.html): each paragraph becomes a step beside a drawing; a paragraph
# can be split before a given sentence so that it gets two drawings
STORIES = {
    "afterword": [("illusion", None), ("selection", "Natural selection is a statistical tool"), ("anchor", None),
                  ("barrier", None), ("orbit", "Mathematics, the tool that let Isaac Newton"), ("noise", None),
                  ("questions", None)],
}


def story(body, plan):
    soup = BeautifulSoup(body, "html.parser")
    sec = soup.find("section")
    paras = [p for p in sec.find_all("p", recursive=False)]
    chunks = []                                     # (html) in reading order, split where the plan says
    splits = {text: scene for scene, text in plan if text}
    for p in paras:
        html_ = re.sub(r"\s+", " ", "".join(str(c) for c in p.contents))
        cut = next((t for t in splits if t in html_), None)
        if cut:
            i = html_.index(cut)
            chunks += [html_[:i].rstrip(), html_[i:]]
        else:
            chunks.append(html_)
    if len(chunks) != len(plan):
        raise SystemExit(f"story plan has {len(plan)} steps, text has {len(chunks)}")
    for p in paras:
        p.decompose()
    steps = soup.new_tag("div", attrs={"class": "steps"})
    for (scene, _), c in zip(plan, chunks):
        st = soup.new_tag("div", attrs={"class": "step", "data-scene": scene})
        st.append(BeautifulSoup(f"<p>{c}</p>", "html.parser"))
        steps.append(st)
    h1 = sec.find("h1")
    h1.insert_after(steps)
    return str(soup)


def fingerprint():
    """a hash of the LaTeX that makes the pages, and the newest time any of it was edited"""
    import hashlib
    files = sorted([SRC / "combined_dissertation.tex", *(SRC / "chapters").glob("*.tex")])
    h = hashlib.sha256()
    for f in files:
        h.update(f.name.encode()); h.update(f.read_bytes())
    newest = max(f.stat().st_mtime for f in files)
    return h.hexdigest()[:16], newest


def guard():
    """Refuse to overwrite pages built from another copy of the dissertation. Two copies once diverged: the site's
    pages came from one, a regeneration from the other would have silently undone its wording. --force overrides."""
    sha, newest = fingerprint()
    if FORCE:
        return sha
    if not STAMP.exists():
        sys.exit("build.py: the pages on the site carry no record of the LaTeX they came from (they were built\n"
                 "elsewhere), so this source may be older or newer than them. Compare the two, then rerun with --force.")
    old = json.loads(STAMP.read_text())
    if old.get("tex") != sha and newest < old.get("built", 0):
        sys.exit(f"build.py: this LaTeX differs from what the pages were built from, and none of it was edited since\n"
                 f"that build ({old.get('source')}). It is probably an older copy. Rerun with --force to use it anyway.")
    return sha


def main():
    sha = guard()
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)
    labels = read_aux(SRC / "combined_dissertation.aux")
    toc = read_toc(SRC / "combined_dissertation.toc")
    bibl = read_bbl(SRC / "combined_dissertation.bbl")
    bib = {b["key"]: b for b in bibl}
    print(f"{len(labels)} labels, {len(toc)} toc lines, {len(bibl)} references")

    (WORK / "web.tex").write_text(web_master(bib))
    print("tikz pictures:", TIKZ["n"])
    subprocess.run(["pandoc", str(WORK / "web.tex"), "-f", "latex", "-t", "html5", "--mathjax", "--section-divs",
                    "--top-level-division=chapter", "--reference-location=section", "-o", str(WORK / "all.html")],
                   check=True, cwd=SRC)
    soup = BeautifulSoup((WORK / "all.html").read_text(), "html.parser")
    media()

    stats = {"eq": 0, "thm": 0, "thm_renum": 0, "heads": 0, "refs": 0, "refs_missing": 0}
    fix_math(soup, labels, stats)
    fix_headings(soup, toc, stats)
    fix_theorems(soup, labels, stats)
    fix_figures(soup, labels)
    fix_tables(soup)

    # ---- split into pages
    sections = soup.find_all("section", recursive=False)
    pages, used = [], set()
    for slug, title, starts in PAGES:
        secs = [s for s in sections if s.find("h1") and any(s.find("h1").get_text(" ", strip=True).replace("Chapter", "").lstrip(" 0123456789A").startswith(x) or
                                                              s.find("h1").get_text(" ", strip=True).find(x) >= 0 for x in starts) and id(s) not in used]
        for s in secs:
            used.add(id(s))
        pages.append({"slug": slug, "title": title, "secs": secs})
    left = [s.find("h1").get_text(" ", strip=True) for s in sections if id(s) not in used]
    if left:
        print("sections on no page:", left, file=sys.stderr)

    # ---- where every id lives
    where = {}
    for p in pages:
        for s in p["secs"]:
            for e in s.find_all(id=True):
                where.setdefault(e["id"], p["slug"])
            if s.get("id"):
                where.setdefault(s["id"], p["slug"])

    # ---- cross-references: LaTeX's numbers, links across pages
    for p in pages:
        for s in p["secs"]:
            for a in s.find_all("a", attrs={"data-reference": True}):
                key, kind = a["data-reference"], a.get("data-reference-type", "ref")
                num = labels.get(key, {}).get("num")
                if not num:
                    stats["refs_missing"] += 1
                    print("  no number for", key, file=sys.stderr)
                    continue
                a.string = f"({num})" if kind == "eqref" else num
                tgt = where.get(key)
                a["href"] = (f"#{key}" if tgt == p["slug"] else f"{BASE}/{tgt}/#{key}") if tgt else a["href"]
                if not tgt:
                    print("  no anchor for", key, file=sys.stderr)
                for k in ("data-reference", "data-reference-type"):
                    a.attrs.pop(k, None)
                a["class"] = ["xref"]
                stats["refs"] += 1
    print(stats)

    # ---- write
    frag = SITE / "data" / "dissertation"
    if frag.exists():
        shutil.rmtree(frag)
    frag.mkdir(parents=True)
    if CONTENT.exists():
        for f in CONTENT.glob("*.md"):         # only the pages this script wrote; the illustrated stories stay
            if f.name != "_index.md" and 'template = "thesis.html"' in f.read_text():
                f.unlink()
    CONTENT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for w, p in enumerate(pages, 1):
        h1s = [s.find("h1") for s in p["secs"]]
        first = h1s[0]
        num = first.get("data-num", "") if first else ""
        name = re.sub(r"^(Chapter|Appendix)\s+\S+\s*", "", first.get_text(" ", strip=True)) if first else p["title"]
        title = p["title"] or name
        label = (f"Chapter {num}" if num.isdigit() else f"Appendix {num}" if num else "")
        sections = []
        for s in p["secs"]:
            for h in s.find_all(["h1", "h2"]):
                sec = h.find_parent("section")
                hc = BeautifulSoup(str(h), "html.parser")
                for sn in hc.select(".secnum"):
                    sn.decompose()
                sections.append({"id": sec.get("id", "") if sec else "", "level": int(h.name[1]),
                                 "num": h.get("data-num", ""), "title": " ".join(hc.get_text().split())})
        body = "\n".join(str(s) for s in p["secs"])
        if p["slug"] in STORIES:
            body = story(body, STORIES[p["slug"]])
        (frag / f"{p['slug']}.html").write_text(body)
        manifest.append({"slug": p["slug"], "title": title, "label": label, "sections": sections, "weight": w})
        fm = ["+++", f'title = {json.dumps((label + ": " if label else "") + title)}', f"weight = {w}",
              f'path = "dissertation/{p["slug"]}"', 'template = "thesis.html"', "[extra]", "math = false",
              f'slug = "{p["slug"]}"', f'short = {json.dumps(title)}', f'label = "{label}"', "+++", ""]
        if p["slug"] in STORIES:
            fm[4] = 'template = "thesis-story.html"'
        (CONTENT / f"{p['slug']}.md").write_text("\n".join(fm))
    # the bibliography, from the .bbl, run through pandoc for its accents and emphasis
    items = []
    for b in bibl:
        body = re.sub(r"\\newblock\s*", " ", b["body"])
        body = re.sub(r"\\penalty0\s*", "", body)
        items.append(f"\\item[{b['n']}] \\label{{ref-{b['key']}}} {body}")
    btex = "\\documentclass{article}\\begin{document}\\begin{description}" + "\n".join(items) + "\\end{description}\\end{document}"
    (WORK / "bib.tex").write_text(btex)
    bhtml = subprocess.run(["pandoc", str(WORK / "bib.tex"), "-f", "latex", "-t", "html5", "--mathjax"], capture_output=True, text=True, check=True).stdout
    bs = BeautifulSoup(bhtml, "html.parser")
    out = ['<section class="level1 refs"><h1>References</h1><ol class="bib">']
    for dt, b in zip(bs.find_all("dt"), bibl):
        dd = dt.find_next_sibling("dd")
        inner = "".join(str(c) for c in dd.contents) if dd else ""
        inner = re.sub(r"</?p>", "", inner).strip()
        out.append(f'<li id="ref-{b["key"]}" value="{b["n"]}"><span class="bn">[{b["n"]}]</span> {inner}</li>')
    out.append("</ol></section>")
    (frag / "references.html").write_text("\n".join(out))
    manifest.append({"slug": "references", "title": "References", "label": "", "sections": [], "weight": len(pages) + 1})
    (CONTENT / "references.md").write_text("\n".join(["+++", 'title = "References"', f"weight = {len(pages) + 1}",
                                                     'path = "dissertation/references"', 'template = "thesis.html"', "[extra]",
                                                     "math = false", 'slug = "references"', 'short = "References"', 'label = ""', "+++", ""]))
    (frag / "manifest.json").write_text(json.dumps(manifest, indent=1))
    subprocess.run(["node", str(Path(__file__).parent / "prerender.js")], check=True)
    import time
    STAMP.write_text(json.dumps({"tex": sha, "built": time.time(), "source": str(SRC)}) + "\n")
    # figures re-rendered for the web replace their print conversions (tools/dissertation/webfigs.py)
    subprocess.run([sys.executable, str(Path(__file__).parent / "webfigs.py")], check=True)
    # what rests on what, for the "used in" lines, assumption tracing, the map and the paths
    subprocess.run([sys.executable, str(Path(__file__).parent / "deps.py")], check=True)
    pdf = SRC / "combined_dissertation.pdf"
    if pdf.exists():
        shutil.copy(pdf, STATIC / "babichenko-dissertation.pdf")
    print("pages:", [(m["slug"], m["label"], m["title"][:40], len(m["sections"])) for m in manifest])


if __name__ == "__main__":
    main()
    import search_index                   # the search box's index, from the fragments just written
    search_index.main()
