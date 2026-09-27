"""What rests on what: the dissertation's results and the references between them.

Reads the chapter pages (data/dissertation/*.html, in the order of manifest.json) and writes
static/dissertation/deps.json, which the pages use for "used in" lines, assumption tracing, the map
(/dissertation/map/) and the "just what you need" paths (/dissertation/path/):

    nodes  every numbered result (assumption, definition, lemma, proposition, theorem, corollary, conjecture),
           with its label ("Theorem 4.6"), its name if it has one, its page and a plain-text opening
    edges  [a, b]: a's statement or proof refers to b, directly or through an equation displayed inside b
           (b is also an equation of the running text when that is what a proof cites: those equations are
           nodes too, kind "equation", with the words that introduce them as their text)

A result's proof is the block that follows it ("Proof." as a div or a paragraph), or a block headed "Proof of
<that result>" wherever it stands. A proof that points elsewhere ("Proof. Appendix 1.B.2: ...") also takes in the
references of the section it names (with its unnumbered paragraphs, not its numbered subsections).

Remarks are left out: they comment on results rather than being used by them. Nothing is inferred beyond
the references the text itself makes. build.py runs this after writing the pages; it can also be run alone:

    python3 tools/dissertation/deps.py
"""
import json
import re
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[2]
PAGES = ROOT / "data" / "dissertation"
OUT = ROOT / "static" / "dissertation" / "deps.json"
KINDS = ("assumption", "definition", "lemma", "proposition", "theorem", "corollary", "conjecture")


def target(href, slug):
    """(page, id) a cross-reference points at"""
    page, _, frag = href.partition("#")
    m = re.match(r"/dissertation/([\w-]+)/?$", page)
    return (m.group(1) if m else slug), frag


def paren(s):
    """the balanced (...) group at the start of s, without its outer parentheses"""
    s = s.lstrip()
    if not s.startswith("("):
        return ""
    depth = 0
    for i, ch in enumerate(s):
        depth += ch == "("
        depth -= ch == ")"
        if depth == 0:
            return " ".join(s[1:i].split())
    return ""


def prose(el, stop=None):
    """an element's words, each piece of maths shown as "…", up to `stop` if that lies inside it"""
    out, last_math = [], None
    for t in el.find_all(string=True):
        if stop is not None and (stop in t.parents or stop in t.find_all_previous()):
            break
        m = t.find_parent(class_="math")
        if m is None:
            out.append(str(t))
        elif m is not last_math:
            out.append(" … ")
            last_math = m
    return " ".join("".join(out).split())


TAG = re.compile(r"\([\dA-Z]+(\.[\dA-Z]+)*\)")


def unannotated(soup):
    """the page without the TeX source MathML carries alongside each formula (it is not what the reader sees)"""
    for a in soup.find_all("annotation"):
        a.decompose()
    return soup


def display_of(eq):
    """the typeset display an equation label belongs to (a label on one line of a block is an empty anchor)"""
    if "display" in eq.get("class", []):
        return eq
    return eq.find_parent(class_="display") or eq.find_next(class_="display") or eq


def tag_of(eq):
    """a displayed equation's first printed number, "(1.3.2)", or "" (KaTeX's HTML or MathML)"""
    eq = display_of(eq)
    tag = eq.select_one(".tag")
    if tag is not None:
        m = TAG.search(tag.get_text("", strip=True))
        return m.group(0) if m else ""
    for mt in eq.find_all("mtext"):
        t = mt.get_text("", strip=True)
        if TAG.fullmatch(t) and mt.parent.name == "mtd" and len(mt.parent.find_all(True)) == 1:
            return t
    return ""




def is_proof(el):
    """a proof block: a div.proof, or a paragraph that opens with "Proof." """
    if el is None or el.name is None:
        return False
    if "proof" in el.get("class", []):
        return True
    em = el.find(True) if el.name == "p" else None
    return em is not None and em.name == "em" and em.get_text(strip=True).startswith("Proof")


def proof_of(el):
    """the xref in a proof headed "Proof of <result>.", or None"""
    head = el.find("em")
    if head is None or not head.get_text(" ", strip=True).startswith("Proof of"):
        return None
    return head.find("a", class_="xref", href=True)


def pointed(proof):
    """the sections a proof sends the reader to ("Appendix 1.B.2"): [(href, where the link is)]"""
    out = []
    for a in proof.select("a.xref[href]"):
        before = a.find_previous(string=True)
        if before is not None and re.search(r"Appendix\s*$", str(before)):
            out.append(a["href"])
    return out


def section_refs(sec):
    """the cross-references of a section and its unnumbered paragraphs, not of its numbered subsections"""
    def numbered(s):
        h = s.find(re.compile(r"^h[1-6]$"))
        return h is not None and re.match(r"[\dA-Z]+(\.[\dA-Z]+)+\s", h.get_text(" ", strip=True) + " ")
    return [a for a in sec.select("a.xref[href]")
            if all(not numbered(s) for s in a.find_parents("section") if s is not sec and sec in s.find_parents("section"))]


def lead_in(eq):
    """the prose just before a displayed equation, its maths left out"""
    box = eq.find_parent(["p", "li"])
    t = prose(box, stop=eq) if box is not None else ""
    if len(t) < 60:                                   # a display that opens its paragraph: add the one before
        prev = (box or eq).find_previous_sibling(["p", "li"])
        if prev is not None:
            t = (prose(prev) + " " + t).strip()
    t = t.rstrip(" ,:;")
    return ("…" + t[-160:].lstrip()) if len(t) > 160 else t


def main():
    manifest = json.loads((PAGES / "manifest.json").read_text())
    nodes, owner, refs, own_proof, pointers = [], {}, {}, {}, {}
    soups = {}
    for m in manifest:
        slug = m["slug"]
        f = PAGES / f"{slug}.html"
        if not f.exists():
            continue
        soup = soups[slug] = unannotated(BeautifulSoup(f.read_text(), "html.parser"))
        for t in soup.select(".thm"):
            kind = next((k for k in KINDS if f"thm-{k}" in t.get("class", [])), None)
            if not kind or not t.get("id"):
                continue
            head = t.find("strong")
            label = head.get_text(" ", strip=True) if head else kind.title()
            # the name in parentheses right after the label, when the result has one
            text = " ".join(t.get_text(" ", strip=True).split())
            text = text[len(label):].lstrip(" .")
            name = paren(text)
            if name:
                text = text[len(name) + 2:].lstrip(" .")
            nid = t["id"]
            nodes.append({"id": nid, "kind": kind, "label": label, "name": name, "page": slug,
                          "chapter": m.get("label") or m.get("title"), "text": text[:220]})
            for eq in t.select("[id^='eq:']"):
                owner[eq["id"]] = nid
            # its proof, when one follows it (a "Proof of <another result>" block belongs to that one)
            proof = t.find_next_sibling()
            while proof is not None and proof.name is None:
                proof = proof.find_next_sibling()
            parts = [t] + ([proof] if is_proof(proof) and proof_of(proof) is None else [])
            refs.setdefault(nid, []).extend((target(a["href"], slug), "proof" if p is not t else "statement")
                                            for p in parts for a in p.select("a.xref[href]"))
            for p in parts[1:]:
                own_proof.setdefault(nid, set()).update(e["id"] for e in p.select("[id^='eq:']"))
                pointers.setdefault(nid, []).extend(target(h, slug) for h in pointed(p))
        # proofs set apart from their results: "Proof of Theorem 4.6." wherever it stands
        for p in soup.find_all(["div", "p"]):
            if not is_proof(p) or (p.name == "p" and p.find_parent("div", class_="proof")):
                continue
            a = proof_of(p)
            if a is None:
                continue
            nid = target(a["href"], slug)[1]
            refs.setdefault(nid, []).extend((target(x["href"], slug), "proof") for x in p.select("a.xref[href]") if x is not a)
            own_proof.setdefault(nid, set()).update(e["id"] for e in p.select("[id^='eq:']"))
            pointers.setdefault(nid, []).extend(target(h, slug) for h in pointed(p))
    # a proof that sends the reader to a section takes in that section's references
    for nid, secs in pointers.items():
        for page, frag in secs:
            if page not in soups:
                continue
            sec = soups[page].find(id=frag)
            if sec is None or sec.name != "section":
                continue
            refs.setdefault(nid, []).extend((target(a["href"], page), "proof") for a in section_refs(sec))
    ids = {n["id"] for n in nodes}
    edges, free = set(), {}
    for nid, rs in refs.items():
        if nid not in ids:
            continue
        for (page, frag), where in rs:
            if frag in own_proof.get(nid, ()):            # an equation displayed in its own proof
                continue
            dep = frag if frag in ids else owner.get(frag)
            if dep is None and frag.startswith("eq:"):
                dep = frag
                free[frag] = page
            if dep and dep != nid:
                edges.add((nid, dep))
    # how often each equation is cited anywhere in the text, results or not: the ones the dissertation leans on
    cites, where = {}, {}
    for m in manifest:
        f = PAGES / f"{m['slug']}.html"
        if not f.exists():
            continue
        for a in soups[m["slug"]].select('a.xref[href*="#eq:"]'):
            eid = a["href"].split("#", 1)[1]
            cites[eid] = cites.get(eid, 0) + 1
            t = " ".join(a.get_text("", strip=True).split())
            if re.fullmatch(r"\(?[\dA-Z]+(\.[\dA-Z]+)*\)?", t):
                where.setdefault(eid, {}).setdefault(t.strip("()"), 0)
                where[eid][t.strip("()")] += 1
    # the number the citing links print is the equation's own, even inside a block with several numbers
    cited_as = {e: max(w, key=w.get) for e, w in where.items() if w}
    # the running-text equations the results cite, as nodes of their own
    order = {m["slug"]: i for i, m in enumerate(manifest)}
    for eid, page in sorted(free.items(), key=lambda kv: order.get(kv[1], 99)):
        eq = soups[page].find(id=eid) if page in soups else None
        if eq is None:
            edges = {e for e in edges if e[1] != eid}
            continue
        num = f"({cited_as[eid]})" if eid in cited_as else tag_of(eq)
        chap = next((m.get("label") or m.get("title") for m in manifest if m["slug"] == page), "")
        nodes.append({"id": eid, "kind": "equation", "label": f"Equation {num}".strip(), "name": "",
                      "page": page, "chapter": chap, "text": lead_in(eq)})
    # document order: pages in the manifest's order, then position on the page
    pos = {}
    for page in (m["slug"] for m in manifest):
        f = PAGES / f"{page}.html"
        if not f.exists():
            continue
        html = f.read_text()
        for n in nodes:
            if n["page"] == page:
                pos[n["id"]] = (order[page], html.find(f'id="{n["id"]}"'))
    nodes.sort(key=lambda n: pos.get(n["id"], (99, 0)))
    top = sorted((e for e in cites if cites[e] >= 3), key=lambda e: -cites[e])
    lean = []
    for eid in top:
        n = next((x for x in nodes if x["id"] == eid), None)
        eq = None
        for m in manifest:                             # the equation on its page (also when no result cites it)
            if m["slug"] in soups and soups[m["slug"]].find(id=eid) is not None:
                eq = soups[m["slug"]].find(id=eid)
                if n is None:
                    n = {"id": eid, "label": f"Equation {tag_of(eq)}".strip(),
                         "page": m["slug"], "chapter": m.get("label") or m.get("title"), "text": lead_in(eq)}
                break
        if n:
            # the number the citing links print is the equation's own, even inside a block with several numbers
            if eid in cited_as:
                n = dict(n, label=f"Equation ({cited_as[eid]})")
            block = id(display_of(eq)) if eq is not None else eid
            same = next((x for x in lean if x["_block"] == block), None)
            if same is not None:                       # two numbers of one displayed block: one entry, both numbers
                same["also"].append({"id": eid, "label": n["label"], "cited": cites[eid]})
                continue
            lean.append({"_block": block, "id": eid, "label": n["label"], "page": n["page"], "chapter": n["chapter"],
                         "text": n["text"], "cited": cites[eid], "also": [],
                         "block": (display_of(eq).get("id") or eid) if eq is not None else eid})
    for x in lean:
        del x["_block"]
        if not x["also"]:
            del x["also"]
    OUT.write_text(json.dumps({"nodes": nodes, "edges": sorted(edges), "cites": {e: c for e, c in cites.items() if c >= 3},
                               "lean": lean[:12]}, ensure_ascii=False, separators=(",", ":")))
    k = sum(n["kind"] == "equation" for n in nodes)
    print(f"deps: {len(nodes) - k} results and {k} cited equations, {len(edges)} references -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
