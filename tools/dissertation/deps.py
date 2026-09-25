"""What rests on what: the dissertation's results and the references between them.

Reads the chapter pages (data/dissertation/*.html, in the order of manifest.json) and writes
static/dissertation/deps.json, which the pages use for "used in" lines, assumption tracing, the map
(/dissertation/map/) and the "just what you need" paths (/dissertation/path/):

    nodes  every numbered result (assumption, definition, lemma, proposition, theorem, corollary, conjecture),
           with its label ("Theorem 4.6"), its name if it has one, its page and a plain-text opening
    edges  [a, b]: a's statement or proof refers to b, directly or through an equation displayed inside b
           (b is also an equation of the running text when that is what a proof cites: those equations are
           nodes too, kind "equation", with the words that introduce them as their text)

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
    nodes, owner, refs = [], {}, {}
    for m in manifest:
        slug = m["slug"]
        f = PAGES / f"{slug}.html"
        if not f.exists():
            continue
        soup = BeautifulSoup(f.read_text(), "html.parser")
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
            # its proof, when one follows it
            proof = t.find_next_sibling()
            while proof is not None and proof.name is None:
                proof = proof.find_next_sibling()
            parts = [t] + ([proof] if proof is not None and "proof" in proof.get("class", []) else [])
            refs[nid] = [(target(a["href"], slug), "proof" if p is not t else "statement")
                         for p in parts for a in p.select("a.xref[href]")]
    ids = {n["id"] for n in nodes}
    edges, free = set(), {}
    for nid, rs in refs.items():
        for (page, frag), where in rs:
            dep = frag if frag in ids else owner.get(frag)
            if dep is None and frag.startswith("eq:"):
                dep = frag
                free[frag] = page
            if dep and dep != nid:
                edges.add((nid, dep))
    # the running-text equations the results cite, as nodes of their own
    soups = {}
    order = {m["slug"]: i for i, m in enumerate(manifest)}
    for eid, page in sorted(free.items(), key=lambda kv: order.get(kv[1], 99)):
        if page not in soups:
            soups[page] = BeautifulSoup((PAGES / f"{page}.html").read_text(), "html.parser")
        eq = soups[page].find(id=eid)
        if eq is None:
            edges = {e for e in edges if e[1] != eid}
            continue
        tag = eq.select_one(".tag")
        num = " ".join(tag.get_text("", strip=True).split()) if tag else ""
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
    OUT.write_text(json.dumps({"nodes": nodes, "edges": sorted(edges)}, ensure_ascii=False, separators=(",", ":")))
    k = sum(n["kind"] == "equation" for n in nodes)
    print(f"deps: {len(nodes) - k} results and {k} cited equations, {len(edges)} references -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
