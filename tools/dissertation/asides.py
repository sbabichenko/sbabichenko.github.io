#!/usr/bin/env python3
"""Notes added to the web dissertation after it was written, each folded shut until opened.  Each file in asides/
names the element it follows and the page it lives on (two comment lines), and holds a <details> block with TeX in
\\( \\) and \\[ \\].  This inserts every aside into its page's fragment in data/dissertation (replacing an earlier copy
of itself, so it can be rerun), typesetting its math as the rest of the page is (KaTeX MathML, asides_math.js).
The PDF is untouched.  build.py runs it after prerender.js; alone: python3 tools/dissertation/asides.py"""
import json, re, subprocess
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE.parents[1] / "data" / "dissertation"


def typeset(html):
    tex = re.findall(r"\\\((.+?)\\\)|\\\[(.+?)\\\]", html, flags=re.S)
    items = [{"tex": a or b, "display": bool(b)} for a, b in tex]
    out = json.loads(subprocess.run(["node", str(HERE / "asides_math.js")], input=json.dumps(items),
                                    capture_output=True, text=True, check=True).stdout)
    it = iter(out)
    return re.sub(r"\\\((.+?)\\\)|\\\[(.+?)\\\]", lambda m: next(it), html, flags=re.S)


def main():
    for f in sorted((HERE / "asides").glob("*.html")):
        src = f.read_text()
        after = re.search(r"<!--\s*after:\s*(\S+)\s*-->", src).group(1)
        page = re.search(r"<!--\s*page:\s*(\S+)\s*-->", src).group(1)
        block = typeset(re.sub(r"<!--.*?-->\n?", "", src))
        path = DATA / f"{page}.html"
        html = path.read_text()
        aid = re.search(r'<details[^>]*\bid="([^"]+)"', block).group(1)
        html = re.sub(r'\n?<details[^>]*\bid="' + re.escape(aid) + r'".*?</details>', "", html, flags=re.S)   # an earlier copy
        # the end of the element it follows, by counting its tag's nesting (a string edit: the page stays byte-identical)
        m = re.search(r'<(\w+)[^>]*\bid="' + re.escape(after) + r'"', html)
        tag, depth, k = m.group(1), 0, m.start()
        for t in re.finditer(r"<(/?)" + tag + r"\b[^>]*>", html[k:]):
            depth += -1 if t.group(1) else 1
            if depth == 0:
                end = k + t.end()
                break
        html = html[:end] + "\n" + block.strip() + html[end:]
        path.write_text(html)
        new = {"id": aid}
        print(f"aside {new['id']} after {after} on {page}")


if __name__ == "__main__":
    main()
