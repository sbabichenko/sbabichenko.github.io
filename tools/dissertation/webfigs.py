"""Swap the dissertation pages' print figures for their web versions.

A figure whose script has been re-rendered for the site (static/dissertation/media/web/<name>.svg and
<name>-dark.svg, with -narrow and -narrow-dark phone layouts where those came out clean, made by running the
chapter's figure scripts through the webstyle harness) is shown as a light and a dark <picture>, in place of the .webp
converted from the print PDF. Figures without web versions (the TikZ diagrams, the hand-made ones) are left
as they are. build.py calls this last; it can also be run on its own:

    python3 tools/dissertation/webfigs.py
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / "static" / "dissertation" / "media" / "web"
PAGES = ROOT / "data" / "dissertation"
URL = "/dissertation/media/web/"
IMG = re.compile(r'<img loading="lazy" src="/dissertation/media/([\w.-]+)\.webp"\s*/>')


def picture(name, theme):
    s = "" if theme == "light" else "-dark"
    # a phone layout only where one came out clean (the harness skips the rest; phones then get the wide one)
    phone = (f'<source media="(max-width: 600px)" srcset="{URL}{name}-narrow{s}.svg"/>'
             if (WEB / f"{name}-narrow{s}.svg").exists() else "")
    return f'<picture class="wf-{theme}">{phone}<img loading="lazy" src="{URL}{name}{s}.svg" alt=""/></picture>'


def rewrite(html):
    def swap(m):
        name = m.group(1)
        if not all((WEB / f"{name}{v}.svg").exists() for v in ("", "-dark")):
            return m.group(0)
        return picture(name, "light") + picture(name, "dark")
    html = IMG.sub(swap, html)
    # mark the figures that now hold web versions, so the stylesheet drops their white print box
    return re.sub(r'<figure([^>]*)>((?:(?!</figure>).)*?<picture class="wf-light">)',
                  lambda m: f'<figure{m.group(1) if "webfig" in m.group(1) else _add_class(m.group(1))}>{m.group(2)}',
                  html, flags=re.S)


def _add_class(attrs):
    if 'class="' in attrs:
        return attrs.replace('class="', 'class="webfig ', 1)
    return attrs + ' class="webfig"'


if __name__ == "__main__":
    n = 0
    for f in sorted(PAGES.glob("*.html")):
        old = f.read_text()
        new = rewrite(old)
        if new != old:
            f.write_text(new)
        n += new.count('class="wf-light"')
    print(f"{n} figures use their web versions")
