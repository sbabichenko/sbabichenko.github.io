"""Web versions of the dissertation's TikZ diagrams (static/dissertation/media/tikz-N.svg, cairo SVGs of the print
figures): a light and a dark SVG each, on a transparent ground, in the web figures' inks, so they sit on the page's
paper like the re-rendered plots (webstyle.py) and need no blend or inversion.

    python3 tools/dissertation/webstyle/tikz_web.py        (render_all.sh runs it)

writes static/dissertation/media/web/tikz-N.svg and tikz-N-dark.svg. Only colours change, never the drawing:

  black and greys          the theme's ink ramp (as webstyle._ink_for)
  colours                  kept in light; lifted for the dark ground in dark (as webstyle._brighten)
  pure blue (link text)    the page's link accent
  white text               the paper colour (a numeral set on an ink-filled node)
  light fills (node and    painted as a translucent tint of their own hue, which over white gives back the print
  label boxes, shading)    colour exactly; and whatever the diagram drew before them is cut away under them (an SVG
                           mask), so a line running under a label box stops at the box, as it did under the opaque
                           white, while the box itself shows the paper. A white box is only the cut.
"""
import colorsys
import copy
import re
from pathlib import Path

from lxml import etree

ROOT = Path(__file__).resolve().parents[3]
MEDIA = ROOT / "static" / "dissertation" / "media"
OUT = MEDIA / "web"
SVG = "http://www.w3.org/2000/svg"
XLINK = "{http://www.w3.org/1999/xlink}href"

# the web figures' inks and paper (webstyle.py INK, PAGE) and the page's link accent (thesis.css --accent)
INK = {"light": ("#1d1d22", "#55555f", "#d9d7cf"), "dark": ("#e4e4ea", "#a4a4b0", "#44444c")}
PAGE = {"light": "#fcf9f2", "dark": "#27282b"}
ACCENT = {"light": "#1f3fd0", "dark": "#8fb0ff"}
TINT_DARK = 1.0          # a tint's opacity in dark, relative to light: an ink tint lightens the dark ground as much
                         # as it darkens the light one
RGB = re.compile(r"rgb\(\s*([\d.]+)%,\s*([\d.]+)%,\s*([\d.]+)%\s*\)")


def _hex(c):
    return c.lstrip("#")


def _rgb(h):
    h = _hex(h)
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


def _css(c):
    return "#%02x%02x%02x" % tuple(round(max(0, min(1, v)) * 255) for v in c)


def _parse(v):
    m = RGB.fullmatch(v.strip()) if v else None
    return tuple(float(x) / 100 for x in m.groups()) if m else None


def _is_grey(c):
    return colorsys.rgb_to_hls(*c)[2] < 0.18


def _ink_for(c, theme):
    """a grey of the print figure on the theme's ink ramp (webstyle._ink_for)"""
    l = sum(c) / 3
    fg, mid, rule = [_rgb(x) for x in INK[theme]]
    t = min(1.0, l / 0.75)
    lo, hi = (fg, mid) if t < 0.5 else (mid, rule)
    u = t * 2 if t < 0.5 else (t - 0.5) * 2
    return tuple(lo[i] + (hi[i] - lo[i]) * u for i in range(3))


def _brighten(c):
    """a colour lifted to read on the dark ground, hue kept. Lifted further than webstyle._brighten does for the
    plots' thick curves: here it is thin text and hairlines, which at the plots' lightness read dim and harsh"""
    h, l, s = colorsys.rgb_to_hls(*c)
    return colorsys.hls_to_rgb(h, max(l, 0.68), min(s, 0.85))


def _ink(c, theme):
    """the colour of a line or a glyph"""
    if c == (0.0, 0.0, 1.0):                       # hyperref's link blue
        return ACCENT[theme]
    if _is_grey(c):
        return _css(_ink_for(c, theme))
    return _css(c if theme == "light" else _brighten(c))


def _is_tint(c):
    """a fill light enough to be a box, a node's shading or white: not ink"""
    return min(c) > 0.7


def _tint(c, theme):
    """(colour, opacity) of a light fill: over white, colour at that opacity gives back the print colour exactly"""
    a = 1 - min(c)
    if a < 0.004:
        return None
    f = tuple((c[i] - (1 - a)) / a for i in range(3))
    if _is_grey(c):
        return _css(_ink_for((0, 0, 0), theme)), a * (1 if theme == "light" else TINT_DARK)
    return (_css(f), a) if theme == "light" else (_css(_brighten(f)), min(1.0, a * TINT_DARK * 1.3))


def _shapes(item):
    """the drawn elements of a top-level item (cairo writes a flat list: paths, and groups of glyphs or a clip)"""
    return [item] if etree.QName(item).localname != "g" else [e for e in item.iter() if e is not item]


def convert(src, theme):
    tree = etree.parse(str(src))
    root = tree.getroot()
    vb = [float(x) for x in root.get("viewBox").split()]
    defs = root.find(f"{{{SVG}}}defs")
    items = [c for c in root if c is not defs and isinstance(c.tag, str)]
    for c in items:
        root.remove(c)
    out, masks = [], 0
    for item in items:
        # a glyph group: its fill is text colour
        glyphs = etree.QName(item).localname == "g" and item.find(f"{{{SVG}}}use") is not None
        knock = None
        for e in [item] + _shapes(item):
            s = _parse(e.get("stroke"))
            if s is not None:
                e.set("stroke", _ink(s, theme))
            f = _parse(e.get("fill"))
            if f is None:
                continue
            if glyphs and min(f) > 0.93:
                e.set("fill", PAGE[theme])
            elif not glyphs and _is_tint(f) and e.get("d"):
                knock = e
            else:
                e.set("fill", _ink(f, theme))
        if knock is not None:
            # cut everything drawn so far away under the fill
            masks += 1
            mid = f"ko{masks}"
            m = etree.SubElement(defs, f"{{{SVG}}}mask", id=mid, maskUnits="userSpaceOnUse",
                                 x=str(vb[0] - 10), y=str(vb[1] - 10), width=str(vb[2] + 20), height=str(vb[3] + 20))
            etree.SubElement(m, f"{{{SVG}}}rect", x=str(vb[0] - 10), y=str(vb[1] - 10), width=str(vb[2] + 20),
                             height=str(vb[3] + 20), fill="white")
            cut = copy.deepcopy(item)
            for e in [cut] + _shapes(cut):
                if e.get("fill") not in (None, "none"):
                    e.set("fill", "black")
                    e.set("fill-opacity", "1")
                for a in ("stroke", "stroke-width", "stroke-opacity"):
                    e.attrib.pop(a, None)
            if cut.get("clip-path"):
                cut.attrib.pop("clip-path")        # clip paths live in defs, outside the mask: re-apply by wrapping
                g = etree.SubElement(m, f"{{{SVG}}}g", {"clip-path": item.get("clip-path")})
                g.append(cut)
            else:
                m.append(cut)
            if out:
                g = etree.Element(f"{{{SVG}}}g", mask=f"url(#{mid})")
                for o in out:
                    g.append(o)
                out = [g]
            # then the fill itself as a tint, and its outline (if the same element drew one) on top
            f = _parse(knock.get("fill"))
            t = _tint(f, theme)
            if knock.get("stroke") not in (None, "none"):
                line = copy.deepcopy(knock)
                line.set("fill", "none")
                line.attrib.pop("fill-opacity", None)
                line.attrib.pop("fill-rule", None)
                for a in ("stroke", "stroke-width", "stroke-opacity", "stroke-linecap", "stroke-linejoin",
                          "stroke-miterlimit", "stroke-dasharray", "stroke-dashoffset"):
                    knock.attrib.pop(a, None)
            else:
                line = None
            if t is None:
                knock.getparent().remove(knock) if knock is not item else None
                if knock is item:
                    item = None
            else:
                knock.set("fill", t[0])
                knock.set("fill-opacity", "%.4f" % t[1])
            if item is not None:
                out.append(item)
            if line is not None:
                if knock is item or item is None:
                    out.append(line)
                else:                              # the outline belongs inside the same clip group
                    knock.addnext(line)
            continue
        out.append(item)
    for o in out:
        root.append(o)
    return etree.tostring(tree, xml_declaration=True, encoding="UTF-8")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    n = 0
    for src in sorted(MEDIA.glob("tikz-*.svg")):
        for theme, suffix in (("light", ""), ("dark", "-dark")):
            (OUT / f"{src.stem}{suffix}.svg").write_bytes(convert(src, theme))
        n += 1
    print(f"{n} TikZ diagrams written in light and dark")
