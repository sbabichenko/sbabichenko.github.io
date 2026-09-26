"""Alt text for the dissertation's figures, written to data/dissertation/figure_alts.json (figure id -> alt), which
webfigs.py applies to every figure image on each build.  The alt is the first sentence of the caption as the site shows
it, with the caption's math written out as readable text from its tex source (KaTeX leaves no source in the pages).

    python3 tools/dissertation/figalt.py /path/to/dissertation      # rerun when captions change
"""
import sys
from pathlib import Path
import re, json, glob, html
from bs4 import BeautifulSoup

SUP = str.maketrans("0123456789+-=()ni*", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿⁱ*")
SUB = str.maketrans("0123456789+-=()aeijkotuvx", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑᵢⱼₖₒₜᵤᵥₓ")
GREEK = {r"\alpha": "α", r"\beta": "β", r"\gamma": "γ", r"\delta": "δ", r"\epsilon": "ε", r"\varepsilon": "ε", r"\zeta": "ζ",
         r"\eta": "η", r"\theta": "θ", r"\kappa": "κ", r"\lambda": "λ", r"\mu": "μ", r"\nu": "ν", r"\xi": "ξ", r"\pi": "π",
         r"\rho": "ρ", r"\sigma": "σ", r"\tau": "τ", r"\phi": "φ", r"\varphi": "φ", r"\chi": "χ", r"\psi": "ψ", r"\omega": "ω",
         r"\Gamma": "Γ", r"\Delta": "Δ", r"\Lambda": "Λ", r"\Sigma": "Σ", r"\Phi": "Φ", r"\Pi": "Π", r"\Omega": "Ω",
         r"\le": "≤", r"\leq": "≤", r"\ge": "≥", r"\geq": "≥", r"\pm": "±", r"\times": "×", r"\cdot": "·", r"\to": "→",
         r"\infty": "∞", r"\in": "∈", r"\approx": "≈", r"\neq": "≠", r"\ell": "ℓ", r"\bullet": "•", r"\dots": "…", r"\ldots": "…",
         r"\partial": "∂", r"\sim": "~", r"\mid": "|", r"\,": " ", r"\;": " ", r"\!": "", r"\ ": " ", r"\quad": " "}

def group(s, i):
    """the argument at s[i]: a braced group or one token; returns (text, next index)"""
    while i < len(s) and s[i] == " ": i += 1
    if i >= len(s): return "", i
    if s[i] == "{":
        d = 0
        for j in range(i, len(s)):
            if s[j] == "{": d += 1
            elif s[j] == "}":
                d -= 1
                if d == 0: return s[i + 1:j], j + 1
        return s[i + 1:], len(s)
    if s[i] == "\\":
        m = re.match(r"\\[A-Za-z]+|\\.", s[i:])
        return m.group(0), i + len(m.group(0))
    return s[i], i + 1

def script(t, table):
    t = tex(t)
    return t.translate(table) if all(c in table_keys(table) for c in t) else ("^" if table is SUP else "_") + (t if len(t) == 1 else "(" + t + ")")

def table_keys(table): return {chr(k) for k in table}

def tex(s):
    out, i = [], 0
    while i < len(s):
        c = s[i]
        if c == "\\":
            m = re.match(r"\\[A-Za-z]+|\\.", s[i:]); cmd = m.group(0); i += len(cmd)
            if cmd in (r"\bar", r"\overline", r"\hat", r"\widehat", r"\tilde", r"\widetilde", r"\dot"):
                a, i = group(s, i); mark = {"\\bar": "\u0304", "\\overline": "\u0304", "\\hat": "\u0302", "\\widehat": "\u0302",
                                            "\\tilde": "\u0303", "\\widetilde": "\u0303", "\\dot": "\u0307"}[cmd]
                out.append(tex(a) + mark)
            elif cmd in (r"\mathcal", r"\mathrm", r"\mathbf", r"\mathbb", r"\mathsf", r"\text", r"\textrm", r"\operatorname", r"\boldsymbol", r"\mathit"):
                a, i = group(s, i); out.append(tex(a))
            elif cmd in (r"\frac", r"\tfrac", r"\dfrac"):
                a, i = group(s, i); b, i = group(s, i); out.append(f"{tex(a)}/{tex(b)}")
            elif cmd == r"\sqrt":
                a, i = group(s, i); out.append(f"√{tex(a)}")
            elif cmd in (r"\left", r"\right", r"\big", r"\Big", r"\bigl", r"\bigr", r"\displaystyle"): pass
            elif cmd in (r"\{", r"\}"): out.append(cmd[1])
            elif cmd == r"\ref" or cmd == r"\eqref":
                a, i = group(s, i); out.append("")
            else: out.append(GREEK.get(cmd, ""))
        elif c in "^_":
            a, i = group(s, i + 1); out.append(script(a, SUP if c == "^" else SUB))
        elif c in "{}": i += 1
        elif c == "~": out.append(" "); i += 1
        else: out.append(c); i += 1
    return re.sub(r"\s+", " ", "".join(out)).strip()

MATH = re.compile(r"\$\$(.+?)\$\$|\$(.+?)\$|\\\((.+?)\\\)|\\\[(.+?)\\\]", re.S)
def tex_math(cap): return [next(g for g in m.groups() if g is not None) for m in MATH.finditer(cap)]
def tex_plain(cap):
    s = MATH.sub(lambda m: tex(next(g for g in m.groups() if g is not None)), cap)
    s = re.sub(r"~\\ref\{[^}]*\}|\\ref\{[^}]*\}|\\eqref\{[^}]*\}", "", s)
    s = re.sub(r"\\(emph|textit|textbf)\{([^}]*)\}", r"\2", s).replace("~", " ").replace("--", "–").replace("``", "“").replace("''", "”")
    return re.sub(r"\s+", " ", s).strip()

def first_sentence(t):
    m = re.search(r"(.+?[.!?])(\s+[A-Z(]|$)", t)
    return (m.group(1) if m else t).strip()

def alt_for(fig, texcap):
    cap = fig.find("figcaption")
    if cap is None: return None
    cap = BeautifulSoup(str(cap), "html.parser").figcaption
    num = cap.find(class_="fignum")
    if num: num.decompose()
    maths = cap.select("span.math")
    tm = tex_math(texcap) if texcap else []
    if texcap is not None and len(maths) == len(tm):
        for sp, t in zip(maths, tm): sp.replace_with(tex(t))
        text = re.sub(r"\s+", " ", cap.get_text(" ")).strip()
        text = re.sub(r"\s+([,.;:)’])", r"\1", text).replace("( ", "(")
    elif texcap is not None:
        text = tex_plain(texcap)
    else:
        for sp in maths: sp.replace_with("")
        text = re.sub(r"\s+", " ", cap.get_text(" ")).strip()
    return first_sentence(text)


def tex_captions(src):
    """figure label -> caption source, from every chapter"""
    out = {}
    for f in list(Path(src, "chapters").glob("*.tex")) + [Path(src, "intro.tex")]:
        if not f.exists(): continue
        s = f.read_text()
        for m in re.finditer(r"\\begin\{figure\*?\}(.*?)\\end\{figure\*?\}", s, re.S):
            body = m.group(1); lab = re.search(r"\\label\{(fig:[^}]*)\}", body); c = body.find("\\caption")
            if not lab or c < 0: continue
            cap, _ = group(body, body.find("{", c))
            out[lab.group(1)] = cap
    return out


if __name__ == "__main__":
    site = Path(__file__).resolve().parents[2]
    caps = tex_captions(sys.argv[1] if len(sys.argv) > 1 else Path.home() / "dissertation")
    alts = {}
    for f in sorted((site / "data" / "dissertation").glob("*.html")):
        for fig in BeautifulSoup(f.read_text(), "html.parser").find_all("figure"):
            if fig.get("id"):
                a = alt_for(fig, caps.get(fig["id"]))
                if a: alts[fig["id"]] = a
    (site / "data" / "dissertation" / "figure_alts.json").write_text(json.dumps(alts, ensure_ascii=False, indent=1) + "\n")
    print(f"{len(alts)} figure alts")
