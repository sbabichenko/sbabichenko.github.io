"""Web versions of the dissertation's figures.

Run a figure script through this module (run.py does it): Figure.savefig is replaced so that, instead of the
print PDF, each figure is written as four SVGs for the site,

    <name>.svg          light, the figure's own layout, text sized for a ~740 px column
    <name>-dark.svg     the same, recoloured for the site's dark theme
    <name>-narrow.svg   phones: side-by-side panels stacked into one column, text sized for ~360 px
    <name>-narrow-dark.svg

Text is set in Newsreader (the site's face) with STIX maths, backgrounds are transparent, and nothing the
script computed is changed: only sizes, colours, fonts and the arrangement of whole panels.
"""
import os, colorsys, statistics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.figure import Figure
from matplotlib.text import Text
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.collections import Collection
from matplotlib.colors import to_rgba
from matplotlib.gridspec import GridSpec
from matplotlib.legend import Legend

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("WEBFIG_OUT", os.path.join(HERE, "out"))
FONTS = os.environ.get("WEBFIG_FONTS", os.path.join(HERE, "fonts"))     # static TTFs made by render_all.sh
for f in ("Newsreader-normal.ttf", "Newsreader-italic.ttf"):
    font_manager.fontManager.addfont(os.path.join(FONTS, f))
FONT = {"svg.hashsalt": "sbabichenko", "font.family": "serif", "font.serif": ["Newsreader 16pt", "DejaVu Serif"], "mathtext.fontset": "stix",
        "svg.fonttype": "path", "axes.unicode_minus": True}

matplotlib.rcParams.update(FONT)

# the site's inks (static/dissertation/thesis.css): body text, secondary text, rules
INK = {"light": ("#1d1d22", "#55555f", "#d9d7cf"), "dark": ("#e4e4ea", "#a4a4b0", "#44444c")}
WIDE_IN, NARROW_IN = 7.4, 4.6          # figure widths in inches; the column is ~740 px, a phone ~360 px
WIDE_PT, NARROW_PT = 10.5, 12.0         # the median text size after scaling
PAGE = {"light": "#fcf9f2", "dark": "#27282b"}   # the paper behind the figures as drawn (--paper in thesis.css, under its grain)

# per-figure adjustments for the web versions: placement and drawing only, never the data or the words
#   legend: {panel index: (loc, bbox_to_anchor, text size factor)}   move a legend off the curves the larger text
#                                                   pushes it onto, and set its text a little smaller
#   dash:   [series label, ...]                     draw a series dashed and on top, where it lies under another
#   halo:   True                                    set the annotations on the page colour where lines cross them
#   nudge:  {label text: (dx, dy)}                  move a label, in data units, off a box the larger text runs into
#   max_scale: k                                    in the wide version, enlarge the text no further than k (a diagram
#                                                   whose labels sit close to its lines)
#   repel:  True                                    part inline labels that sit on top of each other
FIGURES = {
    "fig_kb_multiasset": {"legend": {2: ("upper right", (1.0, 0.93), 0.9)}},
    "figA_transmission": {"dash": ["exogenous signal"]},
    "fig_ch5_schematic": {"halo": True, "nudge": {"order $o^{v-1}_t$": (0.0, -0.12)}, "max_scale": 1.17},
    "fig8_barD1_vs_p": {"repel": True, "halo": True},
    "fig12_costs_private_vs_pooled": {"legend": {0: ("lower left", (0.0, 0.0), 0.8), 3: ("upper center", (0.5, -0.36), 0.9)}},
    "fig13_precision_decomposition": {"legend": {0: ("center right", (1.0, 0.69), 0.72), 2: ("center right", (1.0, 0.645), 0.7)}},
}


def _is_ink(c):
    """near-black or grey, the colours that mean 'text/axis ink' rather than a data series"""
    r, g, b, a = to_rgba(c)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return a > 0 and s < 0.18 and l < 0.75


def _is_paper(c):
    r, g, b, a = to_rgba(c)
    return a > 0 and min(r, g, b) > 0.93


def _ink_for(c, theme):
    """map a grey of the print figure onto the theme's ink ramp, keeping its alpha"""
    r, g, b, a = to_rgba(c)
    l = (r + g + b) / 3
    fg, mid, rule = [to_rgba(x) for x in INK[theme]]
    t = min(1.0, l / 0.75)                          # 0 = black ... 1 = light grey
    lo, hi = (fg, mid) if t < 0.5 else (mid, rule)
    u = t * 2 if t < 0.5 else (t - 0.5) * 2
    mix = [lo[i] + (hi[i] - lo[i]) * u for i in range(3)]
    return (*mix, a)


def _brighten(c):
    """a data colour made to read on the dark ground: lift its lightness, keep its hue"""
    r, g, b, a = to_rgba(c)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    if l < 0.62:
        l = 0.62 + (l - 0.62) * 0.35
    return (*colorsys.hls_to_rgb(h, l, s), a)


def _recolor(fig, theme):
    fig.patch.set_alpha(0)
    for ax in fig.axes:
        ax.patch.set_alpha(0)
    ink = lambda c: _ink_for(c, theme)
    data = (lambda c: c) if theme == "light" else _brighten
    for o in fig.findobj():
        if isinstance(o, Text):
            c = o.get_color()
            o.set_color(ink(c) if _is_ink(c) else data(c))
            bb = o.get_bbox_patch()
            if bb is not None:
                if _is_paper(bb.get_facecolor()):
                    bb.set_facecolor("none" if theme == "dark" else bb.get_facecolor())
                if _is_ink(bb.get_edgecolor()):
                    bb.set_edgecolor(ink(bb.get_edgecolor()))
        elif isinstance(o, Line2D):
            for get, put in ((o.get_color, o.set_color), (o.get_markerfacecolor, o.set_markerfacecolor),
                             (o.get_markeredgecolor, o.set_markeredgecolor)):
                c = get()
                if isinstance(c, str) and c in ("none", "auto"):
                    continue
                put(ink(c) if _is_ink(c) else data(c))
        elif isinstance(o, Patch):
            if o is fig.patch or any(o is ax.patch for ax in fig.axes):
                continue
            fc, ec = o.get_facecolor(), o.get_edgecolor()
            if _is_paper(fc):
                o.set_facecolor((0, 0, 0, 0) if theme == "dark" else fc)
            elif fc[3] > 0:
                o.set_facecolor(ink(fc) if _is_ink(fc) else data(fc))
            if ec[3] > 0:
                o.set_edgecolor(ink(ec) if _is_ink(ec) else data(ec))
        elif isinstance(o, Collection) and o.get_array() is None:     # not a heat map: those keep their colours
            try:
                fcs, ecs = o.get_facecolor(), o.get_edgecolor()
                if len(fcs):
                    o.set_facecolor([ink(c) if _is_ink(c) else ((0, 0, 0, 0) if theme == "dark" and _is_paper(c) else data(c)) for c in fcs])
                if len(ecs):
                    o.set_edgecolor([ink(c) if _is_ink(c) else data(c) for c in ecs])
            except Exception:
                pass
    for o in fig.findobj(Text):                       # an annotation's arrow is not among the figure's children
        ap = getattr(o, "arrow_patch", None)
        if ap is not None:
            for get, put in ((ap.get_edgecolor, ap.set_edgecolor), (ap.get_facecolor, ap.set_facecolor)):
                c = get()
                if c[3] > 0:
                    put(ink(c) if _is_ink(c) else data(c))
    for ax in fig.axes:
        for s in ax.spines.values():
            s.set_edgecolor(ink(s.get_edgecolor()))
        ax.tick_params(colors=INK[theme][1], which="both")
        for lab in ax.get_xticklabels() + ax.get_yticklabels():
            lab.set_color(INK[theme][1])
        for g in ax.get_xgridlines() + ax.get_ygridlines():
            g.set_color(INK[theme][2])
        leg = ax.get_legend()
        if leg is not None:
            leg.get_frame().set_facecolor("none")
            leg.get_frame().set_edgecolor(INK[theme][2])
    for leg in fig.legends:
        leg.get_frame().set_facecolor("none")
        leg.get_frame().set_edgecolor(INK[theme][2])


def _texts(fig):
    return [t for t in fig.findobj(Text) if t.get_text().strip()]


def _scale_text(fig, target):
    fig.canvas.draw()                                   # materialise tick labels at their final sizes
    ts = _texts(fig)
    if not ts:
        return
    k = target / statistics.median(t.get_fontsize() for t in ts)
    for t in ts:
        t.set_fontsize(t.get_fontsize() * k)
        t.set_fontfamily("serif")
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            labs = axis.get_ticklabels()
            if labs:
                axis.set_tick_params(labelsize=labs[0].get_fontsize())
        ax.tick_params(length=3, width=0.6)


def _main_axes(fig):
    """the panels: axes on a gridspec that are not colorbars or insets"""
    out = []
    for ax in fig.axes:
        ss = ax.get_subplotspec()
        if ss is None or getattr(ax, "_colorbar", None) is not None or ax.get_label() == "<colorbar>":
            continue
        out.append(ax)
    return out


def _stack(fig):
    """side-by-side panels into one column, for a phone; a figure already one panel wide is left alone"""
    axes = _main_axes(fig)
    if len(axes) < 2:
        return 1, 1
    grids = {id(ax.get_subplotspec().get_gridspec()) for ax in axes}
    gs0 = axes[0].get_subplotspec().get_gridspec()
    nrows, ncols = gs0.get_geometry()
    if len(grids) != 1 or ncols < 2:
        return 1, 1
    order = sorted(axes, key=lambda a: (a.get_subplotspec().rowspan.start, a.get_subplotspec().colspan.start))
    n = len(order)
    gs = GridSpec(n, 1, figure=fig)
    for i, ax in enumerate(order):
        ax.set_subplotspec(gs[i])
        # panels that shared an axis hid their inner tick labels; stacked, each needs its own
        ax.yaxis.set_tick_params(labelleft=True)
        ax.xaxis.set_tick_params(labelbottom=True)
    return n, ncols


def _layout(fig):
    eng = fig.get_layout_engine()
    if eng is None or type(eng).__name__ == "TightLayoutEngine":
        try:
            fig.tight_layout(pad=0.4)
        except Exception:
            pass


_last_hits = []


def _undrawn(fig):
    """tick labels matplotlib keeps for ticks outside the axis limits: laid out, never drawn"""
    out = set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            lo, hi = sorted(axis.get_view_interval())
            eps = (hi - lo) * 1e-6
            for ticks in (axis.get_major_ticks(), axis.get_minor_ticks()):
                for tk in ticks:
                    loc = tk.get_loc()
                    if loc is None or loc < lo - eps or loc > hi + eps:
                        out.add(id(tk.label1)); out.add(id(tk.label2))
    return out


def _problems(fig):
    """text that collides with other text, after layout (text past the edge is fine: the tight crop keeps it)"""
    global _last_hits
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    skip = _undrawn(fig)
    boxes = []
    for t in _texts(fig):
        if id(t) in skip or not t.get_visible() or t.get_alpha() == 0:
            continue
        try:
            b = t.get_window_extent(r)
        except Exception:
            continue
        if b.width >= 1 and b.height >= 1:
            boxes.append((t, b))
    hits = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            (t1, a), (t2, b) = boxes[i], boxes[j]
            w = min(a.x1, b.x1) - max(a.x0, b.x0)
            h = min(a.y1, b.y1) - max(a.y0, b.y0)
            same = t1.get_text() == t2.get_text() and abs(a.x0 - b.x0) < 2 and abs(a.y0 - b.y0) < 2
            if w > 1.5 and h > 1.5 and not same:          # a twin axis repeats its partner's labels in place
                hits.append(f"{t1.get_text()[:18]!r}~{t2.get_text()[:18]!r}")
    # an axis label or title running into another panel
    panels = [(ax, ax.patch.get_window_extent(r)) for ax in fig.axes if ax.get_visible()]
    for ax in fig.axes:
        for lab in (ax.xaxis.label, ax.yaxis.label, ax.title):
            if not (lab.get_visible() and lab.get_text().strip()):
                continue
            b = lab.get_window_extent(r)
            for other, pb in panels:
                if other is ax:
                    continue
                w = min(b.x1, pb.x1) - max(b.x0, pb.x0); h = min(b.y1, pb.y1) - max(b.y0, pb.y0)
                if w > 2 and h > 2:
                    hits.append(f"{lab.get_text()[:18]!r} into another panel")
    # an axis label longer than its axes runs off the figure (the tight crop does not always rescue it)
    H = fig.bbox.height
    for ax in fig.axes:
        yl = ax.yaxis.label
        if yl.get_visible() and yl.get_text().strip():
            b = yl.get_window_extent(r)
            if b.y1 > H + 1 or b.y0 < -1:
                hits.append(f"{yl.get_text()[:18]!r} off the figure")
    _last_hits = hits
    return len(hits)


def _fit(fig, target, allowed, relayout_first=False, cap=float("inf")):
    """grow the text toward target (median size, pt) only as far as it adds no collisions beyond `allowed`
    (what the figure had to begin with); tries the script's own layout before re-running tight_layout"""
    base = {id(t): t.get_fontsize() for t in fig.findobj(Text)}
    tick = {}
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            labs = axis.get_ticklabels()
            tick[id(axis)] = labs[0].get_fontsize() if labs else None
    sp = fig.subplotpars
    pars = dict(left=sp.left, right=sp.right, bottom=sp.bottom, top=sp.top, wspace=sp.wspace, hspace=sp.hspace)
    sizes = [t.get_fontsize() for t in _texts(fig)]
    k0 = min(cap, max(1.0, target / statistics.median(sizes)) if sizes else 1.0)
    tries = sorted({k0 * f for f in (1, 0.9, 0.8, 0.7, 0.6, 0.5) if k0 * f >= 1} | {1.0}, reverse=True)
    n = None
    for k in tries:
        for t in fig.findobj(Text):
            if id(t) in base:
                t.set_fontsize(base[id(t)] * k)
        for ax in fig.axes:
            for axis in (ax.xaxis, ax.yaxis):
                if tick.get(id(axis)):
                    axis.set_tick_params(labelsize=tick[id(axis)] * k)
        for relayout in ((True, False) if relayout_first else (False, True)):
            if fig.get_layout_engine() is None:
                fig.subplots_adjust(**pars)
            if relayout:
                _layout(fig)
            n = _problems(fig)
            if n <= allowed:
                return k, n
    return 1.0, n


def _adjust(fig, name):
    """the FIGURES placement changes that come before the text is sized: legends moved, series dashed"""
    opt = FIGURES.get(name, {})
    panels = sorted(_main_axes(fig), key=lambda a: (a.get_subplotspec().rowspan.start, a.get_subplotspec().colspan.start))
    for i, (loc, anchor, _) in opt.get("legend", {}).items():
        leg = panels[i].get_legend()
        leg._loc = Legend.codes[loc]
        leg.set_bbox_to_anchor(anchor, transform=panels[i].transAxes)
    for ax in fig.axes:
        for t in ax.texts:
            if t.get_text() in opt.get("nudge", {}):
                dx, dy = opt["nudge"][t.get_text()]
                x, y = t.get_position()
                t.set_position((x + dx, y + dy))
    dashed = set(opt.get("dash", ()))
    for ax in fig.axes if dashed else ():
        lines = ax.get_lines()
        top = max((l.get_zorder() for l in lines), default=2)
        for l in lines:
            if l.get_label() in dashed:
                l.set_linestyle((0, (3, 2))); l.set_zorder(top + 0.1)
        leg = ax.get_legend()
        if leg is not None:
            for h, t in zip(getattr(leg, "legend_handles", None) or leg.legendHandles, leg.get_texts()):
                if t.get_text() in dashed:
                    h.set_linestyle((0, (3, 2)))


def _repel(fig):
    """inline labels (text boxes placed on the data) that overlap: push each pair apart vertically, half each"""
    for _ in range(4):
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        moved = False
        for ax in fig.axes:
            ts = [t for t in ax.texts if t.get_text().strip() and t.get_bbox_patch() is not None]
            for i in range(len(ts)):
                for j in range(i + 1, len(ts)):
                    a, b = (t.get_bbox_patch().get_window_extent(r) for t in (ts[i], ts[j]))
                    w = min(a.x1, b.x1) - max(a.x0, b.x0); h = min(a.y1, b.y1) - max(a.y0, b.y0)
                    if w > 0 and h > 0:
                        lo, hi = (ts[i], ts[j]) if a.y0 < b.y0 else (ts[j], ts[i])
                        for t, dy in ((lo, -(h / 2 + 0.5)), (hi, h / 2 + 0.5)):
                            x, y = ax.transData.transform(t.get_position())
                            t.set_position(ax.transData.inverted().transform((x, y + dy)))
                        moved = True
        if not moved:
            return


def _halo(fig, theme):
    """annotations set on a patch of the page colour, so a line running under a label does not strike through it
    (a box, not a stroke path effect: with svg.fonttype=path a path effect writes every glyph out in full)"""
    for ax in fig.axes:
        for t in ax.texts:
            if not t.get_text().strip():               # an arrow drawn as an empty annotation keeps its own look
                continue
            bb = t.get_bbox_patch()
            if bb is not None:
                bb.set_facecolor(PAGE[theme])
            else:
                t.set_bbox(dict(facecolor=PAGE[theme], edgecolor="none", pad=0.6))


def _axis_labels(fig):
    """axis labels, which matplotlib (3.6) collapses to a point when it finds the tight crop, so a label longer
    than its axes is cut off at the figure's edge; one longer than everything else in the figure is set smaller"""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    tb = fig.get_tightbbox(r)
    W, H = tb.width * fig.dpi, tb.height * fig.dpi
    labels = []
    for ax in fig.axes:
        for lab, long in ((ax.xaxis.label, "width"), (ax.yaxis.label, "height")):
            if not (ax.get_visible() and lab.get_visible() and lab.get_text().strip()):
                continue
            size = getattr(lab.get_window_extent(r), long)
            room = W if long == "width" else H
            text = lab.get_text()
            if size > room and ", " in text and "\n" not in text:     # first break it at the comma nearest its middle
                i = min((i for i in range(len(text)) if text.startswith(", ", i)), key=lambda i: abs(i - len(text) / 2))
                lab.set_text(text[:i + 1] + "\n" + text[i + 2:])
                lab.set_linespacing(1.1)
                size = getattr(lab.get_window_extent(r), long)
            if size > room:
                lab.set_fontsize(lab.get_fontsize() * 0.97 * room / size)
            labels.append(lab)
    return labels


def _variant(fig, name, narrow, theme):
    with plt.rc_context(FONT):
        for t in fig.findobj(Text):
            t.set_fontfamily("serif")
        opt = FIGURES.get(name, {})
        _adjust(fig, name)
        allowed = _problems(fig)                      # collisions the figure already has as drawn
        for ax in fig.axes:
            ax.tick_params(length=3, width=0.6)
        w, h = fig.get_size_inches()
        # the text size that reads at ~12 px where the figure is shown: the column (740 px) or a phone (360 px)
        if narrow:
            snapshot = [(ax, ax.get_subplotspec()) for ax in fig.axes]
            rows, ncols = _stack(fig)
            if rows > 1:
                per = min(max(NARROW_IN * h / (w / ncols), 2.4), 3.4)
                fig.set_size_inches(NARROW_IN, per * rows + (0.35 if fig._suptitle is not None else 0))
            else:
                fig.set_size_inches(NARROW_IN, h * NARROW_IN / w)
            k, bad = _fit(fig, NARROW_PT, allowed, relayout_first=rows > 1)
            if bad > allowed and rows > 1:                      # the stacked layout will not come clean: keep the original
                for ax, ss in snapshot:
                    if ss is not None:
                        ax.set_subplotspec(ss)
                fig.set_size_inches(NARROW_IN, h * NARROW_IN / w)
                k, bad = _fit(fig, NARROW_PT, allowed)
            if bad > allowed or k < 1.05:             # no clean phone layout, or no gain: phones get the wide version
                with open(os.path.join(OUT, "report.txt"), "a") as f:
                    f.write(f"{name}-narrow{'-dark' if theme == 'dark' else ''}\tSKIPPED\t{' ; '.join(_last_hits[:3])}\n")
                return
        else:
            k, bad = _fit(fig, WIDE_PT * w / WIDE_IN, allowed, cap=opt.get("max_scale", float("inf")))   # the figure keeps its size; text is set for its display width
        if opt.get("legend"):
            panels = sorted(_main_axes(fig), key=lambda a: (a.get_subplotspec().rowspan.start, a.get_subplotspec().colspan.start))
            for i, (_, _, f) in opt["legend"].items():
                for t in panels[i].get_legend().get_texts():
                    t.set_fontsize(t.get_fontsize() * f)
        if opt.get("repel"):
            _repel(fig)
        _recolor(fig, theme)
        if opt.get("halo"):
            _halo(fig, theme)
        labels = _axis_labels(fig)
        suffix = ("-narrow" if narrow else "") + ("-dark" if theme == "dark" else "")
        path = os.path.join(OUT, name + suffix + ".svg")
        _orig_savefig(fig, path, format="svg", transparent=True, bbox_inches="tight", pad_inches=0.04,
                      bbox_extra_artists=fig.get_default_bbox_extra_artists() + labels, metadata={"Date": None})
        if narrow:                                    # a side legend can leave it wider than a phone
            import re as _re
            width = float(_re.search(r'width="([\d.]+)pt"', open(path).read(1000)).group(1))
            if width > NARROW_IN * 72 * 1.2:
                os.remove(path)
                with open(os.path.join(OUT, "report.txt"), "a") as f:
                    f.write(f"{name}{suffix}\tSKIPPED\twider than a phone ({width / 72:.1f} in)\n")
                return
        with open(os.path.join(OUT, "report.txt"), "a") as f:
            f.write(f"{name}{suffix}\tscale {k:.2f}\tproblems {bad} (had {allowed})\t{' ; '.join(_last_hits[:4]) if bad else ''}\n")


_orig_savefig = Figure.savefig
_done = set()


def savefig(self, fname, *a, **kw):
    name = os.path.splitext(os.path.basename(str(fname)))[0]
    if name in _done:
        return
    _done.add(name)
    os.makedirs(OUT, exist_ok=True)
    # one variant per run of the script (run.py runs it once for each): restyling mutates the figure, and
    # copying a figure (pickle) can drag in far more than the figure
    v = os.environ.get("WEBFIG_VARIANT", "wide-light")
    _variant(self, name, v.startswith("narrow"), v.split("-")[1])
    print("web figure:", name, v)


Figure.savefig = savefig
plt.savefig = lambda *a, **kw: plt.gcf().savefig(*a, **kw)
