# The dissertation on the site

`/dissertation/` is generated from the LaTeX sources. To refresh it after editing the dissertation:

    cd ~/dissertation
    pdflatex combined_dissertation && bibtex combined_dissertation && pdflatex combined_dissertation && pdflatex combined_dissertation
    # the TikZ pictures, one PDF each (tikz external); build.py turns them into SVGs
    sed 's|\\usetikzlibrary{arrows.meta,positioning,matrix}|\\usetikzlibrary{arrows.meta,positioning,matrix,external}\\tikzexternalize[prefix=tikzext/]|' combined_dissertation.tex > ext.tex
    mkdir -p tikzext && pdflatex -shell-escape ext.tex
    cd -
    (cd tools/dissertation && npm install)
    python3 tools/dissertation/build.py ~/dissertation

What it writes: `content/dissertation/*.md` (front matter only), `data/dissertation/*.html` (the page bodies,
math already typeset), `data/dissertation/manifest.json` (the contents), and `static/dissertation/media/`
(figures). It prints the counts it checks: labels read, theorem numbers, equation tags, cross-references
resolved, formulas typeset. All of LaTeX's numbering comes from the `.aux`, `.toc` and `.bbl`, so the web
pages and the PDF agree.

Needs pandoc 3, poppler (`pdftocairo`), Pillow and BeautifulSoup.

## Web versions of the figures

The chapters' scripted figures are re-rendered for the site rather than converted from the print PDFs:

    tools/dissertation/webstyle/render_all.sh ~/dissertation

runs every script in the dissertation's `numerics/regen_figs.sh` through `webstyle/webstyle.py`. That makes four
SVGs per figure in `static/dissertation/media/web/`: light and dark, wide and phone (side-by-side panels
stacked). Each is in Newsreader, with text enlarged only as far as nothing collides. Phone layouts that would
not come out clean are skipped, and phones then get the wide one. `webfigs.py` then points the chapter pages at
them; `build.py` runs it after every regeneration. Figures without a script (the TikZ diagrams, fig2, fig12,
fig13) keep their print conversions.

## Which copy of the dissertation

`build.py` records a fingerprint of the LaTeX it built from in `data/dissertation/source.json` and refuses to
overwrite the pages from different LaTeX that is no newer, or when there is no record at all (the pages online
in September 2026 were built from another machine's copy). Compare the copies, then pass `--force`.
