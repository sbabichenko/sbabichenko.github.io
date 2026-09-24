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
