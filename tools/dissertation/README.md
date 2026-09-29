# The dissertation on the site

`/dissertation/` is generated from the LaTeX sources. To refresh it after editing the dissertation:

    cd ~/dissertation
    pdflatex combined_dissertation && bibtex combined_dissertation && pdflatex combined_dissertation && pdflatex combined_dissertation
    # the TikZ pictures, one PDF each (tikz external); build.py turns them into SVGs
    sed 's|\\usetikzlibrary{arrows.meta,positioning,matrix}|\\usetikzlibrary{arrows.meta,positioning,matrix,external}\\tikzexternalize[prefix=tikzext/]|' combined_dissertation.tex > ext.tex
    mkdir -p tikzext && pdflatex -shell-escape ext.tex
    # each picture is its own job and reads only its own .aux, so a \ref inside one comes out ?? (Fig 1.6):
    # give every picture the main labels and remake it
    for f in tikzext/ext-figure*.pdf; do j="${f%.pdf}"; cp combined_dissertation.aux "$j.aux"; rm "$f"
      pdflatex -shell-escape -halt-on-error -interaction=batchmode -jobname "$j" "\def\tikzexternalrealjob{ext}\input{ext}"; done
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
them; `build.py` runs it after every regeneration. Every figure now has a web version, so the print conversions
`build.py` writes (`media/*.webp`) are not kept; the TikZ diagrams' `media/tikz-N.svg` are, as `tikz_web.py`'s input.

## Which copy of the dissertation

`build.py` records a fingerprint of the LaTeX it built from in `data/dissertation/source.json` and refuses to
overwrite the pages from different LaTeX that is no newer, or when there is no record at all (the pages online
in September 2026 were built from another machine's copy). Compare the copies, then pass `--force`.

## Checking for drift

The pages on the site have been edited after they were built, and the LaTeX has been edited since, so before
regenerating from the LaTeX, see where the two differ:

    PANDOC_DIR=/path/to/pandoc-3/bin python3 tools/dissertation/drift.py ~/dissertation --out drift.txt

It builds the LaTeX into a scratch copy of the site (never into the site itself) and reports, page by page, the
paragraphs, headings and captions that differ, with a word-level diff (`[-site-] {+LaTeX+}`), and any that exist on
only one side. Formulas are compared by the glyphs KaTeX drew for them. Carry the site's wording back into the LaTeX
wherever it is the newer one, and only then run `build.py --force`. `--built DIR` reuses a scratch copy kept with
`--keep`; `--only chapter-1,chapter-2` limits the report.


## Notes added after the dissertation

`tools/dissertation/asides/*.html` holds notes that exist only on the web, each a `<details>` block folded shut, with
two comment lines naming the element it follows and its page. `asides.py` (run by `build.py` after the deps) inserts
them as a plain string edit, so the page is otherwise byte-identical, and typesets their math like the rest. The PDF
never has them. The first, `cara-operator.html`, follows Theorem 1.18.
