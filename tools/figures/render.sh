#!/usr/bin/env bash
# Render a figure whose code lives here (tools/figures/<fig>/<fig>.py and its model <fig>.yaml) for the site:
#
#   tools/figures/render.sh fig1_2            the four web SVGs (light, dark, and the phone layouts) in
#                                             static/dissertation/media/web/, made by running the script through
#                                             the webstyle harness, and the script and model copied to
#                                             static/dissertation/figcode/ for the figure's Inspect panel
#   tools/figures/render.sh fig1_2 --print    also the print PDF, fonts outlined, into ~/dissertation/figs/
#
# The script is run as it is shown on the page: nothing is edited on the way. NS_PY is the Python with noisestate
# (1.1 or later) and matplotlib; default ~/Projects/noisestate/.venv/bin/python.
set -euo pipefail
FIG="$1"; PRINT="${2:-}"
HERE="$(cd "$(dirname "$0")" && pwd)"; SITE="$(cd "$HERE/../.." && pwd)"; SRC="$HERE/$FIG"
NS_PY="${NS_PY:-$HOME/Projects/noisestate/.venv/bin/python}"; DISS="${DISS:-$HOME/dissertation}"
WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/run" "$WORK/out" "$WORK/fonts"; cp "$SRC"/"$FIG".py "$SRC"/"$FIG".yaml "$WORK/run/"
# Newsreader as static TTFs, as tools/dissertation/webstyle/render_all.sh makes them
/usr/bin/python3 - "$SITE/static/fonts" "$WORK/fonts" <<'PY'
import sys
from fontTools.ttLib import TTFont
from fontTools.varLib import instancer
src, dst = sys.argv[1:]
for style in ("normal", "italic"):
    f = TTFont(f"{src}/newsreader-latin-opsz-{style}.woff2"); f.flavor = None
    instancer.instantiateVariableFont(f, {"wght": 400, "opsz": 12}).save(f"{dst}/Newsreader-{style}.ttf")
PY
WEBFIG_OUT="$WORK/out" WEBFIG_FONTS="$WORK/fonts" "$NS_PY" "$SITE/tools/dissertation/webstyle/run.py" "$WORK/run/$FIG.py" | grep -v '^wrote\|^web figure' || true
mkdir -p "$SITE/static/dissertation/media/web" "$SITE/static/dissertation/figcode"
cp "$WORK/out/"*.svg "$SITE/static/dissertation/media/web/"
cp "$SRC/$FIG.py" "$SRC/$FIG.yaml" "$SITE/static/dissertation/figcode/"
echo "web: $(cd "$WORK/out" && ls *.svg | tr '\n' ' ')"
if [ "$PRINT" = "--print" ]; then
  (cd "$WORK/run" && "$NS_PY" "$FIG.py")
  for f in "$WORK/run/"*.pdf; do
    gs -q -dNOPAUSE -dBATCH -sDEVICE=pdfwrite -dNoOutputFonts -dCompatibilityLevel=1.5 -sOutputFile="$DISS/figs/$(basename "$f")" "$f"
    echo "print: $DISS/figs/$(basename "$f")"
  done
fi
