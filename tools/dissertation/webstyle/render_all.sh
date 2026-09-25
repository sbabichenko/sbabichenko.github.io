#!/bin/bash
# Re-render the dissertation's scripted figures for the site: four SVGs each (light/dark, wide/phone) into
# static/dissertation/media/web/, then point the chapter pages at them (../webfigs.py).
#
#   tools/dissertation/webstyle/render_all.sh [~/dissertation]
#
# Runs every script listed in the dissertation's numerics/regen_figs.sh through webstyle.py, in a symlinked copy
# of numerics/ so nothing in the dissertation tree is written (the scripts only read their data; webstyle replaces
# savefig, so no PDF is made). Each script is capped at 6 GB and 10 minutes. out/report.txt in the work directory
# says, per figure, how far the text was enlarged and which phone layouts were skipped (text that would collide).
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"; SITE="$(cd "$HERE/../../.." && pwd)"
DISS="${1:-$HOME/dissertation}"
WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
SYSPY=/usr/bin/python3; KBPY=$HOME/.kbenv/bin/python3        # the two interpreters regen_figs.sh uses
cp -rs "$DISS/numerics" "$WORK/numerics"; mkdir -p "$WORK/figs/ch5" "$WORK/out" "$WORK/fonts"
# Newsreader, the site's face, as static TTFs matplotlib can load (the site ships variable woff2)
$SYSPY - "$SITE/static/fonts" "$WORK/fonts" <<'PY'
import sys
from fontTools.ttLib import TTFont
from fontTools.varLib import instancer
src, dst = sys.argv[1:]
for style in ("normal", "italic"):
    f = TTFont(f"{src}/newsreader-latin-opsz-{style}.woff2"); f.flavor = None
    instancer.instantiateVariableFont(f, {"wght": 400, "opsz": 12}).save(f"{dst}/Newsreader-{style}.ttf")
PY
export WEBFIG_OUT="$WORK/out" WEBFIG_FONTS="$WORK/fonts"
grep '^run ' "$DISS/numerics/regen_figs.sh" | while read -r _ py script _; do
  case "$py" in '"$KBPY"') py=$KBPY ;; *) py=$SYSPY ;; esac
  echo "== $script"
  ( ulimit -v 6000000; timeout 600 "$py" "$HERE/run.py" "$WORK/numerics/$script" | grep -v '^wrote\|^web figure' || true )
done
cat "$WORK/out/report.txt"
rm -f "$SITE/static/dissertation/media/web/"*.svg; mkdir -p "$SITE/static/dissertation/media/web"
cp "$WORK/out/"*.svg "$SITE/static/dissertation/media/web/"
python3 "$SITE/tools/dissertation/webfigs.py"
