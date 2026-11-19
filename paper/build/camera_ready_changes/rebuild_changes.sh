#!/usr/bin/env bash
# Rebuild changes_camera_ready.pdf: the whole paper with every change from the arXiv v1 build to the
# current camera-ready draft marked up by latexdiff -- red struck-through removed, blue underlined
# added -- in the format of paper/build/arxiv_changes/changes_arxiv.pdf.
#
#   bash paper/build/camera_ready_changes/rebuild_changes.sh                  # appendices aligned
#   bash paper/build/camera_ready_changes/rebuild_changes.sh --keep-old-order # raw order, T1 visible
#
# "Before" is commit 54638f7, the arXiv v1 build the camera-ready roadmap (2026-10-10) was written
# against: its main.tex, refs.bib, tables and figures, reassembled under old/. "After" is the working
# tree, copied to new/. Both sides are compiled once in their own directory so --flatten inlines each
# side's own fresh .bbl; paper/main.bbl is never read (it can be stale: build_and_check.sh no longer
# writes it).
#
# By default the old appendices are put into the new order first (reorder_old_appendices.py), because
# the camera-ready moved all ten (T1) and a raw diff would show each one deleted and re-added, hiding
# the edits made inside them. The reading note on page 1 says so.
#
# latexdiff settings are those of the arXiv changes document: --graphics-markup=both typesets a
# replaced figure crossed out beside the new one; PICTUREENV includes tabular so a table whose column
# layout changed (Table 1, K5) compiles, and patch_diff.py prints its old version in red above it.
set -u
BEFORE=54638f7
REORDER=reordered
for arg in "$@"; do
  case "$arg" in
    --keep-old-order) REORDER=raw ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

PY="${PYTHON:-/c/Users/user1/.venvs/astro/Scripts/python.exe}"
cd "$(dirname "$0")/../.." || exit 1          # paper/
W=build/camera_ready_changes
OLD=$W/old
NEW=$W/new
rm -rf "$OLD" "$NEW"
mkdir -p "$OLD/tables" "$OLD/figures" "$NEW"

# --- old side: the arXiv v1 tree at $BEFORE
for f in main.tex refs.bib neurips_2026.sty; do git show "$BEFORE:paper/$f" > "$OLD/$f"; done
for f in $(git ls-tree --name-only "$BEFORE" tables/ figures/); do   # paths relative to paper/
  git show "$BEFORE:paper/$f" > "$OLD/$f"
done
[ -s "$OLD/tables/table1_scorecard.tex" ] || { echo "FAIL: old tables not extracted" >&2; exit 1; }

# --- new side: the working tree
cp main.tex refs.bib neurips_2026.sty "$NEW/"
cp -r tables figures "$NEW/"

# Figures regenerated under the same file name since $BEFORE: point the old document at its own copy
# under a distinct name, so latexdiff marks the figure as replaced instead of silently printing the
# new one in both places. The copy goes into new/figures too, where the diff is compiled.
for fig in $(cd figures && ls *.pdf); do
  if ! git diff --quiet "$BEFORE" -- "figures/$fig" 2>/dev/null; then
    cp "$OLD/figures/$fig" "$OLD/figures/v1_$fig"
    cp "$OLD/figures/$fig" "$NEW/figures/v1_$fig"
    sed -i "s#{figures/$fig}#{figures/v1_$fig}#" "$OLD/main.tex"
    echo "figure replaced since $BEFORE: $fig"
  fi
done

if [ "$REORDER" = reordered ]; then
  "$PY" -u $W/reorder_old_appendices.py "$OLD/main.tex" "$NEW/main.tex" || exit 1
fi

# Each side compiled once in its own directory, for its own main.bbl.
for side in "$OLD" "$NEW"; do
  ( cd "$side" && pdflatex -interaction=batchmode main.tex >/dev/null 2>&1; bibtex main >/dev/null 2>&1 )
  [ -s "$side/main.bbl" ] || { echo "FAIL: no main.bbl in $side" >&2; exit 1; }
done

latexdiff-so --flatten --graphics-markup=both \
  --config 'PICTUREENV=picture|DIFnomarkup|tabular' \
  "$OLD/main.tex" "$NEW/main.tex" 2>&1 | grep -av "MiKTeX" > $W/main_diff.tex
"$PY" -u $W/patch_diff.py "$BEFORE" "$REORDER" || exit 1

# No bibtex pass: --flatten has already inlined the bibliography as a thebibliography environment.
cp $W/main_diff.tex "$NEW/changes_camera_ready.tex"
( cd "$NEW" && for pass in 1 2 3; do
    pdflatex -interaction=batchmode changes_camera_ready.tex | grep -aE "^!" || true
  done )
[ -f "$NEW/changes_camera_ready.pdf" ] || { echo "FAIL: no PDF; see $NEW/changes_camera_ready.log" >&2; exit 1; }
mv "$NEW/changes_camera_ready.pdf" $W/changes_camera_ready.pdf
cp "$OLD/main.tex" $W/main_v1.tex
cp "$NEW/main.tex" $W/main_camera_ready.tex
echo "undefined references: $(grep -cE 'LaTeX Warning: (Reference|Citation).*undefined' "$NEW/changes_camera_ready.log")"
pdfinfo $W/changes_camera_ready.pdf | grep Pages
echo "wrote paper/$W/changes_camera_ready.pdf"
