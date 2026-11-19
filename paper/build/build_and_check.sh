#!/usr/bin/env bash
# Build the paper and print the format checks: overfull boxes, undefined references, page count,
# the five-page body boundary and the first-page notice.
#
# One source, two builds (camera-ready roadmap D35). paper/main.tex is the single source and carries
# the preprint (arXiv) style option; the camera-ready build is generated from it by swapping that
# one option for [final] and setting the first-page notice. Nothing else differs.
#
#   bash paper/build/build_and_check.sh                  # preprint  -> paper/build/out/preprint/
#   bash paper/build/build_and_check.sh --camera-ready   # ML4PS     -> paper/build/out/camera_ready/
#   bash paper/build/build_and_check.sh --install        # preprint, then copy its PDF to paper/main.pdf
#
# Page rule (ML4PS 2026 decision mail): the body may run to five pages, excluding references. The
# check is that no body text reaches page 6: the "References" heading must sit on page 5 or
# earlier, or be the first non-blank line of page 6. A page count alone cannot tell you this.
#
# The ML4PS 2026 site calls the workshop "independent ... not affiliated with NeurIPS" and gives no
# camera-ready footer, so the notice is the submitted one without "Submitted to" (author's choice,
# 2026-11-16), not the style file's NeurIPS workshop footer.
set -u

MODE="preprint"
INSTALL=0
for arg in "$@"; do
  case "$arg" in
    --camera-ready) MODE="camera_ready" ;;
    --install) INSTALL=1 ;;
    -h|--help) sed -n '2,19p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done
if [ "$INSTALL" = 1 ] && [ "$MODE" != "preprint" ]; then
  echo "--install copies the preprint build only; drop --camera-ready" >&2; exit 2
fi

BODY_PAGES="${BODY_PAGES:-5}"   # overridable only to exercise the FAIL path
CAMERA_NOTICE="9th Workshop on Machine Learning and the Physical Sciences (ML4PS 2026)."
PREPRINT_NOTICE="Preprint."

PAPER="$(cd "$(dirname "$0")/.." && pwd)" || exit 1
OUT="$PAPER/build/out/$MODE"
mkdir -p "$OUT"
cd "$PAPER" || exit 1   # tables/, figures/ and the style file resolve relative to paper/

if [ "$MODE" = "camera_ready" ]; then
  SRC="$OUT/main.tex"
  # Swap the one style line for [final] plus the notice; refuse if the line is not there exactly once.
  awk -v notice="$CAMERA_NOTICE" '
    /^\\usepackage\[preprint\]\{neurips_2026\}[[:space:]]*$/ {
      print "\\usepackage[final]{neurips_2026}"
      print "\\makeatletter"
      print "\\renewcommand{\\@noticestring}{" notice "}"
      print "\\makeatother"
      n++; next
    }
    { print }
    END { if (n != 1) exit 3 }
  ' main.tex > "$SRC" || { echo "FAIL: expected exactly one \\usepackage[preprint]{neurips_2026} line in main.tex" >&2; exit 1; }
  NOTICE="$CAMERA_NOTICE"
else
  SRC="$PAPER/main.tex"
  NOTICE="$PREPRINT_NOTICE"
fi

# pdflatex runs inside $OUT, with paper/ appended to the input path for the style file, tables/ and
# figures/. Running it from paper/ with -output-directory instead lets TeX pick up a stale
# paper/main.bbl ahead of the fresh $OUT/main.bbl (current directory is searched first), which
# silently printed an outdated reference list on 2026-11-16.
PAPER_NATIVE="$(cygpath -m "$PAPER" 2>/dev/null || echo "$PAPER")"   # MiKTeX is a native Windows binary
SRC_NATIVE="$(cygpath -m "$SRC" 2>/dev/null || echo "$SRC")"
latex_pass() {
  ( cd "$OUT" && pdflatex -interaction=nonstopmode --include-directory="$PAPER_NATIVE" "$SRC_NATIVE" \
      | grep -E "^!" || true )
}

echo "=== $MODE build -> $OUT"
rm -f "$OUT/main.aux" "$OUT/main.bbl"   # no state carried over from an earlier build
latex_pass
cp refs.bib "$OUT/refs.bib"   # MiKTeX bibtex ignores BIBINPUTS here, so the .bib goes beside the .aux
( cd "$OUT" && bibtex main | grep -iE "warning|error" || true )
latex_pass
latex_pass
# The .bbl actually typeset must be this build's: compare the entry count with the cited keys.
if ! grep -q "main.bbl" "$OUT/main.log" || [ ! -f "$OUT/main.bbl" ]; then
  echo "FAIL: main.bbl was not read from $OUT"; exit 1
fi

PDF="$OUT/main.pdf"
LOG="$OUT/main.log"
[ -f "$PDF" ] || { echo "FAIL: no PDF produced; see $LOG" >&2; exit 1; }

STATUS=0
OVERFULL=$(grep -c "Overfull" "$LOG")
UNDEF=$(grep -cE "LaTeX Warning: (Reference|Citation).*undefined" "$LOG")
PAGES=$(pdfinfo "$PDF" | awk '/^Pages:/ {print $2}')
echo "pages: $PAGES"
echo "overfull boxes: $OVERFULL"
echo "undefined references/citations: $UNDEF"
[ "$OVERFULL" -eq 0 ] || STATUS=1
[ "$UNDEF" -eq 0 ] || STATUS=1

# Page holding the "References" heading (first line that is exactly that word).
REF_PAGE=""
for p in $(seq 1 "$PAGES"); do
  if pdftotext -layout -f "$p" -l "$p" "$PDF" - | grep -qE '^[[:space:]]*References[[:space:]]*$'; then
    REF_PAGE=$p; break
  fi
done
if [ -z "$REF_PAGE" ]; then
  echo "FAIL: no 'References' heading found"; STATUS=1
elif [ "$REF_PAGE" -le "$BODY_PAGES" ]; then
  ABOVE=$(pdftotext -layout -f "$REF_PAGE" -l "$REF_PAGE" "$PDF" - | grep -v '^[[:space:]]*$' \
          | sed -n '/^[[:space:]]*References[[:space:]]*$/q;p' | wc -l)
  echo "PASS body boundary: References on page $REF_PAGE, below $ABOVE non-blank body lines"
else
  FIRST=$(pdftotext -layout -f "$REF_PAGE" -l "$REF_PAGE" "$PDF" - | grep -v '^[[:space:]]*$' | sed -n '1p')
  if [ "$REF_PAGE" -eq $((BODY_PAGES + 1)) ] && echo "$FIRST" | grep -qE '^[[:space:]]*References[[:space:]]*$'; then
    echo "PASS body boundary: References opens page $REF_PAGE"
  else
    echo "FAIL body boundary: References on page $REF_PAGE; body text runs past page $BODY_PAGES"
    echo "  first line of page $REF_PAGE: $FIRST"
    STATUS=1
  fi
fi

FOOT=$(pdftotext -layout -f 1 -l 1 "$PDF" - | grep -v '^[[:space:]]*$' | sed -n '$p' | sed 's/^[[:space:]]*//')
echo "first-page notice: $FOOT"
if [ "$FOOT" != "$NOTICE" ]; then
  echo "FAIL: notice should read '$NOTICE'"; STATUS=1
fi

if [ "$INSTALL" = 1 ]; then
  if [ "$STATUS" -eq 0 ]; then
    cp "$PDF" "$PAPER/main.pdf" && echo "installed $PDF -> paper/main.pdf"
  else
    echo "not installing: a check failed"
  fi
fi

[ "$STATUS" -eq 0 ] && echo "ALL CHECKS PASS" || echo "SOME CHECKS FAILED"
exit "$STATUS"
