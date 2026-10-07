#!/usr/bin/env bash
# Rebuild main.pdf and print the format checks: overfull boxes, page count, page-4/5 boundary.
# Since 2026-10-06 the 4-page body limit applies again (the arXiv version must line up with the
# formal publication): the first line printed for page 5 must be "References".
# Run from anywhere: bash paper/build/build_and_check.sh
cd "$(dirname "$0")/.." || exit 1
pdflatex -interaction=nonstopmode main.tex | grep -E "^!|Warning"
bibtex main | grep -iE "warning|error"
pdflatex -interaction=nonstopmode main.tex | grep -E "^!"
pdflatex -interaction=nonstopmode main.tex | grep -E "^!|undefined"
echo "overfull boxes: $(grep -c Overfull main.log)"
pdfinfo main.pdf | grep Pages
echo "--- last lines of page 4"
pdftotext -layout -f 4 -l 4 main.pdf - | grep -v '^\s*$' | tail -n 3
echo "--- first lines of page 5 (must be References: 4-page body limit)"
pdftotext -layout -f 5 -l 5 main.pdf - | grep -v '^\s*$' | head -n 2
echo "--- first-page notice (arXiv build: must read 'Preprint.')"
pdftotext -layout -f 1 -l 1 main.pdf - | grep -v '^\s*$' | tail -n 1
