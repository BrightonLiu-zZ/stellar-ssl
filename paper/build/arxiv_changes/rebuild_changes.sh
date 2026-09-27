#!/usr/bin/env bash
# Rebuild changes_arxiv.pdf: a two-page cover listing what changed between the ML4PS submission and
# the arXiv version, followed by the body only, marked up by latexdiff. Run from anywhere.
#
# "Before" is the frozen submission in build/submitted_ml4ps_2026/, never a review wave.
# "After" is the live main.tex. make_excerpt.py cuts the diff at the end of the body and rewrites the
# appendix cross-references, so nothing prints as "??".
cd "$(dirname "$0")/../.." || exit 1          # paper/
W=build/arxiv_changes

cp build/submitted_ml4ps_2026/main.tex $W/main_ml4ps.tex
cp main.tex $W/main_arxiv.tex
latexdiff-so $W/main_ml4ps.tex main.tex > $W/main_diff.tex 2>/dev/null
python -u $W/make_excerpt.py

cp $W/excerpt_diff.tex .
pdflatex -interaction=batchmode excerpt_diff.tex | grep -aE "^!"
bibtex excerpt_diff | grep -aiE "error"
pdflatex -interaction=batchmode excerpt_diff.tex | grep -aE "^!"
pdflatex -interaction=batchmode excerpt_diff.tex | grep -aE "^!|undefined"
(cd $W && pdflatex -interaction=batchmode cover.tex | grep -aE "^!")

pdfunite $W/cover.pdf excerpt_diff.pdf $W/changes_arxiv.pdf 2>&1 | grep -av "MiKTeX"
rm -f excerpt_diff.* $W/cover.pdf $W/cover.aux $W/cover.log $W/ex-*.png $W/cv-*.png $W/e1.log
pdfinfo $W/changes_arxiv.pdf | grep Pages
