#!/usr/bin/env bash
# Rebuild changes_arxiv.pdf: the whole arXiv paper with every change from the ML4PS submission
# marked up by latexdiff -- red struck-through for removed, blue underlined for added. No cover, no
# commentary. Run from anywhere:  bash paper/build/arxiv_changes/rebuild_changes.sh
#
# "Before" is the frozen submission in build/submitted_ml4ps_2026/ plus the table files as they stood
# at that commit, reassembled under build/arxiv_changes/old/ so that --flatten expands each version's
# own \input files. "After" is the live main.tex.
#
# Two latexdiff settings earn their place:
#   --graphics-markup=both   also typesets deleted figures, at half scale with a red cross. The
#                            default prints only the new ones, which would hide the figure the arXiv
#                            version replaced.
#   PICTUREENV includes tabular   a table whose column specification changed cannot be marked up
#                            cell by cell: \multicolumn and \cmidrule inside \DIFadd produce
#                            "Misplaced \noalign" and the build dies. Treating each tabular as one
#                            atomic block compiles, and patch_diff.py then prints the old
#                            table in red above the new one, which latexdiff alone would only leave
#                            behind as a comment.
cd "$(dirname "$0")/../.." || exit 1          # paper/
W=build/arxiv_changes

# The ML4PS side: its main.tex and the table files of that commit, side by side, for --flatten.
rm -rf $W/old
mkdir -p $W/old/tables
cp build/submitted_ml4ps_2026/main.tex $W/old/main.tex
SUBMITTED=fd738d6
for f in table1_scorecard tableC_bootstrap table_readout table_unseen; do
  git show $SUBMITTED:paper/tables/$f.tex > $W/old/tables/$f.tex
done
cp build/submitted_ml4ps_2026/main.tex $W/main_ml4ps.tex
cp main.tex $W/main_arxiv.tex

# --flatten also expands \bibliography{refs} into the .bbl beside the file being processed. The live
# paper has main.bbl; the reassembled ML4PS tree does not, so without this the whole reference list
# would come out marked as added. Compiling the old document once produces its own main.bbl, and the
# two lists then diff to nothing, which is correct: refs.bib has not changed.
cp refs.bib neurips_2026.sty $W/old/
cp -r figures $W/old/figures
(cd $W/old && pdflatex -interaction=batchmode main.tex >/dev/null 2>&1; bibtex main >/dev/null 2>&1)

latexdiff-so --flatten --graphics-markup=both \
  --config 'PICTUREENV=picture|DIFnomarkup|tabular' \
  $W/old/main.tex main.tex 2>&1 | grep -av "MiKTeX" > $W/main_diff.tex
python -u $W/patch_diff.py

# No bibtex pass: --flatten has already inlined the bibliography as a thebibliography environment.
cp $W/main_diff.tex changes_arxiv.tex
pdflatex -interaction=batchmode changes_arxiv.tex | grep -aE "^!"
pdflatex -interaction=batchmode changes_arxiv.tex | grep -aE "^!"
pdflatex -interaction=batchmode changes_arxiv.tex | grep -aE "^!|undefined"
mv changes_arxiv.pdf $W/changes_arxiv.pdf
rm -f changes_arxiv.tex changes_arxiv.aux changes_arxiv.bbl changes_arxiv.blg changes_arxiv.log changes_arxiv.out
echo "overfull boxes and page count:"
pdfinfo $W/changes_arxiv.pdf | grep Pages
