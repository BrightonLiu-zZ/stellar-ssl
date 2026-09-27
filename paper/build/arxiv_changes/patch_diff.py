"""Two repairs to latexdiff's output, both forced by changes latexdiff cannot express.

1. **A table whose column specification changed.** Wrapping \\multicolumn or \\cmidrule in \\DIFadd
   breaks the alignment and pdflatex dies with "Misplaced \\noalign", so the build runs latexdiff
   with `tabular` in PICTUREENV, which treats each table as one atomic block. That compiles, but
   latexdiff then leaves the old table behind as a `%DIFDELCMD <` comment and typesets only the new
   one, so the reader sees no sign of what it replaced. This turns the comment back into LaTeX and
   prints it in red above the new table, which is the convention the surrounding prose already uses.

2. **A reference to a label that exists only in the ML4PS version.** The old Appendix C figure
   carried \\label{fig:valloss}; the arXiv version splits it into fig:valloss_a (now Figure 2 in the
   body) and fig:valloss_b. The deleted sentence still says \\ref{fig:valloss}, whose label survives
   only inside a comment, so it prints as "Figure ??". Each such reference is retargeted at the
   figure that replaced it, so the number the deleted sentence shows is a figure the reader can
   actually turn to.

Anything latexdiff marked up successfully is left exactly as it is.
"""
from __future__ import annotations

import re
from pathlib import Path

HERE = Path(__file__).resolve().parent

# \DIFdelbeginFL, then one or more fully commented-out lines, then a bare %%% sentinel, then
# \DIFdelendFL. The commented lines are the old table.
DELETED_BLOCK = re.compile(
    r"\\DIFdelbeginFL[ \t]*\n?(?P<block>(?:%DIFDELCMD <.*\n)+)(?:%%%\n)?\\DIFdelendFL",
)
COMMENT_PREFIX = re.compile(r"^%DIFDELCMD <[ ]?")

BEFORE_LABEL = r"{\scriptsize\textcolor{red}{\textbf{ML4PS version:}}}\par\nobreak\smallskip"
AFTER_LABEL = r"{\scriptsize\textcolor{blue}{\textbf{arXiv version:}}}\par\nobreak\smallskip"

# Label in the ML4PS version -> the label of what replaced it in the arXiv version.
RETARGET_REFS = {"fig:valloss": "fig:valloss_b"}


def uncomment(block: str) -> list[str]:
    lines = []
    for raw in block.splitlines():
        line = COMMENT_PREFIX.sub("", raw)
        if line.strip() in {"", "%%%"}:
            continue
        lines.append(line)
    return lines


def restore_deleted_tables(src: str) -> tuple[str, int]:
    restored = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal restored
        lines = uncomment(match.group("block"))
        if not any(r"\begin{tabular}" in line for line in lines):
            return match.group(0)  # not a table; leave latexdiff's own markup alone
        restored += 1
        old = "\n".join(lines)
        return (
            "\\DIFdelbeginFL\n"
            f"{BEFORE_LABEL}\n"
            "\\begingroup\\color{red}\n"
            f"{old}\n"
            "\\endgroup\\par\\medskip\n"
            f"{AFTER_LABEL}\n"
            "\\DIFdelendFL"
        )

    return DELETED_BLOCK.sub(replace, src), restored


def retarget_dead_refs(src: str) -> tuple[str, int]:
    retargeted = 0
    for old_label, new_label in RETARGET_REFS.items():
        src, n = re.subn(
            re.escape(f"\\ref{{{old_label}}}"), f"\\\\ref{{{new_label}}}", src
        )
        retargeted += n
    return src, retargeted


def main() -> None:
    path = HERE / "main_diff.tex"
    src = path.read_text(encoding="utf-8")

    src, restored = restore_deleted_tables(src)
    src, retargeted = retarget_dead_refs(src)

    path.write_text(src, encoding="utf-8")
    print(f"restored {restored} deleted table(s), retargeted {retargeted} dead reference(s)")


if __name__ == "__main__":
    main()
