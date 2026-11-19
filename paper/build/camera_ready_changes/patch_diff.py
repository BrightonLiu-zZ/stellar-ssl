"""One repair to latexdiff's output: show the old version of a table whose column layout changed.

Same repair as paper/build/arxiv_changes/patch_diff.py. Wrapping \\multicolumn or \\cmidrule in
\\DIFadd breaks the alignment ("Misplaced \\noalign"), so latexdiff runs with `tabular` in
PICTUREENV and treats each table as one block. It then leaves the old table only as a
`%DIFDELCMD <` comment; this turns it back into LaTeX and prints it in red above the new table.
In the camera-ready this is Table 1, which gained a pool column (roadmap K5).

No reference retargeting is needed here: no label of the arXiv v1 version was removed.
"""
from __future__ import annotations

import re
from pathlib import Path

HERE = Path(__file__).resolve().parent

DELETED_BLOCK = re.compile(
    r"\\DIFdelbeginFL[ \t]*\n?(?P<block>(?:%DIFDELCMD <.*\n)+)(?:%%%\n)?\\DIFdelendFL",
)
COMMENT_PREFIX = re.compile(r"^%DIFDELCMD <[ ]?")

BEFORE_LABEL = r"{\scriptsize\textcolor{red}{\textbf{arXiv v1:}}}\par\nobreak\smallskip"
AFTER_LABEL = r"{\scriptsize\textcolor{blue}{\textbf{camera-ready:}}}\par\nobreak\smallskip"


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


def insert_note(src: str, before: str, reordered: bool) -> str:
    """One reading note under the title: what the colours mean, the baseline, the appendix order."""
    order = (
        "The appendices were reordered into first-citation order (A--J); the arXiv v1 text is shown in "
        "the new order so that the edits inside each appendix are marked rather than the whole appendix."
        if reordered else
        "The appendices were reordered into first-citation order; each moved appendix shows as deleted "
        "and re-added."
    )
    note = (
        "\\noindent{\\footnotesize\\textit{Changes from the arXiv v1 build (commit "
        f"\\texttt{{{before}}}) to the camera-ready draft, marked by latexdiff: "
        "\\textcolor{red}{red struck-through} removed, \\textcolor{blue}{blue underlined} added. "
        f"{order}}}}}\\par\\medskip\n"
    )
    marker = "\\maketitle"
    if src.count(marker) != 1:
        raise ValueError("expected exactly one \\maketitle")
    return src.replace(marker, marker + "\n" + note, 1)


def main() -> None:
    import sys
    before = sys.argv[1] if len(sys.argv) > 1 else "?"
    reordered = (sys.argv[2] if len(sys.argv) > 2 else "reordered") == "reordered"
    path = HERE / "main_diff.tex"
    src = path.read_text(encoding="utf-8")
    src, restored = restore_deleted_tables(src)
    src = insert_note(src, before, reordered)
    path.write_text(src, encoding="utf-8", newline="")
    print(f"restored {restored} deleted table(s); reading note inserted")


if __name__ == "__main__":
    main()
