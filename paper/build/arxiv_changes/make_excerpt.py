"""Cut the latexdiff output down to the body, so the marked-up pages stay readable.

The appendices are mostly new in the arXiv version, so marking them adds pages without adding
information: the cover page lists them instead. What is left is the preamble, the title block and
sections 1-4, which is where the sentence-level edits live.

Cross-references into the appendices would print as "??" once the appendices are cut, so they are
replaced by the letter the arXiv version gives each one. The letters are the arXiv version's
throughout, including inside deleted (red) text, where the ML4PS version used a different letter;
the cover states this.
"""
from __future__ import annotations

import re
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Appendix label -> the letter it carries in the arXiv version, in \appendix order.
APPENDIX_LETTERS: dict[str, str] = {
    "app:training": "A",
    "app:pools": "B",
    "app:negatives": "C",
    "app:readout": "D",
    "app:valloss": "E",
    "app:artifact": "F",
    "app:bootstrap": "G",
    "app:features": "H",
    "app:unseen": "I",
    "app:rotpool": "J",
}

# The body ends here; everything from this line on is bibliography and appendices.
CUT_MARKER = r"\raggedbottom"

BS = chr(92)
TAIL = (
    f"\n{BS}clearpage\n"
    f"{BS}bibliographystyle{{plainnat}}\n"
    f"{BS}bibliography{{refs}}\n"
    f"{BS}end{{document}}\n"
)


def main() -> None:
    src = (HERE / "main_diff.tex").read_text(encoding="utf-8")

    cut = src.index(CUT_MARKER)
    body = src[:cut] + TAIL

    # \ref{app:x} and \ref{tab:readout} point outside the excerpt; print the letter/number instead.
    def replace_ref(match: re.Match[str]) -> str:
        return APPENDIX_LETTERS[match.group(1)]

    body = re.sub(r"\\ref\{(app:[a-z]+)\}", replace_ref, body)
    body = body.replace(r"\ref{tab:readout}", "D.1")

    out = HERE / "excerpt_diff.tex"
    out.write_text(body, encoding="utf-8")

    unresolved = sorted(set(re.findall(r"\\ref\{([^}]+)\}", body)))
    print(f"wrote {out.name}, {len(body.splitlines())} lines")
    print("references still resolved from inside the excerpt:", unresolved)


if __name__ == "__main__":
    main()
