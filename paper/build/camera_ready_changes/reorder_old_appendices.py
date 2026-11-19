"""Put the old document's appendix \\section blocks into the new document's order, in place.

The camera-ready reordered all ten appendices into body first-citation order (roadmap T1). A diff
of the two files as they stand would show every appendix deleted and re-added, burying the edits
made inside them (R1-R9). Reordering the old blocks first lets latexdiff mark up only what changed
within each appendix; the move itself is stated in the changes document's preamble note.

Usage: python reorder_old_appendices.py OLD.tex NEW.tex   (OLD.tex is rewritten)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path


def appendix_blocks(text: str) -> tuple[str, dict[str, str], list[str], str]:
    """Split a document into (head through \\appendix, {label: block}, label order, tail)."""
    head, sep, rest = text.partition("\\appendix")
    if not sep:
        raise ValueError("no \\appendix in document")
    body, sep_end, tail = rest.rpartition("\\end{document}")
    if not sep_end:
        raise ValueError("no \\end{document}")
    starts = [m.start() for m in re.finditer(r"^\\section\{", body, flags=re.M)]
    if not starts:
        raise ValueError("no appendix \\section blocks")
    lead = body[: starts[0]]
    blocks: dict[str, str] = {}
    order: list[str] = []
    for i, start in enumerate(starts):
        block = body[start: starts[i + 1] if i + 1 < len(starts) else len(body)]
        labels = re.findall(r"\\label\{(app:[A-Za-z]+)\}", block)
        if len(labels) != 1:
            raise ValueError(f"appendix block without exactly one app: label: {labels}")
        blocks[labels[0]] = block
        order.append(labels[0])
    return head + sep + lead, blocks, order, sep_end + tail


def main() -> int:
    old_path, new_path = Path(sys.argv[1]), Path(sys.argv[2])
    old_text = old_path.read_text(encoding="utf-8")
    head, old_blocks, old_order, tail = appendix_blocks(old_text)
    _, _, new_order, _ = appendix_blocks(new_path.read_text(encoding="utf-8"))
    if set(old_order) != set(new_order):
        raise ValueError(f"appendix labels differ: old {old_order}, new {new_order}")
    nl = "\r\n" if "\r\n" in old_text else "\n"
    pieces = []
    for label in new_order:
        block = old_blocks[label]
        pieces.append(block if block.endswith(nl + nl) else block.rstrip("\r\n") + nl + nl)
    new_text = head + "".join(pieces) + tail
    if sorted(new_text.split()) != sorted(old_text.split()):
        raise ValueError("reordering changed the content")
    old_path.write_text(new_text, encoding="utf-8", newline="")
    print("old appendix order:", " ".join(old_order))
    print("now:               ", " ".join(new_order))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
