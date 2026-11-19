"""List every link in a built PDF: anchor text, named destination, landing page and landing text.

Usage: uv run --with pymupdf python paper/build/check_links.py [PDF]
The default is the preprint build of build_and_check.sh; pass
paper/build/out/camera_ready/main.pdf for the workshop build. paper/main.pdf is only refreshed by
build_and_check.sh --install, so it is not the default.
"""

import sys
import fitz
sys.stdout.reconfigure(encoding="utf-8")

pdf_path = sys.argv[1] if len(sys.argv) > 1 else r"C:/git_repo/Stellar-World-Model/paper/build/out/preprint/main.pdf"
doc = fitz.open(pdf_path)
print("checking", pdf_path)
names = doc.resolve_names()
count = 0
for pno, page in enumerate(doc, 1):
    for link in page.get_links():
        named = link.get("nameddest")
        dest = named or link.get("uri") or str(link.get("page"))
        anchor = page.get_textbox(link["from"]).replace("\n", " ").strip()
        target = names.get(named, {}) if named else {}
        landing_page = None
        snippet = ""
        if target:
            tpage = doc[target["page"]]
            landing_page = target["page"] + 1
            y = tpage.rect.height - target.get("to", (0, 0))[1]
            snippet = tpage.get_textbox(fitz.Rect(0, y - 2, tpage.rect.width, y + 30)).replace("\n", " ")[:80]
        print(f"p{pno} | {dest:<26} | {anchor[:38]:<38} | ->p{landing_page} | {snippet}")
        count += 1
print(count, "links")
