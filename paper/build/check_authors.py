"""Compare every author named in refs.bib, in order, with the arXiv author list; also compare titles."""

import re
import sys
import subprocess
import time
import unicodedata
import urllib.request
import xml.etree.ElementTree as ET

sys.stdout.reconfigure(encoding="utf-8")
atom = "{http://www.w3.org/2005/Atom}"
bib = open(r"C:/git_repo/Stellar-World-Model/paper/refs.bib", encoding="utf-8").read()

# Entries whose .bib title is the journal's published title, which differs from the arXiv title
# (camera-ready T10, 2026-11-16; published records read from Crossref). A title mismatch on these
# keys prints [OK-PUBLISHED] with both titles; any other mismatch, and any author problem, is still
# [CHECK].
published_title = {
    "zuo2025falco": "AJ 171, 10 (2026), doi:10.3847/1538-3881/ae1467",
    "donoso2025astromer2": "A&A 707, A170 (2026), doi:10.1051/0004-6361/202554026",
}

latex_accents = {r"\'{a}": "a", r"\'{o}": "o", r"\'{e}": "e", r"\v{s}": "s", r"\v{Z}": "Z",
                 r"\'{c}": "c", r"\"{e}": "e", r"\'{A}": "A"}


def norm(text: str) -> str:
    for k, v in latex_accents.items():
        text = text.replace(k, v)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9 ]", "", text.lower().replace("-", " ")).split()


for entry in re.findall(r"@\w+\{(\w+),(.*?)\n\}", bib, re.S):
    key, body = entry
    arxiv_id = re.search(r"arXiv:([\d.]+)", body).group(1)
    authors = re.search(r"author\s*=\s*\{(.*?)\},\n", body, re.S).group(1)
    title = re.search(r"title\s*=\s*\{(.*?)\},\n", body, re.S).group(1)
    named = [a.strip() for a in re.split(r"\s+and\s+", " ".join(authors.split())) if a.strip() != "others"]
    truncated = "others" in authors
    url = "https://export.arxiv.org/api/query?id_list=" + arxiv_id
    root = ET.fromstring(subprocess.run(["curl", "-sL", url], capture_output=True, check=True).stdout)
    e = root.find(f"{atom}entry")
    got = [a.find(f"{atom}name").text for a in e.findall(f"{atom}author")]
    got_title = " ".join(e.find(f"{atom}title").text.split())
    problems = []
    for i, name in enumerate(named):
        family = norm(name.split(",")[0])[-1]
        if i >= len(got) or family not in norm(got[i]):
            problems.append(f"author {i + 1}: bib '{name}' vs arXiv '{got[i] if i < len(got) else None}'")
        elif "," in name:
            given = norm(name.split(",")[1])
            if given and given[0][0] != norm(got[i])[0][0]:
                problems.append(f"author {i + 1} given name: bib '{name}' vs arXiv '{got[i]}'")
    if not truncated and len(named) != len(got):
        problems.append(f"bib lists {len(named)} authors with no 'others', arXiv has {len(got)}")
    title_note = None
    if norm(re.sub(r"[{}\\]", "", title)) != norm(got_title):
        if key in published_title:
            title_note = f"published title ({published_title[key]}); arXiv '{got_title}'"
        else:
            problems.append(f"title differs: arXiv '{got_title}'")
    status = "CHECK" if problems else ("OK-PUBLISHED" if title_note else "OK")
    print(f"[{status}] {key} ({len(named)}/{len(got)} authors named)")
    for p in problems + ([title_note] if title_note else []):
        print("     ", p)
    time.sleep(3.5)
