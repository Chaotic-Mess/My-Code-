# Build data/processed/corpus.txt from data/raw/*.txt (skip dialogue files)
import os, re, glob, hashlib

RAW = "data/raw"
OUT = "data/processed/corpus.txt"
EXCLUDE_NAMES = {"dialogue_lines.txt", "dialogs.txt", "dialogue.txt"}

def clean(txt: str) -> str:
    txt = txt.replace("\r\n","\n").replace("\r","\n")
    txt = re.sub(r"[ \t]+\n","\n", txt)
    txt = re.sub(r"\n{3,}","\n\n", txt)
    return txt.strip()

def para_hash(p: str) -> str:
    return hashlib.sha1(p.strip().encode("utf-8")).hexdigest()

def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    paragraphs, seen = [], set()

    for p in glob.glob(os.path.join(RAW,"**","*.txt"), recursive=True):
        name = os.path.basename(p).lower()
        if name in EXCLUDE_NAMES or "dialog" in name:
            continue
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            txt = clean(f.read())
        for para in [x for x in txt.split("\n\n") if x.strip()]:
            h = para_hash(para)
            if h not in seen:
                seen.add(h); paragraphs.append(para)

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n\n".join(paragraphs) + "\n")
    print(f"wrote {OUT} | {len(paragraphs)} unique paragraphs")

if __name__ == "__main__":
    main()
