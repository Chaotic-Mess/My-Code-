# V2/src/get_sample_corpus.py
# Robust downloader for a compact, engineering-leaning starter corpus.
# Usage:  python -m src.get_sample_corpus
import os, re, time, json
from pathlib import Path

try:
    import requests
except Exception:
    requests = None

import urllib.request, urllib.parse

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "raw" / "samples"
ATTRIB = OUT_DIR / "ATTRIBUTION.txt"

# Engineering + CS topics, compact but useful
WIKI_TITLES = [
    "Calculus",
    "Linear algebra",
    "Differential equation",
    "Classical mechanics",
    "Thermodynamics",
    "Electricity",
    "Magnetism",
    "Control theory",
    "Materials science",
    "Probability",
    "Statistics",
    "Algorithm",
    "Data structure",
    "Operating system",
    "Computer science",
]

GUTENBERG = [
    ("art_of_war_sun_tzu.txt", "https://www.gutenberg.org/cache/epub/132/pg132.txt"),
    ("alice_in_wonderland_lewis_carroll.txt", "https://www.gutenberg.org/files/11/11-0.txt"),
]

GITHUB = [
    ("tiny_shakespeare.txt", "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"),
]

UA = {"User-Agent": "ARES-local-corpus/1.0 (+local use)"}

def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def _requests_get(url: str, params=None):
    if requests is None:
        # Fallback: urllib
        if params:
            url = url + "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=25) as resp:
            return resp.read(), resp.status
    else:
        r = requests.get(url, params=params, headers=UA, timeout=25, allow_redirects=True)
        return r.content, r.status_code

def fetch_bytes(url: str, params=None, retries=3):
    last_err = None
    for i in range(retries):
        try:
            data, status = _requests_get(url, params)
            if 200 <= status < 300 and data:
                return data
            last_err = f"HTTP {status}"
        except Exception as e:
            last_err = str(e)
        time.sleep(1.5 * (i + 1))
    raise RuntimeError(last_err or "download failed")

def wiki_fetch_plain(title: str) -> bytes | None:
    """Try REST 'plain' endpoint with redirect."""
    base = "https://en.wikipedia.org/api/rest_v1/page/plain/"
    url = base + urllib.parse.quote(title)
    try:
        return fetch_bytes(url, params={"redirect": "true"})
    except Exception:
        return None

def wiki_fetch_extract(title: str) -> bytes | None:
    """Fallback: Action API extract as plaintext JSON → bytes."""
    api = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": "1",
        "redirects": "1",
        "format": "json",
        "titles": title,
    }
    try:
        raw = fetch_bytes(api, params=params)
        data = json.loads(raw.decode("utf-8", errors="ignore"))
        pages = data.get("query", {}).get("pages", {})
        if not pages:
            return None
        # grab first page
        page = next(iter(pages.values()))
        extract = page.get("extract", "")
        if not extract:
            return None
        return extract.encode("utf-8")
    except Exception:
        return None

def sanitize_filename(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("_").lower()

def save_bytes(path: Path, data: bytes):
    path.write_bytes(data)
    kb = len(data) / 1024
    print(f"✓ {path.name:50s}  {kb:7.1f} KB")

def get_wikipedia_pages():
    print("Fetching Wikipedia topics...")
    for title in WIKI_TITLES:
        data = wiki_fetch_plain(title)
        if data is None:
            data = wiki_fetch_extract(title)
        if data is None:
            print(f"× Skipped (not found): {title}")
            continue
        fname = sanitize_filename(f"wikipedia_{title}.txt")
        save_bytes(OUT_DIR / fname, data)

def get_simple_texts():
    print("Fetching public-domain/simple texts...")
    for fname, url in GUTENBERG + GITHUB:
        try:
            data = fetch_bytes(url)
            save_bytes(OUT_DIR / fname, data)
        except Exception as e:
            print(f"× Skipped {fname}: {e}")

def write_attribution():
    text = f"""\
This 'samples' subcorpus was downloaded for local/offline training of ARES.

Wikipedia pages (via REST or Action API):
{", ".join(WIKI_TITLES)}
License: CC BY-SA (ShareAlike applies if you redistribute derivatives).
REST endpoint: https://en.wikipedia.org/api/rest_v1/page/plain/<TITLE>?redirect=true
Action API:    https://en.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext=1&titles=<TITLE>

Public-domain texts:
- The Art of War — Project Gutenberg (pg132)
- Alice's Adventures in Wonderland — Project Gutenberg (11-0)
- Tiny Shakespeare — Public domain (Shakespeare); mirror via GitHub (karpathy/char-rnn)

Generated locally for private use. Timestamp: {time.ctime()}
"""
    ATTRIB.write_text(text, encoding="utf-8")
    print(f"\n✓ Wrote {ATTRIB.name}")

def main():
    ensure_dir(OUT_DIR)
    print(f"Downloading to: {OUT_DIR}\n")
    get_wikipedia_pages()
    get_simple_texts()
    write_attribution()
    print("\nDone. Next run:\n"
          "  python -m src.prepare_data\n"
          "  python -m src.tokenizer_train\n"
          "  python -m src.ingest_dialogue --in data/raw/dialogue_lines.txt  # if you have it\n"
          "  python -m src.make_instructions                               # optional\n"
          "  python -m src.train_sft\n"
          "  python -m src.chat")

if __name__ == "__main__":
    main()
