# Convert a tab-separated "User: ... \t AI: ..." file to instructions.jsonl
# Run: python -m src.ingest_dialogue --in data/raw/dialogue_lines.txt
import csv, json, os, re, argparse

SYSTEM = "You are ARES, a concise, helpful, 100% local assistant."

def strip_role(s: str) -> str:
    return re.sub(r"^\s*(user|ai|ares)\s*:\s*", "", s, flags=re.I).strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_file", default="data/raw/dialogue_lines.txt")
    ap.add_argument("--out", dest="out_file", default="data/processed/instructions.jsonl")
    ap.add_argument("--system", default=SYSTEM)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    n = 0
    with open(args.in_file, "r", encoding="utf-8") as fin,              open(args.out_file, "w", encoding="utf-8") as fout:
        reader = csv.reader(fin, delimiter="\t")
        for row in reader:
            if not row or len(row) < 2: continue
            user, ai = strip_role(row[0]), strip_role(row[1])
            prompt = f"<|system|> {args.system}\n<|user|> {user}\n<|assistant|> "
            json.dump({"prompt": prompt, "response": ai}, fout, ensure_ascii=False)
            fout.write("\n"); n += 1
    print(f"wrote {args.out_file} with {n} pairs")

if __name__ == "__main__":
    main()
