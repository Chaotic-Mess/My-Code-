import json

def read_corpus_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f: return f.read()

def read_instructions_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f: yield json.loads(line)
