# Create synthetic instruction pairs to augment your dialogue set
# Run: python -m src.make_instructions
import json, os, random

OUT = "data/processed/instructions.jsonl"
SYSTEM = "You are ARES, a concise, helpful, 100% local assistant."

def arith():
    a,b = random.randint(10,99), random.randint(10,99)
    return (f"What is {a} + {b}?", str(a+b))

def classify():
    x = random.choice(["cat","dog","car","tree"])
    return (f"Is '{x}' an animal? Answer yes or no.", "yes" if x in ["cat","dog"] else "no")

def sort_nums():
    nums = random.sample(range(1,50), k=5)
    return (f"Sort these numbers ascending: {nums}", str(sorted(nums)))

def reverse_str():
    s = random.choice(["ARES","local only","arrays rule"])
    return (f"Reverse this string: '{s}'", s[::-1])

def format_pair(prompt, response):
    return {"prompt": f"<|system|> {SYSTEM}\n<|user|> {prompt}\n<|assistant|> ", "response": response}

def main(n=300):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    gens = [arith, classify, sort_nums, reverse_str]
    with open(OUT, "a", encoding="utf-8") as f:
        for _ in range(n):
            p,r = random.choice(gens)()
            json.dump(format_pair(p,r), f); f.write("\n")
    print(f"appended {n} pairs to {OUT}")

if __name__ == "__main__":
    main()
