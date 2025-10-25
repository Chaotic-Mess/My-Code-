# app.py
from http.server import SimpleHTTPRequestHandler, HTTPServer
import os, json
from model.model import TinyCharRNN
from model.tokenizer import CharTokenizer 

# ---- tiny model setup ----
DATA_PATH = os.path.join("data", "tiny_shakespeare.txt")
WEIGHTS_PATH = os.path.join("weights", "model.json")

# build tokenizer from the same corpus
with open(os.path.join("data","tiny_shakespeare.txt"), encoding="utf-8") as f:
    corpus = f.read()
tokenizer = CharTokenizer(corpus)

# load trained weights
model = TinyCharRNN(len(tokenizer.stoi))
try:
    model = TinyCharRNN.load(os.path.join("weights","model.json"))
except Exception as e:
    print("Weights not found, using random-initialized model.", e)

# ---- HTTP handler ----
class ChatHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.path = "static/index.html"
        elif self.path == "/style.css":
            self.path = "static/style.css"
        elif self.path == "/main.js":
            self.path = "static/main.js"
        return super().do_GET()

    def do_POST(self):
        if self.path != "/chat":
            self.send_error(404, "Unknown endpoint")
            return
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        msg = body.get("message", "")
        reply = model.generate(
            tokenizer,
            seed=msg,
            max_new=160,
            temperature=0.8,   # safer, more coherent by default (0.6..0.9)
            top_k=50           # trim the ultra‑low‑prob tail
        )

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"response": reply}).encode("utf-8"))

def run():
    os.chdir(os.path.dirname(__file__))
    port = 8000
    print(f"ARES_AI running → http://localhost:{port}")
    HTTPServer(("0.0.0.0", port), ChatHandler).serve_forever()

if __name__ == "__main__":
    run()
