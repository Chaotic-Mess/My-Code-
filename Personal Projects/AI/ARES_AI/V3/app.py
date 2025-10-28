# app.py — ARES_V3.5 local chat interface
from http.server import SimpleHTTPRequestHandler, HTTPServer
import os, json
from model.model import TinyCharRNN
from model.tokenizer import CharTokenizer

def build_tokenizer():
    tok_path = os.path.join("weights","tokenizer.json")
    if os.path.exists(tok_path):
        data=json.load(open(tok_path,"r",encoding="utf-8"))
        t=CharTokenizer("")
        t.stoi={k:int(v) if isinstance(v,str) and v.isdigit() else v for k,v in data["stoi"].items()}
        t.itos={int(k):v for k,v in data["itos"].items()}
        print("[tokenizer] loaded from saved vocab")
        return t
    texts=[]
    for n in os.listdir("data"):
        if n.endswith(".txt"):
            with open(os.path.join("data",n),encoding="utf-8") as f: texts.append(f.read())
    print("[tokenizer] built from data folder")
    return CharTokenizer("\n\n".join(texts))

tokenizer=build_tokenizer()
model=TinyCharRNN(len(tokenizer.stoi))
try:
    model=TinyCharRNN.load(os.path.join("weights","model.json"))
    print("[model] weights loaded")
except Exception as e:
    print("[model] using random weights:",e)

class ChatHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/","/index.html"): self.path="static/index.html"
        elif self.path=="/style.css": self.path="static/style.css"
        elif self.path=="/main.js": self.path="static/main.js"
        return super().do_GET()
    def do_POST(self):
        if self.path!="/chat": return self.send_error(404)
        ln=int(self.headers.get("Content-Length",0))
        body=json.loads(self.rfile.read(ln))
        msg=body.get("message","")
        reply = model.generate(
            tokenizer,
            seed=msg,
            max_new=160,
            temperature=0.8,
            top_k=50
        )
        self.send_response(200)
        self.send_header("Content-Type","application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"response":reply}).encode("utf-8"))

def run():
    port=8000
    print(f"ARES_AI → http://localhost:{port}")
    HTTPServer(("0.0.0.0",port),ChatHandler).serve_forever()

if __name__=="__main__":
    run()
