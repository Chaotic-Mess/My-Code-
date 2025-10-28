class CharTokenizer:
    def __init__(self,text):
        chars=sorted(list(set(text)))
        self.stoi={ch:i for i,ch in enumerate(chars)}
        self.itos={i:ch for ch,i in self.stoi.items()}
    def encode(self,text): return [self.stoi[c] for c in text if c in self.stoi]
    def decode(self,tokens): return ''.join(self.itos.get(t,'?') for t in tokens)
