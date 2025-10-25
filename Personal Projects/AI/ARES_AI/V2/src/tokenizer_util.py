from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder  # ⬅ add this

class AresTokenizer:
    def __init__(self, path="tokenizer/tokenizer.json"):
        self.tok = Tokenizer.from_file(path)
        # Ensure proper decoding for ByteLevel BPE (fixes the 'Ġ' artifacts)
        try:
            # If a decoder wasn’t saved, attach one now:
            if self.tok.decoder is None:
                self.tok.decoder = ByteLevelDecoder()
        except Exception:
            # older versions don’t expose .decoder as None; forcing is fine
            self.tok.decoder = ByteLevelDecoder()

        def need(tok):
            i = self.tok.token_to_id(tok)
            if i is None: raise ValueError(f"Missing special token: {tok}")
            return i
        self.pad = need("[PAD]")
        self.bos = need("<|bos|>")
        self.eos = need("<|eos|>")
        self.system = need("<|system|>")
        self.user = need("<|user|>")
        self.assistant = need("<|assistant|>")

    @property
    def vocab_size(self): return self.tok.get_vocab_size()

    def encode_ids(self, text, add_special=True):
        ids = self.tok.encode(text).ids
        return ([self.bos] + ids + [self.eos]) if add_special else ids

    def decode(self, ids):
        # Drop BOS/EOS/PAD etc. when turning ids back into text
        return self.tok.decode(ids, skip_special_tokens=True)
