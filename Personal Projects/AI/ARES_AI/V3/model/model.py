# model/model.py — minimal char RNN (pure Python, stdlib-only)
import json, random, os
from mymath import (
    randn_matrix, zeros_matrix, zeros_vec, vecTmat,
    add_inplace, tanh, dtanh, softmax, cross_entropy, clip_vec, clip_mat
)

def _safe_save_json(path, data):
    """Write JSON atomically to avoid partial files on Ctrl+C or crashes."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(tmp, path)

class TinyCharRNN:
    """
    One-layer Elman RNN:
      h_t = tanh( E[x_t] + Whh^T @ h_{t-1} + b_h )
      y_t = softmax( Why^T @ h_t + b_y )
    BPTT on short sequences. All pure Python lists.
    """
    def __init__(self, vocab_size, hidden=128, lr=0.03, seed=42):
        random.seed(seed)
        self.vocab_size = vocab_size
        self.hidden = hidden
        self.lr = lr

        # Parameters
        self.E   = randn_matrix(vocab_size, hidden, 0.08)      # embeddings
        self.Whh = randn_matrix(hidden,     hidden, 0.08)      # recurrent (column-major in vecTmat use)
        self.Why = randn_matrix(hidden,     vocab_size, 0.08)  # to logits
        self.bh  = zeros_vec(hidden)
        self.by  = zeros_vec(vocab_size)

    # ---------- forward one step ----------
    def _step(self, idx, h_prev):
        # x embedding
        x = self.E[idx][:]  # copy row
        # h_t = tanh( x + Whh^T @ h_prev + b_h )
        Whh_h = vecTmat(h_prev, self.Whh)  # h_prev^T * Whh
        pre = [xi + wi + bi for xi, wi, bi in zip(x, Whh_h, self.bh)]
        h = tanh(pre)
        # logits = Why^T @ h + b_y
        logits = vecTmat(h, self.Why)
        for j in range(len(logits)):
            logits[j] += self.by[j]
        probs = softmax(logits)
        return h, probs

    # ---------- forward over a sequence ----------
    def forward(self, idx_seq, h0=None):
        h = [0.0]*self.hidden if h0 is None else h0[:]
        hs, ps = [], []
        for idx in idx_seq:
            h, p = self._step(idx, h)
            hs.append(h); ps.append(p)
        return hs, ps

    # ---------- backward (BPTT) ----------
    def train_step(self, idx_seq, tgt_seq):
        # Forward
        hs, ps = self.forward(idx_seq)
        loss = 0.0
        for p, t in zip(ps, tgt_seq):
            loss += cross_entropy(p, t)

        # Grad buffers
        dE   = zeros_matrix(self.vocab_size, self.hidden)
        dWhh = zeros_matrix(self.hidden, self.hidden)
        dWhy = zeros_matrix(self.hidden, self.vocab_size)
        dbh  = zeros_vec(self.hidden)
        dby  = zeros_vec(self.vocab_size)

        dh_next = zeros_vec(self.hidden)

        # BPTT
        for t in reversed(range(len(idx_seq))):
            h = hs[t]
            p = ps[t]
            y = [0.0]*self.vocab_size; y[tgt_seq[t]] = 1.0

            # dL/dlogits = p - y
            dlog = [p[j] - y[j] for j in range(self.vocab_size)]

            # dWhy += h * (p - y)
            for i in range(self.hidden):
                hi = h[i]
                row = dWhy[i]
                for j in range(self.vocab_size):
                    row[j] += hi * dlog[j]
            # dby
            add_inplace(dby, dlog)

            # dh = Why * dlog + dh_next
            dh = [0.0]*self.hidden
            for i in range(self.hidden):
                s = 0.0
                Wi = self.Why[i]
                for j in range(self.vocab_size):
                    s += Wi[j] * dlog[j]
                dh[i] = s + dh_next[i]

            # back through tanh
            dt = dtanh(h)
            dpre = [dh[i] * dt[i] for i in range(self.hidden)]

            # dbh
            add_inplace(dbh, dpre)

            # dWhh += h_{t-1} outer dpre
            h_prev = hs[t-1] if t > 0 else [0.0]*self.hidden
            for i in range(self.hidden):
                hp = h_prev[i]
                row = dWhh[i]
                for j in range(self.hidden):
                    row[j] += hp * dpre[j]

            # dE row for x_t
            idx = idx_seq[t]
            rowE = dE[idx]
            for j in range(self.hidden):
                rowE[j] += dpre[j]

            # propagate to previous hidden
            dh_prev = [0.0]*self.hidden
            for i in range(self.hidden):
                s = 0.0
                for j in range(self.hidden):
                    s += dpre[j] * self.Whh[i][j]
                dh_prev[i] = s
            dh_next = dh_prev

        # Clip to avoid exploding grads
        clip_mat(dWhy, 0.25); clip_mat(dWhh, 0.25); clip_mat(dE, 0.25)
        clip_vec(dbh, 0.25);  clip_vec(dby, 0.25)

        # SGD update
        eta = self.lr
        for i in range(self.hidden):
            row = self.Why[i]
            drow = dWhy[i]
            for j in range(self.vocab_size):
                row[j] -= eta * drow[j]
        for i in range(self.hidden):
            row = self.Whh[i]
            drow = dWhh[i]
            for j in range(self.hidden):
                row[j] -= eta * drow[j]
        for i in range(self.vocab_size):
            row = self.E[i]
            drow = dE[i]
            for j in range(self.hidden):
                row[j] -= eta * drow[j]
        for i in range(self.hidden):
            self.bh[i] -= eta * dbh[i]
        for j in range(self.vocab_size):
            self.by[j] -= eta * dby[j]

        return loss / len(idx_seq)

    # ---------- sampling helpers (temperature + top-k) ----------
    def _pick(self, probs, temperature=1.0, top_k=None):
        """
        Returns an index sampled from `probs` using temperature scaling and optional top-k filtering.
        - temperature <= 0: greedy (argmax)
        - 0 < temperature ~ 0.6..0.9: safer/more coherent
        - temperature > 1.0: more creative/chaotic
        - top_k: restrict to k highest-probability tokens before sampling
        """
        # greedy if temperature<=0
        if temperature is None or temperature <= 0:
            best = 0; bestp = -1.0
            for i, p in enumerate(probs):
                if p > bestp:
                    bestp = p; best = i
            return best

        # temperature scaling directly on probabilities
        T = temperature if temperature > 1e-8 else 1e-8
        scaled = []
        s = 0.0
        for p in probs:
            q = p ** (1.0 / T)
            scaled.append(q); s += q
        invs = 1.0 / s
        for i in range(len(scaled)):
            scaled[i] *= invs

        # optional top-k truncation
        if top_k is not None and 0 < top_k < len(scaled):
            idxs = list(range(len(scaled)))
            idxs.sort(key=lambda i: scaled[i], reverse=True)
            idxs = idxs[:top_k]
            s2 = 0.0
            for i in idxs:
                s2 += scaled[i]
            r = random.random()
            acc = 0.0
            for i in idxs:
                acc += scaled[i] / s2
                if r <= acc:
                    return i
            return idxs[-1]

        # full categorical
        r = random.random()
        acc = 0.0
        for i, p in enumerate(scaled):
            acc += p
            if r <= acc:
                return i
        return len(scaled) - 1

    # ---------- generation ----------
    def generate(self, tokenizer, seed="A", max_new=200, temperature=1.0, top_k=None):
        """
        Generate text continuing from `seed`.
        - tokenizer: must provide encode(str)->List[int], decode(List[int])->str
        - seed: initial text to prime hidden state
        - max_new: number of new tokens to append
        - temperature/top_k: sampling controls (see _pick)
        """
        # prime hidden with the seed (limit to last 64 chars to bound warmup time)
        h = [0.0]*self.hidden
        out = tokenizer.encode(seed)
        for idx in out[-min(len(out), 64):]:
            h, _ = self._step(idx, h)

        # continue
        idx = out[-1] if out else 0
        for _ in range(max_new):
            h, probs = self._step(idx, h)
            idx = self._pick(probs, temperature=temperature, top_k=top_k)
            out.append(idx)
        return tokenizer.decode(out)

    # ---------- persistence ----------
    def save(self, path):
        data = {
            "vocab_size": self.vocab_size, "hidden": self.hidden,
            "E": self.E, "Whh": self.Whh, "Why": self.Why, "bh": self.bh, "by": self.by,
            "lr": self.lr,
        }
        _safe_save_json(path, data)

    @staticmethod
    def load(path):
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        m = TinyCharRNN(d["vocab_size"], d["hidden"], lr=d.get("lr", 0.03))
        m.E, m.Whh, m.Why, m.bh, m.by = d["E"], d["Whh"], d["Why"], d["bh"], d["by"]
        return m
