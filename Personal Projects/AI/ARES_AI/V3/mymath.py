# mymath.py — tiny math helpers (pure Python)
import math, random

def randn_matrix(r, c, scale=0.05):
    return [[(random.random()*2-1)*scale for _ in range(c)] for _ in range(r)]

def zeros_matrix(r, c): return [[0.0]*c for _ in range(r)]
def zeros_vec(n): return [0.0]*n

def matvec(M, v):
    out = [0.0]*len(M)
    for i,row in enumerate(M):
        s = 0.0
        for a,b in zip(row, v): s += a*b
        out[i] = s
    return out

def vecTmat(v, M):  # v^T * M  -> vector length = len(M[0])
    out = [0.0]*len(M[0])
    for j in range(len(M[0])):
        s = 0.0
        for i in range(len(M)):
            s += v[i]*M[i][j]
        out[j] = s
    return out

def add_inplace(a, b):
    for i in range(len(a)): a[i] += b[i]

def tanh(v):  return [math.tanh(x) for x in v]
def dtanh(h): return [1.0 - x*x for x in h]  # h already tanh'ed

def softmax(logits):
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s = sum(exps)
    return [e/s for e in exps]

def cross_entropy(probs, target_idx, eps=1e-9):
    return -math.log(probs[target_idx] + eps)

def clip_vec(v, th=1.0):
    for i,x in enumerate(v):
        if x >  th: v[i] =  th
        if x < -th: v[i] = -th

def clip_mat(M, th=1.0):
    for row in M: clip_vec(row, th)
