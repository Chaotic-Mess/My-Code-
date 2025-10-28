import random, math

def zeros(rows, cols): return [[0.0]*cols for _ in range(rows)]

def randn(rows, cols, scale=0.02):
    return [[(random.random()*2-1)*scale for _ in range(cols)] for _ in range(rows)]

def dot(v1, v2): return sum(x*y for x,y in zip(v1, v2))

def matmul(A, B):
    m, n, p = len(A), len(A[0]), len(B[0])
    out = zeros(m, p)
    for i in range(m):
        for j in range(p):
            for k in range(n):
                out[i][j] += A[i][k]*B[k][j]
    return out

def softmax(v):
    e = [math.exp(x - max(v)) for x in v]
    s = sum(e)
    return [x/s for x in e]

def cross_entropy(pred, target_index):
    return -math.log(pred[target_index] + 1e-9)
