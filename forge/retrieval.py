"""BM25 (pure python, no deps) with optional dense hybrid via Ollama embeddings."""
import math, os, re
from collections import Counter

STOP = set("the a an of to and or in on for is are was were be by with as at it this that from your you i we our can may what how do does".split())


def tok(s):
    return [w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP]


class Retriever:
    def __init__(self, chunks, mode="bm25", k1=1.5, b=0.75):
        self.chunks, self.mode, self.k1, self.b = chunks, mode, k1, b
        self.toks = [tok(c.text) for c in chunks]
        self.df = Counter(w for t in self.toks for w in set(t))
        self.avg = sum(len(t) for t in self.toks) / max(len(self.toks), 1)
        self.emb = [embed(c.text) for c in chunks] if mode == "hybrid" else None

    def _bm25(self, q):
        n, qt = len(self.chunks), tok(q)
        scores = []
        for t in self.toks:
            tf, s = Counter(t), 0.0
            for w in qt:
                if w in tf:
                    idf = math.log(1 + (n - self.df[w] + 0.5) / (self.df[w] + 0.5))
                    s += idf * tf[w] * (self.k1 + 1) / (tf[w] + self.k1 * (1 - self.b + self.b * len(t) / max(self.avg, 1)))
            scores.append(s)
        return scores

    def search(self, query, top_k=4):
        """Returns [(chunk, score 0..1)]. Scores are normalised so thresholds are comparable."""
        sc = self._bm25(query)
        top = max(sc) if sc and max(sc) > 0 else 1.0
        sc = [s / top for s in sc]
        if self.emb is not None:
            qe = embed(query)
            sc = [0.5 * a + 0.5 * cos(qe, e) for a, e in zip(sc, self.emb)]
        idx = sorted(range(len(sc)), key=lambda i: -sc[i])[:top_k]
        return [(self.chunks[i], sc[i]) for i in idx]


def embed(text):
    import requests
    base = os.getenv("FORGE_EMBED_URL", "http://localhost:11434")
    r = requests.post(f"{base}/api/embeddings",
                      json={"model": os.getenv("FORGE_EMBED_MODEL", "nomic-embed-text"), "prompt": text}, timeout=120)
    return r.json()["embedding"]


def cos(a, b):
    import numpy as np
    a, b = np.array(a), np.array(b)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
