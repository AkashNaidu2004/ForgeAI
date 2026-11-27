"""Deterministic text matching shared by the eval generator (verify facts) and the judge (find facts in answers)."""
import re
from itertools import product

STOPW = set("the a an of to and or in on for is are was were be by with as at it this that from per".split())
_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")


def canon_num(s):
    s = s.replace(",", "")
    try:
        f = float(s)
    except ValueError:
        return s
    return format(f, "f").rstrip("0").rstrip(".") if "." in format(f, "f") else format(f, "f")


def normalize(text):
    t = text.lower().replace("%", " percent ").replace("$", " ")
    t = _NUM.sub(lambda m: " " + canon_num(m.group(0)) + " ", t)
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9. ]", " ", t)).strip()


def _stem(w):
    for suf in ("ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[:-len(suf)]
    return w


def tokens(text):
    return [t.rstrip(".") for t in re.findall(r"\d+(?:\.\d+)?|[a-z]+", normalize(text))]


def numbers_in(text):
    return [canon_num(m.group(0)) for m in _NUM.finditer(text)]


def fact_in_text(fact, text, cov=0.8):
    """True if every number in `fact` appears in `text` and >= cov of its other content words do (stemmed)."""
    ft = [t for t in tokens(fact) if t not in STOPW]
    if not ft:
        return False
    tt = set(tokens(text))
    ts = {_stem(t) for t in tt}
    nums = [t for t in ft if t[0].isdigit()]
    words = [t for t in ft if not t[0].isdigit()]
    if any(n not in tt for n in nums):
        return False
    return not words or sum(_stem(w) in ts for w in words) / len(words) >= cov


def norm_ws(s):
    return normalize(s)


def _ops(a, b):
    out = [a + b, a - b, b - a, a * b, a * b / 100, a * (1 - b / 100), a * (1 + b / 100)]
    if b:
        out.append(a / b)
    if a:
        out.append(b / a)
    return out


def derivable(x, base, small):
    """Can number x be produced by one or two arithmetic steps (incl. percentages) from numbers in play?
    Lets the judge accept '79,800' derived from 200*420 and a 5% discount, while flagging invented figures."""
    base = [float(b) for b in list(dict.fromkeys(base))[:80]]
    small = [float(s) for s in dict.fromkeys(small)][:12]
    close = lambda v: abs(v - x) <= max(0.011, abs(x) * 1e-6)
    for a, b in product(base, base):
        if any(close(v) for v in _ops(a, b)):
            return True
    l1 = {v for a, b in product(small, small) for v in _ops(a, b)} | set(small)
    for a, b in product(l1, base):
        if any(close(v) for v in _ops(a, b)):
            return True
    return False
