"""Requirement 4: eval generator. Builds the test set FROM THE DOCUMENTS with prompts in prompts/evalgen_*.txt.
Because the judge will trust these cases, every case is validated in code before it is kept:
  - key facts must literally appear in the source passage(s)  (kills hallucinated expectations)
  - multi-document cases must have a fact verified in EACH passage
  - near-miss refusal cases are dropped if the documents seem to answer them
  - duplicates and answer-leaking questions are dropped
Dropped candidates are logged, so the report can say how much of the generator's output was rejected."""
import json, os, random
from collections import Counter
from .docs import chunk_all
from .retrieval import Retriever
from .guard import gather, union_coverage
from .prompts import render
from .textmatch import fact_in_text, tokens


def _dup(q, seen):
    """Near-duplicate = high word overlap AND the same numbers/IDs. 'price of SP-1001' vs 'price of SP-1002' are different tests."""
    a = set(tokens(q))
    nums = {t for t in a if t[0].isdigit()}
    return any(a and len(a & s) / len(a | s) >= 0.7 and nums == {t for t in s if t[0].isdigit()} for s in seen)


def _cycle(chunks, rng, min_words=25):
    good = [c for c in chunks if len(c.text.split()) >= min_words] or chunks
    by_doc = {}
    for c in good:
        by_doc.setdefault(c.source, []).append(c)
    for v in by_doc.values():
        rng.shuffle(v)
    while True:
        for src in sorted(by_doc):
            v = by_doc[src]
            v.append(v.pop(0))
            yield v[-1]


def _evidence(hits):
    """All retrieved text per source file (several chunks can come from one document)."""
    ev = {}
    for c, _ in hits:
        ev[c.source] = (ev[c.source] + "\n" + c.text) if c.source in ev else c.text
    return ev


class Gen:
    def __init__(self, req, docs, llm, seed=7):
        self.req, self.llm, self.rng = req, llm, random.Random(seed)
        self.chunks = chunk_all(docs, 150, 30)
        self.retr = Retriever(self.chunks)
        self.cases, self.seen, self.dropped = [], [], Counter()
        self.counts, self.docs = Counter(), docs

    def _llm(self, prompt):
        try:
            r = self.llm.json([{"role": "user", "content": prompt}], temperature=0.4)
            return r if isinstance(r, dict) else {}
        except Exception:
            self.dropped["llm_error"] += 1
            return {}

    def _add(self, typ, q, facts, sources, ref, evidence, behavior="answer", **extra):
        self.counts[typ] += 1
        self.seen.append(set(tokens(q)))
        self.cases.append({"id": f"{typ[:3]}-{self.counts[typ]:02d}", "type": typ, "question": q.strip(),
                           "expected_behavior": behavior, "key_facts": facts, "expected_sources": sources,
                           "reference_answer": ref, "evidence": evidence, **extra})

    def _check(self, r, texts):
        if not r or r.get("skip") or not str(r.get("question", "")).strip():
            self.dropped["skipped_by_model"] += 1
            return None
        q = str(r["question"])
        facts = [str(f) for f in r.get("key_facts", []) if str(f).strip()][:3]
        if not facts:
            self.dropped["no_key_facts"] += 1
            return None
        if _dup(q, self.seen):
            self.dropped["duplicate_question"] += 1
            return None
        alltext = " ".join(texts)
        if not all(fact_in_text(f, alltext) for f in facts):
            self.dropped["fact_not_in_source"] += 1
            return None
        if all(fact_in_text(f, q) for f in facts):
            self.dropped["question_leaks_answer"] += 1
            return None
        return q, facts, str(r.get("reference_answer", ""))

    def answerable(self, n, style):
        typ = "answerable" if style == "direct" else "ambiguous"
        it, tries = _cycle(self.chunks, self.rng), 0
        while self.counts[typ] < n and tries < n * 4:
            tries += 1
            c = next(it)
            r = self._llm(render("evalgen_answerable", source=c.source, passage=c.text, style=style))
            ok = self._check(r, [c.text])
            if ok:
                self._add(typ, ok[0], ok[1], [c.source], ok[2], {c.source: c.text})

    def multi(self, n):
        it, tries = _cycle(self.chunks, self.rng), 0
        while self.counts["multi_doc"] < n and tries < n * 4:
            tries += 1
            a = next(it)
            ranked = self.retr.search(a.text, top_k=len(self.chunks))
            b = next((c for c, _ in ranked if c.source != a.source), None)
            if not b:
                return
            r = self._llm(render("evalgen_multi", source_a=a.source, passage_a=a.text, source_b=b.source, passage_b=b.text))
            ok = self._check(r, [a.text, b.text])
            if ok and any(fact_in_text(f, a.text) for f in ok[1]) and any(fact_in_text(f, b.text) for f in ok[1]):
                self._add("multi_doc", ok[0], ok[1], sorted({a.source, b.source}), ok[2], {a.source: a.text, b.source: b.text})
            elif ok:
                self.dropped["multi_not_spanning_both"] += 1

    def partial(self, n):
        for topic, note in list(self.req.get("partial_notes", {}).items())[:n]:
            hits = gather(self.retr, topic + " " + note)
            r = self._llm(render("evalgen_partial", note=note, passages="\n\n".join(f"[{c.source}]\n{c.text}" for c, _ in hits)))
            ok = self._check(r, [c.text for c, _ in hits])
            if ok and str(r.get("missing_item", "")).strip():
                self._add("partial", ok[0], ok[1], sorted({c.source for c, _ in hits if any(fact_in_text(f, c.text) for f in ok[1])}),
                          ok[2], _evidence(hits), missing_item=str(r["missing_item"]))

    def refuse(self, n):
        req = self.req
        r = self._llm(render("evalgen_refuse", scope=req["scope"], n=n + 4,
                             topics="; ".join(t["topic"] for t in req["must_answer_topics"]),
                             refuse="; ".join(req["must_refuse_topics"])))
        for c in r.get("cases", []) if isinstance(r.get("cases"), list) else []:
            if self.counts["out_of_scope"] >= n:
                break
            q = str(c.get("question", "")).strip() if isinstance(c, dict) else ""
            if not q or _dup(q, self.seen):
                self.dropped["refuse_empty_or_duplicate"] += 1
                continue
            cov, _ = union_coverage(q, gather(self.retr, q)[:1])  # best-chunk coverage
            if str(c.get("category")) == "near_miss" and cov >= 0.7:
                self.dropped["near_miss_maybe_answerable"] += 1
                continue
            self._add("out_of_scope", q, [], [], "", {}, behavior="refuse", category=str(c.get("category", "")))


def generate_evalset(req, docs, llm, n=24, seed=7):
    g = Gen(req, docs, llm, seed)
    n_ref = 6
    n_part = min(2, len(req.get("partial_notes", {})))
    n_multi = 4 if len({d.name for d in docs}) > 1 else 0
    n_amb = 3
    n_ans = max(n - n_ref - n_part - n_multi - n_amb, 8)
    g.answerable(n_ans, "direct"); g.answerable(n_amb, "ambiguous"); g.multi(n_multi); g.partial(n_part); g.refuse(n_ref)
    return {"meta": {"generated_with": llm.model, "seed": seed, "n_cases": len(g.cases), "target": n,
                     "by_type": dict(Counter(c["type"] for c in g.cases)),
                     "prompts": [f"prompts/evalgen_{p}.txt" for p in ("answerable", "multi", "partial", "refuse")],
                     "dropped_by_validation": dict(g.dropped)}, "cases": g.cases}


def run_evalgen(req_path, out_dir, llm, n=24):
    from .docs import load_folder
    req = json.load(open(req_path, encoding="utf-8"))
    es = generate_evalset(req, load_folder(req["docs_path"]), llm, n)
    os.makedirs(out_dir, exist_ok=True)
    json.dump(es, open(os.path.join(out_dir, "eval_set.json"), "w", encoding="utf-8"), indent=2)
    m = es["meta"]
    print(f"[evalgen] {m['n_cases']} cases (target {m['target']}) by type {m['by_type']}; rejected by validation: {m['dropped_by_validation']}")
    if m["n_cases"] < 20:
        print("   WARNING: fewer than 20 valid cases; add documents or re-run with a larger --n")
    return es
