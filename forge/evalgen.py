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
from .llm import RateLimitError
from .budget import InfraError


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
    def __init__(self, req, docs, llm, seed=7, verify=True):
        self.req, self.llm, self.rng = req, llm, random.Random(seed)
        self.chunks = chunk_all(docs, 150, 30)
        self.retr = Retriever(self.chunks)
        self.cases, self.seen, self.dropped = [], [], Counter()
        self.counts, self.docs = Counter(), docs
        if not self.chunks:
            raise ValueError("No document chunks available for evaluation generation")
        self.verify = verify
        self.audit = []
        self.checkpoint = None
        self.calls = 0
        self.topics = [t["topic"] for t in req.get("must_answer_topics", [])]
        self.covered = Counter()

    def _llm(self, prompt):
        self.calls += 1
        if self.verify:
            print(f"[evalgen] model call {self.calls}; accepted {len(self.cases)}", flush=True)
        try:
            r = self.llm.json([{"role": "user", "content": prompt}], temperature=0.4, cache_salt=self.calls)
            return r if isinstance(r, dict) else {}
        except InfraError:
            raise
        except Exception:
            self.dropped["llm_error"] += 1
            return {}

    def _add(self, typ, q, facts, sources, ref, evidence, behavior="answer", **extra):
        self.counts[typ] += 1
        topic = getattr(self, "pending_topic", None) if typ != "out_of_scope" else None
        if isinstance(topic, str) and topic in self.topics:
            self.covered[topic] += 1
            extra["topic"] = topic
        self.pending_topic = None
        self.seen.append(set(tokens(q)))
        self.cases.append({"id": f"{typ[:3]}-{self.counts[typ]:02d}", "type": typ, "question": q.strip(),
                           "expected_behavior": behavior, "key_facts": facts, "expected_sources": sources,
                           "reference_answer": ref, "evidence": evidence, **extra})

        if self.checkpoint:
            self.checkpoint(self)

    def _check(self, r, texts, verdict=None, preflight=False):
        if not r or r.get("skip") or not str(r.get("question", "")).strip():
            self.dropped["skipped_by_model"] += 1
            return None
        q = str(r["question"])
        facts = [f for f in r.get("key_facts", []) if isinstance(f, str) and f.strip()][:3] if isinstance(r.get("key_facts"), list) else []
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
        ref = str(r.get("reference_answer", ""))
        if preflight:
            return q, facts, ref
        if self.verify:
            v = verdict if verdict is not None else self._llm(render("evalgen_verify", question=q, facts=json.dumps(facts), reference=ref,
                passages=alltext, scope=self.req["scope"], topics=json.dumps(self.topics),
                prior=json.dumps([{"question": c["question"], "facts": c["key_facts"]} for c in self.cases][-12:]),
                missing=r.get("missing_item", "")))
            self.audit.append({"question": q, "verification": v})
            if v.get("valid") is not True or v.get("duplicate") is not False:
                self.dropped["verification_rejected"] += 1
                return None
            revised = v.get("key_facts")
            if not isinstance(revised, list) or not revised or not all(isinstance(f, str) and f.strip() and fact_in_text(f, alltext) for f in revised):
                self.dropped["verification_invalid_facts"] += 1
                return None
            facts = revised[:3]
            ref = v.get("reference_answer", "")
            if not isinstance(ref, str) or not ref.strip():
                self.dropped["verification_invalid_reference"] += 1
                return None
            if not all(fact_in_text(f, ref) for f in facts):
                self.dropped["reference_missing_fact"] += 1
                return None
            if all(fact_in_text(f, q) for f in facts):
                self.dropped["question_leaks_answer"] += 1
                return None
            self.pending_topic = v.get("topic")
        return q, facts, ref

    def batch(self, typ, n, evidence, prompt):
        """Small independent batches, with indexed verification and unchanged fact checks."""
        prior = [{"question": c["question"], "facts": c["key_facts"]} for c in self.cases]
        raw = self._llm(render("evalgen_batch", task=prompt, count=min(4, n + 1), prior=json.dumps(prior)))
        candidates = raw.get("cases", [])
        if not isinstance(candidates, list):
            return
        pending = []
        for r in candidates[:4]:
            if not isinstance(r, dict):
                continue
            ok = self._check(r, list(evidence.values()), preflight=True)
            if ok:
                pending.append(r)
        if not pending:
            return
        audit = self._llm(render("evalgen_verify_batch", scope=self.req["scope"], topics=json.dumps(self.topics),
            passages=json.dumps(evidence), prior=json.dumps(prior), candidates=json.dumps([dict(r, index=i) for i,r in enumerate(pending)])))
        verdicts = audit.get("verdicts", [])
        if not isinstance(verdicts, list):
            return
        for i, r in enumerate(pending):
            if n <= 0:
                break
            matches = [v for v in verdicts if isinstance(v, dict) and type(v.get("index")) is int and v["index"] == i]
            if len(matches) != 1:
                self.dropped["missing_batch_verdict"] += 1
                continue
            ok = self._check(r, list(evidence.values()), verdict=matches[0])
            if not ok:
                continue
            q, facts, ref = ok
            if typ == "multi_doc" and (len(evidence) < 2 or not all(any(fact_in_text(f,t) and not any(fact_in_text(f,u) for k,u in evidence.items() if k != src) for f in facts) for src,t in evidence.items())):
                self.dropped["multi_not_spanning_both"] += 1
                continue
            self._add(typ, q, facts, sorted(evidence), ref, evidence)
            n -= 1
        print(f"[evalgen] batch complete: {typ}; accepted {len(self.cases)}; rejected {dict(self.dropped)}", flush=True)

    def batched_answerable(self, n, style):
        typ = "answerable" if style == "direct" else "ambiguous"
        it = _cycle(self.chunks, self.rng)
        for _ in range(max(4, n)):
            if self.counts[typ] >= n:
                break
            c = next(it)
            topics = sorted(self.topics, key=lambda t: self.covered[t])
            self.batch(typ, n-self.counts[typ], {c.source:c.text}, render("evalgen_answerable", source=c.source, passage=c.text, style=style, topic="; ".join(topics), scope=self.req["scope"]))

    def batched_multi(self, n):
        it = _cycle(self.chunks, self.rng)
        for _ in range(max(4,n*2)):
            if self.counts["multi_doc"] >= n:
                break
            a = next(it)
            b = next((c for c,_ in self.retr.search(a.text, top_k=len(self.chunks)) if c.source != a.source), None)
            if b is None:
                return
            evidence = {a.source:a.text,b.source:b.text}
            prior = [{"question":c["question"],"facts":c["key_facts"]} for c in self.cases]
            raw = self._llm(render("evalgen_evidence_multi", count=min(4,n-self.counts["multi_doc"]), scope=self.req["scope"], a=a.text,b=b.text,prior=json.dumps(prior)))
            pending = []
            for r in raw.get("cases", [])[:4] if isinstance(raw.get("cases"),list) else []:
                if not isinstance(r,dict):
                    continue
                fa,fb = r.get("fact_a"),r.get("fact_b")
                if not all(isinstance(f,str) and len(f.split()) >= 3 and f in t for f,t in ((fa,a.text),(fb,b.text))) or fact_in_text(fa,b.text) or fact_in_text(fb,a.text):
                    self.dropped["invalid_evidence_pair"] += 1
                    continue
                candidate = {"question":r.get("question",""),"key_facts":[fa,fb],"reference_answer":fa+"; "+fb}
                if self._check(candidate,[a.text,b.text],preflight=True):
                    pending.append(candidate)
            if pending:
                audit = self._llm(render("evalgen_evidence_audit", scope=self.req["scope"], prior=json.dumps(prior), candidates=json.dumps([dict(r,index=i) for i,r in enumerate(pending)])))
                verdicts = audit.get("verdicts",[])
                for i,r in enumerate(pending):
                    if self.counts["multi_doc"] >= n:
                        break
                    matches = [v for v in verdicts if isinstance(v,dict) and type(v.get("index")) is int and v["index"]==i] if isinstance(verdicts,list) else []
                    self.audit.append({"question":r["question"],"verification":matches})
                    if len(matches)!=1 or matches[0].get("valid") is not True or matches[0].get("duplicate") is not False:
                        self.dropped["verification_rejected"] += 1
                        continue
                    if not self._check(r,[a.text,b.text],preflight=True):
                        continue
                    if any(fact_in_text(f,r["question"]) for f in r["key_facts"]):
                        self.dropped["question_leaks_answer"] += 1
                        continue
                    self._add("multi_doc",r["question"],r["key_facts"],sorted(evidence),r["reference_answer"],evidence)
            print(f"[evalgen] evidence batch: multi_doc; accepted {len(self.cases)}; rejected {dict(self.dropped)}",flush=True)

    def answerable(self, n, style):
        typ = "answerable" if style == "direct" else "ambiguous"
        it, tries = _cycle(self.chunks, self.rng), 0
        while self.counts[typ] < n and tries < n * 2:
            tries += 1
            c = next(it)
            topic = min(self.topics, key=lambda t: self.covered[t]) if self.topics else ""
            if self.verify and topic and tries % 2:
                hits = gather(self.retr, topic)
                if hits:
                    c = hits[0][0]
            r = self._llm(render("evalgen_answerable", source=c.source, passage=c.text, style=style, topic=topic, scope=self.req["scope"]))
            ok = self._check(r, [c.text])
            if ok:
                self._add(typ, ok[0], ok[1], [c.source], ok[2], {c.source: c.text})

    def multi(self, n):
        it, tries = _cycle(self.chunks, self.rng), 0
        while self.counts["multi_doc"] < n and tries < n * 2:
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


def generate_evalset(req, docs, llm, n=24, seed=7, verify=True, checkpoint=None):
    g = Gen(req, docs, llm, seed, verify=verify)
    g.checkpoint = checkpoint
    n_ref = 6
    n_part = min(2, len(req.get("partial_notes", {})))
    n_multi = 4 if len({d.name for d in docs}) > 1 else 0
    n_amb = 3
    n_ans = max(n - n_ref - n_part - n_multi - n_amb, 8)
    if verify:
        g.batched_multi(n_multi)
        g.batched_answerable(n_amb, "ambiguous")
        g.batched_answerable(n_ans, "direct")
    else:
        g.answerable(n_ans, "direct"); g.answerable(n_amb, "ambiguous"); g.multi(n_multi)
    g.partial(n_part); g.refuse(n_ref)
    return {"meta": {"verification_enabled": g.verify, "verification_audit": g.audit, "topic_coverage": dict(g.covered), "uncovered_topics": [t for t in g.topics if not g.covered[t]], "generated_with": llm.model, "seed": seed, "n_cases": len(g.cases), "target": n,
                     "by_type": dict(Counter(c["type"] for c in g.cases)),
                     "prompts": [f"prompts/evalgen_{p}.txt" for p in ("answerable", "multi", "partial", "refuse", "verify", "batch", "verify_batch", "evidence_multi", "evidence_audit")],
                     "required_types": ["answerable", "ambiguous", "out_of_scope"] + (["multi_doc"] if n_multi else []) + (["partial"] if n_part else []), "model_calls": g.calls, "dropped_by_validation": dict(g.dropped)}, "cases": g.cases}


class EvalCoverageError(RuntimeError):
    pass


def run_evalgen(req_path, out_dir, llm, n=24):
    from .docs import load_folder
    req = json.load(open(req_path, encoding="utf-8"))
    os.makedirs(out_dir, exist_ok=True)
    def checkpoint(g):
        path = os.path.join(out_dir, "eval_partial.json")
        with open(path + ".tmp", "w", encoding="utf-8") as f:
            json.dump({"status": "incomplete", "cases": g.cases, "dropped": dict(g.dropped), "calls": g.calls}, f, indent=2)
        os.replace(path + ".tmp", path)
    es = generate_evalset(req, load_folder(req["docs_path"]), llm, n, checkpoint=checkpoint)
    json.dump(es, open(os.path.join(out_dir, "eval_set.json"), "w", encoding="utf-8"), indent=2)
    m = es["meta"]
    print(f"[evalgen] {m['n_cases']} cases (target {m['target']}) by type {m['by_type']}; rejected by validation: {m['dropped_by_validation']}")
    missing = [t for t in m["required_types"] if not m["by_type"].get(t)]
    if m["n_cases"] < 20 or missing:
        os.replace(os.path.join(out_dir, "eval_set.json"), os.path.join(out_dir, "eval_incomplete.json"))
        raise EvalCoverageError(f"Incomplete evaluation: {m['n_cases']} valid cases; missing categories {missing}. See eval_incomplete.json.")
    return es
