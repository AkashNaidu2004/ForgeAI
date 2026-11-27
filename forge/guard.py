"""Requirement 2: groundability guard.
For every capability the brief asks for we combine THREE signals so no single weak model decides alone:
  1. deterministic word coverage in the best retrieved chunk
  2. an LLM verdict (full/partial/none), preferably from the judge model (different family)
  3. verification that the LLM's quoted evidence really exists in the cited source (catches invented evidence)
Unsupported topics are moved to must_refuse so the generated agent cannot hallucinate the gap away."""
import re
from .docs import chunk_all
from .retrieval import Retriever, coverage
from .prompts import render

LEVEL = {"none": 0, "partial": 1, "full": 2}
NAME = {0: "ungrounded", 1: "partial", 2: "grounded"}


def _words(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def evidence_verified(evidence, source, hits):
    et = _words(evidence)
    if len(et) < 3:
        return False
    pool = set()
    for c, _ in hits:
        if not source or c.source == source:
            pool |= set(_words(c.text))
    return bool(pool) and sum(w in pool for w in et) / len(et) >= 0.85


def check_topic(t, retr, llm):
    cov, hits = coverage(retr, f"{t['topic']} {t['example_question']}")
    passages = "\n\n".join(f"[{c.source}]\n{c.text}" for c, _ in hits)
    v = llm.json([{"role": "user", "content": render("ground_check", topic=t["topic"],
                                                     question=t["example_question"], passages=passages)}], temperature=0)
    lv = LEVEL.get(str(v.get("verdict", "none")).lower(), 0)
    evidence, source = str(v.get("evidence", "")), str(v.get("source", "")).strip("[] ")
    verified = evidence_verified(evidence, source, hits)
    final = lv
    if lv and not verified:
        final -= 1                      # claimed support with no checkable quote: downgrade
    if cov < 0.2:
        final = min(final, 1)           # almost none of the topic's words appear anywhere together
    if cov < 0.1:
        final = 0
    return {"topic": t["topic"], "example_question": t["example_question"], "verdict": NAME[final],
            "llm_verdict": NAME[lv], "coverage": round(cov, 2), "evidence": evidence,
            "evidence_verified": verified, "source": source, "missing": str(v.get("missing", ""))}


def check_groundability(req, docs, llm):
    retr = Retriever(chunk_all(docs, 150, 30))
    results = [check_topic(t, retr, llm) for t in req["must_answer_topics"]]
    keep, notes, gaps = [], {}, []
    for t, r in zip(req["must_answer_topics"], results):
        if r["verdict"] == "ungrounded":
            gaps.append(r)
            req["must_refuse_topics"].append(f"{t['topic']} (not covered by the documents)")
        else:
            keep.append(t)
            if r["verdict"] == "partial":
                notes[t["topic"]] = r["missing"] or "only partly covered"
    n = max(len(results), 1)
    score = sum(1.0 if r["verdict"] == "grounded" else 0.5 if r["verdict"] == "partial" else 0 for r in results) / n
    status = "blocked" if score < 0.4 else "ok" if all(r["verdict"] == "grounded" for r in results) else "partial"
    req["must_answer_topics"] = keep
    req["partial_notes"] = notes
    req["ungrounded_requests"] = gaps
    req["groundability"] = {"status": status, "score": round(score, 2), "topics": results}
    return req
