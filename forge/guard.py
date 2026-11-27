"""Requirement 2: groundability guard.
For every capability the brief asks for we combine signals so no single weak model decides alone:
  1. deterministic word coverage across the retrieved passages
  2. an LLM verdict (full / partial / none), preferably from the judge model (different family)
  3. verification that the LLM's quoted evidence really exists in the cited source (catches invented evidence)
Design points (learned from live runs):
  - passages are gathered from EVERY document, not just the global top-k, so cross-document capabilities
    (price from a catalogue + discount from supplier terms) can be judged
  - verdicts are conditional: "partial" keeps the capability and records exactly what is / is not supported
  - tool-aware: if the agent will have a calculator or lookup tool, arithmetic over supported inputs is supported
  - only "none" rejects a capability; rejected capabilities go to must_refuse and the scope is repaired"""
import re
from .docs import chunk_all
from .retrieval import Retriever, STOP, tok
from .prompts import render

LEVEL = {"none": 0, "partial": 1, "full": 2}
NAME = {0: "ungrounded", 1: "partial", 2: "grounded"}

TOOL_NOTES = {
    "needs_calculation": "The assistant WILL have a calculator tool. Arithmetic (totals, percentages, discounts, unit costs) is supported as long as the input numbers, rates and rules are present in the passages; the final number need not be written down.",
    "needs_table_lookup": "The assistant WILL have a lookup tool for CSV tables. Looking up rows is supported if the table's documents appear among the passages.",
    "needs_current_date": "The assistant WILL have a current-date tool. Date arithmetic is supported as long as the rules (deadlines, periods) are in the passages.",
}


def _words(s):
    return re.findall(r"[a-z0-9]+", s.lower().replace("%", " percent "))


def evidence_verified(evidence, source, hits):
    et = _words(evidence)
    if len(et) < 3:
        return False
    pool = set()
    for c, _ in hits:
        if not source or c.source == source:
            pool |= set(_words(c.text))
    return bool(pool) and sum(w in pool for w in et) / len(et) >= 0.8


def gather(retr, query, k=4, per_doc_min=0.05, cap=7):
    """Top-k overall PLUS the best chunk of every other document that matches at all."""
    ranked = retr.search(query, top_k=len(retr.chunks))
    chosen, seen = ranked[:k], {c.source for c, _ in ranked[:k]}
    for c, s in ranked[k:]:
        if c.source not in seen and s >= per_doc_min and len(chosen) < cap:
            chosen.append((c, s))
            seen.add(c.source)
    return chosen


def union_coverage(query, hits):
    qt = set(tok(query))
    if not qt:
        return 0.0, 0.0
    pool = set()
    best = 0.0
    for c, _ in hits:
        ct = set(tok(c.text))
        pool |= ct
        best = max(best, len(qt & ct) / len(qt))
    return len(qt & pool) / len(qt), best


def _stem(w):
    for suf in ("ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[:-len(suf)]
    return w


def _terms(text):
    return {_stem(w) for w in re.findall(r"[a-z]+", text.lower()) if w not in STOP and len(w) > 2}


def repair_scope(req, gaps, llm):
    """The scope was written BEFORE the coverage check, so it may still promise rejected capabilities.
    Rewrite it, then verify in code that no term unique to a rejected topic survives; else use a template."""
    kept = req["must_answer_topics"]
    kept_terms = set().union(*[_terms(t["topic"] + " " + t["example_question"]) for t in kept]) if kept else set()
    bad = set().union(*[_terms(g["topic"] + " " + g["example_question"]) for g in gaps]) - kept_terms
    r = llm.json([{"role": "user", "content": render(
        "scope_repair", scope=req["scope"], audience=req["audience"],
        kept="\n".join(f"- {t['topic']}" for t in kept), removed="\n".join(f"- {g['topic']}" for g in gaps))}], temperature=0)
    new, method = str(r.get("scope", "")).strip(), "llm"
    if not new or (_terms(new) & bad):
        new = f"Answers questions from {req['audience']} about: {', '.join(t['topic'] for t in kept)}. Anything else is declined."
        method = "deterministic fallback (model scope still mentioned a removed capability)"
    req["original_scope"], req["scope"] = req["scope"], new
    req["scope_repair"] = {"method": method, "removed": [g["topic"] for g in gaps]}


def check_topic(t, retr, llm, signals=None):
    query = f"{t['topic']} {t['example_question']}"
    hits = gather(retr, query)
    cov, best = union_coverage(query, hits)
    passages = "\n\n".join(f"[{c.source}]\n{c.text}" for c, _ in hits)
    note = "\n".join(TOOL_NOTES[k] for k in TOOL_NOTES if (signals or {}).get(k))
    v = llm.json([{"role": "user", "content": render("ground_check", topic=t["topic"], question=t["example_question"],
                                                     passages=passages, tools_note=note)}], temperature=0)
    lv = LEVEL.get(str(v.get("verdict", "none")).lower(), 0)
    evidence, source = str(v.get("evidence", "")), str(v.get("source", "")).strip("[] ")
    verified = evidence_verified(evidence, source, hits)
    final = lv
    if lv == 2 and not verified:
        final = 1                       # claimed full support with no checkable quote
    if lv == 1 and not verified and cov < 0.3:
        final = 0                       # weak claim, no quote, little word overlap: reject
    if cov < 0.2:
        final = min(final, 1)
    if cov < 0.1:
        final = 0
    return {"topic": t["topic"], "example_question": t["example_question"], "verdict": NAME[final],
            "llm_verdict": NAME[lv], "coverage": round(cov, 2), "coverage_best_chunk": round(best, 2),
            "sources_seen": sorted({c.source for c, _ in hits}), "evidence": evidence, "evidence_verified": verified,
            "source": source, "supported": str(v.get("supported", "")), "missing": str(v.get("missing", ""))}


def check_groundability(req, docs, llm):
    retr = Retriever(chunk_all(docs, 150, 30))
    signals = req.get("tool_signals", {})
    results = [check_topic(t, retr, llm, signals) for t in req["must_answer_topics"]]
    keep, notes, gaps = [], {}, []
    for t, r in zip(req["must_answer_topics"], results):
        if r["verdict"] == "ungrounded":
            gaps.append(r)
            req["must_refuse_topics"].append(f"{t['topic']} (not covered by the documents)")
        else:
            keep.append(t)
            if r["verdict"] == "partial":
                notes[t["topic"]] = (f"supported: {r['supported'] or 'see documents'}; "
                                     f"NOT supported (say so, never guess): {r['missing'] or 'unspecified'}")
    n = max(len(results), 1)
    score = sum(1.0 if r["verdict"] == "grounded" else 0.5 if r["verdict"] == "partial" else 0 for r in results) / n
    status = "blocked" if score < 0.4 else "ok" if all(r["verdict"] == "grounded" for r in results) else "partial"
    req["must_answer_topics"] = keep
    req["partial_notes"] = notes
    req["ungrounded_requests"] = gaps
    if gaps and keep:
        repair_scope(req, gaps, llm)
    req["groundability"] = {"status": status, "score": round(score, 2), "topics": results}
    return req
