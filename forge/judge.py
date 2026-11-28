"""Requirement 5: judge. Deterministic checks first; the LLM only handles what code cannot (paraphrase, unsupported prose).

Defences against known LLM-judge failure modes:
  position bias      : pointwise grading (no A/B order to bias); any pairwise use must run both orders
  verbosity bias     : claim-by-claim rubric that says length is irrelevant, + canary test with padded wrong answers
  self-preference    : judge model should differ from the agent model (warned in evaluate.py)
  judge hallucination: every 'stated' verdict must carry a quote that verifiably exists in the answer; 'unsupported
                       claims' made of numbers that exist/derive from the material are overruled
  trust calibration  : canary answers (gold / wrong / padded-wrong / fabricated) measure the judge before it scores
                       anything; if it is unreliable the run falls back to deterministic checks only"""
import json, random, re
from .prompts import render
from .budget import InfraError
from .textmatch import fact_in_text, numbers_in, norm_ws, derivable

REFUSAL_RE = re.compile(r"\b(cannot|can't|can not|unable to|not (?:contained|covered|mentioned|available|provided|found|documented|specified)|no information|outside (?:the |my )?scope|don't have|do not have)\b", re.I)
IGNORE_NUMS = {"0", "1"}


def looks_like_refusal(text):
    return bool(REFUSAL_RE.search(text))


def llm_check(case, answer, context_text, llm):
    facts = case.get("key_facts", [])
    missing = case.get("missing_item")
    r = llm.json([{"role": "user", "content": render(
        "judge_case", question=case["question"], answer=answer, context=context_text or "(nothing)",
        facts="\n".join(f"- {f}" for f in facts) or "(none)",
        missing_line=f"MISSING ITEM (the answer must say this is unavailable): {missing}\n" if missing else "")}], temperature=0)
    entries = r.get("facts", []) if isinstance(r.get("facts"), list) else []
    na, stated = norm_ws(answer), {}
    for i, f in enumerate(facts):
        e = next((x for x in entries if isinstance(x, dict) and norm_ws(str(x.get("fact", ""))) == norm_ws(f)),
                 {})
        q = norm_ws(str(e.get("quote", "")))
        stated[f] = e.get("stated") is True and bool(q) and q in na and all(n in numbers_in(q) for n in numbers_in(f))      # quote must really occur in the answer
    ctx_nums = set(numbers_in(context_text))
    claims = []
    for c in r.get("unsupported_claims", []) if isinstance(r.get("unsupported_claims"), list) else []:
        c = str(c).strip()
        nums = [n for n in numbers_in(c) if n not in IGNORE_NUMS]
        if c and not fact_in_text(c, context_text, cov=1.0):         # overrule over-flagging of numbers that exist
            claims.append(c)
    return {"facts": stated, "unsupported_claims": claims, "flags_missing": bool(r.get("flags_missing"))}


def judge_case(case, res, llm=None, doc_names=()):
    ans, flag = res.get("answer", ""), bool(res.get("refused"))
    ctx_items = res.get("context", [])
    ctx_text = "\n".join(x["text"] for x in ctx_items)
    tools = res.get("tool_calls", [])
    tool_text = "\n".join(f"{t['tool']} {json.dumps(t.get('args', {}))} -> {t['result']}" for t in tools)
    material = (ctx_text + "\n" + tool_text).strip()
    cited, sig = list(res.get("sources", [])), []
    out = {"id": case["id"], "type": case["type"], "question": case["question"], "expected": case["expected_behavior"],
           "answer": ans, "refused": flag, "sources": cited, "tool_calls": tools,
           "retrieved": [x["id"] for x in ctx_items], "signals": sig, "judge_used": False}
    if res.get("error"):
        sig.append("error")
        out.update(refusal_ok=False, correct=False, grounded=False, passed=False, error=res["error"])
        return out

    if case["expected_behavior"] == "refuse":
        if not flag:
            sig.append("answered_unanswerable")
            if looks_like_refusal(ans):
                sig.append("refusal_text_without_flag")
        out.update(refusal_ok=flag, correct=None, grounded=None, passed=flag)
        return out

    facts = case["key_facts"]
    exp_src = case.get("expected_sources", [])
    retrieved_src = {x["source"] for x in ctx_items}
    out["retrieval_hit"] = all(s in retrieved_src for s in exp_src)
    out["facts_in_context"] = round(sum(fact_in_text(f, material) for f in facts) / max(len(facts), 1), 2)
    if flag:
        sig.append("wrong_refusal")
        out.update(refusal_ok=False, correct=False, grounded=True, source_ok=False, passed=False, fact_recall=0.0)
        return out

    det = {f: fact_in_text(f, ans) for f in facts}
    stated = dict(det)
    need_llm = llm is not None
    chk = None
    if need_llm:
        try:
            chk = llm_check(case, ans, material, llm)
            out["judge_used"] = True
            for f in facts:
                stated[f] = stated[f] or chk["facts"].get(f, False)
        except InfraError:
            raise                      # do not silently downgrade to deterministic-only scoring
        except Exception as e:
            sig.append(f"judge_error:{type(e).__name__}")
    correct = all(stated.values()) and bool(facts)
    out["fact_recall"] = round(sum(stated.values()) / max(len(facts), 1), 2)
    out["facts_stated"] = stated

    missing_ok = True
    if case.get("missing_item"):
        missing_ok = bool(chk and chk["flags_missing"]) if chk is not None else (looks_like_refusal(ans) and fact_in_text(case["missing_item"], ans))
        if not missing_ok:
            sig.append("missing_item_not_flagged")
    bad_cite = [c for c in cited if doc_names and c not in doc_names]
    source_ok = all(s in cited for s in exp_src)
    if bad_cite:
        sig.append("invented_citation")
    if not source_ok:
        sig.append("expected_source_not_cited")

    small = numbers_in(case["question"] + " " + tool_text)
    base = small + numbers_in(ctx_text)       # question/fact/tool numbers first, then every number in the retrieved context
    suspects = [n for n in numbers_in(ans) if n not in IGNORE_NUMS and n not in set(base)
                and not derivable(float(n), base, base)]
    if suspects:
        sig.append("unsupported_numbers")
    claims = chk["unsupported_claims"] if chk else []
    if claims:
        sig.append("unsupported_claims")
    if not correct:
        sig.append("facts_missing_from_answer")
    grounded = not suspects and not claims and not bad_cite
    out.update(refusal_ok=True, correct=correct, grounded=grounded, source_ok=source_ok,
               unsupported_numbers=suspects, unsupported_claims=claims,
               passed=bool(correct and grounded and source_ok and missing_ok))
    return out


# ---------------------------------------------------------------- canaries
FILLER = ("To give you the fullest possible picture, it is worth noting that organisations maintain detailed policies "
          "so that every member of staff is treated fairly and consistently, and that it is always sensible to review "
          "the relevant documentation carefully, consult colleagues where appropriate, and keep a record of any decision. ") * 2


def run_canaries(cases, llm, n=3, seed=3):
    """Measure the judge itself on answers whose verdict we know. Tests the LLM part of the judge, not the string matching."""
    pool = [c for c in cases if c["expected_behavior"] == "answer" and c["key_facts"] and c.get("reference_answer") and c.get("evidence")]
    picks = random.Random(seed).sample(pool, min(n, len(pool)))
    rows = []
    for c in picks:
        ctx = "\n\n".join(c["evidence"].values())
        wrong = "According to the documents the answer is 9173, and nothing else applies."
        variants = [("gold", c["reference_answer"]), ("wrong", wrong), ("padded_wrong", wrong + " " + FILLER),
                    ("fabricated", c["reference_answer"] + " Additionally, staff receive a guaranteed annual bonus of 4821 USD every March.")]
        for kind, ans in variants:
            try:
                k = llm_check(c, ans, ctx, llm)
                all_stated = bool(k["facts"]) and all(k["facts"].values())
                ok = {"gold": all_stated and not k["unsupported_claims"], "wrong": not all_stated,
                      "padded_wrong": not all_stated, "fabricated": bool(k["unsupported_claims"])}[kind]
            except InfraError:
                raise
            except Exception:
                ok = False
            rows.append({"case": c["id"], "kind": kind, "correct": ok})
    acc = sum(r["correct"] for r in rows) / max(len(rows), 1)
    by = {}
    for r in rows:
        by.setdefault(r["kind"], []).append(r["correct"])
    return {"accuracy": round(acc, 2), "n": len(rows), "by_kind": {k: round(sum(v) / len(v), 2) for k, v in by.items()},
            "reliable": bool(rows) and acc >= 0.75 and all(sum(v) / len(v) >= 2/3 for v in by.values())}
