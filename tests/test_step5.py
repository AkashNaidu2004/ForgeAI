"""Offline tests for the optimizer + report. Run: python tests/test_step5.py"""
import json, os, sys, tempfile, types
os.environ["FORGE_PROVIDER"] = "mock"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import forge.optimizer as opt
from forge.generator import render_system_prompt
from forge.report import build_report

CORPUS = "Leave allowance is 24 days per year. Sick allowance is 10 days. Meal allowance is 40 USD per day."
tmp = tempfile.mkdtemp()
docs = os.path.join(tmp, "docs"); os.makedirs(docs)
open(os.path.join(docs, "alpha.txt"), "w").write(CORPUS)

# ---------- cases: each kind exercises one triage category
KINDS = {"wrongref": ("24 days", "Staff get 24 days of leave per year."), "retr": ("10 days", "Sick allowance is 10 days."),
         "ok": ("40 USD", "The meal allowance is 40 USD per day."), "defect": ("99 weeks", "It is 5 weeks."),
         "refuse_ok": (None, None), "unans": (None, None)}
plan = ["wrongref"] * 3 + ["retr"] * 2 + ["ok"] * 3 + ["defect"] + ["refuse_ok"] + ["unans"] * 2
cases, kind_of = [], {}
for i, k in enumerate(plan, 1):
    q = f"question {i} about {k}"
    kind_of[q] = k
    fact = KINDS[k][0]
    cases.append({"id": f"c-{i:02d}", "type": "out_of_scope" if fact is None else "answerable", "question": q,
                  "expected_behavior": "refuse" if fact is None else "answer", "key_facts": [fact] if fact else [],
                  "expected_sources": ["alpha.txt"] if fact else [], "reference_answer": "", "evidence": {}})


class StubAgent:
    """Behaviour depends on the definition, so patches can really change the outcome."""
    def __init__(self, d, llm=None):
        self.d, self.llm = d, types.SimpleNamespace(model="stub-agent")
    def answer(self, q):
        sp, k, kind = self.d["system_prompt"], self.d["retrieval"]["top_k"], kind_of[q]
        ctx = [] if (kind == "retr" and k < 3) else [{"id": "alpha#0", "source": "alpha.txt", "text": CORPUS}]
        refuse = {"answer": "Not in the documents.", "sources": [], "refused": True}
        if kind == "refuse_ok" or (kind == "wrongref" and "Refuse only when none" not in sp) or (kind == "retr" and not ctx):
            r = refuse
        elif kind == "unans":
            r = refuse if "unstated" in sp else {"answer": "The bonus is 500 USD.", "sources": [], "refused": False}
        else:
            r = {"answer": KINDS[kind][1], "sources": ["alpha.txt"], "refused": False}
        return dict(r, context=ctx, tool_calls=[])


def critic(m):
    u = m[-1]["content"]
    if "FAILURE KIND: wrong_refusal" in u:
        return json.dumps({"rule": "Always answer 24 days questions from the passages."})         # has digits: must be rejected
    if "FAILURE KIND: answered_unanswerable" in u:
        return json.dumps({"rule": "Refuse whenever the requested detail is unstated in the passages, and say what is missing."})
    return json.dumps({"rule": "Use every relevant passage."})
critic_llm = types.SimpleNamespace(model="critic", json=lambda m, **k: json.loads(critic(m)))
opt.Agent = StubAgent

# ---------- files a real run would have produced
secs = {"role": "You are an HR assistant.", "scope": "Scope: HR", "grounding_rules": "GROUNDING RULES:\n1. Use context only.",
        "refusal_rules": "REFUSE when asked about:\n- salaries"}
d0 = {"version": 0, "name": "hr", "docs_path": docs, "prompt_sections": secs, "system_prompt": render_system_prompt(secs), "tools": [],
      "retrieval": {"chunk_size": 150, "overlap": 30, "top_k": 1, "mode": "bm25", "min_score": 0.0},
      "tool_decisions": [{"tool": "calculator", "reason": "not attached: no evidence in brief/topics"}]}
run = os.path.join(tmp, "run"); os.makedirs(run)
J = lambda n, o: json.dump(o, open(os.path.join(run, n), "w"), indent=2)
J("agent_v0.json", d0)
J("eval_set.json", {"meta": {"n_cases": len(cases), "by_type": {"answerable": 9, "out_of_scope": 3}, "generated_with": "mock",
                              "dropped_by_validation": {"fact_not_in_source": 1}}, "cases": cases})
J("requirements.json", {"agent_name": "hr", "brief": "HR helper", "scope": "HR policy answers", "docs_path": docs,
                        "must_answer_topics": [{"topic": "leave", "example_question": "x"}], "must_refuse_topics": ["flight booking"],
                        "partial_notes": {}, "groundability": {"status": "ok", "topics": [{"topic": "leave", "verdict": "grounded"}]}})

# ---------- A. split is deterministic and disjoint
dev, held = opt.split(cases)
assert dev and held and not ({c["id"] for c in dev} & {c["id"] for c in held}) and len(dev) + len(held) == len(cases)
assert opt.split(cases) == (dev, held)

# ---------- B. triage categories
corpus = CORPUS
base = {"id": "x", "type": "answerable", "question": "q", "answer": "a", "signals": [], "refused": False, "expected": "answer"}
assert opt.triage(dict(base, refused=True, signals=["wrong_refusal"], facts_in_context=1.0), {"key_facts": ["24 days"]}, corpus)[0] == "wrong_refusal"
assert opt.triage(dict(base, refused=True, signals=["wrong_refusal"], facts_in_context=0.0), {"key_facts": ["24 days"]}, corpus)[0] == "retrieval_miss"
assert opt.triage(base, {"key_facts": ["99 weeks"]}, corpus) == ("eval_defect_suspect", False, "an expected key fact does not appear in any document")
assert opt.triage(dict(base, signals=["unsupported_numbers"]), {"key_facts": ["24 days"]}, corpus)[0] == "hallucination"
assert opt.triage(dict(base, signals=["facts_missing_from_answer"], facts_in_context=1.0, fact_recall=0.5, judge_used=False), {"key_facts": ["24 days"]}, corpus)[1] is False
assert opt.triage(dict(base, signals=["facts_missing_from_answer"], facts_in_context=1.0, fact_recall=0.5, judge_used=True), {"key_facts": ["24 days"]}, corpus)[0] == "incomplete_answer"
ref = dict(base, expected="refuse", answer="The bonus is 500 USD.", signals=["answered_unanswerable"])
assert opt.triage(ref, {"key_facts": []}, corpus)[0] == "answered_unanswerable"
assert opt.triage(dict(ref, answer="Leave is 24 days."), {"key_facts": []}, corpus)[0] == "eval_defect_suspect"
assert opt.triage(dict(base, signals=["error"], error="boom"), {"key_facts": []}, corpus)[1] is False

# ---------- C. the loop
log = opt.optimize(os.path.join(run, "agent_v0.json"), os.path.join(run, "eval_set.json"), run, None, agent_llm=critic_llm, log=lambda *a: None)
v0, vf = log["versions"][0], log["versions"][-1]
assert len(log["iterations"]) <= 3
assert vf["overall"]["passed"] > v0["overall"]["passed"], "the optimizer should have fixed genuine defects"
assert vf["overall"]["n"] == v0["overall"]["n"] == len(cases), "official score must keep every case"
dev_ids, held_ids = set(log["split"]["dev"]), set(log["split"]["held_out"])
for it in log["iterations"]:
    for f in it["dev_failures"]:
        assert f["id"] in dev_ids, "optimizer diagnosed a held-out case"
    for p in it.get("patches", []):
        assert set(p["targets"]) <= dev_ids, "a patch targeted a held-out case"
        if "rule" in p:
            assert not any(ch.isdigit() for ch in p["rule"]), "rule contains eval-specific numbers"
patches = [p for it in log["iterations"] for p in it.get("patches", [])]
cats = {p["category"]: p for p in patches}
assert cats["wrong_refusal"]["rule_source"] == "template", "digit-laden critic rule must fall back to the template"
assert "unans" and cats.get("answered_unanswerable", {}).get("rule_source") == "critic"
assert cats["retrieval_miss"]["after"]["top_k"] == 3 and cats["retrieval_miss"]["before"]["top_k"] == 1
defect = [f for f in log["final_failures"] if f["category"] == "eval_defect_suspect"]
assert defect and not defect[0]["genuine_agent_defect"], "suspected eval defect must stay in the score and be listed"
assert not any(p["category"] == "eval_defect_suspect" for p in patches), "the agent must not be patched for a test defect"
fin = json.load(open(os.path.join(run, "agent_final.json")))
assert fin["version"] == log["final_version"] and "Refuse only when none" in fin["system_prompt"] and fin["retrieval"]["top_k"] == 3
assert json.load(open(os.path.join(run, "agent_v0.json")))["retrieval"]["top_k"] == 1, "v0 must be preserved"

# ---------- D. a patch that does not help is rejected and reverted
class Useless(StubAgent):
    def answer(self, q):
        return dict(super().answer(q), answer="Not in the documents.", sources=[], refused=True)
opt.Agent = Useless
run2 = os.path.join(tmp, "run2"); os.makedirs(run2)
log2 = opt.optimize(os.path.join(run, "agent_v0.json"), os.path.join(run, "eval_set.json"), run2, None, agent_llm=critic_llm, log=lambda *a: None)
assert log2["iterations"] and not log2["iterations"][-1].get("accepted", False) or "stopped" in log2["iterations"][-1]["outcome"]
assert log2["final_version"] == 0, "a rejected iteration must be reverted"
opt.Agent = StubAgent

# ---------- E. report
txt = build_report(run, {"calls": 5, "seconds": 1.0})
for needle in ("## 1. Headline", "Held-out cases", "## 7. What still fails", "eval_defect_suspect", "suspected test defect",
               "## 8. Limitations", "agent_final.json", "wrong_refusal", "Official score"):
    assert needle in txt, needle
print("STEP 5 TESTS PASSED")
