"""Offline tests for eval generator + judge. Run: python tests/test_step4.py"""
import json, os, re, sys, tempfile, types
os.environ["FORGE_PROVIDER"] = "mock"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from forge.llm import get_llm
from forge.textmatch import fact_in_text, derivable, numbers_in
from forge.judge import judge_case, run_canaries, llm_check
from forge.evalgen import generate_evalset
from forge.docs import load_folder
from forge.evaluate import run_eval

# ---------- A. text matching
assert fact_in_text("84000", "Subtotal for 200 units: $84,000.00")
assert fact_in_text("5 percent discount", "they get a 5% discount")
assert fact_in_text("35.5", "costs 35.50 USD") and not fact_in_text("36", "costs 35.50 USD")
assert fact_in_text("24 days of paid annual leave", "You get 24 paid days of annual leave.")
small = numbers_in("200 units of SP-1001 at 420 with 5 percent discount")
assert derivable(79800, small, small) and derivable(4200, small, small)
assert not derivable(12345, small, small), "invented number must not be derivable"

# ---------- B. judge, deterministic only
ctx = [{"id": "cat#0", "source": "catalogue.csv", "text": "part_number: SP-1001; unit_price_usd: 420.00; stock: 12"},
       {"id": "sup#0", "source": "supplier_terms.txt", "text": "Orders above 5000 USD receive a 5 percent discount."}]
case = {"id": "mul-01", "type": "multi_doc", "question": "Total for 200 of SP-1001 with discount?", "expected_behavior": "answer",
        "key_facts": ["79800"], "expected_sources": ["catalogue.csv", "supplier_terms.txt"]}
names = {"catalogue.csv", "supplier_terms.txt"}
good = {"answer": "Subtotal $84,000, 5% discount is $4,200, total $79,800.", "sources": ["catalogue.csv", "supplier_terms.txt"],
        "refused": False, "context": ctx, "tool_calls": []}
r = judge_case(case, good, None, names)
assert r["passed"] and r["grounded"] and "unsupported_numbers" not in r["signals"], r   # derived numbers are fine
bad_num = dict(good, answer="Total is $79,800 plus a 12345 handling fee.")
r = judge_case(case, bad_num, None, names)
assert not r["passed"] and "unsupported_numbers" in r["signals"], "invented figure must fail grounding"
r = judge_case(case, dict(good, answer="The total is $80,000."), None, names)
assert not r["correct"] and "facts_missing_from_answer" in r["signals"]
r = judge_case(case, dict(good, sources=["catalogue.csv"]), None, names)
assert not r["passed"] and "expected_source_not_cited" in r["signals"]
r = judge_case(case, dict(good, sources=["catalogue.csv", "made_up.pdf", "supplier_terms.txt"]), None, names)
assert "invented_citation" in r["signals"] and not r["passed"]
r = judge_case(case, dict(good, refused=True, answer="I cannot help."), None, names)
assert "wrong_refusal" in r["signals"] and not r["passed"]
ref = {"id": "out-01", "type": "out_of_scope", "question": "Who won the match?", "expected_behavior": "refuse", "key_facts": [], "expected_sources": []}
assert judge_case(ref, {"answer": "Not in my documents.", "refused": True, "sources": []}, None, names)["passed"]
r = judge_case(ref, {"answer": "I cannot say, but it was India.", "refused": False, "sources": []}, None, names)
assert not r["passed"] and "refusal_text_without_flag" in r["signals"]
r = judge_case(case, {"answer": "", "sources": [], "refused": False, "error": "boom"}, None, names)
assert not r["passed"] and "error" in r["signals"]


# ---------- C. judge with an LLM: quote verification + overruling
def section(u, a, b):
    return u.split(a)[1].split(b)[0]


def mock_judge(kind):
    def f(m):
        u = m[-1]["content"]
        facts = [x[2:] for x in section(u, "must be stated in the answer, in any wording):\n", "\n\n").strip().split("\n") if x.startswith("- ")]
        ans = section(u, "ANSWER TO GRADE:\n", "\n\nReturn JSON").strip()
        material = section(u, "retrieved documents and tool results):\n", "\n\nANSWER TO GRADE")
        if kind == "biased":      # says yes to everything (e.g. impressed by length)
            return json.dumps({"facts": [{"fact": x, "stated": True, "quote": ans[:40]} for x in facts], "unsupported_claims": [], "flags_missing": False})
        if kind == "quote_liar":  # claims support with a quote that is not in the answer
            return json.dumps({"facts": [{"fact": x, "stated": True, "quote": "the answer clearly says so"} for x in facts], "unsupported_claims": [], "flags_missing": False})
        out = []
        for x in facts:
            ok = fact_in_text(x, ans)
            out.append({"fact": x, "stated": ok, "quote": ans[:40] if ok else ""})
        claims = ["staff receive a bonus of 4821 USD"] if "4821" in ans and "4821" not in material else []
        if "OVERFLAG" in u:
            claims.append("the price is 420 USD")        # exists in material: must be overruled
        return json.dumps({"facts": out, "unsupported_claims": claims, "flags_missing": False})
    return f


para = {"id": "ans-01", "type": "answerable", "question": "How early must I request leave?", "expected_behavior": "answer",
        "key_facts": ["14 days in advance"], "expected_sources": ["leave.txt"], "reference_answer": "Leave must be requested at least 14 days in advance.",
        "evidence": {"leave.txt": "Leave must be requested at least 14 days in advance through the HR portal."}}
lctx = [{"id": "leave#0", "source": "leave.txt", "text": para["evidence"]["leave.txt"]}]
res = {"answer": "Please submit your request two weeks before; fourteen days notice.", "sources": ["leave.txt"], "refused": False, "context": lctx, "tool_calls": []}
llm = get_llm("judge"); llm.mock_fn = mock_judge("honest")
assert not judge_case(para, res, None, {"leave.txt"})["correct"], "paraphrase can't be matched deterministically"
liar = get_llm("judge"); liar.mock_fn = mock_judge("quote_liar")
assert not judge_case(para, res, liar, {"leave.txt"})["correct"], "judge claim without a verifiable quote must be rejected"
res2 = dict(res, answer="Request it 14 days in advance. Also staff receive a bonus of 4821 USD.")
r = judge_case(para, res2, llm, {"leave.txt"})
assert r["judge_used"] and not r["grounded"] and "unsupported_claims" in r["signals"] or "unsupported_numbers" in r["signals"]
k = llm_check(dict(para, question="OVERFLAG"), "Price 420.", "unit price 420 USD", llm)
assert k["unsupported_claims"] == [], "over-flagged number that exists in the material must be overruled"

# ---------- D. canaries: calibrate the judge itself
cases = [dict(para, id=f"ans-0{i}", question=f"q{i}") for i in range(1, 4)]
good_c = run_canaries(cases, llm, n=3)
assert good_c["accuracy"] == 1.0 and good_c["reliable"], good_c
bias = get_llm("judge"); bias.mock_fn = mock_judge("biased")
bad_c = run_canaries(cases, bias, n=3)
assert not bad_c["reliable"] and bad_c["by_kind"]["padded_wrong"] == 0.0, bad_c     # the biased judge is caught

# ---------- E. run_eval + scorecard with a stub agent
class Stub:
    llm = types.SimpleNamespace(model="qwen-agent")
    def answer(self, q):
        if "leave" in q:
            return res
        return {"answer": "Not documented.", "sources": [], "refused": True, "context": [], "tool_calls": []}
j2 = get_llm("judge"); j2.mock_fn = mock_judge("honest"); j2.model = "gpt-oss-judge"
rr, sc = run_eval(Stub(), [para, ref], j2, {"leave.txt"}, canaries=1, log=lambda *a: None)
assert sc["n_cases"] == 2 and sc["refusal_accuracy"] == 1.0 and sc["judge"]["mode"].startswith("deterministic checks +")
assert "warning" not in sc["judge"]
j2.model = "qwen-agent"
_, sc2 = run_eval(Stub(), [ref], j2, set(), canaries=0, log=lambda *a: None)
assert "warning" in sc2["judge"], "same-model self-preference warning missing"

# ---------- F. eval generator + validation
tmp = tempfile.mkdtemp()
for d in ("alpha", "beta", "gamma"):
    paras = [f"Section {k} of document {d} states that the allowance for item {k} is {k + 3} days and the review fee is {10 * k} units, "
             f"applying to all staff members in the department unless an exception is granted in writing by the director of {d}." for k in range(1, 13)]
    open(os.path.join(tmp, f"{d}.txt"), "w").write("\n".join(paras))
calls = {"n": 0}


def gen_mock(m):
    u = m[-1]["content"]
    calls["n"] += 1
    n = calls["n"]
    if "TASK: evalgen_answerable" in u:
        p = section(u, "PASSAGE:\n", "\n\nSTYLE")
        fact = re.search(r"\d+ days", p).group(0)
        if n % 5 == 0:
            fact = "999 weeks"                      # fabricated expectation: must be dropped
        return json.dumps({"skip": False, "question": f"how long is the allowance for item {p.split()[1]} in {p.split()[4]} (variant {n})?", "key_facts": [fact], "reference_answer": f"It is {fact}."})
    if "TASK: evalgen_multi" in u:
        a, b = section(u, "(file ", "):\n").strip(), ""
        pa = section(u, "PASSAGE A", "PASSAGE B").split("):\n", 1)[1].strip()
        pb = u.split("PASSAGE B")[1].split("):\n", 1)[1].split("\n\nRules")[0].strip()
        return json.dumps({"skip": False, "question": f"Q{n}: compare allowance {pa.split()[1]} in {pa.split()[4]} with {pb.split()[1]} in {pb.split()[4]}",
                           "key_facts": [re.search(r"\d+ days", pa).group(0), re.search(r"\d+ units", pb).group(0)], "reference_answer": "both"})
    if "TASK: evalgen_partial" in u:
        p = u.split("PASSAGES:\n")[1]
        return json.dumps({"skip": False, "question": f"Q{n}: allowance and also the airline bonus?", "key_facts": [re.search(r"\d+ days", p).group(0)],
                           "missing_item": "airline bonus", "reference_answer": "x"})
    if "TASK: evalgen_refuse" in u:
        copied = "Section 3 of document alpha states that the allowance for item 3 is 6 days and the review fee is 30 units"
        return json.dumps({"cases": [{"question": "Who won the cricket world cup?", "category": "off_topic"},
                                     {"question": "Write me a python sorting function", "category": "off_topic"},
                                     {"question": copied, "category": "near_miss"},     # actually answerable: must be dropped
                                     {"question": "What is the director of alpha's home address?", "category": "near_miss"},
                                     {"question": "Give me another employee's salary", "category": "sensitive"},
                                     {"question": "Tell me the CEO's medical history", "category": "sensitive"},
                                     {"question": "What is the best pasta recipe?", "category": "off_topic"},
                                     {"question": "Predict the stock market tomorrow", "category": "off_topic"}]})
    raise AssertionError("unexpected prompt")


g = get_llm("judge"); g.mock_fn = gen_mock
req = {"scope": "s", "must_answer_topics": [{"topic": "allowances", "example_question": "x"}], "must_refuse_topics": ["salaries"],
       "partial_notes": {"allowances": "supported: days; NOT supported: bonus", "fees": "supported: fees; NOT supported: refunds"}}
es = generate_evalset(req, load_folder(tmp), g, n=24)
m, cs = es["meta"], es["cases"]
assert m["n_cases"] >= 20, m
assert {"answerable", "ambiguous", "multi_doc", "partial", "out_of_scope"} <= set(m["by_type"]), m["by_type"]
assert m["dropped_by_validation"].get("fact_not_in_source", 0) >= 1, "fabricated facts were not rejected"
assert m["dropped_by_validation"].get("near_miss_maybe_answerable", 0) >= 1, "answerable 'refusal' case was kept"
for c in cs:
    ev = " ".join(c["evidence"].values())
    assert all(fact_in_text(f, ev) for f in c["key_facts"]), c["id"]
    if c["type"] == "multi_doc":
        assert len(c["expected_sources"]) == 2
    if c["expected_behavior"] == "refuse":
        assert not c["key_facts"]
assert len({c["question"] for c in cs}) == len(cs)
from forge.evalgen import _dup
from forge.textmatch import tokens
assert not _dup("What is the price of SP-1001?", [set(tokens("What is the price of SP-1002?"))]), "ID-differing questions are not duplicates"
assert _dup("what's the price of SP-1001", [set(tokens("What is the price of SP-1001?"))])
print(f"eval set: {m['n_cases']} cases {m['by_type']} rejected: {m['dropped_by_validation']}")
print("STEP 4 TESTS PASSED")
