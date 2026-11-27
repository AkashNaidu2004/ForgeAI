"""Regression test for the live failure: 'order total with supplier discount' was rejected although the discount rule
exists in supplier_terms.txt. Run: python tests/test_guard_conditional.py"""
import json, os, sys
os.environ["FORGE_PROVIDER"] = "mock"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from forge.llm import get_llm
from forge.docs import load_folder
from forge.guard import check_groundability
from forge.generator import consistency_warnings

SEEN = {}


def fake(m):
    u = m[-1]["content"]
    if "TASK: ground_check" in u:
        topic = u.split("TOPIC:")[1].split("\n")[0]
        SEEN[topic.strip()] = u
        if "discount" in topic:   # supported for MarineParts, unsupported for other suppliers
            return json.dumps({"verdict": "partial", "evidence": "Orders above 5000 USD receive a 5% discount",
                               "source": "supplier_terms.txt", "supported": "5 percent discount on MarineParts orders above 5000 USD",
                               "missing": "no discount terms for OceanFlow Supplies"})
        return json.dumps({"verdict": "none", "evidence": "", "source": "", "supported": "", "missing": "nothing about it"})
    if "TASK: scope_repair" in u:
        return json.dumps({"scope": "Answers ship engineers' questions about spare parts, prices and supplier terms."})
    raise AssertionError("unexpected prompt")


llm = get_llm("gen"); llm.mock_fn = fake
req = {"agent_name": "spares", "brief": "b", "scope": "Spares help plus live sea weather.", "audience": "ship engineers",
       "tone": "t", "must_refuse_topics": ["legal advice"],
       "tool_signals": {"needs_calculation": True, "needs_table_lookup": True},
       "must_answer_topics": [
           {"topic": "order total with supplier discount",
            "example_question": "What is the total for 10 units of SP-1001 from OceanFlow with the discount?"},
           {"topic": "live sea weather forecast", "example_question": "What is the sea state tomorrow?"}]}
req = check_groundability(req, load_folder("samples/ship_spares"), llm)

v = {t["topic"]: t for t in req["groundability"]["topics"]}
d = v["order total with supplier discount"]
assert d["verdict"] == "partial", d                       # kept, not rejected
assert d["evidence_verified"], "'5%' vs '5 percent' must still verify"
assert {"catalogue.csv", "supplier_terms.txt"} <= set(d["sources_seen"]), d["sources_seen"]   # cross-document passages
assert "calculator tool" in SEEN["order total with supplier discount"], "tool-aware note missing"
assert "OceanFlow" in req["partial_notes"]["order total with supplier discount"]
assert v["live sea weather forecast"]["verdict"] == "ungrounded"
assert [t["topic"] for t in req["must_answer_topics"]] == ["order total with supplier discount"]
assert "weather" not in req["scope"].lower()

bad = {"must_answer_topics": [{"topic": "order cost with supplier discount", "example_question": "x"}],
       "must_refuse_topics": ["order totals with supplier discounts (not covered by the documents)"]}
assert consistency_warnings(bad), "refuse/answer contradiction not detected"
print("GUARD CONDITIONAL TESTS PASSED")
