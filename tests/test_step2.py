"""Offline test of intake + guard using a scripted mock LLM. Run: python tests/test_step2.py"""
import json, os, shutil, sys
os.environ["FORGE_PROVIDER"] = "mock"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from forge.llm import get_llm
from forge.intake import run_intake


def fake(m):
    u = m[-1]["content"]
    if "TASK: doc_profile" in u:
        return json.dumps({"summary": "HR policy.", "topics": ["leave", "travel"]})
    if "TASK: intake_questions" in u:
        return json.dumps({"questions": [{"question": f"q{i}?", "why": "w", "default": "d"} for i in range(8)]})  # model ignores the cap
    if "TASK: intake_requirements" in u:
        return json.dumps({"agent_name": "HR Helper!", "scope": "Answers HR questions on leave and also books business flights for staff.", "audience": "a", "tone": "t",
                           "must_answer_topics": [
                               {"topic": "annual leave entitlement", "example_question": "How many days of annual leave do I get per year?"},
                               {"topic": "sick leave certificate", "example_question": "When do I need a medical certificate for sick leave?"},
                               {"topic": "flight booking on my behalf", "example_question": "Book me a flight to Mumbai"}],
                           "must_refuse_topics": ["legal advice", "salary of others", "medical advice"], "tool_signals": {}})
    if "TASK: ground_check" in u:
        if "annual leave" in u.split("TOPIC:")[1].split("\n")[0]:
            return json.dumps({"verdict": "full", "evidence": "24 days of paid annual leave per calendar year", "source": "leave_policy.txt", "missing": ""})
        if "sick leave" in u.split("TOPIC:")[1].split("\n")[0]:   # invented quote -> must be downgraded
            return json.dumps({"verdict": "full", "evidence": "a certificate is always optional for any sickness", "source": "leave_policy.txt", "missing": ""})
        return json.dumps({"verdict": "none", "evidence": "", "source": "", "missing": "no booking process in documents"})
    if "TASK: scope_repair" in u:
        if CLEAN:
            return json.dumps({"scope": "Answers staff questions about annual leave and sick leave policy."})
        return json.dumps({"scope": "Answers leave questions and handles flight booking."})  # model fails to repair
    raise AssertionError("unexpected prompt")


CLEAN = False
llm = get_llm("gen"); llm.mock_fn = fake
out = "/tmp/step2_out"; shutil.rmtree(out, ignore_errors=True)
req = run_intake("HR assistant for shore staff", "samples/hr_policy", out, llm, unattended=True, max_q=5, force=True)
v = {t["topic"]: t["verdict"] for t in req["groundability"]["topics"]}
assert len(req["clarifications"]) == 5, "question cap not enforced"
assert v["annual leave entitlement"] == "grounded"
assert v["sick leave certificate"] != "grounded", "invented evidence was not caught"
assert v["flight booking on my behalf"] == "ungrounded"
assert any("flight" in t for t in req["must_refuse_topics"]), "gap not moved to must_refuse"
assert all(t["topic"] != "flight booking on my behalf" for t in req["must_answer_topics"])
assert req["agent_name"] == "hr_helper_"
assert "flight" not in req["scope"].lower() and "booking" not in req["scope"].lower(), req["scope"]
assert req["scope_repair"]["method"].startswith("deterministic"), "bad model scope should trigger fallback"
CLEAN = True
req2 = run_intake("HR assistant for shore staff", "samples/hr_policy", out + "2", llm, unattended=True, max_q=5, force=True)
assert req2["scope_repair"]["method"] == "llm" and "flight" not in req2["scope"].lower()
print(open(out + "/requirements.md").read())
print("STEP 2 TESTS PASSED")
