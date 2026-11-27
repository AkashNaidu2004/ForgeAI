"""Offline test of the generator. A lying mock LLM tries to attach a weather tool to every agent.
Run: python tests/test_step3.py"""
import json, os, sys
os.environ["FORGE_PROVIDER"] = "mock"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from forge.llm import get_llm
from forge.generator import generate_agent
from forge.agent import Agent, load_definition


def fake(m):
    u = m[-1]["content"]
    if "TASK: generate_persona" in u:
        return json.dumps({"role_line": "You are a helpful assistant.", "style_rules": ["Be brief."]})
    if "TASK: tool_selection" in u:   # over-eager: says yes to everything
        names = ["calculator", "current_date", "csv_lookup", "weather_api"]
        return json.dumps({"tools": [{"name": n, "use": True, "justification": "might be useful"} for n in names]})
    return json.dumps({"answer": "ok", "sources": [], "refused": True})


def req(brief, docs, topics, signals):
    return {"agent_name": "t", "brief": brief, "scope": brief, "audience": "staff", "tone": "friendly",
            "docs_path": docs, "must_answer_topics": [{"topic": t, "example_question": t} for t in topics],
            "must_refuse_topics": ["legal advice", "medical advice", "personal data"],
            "partial_notes": {}, "tool_signals": signals}


llm = get_llm("gen"); llm.mock_fn = fake
hr = generate_agent(req("HR policy questions for shore staff", "samples/hr_policy",
                        ["annual leave", "sick leave", "travel expense claims"], {}), llm)
assert hr["tools"] == [], f"HR agent got tools: {hr['tools']}"
sp = generate_agent(req("Ship spares assistant: look up part numbers in the catalogue and calculate order cost with supplier discounts",
                        "samples/ship_spares", ["part lookup", "order cost with discount", "delivery terms"],
                        {"needs_table_lookup": True, "needs_calculation": True}), llm)
assert set(sp["tools"]) == {"csv_lookup", "calculator"}, sp["tools"]
assert "weather_api" not in sp["tools"] and "current_date" not in sp["tools"]
assert "catalogue.csv" in sp["system_prompt"] and "unit_price_usd" in sp["system_prompt"]
json.dump(sp, open("/tmp/sp_def.json", "w"))
a = Agent(load_definition("/tmp/sp_def.json"), llm)          # definition must load in the generic runtime
assert a.answer("price of SP-1001")["refused"] is True
print("HR tools:", hr["tools"], "| Spares tools:", sp["tools"])
print("STEP 3 TESTS PASSED")
