"""Offline tests for the API budget layer. Run: python tests/test_budget.py
Covers: pacing, daily cap + persistence + UTC rollover, 429 handling (per-minute vs daily), cache correctness and
staleness, Ollama fallback, infra errors never scored as agent failures, per-stage accounting, and optimizer
re-baselining after a backend change. Nothing here calls a real API; the 'provider' is simulated."""
import json, os, sys, tempfile, types
from unittest.mock import patch
os.environ["FORGE_PROVIDER"] = "mock"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import requests
import forge.llm as L
from forge.budget import Budget, BudgetExhausted, RateLimitError, LLMUnavailable, InfraError

for k in ("FORGE_RPM", "FORGE_TPM", "FORGE_RPD", "FORGE_TPD", "FORGE_LIMIT_MARGIN", "FORGE_FALLBACK", "FORGE_CACHE"):
    os.environ.pop(k, None)
quiet = lambda *a, **k: None


class Clock:
    def __init__(self): self.t, self.slept = 1000.0, 0.0
    def now(self): return self.t
    def sleep(self, s): self.t += s; self.slept += s


def mk_budget(day="d1", **env):
    c = Clock()
    for k, v in env.items():
        os.environ[k] = str(v)
    b = Budget(state_dir=tempfile.mkdtemp(), clock=c.now, sleep=c.sleep, day_fn=lambda: day, log=quiet)
    return b, c


def clear_env():
    for k in ("FORGE_RPM", "FORGE_TPM", "FORGE_RPD", "FORGE_TPD", "FORGE_LIMIT_MARGIN", "FORGE_FALLBACK", "FORGE_CACHE"):
        os.environ.pop(k, None)


class Resp:
    def __init__(self, status=200, text="", headers=None, body=None):
        self.status_code, self.text, self.headers, self._b = status, text, headers or {}, body
    def json(self): return self._b
    def raise_for_status(self):
        if self.status_code >= 400: raise requests.HTTPError(str(self.status_code))


def ok(text, pt=100, ct=20):
    return Resp(200, body={"choices": [{"message": {"content": text}}], "usage": {"prompt_tokens": pt, "completion_tokens": ct}})


def hosted(role="gen", model="big-model"):
    l = L.LLM(role); l.provider, l.model, l._base_url = "openai", model, "http://x"
    return l


M = [{"role": "user", "content": "hello"}]

# ---------- 1. pacing: requests per minute
b, c = mk_budget(FORGE_RPM=2, FORGE_TPM=1000000, FORGE_LIMIT_MARGIN=1)
for _ in range(2):
    b.commit(b.reserve("m", 10), 5, 5)
assert c.slept == 0, "two calls fit inside the minute"
h = b.reserve("m", 10)
assert c.slept >= 59.9, "third call must wait for the window to clear"
b.commit(h, 5, 5)
clear_env()

# ---------- 2. pacing: tokens per minute
b, c = mk_budget(FORGE_RPM=100, FORGE_TPM=1000, FORGE_LIMIT_MARGIN=1)
b.commit(b.reserve("m", 600), 300, 300)
b.commit(b.reserve("m", 300), 150, 150)
assert c.slept == 0
b.commit(b.reserve("m", 600), 300, 300)
assert c.slept >= 59.9, "token window must hold the call back"
clear_env()

# ---------- 3. daily cap, persistence across restarts, UTC rollover
b, c = mk_budget(day="d1", FORGE_TPD=1000, FORGE_RPD=50, FORGE_LIMIT_MARGIN=1)
b.commit(b.reserve("m", 600), 300, 300)
try:
    b.reserve("m", 600); raise AssertionError("must refuse BEFORE sending when the daily cap would be exceeded")
except BudgetExhausted:
    pass
b2 = Budget(state_dir=b.state_dir, clock=c.now, sleep=c.sleep, day_fn=lambda: "d1", log=quiet)
assert b2.used_today("m")["tokens"] == 600, "usage must survive a restart"
b3 = Budget(state_dir=b.state_dir, clock=c.now, sleep=c.sleep, day_fn=lambda: "d2", log=quiet)
assert b3.used_today("m")["tokens"] == 0, "usage must reset on a new UTC day"
b3.commit(b3.reserve("m", 600), 300, 300)
clear_env()

# ---------- 4. provider 429s: per-minute waits, daily stops at once, and is remembered
b, c = mk_budget(FORGE_TPD=100000)
with patch.object(L, "BUDGET", b), patch.object(L.time, "sleep") as sl:
    l = hosted()
    with patch.object(L.requests, "post", side_effect=[Resp(429, "tokens per minute (TPM)", {"Retry-After": "5"}), ok("fine")]) as post:
        assert l.chat(M) == "fine" and post.call_count == 2 and sl.call_count == 1
    daily = Resp(429, "Rate limit reached ... on tokens per day (TPD): Limit 200000, Used 199990", {"Retry-After": "600"})
    with patch.object(L.requests, "post", return_value=daily) as post:
        try: l.chat([{"role": "user", "content": "other"}]); raise AssertionError("daily limit must raise")
        except BudgetExhausted: pass
        assert post.call_count == 1, "a daily limit must not be retried"
    with patch.object(L.requests, "post") as post:
        try: l.chat([{"role": "user", "content": "third"}]); raise AssertionError("must fail fast now")
        except BudgetExhausted: pass
        assert post.call_count == 0, "after a daily 429 we must not even ask the provider again"

# ---------- 5. other infrastructure errors are typed, bounded and never look like agent failures
b, c = mk_budget()
with patch.object(L, "BUDGET", b), patch.object(L.time, "sleep"):
    l = hosted()
    with patch.object(L.requests, "post", side_effect=requests.ConnectionError("down")) as post:
        try: l.chat(M); raise AssertionError
        except LLMUnavailable: pass
        assert post.call_count == 3
    with patch.object(L.requests, "post", return_value=Resp(404, "model not found")):
        try: l.chat([{"role": "user", "content": "x"}]); raise AssertionError
        except LLMUnavailable as e: assert "model id" in str(e)
assert issubclass(BudgetExhausted, RateLimitError) and issubclass(RateLimitError, InfraError) and issubclass(LLMUnavailable, InfraError)

# ---------- 6. cache: hits are free; anything that changes the request is a miss; bad JSON is evicted
b, c = mk_budget()
with patch.object(L, "BUDGET", b):
    l = hosted()
    with patch.object(L.requests, "post", return_value=ok("answer-1")) as post:
        assert l.chat(M, temperature=0.2) == "answer-1"
        assert l.chat(M, temperature=0.2) == "answer-1" and post.call_count == 1, "identical request must be served from cache"
        l.chat(M, temperature=0.9); assert post.call_count == 2, "temperature is part of the key"
        l.chat([{"role": "user", "content": "hello!"}], temperature=0.2); assert post.call_count == 3, "changed prompt must not hit a stale entry"
        l.chat(M, temperature=0.2, cache_salt=7); assert post.call_count == 4, "salt separates deliberate repeats"
        hosted(model="other-model").chat(M, temperature=0.2); assert post.call_count == 5, "model id is part of the key"
        assert b.stages["unscoped"]["cache_hits"] == 1 and b.stages["unscoped"]["calls"] == 5
    assert b.used_today("big-model")["requests"] == 4, "cache hits must not consume daily budget"
    n_before = sum(len(f) for _, _, f in os.walk(b.cache_dir))
    with patch.object(L.requests, "post", return_value=ok("this is not json at all")) as post:
        try: l.json([{"role": "user", "content": "give json"}]); raise AssertionError
        except ValueError: pass
    n_after = sum(len(f) for _, _, f in os.walk(b.cache_dir))
    assert n_after == n_before, "unparseable replies must be evicted so a re-run asks again"
    os.environ["FORGE_CACHE"] = "0"
    with patch.object(L.requests, "post", return_value=ok("z")) as post:
        l.chat(M, temperature=0.2); l.chat(M, temperature=0.2); assert post.call_count == 2, "FORGE_CACHE=0 disables caching"
    clear_env()

# ---------- 7. Ollama fallback: only when enabled, sticky, logged, and reported under the REAL model name
def route(url, **kw):
    if "chat/completions" in url:
        return Resp(429, "tokens per day (TPD) reached", {"Retry-After": "900"})
    return Resp(200, body={"message": {"content": "local answer"}, "prompt_eval_count": 50, "eval_count": 10})

b, c = mk_budget()
with patch.object(L, "BUDGET", b):
    l = hosted()
    with patch.object(L.requests, "post", side_effect=route):
        try: l.chat(M); raise AssertionError("without FORGE_FALLBACK the daily limit must surface")
        except BudgetExhausted: pass
    os.environ["FORGE_FALLBACK"] = "ollama"
    b2, _ = mk_budget(); L.BUDGET = b2
    l, j = hosted("gen", "big-model"), hosted("judge", "judge-model")
    with patch.object(L.requests, "post", side_effect=route), patch.object(L.requests, "get", return_value=Resp(200)):
        assert l.chat(M) == "local answer"
        assert (l.provider, l.model) == ("ollama", "qwen2.5:7b"), "model name must reflect who actually answered"
        assert (j.provider, j.model) == ("ollama", "llama3.2:3b"), "other roles switch too, with their own fallback model"
        assert b2.fallback_active and [e["type"] for e in b2.events].count("backend_fallback") == 1
        assert hosted().chat([{"role": "user", "content": "again"}]) == "local answer" and len([e for e in b2.events if e["type"] == "backend_fallback"]) == 1
    b3, _ = mk_budget(); L.BUDGET = b3
    with patch.object(L.LLM, "_post_openai", side_effect=RateLimitError("retry exhausted")), patch.object(L.LLM, "_post_ollama", return_value=("local answer", 1, 1)), patch.object(L.requests, "get", return_value=Resp(200)):
        assert hosted().chat(M) == "local answer"
        assert b3.fallback_active
    b4, _ = mk_budget(); L.BUDGET = b4
    with patch.object(L.requests, "post", side_effect=route), patch.object(L.requests, "get", side_effect=requests.ConnectionError("no ollama")):
        try: hosted().chat(M); raise AssertionError
        except LLMUnavailable as e: assert "Ollama" in str(e)
    clear_env()

# ---------- 8. infrastructure errors are never scored as agent failures
from forge.evaluate import run_eval
from forge.judge import judge_case
case = {"id": "c1", "type": "answerable", "question": "q", "expected_behavior": "answer", "key_facts": ["24 days"],
        "expected_sources": ["a.txt"], "reference_answer": "24 days", "evidence": {}}
def agent_raising(exc):
    return types.SimpleNamespace(llm=types.SimpleNamespace(model="a"), answer=lambda q: (_ for _ in ()).throw(exc))
for exc in (BudgetExhausted("quota"), RateLimitError("429"), LLMUnavailable("down")):
    try: run_eval(agent_raising(exc), [case], None, ["a.txt"], 0, quiet); raise AssertionError("must abort, not score")
    except InfraError: pass
res, sc = run_eval(agent_raising(ValueError("model produced garbage")), [case], None, ["a.txt"], 0, quiet)
assert res[0]["passed"] is False and "error" in res[0]["signals"], "a model-side parse failure is still a (flagged) failure"
res_ok = {"answer": "You get 24 days.", "sources": ["a.txt"], "refused": False, "context": [{"id": "a#0", "source": "a.txt", "text": "24 days"}], "tool_calls": []}
bad_judge = types.SimpleNamespace(model="j", json=lambda m, **k: (_ for _ in ()).throw(RateLimitError("429")))
try: judge_case(case, res_ok, bad_judge, ["a.txt"]); raise AssertionError("judge must not silently fall back to deterministic-only")
except InfraError: pass

# ---------- 9. per-stage accounting and additive usage.json
b, c = mk_budget()
with b.stage("evalgen"):
    b.commit(b.reserve("m", 50), 40, 10)
    b.commit(b.reserve("m", 50), 30, 10)
with b.stage("optimize"):
    b.note_cache_hit()
assert b.stages["evalgen"]["calls"] == 2 and b.stages["evalgen"]["prompt_tokens"] == 70 and b.stages["optimize"]["cache_hits"] == 1
run = tempfile.mkdtemp()
u = b.dump_usage(run)
with b.stage("evalgen"):
    b.commit(b.reserve("m", 50), 5, 5)
u = b.dump_usage(run)
assert u["stages"]["evalgen"]["calls"] == 3 and u["totals"]["calls"] == 3 and u["totals"]["cache_hits"] == 1, "dumps add up, no double counting"

# ---------- 10. optimizer: a backend change forces a fresh baseline; old and new scores are never compared
import forge.optimizer as opt
from forge.generator import render_system_prompt
from forge.report import build_report
CORPUS = "Leave allowance is 24 days per year. Sick allowance is 10 days. Meal allowance is 40 USD per day."
tmp = tempfile.mkdtemp(); docs = os.path.join(tmp, "docs"); os.makedirs(docs)
open(os.path.join(docs, "alpha.txt"), "w").write(CORPUS)
KINDS = {"wrongref": ("24 days", "Staff get 24 days of leave per year."), "ok": ("40 USD", "The meal allowance is 40 USD per day."),
         "refuse_ok": (None, None)}
plan = ["wrongref"] * 4 + ["ok"] * 6 + ["refuse_ok"] * 2
cases, kind_of = [], {}
for i, k in enumerate(plan, 1):
    q = f"question {i} about {k}"; kind_of[q] = k; fact = KINDS[k][0]
    cases.append({"id": f"c-{i:02d}", "type": "out_of_scope" if fact is None else "answerable", "question": q,
                  "expected_behavior": "refuse" if fact is None else "answer", "key_facts": [fact] if fact else [],
                  "expected_sources": ["alpha.txt"] if fact else [], "reference_answer": "", "evidence": {}})
backend = types.SimpleNamespace(provider="openai", model="big-hosted",
                                json=lambda m, **k: {"rule": "Use every relevant passage when answering."})
calls = {"n": 0}
FLIP_AT = len(cases) + 5                       # flip part-way through the first candidate evaluation


class StubAgent:
    def __init__(self, d, llm=None):
        self.d, self.llm = d, backend
    def answer(self, q):
        calls["n"] += 1
        if calls["n"] == FLIP_AT:
            backend.provider, backend.model = "ollama", "small-local"      # simulated Ollama fallback
        kind, sp = kind_of[q], self.d["system_prompt"]
        ctx = [{"id": "alpha#0", "source": "alpha.txt", "text": CORPUS}]
        refuse = {"answer": "Not in the documents.", "sources": [], "refused": True}
        if kind == "refuse_ok" or (kind == "wrongref" and "Refuse only when none" not in sp):
            r = refuse
        elif kind == "ok" and backend.model == "small-local":
            r = {"answer": "I am not sure about that.", "sources": ["alpha.txt"], "refused": False}   # weaker model
        else:
            r = {"answer": KINDS[kind][1], "sources": ["alpha.txt"], "refused": False}
        return dict(r, context=ctx, tool_calls=[])
opt.Agent = StubAgent

secs = {"role": "You are an HR assistant.", "scope": "Scope: HR", "grounding_rules": "GROUNDING RULES:\n1. Use context only.",
        "refusal_rules": "REFUSE when asked about:\n- salaries"}
d0 = {"version": 0, "name": "hr", "docs_path": docs, "prompt_sections": secs, "system_prompt": render_system_prompt(secs), "tools": [],
      "retrieval": {"chunk_size": 150, "overlap": 30, "top_k": 4, "mode": "bm25", "min_score": 0.0},
      "tool_decisions": [{"tool": "calculator", "reason": "not attached: no evidence in brief/topics"}]}
run = os.path.join(tmp, "run"); os.makedirs(run)
J = lambda n, o: json.dump(o, open(os.path.join(run, n), "w"), indent=2)
J("agent_v0.json", d0)
J("eval_set.json", {"meta": {"n_cases": len(cases), "by_type": {"answerable": 10, "out_of_scope": 2}, "generated_with": "mock", "dropped_by_validation": {}}, "cases": cases})
J("requirements.json", {"agent_name": "hr", "brief": "HR helper", "scope": "HR policy answers", "docs_path": docs,
                        "must_answer_topics": [{"topic": "leave", "example_question": "x"}], "must_refuse_topics": [], "partial_notes": {},
                        "groundability": {"status": "ok", "topics": [{"topic": "leave", "verdict": "grounded"}]}})
msgs = []
log = opt.optimize(os.path.join(run, "agent_v0.json"), os.path.join(run, "eval_set.json"), run, None, agent_llm=backend, log=msgs.append)
v0 = log["versions"][0]
assert v0["config"]["agent"] == "openai:big-hosted", "v0 was scored by the hosted model"
assert any("discarding the mixed result" in m for m in msgs), "an evaluation that straddles the switch must be discarded"
assert len(log["config_changes"]) == 1, log["config_changes"]
ch = log["config_changes"][0]
assert ch["from"]["agent"] == "openai:big-hosted" and ch["to"]["agent"] == "ollama:small-local"
rb = [s for s in log["versions"] if s.get("rebaseline")]
assert len(rb) == 1 and rb[0]["config"]["agent"] == "ollama:small-local" and rb[0]["replaces_config"]["agent"] == "openai:big-hosted"
assert rb[0]["overall"]["passed"] < v0["overall"]["passed"], "the weaker backend scores lower; the old number must not be used as the baseline"
it1 = log["iterations"][0]
assert it1["dev_before"] == rb[0]["dev"], "iteration 1 must be judged against the RE-SCORED baseline, not v0"
assert it1["rebaseline"]["old_scores"] == v0["overall"]
assert all(s["config"] for s in log["versions"]) and len(log["iterations"]) <= 3, "3-iteration cap and config tagging"
assert log["official_score"]["config"]["agent"] == "ollama:small-local"
assert log["official_score"]["comparable_baseline"]["rebaseline"] is True and log["official_score"]["n"] == len(cases), "no case removed"
build_report(run)
txt = open(os.path.join(run, "report.md"), encoding="utf-8").read()
assert "NOT scored by the same models" in txt and "re-scored after backend change" in txt and "## 9. API usage and backend" in txt

# ---------- 11. an infrastructure abort mid-optimization leaves a clearly-labelled partial log and raises
calls["n"] = 0; backend.provider, backend.model = "openai", "big-hosted"
class QuotaAgent(StubAgent):
    def answer(self, q):
        calls["n"] += 1
        if calls["n"] > len(cases) + 3:
            raise BudgetExhausted("daily budget used up")
        return super().answer(q)
opt.Agent = QuotaAgent
run2 = os.path.join(tmp, "run2"); os.makedirs(run2)
for n in ("agent_v0.json", "eval_set.json"):
    open(os.path.join(run2, n), "w").write(open(os.path.join(run, n)).read())
try:
    opt.optimize(os.path.join(run2, "agent_v0.json"), os.path.join(run2, "eval_set.json"), run2, None, agent_llm=backend, log=quiet)
    raise AssertionError("must raise")
except BudgetExhausted:
    pass
part = json.load(open(os.path.join(run2, "optimization_partial.json")))
assert part["status"].startswith("ABORTED") and not os.path.exists(os.path.join(run2, "optimization_log.json")), "no final log may be written for an aborted run"

# ---------- 12. git hygiene
gi = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".gitignore")).read().split()
assert ".forge_cache/" in gi and ".forge_budget.json" in gi and ".env" in gi
assert "runs/" not in gi, "committed demo runs must not be ignored"
print("BUDGET TESTS PASSED")
