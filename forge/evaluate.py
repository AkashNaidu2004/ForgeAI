"""Run an eval set against an agent definition, judge every case, and compute the scorecard."""
import json, os, time
from .agent import Agent, load_definition
from .judge import judge_case, run_canaries
from .budget import InfraError


def scorecard(results, judge_info):
    n = len(results)
    pct = lambda a, b: round(a / b, 3) if b else None
    ans = [r for r in results if r["expected"] == "answer"]
    ref = [r for r in results if r["expected"] == "refuse"]
    by_type = {}
    for r in results:
        d = by_type.setdefault(r["type"], {"n": 0, "passed": 0})
        d["n"] += 1
        d["passed"] += int(r["passed"])
    return {
        "n_cases": n, "passed": sum(r["passed"] for r in results), "pass_rate": pct(sum(r["passed"] for r in results), n),
        "answer_correct_rate": pct(sum(bool(r["correct"]) for r in ans), len(ans)),
        "grounded_rate": pct(sum(bool(r["grounded"]) for r in ans), len(ans)),
        "refusal_accuracy": pct(sum(r["refusal_ok"] for r in ref), len(ref)),
        "false_refusal_rate": pct(sum("wrong_refusal" in r["signals"] for r in ans), len(ans)),
        "retrieval_hit_rate": pct(sum(bool(r.get("retrieval_hit")) for r in ans), len(ans)),
        "errors": sum("error" in r["signals"] for r in results), "by_type": by_type, "judge": judge_info,
        "failed_ids": [r["id"] for r in results if not r["passed"]]}


def run_eval(agent, cases, judge_llm=None, doc_names=(), canaries=3, log=print):
    info = {"model": getattr(judge_llm, "model", None), "agent_model": getattr(agent.llm, "model", None), "mode": "deterministic only"}
    if judge_llm is not None:
        if judge_llm.model == agent.llm.model:
            info["warning"] = "judge and agent use the SAME model: self-preference bias is possible"
            log("   WARNING: " + info["warning"])
        if canaries:
            c = run_canaries(cases, judge_llm, n=canaries)
            info["canaries"] = c
            log(f"   judge canaries: accuracy {c['accuracy']} {c['by_kind']}")
            if not c["reliable"]:
                info["warning_reliability"] = "judge failed canary tests; LLM verdicts disabled, deterministic checks only"
                log("   WARNING: " + info["warning_reliability"])
                judge_llm = None
        if judge_llm is not None:
            info["mode"] = "deterministic checks + LLM judge (quote-verified)"
    results, t0 = [], time.time()
    for i, c in enumerate(cases, 1):
        try:
            res = agent.answer(c["question"])
        except InfraError:
            raise                      # quota / network problem: abort the stage, never score it as an agent failure
        except Exception as e:
            res = {"answer": "", "sources": [], "refused": False, "context": [], "tool_calls": [], "error": f"{type(e).__name__}: {e}"}
        r = judge_case(c, res, judge_llm, doc_names)
        results.append(r)
        log(f"   [{i}/{len(cases)}] {c['id']:<14} {'PASS' if r['passed'] else 'FAIL'} {r['signals'] or ''}")
    sc = scorecard(results, info)
    sc["seconds"] = round(time.time() - t0, 1)
    return results, sc


def run_evaluate(agent_path, evalset_path, out_path, judge_llm, canaries=3, use_llm=True):
    from .docs import load_folder
    d = load_definition(agent_path)
    es = json.load(open(evalset_path, encoding="utf-8"))
    names = {x.name for x in load_folder(d["docs_path"])}
    results, sc = run_eval(Agent(d), es["cases"], judge_llm if use_llm else None, names, canaries if use_llm else 0)
    json.dump({"agent": agent_path, "scorecard": sc, "results": results}, open(out_path, "w", encoding="utf-8"), indent=2)
    print(f"[evaluate] pass {sc['passed']}/{sc['n_cases']} = {sc['pass_rate']} | correct {sc['answer_correct_rate']} "
          f"grounded {sc['grounded_rate']} refusal_acc {sc['refusal_accuracy']} false_refusal {sc['false_refusal_rate']} "
          f"retrieval_hit {sc['retrieval_hit_rate']}\n   failed: {sc['failed_ids']}\n   saved {out_path}")
    return results, sc
