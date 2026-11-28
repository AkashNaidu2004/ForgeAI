"""ForgeAI command line.
  python -m forge.cli intake   --brief "..." --docs <folder> --out <run dir>
  python -m forge.cli generate --req <run>/requirements.json --out <run>
  python -m forge.cli evalgen  --req <run>/requirements.json --out <run> [--n 24]
  python -m forge.cli evaluate --agent <run>/agent_v0.json --evalset <run>/eval_set.json --out <run>/eval_v0.json
  python -m forge.cli optimize --agent <run>/agent_v0.json --evalset <run>/eval_set.json --out <run> [--max-iter 3]
  python -m forge.cli report   --run <run>
  python -m forge.cli run      --brief "..." --docs <folder> --out <run>      (whole factory, unattended)
  python -m forge.cli budget                                                  (today's hosted-API usage vs limits)
  python -m forge.cli cache-clear                                             (forget cached model responses)"""
import argparse, json, os, sys
from .llm import get_llm
from .budget import BUDGET, InfraError
from .intake import run_intake
from .generator import run_generate
from .evalgen import run_evalgen
from .evaluate import run_evaluate
from .optimizer import optimize
from .report import build_report
from .factory import run_factory


def main():
    ap = argparse.ArgumentParser(prog="forge")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("intake", help="brief + docs -> requirements.json (with groundability check)")
    p.add_argument("--brief"); p.add_argument("--brief-file")
    p.add_argument("--docs", required=True); p.add_argument("--out", required=True)
    p.add_argument("--unattended", action="store_true", help="do not ask; assume defaults")
    p.add_argument("--answers", help="json list of answers, for reproducible demo runs")
    p.add_argument("--max-questions", type=int, default=5)
    p.add_argument("--force", action="store_true", help="continue even if groundability is 'blocked'")
    g = sub.add_parser("generate", help="requirements.json -> agent_v0.json")
    g.add_argument("--req", required=True); g.add_argument("--out", required=True)
    e = sub.add_parser("evalgen", help="requirements.json -> eval_set.json (generated from the documents)")
    e.add_argument("--req", required=True); e.add_argument("--out", required=True); e.add_argument("--n", type=int, default=24)
    v = sub.add_parser("evaluate", help="run + judge an eval set against an agent definition")
    v.add_argument("--agent", required=True); v.add_argument("--evalset", required=True); v.add_argument("--out", required=True)
    v.add_argument("--canaries", type=int, default=3, help="cases used to calibrate the judge (0 = skip)")
    v.add_argument("--no-llm-judge", action="store_true", help="deterministic checks only")
    o = sub.add_parser("optimize", help="critic-optimizer loop (max 3 iterations) -> agent_final.json + optimization_log.json")
    o.add_argument("--agent", required=True); o.add_argument("--evalset", required=True); o.add_argument("--out", required=True)
    o.add_argument("--max-iter", type=int, default=3)
    r = sub.add_parser("report", help="write report.md from a run folder's logs")
    r.add_argument("--run", required=True)
    f = sub.add_parser("run", help="whole factory, unattended: intake -> generate -> evalgen -> optimize -> report")
    f.add_argument("--brief"); f.add_argument("--brief-file"); f.add_argument("--docs", required=True); f.add_argument("--out", required=True)
    f.add_argument("--answers", help="json list of answers (otherwise unattended defaults are used)")
    f.add_argument("--force", action="store_true"); f.add_argument("--n", type=int, default=24); f.add_argument("--max-iter", type=int, default=3)
    sub.add_parser("budget", help="today's hosted-API usage against the configured limits, per model")
    sub.add_parser("cache-clear", help="delete the LLM response cache (.forge_cache/)")
    a = ap.parse_args()
    if a.cmd == "budget":
        return show_budget()
    if a.cmd == "cache-clear":
        print(f"[cache] removed {BUDGET.cache_clear()} cached responses")
        return
    run_dir = getattr(a, "run", None) or getattr(a, "out", None)
    if run_dir and run_dir.endswith(".json"):
        run_dir = os.path.dirname(run_dir) or "."
    code = 0
    try:
        with BUDGET.stage(a.cmd):
            dispatch(a)
    except InfraError as e:
        print(f"\n[forge] STOPPED by infrastructure, not by a quality problem: {type(e).__name__}: {e}\n"
              "        Nothing was scored as a failure. Re-run the same command later (finished calls are cached), "
              "or set FORGE_FALLBACK=ollama.")
        code = 2
    finally:
        if a.cmd != "report":
            BUDGET.dump_usage(run_dir)
    if code:
        sys.exit(code)


def show_budget():
    r = BUDGET.usage_report()
    print(f"[budget] UTC day {r['date']}  (limits from .env: {int(r['limits']['rpm'])} req/min, {int(r['limits']['tpm'])} tok/min, "
          f"{int(r['limits']['rpd'])} req/day, {int(r['limits']['tpd'])} tok/day)")
    if not r["models"]:
        print("   no hosted calls recorded today")
    for m in r["models"]:
        print(f"   {m['model']}: {m['tokens']}/{int(m['tpd'])} tokens used ({m['tokens_left']} left), {m['requests']}/{int(m['rpd'])} requests ({m['requests_left']} left)")
    print("   Per-stage token use for a run is in <run>/usage.json and in report.md section 9.")


def dispatch(a):
    if a.cmd in ("optimize", "report", "run"):
        if a.cmd == "optimize":
            optimize(a.agent, a.evalset, a.out, get_llm("judge"), max_iter=a.max_iter)
        elif a.cmd == "report":
            build_report(a.run)
        else:
            brief = a.brief or open(a.brief_file, encoding="utf-8").read().strip()
            answers = json.load(open(a.answers, encoding="utf-8")) if a.answers else None
            run_factory(brief, a.docs, a.out, answers, a.force, a.n, a.max_iter)
        return
    if a.cmd == "generate":
        run_generate(a.req, a.out, get_llm("gen"))
    elif a.cmd == "evalgen":
        run_evalgen(a.req, a.out, get_llm("judge"), a.n)      # tests written by the judge-family model, not the agent's model
    elif a.cmd == "evaluate":
        run_evaluate(a.agent, a.evalset, a.out, get_llm("judge"), a.canaries, not a.no_llm_judge)
    else:
        brief = a.brief or open(a.brief_file, encoding="utf-8").read().strip()
        answers = json.load(open(a.answers, encoding="utf-8")) if a.answers else None
        run_intake(brief, a.docs, a.out, get_llm("gen"), guard_llm=get_llm("judge"), unattended=a.unattended,
                   answers=answers, max_q=min(a.max_questions, 5), force=a.force)


if __name__ == "__main__":
    main()
