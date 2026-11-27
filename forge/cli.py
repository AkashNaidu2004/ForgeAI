"""ForgeAI command line.
  python -m forge.cli intake   --brief "..." --docs <folder> --out <run dir>
  python -m forge.cli generate --req <run>/requirements.json --out <run>
  python -m forge.cli evalgen  --req <run>/requirements.json --out <run> [--n 24]
  python -m forge.cli evaluate --agent <run>/agent_v0.json --evalset <run>/eval_set.json --out <run>/eval_v0.json"""
import argparse, json
from .llm import get_llm
from .intake import run_intake
from .generator import run_generate
from .evalgen import run_evalgen
from .evaluate import run_evaluate


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
    a = ap.parse_args()
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
