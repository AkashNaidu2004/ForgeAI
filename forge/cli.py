"""python -m forge.cli intake --brief "..." --docs samples/hr_policy --out runs/hr_policy"""
import argparse, json
from .llm import get_llm
from .intake import run_intake
from .generator import run_generate


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
    a = ap.parse_args()
    if a.cmd == "generate":
        run_generate(a.req, a.out, get_llm("gen"))
        return
    brief = a.brief or open(a.brief_file, encoding="utf-8").read().strip()
    answers = json.load(open(a.answers, encoding="utf-8")) if a.answers else None
    if a.cmd == "intake":
        run_intake(brief, a.docs, a.out, get_llm("gen"), guard_llm=get_llm("judge"), unattended=a.unattended,
                   answers=answers, max_q=min(a.max_questions, 5), force=a.force)


if __name__ == "__main__":
    main()
