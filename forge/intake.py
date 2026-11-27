"""Requirement 1: intake interviewer. Brief + docs -> <=5 questions -> structured requirements.json.
Then hands over to the groundability guard (requirement 2)."""
import json, os
from .docs import load_folder
from .prompts import render
from .guard import check_groundability


def _excerpts(text, n=500):
    t = " ".join(text.split())
    if len(t) <= 3 * n:
        return t
    mid = len(t) // 2
    return f"{t[:n]}\n...\n{t[mid:mid + n]}\n...\n{t[-n:]}"


def profile_docs(docs, llm):
    out = []
    for d in docs:
        p = llm.json([{"role": "user", "content": render("doc_profile", name=d.name, kind=d.kind,
                                                         excerpts=_excerpts(d.text))}], temperature=0)
        out.append({"name": d.name, "kind": d.kind, "words": len(d.text.split()),
                    "summary": str(p.get("summary", "")), "topics": [str(x) for x in p.get("topics", [])][:10]})
    return out


def _fmt_profiles(profiles):
    return "\n".join(f"- {p['name']} ({p['kind']}, {p['words']} words): {p['summary']} Topics: {', '.join(p['topics'])}"
                     for p in profiles)


def ask_questions(brief, profiles, llm, max_q=5):
    r = llm.json([{"role": "user", "content": render("intake_questions", brief=brief, max_q=max_q,
                                                     profiles=_fmt_profiles(profiles))}], temperature=0.2)
    qs = []
    for q in r.get("questions", []):
        if isinstance(q, dict) and q.get("question"):
            qs.append({"question": str(q["question"]), "why": str(q.get("why", "")),
                       "default": str(q.get("default", "no preference"))})
    return qs[:max_q]  # hard cap in code: never trust the model to respect the limit


def collect_answers(questions, unattended=False, answers=None):
    """Interactive CLI by default. --unattended uses the model's stated defaults (recorded as assumptions).
    --answers <file.json> (a list of strings) makes demo runs reproducible."""
    qa = []
    for i, q in enumerate(questions):
        if answers is not None and i < len(answers) and str(answers[i]).strip():
            a, src = str(answers[i]).strip(), "requester (answers file)"
        elif unattended:
            a, src = q["default"], "assumed default"
        else:
            print(f"\nQ{i + 1}: {q['question']}\n   (why: {q['why']})\n   [Enter = assume: {q['default']}]")
            typed = input("   > ").strip()
            a, src = (typed, "requester") if typed else (q["default"], "assumed default")
        qa.append({**q, "answer": a, "answer_source": src})
    return qa


def _as_list(x):
    return [str(i) for i in x] if isinstance(x, list) else []


def validate_requirements(req):
    """Coerce model output into the schema; small models drift, so enforce shape here."""
    topics = []
    for t in req.get("must_answer_topics", []) if isinstance(req.get("must_answer_topics"), list) else []:
        if isinstance(t, str):
            t = {"topic": t, "example_question": t}
        if isinstance(t, dict) and t.get("topic"):
            topics.append({"topic": str(t["topic"]), "example_question": str(t.get("example_question") or t["topic"])})
    if not topics:
        raise ValueError("requirements draft has no must_answer_topics")
    sig = req.get("tool_signals") if isinstance(req.get("tool_signals"), dict) else {}
    return {
        "agent_name": "".join(c if c.isalnum() else "_" for c in str(req.get("agent_name", "agent")).lower())[:40] or "agent",
        "scope": str(req.get("scope", "")), "audience": str(req.get("audience", "")), "tone": str(req.get("tone", "")),
        "must_answer_topics": topics[:8],
        "must_refuse_topics": _as_list(req.get("must_refuse_topics")),
        "tool_signals": {k: bool(sig.get(k, False)) for k in
                         ("needs_calculation", "needs_live_data", "needs_table_lookup", "needs_current_date")},
    }


def draft_requirements(brief, profiles, qa, llm):
    qa_txt = "\n".join(f"- {x['question']} -> {x['answer']} ({x['answer_source']})" for x in qa) or "(none asked)"
    r = llm.json([{"role": "user", "content": render("intake_requirements", brief=brief, qa=qa_txt,
                                                     profiles=_fmt_profiles(profiles))}], temperature=0.2)
    return validate_requirements(r)


def to_markdown(req):
    g = req["groundability"]
    L = [f"# Requirements: {req['agent_name']}", "", f"**Brief:** {req['brief']}", "",
         f"**Scope:** {req['scope']}", f"**Audience:** {req['audience']}", f"**Tone:** {req['tone']}", "",
         "## Must answer"]
    for t in req["must_answer_topics"]:
        note = req.get("partial_notes", {}).get(t["topic"])
        L.append(f"- {t['topic']}" + (f"  (PARTIAL: {note})" if note else ""))
    L += ["", "## Must refuse"] + [f"- {t}" for t in req["must_refuse_topics"]]
    L += ["", f"## Groundability: {g['status'].upper()}", "",
          "| topic | verdict | coverage | evidence source |", "|---|---|---|---|"]
    for t in g["topics"]:
        L.append(f"| {t['topic']} | {t['verdict']} | {t['coverage']} | {t['source'] or '-'} |")
    if req["ungrounded_requests"]:
        L += ["", "### Requested but NOT supported by the documents (agent will refuse these)"]
        L += [f"- {u['topic']}: {u['missing'] or 'no supporting passage found'}" for u in req["ungrounded_requests"]]
    L += ["", "## Clarifications"] + [f"- Q: {x['question']}\n  A: {x['answer']} ({x['answer_source']})" for x in req["clarifications"]]
    return "\n".join(L) + "\n"


def run_intake(brief, docs_dir, out_dir, llm, guard_llm=None, unattended=False, answers=None, max_q=5, force=False):
    os.makedirs(out_dir, exist_ok=True)
    docs = load_folder(docs_dir)
    if not docs:
        raise SystemExit(f"No readable documents (pdf/txt/md/csv) in {docs_dir}")
    profiles = profile_docs(docs, llm)
    questions = ask_questions(brief, profiles, llm, max_q)
    qa = collect_answers(questions, unattended, answers)
    req = draft_requirements(brief, profiles, qa, llm)
    req.update(brief=brief, docs_path=docs_dir, clarifications=qa)
    req = check_groundability(req, docs, guard_llm or llm)
    w = lambda n, s: open(os.path.join(out_dir, n), "w", encoding="utf-8").write(s)
    w("doc_profiles.json", json.dumps(profiles, indent=2))
    w("requirements.json", json.dumps(req, indent=2))
    w("requirements.md", to_markdown(req))
    print(f"[intake] requirements written to {out_dir} | groundability: {req['groundability']['status']}")
    if req["groundability"]["status"] == "blocked" and not force:
        raise SystemExit("[intake] BLOCKED: the documents cannot support most of this brief. "
                         "See requirements.md. Add documents, narrow the brief, or re-run with --force.")
    return req
