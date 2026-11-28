"""Requirement 7: honest report, generated from the run's own logs (no model writes the numbers).
Everything below is computed from requirements.json, eval_set.json, optimization_log.json and the agent files, so the
report cannot drift from what actually happened. Limitations are derived from the data, not boilerplate only."""
import json, os


def _j(run, name, default=None):
    p = os.path.join(run, name)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else default


def _pct(x):
    return "n/a" if x is None else f"{x:.0%}"


def _cfg(c):
    if not c:
        return "an unrecorded backend"
    return f"agent `{c['agent']}` / judge `{c['judge']}`"


def _cell(s, n=110):
    s = " ".join(str(s).split()).replace("|", "/")
    return s if len(s) <= n else s[:n - 1] + "..."


def limitations(log, es, req, usage):
    held_n = log["versions"][0]["held_out"]["n"]
    out = [f"The eval set was written by an LLM from the same documents, and only {len(es['cases'])} cases were kept after validation. "
           f"Differences of one or two cases are within noise.",
           f"The held-out set has only {held_n} cases, so held-out numbers are indicative, not statistically meaningful.",
           "Model sampling is not fully deterministic: re-running the same agent can change a borderline case's verdict."]
    judge = log["versions"][0]["scorecard"]["judge"]
    if "warning_reliability" in judge:
        out.append("The LLM judge failed its calibration canaries, so scoring used deterministic checks only. Paraphrased correct answers "
                   "may have been marked wrong.")
    if "warning" in judge:
        out.append(judge["warning"] + ".")
    if judge.get("mode", "").startswith("deterministic only"):
        out.append("No LLM judge was used: paraphrased correct answers may be marked wrong.")
    if log.get("config_changes"):
        out.append("The model backend changed mid-run (hosted quota exhausted, local fallback used). Scores from before and after the change come "
                   "from different models and are not compared; the current best agent was re-scored as a fresh baseline. The local models are "
                   "smaller, so absolute scores after the switch may be lower or noisier.")
    n_def = sum(1 for f in log["final_failures"] if f["category"] == "eval_defect_suspect")
    if n_def:
        out.append(f"{n_def} remaining failure(s) are suspected defects of the test itself. They were NOT removed from the score.")
    if any(not it.get("accepted", True) for it in log["iterations"] if "accepted" in it):
        out.append("At least one iteration was rejected because it did not improve the dev score; the agent was reverted.")
    if log["stop_reason"].startswith("no further patch"):
        out.append("The optimizer ran out of automatic patches for the remaining failure categories; those need a human decision.")
    if req.get("groundability", {}).get("status") in ("partial", "blocked"):
        out.append("Part of the brief could not be grounded in the documents; the agent is deliberately configured to refuse those requests.")
    out += ["Retrieval is lexical (BM25) unless embeddings are enabled, so synonyms the documents never use can be missed.",
            "The optimizer can only add rules and tune retrieval parameters. It does not fix missing or contradictory source documents."]
    return out


def build_report(run_dir, usage=None):
    req, es = _j(run_dir, "requirements.json"), _j(run_dir, "eval_set.json")
    log, a0, af = _j(run_dir, "optimization_log.json"), _j(run_dir, "agent_v0.json"), _j(run_dir, "agent_final.json")
    L = []
    w = L.append
    v0, vf = log["versions"][0], log["versions"][-1]
    w(f"# ForgeAI run report: {req['agent_name']}\n")
    w(f"**Brief:** {req['brief']}\n")
    w("## 1. Headline\n")
    changes = log.get("config_changes", [])
    if not changes:
        w(f"- Official score (all {vf['overall']['n']} eval cases, nothing removed): **{v0['overall']['passed']} -> {vf['overall']['passed']} passed** "
          f"({_pct(v0['overall']['rate'])} -> {_pct(vf['overall']['rate'])}).")
        w(f"- Dev cases (the optimizer may learn from these): {v0['dev']['passed']}/{v0['dev']['n']} -> {vf['dev']['passed']}/{vf['dev']['n']}.")
    else:
        base = next(s for s in log["versions"] if s["config"] == vf["config"])
        w(f"- **The model backend changed during this run, so v0 and the final version were NOT scored by the same models.** "
          f"v0 scored {v0['overall']['passed']}/{v0['overall']['n']} on {_cfg(v0['config'])}; that number is not comparable to the final one.")
        w(f"- Official score of the final version (all {vf['overall']['n']} cases, nothing removed), on {_cfg(vf['config'])}: "
          f"**{vf['overall']['passed']}/{vf['overall']['n']}** ({_pct(vf['overall']['rate'])}). "
          f"Comparable starting point on the same backend: {base['overall']['passed']}/{base['overall']['n']} "
          f"({'re-scored after the switch' if base.get('rebaseline') else 'v0'}).")
    ref = v0 if not changes else base
    w(f"- Held-out cases (not diagnosed during optimization or used for patches): {ref['held_out']['passed']}/{ref['held_out']['n']} -> {vf['held_out']['passed']}/{vf['held_out']['n']}"
      f"{'' if not changes else ' (same-backend baseline)'}.")
    w(f"- Final agent: `agent_final.json` (version {log['final_version']}). Optimizer stopped because: {log['stop_reason']}.")
    still = len(log["final_failures"])
    w(f"- Still failing: **{still}** case(s); see section 7.\n")
    g = req["groundability"]
    w("## 2. What was asked and what the documents support\n")
    w(f"Groundability status: **{g['status']}**. Scope as built: {req['scope']}\n")
    verdict = {t["topic"]: t["verdict"] for t in g.get("topics", [])}
    for t in req["must_answer_topics"]:
        w(f"- KEPT ({verdict.get(t['topic'], '?')}): {t['topic']}")
    for t in req["must_refuse_topics"]:
        w(f"- REFUSED: {t}")
    for k, v in req.get("partial_notes", {}).items():
        w(f"- PARTIAL: {k}: {v}")
    w("\n## 3. Tools\n")
    for x in a0["tool_decisions"]:
        w(f"- `{x['tool']}`: {x['reason']}")
    w("\n## 4. Eval set\n")
    m = es["meta"]
    w(f"{m['n_cases']} cases by type: {m['by_type']}. Generated with `{m['generated_with']}` (prompts in `prompts/evalgen_*.txt`). "
      f"Candidates rejected by code validation: {m['dropped_by_validation'] or 'none'}.\n")
    recovery = _j(run_dir, "recovery.json")
    if recovery:
        w(f"**Dataset recovery:** {recovery['description']}\n")
    w("## 5. Judge reliability\n")
    j = v0["scorecard"]["judge"]
    w(f"- Mode: {j['mode']}. Judge model: `{j.get('model')}`; agent model: `{j.get('agent_model')}`.")
    if "canaries" in j:
        w(f"- Canary accuracy {j['canaries']['accuracy']} over {j['canaries']['n']} known answers {j['canaries']['by_kind']}.")
    for k in ("warning", "warning_reliability"):
        if k in j:
            w(f"- WARNING: {j[k]}")
    w("\n## 6. Iterations\n")
    w("| Version | Overall | Dev | Held-out | Scored by |\n|---|---|---|---|---|")
    for s in log["versions"]:
        tag = " (re-scored after backend change)" if s.get("rebaseline") else ""
        w(f"| v{s['version']}{tag} | {s['overall']['passed']}/{s['overall']['n']} | {s['dev']['passed']}/{s['dev']['n']} | {s['held_out']['passed']}/{s['held_out']['n']} | {_cfg(s.get('config'))} |")
    for ch in log.get("config_changes", []):
        w(f"\nBackend change before iteration {ch['before_iteration']}: {_cfg(ch['from'])} -> {_cfg(ch['to'])}. "
          f"v{ch['version']} had scored {ch['old_scores']['passed']}/{ch['old_scores']['n']} on the old backend and was re-scored at "
          f"{ch['rescored']['passed']}/{ch['rescored']['n']} on the new one. Iterations after this point are compared only with the re-scored baseline.")
    w("")
    for it in log["iterations"]:
        w(f"### Iteration {it['iteration']}: {it['outcome']}\n")
        w(f"Dev failures triaged: " + (", ".join(f"{f['id']} ({f['category']}{'' if f['genuine_agent_defect'] else ', not patched'})" for f in it["dev_failures"]) or "none"))
        for p in it.get("patches", []):
            if p["target"] == "retrieval":
                w(f"- Patch [{p['category']}] retrieval {p['change']}: {p['before']} -> {p['after']}. Fixed {p['fixed'] or 'none'}; still failing {p['still_failing'] or 'none'}.")
            else:
                w(f"- Patch [{p['category']}] `{p['target']}` ({p['rule_source']}): \"{p['rule']}\" Fixed {p['fixed'] or 'none'}; still failing {p['still_failing'] or 'none'}.")
        if "dev_after" in it:
            w(f"- Dev {it['dev_before']['passed']} -> {it['dev_after']['passed']}; held-out {it['held_out_before']['passed']} -> {it['held_out_after']['passed']}; "
              f"regressions: {it['regressions'] or 'none'}.")
        w("")
    w("## 7. What still fails, and why\n")
    if not log["final_failures"]:
        w("Nothing failed in the final run. That still does not prove the agent is correct beyond these cases.\n")
    else:
        w("| Case | Split | Type | Triage | Genuine agent defect? | Note |\n|---|---|---|---|---|---|")
        for f in log["final_failures"]:
            w(f"| {f['id']} | {f['split']} | {f['type']} | {f['category']} | {'yes' if f['genuine_agent_defect'] else 'no (suspected test defect)'} | {_cell(f['note'] or ', '.join(f['signals']))} |")
        w("\nQuestions and answers for the failures above:\n")
        for f in log["final_failures"]:
            w(f"- **{f['id']}** Q: {_cell(f['question'], 160)}\n  A: {_cell(f['answer'], 240)}")
    w("\n## 8. Limitations\n")
    for x in limitations(log, es, req, usage):
        w(f"- {x}")
    u = _j(run_dir, "usage.json")
    w("\n## 9. API usage and backend\n")
    if u:
        w("Model calls actually sent (cache hits cost nothing). Tokens are the provider's own counts when it returned them, otherwise estimates.\n")
        w("| Stage | Calls | Cache hits | Prompt tokens | Completion tokens | Waited (s) |\n|---|---|---|---|---|---|")
        for name, s in u["stages"].items():
            w(f"| {name} | {s['calls']} | {s['cache_hits']} | {s['prompt_tokens']} | {s['completion_tokens']} | {s['waited_s']:.0f} |")
        t = u["totals"]
        w(f"| **total** | {t['calls']} | {t['cache_hits']} | {t['prompt_tokens']} | {t['completion_tokens']} | {t['waited_s']:.0f} |")
        w(f"\nPer-minute pacing used limits of {int(u['limits']['rpm'])} requests and {int(u['limits']['tpm'])} tokens per model "
          f"(x{u['limits']['margin']} safety margin); daily cap {int(u['limits']['tpd'])} tokens. "
          "If a run was resumed, calls from every attempt are added together.")
        ev = [e for e in u.get("events", []) if e["type"] in ("backend_fallback", "daily_budget")]
        if ev:
            w("\nBackend and quota events:")
            for e in ev:
                w(f"- {e['time']} during `{e['stage']}`: {e['type']}: {e['detail']}")
        else:
            w("\nNo quota exhaustion or backend switch occurred.")
    else:
        w("No usage.json was found for this run, so per-stage API usage is not available.")
    if usage and not u:
        w(f"\nLLM usage for the whole process: {usage['calls']} calls, {usage['seconds']:.0f} s of model time.")
    txt = "\n".join(L) + "\n"
    open(os.path.join(run_dir, "report.md"), "w", encoding="utf-8").write(txt)
    print(f"[report] wrote {os.path.join(run_dir, 'report.md')}")
    return txt
