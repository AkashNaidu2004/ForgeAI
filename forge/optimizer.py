"""Requirement 6: critic-optimizer loop (max 3 iterations, every change logged).

Principles
  * Triage first. Each failure is classified; only failures judged to be GENUINE agent defects are fixed. Failures that
    look like defects of the test itself are labelled `eval_defect_suspect`: they stay in the official score and are
    listed in the report, but the optimizer does not "fix" the agent for them.
  * Dev / held-out split. The optimizer only sees (and may learn from) the dev cases. The held-out cases (~25%) are
    scored every iteration but never diagnosed, so the report can show whether gains generalise or are overfitting.
  * One targeted patch per failure category, never a prompt rewrite. A patch touches one named prompt section or one
    retrieval parameter. Rules written by the critic LLM are code-validated (no digits, no eval key facts) so the
    eval set's answers cannot leak into the prompt; otherwise a fixed template is used.
  * An iteration is accepted only if the dev pass count goes up. Otherwise it is reverted and logged as rejected.
"""
import copy, json, os, zlib
from .agent import Agent, load_definition
from .docs import load_folder
from .evaluate import run_eval
from .llm import get_llm
from .generator import render_system_prompt
from .judge import looks_like_refusal
from .prompts import render
from .budget import InfraError
from .textmatch import fact_in_text, numbers_in

MAX_ITER = 3

CATEGORIES = {   # category -> (what it means, genuine agent defect?)
    "retrieval_miss": ("the facts needed were not in the retrieved passages", True),
    "wrong_refusal": ("the assistant refused although the answer was in the retrieved passages", True),
    "answered_unanswerable": ("the assistant answered something the documents do not support", True),
    "refusal_flag": ("the answer text declined but refused=true was not set", True),
    "hallucination": ("the answer contains numbers, claims or filenames not supported by the material", True),
    "partial_gap": ("the answer did not say which part of the question the documents do not cover", True),
    "incomplete_answer": ("the answer left out a relevant fact that was in the retrieved passages", True),
    "citation_gap": ("the answer did not cite a document it needed", True),
    "eval_defect_suspect": ("the test case itself looks wrong or too strict; the agent may be right", False),
    "infrastructure_error": ("an API or JSON error, not a quality problem", False),
    "other": ("unclassified failure", True),
}

TEMPLATES = {   # fallback rules if the critic's rule fails validation
    "wrong_refusal": ("answer_policy", "If any retrieved passage contains information that answers the question, answer from it. Refuse only when none of the passages address the question."),
    "answered_unanswerable": ("refusal_rules", "- a specific item, person or detail that is not stated in the passages, even if the general topic is covered (say what is missing)"),
    "refusal_flag": ("contract_policy", "Whenever you decline or say the information is unavailable, you must also set refused=true and sources=[]."),
    "hallucination": ("numbers_policy", "Every number, name, date and filename in your answer must appear in the context or in a tool result. Use the calculator tool for any arithmetic instead of computing it yourself."),
    "partial_gap": ("partial_rules", "- if one part of a question is not covered, answer the covered part and state explicitly which part is not in the documents"),
    "incomplete_answer": ("completeness_policy", "Answer every part of the question and include every directly relevant fact from the passages, such as limits, conditions and availability."),
    "citation_gap": ("citation_policy", "Cite every document whose content you used, including documents that supplied only a rate, rule or condition."),
    "other": ("completeness_policy", "Re-read the passages before answering and use every relevant fact they contain."),
}


# ------------------------------------------------------------------ split + triage
def split(cases):
    """Deterministic dev / held-out split (about one case in four is held out)."""
    held = [c for c in cases if zlib.crc32(c["id"].encode()) % 4 == 0]
    if len(held) < 3 and len(cases) >= 8:
        held = cases[3::4]
    ids = {c["id"] for c in held}
    return [c for c in cases if c["id"] not in ids], held


def triage(r, case, corpus):
    """-> (category, genuine, note). Pure code, no LLM, so the classification is auditable."""
    s, facts = set(r["signals"]), case.get("key_facts", [])
    if "error" in s:
        return "infrastructure_error", False, r.get("error", "")
    if r["expected"] == "refuse":
        if "refusal_text_without_flag" in s:
            return "refusal_flag", True, "text declines but refused=false"
        ans_nums, doc_nums = numbers_in(r["answer"]), set(numbers_in(corpus))
        if ans_nums and all(n in doc_nums for n in ans_nums) and not looks_like_refusal(r["answer"]):
            return "eval_defect_suspect", False, "every figure in the answer occurs in the documents: the question may be answerable"
        return "answered_unanswerable", True, ""
    if facts and not all(fact_in_text(f, corpus) for f in facts):
        return "eval_defect_suspect", False, "an expected key fact does not appear in any document"
    if r["refused"]:
        return ("retrieval_miss", True, "") if r.get("facts_in_context", 1) < 1 else ("wrong_refusal", True, "")
    if s & {"unsupported_numbers", "unsupported_claims", "invented_citation"}:
        return "hallucination", True, ", ".join(sorted(s & {"unsupported_numbers", "unsupported_claims", "invented_citation"}))
    if "missing_item_not_flagged" in s:
        return "partial_gap", True, ""
    if "facts_missing_from_answer" in s:
        if r.get("facts_in_context", 1) < 1:
            return "retrieval_miss", True, ""
        if not r.get("judge_used") and r.get("fact_recall", 0) >= 0.5:
            return "eval_defect_suspect", False, "deterministic matching only; a paraphrase may have been missed"
        return "incomplete_answer", True, f"fact recall {r.get('fact_recall')}"
    if "expected_source_not_cited" in s:
        return "citation_gap", True, ""
    return "other", True, ", ".join(sorted(s))


# ------------------------------------------------------------------ patches
def critic_rule(cat, examples, banned_facts, llm):
    """Ask the critic for a general rule; return (rule, source). Falls back to the template if validation fails."""
    ex = "\n".join(f"Q: {q}\nA: {a[:300]}" for q, a in examples[:2])
    try:
        rule = str(llm.json([{"role": "user", "content": render(
            "critic_rule", category=cat, description=CATEGORIES[cat][0], examples=ex)}], temperature=0.2).get("rule", "")).strip()
    except InfraError:
        raise
    except Exception:
        return None, "template"
    ok = 8 <= len(rule.split()) <= 45 and not any(ch.isdigit() for ch in rule) \
        and not any(fact_in_text(f, rule) for f in banned_facts if len(f.split()) >= 2) and "\n" not in rule
    return (rule, "critic") if ok else (None, "template")


def _add(sections, name, rule):
    before = sections.get(name, "")
    if name in ("refusal_rules", "partial_rules"):
        sections[name] = (before + "\n" + rule) if before else rule
    else:
        sections[name] = ((before + "\n") if before else "") + rule
    return before


def apply_patches(d, groups, critic_llm, banned_facts, version):
    """groups: {category: [(result, case)]} of GENUINE dev failures. Returns (new definition, patch records)."""
    nd, patches = copy.deepcopy(d), []
    nd["version"] = version
    secs = nd["prompt_sections"]
    n_retr = sum(1 for h in d.get("patch_history", []) if h["category"] == "retrieval_miss")
    for cat, items in groups.items():
        ids = [c["id"] for _, c in items]
        if cat == "retrieval_miss":
            r = nd["retrieval"]
            before = {k: r[k] for k in ("chunk_size", "overlap", "top_k", "mode")}
            if n_retr == 0 and r["top_k"] < 8:
                r["top_k"] = min(8, r["top_k"] + 2); what = "top_k +2"
            elif n_retr <= 1 and r["chunk_size"] < 300:
                r["chunk_size"] = min(300, int(r["chunk_size"] * 1.5)); r["overlap"] = r["chunk_size"] // 5; what = "chunk_size x1.5"
            elif r["mode"] == "bm25" and os.getenv("FORGE_USE_EMBEDDINGS") == "1":
                r["mode"] = "hybrid"; what = "bm25 -> hybrid"
            elif r["top_k"] < 8:
                r["top_k"] += 1; what = "top_k +1"
            else:
                continue                      # nothing left to try for retrieval
            n_retr += 1
            patches.append({"category": cat, "target": "retrieval", "change": what, "before": before,
                            "after": {k: r[k] for k in before}, "rule_source": "fixed schedule", "targets": ids})
            continue
        section, template = TEMPLATES.get(cat, TEMPLATES["other"])
        rule, src = critic_rule(cat, [(c["question"], r["answer"]) for r, c in items], banned_facts, critic_llm)
        if rule is None:
            rule = template
        elif section in ("refusal_rules", "partial_rules") and not rule.startswith("-"):
            rule = "- " + rule
        before = _add(secs, section, rule)
        patches.append({"category": cat, "target": f"prompt_sections.{section}", "change": "added rule", "rule": rule,
                        "before_len": len(before), "after_len": len(secs[section]), "rule_source": src, "targets": ids})
    nd["system_prompt"] = render_system_prompt(secs)
    nd["patch_history"] = d.get("patch_history", []) + [{"version": version, **p} for p in patches]
    return nd, patches


# ------------------------------------------------------------------ the loop
def _rate(results, ids):
    rs = [r for r in results if r["id"] in ids]
    return {"n": len(rs), "passed": sum(r["passed"] for r in rs), "rate": round(sum(r["passed"] for r in rs) / len(rs), 3) if rs else None}


def _snapshot(version, results, sc, dev_ids, held_ids, config=None):
    return {"version": version, "config": config, "overall": _rate(results, {r["id"] for r in results}), "dev": _rate(results, dev_ids),
            "held_out": _rate(results, held_ids), "scorecard": sc}


def _llm_sig(l):
    """Identity of the model that actually answers (provider:model). After an Ollama fallback this names the local model."""
    return "deterministic" if l is None else f"{getattr(l, 'provider', '?')}:{getattr(l, 'model', '?')}"


def classify_all(results, cases, corpus, dev_ids):
    by = {c["id"]: c for c in cases}
    rows = []
    for r in results:
        if r["passed"]:
            continue
        cat, genuine, note = triage(r, by[r["id"]], corpus)
        rows.append({"id": r["id"], "type": r["type"], "split": "dev" if r["id"] in dev_ids else "held_out", "category": cat,
                     "genuine_agent_defect": genuine, "note": note, "signals": r["signals"], "question": r["question"],
                     "answer": r["answer"][:400]})
    return rows


def optimize(agent_path, evalset_path, out_dir, judge_llm, agent_llm=None, max_iter=MAX_ITER, log=print):
    """Loop (max 3 iterations). Scores are only ever compared when they were measured by the SAME backend (agent model +
    judge model + judge mode). If the backend changes (for example the hosted quota runs out and the local fallback takes
    over) the current best agent is re-scored on the new backend as a fresh baseline; the old scores stay in the log."""
    max_iter = max(0, min(max_iter, MAX_ITER))
    os.makedirs(out_dir, exist_ok=True)
    save = lambda n, o: json.dump(o, open(os.path.join(out_dir, n), "w", encoding="utf-8"), indent=2)
    d = load_definition(agent_path)
    cases = json.load(open(evalset_path, encoding="utf-8"))["cases"]
    dev, held = split(cases)
    dev_ids, held_ids = {c["id"] for c in dev}, {c["id"] for c in held}
    docs = load_folder(d["docs_path"])
    names, corpus = {x.name for x in docs}, "\n".join(x.text for x in docs)
    banned = [f for c in dev for f in c.get("key_facts", [])]
    probe = agent_llm or get_llm("gen")             # same effective model as the Agent objects built below
    critic = judge_llm or agent_llm or Agent(d).llm
    state = {"jl": judge_llm}

    raw_cfg = lambda: (_llm_sig(probe), _llm_sig(judge_llm))                        # detects a switch DURING an evaluation
    eff_cfg = lambda: {"agent": _llm_sig(probe), "judge": _llm_sig(state["jl"])}    # what the scores are attributed to

    def scored(defn, canaries):
        """Evaluate every case under ONE backend. If the backend changed mid-way the mixed result is discarded and redone."""
        for attempt in range(3):
            c0 = raw_cfg()
            res, sc = run_eval(Agent(defn, agent_llm), cases, state["jl"], names, canaries, log)
            if canaries and "warning_reliability" in sc["judge"]:
                state["jl"] = None                    # judge failed calibration: deterministic only from now on
            if raw_cfg() == c0:
                return res, sc, eff_cfg()
            log("[optimize] the backend changed during this evaluation; discarding the mixed result and re-scoring on the new backend")
        raise InfraError("backend kept changing during evaluation")

    snaps, iterations, config_changes = [], [], []
    stop, cur, cur_res = "reached max iterations", d, None
    try:
        log(f"[optimize] {len(cases)} cases: {len(dev)} dev (optimizer may learn from these), {len(held)} held-out (scored, never diagnosed)")
        results, sc, cfg0 = scored(d, 3)
        save("eval_v0.json", {"agent": agent_path, "scorecard": sc, "results": results, "config": cfg0})
        snaps.append(_snapshot(0, results, sc, dev_ids, held_ids, cfg0))
        cur_res = results
        for i in range(1, max_iter + 1):
            triaged = classify_all([r for r in cur_res if r["id"] in dev_ids], cases, corpus, dev_ids)
            dev_fail = [t for t in triaged if t["split"] == "dev"]
            by_res = {r["id"]: r for r in cur_res}
            by_case = {c["id"]: c for c in cases}
            groups = {}
            for t in dev_fail:
                if t["genuine_agent_defect"]:
                    groups.setdefault(t["category"], []).append((by_res[t["id"]], by_case[t["id"]]))
            entry = {"iteration": i, "from_version": cur["version"], "dev_failures": [{k: t[k] for k in ("id", "category", "genuine_agent_defect", "note")} for t in dev_fail]}
            if not groups:
                stop = "no genuine agent defects left among dev failures"
                iterations.append(dict(entry, outcome="stopped: " + stop)); break
            cand, patches = apply_patches(cur, groups, critic, banned, i)
            if not patches:
                stop = "no further patch available for the remaining failure categories"
                iterations.append(dict(entry, outcome="stopped: " + stop)); break
            log(f"[optimize] iteration {i}: patching {[p['category'] for p in patches]}")
            new_res, new_sc, new_cfg = scored(cand, 0)
            if new_cfg != snaps[-1]["config"]:
                # Backend changed since the baseline was measured: those scores are not comparable. Re-score the CURRENT BEST
                # agent on the new backend (recalibrating the judge), then compare the candidate against that fresh baseline.
                prev = snaps[-1]
                log(f"[optimize] backend changed {prev['config']} -> {new_cfg}: re-scoring v{cur['version']} as a fresh baseline")
                b_res, b_sc, b_cfg = scored(cur, 3)
                if b_cfg != new_cfg:                      # judge calibration changed the mode: candidate must be re-scored too
                    new_res, new_sc, new_cfg = scored(cand, 0)
                if b_cfg != new_cfg:
                    raise InfraError("could not obtain a baseline and a candidate score on the same backend")
                rb = _snapshot(cur["version"], b_res, b_sc, dev_ids, held_ids, b_cfg)
                rb.update(rebaseline=True, replaces_config=prev["config"], previous_scores={"overall": prev["overall"], "dev": prev["dev"], "held_out": prev["held_out"]})
                snaps.append(rb)
                cur_res = b_res
                change = {"before_iteration": i, "from": prev["config"], "to": b_cfg, "version": cur["version"],
                          "old_scores": prev["overall"], "rescored": rb["overall"]}
                config_changes.append(change)
                entry["rebaseline"] = change
                save(f"eval_v{cur['version']}_rebaseline_{len(config_changes)}.json", {"agent": f"agent_v{cur['version']}", "scorecard": b_sc, "results": b_res, "config": b_cfg})
            before = {r["id"]: r["passed"] for r in cur_res}
            after = {r["id"]: r["passed"] for r in new_res}
            for p in patches:
                p["fixed"] = [x for x in p["targets"] if after[x]]
                p["still_failing"] = [x for x in p["targets"] if not after[x]]
            regress = [x for x in after if before[x] and not after[x]]
            snap = _snapshot(i, new_res, new_sc, dev_ids, held_ids, new_cfg)
            # `snaps[-1]` is always the last ACCEPTED (or freshly re-baselined) version on the SAME backend.
            accepted = snap["dev"]["passed"] > snaps[-1]["dev"]["passed"]
            entry.update(patches=patches, regressions=regress, newly_passing=[x for x in after if after[x] and not before[x]],
                         dev_before=snaps[-1]["dev"], dev_after=snap["dev"], held_out_before=snaps[-1]["held_out"],
                         held_out_after=snap["held_out"], overall_before=snaps[-1]["overall"], overall_after=snap["overall"],
                         config=new_cfg, accepted=accepted, outcome="accepted" if accepted else "REJECTED and reverted: dev pass count did not improve")
            iterations.append(entry)
            save(f"agent_v{i}.json", cand)
            save(f"eval_v{i}.json", {"agent": f"agent_v{i}.json", "scorecard": new_sc, "results": new_res, "config": new_cfg})
            log(f"   dev {snap['dev']['passed']}/{snap['dev']['n']} (was {snaps[-1]['dev']['passed']}), held-out {snap['held_out']['passed']}/{snap['held_out']['n']} "
                f"(was {snaps[-1]['held_out']['passed']}), regressions {regress or 'none'} -> {entry['outcome']}")
            if not accepted:
                stop = "last iteration did not improve the dev score; reverted"
                break
            cur, cur_res = cand, new_res
            snaps.append(snap)
    except InfraError as e:
        save("optimization_partial.json", {"status": "ABORTED (infrastructure, not a quality result)", "reason": f"{type(e).__name__}: {e}",
                                          "versions": snaps, "iterations": iterations, "config_changes": config_changes,
                                          "note": "No score from this file is final. Re-run the same command after the quota resets: finished model calls are cached."})
        log(f"[optimize] ABORTED: {e}\n   Partial log saved. Re-run the same command later; calls already made are cached and cost nothing.")
        raise
    save("agent_final.json", cur)
    final_fail = classify_all(cur_res, cases, corpus, dev_ids)
    f = snaps[-1]
    comparable = next(s for s in snaps if s["config"] == f["config"])      # earliest score taken on the final backend
    out = {"agent_in": agent_path, "evalset": evalset_path, "max_iterations": max_iter, "stop_reason": stop,
           "split": {"dev": sorted(dev_ids), "held_out": sorted(held_ids)}, "versions": snaps, "iterations": iterations,
           "config_changes": config_changes, "final_version": cur["version"], "final_failures": final_fail,
           "official_score": {"note": "all cases count, including suspected eval defects", "config": f["config"],
                              "comparable_baseline": {"version": comparable["version"], "rebaseline": bool(comparable.get("rebaseline")), **comparable["overall"]},
                              **f["overall"]}}
    save("optimization_log.json", out)
    log(f"[optimize] v0 -> v{cur['version']}: overall {snaps[0]['overall']['passed']}/{snaps[0]['overall']['n']} -> {f['overall']['passed']}/{f['overall']['n']}; "
        f"held-out {snaps[0]['held_out']['passed']}/{snaps[0]['held_out']['n']} -> {f['held_out']['passed']}/{f['held_out']['n']}; stop: {stop}")
    if config_changes:
        log(f"[optimize] NOTE: the backend changed {len(config_changes)} time(s); only scores on the same backend are comparable "
            f"(baseline for the final version: {comparable['overall']['passed']}/{comparable['overall']['n']}).")
    return out
