"""Requirement 3: agent generator. requirements.json + docs -> declarative agent definition.
Design: the LLM only writes the persona and proposes tools. Everything safety-critical (grounding rules, refusal
rules, tool attachment, retrieval params) is assembled or VERIFIED in code, because a small model cannot be trusted
to remember them. The system prompt is kept as named sections so the optimizer can patch one section at a time."""
import json, os, re
from .docs import load_folder
from .prompts import render
from .tools import REGISTRY

SIGNAL = {"calculator": "needs_calculation", "current_date": "needs_current_date",
          "weather_api": "needs_live_data", "csv_lookup": "needs_table_lookup"}

KEYWORDS = {
    "calculator": ["calculate", "calculation", "total cost", "totals", "cost", "price", "pricing", "quote", "quantity",
                   "percentage", "percent", "budget", "convert", "how much", "discount", "tax", "margin"],
    "current_date": ["today", "current date", "deadline", "days until", "days left", "expiry", "expires", "overdue", "due date", "tomorrow"],
    "weather_api": ["weather", "forecast", "temperature", "rain", "wind speed", "storm", "sea state", "humidity"],
    "csv_lookup": ["catalogue", "catalog", "part number", "part no", "price list", "stock", "inventory", "sku", "spares", "lookup", "look up"],
}

GROUNDING_RULES = [
    "Answer ONLY from the CONTEXT passages supplied with each question (and tool results). Never use outside knowledge.",
    "Never guess numbers, dates, names, limits or policies. Quote figures exactly as written in the context.",
    "If the context does not contain the answer, set refused=true and say what you could not find. Do not improvise.",
    "List every document you used in sources, using its exact filename. Do not list documents you did not use.",
    "If answering needs facts from two documents, combine them and cite both.",
    "If the question is ambiguous, state the ambiguity in one sentence, then answer the most likely reading using the context.",
]


def _text_for_keywords(req):
    # Uses the repaired scope and kept topics, NOT the raw brief (it still contains rejected requests).
    parts = [req["scope"]] + [t["topic"] + " " + t["example_question"] for t in req["must_answer_topics"]]
    return " ".join(parts).lower()


def _hits(tool, text):
    return [k for k in KEYWORDS[tool] if re.search(r"\b" + re.escape(k), text)]


def csv_info(docs):
    info = []
    for d in docs:
        if d.kind == "csv" and d.text.strip():
            first = d.text.split("\n")[0]
            info.append((d.name, [seg.split(":")[0].strip() for seg in first.split("; ")]))
    return info


def select_tools(req, docs, llm):
    """LLM proposes, code disposes: a tool is attached only with (a) a keyword in the brief/topics, (b) an LLM yes
    or a requirement signal, and (c) its preconditions met. This blocks 'HR agent with a weather tool'."""
    registry = "\n".join(f"- {n}: {t['description']}" for n, t in REGISTRY.items())
    prop = llm.json([{"role": "user", "content": render(
        "tool_selection", brief=req["scope"], scope=req["scope"], files=", ".join(d.name for d in docs),
        topics="\n".join(f"- {t['topic']}" for t in req["must_answer_topics"]), registry=registry)}], temperature=0)
    by_name = {str(t.get("name")): t for t in prop.get("tools", []) if isinstance(t, dict)}
    text, decisions, attached = _text_for_keywords(req), [], []
    for name in REGISTRY:
        p = by_name.get(name, {})
        llm_use = bool(p.get("use")) and bool(str(p.get("justification", "")).strip())
        signal = bool(req.get("tool_signals", {}).get(SIGNAL[name]))
        hits = _hits(name, text)
        pre = any(d.kind == "csv" for d in docs) if name == "csv_lookup" else True
        ok = bool(hits) and (llm_use or signal) and pre
        why = (f"attached: keywords {hits}; " + ("model justified: " + str(p.get("justification", "")) if llm_use else "requirement signal")) if ok else \
              "not attached: " + ("no CSV document in folder" if not pre else "no evidence in brief/topics" if not hits else "neither model nor requirements asked for it")
        decisions.append({"tool": name, "llm_use": llm_use, "signal": signal, "keyword_hits": hits,
                          "precondition_ok": pre, "attached": ok, "reason": why})
        if ok:
            attached.append(name)
    return attached, decisions


def choose_retrieval(docs):
    words = sum(len(d.text.split()) for d in docs)
    csv_share = sum(len(d.text.split()) for d in docs if d.kind == "csv") / max(words, 1)
    size = 120 if csv_share > 0.5 else 150 if words < 2000 else 200
    top_k = 3 if words < 1500 else 5 if words > 20000 else 4
    mode = "hybrid" if os.getenv("FORGE_USE_EMBEDDINGS") == "1" else "bm25"
    return {"chunk_size": size, "overlap": size // 5, "top_k": top_k, "mode": mode, "min_score": 0.0,
            "rationale": f"{words} words, {csv_share:.0%} tabular; top_k scales with corpus size; bm25 unless embeddings enabled"}


def build_sections(req, persona, tools, docs):
    refuse = [f"- {t}" for t in req["must_refuse_topics"]]
    s = {
        "role": persona["role_line"],
        "scope": f"Scope: {req['scope']}\nAudience: {req['audience']}",
        "grounding_rules": "GROUNDING RULES:\n" + "\n".join(f"{i + 1}. {r}" for i, r in enumerate(GROUNDING_RULES)),
        "refusal_rules": "REFUSE (refused=true, one polite sentence, no guessing) when asked about:\n" + "\n".join(refuse) +
                         "\n- anything not supported by the documents",
        "style_rules": f"STYLE (tone: {req['tone']}):\n" + "\n".join(f"- {r}" for r in persona["style_rules"]),
    }
    if req.get("partial_notes"):
        s["partial_rules"] = "PARTIAL COVERAGE (give what the documents say, then state what is missing):\n" + \
            "\n".join(f"- {k}: {v}" for k, v in req["partial_notes"].items())
    if tools:
        lines = ["TOOL RULES: use a tool only when the question needs it. Never invent a tool result. "
                 "After a tool result, give the final answer and still cite the documents you used."]
        for n in tools:
            lines.append(f"- {n}: {REGISTRY[n]['description']}")
        for fname, cols in csv_info(docs):
            lines.append(f"- csv_lookup file '{fname}' has columns: {', '.join(cols)}")
        s["tool_rules"] = "\n".join(lines)
    return s


ORDER = ["role", "scope", "grounding_rules", "refusal_rules", "partial_rules", "tool_rules", "style_rules"]


def render_system_prompt(sections):
    return "\n\n".join(sections[k] for k in ORDER if k in sections)


def generate_agent(req, llm):
    docs = load_folder(req["docs_path"])
    persona = llm.json([{"role": "user", "content": render("generate_persona", scope=req["scope"],
                                                          audience=req["audience"], tone=req["tone"])}], temperature=0.2)
    persona = {"role_line": str(persona.get("role_line") or f"You are an assistant for {req['audience']}."),
               "style_rules": [str(x) for x in persona.get("style_rules", [])][:4] or ["Be concise and accurate."]}
    tools, decisions = select_tools(req, docs, llm)
    sections = build_sections(req, persona, tools, docs)
    definition = {
        "version": 0, "name": req["agent_name"], "description": req["scope"], "docs_path": req["docs_path"],
        "prompt_sections": sections, "system_prompt": render_system_prompt(sections),
        "retrieval": choose_retrieval(docs), "tools": tools,
        "output_contract": {"fields": {"answer": "string", "sources": "list of document filenames", "refused": "bool"}},
        "refusal_message": "I can't help with that because it isn't covered by the documents I have.",
        "tool_decisions": decisions,
    }
    return definition


def run_generate(req_path, out_dir, llm):
    req = json.load(open(req_path, encoding="utf-8"))
    d = generate_agent(req, llm)
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, "agent_v0.json")
    json.dump(d, open(p, "w", encoding="utf-8"), indent=2)
    print(f"[generate] {p} | tools attached: {d['tools'] or 'none'}")
    for x in d["tool_decisions"]:
        print(f"   {x['tool']:<13} {x['reason']}")
    return d
