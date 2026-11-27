"""Generic agent runtime: loads ANY definition (json) and answers questions with retrieval + tools.

Definition schema:
{ name, description, system_prompt, docs_path,
  retrieval: {chunk_size, overlap, top_k, mode, min_score},
  tools: [names], output_contract: {fields:[answer, sources, refused]}, refusal_message }
"""
import json, os
from .docs import load_folder, chunk_all
from .retrieval import Retriever
from .tools import REGISTRY, run_tool
from .llm import get_llm

MAX_TOOL_STEPS = 2

CONTRACT = ('Reply with ONLY a JSON object: {"answer": str, "sources": [document filenames you used], '
            '"refused": bool}. If the context does not contain the answer, or the question is outside '
            'your scope, set "refused": true, "sources": [] and say so briefly in "answer". '
            'To use a tool instead, reply {"tool": name, "args": {...}}.')


def load_definition(path):
    d = json.load(open(path, encoding="utf-8"))
    d.setdefault("tools", [])
    d.setdefault("retrieval", {})
    for k, v in dict(chunk_size=200, overlap=40, top_k=4, mode="bm25", min_score=0.0).items():
        d["retrieval"].setdefault(k, v)
    return d


class Agent:
    def __init__(self, definition, llm=None):
        self.d = definition
        r = self.d["retrieval"]
        docs = load_folder(self.d["docs_path"])
        self.retriever = Retriever(chunk_all(docs, r["chunk_size"], r["overlap"]), r["mode"])
        self.llm = llm or get_llm("gen")

    def _tool_block(self):
        if not self.d["tools"]:
            return "You have no tools."
        lines = [f'- {t}: {REGISTRY[t]["description"]} args={json.dumps(REGISTRY[t]["args"])}' for t in self.d["tools"]]
        return "Tools you may call:\n" + "\n".join(lines)

    def answer(self, question):
        """Returns dict: answer, sources, refused, retrieved (chunk ids + scores), tool_calls."""
        r = self.d["retrieval"]
        hits = [(c, s) for c, s in self.retriever.search(question, r["top_k"]) if s >= r["min_score"]]
        ctx = "\n\n".join(f"[{c.source}]\n{c.text}" for c, _ in hits) or "(nothing relevant retrieved)"
        sys = f'{self.d["system_prompt"]}\n\n{self._tool_block()}\n\n{CONTRACT}'
        msgs = [{"role": "system", "content": sys},
                {"role": "user", "content": f"CONTEXT:\n{ctx}\n\nQUESTION: {question}"}]
        calls, out = [], {}
        for _ in range(MAX_TOOL_STEPS + 1):
            out = self.llm.json(msgs)
            if "tool" in out and out["tool"] in self.d["tools"] and len(calls) < MAX_TOOL_STEPS:
                res = run_tool(out["tool"], out.get("args", {}), self.d["docs_path"])
                calls.append({"tool": out["tool"], "args": out.get("args", {}), "result": res})
                msgs += [{"role": "assistant", "content": json.dumps(out)},
                         {"role": "user", "content": f"TOOL RESULT: {res}\nNow give the final JSON answer."}]
                continue
            break
        return {"answer": str(out.get("answer", "")), "sources": out.get("sources", []) or [],
                "refused": bool(out.get("refused", False)),
                "retrieved": [(c.id, round(s, 3)) for c, s in hits], "tool_calls": calls}
