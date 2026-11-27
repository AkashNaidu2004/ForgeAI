"""Thin LLM wrapper. Providers: ollama (local), openai (any OpenAI-compatible API), mock (tests).

Env vars:
  FORGE_PROVIDER    ollama | openai          (default ollama)
  FORGE_MODEL       generator model name     (default qwen2.5:7b)
  FORGE_JUDGE_MODEL judge model name         (use a different model family than the generator)
  FORGE_BASE_URL    API base URL
  FORGE_API_KEY     key for openai-compatible providers
"""
import json, os, re, time
import requests

USAGE = {"calls": 0, "seconds": 0.0, "prompt_chars": 0, "completion_chars": 0}


class LLM:
    def __init__(self, role="gen", temperature=0.2):
        self.provider = os.getenv("FORGE_PROVIDER", "ollama")
        gen_model = os.getenv("FORGE_MODEL", "qwen2.5:7b")
        self.model = gen_model if role == "gen" else os.getenv("FORGE_JUDGE_MODEL", gen_model)
        self.base_url = os.getenv(
            "FORGE_BASE_URL",
            "http://localhost:11434" if self.provider == "ollama" else "https://api.groq.com/openai/v1",
        )
        self.api_key = os.getenv("FORGE_API_KEY", "")
        self.temperature = temperature
        self.mock_fn = None  # tests: fn(messages) -> str

    def chat(self, messages, json_mode=False, temperature=None):
        t0 = time.time()
        temp = self.temperature if temperature is None else temperature
        if self.provider == "mock":
            out = self.mock_fn(messages)
        elif self.provider == "ollama":
            body = {"model": self.model, "messages": messages, "stream": False,
                    "options": {"temperature": temp}}
            if json_mode:
                body["format"] = "json"
            r = requests.post(f"{self.base_url}/api/chat", json=body, timeout=300)
            r.raise_for_status()
            out = r.json()["message"]["content"]
        else:
            body = {"model": self.model, "messages": messages, "temperature": temp}
            if json_mode:
                body["response_format"] = {"type": "json_object"}
            r = requests.post(f"{self.base_url}/chat/completions", json=body, timeout=300,
                              headers={"Authorization": f"Bearer {self.api_key}"})
            r.raise_for_status()
            out = r.json()["choices"][0]["message"]["content"]
        USAGE["calls"] += 1
        USAGE["seconds"] += time.time() - t0
        USAGE["prompt_chars"] += sum(len(m["content"]) for m in messages)
        USAGE["completion_chars"] += len(out)
        return out

    def json(self, messages, retries=2, temperature=None):
        """Ask for JSON; small models often wrap or garble it, so parse defensively and retry."""
        last = ""
        for _ in range(retries + 1):
            last = self.chat(messages, json_mode=True, temperature=temperature)
            obj = extract_json(last)
            if obj is not None:
                return obj
            messages = messages + [
                {"role": "assistant", "content": last},
                {"role": "user", "content": "That was not valid JSON. Reply with ONLY a valid JSON object."},
            ]
        raise ValueError(f"LLM never returned valid JSON. Last output: {last[:300]}")


def extract_json(text):
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    for cand in (text, _between(text, "{", "}"), _between(text, "[", "]")):
        if cand:
            try:
                return json.loads(cand)
            except Exception:
                pass
    return None


def _between(text, a, b):
    i, j = text.find(a), text.rfind(b)
    return text[i:j + 1] if i != -1 and j > i else None


def get_llm(role="gen", **kw):
    return LLM(role, **kw)
