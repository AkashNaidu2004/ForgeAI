"""Thin LLM wrapper. Providers: ollama (local), openai (any OpenAI-compatible API, e.g. Groq), mock (tests).

Config comes from env vars or a .env file in the project root (see .env.example).
  FORGE_PROVIDER ollama|openai | FORGE_MODEL | FORGE_JUDGE_MODEL | FORGE_BASE_URL | FORGE_API_KEY
"""
import json, os, re, time
import requests

USAGE = {"calls": 0, "seconds": 0.0, "prompt_chars": 0, "completion_chars": 0}


def _load_dotenv():
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if os.path.exists(p):
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()


class LLM:
    def __init__(self, role="gen", temperature=0.2):
        self.provider = os.getenv("FORGE_PROVIDER", "ollama")
        gen_model = os.getenv("FORGE_MODEL", "qwen2.5:7b")
        self.model = gen_model if role == "gen" else os.getenv("FORGE_JUDGE_MODEL", gen_model)
        self.role = role
        self.base_url = os.getenv(
            "FORGE_BASE_URL",
            "http://localhost:11434" if self.provider == "ollama" else "https://api.groq.com/openai/v1",
        )
        self.api_key = os.getenv("FORGE_API_KEY", "")
        self.temperature = temperature
        self.mock_fn = None  # tests: fn(messages) -> str

    def _post_openai(self, messages, temp, json_mode):
        body = {"model": self.model, "messages": messages, "temperature": temp}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        hdr = {"Authorization": f"Bearer {self.api_key}"}
        for attempt in range(4):
            r = requests.post(f"{self.base_url}/chat/completions", json=body, headers=hdr, timeout=300)
            if r.status_code == 429:               # free-tier rate limit: back off and retry
                time.sleep(2 ** attempt * 3)
                continue
            if r.status_code == 400 and "response_format" in body:
                body.pop("response_format")        # model does not support JSON mode: rely on prompt + parser
                continue
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        raise RuntimeError("LLM API kept failing (rate limit?)")

    def chat(self, messages, json_mode=False, temperature=None):
        t0 = time.time()
        temp = self.temperature if temperature is None else temperature
        if self.provider == "mock":
            out = self.mock_fn(messages)
        elif self.provider == "ollama":
            body = {"model": self.model, "messages": messages, "stream": False, "options": {"temperature": temp}}
            if json_mode:
                body["format"] = "json"
            r = requests.post(f"{self.base_url}/api/chat", json=body, timeout=600)
            r.raise_for_status()
            out = r.json()["message"]["content"]
        else:
            out = self._post_openai(messages, temp, json_mode)
        out = re.sub(r"<think>.*?</think>", "", out or "", flags=re.S).strip()  # reasoning models
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
