"""Thin LLM wrapper. Providers: ollama (local), openai (any OpenAI-compatible API, e.g. Groq), mock (tests).

Config comes from env vars or a .env file in the project root (see .env.example).
  FORGE_PROVIDER ollama|openai | FORGE_MODEL | FORGE_JUDGE_MODEL | FORGE_BASE_URL | FORGE_API_KEY
Hosted calls go through forge.budget (pacing, daily quota, cache). Optional local fallback: FORGE_FALLBACK=ollama.
"""
import json, os, re, time
import requests
from .budget import BUDGET, InfraError, RateLimitError, BudgetExhausted, LLMUnavailable, est_tokens  # noqa: F401 (re-exported)

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


MAX_WAIT = 90          # longest single wait we accept after a per-minute 429 (seconds)
CACHE_ON = lambda: os.getenv("FORGE_CACHE", "1") != "0"
COMPLETION_RESERVE = lambda: int(os.getenv("FORGE_COMPLETION_RESERVE", "800"))   # tokens assumed for the reply when pacing


def _daily_limit_hit(resp):
    txt = resp.text if isinstance(getattr(resp, "text", None), str) else ""
    t = txt.lower()
    return any(k in t for k in ("per day", "(tpd)", "(rpd)", "daily"))


def _fallback_models(role):
    gen = os.getenv("FORGE_FALLBACK_MODEL", "qwen2.5:7b")
    return gen if role == "gen" else os.getenv("FORGE_FALLBACK_JUDGE_MODEL", "llama3.2:3b")


class LLM:
    def __init__(self, role="gen", temperature=0.2):
        self._provider = os.getenv("FORGE_PROVIDER", "ollama")
        gen_model = os.getenv("FORGE_MODEL", "qwen2.5:7b")
        self._model = gen_model if role == "gen" else os.getenv("FORGE_JUDGE_MODEL", gen_model)
        self.role = role
        self._base_url = os.getenv(
            "FORGE_BASE_URL",
            "http://localhost:11434" if self._provider == "ollama" else "https://api.groq.com/openai/v1",
        )
        self.api_key = os.getenv("FORGE_API_KEY", "")
        self.temperature = temperature
        self.mock_fn = None  # tests: fn(messages) -> str
        self.last_key = None

    # provider / model / base_url report the EFFECTIVE backend: after a fallback switch they name the local model,
    # so reports and judge-vs-agent comparisons never show a model that did not actually answer.
    def _fallen_back(self):
        return BUDGET.fallback_active and self._provider == "openai"

    provider = property(lambda s: "ollama" if s._fallen_back() else s._provider, lambda s, v: setattr(s, "_provider", v))
    model = property(lambda s: _fallback_models(s.role) if s._fallen_back() else s._model, lambda s, v: setattr(s, "_model", v))
    base_url = property(lambda s: os.getenv("FORGE_FALLBACK_URL", "http://localhost:11434") if s._fallen_back() else s._base_url,
                        lambda s, v: setattr(s, "_base_url", v))

    # ------------------------------------------------------------------ hosted (OpenAI-compatible)
    def _post_openai(self, messages, temp, json_mode):
        """-> (text, prompt_tokens, completion_tokens). Paced by the budget; bounded retries; never loops forever."""
        body = {"model": self.model, "messages": messages, "temperature": temp}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        hdr = {"Authorization": f"Bearer {self.api_key}"}
        est = est_tokens(messages) + COMPLETION_RESERVE()
        handle = BUDGET.reserve(self.model, est)
        t0 = time.time()
        try:
            transient = 0
            for attempt in range(4):
                try:
                    r = requests.post(f"{self.base_url}/chat/completions", json=body, headers=hdr, timeout=(10, 90))
                except requests.RequestException as e:
                    transient += 1
                    if transient > 2:
                        raise LLMUnavailable(f"network error talking to {self.base_url}: {type(e).__name__}") from e
                    time.sleep(2 * transient)
                    continue
                if r.status_code == 429:               # free-tier rate limit
                    if _daily_limit_hit(r):
                        BUDGET.mark_exhausted(self.model)
                        raise BudgetExhausted(f"provider reports the DAILY limit for {self.model} is reached")
                    try:
                        delay = max(3.0, float(r.headers.get("Retry-After", 2 ** attempt * 3)))
                    except (ValueError, TypeError):
                        delay = 2 ** attempt * 3
                    if attempt == 3 or delay > MAX_WAIT:
                        raise RateLimitError("HTTP 429: retry budget exhausted or provider requests a wait longer than "
                                             f"{MAX_WAIT}s. Check provider quota before retrying.")
                    print(f"[llm] HTTP 429; retry {attempt+1}/3 in {delay:.0f}s", flush=True)
                    time.sleep(delay)
                    BUDGET.note_wait(delay)
                    continue
                if r.status_code == 400 and "response_format" in body:
                    body.pop("response_format")        # model does not support JSON mode: rely on prompt + parser
                    continue
                if r.status_code >= 500:
                    transient += 1
                    if transient > 2:
                        raise LLMUnavailable(f"HTTP {r.status_code} from provider")
                    time.sleep(2 * transient)
                    continue
                if r.status_code >= 400:
                    detail = r.text[:200] if isinstance(r.text, str) else ""
                    raise LLMUnavailable(f"HTTP {r.status_code} for model {self.model}: {detail} (wrong model id or key?)")
                data = r.json()
                text = data["choices"][0]["message"]["content"] or ""
                u = data.get("usage") or {}
                pt = int(u.get("prompt_tokens") or est_tokens(messages))
                ct = int(u.get("completion_tokens") or est_tokens(text))
                BUDGET.commit(handle, pt, ct, time.time() - t0)
                return text, pt, ct
            raise LLMUnavailable("LLM API kept failing")
        except BaseException:
            BUDGET.release(handle)
            raise

    def _post_ollama(self, messages, temp, json_mode):
        body = {"model": self.model, "messages": messages, "stream": False, "options": {"temperature": temp}}
        if json_mode:
            body["format"] = "json"
        t0 = time.time()
        try:
            r = requests.post(f"{self.base_url}/api/chat", json=body, timeout=600)
            r.raise_for_status()
        except requests.RequestException as e:
            raise LLMUnavailable(f"Ollama call failed ({type(e).__name__}); is `ollama serve` running and the model pulled?") from e
        d = r.json()
        text = d["message"]["content"]
        pt, ct = int(d.get("prompt_eval_count") or est_tokens(messages)), int(d.get("eval_count") or est_tokens(text))
        BUDGET.note_local_call(pt, ct, time.time() - t0)
        return text, pt, ct

    def _try_fallback(self, err):
        if os.getenv("FORGE_FALLBACK", "").lower() != "ollama" or self._provider != "openai" or BUDGET.fallback_active:
            return False
        try:
            requests.get(os.getenv("FORGE_FALLBACK_URL", "http://localhost:11434") + "/api/tags", timeout=3).raise_for_status()
        except requests.RequestException as e:
            raise LLMUnavailable(f"hosted budget exhausted ({err}) and the Ollama fallback is not reachable") from e
        BUDGET.activate_fallback(str(err), f"hosted `{self._model}` unavailable ({err}); continuing on local Ollama "
                                           f"(gen `{_fallback_models('gen')}`, judge `{_fallback_models('judge')}`)")
        return True

    # ------------------------------------------------------------------ public
    def chat(self, messages, json_mode=False, temperature=None, cache_salt=None):
        t0 = time.time()
        temp = self.temperature if temperature is None else temperature
        key = None
        for _ in range(2):                      # second pass only after a fallback switch
            if self.provider == "mock":
                out, key = self.mock_fn(messages), None
                break
            key = BUDGET.cache_key(self.provider, self.model, temp, json_mode, messages, cache_salt) if CACHE_ON() else None
            if key:
                hit = BUDGET.cache_get(key)
                if hit is not None:
                    BUDGET.note_cache_hit()
                    self.last_key = key
                    return hit
            try:
                if self.provider == "ollama":
                    out, _, _ = self._post_ollama(messages, temp, json_mode)
                else:
                    out, _, _ = self._post_openai(messages, temp, json_mode)
                break
            except RateLimitError as e:
                if self._try_fallback(e):
                    continue
                raise
        out = re.sub(r"<think>.*?</think>", "", out or "", flags=re.S).strip()  # reasoning models
        if key and out:
            BUDGET.cache_put(key, out)
        self.last_key = key
        USAGE["calls"] += 1
        USAGE["seconds"] += time.time() - t0
        USAGE["prompt_chars"] += sum(len(m["content"]) for m in messages)
        USAGE["completion_chars"] += len(out)
        return out

    def json(self, messages, retries=2, temperature=None, cache_salt=None):
        """Ask for JSON; small models often wrap or garble it, so parse defensively and retry.
        A cached reply that does not parse is evicted, so a bad answer is never replayed on a re-run."""
        last = ""
        for _ in range(retries + 1):
            last = self.chat(messages, json_mode=True, temperature=temperature, cache_salt=cache_salt)
            obj = extract_json(last)
            if obj is not None:
                return obj
            if self.last_key:
                BUDGET.cache_evict(self.last_key)
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
