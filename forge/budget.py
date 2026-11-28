"""API budget control for hosted LLM providers (free tiers have tight per-minute and per-day limits).

What lives here
  * Pacing      : before each hosted call, wait until it fits inside the per-minute request/token windows.
  * Daily quota : usage is persisted in .forge_budget.json (UTC day), so a restart does not forget what was spent.
                  When a call would exceed the daily cap, BudgetExhausted is raised BEFORE sending it.
  * Accounting  : calls, cache hits, tokens and waiting time per pipeline stage (`with BUDGET.stage("evalgen")`).
  * Cache       : content-addressed response cache in .forge_cache/ so an interrupted run can be repeated for free.
  * Fallback    : a process-wide flag. Once the hosted budget is gone and FORGE_FALLBACK=ollama, every LLM object
                  switches to the local model for the rest of the process (never back, so one run never flip-flops).

Limits are the values you set in .env (FORGE_RPM, FORGE_TPM, FORGE_RPD, FORGE_TPD). They are applied per model id, as
Groq does, and scaled by FORGE_LIMIT_MARGIN (default 0.85) so we stay below the provider's own counter.
Token counts come from the provider's `usage` field when it is returned; otherwise they are estimated as chars/4.
"""
import contextlib, hashlib, json, os, time
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class InfraError(RuntimeError):
    """Anything that is the infrastructure's fault, never the agent's: it must abort a stage, not be scored."""


class RateLimitError(InfraError):
    pass


class BudgetExhausted(RateLimitError):
    """Daily quota gone (locally counted or reported by the provider). Retrying today will not help."""


class LLMUnavailable(InfraError):
    """Network failure, 5xx, bad key, unknown model id."""


def _utc_day():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def est_tokens(text_or_messages):
    if isinstance(text_or_messages, str):
        n = len(text_or_messages)
    else:
        n = sum(len(m.get("content", "")) for m in text_or_messages)
    return max(1, (n + 3) // 4)


def _empty_stage():
    return {"calls": 0, "cache_hits": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0, "waited_s": 0.0}


class Budget:
    def __init__(self, state_dir=None, clock=time.time, sleep=time.sleep, day_fn=_utc_day, log=print):
        self.state_dir = state_dir if state_dir is not None else os.getenv("FORGE_STATE_DIR", ROOT)
        self.clock, self.sleep, self.day_fn, self.log = clock, sleep, day_fn, log
        self.persist = True
        self.windows = {}                  # model -> [(t, tokens)] for the last 60 s
        self.stages = {}                   # stage -> counters (this process)
        self.events = []                   # backend changes, quota hits (this process)
        self.current_stage = "unscoped"
        self.fallback_active = False
        self.fallback_reason = None
        self._state = None

    # ------------------------------------------------------------------ config
    @property
    def state_path(self):
        return os.path.join(self.state_dir, ".forge_budget.json")

    @property
    def cache_dir(self):
        return os.path.join(self.state_dir, ".forge_cache")

    def limits(self, model=None):
        g = lambda k, d: float(os.getenv(k, d))
        m = g("FORGE_LIMIT_MARGIN", 0.85)
        return {"rpm": g("FORGE_RPM", 30), "tpm": g("FORGE_TPM", 8000), "rpd": g("FORGE_RPD", 1000), "tpd": g("FORGE_TPD", 200000),
                "margin": m}

    # ------------------------------------------------------------------ persistent daily usage
    def _load(self):
        today = self.day_fn()
        if self._state is None or self._state.get("date") != today:
            st = {"date": today, "models": {}}
            if self.persist and os.path.exists(self.state_path):
                try:
                    old = json.load(open(self.state_path, encoding="utf-8"))
                    if old.get("date") == today:
                        st = old
                except Exception:
                    pass                    # corrupt file: start the day fresh rather than crash
            self._state = st
        return self._state

    def _save(self):
        if not self.persist:
            return
        try:
            os.makedirs(self.state_dir, exist_ok=True)
            tmp = self.state_path + ".tmp"
            json.dump(self._state, open(tmp, "w", encoding="utf-8"), indent=1)
            os.replace(tmp, self.state_path)
        except OSError:
            pass                            # accounting must never crash a run

    def used_today(self, model):
        return dict(self._load()["models"].get(model, {"requests": 0, "tokens": 0}))

    def usage_report(self):
        """For `forge budget`: today's usage against the configured limits, per model."""
        st, lim = self._load(), self.limits()
        rows = []
        for model, u in sorted(st["models"].items()):
            rows.append({"model": model, "requests": u["requests"], "rpd": lim["rpd"], "tokens": u["tokens"], "tpd": lim["tpd"],
                         "tokens_left": max(0, int(lim["tpd"] - u["tokens"])), "requests_left": max(0, int(lim["rpd"] - u["requests"]))})
        return {"date": st["date"], "models": rows, "limits": lim}

    # ------------------------------------------------------------------ pacing + reservation
    def reserve(self, model, est):
        """Block until a call of ~`est` tokens fits; raise BudgetExhausted if today's cap would be exceeded.
        Returns a handle to pass to commit() or release()."""
        lim = self.limits(model)
        u = self._load()["models"].setdefault(model, {"requests": 0, "tokens": 0})
        if u["requests"] + 1 > lim["rpd"] or u["tokens"] + est > lim["tpd"]:
            self._event("daily_budget", f"{model}: {u['tokens']}/{int(lim['tpd'])} tokens, {u['requests']}/{int(lim['rpd'])} requests used today (UTC)")
            raise BudgetExhausted(f"daily budget for {model} is used up ({u['tokens']}/{int(lim['tpd'])} tokens, "
                                  f"{u['requests']}/{int(lim['rpd'])} requests, UTC day {self._state['date']}). Resets at 00:00 UTC.")
        win, waited = self.windows.setdefault(model, []), 0.0
        rpm, tpm = lim["rpm"] * lim["margin"], lim["tpm"] * lim["margin"]
        if est > tpm and not win:
            self.log(f"[budget] warning: one call (~{est} tokens) is larger than the per-minute allowance ({int(tpm)}); sending anyway")
        while True:
            now = self.clock()
            win[:] = [(t, k) for t, k in win if now - t < 60.0]
            if len(win) + 1 <= rpm and (sum(k for _, k in win) + est <= tpm or not win):
                break
            wait = max(0.05, win[0][0] + 60.0 - now + 0.05)
            if wait >= 1.0:
                self.log(f"[budget] pacing: waiting {wait:.0f}s to stay inside per-minute limits for {model}")
            self.sleep(wait)
            waited += wait
        handle = {"model": model, "t": self.clock(), "est": est}
        win.append((handle["t"], est))
        handle["entry"] = win[-1]
        if waited:
            self._stage()["waited_s"] += waited
        return handle

    def commit(self, handle, prompt_tokens, completion_tokens, seconds=0.0):
        actual = int(prompt_tokens) + int(completion_tokens)
        win = self.windows.get(handle["model"], [])
        if handle["entry"] in win:                        # replace the estimate by the real count
            win[win.index(handle["entry"])] = (handle["t"], actual)
        u = self._load()["models"].setdefault(handle["model"], {"requests": 0, "tokens": 0})
        u["requests"] += 1
        u["tokens"] += actual
        s = self._stage()
        s["calls"] += 1
        s["prompt_tokens"] += int(prompt_tokens)
        s["completion_tokens"] += int(completion_tokens)
        s["seconds"] += seconds
        self._save()

    def release(self, handle):
        win = self.windows.get(handle["model"], [])
        if handle["entry"] in win:
            win.remove(handle["entry"])

    def mark_exhausted(self, model):
        """The provider said the daily limit is hit: remember it so we do not keep asking."""
        lim = self.limits(model)
        u = self._load()["models"].setdefault(model, {"requests": 0, "tokens": 0})
        u["tokens"] = max(u["tokens"], int(lim["tpd"]))
        self._event("daily_budget", f"{model}: provider reported the daily limit")
        self._save()

    # ------------------------------------------------------------------ stages / events
    def _stage(self):
        return self.stages.setdefault(self.current_stage, _empty_stage())

    @contextlib.contextmanager
    def stage(self, name):
        prev, self.current_stage = self.current_stage, name
        self.stages.setdefault(name, _empty_stage())
        try:
            yield
        finally:
            self.current_stage = prev

    def note_cache_hit(self):
        self._stage()["cache_hits"] += 1

    def note_local_call(self, prompt_tokens, completion_tokens, seconds):
        s = self._stage()
        s["calls"] += 1
        s["prompt_tokens"] += int(prompt_tokens)
        s["completion_tokens"] += int(completion_tokens)
        s["seconds"] += seconds

    def note_wait(self, seconds):
        self._stage()["waited_s"] += seconds

    def _event(self, kind, detail):
        self.events.append({"time": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "stage": self.current_stage, "type": kind, "detail": detail})

    def activate_fallback(self, reason, description):
        self.fallback_active, self.fallback_reason = True, reason
        self._event("backend_fallback", description)
        self.log(f"[budget] BACKEND SWITCH: {description}. Scores measured before and after this point come from different models.")

    # ------------------------------------------------------------------ response cache
    def cache_key(self, provider, model, temperature, json_mode, messages, salt=None):
        raw = json.dumps([provider, model, round(float(temperature), 3), bool(json_mode), messages, salt], sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def cache_get(self, key):
        p = os.path.join(self.cache_dir, key[:2], key + ".json")
        try:
            return json.load(open(p, encoding="utf-8"))["response"]
        except Exception:
            return None

    def cache_put(self, key, response):
        d = os.path.join(self.cache_dir, key[:2])
        try:
            os.makedirs(d, exist_ok=True)
            tmp = os.path.join(d, key + ".tmp")
            json.dump({"response": response}, open(tmp, "w", encoding="utf-8"))
            os.replace(tmp, os.path.join(d, key + ".json"))
        except OSError:
            pass

    def cache_evict(self, key):
        try:
            os.remove(os.path.join(self.cache_dir, key[:2], key + ".json"))
        except OSError:
            pass

    def cache_clear(self):
        n = 0
        for root, _, files in os.walk(self.cache_dir):
            for f in files:
                os.remove(os.path.join(root, f)); n += 1
        return n

    # ------------------------------------------------------------------ run usage file
    def dump_usage(self, run_dir):
        """Merge this process's counters into <run_dir>/usage.json (additive across commands of one run)."""
        if not run_dir or not os.path.isdir(run_dir):
            return None
        p = os.path.join(run_dir, "usage.json")
        try:
            cur = json.load(open(p, encoding="utf-8"))
        except Exception:
            cur = {"stages": {}, "events": []}
        for name, s in self.stages.items():
            c = cur["stages"].setdefault(name, _empty_stage())
            for k, v in s.items():
                c[k] = round(c[k] + v, 2) if isinstance(v, float) else c[k] + v
        cur["events"] += [e for e in self.events if e not in cur["events"]]
        tot = _empty_stage()
        for s in cur["stages"].values():
            for k in tot:
                tot[k] += s[k]
        cur["totals"] = {k: (round(v, 1) if isinstance(v, float) else v) for k, v in tot.items()}
        cur["limits"] = self.limits()
        json.dump(cur, open(p, "w", encoding="utf-8"), indent=2)
        self.stages, self.events = {}, []         # so a second dump in one process does not double count
        return cur


BUDGET = Budget()
