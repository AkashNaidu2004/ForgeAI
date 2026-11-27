import os
_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prompts")


def render(_template, **kw):
    """Load prompts/<template>.txt and substitute {{key}} placeholders (JSON braces stay untouched)."""
    t = open(os.path.join(_DIR, _template + ".txt"), encoding="utf-8").read()
    for k, v in kw.items():
        t = t.replace("{{" + k + "}}", str(v))
    return t
