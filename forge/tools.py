"""Tool registry. The generator picks from REGISTRY by name; the runner executes them.
Each tool: description (shown to the model + used by the tool selector), args spec, and fn."""
import ast, csv, datetime, operator, os
import requests

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.Mod: operator.mod, ast.USub: operator.neg}


def calculator(expression, **_):
    """Safe arithmetic evaluator (no eval)."""
    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.operand))
        raise ValueError("unsupported expression")
    return str(ev(ast.parse(expression, mode="eval").body))


def current_date(**_):
    return datetime.date.today().isoformat()


def csv_lookup(file, column, value, docs_dir=".", **_):
    """Exact/contains match on one column of a CSV in the agent's document folder."""
    path = os.path.join(docs_dir, os.path.basename(file))
    with open(path, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if str(value).lower() in str(r.get(column, "")).lower()]
    return "\n".join("; ".join(f"{k}: {v}" for k, v in r.items()) for r in rows[:5]) or "no matching rows"


def http_weather(latitude, longitude, **_):
    """Keyless open-meteo current weather."""
    r = requests.get("https://api.open-meteo.com/v1/forecast",
                     params={"latitude": latitude, "longitude": longitude, "current_weather": True}, timeout=15)
    return str(r.json().get("current_weather"))


REGISTRY = {
    "calculator": {"fn": calculator, "args": {"expression": "arithmetic expression, e.g. 12*4.5"},
                   "description": "Evaluate arithmetic: totals, percentages, unit costs, quantity x price."},
    "current_date": {"fn": current_date, "args": {},
                     "description": "Return today's date. For deadlines, notice periods, expiry, 'how many days until'."},
    "csv_lookup": {"fn": csv_lookup, "args": {"file": "csv filename", "column": "column name", "value": "text to match"},
                   "description": "Look up rows in a tabular CSV (catalogue, price list, part numbers) by column value."},
    "weather_api": {"fn": http_weather, "args": {"latitude": "float", "longitude": "float"},
                    "description": "Live weather conditions for a location (open-meteo). Only for briefs about weather, travel or outdoor operations."},
}


def run_tool(name, args, docs_dir="."):
    if name not in REGISTRY:
        return f"error: unknown tool {name}"
    try:
        return REGISTRY[name]["fn"](**args, docs_dir=docs_dir)
    except Exception as e:
        return f"error: {e}"
