"""Requirement 9 helper: the whole factory, unattended, in one command.
  brief + docs -> intake -> groundability -> generate -> evalgen -> evaluate/optimize (max 3) -> report.md
Every stage writes its artifacts into the run folder, so the folder is the evidence."""
import json, os
from .llm import get_llm, USAGE
from .budget import BUDGET, InfraError
from .intake import run_intake
from .generator import run_generate
from .evalgen import run_evalgen, EvalCoverageError
from .optimizer import optimize
from .report import build_report


def run_factory(brief, docs, out, answers=None, force=False, n_cases=24, max_iter=3):
    os.makedirs(out, exist_ok=True)
    gen, judge = get_llm("gen"), get_llm("judge")
    try:
        with BUDGET.stage("intake"):
            run_intake(brief, docs, out, gen, guard_llm=judge, unattended=answers is None, answers=answers, force=force)
    except SystemExit as e:                      # BLOCKED: say so honestly instead of building a useless agent
        msg = str(e)
        open(os.path.join(out, "report.md"), "w", encoding="utf-8").write(
            f"# ForgeAI run report\n\n**Brief:** {brief}\n\n## Result: BLOCKED\n\n{msg}\n\nNo agent was built. See requirements.md for the topic-by-topic evidence.\n")
        print(msg)
        return None
    except InfraError as e:
        return _incomplete(out, brief, "intake", e)
    req_path = os.path.join(out, "requirements.json")
    stage = "intake"
    try:
        stage = "generate"
        with BUDGET.stage(stage):
            run_generate(req_path, out, gen)
        stage = "evalgen"
        with BUDGET.stage(stage):
            run_evalgen(req_path, out, judge, n_cases)
        stage = "optimize"
        with BUDGET.stage(stage):
            optimize(os.path.join(out, "agent_v0.json"), os.path.join(out, "eval_set.json"), out, judge, max_iter=max_iter)
    except InfraError as e:
        return _incomplete(out, brief, stage, e)
    except EvalCoverageError as e:
        BUDGET.dump_usage(out)
        open(os.path.join(out, "report.md"), "w", encoding="utf-8").write(f"# ForgeAI run report\n\n## Result: INCOMPLETE\n\n{e}\n")
        raise
    BUDGET.dump_usage(out)                       # the report reads usage.json
    return build_report(out, dict(USAGE))


def _incomplete(out, brief, stage, err):
    """A quota or network failure is reported as INCOMPLETE. It is never turned into a quality score."""
    BUDGET.dump_usage(out)
    msg = (f"The run stopped during `{stage}` because of an infrastructure problem, not a quality result:\n\n`{type(err).__name__}: {err}`\n\n"
           "No scores from this run are final. Re-run the same command after the quota resets (00:00 UTC) or with FORGE_FALLBACK=ollama; "
           "model calls that already finished are cached and are not repeated.\n")
    open(os.path.join(out, "report.md"), "w", encoding="utf-8").write(f"# ForgeAI run report\n\n**Brief:** {brief}\n\n## Result: INCOMPLETE\n\n{msg}")
    print(f"[factory] INCOMPLETE at {stage}: {err}")
    return None
