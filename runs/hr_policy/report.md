# ForgeAI run report: shore_hr_travel_assistant

**Brief:** I need an assistant that answers HR policy questions for shore staff, and can also book flights

## 1. Headline

- Official score (all 22 eval cases, nothing removed): **19 -> 19 passed** (86% -> 86%).
- Dev cases (the optimizer may learn from these): 15/17 -> 15/17.
- Held-out cases (not diagnosed during optimization or used for patches): 4/5 -> 4/5.
- Final agent: `agent_final.json` (version 0). Optimizer stopped because: last iteration did not improve the dev score; reverted.
- Still failing: **3** case(s); see section 7.

## 2. What was asked and what the documents support

Groundability status: **partial**. Scope as built: The assistant provides information on annual leave entitlements and the procedures for requesting leave. It also covers sick leave requirements, travel expense allowances, and expense reporting deadlines.

- KEPT (grounded): Annual leave entitlements
- KEPT (grounded): Leave request procedures
- KEPT (grounded): Sick leave requirements
- KEPT (grounded): Travel expense allowances
- KEPT (grounded): Expense reporting deadlines
- REFUSED: Requests from employees not classified as shore staff
- REFUSED: Flight booking requests that violate travel policy (e.g., business class)
- REFUSED: HR policy questions outside of leave and travel (e.g., benefits, code of conduct)
- REFUSED: Direct execution of flight bookings or financial transactions
- REFUSED: Flight booking procedures (not covered by the documents)

## 3. Tools

- `calculator`: not attached: no evidence in brief/topics
- `current_date`: not attached: neither model nor requirements asked for it
- `csv_lookup`: not attached: no CSV document in folder
- `weather_api`: not attached: no evidence in brief/topics

## 4. Eval set

22 cases by type: {'multi_doc': 3, 'ambiguous': 2, 'answerable': 11, 'out_of_scope': 6}. Generated with `openai/gpt-oss-20b` (prompts in `prompts/evalgen_*.txt`). Candidates rejected by code validation: {'verification_rejected': 12, 'missing_batch_verdict': 1, 'duplicate_question': 3}.

## 5. Judge reliability

- Mode: deterministic checks + LLM judge (quote-verified). Judge model: `llama3.2:3b`; agent model: `qwen2.5:7b`.
- Canary accuracy 0.92 over 12 known answers {'gold': 1.0, 'wrong': 1.0, 'padded_wrong': 1.0, 'fabricated': 0.67}.

## 6. Iterations

| Version | Overall | Dev | Held-out | Scored by |
|---|---|---|---|---|
| v0 | 19/22 | 15/17 | 4/5 | agent `ollama:qwen2.5:7b` / judge `ollama:llama3.2:3b` |

### Iteration 1: REJECTED and reverted: dev pass count did not improve

Dev failures triaged: amb-01 (incomplete_answer), out-05 (refusal_flag)
- Patch [incomplete_answer] `prompt_sections.completeness_policy` (critic): "Ensure all relevant facts from the retrieved passages are incorporated into your answer, and if any information is missing, provide a clear indication of what is missing." Fixed none; still failing ['amb-01'].
- Patch [refusal_flag] `prompt_sections.contract_policy` (template): "Whenever you decline or say the information is unavailable, you must also set refused=true and sources=[]." Fixed none; still failing ['out-05'].
- Dev 15 -> 14; held-out 4 -> 4; regressions: ['out-03'].

## 7. What still fails, and why

| Case | Split | Type | Triage | Genuine agent defect? | Note |
|---|---|---|---|---|---|
| amb-01 | dev | ambiguous | incomplete_answer | yes | fact recall 0.0 |
| amb-02 | held_out | ambiguous | incomplete_answer | yes | fact recall 0.0 |
| out-05 | dev | out_of_scope | refusal_flag | yes | text declines but refused=false |

Questions and answers for the failures above:

- **amb-01** Q: How many days can I carry over?
  A: You can carry over up to 5 unused days of annual leave into the next year.
- **amb-02** Q: What do I need if I’m sick for more than two days?
  A: If you are sick for more than two consecutive days, you need a medical certificate.
- **out-05** Q: What is the procedure for taking a leave of absence for a family emergency?
  A: The procedure for taking a leave of absence for a family emergency is similar to requesting annual or sick leave. It must be requested at least 14 days in advance through the HR portal. If the leave is for a family emergency and the notice...

## 8. Limitations

- The eval set was written by an LLM from the same documents, and only 22 cases were kept after validation. Differences of one or two cases are within noise.
- The held-out set has only 5 cases, so held-out numbers are indicative, not statistically meaningful.
- Model sampling is not fully deterministic: re-running the same agent can change a borderline case's verdict.
- At least one iteration was rejected because it did not improve the dev score; the agent was reverted.
- Part of the brief could not be grounded in the documents; the agent is deliberately configured to refuse those requests.
- Retrieval is lexical (BM25) unless embeddings are enabled, so synonyms the documents never use can be missed.
- The optimizer can only add rules and tune retrieval parameters. It does not fix missing or contradictory source documents.

## 9. API usage and backend

Model calls actually sent (cache hits cost nothing). Tokens are the provider's own counts when it returned them, otherwise estimates.

| Stage | Calls | Cache hits | Prompt tokens | Completion tokens | Waited (s) |
|---|---|---|---|---|---|
| evalgen | 41 | 26 | 36520 | 49056 | 714 |
| evaluate | 72 | 28 | 39327 | 10738 | 110 |
| optimize | 26 | 64 | 16463 | 1442 | 0 |
| **total** | 139 | 118 | 92310 | 61236 | 824 |

Per-minute pacing used limits of 30 requests and 8000 tokens per model (x0.85 safety margin); daily cap 200000 tokens. If a run was resumed, calls from every attempt are added together.

Backend and quota events:
- 2026-11-28T10:20:01Z during `evaluate`: daily_budget: openai/gpt-oss-20b: provider reported the daily limit
- 2026-11-28T10:20:04Z during `evaluate`: backend_fallback: hosted `openai/gpt-oss-20b` unavailable (provider reports the DAILY limit for openai/gpt-oss-20b is reached); continuing on local Ollama (gen `qwen2.5:7b`, judge `llama3.2:3b`)
