# ForgeAI run report: ship_engineer_assistant

**Brief:** Build an assistant for ship engineers to look up spare part numbers, prices, stock availability and supplier terms, and calculate order totals with applicable supplier discounts. Do not invent prices, discounts or supplier policies.

## 1. Headline

- Official score (all 24 eval cases, nothing removed): **11 -> 12 passed** (46% -> 50%).
- Dev cases (the optimizer may learn from these): 9/18 -> 11/18.
- Held-out cases (not diagnosed during optimization or used for patches): 2/6 -> 1/6.
- Final agent: `agent_final.json` (version 1). Optimizer stopped because: last iteration did not improve the dev score; reverted.
- Still failing: **12** case(s); see section 7.

## 2. What was asked and what the documents support

Groundability status: **partial**. Scope as built: This assistant helps ship engineers find spare part numbers, check prices, stock availability, and supplier terms, and calculate order totals with applicable discounts. It stops at providing real-time notifications or handling multiple currencies.

- KEPT (partial): Part lookup
- KEPT (grounded): Price inquiry
- KEPT (partial): Stock check
- KEPT (partial): Supplier terms
- KEPT (grounded): Order total calculation
- REFUSED: Legal advice
- REFUSED: Medical information
- REFUSED: Personal data
- REFUSED: Real-time stock updates
- PARTIAL: Part lookup: supported: lookup for part number SP-1001; NOT supported (say so, never guess): part number for a marine engine piston ring, specific unit price and stock information for this part, supplier information for this part
- PARTIAL: Stock check: supported: looking up rows is supported if the table's documents appear among the passages; NOT supported (say so, never guess): unspecified
- PARTIAL: Supplier terms: supported: Delivery within 21 days for OceanFlow Supplies; NOT supported (say so, never guess): Delivery times for emergency orders from Supplier B, including the extra 15 percent charge for air freight

## 3. Tools

- `calculator`: attached: keywords ['calculate', 'calculation', 'total cost', 'totals', 'cost', 'price', 'how much', 'discount']; model justified: The assistant needs to calculate order totals with applicable discounts, which requires arithmetic evaluation capabilities.
- `current_date`: not attached: no evidence in brief/topics
- `csv_lookup`: attached: keywords ['part number', 'stock', 'lookup']; model justified: The assistant needs to look up part numbers, prices, and stock availability, which can be done using the CSV file 'catalogue.csv'.
- `weather_api`: not attached: no evidence in brief/topics

## 4. Eval set

24 cases by type: {'answerable': 9, 'ambiguous': 3, 'multi_doc': 4, 'partial': 2, 'out_of_scope': 6}. Generated with `openai/gpt-oss-20b` (prompts in `prompts/evalgen_*.txt`). Candidates rejected by code validation: {'near_miss_maybe_answerable': 1}.

**Dataset recovery:** The unattended evaluation generation attempt produced 12 valid cases and failed coverage validation. The earlier automatically generated 24-case dataset from runs/test_spares was reused for staged recovery on the same source documents. The incomplete attempt is preserved in eval_incomplete.json and eval_partial.json. This is not a successful uninterrupted end-to-end run.

## 5. Judge reliability

- Mode: deterministic only. Judge model: `llama3.2:3b`; agent model: `qwen2.5:7b`.
- Canary accuracy 0.67 over 12 known answers {'gold': 0.0, 'wrong': 1.0, 'padded_wrong': 1.0, 'fabricated': 0.67}.
- WARNING: judge failed canary tests; LLM verdicts disabled, deterministic checks only

## 6. Iterations

| Version | Overall | Dev | Held-out | Scored by |
|---|---|---|---|---|
| v0 | 11/24 | 9/18 | 2/6 | agent `ollama:qwen2.5:7b` / judge `deterministic` |
| v1 | 12/24 | 11/18 | 1/6 | agent `ollama:qwen2.5:7b` / judge `deterministic` |

### Iteration 1: accepted

Dev failures triaged: ans-04 (eval_defect_suspect, not patched), ans-09 (eval_defect_suspect, not patched), amb-01 (incomplete_answer), amb-03 (eval_defect_suspect, not patched), mul-01 (incomplete_answer), mul-02 (incomplete_answer), mul-03 (incomplete_answer), mul-04 (incomplete_answer), par-01 (partial_gap)
- Patch [incomplete_answer] `prompt_sections.completeness_policy` (critic): "Ensure all relevant information from the retrieved passages is included in the answer, and if any information is missing, provide a clear indication of what is missing." Fixed none; still failing ['amb-01', 'mul-01', 'mul-02', 'mul-03', 'mul-04'].
- Patch [partial_gap] `prompt_sections.partial_rules` (critic): "- Always specify which part of the question the documents do not cover when an answer is incomplete." Fixed none; still failing ['par-01'].
- Dev 9 -> 11; held-out 2 -> 1; regressions: ['ans-03'].

### Iteration 2: REJECTED and reverted: dev pass count did not improve

Dev failures triaged: ans-09 (eval_defect_suspect, not patched), amb-01 (incomplete_answer), mul-01 (incomplete_answer), mul-02 (incomplete_answer), mul-03 (incomplete_answer), mul-04 (incomplete_answer), par-01 (partial_gap)
- Patch [incomplete_answer] `prompt_sections.completeness_policy` (critic): "Ensure all relevant information from the retrieved passages is included in the answer, and if any information is missing, provide a clear indication of what is missing." Fixed none; still failing ['amb-01', 'mul-01', 'mul-02', 'mul-03', 'mul-04'].
- Patch [partial_gap] `prompt_sections.partial_rules` (critic): "- Clearly indicate which parts of the question the documents do not cover in your answer." Fixed none; still failing ['par-01'].
- Dev 11 -> 10; held-out 1 -> 2; regressions: ['ans-02'].

## 7. What still fails, and why

| Case | Split | Type | Triage | Genuine agent defect? | Note |
|---|---|---|---|---|---|
| ans-01 | held_out | answerable | incomplete_answer | yes | fact recall 0.0 |
| ans-03 | held_out | answerable | eval_defect_suspect | no (suspected test defect) | deterministic matching only; a paraphrase may have been missed |
| ans-08 | held_out | answerable | eval_defect_suspect | no (suspected test defect) | deterministic matching only; a paraphrase may have been missed |
| ans-09 | dev | answerable | eval_defect_suspect | no (suspected test defect) | deterministic matching only; a paraphrase may have been missed |
| amb-01 | dev | ambiguous | incomplete_answer | yes | fact recall 0.33 |
| amb-02 | held_out | ambiguous | eval_defect_suspect | no (suspected test defect) | deterministic matching only; a paraphrase may have been missed |
| mul-01 | dev | multi_doc | incomplete_answer | yes | fact recall 0.0 |
| mul-02 | dev | multi_doc | incomplete_answer | yes | fact recall 0.0 |
| mul-03 | dev | multi_doc | incomplete_answer | yes | fact recall 0.0 |
| mul-04 | dev | multi_doc | incomplete_answer | yes | fact recall 0.0 |
| par-01 | dev | partial | partial_gap | yes | missing_item_not_flagged, facts_missing_from_answer |
| par-02 | held_out | partial | incomplete_answer | yes | fact recall 0.0 |

Questions and answers for the failures above:

- **ans-01** Q: How much does the Fuel injector nozzle cost?
  A: The Fuel injector nozzle costs 420.00 USD.
- **ans-03** Q: What is the unit price in USD and the stock quantity for the Fuel injector nozzle?
  A: Unit price: $420.00; Stock quantity: 12
- **ans-08** Q: How much discount does MarineParts Ltd give on orders over $5,000?
  A: MarineParts Ltd gives a 5 percent discount on orders over $5,000.
- **ans-09** Q: What is the stock of the fuel injector nozzle?
  A: The stock of the fuel injector nozzle (part_number: SP-1001) is 12.
- **amb-01** Q: How much does the part with 25 in stock cost?
  A: The part with 25 in stock is the Centrifugal pump seal kit (part_number: SP-2001) and it costs 128.00 USD.
- **amb-02** Q: If I place a big order from MarineParts, how fast will it arrive, will I get a discount, and can I return it unopened?
  A: Your order from MarineParts Ltd will be delivered within 14 days. Since the order is big (above 5000 USD), you will receive a 5 percent discount. Returns are accepted within 30 days if the items are unopened.
- **mul-01** Q: If a customer orders all 12 units of part SP-1001 (Fuel injector nozzle) from MarineParts Ltd, will they qualify for the 5 percent discount, and what is the to...
  A: The customer will qualify for the 5 percent discount. The total cost after discount is 5,040.00 USD.
- **mul-02** Q: How much would a customer pay for 15 units of part SP-1001 from MarineParts Ltd after the discount is applied?
  A: The customer would pay $5,880.00 for 15 units of part SP-1001 from MarineParts Ltd after the discount is applied.
- **mul-03** Q: If a customer orders 15 units of part SP-1001 from MarineParts Ltd, what is the total cost before discount, and will the order qualify for the 5% discount?
  A: The total cost before discount is $6300.00. The order does not qualify for the 5% discount as the order amount is less than 5000 USD.
- **mul-04** Q: If a customer orders 15 units of part SP-1001 from MarineParts Ltd, what is the total cost after discount and how long will delivery take?
  A: The total cost for 15 units of part SP-1001 from MarineParts Ltd is $6,300.00 after a 5% discount. Delivery will take 14 days. Total cost calculation: 15 units * $420.00/unit = $6,300.00 Discount: $6,300.00 * 5% = $315.00 Discounted total:...
- **par-01** Q: What are the unit prices of parts SP-1001 and SP-1002, and what discount rule applies to MarineParts Ltd orders above $5000? Also, what discount rule applies t...
  A: Unit prices: SP-1001: $420.00, SP-1002: $35.50. Discount rule for MarineParts Ltd: 5 percent discount for orders above $5000. Discount rule for OceanFlow Supplies: No specific discount mentioned in the documents.
- **par-02** Q: What discount does MarineParts Ltd offer for orders over $5,000, and does the document specify discount policies for OceanFlow Supplies?
  A: MarineParts Ltd offers a 5 percent discount for orders over $5,000. OceanFlow Supplies' discount policies are not specified in the provided documents.

## 8. Limitations

- The eval set was written by an LLM from the same documents, and only 24 cases were kept after validation. Differences of one or two cases are within noise.
- The held-out set has only 6 cases, so held-out numbers are indicative, not statistically meaningful.
- Model sampling is not fully deterministic: re-running the same agent can change a borderline case's verdict.
- The LLM judge failed its calibration canaries, so scoring used deterministic checks only. Paraphrased correct answers may have been marked wrong.
- No LLM judge was used: paraphrased correct answers may be marked wrong.
- 4 remaining failure(s) are suspected defects of the test itself. They were NOT removed from the score.
- At least one iteration was rejected because it did not improve the dev score; the agent was reverted.
- Part of the brief could not be grounded in the documents; the agent is deliberately configured to refuse those requests.
- Retrieval is lexical (BM25) unless embeddings are enabled, so synonyms the documents never use can be missed.
- The optimizer can only add rules and tune retrieval parameters. It does not fix missing or contradictory source documents.

## 9. API usage and backend

Model calls actually sent (cache hits cost nothing). Tokens are the provider's own counts when it returned them, otherwise estimates.

| Stage | Calls | Cache hits | Prompt tokens | Completion tokens | Waited (s) |
|---|---|---|---|---|---|
| run | 0 | 0 | 0 | 0 | 0 |
| intake | 9 | 0 | 4249 | 1005 | 0 |
| generate | 2 | 0 | 558 | 254 | 0 |
| evalgen | 39 | 0 | 21550 | 6249 | 0 |
| optimize | 87 | 1 | 87006 | 4889 | 0 |
| **total** | 137 | 1 | 113363 | 12397 | 0 |

Per-minute pacing used limits of 30 requests and 8000 tokens per model (x0.85 safety margin); daily cap 200000 tokens. If a run was resumed, calls from every attempt are added together.

No quota exhaustion or backend switch occurred.
