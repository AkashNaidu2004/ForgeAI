\# ForgeAI — The Agent That Builds Agents



\## Project Overview



ForgeAI is an AI-powered agent factory that transforms natural language requirements and domain-specific documents into specialized AI assistants. It generates agent configurations, evaluates their performance and applies targeted improvements through a critic–optimizer loop.



\## Key Features



\- Automated requirement gathering and clarification.

\- Document ingestion and profiling.

\- Groundability verification to identify unsupported requirements.

\- Declarative agent generation.

\- Document retrieval and dynamic tool selection.

\- Automatic evaluation dataset generation.

\- Calibrated LLM judge with deterministic fallback.

\- Critic–optimizer loop with a maximum of three iterations.

\- Regression detection and automatic rollback.

\- Token budgeting, response caching and API fallback.

\- Interactive command-line chat interface.

\- Evaluation reports and optimization logs.



\## System Architecture



```text

User Brief + Documents

&#x20;         |

&#x20;         v

&#x20;   Intake Interviewer

&#x20;         |

&#x20;         v

&#x20;  Groundability Guard

&#x20;         |

&#x20;         v

&#x20;    Agent Generator

&#x20;         |

&#x20;         v

&#x20;  Evaluation Generator

&#x20;         |

&#x20;         v

&#x20;    Automated Judge

&#x20;         |

&#x20;         v

&#x20;   Critic–Optimizer

&#x20;         |

&#x20;         v

&#x20;Final Agent + QA Report

&#x20;         |

&#x20;         v

&#x20; Interactive Chat CLI

```



\## Technology Stack



| Component | Technology |

|---|---|

| Programming language | Python |

| Local model runtime | Ollama |

| Generator model | Qwen 2.5 7B |

| Judge model | Llama 3.2 3B |

| Remote inference | Groq |

| Document processing | PyPDF |

| Numerical processing | NumPy |

| Configuration | JSON, YAML and environment variables |

| Version control | Git and GitHub |



\## Installation



Clone the repository:



```bash

git clone https://github.com/AkashNaidu2004/ForgeAI.git

cd ForgeAI

pip install -r requirements.txt

```



Install Ollama and download the required models:



```bash

ollama pull qwen2.5:7b

ollama pull llama3.2:3b

```



Create a `.env` file using `.env.example` as a reference.



For local inference, configure:



```env

FORGE\_PROVIDER=ollama

FORGE\_BASE\_URL=http://localhost:11434

FORGE\_MODEL=qwen2.5:7b

FORGE\_JUDGE\_MODEL=llama3.2:3b

```



Remote inference can also be configured using a compatible API endpoint and an API key. Never commit private credentials.



\## Running ForgeAI



Generate and evaluate an agent using a brief and document folder:



```bash

python -m forge.cli run --brief "Build an HR policy assistant" --docs samples/hr\_policy --out runs/test\_new\_hr

```



The pipeline attempts to complete requirement gathering, agent generation, evaluation and optimization automatically. If evaluation coverage is insufficient, it stops and records the incomplete results.



\## Interactive Chat



Run the generated HR assistant:



```bash

python -m runner.chat runs/hr\_policy/agent\_final.json

```



Run the ship spares assistant:



```bash

python -m runner.chat runs/ship\_spares/agent\_final.json

```



\## Demonstration Results



| Metric | HR Policy Assistant | Ship Spares Assistant |

|---|---:|---:|

| Evaluation cases | 22 | 24 |

| Baseline passed | 19 | 11 |

| Final passed | 19 | 12 |

| Baseline score | 86.36% | 45.83% |

| Final score | 86.36% | 50.00% |

| Improvement | 0.00 pp | 4.17 pp |



\### Demonstration 1: HR Policy Assistant



The assistant answers questions about employee leave policies, sick leave, business travel and expense rules.



The optimization candidate did not improve the development score, so the original agent was retained.



\### Demonstration 2: Ship Spares Assistant



The assistant supports spare part lookup, stock availability, supplier information and order calculations.



One optimization candidate was accepted and a subsequent candidate was rejected and reverted.



The latest automatic evaluation generation attempt did not produce sufficient valid cases. A previously automatically generated 24-case dataset was reused, with recovery details recorded in the demonstration artifacts.



\## Testing



The project includes offline regression tests covering intake, groundability, agent generation, evaluation, optimization and API reliability.



Example commands:



```bash

python tests/test\_step2.py

python tests/test\_step3.py

python tests/test\_step4.py

python tests/test\_step5.py

python tests/test\_budget.py

```



\## Project Structure



```text

ForgeAI/

├── forge/

│   ├── intake.py

│   ├── guard.py

│   ├── generator.py

│   ├── agent.py

│   ├── retrieval.py

│   ├── tools.py

│   ├── evalgen.py

│   ├── evaluate.py

│   ├── judge.py

│   ├── optimizer.py

│   ├── factory.py

│   ├── budget.py

│   ├── report.py

│   └── cli.py

├── prompts/

├── runner/

├── samples/

├── tests/

├── runs/

│   ├── hr\_policy/

│   └── ship\_spares/

├── .env.example

├── .gitignore

├── requirements.txt

└── README.md

```



\## Known Limitations



\- Automatic evaluation generation can produce duplicate questions or insufficient valid cases.

\- An unseen library policy test generated only 13 valid evaluation cases and did not complete the full pipeline.

\- Explicit calculation requirements may be lost during intake.

\- Calculator usage is not consistently enforced.

\- Deterministic scoring can reject factually correct paraphrased answers.

\- The ship spares judge failed calibration, resulting in deterministic-only scoring.

\- Held-out performance declined during the accepted ship spares optimization.

\- Some unsupported questions can still produce hallucinated answers.

\- Fully reliable unattended execution has not yet been demonstrated.



These limitations are documented rather than hidden or removed from the official evaluation results.



\## Future Improvements



\- Evidence-driven evaluation generation with improved question diversity.

\- Reliable preservation of calculation requirements.

\- Mandatory calculator invocation for arithmetic tasks.

\- Stronger grounding and semantic answer verification.

\- Improved regression-aware optimization.

\- More reliable execution across unseen document domains.

\- Web-based interactive interface.



\## Assignment



Developed for the OceanAI Junior GenAI Engineer assessment.



\*\*Developer:\*\* Akash Naidu



\*\*Project:\*\* ForgeAI — The Agent That Builds Agents

