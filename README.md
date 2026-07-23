# Customer Support Bot — LangChain Take-Home

A LangGraph-based customer support agent built and evaluated with LangSmith, 
as part of the LangChain Senior Technical Support Engineer take-home assignment.

---

## What I Built

A customer support bot that answers common support questions using a three-node 
LangGraph workflow. The agent is intentionally scoped to be simple and auditable 
— the goal was to build something I could evaluate meaningfully, not something 
impressive on the surface.

The bot handles a range of realistic support scenarios: standard how-to questions, 
ambiguous inputs that need clarification, account-specific requests it can't fulfill, 
out-of-scope questions, compound questions, frustrated customers, and security concerns.

**How it works:**

The agent uses LangGraph's `StateGraph` to wire three nodes in sequence:

- `receive` — accepts the incoming question
- `answer` — passes it to an LLM (Groq/Llama 3.1 8b) with a customer support 
  system prompt including an explicit anti-hallucination guardrail
- `output` — returns the final response

State is passed between nodes as a typed `TypedDict` with two fields: 
`question` and `answer`. The graph is defined once in `support_agent.py` 
and imported by both `agent.py` and `evaluate.py` to prevent prompt drift 
between the agent and its evaluation.

---

## Evaluation

The agent was evaluated against an 18-question dataset covering 6 scenario categories:

- Standard support questions (password reset, login, cancellation, exports)
- Edge cases — account-specific data the bot cannot access
- Ambiguous inputs — questions requiring clarification
- Out-of-scope requests
- Compound questions
- Escalation/frustration and security concerns

**Three evaluators, all LLM-as-a-judge using Llama 3.3 70b:**

- **Helpfulness** — does the response address the question with actionable steps?
- **Factuality** — does it avoid fabricating placeholder information?
- **Conciseness** — is it appropriately direct without unnecessary verbosity?

Each example is run 3 times and scores are averaged to reduce judge variance.

**v8 results (final):**

| Metric | Avg Score |
|---|---|
| Helpfulness | TBD |
| Factuality | TBD |
| Conciseness | TBD |

**Experiment progression:**

| Version | Key change |
|---|---|
| v1 | Baseline — plain f-string prompt, single helpfulness evaluator |
| v2 | Improved system prompt (mislabeled — ran v1 prompt by mistake) |
| v3 | Dual evaluators: helpfulness + factuality |
| v4 | Granular 5-band rubric on both evaluators |
| v5 | Added num_repetitions=3 for averaged scores |
| v6 | Refactored to shared agent module |
| v7 | Expanded dataset from 6 to 18 questions |
| v8 | Added conciseness evaluator |

---

## Project Structure

support_agent.py # Shared LangGraph agent — imported by agent.py and evaluate.py
agent.py # Run the agent manually against a single question
evaluate.py # Run evaluation experiments against the full dataset
create_dataset.py # Create the 18-question dataset in LangSmith
friction_log # Observations from a new user's perspective

---

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file:

GROQ_API_KEY=your_key_here
LANGSMITH_API_KEY=your_key_here
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=your_project_name

---

## Run

```bash
# Create the dataset in LangSmith (first time only)
python3 create_dataset.py

# Run the agent against a single question
python3 agent.py

# Run the full evaluation experiment
python3 evaluate.py
```