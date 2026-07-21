# LangChain Take-Home — Customer Support Bot

A simple LangGraph-based customer support agent evaluated with LangSmith.

## What I Built

A three-node LangGraph agent that answers customer support questions:
- **receive** — accepts the incoming question
- **answer** — passes it to an LLM (Groq/Llama 3.1) with a support agent system prompt
- **output** — returns the response

## Files

- `agent.py` — the LangGraph agent
- `create_dataset.py` — creates the evaluation dataset in LangSmith via SDK
- `evaluate.py` — runs the evaluation experiment via SDK using LLM-as-a-judge
- `friction_log` — observations from a new user's perspective

## Evaluation

Ran two evaluation experiments against a 6-question customer support dataset:

- **SDK experiment** (`support-bot-v1`): avg helpfulness score **0.75** using LLM-as-a-judge (Llama 3.3 70b as judge, Llama 3.1 8b as agent)
- **UI experiment**: same dataset run via LangSmith Playground with Groq

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install langchain-groq langgraph langsmith python-dotenv
```

Create a `.env` file:

```GROQ_API_KEY=your_key_here
LANGSMITH_API_KEY=your_key_here
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=your_project_name
```

## Run

```bash
python3 agent.py                # run the agent
python3 create_dataset.py       # create the dataset in LangSmith
python3 evaluate.py             # run the evaluation experiment
```