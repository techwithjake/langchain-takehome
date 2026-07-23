from dotenv import load_dotenv                  # Import function to read variables from a .env file
from langsmith import evaluate                  # LangSmith's evaluate() harness for running experiments against a dataset
from langchain_groq import ChatGroq             # Groq's chat model wrapper for LangChain
from support_agent import build_graph, State    # Shared graph builder + state schema — guarantees this eval
                                                 # runs the exact same agent as agent.py, no copy-paste drift

load_dotenv()  # Load environment variables (e.g. GROQ_API_KEY, LANGCHAIN_API_KEY) from a .env file

app = build_graph()  # Single shared graph instance, built once and reused across all dataset examples

# Target function - this is what gets run against each dataset example
def run_agent(inputs: dict) -> dict:
    result = app.invoke({"question": inputs["question"], "answer": ""})
    # Run the full graph for one dataset example, using the "question" field from that example as input
    return {"answer": result["answer"]}
    # Return only the "answer" key — this becomes the "outputs" dict for the evaluators below

# LLM-as-a-judge evaluator
judge = ChatGroq(model="llama-3.3-70b-versatile")
# A separate, stronger model acts as the grader here (70B vs. the 8B agent), rather than reusing the
# agent's own model — reduces the risk of the judge sharing the same blind spots/biases as what it's grading
# Reused by both evaluators below

def helpfulness_evaluator(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    # LangSmith calls this once per example, passing: the original inputs, the agent's outputs
    # (from run_agent), and the reference_outputs (the "expected" field from the dataset)
    prompt = f"""You are evaluating a customer support response.

Question: {inputs["question"]}
Expected theme: {reference_outputs["expected"]}
Actual answer: {outputs["answer"]}

Rate the answer on helpfulness from 0 to 1:
- 1.0: Fully helpful, directly addresses the question with clear actionable steps
- 0.75: Mostly helpful, addresses the main question but misses some aspects
- 0.5: Partially helpful, addresses some aspects but leaves key questions unanswered
- 0.25: Minimally helpful, tangentially related but doesn't address the question
- 0.0: Not helpful, misses the point entirely

Respond with ONLY a number between 0 and 1. Nothing else."""
    # 5-band rubric (vs. the earlier 3-band 0/0.5/1.0 version) — gives the judge room for intermediate
    # scores, which surfaces gradual improvement/regression that a sparser rubric would flatten out

    response = judge.invoke(prompt)  # Ask the judge LLM to grade this example

    try:
        # Strip common prefixes like "Score: 0.8" or "0.8/1"
        raw = response.content.strip()
        cleaned = raw.split(":")[-1].split("/")[0].strip()
        # split(":")[-1] discards any "Score:" style prefix, keeping only what's after the last colon
        # split("/")[0] then discards a trailing "/1" style suffix, keeping only the numerator
        score = float(cleaned)
        score = max(0.0, min(1.0, score))  # Clamp into valid [0, 1] range in case the judge drifts outside it
    except ValueError:
        print(f"Warning: Could not parse judge score from response: '{response.content}' — defaulting to 0.5")
        score = 0.5
        # Fallback: if the judge's response still doesn't reduce to a clean float after cleanup
        # (e.g. prose with no colon/slash separator), log it so you can see how often this happens,
        # and default to a neutral 0.5 rather than letting the whole evaluation run crash

    return {"key": "helpfulness", "score": score}
    # LangSmith expects evaluators to return a dict with "key" (metric name) and "score" (the value),
    # which is how it knows to log this as a "helpfulness" score in the experiment results

def factuality_evaluator(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    # Second evaluator: same signature, grades a different dimension — whether the answer
    # fabricates specifics rather than admitting it doesn't know
    prompt = f"""You are evaluating whether a customer support response contains 
fabricated or placeholder information.

Question: {inputs["question"]}
Response: {outputs["answer"]}

Check if the response contains any of the following:
- Placeholder email addresses (e.g. support@yourcompany.com, support@[domain].com)
- Placeholder phone numbers (e.g. 1-800-SUPPORT, 1-800-[number])
- Fabricated URLs or website addresses
- Made-up pricing, dates, or account-specific details
- Any text in brackets suggesting placeholder content (e.g. [company name])

Score from 0 to 1:
- 1.0: No fabricated information, all specific claims are accurate or appropriately deferred
- 0.75: Minor uncertainty but no clear fabrication
- 0.5: Contains some placeholder or uncertain content that could mislead
- 0.25: Contains likely fabricated specifics
- 0.0: Contains clearly fabricated specifics presented as real information

Respond with ONLY a number between 0 and 1. Nothing else."""
    # Concrete checklist of hallucination patterns (placeholder emails/phones/URLs, made-up pricing,
    # bracketed placeholders), now also on a 5-band scale — this is the metric that directly tests
    # the guardrail's intended effect

    response = judge.invoke(prompt)  # Reuses the same judge instance as helpfulness_evaluator

    try:
        raw = response.content.strip()
        cleaned = raw.split(":")[-1].split("/")[0].strip()
        score = float(cleaned)
        score = max(0.0, min(1.0, score))
    except ValueError:
        print(f"Warning: Could not parse factuality score from: '{response.content}' — defaulting to 0.5")
        score = 0.5
        # Same parsing/fallback pattern as helpfulness_evaluator

    return {"key": "factuality", "score": score}
    # Logged as a separate "factuality" metric in LangSmith, alongside "helpfulness"

def conciseness_evaluator(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    # Third evaluator: grades whether the response is appropriately concise
    # or overwhelming the customer with unnecessary length
    prompt = f"""You are evaluating whether a customer support response is appropriately concise.

Question: {inputs["question"]}
Response: {outputs["answer"]}

A good customer support response gets to the point quickly without unnecessary preamble,
repetition, or over-explanation. It should be as long as needed and no longer.

Score from 0 to 1:
- 1.0: Perfectly concise — clear, direct, no unnecessary content
- 0.75: Mostly concise — minor unnecessary content but doesn't obscure the answer
- 0.5: Somewhat verbose — answer is present but buried in unnecessary content
- 0.25: Very verbose — excessive length makes the response hard to follow
- 0.0: Completely inappropriate length — either far too long or too short to be useful

Respond with ONLY a number between 0 and 1. Nothing else."""

    response = judge.invoke(prompt)

    try:
        raw = response.content.strip()
        cleaned = raw.split(":")[-1].split("/")[0].strip()
        score = float(cleaned)
        score = max(0.0, min(1.0, score))
    except ValueError:
        print(f"Warning: Could not parse conciseness score from: '{response.content}' — defaulting to 0.5")
        score = 0.5

    return {"key": "conciseness", "score": score}

# Run the evaluation against the dataset
results = evaluate(
    run_agent,
    data="Customer Support QA",
    evaluators=[helpfulness_evaluator, factuality_evaluator, conciseness_evaluator],
    num_repetitions=3,
    # Run each example 3 times and average the scores — addresses the judge-score variance found
    # earlier, where identical runs on an unchanged agent produced meaningfully different scores.
    # A single run isn't reliable in isolation; repetitions give a more trustworthy signal.
    experiment_prefix="support-bot-v8-conciseness"
    # Label documents *why* this run differs from prior ones (shared module + repetitions + 5-band
    # rubric), not just a version bump — makes the experiment list in LangSmith self-explanatory later
)
# Fetches all examples from the dataset, runs run_agent on each (3x per example), scores each with
# both evaluators, and logs everything as a named "experiment" in LangSmith for viewing/comparison

print("\nEvaluation complete!")