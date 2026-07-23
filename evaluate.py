from dotenv import load_dotenv                  # Import function to read variables from a .env file
from langsmith import evaluate                  # Import LangSmith's evaluate() harness for running experiments against a dataset
from langchain_groq import ChatGroq             # Groq's chat model wrapper for LangChain
from langgraph.graph import StateGraph, END     # StateGraph builds the workflow; END marks a terminal node
from typing import TypedDict                    # For defining a typed dict schema for the graph's state

load_dotenv()  # Load environment variables (e.g. GROQ_API_KEY, LANGCHAIN_API_KEY) from a .env file

# Same agent setup as agent.py
class State(TypedDict):
    question: str   # The user's incoming question
    answer: str      # The LLM's answer, filled in later

llm = ChatGroq(model="llama-3.1-8b-instant")  # The agent's own LLM — this is what's being evaluated

def receive_question(state: State) -> State:
    return state  # No-op node: just passes state through (logging removed vs. agent.py, since this runs in an eval loop)

def generate_answer(state: State) -> State:
    response = llm.invoke(f"You are a helpful customer support agent. Answer this question: {state['question']}")
    return {"question": state["question"], "answer": response.content}
    # Calls the LLM and returns a new state dict with the generated answer added

def output_answer(state: State) -> State:
    return state  # No-op node: just passes state through unchanged

graph = StateGraph(State)                        # Create a new graph with the State schema
graph.add_node("receive", receive_question)      # Register each node with its backing function
graph.add_node("answer", generate_answer)
graph.add_node("output", output_answer)
graph.set_entry_point("receive")                 # Execution starts at "receive"
graph.add_edge("receive", "answer")              # receive -> answer
graph.add_edge("answer", "output")               # answer -> output
graph.add_edge("output", END)                    # output -> END (terminate)
app = graph.compile()                            # Compile into a runnable graph object

# Target function - this is what gets run against each dataset example
def run_agent(inputs: dict) -> dict:
    result = app.invoke({"question": inputs["question"], "answer": ""})
    # Run the full graph for one dataset example, using the "question" field from that example as input
    return {"answer": result["answer"]}
    # Return only the "answer" key — this becomes the "outputs" dict for the evaluator below

# LLM-as-a-judge evaluator
judge = ChatGroq(model="llama-3.3-70b-versatile")
# A separate, stronger model acts as the grader here (70B vs. the 8B agent), rather than reusing the
# agent's own model — reduces the risk of the judge sharing the same blind spots/biases as what it's grading

def helpfulness_evaluator(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    # LangSmith calls this once per example, passing: the original inputs, the agent's outputs
    # (from run_agent), and the reference_outputs (the "expected" field from the dataset)
    prompt = f"""You are evaluating a customer support response.

Question: {inputs["question"]}
Expected theme: {reference_outputs["expected"]}
Actual answer: {outputs["answer"]}

Rate the answer on helpfulness from 0 to 1:
- 1.0: Fully helpful, addresses the question completely
- 0.5: Partially helpful, addresses some aspects  
- 0.0: Not helpful, misses the point entirely

Respond with ONLY a number between 0 and 1. Nothing else."""
    # Prompt instructs the judge LLM to compare the actual answer against the expected theme
    # and output a bare numeric score, so the response can be parsed directly

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

# Run the evaluation against the dataset
results = evaluate(
    run_agent,
    data="Customer Support QA",
    evaluators=[helpfulness_evaluator],
    experiment_prefix="support-bot-v2-improved-prompt"
)
# Fetches all examples from the dataset, runs run_agent on each, scores each with helpfulness_evaluator,
# and logs everything as a named "experiment" in LangSmith for viewing/comparison

print("\nEvaluation complete!")