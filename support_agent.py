from dotenv import load_dotenv          # Import function to read variables from a .env file
from langchain_groq import ChatGroq     # Groq's chat model wrapper for LangChain
from langgraph.graph import StateGraph, END  # StateGraph builds the workflow; END marks a terminal node
from langchain_core.messages import SystemMessage, HumanMessage
# SystemMessage sets persistent behavioral instructions; HumanMessage wraps the actual per-request input
from typing import TypedDict            # TypedDict lets us define a typed dict schema for the graph's state

load_dotenv()  # Load environment variables (e.g. GROQ_API_KEY) before anything below needs them

# Define the state - what gets passed between nodes
class State(TypedDict):
    question: str   # The user's incoming question
    answer: str      # The LLM's answer, filled in later

# Initialize the LLM once at module load, not inside generate_answer — avoids creating a new
# client on every single node call. load_dotenv() above already ran, so GROQ_API_KEY is available
# by the time this line executes, regardless of which script imports this module first.
llm = ChatGroq(model="llama-3.1-8b-instant")

# Node 1: receive the question
def receive_question(state: State) -> State:
    return state  # No-op passthrough — kept as an explicit node for clarity/extensibility (e.g. logging, validation)

# Node 2: generate an answer
def generate_answer(state: State) -> State:
    messages = [
        SystemMessage(content="""You are a helpful customer support agent. 
Answer customer questions clearly and concisely.

Important: If you don't have specific company information such as contact 
details, pricing, or account-specific data, say so honestly rather than 
making up placeholder information. Direct the customer to find that 
information on the company's official website or documentation."""),
        # Guardrail instruction: tells the model to admit uncertainty rather than fabricate
        # contact info/pricing/account details
        HumanMessage(content=state["question"])
        # The actual customer question, kept separate from the system instructions
    ]
    response = llm.invoke(messages)  # Call the shared LLM client with the structured message list
    return {"question": state["question"], "answer": response.content}
    # Return a new state dict with the generated answer added

# Node 3: output the answer
def output_answer(state: State) -> State:
    return state  # No-op passthrough — graph is about to end anyway

# Build the graph
def build_graph():
    """Construct and compile the support-agent graph.

    Defined as a function (rather than module-level globals) so both agent.py and evaluate.py
    can call build_graph() and always get an identically-configured graph — this is what prevents
    the agent code and eval code from silently drifting apart, as happened in earlier versions
    where generate_answer was copy-pasted into evaluate.py and one copy fell out of sync.
    """
    graph = StateGraph(State)                        # Create a new graph whose shared state follows the State schema
    graph.add_node("receive", receive_question)       # Register "receive" node, backed by receive_question()
    graph.add_node("answer", generate_answer)         # Register "answer" node, backed by generate_answer()
    graph.add_node("output", output_answer)           # Register "output" node, backed by output_answer()

    graph.set_entry_point("receive")        # Execution starts at the "receive" node
    graph.add_edge("receive", "answer")     # After "receive" finishes, go to "answer"
    graph.add_edge("answer", "output")      # After "answer" finishes, go to "output"
    graph.add_edge("output", END)           # After "output" finishes, terminate the graph

    return graph.compile()  # Compile the graph definition into a runnable object