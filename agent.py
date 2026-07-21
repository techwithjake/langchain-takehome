from dotenv import load_dotenv          # Import function to read variables from a .env file
from langchain_groq import ChatGroq     # Import Groq's chat model wrapper for LangChain
from langgraph.graph import StateGraph, END  # StateGraph builds the workflow; END marks a terminal node
from typing import TypedDict            # TypedDict lets us define a typed dict schema for the graph's state

load_dotenv()  # Load environment variables (e.g. GROQ_API_KEY) from a .env file into the process environment

# Define the state - what gets passed between nodes
class State(TypedDict):
    question: str   # The user's incoming question
    answer: str      # The LLM's answer, filled in later

# Initialize the LLM
llm = ChatGroq(model="llama-3.1-8b-instant")  # Create a client for Groq's hosted Llama 3.1 8B model

# Node 1: receive the question
def receive_question(state: State) -> State:
    print(f"Question received: {state['question']}")  # Log the incoming question to the console
    return state  # Pass the state through unchanged to the next node

# Node 2: generate an answer
def generate_answer(state: State) -> State:
    response = llm.invoke(f"You are a helpful customer support agent. Answer this question: {state['question']}")
    # Call the LLM with a prompt combining a role instruction and the user's question; returns a response object
    return {"question": state["question"], "answer": response.content}
    # Build a new state dict: keep the original question, add the LLM's text output as "answer"

# Node 3: output the answer
def output_answer(state: State) -> State:
    print(f"Answer: {state['answer']}")  # Print the final answer to the console
    return state  # Pass the state through unchanged (graph is about to end anyway)

# Build the graph
graph = StateGraph(State)  # Create a new graph whose shared state follows the State schema
graph.add_node("receive", receive_question)  # Register "receive" node, backed by receive_question()
graph.add_node("answer", generate_answer)    # Register "answer" node, backed by generate_answer()
graph.add_node("output", output_answer)      # Register "output" node, backed by output_answer()

graph.set_entry_point("receive")        # Tell the graph execution starts at the "receive" node
graph.add_edge("receive", "answer")     # After "receive" finishes, go to "answer"
graph.add_edge("answer", "output")      # After "answer" finishes, go to "output"
graph.add_edge("output", END)           # After "output" finishes, terminate the graph

app = graph.compile()  # Compile the graph definition into a runnable object

# Run it
result = app.invoke({"question": "How do I contact support?", "answer": ""})
# Execute the graph starting from this initial state; runs receive -> answer -> output in sequence
print("\nFinal state:", result)  # Print the final state dict after the graph finishes running