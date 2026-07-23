from dotenv import load_dotenv          # Import function to read variables from a .env file
from langchain_groq import ChatGroq     # Import Groq's chat model wrapper for LangChain
from langgraph.graph import StateGraph, END  # StateGraph builds the workflow; END marks a terminal node
from typing import TypedDict            # TypedDict lets us define a typed dict schema for the graph's state
from langchain_core.messages import SystemMessage, HumanMessage
# SystemMessage sets the model's role/behavior instructions; HumanMessage wraps the actual user input —
# using these instead of an f-string gives a proper structured chat format instead of one flat prompt

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
    messages = [
    SystemMessage(content="""You are a helpful customer support agent. 
    Answer customer questions clearly and concisely.

    Important: If you don't have specific company information such as contact 
    details, pricing, or account-specific data, say so honestly rather than 
    making up placeholder information. Direct the customer to find that 
    information on the company's official website or documentation."""),
    # System instructions now explicitly guard against hallucinating specifics (contact info, pricing,
    # account data) the model has no real access to — pushes it to defer to official sources instead
        HumanMessage(content=state["question"])
        # The actual customer question, passed as a separate message rather than embedded in the system prompt
    ]
    response = llm.invoke(messages)  # Call the LLM with the structured message list instead of a single string
    return {"question": state["question"], "answer": response.content}  # return updated state

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
result = app.invoke({"question": "How do I reset my password?", "answer": ""})
# Execute the graph starting from this initial state; runs receive -> answer -> output in sequence
print("\nFinal state:", result)  # Print the final state dict after the graph finishes running