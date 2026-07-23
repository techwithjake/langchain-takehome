from support_agent import build_graph  # Import the shared graph builder — same graph evaluate.py uses

app = build_graph()  # Build and compile the graph (loads env vars, wires up nodes, etc. — see support_agent.py)

# Run it once, manually, with a single example question
result = app.invoke({"question": "How do I reset my password?", "answer": ""})
# Execute the graph starting from this initial state; runs receive -> answer -> output in sequence

print(f"Question received: {result['question']}")  # Echo the question back for visibility
print(f"Answer: {result['answer']}")                # Print the generated answer
print("\nFinal state:", result)                     # Print the full final state dict for debugging