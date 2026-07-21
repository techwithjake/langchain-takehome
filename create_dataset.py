from dotenv import load_dotenv          # Import function to read variables from a .env file
from langsmith import Client            # Import LangSmith's client for managing datasets, runs, and evaluations

load_dotenv()  # Load environment variables (e.g. LANGCHAIN_API_KEY) from a .env file into the process environment

# Initialize the LangSmith client
client = Client()  # Create a client instance, authenticated via the env vars loaded above

# Define our test dataset - questions and expected answer themes
examples = [
    {
        "question": "How do I reset my password?",
        "expected": "Should explain password reset steps including email link method"
    },
    {
        "question": "How do I cancel my subscription?",
        "expected": "Should explain cancellation process and any relevant policies"
    },
    {
        "question": "Why is my payment failing?",
        "expected": "Should suggest common payment failure reasons and troubleshooting steps"
    },
    {
        "question": "How do I export my data?",
        "expected": "Should explain data export options and process"
    },
    {
        "question": "I can't log into my account",
        "expected": "Should provide login troubleshooting steps"
    },
    {
        "question": "How do I contact support?",
        "expected": "Should provide support contact methods"
    },
]
# A list of dicts, each pairing a sample customer question with a description of what a good answer should cover.
# Note: these "expected" values are descriptions/criteria, not literal expected strings — useful for an LLM-graded eval later.

# Create the dataset in LangSmith
dataset = client.create_dataset(
    dataset_name="Customer Support QA",         # Name shown for this dataset in the LangSmith UI
    description="Test dataset for customer support bot evaluation"
)
# Creates an empty dataset "container" in LangSmith and returns an object representing it (with a generated .id)

# Add examples to the dataset
for example in examples:
    client.create_example(
        inputs={"question": example["question"]},   # The input the bot will receive during evaluation
        outputs={"expected": example["expected"]},   # The reference/expected output tied to that input
        dataset_id=dataset.id                         # Associates this example with the dataset created above
    )
    print(f"Added: {example['question']}")  # Log each example as it's uploaded, for visibility during the loop

print(f"\nDataset created successfully: {dataset.id}")  # Confirm completion and show the dataset's unique ID
print(f"View it at: https://smith.langchain.com/datasets/{dataset.id}")  # Print a direct link to view the dataset in the LangSmith web UI