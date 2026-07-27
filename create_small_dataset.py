from dotenv import load_dotenv          # Load environment variables from .env
from langsmith import Client            # LangSmith client for dataset management

load_dotenv()

client = Client()

DATASET_NAME = "Customer Support QA - Small"

# Check if dataset already exists to avoid duplicates on re-runs
existing = [d for d in client.list_datasets() if d.name == DATASET_NAME]
if existing:
    print(f"Dataset '{DATASET_NAME}' already exists — skipping creation.")
    print(f"View at: https://smith.langchain.com/datasets/{existing[0].id}")
    exit(0)

# 5 questions across 5 distinct scenario categories
# Chosen to demonstrate range without exceeding Playground TPM limits
examples = [
    {
        "question": "How do I reset my password?",
        "expected": "Should explain password reset steps including email link method"
    },
    {
        "question": "What is my current account balance?",
        "expected": "Should acknowledge it cannot access account-specific data and direct to official account portal"
    },
    {
        "question": "It's not working.",
        "expected": "Should ask clarifying questions to understand what specifically isn't working"
    },
    {
        "question": "I think someone hacked my account.",
        "expected": "Should treat as urgent, advise immediate password reset, enable 2FA, and contact support"
    },
    {
        "question": "Can you write me a poem?",
        "expected": "Should politely decline and redirect to customer support topics"
    },
]

dataset = client.create_dataset(
    dataset_name=DATASET_NAME,
    description="Reduced 5-question dataset for UI Playground evaluation — one example per scenario category, scoped to stay within free tier TPM limits."
)

client.create_examples(
    inputs=[{"question": ex["question"]} for ex in examples],
    outputs=[{"expected": ex["expected"]} for ex in examples],
    dataset_id=dataset.id
)

print(f"Created dataset '{DATASET_NAME}' with {len(examples)} examples.")
print(f"View at: https://smith.langchain.com/datasets/{dataset.id}")