from dotenv import load_dotenv          # Load environment variables from .env
from langsmith import Client            # LangSmith client for dataset management

load_dotenv()

client = Client()

DATASET_NAME = "Customer Support QA"

# Check if dataset already exists to avoid duplicates on re-runs
existing = [d for d in client.list_datasets() if d.name == DATASET_NAME]
if existing:
    print(f"Dataset '{DATASET_NAME}' already exists — skipping creation.")
    print(f"View at: https://smith.langchain.com/datasets/{existing[0].id}")
    exit(0)

# Full 18-question dataset covering a broad range of support scenarios
examples = [
    # Standard support questions
    {"question": "How do I reset my password?",
     "expected": "Should explain password reset steps including email link method"},
    {"question": "How do I cancel my subscription?",
     "expected": "Should explain cancellation process and any relevant policies"},
    {"question": "Why is my payment failing?",
     "expected": "Should suggest common payment failure reasons and troubleshooting steps"},
    {"question": "How do I export my data?",
     "expected": "Should explain data export options and process"},
    {"question": "I can't log into my account",
     "expected": "Should provide login troubleshooting steps"},
    {"question": "How do I contact support?",
     "expected": "Should provide support contact methods"},

    # Edge cases — account-specific data the bot cannot access
    {"question": "What is my current account balance?",
     "expected": "Should acknowledge it cannot access account-specific data and direct to official account portal"},
    {"question": "Can you process a refund for my last payment?",
     "expected": "Should explain it cannot process transactions and direct to support or billing team"},
    {"question": "What is my subscription renewal date?",
     "expected": "Should acknowledge it cannot access account data and suggest checking account settings"},

    # Ambiguous questions — should ask clarifying questions
    {"question": "It's not working.",
     "expected": "Should ask clarifying questions to understand what specifically isn't working"},
    {"question": "I need help.",
     "expected": "Should ask clarifying questions to understand what kind of help is needed"},

    # Out of scope — should redirect politely
    {"question": "What's the weather like today?",
     "expected": "Should politely explain this is outside its scope as a customer support agent"},
    {"question": "Can you write me a poem?",
     "expected": "Should politely decline and redirect to customer support topics"},

    # Compound questions — should address both parts
    {"question": "How do I reset my password and update my billing information?",
     "expected": "Should address both parts: password reset steps and how to update billing info"},
    {"question": "Can I cancel my subscription and get a refund?",
     "expected": "Should address cancellation process and explain it cannot process refunds directly"},

    # Escalation / frustration
    {"question": "I've been waiting 3 days for a response and nobody is helping me.",
     "expected": "Should acknowledge frustration empathetically and provide escalation options"},
    {"question": "This is the worst service I've ever experienced.",
     "expected": "Should respond empathetically, acknowledge the frustration, and offer to help resolve the issue"},

    # Security concern — should treat as urgent
    {"question": "I think someone hacked my account.",
     "expected": "Should treat as urgent, advise immediate password reset, enable 2FA, and contact support"},
]

dataset = client.create_dataset(
    dataset_name=DATASET_NAME,
    description="Customer support bot evaluation dataset — 18 questions across standard support, edge cases, ambiguous input, out-of-scope, compound questions, escalation, and security scenarios."
)

client.create_examples(
    inputs=[{"question": ex["question"]} for ex in examples],
    outputs=[{"expected": ex["expected"]} for ex in examples],
    dataset_id=dataset.id
)

print(f"Created dataset '{DATASET_NAME}' with {len(examples)} examples.")
print(f"View at: https://smith.langchain.com/datasets/{dataset.id}")
