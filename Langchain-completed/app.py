# Create a LANGSMITH_API_KEY in Settings > API Keys
from langsmith import Client
client = Client(api_key="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")
prompt = client.pull_prompt("hwchase17/openai-functions-agent", include_model=True)
print(prompt)