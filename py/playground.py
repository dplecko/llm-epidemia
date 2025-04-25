import os
from openai import OpenAI

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

response = client.responses.create(
    model="o3",
    input="How do I check if a Python object is an instance of a class?",
    reasoning={"effort": "low"},
    tools=[{"type": "web_search_preview"}],
)

print(response.output_text)
