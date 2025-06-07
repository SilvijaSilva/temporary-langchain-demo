import openai
import os

# Replace with your actual OpenAI API key
# openai.api_key = "YOUR_OPENAI_API_KEY"
from dotenv import load_dotenv
from rich import print

load_dotenv()  

token = os.getenv("MY_TOKEN")
endpoint = "https://models.github.ai/inference"
model = "openai/gpt-4.1-nano"

client = openai.OpenAI(
    api_key=token,
    base_url=endpoint)

response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What are three benefits of using Python for data analysis?"}
            ],
        temperature=0.7,
    )

print(response.choices[0].message.content)
