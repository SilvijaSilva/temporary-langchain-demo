import openai
import os

# Replace with your actual OpenAI API key
# openai.api_key = "YOUR_OPENAI_API_KEY"
from dotenv import load_dotenv

load_dotenv()  

token = os.getenv("MY_TOKEN")
endpoint = "https://models.github.ai/inference"
model = "openai/gpt-4.1-nano"

client = ChatOpenAI(
    base_url=endpoint,
    api_key=token,
    model=model,
)

response = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Once upon a time,"
        }
    ]
)

print(response.choices[0].message.content.strip())
