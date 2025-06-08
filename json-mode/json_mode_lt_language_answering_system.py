import openai
import os
from pydantic import BaseModel

from dotenv import load_dotenv
from rich import print

load_dotenv()  # Load environment variables from .env file

token = os.getenv("MY_TOKEN")  # Replace with your actual token
endpoint = "https://models.github.ai/inference"
model = "openai/gpt-4.1-nano"

# complaint_letter = """Dear Customer Service,
# I am writing to express my dissatisfaction with the recent purchase I made from your store.
#  The product arrived damaged and did not match the description provided on your website. 
#  I expected a much higher quality based on the price I paid. I would like a full refund and an apology for the inconvenience caused.
# Thank you for your attention to this matter.
# Sincerely,"""

# love_letter = """ I love you more than words can express."""

user_question = input("Enter your question in Lithuanian: ")

# Initialize the OpenAI client
client = openai.OpenAI(
    api_key=token,
    base_url=endpoint)

class LithuanianQuestionCheck(BaseModel):
    """
    # LithuanianQuestionCheck is a data model representing the analysis of a question 
    # if it is in Lithuanian language or not.

    Attributes:
        is_lithuanian (bool): Indicates whether the question is identified as being in Lithuanian.
        is_lithuanian_reason (str): Provides the reason or explanation for classifying the question as being in Lithuanian or not.

        This model is intended for use in AI-powered document analysis systems to help classify and explain the nature of correspondence.
        Also mention why it is not a complaint letter.
        confidence_score (float): A confidence score indicating the model's certainty about this classification 
        if it is a complaint letter or not.
        The score is up to 1 where 1 is the highest confidence.
    """
    is_lithuanian_language: bool
    is_lithuanian_reason: str
    confidence_score: float

response = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": """You are a helpful assistant that recognises Lithuanian language questions. 
             Please let us know if the question is in Lithuanian or not? Please response with True or False."""},
            {"role": "user", "content": user_question}
        ],
        temperature=0.7,
        response_format=LithuanianQuestionCheck
    )

letter_info = response.choices[0].message.parsed

if letter_info is not None and letter_info.is_lithuanian_language:
    print("[bold green]This is a Lithuanian language question.[/bold green]")
    print(f"Reason: {letter_info.is_lithuanian_reason} with confidence score {letter_info.confidence_score:.2f}")
else:
    print("[bold red]This is not a Lithuanian language question.[/bold red]")
    print(f"Reason: {letter_info.is_lithuanian_reason} with confidence score {letter_info.confidence_score:.2f}")