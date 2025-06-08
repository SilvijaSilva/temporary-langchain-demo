import openai
import os
from pydantic import BaseModel

from dotenv import load_dotenv
from rich import print

load_dotenv()  # Load environment variables from .env file

token = os.getenv("MY_TOKEN")  # Replace with your actual token
endpoint = "https://models.github.ai/inference"
model = "openai/gpt-4.1-nano"

# Complaint letter recogniser

# Generate me a mock complaint letter
complaint_letter = """Dear Customer Service,
I am writing to express my dissatisfaction with the recent purchase I made from your store.
 The product arrived damaged and did not match the description provided on your website. 
 I expected a much higher quality based on the price I paid. I would like a full refund and an apology for the inconvenience caused.
Thank you for your attention to this matter.
Sincerely,"""

love_letter = """ I love you more than words can express."""

# Initialize the OpenAI client
client = openai.OpenAI(
    api_key=token,
    base_url=endpoint)

class LetterInformation(BaseModel):
    """
    LetterInformation is a data model representing the analysis of a letter's content.

    Attributes:
        is_complaint (bool): Indicates whether the letter is identified as a complaint.
        is_complaint_reason (str): Provides the reason or explanation for classifying the letter as a complaint or not.

        This model is intended for use in AI-powered document analysis systems to help classify and explain the nature of correspondence.
        Also mention why it is not a complaint letter.
        confidence_score (float): A confidence score indicating the model's certainty about this classification 
        if it is a complaint letter or not.
        The score is up to 1 where 1 is the highest confidence.
    """
    is_complaint: bool
    is_complaint_reason: str
    confidence_score: float

response = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": """You are a helpful assistant that recognises complaint letters. 
             Please let us know if the letter is a complaint or not? Please response with True or False."""},
            {"role": "user", "content": complaint_letter}
        ],
        temperature=0.7,
        response_format=LetterInformation
    )

letter_info = response.choices[0].message.parsed

if letter_info is not None and letter_info.is_complaint:
    print("[bold red]This is a complaint letter.[/bold red]")
    print(f"Reason: {letter_info.is_complaint_reason} with confidence score {letter_info.confidence_score:.2f}")
else:
    print("[bold green]This is not a complaint letter.[/bold green]")
    print(f"Reason: {letter_info.is_complaint_reason} with confidence score {letter_info.confidence_score:.2f}")