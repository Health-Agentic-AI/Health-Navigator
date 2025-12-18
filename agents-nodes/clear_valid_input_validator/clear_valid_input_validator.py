from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict, Literal
from dotenv import load_dotenv
import os

load_dotenv('../../credentials.env')


def system_prompt_builder(input_text, available_attachments):
    system_prompt = f"""
    You are a medical input classifier. Classify into ONE label.

    TEXT_VALID_ATTACHMENT_VALID
    - Text shows medical intent (symptoms, diagnosis, treatment, test results, health questions)
    - Attachments absent OR have medical-relevant names

    TEXT_VALID_ATTACHMENT_NOT_VALID
    - Text shows medical intent
    - Attachments present with clearly non-medical names (vacation.jpg, recipe.pdf)

    TEXT_NOT_VALID_ATTACHMENT_VALID
    - Text is non-medical (greetings, jokes, coding, sports, general chat)
    - Attachments have medical names (xray, report, scan, lab, test)
    - User likely uploaded wrong files or is confused

    TEXT_NOT_VALID_ATTACHMENT_NOT_VALID
    - Text is non-medical
    - Attachments absent OR clearly non-medical

    Rules:
    - "xray", "report", "scan", "lab", "test" in filenames = medical attachment
    - When text is invalid but attachments are medical, still classify as TEXT_NOT_VALID_ATTACHMENT_VALID

    user input: {input_text}
    available attachments: {available_attachments}
    """
    return system_prompt


class MedicalInputCheck(TypedDict):
    input_classification: Literal["TEXT_VALID_ATTACHMENT_VALID", "TEXT_VALID_ATTACHMENT_NOT_VALID", "TEXT_NOT_VALID_ATTACHMENT_VALID", "TEXT_NOT_VALID_ATTACHMENT_NOT_VALID"]


structured_llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    
    ).with_structured_output(MedicalInputCheck)


def validate_medical_input_agent(input_text: str, available_attachments=None):

    available_attachments = available_attachments if available_attachments else "The user did not provide any attachments."

    result = structured_llm.invoke([
        ("system", system_prompt_builder(input_text, available_attachments)),
        ("human", input_text)
    ])

    return result['input_classification']
