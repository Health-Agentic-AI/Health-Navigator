from typing import TypedDict, Literal, Iterable
from functools import lru_cache
from dotenv import load_dotenv
import os
import re
from langchain_core.exceptions import OutputParserException
from app.workflow.llm_provider import create_langchain_chat_model

load_dotenv(os.path.join(os.getcwd(), 'credentials.env'))


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
    - Note (important): for the attachments to be valid they all must be valid if one is invalid then all of them are invalid (ATTACHMENT_NOT_VALID)
"""

system_prompt_text_only = f"""
You are a medical text validator. Classify the input into ONE label.


TEXT_VALID
- Text shows medical intent (symptoms, diagnosis, treatment, test results, health questions)

TEXT_NOT_VALID
- Text is non-medical (greetings, jokes, coding, sports, general chat)

Rules:
- Don't be too strict, classify it as not valid only if the input is very clearly not a midical input
- Output exactly one label: TEXT_VALID or TEXT_NOT_VALID

Note: you will recieve a title for the text, take it as a small hint
Note: the input could be an extracted text from an image using an ocr system or could be extracted from attachments (pdf, csv, etc...)
"""



class MedicalInputCheck(TypedDict):
    input_classification: Literal["TEXT_VALID_ATTACHMENT_VALID", "TEXT_VALID_ATTACHMENT_NOT_VALID", "TEXT_NOT_VALID_ATTACHMENT_VALID", "TEXT_NOT_VALID_ATTACHMENT_NOT_VALID"]

class TextOnlyInputValidator(TypedDict):
    input_classification: Literal["TEXT_VALID", "TEXT_NOT_VALID"]

FIRST_INPUT_LABELS = (
    "TEXT_VALID_ATTACHMENT_VALID",
    "TEXT_VALID_ATTACHMENT_NOT_VALID",
    "TEXT_NOT_VALID_ATTACHMENT_VALID",
    "TEXT_NOT_VALID_ATTACHMENT_NOT_VALID",
)

TEXT_ONLY_LABELS = (
    "TEXT_VALID",
    "TEXT_NOT_VALID",
)


@lru_cache
def get_structured_llm():
    return create_langchain_chat_model(
        google_model="gemini-2.5-flash-lite-preview-09-2025",
    ).with_structured_output(MedicalInputCheck)


@lru_cache
def get_structured_llm_text_only():
    return create_langchain_chat_model(
        google_model="gemini-2.5-flash-lite-preview-09-2025",
    ).with_structured_output(TextOnlyInputValidator)


@lru_cache
def get_raw_llm():
    return create_langchain_chat_model(
        google_model="gemini-2.5-flash-lite-preview-09-2025",
    )


def _extract_label(raw_output: str, allowed_labels: Iterable[str]) -> str | None:
    text = str(raw_output).strip().upper()
    for label in sorted(allowed_labels, key=len, reverse=True):
        if re.search(rf"\b{re.escape(label)}\b", text):
            return label
    return None


def validate_first_input(input_text: str, available_attachments=None):

    available_attachments = available_attachments if available_attachments else "The user did not provide any attachments."
    messages = [
        ("system", system_prompt),
        ("human", f"Text: {input_text}\nAttachments: {available_attachments}"),
    ]

    try:
        structured_llm = get_structured_llm()
        result = structured_llm.invoke(messages)
        return result["input_classification"]
    except (OutputParserException, KeyError, TypeError, ValueError):
        raw_llm = get_raw_llm()
        raw_response = raw_llm.invoke(messages)
        raw_content = getattr(raw_response, "content", raw_response)
        label = _extract_label(raw_content, FIRST_INPUT_LABELS)
        return label or "TEXT_NOT_VALID_ATTACHMENT_NOT_VALID"

def validate_input_text_only(title, input_text: str):

    messages = [
        ("system", system_prompt_text_only),
        ("human", f"Title: {title}, Input Text: {input_text}"),
    ]

    try:
        structured_llm_text_only = get_structured_llm_text_only()
        result = structured_llm_text_only.invoke(messages)
        return result["input_classification"]
    except (OutputParserException, KeyError, TypeError, ValueError):
        raw_llm = get_raw_llm()
        raw_response = raw_llm.invoke(messages)
        raw_content = getattr(raw_response, "content", raw_response)
        label = _extract_label(raw_content, TEXT_ONLY_LABELS)
        return label or "TEXT_NOT_VALID"
