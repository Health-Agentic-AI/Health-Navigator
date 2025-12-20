from typing import TypedDict, Literal
from dotenv import load_dotenv
from google import genai
from google.genai import types
import json
import os

load_dotenv('C:\My Projects\Health-Navigator\credentials.env')

class MedicalInputCheck(TypedDict):
    image_type: Literal['chest_xray', 'colon_tissue', 'text', 'not_valid']

system_prompt = """
    You are a medical image classifier. Analyze the provided image and its title, then classify it into exactly one category.

    Classification Rules:
    - chest_xray: ONLY if the image is a chest X-ray (frontal or lateral view of thorax). Any other type of X-ray → not_valid
    - colon_tissue: ONLY if the image shows colon pathology tissue/histology slides
    - text: If the image primarily contains text (documents, screenshots, diagrams with text, etc.) - regardless of content relevance
    - not_valid: Everything else (other medical images, non-medical images, unclear images, other anatomical X-rays, etc.)

    Important:
    - When unsure, default to not_valid
    - Consider both image content and title, and take the title as a strong hint towards classification

    Output only one of these classes: 'chest_xray', 'colon_tissue', 'text', 'not_valid'
"""


client = genai.Client()

def validate_image(image_title, image_path):
    with open(image_path, 'rb') as f:
        image_bytes = f.read()


    response = client.models.generate_content(
        model='gemini-3-flash-preview',
        contents=[
            types.Part.from_bytes(
                data=image_bytes,
                mime_type='image/jpeg',
            ),
            f"Image Title: {image_title}"
        ],
        config=types.GenerateContentConfig(
            system_instruction=system_prompt
            response_mime_type="application/json",
            response_schema=MedicalInputCheck
        )
    )
    json_response = json.loads(response.text)
    return json_response['image_type']
