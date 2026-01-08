import os
from dotenv import load_dotenv
from google.cloud import vision

load_dotenv(r'C:\My Projects\Health-Navigator\credentials.env')

client = vision.ImageAnnotatorClient()

def extract_text(path):
    """Detects text in the file."""


    with open(path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations
    full_text = ''

    for text in texts:
        full_text += f'\n"{text.description}"\n'

        vertices = [
            f"({vertex.x},{vertex.y})" for vertex in text.bounding_poly.vertices
        ]

        full_text += f'"bounds: {",".join(vertices)}"'

    if response.error.message:
        raise Exception(
            "{}\nFor more info on error messages, check: "
            "https://cloud.google.com/apis/design/errors".format(response.error.message)
        )
    return full_text

