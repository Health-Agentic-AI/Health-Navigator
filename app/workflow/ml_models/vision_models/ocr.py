import os
import io
from dotenv import load_dotenv
from google.cloud import vision
import fitz  # PyMuPDF

load_dotenv(os.path.join(os.getcwd(), 'credentials.env'))

def get_vision_client():
    return vision.ImageAnnotatorClient()

def extract_text(path):
    """Detects text in the file."""
    client = get_vision_client()
    
    # Handle PDF files
    if path.lower().endswith('.pdf'):
        pdf_document = fitz.open(path)
        page = pdf_document[0]  # First page
        pix = page.get_pixmap(dpi=300)  # Higher DPI for better OCR
        content = pix.tobytes("png")
        pdf_document.close()
    else:
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