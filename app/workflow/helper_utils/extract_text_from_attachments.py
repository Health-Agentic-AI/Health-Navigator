import PyPDF2
import pandas as pd
import docx
import csv

def extract_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def extract_csv(file_path):
    df = pd.read_csv(file_path)
    return df.to_string()

def extract_excel(file_path):
    df = pd.read_excel(file_path)
    return df.to_string()

def extract_docx(file_path):
    doc = docx.Document(file_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

def extract_text_from_file(file_path):
    if file_path.endswith('.pdf'):
        return extract_pdf(file_path)
    elif file_path.endswith('.csv'):
        return extract_csv(file_path)
    elif file_path.endswith(('.xlsx', '.xls')):
        return extract_excel(file_path)
    elif file_path.endswith('.docx'):
        return extract_docx(file_path)
    elif file_path.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        raise ValueError(f"Unsupported file type: {file_path}")