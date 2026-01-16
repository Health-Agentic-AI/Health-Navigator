# Health-Navigator

Health-Navigator is an intelligent medical workflow system designed to process, validate, and route medical queries and images. It leverages **LangGraph** for orchestration, **Google Gemini** for advanced reasoning, and specialized **PyTorch** vision models to analyze user inputs comprehensively.

## Workflow

The system takes user input (text and optional attachments), validates the medical intent, and routes images to the appropriate specialized models for analysis. It employs a multi-agent architecture where agents collaborate to retrieve data, analyze images, and formulate medical assessments.

![Health-Navigator Workflow](Workflow%20Diagram/Workflow%20Diagram.drawio.png)

## Technology Stack

*   **Orchestration**: [LangGraph](https://langchain-ai.github.io/langgraph/) (StateGraph, MemorySaver)
*   **LLM**: Google Gemini (via `langchain-google-genai`)
*   **Backend**: Flask
*   **Database**:
    *   **PostgreSQL**: For structured patient data (medications, labs, vitals).
    *   **Hybrid Vector DB**: For unstructured documents (clinical notes, reports) using Semantic + BM25 search.
*   **Computer Vision**: PyTorch (ResNet18-based custom models).

## System Architecture & Agents

The workflow is composed of several specialized nodes and agents:

### 1. Input Processing & Validation
*   **Input Validation**: `first_input_validation_node` and `second_input_validation_node` ensure that both the initial prompt and any extracted text from files are valid and medically relevant.
*   **Intelligent Fallback**: The `input_not_valid_fallback_node` uses an LLM to generate specific, helpful error messages for the user if their input is rejected (e.g., "The file format is valid, but the content does not appear to be medical").
*   **Image Routing**: The `input_image_classification_node` distinguishes between Chest X-rays, Colon Tissue slides, and Text-heavy images (for OCR), ensuring they are sent to the correct model.

### 2. Intelligent Agents

#### 🧠 Numerical Models Agent
Analyzes structured patient data to predict health risks using neural networks.
*   **Heart Disease Prediction**: Analyzes 19 clinical features (including BMI, Age, Smoker status, Diabetes history, etc.) to output a binary prediction and probability score for heart disease.

#### 👁️ Vision Models Agent
Wrapper around specialized deep learning models for diagnostic image analysis.
*   **Chest X-Ray Classifier**: A multi-label classifier capable of detecting 14 thoracic conditions, including:
    *   Pneumonia, Pneumothorax, Effusion, Infiltration, Cardiomegaly, and others.
*   **Colon Tissue Classifier**: Classifies histopathology slides into 9 tissue types, such as:
    *   Normal Colon Mucosa, Adipose, Debris, and Colorectal Adenocarcinoma Epithelium.

#### 🗄️ Information Retriever Agent (RAG)
A specialized agent responsible for gathering patient context without hallucinating.
*   **Hybrid Search**: Retrieves information from both the SQL database and the Vector store.
*   **Human-in-the-Loop**: If the Medical Agent requests information that isn't in the database, this agent uses the `ask_user_for_info` tool to pause the workflow and request clarification directly from the user.

#### 🩺 Medical Agent
The central reasoning engine that acts as the "doctor" in the loop.
*   **Clinical Assessment**: Synthesizes the outputs from the Numerical Agent, Vision Agent, and Retriever Agent to form a cohesive medical analysis.
*   **Reflection Loop**: If the available information is insufficient for a safe assessment, it enters a reflection loop, instructing the Retriever Agent to find the missing data (either from the DB or the user).

### 3. Output Refinement
*   **Refiner Node**: The `output_refiner_node` takes the raw clinical output and reformats it to be user-friendly and empathetic, while strictly adhering to the original medical facts (no hallucinations or alterations of diagnoses).

## Project Structure

*   `app/workflow/workflow.py`: The main production workflow definition using LangGraph.
*   `app/workflow/agents/`: Contains the logic for the specific agents (`medical_agent`, `vision_models_agent`, etc.).
*   `app/workflow/ml_models/`: Stores the PyTorch model definitions and weights.
*   `app/workflow/helper_utils/`: Utilities for input validation, OCR, and file processing.
*   `Workflow Diagram/`: Contains the project's architectural diagrams.

## Disclaimer
Health-Navigator is an AI-assisted tool developed for experimentation and demonstration purposes. It is **not** a substitute for professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider for medical concerns.
