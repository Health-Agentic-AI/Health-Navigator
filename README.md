# Health-Navigator

Health-Navigator is an intelligent medical workflow system designed to process, validate, and route medical queries and images. It leverages **LangGraph** for orchestration, **Google Gemini** for advanced reasoning, and specialized **PyTorch** vision models to analyze user inputs comprehensively.

## The Core Idea

Health-Navigator emulates how a medical team collaborates on a complex case. Just as a doctor might consult with radiologists, request additional tests, review patient history, and iteratively refine their understanding—this system uses multiple specialized agents that work together through a **reflection loop** to arrive at a comprehensive medical assessment.

### Key Principles

| Principle | Description |
|-----------|-------------|
| **Specialization** | Each agent has a specific expertise (vision analysis, numerical risk scoring, information retrieval, clinical reasoning) |
| **Collaboration** | Agents don't work in isolation—they build on each other's outputs |
| **Reflection** | The Medical Agent can request more information, triggering retrieval cycles until satisfied |
| **Human-in-the-Loop** | When data is missing, the workflow pauses to ask the user—then resumes with enriched context |
| **Multi-Modal** | Processes text, images (X-rays, tissue slides), and structured patient data simultaneously |

## Workflow Overview

The system takes user input (text and optional attachments), validates the medical intent, and routes images to the appropriate specialized models for analysis. It employs a multi-agent architecture where agents collaborate to retrieve data, analyze images, and formulate medical assessments.

![Health-Navigator Workflow](Workflow%20Diagram/Workflow%20Diagram.drawio.png)

## Technology Stack

*   **Orchestration**: [LangGraph](https://langchain-ai.github.io/langgraph/) (StateGraph, MemorySaver) for managing state, cycles, and persistence.
*   **LLM**: Google Gemini (via `langchain-google-genai`) for reasoning, validation, and medical assessment.
*   **Backend**: Flask for the API and application server.
*   **Database**:
    *   **PostgreSQL**: Stores structured patient data (medications, appointments, labs, vitals).
    *   **Hybrid Vector DB**: Stores unstructured documents (clinical notes, reports, scans). It utilizes a **Hybrid Search Strategy** combining **Semantic Search** (embeddings) for conceptual matching and **BM25** for precise keyword matching.
*   **Computer Vision**: PyTorch (ResNet18-based custom models) for X-ray and tissue analysis.

## System Architecture & Agents

The workflow is composed of several specialized nodes and agents that interact dynamically:

### 1. Input Processing & Validation
*   **Input Validation**: `first_input_validation_node` and `second_input_validation_node` ensure that both the initial prompt and any extracted text from files are valid and medically relevant.
*   **Intelligent Fallback**: The `input_not_valid_fallback_node` uses an LLM to generate specific, helpful error messages for the user if their input is rejected (e.g., "The file format is valid, but the content does not appear to be medical").
*   **Image Routing**: The `input_image_classification_node` distinguishes between Chest X-rays, Colon Tissue slides, and Text-heavy images (for OCR), ensuring they are sent to the correct model.

---

### 2. The Reflection Loop: Medical Agent ↔ Information Retriever

This is the **core intelligence** of the system. The Medical Agent and Information Retriever Agent engage in an iterative dialogue, much like a physician consulting with a medical records specialist.

#### How It Works

```
┌─────────────────────────────────────────────────────────────────────┐
│                    REFLECTION LOOP (max N iterations)                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌─────────────────┐      INFORMATION REQUEST      ┌─────────────┐ │
│   │  Medical Agent  │ ─────────────────────────────▶│  Retriever  │ │
│   │                 │   "I need patient's blood    │    Agent    │ │
│   │  Analyzes case  │    pressure readings from     │             │ │
│   │  & identifies  │    the past 6 months"         │  Searches:  │ │
│   │  gaps in info  │◀─────────────────────────────│  • Vector DB│ │
│   └─────────────────┘      RETRIEVED DATA          │  • SQL DB   │ │
│            │                  + new context         │  • Ask User│ │
│            │                                       └─────────────┘ │
│            │                                                      │
│            ▼                                                      │
│   ┌─────────────────┐                                            │
│   │  Satisfied?     │    No ───────────────────────────────┐     │
│   └─────────────────┘                                       │     │
│            │ Yes                                            │     │
│            ▼                                                │     │
│   ┌─────────────────┐                                       │     │
│   │  Generate Final │◀──────────────────────────────────────┘     │
│   │  Assessment     │                                              │
│   └─────────────────┘                                              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

#### Step-by-Step Breakdown

| Step | Agent | Action | Example |
|------|-------|--------|---------|
| **1** | Medical Agent | Reviews all available data (vision results, numerical scores, retrieved records) and identifies what's missing | "I see the X-ray shows possible cardiomegaly, but I don't have the patient's baseline heart function or history of hypertension" |
| **2** | Medical Agent | Issues a specific information request | `needs_more_info(request="patient_hypertension_history_and_baseline_ekg")` |
| **3** | Information Retriever | Receives the request and searches multiple sources | • Queries Vector DB for clinical notes mentioning "hypertension"<br>• Queries SQL for blood pressure readings<br>• If still insufficient, calls `ask_user_for_info` |
| **4** | Information Retriever | Returns aggregated findings | "Found: Patient diagnosed with Stage 1 hypertension in 2021, currently on Lisinopril. Last BP reading was 145/95." |
| **5** | Medical Agent | Incorporates new data and re-evaluates | "With hypertension history, the cardiomegaly finding is more concerning. I should also check for medications affecting cardiac function..." |
| **6** | Loop | Steps 1-5 repeat until Medical Agent has sufficient context OR maximum iterations reached | — |
| **7** | Medical Agent | Produces final comprehensive assessment | — |

#### Why This Matters

- **Avoids Premature Conclusions**: The agent doesn't make diagnoses with partial information
- **Mimics Real Clinical Workflow**: Doctors routinely request additional tests and records
- **Adaptive Information Gathering**: Each request is informed by previous findings—not a static checklist
- **Safety Valve**: The `ask_user_for_info` mechanism ensures the system acknowledges when it truly doesn't know something

---

### 3. Intelligent Agents

#### 🧠 Numerical Models Agent
Analyzes structured patient data to predict health risks using neural networks.
*   **Heart Disease Prediction**: Uses 19 clinical features (BMI, Age, Smoker status, Diabetes history, etc.) → binary prediction + probability score
*   **Stroke Prediction**: Uses 17 symptom/risk-factor features (age, chest pain, high blood pressure, irregular heartbeat, etc.) → binary prediction + probability score
*   **Cancer Prediction**: Uses 8 demographic and lifestyle features (Age, BMI, Smoking, GeneticRisk, etc.) → binary prediction + probability score

#### 👁️ Vision Models Agent
Wrapper around specialized deep learning models for diagnostic image analysis.
*   **Chest X-Ray Classifier**: Multi-label classifier detecting 14 thoracic conditions (Pneumonia, Pneumothorax, Effusion, Infiltration, Cardiomegaly, etc.)
*   **Colon Tissue Classifier**: Classifies histopathology slides into 9 tissue types (Normal Colon Mucosa, Adipose, Colorectal Adenocarcinoma Epithelium, etc.)

#### 🗄️ Information Retriever Agent (RAG)
A specialized agent responsible for gathering patient context without hallucinating.
*   **Hybrid Retrieval Strategy**: Combines **Semantic Search** (embeddings) for conceptual matching with **BM25** for precise keyword matching
*   **Human-in-the-Loop (HITL)**: If critical information is missing from databases, uses `ask_user_for_info` tool to pause the workflow and request input from the user

#### 🩺 Medical Agent
The central reasoning engine that acts as the "doctor" in the loop.
*   **Clinical Assessment**: Synthesizes outputs from all other agents into a cohesive medical analysis
*   **Reflection Loop Driver**: Determines when more information is needed and drives the iterative retrieval process

### 4. Output Refinement
*   **Refiner Node**: The `output_refiner_node` takes the raw clinical output and reformats it to be user-friendly and empathetic, while strictly adhering to the original medical facts (no hallucinations or alterations of diagnoses).

## Project Structure

```
app/
├── workflow/
│   ├── workflow.py          # Main LangGraph workflow definition
│   ├── agents/              # Agent implementations
│   │   ├── medical_agent.py
│   │   ├── vision_models_agent.py
│   │   ├── numerical_models_agent.py
│   │   └── info_retriever_agent.py
│   ├── ml_models/           # PyTorch model definitions & weights
│   └── helper_utils/        # Input validation, OCR, file processing
├── models.py                # SQLAlchemy database models
├── routes.py                # Flask routes & endpoints
└── templates/               # Jinja2 templates
Workflow Diagram/            # Architectural diagrams
results/                     # Trained model artifacts (.pth, .pkl files)
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set Flask app
export FLASK_APP=app         # Linux/Mac
set FLASK_APP=app            # Windows

# Run the development server
flask run
# Or use: make dev
```

**Note**: Ensure model artifacts exist in `results/` directory:
- `results/stroke_model.pth`
- `results/cancer_model.pth` / `results/heart_model.pth`
- `results/scaler.pkl`, `results/feature_names.pkl`

## Disclaimer

Health-Navigator is an AI-assisted tool developed for experimentation and demonstration purposes. It is **not** a substitute for professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider for medical concerns.
