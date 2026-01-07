# Health-Navigator

Health-Navigator is an intelligent medical workflow system designed to process, validate, and route medical queries and images. It leverages `langgraph` for orchestration and utilizes Large Language Models (LLMs) and specialized vision models to analyze user inputs.

## Workflow

The system takes user input (text and optional attachments), validates the medical intent, and routes images to the appropriate specialized models for analysis.

![Health-Navigator Workflow](<Workflow Diagram/Workflow Diagram.drawio.png>)

## Key Features

### 1. Intelligent Input Validation
The system uses an LLM-based agent (`agents_nodes/clear_valid_input_validator`) to validate user inputs. It classifies inputs into:
*   **Medical Intent**: Text showing symptoms, diagnosis, treatment, etc.
*   **Valid Attachments**: Files with medical-relevant names or contents.
*   **Invalid/Non-medical**: Filters out non-medical queries or unrelated files.

### 2. Image Routing & Classification
Images are analyzed by an Image Classification Agent (`vision_models/input_image_classification`) to determine their type before being sent to specific diagnostic models. It distinguishes between:
*   **Chest X-rays**: Forwarded to the X-ray analysis model.
*   **Colon Tissue**: Forwarded to the colon pathology classifier.
*   **Text Images**: Identified for potential OCR processing.
*   **Invalid Images**: Filtered out.

### 3. Specialized Vision Models
The project currently integrates two specialized deep learning models:

*   **Chest X-ray Classifier**:
    *   Based on ResNet18.
    *   Detects 14 conditions including Pneumonia, Cardiomegaly, Effusion, and Infiltration.
*   **Colon Tissue Classifier**:
    *   Based on ResNet18.
    *   Classifies tissue slides into 9 categories such as Normal Colon Mucosa, Adipose, Debris, and Colorectal Adenocarcinoma Epithelium.

## Project Structure

*   `workflow.ipynb`: The main notebook containing the current workflow logic and experimentation.
*   `agents_nodes/`: Contains the logic for input validation agents.
*   `vision_models/`: Contains the specialized image classification models and the routing classifier.
*   `Workflow Diagram/`: Contains the project's architectural diagrams.

## Note
This project is currently in the experimentation phase. The core logic and workflow definitions are primarily located in `workflow.ipynb`.
