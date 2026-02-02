import sys
sys.path.append(r"C:\My Projects\Health-Navigator")
import os

from dotenv import load_dotenv
load_dotenv(os.path.join(os.getcwd(), 'credentials.env'))

os.environ["LANGCHAIN_TRACING_V2"]
os.environ["LANGCHAIN_ENDPOINT"]
os.environ["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_PROJECT"]

from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any
from typing import Literal, List, Dict, Any, Annotated
from typing_extensions import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.types import Command, interrupt
import operator
import logging

# Configure logging
logger = logging.getLogger(__name__)

from app.workflow.helper_utils.clear_valid_input_validator.input_validator import validate_first_input, validate_input_text_only
from app.workflow.helper_utils.extract_text_from_attachments import extract_text_from_file

from app.workflow.ml_models.vision_models.input_image_classification.image_classifier import classify_image
from app.workflow.ml_models.vision_models.ocr import extract_text

from app.workflow.agents.numerical_models_agent.agent import invoke_agent as invoke_numerical_models_agent
from app.workflow.agents.vision_models_agent.agent import invoke_agent as invoke_vision_models_agent

from app.workflow.agents.information_retriever_agent.agent import invoke_db_retriever_agent
from app.workflow.agents.medical_agent.agent import invoke_medical_agent



class AgentState(TypedDict):
    # Initial inputs
    input_prompt: str
    attachments: Dict[str, Any]
    user_id: str
    thread_id: str
    
    # Validation
    input_validation_result: str
    coming_from_validation: str
    coming_from_extracted_text: Annotated[List[str], operator.add]
    extracted_text_from_images_or_attachments_validation_results: Annotated[List[str], operator.add]
    
    # Image processing
    input_images_titles_and_paths: Dict[str, str]
    input_images_classification_results: List[List[str]]
    text_images: List[List[str]]
    medical_images: List[List[str]]
    extracted_text_from_images: List[List[str]]
    
    # File processing
    input_files_titles_and_paths: Dict[str, str]
    has_files: bool
    extracted_text_from_attachments: List[List[str]]
    
    # Agent outputs
    numerical_models_agent_output: str
    medical_vision_models_agent_output: str
    models_agents_aggregated_output: str
    
    # Database and medical agent
    db_query_results: str
    medical_agent_output: str
    medical_agent_needs_info: bool
    info_request: str
    
    # Reflection loop
    reflection_count: int
    max_reflections: int
    conversation_history: List[Dict[str, Any]]

    final_refined_medical_output: str

def first_input_validation_node(state: AgentState):
    logger.info("Entering Node: first_input_validation_node")
    input_prompt = state['input_prompt']
    attachments = list(state.get("attachments", {}).keys())

    first_input_validation_result = validate_first_input(input_prompt, attachments)
    state["input_validation_result"] = first_input_validation_result
    state["coming_from_validation"] = "first_input_text"

    logger.debug(f"Validation result: {first_input_validation_result}", extra={"input_prompt_length": len(input_prompt), "attachments_count": len(attachments)})
    
    # Determine routing
    if first_input_validation_result != "TEXT_VALID_ATTACHMENT_VALID":
        logger.debug("Routing to input_not_valid_fallback_node", extra={"validation_result": first_input_validation_result})
        return Command(
            update=state,
            goto=["input_not_valid_fallback_node"]
        )
    
    # Route based on input type
    text_provided = bool(input_prompt)
    attachments_files = {}
    attachments_images = {}
    
    for key, value in state.get('attachments', {}).items():
        if value.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
            attachments_images[key] = value
        else:
            attachments_files[key] = value
    
    state["input_images_titles_and_paths"] = attachments_images
    state["input_files_titles_and_paths"] = attachments_files
    state["has_files"] = bool(attachments_files)
    
    # Determine next nodes
    next_nodes = []
    if attachments_files:
        next_nodes.append("extract_text_from_files_node")
    if attachments_images:
        next_nodes.append("input_image_classification_node")
    if text_provided and not attachments_files and not attachments_images:
        next_nodes.append("numerical_models_agent_node")
    
    logger.debug(f"Exiting first_input_validation_node. Routing to: {next_nodes}")
    return Command(
        update={
            "input_validation_result": state["input_validation_result"],
            "input_images_titles_and_paths": state["input_images_titles_and_paths"],
            "input_files_titles_and_paths": state["input_files_titles_and_paths"],
            "has_files": state["has_files"]
        },
        goto=next_nodes
    )

def second_input_validation_node(state: AgentState):
    logger.info("Entering Node: second_input_validation_node")
    coming_from_validation = state["coming_from_extracted_text"]  # Now a list
    
    # Will contain ["images"] or ["attachments"] or ["images", "attachments"]
    full_input_text = []
    
    if "images" in coming_from_validation:
        full_input_text.extend(state["extracted_text_from_images"])
    if "attachments" in coming_from_validation:
        full_input_text.extend(state["extracted_text_from_attachments"])

    full_validation_results = []

    for input_text in full_input_text:
        title = input_text[0]
        extracted_text = input_text[1]

        one_validation_results = validate_input_text_only(title, extracted_text)
        full_validation_results.append(one_validation_results)
        
    state["extracted_text_from_images_or_attachments_validation_results"] = full_validation_results

    # Route using Command
    return second_input_validation_route(state)

def input_not_valid_fallback_node(state: AgentState):
    """
    Handles invalid input cases by using an LLM to generate a helpful and specific error message.
    """
    logger.info("Entering Node: input_not_valid_fallback_node")
    validation_results = state.get("input_validation_result", "UNKNOWN_ERROR")
    error_context = ""

    if state["coming_from_validation"] == "first_input_text":
        if validation_results == "TEXT_VALID_ATTACHMENT_NOT_VALID":
            error_context = "The text input is valid, but the attachment format is not supported or corrupt."
        elif validation_results == "TEXT_NOT_VALID_ATTACHMENT_VALID":
            error_context = "The attachment is valid, but the text input is missing or invalid."
        elif validation_results == "TEXT_NOT_VALID_ATTACHMENT_NOT_VALID":
            error_context = "Both the text input and the attachment are invalid."
    
    elif state["coming_from_validation"] == "second_input_text":
        error_context = "The text extracted from the provided file or image is invalid or empty."

    elif state["coming_from_validation"] == "input_image":
        error_context = "The provided image could not be processed or is not a valid medical image."

    # Initialize LLM for error generation
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-preview-09-2025",
        google_api_key=os.environ["GOOGLE_API_KEY"],
    )

    system_prompt = """You are a helpful assistant for a medical application.
    Your task is to explain to the user why their input was invalid based on the provided technical context.
    Be polite, clear, and offer a suggestion on how to fix it.
    Keep the message short and user-friendly."""

    messages = [
        ("system", system_prompt),
        ("human", f"Technical Context: {error_context}\n\nPlease generate a user-friendly error message.")
    ]

    response = llm.invoke(messages)
    error_message = response.content

    # Set the final output to the error message so the frontend can display it
    state["final_refined_medical_output"] = error_message

    logger.debug("Exiting input_not_valid_fallback_node")
    return state

def input_image_classification_node(state: AgentState):
    logger.info("Entering Node: input_image_classification_node")
    input_images_titles_and_paths = state["input_images_titles_and_paths"]

    results = []
    medical_images = []
    text_images = []

    for title, path in input_images_titles_and_paths.items():
        classification = classify_image(title, path)
        
        result = [title, path, classification]

        if classification == "text":
            text_images.append(result)
        elif classification == "not_valid_image":
            pass
        else:
            medical_images.append(result)

        results.append(result)

    if text_images:
        state["text_images"] = text_images
        
    if medical_images:
        state["medical_images"] = medical_images

    state["input_images_classification_results"] = results

    logger.debug(f"Processed {len(input_images_titles_and_paths)} images")
    logger.debug("Exiting input_image_classification_node")

    # Route using Command
    return input_image_classification_route(state)

def extract_text_from_images_node(state: AgentState):
    logger.info("Entering Node: extract_text_from_images_node")
    full_images_results = state["text_images"]
    full_extracted_text = []

    for image_result in full_images_results:
        
        # image_result[0] = title
        # image_result[1] = path
        # image_result[2] = classification
        title = image_result[0]
        path = image_result[1]

        extracted_text = extract_text(path)

        full_extracted_text.append([title, extracted_text])

    logger.debug(f"Extracted text from {len(full_extracted_text)} images")
    return {
        "extracted_text_from_images": full_extracted_text,
        "coming_from_extracted_text": ["images"]  # Now a list
    }

def extract_text_from_files_node(state: AgentState):
    logger.info("Entering Node: extract_text_from_files_node")
    full_files = state["input_files_titles_and_paths"]
    full_extracted_text = []

    for title, path in full_files.items():
        extracted_text = extract_text_from_file(path)

        full_extracted_text.append([title, extracted_text])

    logger.debug(f"Extracted text from {len(full_extracted_text)} files")
    return {
        "extracted_text_from_attachments": full_extracted_text,
        "coming_from_extracted_text": ["attachments"]  # Now a list
    }


def numerical_models_agent_node(state: AgentState):
    logger.info("Entering Node: numerical_models_agent_node")
    extracted_text = state.get("extracted_text_from_images_or_attachments_validation_results", "No extracted text")
    input_prompt = state['input_prompt'] if state['input_prompt'] else 'No input prompt'

    full_input = f"Input Prompt: {input_prompt}\n\n Extracted Text: {extracted_text}"


    result = invoke_numerical_models_agent(user_input=full_input, user_id=state["user_id"])

    logger.debug("Exiting numerical_models_agent_node")
    return {"numerical_models_agent_output": result}


def medical_vision_models_agent_node(state: AgentState):
    logger.info("Entering Node: medical_vision_models_agent_node")
    # very important note here: the paths of the images or anything in the workflow should contain \\ and not \
    images = state["medical_images"]
    
    results = invoke_vision_models_agent(str(images))

    logger.debug("Exiting medical_vision_models_agent_node")
    return {"medical_vision_models_agent_output": results}


def initialize_reflection_state(state: AgentState) -> AgentState:
    """Initialize reflection-related fields if not present."""
    if "reflection_count" not in state:
        state["reflection_count"] = 0
    if "max_reflections" not in state:
        state["max_reflections"] = 5
    if "conversation_history" not in state:
        state["conversation_history"] = []
    return state

def models_agents_output_aggregator_node(state: AgentState):
    """Enhanced version of your models_agents_output_aggregator_node"""
    logger.info("Entering Node: models_agents_output_aggregator_node")
    vision_agent_input = str(state.get("medical_images", "No medical images"))
    vision_agent_output = str(state.get("medical_vision_models_agent_output", "No vision agent output"))
    input_prompt = str(state.get('input_prompt', 'No input prompt'))
    numerical_agent_input = str(state.get("extracted_text_from_images_or_attachments_validation_results", "No extracted text"))
    numerical_agent_output = str(state.get("numerical_models_agent_output", "No numerical agent output"))

    full_aggregated_output = f"""
    Input Prompt: {input_prompt}
    
    Numerical Analysis:
    Input: {numerical_agent_input}
    Output: {numerical_agent_output}
    
    Vision Analysis:
    Input: {vision_agent_input}
    Output: {vision_agent_output}
    """
    
    state["models_agents_aggregated_output"] = full_aggregated_output
    
    # Initialize reflection state
    state = initialize_reflection_state(state)

    logger.debug("Exiting models_agents_output_aggregator_node")
    return state

def db_retriever_agent_node(state: AgentState):
    """
    Retrieves information from both vector and relational databases,
    and determines whether to query more, ask user, or provide results.
    """
    logger.info("Entering Node: db_retriever_agent_node")
    aggregated_output = state.get("models_agents_aggregated_output", "")
    info_request = state.get("info_request", "")
    reflection_count = state.get("reflection_count", 0)
    max_reflections = state.get("max_reflections", 5)
    user_id = state["user_id"]
    thread_id = state["thread_id"]

    logger.debug(f"Reflection count: {reflection_count}/{max_reflections}")

    result = invoke_db_retriever_agent(
        aggregated_output=aggregated_output,
        info_request=info_request,
        reflection_count=reflection_count,
        max_reflections=max_reflections,
        user_id=user_id,
        conversation_history=state["conversation_history"],
        checkpointer=checkpointer,
        thread_id=thread_id
    )
    
    # Update state
    state["conversation_history"] = result["conversation_history"]
    response_text = result["response"][0]['text']
    
    # ✅ CHECK IF AGENT NEEDS USER INPUT
    question = None
    for msg in reversed(result["conversation_history"]):
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            for tool_call in msg.tool_calls:
                if tool_call['name'] == 'ask_user_for_info':
                    question = tool_call['args']['request']
                    break
            if question:
                break

    if question:
        logger.info(f"Triggering interrupt with question")

        # Interrupt at the MAIN WORKFLOW level
        user_response = interrupt(question)

        logger.info("Received user response")
        
        # Add the user's response to conversation history
        state["conversation_history"].append({
            "role": "user", 
            "content": user_response
        })
        
        # Store the response for the medical agent
        state["db_query_results"] = f"USER PROVIDED INFORMATION:\n{user_response}"
        state["medical_agent_needs_info"] = False
    else:
        state["db_query_results"] = response_text
        state["medical_agent_needs_info"] = result["needs_more_info"]

    logger.debug(f"Exiting db_retriever_agent_node. needs_more_info: {state['medical_agent_needs_info']}")
    return state

def medical_agent_node(state: AgentState):
    """
    Analyzes patient information and provides medical assessment.
    Requests more information when critical data is missing.
    """
    logger.info("Entering Node: medical_agent_node")
    
    # Prepare context
    aggregated_output = state["models_agents_aggregated_output"]
    db_results = state.get("db_query_results", "No database results available")
    reflection_count = state.get("reflection_count", 0)
    max_reflections = state.get("max_reflections", 5)
    
    # Invoke the medical agent
    result = invoke_medical_agent(
        aggregated_output=aggregated_output,
        db_results=db_results,
        reflection_count=reflection_count,
        max_reflections=max_reflections,
        conversation_history=state["conversation_history"]
    )
    
    # Update state with results
    state["conversation_history"] = result["conversation_history"]
    
    # ✅ CHECK IF MEDICAL AGENT NEEDS MORE INFO
    # Trigger keywords that indicate need for more information:
    NEED_MORE_INFO_TRIGGERS = [
        "NEED_MORE_INFO",
        "INFORMATION NEEDED:",
        "NEED_MORE_INFORMATION",
        "REQUIRE ADDITIONAL",
        "DATA DISCREPANCY IDENTIFIED"
    ]
    
    response_text = result["response"][0]['text']
    needs_more_info = any(trigger in response_text for trigger in NEED_MORE_INFO_TRIGGERS)
    
    if needs_more_info and reflection_count < max_reflections:
        # Extract the specific information request
        info_request = extract_info_request(response_text)

        state["medical_agent_needs_info"] = True
        state["info_request"] = info_request
        state["reflection_count"] = reflection_count + 1

        logger.info(f"Medical Agent requested more info, reflection count: {state['reflection_count']}")
        
    elif reflection_count >= max_reflections and needs_more_info:
        # At max reflections but still needs info - force completion
        logger.info(f"Max reflections reached ({max_reflections}), forcing final response")
        
        # Optionally: re-invoke medical agent with instruction to provide best-possible assessment
        final_result = invoke_medical_agent(
            aggregated_output=aggregated_output,
            db_results=db_results + "\n\n[MAX REFLECTIONS REACHED - Provide best assessment with available data]",
            reflection_count=max_reflections,
            max_reflections=max_reflections,
            conversation_history=state["conversation_history"],
            force_completion=True  # Add this flag to your invoke function
        )
        
        state["medical_agent_needs_info"] = False
        state["medical_agent_output"] = final_result["response"][0]['text']
        state["conversation_history"] = final_result["conversation_history"]
        
    else:
        # Has enough info or didn't request more
        state["medical_agent_needs_info"] = False
        state["medical_agent_output"] = response_text

    logger.debug("Exiting medical_agent_node")
    return state


def extract_info_request(response_text: str) -> str:
    """
    Extract the specific information request from medical agent's response.
    Looks for content after NEED_MORE_INFO or similar triggers.
    """
    # Look for the information request section
    patterns = [
        r"NEED_MORE_INFO\s*\n\s*\*\*INFORMATION NEEDED:\*\*\s*\n(.*?)\n\s*\*\*CLINICAL JUSTIFICATION:\*\*",
        r"INFORMATION NEEDED:\s*\n(.*?)\n\s*\*\*CLINICAL JUSTIFICATION:\*\*",
        r"NEED_MORE_INFO[:\s]*\n(.*?)(?:\n\n|\Z)",
        r"DATA DISCREPANCY IDENTIFIED.*?NEED_MORE_INFO:\s*(.*?)(?:\n\n|\Z)"
    ]
    
    import re
    for pattern in patterns:
        match = re.search(pattern, response_text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    # Fallback: return everything after first trigger found
    for trigger in ["NEED_MORE_INFO", "INFORMATION NEEDED:", "DATA DISCREPANCY"]:
        if trigger in response_text:
            idx = response_text.index(trigger)
            return response_text[idx:idx+500].strip()  # Get next 500 chars
    
    return "Additional information needed for accurate assessment"



def output_refiner_node(state: AgentState):
    """
    Refines the medical agent output to be more user-friendly while
    preserving all medical accuracy and recommendations.
    """
    logger.info("Entering Node: output_refiner_node")
    medical_output = state.get("medical_agent_output", "No medical output available")
    
    # Initialize LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-preview-09-2025",
        google_api_key=os.environ["GOOGLE_API_KEY"],
    )
    
    # System prompt with strict medical preservation instructions
    system_prompt = """You are a medical communication expert. Your task is to make medical information more accessible and user-friendly while maintaining ABSOLUTE medical accuracy.

    STRICT RULES - YOU MUST FOLLOW THESE:
    1. DO NOT change, modify, or omit ANY medical diagnoses, test results, measurements, or clinical findings
    2. DO NOT alter any medication names, dosages, or treatment recommendations
    3. DO NOT change any medical terminology that is critical to understanding the diagnosis
    4. DO NOT add any medical advice or recommendations that weren't in the original output
    5. DO NOT remove any warnings, precautions, or important medical information

    WHAT YOU CAN DO:
    - Add simple explanations of complex medical terms in parentheses
    - Organize information with clear headings and structure
    - Use more conversational language for non-medical connecting text
    - Add brief context to help patients understand their results
    - Break down long paragraphs into digestible sections
    - Use bullet points or numbering for clarity where appropriate

    ## Output Formatting Guidelines

    When generating responses, use the following markdown-style formatting that will be rendered in the user interface:

    **Supported Formatting:**
    - Use **double asterisks** for bold text: **bold text**
    - Use ## for main headings and ### for subheadings
    - Use *** or --- on a single line for horizontal dividers
    - Use * or - at the start of lines for bullet points
    - Use double line breaks to separate paragraphs

    **DO NOT USE:**
    - Markdown tables with pipes (| Column | Column |) - Instead, describe information in bullet points or simple text
    - Complex nested formatting or multiple formatting styles on the same text
    - Code blocks with ``` backticks
    - Numbered lists (use bullets instead)
    - Inline code with single backticks

    Keep formatting simple and one level deep for optimal display in the web interface.

    Your goal: Make the information easier to understand while keeping every medical fact exactly as stated."""
    
    # Create messages with system and user prompts
    messages = [
        ("system", system_prompt),
        ("human", f"Please refine this medical information to be more user-friendly:\n\n{medical_output}")
    ]
    
    # Invoke LLM
    response = llm.invoke(messages)
    
    # Extract refined content
    refined_output = response.content
    # Normalize excessive blank lines to reduce spacing in the UI
    import re
    refined_output = re.sub(r"\r\n", "\n", str(refined_output))
    refined_output = re.sub(r"\n{2,}", "\n", refined_output).strip()
    
    # Store refined output in state
    state["final_refined_medical_output"] = refined_output

    logger.debug("Exiting output_refiner_node")
    return state


def should_continue_reflection(state: AgentState) -> Literal["db_retriever_agent_node", "output_refiner_node"]:
    """
    Determines whether to continue reflection loop or end.
    """
    if state.get("medical_agent_needs_info", False):
        logger.debug("Routing to db_retriever_agent_node (Needs Info)")
        return "db_retriever_agent_node"
    logger.debug("Routing to output_refiner_node (Analysis Complete)")
    return "output_refiner_node"


def input_image_classification_route(state: AgentState):
    full_images_results = state["input_images_classification_results"]

    for image_result in full_images_results:
        if image_result[2] == "not_valid_image":
            state["coming_from_validation"] = "input_image"
            logger.debug("Routing to input_not_valid_fallback_node (Invalid Image)")
            return Command(update=state, goto=["input_not_valid_fallback_node"])
    
    nodes_to_return = []
    if state.get("text_images"):
        nodes_to_return.append("extract_text_from_images_node")

    if state.get("medical_images"):
        nodes_to_return.append("medical_vision_models_agent_node")

    if state.get("input_prompt") and not state.get("text_images"):
        nodes_to_return.append("numerical_models_agent_node")

    logger.debug(f"Routing to: {nodes_to_return}")
    return Command(
        update={
            "input_images_classification_results": state["input_images_classification_results"],
            "text_images": state.get("text_images"),
            "medical_images": state.get("medical_images")
        },
        goto=nodes_to_return
    )


def second_input_validation_route(state: AgentState):
    full_validation_results = state["extracted_text_from_images_or_attachments_validation_results"]

    for validation_result in full_validation_results:
        if validation_result != "TEXT_VALID":
            state["coming_from_validation"] = "second_input_text"
            logger.debug("Routing to input_not_valid_fallback_node (Invalid Text)")
            return Command(update=state, goto=["input_not_valid_fallback_node"])


    logger.debug("Routing to numerical_models_agent_node")
    return Command(
    update={
        "extracted_text_from_images_or_attachments_validation_results": state["extracted_text_from_images_or_attachments_validation_results"]
    },
    goto=["numerical_models_agent_node"]
    )



workflow = StateGraph(AgentState)


workflow.add_node("first_input_validation_node", first_input_validation_node)
workflow.add_node("input_not_valid_fallback_node", input_not_valid_fallback_node)
workflow.add_node("input_image_classification_node", input_image_classification_node)
workflow.add_node("second_input_validation_node", second_input_validation_node)
workflow.add_node("extract_text_from_images_node", extract_text_from_images_node)
workflow.add_node("extract_text_from_files_node", extract_text_from_files_node)
workflow.add_node("numerical_models_agent_node", numerical_models_agent_node)
workflow.add_node("medical_vision_models_agent_node", medical_vision_models_agent_node)
workflow.add_node("output_refiner_node", output_refiner_node)

workflow.add_node("models_agents_output_aggregator_node", models_agents_output_aggregator_node)
workflow.add_node("db_retriever_agent_node", db_retriever_agent_node)
workflow.add_node("medical_agent_node", medical_agent_node)


workflow.add_edge("extract_text_from_images_node", "second_input_validation_node")
workflow.add_edge("extract_text_from_files_node", "second_input_validation_node")

workflow.add_edge("numerical_models_agent_node", "models_agents_output_aggregator_node")
workflow.add_edge("medical_vision_models_agent_node", "models_agents_output_aggregator_node")
workflow.add_edge("models_agents_output_aggregator_node", "db_retriever_agent_node")
workflow.add_edge("db_retriever_agent_node", "medical_agent_node")

workflow.add_edge("output_refiner_node", END)


workflow.add_conditional_edges(
    "medical_agent_node",
    should_continue_reflection,
    {
        "db_retriever_agent_node": "db_retriever_agent_node",
        "output_refiner_node": "output_refiner_node"
    }
)



from langgraph.checkpoint.memory import MemorySaver

workflow.set_entry_point("first_input_validation_node")

# Use MemorySaver for state persistence to support interrupts
checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)

def run_workflow(initial_state, thread_id: str):
    config = {
        "configurable": {"thread_id": thread_id},
        "metadata": {"user_id": initial_state.get("user_id"), "flow_type": "medical_analysis"},
        "recursion_limit": 300  # Add this
    }
    returned_state = app.invoke(initial_state, config=config)
    return returned_state
