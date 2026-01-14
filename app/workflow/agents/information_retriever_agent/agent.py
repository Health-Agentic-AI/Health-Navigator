import os
from typing import Dict, Any
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_core.tools import tool
from langchain.agents import create_agent
from langgraph.types import interrupt
import time 

import sys
sys.path.append(r"C:\My Projects\Health-Navigator")

from dotenv import load_dotenv
# Load generic path for .env
load_dotenv(os.path.join(os.getcwd(), 'credentials.env'))

from app.workflow.vectordb.vectordb import HybridVectorDB


@tool
def retrieve_from_vector_db(
    user_id: str,
    query: str,
    top_k: int = 100,
    filters: dict = None,
    date: str = None,
    date_filter: str = None
) -> list:
    """
    Retrieves relevant medical records and health documents from a user's personal vector database.
    
    Uses hybrid search (semantic + BM25) to find the most relevant information based on the query.
    Each user has their own isolated database identified by user_id.
    
    Args:
        user_id (str): Unique identifier for the user whose database to search
        query (str): Search query to find relevant documents
        top_k (int, optional): Number of results to return. Defaults to 100, to make sure you return all relevant results.
        filters (dict, optional): Metadata filters to apply, e.g., {'type': 'prescription', 'doctor': 'Dr. Smith'}
        date (str, optional): Date for filtering in 'YYYY-MM-DD' format
        date_filter (str, optional): How to filter by date - 'before', 'at', or 'after'. Requires date parameter, you can only pass these three values ('before', 'at', 'after').
    
    Returns:
        list: List of dictionaries, each containing:
            - text (str): The retrieved document text
            - metadata (dict): Document metadata (type, date, doctor, etc.)
            - score (float): Relevance score
    
    Example:
        # Basic search
        results = retrieve_from_vector_db(user_id="user123", query="blood pressure medications")
        
        # Search with filters and date
        results = retrieve_from_vector_db(
            user_id="user123",
            query="lab results",
            top_k=5,
            filters={'type': 'lab_report'},
            date="2024-01-01",
            date_filter="after"
        )
    """
    print(f"DEBUG: retrieve_from_vector_db called. Query: {query}, User ID: {user_id}")
    db = HybridVectorDB(user_id=user_id)
    
    results = db.retrieve(
        query=query,
        top_k=top_k,
        filters=filters,
        date=date,
        date_filter=date_filter
    )
    print(f"DEBUG: retrieve_from_vector_db found {len(results)} results.")
    return results


@tool
def add_to_vector_db(
    user_id: str,
    text: str,
    metadata: dict = None
) -> bool:
    """
    Adds and indexes new medical text or health documents to a user's personal vector database.
    
    The function processes the text into nodes, extracts embeddings for semantic search, 
    and updates the BM25 index for hybrid retrieval. Dates in metadata are automatically 
    converted to integers for efficient filtering.
    
    Args:
        user_id (str): Unique identifier for the user whose database to update.
        text (str): The actual document content or medical note to be stored.
        metadata (dict, optional): Additional context such as {'type': 'prescription', 
                                   'date': '2024-05-20', 'doctor': 'Dr. Jordan'}.
    
    Returns:
        bool: True if the document was successfully indexed, False otherwise.
        
    Example:
        # Adding a new lab result
        success = add_to_vector_db(
            user_id="user123",
            text="Patient blood glucose levels are within normal range (95 mg/dL).",
            metadata={'type': 'lab_report', 'date': '2024-06-12'}
        )
    """
    try:
        db = HybridVectorDB(user_id=user_id)
        return db.add_text(text=text, metadata=metadata)
    except Exception as e:
        print(f"Failed to initialize database for user {user_id}: {e}")
        return False

@tool
def ask_user_for_info(request: str) -> str:
    """
    Request information from the user when it's not available in databases.
    Use this when you need subjective information, recent events not in records,
    current symptoms, or clarification that only the user can provide.
    
    Args:
        request: Specific question to ask the user (be clear and concise)
        
    Returns:
        User's response
    """
    # Trigger an interrupt to ask the user via frontend
    user_response = interrupt(request)
    return user_response

# Helpers for lazy loading
def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-3-pro-preview",
        google_api_key=os.environ.get("GOOGLE_API_KEY"),
        name="Information Retriever Agent"
    )

def get_sql_db():
    try:
        pg_uri = f'postgresql+psycopg2://{os.environ.get("POSTGRES_USERNAME")}:{os.environ.get("POSTGRES_PASSWORD")}@{os.environ.get("POSTGRES_HOST")}:{os.environ.get("POSTGRES_PORT")}/{os.environ.get("DATABASE_NAME")}'
        return SQLDatabase.from_uri(pg_uri)
    except Exception as e:
        print(f"Failed to connect to DB: {e}")
        return None

# DB Retriever Agent System Prompt
DB_RETRIEVER_SYSTEM_PROMPT = """You are a Medical Information Retrieval Specialist with access to vector databases, relational databases, and the ability to request information directly from users.

## Available Tools:
- **retrieve_from_vector_db**: Search medical knowledge base and patient documents
- **add_to_vector_db**: Store new information for future reference
- **SQL Database Tools**: Query structured patient records, lab results, medications, appointments, etc.
- **ask_user_for_info**: Request information directly from the user when not in databases

## Decision Framework:

### Query Database When:
- Patient's medical history, past diagnoses, procedures, medications
- Lab results, imaging reports, clinical notes
- Medical knowledge from guidelines or research
- Similar cases or treatment protocols
- Demographic or administrative data

### Use ask_user_for_info Tool When:
- Subjective information (current symptoms, pain levels, concerns)
- Recent events not yet documented
- Current home medications or lifestyle factors
- Personal preferences or context
- Clarification needed
- Family history not in records

## Workflow:
1. Start by querying databases for available information
2. Use ask_user_for_info tool if critical information is missing
3. Continue until you have comprehensive information
4. When complete, respond with:
```
INFORMATION_COMPLETE

**VECTOR DATABASE RESULTS:**
[Medical knowledge, research, guidelines]

**RELATIONAL DATABASE RESULTS:**
[Patient records, history, test results]

**USER PROVIDED INFORMATION:**
[Information gathered via ask_user_for_info tool]

**SUMMARY:** [Brief synthesis of all retrieved information]
```

## Guidelines:
- Always query databases BEFORE using ask_user_for_info
- Be specific in questions to users - explain why information is needed
- Current iteration: {reflection_count}/{max_reflections}
- At max iterations, work with available information
- Only retrieve and organize - do NOT provide medical advice

Current date and time: {date_time} in format YYYY-MM-DD HH:MM:SS

Remember: The Medical Agent handles clinical analysis. Your role is comprehensive information retrieval."""


def invoke_db_retriever_agent(
    aggregated_output: str,
    info_request: str,
    reflection_count: int,
    max_reflections: int,
    user_id: str,
    conversation_history: list,
) -> Dict[str, Any]:
    """
    Invokes the DB retriever agent to gather information from databases.
    
    Returns:
        dict with keys: 'response', 'conversation_history', 'needs_more_info'
    """
    # Lazy load tools/resources
    llm = get_llm()
    sql_db = get_sql_db()

    if not sql_db:
        # Fallback if DB not available (e.g. build time) or throw error
        # Assuming for now we want to proceed or crash
        pass

    sql_toolkit = SQLDatabaseToolkit(db=sql_db, llm=llm)
    sql_tools = sql_toolkit.get_tools()

    # Combine SQL tools with vector DB tools
    all_tools = sql_tools + [retrieve_from_vector_db, add_to_vector_db, ask_user_for_info]

    # Build initial query context
    if info_request:
        query_context = f"""
        Initial Medical Analysis Context:
        {aggregated_output}
        
        Medical Agent's Information Request:
        {info_request}
        
        Task: Retrieve the specific information requested by the Medical Agent.
        Determine if this information exists in databases or needs to be obtained from the user.
        """
    else:
        query_context = f"""
        Medical Analysis Context:
        {aggregated_output}
        
        Task: Gather comprehensive patient information from all available databases
        to support medical assessment.
        """
    date_time = time.strftime('%Y-%m-%d %H:%M:%S')
    # Format system prompt with current iteration info
    formatted_system_prompt = DB_RETRIEVER_SYSTEM_PROMPT.format(
        reflection_count=reflection_count,
        max_reflections=max_reflections,
        date_time=date_time
    )
    
    retriever_agent = create_agent(
        llm,
        tools=all_tools,
        system_prompt=formatted_system_prompt
    )
    # Prepare messages for agent
    agent_input = {
        "messages": conversation_history + [
            HumanMessage(content=query_context)
        ]
    }
    
    # Invoke agent
    print(f"\n--- Information Retriever Agent Invoked ---")
    print(f"DEBUG: Query Context: {query_context[:200]}...")

    result = retriever_agent.invoke(agent_input)
    agent_response = result["messages"][-1].content
    
    # Determine if information is complete
    needs_more_info = "INFORMATION_COMPLETE" not in agent_response
    
    print(f"DEBUG: Retriever Agent Response: {agent_response[:200]}...")
    print(f"DEBUG: Needs More Info: {needs_more_info}")

    return {
        'response': agent_response,
        'conversation_history': result["messages"],
        'needs_more_info': needs_more_info
    }
