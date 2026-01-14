import os
from typing import Dict, Any
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_core.tools import tool
from langchain.agents import create_agent
from langgraph.types import interrupt
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import functools
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
        model="gemini-3-flash-preview",
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

## CRITICAL SECURITY AND DATA PRIVACY CONTEXT
- **CURRENT USER ID:** "{user_id}"
- **STRICT REQUIREMENT:** You must ONLY query, retrieve, or access data belonging to this specific user_id.
- **SQL QUERIES:** ALWAYS include `WHERE user_id = '{user_id}'` (or equivalent column) in every SQL query. Never select all rows without filtering by user.
- **VECTOR DB:** Always pass `user_id="{user_id}"` when calling `retrieve_from_vector_db` or `add_to_vector_db`.

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

## Query Efficiency Rules:
- Make SPECIFIC, targeted queries - not broad exploratory queries
- After 2-4 database queries, you should have enough context to decide if user input is needed
- Do NOT repeatedly query the same data with slight variations
- If you've already retrieved medical history, lab results, and medications - that's comprehensive
- Better to ask user for missing details than keep searching databases

## Workflow:
1. Make 1-3 targeted database queries (SQL and/or Vector DB) to find relevant information
2. If you find sufficient information OR have made 3+ queries, proceed to step 3
3. If critical information is still missing after database queries, use ask_user_for_info ONCE or TWICE to gather necessary details 
4. After getting user response OR if no critical info is missing, respond with:

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

Remember: The Medical Agent handles clinical analysis. Your role is comprehensive information retrieval for User: {user_id}."""


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
    def _run_agent():
        llm = get_llm()
        sql_db = get_sql_db()
        print(f"DEBUG: SQL DB connection successful: {sql_db is not None}")

        if not sql_db:
            raise Exception("SQL Database connection failed")

        sql_toolkit = SQLDatabaseToolkit(db=sql_db, llm=llm)
        sql_tools = sql_toolkit.get_tools()

        print(f"DEBUG: Number of SQL tools: {len(sql_tools)}")
        print(f"DEBUG: Number of total tools: {len(sql_tools) + 3}")

        all_tools = sql_tools + [retrieve_from_vector_db, add_to_vector_db, ask_user_for_info]

        if info_request:
            query_context = f"""
            Initial Medical Analysis Context:
            {aggregated_output}
            
            Medical Agent's Information Request:
            {info_request}
            
            Task: Retrieve the specific information requested by the Medical Agent for User ID: {user_id}.
            Determine if this information exists in databases or needs to be obtained from the user.
            """
        else:
            query_context = f"""
            Medical Analysis Context:
            {aggregated_output}
            
            Task: Gather comprehensive patient information from all available databases
            to support medical assessment for User ID: {user_id}.
            """
        
        date_time = time.strftime('%Y-%m-%d %H:%M:%S')
        formatted_system_prompt = DB_RETRIEVER_SYSTEM_PROMPT.format(
            reflection_count=reflection_count,
            max_reflections=max_reflections,
            date_time=date_time,
            user_id=user_id
        )
        
        retriever_agent = create_agent(
            llm,
            tools=all_tools,
            system_prompt=formatted_system_prompt,
            max_iterations=10
        )

        agent_input = {
            "messages": conversation_history + [
                HumanMessage(content=query_context)
            ]
        }
        
        print(f"DEBUG: About to invoke retriever agent...")
        print(f"DEBUG: Agent input messages count: {len(agent_input['messages'])}")

        result = retriever_agent.invoke(agent_input)

        print(f"DEBUG: Agent returned successfully")
        print(f"DEBUG: Result message count: {len(result['messages'])}")
        
        return result

    print(f"\n--- Information Retriever Agent Invoked ---")
    print(f"DEBUG: Query Context: {(aggregated_output[:200] if aggregated_output else 'None')}...")
    
    # Run with timeout
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_run_agent)
        try:
            result = future.result(timeout=30)  # 30 second timeout
        except FuturesTimeoutError:
            print("ERROR: DB Retriever Agent timed out after 30 seconds")
            # Return what we have from aggregated output
            timeout_response = f"""INFORMATION_COMPLETE

        **VECTOR DATABASE RESULTS:**
        Query timed out - using information from previous analysis.

        **RELATIONAL DATABASE RESULTS:**
        Query timed out - using information from previous analysis.

        **USER PROVIDED INFORMATION:**
        None

        **SUMMARY:** Database retrieval exceeded time limit. Proceeding with information from initial analysis: {aggregated_output[:500]}..."""
            
            return {
                'response': [{'text': timeout_response}],
                'conversation_history': conversation_history,
                'needs_more_info': False
            }
        except Exception as e:
            print(f"ERROR: DB Retriever Agent failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                'response': [{'text': f'INFORMATION_COMPLETE\n\nDatabase error: {str(e)}'}],
                'conversation_history': conversation_history,
                'needs_more_info': False
            }
    
    # Extract content properly
    last_message = result["messages"][-1]
    if isinstance(last_message.content, list):
        agent_response = last_message.content[0].get('text', str(last_message.content))
    else:
        agent_response = str(last_message.content)
    
    needs_more_info = "INFORMATION_COMPLETE" not in agent_response
    
    print(f"DEBUG: Retriever Agent Response: {agent_response[:200]}...")
    print(f"DEBUG: Needs More Info: {needs_more_info}")

    return {
        'response': [{'text': agent_response}],
        'conversation_history': result["messages"],
        'needs_more_info': needs_more_info
    }