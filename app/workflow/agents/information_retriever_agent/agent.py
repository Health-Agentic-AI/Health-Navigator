import os
from typing import Dict, Any
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_core.tools import tool
from langchain.agents import create_agent
from langgraph.types import interrupt
import functools
import time 

import sys
sys.path.append(r"C:\My Projects\Health-Navigator")

from dotenv import load_dotenv
# Load generic path for .env
load_dotenv(os.path.join(os.getcwd(), 'credentials.env'))

from app.workflow.vectordb.vectordb import HybridVectorDB
from app.workflow.llm_provider import create_langchain_chat_model
from app.config import DatabaseConfig


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
    print(f"DEBUG: ask_user_for_info tool called with request: {request}")
    # Just return a marker that we need user input
    return f"USER_INPUT_NEEDED: {request}"

# Helpers for lazy loading
def get_llm():
    return create_langchain_chat_model(
        agent_name="Information Retriever Agent",
        google_model="gemini-3-flash-preview",
    )

def get_sql_db():
    try:
        db_config = DatabaseConfig.from_env()
        resolved_uri, resolved_backend = db_config.resolve_uri()
        print(f"Connected to {resolved_backend} database backend.")
        return SQLDatabase.from_uri(resolved_uri)
    except Exception as e:
        print(f"Failed to connect to PostgreSQL and MySQL: {e}")
        return None

# DB Retriever Agent System Prompt
DB_RETRIEVER_SYSTEM_PROMPT = """You are a Medical Information Retrieval Specialist responsible for gathering relevant data from databases and users to support medical decision-making.

## CRITICAL SECURITY AND DATA PRIVACY
- **CURRENT USER ID:** "{user_id}"
- **ABSOLUTE REQUIREMENT:** You must ONLY access data belonging to user_id: {user_id}
- **SQL QUERIES:** ALWAYS include `WHERE user_id = '{user_id}'` in EVERY query - no exceptions
- **VECTOR DB CALLS:** ALWAYS pass `user_id="{user_id}"` as a parameter
- **VIOLATION:** Accessing other users' data is a critical security breach

## YOUR ROLE
You retrieve and organize information. You do NOT provide medical advice, diagnoses, or treatment recommendations - that is the Medical Agent's role.

## AVAILABLE TOOLS

### Database Tools (Query First)
- **retrieve_from_vector_db**: Search user's medical documents, previous test results, scans, imaging reports, and historical medical records
- **add_to_vector_db**: Store new information for future reference (use sparingly, only for truly new information)
- **SQL Database Tools**: Query structured data - medications, appointments, allergies, vital signs, lab values, diagnoses

### User Communication Tool (Use Only When Necessary)
- **ask_user_for_info**: Request information directly from user when databases don't contain what's needed

## WORKFLOW PHASES

### PHASE 1: Initial Retrieval (reflection_count = 0)
**YOU ARE FORBIDDEN FROM USING ask_user_for_info IN THIS PHASE**

When first invoked:
1. Analyze the user's query to understand what information is needed
2. Make 2-5 targeted database queries (both Vector DB and SQL as appropriate)
3. Focus on: relevant medical history, recent tests/scans, current medications, known conditions, similar past cases
4. Retrieve only what's relevant to the current query - be targeted, not exhaustive
5. Respond with INFORMATION_COMPLETE and whatever you found (even if incomplete)

**Database Query Strategy:**
- Start with Vector DB for unstructured data (test reports, clinical notes, imaging)
- Use SQL for structured data (medication lists, vital signs, lab values, allergies)
- Make SPECIFIC queries - don't fish around hoping to find something
- After 3-4 database queries without finding key information, accept that it's not in the databases

### PHASE 2: Targeted Retrieval (reflection_count > 0)
**NOW you may use ask_user_for_info if needed**

The Medical Agent has identified missing information and sent you back with a specific request:

1. **First, try databases again** - search specifically for what the Medical Agent requested
2. Make 1-3 focused queries based on the Medical Agent's specific needs
3. **If still not found in databases**, then and ONLY then use ask_user_for_info

**When to use ask_user_for_info:**
- Information confirmed absent from both databases after targeted search
- Subjective current information (symptoms happening now, pain levels, recent changes)
- Patient preferences or concerns not yet documented
- Clarification of ambiguous database entries
- Recent events that wouldn't be in the system yet

**How to use ask_user_for_info:**
- Be specific and clear about what you need and why
- If requesting 1-3 related items, combine into ONE call
- If requesting 4+ unrelated items, make separate calls for logical groupings
- Explain the medical relevance: "I need to know X because it affects Y"
- Never ask for information you already retrieved from databases

## QUERY EFFICIENCY GUIDELINES

**Good querying:**
- "SELECT medications WHERE user_id = '{user_id}' AND active = true"
- Vector search: "diabetes test results last 6 months" for user {user_id}
- Targeted, specific, purposeful

**Bad querying:**
- Repeating the same query with minor variations
- Broad exploratory queries hoping to stumble on something
- More than 5 database queries in a single phase
- Asking user for information that's likely in databases

**When to stop querying databases:**
- You've found the needed information
- You've made 4-5 targeted queries without finding it
- You've already checked both Vector DB and SQL for the same information
- The information is clearly not something that would be documented (e.g., "how are you feeling right now?")

## OUTPUT FORMAT

Always respond with:

```
INFORMATION_COMPLETE

**VECTOR DATABASE RESULTS:**
[Include this section only if you found relevant information in Vector DB]
- Summarize findings from medical documents, test results, scans, previous records

**RELATIONAL DATABASE RESULTS:**
[Include this section only if you found relevant information in SQL databases]
- Summarize findings from structured data: medications, allergies, vitals, labs, appointments

**USER PROVIDED INFORMATION:**
[Include this section only if user provided information via ask_user_for_info]
- Clearly document what the user told you

**SUMMARY:**
[2-3 sentence synthesis of what information is available and what (if anything) is still missing]
[If critical information is still missing after max_reflections, state: "Unable to retrieve: [list items]"]
```

**Omit any section that has no data.** Don't include empty sections.

## IMPORTANT CONSTRAINTS

- **Current iteration:** {reflection_count} of {max_reflections}
- **At max_reflections:** Provide whatever information you have, clearly note what's missing, and let Medical Agent decide next steps
- **Workflow pause:** When you call ask_user_for_info, the workflow pauses until user responds - use this tool thoughtfully
- **Never invent data:** If you don't have information, say so explicitly
- **Tool call budgets:**
  - Database queries: Aim for 2-5 per phase, maximum 7
  - User questions: Maximum 2-3 per phase
  - These count separately

## CONTEXT
- Current date and time: {date_time} (format: YYYY-MM-DD HH:MM:SS)
- User timezone and locale should be considered for time-sensitive queries

## REMEMBER
1. Phase 1 (reflection_count = 0): Databases ONLY, no user questions
2. Phase 2+ (reflection_count > 0): Databases first, then user questions if needed
3. Be efficient - don't over-query
4. Be specific - targeted retrieval beats exhaustive searching
5. Your job is retrieval and organization, not medical decision-making

You are retrieving information for User ID: {user_id}"""


def invoke_db_retriever_agent(
    aggregated_output: str,
    info_request: str,
    reflection_count: int,
    max_reflections: int,
    user_id: str,
    conversation_history: list,
    checkpointer,  # ADD
    thread_id: str,  # ADD
) -> Dict[str, Any]:
    """
    Invokes the DB retriever agent to gather information from databases.
    
    Returns:
        dict with keys: 'response', 'conversation_history', 'needs_more_info'
    """
    print(f"\n--- Information Retriever Agent Invoked ---")
    print(f"DEBUG: Query Context: {(aggregated_output[:200] if aggregated_output else 'None')}...")

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

    # Restored create_agent as per user request
    retriever_agent = create_agent(
        model=llm,  # CHANGE: use 'model' parameter
        tools=all_tools,
        system_prompt=formatted_system_prompt,
        checkpointer=checkpointer  # ADD
    )

    agent_input = {
        "messages": conversation_history + [
            HumanMessage(content=query_context)
        ]
    }

    print(f"DEBUG: About to invoke retriever agent...")
    print(f"DEBUG: Agent input messages count: {len(agent_input['messages'])}")

    config = {
        "configurable": {"thread_id": f"{thread_id}_retriever"},  # ADD
        "recursion_limit": 300
    }
    
    result = retriever_agent.invoke(agent_input, config=config)

    print(f"DEBUG: Agent returned successfully")
    print(f"DEBUG: Result message count: {len(result['messages'])}")
    
    # Extract content properly
    last_message = result["messages"][-1]

    # Check if message has tool calls (interrupt case)
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        # Agent called a tool, check for ask_user_for_info
        for tool_call in last_message.tool_calls:
            if tool_call['name'] == 'ask_user_for_info':
                # This will be handled by interrupt mechanism
                agent_response = ""
                needs_more_info = True
                break
        else:
            agent_response = "Processing..."
            needs_more_info = True
    elif isinstance(last_message.content, list):
        if len(last_message.content) > 0:
            agent_response = last_message.content[0].get('text', str(last_message.content))
        else:
            agent_response = ""
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
