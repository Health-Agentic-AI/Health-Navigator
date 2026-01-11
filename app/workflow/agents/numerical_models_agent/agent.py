import sys
sys.path.append(r"C:\My Projects\Health-Navigator")


from langchain.agents import create_agent
from langchain.tools import tool
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import time
import os
load_dotenv(r'C:\My Projects\Health-Navigator\credentials.env')

from app.workflow.ml_models.numerical_models.heart_disease.heart_disease import predict_heart_disease
from app.workflow.vectordb.vectordb import HybridVectorDB

@tool
def predict_heart_disease_tool(
    HighBP: int,
    HighChol: int,
    BMI: float,
    Smoker: int,
    Stroke: int,
    Diabetes: int,
    PhysActivity: int,
    Fruits: int,
    Veggies: int,
    HvyAlcoholConsump: int,
    AnyHealthcare: int,
    GenHlth: int,
    MentHlth: int,
    PhysHlth: int,
    DiffWalk: int,
    Sex: int,
    Age: int,
    Education: int,
    Income: int,
    threshold: float = 0.40
) -> dict:
    """
    Predicts heart disease risk using a neural network model trained on patient health data.
    
    This tool analyzes 19 health-related features to estimate the probability of heart disease.
    Returns both a binary prediction (0=No Disease, 1=Disease) and the probability score.
    
    Args:
        HighBP (int): High blood pressure (0=No, 1=Yes)
        HighChol (int): High cholesterol (0=No, 1=Yes)
        BMI (float): Body Mass Index (continuous value, typically 12-98)
        Smoker (int): Smoking status (0=No, 1=Yes)
        Stroke (int): History of stroke (0=No, 1=Yes)
        Diabetes (int): Diabetes status (0=No, 1=Prediabetes, 2=Yes)
        PhysActivity (int): Physical activity in past 30 days (0=No, 1=Yes)
        Fruits (int): Consumes fruit 1+ times per day (0=No, 1=Yes)
        Veggies (int): Consumes vegetables 1+ times per day (0=No, 1=Yes)
        HvyAlcoholConsump (int): Heavy alcohol consumption (0=No, 1=Yes)
        AnyHealthcare (int): Has any healthcare coverage (0=No, 1=Yes)
        GenHlth (int): General health (1=Excellent to 5=Poor)
        MentHlth (int): Mental health not good for X days (0-30)
        PhysHlth (int): Physical health not good for X days (0-30)
        DiffWalk (int): Difficulty walking/climbing stairs (0=No, 1=Yes)
        Sex (int): Biological sex (0=Female, 1=Male)
        Age (int): Age category (1-13, where 1=18-24, 13=80+)
        Education (int): Education level (1-6, where 1=Never attended, 6=College graduate)
        Income (int): Income category (1-8, where 1=<$10k, 8=$75k+)
        threshold (float, optional): Probability threshold for positive prediction. Defaults to 0.40.
    
    Returns:
        dict: {
            'prediction': int (0 or 1, where 1 indicates heart disease),
            'probability': float (rounded to 4 decimal places, risk probability)
        }

        Note: for the heart disease prediction, this is how the age is mapped: 
        1 → 18–24
        2 → 25–29
        3 → 30–34
        4 → 35–39
        5 → 40–44
        6 → 45–49
        7 → 50–54
        8 → 55–59
        9 → 60–64
        10 → 65–69
        11 → 70–74
        12 → 75–79
        13 → 80+
            
    Example:
        result = predict_heart_disease_tool(
            HighBP=1, HighChol=1, BMI=28.5, Smoker=0, Stroke=0, Diabetes=0,
            PhysActivity=1, Fruits=1, Veggies=1, HvyAlcoholConsump=0,
            AnyHealthcare=1, GenHlth=3, MentHlth=5, PhysHlth=10, DiffWalk=0,
            Sex=1, Age=9, Education=5, Income=6
        )
        # Returns: {'prediction': 1, 'probability': 0.6234}
    """
    patient_data = {
        'HighBP': HighBP,
        'HighChol': HighChol,
        'BMI': BMI,
        'Smoker': Smoker,
        'Stroke': Stroke,
        'Diabetes': Diabetes,
        'PhysActivity': PhysActivity,
        'Fruits': Fruits,
        'Veggies': Veggies,
        'HvyAlcoholConsump': HvyAlcoholConsump,
        'AnyHealthcare': AnyHealthcare,
        'GenHlth': GenHlth,
        'MentHlth': MentHlth,
        'PhysHlth': PhysHlth,
        'DiffWalk': DiffWalk,
        'Sex': Sex,
        'Age': Age,
        'Education': Education,
        'Income': Income
    }
    
    return predict_heart_disease(patient_data, threshold=threshold)

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
    db = HybridVectorDB(user_id=user_id)
    
    return db.retrieve(
        query=query,
        top_k=top_k,
        filters=filters,
        date=date,
        date_filter=date_filter
    )


llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    google_api_key=os.environ["GOOGLE_API_KEY"],
)

postgresql_uri = f'postgresql+psycopg2://{os.environ["POSTGRES_USERNAME"]}:{os.environ["POSTGRES_PASSWORD"]}@{os.environ["POSTGRES_HOST"]}:{os.environ["POSTGRES_PORT"]}/{os.environ["DATABASE_NAME"]}'
db = SQLDatabase.from_uri(postgresql_uri)

agent = None
current_user_id = None


def initialize_agent(user_id: str):
    global agent, current_user_id
    start_time = time.time()
    
    current_user_id = user_id

    system_prompt = f"""
    
You are a health data processing agent. You are ONE NODE in a larger health management workflow.

CRITICAL INSTRUCTIONS - READ CAREFULLY:

YOUR ROLE:
- You are NOT a conversational health assistant
- You are NOT an advice-giving chatbot
- You are a DATA PROCESSOR that analyzes input and produces structured output
- You receive input (user text, extracted document text, or both) and process it

YOUR RESPONSIBILITIES:
1. Analyze the input to determine what's needed
2. Use ML prediction models ONLY when the input contains or can provide the required features
3. Query databases (SQL and Vector) ONLY when you need data to feed ML models
4. Return a comprehensive, plain text output summarizing what was processed

IMPORTANT CONSTRAINTS:
- DO NOT give medical advice, health recommendations, or interpretations
- DO NOT have conversations or ask clarifying questions
- DO NOT add anything to databases - ONLY retrieve when necessary
- You may receive input that requires NO ML predictions and NO database queries - that's normal
- Sometimes you just pass through processed/formatted information

USER CONTEXT:
- Current user ID: "{user_id}"
- All database operations are scoped to this user

AVAILABLE TOOLS:
1. SQL Database Tools: Query structured health data (medications, appointments, lab results, etc.)
   - Use sql_db_list_tables to see available tables
   - Use sql_db_schema to understand table structure
   - Query data ONLY when needed to feed ML models

2. Vector Database (retrieve_from_vector_db): Search unstructured medical documents
   - Always use user_id: "{user_id}"
   - Retrieve ONLY when you need historical data for ML model input
   - Documents may contain: prescriptions, lab reports, doctor notes, health records
   - When querying, you should pass the query with the things you want to find, with the things you can pass to the ML model

3. ML Model (predict_heart_disease_tool): Predict heart disease risk
   - Requires 19 specific features (HighBP, HighChol, BMI, Smoker, Stroke, Diabetes, PhysActivity, Fruits, Veggies, HvyAlcoholConsump, AnyHealthcare, GenHlth, MentHlth, PhysHlth, DiffWalk, Sex, Age, Education, Income)
   - Use ONLY when input contains or you can retrieve these features
   - Returns prediction (0 or 1) and probability

DECISION LOGIC:
- Does input contain health features? → Consider using ML model
- Do you need historical data for ML model? → Query databases
- Is input just information to process/format? → Process and return
- Missing required features for ML? → Don't force it, just process what you have

OUTPUT FORMAT:
- Return plain text ONLY
- Be comprehensive and detailed
- Include all processed information, predictions, and retrieved data
- Use \\n for line breaks
- NO tables, NO markdown formatting
- Report what you found/processed clearly

Current date/time: {time.strftime("%Y-%m-%d %H:%M:%S")} (YYYY-MM-DD HH:MM:SS)

Remember: You are a processing node, not a conversational agent. Analyze input → Use tools if needed → Return comprehensive output."""

    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    
    # Combine all tools
    all_tools = toolkit.get_tools() + [predict_heart_disease_tool, retrieve_from_vector_db]
    
    agent = create_agent(llm, all_tools, system_prompt=system_prompt)
    
    end_time = time.time()
    initialization_time = end_time - start_time
    print(f"Agent initialized in {initialization_time:.2f} seconds.")


def invoke_agent(user_input: str, user_id: str) -> str:
    """
    Process user input through the agent.
    
    Args:
        user_input: User query or extracted text from documents/images
        
    Returns:
        Comprehensive plain text output
    """


    if agent == None:
        initialize_agent(user_id=user_id)

    response = agent.invoke({
        "messages": [
            {"role": "user", "content": str(user_input)}
        ]
    })
    
    return response['messages'][-1].content[0]['text']