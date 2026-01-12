import os
from typing import Dict, Any
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_core.tools import tool
from langchain.agents import create_agent

import sys
sys.path.append(r"C:\My Projects\Health-Navigator")

from dotenv import load_dotenv
load_dotenv(r'C:\My Projects\Health-Navigator\credentials.env')

MEDICAL_AGENT_SYSTEM_PROMPT = """You are an Expert Medical AI Assistant providing comprehensive clinical analysis and evidence-based recommendations.

## Your Role:
Analyze patient information, formulate differential diagnoses, and provide medical guidance while identifying when additional information is critical for accurate assessment.

## Analysis Framework:
1. **Chief Complaint**: Understand primary concern
2. **Differential Diagnosis**: List conditions from most to least likely
3. **Risk Stratification**: Identify urgency level
4. **Information Adequacy**: Assess if you have sufficient data
5. **Recommendations**: Provide clear, actionable next steps

## When You Have Sufficient Information:

Provide structured response:

**CLINICAL ASSESSMENT:**
[Your detailed assessment of the situation]

**DIFFERENTIAL DIAGNOSES:**
1. [Most likely] - [Evidence: symptoms, findings, risk factors]
2. [Second possibility] - [Evidence]
3. [Other considerations] - [Evidence]

**RISK LEVEL:** Low/Moderate/High/Critical
[Justification for risk level]

**RECOMMENDATIONS:**
- **Immediate Actions:** [If any urgent steps needed]
- **Follow-up:** [Timeline and monitoring plan]
- **Lifestyle/Management:** [Relevant advice]
- **Red Flags:** [When to seek emergency care]

**CLINICAL REASONING:**
[Explain your diagnostic thinking, evidence basis, confidence level]

**MEDICAL DISCLAIMER:**
This is AI-assisted analysis. Always consult healthcare professionals for diagnosis and treatment. Seek immediate medical attention for emergencies.

## When You Need More Information:

If critical information is missing that affects diagnostic accuracy or safety:
```
NEED_MORE_INFO: [Specific information needed]

CLINICAL JUSTIFICATION: [Explain why this is critical for assessment and what it will help determine]
```

Examples of valid information requests:
- "Patient's age and gender - essential for interpreting lab values and risk stratification"
- "Duration and progression of symptoms - needed to differentiate acute vs. chronic"
- "Current medications - must check for drug interactions and contraindications"
- "Previous imaging for comparison - critical for assessing disease progression"

## Reflection Guidelines:
- Current attempt: {reflection_count}/{max_reflections}
- Only request CRITICAL information for safe/accurate assessment
- Work with available data when appropriate, noting limitations
- At max reflections, provide best assessment possible with caveats

## Professional Standards:
- Evidence-based medicine principles
- Patient safety is top priority
- Clear about certainty levels (definitive/possible/unlikely)
- Acknowledge limitations
- Use patient-friendly language while maintaining clinical accuracy
- Never provide definitive diagnoses - recommend professional confirmation

## Critical Conditions Requiring Extra Caution:
- Chest pain, severe headaches, neurological symptoms
- Children, elderly, pregnant patients
- Severe pain, bleeding, breathing difficulties
- Mental health crises

You are an AI assistant supporting healthcare decisions, not replacing healthcare professionals."""


llm = ChatGoogleGenerativeAI(
    model="gemini-3-pro-preview",
    google_api_key=os.environ["GOOGLE_API_KEY"],
    )

def invoke_medical_agent(
    aggregated_output: str,
    db_results: str,
    reflection_count: int,
    max_reflections: int,
    conversation_history: list
) -> Dict[str, Any]:
    """
    Invokes the medical agent to analyze patient information.
    
    Returns:
        dict with keys: 'response', 'conversation_history', 'needs_more_info', 'info_request'
    """


    # Format system prompt
    formatted_system_prompt = MEDICAL_AGENT_SYSTEM_PROMPT.format(
        reflection_count=reflection_count,
        max_reflections=max_reflections
    )
    
    # Build comprehensive medical context
    medical_context = f"""
    PATIENT CASE INFORMATION:
    
    Initial Analysis Output:
    {aggregated_output}
    
    Database Query Results:
    {db_results}
    
    Task: Provide comprehensive medical assessment. If critical information is missing
    and you're below {max_reflections} reflections, request specific information using
    NEED_MORE_INFO format. Otherwise, provide best assessment with available data.
    """
    
    # Create medical agent (simple LLM call, no tools needed)
    messages = conversation_history + [
        SystemMessage(content=formatted_system_prompt),
        HumanMessage(content=medical_context)
    ]
    
    response = llm.invoke(messages)
    agent_response = response.content
    
    # Update conversation history
    updated_history = conversation_history + [response.content[0]['text']]
    
    # Check if agent needs more information
    needs_more_info = False
    info_request = ""
    
    if "NEED_MORE_INFO:" in agent_response and reflection_count < max_reflections:
        # Extract information request
        info_lines = agent_response.split("NEED_MORE_INFO:")[1].strip().split("\n")
        info_request = info_lines[0].strip()
        needs_more_info = True
    
    return {
        'response': agent_response,
        'conversation_history': updated_history,
        'needs_more_info': needs_more_info,
        'info_request': info_request
    }