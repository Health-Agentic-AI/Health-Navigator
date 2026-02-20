import os
import time
from typing import Any, Dict, List
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv
from langchain_core.tools import tool

import sys
sys.path.append(r"C:\My Projects\Health-Navigator")

from app.workflow.llm_provider import create_langchain_chat_model


load_dotenv(os.path.join(os.getcwd(), "app", "credentials.env"))

TAVILY_SEARCH_URL = "https://api.tavily.com/search"
HIGH_AUTHORITY_MEDICAL_DOMAINS = [
    "nih.gov",
    "cdc.gov",
    "who.int",
    "mayoclinic.org",
    "nejm.org",
    "thelancet.com",
    "bmj.com",
    "jamanetwork.com",
    "medlineplus.gov",
    "fda.gov",
]

def _get_research_llm():
    return create_langchain_chat_model(
        agent_name="Medical Research Summarizer",
        google_model="gemini-3-flash-preview",
    )


def _extract_domain(url: str) -> str:
    try:
        parsed = urlparse(url)
        return (parsed.netloc or "").replace("www.", "")
    except Exception:
        return ""


def _tavily_search(query: str, max_results: int) -> List[Dict[str, Any]]:
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY is required for medical_fact_check_tool.")

    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": "advanced",
        "max_results": max(1, min(int(max_results), 8)),
        "include_domains": HIGH_AUTHORITY_MEDICAL_DOMAINS,
        "include_raw_content": False,
        "include_answer": False,
    }

    response = requests.post(TAVILY_SEARCH_URL, json=payload, timeout=45)
    response.raise_for_status()
    data = response.json()
    return data.get("results", []) or []


def run_medical_research(query: str, max_results: int = 5) -> str:
    try:
        raw_results = _tavily_search(query, max_results)
    except Exception as exc:
        return (
            "Research lookup failed. Continue with internal medical reasoning and clearly "
            f"state uncertainty. Error details: {exc}"
        )

    if not raw_results:
        return (
            "No high-authority medical web results were found for this query. "
            "Continue with internal medical reasoning and clearly state uncertainty."
        )

    compact_results: List[Dict[str, str]] = []
    for item in raw_results:
        compact_results.append(
            {
                "title": str(item.get("title", "")).strip(),
                "domain": _extract_domain(str(item.get("url", "")).strip()),
                "published_date": str(
                    item.get("published_date") or item.get("date") or "Unknown"
                ).strip(),
                "content": str(item.get("content", "")).strip(),
            }
        )

    summarizer_prompt = f"""
You are summarizing trusted medical web findings for another medical AI agent.

Rules:
1) Keep only clinically relevant facts that help verify or correct a medical claim.
2) Return concise output with NO URLs.
3) For each source mentioned, include domain and published date when available.
4) If sources conflict, call out the conflict and say what is most reliable.
5) If evidence is weak or incomplete, explicitly say so.
6) Do not invent facts.

Current date/time: {time.strftime("%Y-%m-%d %H:%M:%S")}

User's medical fact-check query:
{query}

Search findings:
{compact_results}

Output format:
MEDICAL_FACT_CHECK_SUMMARY:
- [fact]
- [fact]

SOURCE_TIMELINE:
- [domain] | Published: [date] | [short relevance note]
- [domain] | Published: [date] | [short relevance note]

CONFIDENCE:
[High/Moderate/Low] - [1 sentence reason]
""".strip()

    print(
        "DEBUG: summarize_results_node invoking research LLM "
        f"(query='{query[:120]}...', sources={len(compact_results)})"
    )
    llm = _get_research_llm()
    response = llm.invoke([("human", summarizer_prompt)])

    if isinstance(response.content, list):
        return str(response.content[0].get("text", response.content))
    return str(response.content)


@tool
def medical_fact_check_tool(query: str, max_results: int = 5) -> str:
    """
    Search trusted medical web sources and return a concise fact-check summary.

    Use this tool when medical facts are uncertain, subtle, or potentially outdated.
    It returns summarized evidence with source domains and publish dates only (no URLs).

    Args:
        query (str): The medical claim or question to fact-check.
        max_results (int): Maximum number of sources to retrieve. Defaults to 5.

    Returns:
        str: A concise fact-check summary with evidence drawn from trusted medical
             sources, including source domains and publish dates.
    """
    print(
        "DEBUG: medical_fact_check_tool invoked "
        f"(query='{query[:160]}...', max_results={max_results})"
    )
    return run_medical_research(query=query, max_results=max_results)
