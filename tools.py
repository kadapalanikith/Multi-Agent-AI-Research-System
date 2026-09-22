"""
=============================================================================
Multi-Agent AI Research System - External Tools & Integration Layer
=============================================================================
This module defines the callable tools that autonomous agents utilize to
interact with the external world (the Web).

In LangChain, a "Tool" is an abstraction that wraps an external capability
(such as an API endpoint, a web scraper, or a database query) with a schema
that an LLM (Large Language Model) can interpret and invoke autonomously.

Tools Defined:
1. web_search: Queries the Tavily Search API for real-time web references.
2. scrape_url: Fetches and sanitizes raw HTML into readable plain text.
=============================================================================
"""

import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.tools import tool
from tavily import TavilyClient

# -----------------------------------------------------------------------------
# Configuration & Initialization
# -----------------------------------------------------------------------------
# Load environment variables from the local .env file (e.g. TAVILY_API_KEY, MISTRAL_API_KEY)
load_dotenv()

# Initialize the Tavily API client using the API key stored in the environment.
# Tavily is a search engine purpose-built for AI agents and LLMs, providing
# clean, structured, and factual search results optimized for agent consumption.
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


# -----------------------------------------------------------------------------
# Tool 1: Web Search via Tavily
# -----------------------------------------------------------------------------
@tool
def web_search(query: str) -> str:
    """Search the web for recent and reliable information on a topic. Returns Titles, URLs, and snippets."""
    # The docstring above is critical: LangChain agents read this description to decide
    # when and how to call this tool during autonomous decision-making.
    try:
        # Perform search query via Tavily, retrieving up to 5 authoritative results
        results = tavily_client.search(query=query, max_results=5)
        out = []
        
        # Iterate over the retrieved search hits and format them into structured text blocks
        for r in results.get("results", []):
            out.append(
                f"Title: {r.get('title', 'N/A')}\n"
                f"URL: {r.get('url', 'N/A')}\n"
                f"Snippet: {r.get('content', '')[:300]}\n"  # Truncate snippet to 300 chars to conserve LLM context window
            )
        
        # Join formatted source blocks with a clear separator delimiter '----'
        return "\n----\n".join(out) if out else "No results found."
    except Exception as e:
        # Gracefully capture and return exceptions as text so the agent can self-correct or notify
        return f"Search failed: {str(e)}"


# -----------------------------------------------------------------------------
# Tool 2: Deep Web Scraper via BeautifulSoup
# -----------------------------------------------------------------------------
@tool
def scrape_url(url: str) -> str:
    """Scrape and return clean text content from a given URL for deeper reading."""
    # This tool enables the Reader Agent to visit specific links found by the Search Agent
    # and extract in-depth article body text rather than relying solely on short snippets.
    try:
        # Send an HTTP GET request to the target webpage.
        # - timeout=8: Prevents hanging if the target server is slow or unresponsive.
        # - User-Agent: Identifies request as a standard browser to avoid automated bot blockers.
        resp = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()  # Raises an HTTPError if the response was an error code (4xx or 5xx)
        
        # Parse the raw HTML markup into a BeautifulSoup DOM tree
        soup = BeautifulSoup(resp.text, "html.parser")
        
        # Sanitize HTML: Decompose and remove irrelevant structural and aesthetic elements
        # (scripts, stylesheet links, navigation headers, and footer menus) to isolate main content
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
            
        # Extract text separated by single spaces, stripping extra whitespace,
        # and limit content to 3,000 characters to fit efficiently into LLM token budgets.
        return soup.get_text(separator=" ", strip=True)[:3000]
    except Exception as e:
        # Return error diagnostic as string if fetching or parsing fails
        return f"Could not scrape URL: {str(e)}"