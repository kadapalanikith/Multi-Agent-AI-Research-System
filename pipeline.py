"""
=============================================================================
Multi-Agent AI Research System - Pipeline Orchestration Engine
=============================================================================
This module orchestrates the sequential execution flow of the autonomous
multi-agent research team:

Orchestration Workflow:
  1. Search Agent:
     Finds authoritative web articles and sources using the Tavily Search API.
  2. Reader Agent:
     Selects the top URL and extracts deep semantic text content via BeautifulSoup.
  3. Writer Chain:
     Synthesizes gathered search snippets and scraped text into a structured report.
  4. Critic Chain:
     Conducts an objective peer-review, scoring the report and suggesting improvements.

Shared State Dictionary:
The pipeline maintains a shared state dictionary (`state = {}`) that preserves
intermediate outputs (search results, scraped content, draft report, critic feedback)
as data transitions from one cognitive stage to the next.
=============================================================================
"""

from agents import build_search_agent, build_reader_agent, writer_chain, critic_chain


def run_research_pipeline(topic: str) -> dict:
    """
    Executes the end-to-end multi-agent research pipeline for a given topic.

    Parameters:
        topic (str): The research subject or question to investigate.

    Returns:
        dict: The final pipeline state containing:
              - 'search_result': Extracted search snippets and URLs from Tavily
              - 'scraped_content': In-depth plain text scraped from the chosen source
              - 'report': The synthesized Markdown research report from Mistral AI
              - 'feedback': The peer-review critique and score from the Critic chain
    """
    # Initialize the blackboard / state dictionary to store results from each stage
    state = {}
    
    # -------------------------------------------------------------------------
    # Step 1: Search Agent Execution
    # -------------------------------------------------------------------------
    # The Search Agent queries Tavily to retrieve top authoritative web results,
    # titles, URLs, and summaries relevant to the requested topic.
    print("\n"+" ="*50)
    print("step 1: Search Agent is working...")
    print("\n"+" ="*50)
    
    # Build a fresh instance of the search agent
    search_agent = build_search_agent()
    
    # Invoke the agent with a user prompt guiding it to discover reliable information
    search_result = search_agent.invoke({
        "messages": [("user", f"Find recent,reliable and detailed information about topic: {topic}")]
    })
    
    # In LangChain agents, the conversation history is returned as a list of 'messages'.
    # The last message represents the agent's concluding synthesis of the search tool output.
    last_search_msg = search_result['messages'][-1]
    state["search_result"] = getattr(last_search_msg, 'content', str(last_search_msg))
    
    print("\n search result: \n", state["search_result"])
    
    # -------------------------------------------------------------------------
    # Step 2: Reader Agent Execution
    # -------------------------------------------------------------------------
    # The Reader Agent reviews the search findings, selects the most relevant URL,
    # and autonomously invokes the `scrape_url` tool to read the webpage in depth.
    print("\n"+" ="*50)
    print("step 2: Reader Agent is working...")
    print("\n"+" ="*50)
    
    # Build a fresh instance of the reader agent equipped with web scraping tools
    reader_agent = build_reader_agent()
    
    # Provide the initial search results (truncated to 800 chars for prompt compactness)
    # and direct the reader agent to scrape the best link.
    reader_result = reader_agent.invoke({
        "messages": [("user",
            f"Based on the following search results about '{topic}', "
            f"pick the most relevant URL and scrape it for deeper content.\n\n"
            f"Search Results:\n{state['search_result'][:800]}"
        )]
    })
    
    # Extract the agent's final message containing the sanitized scraped content
    last_reader_msg = reader_result['messages'][-1]
    state["scraped_content"] = getattr(last_reader_msg, 'content', str(last_reader_msg))  
    
    print("\n scraped content: \n", state["scraped_content"])
    
    # -------------------------------------------------------------------------
    # Step 3: Writer Chain Execution
    # -------------------------------------------------------------------------
    # The Writer Chain takes both the broad search snippets and deep scraped text,
    # synthesizing them into a cohesive, publication-quality research report.
    print("\n"+" ="*50)
    print("step 3: Writer Chain is working...")
    print("\n"+" ="*50)
    
    # Combine the multi-source evidence into a single comprehensive research context
    research_combined = (
        f"Search Results:\n{state['search_result']}\n\n"
        f"Scraped Content:\n{state['scraped_content']}" 
    ) 
    
    # Invoke the deterministic writer chain (Prompt Template -> Mistral LLM -> Parser)
    state["report"] = writer_chain.invoke({
        "topic": topic,
        "research": research_combined
    })
    
    print("\n Final report: \n", state["report"])
    
    # -------------------------------------------------------------------------
    # Step 4: Critic Chain Execution
    # -------------------------------------------------------------------------
    # The Critic Chain acts as an independent peer reviewer, evaluating the
    # drafted report for clarity, factual depth, structure, and citation integrity.
    print("\n"+" ="*50)
    print("step 4: Critic Chain is working...")
    print("\n"+" ="*50)
    
    # Invoke the critic chain to obtain strict constructive feedback and a 10-point score
    state["feedback"] = critic_chain.invoke({
        "report": state["report"]
    })
     
    print("\n Critic feedback: \n", state["feedback"])
    
    # Return the aggregated state containing all artifacts produced across the pipeline
    return state


# -----------------------------------------------------------------------------
# CLI Entry Point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Prompt the user for a research query via the command-line interface
    topic = input("\n Enter a research topic: ")
    run_research_pipeline(topic)