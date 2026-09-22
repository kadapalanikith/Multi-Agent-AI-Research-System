"""
=============================================================================
Multi-Agent AI Research System - Agent & Chain Definitions
=============================================================================
This module defines the core cognitive workers (agents and chains) that power
the research lifecycle:

1. Search Agent:
   An autonomous agent equipped with Tavily Web Search to discover pertinent
   articles, papers, and URLs for a designated topic.
   
2. Reader Agent:
   An autonomous agent equipped with BeautifulSoup web scraping to drill down
   into the most relevant URL and extract rich content.
   
3. Writer Chain:
   A deterministic LangChain Expression Language (LCEL) chain that synthesizes
   the gathered research into a structured, publication-quality report.
   
4. Critic Chain:
   An evaluator chain that performs peer-review on the generated report,
   grading it on a 10-point scale with actionable feedback and strengths.
=============================================================================
"""

import os
from langchain.agents import create_agent
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools import web_search, scrape_url
from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# Environment & Model Initialization
# -----------------------------------------------------------------------------
# Load API credentials from the local .env configuration file
load_dotenv()

# Select the Mistral model to use (defaulting to "ministral-8b-latest")
# This can be customized via the MISTRAL_MODEL environment variable in .env
model_name = os.getenv("MISTRAL_MODEL", "ministral-8b-latest")

# Initialize the Mistral AI Chat LLM wrapper.
# - temperature=0: Ensures deterministic, factual, and reproducible outputs,
#   minimizing hallucinations which is vital for academic and empirical research.
llm = ChatMistralAI(model=model_name, temperature=0)


# -----------------------------------------------------------------------------
# Agent 1: Search Agent Factory
# -----------------------------------------------------------------------------
# An autonomous agent is an LLM with access to tools that can make dynamic decisions
# on how many times to invoke a tool and how to interpret the results.
def build_search_agent():
    """
    Constructs and returns the Search Agent.
    Binds the ChatMistralAI model with the web_search tool (Tavily Search API).
    """
    return create_agent(
        model=llm,
        tools=[web_search]  # The agent can autonomously call the Tavily search tool
    )


# -----------------------------------------------------------------------------
# Agent 2: Reader Agent Factory
# -----------------------------------------------------------------------------
def build_reader_agent():
    """
    Constructs and returns the Reader Agent.
    Binds the ChatMistralAI model with the scrape_url tool (BeautifulSoup scraper).
    """
    return create_agent(
        model=llm,
        tools=[scrape_url]  # The agent can autonomously fetch and sanitize webpage DOM
    )


# -----------------------------------------------------------------------------
# Chain 3: Research Report Writer Chain
# -----------------------------------------------------------------------------
# Unlike autonomous agents with open-ended tool loops, a Chain is a predictable,
# direct pipeline: Prompt Template -> Large Language Model -> Output Parser.

# Define the prompt template guiding the LLM in synthesizing research notes
writer_prompt = ChatPromptTemplate.from_messages([
    # System role sets the persona, instructions, and standard of quality
    ("system", "You are an expert research writer. Write clear, structured and insightful reports."),
    # Human role passes dynamic input parameters: {topic} and {research}
    ("human", """Write a detailed research report on the topic below.

Topic: {topic}

Research Gathered:
{research}

Structure the report as:
- Introduction
- Key Findings (minimum 3 well-explained points)
- Conclusion
- Sources (list all URLs found in the research)
 
Be detailed, factual and professional."""),
])

# LangChain Expression Language (LCEL) Pipe Syntax:
# - writer_prompt formats the user inputs into structured messages.
# - llm receives the formatted messages and invokes Mistral AI.
# - StrOutputParser() converts the resulting AIMessage response into a plain Python string.
writer_chain = writer_prompt | llm | StrOutputParser()


# -----------------------------------------------------------------------------
# Chain 4: Research Report Critic Chain
# -----------------------------------------------------------------------------
# The Critic acts as an objective quality assurance reviewer, assessing the
# drafted report against clarity, factual grounding, and structure.

critic_prompt = ChatPromptTemplate.from_messages([
    # System role enforces strict objectivity and constructive critique
    ("system", "You are a sharp and constructive research critic. Be honest and specific."),
    # Human role provides the drafted report and instructs the exact output schema
    ("human", """Review the research report below and evaluate it strictly.

Report:
{report}

Respond in this exact format:

Score: X/10

Strengths:
- ...
- ...

Areas to Improve:
- ...
- ...

One line verdict:
..."""),
])

# Assemble the Critic Chain via LCEL pipe syntax
critic_chain = critic_prompt | llm | StrOutputParser()