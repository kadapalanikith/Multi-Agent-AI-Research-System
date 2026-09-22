"""
=============================================================================
Multi-Agent AI Research System - Web Studio Application Server
=============================================================================
This module provides a lightweight, zero-dependency HTTP server powered by
Python's built-in `http.server.ThreadingHTTPServer`. It supports real-time
Server-Sent Events (SSE) streaming for multi-agent execution telemetry,
research report rendering, and live pipeline orchestration.

Key Capabilities:
1. Zero external server dependencies - runs out-of-the-box on standard Python 3.
2. Serves the AI-Native frontend interface at GET /
3. Streams live agent telemetry step-by-step via GET /api/stream (SSE)
4. Validates environment configurations and API health via GET /api/status
5. Supports both live LangGraph/Mistral agents and interactive demo simulation.
=============================================================================
"""

import os
import sys
import json
import time
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from http.server import HTTPServer, ThreadingHTTPServer, BaseHTTPRequestHandler
from dotenv import load_dotenv

# Load environment variables (.env)
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"


def check_api_status() -> dict:
    """
    Examines the active environment to verify whether API credentials
    for Mistral AI and Tavily Search are present and non-placeholder.
    """
    tavily_key = os.getenv("TAVILY_API_KEY", "").strip()
    mistral_key = os.getenv("MISTRAL_API_KEY", "").strip()
    
    tavily_valid = bool(tavily_key and "your_tavily" not in tavily_key.lower())
    mistral_valid = bool(mistral_key and "your_mistral" not in mistral_key.lower())
    
    return {
        "tavily_ready": tavily_valid,
        "mistral_ready": mistral_valid,
        "model": os.getenv("MISTRAL_MODEL", "ministral-8b-latest")
    }


class ResearchStudioHandler(BaseHTTPRequestHandler):
    """
    HTTP Request Handler serving the Web Studio SPA and SSE telemetry stream.
    """

    def do_GET(self):
        parsed_url = urlparse(self.path)
        path = parsed_url.path
        query_params = parse_qs(parsed_url.query)

        # ---------------------------------------------------------------------
        # Route 1: Primary Homepage (GET /)
        # ---------------------------------------------------------------------
        if path in ("/", "/index.html"):
            index_file = TEMPLATES_DIR / "index.html"
            if not index_file.exists():
                self.send_error(500, "templates/index.html not found.")
                return
            
            content = index_file.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            return

        # ---------------------------------------------------------------------
        # Route 2: API Connection Status (GET /api/status)
        # ---------------------------------------------------------------------
        if path == "/api/status":
            status_data = check_api_status()
            payload = json.dumps(status_data).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(payload)
            return

        # ---------------------------------------------------------------------
        # Route 3: Real-Time Telemetry Stream (GET /api/stream?topic=...&demo=...)
        # ---------------------------------------------------------------------
        if path == "/api/stream":
            topic = query_params.get("topic", [""])[0].strip()
            if not topic:
                self.send_error(400, "Missing required query parameter: 'topic'")
                return
            
            use_demo = query_params.get("demo", ["false"])[0].lower() in ["true", "1", "yes"]

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            self.handle_sse_stream(topic, use_demo)
            return

        # Fallback 404
        self.send_error(404, "Endpoint not found.")

    def send_sse(self, data_dict: dict, event: str = None) -> bool:
        """
        Helper method to format and flush a single Server-Sent Event to the client.
        """
        try:
            if event:
                msg = f"event: {event}\ndata: {json.dumps(data_dict)}\n\n"
            else:
                msg = f"data: {json.dumps(data_dict)}\n\n"
            self.wfile.write(msg.encode("utf-8"))
            self.wfile.flush()
            return True
        except (BrokenPipeError, ConnectionResetError):
            return False

    def handle_sse_stream(self, topic: str, use_demo: bool):
        """
        Orchestrates either live agent execution or high-fidelity simulation
        while streaming telemetry packets to the frontend.
        """
        status = check_api_status()
        keys_ready = status["tavily_ready"] and status["mistral_ready"]

        # Run demo simulation if toggled OR if credentials are missing
        if use_demo or not keys_ready:
            self.run_simulated_pipeline(topic)
            return

        # Run live agent pipeline
        try:
            from agents import build_search_agent, build_reader_agent, writer_chain, critic_chain

            # Step 1: Search Agent
            self.send_sse({
                "step": 1,
                "agent": "Search Agent",
                "message": f"Initializing Tavily Web Search for topic: \"{topic}\"",
                "level": "info"
            })
            time.sleep(0.3)

            search_agent = build_search_agent()
            search_result = search_agent.invoke({
                "messages": [("user", f"Find recent, reliable and detailed information about topic: {topic}")]
            })
            last_search_msg = search_result['messages'][-1]
            search_content = getattr(last_search_msg, 'content', str(last_search_msg))

            # Parse discovered snippets
            parsed_sources = []
            for block in search_content.split("----"):
                if "URL:" in block:
                    title = "Web Reference"
                    url = "#"
                    snippet = block.strip()
                    for line in block.strip().split("\n"):
                        if line.startswith("Title:"):
                            title = line.replace("Title:", "").strip()
                        elif line.startswith("URL:"):
                            url = line.replace("URL:", "").strip()
                        elif line.startswith("Snippet:"):
                            snippet = line.replace("Snippet:", "").strip()
                    parsed_sources.append({"title": title, "url": url, "snippet": snippet})

            self.send_sse({
                "step": 1,
                "agent": "Search Agent",
                "message": f"Discovered {len(parsed_sources)} verified sources via Tavily API.",
                "level": "success",
                "metric": f"{len(parsed_sources)} sources",
                "payload": {"sources": parsed_sources}
            })
            time.sleep(0.4)

            # Step 2: Reader Agent
            self.send_sse({
                "step": 2,
                "agent": "Reader Agent",
                "message": "Selecting most authoritative URL for deep scraping & HTML sanitization...",
                "level": "info"
            })
            time.sleep(0.3)

            reader_agent = build_reader_agent()
            reader_result = reader_agent.invoke({
                "messages": [("user",
                    f"Based on the following search results about '{topic}', "
                    f"pick the most relevant URL and scrape it for deeper content.\n\n"
                    f"Search Results:\n{search_content[:800]}"
                )]
            })
            last_reader_msg = reader_result['messages'][-1]
            scraped_content = getattr(last_reader_msg, 'content', str(last_reader_msg))

            self.send_sse({
                "step": 2,
                "agent": "Reader Agent",
                "message": f"Extracted clean text content ({len(scraped_content)} characters).",
                "level": "success",
                "metric": f"{len(scraped_content)} chars"
            })
            time.sleep(0.4)

            # Step 3: Writer Chain
            self.send_sse({
                "step": 3,
                "agent": "Writer Chain",
                "message": "Synthesizing research into structured report using Mistral LLM...",
                "level": "info"
            })
            time.sleep(0.3)

            research_combined = f"Search Results:\n{search_content}\n\nScraped Content:\n{scraped_content}"
            report_content = writer_chain.invoke({
                "topic": topic,
                "research": research_combined
            })
            word_count = len(report_content.split())

            self.send_sse({
                "step": 3,
                "agent": "Writer Chain",
                "message": f"Drafted comprehensive research report ({word_count} words).",
                "level": "success",
                "metric": f"{word_count} words",
                "payload": {"partial_report": report_content}
            })
            time.sleep(0.4)

            # Step 4: Critic Chain
            self.send_sse({
                "step": 4,
                "agent": "Critic Chain",
                "message": "Evaluating report factual accuracy and depth against 10-point rubric...",
                "level": "info"
            })
            time.sleep(0.3)

            feedback_content = critic_chain.invoke({
                "report": report_content
            })

            self.send_sse({
                "step": 4,
                "agent": "Critic Chain",
                "message": "Peer review assessment finalized.",
                "level": "success",
                "payload": {"feedback": feedback_content}
            })
            time.sleep(0.4)

            # Final completion packet
            final_state = {
                "search_result": search_content,
                "scraped_content": scraped_content,
                "report": report_content,
                "feedback": feedback_content
            }
            self.send_sse(final_state, event="complete")

        except Exception as exc:
            err_msg = f"Pipeline execution error: {str(exc)}"
            self.send_sse({"step": 0, "agent": "System", "message": err_msg, "level": "error"})

    def run_simulated_pipeline(self, topic: str):
        """
        Simulates the 4-step multi-agent pipeline with realistic research content.
        """
        # Step 1: Search Simulation
        self.send_sse({
            "step": 1,
            "agent": "Search Agent",
            "message": f"[Demo] Querying Tavily Web Index for \"{topic}\"...",
            "level": "info"
        })
        time.sleep(1.0)

        slug = topic.lower().replace(" ", "-")
        mock_sources = [
            {
                "title": f"Advancements & Breakthroughs in {topic} (2026)",
                "url": f"https://www.nature.com/articles/{slug}-2026",
                "snippet": f"State-of-the-art analysis demonstrating exponential growth and new architectural benchmarks in {topic}."
            },
            {
                "title": f"Industrial Applications and Strategic Roadmap for {topic}",
                "url": f"https://arxiv.org/abs/2601.{slug[:4]}01",
                "snippet": f"A comprehensive empirical survey on how enterprise systems and academic labs are operationalizing {topic}."
            },
            {
                "title": f"Global Technology Review: {topic} Executive Briefing",
                "url": f"https://technologyreview.com/hub/{slug}",
                "snippet": f"Critical insights and key risk vectors discovered in recent testing deployments of {topic}."
            }
        ]

        self.send_sse({
            "step": 1,
            "agent": "Search Agent",
            "message": f"[Demo] Tavily retrieved 3 high-confidence sources.",
            "level": "success",
            "metric": "3 sources",
            "payload": {"sources": mock_sources}
        })
        time.sleep(1.0)

        # Step 2: Reader Simulation
        top_url = mock_sources[0]["url"]
        self.send_sse({
            "step": 2,
            "agent": "Reader Agent",
            "message": f"[Demo] Selecting top reference: {top_url}",
            "level": "info"
        })
        time.sleep(0.7)
        self.send_sse({
            "step": 2,
            "agent": "Reader Agent",
            "message": "[Demo] Scraping HTML, stripping ads/scripts, extracting clean semantic text...",
            "level": "info"
        })
        time.sleep(0.9)
        self.send_sse({
            "step": 2,
            "agent": "Reader Agent",
            "message": "[Demo] Extracted 2,840 characters of rich literature.",
            "level": "success",
            "metric": "2,840 chars"
        })
        time.sleep(1.0)

        # Step 3: Writer Simulation
        self.send_sse({
            "step": 3,
            "agent": "Writer Chain",
            "message": "[Demo] Initializing Mistral-8B synthesis chain...",
            "level": "info"
        })
        time.sleep(1.2)

        mock_report = f"""# Research Report: {topic}

## Introduction
The rapid evolution of **{topic}** has marked a transformative milestone in modern science and computation. Recent developments highlight a shift from speculative experimentation to resilient, real-world deployment across multidisciplinary sectors.

## Key Findings

### 1. Foundational Architecture & Paradigm Evolution
Recent empirical data demonstrates significant efficiency gains, driven by novel algorithmic frameworks and hardware co-design. Benchmarks indicate up to a 4.2x speedup in throughput while maintaining rigorous reliability guarantees.

### 2. Autonomous Multi-Agent Coordination
Integration with distributed cognitive workflows allows complex tasks to be decomposed into autonomous sub-routines. Specialized agents operate collaboratively—separating retrieval, contextual validation, synthesis, and verification.

### 3. Safety, Scalability, and Industry Adoption
Enterprise organizations are increasingly standardizing on verified safety guardrails and reproducible pipelines. Key bottlenecks in latency and context preservation have been systematically resolved with adaptive caching and selective retrieval.

## Conclusion
The future trajectory of **{topic}** points toward ubiquitous integration within intelligence platforms. Continuous verification protocols and structured peer reviews will remain paramount to ensuring high-fidelity outcomes.

## Sources
- [Nature Journal: Advancements in {topic}]({mock_sources[0]['url']})
- [arXiv Research Repository: Empirical Frameworks]({mock_sources[1]['url']})
- [MIT Technology Review: Global Briefing]({mock_sources[2]['url']})
"""

        self.send_sse({
            "step": 3,
            "agent": "Writer Chain",
            "message": "[Demo] Research report draft generated (482 words).",
            "level": "success",
            "metric": "482 words",
            "payload": {"partial_report": mock_report}
        })
        time.sleep(1.0)

        # Step 4: Critic Simulation
        self.send_sse({
            "step": 4,
            "agent": "Critic Chain",
            "message": "[Demo] Critic agent reviewing draft against clarity, structure, and citations...",
            "level": "info"
        })
        time.sleep(1.0)

        mock_feedback = """Score: 9.2/10

Strengths:
- Clear and logical multi-tier structural progression from introduction to verified sources.
- Thoroughly addresses architectural paradigms and autonomous coordination with concrete metrics.
- High citation integrity with direct links to academic and industrial publications.

Areas to Improve:
- Consider incorporating quantitative ablation comparisons for alternative implementations.
- Expand on potential edge-case regulatory constraints in subsequent revisions.

One line verdict:
An exceptionally articulated, rigorous, and actionable research report."""

        self.send_sse({
            "step": 4,
            "agent": "Critic Chain",
            "message": "[Demo] Peer review complete. Score: 9.2/10.",
            "level": "success",
            "payload": {"feedback": mock_feedback}
        })
        time.sleep(0.5)

        final_state = {
            "search_result": "Simulated Tavily search payload.",
            "scraped_content": "Simulated deep scraped text payload.",
            "report": mock_report,
            "feedback": mock_feedback
        }
        self.send_sse(final_state, event="complete")

    def log_message(self, format, *args):
        # Clean terminal logging without cluttering the screen with every poll
        if "GET /api/stream" in args[0] or "GET / " in args[0]:
            print(f"[Studio Web Server] {self.address_string()} - {args[0]}")


def run_server(host: str = "127.0.0.1", port: int = 8000):
    server = ThreadingHTTPServer((host, port), ResearchStudioHandler)
    print("\n" + "=" * 65)
    print(" [OK] Multi-Agent AI Research System - Web Studio Launching")
    print(" =" * 65)
    print(f" Server running at: http://{host}:{port}")
    print(f" Interactive UI:     Open http://{host}:{port} in your browser")
    print(" Zero-dependency:   Powered by Python standard library (http.server)")
    print("=" * 65 + "\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n Shutting down Web Studio server cleanly.")
        server.server_close()


if __name__ == "__main__":
    run_server()
