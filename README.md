# 🧠 Multi-Agent AI Research System

An autonomous, multi-agent research intelligence studio powered by **LangGraph**, **LangChain**, **Mistral AI**, and **Tavily Deep Search**.

Equipped with an **award-winning AI-Native Web Studio** featuring real-time Server-Sent Events (SSE) telemetry, interactive agent orchestration graph, bento-grid report reader, and critic peer-review gauge.

---

## 🚀 Architecture Overview

The system decomposes complex research tasks into specialized, coordinated cognitive agents and chains:

```mermaid
flowchart LR
    User([Research Query]) --> SearchAgent[1. Search Agent\nTavily API]
    SearchAgent -->|Top Sources & URLs| ReaderAgent[2. Reader Agent\nBeautifulSoup Deep Scraper]
    ReaderAgent -->|Sanitized Text Content| WriterChain[3. Writer Chain\nMistral-8B LLM]
    WriterChain -->|Structured Report Draft| CriticChain[4. Critic Chain\nPeer-Review & Scoring]
    CriticChain --> Dashboard([Award-Winning\nWeb Studio Dashboard])

    classDef agent fill:#7C3AED,stroke:#DDD6FE,stroke-width:2px,color:#FFFFFF;
    classDef io fill:#0891B2,stroke:#A5F3FC,stroke-width:2px,color:#FFFFFF;
    class SearchAgent,ReaderAgent,WriterChain,CriticChain agent;
    class User,Dashboard io;
```

### Cognitive Team Breakdown:
1. **Search Agent**: Queries Tavily Search API to discover authoritative, real-time sources, snippet previews, and citation URLs.
2. **Reader Agent**: Autonomously inspects search results, chooses the highest-relevance link, and performs deep HTML sanitization and text extraction via BeautifulSoup.
3. **Writer Chain**: Synthesizes multi-source evidence into a publication-quality Markdown report with introduction, structured key findings, conclusion, and verified sources.
4. **Critic Chain**: Evaluates the drafted report against a rigorous 10-point rubric, assessing clarity, factual grounding, and citation integrity with actionable feedback.

---

## ✨ Features

- 🖥️ **Award-Winning Web Studio UI**: AI-Native dark/light responsive interface built with Tailwind CSS, Lucide SVG icons, Space Grotesk, and DM Sans typography.
- ⚡ **Real-Time Streaming Telemetry**: Server-Sent Events (SSE) stream live agent state transitions, word counts, character limits, and execution timers.
- 📊 **Interactive Pipeline Graph**: Visual nodes with glowing radar pulses and status indicators (`Idle` ➔ `Running` ➔ `Completed`).
- 📝 **Markdown Report Studio**: Formatted reader with one-click clipboard copying, Markdown export, and print/PDF support.
- 🎯 **Critic Peer-Review Meter**: Radial score gauge, verdict banner, bulleted strengths, and targeted improvement suggestions.
- 🧪 **Interactive Demo Mode**: Fallback simulation allows immediate evaluation of the full UI without consuming API quotas.
- 💻 **CLI & Web Studio Support**: Run interactively via terminal or launch the real-time Web Studio.

---

## 📁 Project Structure

```
Multi-Agent AI Research System/
├── app.py                  # Web Studio server (Python stdlib http.server + SSE streaming)
├── pipeline.py             # Multi-agent sequential orchestration engine & CLI runner
├── agents.py               # Autonomous agent factories & deterministic LCEL chains
├── tools.py                # Tavily search tool & BeautifulSoup HTML scraper
├── templates/
│   └── index.html          # AI-Native Web Studio Single-Page Application (SPA)
├── design-system/          # UI/UX Pro Max tokens, master spec, and styling rules
│   └── multi-agent-ai-research-system/
│       └── MASTER.md
├── requirements.txt        # Full production dependencies
├── .env.example            # Environment variable template
└── README.md               # Comprehensive project documentation
```

---

## 🛠️ Setup & Installation

### 1. Clone the repository
```bash
git clone <YOUR_REPOSITORY_URL>
cd "Multi-Agent AI Research System"
```

### 2. Create and activate a virtual environment
```bash
python -m venv .venv

# On Windows:
.venv\Scripts\activate

# On macOS/Linux:
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment credentials
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Edit `.env` and provide your API keys:
```ini
# Tavily Search API Key (https://tavily.com)
TAVILY_API_KEY=tvly-xxxxxxxxxxxxxxxxxxxx

# Mistral AI API Key (https://console.mistral.ai)
MISTRAL_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxx

# Optional: Model name (defaults to ministral-8b-latest)
MISTRAL_MODEL=ministral-8b-latest
```

---

## 🚀 Running the System

### Option A: Launch the Interactive Web Studio (Recommended)
```bash
python app.py
```
Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser to experience the real-time research dashboard.

### Option B: Run via Command Line Interface (CLI)
```bash
python pipeline.py
```
Enter your desired topic at the prompt to watch the 4-step pipeline execute directly in your terminal.

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).
