# Multi-Agent AI Research System

An intelligent multi-agent research workflow powered by LangGraph, LangChain, Mistral AI, and Tavily Search.

## Features

- **Web Search**: Automated multi-source query and snippet retrieval using Tavily API.
- **Deep Web Scraping**: Clean text extraction from articles and web pages using BeautifulSoup.
- **Agent Orchestration**: Research graph workflows with LangGraph and Mistral AI models.

## Setup Instructions

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

### 4. Configure environment variables
Copy `.env.example` to `.env` and fill in your API credentials:
```bash
cp .env.example .env
```
Edit `.env` and add:
- `TAVILY_API_KEY`: Get your key from [Tavily](https://tavily.com)
- `MISTRAL_API_KEY`: Get your key from [Mistral AI Console](https://console.mistral.ai)

## License
MIT
