# 📚 Agentic RAG System

This project implements an **intelligent research assistant** that retrieves and synthesizes information using:

1. **ArXiv papers** as the primary knowledge source (**RAG approach**)
2. **Web search (Tavily API)** as a fallback mechanism
3. **LangGraph** for orchestrating the decision-making workflow

## 🎯 Purpose

The system is designed to provide research-backed answers to technical and scientific questions by:

- Prioritizing academic and research papers from ArXiv for scientific queries
- Falling back to web search for recent developments or non-academic topics
- Maintaining conversation context for coherent multi-turn interactions
- Ensuring proper attribution and citations in responses

## 🔑 Prerequisites

To use this system, you'll need:

1. **OpenAI API Key**

   - Required for:
     - Text embeddings (for semantic search)
     - Response generation (GPT-4o-mini)
     - Routing decisions (GPT-4o-mini)
   - Get it from: [OpenAI Platform](https://platform.openai.com)

2. **Tavily API Key**

   - Required for:
     - Web search fallback functionality
     - Real-time information retrieval
     - Academic domain filtering
   - Get it from: [Tavily](https://app.tavily.com)

3. **Python Environment**
   - Python 3.8 or higher
   - Required packages (will be installed automatically):
     - langchain-community
     - langchain_chroma
     - langchain_core
     - langchain_openai
     - langchain_text_splitters
     - langgraph
     - tavily-python
     - openai
     - python-dotenv

---

## 🚀 Features

- **Retrieves academic papers** from **ArXiv** using semantic search.
- **Falls back to web search** if no relevant ArXiv results are found.
- **Synthesizes responses** from multiple sources while ensuring proper attribution.
- **Maintains conversation memory** for multi-turn interactions.

---

## 📌 System Architecture

The workflow consists of several nodes:

1. **Router Node**

   - Decides whether to retrieve information from **ArXiv** or **Web search**.
   - Prioritizes **ArXiv papers** first.
   - Falls back to **web search** if needed.

2. **ArXiv RAG Node**

   - Loads **PDF files** from ArXiv.
   - **Chunks** them effectively using a specialized strategy.
   - **Stores** information in a vector database for **semantic search**.

3. **Web Search Node**

   - Uses the **Tavily API** to fetch web search results.
   - **Optimizes queries** to retrieve relevant information.
   - **Filters and processes** results while ensuring proper attribution.

4. **Answer Synthesis Node**

   - Merges information from ArXiv and Web sources.
   - Uses **context-aware prompts** based on the source.
   - **Generates well-structured responses** with citations.

5. **Conversation Memory Node**

   - Tracks **previous Q&A** for contextual understanding.
   - Updates conversation state dynamically.
   - Maintains a **sliding window** of relevant history.

6. **Workflow Construction**
   - Uses **LangGraph** to define the **state graph**.
   - Integrates all nodes and configures decision paths.

## 🤖 Agentic Workflow Architecture

The user workflow is translated into an agentic system through the following components:

1. **State Management**

   - **Conversation State**: Tracks user queries, system responses, and context
   - **Search State**: Maintains information about current search results and sources
   - **Decision State**: Stores routing decisions and their rationale

2. **Agent Components**

   - **Router Agent**: Makes intelligent decisions about information sources

     - Analyzes query type and context
     - Determines optimal search strategy
     - Handles fallback mechanisms

   - **Search Agent**: Executes information retrieval

     - Manages ArXiv API interactions
     - Handles Tavily web search
     - Processes and filters results

   - **Synthesis Agent**: Combines and formats information
     - Merges multiple sources
     - Ensures proper attribution
     - Generates coherent responses

3. **Feedback Loop**
   - System learns from user interactions
   - Improves routing decisions over time
   - Adapts to user preferences and query patterns

## 📊 Data Requirements and Sources

The system requires and manages several types of data:

1. **Input Data**

   - **User Queries**: Natural language questions and follow-ups
   - **Conversation History**: Previous interactions for context
   - **User Preferences**: Optional settings for search behavior

2. **Knowledge Sources**

   - **ArXiv Papers**:

     - Source: ArXiv API
     - Format: PDF documents
     - Update Frequency: Daily
     - Coverage: Scientific and technical papers

   - **Web Content**:
     - Source: Tavily API
     - Format: Web pages and documents
     - Update Frequency: Real-time
     - Coverage: News, blogs, documentation, etc.

3. **Processed Data**

   - **Embeddings**: Vector representations of text

     - Generated using OpenAI's embedding model
     - Stored in vector database

   - **Chunks**: Processed text segments

     - Size: Optimized for semantic search
     - Metadata: Source, date, relevance score

   - **Citations**: Reference information
     - Paper titles, authors, URLs
     - Web page sources and dates

4. **Output Data**
   - **Responses**: Generated answers with citations
   - **Search Results**: Ranked and filtered information
   - **Conversation Logs**: Interaction history

## 🔍 Example Queries and Responses

The system intelligently routes queries based on their nature:

### 📄 ArXiv-Focused Queries

```python
# Example: Technical/Scientific Query
"What are the latest developments in transformer architecture for NLP?"

# System Response:
# 1. Router prioritizes ArXiv search
# 2. Retrieves relevant papers about transformer architectures
# 3. Synthesizes information with proper citations
```

### 🌐 Web Search-Focused Queries

```python
# Example 1: Recent News/Updates
"What are the latest developments in AI regulation in the EU?"

# System Response:
# 1. Router detects need for recent information
# 2. Uses Tavily API to search current news and official documents
# 3. Provides up-to-date information with source attribution

# Example 2: Non-Academic Topics
"What are the best practices for implementing CI/CD pipelines?"

# System Response:
# 1. Router identifies as practical/implementation query
# 2. Uses web search to find industry best practices
# 3. Combines multiple sources for comprehensive answer
```

### 🔄 Mixed Source Queries

```python
# Example: Hybrid Query
"What are the current trends in quantum computing and their practical applications?"

# System Response:
# 1. Router splits query into academic and practical components
# 2. Uses ArXiv for theoretical/technical aspects
# 3. Uses web search for practical applications and industry news
# 4. Synthesizes information from both sources
```

## 🎓 Conclusion

The Agentic RAG System with ArXiv + Web Fallback represents a powerful approach to information retrieval and synthesis, combining the best of both academic and real-time knowledge sources. By intelligently routing queries and maintaining conversation context, it provides:

- **Comprehensive Answers**: Leveraging both academic papers and current web information
- **Proper Attribution**: Ensuring all sources are properly cited
- **Contextual Understanding**: Maintaining conversation history for coherent interactions
- **Flexible Knowledge Access**: Adapting to different types of queries and information needs

This system is particularly valuable for:

- Researchers seeking both theoretical foundations and practical applications
- Developers looking for up-to-date technical information
- Students and professionals needing comprehensive, well-sourced answers
- Anyone requiring a balance between academic rigor and current information

The modular architecture and use of LangGraph make it easy to extend and adapt the system for specific use cases or additional knowledge sources.

## Bonus: Source code integrated into the n8n workflow

### Install dependencies:
In Terminal
```bash
cd week2/fastapi-n8n
pip install -r requirements.txt
```

### Set environment variables in .env file

```txt
# .env
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key
```

### Run the service:
In Terminal
```txt
uvicorn app.main:app --reload
```

### API Documentation

* Interactive docs available at: http://127.0.0.1:8000/docs
* Health check: http://127.0.0.1:8000/health
* Ask endpoint: POST http://127.0.0.1:8000/ask

### Verify Service Endpoints
In Terminal
```bash
curl -X POST "http://localhost:8000/ask" \
-H "Content-Type: application/json" \
-d '{"question": "Explain quantum computing"}'
```

### Setup n8n Workflow 
To run local LLMs within the workflow, you need to install Ollama, or alternatively, replace the Ollama nodes with an OpenAI nodes

In Terminal
```bash
npx n8n
```

Access n8n UI at: http://127.0.0.1:5678

### Import n8n Workflow
1. Go to n8n UI > Workflows > Import
2. Use the `week2/agentic-rag-demo-n8n.json` JSON workflow provided
3. Set up credentials for:
    * [Gmail API access through the Google OAuth2 single service service](https://docs.n8n.io/integrations/builtin/credentials/google/oauth-single-service/#finish-your-n8n-credential)

### Troubleshooting
1. Service not running:
    * Check API keys in `.env` file
    * Run `uvicorn main:app --reload` from project root
2. n8n connection issues:
    * Verify credentials in n8n
    * Check FastAPI is running when n8n executes workflow
3. Missing results:
    * Ensure Tavily/OpenAI accounts have credits
    * Check API rate limits