import os
from app.services.agent_service import AgenticRAGSystem
from tavily import TavilyClient 
from dotenv import load_dotenv

load_dotenv()

def test_search():
    # Test ArXiv search
    arxiv_processor = AgenticRAGSystem().arxiv_processor
    arxiv_results = arxiv_processor.search_and_retrieve(
        "quantum computing",
        max_docs=3,
        confidence_threshold=0.3
    )
    print(f"ArXiv results: {len(arxiv_results)}")
    for i, doc in enumerate(arxiv_results):
        print(f"\nArXiv Doc {i+1}:")
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")
        print(f"Title: {doc.metadata.get('title', 'Unknown')}")
        print(f"Authors: {doc.metadata.get('authors', 'Unknown')}")
        print(f"Content: {doc.page_content[:200]}...\n")
    
    # Test Tavily search
    tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    web_results = tavily.search(
        query="quantum computing",
        max_results=3,
        include_domains=["arxiv.org", "scholar.google.com", "researchgate.net", "edu"],
        search_depth="advanced"
    ).get("results", [])
    
    print(f"Web results: {len(web_results)}")
    for i, res in enumerate(web_results):
        print(f"\nWeb Result {i+1}:")
        print(f"Title: {res.get('title', 'No title')}")
        print(f"URL: {res.get('url', 'No URL')}")
        print(f"Content: {res.get('content', 'No content')[:200]}...\n")

if __name__ == "__main__":
    test_search()