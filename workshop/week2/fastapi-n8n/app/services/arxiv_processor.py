import re
import requests
from langchain_community.document_loaders import ArxivLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

class ArXivProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", "(?<=\\. )", " ", ""]
        )
        
    def search_and_retrieve(self, query: str, max_docs: int=3, confidence_threshold: float=0.3, k: int=5):
        try:
            # Get paper metadata using Arxiv API
            base_url = "http://export.arxiv.org/api/query?"
            params = {
                "search_query": query,
                "start": 0,
                "max_results": max_docs,
                "sortBy": "relevance",
                "sortOrder": "descending"
            }
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            
            # Parse XML response
            from xml.etree import ElementTree as ET
            root = ET.fromstring(response.content)
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            
            papers = []
            for entry in root.findall('atom:entry', ns):
                paper_id = entry.find('atom:id', ns).text.split('/')[-1]
                title = entry.find('atom:title', ns).text.strip()
                summary = entry.find('atom:summary', ns).text.strip()
                authors = [author.find('atom:name', ns).text 
                           for author in entry.findall('atom:author', ns)]
                
                # Get full text PDF
                pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"
                papers.append({
                    'id': paper_id,
                    'title': title,
                    'summary': summary,
                    'authors': authors,
                    'pdf_url': pdf_url
                })
        except Exception as e:
            print(f"ArXiv API error: {e}")
            return []
        
        all_chunks = []
        for paper in papers:
            try:
                # Create document with proper metadata
                metadata = {
                    'source': f"arXiv:{paper['id']}",
                    'title': paper['title'],
                    'authors': ", ".join(paper['authors']),
                    'url': paper['pdf_url']
                }
                
                # Use summary + first part of content as context
                content = f"{paper['title']}\n\n{paper['summary']}"
                
                # Create document
                doc = Document(page_content=content, metadata=metadata)
                
                # Split into chunks
                chunks = self.text_splitter.split_documents([doc])
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"Error processing paper {paper['id']}: {e}")
        
        if not all_chunks:
            return []
        
        # Create temporary vector store
        vector_store = Chroma.from_documents(
            documents=all_chunks,
            embedding=OpenAIEmbeddings()
        )
        
        # Retrieve relevant chunks
        results = vector_store.similarity_search_with_relevance_scores(query, k=k)
        return [doc for doc, score in results if score >= confidence_threshold]