from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from services.agent_service import AgenticRAGSystem
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Agentic RAG System API",
    description="Research assistant that retrieves and synthesizes information from ArXiv papers and web search",
    version="1.0.0"
)

class QuestionRequest(BaseModel):
    question: str
    conversation_history: str = ""

class AnswerResponse(BaseModel):
    answer: str
    conversation_history: str
    sources: list
    source_type: str

# Initialize the agent system
agent_system = AgenticRAGSystem()

@app.post("/ask", response_model=AnswerResponse, summary="Ask a research question")
async def ask_question(request: QuestionRequest):
    """
    Processes a research question using academic papers and web search, returning a synthesized answer with sources.
    """
    try:
        response = agent_system.ask_question(
            question=request.question,
            conversation_history=request.conversation_history
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.get("/health", summary="Service health check")
def health_check():
    return {"status": "active", "version": "1.0.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)