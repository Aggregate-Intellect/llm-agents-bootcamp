from typing import TypedDict, List, Dict, Optional, Literal, Union
from langchain_core.documents import Document
from langchain.memory import ConversationBufferMemory

class AgentState(TypedDict):
    question: str
    routing_decision: Literal["arxiv", "web", "both"]
    arxiv_results: Optional[List[Document]]
    web_results: Optional[List[Dict]]
    direct_answer: Optional[str]
    answer: str
    conversation_history: str
    memory: ConversationBufferMemory
    next_node: Optional[Literal["web_search", "synthesize"]]