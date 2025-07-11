import os
from typing import Dict, List, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from pyboxen import boxen
from tavily import TavilyClient
from .state import AgentState
from .arxiv_processor import ArXivProcessor
import json
from langchain.memory import ConversationBufferMemory

class AgenticRAGSystem:
    def __init__(self):
        self.router_prompt = self._create_router_prompt()
        self.arxiv_processor = ArXivProcessor()
        self.tavily = TavilyClient()
        self.workflow = self._create_workflow()
        self.app = self.workflow.compile()

    def _create_router_prompt(self):
        return ChatPromptTemplate.from_template("""
        You are a highly specialized research assistant with access to two information sources:
        1. A collection of ArXiv research papers
        2. A web search tool

        Your task is to determine which source(s) would be better to answer the user's question.
        FIRST try to use ArXiv papers for scientific and academic questions.
        ONLY use web search if:
        - The question requires very recent information not likely in research papers
        - The question is about general knowledge, news, or non-academic topics
        - The question asks for information beyond what academic papers would contain
        Choose BOTH if the question requires integrating academic concepts with recent developments, practical applications, or comparing academic views with general information

        Consider the conversation history for context.

        Question: {question}
        Conversation History: {conversation_history}

        Respond with ONLY ONE of these two options:
        "arxiv" - if the question should be answered using research papers
        "web" - if the question requires web search
        "both" - if both ArXiv and web search are needed

        Your decision should be a single word only (either "arxiv", "web" or "both"). Do not include any explanation, reasoning, or additional text in your response.
        """)

    def _create_workflow(self):
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("router", self.router_node)
        workflow.add_node("arxiv_retrieval", self.arxiv_retrieval_node)
        workflow.add_node("web_search", self.web_search_node)
        workflow.add_node("synthesize", self.synthesize_answer_node)
        workflow.add_node("update_memory", self.update_memory_node)
        
        # Set entry point
        workflow.set_entry_point("router")
        
        # Define conditional edges
        workflow.add_conditional_edges(
            "router",
            lambda state: state["routing_decision"],
            {
                "arxiv": "arxiv_retrieval",
                "web": "web_search",
                "both": "arxiv_retrieval"
            }
        )
        
        workflow.add_conditional_edges(
            "arxiv_retrieval",
            lambda state: state.get("next_node", "synthesize"),
            {
                "web_search": "web_search",
                "synthesize": "synthesize"
            }
        )
        
        # Define fixed edges
        workflow.add_edge("web_search", "synthesize")
        workflow.add_edge("synthesize", "update_memory")
        workflow.add_edge("update_memory", END)
        
        return workflow

    def router_node(self, state: AgentState) -> dict:
        llm = ChatOpenAI(model="gpt-4o-mini")
        chain = self.router_prompt | llm | StrOutputParser()
        decision = chain.invoke({
            "question": state["question"],
            "conversation_history": state["conversation_history"]
        }).strip().lower()
        
        if decision not in ["arxiv", "web", "both"]:
            decision = "web"
        
        print(boxen(f"Router decision: {decision}", title=">>> Router Node", color="blue", padding=(1, 2)))
        return {"routing_decision": decision}

    def arxiv_retrieval_node(self, state: AgentState) -> dict:
        question = state["question"]
        routing_decision = state["routing_decision"]
        relevant_docs = []
        next_dest = "synthesize"
        
        try:
            # Use dynamic search instead of pre-loaded papers
            relevant_docs = self.arxiv_processor.search_and_retrieve(question)
            print(boxen(f"Found {len(relevant_docs)} ArXiv chunks", title=">>> ArXiv Retrieval", color="blue", padding=(1, 2)))
        except Exception as e:
            print(boxen(f"ArXiv error: {str(e)}", title=">>> ArXiv Retrieval", color="red", padding=(1, 2)))
        
        if routing_decision == "both":
            next_dest = "web_search"
        
        return {"arxiv_results": relevant_docs, "next_node": next_dest}

    def web_search_node(self, state: AgentState) -> dict:
        web_res = None
        direct_ans = None
        
        try:
            academic_domains = ["arxiv.org", "scholar.google.com", "researchgate.net", "edu"]
            search_response = self.tavily.search(
                query=state["question"],
                max_results=5,
                include_domains=academic_domains,
                search_depth="advanced",
                include_answer=True
            )
            web_res = search_response.get("results", [])
            direct_ans = search_response.get("answer")
            print(boxen(f"Found {len(web_res)} web results", title=">>> Web Search", color="blue", padding=(1, 2)))
        except Exception as e:
            print(boxen(f"Web search error: {str(e)}", title=">>> Web Search", color="red", padding=(1, 2)))
            web_res = []
        
        return {"web_results": web_res, "direct_answer": direct_ans}

    def synthesize_answer_node(self, state: AgentState) -> dict:
        question = state["question"]
        arxiv_results = state.get("arxiv_results", [])
        web_results = state.get("web_results", [])
        direct_answer = state.get("direct_answer", None)
        conversation_history = state["conversation_history"]
        
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        final_answer = ""
        source_type = "None"
        prompt_template = ""
        sources_for_prompt = ""
        
        # 1. Determine source type and prepare context
        if arxiv_results and web_results:
            source_type = "Combined ArXiv and Web"
            arxiv_sources = "\n\n".join([
                f"--- ArXiv Document: {doc.metadata.get('source', 'Unknown')} (Page {doc.metadata.get('page', 'N/A')}) ---\n{doc.page_content}"
                for doc in arxiv_results
            ])
            web_sources = "\n\n".join([
                f"--- Web Source [{i+1}]: {res.get('title', 'N/A')} ---\n{res.get('content', 'N/A')}"
                for i, res in enumerate(web_results)
            ])
            direct_answer_context = f"Tavily suggested direct answer: {direct_answer}" if direct_answer else "No direct answer suggested by Tavily."
            sources_for_prompt = f"ArXiv Extracts:\n{arxiv_sources}\n\nWeb Search Results:\n{web_sources}\n\n{direct_answer_context}"
            prompt_template = """
            You are a knowledgeable research assistant synthesizing information from both academic papers and web search results.

            Task: Answer the user's question comprehensively using ONLY the provided information. Integrate findings, prioritizing ArXiv for core concepts/theory and web search for recent developments, examples, or general context.

            Question:
            {question}

            Conversation History:
            {conversation_history}

            Information Sources:
            {sources}

            Instructions:
            1. Carefully read and understand all provided ArXiv extracts and web search results.
            2. Synthesize a coherent and comprehensive answer addressing the question.
            3. Structure the response logically (e.g., introduction, key points from ArXiv, relevant web context/examples, conclusion). Use markdown formatting (headers, lists, bolding) for readability.
            4. **Crucially, cite your sources accurately within the text**:
                - For ArXiv info: Use (Author et al., Page X) or (Source: Document Source) if author/page unknown.
                - For Web info: Use [Web Source 1], [Web Source 2], etc., corresponding to the numbers provided.
            5. If the provided information is insufficient or contradictory, state that clearly.
            6. DO NOT include information not present in the sources. Base the entire answer strictly on the provided text.
            7. Consider the `direct_answer` suggestion from Tavily but verify against the full web results before incorporating its content.

            Synthesized Answer:
            """
            
        elif arxiv_results:
            source_type = "ArXiv Papers"
            sources_for_prompt = "\n\n".join([
                f"--- Document: {doc.metadata.get('title', 'Unknown')} " \
                f"by {doc.metadata.get('authors', 'Unknown authors')} ---\n" \
                f"{doc.page_content}"
                for doc in arxiv_results
            ])
            # prompt_template = """[Actual prompt template from notebook]"""
            prompt_template = """
            You are a knowledgeable research assistant specializing in mathematical theory and scientific literature analysis.
            Your goal is to generate clean, formatted responses to user questions based solely on the provided ArXiv sources.

            Question:
            {question}

            Relevant Extracts from ArXiv Papers:
            {sources}

            Conversation History:
            {conversation_history}

            Instructions for Synthesizing the Answer:
            1. Read the extracts thoroughly and understand the concepts.
            2. Answer the question comprehensively using ONLY the provided context.
            3. Organize the response into the following markdown sections (if applicable):
                - Summary
                - Key Concepts
                - Theoretical Results
                - Implications / Applications
            4. Cite from the paper in the format: (Author Last Name et al., Year). Example: (Smith et al., 2023)
            5. Avoid repetition, excessive formal tone, or generic commentary. Be clear and concise.
            6. If the provided text lacks enough detail to answer, state it clearly and suggest what additional info is needed.

            Now, write a well-structured, markdown-formatted answer to the question and it should be in a readable format as well.

            Your answer:
            """
            
        elif web_results:
            source_type = "Web Search Results"
            web_sources = "\n\n".join([
                f"--- Web Source [{i+1}]: {res.get('title', 'N/A')} ---\n{res.get('content', 'N/A')}"
                for i, res in enumerate(web_results)
            ])
            direct_answer_context = f"Tavily suggested direct answer: {direct_answer}" if direct_answer else "No direct answer suggested by Tavily."
            sources_for_prompt = f"Web Search Results:\n{web_sources}\n\n{direct_answer_context}"
            # prompt_template = """[Actual prompt template from notebook]"""
            prompt_template = """
            You are a knowledgeable research assistant providing accurate information based on web search results.

            Question: {question}

            Here are relevant web search results:
                {sources}

            Conversation History: {conversation_history}

            Instructions:
            1. Synthesize a comprehensive answer using ONLY the information provided above.
            2. Cite sources using [1], [2], etc. corresponding to the source numbers above.
            3. Consider the `direct_answer` suggestion from Tavily but verify against the full web results before incorporating its content.
            4. If the search results don't contain sufficient information, acknowledge the limitations.
            5. DO NOT make up information not present in the sources. 
            6. Include only facts supported by the sources.
            7. Use markdown formatting for readability.

            Your answer:
            """
            
        else:
            return {"answer": "I could not find relevant information from ArXiv or web search to answer your question."}
        
        # 2. Perform Synthesis
        synthesis_prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = synthesis_prompt | llm | StrOutputParser()
        
        try:
            response = chain.invoke({
                "question": question,
                "sources": sources_for_prompt,
                "conversation_history": conversation_history
            })
            final_answer = response
        except Exception as e:
            final_answer = f"Error synthesizing answer: {str(e)}"
        
        # 3. Add URL Citations if Web Results were used
        if source_type in ["Web Search Results", "Combined ArXiv and Web"] and web_results:
            url_citations = "\n\n**Web Sources:**\n" + "\n".join([
                f"[{i+1}] {res.get('url', 'URL not available')}"
                for i, res in enumerate(web_results)
            ])
            final_answer += url_citations
        
        # 4. Format Final Output
        formatted_output = f"""## Context
        **Question:** {question}
        **Source(s) Used:** {source_type}

        ## Response
        {final_answer}
        """
        
        formatted_output = formatted_output.strip()
        
        return {"answer": formatted_output}

    def update_memory_node(self, state: AgentState) -> dict:
        memory = state['memory']
        memory.save_context(
            {"question": state["question"]},
            {"answer": state["answer"]}
        )
        return {"conversation_history": memory.load_memory_variables({}).get("history", "")}

    def ask_question(self, question: str, conversation_history: str = "") -> dict:
        memory = ConversationBufferMemory(return_messages=False, output_key="answer", input_key="question")
        memory.load_memory_variables({})  # Initialize
        
        initial_state = {
            "question": question,
            "routing_decision": None,
            "arxiv_results": None,
            "web_results": None,
            "direct_answer": None,
            "answer": "",
            "conversation_history": conversation_history,
            "memory": memory,
            "next_node": None
        }
        
        result = self.app.invoke(initial_state)
        return {
            "answer": result["answer"],
            "conversation_history": result["conversation_history"],
            "sources": self._extract_sources(result),
            "source_type": result["routing_decision"]
        }

    def _extract_sources(self, state: AgentState) -> List[Dict]:
        sources = []
        
        # Extract ArXiv sources
        if state.get("arxiv_results"):
            for doc in state["arxiv_results"]:
                sources.append({
                    "type": "arxiv",
                    "id": doc.metadata.get("source", "").replace("arXiv:", ""),
                    "title": doc.metadata.get("title", "Unknown Title"),
                    "authors": doc.metadata.get("authors", "Unknown authors"),
                    "url": doc.metadata.get("url", "")
                })
        
        # Extract web sources
        if state.get("web_results"):
            for i, res in enumerate(state["web_results"]):
                sources.append({
                    "type": "web",
                    "title": res.get("title", "Web Source"),
                    "url": res.get("url", ""),
                    "content": res.get("content", "")[:200] + "..."
                })
        
        return sources