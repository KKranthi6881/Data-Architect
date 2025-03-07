from typing import Dict, List, Any, Optional, Union
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
import logging
import json
import uuid
from datetime import datetime
from pathlib import Path

from src.utils import ChromaDBManager
from src.db.database import ChatDatabase

# Set up logger
logger = logging.getLogger(__name__)

class CodeSearchResult(BaseModel):
    """Model for code search results"""
    file_path: str = Field(description="Path to the file")
    code_snippet: str = Field(description="Relevant code snippet")
    relevance_score: float = Field(description="Relevance score (0-1)")
    explanation: str = Field(description="Why this code is relevant")
    repo_info: Dict[str, Any] = Field(description="Repository information")

class GitHubCodeSearchAgent:
    """Agent for searching GitHub code repositories based on parsed questions"""
    
    def __init__(self, chroma_manager: Optional[ChromaDBManager] = None):
        """Initialize the GitHub code search agent"""
        self.logger = logging.getLogger(__name__)
        self.db = ChatDatabase()
        
        # Initialize ChromaDB manager if not provided
        if chroma_manager:
            self.chroma_manager = chroma_manager
        else:
            self.chroma_manager = ChromaDBManager()
        
        # Initialize LLM
        self.llm = ChatOllama(
            model="llama3.2:latest",
            temperature=0.1,
            base_url="http://localhost:11434",
            timeout=120,
        )
    
    def search_code(self, parsed_question: Dict[str, Any], max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search for relevant code based on the parsed question
        
        Args:
            parsed_question: The output from the question parser
            max_results: Maximum number of results to return
            
        Returns:
            List of code search results
        """
        try:
            # Extract key information from parsed question
            rephrased_question = parsed_question.get("rephrased_question", "")
            key_points = parsed_question.get("key_points", [])
            business_context = parsed_question.get("business_context", {})
            
            # Create search query from parsed question
            search_query = self._create_search_query(rephrased_question, key_points, business_context)
            
            # Search in GitHub collection
            search_results = self._search_github_collection(search_query, max_results)
            
            # Analyze and enhance results with LLM
            enhanced_results = self._enhance_search_results(search_results, parsed_question)
            
            return enhanced_results
            
        except Exception as e:
            self.logger.error(f"Error in GitHub code search: {e}", exc_info=True)
            return []
    
    def _create_search_query(self, rephrased_question: str, key_points: List[str], business_context: Dict[str, Any]) -> str:
        """Create an optimized search query from the parsed question"""
        # Extract key entities from business context
        key_entities = business_context.get("key_entities", [])
        domain = business_context.get("domain", "")
        
        # Combine key information into a search query
        query_parts = [rephrased_question]
        
        # Add key points (limited to first 2)
        if key_points:
            query_parts.extend(key_points[:2])
        
        # Add key entities
        if key_entities:
            query_parts.append(" ".join(key_entities))
        
        # Add domain if available
        if domain:
            query_parts.append(domain)
        
        # Join all parts
        search_query = " ".join(query_parts)
        
        self.logger.info(f"Created search query: {search_query}")
        return search_query
    
    def _search_github_collection(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search the GitHub collection in ChromaDB"""
        try:
            # Get the GitHub collection
            collection = self.chroma_manager.get_or_create_collection("github_documents")
            
            # Search the collection
            results = collection.query(
                query_texts=[query],
                n_results=max_results,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = []
            if results and "documents" in results and results["documents"]:
                documents = results["documents"][0]  # First query results
                metadatas = results["metadatas"][0]  # First query metadatas
                distances = results["distances"][0]  # First query distances
                
                for i, (doc, meta, distance) in enumerate(zip(documents, metadatas, distances)):
                    # Calculate relevance score (1 - normalized distance)
                    relevance = 1.0 - min(distance, 1.0)
                    
                    formatted_results.append({
                        "content": doc,
                        "metadata": meta,
                        "relevance_score": relevance,
                        "rank": i + 1
                    })
            
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"Error searching GitHub collection: {e}", exc_info=True)
            return []
    
    def _enhance_search_results(self, search_results: List[Dict[str, Any]], parsed_question: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Enhance search results with LLM analysis"""
        if not search_results:
            return []
        
        try:
            # Create prompt for LLM
            prompt_template = """
            You are a code analysis expert helping to find relevant code for a business question.
            
            BUSINESS QUESTION:
            {rephrased_question}
            
            KEY POINTS:
            {key_points}
            
            BUSINESS CONTEXT:
            {business_context}
            
            I have found the following code snippets that might be relevant. For each snippet, analyze:
            1. How relevant it is to the business question
            2. What specific parts of the code address the question
            3. How this code could be used to answer the question
            
            CODE SNIPPETS:
            {code_snippets}
            
            For each code snippet, provide:
            - A relevance assessment (0-10 scale)
            - An explanation of why this code is relevant
            - Identification of the most important parts of the code
            
            FORMAT YOUR RESPONSE AS JSON:
            ```json
            [
              {{
                "file_path": "path/to/file.py",
                "relevance_score": 8.5,
                "explanation": "This code is relevant because...",
                "key_parts": ["line 5-10: defines the main function", "line 15-20: processes the data"],
                "repo_info": {{
                  "repo_name": "example/repo",
                  "language": "python"
                }}
              }},
              ...
            ]
            ```
            """
            
            # Format code snippets for the prompt
            code_snippet_texts = []
            for i, result in enumerate(search_results):
                file_path = result["metadata"].get("file_path", "unknown_file")
                repo_url = result["metadata"].get("repo_url", "unknown_repo")
                content = result["content"]
                
                snippet_text = f"SNIPPET {i+1}:\nFile: {file_path}\nRepo: {repo_url}\n\n```\n{content}\n```\n"
                code_snippet_texts.append(snippet_text)
            
            # Format the prompt
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["rephrased_question", "key_points", "business_context", "code_snippets"]
            )
            
            formatted_prompt = prompt.format(
                rephrased_question=parsed_question.get("rephrased_question", ""),
                key_points="\n".join([f"- {point}" for point in parsed_question.get("key_points", [])]),
                business_context=json.dumps(parsed_question.get("business_context", {}), indent=2),
                code_snippets="\n\n".join(code_snippet_texts)
            )
            
            # Get LLM response
            response = self.llm.invoke(formatted_prompt)
            response_text = response.content if hasattr(response, "content") else str(response)
            
            # Extract JSON from response
            json_start = response_text.find("```json")
            json_end = response_text.rfind("```")
            
            if json_start != -1 and json_end != -1 and json_end > json_start:
                json_text = response_text[json_start + 7:json_end].strip()
                enhanced_results = json.loads(json_text)
                
                # Ensure all results have the required fields
                for result in enhanced_results:
                    if "relevance_score" not in result:
                        result["relevance_score"] = 0.0
                    if "explanation" not in result:
                        result["explanation"] = "No explanation provided"
                    if "file_path" not in result:
                        result["file_path"] = "unknown_file"
                    if "repo_info" not in result:
                        result["repo_info"] = {}
                
                return enhanced_results
            else:
                # Fallback if JSON parsing fails
                self.logger.warning("Failed to parse LLM response as JSON, using raw search results")
                return [
                    {
                        "file_path": result["metadata"].get("file_path", "unknown_file"),
                        "code_snippet": result["content"],
                        "relevance_score": result["relevance_score"],
                        "explanation": "Automatically extracted from code repository",
                        "repo_info": {
                            "repo_url": result["metadata"].get("repo_url", "unknown_repo"),
                            "language": result["metadata"].get("language", "unknown")
                        }
                    }
                    for result in search_results
                ]
                
        except Exception as e:
            self.logger.error(f"Error enhancing search results: {e}", exc_info=True)
            # Return basic results if enhancement fails
            return [
                {
                    "file_path": result["metadata"].get("file_path", "unknown_file"),
                    "code_snippet": result["content"],
                    "relevance_score": result["relevance_score"],
                    "explanation": "Error enhancing results",
                    "repo_info": {
                        "repo_url": result["metadata"].get("repo_url", "unknown_repo"),
                        "language": result["metadata"].get("language", "unknown")
                    }
                }
                for result in search_results
            ]
    
    def save_search_results(self, thread_id: str, conversation_id: str, parsed_question: Dict[str, Any], search_results: List[Dict[str, Any]]):
        """Save search results to the thread directory"""
        try:
            # Get thread directory
            thread_dir = Path(self.db.db_path).parent / "threads" / thread_id
            thread_dir.mkdir(exist_ok=True)
            
            # Create a search results file
            search_results_file = thread_dir / f"search_results_{conversation_id}.json"
            
            # Prepare data for JSON storage
            json_data = {
                "id": str(uuid.uuid4()),
                "thread_id": thread_id,
                "conversation_id": conversation_id,
                "timestamp": datetime.now().isoformat(),
                "parsed_question": parsed_question,
                "search_results": search_results
            }
            
            # Write to file
            with open(search_results_file, 'w') as f:
                json.dump(json_data, f, indent=2)
                
            self.logger.info(f"Saved search results to {search_results_file}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving search results: {e}", exc_info=True)
            return False