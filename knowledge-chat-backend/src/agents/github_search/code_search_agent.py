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
import re

from src.utils import ChromaDBManager
from src.db.database import ChatDatabase
from src.tools import SearchTools

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
    
    def __init__(self):
        """Initialize the GitHub code search agent"""
        self.logger = logging.getLogger(__name__)
        self.db = ChatDatabase()
        
        # Create ChromaDBManager and SearchTools
        self.chroma_manager = ChromaDBManager()
        self.search_tools = SearchTools(self.chroma_manager)
        
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
            # Create search query from parsed question
            search_query = self._create_search_query(parsed_question)
            self.logger.info(f"Created code search query: {search_query}")
            
            # Use the search_tools to search for GitHub repositories
            search_results = self.search_tools.search_github_repos(search_query, limit=max_results)
            
            # Check if we got valid results
            if search_results.get("status") != "success" or not search_results.get("results"):
                self.logger.warning(f"No GitHub code search results found or error in search")
                return []
            
            # Format the results for enhancement
            formatted_results = search_results.get("results", [])
            
            # Enhance results with LLM
            enhanced_results = self._enhance_search_results(formatted_results, parsed_question)
            
            return enhanced_results
            
        except Exception as e:
            self.logger.error(f"Error in GitHub code search: {e}", exc_info=True)
            return []
    
    def _create_search_query(self, parsed_question: Dict[str, Any]) -> str:
        """Create a search query from the parsed question"""
        try:
            # Extract key information from parsed question
            rephrased_question = parsed_question.get("rephrased_question", "")
            key_points = parsed_question.get("key_points", [])
            business_context = parsed_question.get("business_context", {})
            
            # Build search query
            query_parts = [rephrased_question]
            
            # Add key points (limited to first 3)
            if key_points:
                query_parts.extend(key_points[:3])
            
            # Add key entities from business context
            if business_context and "key_entities" in business_context:
                entities = business_context.get("key_entities", [])
                if entities and isinstance(entities, list):
                    query_parts.append(" ".join(entities[:5]))
            
            # Join all parts
            search_query = " ".join(query_parts)
            
            return search_query
            
        except Exception as e:
            self.logger.error(f"Error creating code search query: {e}")
            return parsed_question.get("rephrased_question", "")
    
    def _enhance_search_results(self, search_results: List[Dict[str, Any]], parsed_question: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Enhance search results with LLM analysis"""
        try:
            if not search_results:
                return []
            
            # Create prompt for LLM
            prompt_template = """
            You are an expert code analyst helping to find relevant code for a business question.
            
            BUSINESS QUESTION:
            {rephrased_question}
            
            KEY POINTS:
            {key_points}
            
            BUSINESS CONTEXT:
            {business_context}
            
            I've found some code that might be relevant. For each code snippet, analyze:
            1. How relevant it is to the business question (score 0-10)
            2. Why it's relevant or what insights it provides
            3. How it could be used or adapted to address the business need
            
            CODE SNIPPETS:
            {code_snippets}
            
            Provide your analysis in the following JSON format:
            ```json
            [
              {{
                "file_path": "path/to/file.py",
                "code_snippet": "def example_function():\\n    return 'example'",
                "relevance_score": 8.5,
                "explanation": "This code is relevant because...",
                "repo_info": {{
                  "repo_name": "example/repo",
                  "language": "python"
                }}
              }},
              ...
            ]
            ```
            
            IMPORTANT: Your response must be valid JSON that can be parsed. Do not include any text outside the JSON block.
            """
            
            # Format code snippets for the prompt
            code_snippet_texts = []
            for i, result in enumerate(search_results):
                # Extract code information
                file_path = result.get("file", {}).get("path", "unknown_file")
                repo_name = result.get("repository", {}).get("name", "unknown_repo")
                repo_url = result.get("repository", {}).get("url", "unknown_url")
                language = result.get("file", {}).get("language", "unknown_language")
                content = result.get("content", "")
                
                # Create code snippet text
                snippet_text = f"SNIPPET {i+1}:\nFile: {file_path}\nRepo: {repo_name} ({repo_url})\nLanguage: {language}\n\n```\n{content}\n```\n"
                code_snippet_texts.append(snippet_text)
            
            # Format the prompt
            prompt = PromptTemplate.from_template(prompt_template).format(
                rephrased_question=parsed_question.get("rephrased_question", ""),
                key_points="\n".join([f"- {point}" for point in parsed_question.get("key_points", [])]),
                business_context=json.dumps(parsed_question.get("business_context", {}), indent=2),
                code_snippets="\n".join(code_snippet_texts)
            )
            
            # Get LLM response
            response = self.llm.invoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Extract JSON from response
            json_text = self._extract_json(response_text)
            
            if not json_text:
                self.logger.warning("Could not extract JSON from LLM response")
                return self._format_raw_results(search_results)
            
            try:
                enhanced_results = json.loads(json_text)
                # Validate and fix the enhanced results
                return self._validate_enhanced_results(enhanced_results, search_results)
            except json.JSONDecodeError as e:
                self.logger.error(f"Error enhancing code search results: {e}")
                return self._format_raw_results(search_results)
            
        except Exception as e:
            self.logger.error(f"Error enhancing search results: {e}", exc_info=True)
            # Return basic results if enhancement fails
            return self._format_raw_results(search_results)

    def _extract_json(self, text: str) -> str:
        """Extract JSON from LLM response text"""
        # Try to extract JSON from markdown code block
        json_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
        matches = re.findall(json_pattern, text)
        
        if matches:
            return matches[0].strip()
        
        # Try to extract JSON array directly
        array_pattern = r"\[\s*{[\s\S]*}\s*\]"
        matches = re.findall(array_pattern, text)
        
        if matches:
            return matches[0].strip()
        
        # If all else fails, try to extract anything that looks like JSON
        if text.strip().startswith("[") and text.strip().endswith("]"):
            return text.strip()
        
        return ""

    def _validate_enhanced_results(self, enhanced_results: List[Dict[str, Any]], original_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate and fix enhanced results"""
        # Ensure we have a list
        if not isinstance(enhanced_results, list):
            self.logger.warning("Enhanced results is not a list, using raw results")
            return self._format_raw_results(original_results)
        
        # Ensure all results have the required fields
        for i, result in enumerate(enhanced_results):
            if "relevance_score" not in result:
                result["relevance_score"] = 0.7
            elif isinstance(result["relevance_score"], int) or isinstance(result["relevance_score"], float):
                # Normalize score to 0-1 range if it's on a 0-10 scale
                if result["relevance_score"] > 1:
                    result["relevance_score"] = result["relevance_score"] / 10.0
            else:
                result["relevance_score"] = 0.7
            
            if "explanation" not in result:
                result["explanation"] = "Automatically extracted from GitHub repository"
            
            if "file_path" not in result and i < len(original_results):
                result["file_path"] = original_results[i].get("file", {}).get("path", "unknown_file")
            
            if "repo_info" not in result and i < len(original_results):
                result["repo_info"] = {
                    "repo_name": original_results[i].get("repository", {}).get("name", "unknown_repo"),
                    "repo_url": original_results[i].get("repository", {}).get("url", "unknown_url"),
                    "language": original_results[i].get("file", {}).get("language", "unknown_language")
                }
            
            if "code_snippet" not in result and i < len(original_results):
                result["code_snippet"] = original_results[i].get("content", "")
        
        return enhanced_results

    def _format_raw_results(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format raw search results when enhancement fails"""
        return [
            {
                "file_path": result.get("file", {}).get("path", "unknown_file"),
                "code_snippet": result.get("content", ""),
                "relevance_score": 0.7,  # Default relevance score
                "explanation": "Automatically extracted from GitHub repository",
                "repo_info": {
                    "repo_name": result.get("repository", {}).get("name", "unknown_repo"),
                    "repo_url": result.get("repository", {}).get("url", "unknown_url"),
                    "language": result.get("file", {}).get("language", "unknown_language")
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