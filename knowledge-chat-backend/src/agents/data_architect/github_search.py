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
            prompt = f"""
            # Senior Data Engineer Code Analysis

            You are a senior data engineer with 10+ years of experience in implementing data pipelines, ETL processes, and analytics solutions. Analyze the following code snippets to provide actionable implementation guidance.

            ## Business Question
            "{parsed_question.get('rephrased_question', '')}"

            ## Key Points from Question
            {chr(10).join([f"- {point}" for point in parsed_question.get('key_points', [])])}

            ## Business Context
            ```json
            {json.dumps(parsed_question.get('business_context', {}), indent=2)}
            ```

            ## Available Code Snippets
            """
            
            # Format code snippets for the prompt
            for i, result in enumerate(search_results):
                file_path = result.get("file_path", "unknown_file")
                code_snippet = result.get("code_snippet", "")
                repo_info = result.get("repo_info", {})
                
                # Truncate very long code snippets
                if len(code_snippet) > 1500:
                    code_snippet = code_snippet[:1500] + "\n// ... [truncated for brevity] ..."
                
                # Determine language for syntax highlighting
                language = "unknown"
                if file_path.endswith(".py"):
                    language = "python"
                elif file_path.endswith(".sql"):
                    language = "sql"
                elif file_path.endswith(".java"):
                    language = "java"
                elif file_path.endswith(".js"):
                    language = "javascript"
                
                prompt += f"""
            ### Code Snippet {i+1}: {file_path}
            - **Repository**: {repo_info.get('repo_name', 'unknown')}
            - **Language**: {language}
            
            ```{language}
            {code_snippet}
            ```
                """
            
            # Add the rest of the prompt with detailed instructions
            prompt += """
            ## Analysis Instructions
            For each code snippet, provide a comprehensive analysis that would help a data engineer implement or modify the code to address the business question:

            1. **Code Purpose Analysis**:
               - What is the primary function of this code?
               - How does it relate to the business question?
               - What data processing patterns does it implement?

            2. **Implementation Guidance**:
               - How can this code be adapted to address the business question?
               - What specific modifications would be needed?
               - Are there any missing components that need to be added?

            3. **Development Steps**:
               - Provide a step-by-step implementation plan
               - Include specific code blocks that need to be modified
               - Suggest new functions or methods that should be created

            4. **Integration Considerations**:
               - How would this code integrate with other systems?
               - What dependencies or prerequisites are needed?
               - Are there any potential performance issues to address?

            5. **Testing and Validation**:
               - How should the implementation be tested?
               - What edge cases should be considered?
               - What validation checks should be implemented?

            ## Response Format
            For each code snippet, provide your analysis in the following JSON format:

            ```json
            [
              {
                "file_path": "example/path/to/file.py",
                "language": "python",
                "relevance_score": 8.5,
                "code_purpose": "This code implements a data pipeline that extracts data from a source system, transforms it, and loads it into a target database.",
                "implementation_guidance": "This code can be adapted to address the business question by modifying the extraction logic to include additional fields and adding a new transformation step.",
                "development_steps": [
                  {
                    "step": "Modify the extract_data function to include additional fields",
                    "code_block": "def extract_data(source_system):\\n    # Existing code...\\n    # Add new fields\\n    data['new_field'] = source_system.get_new_field()\\n    # Continue with existing code...",
                    "explanation": "Adding the new_field to the extracted data will provide the necessary information for the business question."
                  },
                  {
                    "step": "Add a new transformation function",
                    "code_block": "def transform_for_business_question(data):\\n    # New transformation logic\\n    result = data.groupby(['dimension1', 'dimension2']).agg({'metric': 'sum'})\\n    return result",
                    "explanation": "This new function will aggregate the data according to the business requirements."
                  }
                ],
                "integration_considerations": "This code will need to be integrated with the existing data pipeline. It depends on the source_system module and requires access to the target database.",
                "testing_approach": "The implementation should be tested with a sample dataset that includes edge cases such as missing values and extreme values. Unit tests should be written for the new transformation function."
              }
            ]
            ```

            IMPORTANT: 
            - Your response must be valid JSON that can be parsed
            - Score relevance from 0-10 based on how directly the code addresses the question
            - Provide detailed explanations that show your reasoning
            - Include realistic, executable code blocks in your development steps
            - Focus on practical implementation details that a data engineer could use immediately
            - Do not include any text outside the JSON block
            """
            
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
            self.logger.error(f"Error enhancing code search results: {e}", exc_info=True)
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

    def _validate_enhanced_results(self, enhanced_results: List[Dict[str, Any]], raw_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate and fix enhanced results"""
        if not isinstance(enhanced_results, list):
            self.logger.warning("Enhanced results is not a list, returning raw results")
            return self._format_raw_results(raw_results)
        
        valid_results = []
        
        for i, result in enumerate(enhanced_results):
            if i >= len(raw_results):
                break
            
            # Get the raw result
            raw_result = raw_results[i]
            
            # Create a valid result with required fields
            valid_result = {
                "file_path": result.get("file_path", raw_result.get("file_path", "unknown_file")),
                "language": result.get("language", "unknown"),
                "relevance_score": result.get("relevance_score", 0) / 10.0,  # Normalize to 0-1
                "code_snippet": raw_result.get("code_snippet", ""),
                "repo_info": raw_result.get("repo_info", {}),
                "code_purpose": result.get("code_purpose", ""),
                "implementation_guidance": result.get("implementation_guidance", ""),
                "development_steps": result.get("development_steps", []),
                "integration_considerations": result.get("integration_considerations", ""),
                "testing_approach": result.get("testing_approach", ""),
                "explanation": result.get("explanation", "This code may be relevant to the business question.")
            }
            
            valid_results.append(valid_result)
        
        return valid_results

    def _format_raw_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format raw results when enhancement fails"""
        formatted_results = []
        
        for result in results:
            # Determine language from file extension
            language = "unknown"
            file_path = result.get("file_path", "")
            if file_path.endswith(".py"):
                language = "python"
            elif file_path.endswith(".sql"):
                language = "sql"
            elif file_path.endswith(".java"):
                language = "java"
            elif file_path.endswith(".js"):
                language = "javascript"
            
            formatted_result = {
                "file_path": result.get("file_path", "unknown_file"),
                "language": language,
                "relevance_score": 0.5,  # Default medium relevance
                "code_snippet": result.get("code_snippet", ""),
                "repo_info": result.get("repo_info", {}),
                "code_purpose": "This code was found based on keyword matching.",
                "implementation_guidance": "Review this code to determine if it can be adapted to address the business question.",
                "development_steps": [],
                "integration_considerations": "Further analysis needed to determine integration requirements.",
                "testing_approach": "Standard testing practices should be applied.",
                "explanation": "This code contains keywords related to the business question."
            }
            
            formatted_results.append(formatted_result)
        
        return formatted_results
    
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