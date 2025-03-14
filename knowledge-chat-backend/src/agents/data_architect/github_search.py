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
import os

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
    """Agent for searching GitHub code repositories based on parsed questions with focus on dbt and Snowflake"""
    
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
        
        # Define supported file types and their priorities
        self.file_priorities = {
            ".sql": 10,   # Highest priority for SQL files
            ".yml": 9,    # High priority for dbt schema files
            ".yaml": 9,   # Alternative extension for YAML
            ".py": 7,     # Python files (could be for dbt utils/plugins)
            ".md": 5,     # Documentation files
            ".json": 5,   # Config files
        }
    
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
            search_results = self.search_tools.search_github_repos(search_query, limit=max_results*2)  # Get more results for filtering
            
            # Check if we got valid results
            if search_results.get("status") != "success" or not search_results.get("results"):
                self.logger.warning(f"No GitHub code search results found or error in search")
                return []
            
            # Format and filter the results with focus on dbt and Snowflake
            formatted_results = self._filter_dbt_snowflake_results(search_results.get("results", []), max_results)
            
            # Enhance results with LLM
            enhanced_results = self._enhance_search_results(formatted_results, parsed_question)
            
            # Fetch and add related files for context
            enhanced_results = self._add_related_files(enhanced_results, parsed_question)
            
            return enhanced_results
            
        except Exception as e:
            self.logger.error(f"Error in GitHub code search: {e}", exc_info=True)
            return []
    
    def _filter_dbt_snowflake_results(self, results: List[Dict[str, Any]], max_results: int) -> List[Dict[str, Any]]:
        """Filter results to prioritize dbt and Snowflake content"""
        # Score each result based on relevance to dbt/Snowflake
        scored_results = []
        
        for result in results:
            score = 0
            file_path = result.get("file_path", "").lower()
            code_snippet = result.get("code_snippet", "").lower()
            
            # Score based on file extension
            file_ext = Path(file_path).suffix
            score += self.file_priorities.get(file_ext, 0)
            
            # Score based on directory structure (dbt project structure)
            if "/models/" in file_path:
                score += 5
            if "/macros/" in file_path:
                score += 4
            if "/schema/" in file_path or "/schemas/" in file_path:
                score += 4
            if "/snapshots/" in file_path:
                score += 3
            if "/seeds/" in file_path:
                score += 2
            if "/analyses/" in file_path:
                score += 2
            if "/tests/" in file_path:
                score += 2
            
            # Score based on content keywords (dbt)
            dbt_keywords = ["ref(", "source(", "config(", "macro", "materialized", 
                           "incremental", "snapshot", "seeds", "dbt_utils"]
            for keyword in dbt_keywords:
                if keyword in code_snippet:
                    score += 2
            
            # Score based on content keywords (Snowflake)
            sf_keywords = ["snowflake", "warehouse", "database", "schema", "merge into", 
                          "copy into", "task", "stream", "pipe", "stage", "udf", "procedure"]
            for keyword in sf_keywords:
                if keyword in code_snippet:
                    score += 2
            
            # Add the scored result
            scored_results.append((score, result))
        
        # Sort by score (highest first) and take top results
        scored_results.sort(reverse=True, key=lambda x: x[0])
        return [result for score, result in scored_results[:max_results]]
    
    def _add_related_files(self, search_results: List[Dict[str, Any]], parsed_question: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Add related files to search results to provide better context"""
        try:
            enhanced_results = []
            
            for result in search_results:
                # Add the current result to enhanced results
                result_with_related_files = result.copy()
                result_with_related_files["related_files"] = []
                
                file_path = result.get("file_path", "")
                file_type = self._guess_file_type(file_path)
                
                # If this is a YAML schema file, find the related SQL models
                if file_type == "dbt_schema":
                    # Extract model names from the YAML content
                    model_names = self._extract_model_names_from_yaml(result.get("code_snippet", ""))
                    if model_names:
                        for model_name in model_names:
                            # Search for SQL files with matching model name
                            related_sql_files = self._find_related_sql_files(model_name, search_results)
                            result_with_related_files["related_files"].extend(related_sql_files)
                
                # If this is a SQL model file, find the related YAML schema
                elif file_type in ["dbt_model", "dbt_macro"]:
                    model_name = self._extract_model_name_from_path(file_path)
                    if model_name:
                        # Search for YAML files with this model defined
                        related_yaml_files = self._find_related_yaml_files(model_name, search_results)
                        result_with_related_files["related_files"].extend(related_yaml_files)
                
                enhanced_results.append(result_with_related_files)
            
            return enhanced_results
        
        except Exception as e:
            self.logger.error(f"Error adding related files: {e}", exc_info=True)
            return search_results
    
    def _extract_model_names_from_yaml(self, yaml_content: str) -> List[str]:
        """Extract model names from YAML schema content"""
        try:
            model_names = []
            
            # Try to parse YAML content
            try:
                import yaml
                yaml_data = yaml.safe_load(yaml_content)
                
                # Extract model names from 'models' section
                if yaml_data and 'models' in yaml_data:
                    for model in yaml_data['models']:
                        if 'name' in model:
                            model_names.append(model['name'])
            except:
                # Fallback to regex if YAML parsing fails
                models_section = re.search(r'models:.*?(?:version:|sources:|$)', yaml_content, re.DOTALL)
                if models_section:
                    model_matches = re.finditer(r'^\s*-\s*name:\s*(\w+)', models_section.group(0), re.MULTILINE)
                    for match in model_matches:
                        model_names.append(match.group(1))
            
            return model_names
        
        except Exception as e:
            self.logger.error(f"Error extracting model names from YAML: {e}", exc_info=True)
            return []
    
    def _find_related_sql_files(self, model_name: str, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find SQL files related to a model name"""
        related_files = []
        
        for result in search_results:
            file_path = result.get("file_path", "")
            
            # Skip if this isn't a SQL file
            if not file_path.endswith('.sql'):
                continue
            
            # Match by file name (model_name.sql)
            file_model_name = os.path.basename(file_path).split('.')[0]
            if file_model_name == model_name:
                related_files.append({
                    "file_path": file_path,
                    "relationship": "implements_model",
                    "relevance": 1.0
                })
                continue
            
            # Also check content for references to this model
            code_snippet = result.get("code_snippet", "")
            if f"ref('{model_name}')" in code_snippet or f'ref("{model_name}")' in code_snippet:
                related_files.append({
                    "file_path": file_path,
                    "relationship": "references_model",
                    "relevance": 0.8
                })
        
        return related_files
    
    def _find_related_yaml_files(self, model_name: str, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find YAML schema files related to a model name"""
        related_files = []
        
        for result in search_results:
            file_path = result.get("file_path", "")
            
            # Skip if this isn't a YAML file
            if not (file_path.endswith('.yml') or file_path.endswith('.yaml')):
                continue
            
            # Check content for this model name
            code_snippet = result.get("code_snippet", "")
            
            # Look for model definition pattern
            if f"name: {model_name}" in code_snippet:
                related_files.append({
                    "file_path": file_path,
                    "relationship": "defines_model",
                    "relevance": 1.0
                })
        
        return related_files
    
    def _extract_model_name_from_path(self, file_path: str) -> str:
        """Extract model name from file path"""
        # Simple approach: use the file name without extension
        file_name = os.path.basename(file_path)
        model_name = os.path.splitext(file_name)[0]
        return model_name
    
    def _create_search_query(self, parsed_question: Dict[str, Any]) -> str:
        """Create a search query from the parsed question with focus on dbt and Snowflake"""
        try:
            # Extract key information from parsed question
            rephrased_question = parsed_question.get("rephrased_question", "")
            question_type = parsed_question.get("question_type", "unknown")
            key_points = parsed_question.get("key_points", [])
            business_context = parsed_question.get("business_context", {})
            technical_context = parsed_question.get("technical_context", {})
            
            # Build search query
            query_parts = []
            
            # Add main question
            query_parts.append(rephrased_question)
            
            # Add key points (limited to first 3)
            if key_points:
                query_parts.extend(key_points[:3])
            
            # Add key entities from business context
            if business_context and "key_entities" in business_context:
                entities = business_context.get("key_entities", [])
                if entities and isinstance(entities, list):
                    query_parts.append(" ".join(entities[:5]))
            
            # Add data stack components
            if technical_context and "data_stack" in technical_context:
                data_stack = technical_context.get("data_stack", [])
                if data_stack and isinstance(data_stack, list):
                    query_parts.append(" ".join(data_stack[:3]))
            
            # Add relevant components
            if technical_context and "relevant_components" in technical_context:
                components = technical_context.get("relevant_components", [])
                if components and isinstance(components, list):
                    query_parts.append(" ".join(components[:3]))
            
            # Add dbt and Snowflake specific terms based on question type
            if question_type.lower() in ["technical", "hybrid"]:
                # Add dbt-specific terms
                query_parts.append("dbt model schema")
                
                # Add Snowflake-specific terms
                query_parts.append("snowflake sql")
                
                # Add file type constraints
                query_parts.append("filetype:sql OR filetype:yml OR filetype:yaml")
            
            # Join all parts and ensure we have proper syntax for GitHub search
            search_query = " ".join(query_parts)
            
            return search_query
            
        except Exception as e:
            self.logger.error(f"Error creating code search query: {e}")
            return parsed_question.get("rephrased_question", "")
    
    def _enhance_search_results(self, search_results: List[Dict[str, Any]], parsed_question: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Enhance search results with LLM analysis"""
        try:
            # Format results for LLM
            results_text = "\n\n".join([
                f"File: {r.get('file_path', '')}\n"
                f"Code:\n{r.get('code_snippet', '')}"
                for r in search_results
            ])

            # Create a more structured prompt
            prompt = f"""Analyze these code search results for a dbt/Snowflake implementation question:

Question: {parsed_question.get('original_question', '')}

Code Results:
{results_text}

Provide a JSON response with the following structure for each result:
{{
    "results": [
        {{
            "file_path": "path/to/file",
            "file_type": "dbt_model|dbt_schema|snowflake_script",
            "relevance_score": 0-10,
            "code_purpose": "Brief description of what this code does",
            "key_entities": ["entity1", "entity2"],
            "primary_logic": "Main transformation or business logic",
            "implementation_guidance": "How to adapt this for the current need",
            "development_steps": [
                "Step 1: ...",
                "Step 2: ..."
            ],
            "dependencies": ["dep1", "dep2"],
            "integration_considerations": "Integration notes",
            "testing_strategy": "Recommended tests"
        }}
    ]
}}

Focus on dbt and Snowflake best practices. Response must be valid JSON."""

            # Get LLM response
            response = self.llm.invoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)

            # Extract and validate JSON
            try:
                # First try to find JSON in code blocks
                json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    # Try to extract anything that looks like a JSON array
                    json_str = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', response_text).group(1)

                enhanced_results = json.loads(json_str)

                # Ensure we have a results array
                if isinstance(enhanced_results, dict) and 'results' in enhanced_results:
                    return enhanced_results['results']
                elif isinstance(enhanced_results, list):
                    return enhanced_results
                else:
                    raise ValueError("Invalid response structure")

            except (json.JSONDecodeError, AttributeError, ValueError) as e:
                self.logger.warning(f"Could not parse LLM response as JSON: {e}")
                # Return a formatted version of the raw results
                return [{
                    "file_path": result.get('file_path', ''),
                    "file_type": self._guess_file_type(result.get('file_path', '')),
                    "relevance_score": 5,  # Default medium relevance
                    "code_purpose": "Code found through search",
                    "key_entities": [],
                    "primary_logic": result.get('code_snippet', '')[:200] + "...",
                    "implementation_guidance": "Review code for applicability",
                    "development_steps": ["Review code", "Adapt as needed"],
                    "dependencies": [],
                    "integration_considerations": "Requires further analysis",
                    "testing_strategy": "Add appropriate dbt tests"
                } for result in search_results]

        except Exception as e:
            self.logger.error(f"Error enhancing search results: {e}", exc_info=True)
            return search_results

    def _determine_file_language(self, file_path: str) -> str:
        """Determine the language based on file extension"""
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == ".sql":
            return "sql"
        elif file_ext in [".yml", ".yaml"]:
            return "yaml"
        elif file_ext == ".py":
            return "python"
        elif file_ext == ".md":
            return "markdown"
        elif file_ext == ".json":
            return "json"
        else:
            return "text"

    def _validate_enhanced_results(self, enhanced_results: List[Dict[str, Any]], raw_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate and fix enhanced results with dbt/Snowflake specific fields"""
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
                "file_type": result.get("file_type", self._guess_file_type(raw_result.get("file_path", ""))),
                "data_layer": result.get("data_layer", "unknown"),
                "language": self._determine_file_language(raw_result.get("file_path", "")),
                "relevance_score": result.get("relevance_score", 0) / 10.0,  # Normalize to 0-1
                "code_snippet": raw_result.get("code_snippet", ""),
                "repo_info": raw_result.get("repo_info", {}),
                "code_purpose": result.get("code_purpose", ""),
                "key_entities": result.get("key_entities", []),
                "primary_logic": result.get("primary_logic", ""),
                "implementation_guidance": result.get("implementation_guidance", ""),
                "development_steps": result.get("development_steps", []),
                "dependencies": result.get("dependencies", []),
                "integration_considerations": result.get("integration_considerations", ""),
                "testing_strategy": result.get("testing_strategy", ""),
                "explanation": "This code may be relevant to implementing the requested feature."
            }
            
            # Add related files if present in raw result
            if "related_files" in raw_result:
                valid_result["related_files"] = raw_result["related_files"]
            
            valid_results.append(valid_result)
        
        return valid_results

    def _guess_file_type(self, file_path: str) -> str:
        """Guess the file type based on path and extension"""
        file_path = file_path.lower()
        file_ext = Path(file_path).suffix
        
        if file_ext == ".sql":
            if "/models/" in file_path:
                return "dbt_model"
            elif "/macros/" in file_path:
                return "dbt_macro"
            elif "/analyses/" in file_path:
                return "dbt_analysis"
            else:
                return "snowflake_script"
        elif file_ext in [".yml", ".yaml"]:
            if "/models/" in file_path or "schema.yml" in file_path:
                return "dbt_schema"
            else:
                return "dbt_config"
        else:
            return "other"

    def _format_raw_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format raw results when enhancement fails, with dbt/Snowflake focus"""
        formatted_results = []
        
        for result in results:
            file_path = result.get("file_path", "unknown_file")
            
            # Determine file type and language
            file_type = self._guess_file_type(file_path)
            language = self._determine_file_language(file_path)
            
            # Determine data layer based on path
            data_layer = "unknown"
            if "/staging/" in file_path:
                data_layer = "staging"
            elif "/intermediate/" in file_path:
                data_layer = "intermediate"
            elif "/marts/" in file_path or "/mart/" in file_path:
                data_layer = "marts"
            elif "/core/" in file_path:
                data_layer = "core"
            
            formatted_result = {
                "file_path": file_path,
                "file_type": file_type,
                "data_layer": data_layer,
                "language": language,
                "relevance_score": 0.5,  # Default medium relevance
                "code_snippet": result.get("code_snippet", ""),
                "repo_info": result.get("repo_info", {}),
                "code_purpose": f"This appears to be a {file_type} file that might be relevant to the question.",
                "key_entities": [],
                "primary_logic": "Analysis needed to determine the primary logic.",
                "implementation_guidance": "Review this code to determine if it can be adapted for your needs.",
                "development_steps": [],
                "dependencies": [],
                "integration_considerations": "Further analysis needed to determine integration requirements.",
                "testing_strategy": "Standard dbt testing practices should be applied.",
                "explanation": "This file was found based on keyword matching related to the question."
            }
            
            # Add related files if present
            if "related_files" in result:
                formatted_result["related_files"] = result["related_files"]
            
            formatted_results.append(formatted_result)
        
        return formatted_results

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

    def enhance_code_search_results(self, conversation_id: str) -> None:
        """Enhance code search results with additional context"""
        try:
            # Get conversation data
            conversation = self.db.get_conversation(conversation_id)
            if not conversation:
                self.logger.error(f"Conversation {conversation_id} not found")
                return

            # Handle architect_response parsing
            architect_response = conversation.get('architect_response', '')
            
            # Initialize default structure
            processed_response = {
                'response': '',
                'sections': {},
                'code_results': []
            }

            # Handle different types of architect_response
            if isinstance(architect_response, dict):
                processed_response.update(architect_response)
            elif isinstance(architect_response, str):
                try:
                    # Try to parse if it's a JSON string
                    json_data = json.loads(architect_response)
                    if isinstance(json_data, dict):
                        processed_response.update(json_data)
                    else:
                        processed_response['response'] = json_data
                except json.JSONDecodeError:
                    # If it's not valid JSON, use it as raw response
                    processed_response['response'] = architect_response
            
            # Get code results
            code_results = processed_response.get('code_results', [])
            if not code_results:
                self.logger.info("No code results to enhance")
                return

            # Process each code result
            enhanced_results = []
            for result in code_results:
                try:
                    # Enhance the result with additional context
                    enhanced_result = self._enhance_single_result(result)
                    enhanced_results.append(enhanced_result)
                except Exception as e:
                    self.logger.error(f"Error enhancing single result: {e}")
                    enhanced_results.append(result)  # Keep original if enhancement fails

            # Update the processed response with enhanced results
            processed_response['code_results'] = enhanced_results

            # Save back to database
            self.db.save_conversation(conversation_id, {
                'architect_response': processed_response
            })

            self.logger.info(f"Successfully enhanced code search results for conversation {conversation_id}")

        except Exception as e:
            self.logger.error(f"Error enhancing code search results: {e}", exc_info=True)
            raise

    def _enhance_single_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance a single code search result with additional context"""
        enhanced = result.copy()
        
        try:
            # Add file type information
            file_path = result.get('file_path', '')
            enhanced['file_type'] = self._guess_file_type(file_path)
            enhanced['language'] = self._determine_file_language(file_path)
            
            # Add data layer information if not present
            if 'data_layer' not in enhanced:
                enhanced['data_layer'] = self._determine_data_layer(file_path)
            
            # Add relevance score if not present
            if 'relevance_score' not in enhanced:
                enhanced['relevance_score'] = 0.5  # Default medium relevance
            
            # Add basic metadata if not present
            if 'code_purpose' not in enhanced:
                enhanced['code_purpose'] = f"This appears to be a {enhanced['file_type']} file that might be relevant."
            
            if 'key_entities' not in enhanced:
                enhanced['key_entities'] = []
            
            if 'implementation_guidance' not in enhanced:
                enhanced['implementation_guidance'] = "Review the code structure and adapt as needed."
            
        except Exception as e:
            self.logger.error(f"Error in _enhance_single_result: {e}")
        
        return enhanced