from typing import Dict, List, Any, Annotated, Sequence, TypedDict, Union, Optional, Tuple
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from pathlib import Path
import uuid
import logging
from sqlite3 import connect
from threading import Lock
import time
import traceback
from datetime import datetime
import json
import pprint
from src.tools import SearchTools
from src.db.database import ChatDatabase
from src.utils import ChromaDBManager
from src.dbt_tools import DbtTools, DbtToolsFactory
import re
from urllib.parse import urlparse
import os

# Set up logger
logger = logging.getLogger(__name__)

# Define state type
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "The messages in the conversation"]
    question_analysis: Annotated[Dict, "Parsed question analysis"]
    dbt_results: Annotated[Dict, "DBT model search results"]
    model_details: Annotated[Dict, "Detailed model information"]
    column_details: Annotated[Dict, "Column search results"]
    related_models: Annotated[Dict, "Related model information"]
    content_search: Annotated[Dict, "Content search results"]
    final_response: Annotated[str, "Final formatted response to the user"]

# Define structured outputs for question analysis
class QuestionAnalysis(BaseModel):
    question_type: str = Field(
        description="Type of question (MODEL_INFO, LINEAGE, DEPENDENCIES, CODE_ENHANCEMENT, DOCUMENTATION, DEVELOPMENT, GENERAL)"
    )
    entities: List[str] = Field(
        description="Key entities mentioned in the question (models, columns, etc.)"
    )
    search_terms: List[str] = Field(
        description="Terms to use for content searching",
        default_factory=list
    )
    intent: str = Field(
        description="Primary intent of the question"
    )
    rephrased_question: str = Field(
        description="Rephrased question for better search"
    )

class DataArchitectAgent:
    """
    Data Architect Agent that processes questions about data architecture,
    searches relevant sources, and provides comprehensive answers.
    """
    
    def __init__(self, repo_url: str = "", username: str = "", token: str = ""):
        """Initialize the data architect agent."""
        self.llm = ChatOllama(model="gemma3:latest")
        self.repo_url = repo_url
        self.username = username
        self.token = token
        
        # Initialize DBT tools only if repo_url is provided and not empty
        self.dbt_tools = None
        if repo_url and isinstance(repo_url, str) and repo_url.strip():
            try:
                # Validate GitHub URL format - support both standard and enterprise GitHub URLs
                parsed_url = urlparse(repo_url)
                
                # Check if it has a valid scheme and contains at least domain and path components
                if parsed_url.scheme not in ('https', 'http', 'git') or not parsed_url.netloc or not parsed_url.path:
                    raise ValueError("Invalid repository URL format")

                # Extract path parts to verify it follows owner/repo pattern
                path = parsed_url.path
                if path.endswith('.git'):
                    path = path[:-4]  # Remove .git extension
                
                path_parts = path.strip('/').split('/')
                if len(path_parts) < 2 or not path_parts[0] or not path_parts[1]:
                    raise ValueError("Repository URL must have at least owner/repository format")
                
                # Check if credentials are needed for private repos
                if not (username and token):
                    logger.warning("GitHub credentials not provided. Access to private repositories may be limited.")
                
                # Use the factory to create DBT tools
                self.dbt_tools = DbtToolsFactory.create_dbt_tools(repo_url, username, token)
                logger.info(f"Initialized DBT tools with repository: {repo_url}")
            except ValueError as e:
                logger.error(f"Invalid repository configuration: {str(e)}")
                self.dbt_tools = None
            except Exception as e:
                logger.error(f"Error initializing DBT tools: {str(e)}")
                self.dbt_tools = None
        else:
            logger.info("No valid repository URL provided, skipping DBT tools initialization")
        
        self.agent_graph = self._create_agent_graph()
    
    def process_question(self, question: str, conversation_id: str = None, thread_id: str = None, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process a user question through the agent workflow.
        
        Args:
            question: The user's question text
            conversation_id: Optional conversation ID for tracking
            thread_id: Optional thread ID for grouping conversations
            metadata: Optional metadata for the question
            
        Returns:
            A dictionary with the response and associated data
        """
        try:
            # Create initial message
            user_message = HumanMessage(content=question)
            
            # Initialize state
            initial_state = {
                "messages": [user_message],
                "question_analysis": {},
                "dbt_results": {},
                "model_details": {},
                "column_details": {},
                "related_models": {},
                "content_search": {},
                "final_response": ""
            }
            
            # Process the question through the agent graph
            logger.info(f"Processing question: {question[:100]}...")
            result = self.agent_graph.invoke(initial_state)
            
            # Get the final response
            response = result.get("final_response", "I couldn't generate a response. Please try again.")
            
            # Create metadata for the response
            response_metadata = self._create_response_metadata(result, conversation_id, thread_id, metadata)
            
            return {
                "response": response,
                "conversation_id": conversation_id or str(uuid.uuid4()),
                "processing_time": response_metadata.get("processing_time", 0),
                "question_type": result.get("question_analysis", {}).get("question_type", "GENERAL"),
                "dbt_results": response_metadata.get("dbt_results", {}),
                "relationship_results": response_metadata.get("relationship_results", {})
            }
            
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}", exc_info=True)
            return self._create_error_response(str(e), conversation_id)
    
    def _create_response_metadata(self, result: Dict[str, Any], conversation_id: str, thread_id: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Create metadata for the response."""
        # Extract relevant information from the result
        return {
            "processing_time": metadata.get("processing_time", 0) if metadata else 0,
            "dbt_results": {
                "models_found": len(result.get("dbt_results", {})),
                "columns_found": len(result.get("column_details", {})),
                "content_matches": len(result.get("content_search", {}))
            },
            "relationship_results": result.get("related_models", {}),
            "conversation_id": conversation_id or str(uuid.uuid4()),
            "thread_id": thread_id
        }
    
    def _create_error_response(self, error_message: str, conversation_id: str = None) -> Dict[str, Any]:
        """Create an error response when processing fails."""
        return {
            "response": f"I encountered an error while processing your question: {error_message}. Please try again or contact support if the issue persists.",
            "conversation_id": conversation_id or str(uuid.uuid4()),
            "processing_time": 0,
            "question_type": "ERROR",
            "error": error_message,
            "status": "error"
        }
    
    def _create_agent_graph(self) -> StateGraph:
        """Create the agent workflow graph."""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("parse_question", self._parse_question)
        workflow.add_node("search_models", self._search_models)
        workflow.add_node("search_columns", self._search_columns)
        workflow.add_node("get_model_details", self._get_model_details)
        workflow.add_node("search_content", self._search_content)
        workflow.add_node("generate_response", self._generate_response)
        
        # Define routing function for dynamic routing based on question type
        def route_by_question_type(state: AgentState) -> str:
            question_type = state["question_analysis"].get("question_type", "GENERAL")
            entities = state["question_analysis"].get("entities", [])
            intent = state["question_analysis"].get("intent", "")
            
            logger.info(f"Routing based on question type: {question_type}")
            
            # Check for file paths in entities
            file_paths = [entity for entity in entities if "/" in entity or ".sql" in entity]
            
            # Model information questions
            if question_type == "MODEL_INFO" and entities:
                logger.info("Routing to search_models for MODEL_INFO question")
                return "search_models"
                
            # Lineage and dependency questions
            elif question_type in ["LINEAGE", "DEPENDENCIES"] and entities:
                logger.info("Routing to get_model_details for LINEAGE/DEPENDENCIES question")
                return "get_model_details"
            
            # Code Enhancement questions with file paths should ALWAYS go to get_model_details first
            elif question_type == "CODE_ENHANCEMENT" and file_paths:
                logger.info(f"Routing to get_model_details for CODE_ENHANCEMENT with file paths: {file_paths}")
                return "get_model_details"
            
            # Development and code enhancement questions typically need both model and column details
            elif question_type in ["DEVELOPMENT", "CODE_ENHANCEMENT"] and entities:
                # Look for column-specific entities for column operations
                column_entities = [entity for entity in entities if "_" in entity or "." in entity]
                
                if column_entities:
                    logger.info(f"Routing to search_columns for {question_type} question with column entities")
                    return "search_columns"
                elif file_paths:
                    logger.info(f"Routing to get_model_details for {question_type} question with file paths")
                    return "get_model_details"
                else:
                    logger.info(f"Routing to search_models for {question_type} question with model entities")
                    return "search_models"
                
            # If there are potential column names in the entities
            elif any(len(entity.split('.')) == 2 for entity in entities) or any("_" in entity for entity in entities):
                logger.info("Routing to search_columns for question with column references")
                return "search_columns"
                
            # For all other cases, try content search
            else:
                logger.info("Routing to search_content as fallback")
                return "search_content"
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "parse_question",
            route_by_question_type,
            {
                "search_models": "search_models",
                "get_model_details": "get_model_details",
                "search_columns": "search_columns",
                "search_content": "search_content",
            }
        )
        
        # Add remaining edges
        workflow.add_edge("search_models", "generate_response")
        workflow.add_edge("search_columns", "generate_response")
        workflow.add_edge("get_model_details", "generate_response")
        workflow.add_edge("search_content", "generate_response")
        
        # Set entry point
        workflow.set_entry_point("parse_question")
        
        return workflow.compile()

    def _parse_question(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Parse the user's question to determine intent, entities, and search terms."""
        try:
            # Get the last message, handling both dict and object formats
            last_message = ""
            if "messages" in state:
                messages = state["messages"]
                if isinstance(messages, list) and messages:
                    last_msg = messages[-1]
                    # Handle both dict-like and object formats
                    if hasattr(last_msg, 'content'):
                        last_message = last_msg.content
                    elif isinstance(last_msg, dict) and 'content' in last_msg:
                        last_message = last_msg['content']
                elif isinstance(messages, str):
                    # If messages is just a string (direct question)
                    last_message = messages
            
            # Fallback if we couldn't get the message
            if not last_message:
                logger.warning("Could not extract message content from state. Using fallback.")
                if isinstance(state, dict) and "input" in state:
                    last_message = state["input"]
                else:
                    last_message = "What are the models in this project?"
            
            logger.info(f"Processing question: {last_message}")
            
            # Look for column-specific patterns
            column_patterns = self._extract_column_patterns(last_message)
            
            # Create prompt for question analysis
            prompt = f"""
            Analyze the following question about DBT models and data architecture. Determine:
            
            1. Question Type - Select ONE from:
               - MODEL_INFO: Questions that ask about what a model does, its structure, purpose, or columns. Examples: "What does model X do?", "How is model Y structured?", "What columns are in model Z?"
               
               - LINEAGE: Questions about relationships and how data flows through models. Examples: "What models feed into X?", "What is the lineage of Y?", "How does data flow from A to B?"
               
               - DEPENDENCIES: Questions about what models depend on others or impact analysis. Examples: "What depends on model X?", "What would break if I change Y?", "What are the dependencies of Z?"
               
               - CODE_ENHANCEMENT: Questions about improving, modifying, or optimizing existing code. Examples: "How can I make this query faster?", "How do I modify X to include Y?", "How can I delete/add/change column Z?"
               
               - DOCUMENTATION: Questions about documenting models or generating documentation. Examples: "How should I document X?", "Can you create documentation for Y?", "What should be in the docs for Z?"
               
               - DEVELOPMENT: Questions about implementing new features or models. Examples: "How do I create a new model for X?", "How do I implement Y?", "Can you help me develop Z?", "Give me a script to delete column X"
               
               - GENERAL: Other types of questions that don't fit the above categories.
            
            2. Key Entities - Extract ALL relevant entities:
               - Model names (e.g., "orders", "customers", "financial_metrics")
               - Column names (e.g., "order_id", "customer_name", "total_value")
               - Combined entities (e.g., "orders.order_id", "customers.email")
               - Any specific tables, schemas, or datasets mentioned
               - File paths if mentioned (e.g., "models/marts/core/customers.sql")
            
            3. Search Terms - Key words to search for in code/docs:
               - Technical terms (e.g., "materialization", "incremental", "full refresh")
               - Business concepts (e.g., "revenue calculation", "user retention")
               - Any specific code patterns or logic mentioned
               - Column calculation terms (e.g., "gross_amount", "net_sales", "discount")
               - For code changes, include terms like "add", "delete", "modify", "change", "alter"
            
            4. Primary Intent - The core goal of the question
               - For development/enhancement questions, include the action intent (add/remove/modify)
            
            5. Rephrased Question - A clear, searchable version of the question
            
            Question: {last_message}
            
            {f"Detected Column Patterns: {', '.join(column_patterns)}" if column_patterns else ""}
            
            IMPORTANT CLASSIFICATION GUIDELINES:
            - If the question asks how to delete, add, modify, or implement something, it's likely CODE_ENHANCEMENT or DEVELOPMENT, not MODEL_INFO
            - If the question asks for a script, code changes, or implementation, it's likely DEVELOPMENT
            - If the question asks to optimize or improve existing code, it's CODE_ENHANCEMENT
            - If the question only asks what a model/column does, then it's MODEL_INFO
            
            Return ONLY a JSON object with these fields:
            {{
                "question_type": "MODEL_INFO", // One type from the list above
                "entities": ["entity1", "entity2"], // All identified entities
                "search_terms": ["term1", "term2"], // Terms for content search
                "intent": "intent description",
                "rephrased_question": "rephrased version"
            }}
            
            Be precise and comprehensive in identifying entities and search terms, as they will be used to search DBT code repositories.
            If column names or calculations are mentioned (like "order_amount", "gross_sales"), be sure to include them as entities or search terms.
            """
            
            # Create messages list for the LLM call
            prompt_messages = [
                SystemMessage(content="You are an expert at analyzing questions about data architecture and DBT models."),
                HumanMessage(content=prompt)
            ]
            
            # Get analysis from LLM
            response = self._safe_llm_call(prompt_messages)
            
            # Clean the response to ensure it's valid JSON
            clean_response = response.strip()
            
            # Extract JSON from the response, handling various formats
            json_str = self._extract_json_from_response(clean_response)
            
            try:
                # Parse the response
                analysis = QuestionAnalysis.parse_raw(json_str)
                
                # Add any detected column patterns that might have been missed
                if column_patterns:
                    existing_entities = set(analysis.entities)
                    for pattern in column_patterns:
                        if pattern not in existing_entities:
                            analysis.entities.append(pattern)
                    
                    existing_terms = set(analysis.search_terms)
                    for pattern in column_patterns:
                        if pattern not in existing_terms:
                            analysis.search_terms.append(pattern)
                
                # Log the question analysis for debugging
                logger.info(f"Question classified as: {analysis.question_type}")
                logger.info(f"Entities identified: {', '.join(analysis.entities)}")
                logger.info(f"Search terms: {', '.join(analysis.search_terms)}")
                logger.info(f"Intent: {analysis.intent}")
                
                # Apply fallback classification rules for common patterns
                original_type = analysis.question_type
                
                # Keywords that strongly indicate specific question types
                development_keywords = ["create", "implement", "delete", "add", "remove", "script", "write", "build"]
                enhancement_keywords = ["change", "modify", "improve", "optimize", "update", "alter", "fix", "enhance"]
                
                # Check for development and code enhancement keywords in the original question
                lowercase_question = last_message.lower()
                
                # Check for development patterns
                if original_type not in ["DEVELOPMENT", "CODE_ENHANCEMENT"]:
                    if any(keyword in lowercase_question for keyword in development_keywords):
                        analysis.question_type = "DEVELOPMENT"
                        logger.info(f"Reclassified question from {original_type} to DEVELOPMENT based on keywords")
                    elif any(keyword in lowercase_question for keyword in enhancement_keywords):
                        analysis.question_type = "CODE_ENHANCEMENT"
                        logger.info(f"Reclassified question from {original_type} to CODE_ENHANCEMENT based on keywords")
                
                # Update state
                state["question_analysis"] = analysis.dict()
                
                # Initialize other state fields
                state["dbt_results"] = {}
                state["model_details"] = {}
                state["column_details"] = {}
                state["related_models"] = {}
                state["content_search"] = {}
                
                return state
            except Exception as parsing_error:
                logger.error(f"Error parsing LLM response as JSON: {parsing_error}")
                logger.error(f"Raw response: {json_str}")
                
                # Attempt to extract basic info from non-JSON response by direct pattern matching
                fallback_analysis = self._create_fallback_analysis(last_message, clean_response, column_patterns)
                
                logger.info(f"Using fallback analysis with question type: {fallback_analysis['question_type']}")
                state["question_analysis"] = fallback_analysis
                
                # Initialize other state fields
                state["dbt_results"] = {}
                state["model_details"] = {}
                state["column_details"] = {}
                state["related_models"] = {}
                state["content_search"] = {}
                
                return state
            
        except Exception as e:
            logger.error(f"Error parsing question: {str(e)}")
            # Ensure we have a fallback message
            last_message = "What are the main models in this project?"
            if "messages" in state and isinstance(state["messages"], list) and state["messages"]:
                try:
                    last_msg = state["messages"][-1]
                    if hasattr(last_msg, 'content'):
                        last_message = last_msg.content
                    elif isinstance(last_msg, dict) and 'content' in last_msg:
                        last_message = last_msg['content']
                except:
                    pass
            
            state["question_analysis"] = {
                "question_type": "GENERAL",
                "entities": [],
                "search_terms": [],
                "intent": "general inquiry",
                "rephrased_question": last_message
            }
            
            # Initialize other state fields
            state["dbt_results"] = {}
            state["model_details"] = {}
            state["column_details"] = {}
            state["related_models"] = {}
            state["content_search"] = {}
            
            return state

    def _extract_column_patterns(self, text: str) -> List[str]:
        """Extract potential column name patterns from text."""
        # Pattern for typical column names (snake_case identifiers)
        column_pattern = re.compile(r'\b[a-z][a-z0-9_]*(?:_[a-z0-9]+)+\b')
        potential_columns = column_pattern.findall(text.lower())
        
        # Filter out common words that match the pattern but aren't columns
        stop_words = ['for_the', 'in_the', 'with_the', 'how_to', 'what_is', 'such_as']
        columns = [col for col in potential_columns if col not in stop_words]
        
        # Also look for dotted notation (table.column)
        dotted_pattern = re.compile(r'\b([a-z][a-z0-9_]*)\.([a-z][a-z0-9_]*)\b')
        dotted_matches = dotted_pattern.findall(text.lower())
        
        # Add both the full reference and just the column name
        for table, column in dotted_matches:
            if f"{table}.{column}" not in columns:
                columns.append(f"{table}.{column}")
            if column not in columns:
                columns.append(column)
        
        return columns

    def _search_models(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Search for models based on entities in the question."""
        try:
            # Get entities from question analysis
            entities = state.get("question_analysis", {}).get("entities", [])
            
            # Skip if no entities or no DBT tools
            if not entities or not self.dbt_tools:
                logger.warning("No entities to search for models or no DBT tools available")
                state["dbt_results"] = []
                return state
            
            # Initialize results structure
            all_results = []
            
            # Search for each entity
            for entity in entities:
                logger.info(f"Searching for model entity: {entity}")
                
                # Check if the entity looks like a file path
                if '/' in entity or entity.endswith('.sql'):
                    logger.info(f"Searching for file path entity: {entity}")
                    try:
                        # Use search_file_path instead of search_by_file_path
                        path_results = self.dbt_tools.search_file_path(entity)
                        
                        # Process path results
                        if path_results:
                            for result in path_results:
                                # Convert result to dictionary if it's not already
                                if not isinstance(result, dict):
                                    try:
                                        result_dict = result.__dict__.copy() if hasattr(result, '__dict__') else {"model_name": str(result)}
                                        result_dict["match_type"] = "path"
                                        all_results.append(result_dict)
                                    except Exception as e:
                                        logger.error(f"Error converting path result to dict: {str(e)}")
                                else:
                                    result["match_type"] = "path"
                                    all_results.append(result)
                            
                            logger.info(f"Found {len(path_results)} models for path entity: {entity}")
                    except Exception as e:
                        logger.error(f"Error searching file path: {str(e)}")
                
                # Search as a model name
                try:
                    model_results = self.dbt_tools.search_model(entity, search_mode="parse")
                    
                    # Process model results
                    if model_results:
                        for result in model_results:
                            # Convert result to dictionary if it's not already
                            if not isinstance(result, dict):
                                try:
                                    result_dict = result.__dict__.copy() if hasattr(result, '__dict__') else {"model_name": str(result)}
                                    result_dict["match_type"] = "model"
                                    all_results.append(result_dict)
                                except Exception as e:
                                    logger.error(f"Error converting model result to dict: {str(e)}")
                            else:
                                result["match_type"] = "model"
                                all_results.append(result)
                        
                        logger.info(f"Found {len(model_results)} models for entity: {entity}")
                except Exception as e:
                    logger.error(f"Error searching model: {str(e)}")
            
            # Deduplicate results by file path
            unique_results = []
            seen_paths = set()
            
            for result in all_results:
                file_path = result.get("file_path", "")
                model_name = result.get("model_name", "")
                
                # Create a unique key for this result
                unique_key = f"{file_path}|{model_name}"
                
                if unique_key and unique_key not in seen_paths:
                    seen_paths.add(unique_key)
                    unique_results.append(result)
            
            # Update state with the results
            logger.info(f"Found {len(unique_results)} unique model results")
            state["dbt_results"] = unique_results
            
            return state
            
        except Exception as e:
            logger.error(f"Error searching for models: {str(e)}")
            state["dbt_results"] = []
            return state
    
    def _search_columns(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Search for columns in DBT models."""
        try:
            analysis = state["question_analysis"]
            question = state["messages"][-1].content
            entities = analysis.get("entities", [])
            search_terms = analysis.get("search_terms", [])
            
            results = {}
            
            # Only proceed with DBT search if we have DBT tools initialized
            if self.dbt_tools:
                # Process column name extraction
                column_names = []
                
                # First check entities that look like column names (containing underscores or dots)
                for entity in entities:
                    if '_' in entity or '.' in entity:
                        column_names.append(entity)
                
                # If no column names found in entities, try to extract from the question
                if not column_names:
                    column_names = self._extract_column_patterns(question)
                
                # Add relevant search terms that might be parts of column names
                potential_column_terms = [term for term in search_terms 
                                       if len(term) > 3 
                                       and term.lower() not in COMMON_STOP_WORDS
                                       and any(w in term.lower() for w in ['amount', 'total', 'sum', 'price', 'gross', 'net', 'tax', 'discount', 'sales'])]
                                       
                for term in potential_column_terms:
                    if term not in column_names:
                        column_names.append(term)
                
                # Log column names found for debugging
                logger.info(f"Searching for columns: {', '.join(column_names)}")
                
                # Search for each column
                for column_name in column_names:
                    # For dotted column names (e.g., order_item_summary.item_discount_amount)
                    # search for both the full name and just the column part
                    search_names = [column_name]
                    if '.' in column_name:
                        parts = column_name.split('.')
                        if len(parts) == 2:
                            table_name, col_name = parts
                            # Add the column name without the table prefix
                            if col_name not in search_names:
                                search_names.append(col_name)
                            # Also search for the table name
                            if table_name not in search_names:
                                search_names.append(table_name)
                    
                    # Search for each variation of the column name
                    for search_name in search_names:
                        logger.info(f"Searching for column: {search_name}")
                        column_results = self.dbt_tools.search_column(search_name)
                        
                        if column_results:
                            logger.info(f"Found {len(column_results)} results for column '{search_name}'")
                            
                            # For each match, extract calculation logic if possible
                            enriched_results = []
                            model_names = set()
                            
                            for result in column_results:
                                # Track which models contain this column
                                if result.model_name:
                                    model_names.add(result.model_name)
                                
                                # Convert to dictionary for easier manipulation
                                result_dict = result.__dict__.copy()
                                
                                # If we have a calculation context, try to clean it up for better display
                                if hasattr(result, 'calculation') and result.calculation:
                                    # Add a clean version of the calculation
                                    result_dict["clean_calculation"] = self._format_calculation(result.calculation)
                                
                                enriched_results.append(result_dict)
                            
                            # For each model that contains this column, try to extract the calculation
                            for model_name in model_names:
                                # Try to get explicit calculation definition
                                calculation = self.dbt_tools.extract_specific_calculation(model_name, search_name)
                                if calculation:
                                    # Add this as an additional result with type "explicit_calculation"
                                    enriched_results.append({
                                        "model_name": model_name,
                                        "match_type": "explicit_calculation",
                                        "column_name": search_name,
                                        "calculation": calculation,
                                        "clean_calculation": self._format_calculation(calculation)
                                    })
                            
                            results[search_name] = enriched_results
                    
                    # If we've found results for any variation, add them to the original column name
                    if not column_name in results and any(name in results for name in search_names):
                        for name in search_names:
                            if name in results:
                                results[column_name] = results[name]
                                break
                
                # If no direct column results, try content search
                if not results:
                    logger.info("No column results found, trying content search")
                    
                    # Try to search for column-like terms in code content
                    for column_name in column_names:
                        # Look for both the full name and parts
                        content_results = self.dbt_tools.search_content(column_name)
                        
                        if content_results:
                            logger.info(f"Found {len(content_results)} content matches for '{column_name}'")
                            results[f"content:{column_name}"] = [result.__dict__ for result in content_results]
                        else:
                            # Try with each meaningful part of the column name
                            parts = column_name.split('_')
                            for part in parts:
                                if len(part) > 3 and part.lower() not in COMMON_STOP_WORDS:
                                    part_results = self.dbt_tools.search_content(part)
                                    if part_results:
                                        logger.info(f"Found {len(part_results)} content matches for part '{part}'")
                                        results[f"content:{part}"] = [result.__dict__ for result in part_results]
                                        break
            else:
                results = self._get_dbt_tools_error()
            
            # Update state
            state["column_details"] = results
            
            # If we found column results, get model details for those models
            if results and not any("error" in k for k in results.keys()):
                self._get_related_model_info(state, results)
            
            return state
            
        except Exception as e:
            logger.error(f"Error searching columns: {str(e)}")
            state["column_details"] = {
                "error": str(e),
                "status": "error",
                "message": f"Error searching columns: {str(e)}"
            }
            return state
    
    def _format_calculation(self, calculation: str) -> str:
        """Format a calculation string for better readability."""
        # Remove extra whitespace
        calculation = re.sub(r'\s+', ' ', calculation.strip())
        
        # Handle case when the calculation has SQL keywords
        if any(keyword in calculation.lower() for keyword in ['select', 'from', 'where', 'join']):
            # This is likely a larger SQL snippet, return as is
            return calculation
        
        # Clean up common patterns:
        # Remove leading commas
        calculation = re.sub(r'^\s*,\s*', '', calculation)
        
        # If it's an 'as column_name' format, focus on the calculation part
        if ' as ' in calculation.lower():
            parts = re.split(r'\s+as\s+', calculation, flags=re.IGNORECASE)
            if len(parts) == 2:
                calculation = parts[0].strip()
        
        # Remove outer parentheses if they wrap the whole expression
        if calculation.startswith('(') and calculation.endswith(')'):
            inner = calculation[1:-1].strip()
            if inner.count('(') == inner.count(')'):  # Balanced parentheses inside
                calculation = inner
        
        return calculation
    
    def _get_related_model_info(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Get related model information based on search results."""
        try:
            # Get search results
            dbt_results = state.get("dbt_results", [])
            model_details = state.get("model_details", {})
            
            # Initialize related models container
            related_models = {}
            
            # Process each found model for dependencies
            # Handle cases where dbt_results is a list (newer implementation) or dict (older)
            if isinstance(dbt_results, list):
                logger.info(f"Processing related models for {len(dbt_results)} model results in list format")
                
                for result in dbt_results:
                    model_name = result.get("model_name", "")
                    file_path = result.get("file_path", "")
                    
                    if not model_name and not file_path:
                        continue
                    
                    if model_name not in related_models:
                        related_models[model_name] = {"upstream": [], "downstream": []}
                    
                    # Get dependencies for this model
                    try:
                        if model_name and self.dbt_tools:
                            dependencies = self.dbt_tools.find_related_models(model_name)
                            
                            if dependencies:
                                # Add upstream models
                                if "upstream" in dependencies and dependencies["upstream"]:
                                    related_models[model_name]["upstream"] = dependencies["upstream"]
                                
                                # Add downstream models
                                if "downstream" in dependencies and dependencies["downstream"]:
                                    related_models[model_name]["downstream"] = dependencies["downstream"]
                    except Exception as e:
                        logger.warning(f"Error getting dependencies for {model_name}: {str(e)}")
            
            # Handle model details if available
            for model_name, details in model_details.items():
                if model_name in related_models:
                    continue  # Skip if already processed
                
                if "dependencies" in details:
                    related_models[model_name] = details["dependencies"]
            
            # Update state with related models
            state["related_models"] = related_models
            
            logger.info(f"Gathered related model information for {len(related_models)} models")
            return state
        
        except Exception as e:
            logger.error(f"Error getting related model info: {str(e)}")
            state["related_models"] = {}
            return state
    
    def _search_content(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Search for content in DBT models."""
        try:
            analysis = state["question_analysis"]
            question = state["messages"][-1].content
            entities = analysis.get("entities", [])
            search_terms = analysis.get("search_terms", [])
            
            results = {}
            
            # Only proceed with DBT search if we have DBT tools initialized
            if self.dbt_tools:
                # Add sales-related terms that might be relevant based on common patterns
                sales_terms = []
                if any('sales' in term.lower() or 'amount' in term.lower() or 'discount' in term.lower() 
                       or 'tax' in term.lower() or 'gross' in term.lower() or 'net' in term.lower() 
                       for term in search_terms + entities):
                    sales_terms = ['sales', 'amount', 'discount', 'tax', 'gross', 'net', 'item']
                
                # Filter out common stop words and short terms
                filtered_terms = [term for term in search_terms 
                               if len(term) > 3 and term.lower() not in COMMON_STOP_WORDS]
                
                # Add entities as search terms if they seem relevant
                filtered_terms.extend([entity for entity in entities 
                                   if len(entity) > 3 and entity.lower() not in COMMON_STOP_WORDS
                                   and entity not in filtered_terms])
                
                # Add the sales terms if relevant
                for term in sales_terms:
                    if term not in filtered_terms:
                        filtered_terms.append(term)
                
                # If no usable search terms, extract from question directly
                if not filtered_terms:
                    # Look for keywords in the question that might be relevant
                    question_words = question.lower().split()
                    relevant_keywords = [word for word in question_words 
                                      if len(word) > 3 
                                      and word not in COMMON_STOP_WORDS
                                      and any(c.isalpha() for c in word)]  # Ensure it has letters
                    filtered_terms.extend(relevant_keywords)
                
                # Get the most specific terms first (longer and less common terms)
                filtered_terms.sort(key=lambda x: len(x), reverse=True)
                
                # Log what we're searching for
                logger.info(f"Content search using terms: {', '.join(filtered_terms[:5])}" + 
                           (f" and {len(filtered_terms) - 5} more" if len(filtered_terms) > 5 else ""))
                
                # Search for specific multi-word phrases first (more precise)
                for entity in entities:
                    if ' ' in entity and len(entity) > 8:  # Only longer phrases
                        content_results = self.dbt_tools.search_content(entity)
                        if content_results:
                            logger.info(f"Found {len(content_results)} results for phrase '{entity}'")
                            results[entity] = [result.__dict__ for result in content_results]
                
                # Search for each term individually
                for term in filtered_terms[:5]:  # Limit to top 5 terms for efficiency
                    if term not in results:  # Skip if already found
                        content_results = self.dbt_tools.search_content(term)
                        if content_results:
                            logger.info(f"Found {len(content_results)} results for term '{term}'")
                            results[term] = [result.__dict__ for result in content_results]
                
                # Special search for specific column calculations
                calculation_terms = ['gross_item_sales_amount', 'item_discount_amount', 'item_tax_amount', 'net_item_sales_amount']
                if any(term in question.lower() for term in ['calculation', 'formula', 'compute', 'derive']):
                    for calc_term in calculation_terms:
                        if calc_term not in results and calc_term.lower() in question.lower():
                            logger.info(f"Searching for calculation term '{calc_term}'")
                            calc_results = self.dbt_tools.search_content(calc_term)
                            if calc_results:
                                logger.info(f"Found {len(calc_results)} results for calculation '{calc_term}'")
                                results[calc_term] = [result.__dict__ for result in calc_results]
                
                # If no results, try path-based search
                if not results:
                    logger.info("No content results found, trying path-based search")
                    for term in filtered_terms[:3]:
                        path_results = self.dbt_tools.search_file_path(term)
                        if path_results:
                            logger.info(f"Found {len(path_results)} path results for '{term}'")
                            results[f"path:{term}"] = [result.__dict__ for result in path_results]
            else:
                results = self._get_dbt_tools_error()
            
            # Update state
            state["content_search"] = results
            
            return state
            
        except Exception as e:
            logger.error(f"Error searching content: {str(e)}")
            state["content_search"] = {
                "error": str(e),
                "status": "error",
                "message": f"Error searching content: {str(e)}"
            }
            return state
    
    def _get_dbt_tools_error(self) -> Dict[str, Any]:
        """Get a standardized error for when DBT tools are not available."""
        return {
            "status": "no_dbt_tools",
            "message": "Unable to search DBT models. Please configure a GitHub repository containing your DBT models.",
            "setup_required": True,
            "setup_steps": [
                "Configure a GitHub repository in the settings",
                "Ensure the repository contains your DBT models",
                "Provide valid GitHub credentials"
            ]
        }

    def _generate_response(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate response using LLM based on search results."""
        try:
            # Get parsed question
            question_analysis = state.get("question_analysis", {})
            question = question_analysis.get("original_question", "")
            question_type = question_analysis.get("question_type", "GENERAL")
            entities = question_analysis.get("entities", [])
            
            # Log core information
            logger.info(f"Generating response for {question_type} question with {len(entities)} entities")
            
            # Get search results
            search_results = self._format_results_for_prompt(state)
            
            # Check if we need more model-specific information
            model_details = state.get("model_details", {})
            has_model_content = False
            model_paths = []
            model_content = ""
            model_file_path = ""
            
            # Check if we have model details and content
            if model_details:
                for model_name, details in model_details.items():
                    model_info = details.get("model", {})
                    if isinstance(model_info, dict) and model_info.get("content"):
                        has_model_content = True
                        model_file_path = model_info.get("file_path", "unknown")
                        model_paths.append(model_file_path)
                        
                        # For CODE_ENHANCEMENT, get the specific model content
                        if question_type == "CODE_ENHANCEMENT":
                            model_content = model_info.get("content", "")
                            break
            
            # If we have file paths but no model details, try to get model information using the new method
            if (not has_model_content or not model_content) and entities:
                # Try using our new helper method for each entity
                for entity in entities:
                    logger.info(f"Using _find_model_content for entity: {entity}")
                    found, content, file_path = self._find_model_content(entity)
                    if found:
                        has_model_content = True
                        model_content = content
                        model_file_path = file_path
                        model_paths = [file_path]
                        logger.info(f"Found model content using _find_model_content for {entity} at {file_path}")
                        break
                
                # If we still don't have content, extract paths from entities as before
                if not has_model_content:
                    # Extract paths from entities - they might be file paths or file names
                    for entity in entities:
                        if '/' in entity or entity.endswith('.sql'):
                            model_paths.append(entity)
                    
                    # Try the original method as fallback
                    if model_paths and self.dbt_tools:
                        logger.info(f"Searching for model content from entity paths: {model_paths}")
                        try:
                            # Use output mode for precise matching
                            for path in model_paths:
                                model_name = path
                                # Remove .sql extension if present
                                if model_name.endswith('.sql'):
                                    model_name = model_name[:-4]
                                
                                # Get the base name if it's a path
                                if '/' in model_name:
                                    model_name = os.path.basename(model_name)
                                
                                results = self.dbt_tools.search_model(model_name, search_mode="output")
                                if results and len(results) > 0:
                                    # Take the first result as our model
                                    result = results[0]
                                    # Use 'content' attribute instead of 'sql_content'
                                    if hasattr(result, 'content') and result.content:
                                        has_model_content = True
                                        model_content = result.content
                                        model_file_path = result.file_path if hasattr(result, 'file_path') else path
                                        logger.info(f"Found model content for {model_name}")
                                        break
                                    elif isinstance(result, dict) and 'content' in result and result['content']:
                                        has_model_content = True
                                        model_content = result['content']
                                        model_file_path = result.get('file_path', path)
                                        logger.info(f"Found model content for {model_name}")
                                        break
                        except Exception as e:
                            logger.warning(f"Error getting model content: {str(e)}")
                            
                    # If we still don't have content, try direct file access
                    if not has_model_content and model_paths and self.dbt_tools:
                        logger.info(f"Trying direct file access for paths: {model_paths}")
                        try:
                            for path in model_paths:
                                content = self.dbt_tools.get_file_content(path)
                                if content:
                                    has_model_content = True
                                    model_content = content
                                    model_file_path = path
                                    logger.info(f"Found content via direct file access for {path}")
                                    break
                        except Exception as e:
                            logger.warning(f"Error accessing file content: {str(e)}")
            
            # Get the request type and prepare prompt instructions
            instructions = self._get_instructions_for_type(question_type, question)
            
            # Add specific model content for CODE_ENHANCEMENT
            if question_type == "CODE_ENHANCEMENT" and model_content:
                # Add special DBT-specific instructions for code enhancement
                dbt_enhancement_instructions = f"""
                # DBT-Specific Requirements
                - Your response MUST preserve DBT's config blocks, doc blocks, and jinja templating
                - The enhanced model MUST maintain exact compatibility with existing references
                - If improving an incremental model, maintain the incremental logic
                - Provide specific Snowflake/DBT optimization techniques
                - Include complete enhanced model code, not just the changes
                
                Analyze this EXACT model code:
                
                ```sql
                {model_content}
                ```
                """.format(model_content=model_content)
                
                instructions += "\n" + dbt_enhancement_instructions
            elif (question_type == "DOCUMENTATION" or question_type == "MODEL_INFO") and model_content:
                # Add special DBT-specific instructions for documentation
                dbt_documentation_instructions = """
                # DBT-Specific Documentation Requirements
                - The documentation must accurately describe THIS EXACT model
                - Focus on the purpose, structure, and key columns of the model
                - Include details about the model's materialization strategy (especially if incremental)
                - Document the primary/unique keys and their significance
                - Explain important joins and relationships with other models
                
                Analyze this EXACT model code:
                
                ```sql
                {model_content}
                ```
                """.format(model_content=model_content)
                
                instructions += "\n" + dbt_documentation_instructions
            
            # Create messages with system message, context, and question
            system_message = """
            You are a helpful data architect assistant that provides clear, accurate responses to DBT-related questions.
            Your expertise is in analyzing and optimizing data modeling code and helping data teams work effectively with DBT.
            
            When generating responses:
            - Focus on clarity and accuracy
            - Provide specific, actionable recommendations
            - Include SQL code examples where relevant
            - Explain your reasoning
            - Cite specific documentation or best practices when available
            
            For code enhancement or optimization, focus specifically on:
            1. Performance improvements 
            2. Maintainability enhancements
            3. Following DBT best practices
            4. Providing the complete enhanced code
            
            For documentation, focus on:
            1. Clear explanations of purpose and functionality
            2. Details about columns and their uses
            3. Information about dependencies and relationships
            4. Business context if available
            """
            
            # Create the messages with detailed context
            messages = [
                SystemMessage(content=system_message),
                HumanMessage(content=f"""
                # Question
                {question}
                
                # Question Type
                {question_type}
                
                # Search Results
                {search_results}
                
                # Instructions
                {instructions}
                
                Please provide a comprehensive, well-structured response based on the provided information and search results.
                """)
            ]
            
            # Call the LLM to generate response
            response = self._safe_llm_call(messages)
            
            # For CODE_ENHANCEMENT, validate the response
            if question_type == "CODE_ENHANCEMENT" and model_content:
                response = self._validate_code_enhancement_response(response, model_content, model_paths[0] if model_paths else "", question)
            
            # Store response
            state["final_response"] = response
            
            logger.info("Response generated successfully")
            return state
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            state["final_response"] = f"I apologize, but I had an issue generating a response. Error: {str(e)}"
            return state

    def _get_code_enhancement_instructions(self, query: str) -> str:
        """Get instructions for code enhancement requests."""
        return """
        # Code Enhancement Instructions

        ## Analysis Requirements
        1. **Examine the exact DBT model code** provided in the search results
        2. **Identify performance bottlenecks** specifically within the SQL logic of the model
        3. **Evaluate the model's materialization strategy** and config parameters
        4. **Assess incremental logic** if present (e.g., unique_key settings, incremental filter conditions)
        5. **Review join patterns and filter conditions** for optimization opportunities

        ## Response Format
        Provide a detailed response with these sections:

        ### 1. Current Model Analysis
        - Summarize the current model's purpose and structure
        - Identify the materialization strategy (table, view, incremental, etc.)
        - Note key design patterns used and potential bottlenecks

        ### 2. Performance Optimization Recommendations
        - **Query structure improvements**: CTEs, join order, predicate pushdown
        - **Snowflake-specific optimizations**: clustering keys, partition pruning
        - **DBT-specific optimizations**: materialization changes, incremental strategy adjustments
        - **Indexing strategy recommendations** (if applicable)

        ### 3. Complete Enhanced Model Code
        - IMPORTANT: Provide the complete SQL code with all recommended changes
        - NOT just snippets or explanations of changes
        - PRESERVE all existing DBT jinja patterns, macros, and comment structure
        - Maintain config blocks with any modifications clearly indicated
        - Use markdown SQL code blocks for the enhanced model

        ### 4. Implementation Notes
        - Highlight specific changes made and their expected impact
        - Note any dependencies affected and how to handle them
        - Provide testing recommendations for the enhanced model

        ## Critical Requirements
        - Your recommendations must be **specific to the actual model code** provided, not generic advice
        - ALL SQL code must be **complete and executable** within the DBT framework
        - Preserve existing model structure while optimizing performance
        - Maintain compatibility with existing references to this model
        """

    def _get_documentation_instructions(self, query: str) -> str:
        """Get instructions for documentation requests."""
        return """
        # Documentation Instructions

        ## Analysis Requirements
        1. **Review the exact DBT model code** provided in the search results
        2. **Examine schema YAML** and any doc blocks if available
        3. **Identify key columns and their business meanings**
        4. **Determine the model's purpose** in the overall data pipeline
        5. **Note dependencies** (upstream/downstream) and their relationships

        ## Response Format
        Create comprehensive documentation with these sections:

        ### 1. Model Overview
        - Name and physical location in the DBT project
        - Business purpose and key use cases
        - Materialization strategy and refresh pattern
        - Data volume and granularity information

        ### 2. Technical Details
        - Schema information with column descriptions
        - Primary/unique keys and their significance
        - Important joins and filters explained
        - Performance considerations
        - Any special handling or edge cases

        ### 3. Lineage & Dependencies
        - List upstream dependencies with descriptions
        - List downstream dependencies with descriptions
        - How this model fits in the overall data architecture

        ### 4. SQL Breakdown
        - Explanation of complex SQL patterns used
        - Description of CTEs and their purposes
        - Explanation of business logic implemented in code
        - Include relevant SQL snippets with explanations

        ## Critical Requirements
        - Documentation must be **specific to the actual model provided**, not generic
        - Include technical details AND business context
        - Make the documentation useful for both technical and business users
        - Use clear, concise language with proper formatting
        - Focus on information that helps users understand and work with the model
        """

    def _safe_llm_call(self, messages: List[BaseMessage], max_retries: int = 2) -> str:
        """Safely call LLM with retry mechanism and error handling"""
        retries = 0
        while retries <= max_retries:
            try:
                response = self.llm.invoke(messages)
                content = response.content

                # Check if this is a code enhancement task with framework-specific content
                is_code_enhancement = any(message.content and "enhance the following code" in message.content.lower() for message in messages)
                
                # Check if this is a lineage task
                is_lineage_task = any(message.content and ("data lineage" in message.content.lower() or "model lineage" in message.content.lower()) for message in messages)
                
                # Framework mentions that don't belong in a DBT context
                framework_mentions = ["Laravel", "Django", "Rails", "Java", "Node.js", "Express", "Spring Boot", "Flask"]
                
                # Generic lineage phrases that indicate a non-specific response
                generic_lineage_phrases = [
                    "lineage overview",
                    "upstream dependencies",
                    "downstream dependencies",
                    "critical path analysis", 
                    "the model of interest",
                    "highlight the model in your visualization",
                    "visualization",
                    "ALWAYS INCLUDE"
                ]
                
                # DBT model path patterns
                dbt_model_path_pattern = re.compile(r'models/[a-zA-Z0-9_/]+\.sql')
                
                # Check for framework-specific content in code enhancements
                if is_code_enhancement and any(framework in content for framework in framework_mentions):
                    correction = """
                    THE SOLUTION PROVIDED IS NOT APPROPRIATE FOR A DBT PROJECT.
                    
                    DBT (data build tool) is SQL-first and specifically designed for data transformation in the data warehouse.
                    It does not use web frameworks like Laravel, Django, etc.
                    
                    CORRECT APPROACH:
                    1. Make direct edits to the SQL file in the models directory 
                    2. Focus on SQL transformations, not web application code
                    3. Follow DBT best practices for model structure and reference
                    
                    Please revise your answer to only include DBT-specific SQL transformations.
                    """
                    content += "\n\n" + correction
                
                # Check for generic lineage responses
                if is_lineage_task:
                    # Extract file path mentions
                    model_paths = dbt_model_path_pattern.findall(content)
                    
                    # Check for generic phrases or lacking sufficient concrete paths
                    has_generic_phrases = any(phrase in content for phrase in generic_lineage_phrases)
                    
                    # If there are fewer than 2 model paths or generic phrases are present, flag as generic
                    if len(model_paths) < 2 or has_generic_phrases:
                        correction = """
                        THE LINEAGE INFORMATION PROVIDED APPEARS TO BE GENERIC AND NOT SPECIFIC TO THE REPOSITORY.
                        
                        IMPORTANT CORRECTION:
                        
                        Your response should ONLY include actual lineage information extracted from the Git repository.
                        Be specific about:
                        1. The exact file paths found in the repository
                        2. True upstream dependencies (models referenced with ref() or source())
                        3. True downstream dependencies (models that reference this model)
                        4. Column-level dependencies where possible
                        
                        Example of correct paths in this repository:
                        - models/marts/intermediate/order_items.sql
                        - models/staging/tpch/stg_tpch_orders.sql
                        - models/marts/core/fct_order_items.sql
                        
                        If the information cannot be found in the repository, please state that clearly instead of providing generalized descriptions.
                        """
                        content += "\n\n" + correction
                
                return content
            except Exception as e:
                retries += 1
                if retries > max_retries:
                    # If all retries fail, return an error message
                    return f"I apologize, but I encountered an error while processing your request. Please try again or rephrase your question. Error details: {str(e)}"
                # Wait briefly before retrying
                time.sleep(1)

    def _format_results_for_prompt(self, state: Dict[str, Any]) -> str:
        """Format the search results for inclusion in the prompt."""
        formatted_text = ""
        
        # Format DBT results
        if "dbt_results" in state and state["dbt_results"]:
            dbt_results = state["dbt_results"]
            
            # Handle case where dbt_results could be a list (from newer implementation) or dict (from older)
            if isinstance(dbt_results, list):
                formatted_text += "## DBT Models\n\n"
                for result in dbt_results:
                    model_name = result.get("model_name", "")
                    file_path = result.get("file_path", "")
                    description = result.get("description", "")
                    match_type = result.get("match_type", "")
                    
                    if model_name:
                        formatted_text += f"### Model: {model_name}\n"
                    if file_path:
                        formatted_text += f"Path: {file_path}\n"
                    if description:
                        formatted_text += f"Description: {description}\n"
                    if match_type:
                        formatted_text += f"Match Type: {match_type}\n"
                    
                    formatted_text += "\n"
            elif isinstance(dbt_results, dict):
                # Original implementation for backward compatibility
                formatted_text += "## DBT Models\n\n"
                for entity, results in dbt_results.items():
                    if isinstance(results, list):
                        for result in results:
                            model_name = result.get("model_name", "")
                            file_path = result.get("file_path", "")
                            description = result.get("description", "")
                            
                            if model_name:
                                formatted_text += f"### Model: {model_name}\n"
                            if file_path:
                                formatted_text += f"Path: {file_path}\n"
                            if description:
                                formatted_text += f"Description: {description}\n"
                            
                            formatted_text += "\n"
                    elif isinstance(results, dict) and "error" not in results:
                        model_name = results.get("model_name", "")
                        file_path = results.get("file_path", "")
                        description = results.get("description", "")
                        
                        if model_name:
                            formatted_text += f"### Model: {model_name}\n"
                        if file_path:
                            formatted_text += f"Path: {file_path}\n"
                        if description:
                            formatted_text += f"Description: {description}\n"
                        
                        formatted_text += "\n"
            else:
                # Handle case where dbt_results is neither a list nor a dict
                formatted_text += "No structured DBT results found.\n\n"
        
        # Format model details
        if "model_details" in state and state["model_details"]:
            model_details = state["model_details"]
            
            formatted_text += "## Model Details\n\n"
            
            for model_name, details in model_details.items():
                formatted_text += f"### {model_name}\n"
                
                # Extract model info
                if "model" in details:
                    model = details["model"]
                    
                    if isinstance(model, dict):
                        if "file_path" in model:
                            formatted_text += f"Path: {model['file_path']}\n"
                        if "model_type" in model:
                            formatted_text += f"Type: {model['model_type']}\n"
                        if "description" in model:
                            formatted_text += f"Description: {model['description']}\n"
                
                # Add schema information if available
                if "schema" in details and details["schema"]:
                    schema = details["schema"]
                    
                    if "description" in schema and schema["description"]:
                        formatted_text += f"Schema Description: {schema['description']}\n"
                    
                    if "columns" in schema and schema["columns"]:
                        formatted_text += "Columns:\n"
                        for column in schema["columns"]:
                            col_name = column.get("name", "Unknown")
                            col_desc = column.get("description", "No description")
                            formatted_text += f"- {col_name}: {col_desc}\n"
                
                # Add dependency information
                if "dependencies" in details and details["dependencies"]:
                    deps = details["dependencies"]
                    
                    if "upstream" in deps and deps["upstream"]:
                        formatted_text += "Upstream: " + ", ".join(deps["upstream"]) + "\n"
                    
                    if "downstream" in deps and deps["downstream"]:
                        formatted_text += "Downstream: " + ", ".join(deps["downstream"]) + "\n"
                
                # Add column-specific info
                if "column_info" in details and details["column_info"]:
                    col_info = details["column_info"]
                    formatted_text += f"Column: {col_info.get('name', 'Unknown')}\n"
                    if "calculation" in col_info and col_info["calculation"]:
                        formatted_text += f"Calculation: {col_info['calculation']}\n"
                
                formatted_text += "\n"
        
        # Format column search results
        if "column_details" in state and state["column_details"]:
            column_details = state["column_details"]
            
            formatted_text += "## Column Search Results\n\n"
            
            # Handle column_details as either a dict or a list
            if isinstance(column_details, dict):
                for column_name, results in column_details.items():
                    formatted_text += f"### Column: {column_name}\n"
                    
                    if isinstance(results, list):
                        for result in results:
                            model_name = result.get("model_name", "")
                            file_path = result.get("file_path", "")
                            calculation = result.get("calculation", "")
                            
                            if model_name:
                                formatted_text += f"Model: {model_name}\n"
                            if file_path:
                                formatted_text += f"Path: {file_path}\n"
                            if calculation:
                                calc_formatted = self._format_calculation(calculation)
                                formatted_text += f"Calculation:\n```sql\n{calc_formatted}\n```\n"
                            
                            formatted_text += "\n"
            elif isinstance(column_details, list):
                for result in column_details:
                    column_name = result.get("column_name", "")
                    model_name = result.get("model_name", "")
                    file_path = result.get("file_path", "")
                    calculation = result.get("calculation", "")
                    
                    if column_name:
                        formatted_text += f"### Column: {column_name}\n"
                    if model_name:
                        formatted_text += f"Model: {model_name}\n"
                    if file_path:
                        formatted_text += f"Path: {file_path}\n"
                    if calculation:
                        calc_formatted = self._format_calculation(calculation)
                        formatted_text += f"Calculation:\n```sql\n{calc_formatted}\n```\n"
                    
                    formatted_text += "\n"
        
        # Format content search results
        if "content_search" in state and state["content_search"]:
            content_search = state["content_search"]
            
            formatted_text += "## Content Search Results\n\n"
            
            if isinstance(content_search, list):
                for result in content_search:
                    file_path = result.get("file_path", "")
                    search_text = result.get("search_text", "")
                    contexts = result.get("match_contexts", [])
                    
                    if file_path:
                        formatted_text += f"### File: {file_path}\n"
                    if search_text:
                        formatted_text += f"Search Term: '{search_text}'\n"
                    
                    if contexts:
                        formatted_text += "Matches:\n"
                        for i, context in enumerate(contexts[:3]):  # Limit to first 3 contexts for brevity
                            formatted_text += f"```\n{context}\n```\n"
                        
                        if len(contexts) > 3:
                            formatted_text += f"...and {len(contexts) - 3} more matches\n"
                    
                    formatted_text += "\n"
            elif isinstance(content_search, dict):
                for search_term, results in content_search.items():
                    formatted_text += f"### Search Term: '{search_term}'\n"
                    
                    if isinstance(results, list):
                        for result in results:
                            file_path = result.get("file_path", "")
                            contexts = result.get("match_contexts", [])
                            
                            if file_path:
                                formatted_text += f"File: {file_path}\n"
                            
                            if contexts:
                                formatted_text += "Matches:\n"
                                for context in contexts[:2]:  # Limit to first 2 contexts
                                    formatted_text += f"```\n{context}\n```\n"
                            
                            formatted_text += "\n"
            
        # Check if we have any content to return
        if not formatted_text:
            formatted_text = "No search results were found to include in the prompt!"
            logger.warning("No search results were found to include in the prompt!")
        
        # Log the size of the formatted text for debugging
        logger.info(f"Formatted prompt contains {len(formatted_text)} characters")
        
        return formatted_text
    
    def _get_example_path_from_results(self, results: Dict[str, Any]) -> str:
        """Extract an example file path from results to show in the prompt."""
        # Try to find a file path in the model search results
        if "model_search" in results:
            model_results = results["model_search"]
            if "results" in model_results and model_results["results"]:
                first_result = model_results["results"][0]
                if isinstance(first_result, dict) and "file_path" in first_result:
                    return first_result["file_path"]
        
        # If no file path found, return a generic example
        return "models/path/to/model.sql"

    def _get_instructions_for_type(self, question_type: str, question: str) -> str:
        """Get specific instructions based on question type."""
        instruction_map = {
            "MODEL_INFO": self._get_model_explanation_instructions,
            "LINEAGE": self._get_lineage_instructions,
            "DEPENDENCIES": self._get_dependency_instructions,
            "CODE_ENHANCEMENT": self._get_code_enhancement_instructions,
            "DOCUMENTATION": self._get_documentation_instructions,
            "DEVELOPMENT": self._get_development_instructions,
            "GENERAL": self._get_general_instructions
        }
        
        instruction_func = instruction_map.get(question_type, self._get_general_instructions)
        return instruction_func(question)

    def _get_model_explanation_instructions(self, query: str) -> str:
        """Get instructions for model explanation questions."""
        return """
        When explaining a dbt model, you must focus on comprehensive understanding:
        
        1. CONTEXT AND PURPOSE
        - Explain the business purpose of the model
        - Describe what this model represents in the data architecture
        - Include how this model fits in the broader data pipeline
        
        2. TECHNICAL IMPLEMENTATION
        - Show the EXACT file path of the model from the search results - do not make this up
        - Explain the materialization type (view, table, incremental, etc.) based on the actual code or configuration
        - Break down complex SQL patterns or functions
        - Identify data sources (FROM clauses, JOINs, CTEs,.YML schemas)
        
        3. CODE WALKTHROUGH
        - Provide ONLY the SQL code that appears in the search results - never fabricate code
        - Break down the query into logical sections
        - Explain each section's purpose with clear explanations
        - For complex transformations, walk through the step-by-step logic
        
        4. DEPENDENCIES AND RELATIONSHIPS
        - List ONLY models referenced in the search results with their EXACT file paths
        - Show upstream dependencies (models this one relies on) ONLY if found in the search results
        - Show downstream dependencies (models that rely on this one) ONLY if found in the search results
        - Visualize the dependency chain if complex
        
        5. SCHEMA AND STRUCTURE
        - List ONLY columns that appear in the search results with their descriptions
        - Identify primary/foreign keys or unique identifiers ONLY if mentioned in the search results
        - Show any tests or assertions applied to the model ONLY if found in the search results
        - Present the model YAML configuration ONLY if available in the search results
        
        FOR ANY SQL CODE EXAMPLES:
        - Include the EXACT file path from the search results in a comment at the top
        - NEVER fabricate or guess at code not explicitly included in the search results
        - If a piece of information is not in the search results, clearly state this limitation
        
        MOST IMPORTANTLY:
        - USE ONLY information explicitly provided in the search results
        - NEVER make up file paths, code, or model descriptions
        - ALWAYS acknowledge when information is not available rather than guessing
        """

    def _get_lineage_instructions(self, query: str) -> str:
        """Get instructions for lineage queries"""
        return """
You need to analyze and visualize the data lineage for the requested model from the Git repository.

1. MOST IMPORTANT: Provide ONLY ACTUAL lineage information extracted from the Git repository. 
   DO NOT generate hypothetical lineage. If information is not found, say so explicitly.

2. First identify the EXACT model file path in the Git repository. For example:
   - models/marts/intermediate/order_items.sql
   - models/staging/tpch/stg_tpch_orders.sql
   
3. Extract model information:
   - The exact model definition (SQL code)
   - All upstream dependencies using ref() or source() functions
   - Downstream dependencies (models that reference this model)
   - Column definitions and calculations

4. Present the lineage in a clearly formatted section titled "LINEAGE" that includes:
   - Full file paths (e.g., models/marts/intermediate/order_items.sql)
   - Model types (staging, intermediate, core, mart) based on their paths
   - Key transformations performed (joins, calculations, etc.)
   - Exact column names and their transformations

5. Create a visualization using text that clearly shows:
   ```
   Upstream Flow:
   models/staging/tpch/stg_tpch_orders.sql → models/marts/intermediate/order_items.sql ← models/staging/tpch/stg_tpch_line_items.sql
   
   Detailed Model:
   └── models/marts/intermediate/order_items.sql
       ├── [order_key] (primary key, from stg_tpch_orders)
       ├── [base_price] (calculated from extended_price/quantity)
       └── [gross_item_sales_amount] (line_item.extended_price)
       
   Downstream Flow:
   └── models/marts/core/fct_order_items.sql
       └── models/marts/aggregates/agg_ship_modes_hardcoded_pivot.sql
   └── models/marts/core/fct_orders.sql
   ```

6. For column-level lineage, identify:
   - Source of each column (upstream model or calculation)
   - Transformation logic (formula or reference)
   - Downstream usage

REMEMBER: 
- ONLY reference models and sources that ACTUALLY EXIST in the Git repository
- Use EXACT file paths from the repository structure
- NEVER invent connections that don't appear in the code
- For column calculations, quote the ACTUAL SQL from the model file
- If you cannot find certain information, explicitly state what is missing
"""

    def _get_dependency_instructions(self, query: str) -> str:
        """Get instructions for dependency questions."""
        return """
        When analyzing dependencies, provide a detailed dependency assessment:
        
        1. DIRECT DEPENDENCIES
        - List all immediate upstream dependencies (models this one references)
          * Include the full file path for each (models/path/to/model.sql)
          * Show the exact SQL where the reference occurs
          * Explain what data each dependency provides
        - List all immediate downstream dependencies (models that reference this one)
          * Include the full file path for each
          * Explain how each uses this model's data
        
        2. DEPENDENCY IMPACT ANALYSIS
        - Explain cascading effects of changes
          * How changes to the model would affect downstream models
          * Which upstream changes would impact this model
        - Identify critical dependencies
          * Which dependencies are most critical to this model's function
          * Which downstream models most critically depend on this one
        
        3. OPTIMIZATION OPPORTUNITIES
        - Identify potential improvements:
          * Materialization changes (view to table, incremental, etc.)
          * Refactoring opportunities (splitting or combining models)
          * Performance enhancements (indexing, partitioning)
        - Suggest tests to ensure data integrity
        
        4. TRANSITIVE DEPENDENCIES
        - Show the extended dependency chain (dependencies of dependencies)
        - Identify circular references or complex dependency patterns
        - Map out the complete dependency tree
        
        5. RISK ASSESSMENT
        - Highlight potential failure points
        - Identify models with many downstream dependencies
        - Note any stale or outdated references
        
        ALWAYS INCLUDE:
        - Complete file paths for every model mentioned
        - Clear distinction between upstream and downstream dependencies
        - Practical advice for managing dependencies
        - Specific code examples with syntax highlighting
        """

    def _get_development_instructions(self, query: str) -> str:
        """Get instructions for development tasks."""
        return """
        Provide CONCISE and PRACTICAL development guidance with a focus on CODE:
        
        1. REQUIREMENTS SUMMARY
        - In 2-3 sentences, summarize what needs to be developed/modified
        - Identify the exact file(s) to create or change
        - Focus on WHAT to implement, not extensive background
        
        2. DIRECT CODE IMPLEMENTATION
        - Provide complete, ready-to-use code blocks:
          ```sql
          -- File: models/path/to/model.sql
          -- Implement the feature
          SELECT
            field1,
            field2,
            calculated_field
          FROM source_table
          ```
        
        3. CLEAR STEP-BY-STEP PROCESS
        - Number the implementation steps precisely:
          1. Create/open file at exact path
          2. Add/modify the specific code
          3. Save the file
          4. Run specific commands to test
        
        4. CODE VALIDATION
        - Include brief validation checks:
          * Syntax verification points
          * Expected output examples
          * Simple test queries to confirm correctness
          * Common errors to avoid
        
        5. MINIMAL CONTEXT AND BACKGROUND
        - Provide only essential context needed to understand the code
        - Focus on IMPLEMENTATION, not theory
        - Include dependencies only if directly required
        
        FOR COLUMN DELETION (if applicable):
        - Show exact before/after SQL demonstrating the column removal
        - Verify no dependencies on the removed column
        - Check for any downstream impacts
        
        IMPORTANT:
        - Use EXACT file paths from search results
        - Must be focus on Dbt specific, never generalize it or do not assume with other frameworks.
        - Provide COMPLETE code blocks, not fragments
        - Validate logic and syntax before providing final code
        - Be DIRECT and PRACTICAL - focus on implementation
        """

    def _get_general_instructions(self, query: str) -> str:
        """Get instructions for general questions."""
        return """
        Provide a focused, detailed response to this general dbt question:
        
        1. DIRECT ANSWER
        - Answer the question explicitly and clearly
        - Provide specific, actionable information
        - Address all aspects of the question
        - Be concise but comprehensive
        
        2. TECHNICAL CONTEXT
        - Explain relevant dbt concepts
        - Show how dbt handles this scenario
        - Reference dbt best practices
        - Include file paths where relevant
        
        3. CODE EXAMPLES
        - Provide practical code examples:
          * SQL queries with proper syntax highlighting
          * YAML configurations if relevant
          * Command line examples if needed
          * dbt macro examples if applicable
        - For each example:
          * Explain what it does
          * Highlight key parts
          * Show expected output
        
        4. BEST PRACTICES
        - Recommend optimal approaches
        - Explain why they're considered best practices
        - Mention alternatives and trade-offs
        - Provide implementation guidance
        
        5. NEXT STEPS
        - Suggest follow-up actions
        - Recommend related topics to explore
        - Point to relevant documentation
        - Offer testing strategies if applicable
        
        REMEMBER TO:
        - Use clear, technical language
        - Include code examples with syntax highlighting
        - Be specific about file paths and model names
        - Focus on actionable advice
        """

    def _extract_json_from_response(self, response: str) -> str:
        """Extract a valid JSON string from a potentially noisy LLM response."""
        # Handle usual JSON code blocks
        if '```json' in response:
            # Get content between ```json and ```
            match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if match:
                return match.group(1).strip()
                
        # Handle just code blocks without language specification
        if '```' in response:
            # Get content between ``` and ```
            match = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)
            if match:
                candidate = match.group(1).strip()
                # Check if it looks like JSON
                if candidate.startswith('{') and candidate.endswith('}'):
                    return candidate
                
        # If the response itself is a JSON object
        if response.strip().startswith('{') and response.strip().endswith('}'):
            return response.strip()
            
        # Handle malformed responses with a JSON object buried in them
        match = re.search(r'(\{.*"question_type".*\})', response, re.DOTALL)
        if match:
            return match.group(1).strip()
            
        # Handle thinking steps by removing them
        if '<think>' in response.lower():
            # Extract part after thinking
            match = re.search(r'(?:>|think>)(.*?)(?:$|\{)', response, re.DOTALL | re.IGNORECASE)
            if match and '{' in response:
                # Find the json part
                json_start = response.find('{', match.start())
                json_end = response.rfind('}') + 1
                if json_start < json_end:
                    return response[json_start:json_end].strip()
        
        # Default - return the original response as is
        return response
    
    def _create_fallback_analysis(self, question: str, llm_response: str, column_patterns: List[str]) -> Dict[str, Any]:
        """Create a fallback question analysis when JSON parsing fails."""
        # Default values
        fallback = {
            "question_type": "GENERAL",
            "entities": [],
            "search_terms": [],
            "intent": "general inquiry",
            "rephrased_question": question
        }
        
        # Add any column patterns as entities
        if column_patterns:
            fallback["entities"].extend(column_patterns)
        
        # Extract file paths as entities
        file_paths = re.findall(r'models/[a-zA-Z0-9_/]+\.sql', question)
        if file_paths:
            fallback["entities"].extend(file_paths)
        
        # Look for models mentioned in the question
        model_matches = re.findall(r'\b([a-z][a-z0-9_]+)\.sql\b', question)
        fallback["entities"].extend(model_matches)
        
        # Try to classify based on keywords in the question
        lowercase_question = question.lower()
        
        # Check for development indicators
        if any(keyword in lowercase_question for keyword in ["create", "implement", "delete", "add", "remove", "script", "write", "build"]):
            fallback["question_type"] = "DEVELOPMENT"
            fallback["intent"] = "development request"
            
        # Check for code enhancement indicators
        elif any(keyword in lowercase_question for keyword in ["change", "modify", "improve", "optimize", "update", "alter", "fix", "enhance"]):
            fallback["question_type"] = "CODE_ENHANCEMENT"
            fallback["intent"] = "code enhancement request"
            
        # Check for model info indicators
        elif any(keyword in lowercase_question for keyword in ["what is", "explain", "show me", "how does", "tell me about"]):
            fallback["question_type"] = "MODEL_INFO"
            fallback["intent"] = "information request"
            
        # Extract search terms
        words = re.findall(r'\b[a-z][a-z0-9_]{3,}\b', lowercase_question)
        fallback["search_terms"] = [w for w in words if w not in COMMON_STOP_WORDS]
        
        # Add the extracted entities to search terms if not already there
        for entity in fallback["entities"]:
            if entity not in fallback["search_terms"]:
                fallback["search_terms"].append(entity)
                
        return fallback

    def _get_model_details(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Get detailed information for models identified in search results"""
        try:
            # Extract search results
            dbt_results = state.get("dbt_results", {})
            
            # Initialize model details container
            model_details = {}
            
            # Process search results to gather detailed information
            # Handle case where dbt_results could be a list (newer implementation)
            if isinstance(dbt_results, list):
                logger.info(f"Processing model details from list-style results with {len(dbt_results)} items")
                
                for result in dbt_results:
                    model_name = result.get("model_name", "")
                    file_path = result.get("file_path", "")
                    
                    # Skip entries without model name or path
                    if not model_name or not file_path:
                        continue
                    
                    # Skip if model already processed
                    if model_name in model_details:
                        continue
                    
                    # Add model to details
                    model_details[model_name] = {
                        "model": {
                            "name": model_name,
                            "file_path": file_path,
                            "content": result.get("content", ""),
                            "description": result.get("description", ""),
                            "model_type": self._extract_model_type(result.get("content", ""))
                        },
                        "schema": result.get("yaml_content", {}),
                        "dependencies": self._get_model_dependencies(model_name, file_path)
                    }
                    
                    # Add schema info if available in search result
                    if "schema_info" in result and result["schema_info"]:
                        model_details[model_name]["schema"] = result["schema_info"]
            
            # Handle case where dbt_results is a dictionary (older implementation)
            elif isinstance(dbt_results, dict):
                logger.info(f"Processing model details from dictionary-style results with {len(dbt_results)} entities")
                
                for entity, results in dbt_results.items():
                    # Handle list of results for an entity
                    if isinstance(results, list):
                        for result in results:
                            if not isinstance(result, dict):
                                continue
                            
                            model_name = result.get("model_name", "")
                            file_path = result.get("file_path", "")
                            
                            # Skip entries without model name or path
                            if not model_name or not file_path:
                                continue
                            
                            # Skip if model already processed
                            if model_name in model_details:
                                continue
                            
                            # Add model to details
                            model_details[model_name] = {
                                "model": {
                                    "name": model_name,
                                    "file_path": file_path,
                                    "content": result.get("content", ""),
                                    "description": result.get("description", ""),
                                    "model_type": self._extract_model_type(result.get("content", ""))
                                },
                                "schema": result.get("yaml_content", {}),
                                "dependencies": self._get_model_dependencies(model_name, file_path)
                            }
                            
                            # Add schema info if available in search result
                            if "schema_info" in result and result["schema_info"]:
                                model_details[model_name]["schema"] = result["schema_info"]
                    
                    # Handle single result for an entity
                    elif isinstance(results, dict) and "error" not in results:
                        model_name = results.get("model_name", "")
                        file_path = results.get("file_path", "")
                        
                        # Skip entries without model name or path
                        if not model_name or not file_path:
                            continue
                        
                        # Skip if model already processed
                        if model_name in model_details:
                            continue
                        
                        # Add model to details
                        model_details[model_name] = {
                            "model": {
                                "name": model_name,
                                "file_path": file_path,
                                "content": results.get("content", ""),
                                "description": results.get("description", ""),
                                "model_type": self._extract_model_type(results.get("content", ""))
                            },
                            "schema": results.get("yaml_content", {}),
                            "dependencies": self._get_model_dependencies(model_name, file_path)
                        }
                        
                        # Add schema info if available in search result
                        if "schema_info" in results and results["schema_info"]:
                            model_details[model_name]["schema"] = results["schema_info"]
            
            # Add a special case to handle if model_details is empty but we have content in the search results
            if not model_details and isinstance(dbt_results, list) and len(dbt_results) > 0:
                # Try to extract model details from the first result's content
                first_result = dbt_results[0]
                if "content" in first_result and first_result["content"]:
                    file_path = first_result.get("file_path", "unknown_path")
                    model_name = first_result.get("model_name", os.path.basename(file_path).replace('.sql', ''))
                    
                    model_details[model_name] = {
                        "model": {
                            "name": model_name,
                            "file_path": file_path,
                            "content": first_result["content"],
                            "description": first_result.get("description", ""),
                            "model_type": self._extract_model_type(first_result["content"])
                        },
                        "schema": first_result.get("yaml_content", {}),
                        "dependencies": {}
                    }
                    
                    logger.info(f"Created model details from content for {model_name}")
            
            logger.info(f"Gathered detailed information for {len(model_details)} models")
            
            # Update state with model details
            state["model_details"] = model_details
            return state
            
        except Exception as e:
            logger.error(f"Error getting model details: {str(e)}")
            state["model_details"] = {}
            return state
    
    def _extract_model_type(self, content: str) -> str:
        """Extract model materialization type from SQL content"""
        if not content:
            return "view"  # Default type
        
        # Look for config materialized parameter
        materialized_match = re.search(r'{{\s*config\s*\(\s*materialized\s*=\s*[\'"](\w+)[\'"]', content)
        if materialized_match:
            return materialized_match.group(1)
        
        # Look for table or view keywords
        if re.search(r'create\s+table', content, re.IGNORECASE):
            return "table"
        elif re.search(r'create\s+view', content, re.IGNORECASE):
            return "view"
        
        return "view"  # Default type
    
    def _get_model_dependencies(self, model_name: str, file_path: str) -> Dict[str, List[str]]:
        """Get upstream and downstream dependencies for a model"""
        try:
            if not self.dbt_tools:
                return {"upstream": [], "downstream": []}
            
            dependencies = self.dbt_tools.find_related_models(model_name)
            if not dependencies:
                # Try with the path
                model_path = file_path
                if model_path.endswith('.sql'):
                    model_path = model_path[:-4]
                dependencies = self.dbt_tools.find_related_models(model_path)
            
            if not dependencies:
                return {"upstream": [], "downstream": []}
            
            return dependencies
            
        except Exception as e:
            logger.warning(f"Error getting dependencies for {model_name}: {str(e)}")
            return {"upstream": [], "downstream": []}

    def _validate_code_enhancement_response(self, response: str, original_code: str, model_path: str, question: str) -> str:
        """Validate and correct code enhancement responses to ensure they follow the required format."""
        try:
            # Check if the model path appears in the response
            if model_path and model_path not in response:
                logger.warning("Code enhancement response does not include the correct model path")
                response = f"Model path: {model_path}\n\n" + response
            
            # Check if the response contains generic-looking code (doesn't match the original structure)
            # We can check for specific patterns in the original code that should be preserved
            
            # 1. Check for CTE patterns
            cte_pattern = re.compile(r'with\s+\w+\s+as\s+\(', re.IGNORECASE)
            original_has_cte = bool(cte_pattern.search(original_code))
            response_has_cte = bool(cte_pattern.search(response))
            
            # 2. Check for config blocks
            config_pattern = re.compile(r'\{\{\s*config\(.*?\)\s*\}\}', re.DOTALL)
            original_has_config = bool(config_pattern.search(original_code))
            response_has_config = bool(config_pattern.search(response))
            
            # 3. Check for ref patterns
            ref_pattern = re.compile(r'\{\{\s*ref\([\'"].*?[\'"]\)\s*\}\}')
            original_refs = ref_pattern.findall(original_code)
            
            # Detect issues with the response
            issues = []
            
            if original_has_cte and not response_has_cte:
                issues.append("missing CTE structure")
            
            if original_has_config and not response_has_config:
                issues.append("missing config block")
                
            # Extract complete code blocks from the response
            code_blocks = re.findall(r'```sql\s*(.*?)\s*```', response, re.DOTALL)
            complete_model_blocks = [block for block in code_blocks if len(block.split('\n')) > 10 and ('{{' in block or 'select' in block.lower())]
            
            # If we found issues and have a complete model block, add a correction
            if issues and not complete_model_blocks:
                logger.warning(f"Code enhancement issues detected: {', '.join(issues)}")
                
                # Create messages for the correction LLM call
                correction_messages = [
                    SystemMessage(content=f"""
                    You are a Senior DBT SQL Developer. A previous response to enhance a model was not properly formatted.
                    
                    The ORIGINAL model code is:
                    ```sql
                    {original_code}
                    ```
                    
                    The enhancement request was: {question}
                    
                    The previous response had these issues: {', '.join(issues)}
                    
                    Please create a CORRECTLY FORMATTED response that:
                    1. Provides a proper analysis of the model structure
                    2. Clearly shows what changes are being made
                    3. PRESERVES the EXACT structure of the original model (CTEs, config blocks, etc.)
                    4. ONLY makes the specific changes requested in the enhancement
                    5. Includes the FULL modified code that can be directly used
                    
                    IMPORTANT: Do NOT create a generic model. ONLY modify the exact code provided.
                    """),
                    HumanMessage(content=f"Previous response: {response}")
                ]
                
                # Get corrected response
                corrected_response = self._safe_llm_call(correction_messages)
                
                # Add a note about the correction
                final_response = f"""
                I detected issues with my previous response that didn't properly preserve the original model structure.
                
                Here's a corrected enhancement:
                
                {corrected_response}
                """
                
                return final_response
            
            # If we have a complete model that looks reasonable, keep the response as is
            return response
            
        except Exception as e:
            logger.error(f"Error validating code enhancement: {str(e)}")
            # Return the original response if validation fails
            return response

    def _find_model_content(self, path_or_model: str) -> Tuple[bool, str, str]:
        """
        Find content for a model given a path or model name.
        
        Args:
            path_or_model: Path to a model file or model name
            
        Returns:
            Tuple of (success, content, file_path)
        """
        if not self.dbt_tools:
            return False, "", ""
            
        # Try multiple approaches to get model content
        try:
            # Approach 1: Direct file access if it looks like a path
            if '/' in path_or_model or path_or_model.endswith('.sql'):
                content = self.dbt_tools.get_file_content(path_or_model)
                if content:
                    logger.info(f"Found model content via direct file access: {path_or_model}")
                    return True, content, path_or_model
                    
            # Approach 2: Search as a model name
            model_name = path_or_model
            # Remove .sql extension if present
            if model_name.endswith('.sql'):
                model_name = model_name[:-4]
                
            # Get the base name if it's a path
            if '/' in model_name:
                model_name = os.path.basename(model_name)
                
            # Search with output mode for precise matching
            results = self.dbt_tools.search_model(model_name, search_mode="output")
            if results and len(results) > 0:
                result = results[0]
                
                # Get content from the result
                content = ""
                file_path = ""
                
                if hasattr(result, 'content') and result.content:
                    content = result.content
                    file_path = result.file_path if hasattr(result, 'file_path') else path_or_model
                    logger.info(f"Found model content via model search: {model_name}")
                    return True, content, file_path
                elif isinstance(result, dict):
                    if 'content' in result and result['content']:
                        content = result['content']
                        file_path = result.get('file_path', path_or_model)
                        logger.info(f"Found model content via model search (dict): {model_name}")
                        return True, content, file_path
                
            # Approach 3: Try path variations for file paths
            if '/' in path_or_model:
                path_pattern = path_or_model
                # Generate variations of the path
                path_variations = []
                
                # Try with and without .sql extension
                if path_pattern.endswith('.sql'):
                    base_path = path_pattern[:-4]
                    path_variations.append(base_path)
                else:
                    path_variations.append(f"{path_pattern}.sql")
                    
                # Try with models/ prefix if not present
                if not path_pattern.startswith('models/'):
                    path_variations.append(f"models/{path_pattern}")
                    
                # Try marts prefix for dimension models
                if 'dim_' in path_pattern or 'fact_' in path_pattern:
                    # Extract the model name
                    model_name = os.path.basename(path_pattern).replace('.sql', '')
                    path_variations.append(f"models/marts/core/{model_name}.sql")
                    path_variations.append(f"models/marts/{model_name}.sql")
                    path_variations.append(f"marts/core/{model_name}.sql")
                
                # Try each variation
                for variation in path_variations:
                    content = self.dbt_tools.get_file_content(variation)
                    if content:
                        logger.info(f"Found model content via path variation: {variation}")
                        return True, content, variation
                        
            logger.warning(f"Could not find model content for: {path_or_model}")
            return False, "", ""
            
        except Exception as e:
            logger.error(f"Error finding model content: {str(e)}")
            return False, "", ""

# Define common stop words to exclude from search terms
COMMON_STOP_WORDS = {
    'the', 'and', 'but', 'for', 'nor', 'or', 'so', 'yet', 
    'with', 'about', 'into', 'over', 'after', 'when', 'where', 'why',
    'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'some',
    'such', 'than', 'that', 'these', 'this', 'those', 'which', 'whose',
    'while', 'what', 'find', 'does', 'show', 'tell', 'use', 'used', 'using'
}

def create_data_architect_agent(repo_url_or_tools: Union[str, SearchTools] = "", username: str = "", token: str = ""):
    """Factory function to create a data architect agent.
    
    Args:
        repo_url_or_tools: Either a repository URL string or a SearchTools object
        username: GitHub username (optional)
        token: GitHub token (optional)
    """
    # If repo_url_or_tools is a SearchTools object, extract the repo URL from the database
    if not isinstance(repo_url_or_tools, str):
        # Get the latest GitHub configuration from the database
        try:
            repo_url, username, token = DbtToolsFactory.get_tools_from_db()
            if repo_url:
                return DataArchitectAgent(repo_url, username, token)
        except Exception as e:
            logger.error(f"Error getting GitHub configuration: {str(e)}")
    
    # If we get here, either repo_url_or_tools is a string or we couldn't get the config
    return DataArchitectAgent(
        repo_url=repo_url_or_tools if isinstance(repo_url_or_tools, str) else "",
        username=username,
        token=token
    ) 