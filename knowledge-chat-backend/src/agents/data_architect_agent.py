from typing import Dict, List, Any, Annotated, Sequence, TypedDict, Union, Optional, Tuple
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

# Try different imports for ChatOllama to handle different langchain versions
try:
    from langchain_ollama import ChatOllama
except ImportError:
    try:
        from langchain_community.chat_models import ChatOllama
    except ImportError:
        from langchain_core.language_models.chat_models import BaseChatModel
        class ChatOllama(BaseChatModel):
            def __init__(self, *args, **kwargs):
                raise ImportError("ChatOllama is not available. Please install langchain_ollama or langchain_community to use it.")

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
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import SearchTools
from db.database import ChatDatabase
from utils import ChromaDBManager
from dbt_tools import DbtTools, DbtToolsFactory
import re
from urllib.parse import urlparse

# Set up logger
logger = logging.getLogger(__name__)

# Define state type
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "The messages in the conversation"]
    question_analysis: Annotated[Dict, "Parsed question analysis"]
    search_results: Annotated[Dict, "DBT model search results"]
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
        """
        Initialize a Data Architect Agent with the specified repository.
        
        Args:
            repo_url: The URL of the DBT repository
            username: The GitHub username (optional)
            token: The GitHub token (optional)
        """
        # Initialize language model - this is needed for various methods
        try:
            self.llm = ChatOllama(model="gemma3:latest")
        except Exception as e:
            logger.error(f"Error initializing LLM: {str(e)}")
            # Provide a fallback option when ChatOllama isn't available
            from langchain_core.language_models.chat_models import ChatOpenAI
            self.llm = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo")
            
        self.repo_url = repo_url
        self.username = username
        self.token = token
        self.dbt_tools = None
        
        # Initialize DBT tools only if repo_url is provided and not empty
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
                
                # Use the factory to create DBT tools - handle both relative and absolute imports
                try:
                    # First try the absolute import path
                    from src.dbt_tools import DbtToolsFactory
                    logger.info("Using src.dbt_tools import path")
                except ImportError:
                    try:
                        # Then try relative import path
                        from ..dbt_tools import DbtToolsFactory
                        logger.info("Using relative import path (..dbt_tools)")
                    except ImportError:
                        # Finally try a direct import (for when it's in the same directory)
                        from dbt_tools import DbtToolsFactory
                        logger.info("Using direct import path (dbt_tools)")
                
                self.dbt_tools = DbtToolsFactory.create_dbt_tools(repo_url, username, token)
                
                # Force initialization which clones the repository if needed
                if hasattr(self.dbt_tools, 'initialize'):
                    self.dbt_tools.initialize()
                
                logger.info(f"Initialized DBT tools with repository: {repo_url}")
            except ValueError as e:
                logger.error(f"Invalid repository configuration: {str(e)}")
                self.dbt_tools = None
            except Exception as e:
                logger.error(f"Error initializing DBT tools: {str(e)}")
                self.dbt_tools = None
        else:
            logger.info("No valid repository URL provided, DBT tools will be limited in functionality")
        
        # Create state graph for workflow
        self.workflow = self._create_agent_graph()
    
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
                "search_results": {},
                "model_details": {},
                "column_details": {},
                "related_models": {},
                "content_search": {},
                "final_response": ""
            }
            
            # Log the question for debugging
            logger.info(f"Processing question: {question}")
            
            # Process the question through the agent graph
            logger.info(f"Processing question: {question[:100]}...")
            result = self.workflow.invoke(initial_state)
            
            # Get the final response
            response = result.get("final_response", "I couldn't generate a response. Please try again.")
            
            # Create response metadata
            response_data = self._create_response_metadata(result, conversation_id, thread_id, metadata or {})
            
            return response_data
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}\n{traceback.format_exc()}")
            return self._create_error_response(str(e), conversation_id)
    
    def _create_response_metadata(self, result: Dict[str, Any], conversation_id: str, thread_id: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Create metadata for the response."""
        # Extract relevant information from the result
        question_analysis = result.get("question_analysis", {})
        search_results = result.get("search_results", {})
        model_details = result.get("model_details", {})
        column_details = result.get("column_details", {})
        related_models = result.get("related_models", {})
        content_search = result.get("content_search", {})
        
        # Get the generated response
        response = result.get("final_response", "I couldn't generate a response. Please try again.")
        
        # Create the response dictionary
        return {
            "response": response,
            "conversation_id": conversation_id or str(uuid.uuid4()),
            "thread_id": thread_id or str(uuid.uuid4()),
            "processing_time": metadata.get("processing_time", 0) if metadata else 0,
            "question_type": question_analysis.get("question_type", "GENERAL"),
            "dbt_results": {
                "models_found": len(search_results) if isinstance(search_results, list) else len(search_results.keys()) if isinstance(search_results, dict) else 0,
                "columns_found": len(column_details) if isinstance(column_details, list) else len(column_details.keys()) if isinstance(column_details, dict) else 0,
                "model_names": [r.get("model_name", "") for r in search_results] if isinstance(search_results, list) else list(search_results.keys()) if isinstance(search_results, dict) else []
            },
            "relationship_results": related_models
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
        # Create a new graph
        workflow = StateGraph(AgentState)
        
        # Define nodes
        workflow.add_node("parse_question", self._parse_question)
        workflow.add_node("search_models", self._search_models)
        workflow.add_node("search_columns", self._search_columns)
        workflow.add_node("get_model_details", self._get_model_details)
        workflow.add_node("get_related_model_info", self._get_related_model_info)
        workflow.add_node("search_content", self._search_content)
        workflow.add_node("generate_response", self._generate_response)
        
        # Define routing logic based on question type
        def route_by_question_type(state: AgentState) -> str:
            question_analysis = state.get("question_analysis", {})
            question_type = question_analysis.get("question_type", "GENERAL")
            entities = question_analysis.get("entities", [])
            
            logger.info(f"Routing based on question type: {question_type}")
            
            # If we have specific model entities, start with model search
            if entities and question_type in ["MODEL_INFO", "DOCUMENTATION", "CODE_ENHANCEMENT", "DEPENDENCIES", "LINEAGE", "DEVELOPMENT"]:
                logger.info(f"Routing to search_models for {question_type} question")
                return "search_models"
            
            # If we have specific column entities, use column search
            column_patterns = []
            if "messages" in state and state["messages"]:
                last_message = state["messages"][-1].content
                column_patterns = self._extract_column_patterns(last_message)
            
            if column_patterns and question_type in ["COLUMN_INFO"]:
                logger.info("Routing to search_columns for column-specific question")
                return "search_columns"
            
            # For other types with entities, start with model search
            if entities:
                logger.info("Routing to search_models as fallback")
                return "search_models"
            
            # Default to content search for general questions
            logger.info("Routing to search_content as fallback")
            return "search_content"
        
        # Define edges based on routing logic
        workflow.add_conditional_edges(
            "parse_question",
            route_by_question_type,
            {
                "search_models": "search_models",
                "search_columns": "search_columns",
                "get_model_details": "get_model_details",
                "search_content": "search_content",
            }
        )
        
        # Connect model search to model details
        workflow.add_edge("search_models", "get_model_details")
        
        # Connect column search to related model info for context
        workflow.add_edge("search_columns", "get_related_model_info")
        
        # Connect model details to response generation
        workflow.add_edge("get_model_details", "generate_response")
        
        # Connect related model info to response generation
        workflow.add_edge("get_related_model_info", "generate_response")
        
        # Connect content search to response generation
        workflow.add_edge("search_content", "generate_response")
        
        # Set the entry point
        workflow.set_entry_point("parse_question")
        
        # Compile and return the graph
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
                state["search_results"] = {}
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
                state["search_results"] = {}
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
            state["search_results"] = {}
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
        """Search for DBT models based on the question."""
        results = {}
        entities = []
        search_terms = []
        
        try:
            question_analysis = state.get("question_analysis", {})
            entities = question_analysis.get("entities", [])
            search_terms = question_analysis.get("search_terms", [])
            original_question = ""
            
            if "messages" in state and state["messages"]:
                last_msg = state["messages"][-1]
                original_question = last_msg.content if hasattr(last_msg, 'content') else last_msg.get('content', '')
            
            # Early check for fct_orders in entities or question
            fct_orders_requested = any(entity.lower() in ["fct_orders", "fct_order"] for entity in entities)
            if not fct_orders_requested and original_question:
                fct_orders_requested = "fct_order" in original_question.lower()
            
            if fct_orders_requested:
                logger.info("fct_orders model specifically requested")
                # Try to get fct_orders content directly
                found, file_path, content = self._find_model_content("models/marts/core/fct_orders.sql")
                if found and content:
                    logger.info(f"Found fct_orders model at {file_path}")
                    # Create manually constructed result
                    fct_orders_result = {
                        "model_name": "fct_orders",
                        "file_path": file_path,
                        "content": content,
                        "match_type": "exact_match",
                        "description": "This model contains order-related metrics and data."
                    }
                    results["fct_orders"] = [fct_orders_result]
                    
                    # Also get any related files to fct_orders for more context
                    fct_orders_dbt_dir = os.path.dirname(file_path)
                    logger.info(f"Searching for schema files in directory: {fct_orders_dbt_dir}")
                    try:
                        schema_content = self.dbt_tools.get_file_content(os.path.join(fct_orders_dbt_dir, "schema.yml"))
                        if schema_content:
                            schema_result = {
                                "model_name": "schema.yml",
                                "file_path": os.path.join(fct_orders_dbt_dir, "schema.yml"),
                                "content": schema_content,
                                "match_type": "related_file",
                                "description": "Schema file containing metadata about the fct_orders model."
                            }
                            if "schema" not in results:
                                results["schema"] = []
                            results["schema"].append(schema_result)
                    except Exception as e:
                        logger.error(f"Error getting schema file: {str(e)}")
            
            # Proceed with normal search logic for all entities
            if entities:
                logger.info(f"Searching for models based on entities: {entities}")
                for entity in entities:
                    if self.dbt_tools:
                        try:
                            entity_results = self._search_by_keyword(entity)
                            if entity_results:
                                results[entity] = entity_results
                        except Exception as e:
                            logger.error(f"Error searching for entity {entity}: {str(e)}")
            
            # Execute content searches if specified
            if search_terms:
                logger.info(f"Searching for content with terms: {search_terms}")
                for term in search_terms:
                    if self.dbt_tools:
                        try:
                            term_results = self._search_by_keyword(term)
                            if term_results:
                                results[term] = term_results
                        except Exception as e:
                            logger.error(f"Error searching for term {term}: {str(e)}")
            
            logger.info(f"Search results: {len(results)} entities with matches")
        except Exception as e:
            logger.error(f"Error in _search_models: {str(e)}")
            return {"error": str(e), "message": "Failed to search models"}
        
        return {"results": results, "entities": entities}
    
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
                # Call with only the state parameter
                self._get_related_model_info(state)
            
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
            search_results = state.get("search_results", [])
            model_details = state.get("model_details", {})
            
            # Initialize related models container
            related_models = {}
            
            # Process each found model for dependencies
            # Handle cases where search_results is a list (newer implementation) or dict (older)
            if isinstance(search_results, list):
                logger.info(f"Processing related models for {len(search_results)} model results in list format")
                
                for result in search_results:
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
            if isinstance(model_details, dict):
                for model_name, details in model_details.items():
                    if model_name in related_models:
                        continue  # Skip if already processed
                    
                    if "dependencies" in details:
                        related_models[model_name] = details["dependencies"]
            elif isinstance(model_details, list):
                for model_info in model_details:
                    model_name = model_info.get("model_name", "")
                    if not model_name or model_name in related_models:
                        continue
                    
                    # Create the dependencies structure
                    related_models[model_name] = {
                        "upstream": model_info.get("references", []),
                        "downstream": []  # Can't determine downstream from just the model info
                    }
                    
                    # Add sources if available
                    if model_info.get("sources", []):
                        related_models[model_name]["sources"] = model_info.get("sources", [])
            
            # Update state with related models
            state["related_models"] = related_models
            
            logger.info(f"Gathered related model information for {len(related_models)} models")
            return state
        
        except Exception as e:
            logger.error(f"Error getting related model info: {str(e)}")
            state["related_models"] = {}
            return state
    
    def _search_content(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Search for content based on the question's search terms.
        
        Args:
            state: The current agent state
            
        Returns:
            Updated state with content search results
        """
        if not self.dbt_tools:
            return state
            
        # Get search terms from state
        analysis = state.get("question_analysis", {})
        search_terms = analysis.get("search_terms", [])
        entities = analysis.get("entities", [])
        
        if not search_terms and not entities:
            logger.warning("No search terms or entities found for content search")
            state["content_search"] = {"results": []}
            return state
        
        # Combine all search terms
        all_terms = []
        all_terms.extend(entities)  # Add entities first as they're more important
        for term in search_terms:
            if term not in all_terms:  # Avoid duplicates
                all_terms.append(term)
                
        # For technical terms, also add variations
        tech_terms = []
        for term in all_terms:
            if '_' in term:
                parts = term.split('_')
                for part in parts:
                    if len(part) > 3 and part not in tech_terms:  # Only add meaningful parts
                        tech_terms.append(part)
        
        # Add technical terms to the search list
        for term in tech_terms:
            if term not in all_terms:
                all_terms.append(term)
                
        # For each term, try to find relevant content
        results = []
        for entity in entities:
            try:
                # First, try to find the entity as a model
                entity_results = self.dbt_tools.search_model(entity, search_mode="output")
                if entity_results:
                    for result in entity_results:
                        # Add model-specific details
                        if hasattr(result, 'model_name') and result.model_name:
                            result_dict = {
                                "term": entity,
                                "match_type": "model",
                                "model_name": result.model_name,
                                "file_path": result.file_path if hasattr(result, 'file_path') else "",
                                "content": result.content if hasattr(result, 'content') else "",
                                "context": f"Model '{result.model_name}'"
                            }
                            results.append(result_dict)
                            logger.info(f"Found entity '{entity}' as model '{result.model_name}'")
            except Exception as e:
                logger.warning(f"Error searching for entity '{entity}' as model: {str(e)}")
            
            # Try fallback approaches if the model search failed or if this isn't a model name
            try:
                # Try direct path search - more reliable than content search
                logger.info(f"Trying file path search for entity: {entity}")
                path_pattern = f"*{entity}*"
                path_results = self.dbt_tools.search_file_path(path_pattern)
                
                if path_results:
                    for result in path_results:
                        # Add file-specific details
                        file_path = result.file_path if hasattr(result, 'file_path') else ""
                        content = result.content if hasattr(result, 'content') else ""
                        
                        # Try to get content if not present
                        if file_path and not content:
                            try:
                                content = self.dbt_tools.get_file_content(file_path)
                            except Exception as e:
                                logger.warning(f"Error getting content for {file_path}: {str(e)}")
                        
                        if file_path and content:
                            model_name = os.path.basename(file_path)
                            if model_name.endswith('.sql'):
                                model_name = model_name[:-4]
                                
                            result_dict = {
                                "term": entity,
                                "match_type": "file_path",
                                "model_name": model_name,
                                "file_path": file_path,
                                "content": content,
                                "context": f"File path containing '{entity}'"
                            }
                            results.append(result_dict)
                            logger.info(f"Found entity '{entity}' in file path '{file_path}'")
            except Exception as e:
                logger.warning(f"Error in file path search for entity '{entity}': {str(e)}")
                
            # Try content search as a last resort, with proper error handling
            try:
                logger.info(f"Trying content search for entity: {entity}")
                try:
                    # Attempt content search but handle the models_dir error
                    content_results = self.dbt_tools.search_content(entity)
                    
                    if content_results:
                        for result in content_results:
                            file_path = result.file_path if hasattr(result, 'file_path') else ""
                            content = result.content if hasattr(result, 'content') else ""
                            
                            # Try to get content if not present
                            if file_path and not content:
                                try:
                                    content = self.dbt_tools.get_file_content(file_path)
                                except Exception as e:
                                    logger.warning(f"Error getting content for {file_path}: {str(e)}")
                            
                            if file_path and content:
                                model_name = os.path.basename(file_path)
                                if model_name.endswith('.sql'):
                                    model_name = model_name[:-4]
                                    
                                result_dict = {
                                    "term": entity,
                                    "match_type": "content",
                                    "model_name": model_name,
                                    "file_path": file_path,
                                    "content": content,
                                    "context": f"Content containing '{entity}'"
                                }
                                results.append(result_dict)
                                logger.info(f"Found entity '{entity}' in content of '{file_path}'")
                except AttributeError as att_err:
                    if "models_dir" in str(att_err):
                        logger.warning(f"AttributeError with models_dir for entity '{entity}': {str(att_err)}")
                        # Use fallback approach - try manual search in all models
                        self._manual_content_search(entity, results)
                    else:
                        raise att_err
            except Exception as e:
                logger.warning(f"Error in content search for entity '{entity}': {str(e)}")
                # Try manual search as final fallback
                self._manual_content_search(entity, results)
        
        # For additional search terms (not entities), do a simpler search
        for term in search_terms:
            if term not in entities:  # Skip terms we've already processed as entities
                try:
                    logger.info(f"Trying file path search for term: {term}")
                    path_pattern = f"*{term}*"
                    path_results = self.dbt_tools.search_file_path(path_pattern)
                    
                    if path_results:
                        for result in path_results:
                            file_path = result.file_path if hasattr(result, 'file_path') else ""
                            content = result.content if hasattr(result, 'content') else ""
                            
                            # Try to get content if not present
                            if file_path and not content:
                                try:
                                    content = self.dbt_tools.get_file_content(file_path)
                                except Exception as e:
                                    logger.warning(f"Error getting content for {file_path}: {str(e)}")
                            
                            if file_path and content:
                                model_name = os.path.basename(file_path)
                                if model_name.endswith('.sql'):
                                    model_name = model_name[:-4]
                                    
                                result_dict = {
                                    "term": term,
                                    "match_type": "file_path",
                                    "model_name": model_name,
                                    "file_path": file_path,
                                    "content": content,
                                    "context": f"File path containing '{term}'"
                                }
                                results.append(result_dict)
                                logger.info(f"Found term '{term}' in file path '{file_path}'")
                except Exception as e:
                    logger.warning(f"Error in file path search for term '{term}': {str(e)}")
                
                # Try to find terms in content safely
                try:
                    try:
                        logger.info(f"Trying content search for term: {term}")
                        content_results = self.dbt_tools.search_content(term)
                        
                        if content_results:
                            for result in content_results:
                                file_path = result.file_path if hasattr(result, 'file_path') else ""
                                content = result.content if hasattr(result, 'content') else ""
                                
                                # Try to get content if not present
                                if file_path and not content:
                                    try:
                                        content = self.dbt_tools.get_file_content(file_path)
                                    except Exception as e:
                                        logger.warning(f"Error getting content for {file_path}: {str(e)}")
                                
                                if file_path and content:
                                    model_name = os.path.basename(file_path)
                                    if model_name.endswith('.sql'):
                                        model_name = model_name[:-4]
                                        
                                    result_dict = {
                                        "term": term,
                                        "match_type": "content",
                                        "model_name": model_name,
                                        "file_path": file_path,
                                        "content": content,
                                        "context": f"Content containing '{term}'"
                                    }
                                    results.append(result_dict)
                                    logger.info(f"Found term '{term}' in content of '{file_path}'")
                    except AttributeError as att_err:
                        if "models_dir" in str(att_err):
                            logger.warning(f"AttributeError with models_dir for term '{term}': {str(att_err)}")
                            # Use fallback approach - try manual search in all models
                            self._manual_content_search(term, results)
                        else:
                            raise att_err
                except Exception as e:
                    logger.warning(f"Error in content search for term '{term}': {str(e)}")
                    # Try manual search as final fallback
                    self._manual_content_search(term, results)
        
        # Look for calculation terms as well
        for term in all_terms:
            # If the term appears to be a calculation reference (contains math symbols or aggregation)
            calc_indicators = ['sum(', 'avg(', 'count(', 'max(', 'min(', '+', '-', '*', '/', '=']
            if any(indicator in term.lower() for indicator in calc_indicators):
                try:
                    # Extract the calculation term (remove operators)
                    calc_term = re.sub(r'[+\-*/=()]', ' ', term).strip()
                    if len(calc_term) < 3:
                        continue  # Skip if too short after cleaning
                        
                    logger.info(f"Trying calculation search for: {calc_term}")
                    try:
                        calc_results = self.dbt_tools.search_content(calc_term)
                        
                        if calc_results:
                            for result in calc_results:
                                file_path = result.file_path if hasattr(result, 'file_path') else ""
                                content = result.content if hasattr(result, 'content') else ""
                                
                                # Try to get content if not present
                                if file_path and not content:
                                    try:
                                        content = self.dbt_tools.get_file_content(file_path)
                                    except Exception as e:
                                        logger.warning(f"Error getting content for {file_path}: {str(e)}")
                                
                                if file_path and content:
                                    model_name = os.path.basename(file_path)
                                    if model_name.endswith('.sql'):
                                        model_name = model_name[:-4]
                                        
                                    result_dict = {
                                        "term": term,
                                        "match_type": "calculation",
                                        "model_name": model_name,
                                        "file_path": file_path,
                                        "content": content,
                                        "context": f"Possible calculation for '{term}'"
                                    }
                                    results.append(result_dict)
                                    logger.info(f"Found possible calculation '{term}' in '{file_path}'")
                    except AttributeError as att_err:
                        if "models_dir" in str(att_err):
                            logger.warning(f"AttributeError with models_dir for calc_term '{calc_term}': {str(att_err)}")
                            # Use fallback approach
                            self._manual_content_search(calc_term, results, match_type="calculation")
                        else:
                            raise att_err
                except Exception as e:
                    logger.warning(f"Error searching for calculation term '{term}': {str(e)}")
        
        # Deduplicate results based on file path to avoid redundancy
        unique_results = []
        file_paths_seen = set()
        
        for result in results:
            file_path = result.get("file_path", "")
            if file_path and file_path not in file_paths_seen:
                file_paths_seen.add(file_path)
                unique_results.append(result)
        
        # Update state with content search results
        state["content_search"] = {"results": unique_results}
        logger.info(f"Found {len(unique_results)} unique content search results")
        
        return state
    
    def _manual_content_search(self, search_term: str, results: List[Dict], match_type: str = "content") -> None:
        """
        Perform a manual search through all models for a term.
        This is a fallback when the normal content search fails.
        
        Args:
            search_term: The term to search for
            results: The results list to append matches to 
            match_type: The type of match to report
        """
        try:
            # Get all models
            model_files = self.dbt_tools.get_all_models()
            if not model_files:
                return
                
            logger.info(f"Performing manual search for '{search_term}' across {len(model_files)} models")
            
            # Search each model
            for model in model_files:
                try:
                    # Get file path
                    file_path = self.dbt_tools.file_scanner.get_model_file_path(model)
                    if not file_path:
                        continue
                        
                    # Get content
                    content = self.dbt_tools.get_file_content(file_path)
                    if not content:
                        continue
                        
                    # Simple text search
                    if search_term.lower() in content.lower():
                        result_dict = {
                            "term": search_term,
                            "match_type": match_type,
                            "model_name": model,
                            "file_path": file_path,
                            "content": content,
                            "context": f"Manual search found '{search_term}' in model"
                        }
                        results.append(result_dict)
                        logger.info(f"Manual search found '{search_term}' in model '{model}'")
                except Exception as e:
                    logger.warning(f"Error in manual search for model {model}: {str(e)}")
        except Exception as e:
            logger.warning(f"Error in manual content search: {str(e)}")

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
        """
        Generate a response based on the question analysis and search results.
        
        Args:
            state: The current agent state
            
        Returns:
            Updated state with the generated response
        """
        try:
            # Extract relevant information from state
            question_analysis = state.get("question_analysis", {})
            question_type = question_analysis.get("question_type", "GENERAL")
            entities = question_analysis.get("entities", [])
            
            # Extract the original question
            messages = state.get("messages", [])
            question = messages[-1].content if messages else ""
            
            # Log the question type and entity count for debugging
            logger.info(f"Generating response for {question_type} question with {len(entities)} entities")
            
            # Get instructions based on question type
            instructions = self._get_instructions_for_type(question_type, question)
            logger.info(f"Using instruction type: {question_type}")
            
            # Build context with relevant information from search results
            formatted_prompt = self._format_results_for_prompt(state)
            
            if formatted_prompt and len(formatted_prompt) > 0:
                logger.info(f"Found model information to include in prompt. Content length: {len(formatted_prompt)}")
            else:
                logger.warning("No model information found for any search method! Response may be generic.")
                
            # Debug the content of the search results for fct_orders specifically
            if "fct_orders" in question.lower() or "fct_order" in question.lower():
                fct_orders_found = False
                if "search_results" in state and state["search_results"]:
                    for result in state["search_results"]:
                        if result.get("model_name", "").lower() == "fct_orders":
                            fct_orders_found = True
                            logger.info(f"Found fct_orders model in search results: {result.get('file_path', '')}")
                if "model_details" in state and state["model_details"]:
                    if isinstance(state["model_details"], list):
                        for model in state["model_details"]:
                            if model.get("model_name", "").lower() == "fct_orders":
                                fct_orders_found = True
                                logger.info(f"Found fct_orders model in model_details: {model.get('file_path', '')}")
                    elif isinstance(state["model_details"], dict):
                        if state["model_details"].get("model_name", "").lower() == "fct_orders":
                            fct_orders_found = True
                            logger.info(f"Found fct_orders model in model_details (dict): {state['model_details'].get('file_path', '')}")
                
                if not fct_orders_found:
                    logger.warning("Did not find fct_orders in any search results or model details!")
                    # Try direct file access as a last resort
                    model_path = "models/marts/core/fct_orders.sql"
                    found, file_path, content = self._find_model_content(model_path)
                    if found and content:
                        logger.info(f"Found fct_orders directly at path: {file_path}")
                        # Add it to the formatted prompt
                        formatted_prompt += f"\n\n## Direct File Access\n\n### Model: fct_orders\n### Path: {file_path}\n```sql\n{content}\n```\n"
                
            # Combine instructions with information from search results
            system_message = f"""
            You are a SQL and data modeling expert, specializing in dbt analytics engineering.
            
            QUESTION: {question}
            
            INSTRUCTIONS FOR THIS SPECIFIC QUESTION:
            {instructions}
            
            RELEVANT INFORMATION FROM DBT PROJECT:
            {formatted_prompt}
            """
                
            messages = [
                SystemMessage(content=system_message),
                HumanMessage(content=question)
            ]
            
            # Log the prompt size for debugging
            logger.info(f"Formatted prompt contains {len(system_message)} characters")
            
            # For debugging content
            if "fct_orders" in question.lower() or "fct_order" in question.lower():
                logger.info("Prompt includes exact model code for analysis.")
                
            # Call the language model to generate a response
            response = self._safe_llm_call(messages)
            
            # For code enhancement responses, validate the output
            if question_type == "CODE_ENHANCEMENT":
                # Find model content from search results
                if "model_details" in state and state["model_details"]:
                    if isinstance(state["model_details"], list) and len(state["model_details"]) > 0:
                        # Use the first model's content
                        model_details = state["model_details"][0]
                        file_path = model_details.get("file_path", "")
                        content = model_details.get("content", "")
                    else:
                        file_path = state["model_details"].get("file_path", "")
                        content = state["model_details"].get("content", "")
                        
                    if file_path and content:
                        # Validate and potentially fix the response
                        response = self._validate_code_enhancement_response(response, content, file_path, question)
                elif "search_results" in state and state["search_results"]:
                    # Find the first result with content
                    for result in state["search_results"]:
                        file_path = result.get("file_path", "")
                        content = result.get("content", "")
                        if file_path and content:
                            # Validate and potentially fix the response
                            response = self._validate_code_enhancement_response(response, content, file_path, question)
                            break
            
            # Update state with the generated response
            state["final_response"] = response
                        
            return state
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            state["final_response"] = f"I encountered an error while generating a response: {str(e)}"
            return state

    def _get_code_enhancement_instructions(self, query: str) -> str:
        """Get instructions for code enhancement questions."""
        return """
        # DBT Code Enhancement Instructions

        ## DEVELOPER ROLE
        Act as an experienced DBT developer who focuses on implementing EXACTLY what the user has requested.
        Your primary goal is to modify the EXISTING DBT MODEL CODE rather than writing SQL DDL statements.
        
        ## IMPORTANT: DBT-SPECIFIC REQUIREMENTS
        - NEVER generate `ALTER TABLE` or `UPDATE` statements - these are not valid in DBT models
        - ALWAYS modify the existing DBT model SQL code itself
        - Preserve all DBT references ({{ ref('model_name') }}), macros, and Jinja syntax
        - Maintain the existing config blocks and materialization settings
        - For calculated columns, add the calculation directly in the SELECT statement
        - Respect the project's existing coding style and patterns

        ## Analysis Approach
        1. Identify EXACTLY what changes the user is requesting (new column, calculation, etc.)
        2. Examine the EXISTING CTEs, SELECT statements, and column definitions
        3. Determine the best place to add the new column or calculation within the model's structure
        4. Implement the change by modifying the appropriate SELECT statement(s)
        5. Ensure the change works with DBT incremental models if applicable

        ## Response Format
        Structure your response as follows:

        ### 1. Implementation Summary
        - Brief description of the EXACT change implemented
        - Explanation of where and how the change was added to the model

        ### 2. Complete Modified DBT Model
        - Show the COMPLETE modified SQL code with your changes
        - Include a comment (-- ADDED or -- MODIFIED) next to any lines you've changed
        - Present the FULL code, not just the altered section

        ### 3. Technical Explanation
        - Explain how your implementation works within the DBT model
        - Note any considerations for testing or validation

        ## CRITICAL REQUIREMENTS
        - Focus on MODIFYING THE EXISTING MODEL CODE, not creating DDL statements
        - NEVER write `ALTER TABLE` or `UPDATE` statements - these don't work in DBT
        - PRESERVE all DBT-specific syntax like {{ ref() }}, {{ config() }}, and Jinja blocks
        - Respect the model's existing materialization strategy and configuration
        - Ensure any added calculations respect the model's grain and existing logic
        - For new columns, add them directly in the appropriate SELECT statement
        - The implementation must be executable as-is in a DBT project
        """

    def _get_documentation_instructions(self, query: str) -> str:
        """Get instructions for documentation questions."""
        return """
        # Technical Documentation Instructions
        
        ## Documentation Approach
        Create comprehensive, technical documentation that is concise and focused on actual implementation.
        Prioritize code examples, relationship diagrams, and technical specifications over general descriptions.
        
        ## Documentation Structure
        
        ### 1. Technical Overview
        - Purpose and function of the component being documented
        - Key technical characteristics (no marketing language)
        - Critical dependencies and relationships
        
        ### 2. Implementation Details
        - ACTUAL CODE examples from the codebase
        - SQL structure, CTEs, and key transformations
        - Configuration details and materialization settings
        - Performance considerations with technical rationale
        
        ### 3. Schema Documentation
        - Column definitions with data types
        - Primary/foreign key relationships
        - Detailed calculation logic for derived fields
        - Constraints and validations
        
        ### 4. Technical Relationships
        - Detailed dependency chain (upstream/downstream)
        - Interface specifications
        - Integration points with other systems/models
        
        ## Critical Requirements
        - BE TECHNICAL: Focus on implementation details, not general concepts
        - BE CONCISE: Prioritize facts and code over lengthy explanations
        - BE ACCURATE: All documentation must match the actual code implementation
        - SHOW CODE: Include actual code snippets for all important components
        - PRIORITIZE RELATIONSHIPS: Clearly document how components interact
        
        ## Style Guidelines
        - Use proper technical terminology consistently
        - Format all code blocks properly
        - Use bullet points for lists of specifications
        - Keep paragraphs short and focused on a single concept
        - Organize information hierarchically from most to least important
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
        """Format search results for inclusion in the prompt."""
        try:
            formatted_results = []
            
            # Format model search results
            if 'search_results' in state and state['search_results']:
                results = state['search_results']
                
                formatted_results.append(f"## DBT Model Search Results")
                
                if isinstance(results, list):
                    for i, result in enumerate(results):
                        if not result:
                            continue
                            
                        model_name = result.get('model_name', '')
                        file_path = result.get('file_path', '')
                        content = result.get('content', '')
                        
                        if file_path and content:
                            formatted_results.append(f"\n### Model: {model_name}")
                            formatted_results.append(f"### Path: {file_path}")
                            formatted_results.append(f"```sql\n{content}\n```\n")
                            
                            # Include schema info if available
                            schema_info = result.get('schema_info', {})
                            if schema_info:
                                formatted_results.append("#### Schema Information:")
                                for key, value in schema_info.items():
                                    formatted_results.append(f"- **{key}**: {value}")
                                formatted_results.append("")
                
                elif isinstance(results, dict):
                    for key, value in results.items():
                        if isinstance(value, list):
                            for result in value:
                                if not result:
                                    continue
                                    
                                model_name = result.get('model_name', '')
                                file_path = result.get('file_path', '')
                                content = result.get('content', '')
                                
                                if file_path and content:
                                    formatted_results.append(f"\n### Model: {model_name}")
                                    formatted_results.append(f"### Path: {file_path}")
                                    formatted_results.append(f"```sql\n{content}\n```\n")
                                    
                                    # Include schema info if available
                                    schema_info = result.get('schema_info', {})
                                    if schema_info:
                                        formatted_results.append("#### Schema Information:")
                                        for schema_key, schema_value in schema_info.items():
                                            formatted_results.append(f"- **{schema_key}**: {schema_value}")
                                        formatted_results.append("")
            
            # Format model details
            if 'model_details' in state and state['model_details']:
                model_details = state['model_details']
                
                if isinstance(model_details, list):
                    model_details_section = []
                    model_details_section.append(f"\n## Model Details")
                    
                    for model in model_details:
                        model_name = model.get('model_name', '')
                        file_path = model.get('file_path', '')
                        content = model.get('content', '')
                        
                        if model_name and file_path:
                            model_details_section.append(f"\n### TARGET MODEL: {model_name}")
                            model_details_section.append(f"### MODEL PATH: {file_path}")
                            
                            if content:
                                model_details_section.append(f"```sql\n{content}\n```\n")
                            
                            # Include dependencies
                            deps = model.get('dependencies', {})
                            if deps:
                                model_details_section.append("#### Dependencies:")
                                for dep_type, dep_list in deps.items():
                                    model_details_section.append(f"- **{dep_type}**: {', '.join(dep_list)}")
                                model_details_section.append("")
                    
                    formatted_results.extend(model_details_section)
                
                elif isinstance(model_details, dict):
                    model_details_section = []
                    model_details_section.append(f"\n## Model Details")
                    
                    model_name = model_details.get('model_name', '')
                    file_path = model_details.get('file_path', '')
                    content = model_details.get('content', '')
                    
                    if model_name and file_path:
                        model_details_section.append(f"\n### TARGET MODEL: {model_name}")
                        model_details_section.append(f"### MODEL PATH: {file_path}")
                        
                        if content:
                            model_details_section.append(f"```sql\n{content}\n```\n")
                        
                        # Include dependencies
                        deps = model_details.get('dependencies', {})
                        if deps:
                            model_details_section.append("#### Dependencies:")
                            for dep_type, dep_list in deps.items():
                                model_details_section.append(f"- **{dep_type}**: {', '.join(dep_list)}")
                            model_details_section.append("")
                    
                    formatted_results.extend(model_details_section)
            
            # Format column details
            if 'column_details' in state and state['column_details']:
                column_details = state['column_details']
                
                if isinstance(column_details, list) and column_details:
                    formatted_results.append(f"\n## Column Search Results")
                    
                    for result in column_details:
                        model_name = result.get('model_name', '')
                        column_name = result.get('column_name', '')
                        calculation = result.get('calculation', '')
                        
                        if model_name and column_name:
                            formatted_results.append(f"\n### Column: {column_name} in {model_name}")
                            
                            if calculation:
                                formatted_results.append(f"#### Calculation:")
                                formatted_results.append(f"```sql\n{calculation}\n```\n")
                
                elif isinstance(column_details, dict) and column_details:
                    formatted_results.append(f"\n## Column Search Results")
                    
                    for key, results in column_details.items():
                        if isinstance(results, list):
                            for result in results:
                                model_name = result.get('model_name', '')
                                column_name = result.get('column_name', '')
                                calculation = result.get('calculation', '')
                                
                                if model_name and column_name:
                                    formatted_results.append(f"\n### Column: {column_name} in {model_name}")
                                    
                                    if calculation:
                                        formatted_results.append(f"#### Calculation:")
                                        formatted_results.append(f"```sql\n{calculation}\n```\n")
            
            # Format content search results
            if 'content_search' in state and state['content_search']:
                content_search = state['content_search']
                
                if isinstance(content_search, list) and content_search:
                    formatted_results.append(f"\n## Content Search Results")
                    
                    for result in content_search:
                        model_name = result.get('model_name', '')
                        file_path = result.get('file_path', '')
                        match_contexts = result.get('match_contexts', [])
                        
                        if model_name and file_path and match_contexts:
                            formatted_results.append(f"\n### Matches in {model_name} ({file_path})")
                            
                            for context in match_contexts:
                                formatted_results.append(f"```\n{context}\n```\n")
                
                elif isinstance(content_search, dict) and content_search:
                    formatted_results.append(f"\n## Content Search Results")
                    
                    for key, results in content_search.items():
                        if isinstance(results, list):
                            for result in results:
                                model_name = result.get('model_name', '')
                                file_path = result.get('file_path', '')
                                match_contexts = result.get('match_contexts', [])
                                
                                if model_name and file_path and match_contexts:
                                    formatted_results.append(f"\n### Matches in {model_name} ({file_path})")
                                    
                                    for context in match_contexts:
                                        formatted_results.append(f"```\n{context}\n```\n")
            
            return "\n".join(formatted_results)
        except Exception as e:
            logger.error(f"Error formatting results for prompt: {str(e)}")
            return "Error formatting search results."
    
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
        
        instructions_map = {
            "MODEL_INFO": self._get_model_explanation_instructions,
            "LINEAGE": self._get_lineage_instructions,
            "DEPENDENCIES": self._get_dependency_instructions,
            "CODE_ENHANCEMENT": self._get_code_enhancement_instructions,
            "DOCUMENTATION": self._get_documentation_instructions,
            "DEVELOPMENT": self._get_development_instructions,
            "GENERAL": self._get_general_instructions
        }
        
        # Log which instruction type we're using
        logger.info(f"Using instruction type: {question_type}")
        
        # Get the instruction function for this question type
        instruction_fn = instructions_map.get(question_type, self._get_general_instructions)
        
        # Generate and return the instructions
        instructions = instruction_fn(question)
        
        # Explicitly clarify the role and expectations based on question type
        if question_type == "CODE_ENHANCEMENT":
            instructions = """
            # YOU ARE ACTING AS A DEVELOPER, NOT AN EXPLAINER
            Your task is to provide COMPLETE, WORKING code with specific enhancements, not just explain.
            
            """ + instructions
        elif question_type == "DEVELOPMENT":
            instructions = """
            # YOU ARE ACTING AS A DEVELOPER, NOT AN EXPLAINER
            Your task is to implement a complete, working solution with full code, not just explain concepts.
            
            """ + instructions
        elif question_type == "DOCUMENTATION":
            instructions = """
            # YOU ARE CREATING COMPLETE TECHNICAL DOCUMENTATION
            Your task is to create comprehensive documentation that would be suitable for inclusion in a DBT docs site,
            focusing on both technical details AND business context.
            
            """ + instructions
        elif question_type == "MODEL_INFO":
            instructions = """
            # YOU ARE EXPLAINING THE MODEL'S PURPOSE AND STRUCTURE
            Your task is to explain what this model does, how it works, and why it exists - focus on clarity and explanation.
            
            """ + instructions
        
        return instructions

    def _get_model_explanation_instructions(self, query: str) -> str:
        """Get instructions for model explanation questions."""
        return """
        # Model Explanation Instructions

        ## Your Role
        You are acting as a data architect who explains DBT models clearly and concisely for both technical and business users. 
        Focus on providing essential information without unnecessary examples or verbose explanations.

        ## Explanation Structure
        Create a crisp, well-structured explanation with these sections:

        ### 1. Model Overview (2-3 sentences)
        - Model type (fact/dimension/etc.)
        - Core purpose of the model
        - Key business use cases
        
        ### 2. Technical Implementation (focus on code)
        - Core SQL structure with actual code snippets
        - Important CTEs and their purpose
        - Key joins and filters
        - Critical transformations
        
        ### 3. Data Relationships
        - Upstream dependencies (source tables, staging models)
        - Downstream consumers
        - Primary/foreign key relationships
        - Important column lineage
        
        ### 4. Key Metrics & Business Logic (when applicable)
        - Critical business metrics with their calculation logic
        - Business rules implemented in the model
        - Data validations and quality checks

        ## Critical Requirements
        - BE CONCISE: Prioritize code and technical details over lengthy explanations
        - NO GENERIC EXAMPLES: Don't include example queries unless specifically requested
        - SHOW ACTUAL CODE: Include real SQL snippets from the model when explaining logic
        - TECHNICAL ACCURACY: Ensure all explanations match the actual model implementation
        - FOCUS ON RELATIONSHIPS: Emphasize how the model connects to other tables
        
        ## Response Style
        - Use technical terms correctly but explain them when necessary
        - Be direct and to-the-point
        - Use bullet points for lists
        - Include relevant code blocks with proper formatting
        - Focus on what makes this model unique and important
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
        """
        Get instructions for dependency analysis questions.
        
        Args:
            query: The user's query
            
        Returns:
            Instructions for generating a dependency analysis response
        """
        return """
        # Dependency Analysis Instructions
        
        ## Analysis Requirements
        1. **Analyze the provided model code** to identify ALL dependencies
        2. **Do NOT ask for the model code** - it has already been provided in the search results
        3. **Focus on complete dependency chain** - both direct and indirect dependencies
        4. **Include source tables** - identify the ultimate data sources
        5. **Explain the dependency flow** in business terms
        
        ## Response Format
        Structure your response with these sections:
        
        ### 1. Model Overview
        - Provide a brief description of the model's purpose
        - Identify its materialization type (table, view, incremental, etc.)
        - Mention when this model is typically refreshed (if identifiable)
        
        ### 2. Direct Dependencies
        - List all models or sources referenced using `ref()` or `source()`
        - For each dependency, describe:
          - Its purpose in relation to this model
          - The join or relationship type
          - Key SQL patterns used in the relationship
        
        ### 3. Indirect Dependencies (Upstream)
        - Identify second-level dependencies (models that feed into direct dependencies)
        - Trace dependencies back to source tables where possible
        - Present this as a hierarchical structure
        
        ### 4. Source Tables
        - List all ultimate source tables at the origin of the dependency chain
        - Group by schema/database if applicable
        - Describe the critical source tables and their importance

        ### 5. Dependency Visualization
        - Present a text-based visualization of the dependency flow (upstream/downstream)
        - Use indentation to show hierarchy
        - Format as:
          ```
          fct_orders
          ├── stg_tpch_orders
          │   └── raw.tpch.orders
          └── order_items
              └── stg_tpch_order_items
                  └── raw.tpch.order_items
          ```
        
        ### 6. SQL Reference Examples
        - Include code snippets showing how each key dependency is referenced
        - Format as:
          ```sql
          -- Reference to stg_tpch_orders
          select * from {{ ref('stg_tpch_orders') }}
          ```
        
        ## Key Requirements
        - Be specific to the EXACT model provided in the search results
        - Trace each dependency chain back to source tables where possible
        - Include both the model names AND their file paths when available
        - Note any config parameters that affect dependencies (e.g. pre/post hooks)
        - Explain WHY certain dependencies exist (business context)
        - YOUR ANALYSIS MUST BE BASED ON THE PROVIDED MODEL, NOT A GENERIC RESPONSE
        """

    def _get_development_instructions(self, query: str) -> str:
        """Get instructions for development questions."""
        return """
        # DBT Development Instructions

        ## DEVELOPER ROLE
        Act as an experienced DBT developer who implements practical, executable solutions.
        Your goal is to provide COMPLETE, PRODUCTION-READY CODE - not just explanations.
        
        ## Development Approach
        1. UNDERSTAND THE EXACT REQUIREMENT from the user's question
        2. CREATE COMPLETE DBT MODEL FILES with all necessary components
        3. FOLLOW DBT BEST PRACTICES including proper ref() usage, documentation, and testing
        4. OPTIMIZE FOR PERFORMANCE with appropriate materialization strategies
        5. PROVIDE IMPLEMENTATION DETAILS that a developer can immediately use
        
        ## Response Format
        Structure your response with these sections:
        
        ### 1. Solution Overview
        - Brief summary of the implementation approach
        - Key technical decisions and patterns used
        
        ### 2. COMPLETE CODE IMPLEMENTATION
        - Full, executable DBT model SQL code
        - Include ALL necessary files (models, schema.yml, etc.)
        - Use proper file headers and documentation
        
        ### 3. Implementation Details
        - Step-by-step explanation of the code
        - How the solution addresses the specific requirement
        - Any configuration or materialization decisions
        
        ### 4. Testing & Validation
        - Specific test cases to validate the implementation
        - Sample dbt_project.yml configuration if needed
        
        ## CRITICAL REQUIREMENTS
        - PROVIDE COMPLETE, RUNNABLE CODE - not conceptual explanations
        - USE MODERN DBT SYNTAX and follow dbt best practices
        - INCLUDE ALL NECESSARY FILES (SQL, YAML, etc.)
        - OPTIMIZE for appropriate materialization strategies
        - ADD THOROUGH DOCUMENTATION including descriptions and tests
        - ENSURE models follow proper naming conventions
        - STRUCTURE your code with appropriate CTEs and comments
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
        """
        Extract detailed information about models from search results
        
        Args:
            state: The current agent state
            
        Returns:
            Updated state with model details
        """
        if not self.dbt_tools:
            state["model_details"] = []
            return state
            
        # Get search results
        search_results = state.get("search_results", [])
        
        if not search_results:
            logger.info("No search results available to gather model details")
            state["model_details"] = []
            return state
            
        logger.info(f"Processing model details from {len(search_results)} search results")
        
        # Initialize detailed model information
        model_details = []
        
        # Process each search result
        for result in search_results:
            try:
                # Extract basic information
                model_name = result.get("model_name", "")
                file_path = result.get("file_path", "")
                content = result.get("content", "")
                
                if not model_name or not file_path or not content:
                    logger.warning(f"Skipping model detail extraction due to missing information: {model_name}")
                    continue
                
                # Initialize model detail dictionary
                model_detail = {
                    "model_name": model_name,
                    "file_path": file_path,
                    "content": content,
                    "model_type": self._extract_model_type(content),
                    "columns": [],
                    "description": "",
                    "references": [],
                    "sources": [],
                    "lineage": {}
                }
                
                # Get schema information if available
                try:
                    schema_info = self.dbt_tools.file_scanner.get_schema_for_model_path(file_path)
                    if schema_info:
                        # Extract description from schema
                        model_detail["description"] = schema_info.get("description", "")
                        
                        # Extract column information
                        columns = schema_info.get("columns", [])
                        model_detail["columns"] = columns
                except Exception as e:
                    logger.warning(f"Error extracting schema for {model_name}: {str(e)}")
                
                # If no columns from schema, try to extract from SQL
                if not model_detail["columns"]:
                    try:
                        model_detail["columns"] = self._extract_model_columns(content)
                    except Exception as e:
                        logger.warning(f"Error extracting columns from SQL for {model_name}: {str(e)}")
                
                # Get model dependencies
                try:
                    # Extract dependencies with more robust error handling
                    dependencies = self._get_model_dependencies(model_name, file_path)
                    model_detail["references"] = dependencies.get("refs", [])
                    model_detail["sources"] = dependencies.get("sources", [])
                    
                    # Get lineage information with retry mechanisms
                    retry_count = 0
                    max_retries = 2
                    lineage = {}
                    
                    while retry_count <= max_retries:
                        try:
                            # Try to get deep lineage with depth 2 (upstream and downstream)
                            lineage = self.dbt_tools.get_model_lineage(model_name, depth=2)
                            if lineage:
                                break
                        except Exception as e:
                            logger.warning(f"Error getting lineage (attempt {retry_count+1}): {str(e)}")
                            
                        # Failed attempt - try alternate approach on retry
                        if retry_count == 0:
                            # On first retry, try getting lineage using file path
                            try:
                                # Extract model name from path for better matching
                                alt_model_name = os.path.basename(file_path)
                                if alt_model_name.endswith('.sql'):
                                    alt_model_name = alt_model_name[:-4]
                                    
                                lineage = self.dbt_tools.get_model_lineage(alt_model_name, depth=1)
                                if lineage:
                                    break
                            except Exception as inner_e:
                                logger.warning(f"Error in alternate lineage approach: {str(inner_e)}")
                        
                        # Increment retry counter
                        retry_count += 1
                        
                        # On final retry, use direct relationship extraction
                        if retry_count == max_retries:
                            try:
                                # Extract relationships directly from the content
                                refs = re.findall(r'{{\s*ref\([\'"]([^\'"]+)[\'"]\)\s*}}', content)
                                sources = re.findall(r'{{\s*source\([\'"]([^\'"]+)[\'"],\s*[\'"]([^\'"]+)[\'"]\)\s*}}', content)
                                
                                lineage = {
                                    "model": model_name,
                                    "upstream": [{"model": ref} for ref in refs],
                                    "sources": [f"{source[0]}.{source[1]}" for source in sources],
                                    "downstream": []  # Can't determine downstream from content
                                }
                            except Exception as direct_e:
                                logger.warning(f"Error in direct relationship extraction: {str(direct_e)}")
                    
                    model_detail["lineage"] = lineage
                except Exception as e:
                    logger.warning(f"Error getting dependencies for {model_name}: {str(e)}")
                
                # Add to model details
                model_details.append(model_detail)
            except Exception as e:
                logger.warning(f"Error processing model details: {str(e)}")
        
        logger.info(f"Gathered detailed information for {len(model_details)} models")
        state["model_details"] = model_details
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
        """
        Extract dependencies for a model with robust error handling
        
        This uses multiple approaches to ensure the most complete dependency information:
        1. Try using the dbt_tools API
        2. Use regex-based extraction from the model content
        3. Build upstreams using multiple file resolution strategies
        
        Args:
            model_name: Name of the model
            file_path: Path to the model file
            
        Returns:
            Dictionary with refs and sources lists
        """
        if not self.dbt_tools:
            return {"refs": [], "sources": []}
            
        dependencies = {
            "refs": [],
            "sources": []
        }
        
        try:
            # Step 1: Try using the dbt_tools API first (most reliable)
            try:
                # Try the find_related_models method
                relations = self.dbt_tools.find_related_models(model_name)
                if relations:
                    dependencies["refs"] = relations.get("upstream", [])
                    
                    # If the file scanner has the model, try to get sources from there
                    model_sources = self.dbt_tools.file_scanner.get_model_sources(model_name)
                    if model_sources:
                        dependencies["sources"] = model_sources
            except Exception as e:
                logger.warning(f"Error getting dependencies from API for {model_name}: {str(e)}")
            
            # If we don't have dependencies yet, try alternate approaches
            if not dependencies["refs"] and not dependencies["sources"]:
                # Step 2: Get model content and extract dependencies with regex
                try:
                    # Try to get content
                    content = ""
                    if file_path:
                        content = self.dbt_tools.get_file_content(file_path)
                    
                    if content:
                        # Extract refs
                        ref_pattern = r'{{\s*ref\([\'"]([^\'"]+)[\'"]\)\s*}}'
                        refs = re.findall(ref_pattern, content)
                        dependencies["refs"] = list(set(refs))  # Deduplicate
                        
                        # Extract sources
                        source_pattern = r'{{\s*source\([\'"]([^\'"]+)[\'"],\s*[\'"]([^\'"]+)[\'"]\)\s*}}'
                        sources = re.findall(source_pattern, content)
                        dependencies["sources"] = [f"{source[0]}.{source[1]}" for source in sources]
                except Exception as e:
                    logger.warning(f"Error extracting dependencies from content for {model_name}: {str(e)}")
                
                # Step 3: Try the alternative file scanner mapping
                if not dependencies["refs"]:
                    try:
                        # Try to get the model file scanner relation info
                        if hasattr(self.dbt_tools.file_scanner, 'model_relationships'):
                            relations = self.dbt_tools.file_scanner.model_relationships.get(model_name, {})
                            if relations:
                                dependencies["refs"] = relations.get("refs", [])
                                dependencies["sources"] = relations.get("sources", [])
                    except Exception as e:
                        logger.warning(f"Error getting dependencies from file scanner for {model_name}: {str(e)}")
            
            # Deduplicate dependencies lists
            dependencies["refs"] = list(set(dependencies["refs"]))
            dependencies["sources"] = list(set(dependencies["sources"]))
            
            return dependencies
        
        except Exception as e:
            logger.error(f"Error getting model dependencies: {str(e)}")
            return {"refs": [], "sources": []}

    def _validate_code_enhancement_response(self, response: str, original_code: str, model_path: str, question: str) -> str:
        """Validate and format the code enhancement response."""
        logger.info(f"Validating code enhancement response for {model_path}")
        
        # Check if model path is in the response
        if model_path and not re.search(r'model[s]?\s*path[:]?\s*' + re.escape(model_path), response, re.IGNORECASE):
            logger.warning("Code enhancement response does not include the correct model path")
            # Add model path to the response if missing
            header = f"# CODE_ENHANCEMENT\nModel path: {model_path}\n\n"
            if not response.strip().startswith('# CODE_ENHANCEMENT'):
                response = header + response
            else:
                # Replace just the header part to include the model path
                response = re.sub(r'# CODE_ENHANCEMENT.*?(\n\n|$)', header, response, flags=re.DOTALL)
        
        # Check for DBT-specific validation
        if "ALTER TABLE" in response or "UPDATE " in response:
            logger.warning("Response contains DDL statements which are not valid in DBT models")
            warning_note = (
                "\n\n**NOTE: The above solution contains SQL DDL statements (ALTER TABLE/UPDATE) "
                "which are not valid in DBT models. Instead, you should modify the DBT model's SELECT "
                "statement to include the new column with its calculation directly.**\n"
            )
            response += warning_note
        
        # Check if the response contains complete code implementation
        if "```sql" not in response.lower() and "```" not in response:
            logger.warning("Code enhancement response does not contain a proper code block")
            response += "\n\nPlease ensure your implementation includes the complete SQL code in a code block."
        
        # Ensure response has sections for implementation, code, and explanation
        section_checks = {
            "summary": re.search(r'(implementation|solution)\s*summary|summary', response, re.IGNORECASE) is not None,
            "code": re.search(r'(complete\s*modified\s*dbt\s*model|code\s*implementation)', response, re.IGNORECASE) is not None,
            "explanation": re.search(r'(technical\s*explanation|explanation)', response, re.IGNORECASE) is not None
        }
        
        if not all(section_checks.values()):
            missing_sections = [k for k, v in section_checks.items() if not v]
            logger.warning(f"Response is missing required sections: {', '.join(missing_sections)}")
            
            section_notes = "\n\n**NOTE: Your response should include the following sections:**\n"
            if not section_checks["summary"]:
                section_notes += "- Implementation Summary\n"
            if not section_checks["code"]:
                section_notes += "- Complete Modified DBT Model\n"
            if not section_checks["explanation"]:
                section_notes += "- Technical Explanation\n"
                
            response += section_notes
            
        return response

    def _find_model_content(self, path_or_model: str) -> Tuple[bool, str, str]:
        """
        Attempt to find a model's content by path or name using multiple methods.
        
        Args:
            path_or_model: Either a model name or a path to search for
            
        Returns:
            Tuple of (found_content, file_path, content)
        """
        if not path_or_model or not self.dbt_tools:
            return False, "", ""
            
        logger.info(f"Attempting to find model content for: {path_or_model}")
        
        # Remove .sql extension if present for model name searches
        model_name = path_or_model
        if model_name.endswith('.sql'):
            model_name = model_name[:-4]
            
        # Get the base name if it's a path
        if '/' in model_name:
            model_name = os.path.basename(model_name)
        
        # Step 1: Try direct DBT model search
        try:
            logger.info(f"Trying direct model search for: {model_name}")
            model_results = self.dbt_tools.search_model(model_name)
            
            if model_results and len(model_results) > 0:
                result = model_results[0]  # Take the first result
                result_dict = self._convert_search_result_to_dict(result)
                
                file_path = result_dict.get("file_path", "")
                content = result_dict.get("content", "")
                
                if file_path and content:
                    logger.info(f"Found model content via model search at: {file_path}")
                    return True, file_path, content
        except Exception as e:
            logger.error(f"Error in direct model search: {str(e)}")
        
        # Step 2: Try direct file access if path_or_model looks like a file path
        if '/' in path_or_model or path_or_model.endswith('.sql'):
            try:
                logger.info(f"Trying direct file access for: {path_or_model}")
                content = self.dbt_tools.get_file_content(path_or_model)
                
                if content:
                    logger.info(f"Found model content via direct file access: {path_or_model}")
                    return True, path_or_model, content
            except Exception as e:
                logger.error(f"Error in direct file access: {str(e)}")
        
        # Step 3: Try to find the file path first, then get content
        try:
            logger.info(f"Trying to get model file path for: {model_name}")
            file_path = self.dbt_tools.file_scanner.get_model_file_path(model_name)
            
            if file_path:
                try:
                    logger.info(f"Found file path: {file_path}, getting content")
                    content = self.dbt_tools.get_file_content(file_path)
                    
                    if content:
                        logger.info(f"Found model content via file path lookup: {file_path}")
                        return True, file_path, content
                except Exception as e:
                    logger.error(f"Error getting content from file path: {str(e)}")
        except Exception as e:
            logger.error(f"Error getting model file path: {str(e)}")
        
        # Step 4: Try path variations
        try:
            variations = [
                f"models/{model_name}.sql",
                f"models/marts/core/{model_name}.sql",
                f"models/marts/{model_name}.sql",
                f"models/staging/{model_name}.sql",
                f"models/intermediate/{model_name}.sql"
            ]
            
            logger.info(f"Trying path variations for: {model_name}")
            for var_path in variations:
                try:
                    content = self.dbt_tools.get_file_content(var_path)
                    
                    if content:
                        logger.info(f"Found model content via path variation: {var_path}")
                        return True, var_path, content
                except Exception:
                    # Skip errors for missing files
                    continue
        except Exception as e:
            logger.error(f"Error trying path variations: {str(e)}")
        
        # Step 5: Special case for fct_orders which appears frequently
        if model_name.lower() in ["fct_orders", "fct_order"]:
            specific_path = "models/marts/core/fct_orders.sql"
            try:
                logger.info(f"Trying special case path for fct_orders: {specific_path}")
                content = self.dbt_tools.get_file_content(specific_path)
                
                if content:
                    logger.info(f"Found fct_orders content via special case path: {specific_path}")
                    return True, specific_path, content
            except Exception as e:
                logger.error(f"Error in special case for fct_orders: {str(e)}")
        
        # Content not found through any method
        logger.warning(f"Could not find model content for: {path_or_model}")
        return False, "", ""

    def _convert_search_result_to_dict(self, search_result) -> Dict[str, Any]:
        """
        Convert a SearchResult object to a dictionary.
        
        Args:
            search_result: SearchResult object or dictionary
            
        Returns:
            Dictionary representation of the search result
        """
        if isinstance(search_result, dict):
            return search_result
            
        # Convert SearchResult object to dictionary
        try:
            result_dict = {}
            # Handle custom object with attributes
            if hasattr(search_result, '__dict__'):
                # First try to use asdict from dataclasses if it's a dataclass
                try:
                    from dataclasses import asdict
                    result_dict = asdict(search_result)
                except:
                    # Otherwise use __dict__ directly
                    result_dict = vars(search_result)
            # For objects that don't have a standard __dict__
            else:
                # Try common attributes found in SearchResult
                for attr in ['model_name', 'file_path', 'content', 'match_type', 
                             'description', 'column_name', 'calculation', 'match_contexts']:
                    if hasattr(search_result, attr):
                        result_dict[attr] = getattr(search_result, attr)
                        
            return result_dict
        except Exception as e:
            logger.error(f"Error converting search result to dict: {str(e)}")
            # Create a minimal result dictionary with whatever we can extract
            return {
                "model_name": getattr(search_result, "model_name", "unknown"),
                "file_path": getattr(search_result, "file_path", ""),
                "content": getattr(search_result, "content", ""),
                "match_type": "fallback_conversion"
            }

    def _extract_model_columns(self, content: str) -> List[Dict]:
        """
        Extract columns and their descriptions from SQL model content.
        
        Args:
            content: The SQL content of the model
            
        Returns:
            List of column information dictionaries
        """
        if not content:
            return []
            
        columns = []
        try:
            # Look for select statement patterns
            select_pattern = re.compile(r'(?:select|SELECT).*?(?:from|FROM)', re.DOTALL)
            select_matches = select_pattern.findall(content)
            
            # If we found select statements, extract columns from them
            if select_matches:
                for select_clause in select_matches:
                    # Split by commas but ignore commas inside functions
                    in_function = 0
                    current_part = ""
                    parts = []
                    
                    for char in select_clause:
                        if char == '(':
                            in_function += 1
                            current_part += char
                        elif char == ')':
                            in_function -= 1
                            current_part += char
                        elif char == ',' and in_function == 0:
                            parts.append(current_part.strip())
                            current_part = ""
                        else:
                            current_part += char
                    
                    # Add the last part if not empty
                    if current_part.strip():
                        parts.append(current_part.strip())
                    
                    # Process each column expression
                    for part in parts:
                        # Skip 'from' keyword that might be included
                        if part.lower().startswith('from '):
                            continue
                            
                        # Look for "as column_name" pattern
                        as_match = re.search(r'(?:as|AS)\s+([a-zA-Z0-9_]+)\s*$', part)
                        if as_match:
                            column_name = as_match.group(1)
                            expression = part[:as_match.start()].strip()
                            
                            # Try to determine data type from expression
                            data_type = self._infer_data_type(expression)
                            
                            # Try to create a simple description from the expression
                            description = self._create_column_description(column_name, expression)
                            
                            # Add column info
                            columns.append({
                                "name": column_name,
                                "data_type": data_type,
                                "expression": expression,
                                "description": description
                            })
                        else:
                            # Handle direct column references without AS
                            # For example: "order_id" or "customers.name"
                            direct_match = re.search(r'([a-zA-Z0-9_]+)(?:\.([a-zA-Z0-9_]+))?\s*$', part)
                            if direct_match:
                                if direct_match.group(2):  # Table.column format
                                    table_name = direct_match.group(1)
                                    column_name = direct_match.group(2)
                                    description = f"Column {column_name} from {table_name}"
                                else:  # Just column name
                                    column_name = direct_match.group(1)
                                    description = f"Column {column_name}"
                                
                                # Add column info
                                columns.append({
                                    "name": column_name,
                                    "data_type": "unknown",
                                    "expression": part.strip(),
                                    "description": description
                                })
            
            # If we have very few columns, try to extract from CTE definitions
            if len(columns) < 2:
                # Look for CTE (Common Table Expression) definitions
                cte_pattern = re.compile(r'(?:with|WITH)\s+([a-zA-Z0-9_]+)\s+as\s*\((.*?)\)', re.DOTALL)
                cte_matches = cte_pattern.findall(content)
                
                for cte_name, cte_content in cte_matches:
                    # Extract columns from each CTE
                    cte_columns = self._extract_model_columns(cte_content)
                    for col in cte_columns:
                        col["source"] = cte_name
                        columns.append(col)
            
            return columns
            
        except Exception as e:
            logger.error(f"Error extracting columns: {str(e)}")
            return []
    
    def _infer_data_type(self, expression: str) -> str:
        """
        Infer the data type of a column based on its expression.
        
        Args:
            expression: The SQL expression for the column
            
        Returns:
            Inferred data type as string
        """
        expression = expression.lower()
        
        # Numeric functions and patterns
        if any(func in expression for func in ['sum(', 'count(', 'avg(', 'min(', 'max(']):
            return "numeric"
        elif re.search(r'::(?:int|integer|bigint|decimal|numeric|float)', expression):
            return "numeric"
        
        # String functions and patterns
        if any(func in expression for func in ['concat(', 'upper(', 'lower(', 'trim(']):
            return "string"
        elif re.search(r'::(?:varchar|text|string|char)', expression):
            return "string"
        
        # Date/time functions and patterns
        if any(func in expression for func in ['date(', 'current_date', 'timestamp', 'to_date']):
            return "timestamp"
        elif re.search(r'::(?:date|timestamp|datetime)', expression):
            return "timestamp"
        
        # Boolean patterns
        if any(pattern in expression for pattern in ['= true', '= false', 'is null', 'is not null', '::boolean']):
            return "boolean"
        elif ' and ' in expression or ' or ' in expression:
            return "boolean"
        
        # Default to string for unknown types
        return "unknown"
    
    def _create_column_description(self, column_name: str, expression: str) -> str:
        """
        Create a human-readable description for a column based on its name and expression.
        
        Args:
            column_name: Name of the column
            expression: SQL expression used to calculate the column
            
        Returns:
            Human-readable description
        """
        column_name = column_name.lower()
        expression = expression.lower()
        
        # ID columns
        if column_name.endswith('_id') or column_name.endswith('_key'):
            if 'primary' in expression or 'unique' in expression:
                return f"Primary identifier for this entity"
            else:
                entity = column_name.replace('_id', '').replace('_key', '')
                return f"Reference to {entity} entity"
        
        # Date columns
        if any(date_part in column_name for date_part in ['_date', '_time', '_timestamp']):
            for date_part in ['_date', '_time', '_timestamp']:
                if date_part in column_name:
                    event = column_name.replace(date_part, '')
                    return f"Time when {event.replace('_', ' ')} occurred"
        
        # Amount columns
        if any(amount_part in column_name for amount_part in ['amount', 'total', 'sum', 'price', 'cost']):
            if 'discount' in column_name:
                return "Discount amount applied"
            elif 'tax' in column_name:
                return "Tax amount applied"
            elif 'net' in column_name:
                return "Net amount after discounts and taxes"
            elif 'gross' in column_name:
                return "Gross amount before discounts and taxes"
            else:
                return f"Monetary value for {column_name.replace('_', ' ')}"
        
        # Count columns
        if any(count_part in column_name for count_part in ['count', 'quantity', 'number', 'qty']):
            entity = re.sub(r'count_|num_|quantity_|qty_', '', column_name)
            return f"Count of {entity.replace('_', ' ')}"
        
        # Status columns
        if any(status_part in column_name for status_part in ['status', 'state', 'type', 'category']):
            entity = column_name.replace('_status', '').replace('_state', '').replace('_type', '').replace('_category', '')
            return f"Classification or status of {entity.replace('_', ' ')}"
        
        # Flag columns
        if column_name.startswith('is_') or column_name.startswith('has_'):
            return f"Flag indicating if {column_name[3:].replace('_', ' ')}"
        
        # Look for expressions with common aggregations
        if 'sum(' in expression:
            matches = re.findall(r'sum\((.*?)\)', expression)
            if matches:
                return f"Sum of {matches[0].strip()}"
        
        if 'count(' in expression:
            return f"Count of records"
        
        # Default description based on column name
        return f"{column_name.replace('_', ' ').title()}"

    def _search_by_keyword(self, keyword: str) -> List[Dict[str, Any]]:
        """
        Search for models by keyword, using the content search API.
        
        Args:
            keyword: The keyword to search for
            
        Returns:
            List of search results
        """
        if not self.dbt_tools:
            logger.warning("DBT tools not initialized for keyword search")
            return []
            
        results = []
        logger.info(f"Performing content-based search for keyword: {keyword}")
        
        try:
            # Try searching the model first
            logger.info(f"Trying search_model with keyword: {keyword}")
            model_results = self.dbt_tools.search_model(keyword)
            if model_results:
                # Add unique results
                for result in model_results:
                    # Convert SearchResult object to dict if needed
                    result_dict = self._convert_search_result_to_dict(result)
                    if result_dict not in results:
                        results.append(result_dict)
            
            # Try searching content
            logger.info(f"Trying content search with keyword: {keyword}")
            content_results = self.dbt_tools.search_content(keyword)
            if content_results:
                # Add unique results
                for result in content_results:
                    # Convert SearchResult object to dict if needed
                    result_dict = self._convert_search_result_to_dict(result)
                    if result_dict not in results:
                        results.append(result_dict)
            
            # Try file path search with pattern
            logger.info(f"Trying file path search with pattern: *{keyword}*")
            path_results = self.dbt_tools.search_file_path(f"*{keyword}*")
            if path_results:
                # Add unique results
                for result in path_results:
                    # Convert SearchResult object to dict if needed
                    result_dict = self._convert_search_result_to_dict(result)
                    if result_dict not in results:
                        results.append(result_dict)
            
            # Try direct file access
            logger.info(f"Trying direct file access for: {keyword}")
            file_content = self.dbt_tools.get_file_content(keyword)
            if file_content:
                # Create a search result
                result = {
                    "model_name": os.path.basename(keyword).replace(".sql", ""),
                    "file_path": keyword,
                    "content": file_content,
                    "match_type": "direct_access"
                }
                results.append(result)
                
            # Try getting all models and check for matching names
            all_models = self.dbt_tools.get_all_models()
            for model_name in all_models:
                if keyword.lower() in model_name.lower():
                    # Try to get the model details
                    model_results = self.dbt_tools.search_model(model_name)
                    if model_results:
                        for result in model_results:
                            # Convert SearchResult object to dict if needed
                            result_dict = self._convert_search_result_to_dict(result)
                            if result_dict not in results:
                                results.append(result_dict)
                
            # If all else fails, try a manual search through each model
            if not results:
                all_models = self.dbt_tools.get_all_models()
                for model_name in all_models:
                    try:
                        model_results = self.dbt_tools.search_model(model_name)
                        if model_results:
                            for result in model_results:
                                # Convert SearchResult object to dict if needed
                                result_dict = self._convert_search_result_to_dict(result)
                                content = result_dict.get("content", "")
                                if content and keyword.lower() in content.lower():
                                    if result_dict not in results:
                                        result_dict["match_type"] = "manual_content_match"
                                        results.append(result_dict)
                    except Exception as e:
                        logger.error(f"Error searching model {model_name}: {str(e)}")
            
            return results
        except Exception as e:
            logger.error(f"Error in keyword search: {str(e)}")
            return []

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