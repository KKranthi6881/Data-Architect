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
import glob

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
            """Route to the appropriate flow based on question type."""
            question_analysis = state.get("question_analysis", {})
            question_type = question_analysis.get("question_type", "MODEL_INFO") if question_analysis else "MODEL_INFO"
            
            logging.info(f"Routing based on question type: {question_type}")
            
            # Simplify to just five question types
            if question_type == "MODEL_INFO":
                return "search_models"
            elif question_type == "LINEAGE":
                return "search_models"  # Lineage is handled in get_model_details
            elif question_type == "DEPENDENCIES":
                return "search_models"  # Dependencies need model search and details
            elif question_type == "CODE_ENHANCEMENT":
                return "search_models"  # Code enhancement needs model content
            elif question_type == "DEVELOPMENT":
                return "search_models"  # Development needs model content and context
            else:
                # Default to MODEL_INFO for any other type
                logging.info(f"Unknown question type: {question_type}. Defaulting to MODEL_INFO.")
                return "search_models"
        
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
               
               - CODE_ENHANCEMENT: Questions about improving, modifying, or optimizing existing code. Examples: "How can I make this query faster?", "How do I modify X to include Y?", "How can I add/change column Z to an existing model?"
               
               - DEVELOPMENT: Questions about implementing entirely new features or models. Examples: "How do I create a new model for X?", "How do I implement Y?", "Can you help me develop Z from scratch?"
            
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
            - DEVELOPMENT is for creating entirely new models or implementing brand new functionality
            - CODE_ENHANCEMENT is for modifying existing models or adding features to them
            - LINEAGE is specifically about how data flows between models (upstream/downstream)
            - DEPENDENCIES is about understanding what specific models depend on each other
            - MODEL_INFO is for understanding what an existing model does
            
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

    def _search_models(self, state):
        """
        Search for model information related to the question.
        
        Args:
            state (dict): The current state of the conversation.
            
        Returns:
            dict: A dictionary containing search results and entities.
        """
        try:
            entities = state.get("question_analysis", {}).get("entities", [])
            search_terms = state.get("question_analysis", {}).get("search_terms", [])
            
            # Initialize search results and model details
            search_results = {}
            # Initialize model_details as a list, not a dict
            state["model_details"] = []
            
            # Use direct file paths first if they exist
            direct_file_paths = [entity for entity in entities if entity.endswith('.sql') or 'models/' in entity]
            
            if direct_file_paths:
                logging.info(f"Direct file paths found: {direct_file_paths}")
                for file_path in direct_file_paths:
                    model_content = self._find_model_content(file_path)
                    if model_content and isinstance(model_content, tuple) and len(model_content) >= 3:
                        found, path, content = model_content
                        if found and content:
                            # Extract model name from path
                            model_name = path.split('/')[-1].replace('.sql', '')
                            logging.info(f"Found direct model content for: {model_name} at {path}")
                            
                            # Check for dependencies if DEPENDENCIES question
                            if state.get("question_analysis", {}).get("question_type") == "DEPENDENCIES":
                                dependencies = self._get_model_dependencies(model_name, path)
                                if dependencies:
                                    logging.info(f"Found dependencies for {model_name}: {dependencies}")
                                    
                            # Create model detail
                            model_detail = {
                                "model_name": model_name,
                                "file_path": path,
                                "content": content,
                                "model_type": "model",  # Assuming it's a model
                            }
                            
                            # Add dependencies if they exist
                            if 'dependencies' in locals() and dependencies:
                                model_detail["dependencies"] = dependencies
                            
                            # Add to model details
                            state["model_details"].append(model_detail)
                            
                            # Add to search results
                            search_results[model_name] = {
                                "content": content,
                                "file_path": path
                            }
            
            # Continue with regular search if no direct paths or additional info needed
            if not direct_file_paths or not search_results:
                for entity in entities:
                    if self.dbt_tools:
                        try:
                            results = self.dbt_tools.search_model(entity)
                            if results:
                                for result in results:
                                    if isinstance(result, dict):
                                        model_name = result.get("model_name", "")
                                        if model_name:
                                            search_results[model_name] = result
                        except Exception as e:
                            logging.error(f"Error searching for entity {entity}: {str(e)}")
                
                if not search_results and search_terms:
                    for term in search_terms:
                        if self.dbt_tools:
                            try:
                                results = self.dbt_tools.search_model(term)
                                if results:
                                    for result in results:
                                        if isinstance(result, dict):
                                            model_name = result.get("model_name", "")
                                            if model_name:
                                                search_results[model_name] = result
                            except Exception as e:
                                logging.error(f"Error searching for term {term}: {str(e)}")
            
            # Log search results
            if search_results:
                logging.info(f"Found search results: {list(search_results.keys())}")
                
                # Process model details if they don't already exist
                if not state["model_details"]:
                    for model_name, model_info in search_results.items():
                        file_path = model_info.get("file_path", "")
                        
                        # Check for dependencies if DEPENDENCIES question
                        dependencies = None
                        if state.get("question_analysis", {}).get("question_type") == "DEPENDENCIES":
                            dependencies = self._get_model_dependencies(model_name, file_path)
                            if dependencies:
                                logging.info(f"Found dependencies for {model_name}: {dependencies}")
                        
                        # Create model detail
                        model_detail = {
                            "model_name": model_name,
                            "file_path": file_path,
                            "content": model_info.get("content", ""),
                            "model_type": model_info.get("model_type", "model"),
                        }
                        
                        # Add dependencies if they exist
                        if dependencies:
                            model_detail["dependencies"] = dependencies
                        
                        # Add to model details
                        state["model_details"].append(model_detail)
            else:
                logging.warning("No search results found for entities or search terms.")
            
            # Save search results and entities to state
            state["search_results"] = search_results
            state["entities"] = entities
            
            return {"search_results": search_results, "entities": entities}
        
        except Exception as e:
            logging.error(f"Error in _search_models: {str(e)}")
            logging.error(traceback.format_exc())
            state["search_results"] = {}
            state["entities"] = entities
            state["model_details"] = []
            return {"search_results": {}, "entities": entities}
    
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
        """Generate a response based on the analysis and model details."""
        try:
            # Get key information from state
            search_results = state.get("search_results", {})
            model_details = state.get("model_details", [])
            original_question = state["messages"][-1].content
            question_analysis = state.get("question_analysis", {})
            question_type = question_analysis.get("question_type", "MODEL_INFO")
            entities = question_analysis.get("entities", [])
            
            # Special handling for DEPENDENCIES questions with direct file paths
            if question_type == "DEPENDENCIES" and entities and not model_details:
                direct_file_paths = [entity for entity in entities if '/' in entity and (entity.endswith('.sql') or 'models/' in entity)]
                
                if direct_file_paths:
                    for file_path in direct_file_paths:
                        found, path, content = self._find_model_content(file_path)
                        
                        if found:
                            # Extract model name from the path
                            model_name = os.path.splitext(os.path.basename(path))[0]
                            
                            # Get dependencies directly
                            dependencies = self._get_model_dependencies(model_name, path)
                            
                            # Create model detail with dependencies
                            model_detail = {
                                "model_name": model_name,
                                "file_path": path,
                                "content": content,
                                "model_type": self._extract_model_type(content),
                                "dependencies": dependencies
                            }
                            
                            # Add to model_details
                            if not model_details:
                                model_details = []
                            model_details.append(model_detail)
                            state["model_details"] = model_details
                            
                            logging.info(f"Added dependencies for {model_name} in _generate_response")
                            break
            
            # Format search results for prompt
            formatted_results = self._format_results_for_prompt(state)
            
            # Check if we have meaningful results
            has_model_details = model_details and len(model_details) > 0
            has_search_results = search_results and (isinstance(search_results, dict) and any(search_results.values()) or 
                                                     isinstance(search_results, list) and len(search_results) > 0)
            
            if not has_model_details and not has_search_results:
                logging.warning("No model information found for any search method! Response may be generic.")
            
            # Get type-specific instructions
            instruction_prompt = self._get_instructions_for_type(question_type, original_question)
            
            # Create variable to store visualization data for the frontend
            lineage_data = None
            
            # Special formatting for different question types
            # For DEPENDENCIES questions
            if question_type == "DEPENDENCIES" and has_model_details:
                formatted_results = self._format_dependencies_for_prompt(state, model_details, formatted_results)
                lineage_data = self._create_lineage_visualization(model_details)
            
            # For LINEAGE questions
            elif question_type == "LINEAGE" and has_model_details:
                formatted_results = self._format_lineage_for_prompt(state, model_details, formatted_results)
                lineage_data = self._create_lineage_visualization(model_details)
                
            # For CODE_ENHANCEMENT questions
            elif question_type == "CODE_ENHANCEMENT" and has_model_details:
                formatted_results = self._format_code_enhancement_for_prompt(state, model_details, formatted_results)
                
            # For DEVELOPMENT questions
            elif question_type == "DEVELOPMENT" and has_model_details:
                formatted_results = self._format_development_for_prompt(state, model_details, formatted_results)
                
            # For MODEL_INFO questions
            elif question_type == "MODEL_INFO" and has_model_details:
                formatted_results = self._format_model_info_for_prompt(state, model_details, formatted_results)
            
            # Create the prompt for the language model
            prompt_messages = [
                SystemMessage(content=instruction_prompt),
                HumanMessage(content=f"""
                QUESTION: {original_question}
                
                SEARCH RESULTS:
                {formatted_results}
                
                Please provide a comprehensive response to the question based on the search results above.
                """)
            ]
            
            # Call the LLM and handle errors
            try:
                response = self._safe_llm_call(prompt_messages)
                
                # Enhance the response with specialized formatting based on question type
                if question_type == "CODE_ENHANCEMENT":
                    response = self._validate_code_enhancement_response(response, state)
                    
                elif question_type == "MODEL_INFO":
                    response = self._validate_documentation_response(response, question_type)
                    
                elif question_type == "DEPENDENCIES" and has_model_details:
                    # Add dependency visualization if not already present
                    response = self._validate_dependency_response(response, model_details)
                    
                elif question_type == "LINEAGE" and has_model_details:
                    # Add lineage visualization if not already present
                    response = self._validate_lineage_response(response, model_details)
                    
                elif question_type == "DEVELOPMENT":
                    # Ensure development response has code blocks
                    response = self._validate_development_response(response)
                
                # Append visualization data for frontend rendering if applicable
                if lineage_data and (question_type == "DEPENDENCIES" or question_type == "LINEAGE"):
                    # Use a special tag that the frontend can detect and extract for visualization
                    visualization_json = f"\n\n# LINEAGE_VISUALIZATION\n```json\n{json.dumps(lineage_data, indent=2)}\n```"
                    response = response + visualization_json
                    
                    # Also log the visualization data for debugging
                    logging.info(f"Added lineage visualization data with {len(lineage_data.get('models', []))} models and {len(lineage_data.get('edges', []))} edges")
                
                # Set the final response in the state
                state["final_response"] = response
                
                return state
            except Exception as e:
                logging.error(f"Error generating response: {str(e)}")
                state["final_response"] = f"I apologize, but I encountered an error while generating a response. Error details: {str(e)}"
                return state
        except Exception as e:
            logging.error(f"Error in _generate_response: {str(e)}")
            state["final_response"] = f"I apologize, but I encountered an error while processing your request. Error details: {str(e)}"
            return state

    def _get_instructions_for_type(self, question_type: str, question: str) -> str:
        """Get specific instructions based on question type."""
        
        instructions_map = {
            "MODEL_INFO": self._get_model_explanation_instructions,
            "LINEAGE": self._get_lineage_instructions,
            "DEPENDENCIES": self._get_dependency_instructions,
            "CODE_ENHANCEMENT": self._get_code_enhancement_instructions,
            "DEVELOPMENT": self._get_development_instructions
        }
        
        # Log which instruction type we're using
        logging.info(f"Using instruction type: {question_type}")
        
        # Default to MODEL_INFO for any unmapped question types
        if question_type not in instructions_map:
            logging.warning(f"Unknown question type: {question_type}. Defaulting to MODEL_INFO.")
            question_type = "MODEL_INFO"
        
        # Get the instruction function for this question type
        instruction_fn = instructions_map.get(question_type)
        
        # Generate and return the instructions
        instructions = instruction_fn(question)
        
        # Explicitly clarify the role and expectations based on question type
        if question_type == "CODE_ENHANCEMENT":
            instructions = """
            # YOU ARE ACTING AS A DEVELOPER, NOT AN EXPLAINER
            Your task is to provide COMPLETE, WORKING code with specific enhancements, not just explain.
            Focus on providing clear, step-by-step code blocks that show the changes needed.
            Include explanations of WHY you're making each change.
            
            """ + instructions
        elif question_type == "DEVELOPMENT":
            instructions = """
            # YOU ARE ACTING AS A DEVELOPER, NOT AN EXPLAINER
            Your task is to implement a complete, working solution with full code, not just explain concepts.
            Break down your implementation into clear, logical steps.
            Provide complete code blocks that can be executed directly.
            
            """ + instructions
        elif question_type == "MODEL_INFO":
            instructions = """
            # YOU ARE EXPLAINING THE MODEL'S PURPOSE AND STRUCTURE
            Your task is to explain what this model does, how it works, and why it exists - focus on clarity and explanation.
            Use code blocks to illustrate important SQL patterns and transformations.
            
            """ + instructions
        elif question_type == "LINEAGE":
            instructions = """
            # YOU ARE CREATING A VISUAL REPRESENTATION OF DATA FLOW
            Your task is to show how data flows through models, with clear visualizations of upstream and downstream dependencies.
            Use text-based diagrams to illustrate these relationships clearly.
            
            """ + instructions
        elif question_type == "DEPENDENCIES":
            instructions = """
            # YOU ARE ANALYZING MODEL DEPENDENCIES
            Your task is to identify and explain all dependencies of the specified model, both upstream and downstream.
            Use a structured, hierarchical format to show these relationships clearly.
            Include visualizations where helpful.
            
            """ + instructions
        
        return instructions

    def _get_model_explanation_instructions(self, query: str) -> str:
        """Get instructions for model explanation questions."""
        return """
        You are a DBT expert explaining a data model.
        
        # TASK
        You must analyze and explain a DBT model, its purpose, structure, and key transformations.
        
        # APPROACH
        1. Identify the model in question from the context
        2. Analyze its structure and SQL logic
        3. Determine its purpose within the data warehouse
        4. Explain key transformations and business logic
        
        # REQUIREMENTS
        - Clearly explain the model's overall purpose
        - Describe the key business entities it represents
        - Explain important SQL transformations
        - Identify source data and dependencies
        - Note any important calculations or business rules
        
        # EXAMPLE RESPONSE FORMAT
        ## Model Overview: [model_name]
        [Brief explanation of the model's purpose and role]
        
        ## Key Transformations
        [Explanation of main SQL transformations]
        
        ## Business Logic
        [Explanation of key business rules and calculations]
        
        ## Data Sources
        [Description of where the data comes from]
        
        ## Sample SQL
        ```sql
        -- Key portions of SQL with explanations
        ```
        """

    def _get_lineage_instructions(self, query: str) -> str:
        """Get instructions for lineage questions"""
        return """
    You are a DBT expert explaining model lineage.
    
    # TASK
    You must analyze and visualize the lineage of a DBT model, showing upstream and downstream dependencies.
    
    # APPROACH
    1. Identify the model in question from the context
    2. Analyze its direct upstream dependencies (models it references)
    3. Analyze its direct downstream dependencies (models that reference it)
    4. Create a visualization of these relationships
    
    # REQUIREMENTS
    - Clearly identify the central model in question
    - Show all direct upstream dependencies
    - Show all direct downstream dependencies
    - Provide a visual representation of the lineage
    - Explain key dependency relationships and their purpose
    
    # EXAMPLE RESPONSE FORMAT
    ## Model Lineage: [model_name]
    
    [Visual representation of lineage relationships]
    
    ## Upstream Dependencies
    [List and explanation of upstream models]
    
    ## Downstream Dependencies
    [List and explanation of downstream models]
    
    ## Key Relationships
    [Explanation of the most important dependency relationships]
    """

    def _get_dependency_instructions(self, query: str) -> str:
        """Get instructions for dependency questions."""
        return """
    You are a DBT expert analyzing model dependencies.
    
    # TASK
    You must provide a detailed analysis of a model's dependencies, showing what it depends on and what depends on it.
    
    # APPROACH
    1. Identify the model in question from the context
    2. Analyze its direct and indirect upstream dependencies
    3. Analyze its direct and indirect downstream dependencies
    4. Identify source tables and final output consumers
    
    # REQUIREMENTS
    - Clearly identify the central model
    - List and explain all direct upstream dependencies
    - List and explain all direct downstream dependencies
    - Show indirect dependencies where relevant
    - Explain the purpose of key dependency relationships
    - Include a visualization of the dependency network
    
    # EXAMPLE RESPONSE FORMAT
    ## Model Dependencies: [model_name]
    
    [Brief description of the model and its purpose]
    
    ## Direct Dependencies
    
    ### Upstream (models this depends on)
    - [model_name_1]: [brief description of relationship]
    - [model_name_2]: [brief description of relationship]
    
    ### Downstream (models that depend on this)
    - [model_name_3]: [brief description of relationship]
    - [model_name_4]: [brief description of relationship]
    
    ## Source Tables
    [List of original source tables feeding this model]
    
    ## Dependency Visualization
    [Text-based visualization of the dependency network]
    
    ## SQL Reference Example
    ```sql
    -- Example showing how this model references its dependencies
    ```
    """

    def _get_development_instructions(self, query: str) -> str:
        """Get instructions for development questions."""
        return """
    You are a DBT expert assisting with developing new SQL models.
    
    # TASK
    You must create a new, complete DBT model based on the user's requirements.
    
    # APPROACH
    1. Analyze the requirements and identify the necessary data transformations
    2. Design a new model that fulfills these requirements
    3. Implement the model with complete, working SQL code
    4. Follow DBT best practices for naming, structure, and documentation
    
    # REQUIREMENTS
    - Provide a complete implementation of the new model
    - Include properly formatted SQL code that follows DBT standards
    - Add comments explaining key transformations
    - Include appropriate tests and documentation
    - Ensure the model is optimized for performance and maintainability
    
    # EXAMPLE RESPONSE FORMAT
    ## Model Overview
    [Brief explanation of the model's purpose and design]
    
    ## Implementation
    ```sql
    -- Complete SQL implementation of the new model
    ```
    
    ## Tests
    ```yaml
    # Recommended tests for this model
    ```
    
    ## Documentation
    ```yaml
    # Recommended documentation for this model
    ```
    """

    def _get_code_enhancement_instructions(self, query: str) -> str:
        """Get instructions for code enhancement questions."""
        return """
    You are a DBT expert assisting with enhancing SQL code. 
    
    # TASK
    You must provide a complete, working solution to enhance a DBT model based on the user request.

    # APPROACH
    1. Analyze the model content and understand its current structure
    2. Identify the specific enhancements needed based on the user's request
    3. Implement those changes with complete, working SQL code
    4. Ensure your solution aligns with DBT best practices
    
    # REQUIREMENTS
    - Provide a complete implementation of the enhanced model, not just the changed parts
    - Include SQL code that is properly formatted and follows DBT standards
    - Add comments explaining key transformations and the rationale for the changes
    - Ensure your solution is optimized for performance and maintainability
    
    # EXAMPLE RESPONSE FORMAT
    ## Enhancement Summary
    [Brief explanation of what changes you made and why]
    
    ## Implementation
    ```sql
    -- Complete SQL implementation with your enhancements
    ```
    
    ## Key Changes
    [Bullet points explaining specific changes and their purpose]
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
        Get detailed information about a model, including content, columns, and dependencies.
        
        Enhanced for enterprise repositories with:
        - Deep nesting
        - Complex folder structures
        - Multiple file patterns
        - Various naming conventions
        
        Args:
            state: Current state dictionary with search results
            
        Returns:
            Updated state with model details
        """
        try:
            search_results = state.get("search_results", {})
            
            # Extract entities from question analysis
            entities = []
            question_analysis = state.get("question_analysis", {})
            if question_analysis:
                entities = question_analysis.get("entities", [])
            
            # Track model details for state
            model_details = []
            
            # If we don't have search results or they're empty, return early
            if not search_results or not isinstance(search_results, dict) or not any(search_results.values()):
                logger.warning("No search results available to gather model details")
                state["model_details"] = model_details
                return state
            
            # Process each entity's search results
            processed_models = set()  # Track models we've already processed
            
            # First, collect all relevant results across all entities
            all_results = []
            for entity, results in search_results.items():
                if results and isinstance(results, list):
                    for result in results:
                        # Skip duplicates
                        if not isinstance(result, dict):
                            continue
                            
                        model_name = result.get("model_name", "")
                        file_path = result.get("file_path", "")
                        
                        # Skip if we don't have both model name and file path
                        if not model_name or not file_path:
                            continue
                            
                        # Create unique key for this model
                        model_key = f"{model_name}:{file_path}"
                        
                        # Skip if we've already processed this model
                        if model_key in processed_models:
                            continue
                            
                        processed_models.add(model_key)
                        all_results.append(result)
            
            # Process full file paths first (they're more specific)
            full_path_results = [r for r in all_results if '/' in r.get("file_path", "")]
            
            # For each full path, try to get model details
            for result in full_path_results:
                model_name = result.get("model_name", "")
                file_path = result.get("file_path", "")
                content = result.get("content", "")
                
                # Extract just the filename without extension
                if '/' in file_path:
                    short_model_name = os.path.splitext(os.path.basename(file_path))[0]
                else:
                    short_model_name = model_name
                
                # If we don't have content, try to find it
                if not content or content.strip() == "":
                    try:
                        logger.info(f"Getting content for model {model_name} at {file_path}")
                        found, path, content = self._find_model_content(file_path)
                        
                        if found and content:
                            logger.info(f"Found content for {model_name} at {path}")
                            # Update the file path if we found it at a different path
                            file_path = path
                    except Exception as e:
                        logger.error(f"Error getting model content for {model_name}: {str(e)}")
                
                # If we still don't have content, try the model name
                if not content or content.strip() == "":
                    try:
                        logger.info(f"Trying to find content using model name: {short_model_name}")
                        found, path, content = self._find_model_content(short_model_name)
                        
                        if found and content:
                            logger.info(f"Found content for {short_model_name} at {path}")
                            # Update the file path if we found it at a different path
                            file_path = path
                    except Exception as e:
                        logger.error(f"Error getting model content using model name {short_model_name}: {str(e)}")
                
                # If we have content, extract details
                if content and content.strip() != "":
                    try:
                        # Get model type
                        model_type = self._extract_model_type(content)
                        
                        # Get columns and dependencies
                        columns = self._extract_model_columns(content)
                        
                        # Get more thorough dependencies
                        dependencies = self._get_model_dependencies(short_model_name, file_path)
                        
                        # Add model details to the list
                        model_detail = {
                            "model_name": short_model_name,
                            "file_path": file_path,
                            "content": content,
                            "model_type": model_type,
                            "columns": columns,
                            "dependencies": dependencies
                        }
                        
                        model_details.append(model_detail)
                        logger.info(f"Added details for model {short_model_name} ({model_type} with {len(columns)} columns, {len(dependencies.get('upstream', []))} upstreams, {len(dependencies.get('downstream', []))} downstreams)")
                    except Exception as e:
                        logger.error(f"Error extracting details for {short_model_name}: {str(e)}")
            
            # If we didn't get any model details from full paths, try model names
            if not model_details:
                model_name_results = [r for r in all_results if r not in full_path_results]
                
                for result in model_name_results:
                    model_name = result.get("model_name", "")
                    file_path = result.get("file_path", "")
                    content = result.get("content", "")
                    
                    # If we don't have content, try to find it
                    if not content or content.strip() == "":
                        try:
                            logger.info(f"Getting content for model {model_name}")
                            found, path, content = self._find_model_content(model_name)
                            
                            if found and content:
                                logger.info(f"Found content for {model_name} at {path}")
                                # Update the file path
                                file_path = path
                        except Exception as e:
                            logger.error(f"Error getting model content for {model_name}: {str(e)}")
                    
                    # If we have content, extract details
                    if content and content.strip() != "":
                        try:
                            # Get model type
                            model_type = self._extract_model_type(content)
                            
                            # Get columns and dependencies
                            columns = self._extract_model_columns(content)
                            
                            # Get dependencies
                            dependencies = self._get_model_dependencies(model_name, file_path)
                            
                            # Add model details to the list
                            model_detail = {
                                "model_name": model_name,
                                "file_path": file_path,
                                "content": content,
                                "model_type": model_type,
                                "columns": columns,
                                "dependencies": dependencies
                            }
                            
                            model_details.append(model_detail)
                            logger.info(f"Added details for model {model_name} ({model_type} with {len(columns)} columns, {len(dependencies.get('upstream', []))} upstreams, {len(dependencies.get('downstream', []))} downstreams)")
                        except Exception as e:
                            logger.error(f"Error extracting details for {model_name}: {str(e)}")
            
            # Final fallback for entities directly specified in the question
            if not model_details and entities:
                # Try directly looking for models mentioned in the question
                for entity in entities:
                    # Skip entities that look like file extensions or common terms
                    if entity in ['sql', 'yml', 'yaml', 'md', 'txt', 'csv', 'model', 'table', 'view']:
                        continue
                        
                    # Try to find the model
                    try:
                        logger.info(f"Trying to find model directly from entity: {entity}")
                        found, path, content = self._find_model_content(entity)
                        
                        if found and content:
                            logger.info(f"Found content for entity {entity} at {path}")
                            model_name = os.path.splitext(os.path.basename(path))[0]
                            
                            # Get model type
                            model_type = self._extract_model_type(content)
                            
                            # Get columns and dependencies
                            columns = self._extract_model_columns(content)
                            
                            # Get dependencies
                            dependencies = self._get_model_dependencies(model_name, path)
                            
                            # Add model details to the list
                            model_detail = {
                                "model_name": model_name,
                                "file_path": path,
                                "content": content,
                                "model_type": model_type,
                                "columns": columns,
                                "dependencies": dependencies
                            }
                            
                            model_details.append(model_detail)
                            logger.info(f"Added details for entity {entity} as model {model_name}")
                    except Exception as e:
                        logger.error(f"Error getting model details for entity {entity}: {str(e)}")
            
            # Special handling for enterprise file structures
            if not model_details and entities:
                for entity in entities:
                    # Try with known enterprise directory patterns
                    enterprise_patterns = [
                        f"models/marts/aggregates/{entity}.sql",
                        f"models/marts/consumption_metrics/{entity}.sql",
                        f"models/marts/core/{entity}.sql",
                        f"models/marts/intermediate/{entity}.sql",
                        f"models/marts/dimensions/{entity}.sql",
                        f"models/core_models/{entity}.sql",
                        f"models/**/{entity}.sql"  # Recursive glob pattern
                    ]
                    
                    for pattern in enterprise_patterns:
                        try:
                            # Use glob to find matching files
                            glob_pattern = os.path.join(self.dbt_tools.repo_path, pattern.replace('**/', '**/'))
                            matching_files = glob.glob(glob_pattern, recursive=True)
                            
                            if matching_files:
                                file_path = os.path.relpath(matching_files[0], self.dbt_tools.repo_path)
                                logger.info(f"Found enterprise pattern match for {entity} at {file_path}")
                                
                                # Try to get content
                                with open(matching_files[0], 'r') as f:
                                    content = f.read()
                                
                                if content:
                                    model_name = os.path.splitext(os.path.basename(file_path))[0]
                                    
                                    # Get model type
                                    model_type = self._extract_model_type(content)
                                    
                                    # Get columns and dependencies
                                    columns = self._extract_model_columns(content)
                                    
                                    # Get dependencies
                                    dependencies = self._get_model_dependencies(model_name, file_path)
                                    
                                    # Add model details to the list
                                    model_detail = {
                                        "model_name": model_name,
                                        "file_path": file_path,
                                        "content": content,
                                        "model_type": model_type,
                                        "columns": columns,
                                        "dependencies": dependencies
                                    }
                                    
                                    model_details.append(model_detail)
                                    logger.info(f"Added details for enterprise model {model_name}")
                                    break  # Found a match, no need to try other patterns
                        except Exception as e:
                            logger.error(f"Error trying enterprise pattern {pattern} for {entity}: {str(e)}")
            
            # Update state with model details
            state["model_details"] = model_details
            
            return state
        except Exception as e:
            logger.error(f"Error in _get_model_details: {str(e)}")
            state["model_details"] = []
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
        Get dependencies for a model, using its content or file path
        
        Enhanced for enterprise repositories with:
        - Deep nesting
        - Complex dependencies
        - Various reference patterns
        
        Args:
            model_name: Model name
            file_path: Path to the model file
            
        Returns:
            Dictionary with upstream and downstream dependencies
        """
        if not self.dbt_tools:
            return {"upstream": [], "downstream": []}
            
        logger.info(f"Getting dependencies for model: {model_name} at {file_path}")
        
        # Initialize dependency structure
        dependencies = {
            "upstream": [],
            "downstream": [],
            "sources": []
        }
        
        try:
            # Step 1: Try to get model content first
            found, model_path, content = self._find_model_content(file_path)
            
            if not found:
                found, model_path, content = self._find_model_content(model_name)
            
            if found and content:
                logger.info(f"Found model content for dependency analysis at: {model_path}")
                
                # Step 2: Extract refs from content using regex patterns
                # Look for ref('model_name') patterns
                ref_pattern = r"ref\(['\"]([^'\"]+)['\"]\)"
                refs = re.findall(ref_pattern, content)
                
                # Also look for ref('path', 'to', 'model') patterns used in some enterprise repos
                multi_ref_pattern = r"ref\((['\"][^'\"]+['\"](?:\s*,\s*['\"][^'\"]+['\"])+)\)"
                multi_refs = re.findall(multi_ref_pattern, content)
                
                for multi_ref in multi_refs:
                    # Extract individual parts
                    parts = re.findall(r"['\"]([^'\"]+)['\"]", multi_ref)
                    if parts:
                        refs.append('.'.join(parts))  # Join with dots for clarity
                
                # Step 3: Extract sources from content
                # Look for source('source_name', 'table_name') patterns
                source_pattern = r"source\(['\"]([^'\"]+)['\"][^)]*,[^)]*['\"]([^'\"]+)['\"]\)"
                sources = re.findall(source_pattern, content)
                
                # Format sources as "source_name.table_name"
                formatted_sources = [f"{source[0]}.{source[1]}" for source in sources]
                
                # Step 4: Extract direct SQL dependencies
                # Look for direct FROM and JOIN patterns that might be used in enterprise systems
                # This helps catch non-dbt style references
                from_pattern = r"FROM\s+([a-zA-Z0-9_\.]+)"
                join_pattern = r"JOIN\s+([a-zA-Z0-9_\.]+)"
                
                direct_deps = re.findall(from_pattern, content, re.IGNORECASE)
                direct_deps.extend(re.findall(join_pattern, content, re.IGNORECASE))
                
                # Filter out common SQL keywords that might match
                sql_keywords = ['dual', 'lateral', 'unnest', 'values', 'table']
                direct_deps = [dep for dep in direct_deps if dep.lower() not in sql_keywords]
                
                # Step 5: Get upstream and downstream dependencies using dbt_tools
                # Check if the methods exist before trying to call them
                try:
                    # Check if get_upstream_models method exists
                    if hasattr(self.dbt_tools, 'get_upstream_models'):
                        upstream_models = self.dbt_tools.get_upstream_models(model_name)
                        if upstream_models:
                            logger.info(f"Found {len(upstream_models)} upstream models for {model_name}")
                            for model in upstream_models:
                                if model not in dependencies["upstream"]:
                                    dependencies["upstream"].append(model)
                    else:
                        logger.warning("get_upstream_models method not available in dbt_tools")
                except Exception as e:
                    logger.error(f"Error getting upstream models: {str(e)}")
                
                try:
                    # Check if get_downstream_models method exists
                    if hasattr(self.dbt_tools, 'get_downstream_models'):
                        downstream_models = self.dbt_tools.get_downstream_models(model_name)
                        if downstream_models:
                            logger.info(f"Found {len(downstream_models)} downstream models for {model_name}")
                            for model in downstream_models:
                                if model not in dependencies["downstream"]:
                                    dependencies["downstream"].append(model)
                    else:
                        logger.warning("get_downstream_models method not available in dbt_tools")
                except Exception as e:
                    logger.error(f"Error getting downstream models: {str(e)}")
                
                # Step 6: Try to get relationships from file_scanner
                try:
                    if hasattr(self.dbt_tools, 'file_scanner') and hasattr(self.dbt_tools.file_scanner, 'get_model_relationships'):
                        relationships = self.dbt_tools.file_scanner.get_model_relationships()
                        if relationships and model_name in relationships:
                            model_relations = relationships[model_name]
                            
                            # Get upstream models
                            if 'upstream' in model_relations and not dependencies["upstream"]:
                                dependencies["upstream"] = model_relations['upstream']
                                
                            # Get downstream models
                            if 'downstream' in model_relations and not dependencies["downstream"]:
                                dependencies["downstream"] = model_relations['downstream']
                except Exception as e:
                    logger.error(f"Error getting model relationships: {str(e)}")
                
                # Step 7: If no upstream dependencies found yet, use extracted refs
                if not dependencies["upstream"] and refs:
                    logger.info(f"Using extracted refs as upstream dependencies: {refs}")
                    dependencies["upstream"] = refs
                
                # Add sources to dependencies
                if formatted_sources:
                    logger.info(f"Found {len(formatted_sources)} sources: {formatted_sources}")
                    dependencies["sources"] = formatted_sources
                
                # Step 8: Add direct SQL dependencies if no other dependencies were found
                if not dependencies["upstream"] and direct_deps:
                    logger.info(f"Using extracted direct SQL dependencies: {direct_deps}")
                    dependencies["upstream"].extend(direct_deps)
                
                # Step 9: Try to get lineage information as a fallback
                if not dependencies["upstream"] and not dependencies["downstream"]:
                    try:
                        if hasattr(self.dbt_tools, 'get_model_lineage'):
                            lineage = self.dbt_tools.get_model_lineage(model_name)
                            if lineage:
                                logger.info(f"Using model lineage for dependencies")
                                # Extract upstream and downstream from lineage
                                if "upstream" in lineage and lineage["upstream"]:
                                    dependencies["upstream"] = lineage["upstream"]
                                if "downstream" in lineage and lineage["downstream"]:
                                    dependencies["downstream"] = lineage["downstream"]
                    except Exception as e:
                        logger.error(f"Error getting model lineage: {str(e)}")
            else:
                logger.warning(f"Could not find model content for dependency analysis: {model_name}")
                
                # Fallback: Try to get dependencies from the file path pattern
                if '/' in file_path:
                    # For paths like models/marts/core/some_model.sql, assume
                    # it might depend on models/staging/stg_some_*.sql
                    path_parts = file_path.split('/')
                    if len(path_parts) >= 3:
                        base_name = os.path.splitext(path_parts[-1])[0]
                        if base_name.startswith('agg_') or 'mart' in path_parts[-2]:
                            # For aggregation models, they typically depend on staging models
                            logger.info(f"Inferring dependencies from file path pattern for {base_name}")
                            base_name_without_prefix = re.sub(r'^(agg_|fct_|dim_)', '', base_name)
                            dependencies["upstream"].append(f"stg_{base_name_without_prefix}")
                            # For specific enterprise patterns
                            if 'aggregates' in path_parts:
                                # Look for source staging models
                                staging_models = self.dbt_tools.search_model(f"stg_{base_name_without_prefix}")
                                if staging_models:
                                    logger.info(f"Found potential staging dependencies for {base_name}")
                                    for model in staging_models:
                                        model_name = model.model_name if hasattr(model, 'model_name') else ""
                                        if model_name and model_name not in dependencies["upstream"]:
                                            dependencies["upstream"].append(model_name)
        except Exception as e:
            logger.error(f"Error analyzing dependencies: {str(e)}")
            
        return dependencies

    def _validate_code_enhancement_response(self, response: str, state: Dict[str, Any]) -> str:
        """Validate and format code enhancement responses."""
        logger.info("Validating code enhancement response")
        
        # Check if we have any model content
        model_details = state.get("model_details", [])
        model_content = ""
        model_path = ""
        
        if isinstance(model_details, list) and model_details:
            model_content = model_details[0].get("content", "")
            model_path = model_details[0].get("file_path", "")
        elif isinstance(model_details, dict):
            model_content = model_details.get("content", "")
            model_path = model_details.get("file_path", "")
            
        # Add warning if no actual model content was found
        if not model_content:
            warning_message = "\n\n**WARNING: No actual model content was found in the repository. This response is based on generic understanding only and may not be accurate for your specific implementation.**\n\n"
            # Add warning at the beginning of the response to make it prominent
            response = warning_message + response
            logger.warning("No actual model content found for code enhancement response")
            
        # Ensure the model path is in the response
        if model_path and model_path not in response:
            logger.warning(f"Model path {model_path} not found in code enhancement response")
            header = f"## Enhanced Code for {model_path}\n\n"
            response = header + response
            
        # Validate DBT-specific statements
        dbt_patterns = [r"ref\('[^']+'\)", r"source\('[^']+',\s*'[^']+'\)"]
        has_dbt_patterns = any(re.search(pattern, response) for pattern in dbt_patterns)
        
        if not has_dbt_patterns and "```sql" in response:
            logger.warning("Code enhancement response does not contain DBT ref/source patterns")
            note = "\n\n**Note:** Ensure you're using proper DBT ref() and source() functions for referencing other models and sources."
            response += note
            
        # Check for complete code implementation
        if "```sql" in response and "```" in response:
            sql_blocks = re.findall(r"```sql\n(.*?)\n```", response, re.DOTALL)
            if not sql_blocks:
                sql_blocks = re.findall(r"```\n(.*?)\n```", response, re.DOTALL)
                
            if sql_blocks:
                main_sql = sql_blocks[0]
                # Check if the SQL has the basic structure we need
                has_select = "SELECT" in main_sql.upper()
                has_from = "FROM" in main_sql.upper()
                
                if not (has_select and has_from):
                    logger.warning("Code enhancement missing basic SQL structure")
                    note = "\n\n**Note:** The SQL implementation should include complete SELECT and FROM clauses."
                    response += note
        else:
            logger.warning("Code enhancement response missing code blocks")
            note = "\n\n**Note:** The response should include a complete SQL implementation in code blocks."
            response += note
            
        # Check for required sections in the response
        required_sections = ["Implementation", "Explanation", "Changes Made"]
        missing_sections = []
        
        for section in required_sections:
            if not re.search(section, response, re.IGNORECASE):
                missing_sections.append(section)
                
        if missing_sections:
            logger.warning(f"Code enhancement response missing sections: {', '.join(missing_sections)}")
            note = f"\n\n**Note:** This response should include these sections: {', '.join(missing_sections)}"
            response += note
            
        return response

    def _validate_documentation_response(self, response: str, question_type: str) -> str:
        """Validate and format documentation or model info responses."""
        logger.info(f"Validating {question_type} response")
        
        # Add warning if no actual model was found in the search results
        if "No model information found for any search method" in response:
            warning_message = "\n\n**WARNING: No actual model content was found in the repository. This response is based on generic understanding only and may not be accurate for your specific implementation.**\n\n"
            # Add warning at the beginning of the response to make it prominent
            response = warning_message + response
            logger.warning(f"Adding warning to {question_type} response: No model content found")
        
        # Check if the response contains code snippets
        has_code_snippets = "```" in response or "```sql" in response
        
        if not has_code_snippets:
            logger.warning(f"{question_type} response does not contain code snippets")
            note = "\n\n**Note:** This response would be improved by including relevant code snippets from the model."
            response += note
            
        # Check if the response is structured with proper sections
        if question_type == "MODEL_INFO":
            expected_sections = ["Model Overview", "Technical Implementation", "Data Relationships", "Key Metrics"]
        else:  # DOCUMENTATION
            expected_sections = ["Technical Overview", "Implementation Details", "Schema Documentation", "Technical Relationships"]
            
        missing_sections = []
        for section in expected_sections:
            if not re.search(section, response, re.IGNORECASE):
                missing_sections.append(section)
                
        if missing_sections:
            logger.warning(f"{question_type} response is missing sections: {', '.join(missing_sections)}")
            note = f"\n\n**Note:** This response should include these sections: {', '.join(missing_sections)}"
            response += note
            
        return response
            
    def _find_model_content(self, path_or_model: str) -> Tuple[bool, str, str]:
        """
        Attempt to find a model's content by path or name using multiple methods.
        
        Improved to handle complex enterprise repository structures with:
        - Deep nesting
        - Many files (1000+)
        - Varied naming conventions
        - Custom directory structures
        
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
            
        # Extract base name if it's a path
        original_path = path_or_model
        if '/' in model_name:
            model_name = os.path.basename(model_name)
        
        # Step 1: Try direct file access first (fastest for known paths)
        try:
            logger.info(f"Trying direct file access for: {original_path}")
            content = self.dbt_tools.get_file_content(original_path)
            
            if content:
                logger.info(f"Found model content via direct file access: {original_path}")
                return True, original_path, content
        except Exception as e:
            logger.error(f"Error in direct file access: {str(e)}")
            
        # Step 1.5: For specific path patterns, try direct variations
        if "marts/aggregates" in original_path or "marts/core" in original_path:
            parts = original_path.split('/')
            if len(parts) >= 3:
                # Get just the model name from the path
                filename = parts[-1]
                
                # Try both marts/aggregates and marts/core
                specific_paths = [
                    f"models/marts/aggregates/{filename}",
                    f"models/marts/core/{filename}",
                    f"models/marts/intermediate/{filename}"
                ]
                
                for specific_path in specific_paths:
                    try:
                        logger.info(f"Trying specific path: {specific_path}")
                        content = self.dbt_tools.get_file_content(specific_path)
                        
                        if content:
                            logger.info(f"Found model content via specific path: {specific_path}")
                            return True, specific_path, content
                    except Exception as e:
                        logger.error(f"Error in specific path access: {str(e)}")
        
        # Step 2: Try resolving the model from path (enhanced for enterprise repos)
        try:
            logger.info(f"Trying to resolve model from path: {original_path}")
            model_results = self.dbt_tools.resolve_model_from_path(original_path)
            
            if model_results and len(model_results) > 0:
                result = model_results[0]  # Take the first result
                result_dict = self._convert_search_result_to_dict(result)
                
                file_path = result_dict.get("file_path", "")
                content = result_dict.get("content", "")
                
                if file_path and content:
                    logger.info(f"Found model content via path resolution at: {file_path}")
                    return True, file_path, content
        except Exception as e:
            logger.error(f"Error in model path resolution: {str(e)}")
            
        # Step 3: Try direct DBT model search
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
        
        # Step 4: Try advanced file path search with globbing (better for complex repos)
        try:
            logger.info(f"Trying glob file search for: {model_name}")
            # Create glob patterns especially for enterprise repos
            glob_patterns = [
                f"**/{model_name}.sql",
                f"**/models/**/{model_name}.sql",
                f"**/marts/**/{model_name}.sql",
                f"**/marts/aggregates/**/{model_name}.sql",  # Added specific path for aggregates
                f"**/consumption_metrics/**/{model_name}.sql",
                f"**/*{model_name}*.sql"  # More permissive pattern as last resort
            ]
            
            # Add additional patterns if this looks like a path
            if '/' in original_path:
                path_parts = original_path.strip('/').split('/')
                if len(path_parts) > 1:
                    # Try with the last 2-3 parts of the path
                    if len(path_parts) >= 3:
                        # Try with wildcards between directories for flexible matching
                        glob_patterns.append(f"**/{path_parts[-3]}/**/{path_parts[-2]}/**/{path_parts[-1]}")
                    glob_patterns.append(f"**/{path_parts[-2]}/**/{path_parts[-1]}")
            
            # Try each pattern
            for pattern in glob_patterns:
                try:
                    pattern_with_extension = pattern if pattern.endswith('.sql') else f"{pattern}.sql"
                    glob_path = os.path.join(self.dbt_tools.repo_path, pattern_with_extension)
                    matching_files = glob.glob(glob_path, recursive=True)
                    
                    if matching_files:
                        # Take the first match
                        file_path = os.path.relpath(matching_files[0], self.dbt_tools.repo_path)
                        with open(matching_files[0], 'r') as f:
                            content = f.read()
                        
                        logger.info(f"Found model content via glob pattern {pattern} at: {file_path}")
                        return True, file_path, content
                except Exception as e:
                    logger.error(f"Error in glob pattern search with {pattern}: {str(e)}")
        except Exception as e:
            logger.error(f"Error in glob file search: {str(e)}")
        
        # Step 5: Try standard DBT directory paths as fallback
        try:
            logger.info(f"Trying standard DBT directory paths for: {model_name}")
            variations = [
                f"models/{model_name}.sql",
                f"models/marts/core/{model_name}.sql",
                f"models/marts/{model_name}.sql",
                f"models/marts/aggregates/{model_name}.sql",  # Added specific path for aggregates
                f"models/staging/{model_name}.sql",
                f"models/intermediate/{model_name}.sql",
                # Add enterprise-specific paths
                f"models/marts/consumption_metrics/{model_name}.sql",
                f"models/consumption_metrics/{model_name}.sql",
                f"models/core_models/{model_name}.sql",
                f"models/marts/advanced/{model_name}.sql",
                f"models/marts/forecast/{model_name}.sql",
                f"core_models/{model_name}.sql"
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
            
        # Step 6: Last resort - try keyword search
        try:
            logger.info(f"Trying keyword search for: {model_name}")
            search_results = self._search_by_keyword(model_name)
            
            if search_results and len(search_results) > 0:
                result = search_results[0]  # Take the first result
                
                file_path = result.get("file_path", "")
                content = result.get("content", "")
                
                if file_path and content:
                    logger.info(f"Found model content via keyword search at: {file_path}")
                    return True, file_path, content
        except Exception as e:
            logger.error(f"Error in keyword search: {str(e)}")
        
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

    def _format_dependencies_for_prompt(self, state, model_details, formatted_results):
        """Format dependencies information for the prompt."""
        # Get all dependencies from model details
        all_dependencies = {}
        
        for model in model_details:
            model_name = model.get("model_name", "")
            file_path = model.get("file_path", "")
            dependencies = model.get("dependencies", {})
            content = model.get("content", "")
            
            if model_name and dependencies:
                # Add dependency information to the formatted results
                dependency_info = []
                dependency_info.append(f"\n## Dependencies for {model_name}")
                dependency_info.append(f"### File Path: {file_path}")
                
                # Add a code snippet from the content
                if content:
                    content_preview = content[:500] + ("..." if len(content) > 500 else "")
                    dependency_info.append(f"\n### Model Content (Preview):\n```sql\n{content_preview}\n```")
                
                # Add upstream dependencies
                upstream = dependencies.get("upstream", [])
                if upstream:
                    dependency_info.append("\n### Upstream Dependencies:")
                    for dep in upstream:
                        dependency_info.append(f"- {dep}")
                else:
                    dependency_info.append("\n### Upstream Dependencies: None found")
                
                # Add downstream dependencies
                downstream = dependencies.get("downstream", [])
                if downstream:
                    dependency_info.append("\n### Downstream Dependencies:")
                    for dep in downstream:
                        dependency_info.append(f"- {dep}")
                else:
                    dependency_info.append("\n### Downstream Dependencies: None found")
                
                # Add sources
                sources = dependencies.get("sources", [])
                if sources:
                    dependency_info.append("\n### Source Dependencies:")
                    for source in sources:
                        dependency_info.append(f"- {source}")
                
                # Add visualization for dependencies
                if upstream or downstream or sources:
                    dependency_info.append("\n### Dependency Visualization:")
                    
                    # Create a simple text-based visualization
                    viz = [f"```\n{model_name}"]
                    if upstream:
                        for up in upstream:
                            viz.append(f" ├── {up}")
                    if sources:
                        for source in sources:
                            viz.append(f" │  └── {source}")
                    if downstream:
                        for down in downstream:
                            viz.append(f" └── {down}")
                    viz.append("```")
                    
                    dependency_info.append("\n".join(viz))
                
                # Add model-specific dependency info
                all_dependencies[model_name] = "\n".join(dependency_info)
        
        # Add dependency information to the formatted results
        if all_dependencies:
            dependency_section = "\n\n# MODEL DEPENDENCIES\n" + "\n\n".join(all_dependencies.values())
            formatted_results = formatted_results + dependency_section
        
        return formatted_results

    def _format_lineage_for_prompt(self, state, model_details, formatted_results):
        """Format lineage information for the prompt."""
        # Similar structure to dependencies but with different formatting
        lineage_section = "\n\n# MODEL LINEAGE\n"
        
        for model in model_details:
            model_name = model.get("model_name", "")
            file_path = model.get("file_path", "")
            dependencies = model.get("dependencies", {})
            content = model.get("content", "")
            
            if model_name:
                lineage_section += f"\n## Lineage for {model_name}"
                lineage_section += f"\n### File Path: {file_path}"
                
                # Add code preview
                if content:
                    content_preview = content[:500] + ("..." if len(content) > 500 else "")
                    lineage_section += f"\n\n### Model Content (Preview):\n```sql\n{content_preview}\n```"
                
                # Add upstream dependencies
                upstream = dependencies.get("upstream", []) if dependencies else []
                if upstream:
                    lineage_section += "\n\n### Upstream Models:"
                    for up in upstream:
                        lineage_section += f"\n- {up}"
                
                # Add downstream dependencies
                downstream = dependencies.get("downstream", []) if dependencies else []
                if downstream:
                    lineage_section += "\n\n### Downstream Models:"
                    for down in downstream:
                        lineage_section += f"\n- {down}"
                
                # Add lineage visualization
                if upstream or downstream:
                    lineage_section += "\n\n### Lineage Visualization:"
                    lineage_viz = [f"```"]
                    
                    # Add upstream flow
                    if upstream:
                        upstream_flow = " → ".join(upstream) + f" → {model_name}"
                        lineage_viz.append(f"Upstream Flow:\n{upstream_flow}")
                    
                    # Add model details
                    lineage_viz.append(f"\nDetailed Model:\n└── {model_name}")
                    
                    # Add downstream flow
                    if downstream:
                        lineage_viz.append(f"\nDownstream Flow:")
                        for down in downstream:
                            lineage_viz.append(f"└── {down}")
                    
                    lineage_viz.append("```")
                    lineage_section += "\n" + "\n".join(lineage_viz)
        
        # Add lineage section to formatted results
        formatted_results += lineage_section
        
        return formatted_results

    def _format_code_enhancement_for_prompt(self, state, model_details, formatted_results):
        """Format code enhancement information for the prompt."""
        # Add the current code to the formatted results
        code_section = "\n\n# CURRENT CODE\n"
        
        for model in model_details:
            model_name = model.get("model_name", "")
            file_path = model.get("file_path", "")
            content = model.get("content", "")
            
            if model_name and content:
                code_section += f"## Model: {model_name}\n## Path: {file_path}\n\n```sql\n{content}\n```\n"
                
                # Add additional context about the model's structure
                code_section += "\n## Model Structure Analysis:\n"
                
                # Analyze CTEs
                cte_matches = re.findall(r'with\s+(.*?)\s+as\s+\((.*?)\)', content, re.IGNORECASE | re.DOTALL)
                if cte_matches:
                    code_section += "### Common Table Expressions (CTEs):\n"
                    for cte_name, _ in cte_matches:
                        code_section += f"- `{cte_name.strip()}`: Used for intermediate calculations\n"
                
                # Analyze joins
                join_matches = re.findall(r'(inner|left|right|full|cross)?\s*join\s+(.*?)\s+on\s+(.*?)(where|\)|group\s+by|order\s+by|limit|$)', 
                                         content, re.IGNORECASE | re.DOTALL)
                if join_matches:
                    code_section += "\n### Key Joins:\n"
                    for join_type, table, condition, _ in join_matches:
                        join_type = join_type.strip() if join_type else "inner"
                        code_section += f"- {join_type.upper()} JOIN with {table.strip()} on {condition.strip()}\n"
        
        # Add the code section to formatted results
        formatted_results += code_section
        
        return formatted_results

    def _format_development_for_prompt(self, state, model_details, formatted_results):
        """Format development information for the prompt."""
        # Add related models and their structure for context
        dev_section = "\n\n# DEVELOPMENT CONTEXT\n"
        
        # Add examples of similar models
        for model in model_details:
            model_name = model.get("model_name", "")
            file_path = model.get("file_path", "")
            content = model.get("content", "")
            model_type = model.get("model_type", "")
            
            if model_name and content:
                dev_section += f"## Reference Model: {model_name}\n"
                dev_section += f"### File Path: {file_path}\n"
                dev_section += f"### Model Type: {model_type}\n\n"
                dev_section += f"```sql\n{content}\n```\n\n"
        
        # Add development guidelines
        dev_section += "## Development Guidelines:\n"
        dev_section += "- Follow the structure of similar models\n"
        dev_section += "- Use proper DBT references with ref() and source() functions\n"
        dev_section += "- Include appropriate documentation in model files\n"
        dev_section += "- Use CTEs for readability and maintainability\n"
        
        # Add the development section to formatted results
        formatted_results += dev_section
        
    def _format_model_info_for_prompt(self, state, model_details, formatted_results):
        """Format model information for the prompt."""
        # Create a comprehensive model info section
        model_info_section = "\n\n# MODEL INFORMATION\n"
        
        for model in model_details:
            model_name = model.get("model_name", "")
            file_path = model.get("file_path", "")
            content = model.get("content", "")
            model_type = model.get("model_type", "")
            
            if model_name:
                model_info_section += f"## Model: {model_name}\n"
                model_info_section += f"### File Path: {file_path}\n"
                model_info_section += f"### Model Type: {model_type}\n\n"
                
                # Add code preview
                if content:
                    model_info_section += f"### SQL Definition:\n```sql\n{content}\n```\n\n"
                
                # Extract and display model columns
                columns = self._extract_model_columns(content) if content else []
                if columns:
                    model_info_section += "### Key Columns:\n"
                    for column in columns:
                        col_name = column.get("name", "")
                        col_type = column.get("data_type", "")
                        col_desc = column.get("description", "")
                        
                        if col_name:
                            model_info_section += f"- **{col_name}** ({col_type}): {col_desc}\n"
        
        # Add the model info section to formatted results
        formatted_results += model_info_section
        
        return formatted_results

    def _create_lineage_visualization(self, model_details):
        """Create lineage visualization data for frontend rendering.
        
        Generates a complete graph data structure that can be used by the frontend
        LineageGraph component to render an interactive visualization.
        """
        # Initialize the lineage data structure with models and edges arrays
        lineage_data = {
            "models": [],
            "edges": [],
            "columns": []  # Support for column-level lineage
        }
        
        processed_models = set()  # Track already processed models to avoid duplicates
        column_id_counter = 1  # For generating unique column IDs
        
        # Helper function to determine model type
        def determine_model_type(file_path):
            if not file_path:
                return "unknown"
            elif "/staging/" in file_path or "/stg_" in file_path:
                return "staging"
            elif "/intermediate/" in file_path or "/int_" in file_path:
                return "intermediate" 
            elif "/mart/" in file_path or "/core/" in file_path:
                return "mart"
            elif "source:" in file_path:
                return "source"
            else:
                return "model"
        
        # Process each model in the details
        for model in model_details:
            model_name = model.get("model_name", "")
            file_path = model.get("file_path", "")
            model_type = model.get("model_type", determine_model_type(file_path))
            dependencies = model.get("dependencies", {})
            content = model.get("content", "")
            
            # Skip if no model name
            if not model_name:
                continue
                
            # Add the central model if not already processed
            if model_name not in processed_models:
                processed_models.add(model_name)
                
                # Add the model to the lineage data
                lineage_data["models"].append({
                    "id": model_name,
                    "name": model_name,
                    "path": file_path,
                    "type": model_type,
                    "highlight": True  # This is our focal model
                })
                
                # Extract columns if we have content
                if content:
                    try:
                        columns = self._extract_model_columns(content)
                        for col in columns:
                            col_name = col.get("name", "")
                            col_type = "regular"
                            
                            # Determine column type (primary key, calculated, etc.)
                            if "key" in col_name.lower() or "id" in col_name.lower():
                                col_type = "primary_key"
                            elif "sum(" in col.get("expression", "").lower() or "avg(" in col.get("expression", "").lower():
                                col_type = "calculated"
                                
                            # Add column to the visualization
                            lineage_data["columns"].append({
                                "id": f"col{column_id_counter}",
                                "modelId": model_name,
                                "name": col_name,
                                "type": col_type
                            })
                            column_id_counter += 1
                    except Exception as e:
                        logging.warning(f"Error extracting columns for {model_name}: {str(e)}")
                    
            # Process upstream dependencies
            upstream = dependencies.get("upstream", []) if dependencies else []
            for up in upstream:
                # Skip if empty
                if not up:
                    continue
                    
                # Add upstream model if not already processed
                if up not in processed_models:
                    processed_models.add(up)
                    
                    # Find or infer the model type and file path
                    up_type = "upstream"
                    up_path = ""
                    
                    # Look for this model in other model details
                    for other_model in model_details:
                        if other_model.get("model_name", "") == up:
                            up_path = other_model.get("file_path", "")
                            up_type = other_model.get("model_type", determine_model_type(up_path))
                            break
                    
                    lineage_data["models"].append({
                        "id": up,
                        "name": up,
                        "path": up_path,
                        "type": up_type,
                        "highlight": False
                    })
                
                # Add the edge
                lineage_data["edges"].append({
                    "source": up,
                    "target": model_name
                })
            
            # Process downstream dependencies
            downstream = dependencies.get("downstream", []) if dependencies else []
            for down in downstream:
                # Skip if empty
                if not down:
                    continue
                    
                # Add downstream model if not already processed
                if down not in processed_models:
                    processed_models.add(down)
                    
                    # Find or infer the model type and file path
                    down_type = "downstream"
                    down_path = ""
                    
                    # Look for this model in other model details
                    for other_model in model_details:
                        if other_model.get("model_name", "") == down:
                            down_path = other_model.get("file_path", "")
                            down_type = other_model.get("model_type", determine_model_type(down_path))
                            break
                            
                    lineage_data["models"].append({
                        "id": down,
                        "name": down,
                        "path": down_path,
                        "type": down_type,
                        "highlight": False
                    })
                
                # Add the edge
                lineage_data["edges"].append({
                    "source": model_name,
                    "target": down
                })
            
            # Process source dependencies if available
            sources = dependencies.get("sources", []) if dependencies else []
            for source in sources:
                # Skip if empty
                if not source:
                    continue
                    
                source_id = f"source:{source}"
                
                # Add source if not already processed
                if source_id not in processed_models:
                    processed_models.add(source_id)
                    
                    lineage_data["models"].append({
                        "id": source_id,
                        "name": source,
                        "path": f"source:{source}",
                        "type": "source",
                        "highlight": False
                    })
                    
                    # Add the edge
                    lineage_data["edges"].append({
                        "source": source_id,
                        "target": model_name
                    })
        
        return lineage_data

    def _validate_dependency_response(self, response, model_details):
        """Validate and enhance dependency response if needed."""
        # Check if the response already includes a visualization
        if "```" not in response:
            # Find the first model with dependencies
            for model in model_details:
                model_name = model.get("model_name", "")
                dependencies = model.get("dependencies", {})
                
                if model_name and dependencies:
                    upstream = dependencies.get("upstream", [])
                    downstream = dependencies.get("downstream", [])
                    sources = dependencies.get("sources", [])
                    
                    if upstream or downstream or sources:
                        # Add a simple text-based visualization at the beginning
                        viz = [f"```\n{model_name}"]
                        if upstream:
                            for up in upstream:
                                viz.append(f" ├── {up}")
                        if sources:
                            for source in sources:
                                viz.append(f" │  └── {source}")
                        if downstream:
                            for down in downstream:
                                viz.append(f" └── {down}")
                        viz.append("```")
                        
                        # Add to beginning of response
                        response = "\n".join(viz) + "\n\n" + response
                        break
        
        return response

    def _validate_lineage_response(self, response, model_details):
        """Validate and enhance lineage response if needed."""
        # Check if the response already includes a visualization
        if "```" not in response or "Upstream Flow" not in response:
            # Find the first model with dependencies
            for model in model_details:
                model_name = model.get("model_name", "")
                dependencies = model.get("dependencies", {})
                
                if model_name and dependencies:
                    upstream = dependencies.get("upstream", [])
                    downstream = dependencies.get("downstream", [])
                    
                    if upstream or downstream:
                        # Create a lineage visualization at the beginning
                        viz = ["```"]
                        
                        # Add upstream flow if present
                        if upstream:
                            upstream_flow = " → ".join(upstream) + f" → {model_name}"
                            viz.append(f"Upstream Flow:\n{upstream_flow}")
                        
                        # Add model details
                        viz.append(f"\nDetailed Model:\n└── {model_name}")
                        
                        # Add downstream flow if present
                        if downstream:
                            viz.append(f"\nDownstream Flow:")
                            for down in downstream:
                                viz.append(f"└── {down}")
                        
                        viz.append("```")
                        
                        # Add to beginning of response
                        response = "\n".join(viz) + "\n\n" + response
                        break
        
        return response

    def _validate_development_response(self, response):
        """Validate and enhance development response if needed."""
        # Ensure response has code blocks
        if "```sql" not in response and "```" not in response:
            return response + "\n\n```sql\n-- Complete SQL implementation should be here\n-- Please refer to the guidance above for implementation details\n```"
        
        return response

    def _format_results_for_prompt(self, state: Dict[str, Any]) -> str:
        """Format search results for the prompt."""
        results = []
        
        # Get data from state
        search_results = state.get("search_results", {})
        model_details = state.get("model_details", [])
        column_details = state.get("column_details", {})
        related_models = state.get("related_models", {})
        content_search = state.get("content_search", [])
        
        # Format model details
        if model_details:
            if isinstance(model_details, list):
                for model in model_details:
                    model_name = model.get("model_name", "")
                    file_path = model.get("file_path", "")
                    content = model.get("content", "")
                    
                    if content:
                        results.append(f"## Model: {model_name}\n### Path: {file_path}\n```sql\n{content}\n```\n")
            else:
                model_name = model_details.get("model_name", "")
                file_path = model_details.get("file_path", "")
                content = model_details.get("content", "")
                
                if content:
                    results.append(f"## Model: {model_name}\n### Path: {file_path}\n```sql\n{content}\n```\n")
        
        # Format search results if not already covered by model details
        if search_results and not model_details:
            # Check if search_results is a dict
            if isinstance(search_results, dict):
                for entity, results_list in search_results.items():
                    if results_list and isinstance(results_list, list):
                        for result in results_list:
                            if isinstance(result, dict):
                                model_name = result.get("model_name", "")
                                file_path = result.get("file_path", "")
                                content = result.get("content", "")
                                match_type = result.get("match_type", "")
                                
                                if content:
                                    results.append(f"## Search Result: {model_name}\n### Path: {file_path}\n### Match Type: {match_type}\n```sql\n{content}\n```\n")
        
        # Format column details
        if column_details:
            results.append("## Column Details\n")
            for column, details in column_details.items():
                model_name = details.get("model_name", "")
                calculation = details.get("calculation", "")
                
                if calculation:
                    results.append(f"### Column: {column}\n**Model:** {model_name}\n**Calculation:**\n```sql\n{calculation}\n```\n")
        
        # Format related models
        if related_models:
            results.append("## Related Models\n")
            for model_type, models in related_models.items():
                if models:
                    results.append(f"### {model_type.title()} Models\n")
                    for model in models:
                        results.append(f"- {model}\n")
        
        # Format content search results
        if content_search:
            results.append("## Content Search Results\n")
            for result in content_search:
                model_name = result.get("model_name", "")
                file_path = result.get("file_path", "")
                match_contexts = result.get("match_contexts", [])
                
                if match_contexts:
                    results.append(f"### Model: {model_name}\n**Path:** {file_path}\n**Matches:**\n")
                    for context in match_contexts:
                        results.append(f"```\n{context}\n```\n")
        
        return "\n".join(results)

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
                import time
                time.sleep(1)

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