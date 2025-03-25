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
                # Validate GitHub URL format
                if not repo_url.startswith(('https://github.com/', 'git@github.com:')):
                    raise ValueError("Invalid GitHub repository URL format")
                
                # Check if required credentials are provided for private repos
                if repo_url.startswith('https://github.com/') and not (username and token):
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
        """Search for DBT models based on model names."""
        try:
            analysis = state["question_analysis"]
            entities = analysis.get("entities", [])
            
            results = {}
            
            # Only proceed with DBT search if we have DBT tools initialized
            if self.dbt_tools:
                # Search for each model entity
                for entity in entities:
                    # Ignore entities with dots (column references)
                    if '.' in entity:
                        continue
                        
                    # Search for the model
                    model_results = self.dbt_tools.search_model(entity)
                    if model_results:  # Check if the list has any results
                        logger.info(f"Found {len(model_results)} results for model '{entity}'")
                        # Convert SearchResult objects to dictionaries
                        results[entity] = []
                        for result in model_results:
                            # Check if result is already a dict
                            if isinstance(result, dict):
                                results[entity].append(result)
                            else:
                                # Convert dataclass or object to dict
                                try:
                                    # Try dataclass __dict__ first
                                    results[entity].append(result.__dict__)
                                except AttributeError:
                                    # Fall back to vars() for other objects
                                    results[entity].append(vars(result))
                    else:
                        logger.info(f"No results found for model '{entity}'")
            else:
                results = self._get_dbt_tools_error()
                
            # Update state
            state["dbt_results"] = results
            return state
            
        except Exception as e:
            logger.error(f"Error searching models: {str(e)}")
            state["dbt_results"] = {
                "error": str(e),
                "status": "error",
                "message": f"Error searching DBT models: {str(e)}"
            }
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
    
    def _get_related_model_info(self, state: Dict[str, Any], results: Dict[str, Any]) -> None:
        """Get information about related models from column search results."""
        if not self.dbt_tools:
            return
        
        models_to_process = set()
        
        # Collect all model names from the results
        for column_name, column_results in results.items():
            if isinstance(column_results, list):
                for result in column_results:
                    if isinstance(result, dict) and "model_name" in result and result["model_name"]:
                        models_to_process.add(result["model_name"])
        
        # Get details for each model
        model_details = {}
        for model_name in models_to_process:
            model_results = self.dbt_tools.search_model(model_name)
            if model_results:
                # Convert first result to dict
                model_info = model_results[0].__dict__
                
                # Also get dependency information
                dependencies = self.dbt_tools.find_related_models(model_name)
                model_info["dependencies"] = dependencies
                
                model_details[model_name] = {"model": model_info}
        
        # Update state
        state["related_models"] = model_details

    def _get_model_details(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Get detailed model information including dependencies."""
        try:
            analysis = state["question_analysis"]
            entities = analysis.get("entities", [])
            question_type = analysis.get("question_type", "GENERAL")
            
            results = {}
            related_models = {}
            
            # Only proceed with DBT search if we have DBT tools initialized
            if self.dbt_tools:
                # Identify file paths and model names in entities
                file_paths = [entity for entity in entities if "/" in entity]
                model_names = [entity for entity in entities if not "/" in entity and not "." in entity]
                
                # For code enhancement requests, prioritize file paths
                if question_type == "CODE_ENHANCEMENT" and file_paths:
                    logger.info(f"Processing CODE_ENHANCEMENT request for file paths: {file_paths}")
                    
                    for file_path in file_paths:
                        # Extract model name from path - it's usually the last part without .sql
                        path_parts = file_path.split('/')
                        model_name = path_parts[-1].replace('.sql', '') if path_parts else file_path
                        
                        # Search for the model
                        model_results = self.dbt_tools.search_model(model_name)
                        
                        # If no results by model name, try searching by path
                        if not model_results:
                            logger.info(f"No results for '{model_name}', searching by path: {file_path}")
                            path_results = self.dbt_tools.search_file_path(file_path)
                            
                            if path_results:
                                # Get the first result's model name
                                first_result = path_results[0]
                                if hasattr(first_result, 'model_name') and first_result.model_name:
                                    model_name = first_result.model_name
                                    # Now search by this model name
                                    model_results = self.dbt_tools.search_model(model_name)
                        
                        if model_results:
                            logger.info(f"Found model details for '{model_name}' from path '{file_path}'")
                            model_info = model_results[0].__dict__  # Use the first result
                            
                            # Ensure we have the full content for code enhancement
                            if 'content' not in model_info or not model_info['content']:
                                if hasattr(model_results[0], 'file_path') and model_results[0].file_path:
                                    content = self.dbt_tools.get_file_content(model_results[0].file_path)
                                    if content:
                                        model_info['content'] = content
                                        logger.info(f"Added full content for model '{model_name}'")
                            
                            # Add dependency information
                            dependencies = self.dbt_tools.find_related_models(model_name)
                            model_info["dependencies"] = dependencies
                            
                            # Use the file path as the key to maintain the user's reference
                            results[file_path] = model_info
                            
                            # Also add related models
                            related_models[model_name] = dependencies
                        else:
                            logger.warning(f"Could not find model for path: {file_path}")
                            # Still create an entry to maintain structure
                            results[file_path] = {
                                "file_path": file_path,
                                "model_name": model_name,
                                "status": "not_found",
                                "message": f"Could not find model for path: {file_path}"
                            }
                
                # Process any remaining model names (or all models if no file paths)
                remaining_models = model_names if file_paths else entities
                for entity in remaining_models:
                    # Skip entities already processed or with dots (column references)
                    if entity in results or '.' in entity:
                        continue
                        
                    # Get detailed model information
                    model_results = self.dbt_tools.search_model(entity)
                    if model_results:  # Check if the list has any results
                        logger.info(f"Found detailed information for model '{entity}'")
                        model_info = model_results[0].__dict__  # Use the first result
                        
                        # For code enhancement, ensure we have full content
                        if question_type == "CODE_ENHANCEMENT":
                            if 'content' not in model_info or not model_info['content']:
                                if hasattr(model_results[0], 'file_path') and model_results[0].file_path:
                                    content = self.dbt_tools.get_file_content(model_results[0].file_path)
                                    if content:
                                        model_info['content'] = content
                                        logger.info(f"Added full content for model '{entity}'")
                        
                        # Add dependency information
                        dependencies = self.dbt_tools.find_related_models(entity)
                        model_info["dependencies"] = dependencies
                        
                        results[entity] = model_info
                        
                        # Also add related models
                        related_models[entity] = dependencies
                    else:
                        logger.info(f"No detailed information found for model '{entity}'")
            else:
                results = self._get_dbt_tools_error()
                
            # Update state
            state["model_details"] = results
            state["related_models"] = related_models
            return state
            
        except Exception as e:
            logger.error(f"Error getting model details: {str(e)}")
            state["model_details"] = {
                "error": str(e),
                "status": "error",
                "message": f"Error getting model details: {str(e)}"
            }
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
        """Generate a response based on the search results."""
        try:
            question = state["messages"][-1].content
            analysis = state["question_analysis"]
            question_type = analysis.get("question_type", "GENERAL")
            
            # Format results for the prompt
            formatted_results = self._format_results_for_prompt(state)
            
            # Get instructions based on question type
            instructions = self._get_instructions_for_type(question_type, question)
            
            # Add special handling for CODE_ENHANCEMENT requests
            additional_instructions = ""
            if question_type == "CODE_ENHANCEMENT":
                # Check if we have model details with content
                has_model_content = False
                model_code = ""
                model_path = ""
                
                # Extract the actual model code for CODE_ENHANCEMENT
                if "model_details" in state and state["model_details"]:
                    for path, model_data in state["model_details"].items():
                        if isinstance(model_data, dict) and "content" in model_data and model_data["content"]:
                            has_model_content = True
                            model_code = model_data["content"]
                            model_path = path if path else model_data.get("file_path", "Unknown path")
                            break
                
                if has_model_content:
                    # Add specific instructions for the actual model code
                    additional_instructions = f"""
                    
                    SPECIAL CODE ENHANCEMENT INSTRUCTIONS:
                    
                    You are enhancing the FOLLOWING EXACT SQL model:
                    ```sql
                    -- File: {model_path}
                    {model_code}
                    ```
                    
                    IMPORTANT CODE ENHANCEMENT RULES:
                    1. ONLY modify this EXACT code - do not invent a new model
                    2. Maintain the EXACT CTE structure and format
                    3. Use the EXACT column names from the original code
                    4. Format your response as follows:
                       - EXACT MODEL OVERVIEW (file path, purpose, structure)
                       - ENHANCEMENT ANALYSIS (what changes are needed)
                       - PRECISE CODE CHANGES (show before/after code blocks)
                       - COMPLETE ENHANCED MODEL (full model with changes)
                       - VALIDATION APPROACH (queries to test the changes)
                    """
                    
                    logger.info(f"Added specific code enhancement instructions for model: {model_path}")
            
            # Compose the system prompt
            system_prompt = f"""
            You are a Data Architect expert who provides comprehensive answers about DBT models, SQL, and data architecture.
            Focus on providing detailed, accurate information with complete file paths and model lineage.
            
            {instructions}
            {additional_instructions}
            
            Here's information about the user's question:
            - Question Type: {question_type}
            - Key Entities: {', '.join(analysis.get('entities', []))}
            - Search Terms: {', '.join(analysis.get('search_terms', []))}
            - Intent: {analysis.get('intent', 'Unknown')}
            
            Here's what we found in the DBT repository:
            {formatted_results}
            
            IMPORTANT REQUIREMENTS:
            1. Include EXACT, COMPLETE file paths for every model and file you reference
            2. Show specific SQL code sections when relevant to the question
            3. Include column details, calculations, and data types when discussing fields
            4. Always mention dependencies between models
            5. If a column calculation is shown, explain it in detail
            6. Always focus on dbt specific only, never generalize it with other frameworks.

            If the search returned no results or incomplete information, admit the limitations.
            Format your response with Markdown headings, code blocks, and structured sections.
            """
            
            # Log information about the prompt
            logger.info(f"Generating response for question type: {question_type}")
            logger.info(f"Entities identified: {', '.join(analysis.get('entities', ['None']))}")
            logger.info(f"Search terms: {', '.join(analysis.get('search_terms', ['None']))}")
            
            # Check if we have any DBT results
            has_dbt_results = (
                bool(state.get("dbt_results")) or 
                bool(state.get("model_details")) or
                bool(state.get("column_details")) or 
                bool(state.get("content_search"))
            )
            
            if not has_dbt_results:
                logger.warning("No DBT results found for any search method")
                
                # Add a note to the prompt if no results were found
                system_prompt += """
                
                NOTE: No results were found in the DBT repository for this query.
                Be honest about this limitation and suggest alternative approaches.
                Explain what information you would need to provide a more detailed answer.
                """
            
            # Create messages for the chat model
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=question)
            ]
            
            # Generate the response
            response = self._safe_llm_call(messages)
            
            # For CODE_ENHANCEMENT requests, validate and potentially rewrite the response
            if question_type == "CODE_ENHANCEMENT" and has_model_content:
                response = self._validate_code_enhancement_response(response, model_code, model_path, question)
                logger.info("Applied code enhancement validation and correction")
            
            # Update state
            state["final_response"] = response
            
            return state
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            state["final_response"] = f"I encountered an error while generating a response: {str(e)}"
            return state
    
    def _validate_code_enhancement_response(self, response: str, original_code: str, model_path: str, question: str) -> str:
        """Validate and correct code enhancement responses to ensure they follow the required format."""
        try:
            # Check if the model path appears in the response
            if model_path not in response:
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
            complete_model_blocks = [block for block in code_blocks if len(block.split('\n')) > 10 and '{{' in block]
            
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
        """Format search results to highlight file paths and content for the LLM prompt."""
        formatted_text = []
        
        # First add a clear summary of found files and paths
        found_files = []
        
        # Process model search results
        if "dbt_results" in state:
            model_results = state["dbt_results"]
            if model_results and isinstance(model_results, dict):
                for model_name, model_data in model_results.items():
                    if isinstance(model_data, list):
                        for model_result in model_data:
                            if isinstance(model_result, dict) and "file_path" in model_result and "model_name" in model_result:
                                found_files.append({
                                    "type": "model",
                                    "name": model_result["model_name"],
                                    "file_path": model_result["file_path"]
                                })
        
        # Process column search results
        if "column_details" in state:
            column_results = state["column_details"]
            if column_results and isinstance(column_results, dict):
                for column_name, column_data in column_results.items():
                    if isinstance(column_data, list):
                        for col_result in column_data:
                            if isinstance(col_result, dict) and "file_path" in col_result and "model_name" in col_result:
                                found_files.append({
                                    "type": "column",
                                    "name": f"{col_result['model_name']}.{column_name}",
                                    "file_path": col_result["file_path"]
                                })
        
        # Process model details
        if "model_details" in state:
            model_details = state["model_details"]
            for model_name, model_data in model_details.items():
                if isinstance(model_data, dict) and "model" in model_data and "file_path" in model_data["model"]:
                    found_files.append({
                        "type": "model_detail",
                        "name": model_name,
                        "file_path": model_data["model"]["file_path"]
                    })
        
        # Process content search results
        if "content_search" in state:
            content_results = state["content_search"]
            for term, content_data in content_results.items():
                if isinstance(content_data, list):
                    for content_result in content_data:
                        if isinstance(content_result, dict) and "file_path" in content_result:
                            model_name = content_result.get("model_name", "Unknown")
                            found_files.append({
                                "type": "content",
                                "name": f"{model_name} (matches '{term}')",
                                "file_path": content_result["file_path"]
                            })
        
        # Log what was found
        logger.info(f"Search found {len(found_files)} files/models/columns:")
        for file_info in found_files[:5]:  # Log the first 5 to avoid overwhelming logs
            logger.info(f"  - {file_info['type']}: {file_info['name']} in {file_info['file_path']}")
        if len(found_files) > 5:
            logger.info(f"  - ... and {len(found_files) - 5} more results")
            
        # Add the file summary
        if found_files:
            formatted_text.append("## Found Files Summary")
            for file_info in found_files:
                formatted_text.append(f"- {file_info['type'].upper()}: {file_info['name']} → `{file_info['file_path']}`")
            formatted_text.append("")
        
        # Now add the detailed results with explicit sections
        formatted_text.append("## Detailed Search Results")
        
        # Format model search results
        if "dbt_results" in state:
            formatted_text.append("\n### Model Search Results")
            model_results = state["dbt_results"]
            
            for model_name, model_data in model_results.items():
                if isinstance(model_data, list):
                    for idx, model_result in enumerate(model_data, 1):
                        if isinstance(model_result, dict):
                            result_model_name = model_result.get("model_name", f"Unknown model {idx}")
                            file_path = model_result.get("file_path", "Unknown path")
                            
                            formatted_text.append(f"#### Model: {result_model_name}")
                            formatted_text.append(f"**File Path:** `{file_path}`")
                            
                            # Add model content
                            if "content" in model_result and model_result["content"]:
                                formatted_text.append("**SQL Content:**")
                                formatted_text.append("```sql")
                                formatted_text.append(f"-- File: {file_path}")
                                formatted_text.append(model_result["content"])
                                formatted_text.append("```")
                            
                            # Add schema information if available
                            if "schema_info" in model_result and model_result["schema_info"]:
                                formatted_text.append("**Schema Information:**")
                                
                                # Add column details if available
                                if "columns" in model_result["schema_info"] and model_result["schema_info"]["columns"]:
                                    formatted_text.append("**Columns:**")
                                    for column in model_result["schema_info"]["columns"]:
                                        col_name = column.get("name", "Unknown")
                                        col_desc = column.get("description", "No description")
                                        formatted_text.append(f"- `{col_name}`: {col_desc}")
                
                    # Add related models information if available
                    if model_data and isinstance(model_data[0], dict) and "dependencies" in model_data[0]:
                        formatted_text.append("**Related Models:**")
                        related = model_data[0]["dependencies"]
                        
                        if "upstream" in related and related["upstream"]:
                            formatted_text.append("*Upstream Dependencies:*")
                            for upstream in related["upstream"]:
                                formatted_text.append(f"- `{upstream}`")
                        
                        if "downstream" in related and related["downstream"]:
                            formatted_text.append("*Downstream Dependencies:*")
                            for downstream in related["downstream"]:
                                formatted_text.append(f"- `{downstream}`")
        
        # Format column search results
        if "column_details" in state:
            formatted_text.append("\n### Column Search Results")
            column_results = state["column_details"]
            
            for column_name, column_data in column_results.items():
                # Handle "content:" prefix in column names
                display_name = column_name
                if column_name.startswith("content:"):
                    display_name = f"Content match for '{column_name[8:]}'"
                
                formatted_text.append(f"#### Column: {display_name}")
                
                if isinstance(column_data, list):
                    for idx, col_result in enumerate(column_data, 1):
                        if isinstance(col_result, dict):
                            model_name = col_result.get("model_name", f"Unknown model {idx}")
                            file_path = col_result.get("file_path", "Unknown path")
                            match_type = col_result.get("match_type", "Unknown match")
                            
                            formatted_text.append(f"**Found in Model:** {model_name}")
                            formatted_text.append(f"**File Path:** `{file_path}`")
                            formatted_text.append(f"**Match Type:** {match_type}")
                            
                            # Add calculation information
                            if "calculation" in col_result and col_result["calculation"]:
                                formatted_text.append("**Calculation:**")
                                formatted_text.append("```sql")
                                formatted_text.append(col_result["calculation"])
                                formatted_text.append("```")
                            
                            # Add clean calculation if available (more readable)
                            if "clean_calculation" in col_result and col_result["clean_calculation"]:
                                formatted_text.append("**Clean Calculation:**")
                                formatted_text.append("```sql")
                                formatted_text.append(col_result["clean_calculation"])
                                formatted_text.append("```")
                            
                            # Add context information
                            if "match_context" in col_result and col_result["match_context"] and not col_result.get("calculation"):
                                formatted_text.append("**Match Context:**")
                                formatted_text.append("```")
                                formatted_text.append(col_result["match_context"])
                                formatted_text.append("```")
                            
                            # Add description if available
                            if "description" in col_result and col_result["description"]:
                                formatted_text.append(f"**Description:** {col_result['description']}")
        
        # Format model details results
        if "model_details" in state:
            formatted_text.append("\n### Model Details")
            model_details = state["model_details"]
            
            for model_name, model_data in model_details.items():
                if isinstance(model_data, dict) and "model" in model_data:
                    model = model_data["model"]
                    
                    formatted_text.append(f"#### Detailed Model: {model_name}")
                    
                    # Add file path
                    if "file_path" in model:
                        formatted_text.append(f"**File Path:** `{model['file_path']}`")
                    
                    # Add model type
                    if "model_type" in model:
                        formatted_text.append(f"**Materialization:** {model['model_type']}")
                    
                    # Add description
                    if "description" in model and model["description"]:
                        formatted_text.append(f"**Description:** {model['description']}")
                    
                    # Add columns
                    if "columns" in model and model["columns"]:
                        formatted_text.append("**Columns:**")
                        for column in model["columns"]:
                            col_name = column.get("name", "Unknown")
                            col_desc = column.get("description", "No description")
                            formatted_text.append(f"- `{col_name}`: {col_desc}")
                    
                    # Add dependencies
                    if "dependencies" in model and model["dependencies"]:
                        formatted_text.append("**Dependencies:**")
                        dependencies = model["dependencies"]
                        
                        if "upstream" in dependencies and dependencies["upstream"]:
                            formatted_text.append("*Upstream Dependencies:*")
                            for upstream in dependencies["upstream"]:
                                formatted_text.append(f"- `{upstream}`")
                        
                        if "downstream" in dependencies and dependencies["downstream"]:
                            formatted_text.append("*Downstream Dependencies:*")
                            for downstream in dependencies["downstream"]:
                                formatted_text.append(f"- `{downstream}`")
        
        # Format content search results
        if "content_search" in state:
            formatted_text.append("\n### Content Search Results")
            content_results = state["content_search"]
            
            for term, content_data in content_results.items():
                # Handle "path:" prefix in terms
                display_term = term
                if term.startswith("path:"):
                    display_term = f"Path containing '{term[5:]}'"
                    
                formatted_text.append(f"#### Term: {display_term}")
                
                if isinstance(content_data, list):
                    for idx, content_result in enumerate(content_data, 1):
                        if isinstance(content_result, dict):
                            file_path = content_result.get("file_path", "Unknown path")
                            model_name = content_result.get("model_name", "")
                            
                            if model_name:
                                formatted_text.append(f"**Found in Model:** {model_name}")
                            formatted_text.append(f"**File Path:** `{file_path}`")
                            
                            # Add match contexts
                            if "match_contexts" in content_result and content_result["match_contexts"]:
                                formatted_text.append("**Matching Contexts:**")
                                for i, context in enumerate(content_result["match_contexts"][:3], 1):  # Limit to 3 contexts
                                    formatted_text.append(f"*Match {i}:*")
                                    formatted_text.append("```")
                                    formatted_text.append(context)
                                    formatted_text.append("```")
                            
                            # If no match_contexts but context available
                            elif "match_context" in content_result and content_result["match_context"]:
                                formatted_text.append("**Match Context:**")
                                formatted_text.append("```")
                                formatted_text.append(content_result["match_context"])
                                formatted_text.append("```")
        
                            
                            # Add model content
                            if "content" in content_result and content_result["content"]:
                                formatted_text.append("**Full SQL Content:**")
                                formatted_text.append("```sql")
                                formatted_text.append(f"-- File: {file_path}")
                                formatted_text.append(content_result["content"])
                                formatted_text.append("```")
                            
                            formatted_text.append("---")
        
        final_formatted_text = "\n".join(formatted_text)
        
        # Log summary of what's being sent to the LLM
        logger.info(f"Formatted prompt contains {len(final_formatted_text)} characters")
        if not found_files:
            logger.warning("No search results were found to include in the prompt!")
        
        return final_formatted_text
    
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

    def _get_code_enhancement_instructions(self, query: str) -> str:
        """Get instructions for code enhancement tasks."""
        return """
        You are a SENIOR DBT SQL DEVELOPER EXPERT who provides highly specific, contextual model enhancements - exactly like how you provide model information.
        
        <CRITICAL PRINCIPLE>
        When enhancing DBT models, you should ONLY modify the EXACT code provided by the user, just as you would explain that EXACT code when giving model information. 
        NEVER invent new models, columns, or structures that don't appear in the provided SQL.
        </CRITICAL PRINCIPLE>
        
        <THINKING PROCESS>
        First, analyze the provided DBT model in detail:
        1. UNDERSTAND THE EXACT MODEL STRUCTURE
           - Identify materialization type from the config() block
           - Map the exact CTE structure (with, final, etc.) and purpose of each CTE
           - Note exact column names, data types, and calculations
           - Identify exact referenced models ({{ ref() }}) and sources ({{ source() }})
           - Document join conditions and filtering logic
        
        2. PRECISELY UNDERSTAND ENHANCEMENT REQUEST
           - Determine exactly what enhancement is needed for THIS SPECIFIC model
           - Locate exactly which CTE or section needs modification
           - Consider if this affects other parts of the model
        
        3. PLAN PRECISE MODIFICATIONS TO THE EXISTING CODE
           - Identify the minimum changes needed to implement the request
           - Ensure changes align with the existing code style and patterns
           - Preserve all existing functionality while adding the new features
        </THINKING PROCESS>
        
        Now, respond with the same level of specificity as when you provide model information:
        
        1. EXACT MODEL OVERVIEW
           - Identify the exact file path: models/path/to/model.sql
           - Summarize the current model's purpose and structure
           - Note the exact model materializations and configurations
           - List the exact upstream dependencies
        
        2. ENHANCEMENT ANALYSIS
           - Clearly state what specific enhancement is being implemented
           - Identify exactly which part of the model needs to change
           - Explain why the change is being made this way
        
        3. PRECISE CODE CHANGES
           - Show the exact SQL block being modified:
             ```sql
             -- CURRENT CODE IN THE EXACT MODEL:
             select
               order_key,
               sum(gross_item_sales_amount) as gross_item_sales_amount
             from order_item
             group by 1
             
             -- MODIFIED CODE WITH ENHANCEMENTS:
             select
               order_key,
               sum(gross_item_sales_amount) as gross_item_sales_amount,
               avg(gross_item_sales_amount) as avg_gross_item_sales_amount  -- Added calculation
             from order_item
             group by 1
             ```
        
        4. COMPLETE ENHANCED MODEL
           - Provide the complete model SQL with changes integrated
           - Use exact jinja syntax as in the original
           - Include all original CTEs and SQL logic
           - Add helpful comments for any new or modified code
        
        5. VALIDATION APPROACH
           - Provide a validation query specifically for this model
           - Suggest specific dbt tests for the modified/added columns
           - Highlight any potential edge cases
        
        IMPORTANT REQUIREMENTS:
        - NEVER GENERALIZE - only modify the exact code provided
        - Use the EXACT file path from the model
        - Use the EXACT CTE structure from the model
        - Use ONLY column names that exist in the model
        - Reference ONLY models that are in the original code
        - Maintain the EXACT same SQL structure and formatting
        - Ensure your solution addresses the specific enhancement request
        - VERIFY your solution by comparing it against the original code
        - If a column calculation is shown, explain it exactly as it appears
        """

    def _get_documentation_instructions(self, query: str) -> str:
        """Get instructions for documentation tasks."""
        return """
        When generating documentation guidance, focus on comprehensive yet practical documentation:
        
        1. MODEL DOCUMENTATION STRUCTURE
        - Provide a clear documentation template:
          * Model purpose and business context
          * Technical implementation details
          * File path and location in project
          * Materialization strategy and schedule
          * Data sources and destinations
        - For each documentation section:
          * Provide example content
          * Explain what to include and why it matters
        
        2. SCHEMA DOCUMENTATION
        - Create detailed column documentation:
          * Column name, data type, and description
          * Business definition and calculation logic
          * Source of the data
          * Examples of valid values
        - Present as ready-to-use YAML:
          ```yaml
          version: 2
          models:
            - name: model_name
              description: "Detailed model description"
              columns:
                - name: column_name
                  description: "Clear column description"
                  tests:
                    - unique
                    - not_null
          ```
        
        3. CODE COMMENTS
        - Suggest inline SQL comments:
          * Header block explaining the model's purpose
          * Section dividers for complex queries
          * Explanations for complex calculations
          * Logic explanations for business rules
        - Provide examples with proper formatting
        
        4. TESTS AND ASSERTIONS
        - Document recommended tests:
          * Data quality tests (nulls, uniques, etc.)
          * Referential integrity tests
          * Custom business logic tests
          * Performance test criteria
        - Show how to implement in schema files
        
        5. DOCUMENTATION MAINTENANCE
        - Provide guidance on:
          * When to update documentation
          * How to keep docs in sync with code
          * Documentation review process
          * Tools to automate documentation
        
        ALWAYS INCLUDE:
        - Ready-to-use documentation examples
        - Both technical and business-focused documentation
        - Clear file paths for all referenced models
        - YAML schema definitions with proper syntax
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