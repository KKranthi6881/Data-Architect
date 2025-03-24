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
            
            # Model information questions
            if question_type == "MODEL_INFO" and entities:
                return "search_models"
                
            # Lineage and dependency questions
            elif question_type in ["LINEAGE", "DEPENDENCIES"] and entities:
                return "get_model_details"
                
            # If there are potential column names in the entities
            elif any(len(entity.split('.')) == 2 for entity in entities):
                return "search_columns"
                
            # For all other cases, try content search
            else:
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
            # Get the last message
            messages = state["messages"]
            last_message = messages[-1].content
            
            # Look for column-specific patterns
            column_patterns = self._extract_column_patterns(last_message)
            
            # Create prompt for question analysis
            prompt = f"""
            Analyze the following question about DBT models and data architecture. Determine:
            
            1. Question Type - Select ONE from:
               - MODEL_INFO: Questions about model structure, SQL code, purpose, and functionality
               - LINEAGE: Questions about model relationships, dependencies, and data flow
               - DEPENDENCIES: Questions about specific model dependencies and potential impact
               - CODE_ENHANCEMENT: Questions about improving or optimizing DBT code
               - DOCUMENTATION: Questions about generating or improving documentation
               - DEVELOPMENT: Questions about implementation and development tasks
               - GENERAL: Other types of questions
            
            2. Key Entities - Extract ALL relevant entities:
               - Model names (e.g., "orders", "customers", "financial_metrics")
               - Column names (e.g., "order_id", "customer_name", "total_value")
               - Combined entities (e.g., "orders.order_id", "customers.email")
               - Any specific tables, schemas, or datasets mentioned
            
            3. Search Terms - Key words to search for in code/docs:
               - Technical terms (e.g., "materialization", "incremental", "full refresh")
               - Business concepts (e.g., "revenue calculation", "user retention")
               - Any specific code patterns or logic mentioned
               - Column calculation terms (e.g., "gross_amount", "net_sales", "discount")
            
            4. Primary Intent - The core goal of the question
            
            5. Rephrased Question - A clear, searchable version of the question
            
            Question: {last_message}
            
            {f"Detected Column Patterns: {', '.join(column_patterns)}" if column_patterns else ""}
            
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
            response = response.strip()
            if response.startswith('```json'):
                response = response[7:]
            if response.endswith('```'):
                response = response[:-3]
            response = response.strip()
            
            try:
                # Parse the response
                analysis = QuestionAnalysis.parse_raw(response)
                
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
                logger.error(f"Raw response: {response}")
                raise
            
        except Exception as e:
            logger.error(f"Error parsing question: {str(e)}")
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
                        results[entity] = [result.__dict__ for result in model_results]
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
            
            results = {}
            related_models = {}
            
            # Only proceed with DBT search if we have DBT tools initialized
            if self.dbt_tools:
                # Get details for each model entity
                for entity in entities:
                    # Ignore entities with dots (column references)
                    if '.' in entity:
                        continue
                        
                    # Get detailed model information
                    model_results = self.dbt_tools.search_model(entity)
                    if model_results:  # Check if the list has any results
                        logger.info(f"Found detailed information for model '{entity}'")
                        model_info = model_results[0].__dict__  # Use the first result
                        
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
            
            # Compose the system prompt
            system_prompt = f"""
            You are a Data Architect expert who provides comprehensive answers about DBT models, SQL, and data architecture.
            Focus on providing detailed, accurate information with complete file paths and model lineage.
            
            {instructions}
            
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
            
            # Update state
            state["final_response"] = response
            
            return state
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            state["final_response"] = f"I encountered an error while generating a response: {str(e)}"
            return state
    
    def _safe_llm_call(self, messages: List[BaseMessage], max_retries: int = 2) -> str:
        """Make a safe call to the LLM with retry logic."""
        retries = 0
        last_error = None
        
        while retries <= max_retries:
            try:
                response = self.llm(messages)
                return response.content
            except Exception as e:
                last_error = e
                retries += 1
                logger.warning(f"LLM call failed (attempt {retries}): {str(e)}")
                time.sleep(1)  # Short delay before retry
        
        # If we get here, all retries failed
        logger.error(f"All LLM call attempts failed: {str(last_error)}")
        return f"I'm sorry, I encountered an error while processing your request. Please try again later."

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
        - Identify data sources (FROM clauses, JOINs, CTEs)
        
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
        """Get instructions for lineage questions."""
        return """
        When explaining model lineage, provide a complete end-to-end view of the data flow:
        
        1. LINEAGE OVERVIEW
        - Summarize the model's position in the data pipeline
        - Identify its tier or layer (source, staging, intermediate, mart)
        - Explain the general flow of data through this model
        
        2. UPSTREAM DEPENDENCIES
        - List ALL upstream models with file paths (models/path/to/model.sql)
        - For each upstream model, briefly explain:
          * What data it provides
          * How it's joined or referenced
          * Its materialization type
        - Show the direct SQL references (ref() functions)
        - Include any source tables if used directly
        
        3. DOWNSTREAM DEPENDENCIES
        - List ALL downstream models with file paths
        - Explain how each downstream model uses this data
        - Identify critical downstream consumers
        - Note any exposure points (BI dashboards, APIs, etc.)
        
        4. CRITICAL PATH ANALYSIS
        - Identify the critical path of dependencies
        - Highlight any bottlenecks or performance concerns
        - Show run order and execution dependencies
        - Explain timing considerations for scheduling
        
        5. VISUALIZATION
        - Create a text-based diagram of dependencies using markdown
        - Use indentation or symbols to show hierarchy
        - Use arrows (→) to indicate data flow direction
        - Highlight the model of interest in the diagram
        
        ALWAYS INCLUDE:
        - Complete file paths for every model mentioned
        - Model types (staging, intermediate, mart, etc.)
        - Materialization methods (view, table, incremental, etc.)
        - Direct and indirect dependency relationships
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
        When providing code enhancement guidance, focus on practical, actionable improvements:
        
        1. CODE ANALYSIS
        - Analyze the current implementation in detail
          * Show the file path (models/path/to/model.sql)
          * Identify the model's purpose and business context
          * Break down the current SQL structure
        - Identify specific issues:
          * Performance bottlenecks (full table scans, inefficient joins)
          * Maintainability concerns (complex logic, duplication)
          * Readability issues (poor formatting, lack of comments)
          * Design problems (inappropriate materialization, granularity issues)
        
        2. TARGETED ENHANCEMENTS
        - Suggest specific, actionable improvements:
          * SQL optimizations with before/after examples
          * Better indexing or partitioning strategies
          * Refactoring complex expressions
          * Improved commenting and documentation
        - For each suggestion:
          * Explain WHY it's an improvement
          * Show HOW to implement it
          * Describe the BENEFIT it provides
        
        3. IMPLEMENTATION PLAN
        - Prioritize changes by:
          * Impact (high/medium/low)
          * Effort required (high/medium/low)
          * Risk level (high/medium/low)
        - Provide a step-by-step implementation approach
          * What to change first
          * How to test each change
          * What downstream impacts to expect
        
        4. CODE EXAMPLES
        - For each recommendation, provide:
          * Clear BEFORE and AFTER code samples
          * Syntax highlighted SQL blocks
          * Line-by-line explanations of changes
          * Comments explaining the reasoning
        
        5. TESTING STRATEGY
        - Suggest specific tests to validate changes:
          * Row count validations
          * Sum/total checks
          * Data quality assertions
          * Performance benchmarks
        
        ALWAYS INCLUDE:
        - Specific file paths for all code examples
        - Clear, implementable code suggestions
        - Explanations of WHY each change helps
        - Consideration of both performance AND maintainability
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
        When providing development guidance, focus on practical implementation:
        
        1. REQUIREMENTS ANALYSIS
        - Break down the business requirements:
          * Data sources needed
          * Required transformations
          * Expected output structure
          * Business rules and logic
        - Map requirements to technical implementation:
          * Which models to create/modify
          * Dependencies needed
          * Tests to implement
        
        2. IMPLEMENTATION ARCHITECTURE
        - Suggest a clear implementation approach:
          * Model structure and organization
          * File paths and naming conventions
          * Materialization strategies
          * Dependency structure
        - Provide a diagram or visualization:
          * Source → Staging → Intermediate → Mart flow
          * Dependencies between models
          * Data flow direction
        
        3. CODE EXAMPLES
        - Provide complete, executable code examples:
          * Full SQL queries with proper formatting
          * Schema YAML configurations
          * dbt_project.yml settings
          * Custom macros if needed
        - For each code example:
          * Explain each section's purpose
          * Highlight key transformations
          * Note performance considerations
        
        4. TESTING STRATEGY
        - Define comprehensive testing:
          * Standard tests (unique, not_null, etc.)
          * Referential integrity tests
          * Business logic validations
          * Custom data quality checks
        - Show test implementations in YAML:
          ```yaml
          version: 2
          models:
            - name: model_name
              columns:
                - name: column_name
                  tests:
                    - unique
                    - not_null
          ```
        
        5. IMPLEMENTATION STEPS
        - Provide a step-by-step implementation plan:
          * Order of model creation
          * Dependency management
          * Testing procedure
          * Validation approach
        - Include commands to run:
          * dbt run commands
          * dbt test commands
          * Custom script examples
        
        ALWAYS INCLUDE:
        - Complete file paths for all models
        - Full SQL code examples with comments
        - YAML configuration examples
        - Step-by-step implementation instructions
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