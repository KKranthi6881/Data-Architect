from typing import Dict, List, Any, Annotated, Sequence, TypedDict, Union, Optional
from langchain_core.messages import BaseMessage, HumanMessage
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
from src.agents.data_architect.data_architect import DataArchitectAgent
from src.agents.data_architect.question_parser import QuestionParserSystem
from src.utils import ChromaDBManager

# Set up logger
logger = logging.getLogger(__name__)

# Define state type
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "The messages in the conversation"]
    question_analysis: Annotated[Dict, "Parsed question analysis"]
    github_results: Annotated[Dict, "GitHub code search results"]
    sql_results: Annotated[Dict, "SQL code search results"]
    doc_results: Annotated[Dict, "Documentation search results"]
    dbt_results: Annotated[Dict, "DBT model search results"]
    relationship_results: Annotated[Dict, "Relationship search results"]
    combined_analysis: Annotated[str, "Combined analysis of all search results"]
    final_response: Annotated[str, "Final formatted response to the user"]

# Define structured outputs for question analysis
class QuestionAnalysis(BaseModel):
    question_type: str = Field(
        description="Type of question (SCHEMA_INFO, DEVELOPMENT, BUSINESS_LOGIC, DATA_LINEAGE, GENERAL)"
    )
    entities: List[str] = Field(
        description="Key entities mentioned in the question (tables, models, etc.)"
    )
    intent: str = Field(
        description="Primary intent of the question"
    )
    focus_area: str = Field(
        description="Area to focus search on (SQL, DBT, DOCUMENTATION, ALL)"
    )
    rephrased_question: str = Field(
        description="Question rephrased for search clarity"
    )

class DataArchitectAgent:
    """
    Data Architect Agent that processes questions about data architecture,
    searches relevant sources, and provides comprehensive answers.
    """
    
    def __init__(self, tools: SearchTools):
        """Initialize the Data Architect Agent with search tools."""
        self.tools = tools
        self.db = ChatDatabase()
        self._lock = Lock()
        
        # Initialize models with appropriate settings
        self.question_model = ChatOllama(
            model="gemma3:1B",
            temperature=0,
            base_url="http://localhost:11434",
        )
        
        self.analysis_model = ChatOllama(
            model="gemma3:1B",
            temperature=0,
            base_url="http://localhost:11434",
        )
        
        self.response_model = ChatOllama(
            model="gemma3:1B",
            temperature=0.2,
            base_url="http://localhost:11434",
        )
        
        # Initialize parsers
        self.question_parser = PydanticOutputParser(pydantic_object=QuestionAnalysis)
        
        # Create the agent graph
        self.graph = self._create_agent_graph()
    
    def _create_agent_graph(self) -> StateGraph:
        """Create the agent workflow graph."""
        # Create the graph
        graph = StateGraph(AgentState)
        
        # Add nodes
        graph.add_node("parse_question", self._parse_question)
        graph.add_node("search_github", self._search_github)
        graph.add_node("search_sql", self._search_sql)
        graph.add_node("search_docs", self._search_docs)
        graph.add_node("search_dbt", self._search_dbt)
        graph.add_node("search_relationships", self._search_relationships)
        graph.add_node("combine_results", self._combine_results)
        graph.add_node("generate_response", self._generate_response)
        
        # Add edges
        graph.add_edge(START, "parse_question")
        
        # After parsing, search all relevant sources in parallel
        graph.add_edge("parse_question", "search_github")
        graph.add_edge("parse_question", "search_sql")
        graph.add_edge("parse_question", "search_docs")
        graph.add_edge("parse_question", "search_dbt")
        graph.add_edge("parse_question", "search_relationships")
        
        # Combine all search results
        graph.add_edge("search_github", "combine_results")
        graph.add_edge("search_sql", "combine_results")
        graph.add_edge("search_docs", "combine_results")
        graph.add_edge("search_dbt", "combine_results")
        graph.add_edge("search_relationships", "combine_results")
        
        # Generate the final response
        graph.add_edge("combine_results", "generate_response")
        graph.add_edge("generate_response", END)
        
        # Create SQLite saver for checkpoints
        db_path = str(Path(__file__).parent.parent.parent / "chat_history.db")
        conn = connect(db_path, check_same_thread=False)
        checkpointer = SqliteSaver(conn)
        
        # Compile the graph
        return graph.compile(checkpointer=checkpointer)

    def _parse_question(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process and analyze the question."""
        try:
            messages = state['messages']
            if not messages:
                return {}
            
            query = messages[-1].content if isinstance(messages[-1], BaseMessage) else str(messages[-1])
            
            # Log the question processing
            logger.info(f"Processing question: {query[:100]}...")
            
            # Initialize state with start time for tracking processing duration
            result = {
                'start_time': time.time(),
                'original_question': query,
                'question_analysis': {
                    'rephrased_question': query,
                    'question_type': 'technical',
                    'key_points': [],
                    'business_context': {}
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in question processing: {str(e)}", exc_info=True)
            return {
                'question_analysis': {
                    'rephrased_question': query if 'query' in locals() else "",
                    'question_type': 'unknown',
                    'error': str(e)
                }
            }

    def _search_github(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Search for GitHub repository content based on the question analysis."""
        try:
            messages = state.get('messages', [])
            question_analysis = state.get('question_analysis', {})
            if not messages:
                return {"github_results": {}}
            
            # Get the rephrased question for search
            query = question_analysis.get('rephrased_question', '')
            if not query:
                query = messages[-1].content if isinstance(messages[-1], BaseMessage) else str(messages[-1])
            
            # Determine if we should search GitHub based on focus area
            focus_area = question_analysis.get('focus_area', 'ALL')
            if focus_area not in ['ALL', 'CODE', 'GITHUB']:
                logger.info(f"Skipping GitHub search based on focus area: {focus_area}")
                return {"github_results": {"status": "skipped", "results": []}}
            
            # Extract entities to enhance the search
            entities = question_analysis.get('entities', [])
            if entities:
                # Add key entities to the query for better results
                enhanced_query = f"{query} {' '.join(entities)}"
            else:
                enhanced_query = query
            
            logger.info(f"Searching GitHub repositories with query: {enhanced_query}")
            
            # Use the tools directly instead of the GitHub agent
            results = self.tools.search_github_repos(enhanced_query, limit=5)
            
            # Log detailed results for debugging
            self._log_search_results("GitHub", results)
            
            return {"github_results": results}
            
        except Exception as e:
            logger.error(f"Error in GitHub search: {str(e)}", exc_info=True)
            return {"github_results": {"status": "error", "error": str(e), "results": []}}

    def _search_sql(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process SQL code search and analysis."""
        try:
            messages = state['messages']
            if not messages:
                return {}
            
            query = messages[-1].content if isinstance(messages[-1], BaseMessage) else str(messages[-1])
            
            # Use the tools object passed to the constructor
            tools = SearchTools(ChromaDBManager())
            search_results = tools.search_sql_schema(query)
            
            # Log detailed search results
            logger.info(f"Code search query: '{query}'")
            if 'results' in search_results and search_results['results']:
                logger.info(f"Code search returned {len(search_results['results'])} results")
                for i, result in enumerate(search_results['results'][:3], 1):  # Log first 3 results
                    metadata = result.get('metadata', {})
                    source = metadata.get('source', 'Unknown')
                    logger.info(f"Code result {i}: File={source}, Score={metadata.get('score', 'N/A')}")
                    # Log a preview of the content
                    content = result.get('content', '')
                    content_preview = content[:100] + "..." if len(content) > 100 else content
                    logger.info(f"Content preview: {content_preview}")
            else:
                logger.info("Code search returned no results")
            
            # Format results
            sql_context = {"query": query, "results": search_results.get('results', [])}
            sql_analysis = "Code analysis completed. Found {} results.".format(
                len(search_results.get('results', []))
            )
            
            return {
                "sql_results": sql_context,
                "sql_analysis": sql_analysis
            }
            
        except Exception as e:
            logger.error(f"Error in SQL processing: {str(e)}", exc_info=True)
            return {
                "sql_analysis": f"Error during SQL analysis: {str(e)}",
                "sql_results": {"query": query if 'query' in locals() else "", "results": []}
            }

    def _search_docs(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process documentation analysis."""
        try:
            messages = state['messages']
            if not messages:
                return state
            
            query = messages[-1].content if isinstance(messages[-1], BaseMessage) else str(messages[-1])
            
            # Search documentation
            search_results = self.tools.search_documentation(query)
            
            # Format doc context
            doc_snippets = []
            for result in search_results.get('results', []):
                doc_snippets.append(
                    f"Content:\n{result.get('content', '')}\n"
                    f"Metadata: {result.get('metadata', {})}\n"
                )
            
            doc_context = "\n".join(doc_snippets)
            
            # If no doc results, return empty analysis
            if not doc_context:
                return {
                    "doc_results": {"query": query, "results": []},
                    "doc_analysis": "No relevant documentation found."
                }
            
            # Generate analysis
            formatted_prompt = self.doc_prompt.format(
                doc_context=doc_context,
                user_question=query
            )
            response = self.doc_model.invoke(formatted_prompt)
            response_text = response.content if isinstance(response, BaseMessage) else str(response)
            
            try:
                analysis = self.doc_parser.parse(response_text)
                output = f"""
                Documentation Analysis Results:
                
                1. Key Concepts:
                {analysis.key_concepts}
                
                2. Workflows:
                {analysis.workflows}
                
                3. Requirements:
                {analysis.requirements}
                
                4. Additional Context:
                {analysis.additional_context}
                """
            except Exception as parse_error:
                logger.warning(f"Failed to parse doc output: {str(parse_error)}")
                output = response_text
            
            return {
                "doc_results": {"query": query, "results": search_results.get('results', [])},
                "doc_analysis": output
            }
            
        except Exception as e:
            logger.error(f"Error in doc processing: {str(e)}")
            return {
                "doc_analysis": f"Error during documentation analysis: {str(e)}",
                "doc_results": {}
            }

    def _search_dbt(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Search for DBT models based on the question analysis."""
        try:
            messages = state.get('messages', [])
            question_analysis = state.get('question_analysis', {})
            if not messages:
                return {"dbt_results": {}}
            
            # Get the rephrased question for search
            query = question_analysis.get('rephrased_question', '')
            if not query:
                query = messages[-1].content if isinstance(messages[-1], BaseMessage) else str(messages[-1])
            
            # Determine if we should search DBT based on focus area
            focus_area = question_analysis.get('focus_area', 'ALL')
            if focus_area not in ['ALL', 'DBT']:
                logger.info(f"Skipping DBT search based on focus area: {focus_area}")
                return {"dbt_results": {"status": "skipped", "results": []}}
            
            # Extract entities to enhance the search
            entities = question_analysis.get('entities', [])
            if entities:
                # Add key entities to the query for better results
                enhanced_query = f"dbt model {query} {' '.join(entities)}"
            else:
                enhanced_query = f"dbt model {query}"
            
            logger.info(f"Searching DBT models with query: {enhanced_query}")
            
            # Search DBT models
            results = self.tools.search_dbt_models(enhanced_query, limit=5)
            
            # Log detailed results for debugging
            self._log_search_results("DBT", results)
            
            return {"dbt_results": results}
            
        except Exception as e:
            logger.error(f"Error in DBT search: {str(e)}", exc_info=True)
            return {"dbt_results": {"status": "error", "error": str(e), "results": []}}

    def _search_relationships(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Search for relationships between data entities."""
        try:
            messages = state.get('messages', [])
            question_analysis = state.get('question_analysis', {})
            if not messages:
                return {"relationship_results": {}}
            
            # Get the rephrased question for search
            query = question_analysis.get('rephrased_question', '')
            if not query:
                query = messages[-1].content if isinstance(messages[-1], BaseMessage) else str(messages[-1])
            
            # Determine if we should search relationships based on question type
            question_type = question_analysis.get('question_type', 'GENERAL')
            if question_type not in ['DATA_LINEAGE', 'SCHEMA_INFO', 'GENERAL']:
                logger.info(f"Skipping relationship search based on question type: {question_type}")
                return {"relationship_results": {"status": "skipped", "results": []}}
            
            # Extract entities to enhance the search
            entities = question_analysis.get('entities', [])
            if entities:
                # Add key entities to the query for better results
                enhanced_query = f"{query} {' '.join(entities)}"
            else:
                enhanced_query = query
            
            logger.info(f"Searching relationships with query: {enhanced_query}")
            
            # Search relationships
            results = self.tools.search_relationships(enhanced_query)
            
            # Log detailed results for debugging
            self._log_search_results("Relationships", results)
            
            return {"relationship_results": results}
            
        except Exception as e:
            logger.error(f"Error in relationship search: {str(e)}", exc_info=True)
            return {"relationship_results": {"status": "error", "error": str(e), "results": []}}

    def _log_search_results(self, source: str, results: Dict[str, Any]) -> None:
        """Log detailed search results for debugging."""
        try:
            status = results.get('status', 'unknown')
            result_list = results.get('results', [])
            
            logger.info(f"{source} search status: {status}, found {len(result_list)} results")
            
            # Log a preview of each result
            for i, result in enumerate(result_list[:3], 1):  # Log first 3 results
                logger.info(f"{source} result {i}:")
                
                # Log different fields based on result type
                if source == "GitHub":
                    repo_info = result.get('repository', {})
                    file_info = result.get('file', {})
                    logger.info(f"  File: {file_info.get('path', 'Unknown')}")
                    logger.info(f"  Repo: {repo_info.get('name', 'Unknown')}")
                    logger.info(f"  Language: {file_info.get('language', 'Unknown')}")
                    
                    # Log DBT info if available
                    dbt_info = result.get('dbt_info', {})
                    if dbt_info:
                        logger.info(f"  DBT Model: {dbt_info.get('model_name', 'Unknown')}")
                        logger.info(f"  Materialization: {dbt_info.get('materialization', 'Unknown')}")
                
                elif source == "SQL":
                    metadata = result.get('metadata', {})
                    logger.info(f"  Source: {metadata.get('source', 'Unknown')}")
                    logger.info(f"  Tables: {', '.join(metadata.get('tables', []))[:100]}")
                
                elif source == "DBT":
                    dbt_info = result.get('dbt_info', {})
                    if dbt_info:
                        logger.info(f"  Model: {dbt_info.get('model_name', 'Unknown')}")
                        logger.info(f"  References: {', '.join(dbt_info.get('references', []))[:100]}")
                
                elif source == "Relationships":
                    logger.info(f"  Source: {result.get('source', 'Unknown')}")
                    logger.info(f"  Model: {result.get('model', 'Unknown')}")
                
                # Log content preview for all result types
                content = result.get('content', '')
                if content:
                    content_preview = content[:100] + "..." if len(content) > 100 else content
                    logger.info(f"  Content preview: {content_preview}")
                
                logger.info("  ---")
                
        except Exception as e:
            logger.error(f"Error logging {source} search results: {str(e)}")
    
    def _combine_results(self, state: AgentState) -> AgentState:
        """Combine all search results into a comprehensive analysis."""
        try:
            # Extract results from state
            github_results = state.get('github_results', {}).get('results', [])
            sql_results = state.get('sql_results', {}).get('results', [])
            doc_results = state.get('doc_results', {}).get('results', [])
            dbt_results = state.get('dbt_results', {}).get('results', [])
            relationship_results = state.get('relationship_results', {}).get('results', [])
            
            # Log the number of results from each source
            logger.info(f"Combining results: GitHub={len(github_results)}, SQL={len(sql_results)}, "
                       f"Docs={len(doc_results)}, DBT={len(dbt_results)}, "
                       f"Relationships={len(relationship_results)}")
            
            # Check if we have any results at all
            if not any([github_results, sql_results, doc_results, dbt_results, relationship_results]):
                logger.warning("No results found from any source")
                state['combined_analysis'] = "No relevant information found in any of the searched sources."
                return state
            
            # Format each type of result
            formatted_results = []
            
            if github_results:
                formatted_github = self._format_results_for_llm(github_results, "GitHub")
                formatted_results.append(formatted_github)
                
            if sql_results:
                formatted_sql = self._format_results_for_llm(sql_results, "SQL")
                formatted_results.append(formatted_sql)
                
            if doc_results:
                formatted_docs = self._format_results_for_llm(doc_results, "Documentation")
                formatted_results.append(formatted_docs)
                
            if dbt_results:
                formatted_dbt = self._format_results_for_llm(dbt_results, "DBT")
                formatted_results.append(formatted_dbt)
                
            if relationship_results:
                formatted_relationships = self._format_results_for_llm(relationship_results, "Relationships")
                formatted_results.append(formatted_relationships)
            
            # Combine all formatted results
            combined_analysis = "\n\n".join(formatted_results)
            
            # Update the state with the combined analysis
            state['combined_analysis'] = combined_analysis
            
            return state
            
        except Exception as e:
            logger.error(f"Error combining results: {str(e)}", exc_info=True)
            state['combined_analysis'] = f"Error combining results: {str(e)}"
            return state

    def _format_results_for_llm(self, results, source_type):
        """Format search results for the LLM in a readable way."""
        if not results:
            return "No results found."
        
        formatted = f"Found {len(results)} {source_type} results:\n\n"
        
        for i, result in enumerate(results, 1):
            try:
                # Extract metadata properly
                metadata = result.get('metadata', {})
                file_path = metadata.get('file_path', 'Unknown')
                repo_url = metadata.get('repo_url', 'Unknown')
                repo_name = repo_url.split('/')[-1] if repo_url != 'Unknown' else 'Unknown'
                file_type = metadata.get('file_type', 'Unknown')
                
                # Get content and truncate if too long
                content = result.get('content', 'No content available')
                if len(content) > 1500:  # Truncate long content
                    content = content[:1500] + "...(truncated)"
                
                formatted += f"### Result {i}: {file_path}\n"
                formatted += f"**Repository:** {repo_name}\n"
                formatted += f"**File Type:** {file_type}\n"
                formatted += f"**Content:**\n```{file_type}\n{content}\n```\n\n"
            except Exception as e:
                logger.error(f"Error formatting result {i}: {str(e)}")
                formatted += f"### Result {i}: Error formatting result\n\n"
        
        return formatted

    def _safe_llm_call(self, model: ChatOllama, prompt: str, purpose: str = "general", max_retries: int = 2) -> str:
        """Safely call the LLM model with error handling, logging, and retries."""
        last_error = None
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Starting LLM call for {purpose} (attempt {attempt + 1}/{max_retries})...")
                
                # Break prompt into chunks if it's too long
                if len(prompt) > 4000:  # Adjust this threshold based on your model's limits
                    logger.info(f"Large prompt detected ({len(prompt)} chars), breaking into chunks")
                    chunks = [prompt[i:i + 4000] for i in range(0, len(prompt), 4000)]
                    responses = []
                    
                    for i, chunk in enumerate(chunks):
                        logger.info(f"Processing chunk {i + 1}/{len(chunks)}")
                        chunk_response = model.invoke(chunk)
                        chunk_content = chunk_response.content if hasattr(chunk_response, 'content') else str(chunk_response)
                        responses.append(chunk_content.strip())
                    
                    content = " ".join(responses)
                else:
                    response = model.invoke(prompt)
                    content = response.content if hasattr(response, 'content') else str(response)
                
                if not content.strip():
                    raise ValueError(f"Empty response received from LLM during {purpose}")
                
                logger.info(f"Successfully completed LLM call for {purpose}")
                return content.strip()
                
            except Exception as e:
                last_error = e
                logger.warning(f"Error in LLM call for {purpose} (attempt {attempt + 1}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2)  # Wait before retrying
                continue
        
        # If we get here, all retries failed
        logger.error(f"All attempts failed for {purpose}: {str(last_error)}")
        raise last_error

    def _generate_response(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a final response based on the combined analysis."""
        try:
            # Get the combined analysis
            combined_analysis = state.get('combined_analysis', '')
            if not combined_analysis:
                return {
                    "final_response": "I couldn't find any relevant information to answer your question. "
                                    "Please try rephrasing your question or providing more specific details.",
                    "metadata": {
                        "error": "No search results found",
                        "timestamp": datetime.now().isoformat()
                    }
                }
            
            # Get the original question and question analysis
            messages = state.get('messages', [])
            if not messages:
                return {"final_response": "I couldn't process your question. Please try again."}
            
            question = messages[-1].content if isinstance(messages[-1], BaseMessage) else str(messages[-1])
            
            # First, determine the question type using the classifier method
            initial_type = self._classify_question_type(question)
            logger.info(f"Initial question type classification: {initial_type}")
            
            try:
                # Get question type using LLM with retries
                question_type = self._safe_llm_call(
                    self.question_model,
                    self._get_question_type_prompt(question, initial_type),
                    "question classification",
                    max_retries=2
                )
                
                # Validate question type
                valid_types = ["MODEL_EXPLANATION", "IMPLEMENTATION_GUIDE", "ENHANCEMENT", "TROUBLESHOOTING", "GENERAL"]
                if question_type not in valid_types:
                    logger.warning(f"Invalid question type '{question_type}', falling back to initial classification")
                    question_type = initial_type
                    
            except Exception as e:
                logger.warning(f"Error in LLM classification: {str(e)}, falling back to initial classification")
                question_type = initial_type
            
            logger.info(f"Final question type determination: {question_type}")
            
            # Get the appropriate instruction template
            instructions = self._get_instructions_for_type(question_type, question)
            
            try:
                # Generate response using safe LLM call with retries
                response_content = self._safe_llm_call(
                    self.response_model,
                    self._get_response_prompt(question, question_type, combined_analysis, instructions),
                    "response generation",
                    max_retries=2
                )
                
                # Log response preview
                preview = response_content[:200] + "..." if len(response_content) > 200 else response_content
                logger.info(f"Successfully generated response. Preview: {preview}")
                
                return {
                    "final_response": response_content,
                    "metadata": self._create_response_metadata(state, initial_type, question_type)
                }
                
            except Exception as response_error:
                logger.error(f"Error in response generation: {str(response_error)}")
                return self._create_error_response(state, question_type, str(response_error))
            
        except Exception as e:
            logger.error(f"Error in response generation process: {str(e)}", exc_info=True)
            return {
                "final_response": f"I apologize, but I encountered an error while processing your question: {str(e)}",
                "metadata": {
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
            }

    def _get_instructions_for_type(self, question_type: str, question: str) -> str:
        """Get the appropriate instruction template based on question type."""
        if question_type == "MODEL_EXPLANATION":
            return self._get_model_explanation_instructions(question)
        elif question_type == "IMPLEMENTATION_GUIDE":
            return self._get_implementation_instructions(question)
        elif question_type == "ENHANCEMENT":
            return self._get_enhancement_instructions(question)
        elif question_type == "TROUBLESHOOTING":
            return self._get_troubleshooting_instructions(question)
        else:
            return self._get_general_instructions(question)

    def _create_response_metadata(self, state: Dict[str, Any], initial_type: str, question_type: str) -> Dict[str, Any]:
        """Create metadata for the response."""
        return {
            "question_type": question_type,
            "timestamp": datetime.now().isoformat(),
            "sources": {
                "github_results": bool(state.get('github_results')),
                "sql_results": bool(state.get('sql_results')),
                "doc_results": bool(state.get('doc_results')),
                "dbt_results": bool(state.get('dbt_results')),
                "relationship_results": bool(state.get('relationship_results'))
            },
            "processing_info": {
                "initial_classification": initial_type,
                "final_classification": question_type
            }
        }

    def _create_error_response(self, state: Dict[str, Any], question_type: str, error: str) -> Dict[str, Any]:
        """Create an error response with available information."""
        error_response = f"""I apologize, but I encountered an issue while generating the response. 
Here's what I found in the search results:

1. Available Information:
- GitHub Results: {bool(state.get('github_results'))}
- SQL Results: {bool(state.get('sql_results'))}
- DBT Results: {bool(state.get('dbt_results'))}
- Documentation: {bool(state.get('doc_results'))}

2. Question Type: {question_type}

Please try rephrasing your question or providing more specific details."""
        
        return {
            "final_response": error_response,
            "metadata": {
                "error": error,
                "question_type": question_type,
                "timestamp": datetime.now().isoformat()
            }
        }

    def _classify_question_type(self, query: str) -> str:
        """Classify the question type to provide a more tailored response."""
        query_lower = query.lower()
        
        # Model explanation patterns
        model_patterns = [
            "what is", "explain", "describe", "understand", "tell me about", 
            "how does", "purpose of", "definition of", "meaning of",
            "structure of", "fields in", "columns in"
        ]
        
        # Implementation guide patterns
        implementation_patterns = [
            "how to", "how do i", "steps to", "implement", "create", "build",
            "develop", "code for", "example of", "sample of", "template for",
            "pattern for", "approach for", "methodology for", "best way to",
            "how should i", "guidance on", "instructions for"
        ]
        
        # Enhancement patterns
        enhancement_patterns = [
            "improve", "optimize", "enhance", "better", "upgrade",
            "refactor", "modernize", "update", "add feature", "extend",
            "add capability", "add functionality", "make it better",
            "performance improvement", "add new", "extend with",
            "should i add", "how can i add"
        ]
        
        # Troubleshooting patterns
        troubleshooting_patterns = [
            "fix", "issue", "error", "problem", "debug", "troubleshoot",
            "not working", "fails", "broken", "wrong", "incorrect",
            "unexpected", "help with", "resolve", "solution for"
        ]
        
        # Check for each pattern type
        for pattern in model_patterns:
            if pattern in query_lower:
                return "MODEL_EXPLANATION"
                
        for pattern in implementation_patterns:
            if pattern in query_lower:
                return "IMPLEMENTATION_GUIDE"
                
        for pattern in enhancement_patterns:
            if pattern in query_lower:
                return "ENHANCEMENT"
                
        for pattern in troubleshooting_patterns:
            if pattern in query_lower:
                return "TROUBLESHOOTING"
        
        # Default to general explanation
        return "GENERAL"

    def _get_model_explanation_instructions(self, query: str) -> str:
        """Get instructions for DBT model explanation questions."""
        return """
        IMPORTANT: Provide a DBT-specific model explanation based ONLY on the code found in the repository.
        Maximum response length: 500 words.
        
        TASK:
        1. Analyze DBT model structure and dependencies
        2. Explain key transformations and business logic
        3. Document model configurations and tests
        4. Show critical model relationships
        
        FORMAT:
        
        ## Model Overview
        [One paragraph summary of the DBT model's purpose - max 50 words]
        
        ## Model Structure
        
        ### 1. Source Configuration
        ```yaml
        # models/schema.yml
        version: 2
        
        sources:
          [Relevant source definition]
        ```
        
        ### 2. Base Model Definition
        ```sql
        -- models/staging/[model_name].sql
        [Key SQL transformations]
        ```
        
        ### 3. Model Configuration
        ```yaml
        # models/[model_name].yml
        version: 2
        
        models:
          [Model configuration details]
        ```
        
        ## Key Transformations
        1. [Main transformation step]
        2. [Additional transformation step]
        3. [Final transformation step]
        
        ## Dependencies
        ```mermaid
        graph TD
          [Dependency graph showing upstream/downstream models]
        ```
        
        ## Testing & Documentation
        ```yaml
        version: 2
        
        models:
          - name: [model_name]
            description: [model description]
            tests:
              [Existing test configurations]
        ```
        
        ## Usage Examples
        ```sql
        -- Example of how this model is used
        [SQL example]
        ```
        
        Note:
        - All explanations must reference actual code in the repository
        - Include only existing configurations and tests
        - Show real dependencies from DBT manifest
        - If information isn't in the codebase, state "Not found in repository"
        """

    def _get_implementation_instructions(self, query: str) -> str:
        """Get instructions for DBT implementation guide questions."""
        return """
        IMPORTANT: Provide DBT-specific implementation with code examples for each step.
        Focus on practical model development and transformations.
        
        TASK:
        1. Show complete DBT model development
        2. Include all necessary configurations
        3. Show testing and documentation
        4. Provide final working model
        
        FORMAT:
        
        ## Model Development Steps
        
        ### Step 1: Define Source Configuration
        ```yaml
        # models/schema.yml
        [Source configuration code]
        ```
        
        ### Step 2: Create Base Model
        ```sql
        -- models/staging/[model_name].sql
        [Base model SQL transformation]
        ```
        
        ### Step 3: Add Model Configuration
        ```yaml
        # models/[model_name].yml
        [Model configuration with schema and tests]
        ```
        
        ### Step 4: Create Intermediate/Final Model
        ```sql
        -- models/marts/[model_name].sql
        [Final transformation logic]
        ```
        
        ## Complete Implementation
        
        ### Source Configuration
        ```yaml
        version: 2
        
        sources:
          [Complete source definition]
        ```
        
        ### Model SQL
        ```sql
        [Complete model SQL code]
        ```
        
        ### Model Configuration
        ```yaml
        version: 2
        
        models:
          [Complete model configuration]
        ```
        
        ### Model Tests
        ```yaml
        [Complete test configuration]
        ```
        
        ## Required DBT Configurations
        - [DBT project configuration requirement]
        - [Package dependencies]
        - [Database connection]
        
        Note:
        - All SQL must follow DBT best practices
        - Include only code found in search results
        - Each code block must be complete and properly referenced
        - Models must include source references and documentation
        - If specific code is not found, state "DBT code not found in provided results"
        """

    def _get_enhancement_instructions(self, query: str) -> str:
        """Get instructions for DBT model enhancement questions."""
        return """
        IMPORTANT: Provide DBT-specific enhancements based ONLY on the existing code. Focus on practical model improvements.
        Maximum response length: 600 words.
        
        TASK:
        1. Review existing DBT model structure
        2. Identify key enhancement areas
        3. Provide step-by-step implementation
        4. Include complete code blocks
        
        FORMAT:
        
        ## Current Model Analysis
        [Brief analysis of existing model - max 50 words]
        
        ## Enhancement Steps
        
        ### 1. Source Configuration Updates
        ```yaml
        # models/schema.yml
        version: 2
        
        sources:
          [Enhanced source configuration]
        ```
        
        ### 2. Base Model Improvements
        ```sql
        -- models/staging/[model_name].sql
        [Enhanced base model SQL]
        ```
        
        ### 3. Model Configuration Enhancements
        ```yaml
        # models/[model_name].yml
        version: 2
        
        models:
          [Enhanced model configuration with tests]
        ```
        
        ### 4. Intermediate/Final Model Optimizations
        ```sql
        -- models/marts/[model_name].sql
        [Enhanced transformation logic]
        ```
        
        ## Complete Enhanced Implementation
        
        ### Updated Source Configuration
        ```yaml
        version: 2
        
        sources:
          [Complete enhanced source definition]
        ```
        
        ### Enhanced Model SQL
        ```sql
        [Complete enhanced model SQL]
        ```
        
        ### Updated Model Configuration
        ```yaml
        version: 2
        
        models:
          [Complete enhanced model configuration]
        ```
        
        ### Additional Tests
        ```yaml
        version: 2
        
        models:
          - name: [model_name]
            tests:
              [Enhanced test configurations]
        ```
        
        ## Implementation Requirements
        1. [Required DBT package/version]
        2. [Database dependency]
        3. [Other critical requirements]
        
        ## Validation Steps
        1. [How to test the enhancement]
        2. [What to verify]
        3. [Expected outcome]
        
        Note:
        - All enhancements must follow DBT best practices
        - Include only modifications found in search results
        - Each code block must be complete and properly referenced
        - All SQL must be properly formatted and documented
        - If specific enhancement details aren't found, state "Enhancement details not found in provided code"
        """

    def _get_troubleshooting_instructions(self, query: str) -> str:
        """Get instructions for DBT troubleshooting questions."""
        return """
        IMPORTANT: Provide DBT-specific troubleshooting guidance based ONLY on the code in the repository.
        Maximum response length: 400 words.
        
        TASK:
        1. Identify specific DBT model issues
        2. Check configurations and dependencies
        3. Verify SQL transformations
        4. Validate tests and documentation
        
        FORMAT:
        
        ## Issue Analysis
        [Brief description of the DBT-specific issue - max 50 words]
        
        ## Diagnostic Steps
        
        ### 1. Configuration Check
        ```yaml
        # dbt_project.yml
        [Relevant project configuration]
        ```
        
        ### 2. Model Validation
        ```sql
        -- models/[model_path].sql
        [Problematic SQL code]
        ```
        
        ### 3. Dependency Verification
        ```yaml
        # models/schema.yml
        version: 2
        
        models:
          [Relevant model dependencies]
        ```
        
        ## Solution Steps
        
        ### 1. Fix Configuration
        ```yaml
        # Updated configuration
        [Corrected YAML]
        ```
        
        ### 2. Update Model
        ```sql
        -- Corrected SQL
        [Fixed transformation logic]
        ```
        
        ### 3. Add Tests
        ```yaml
        version: 2
        
        models:
          - name: [model_name]
            tests:
              [Additional test coverage]
        ```
        
        ## Verification Process
        1. [DBT command to test fix]
        2. [Expected output]
        3. [Validation query]
        
        ## Prevention Steps
        1. [Recommended test to add]
        2. [Documentation update]
        3. [Best practice to implement]
        
        Note:
        - All solutions must reference actual DBT code
        - Include specific DBT commands to verify fixes
        - Show exact file paths from repository
        - If specific details aren't found, state "Not found in repository"
        """

    def _get_general_instructions(self, query: str) -> str:
        """Get instructions for general questions."""
        return """
        IMPORTANT: Answer based ONLY on the code provided. Keep responses direct and concise.
        Maximum response length: 300 words.
        
        TASK:
        1. Answer the core question directly
        2. Provide relevant code snippet
        3. Give practical context
        4. Note key considerations
        
        FORMAT (keep sections brief):
        
        ## Answer
        [Direct response in 1-2 sentences]
        
        ## Code Context
        ```
        [Most relevant code snippet]
        ```
        [One sentence explaining the code]
        
        ## Key Points
        - [Important point 1]
        - [Important point 2]
        - [Important point 3]
        (Maximum 3 points)
        
        ## Next Steps
        [Optional: 1-2 practical next steps if relevant]
        
        Note: If information isn't in the code, state "Not specified in provided code" rather than making assumptions.
        """

    def _get_question_type_prompt(self, question: str, initial_type: str) -> str:
        """Generate the prompt for question type classification."""
        return f"""
You are a data architecture expert. Analyze the following question and determine the most appropriate response type.

QUESTION:
{question}

AVAILABLE CATEGORIES:
1. MODEL_EXPLANATION - Questions about understanding data models, schemas, structures, or relationships
2. IMPLEMENTATION_GUIDE - Questions about how to implement, create, or set up something
3. ENHANCEMENT - Questions about improving, optimizing, or extending existing functionality
4. TROUBLESHOOTING - Questions about fixing issues, errors, or problems
5. GENERAL - Other types of questions that don't fit above categories

Initial classification: {initial_type}

Respond with ONLY ONE of these exact category names:
MODEL_EXPLANATION
IMPLEMENTATION_GUIDE
ENHANCEMENT
TROUBLESHOOTING
GENERAL
"""

    def _get_response_prompt(self, question: str, question_type: str, combined_analysis: str, instructions: str) -> str:
        """Generate the main response prompt."""
        return f"""
You are an expert data architect assistant. Answer the following question based ONLY on the search results provided.
Do not make up information or reference things not in the search results.

ORIGINAL QUESTION:
{question}

QUESTION TYPE:
{question_type}

SEARCH RESULTS SUMMARY:
{combined_analysis[:4000]}...

THINKING STEPS:
1. ANALYZE AVAILABLE INFORMATION
   - Review all search results carefully
   - Identify relevant code snippets, schemas, and documentation
   - Note any relationships or dependencies mentioned

2. VALIDATE INFORMATION SOURCES
   - Confirm all information comes from the search results
   - Check file paths and references are accurate
   - Verify code examples are complete and relevant

3. ORGANIZE RESPONSE STRUCTURE
   - Follow the template format strictly
   - Include all required sections
   - Ensure logical flow of information

4. PROVIDE SPECIFIC DETAILS
   - Include exact file paths when referencing code
   - Quote relevant code snippets directly from results
   - Reference specific models, tables, or schemas mentioned

5. VERIFY COMPLETENESS
   - Ensure all parts of the question are addressed
   - Check that response follows template format
   - Confirm all statements are supported by search results

RESPONSE INSTRUCTIONS:
{instructions}

IMPORTANT GUIDELINES:
- Use ONLY information from the provided search results
- Include specific file paths and code references
- Format code examples with proper syntax highlighting
- Maintain clear section headers and structure
- If certain information is not found in results, explicitly state that
"""

    def _enhance_response(self, response: str, question_type: str, github_results: List[Dict[str, Any]], sql_results: List[Dict[str, Any]], doc_results: List[Dict[str, Any]], dbt_results: List[Dict[str, Any]], relationship_results: List[Dict[str, Any]]) -> str:
        """
        Enhance the response with file references, more code examples,
        and tailored information based on the question type.
        """
        # If the response already contains good structure, return it as is
        if "##" in response:
            return response
            
        # Start with the original response
        enhanced_response = response
        
        # Add a section header if none exists
        if not enhanced_response.startswith("#"):
            enhanced_response = f"## Response\n\n{enhanced_response}"
        
        # Add code file references if not already included
        if "File:" not in enhanced_response and github_results:
            file_references = "\n\n## Relevant Files\n\n"
            for result in github_results:
                file_path = result.get('file_path', 'Unknown file')
                repo_info = result.get('repo_info', {})
                repo_name = repo_info.get('name', 'Unknown repository')
                file_references += f"- **{file_path}** in repository {repo_name}\n"
            
            enhanced_response += file_references
        
        # Add code examples if not already included and if it's an implementation question
        if question_type == "IMPLEMENTATION_GUIDE" and "```" not in enhanced_response and github_results:
            code_examples = "\n\n## Code Examples\n\n"
            for result in github_results:
                file_path = result.get('file_path', 'Unknown file')
                code_snippet = result.get('code_snippet', 'No code available')
                if code_snippet and len(code_snippet.strip()) > 0:
                    code_examples += f"**From {file_path}:**\n\n```\n{code_snippet}\n```\n\n"
            
            enhanced_response += code_examples
        
        # Add a next steps section for enhancement questions
        if question_type == "ENHANCEMENT" and "Next Steps" not in enhanced_response:
            enhanced_response += "\n\n## Next Steps\n\n"
            enhanced_response += "1. Review the existing code structure thoroughly\n"
            enhanced_response += "2. Create a development branch for your enhancements\n"
            enhanced_response += "3. Implement the suggested changes incrementally\n"
            enhanced_response += "4. Add appropriate tests for new functionality\n"
            enhanced_response += "5. Document your changes in the model's YAML schema file\n"
        
        return enhanced_response

    def process_question(self, question: str, conversation_id: str = None, thread_id: str = None, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process a question and generate a response.
        
        Args:
            question: The question to process
            conversation_id: Optional conversation ID
            thread_id: Optional thread ID
            metadata: Optional metadata
            
        Returns:
            Dictionary with response and metadata
        """
        try:
            logger.info(f"Processing question: {question}...")
            
            # Generate a conversation ID if not provided
            if not conversation_id:
                conversation_id = str(uuid.uuid4())
            
            # Use conversation_id as thread_id if not provided
            if not thread_id:
                thread_id = conversation_id
            
            # Start timing
            start_time = time.time()
            
            # Create a human message from the question
            message = HumanMessage(content=question)
            
            # Initialize the state with the message
            state = {"messages": [message]}
            
            # Run the agent graph
            result = self.graph.invoke(state, {"configurable": {"thread_id": conversation_id}})
            
            # Extract the final response
            response = result.get('final_response', "I couldn't generate a response.")
            
            # Calculate processing time
            processing_time = time.time() - start_time
            logger.info(f"Question processed in {processing_time:.2f} seconds")
            
            # Save the conversation if a database is provided
            if self.db:
                try:
                    self.db.save_conversation({
                        "conversation_id": conversation_id,
                        "thread_id": thread_id,
                        "question": question,
                        "answer": response,
                        "metadata": {
                            "timestamp": datetime.now().isoformat(),
                            "processing_time": processing_time,
                            "question_type": result.get('question_type', 'UNKNOWN'),
                            "user_metadata": metadata
                        }
                    })
                except Exception as e:
                    logger.error(f"Error saving conversation: {str(e)}")
            
            # Return the response
            return {
                "response": response,
                "conversation_id": conversation_id,
                "thread_id": thread_id,
                "processing_time": processing_time,
                "question_type": result.get('question_type', 'UNKNOWN'),
                "github_results": result.get('github_results', {}),
                "sql_results": result.get('sql_results', {}),
                "doc_results": result.get('doc_results', {}),
                "dbt_results": result.get('dbt_results', {}),
                "relationship_results": result.get('relationship_results', {})
            }
            
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            return {
                "response": f"I'm sorry, I encountered an error while processing your question: {str(e)}",
                "conversation_id": conversation_id,
                "thread_id": thread_id,
                "error": str(e)
            }

def create_data_architect_agent(tools: SearchTools):
    """
    Factory function to create a Data Architect Agent.
    
    Args:
        tools: SearchTools instance for searching various data sources
        
    Returns:
        DataArchitectAgent instance
    """
    return DataArchitectAgent(tools) 