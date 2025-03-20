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
        
        # Initialize models
        self.question_model = ChatOllama(
            model="gemma3:latest",
            temperature=0,
            base_url="http://localhost:11434",
            timeout=60,
        )
        
        self.analysis_model = ChatOllama(
            model="gemma3:latest",
            temperature=0,
            base_url="http://localhost:11434",
            timeout=120,
        )
        
        self.response_model = ChatOllama(
            model="gemma3:latest",
            temperature=0.2,
            base_url="http://localhost:11434",
            timeout=180,
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
            logger.info(f"Combining results: GitHub={len(github_results)}, SQL={len(sql_results)}, Docs={len(doc_results)}, DBT={len(dbt_results)}, Relationships={len(relationship_results)}")
            
            # Format the results for the LLM
            formatted_github = self._format_results_for_llm(github_results, "GitHub")
            formatted_sql = self._format_results_for_llm(sql_results, "SQL")
            formatted_docs = self._format_results_for_llm(doc_results, "Documentation")
            formatted_dbt = self._format_results_for_llm(dbt_results, "DBT")
            formatted_relationships = self._format_results_for_llm(relationship_results, "Relationships")
            
            # Create a combined analysis prompt
            combined_analysis = f"""
# Search Results Analysis

## GitHub Code Results
{formatted_github}

## SQL Query Results
{formatted_sql}

## Documentation Results
{formatted_docs}

## DBT Model Results
{formatted_dbt}

## Relationship Results
{formatted_relationships}
"""
            
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

    def _generate_response(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a final response based on the combined analysis."""
        try:
            # Get the combined analysis
            combined_analysis = state.get('combined_analysis', '')
            if not combined_analysis:
                return {"final_response": "I couldn't find any relevant information to answer your question."}
            
            # Get the original question
            messages = state.get('messages', [])
            if not messages:
                return {"final_response": "I couldn't process your question. Please try again."}
            
            question = messages[-1].content if isinstance(messages[-1], BaseMessage) else str(messages[-1])
            
            # Create a prompt for the response generation
            prompt = f"""
You are a helpful data architect assistant. Answer the following question based ONLY on the search results provided.
Do not make up information or reference things not in the search results.

QUESTION:
{question}

SEARCH RESULTS:
{combined_analysis}

INSTRUCTIONS:
1. Provide a direct answer to the question based only on the search results
2. Include relevant code examples from the search results
3. Explain any technical concepts mentioned
4. Format your response with markdown headings and code blocks

Your response should be comprehensive but focused on answering the specific question.
"""
            
            # Generate the response (without timeout parameter)
            logger.info("Generating response with LLM...")
            response = self.response_model.invoke(prompt)
            
            # Extract content from response
            response_content = response.content if hasattr(response, 'content') else str(response)
            
            # Log a preview of the response
            preview = response_content[:100] + "..." if len(response_content) > 100 else response_content
            logger.info(f"Generated response preview: {preview}")
            
            # Return the final response
            return {"final_response": response_content}
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}", exc_info=True)
            return {"final_response": f"I'm sorry, I encountered an error while generating a response: {str(e)}"}

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
        """Get instructions for model explanation questions."""
        return """
        IMPORTANT: Your response should explain the data model based ONLY on the actual code provided above. 
        Focus on clear explanations of structure, relationships, and purpose.
        
        TASK:
        1. Explain what the model is based ONLY on the provided code
        2. Describe its structure, fields, and relationships as shown in the code
        3. Explain the business purpose of the model
        4. Explain how it relates to other models and sources
        5. Provide a summary of the technical implementation shown in the code
        
        FORMAT YOUR RESPONSE AS FOLLOWS:
        
        ## Model Overview
        [Provide a concise overview of what this model represents in the data architecture]
        
        ## Structure and Fields
        [List and explain ONLY the fields/columns actually present in the code]
        
        ## Business Purpose
        [Explain the business purpose of this model]
        
        ## Related Models and Sources
        [Explain how this model relates to other models/sources mentioned in the code]
        
        ## Technical Implementation
        [Explain how the model is technically implemented, including ETL logic]
        
        Include file paths and code references where relevant.
        """

    def _get_implementation_instructions(self, query: str) -> str:
        """Get instructions for implementation guide questions."""
        return """
        IMPORTANT: Your response should provide a practical implementation guide based ONLY on the actual code provided above.
        Focus on concrete steps, code examples, and best practices.
        
        TASK:
        1. Provide a step-by-step implementation guide based on the code examples
        2. Include specific code examples showing how to implement the required functionality
        3. Explain any configuration or setup required
        4. Highlight key considerations and potential challenges
        5. Suggest testing strategies for the implementation
        
        FORMAT YOUR RESPONSE AS FOLLOWS:
        
        ## Implementation Steps
        [Provide a numbered list of implementation steps]
        
        ## Code Examples
        [Provide specific code examples for each step, using the actual code from the search results]
        
        ## Configuration Required
        [Explain any configuration or setup required]
        
        ## Key Considerations
        [Highlight important considerations and potential challenges]
        
        ## Testing Strategy
        [Suggest how to test the implementation]
        
        Include file paths and code references where relevant.
        """

    def _get_enhancement_instructions(self, query: str) -> str:
        """Get instructions for enhancement questions."""
        return """
        IMPORTANT: Your response should suggest enhancements based ONLY on the actual code provided above.
        Focus on practical improvements that build on the existing code.
        
        TASK:
        1. Suggest specific enhancements based on the existing code
        2. Provide sample code for each enhancement
        3. Explain the benefits of each enhancement
        4. Outline implementation steps for each enhancement
        5. Highlight any dependencies or prerequisites
        
        FORMAT YOUR RESPONSE AS FOLLOWS:
        
        ## Suggested Enhancements
        [List suggested enhancements based on the existing code]
        
        ## Enhancement Details
        [For each enhancement, provide:
        1. Description of the enhancement
        2. Sample code showing how to implement it
        3. Benefits of the enhancement
        4. Implementation steps]
        
        ## Implementation Approach
        [Suggest an overall approach for implementing these enhancements]
        
        ## Prerequisites
        [List any dependencies or prerequisites for these enhancements]
        
        Include file paths and code references where relevant.
        """

    def _get_troubleshooting_instructions(self, query: str) -> str:
        """Get instructions for troubleshooting questions."""
        return """
        IMPORTANT: Your response should provide troubleshooting guidance based ONLY on the actual code provided above.
        Focus on identifying potential issues and practical solutions.
        
        TASK:
        1. Identify potential issues in the code that could cause problems
        2. Suggest specific fixes for each issue
        3. Provide diagnostic steps to identify the root cause
        4. Offer code examples for solutions
        5. Suggest preventative measures for the future
        
        FORMAT YOUR RESPONSE AS FOLLOWS:
        
        ## Potential Issues
        [List potential issues identified in the code]
        
        ## Diagnostic Steps
        [Provide steps to diagnose the issues]
        
        ## Recommended Solutions
        [For each issue, provide:
        1. Explanation of the solution
        2. Code example showing the fix
        3. How to verify the fix worked]
        
        ## Preventative Measures
        [Suggest ways to prevent similar issues in the future]
        
        Include file paths and code references where relevant.
        """

    def _get_general_instructions(self, query: str) -> str:
        """Get instructions for general questions."""
        return """
        IMPORTANT: Your response should be based ONLY on the actual code provided above. 
        Do not make assumptions about what a typical implementation might contain if it's not shown in the code.
        
        TASK:
        1. Address the user's question as specifically as possible using the provided code
        2. Provide relevant code examples from the search results
        3. Explain concepts clearly and concisely
        4. Include practical advice based on the code
        5. Highlight relevant file locations and code snippets
        
        FORMAT YOUR RESPONSE AS FOLLOWS:
        
        ## Answer
        [Provide a direct answer to the user's question]
        
        ## Code Examples
        [Include relevant code examples from the search results]
        
        ## Explanation
        [Provide a clear explanation of concepts mentioned]
        
        ## Practical Advice
        [Offer practical advice based on the code]
        
        Include file paths and code references where relevant.
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