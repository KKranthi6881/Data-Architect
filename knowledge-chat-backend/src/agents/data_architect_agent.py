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

from src.tools import SearchTools
from src.db.database import ChatDatabase
from src.agents.data_architect.github_search import GitHubCodeSearchAgent
from src.agents.data_architect.data_architect import DataArchitectAgent
from src.agents.data_architect.question_parser import QuestionParserSystem
from src.utils import ChromaDBManager

# Set up logger
logger = logging.getLogger(__name__)

# Define state type
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "The messages in the conversation"]
    github_context: Annotated[Dict, "GitHub code search results"]
    doc_context: Annotated[Dict, "Documentation search results"]
    code_context: Annotated[Dict, "Code search results"]
    github_analysis: Annotated[str, "GitHub code analysis output"]
    doc_analysis: Annotated[str, "Documentation analysis output"]
    code_analysis: Annotated[str, "Code analysis output"]
    combined_output: Annotated[str, "Combined analysis output"]
    final_summary: Annotated[str, "User-friendly final summary"]
    question_analysis: Annotated[Dict, "Parsed question analysis"]

# Define structured outputs
class GitHubAnalysis(BaseModel):
    dbt_models: Dict[str, Any] = Field(
        description="Dictionary of dbt models and their details"
    )
    schema_definitions: Union[Dict[str, Any], List[Any]] = Field(
        description="Dictionary or list of schema definitions"
    )
    lineage_info: Union[Dict[str, Any], List[Any]] = Field(
        description="Dictionary or list of lineage information"
    )
    code_details: Dict[str, Any] = Field(
        description="Technical implementation details"
    )

class DocAnalysis(BaseModel):
    key_concepts: List[str] = Field(
        description="Key concepts found in documentation"
    )
    workflows: List[str] = Field(
        description="Business workflows described"
    )
    requirements: str = Field(
        description="Business requirements identified"
    )
    additional_context: str = Field(
        description="Additional contextual information"
    )

class CodeAnalysis(BaseModel):
    tables_and_columns: Dict[str, List[str]] = Field(
        description="Dictionary of table names and their columns"
    )
    relationships: List[str] = Field(
        description="List of relationships between tables"
    )
    business_logic: str = Field(
        description="Description of the business logic implemented"
    )
    technical_details: str = Field(
        description="Technical implementation details"
    )

class FinalSummary(BaseModel):
    overview: str = Field(
        description="High-level overview of the analyzed system",
        default=""
    )
    data_model: Dict[str, Any] = Field(
        description="Data model with tables, schemas and relationships",
        default_factory=dict
    )
    dbt_architecture: Dict[str, Any] = Field(
        description="dbt project architecture and key components",
        default_factory=dict
    )
    data_lineage: Union[Dict[str, Any], List[str]] = Field(
        description="Data lineage information",
        default_factory=list
    )
    implementation_details: List[Union[str, Dict[str, Any]]] = Field(
        description="Important technical implementation details",
        default_factory=list
    )
    recommendations: List[Union[str, Dict[str, Any]]] = Field(
        description="Suggested considerations or improvements",
        default_factory=list
    )
    
    class Config:
        extra = "allow"  # Allow extra fields in the response

def create_data_architect_agent(tools: SearchTools):
    # Initialize models
    github_model = ChatOllama(
        model="gemma3:latest ",
        temperature=0,
        base_url="http://localhost:11434",
        timeout=180,
    )
    
    code_model = ChatOllama(
        model="gemma3:latest ",
        temperature=0,
        base_url="http://localhost:11434",
        timeout=120,
    )
    
    doc_model = ChatOllama(
        model="gemma3:latest ",
        temperature=0,
        base_url="http://localhost:11434",
        timeout=120,
    )

    # Create parsers
    github_parser = PydanticOutputParser(pydantic_object=GitHubAnalysis)
    code_parser = PydanticOutputParser(pydantic_object=CodeAnalysis)
    doc_parser = PydanticOutputParser(pydantic_object=DocAnalysis)

    # Initialize specialized agents
    github_search_agent = GitHubCodeSearchAgent()
    question_parser = QuestionParserSystem(tools)
    architect_agent = DataArchitectAgent()

    # Create prompt templates
    github_template = """
    Analyze the following dbt code repositories and provide structured information about them.
    
    CODE TO ANALYZE:
    {github_context}
    
    USER QUESTION:
    {user_question}
    
    Guidelines:
    - Focus on dbt models, schema definitions, and lineage information
    - Identify data models and their relationships
    - Extract schema definitions from YAML files
    - Identify upstream and downstream dependencies
    - Analyze materialization strategies
    - Extract business logic and transformations
    
    Your response MUST be in the following JSON format:
    {format_instructions}
    
    Make sure to include:
    1. All dbt models and their details in the dbt_models field
    2. All schema definitions in the schema_definitions field
    3. Lineage information in the lineage_info field
    4. Implementation details in the code_details field
    
    Response:
    """

    code_template = """
    Analyze the following code and provide structured information about it.
    
    CODE TO ANALYZE:
    {code_context}
    
    USER QUESTION:
    {user_question}
    
    Guidelines:
    - Focus on SQL and Python code structure
    - Identify tables, columns and their relationships
    - Explain technical implementation details
    - Describe the business logic 
    - Provide column level lineage relevant to the user question
    
    Your response MUST be in the following JSON format:
    {format_instructions}
    
    Make sure to include:
    1. All tables and their columns in the tables_and_columns field
    2. All relationships between tables in the relationships field
    3. Clear business logic description in the business_logic field
    4. Implementation details in the technical_details field
    
    Response:
    """

    doc_template = """
    Analyze the following documentation and provide structured information about it.
    
    DOCUMENTATION TO ANALYZE:
    {doc_context}
    
    USER QUESTION:
    {user_question}
    
    Guidelines:
    - Focus on business requirements and workflows
    - Identify key concepts and terminology
    - Extract business rules and processes
    - Note any important considerations
    
    Your response MUST be in the following JSON format:
    {format_instructions}
    
    Response:
    """

    github_prompt = PromptTemplate(
        template=github_template,
        input_variables=["github_context", "user_question"],
        partial_variables={"format_instructions": github_parser.get_format_instructions()}
    )

    code_prompt = PromptTemplate(
        template=code_template,
        input_variables=["code_context", "user_question"],
        partial_variables={"format_instructions": code_parser.get_format_instructions()}
    )

    doc_prompt = PromptTemplate(
        template=doc_template,
        input_variables=["doc_context", "user_question"],
        partial_variables={"format_instructions": doc_parser.get_format_instructions()}
    )

    # Add model for final summary
    summary_model = ChatOllama(
        model="gemma3:latest ",
        temperature=0.2,
        base_url="http://localhost:11434",
        timeout=180,
    )

    # Create parser for final summary
    summary_parser = PydanticOutputParser(pydantic_object=FinalSummary)

    # Create template for final summary
    summary_template = """
    You are an expert data architect specializing in dbt, SQL and Snowflake. Create a comprehensive data architecture analysis based on the following information.
    
    ANALYSIS TO SUMMARIZE:
    {combined_analysis}

    USER QUESTION:
    {user_question}
    
    Guidelines:
    - Create a detailed data architecture assessment
    - Focus on dbt models, schemas, and lineage
    - Clearly explain relationships between tables
    - Highlight key transformations and business logic
    - Provide data lineage information
    - Identify potential data quality issues
    - Suggest architectural improvements
    
    Your response MUST be in the following JSON format:
    {format_instructions}
    
    Response:
    """

    summary_prompt = PromptTemplate(
        template=summary_template,
        input_variables=["combined_analysis", "user_question"],
        partial_variables={"format_instructions": summary_parser.get_format_instructions()}
    )

    def parse_question(state: AgentState) -> Dict:
        """Parse the user question to understand its structure and requirements."""
        try:
            messages = state['messages']
            if not messages:
                return state
            
            query = messages[-1].content if isinstance(messages[-1], BaseMessage) else str(messages[-1])
            
            # Create a simplified question analysis without using QuestionParserSystem
            # This avoids the async/sync mismatch that's causing the error
            parsed_question = {
                "original_question": query,
                "rephrased_question": query,
                "question_type": "technical",  # Default to technical question type
                "key_points": [query],  # Use query as the key point
                "business_context": {
                    "domain": "Data Engineering",
                    "primary_objective": "Understand data structures and architecture",
                    "key_entities": query.split(),
                    "business_impact": "Improved data understanding"
                },
                "technical_context": {
                    "data_stack": ["dbt", "Snowflake", "SQL"],
                    "relevant_components": [],
                    "analysis": query
                }
            }
            
            return {
                "question_analysis": parsed_question
            }
        except Exception as e:
            logger.error(f"Error in question parsing: {str(e)}")
            return {
                "question_analysis": {
                    "original_question": query if 'query' in locals() else "",
                    "rephrased_question": query if 'query' in locals() else "",
                    "question_type": "unknown",
                    "key_points": [],
                    "business_context": {},
                    "technical_context": {}
                }
            }

    def process_github(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process GitHub repositories."""
        try:
            # Create GitHub search agent
            from src.agents.data_architect.github_search import GitHubCodeSearchAgent
            github_search_agent = GitHubCodeSearchAgent()
            
            messages = state['messages']
            question_analysis = state.get('question_analysis', {})
            if not messages:
                return {}
            
            query = messages[-1].content if isinstance(messages[-1], BaseMessage) else str(messages[-1])
            
            # Search GitHub repositories
            logger.info(f"Searching GitHub repositories for: {query[:100]}...")
            search_results = github_search_agent.search_code(
                parsed_question=question_analysis, 
                max_results=5
            )
            
            # Log detailed search results
            self._log_search_results("GitHub", search_results)
            
            # Format results
            github_context = {"query": query, "results": search_results}
            github_analysis = "GitHub analysis completed. Found {} results.".format(len(search_results))
            
            return {
                "github_context": github_context,
                "github_analysis": github_analysis
            }
            
        except Exception as e:
            logger.error(f"Error in GitHub processing: {str(e)}", exc_info=True)
            return {
                "github_analysis": f"Error during GitHub analysis: {str(e)}",
                "github_context": {"query": query if 'query' in locals() else "", "results": []}
            }

    def process_code(state: AgentState) -> Dict:
        """Process code search and analysis."""
        try:
            messages = state['messages']
            if not messages:
                return {}
            
            query = messages[-1].content if isinstance(messages[-1], BaseMessage) else str(messages[-1])
            
            # Use the tools object passed to the constructor
            from src.tools import SearchTools
            tools = SearchTools(ChromaDBManager())
            search_results = tools.search_code(query)
            
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
            code_context = {"query": query, "results": search_results.get('results', [])}
            code_analysis = "Code analysis completed. Found {} results.".format(
                len(search_results.get('results', []))
            )
            
            return {
                "code_context": code_context,
                "code_analysis": code_analysis
            }
            
        except Exception as e:
            logger.error(f"Error in code processing: {str(e)}", exc_info=True)
            return {
                "code_analysis": f"Error during code analysis: {str(e)}",
                "code_context": {"query": query if 'query' in locals() else "", "results": []}
            }

    def process_docs(state: AgentState) -> Dict:
        """Process documentation analysis."""
        try:
            messages = state['messages']
            if not messages:
                return state
            
            query = messages[-1].content if isinstance(messages[-1], BaseMessage) else str(messages[-1])
            
            # Search documentation
            search_results = tools.search_documentation(query)
            
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
                    "doc_context": {"query": query, "results": []},
                    "doc_analysis": "No relevant documentation found."
                }
            
            # Generate analysis
            formatted_prompt = doc_prompt.format(
                doc_context=doc_context,
                user_question=query
            )
            response = doc_model.invoke(formatted_prompt)
            response_text = response.content if isinstance(response, BaseMessage) else str(response)
            
            try:
                analysis = doc_parser.parse(response_text)
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
                "doc_context": {"query": query, "results": search_results.get('results', [])},
                "doc_analysis": output
            }
            
        except Exception as e:
            logger.error(f"Error in doc processing: {str(e)}")
            return {
                "doc_analysis": f"Error during documentation analysis: {str(e)}",
                "doc_context": {}
            }

    def process_dbt(state: AgentState) -> Dict:
        """Process dbt models."""
        try:
            messages = state['messages']
            if not messages:
                return {}
            
            query = messages[-1].content if isinstance(messages[-1], BaseMessage) else str(messages[-1])
            
            # Placeholder for dbt model processing
            return {
                "dbt_context": {"query": query, "results": []},
                "dbt_analysis": "DBT model analysis not implemented yet."
            }
            
        except Exception as e:
            logger.error(f"Error in dbt processing: {str(e)}", exc_info=True)
            return {
                "dbt_analysis": f"Error during dbt analysis: {str(e)}",
                "dbt_context": {"query": query if 'query' in locals() else "", "results": []}
            }

    def _log_search_results(self, search_type: str, results: List[Dict[str, Any]]) -> None:
        """Log detailed information about search results."""
        if not results:
            logger.info(f"{search_type} search returned no results")
            return
            
        logger.info(f"{search_type} search returned {len(results)} results")
        
        # Log details of each result (limited to first 3 for brevity)
        for i, result in enumerate(results[:3], 1):
            file_path = result.get('file_path', 'Unknown file')
            repo_info = result.get('repo_info', {})
            repo_name = repo_info.get('name', 'Unknown repository')
            
            logger.info(f"{search_type} result {i}: File={file_path}, Repo={repo_name}")
            
            # Log a preview of the code snippet
            code_snippet = result.get('code_snippet', '')
            if code_snippet:
                # Extract first few lines for preview
                lines = code_snippet.split('\n')
                preview_lines = lines[:3]  # First 3 lines
                preview = '\n'.join(preview_lines)
                preview += "..." if len(lines) > 3 else ""
                logger.info(f"Code preview: {preview}")
            
            # Add related files if available
            related_files = result.get('related_files', [])
            if related_files:
                related_paths = [f.get('file_path', 'Unknown') for f in related_files[:3]]
                logger.info(f"Related files: {', '.join(related_paths)}")
                
        # If there are more results than we showed in detail
        if len(results) > 3:
            logger.info(f"... and {len(results) - 3} more {search_type} results")
        
        logger.info(f"End of {search_type} search results log")

    def combine_results(state: AgentState) -> Dict:
        """Combine GitHub, code and documentation analysis results."""
        combined = f"""
        Data Architecture Analysis Results
        =================================

        GitHub dbt Code Analysis:
        -----------------------
        {state.get('github_analysis', 'No GitHub analysis available')}

        Code Analysis:
        -------------
        {state.get('code_analysis', 'No code analysis available')}

        Documentation Analysis:
        ---------------------
        {state.get('doc_analysis', 'No documentation analysis available')}
        """
        
        return {"combined_output": combined}

    def create_final_summary(state: AgentState) -> Dict:
        """Create a data architect summary of the analysis."""
        try:
            combined_analysis = state.get('combined_output', '')
            messages = state['messages']
            if not messages or not combined_analysis:
                return {"final_summary": "No analysis available to summarize"}

            query = messages[-1].content if isinstance(messages[-1], BaseMessage) else str(messages[-1])
            
            # Generate summary
            formatted_prompt = summary_prompt.format(
                combined_analysis=combined_analysis,
                user_question=query
            )
            response = summary_model.invoke(formatted_prompt)
            response_text = response.content if isinstance(response, BaseMessage) else str(response)

            try:
                # Parse and validate the summary
                summary = summary_parser.parse(response_text)
                
                # Use the DataArchitectAgent to generate the final response
                github_results = state.get('github_context', {}).get('results', [])
                code_results = state.get('code_context', {}).get('results', [])
                question_analysis = state.get('question_analysis', {})
                
                architect_response = architect_agent.generate_response(
                    parsed_question=question_analysis,
                    schema_results=code_results, 
                    code_results=github_results,
                    original_question=query
                )
                
                # Format the final output
                output = f"""
                {architect_response.get('response', 'No architect response available')}

                Data Architecture Analysis
                ========================
                
                Overview:
                {summary.overview}
                
                Data Model:
                {format_data_model(summary.data_model)}
                
                dbt Architecture:
                {format_dbt_architecture(summary.dbt_architecture)}
                
                Data Lineage:
                {format_data_lineage(summary.data_lineage)}
                
                Implementation Details:
                {format_list(summary.implementation_details)}
                
                Recommendations:
                {format_list(summary.recommendations)}
                """

            except Exception as parse_error:
                logger.warning(f"Failed to parse summary: {str(parse_error)}")
                output = response_text

            return {"final_summary": output}

        except Exception as e:
            logger.error(f"Error in summary creation: {str(e)}")
            return {"final_summary": f"Error creating summary: {str(e)}"}

    def format_data_model(data_model: Dict) -> str:
        """Format data model information in a readable way."""
        if not data_model:
            return "No data model information available."
            
        output = []
        
        # Format tables and their columns
        if isinstance(data_model, dict):
            for table_name, details in data_model.items():
                if isinstance(details, list):
                    # If it's a simple list of columns
                    columns = ", ".join(details)
                    output.append(f"• {table_name}:\n  Columns: {columns}")
                elif isinstance(details, dict):
                    # If it's a detailed table description
                    output.append(f"• {table_name}:")
                    if 'columns' in details:
                        cols = ", ".join(details['columns'])
                        output.append(f"  Columns: {cols}")
                    if 'description' in details:
                        output.append(f"  Purpose: {details['description']}")
                    if 'relationships' in details:
                        rels = ", ".join(details['relationships'])
                        output.append(f"  Relationships: {rels}")
        
        return "\n".join(output) if output else "No structured data model available."

    def format_dbt_architecture(dbt_architecture: Dict) -> str:
        """Format dbt architecture information in a readable way."""
        if not dbt_architecture:
            return "No dbt architecture information available."
            
        output = []
        
        if isinstance(dbt_architecture, dict):
            # Add models information
            if 'models' in dbt_architecture:
                output.append("Models:")
                models = dbt_architecture['models']
                if isinstance(models, dict):
                    for model_name, details in models.items():
                        output.append(f"  • {model_name}:")
                        if isinstance(details, dict):
                            for key, value in details.items():
                                output.append(f"    - {key}: {value}")
                elif isinstance(models, list):
                    for model in models:
                        if isinstance(model, str):
                            output.append(f"  • {model}")
                        elif isinstance(model, dict) and 'name' in model:
                            output.append(f"  • {model['name']}")
            
            # Add sources information
            if 'sources' in dbt_architecture:
                output.append("\nSources:")
                sources = dbt_architecture['sources']
                if isinstance(sources, dict):
                    for source_name, details in sources.items():
                        output.append(f"  • {source_name}:")
                        if isinstance(details, dict):
                            for key, value in details.items():
                                output.append(f"    - {key}: {value}")
                elif isinstance(sources, list):
                    for source in sources:
                        if isinstance(source, str):
                            output.append(f"  • {source}")
                        elif isinstance(source, dict) and 'name' in source:
                            output.append(f"  • {source['name']}")
            
            # Add project structure
            if 'project_structure' in dbt_architecture:
                output.append("\nProject Structure:")
                structure = dbt_architecture['project_structure']
                if isinstance(structure, dict):
                    for dir_name, details in structure.items():
                        output.append(f"  • {dir_name}: {details}")
                elif isinstance(structure, list):
                    for item in structure:
                        output.append(f"  • {item}")
        
        return "\n".join(output) if output else "No structured dbt architecture available."

    def format_data_lineage(data_lineage: Union[Dict[str, Any], List[str]]) -> str:
        """Format data lineage information in a readable way."""
        if not data_lineage:
            return "No data lineage information available."
            
        output = []
        
        # Handle list format
        if isinstance(data_lineage, list):
            for item in data_lineage:
                if isinstance(item, str):
                    output.append(f"• {item}")
                elif isinstance(item, dict):
                    if 'source' in item and 'target' in item:
                        output.append(f"• {item['source']} → {item['target']}")
                    elif 'name' in item:
                        output.append(f"• {item['name']}")
                    elif 'description' in item:
                        output.append(f"• {item['description']}")
                    else:
                        # Convert dict to string representation
                        output.append(f"• {str(item)}")
            
        # Handle dictionary format
        elif isinstance(data_lineage, dict):
            # Check for special fields first
            if 'source_system' in data_lineage:
                output.append(f"• Source System: {data_lineage['source_system']}")
            
            if 'tables_used' in data_lineage:
                tables = data_lineage['tables_used']
                if isinstance(tables, list):
                    output.append(f"• Tables Used: {', '.join(tables)}")
                else:
                    output.append(f"• Tables Used: {tables}")
            
            if 'description' in data_lineage:
                output.append(f"• Description: {data_lineage['description']}")
            
            # Handle standard entity->dependencies format
            for entity, dependencies in data_lineage.items():
                if entity not in ['source_system', 'tables_used', 'description']:
                    output.append(f"• {entity}:")
                    if isinstance(dependencies, list):
                        if dependencies:
                            # Check if these are dictionaries or strings
                            if isinstance(dependencies[0], dict):
                                # Handle dictionary format
                                for dep in dependencies:
                                    if 'source' in dep and 'target' in dep:
                                        output.append(f"  - {dep['source']} → {dep['target']}")
                                    elif 'name' in dep:
                                        output.append(f"  - {dep['name']}")
                            else:
                                # Handle simple string list
                                output.append(f"  - Dependencies: {', '.join(dependencies)}")
                    elif isinstance(dependencies, dict):
                        # Handle dictionary of upstream/downstream
                        if 'upstream' in dependencies:
                            upstream = dependencies['upstream']
                            output.append(f"  - Upstream: {', '.join(upstream) if isinstance(upstream, list) else upstream}")
                        if 'downstream' in dependencies:
                            downstream = dependencies['downstream']
                            output.append(f"  - Downstream: {', '.join(downstream) if isinstance(downstream, list) else downstream}")
        
        return "\n".join(output) if output else "No structured data lineage available."

    def format_list(items: List[Union[str, Dict[str, Any]]]) -> str:
        """Format a list in a readable way."""
        if not items:
            return "No information available."
            
        output = []
        
        for item in items:
            if isinstance(item, dict):
                if 'description' in item:
                    output.append(f"• {item['description']}")
                elif 'value' in item:
                    if 'key' in item:
                        output.append(f"• {item['key']}: {item['value']}")
                    else:
                        output.append(f"• {item['value']}")
                elif 'name' in item:
                    output.append(f"• {item['name']}")
                else:
                    # Fallback to first value
                    first_value = next(iter(item.values()), "No description")
                    output.append(f"• {first_value}")
            else:
                output.append(f"• {item}")
        
        return "\n".join(output) if output else "No information available."

    # Build the graph
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("question_parser", parse_question)
    graph.add_node("github_processor", process_github)
    graph.add_node("code_processor", process_code)
    graph.add_node("doc_processor", process_docs)
    graph.add_node("dbt_processor", process_dbt)
    graph.add_node("combiner", combine_results)
    graph.add_node("summarizer", create_final_summary)

    # Add edges for sequential processing with question parsing first
    graph.add_edge(START, "question_parser")
    
    # Add edges for parallel processing of different data sources
    graph.add_edge("question_parser", "github_processor")
    graph.add_edge("question_parser", "code_processor")
    graph.add_edge("question_parser", "doc_processor")
    graph.add_edge("question_parser", "dbt_processor")
    
    # Combine results from all processors
    graph.add_edge("github_processor", "combiner")
    graph.add_edge("code_processor", "combiner")
    graph.add_edge("doc_processor", "combiner")
    graph.add_edge("dbt_processor", "combiner")
    
    # Create final summary
    graph.add_edge("combiner", "summarizer")
    graph.add_edge("summarizer", END)

    # Create SQLite saver
    db_path = str(Path(__file__).parent.parent.parent / "chat_history.db")
    conn = connect(db_path, check_same_thread=False)  # Allow multi-threading
    checkpointer = SqliteSaver(conn)

    # Update graph compilation to use SQLite
    return graph.compile(checkpointer=checkpointer)

class DataArchitectSystem:
    def __init__(self, tools: SearchTools):
        self.app = create_data_architect_agent(tools)
        self.db = ChatDatabase()
        self._lock = Lock()  # Add thread lock
        
        # Add model definitions for metadata
        self.summary_model = ChatOllama(
            model="gemma3:latest",
            temperature=0.2,
            base_url="http://localhost:11434",
            timeout=180,
        )
        self.github_model = ChatOllama(
            model="gemma3:latest",
            temperature=0,
            base_url="http://localhost:11434",
            timeout=180,
        )
        self.code_model = ChatOllama(
            model="dgemma3:latest",
            temperature=0,
            base_url="http://localhost:11434",
            timeout=120,
        )
        self.dbt_model = ChatOllama(
            model="gemma3:latest",
            temperature=0,
            base_url="http://localhost:11434",
            timeout=120,
        )
        
    def _log_search_results(self, search_type: str, results: List[Dict[str, Any]]) -> None:
        """Log detailed information about search results."""
        if not results:
            logger.info(f"{search_type} search returned no results")
            return
            
        logger.info(f"{search_type} search returned {len(results)} results")
        
        # Log details of each result (limited to first 3 for brevity)
        for i, result in enumerate(results[:3], 1):
            file_path = result.get('file_path', 'Unknown file')
            repo_info = result.get('repo_info', {})
            repo_name = repo_info.get('name', 'Unknown repository')
            
            logger.info(f"{search_type} result {i}: File={file_path}, Repo={repo_name}")
            
            # Log a preview of the code snippet
            code_snippet = result.get('code_snippet', '')
            if code_snippet:
                # Extract first few lines for preview
                lines = code_snippet.split('\n')
                preview_lines = lines[:3]  # First 3 lines
                preview = '\n'.join(preview_lines)
                preview += "..." if len(lines) > 3 else ""
                logger.info(f"Code preview: {preview}")
            
            # Add related files if available
            related_files = result.get('related_files', [])
            if related_files:
                related_paths = [f.get('file_path', 'Unknown') for f in related_files[:3]]
                logger.info(f"Related files: {', '.join(related_paths)}")
                
        # If there are more results than we showed in detail
        if len(results) > 3:
            logger.info(f"... and {len(results) - 3} more {search_type} results")
        
        logger.info(f"End of {search_type} search results log")

    def analyze(self, question: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze a data architecture question and return a detailed response.
        """
        # Generate unique IDs for this conversation
        conversation_id = config.get('conversation_id', str(uuid.uuid4()))
        thread_id = config.get('thread_id', conversation_id)
        
        try:
            logger.info(f"Starting data architect analysis for question: {question[:100]}...")
            
            # Initialize state with relevant contexts
            state = {
                'messages': [question],
                'conversation_id': conversation_id,
                'thread_id': thread_id,
                'config': config or {}
            }
            
            # Run question analysis first
            state.update(self.process_question(state))
            
            # Process all components in parallel for better performance
            github_result = self.process_github(state)
            code_result = self.process_code(state)
            dbt_result = self.process_dbt(state)
            
            # Combine results
            state.update(github_result)
            state.update(code_result)
            state.update(dbt_result)
            
            # Process summary after all other steps
            summary_result = self.process_summary(state)
            state.update(summary_result)
            
            # Generate detailed responses based on the results
            github_details = self.format_github_results(github_result)
            code_details = self.format_code_results(code_result)
            
            # Prepare response data structure
            response = {
                'conversation_id': conversation_id,
                'thread_id': thread_id,
                'question': question,
                'architect_response': state.get('summary', "Unable to generate a summary."),
                'technical_details': {
                    'github_analysis': github_details,
                    'code_analysis': code_details,
                    'dbt_analysis': state.get('dbt_analysis', "No DBT analysis available."),
                    'question_analysis': state.get('question_analysis', {}),
                },
                'code_context': {
                    'github_context': state.get('github_context', {"query": question, "results": []}),
                    'code_context': state.get('code_context', {"query": question, "results": []}),
                    'dbt_context': state.get('dbt_context', {"query": question, "results": []}),
                },
                'metadata': {
                    'processing_time': time.time() - state.get('start_time', time.time()),
                    'model_versions': {
                        'summary': getattr(self.summary_model, 'model_name', 'unknown'),
                        'github': getattr(self.github_model, 'model_name', 'unknown'),
                        'code': getattr(self.code_model, 'model_name', 'unknown'),
                        'dbt': getattr(self.dbt_model, 'model_name', 'unknown'),
                    }
                }
            }
            
            logger.info(f"Data architect analysis completed successfully for conversation {conversation_id}")
            return response
            
        except Exception as e:
            logger.error(f"Error in data architect analysis: {str(e)}", exc_info=True)
            return {
                'conversation_id': conversation_id,
                'thread_id': thread_id,
                'question': question,
                'architect_response': "I encountered an error while analyzing your question. Please try rephrasing or providing more details.",
                'technical_details': {
                    'error': str(e),
                    'traceback': traceback.format_exc()
                },
                'code_context': {'query': question, 'results': []},
                'metadata': {'error': True, 'timestamp': datetime.now().isoformat()}
            }
            
    def format_github_results(self, github_result: Dict[str, Any]) -> str:
        """Format GitHub results into a detailed response."""
        if not github_result:
            return "No GitHub analysis available."
            
        github_context = github_result.get('github_context', {})
        results = github_context.get('results', [])
        
        if not results:
            return "No relevant GitHub repositories found."
            
        # Format the results into a structured response
        formatted_output = ["### GitHub Analysis Results\n"]
        
        for i, result in enumerate(results, 1):
            formatted_output.append(f"**Result {i}**:")
            
            file_path = result.get('file_path', 'Unknown file')
            formatted_output.append(f"- **File**: {file_path}")
            
            repo_info = result.get('repo_info', {})
            repo_name = repo_info.get('name', 'Unknown repository')
            formatted_output.append(f"- **Repository**: {repo_name}")
            
            # Add code snippet
            code_snippet = result.get('code_snippet', 'No code available')
            formatted_output.append(f"- **Code**:\n```\n{code_snippet}\n```\n")
            
            # Add related files if available
            related_files = result.get('related_files', [])
            if related_files:
                related_file_paths = [f.get('file_path', 'Unknown') for f in related_files]
                formatted_output.append(f"- **Related Files**: {', '.join(related_file_paths)}")
            
            formatted_output.append("")  # Add blank line between results
        
        return "\n".join(formatted_output)
    
    def format_code_results(self, code_result: Dict[str, Any]) -> str:
        """Format code search results into a detailed response."""
        if not code_result:
            return "No code analysis available."
            
        code_context = code_result.get('code_context', {})
        results = code_context.get('results', [])
        
        if not results:
            return "No relevant code found in the codebase."
            
        # Format the results into a structured response
        formatted_output = ["### Code Analysis Results\n"]
        
        for i, result in enumerate(results, 1):
            formatted_output.append(f"**Result {i}**:")
            
            # Get content and metadata
            content = result.get('content', 'No content available')
            metadata = result.get('metadata', {})
            
            # Add file path if available
            file_path = metadata.get('source', metadata.get('file_path', 'Unknown file'))
            formatted_output.append(f"- **File**: {file_path}")
            
            # Add code content
            formatted_output.append(f"- **Content**:\n```\n{content}\n```\n")
            
            # Add additional metadata if available
            if 'score' in metadata:
                formatted_output.append(f"- **Relevance Score**: {metadata['score']}")
                
            formatted_output.append("")  # Add blank line between results
        
        return "\n".join(formatted_output)

    def process_question(self, state: Dict[str, Any]) -> Dict[str, Any]:
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

    def process_github(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process GitHub repositories."""
        try:
            # Create GitHub search agent
            from src.agents.data_architect.github_search import GitHubCodeSearchAgent
            github_search_agent = GitHubCodeSearchAgent()
            
            messages = state['messages']
            question_analysis = state.get('question_analysis', {})
            if not messages:
                return {}
            
            query = messages[-1].content if isinstance(messages[-1], BaseMessage) else str(messages[-1])
            
            # Search GitHub repositories
            logger.info(f"Searching GitHub repositories for: {query[:100]}...")
            search_results = github_search_agent.search_code(
                parsed_question=question_analysis, 
                max_results=5
            )
            
            # Log detailed search results
            self._log_search_results("GitHub", search_results)
            
            # Format results
            github_context = {"query": query, "results": search_results}
            github_analysis = "GitHub analysis completed. Found {} results.".format(len(search_results))
            
            return {
                "github_context": github_context,
                "github_analysis": github_analysis
            }
            
        except Exception as e:
            logger.error(f"Error in GitHub processing: {str(e)}", exc_info=True)
            return {
                "github_analysis": f"Error during GitHub analysis: {str(e)}",
                "github_context": {"query": query if 'query' in locals() else "", "results": []}
            }
    
    def process_code(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process code search and analysis."""
        try:
            messages = state['messages']
            if not messages:
                return {}
            
            query = messages[-1].content if isinstance(messages[-1], BaseMessage) else str(messages[-1])
            
            # Use the tools object passed to the constructor
            from src.tools import SearchTools
            tools = SearchTools(ChromaDBManager())
            search_results = tools.search_code(query)
            
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
            code_context = {"query": query, "results": search_results.get('results', [])}
            code_analysis = "Code analysis completed. Found {} results.".format(
                len(search_results.get('results', []))
            )
            
            return {
                "code_context": code_context,
                "code_analysis": code_analysis
            }
            
        except Exception as e:
            logger.error(f"Error in code processing: {str(e)}", exc_info=True)
            return {
                "code_analysis": f"Error during code analysis: {str(e)}",
                "code_context": {"query": query if 'query' in locals() else "", "results": []}
            }
    
    def process_dbt(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process dbt models."""
        try:
            messages = state['messages']
            if not messages:
                return {}
            
            query = messages[-1].content if isinstance(messages[-1], BaseMessage) else str(messages[-1])
            
            # Placeholder for dbt model processing
            return {
                "dbt_context": {"query": query, "results": []},
                "dbt_analysis": "DBT model analysis not implemented yet."
            }
            
        except Exception as e:
            logger.error(f"Error in dbt processing: {str(e)}", exc_info=True)
            return {
                "dbt_analysis": f"Error during dbt analysis: {str(e)}",
                "dbt_context": {"query": query if 'query' in locals() else "", "results": []}
            }
    
    def process_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Create final summary of all analyses with context-aware responses based on question type."""
        try:
            messages = state['messages']
            if not messages:
                return {}
            
            query = messages[-1].content if isinstance(messages[-1], BaseMessage) else str(messages[-1])
            
            # Classify the question type to provide a more tailored response
            question_type = self._classify_question_type(query)
            logger.info(f"Classified question as: {question_type}")
            
            # Get context information
            github_context = state.get('github_context', {})
            code_context = state.get('code_context', {})
            
            # Get GitHub results
            github_results = github_context.get('results', [])
            code_results = code_context.get('results', [])
            
            logger.info(f"Generating summary with {len(github_results)} GitHub results and {len(code_results) if isinstance(code_results, list) else 0} code results")
            
            # Generate prompt for summary LLM
            prompt = f"""
            You are a senior data architect specializing in dbt and Snowflake implementation. 
            You must analyze ONLY the following information about a data architecture question and provide a comprehensive response based SOLELY on the code provided. 
            DO NOT make generalizations or assumptions beyond what is explicitly shown in the code.
            
            USER QUESTION:
            {query}
            
            QUESTION TYPE: {question_type}
            
            GITHUB SEARCH RESULTS:
            """
            
            # Add GitHub results
            if github_results:
                for i, result in enumerate(github_results, 1):
                    file_path = result.get('file_path', 'Unknown file')
                    repo_info = result.get('repo_info', {})
                    repo_name = repo_info.get('name', 'Unknown repository')
                    code_snippet = result.get('code_snippet', 'No code available')
                    
                    prompt += f"""
                    Result {i}:
                    File: {file_path}
                    Repository: {repo_name}
                    Code:
                    ```
                    {code_snippet}
                    ```
                    """
                    
                    # Add related files if available
                    related_files = result.get('related_files', [])
                    if related_files:
                        prompt += "Related files:\n"
                        for rel_file in related_files:
                            rel_path = rel_file.get('file_path', 'Unknown')
                            prompt += f"- {rel_path}\n"
            else:
                prompt += "No GitHub results found.\n"
            
            # Add code search results
            prompt += "\nCODE SEARCH RESULTS:\n"
            if code_results and isinstance(code_results, list) and len(code_results) > 0:
                for i, result in enumerate(code_results, 1):
                    content = result.get('content', 'No content available')
                    metadata = result.get('metadata', {})
                    file_path = metadata.get('source', metadata.get('file_path', 'Unknown file'))
                    
                    prompt += f"""
                    Result {i}:
                    File: {file_path}
                    Content:
                    ```
                    {content}
                    ```
                    """
            else:
                prompt += "No code search results found.\n"
            
            # Add specific instructions based on question type
            if question_type == "MODEL_EXPLANATION":
                prompt += self._get_model_explanation_instructions(query)
            elif question_type == "IMPLEMENTATION_GUIDE":
                prompt += self._get_implementation_instructions(query)
            elif question_type == "ENHANCEMENT":
                prompt += self._get_enhancement_instructions(query)
            elif question_type == "TROUBLESHOOTING":
                prompt += self._get_troubleshooting_instructions(query)
            else:
                prompt += self._get_general_instructions(query)
            
            # Call LLM for analysis
            logger.info("Generating detailed analysis summary...")
            response = self.summary_model.invoke(prompt)
            response_text = response.content if isinstance(response, BaseMessage) else str(response)
            
            # Log a preview of the response
            response_preview = response_text[:500] + "..." if len(response_text) > 500 else response_text
            logger.info(f"Generated summary response: {response_preview}")
            
            # Combine all analyses for the state record
            combined_analysis = f"""
            GitHub Analysis:
            {state.get('github_analysis', 'No GitHub analysis available.')}
            
            Code Analysis:
            {state.get('code_analysis', 'No code analysis available.')}
            
            DBT Analysis:
            {state.get('dbt_analysis', 'No DBT analysis available.')}
            """
            
            # Post-process the response to include file references and examples
            enhanced_response = self._enhance_response(response_text, question_type, github_results, code_results)
            
            return {
                "summary": enhanced_response,
                "combined_analysis": combined_analysis,
                "question_type": question_type
            }
            
        except Exception as e:
            logger.error(f"Error in summary processing: {str(e)}", exc_info=True)
            return {
                "summary": f"Error creating summary: {str(e)}",
                "combined_analysis": "",
                "question_type": "UNKNOWN"
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
    
    def _enhance_response(self, response: str, question_type: str, github_results: List[Dict[str, Any]], code_results: List[Dict[str, Any]]) -> str:
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