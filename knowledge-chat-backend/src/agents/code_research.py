from typing import Dict, List, Any, Annotated, Sequence, TypedDict, Optional
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
from langchain.chains import LLMChain

from src.tools import SearchTools
from src.db.database import ChatDatabase

# Set up logger
logger = logging.getLogger(__name__)

# Define state type
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "The messages in the conversation"]
    code_context: Annotated[Dict, "Code search results"]
    doc_context: Annotated[Dict, "Documentation search results"]
    github_context: Annotated[Dict, "GitHub repository search results"]
    code_analysis: Annotated[str, "Code analysis output"]
    doc_analysis: Annotated[str, "Documentation analysis output"]
    github_analysis: Annotated[str, "GitHub repository analysis output"]
    combined_output: Annotated[str, "Combined analysis output"]
    final_summary: Annotated[str, "User-friendly final summary"]

# Define structured outputs
class CodeAnalysis(BaseModel):
    tables_and_columns: Dict[str, Any] = Field(
        description="Database schema information including tables and their details"
    )
    business_logic: Dict[str, List[Dict[str, str]]] = Field(
        description="Business rules and transformations"
    )
    technical_details: Dict[str, str] = Field(
        description="Technical implementation details"
    )
    business_context: List[Dict[str, str]] = Field(
        description="Business context and comments"
    )

class DocAnalysis(BaseModel):
    key_concepts: List[str] = Field(
        description="Key concepts found in documentation"
    )
    workflows: List[str] = Field(
        description="Business workflows described"
    )
    requirements: Dict[str, List[str]] = Field(
        description="Business and technical requirements"
    )
    additional_context: Dict[str, List[str]] = Field(
        description="Additional contextual information"
    )

class FinalSummary(BaseModel):
    overview: str = Field(
        description="High-level overview of the analyzed system"
    )
    data_model: Dict[str, Any] = Field(
        description="Simplified data model with tables and relationships"
    )
    business_processes: List[str] = Field(
        description="Key business processes identified"
    )
    implementation_notes: List[str] = Field(
        description="Important technical implementation details"
    )
    recommendations: List[str] = Field(
        description="Suggested considerations or improvements"
    )

# Add GitHub Analysis Schema
class GitHubAnalysis(BaseModel):
    repository_info: Dict[str, str] = Field(
        description="Basic repository information"
    )
    file_analysis: List[Dict[str, Any]] = Field(
        description="Analysis of individual files",
        default=[]
    )
    data_architecture: Dict[str, Any] = Field(
        description="Data architecture components found",
        default={}
    )
    implementation_details: Dict[str, Any] = Field(
        description="Implementation patterns and details",
        default={}
    )
    business_context: Dict[str, Any] = Field(
        description="Business context and purpose",
        default={}
    )
    data_lineage: Dict[str, Any] = Field(
        description="Data lineage information",
        default={}
    )



def create_simple_agent(tools: SearchTools):
    # Initialize models
    code_model = ChatOllama(
        model="llama3.2:latest",
        temperature=0,
        base_url="http://localhost:11434",
        timeout=120,
    )
    
    doc_model = ChatOllama(
        model="llama3.2:latest",
        temperature=0,
        base_url="http://localhost:11434",
        timeout=120,
    )

    github_model = ChatOllama(
        model="llama3.2:latest",
        temperature=0,
        base_url="http://localhost:11434",
        timeout=120,
    )

    # Create parsers
    code_parser = PydanticOutputParser(pydantic_object=CodeAnalysis)
    doc_parser = PydanticOutputParser(pydantic_object=DocAnalysis)
    github_parser = PydanticOutputParser(pydantic_object=GitHubAnalysis)

    # Create prompt templates
    code_template = """
    You are a data architect analyzing code structure and schema. Focus on extracting business and technical context from the code.
    
    CODE TO ANALYZE:
    {code_context}
    
    Guidelines:
    - Extract and analyze schema definitions, table structures, and column descriptions
    - Identify business context from code comments and naming conventions
    - Focus on data relationships and dependencies
    - Document data transformations and business rules
    - Look for data quality rules and constraints
    - Identify key business metrics and calculations
    
    Pay special attention to:
    1. Schema Information:
       - Table and column descriptions
       - Data types and constraints
       - Primary/Foreign key relationships
       - Indexing and partitioning strategies
    
    2. Business Logic:
       - Business rules embedded in code
       - Transformation logic and its business purpose
       - Metric calculations and their business significance
       - Data quality checks and validation rules
    
    3. Data Lineage:
       - Source to target mappings
       - Data transformation steps
       - Dependencies between objects
       - Impact analysis information
    
    Your response MUST be in the following JSON format:
    {format_instructions}
    
    Make sure to include:
    1. Detailed table and column metadata with business descriptions
    2. Clear documentation of business rules and data transformations
    3. Complete lineage information showing data flow
    4. Any business context found in comments or naming conventions
    
    Response:
    """

    doc_template = """
    You are a business analyst focusing on extracting business context and requirements from documentation.
    
    DOCUMENTATION TO ANALYZE:
    {doc_context}
    
    Guidelines:
    - Focus on business terminology and definitions
    - Identify key business processes and workflows
    - Extract business rules and requirements
    - Document business metrics and KPIs
    - Capture data governance requirements
    - Note any compliance or regulatory requirements
    
    Pay special attention to:
    1. Business Glossary:
       - Business terms and definitions
       - Domain-specific terminology
       - Business metrics definitions
       - KPI calculations
    
    2. Business Context:
       - Business processes and workflows
       - Business rules and policies
       - Data governance requirements
       - Data quality expectations
    
    3. Requirements:
       - Business requirements
       - Technical requirements
       - Compliance requirements
       - Performance requirements
    
    Your response MUST be in the following JSON format:
    {format_instructions}
    
    Focus on providing clear business context that helps understand:
    1. Why certain data is collected and stored
    2. How the data is used in business processes
    3. What business decisions are made using this data
    4. Who are the key stakeholders and their needs
    
    Response:
    """

    # Add GitHub Analysis Prompt Template
    GITHUB_ANALYSIS_TEMPLATE = """
    You are a data architect analyzing GitHub repository content to understand implementation patterns and data architecture.
    
    CONTEXT:
    The user wants to understand: {query}
    
    GITHUB REPOSITORY CONTENT:
    {github_content}
    
    Focus your analysis on:
    
    1. Repository Structure:
       - Data model definitions and locations
       - SQL/DBT file organization
       - Configuration and environment setup
       - Documentation organization
    
    2. Data Architecture:
       - Database schema definitions
       - Table relationships and dependencies
       - Data transformation patterns
       - Data quality checks
       - Business logic implementation
    
    3. Implementation Details:
       - File locations and URLs for important components
       - Code patterns and best practices
       - Configuration requirements
       - Environment setup
    
    4. Data Lineage:
       - Source to target mappings
       - Transformation logic
       - Dependencies between files
       - Impact analysis
    
    For each relevant file found:
    - Provide the full GitHub URL
    - Explain its purpose and business context
    - Describe how it fits in the data architecture
    - Note any dependencies or relationships
    
    Format your response to match the GitHubAnalysis schema with clear sections for:
    - Repository structure and organization
    - Data architecture components
    - Implementation patterns
    - File locations and URLs
    - Business context and purpose
    - Data lineage information
    
    Response:
    """

    code_prompt = PromptTemplate(
        template=code_template,
        input_variables=["code_context"],
        partial_variables={"format_instructions": code_parser.get_format_instructions()}
    )

    doc_prompt = PromptTemplate(
        template=doc_template,
        input_variables=["doc_context"],
        partial_variables={"format_instructions": doc_parser.get_format_instructions()}
    )

    # Create GitHub prompt template
    github_prompt = PromptTemplate(
        template=GITHUB_ANALYSIS_TEMPLATE,
        input_variables=["query", "github_content"],
        partial_variables={"format_instructions": github_parser.get_format_instructions()}
    )

    # Add new model for final summary
    summary_model = ChatOllama(
        model="llama3.2:latest",
        temperature=0.3,  # Slightly higher temperature for more natural language
        base_url="http://localhost:11434",
        timeout=120,
    )

    # Create parser for final summary
    summary_parser = PydanticOutputParser(pydantic_object=FinalSummary)

    # Create template for final summary
    summary_template = """
    You are a senior data architect creating a comprehensive analysis combining code, documentation, and implementation details.
    
    ANALYSIS TO SUMMARIZE:
    {combined_analysis}
    
    Create a detailed technical document that:
    
    1. Business Context
       - Clear explanation of business purpose
       - Key business processes and workflows
       - Business rules and requirements
       - Important metrics and KPIs
    
    2. Data Architecture
       - Detailed schema information with business context
       - Table and column descriptions
       - Relationships and dependencies
       - Data quality rules and constraints
    
    3. Implementation Details
       - Relevant GitHub file locations (with URLs)
       - Code patterns and best practices
       - Configuration requirements
       - Environment setup
    
    4. Data Lineage
       - Complete source to target mappings
       - Transformation logic and business purpose
       - Dependencies and relationships
       - Impact analysis
    
    5. Recommendations
       - Architecture improvements
       - Best practices alignment
       - Performance considerations
       - Data quality enhancements
    
    For any SQL or data-related questions:
    1. Verify table and column availability
    2. Ensure business context is understood
    3. Consider data quality and constraints
    4. Provide accurate and optimized queries
    5. Include relevant file references
    
    Make your response practical and actionable, including:
    - Specific file locations and URLs
    - Clear business context
    - Complete technical details
    - Implementation considerations
    
    Response:
    """

    summary_prompt = PromptTemplate(
        template=summary_template,
        input_variables=["combined_analysis"],
        partial_variables={"format_instructions": summary_parser.get_format_instructions()}
    )

    def process_code(state: AgentState) -> Dict:
        """Process code analysis."""
        try:
            messages = state['messages']
            if not messages:
                return state
            
            query = messages[-1].content if isinstance(messages[-1], BaseMessage) else str(messages[-1])
            
            # Search code
            search_results = tools.search_code(query)
            
            # Format code context
            code_snippets = []
            for result in search_results.get('results', []):
                code_snippets.append(
                    f"Source: {result['source']}\n"
                    f"Code:\n{result['content']}\n"
                    f"File Info: {result['file_info']}\n"
                )
            
            code_context = "\n".join(code_snippets)
            
            # Generate analysis
            formatted_prompt = code_prompt.format(code_context=code_context)
            response = code_model.invoke(formatted_prompt)
            response_text = response.content if isinstance(response, BaseMessage) else str(response)
            
            try:
                analysis = code_parser.parse(response_text)
                output = {
                    "schema_info": analysis.tables_and_columns,
                    "business_rules": analysis.business_logic,
                    "technical_implementation": analysis.technical_details,
                    "business_context": analysis.business_context
                }
            except Exception as parse_error:
                logger.warning(f"Failed to parse code output: {str(parse_error)}")
                # Provide structured output even when parsing fails
                output = {
                    "schema_info": extract_schema_info(response_text),
                    "business_rules": extract_business_rules(response_text),
                    "technical_implementation": extract_technical_details(response_text),
                    "business_context": extract_business_context(response_text)
                }
            
            return {
                "code_context": {"query": query, "results": search_results.get('results', [])},
                "code_analysis": output
            }
            
        except Exception as e:
            logger.error(f"Error in code processing: {str(e)}")
            return {
                "code_analysis": f"Error during code analysis: {str(e)}",
                "code_context": {}
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
            
            # Generate analysis
            formatted_prompt = doc_prompt.format(doc_context=doc_context)
            response = doc_model.invoke(formatted_prompt)
            response_text = response.content if isinstance(response, BaseMessage) else str(response)
            
            try:
                analysis = doc_parser.parse(response_text)
                output = {
                    "concepts": analysis.key_concepts,
                    "workflows": analysis.workflows,
                    "requirements": analysis.requirements,
                    "context": analysis.additional_context
                }
            except Exception as parse_error:
                logger.warning(f"Failed to parse doc output: {str(parse_error)}")
                # Provide structured output even when parsing fails
                output = {
                    "concepts": extract_concepts(response_text),
                    "workflows": extract_workflows(response_text),
                    "requirements": extract_requirements(response_text),
                    "context": extract_context(response_text)
                }
            
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

    def process_github(state: AgentState) -> Dict:
        """Process GitHub repository content."""
        try:
            github_content = state["github_context"]
            query = state["messages"][0].content
            
            try:
                # Use RunnableSequence instead of deprecated LLMChain
                github_chain = github_prompt | github_model | github_parser
                
                github_analysis = github_chain.invoke({
                    "query": query,
                    "github_content": github_content
                })
                
                output = {
                    "repository": github_analysis.repository_info,
                    "files": github_analysis.file_analysis,
                    "architecture": github_analysis.data_architecture,
                    "implementation": github_analysis.implementation_details,
                    "business_context": github_analysis.business_context,
                    "lineage": github_analysis.data_lineage
                }
            except Exception as parse_error:
                logger.warning(f"Failed to parse GitHub output: {str(parse_error)}")
                # Provide structured output even when parsing fails
                output = {
                    "repository": extract_repo_info(response_text),
                    "files": extract_file_analysis(response_text),
                    "architecture": extract_architecture(response_text),
                    "implementation": extract_implementation(response_text),
                    "business_context": extract_business_info(response_text),
                    "lineage": extract_lineage(response_text)
                }
            
            return {
                "github_context": {"query": query, "content": github_content},
                "github_analysis": output
            }
            
        except Exception as e:
            logger.error(f"Error in GitHub analysis: {str(e)}")
            return {
                "github_analysis": f"Error analyzing GitHub content: {str(e)}",
                "github_context": {}
            }

    def combine_results(state: AgentState) -> Dict:
        """Combine code, documentation, and GitHub analysis results."""
        combined = f"""
        Analysis Results
        ===============

        Code Analysis:
        -------------
        {state.get('code_analysis', 'No code analysis available')}

        Documentation Analysis:
        ---------------------
        {state.get('doc_analysis', 'No documentation analysis available')}
        
        GitHub Repository Analysis:
        ------------------------
        {state.get('github_analysis', 'No GitHub analysis available')}
        """
        
        return {"combined_output": combined}

    def create_final_summary(state: AgentState) -> Dict:
        """Create a final summary incorporating all analysis results."""
        try:
            combined_analysis = state.get('combined_output', '')
            
            # Generate summary using the summary prompt
            formatted_prompt = summary_prompt.format(combined_analysis=combined_analysis)
            response = summary_model.invoke(formatted_prompt)
            response_text = response.content if isinstance(response, BaseMessage) else str(response)
            
            try:
                # Parse the structured summary
                summary = summary_parser.parse(response_text)
                
                # Format data model information
                data_model_text = format_data_model(summary.data_model)
                
                # Create a well-formatted final summary
                final_output = f"""
                Technical Analysis Summary
                ========================

                Overview:
                --------
                {summary.overview}

                Data Model:
                ----------
                {data_model_text}

                Business Processes:
                ------------------
                {chr(10).join(f"• {process}" for process in summary.business_processes)}

                Implementation Details:
                ----------------------
                {chr(10).join(f"• {note}" for note in summary.implementation_notes)}

                Recommendations:
                ---------------
                {chr(10).join(f"• {rec}" for rec in summary.recommendations)}
                """
                
            except Exception as parse_error:
                logger.warning(f"Failed to parse summary output: {str(parse_error)}")
                final_output = response_text
            
            return {"final_summary": final_output}
            
        except Exception as e:
            logger.error(f"Error creating final summary: {str(e)}")
            return {"final_summary": f"Error creating summary: {str(e)}"}

    def format_data_model(data_model: Dict) -> str:
        """Format data model information in a readable way."""
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
        
        return "\n".join(output)

    # Build the graph
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("code_processor", process_code)
    graph.add_node("doc_processor", process_docs)
    graph.add_node("process_github", process_github)
    graph.add_node("combiner", combine_results)
    graph.add_node("summarizer", create_final_summary)

    # Add edges for parallel processing
    graph.add_edge(START, "code_processor")
    graph.add_edge(START, "doc_processor")
    graph.add_edge(START, "process_github")
    graph.add_edge("code_processor", "combiner")
    graph.add_edge("doc_processor", "combiner")
    graph.add_edge("process_github", "combiner")
    graph.add_edge("combiner", "summarizer")
    graph.add_edge("summarizer", END)

    # Create SQLite saver
    db_path = str(Path(__file__).parent.parent.parent / "chat_history.db")
    conn = connect(db_path, check_same_thread=False)  # Allow multi-threading
    checkpointer = SqliteSaver(conn)

    # Update graph compilation to use SQLite
    return graph.compile(checkpointer=checkpointer)

class SimpleAnalysisSystem:
    """A simple system for analyzing code and documentation."""
    
    def __init__(self, tools: SearchTools, db: Optional[ChatDatabase] = None):
        """Initialize the analysis system with search tools."""
        self.tools = tools
        self.db = db
        self.lock = Lock()
        
        # Initialize the app
        self.app = create_simple_agent(tools)

    def analyze(self, query: str, conversation_id: Optional[str] = None, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Process a query through the analysis system."""
        try:
            conv_id = conversation_id or str(uuid.uuid4())
            
            with self.lock:
                # Initialize processing status
                processing_status = {
                    "code": {"status": "processing", "message": "Analyzing code..."},
                    "doc": {"status": "processing", "message": "Processing documentation..."},
                    "github": {"status": "processing", "message": "Checking GitHub repositories..."},
                    "summary": {"status": "waiting", "message": "Waiting for analysis..."}
                }
                
                result = self.app.invoke({
                    "messages": [HumanMessage(content=query)],
                    "code_context": context.get("code_context", {}) if context else {},
                    "doc_context": context.get("doc_context", {}) if context else {},
                    "github_context": context.get("github_context", {}) if context else {},
                    "code_analysis": "",
                    "doc_analysis": "",
                    "github_analysis": "",
                    "combined_output": "",
                    "final_summary": ""
                },
                {"configurable": {"thread_id": conv_id}})
                
                # Update processing status based on results
                processing_status.update({
                    "code": {"status": "completed", "message": "Code analysis complete"},
                    "doc": {"status": "completed", "message": "Documentation analysis complete"},
                    "github": {"status": "completed", "message": "GitHub analysis complete"},
                    "summary": {"status": "completed", "message": "Analysis summarized"}
                })
                
                response_data = {
                    "conversation_id": conv_id,
                    "output": result.get("final_summary", "No response available"),
                    "technical_details": result.get("combined_output", "No technical details available"),
                    "code_analysis": result.get("code_analysis", ""),
                    "doc_analysis": result.get("doc_analysis", ""),
                    "github_analysis": result.get("github_analysis", ""),
                    "sources": {
                        "code": result.get("code_context", {}),
                        "documentation": result.get("doc_context", {}),
                        "github": result.get("github_context", {})
                    },
                    "query": query,
                    "business_context": result.get("business_context", ""),
                    "data_lineage": result.get("data_lineage", {}),
                    "processing_status": processing_status
                }
                
                if self.db:
                    with self.lock:
                        self.db.save_conversation(conv_id, response_data)
                
                return response_data
                
        except Exception as e:
            logger.error(f"Error in analysis: {str(e)}", exc_info=True)
            return {
                "conversation_id": conversation_id or str(uuid.uuid4()),
                "output": f"Error during analysis: {str(e)}",
                "technical_details": "",
                "code_analysis": "",
                "doc_analysis": "",
                "github_analysis": "",
                "sources": {},
                "query": query,
                "business_context": "",
                "data_lineage": {},
                "processing_status": {
                    "code": {"status": "error", "message": "Analysis failed"},
                    "doc": {"status": "error", "message": "Analysis failed"},
                    "github": {"status": "error", "message": "Analysis failed"},
                    "summary": {"status": "error", "message": "Analysis failed"}
                }
            } 

# Add helper functions for extracting information when parsing fails
def extract_schema_info(text: str) -> Dict[str, Any]:
    """Extract schema information from text when parsing fails."""
    # Add implementation to extract schema info using regex or other methods
    return {"tables": [], "relationships": []}

# Add similar helper functions for other extraction needs... 