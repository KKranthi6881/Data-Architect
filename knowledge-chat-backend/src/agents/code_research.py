from typing import Dict, List, Any, Annotated, Sequence, TypedDict
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

from src.tools import SearchTools
from src.db.database import ChatDatabase

# Set up logger
logger = logging.getLogger(__name__)

# Define state type
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "The messages in the conversation"]
    code_context: Annotated[Dict, "Code search results"]
    doc_context: Annotated[Dict, "Documentation search results"]
    code_analysis: Annotated[str, "Code analysis output"]
    doc_analysis: Annotated[str, "Documentation analysis output"]
    combined_output: Annotated[str, "Combined analysis output"]
    final_summary: Annotated[str, "User-friendly final summary"]

# Define structured outputs
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

def create_simple_agent(tools: SearchTools):
    # Initialize models
    code_model = ChatOllama(
        model="deepseek-r1:8b",
        temperature=0,
        base_url="http://localhost:11434",
        timeout=120,
    )
    
    doc_model = ChatOllama(
        model="llama3.2:3b",
        temperature=0,
        base_url="http://localhost:11434",
        timeout=120,
    )

    # Create parsers
    code_parser = PydanticOutputParser(pydantic_object=CodeAnalysis)
    doc_parser = PydanticOutputParser(pydantic_object=DocAnalysis)

    # Create prompt templates
    code_template = """
    Analyze the following code and provide structured information about it.
    
    CODE TO ANALYZE:
    {code_context}
    
    Guidelines:
    - Focus on SQL and Python code structure
    - Identify tables, columns and their relationships related to the user question only.
    - Explain technical implementation details related to the user question only.
    - Describe the business logic 
    - Provide the Column level lineage which is relavent to related to the user question only and code.
    - Don't provide a generalized answers or tables info
    - You must rethink and provide related to the user question.
    
    Your response MUST be in the following JSON format:
    {format_instructions}
    
    Make sure to include the below content must be related to the user question only.:
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
    
    Guidelines:
    - Focus on business requirements and workflows
    - Identify key concepts and terminology
    - Extract business rules and processes
    - Note any important considerations
    - Make sure revist, anlyze and double heck if you miss any table or columns before you confirm the output.
    
    Your response MUST be in the following JSON format:
    {format_instructions}
    
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

    # Add new model for final summary
    summary_model = ChatOllama(
        model="llama3.2:3b",
        temperature=0.3,  # Slightly higher temperature for more natural language
        base_url="http://localhost:11434",
        timeout=120,
    )

    # Create parser for final summary
    summary_parser = PydanticOutputParser(pydantic_object=FinalSummary)

    # Create template for final summary
    summary_template = """
    You are an expert SQL and data analyst. Create a clear technical document analyzing the following code and context.
    
    ANALYSIS TO SUMMARIZE:
    {combined_analysis}

    Guidelines:
    - Create a clear, readable technical document
    - Focus on explaining the business purpose of each query
    - Break down complex SQL logic into simple explanations
    - Clearly describe tables and their relationships
    - Highlight key metrics and calculations
    
    Format your response as a technical document with the following sections:
    
    1. Overview
       - High-level summary of the system
       - Main business objectives
    
    2. Data Model Analysis
       - Tables and their purposes
       - Key columns and their uses
       - Table relationships
    
    3. Business Processes
       - Key processes identified
       - Metrics and calculations
    
    4. Technical Implementation
       - Implementation approach
       - Key technical considerations

    
    Make your analysis clear and actionable. After your analysis 
    
    1. provide suitable sql script based on user question only.
    2. Don't genreate random sql queries
    3. Before you finalize it make sure you analyze the list of tables and columns should cover for the question related task.
    4. Parse the query without any syntax error and provide the final sql query output.
    4. If you do not find the related tables or columns respecitve to the questions Please say I do not have enough info available.
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
                output = f"""
                Code Analysis Results:
                
                1. Tables and Columns:
                {analysis.tables_and_columns}
                
                2. Relationships:
                {analysis.relationships}
                
                3. Business Logic:
                {analysis.business_logic}
                
                4. Technical Details:
                {analysis.technical_details}
                """
            except Exception as parse_error:
                logger.warning(f"Failed to parse code output: {str(parse_error)}")
                output = response_text
            
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

    def combine_results(state: AgentState) -> Dict:
        """Combine code and documentation analysis results."""
        combined = f"""
        Analysis Results
        ===============

        Code Analysis:
        -------------
        {state.get('code_analysis', 'No code analysis available')}

        Documentation Analysis:
        ---------------------
        {state.get('doc_analysis', 'No documentation analysis available')}
        """
        
        return {"combined_output": combined}

    # Add this new function after combine_results
    def create_final_summary(state: AgentState) -> Dict:
        """Create a user-friendly summary of the analysis."""
        try:
            combined_analysis = state.get('combined_output', '')
            if not combined_analysis:
                return {"final_summary": "No analysis available to summarize"}

            # Generate summary
            formatted_prompt = summary_prompt.format(combined_analysis=combined_analysis)
            response = summary_model.invoke(formatted_prompt)
            response_text = response.content if isinstance(response, BaseMessage) else str(response)

            try:
                # Format the response as a technical document
                output = f"""

                                  {response_text}
                ===================================================

                Code Context Details
                ------------------
                The analysis is based on the following code structure:
                
                Tables Found:
                - feature_sequence: Tracks user journey and feature usage
                - feature_adoption: Measures feature adoption metrics
                - subscription_changes: Monitors revenue impact

                Key Metrics:
                - User engagement and success rates
                - Feature adoption rates
                - Revenue impact measurements

                For detailed code implementation, see the code context below.
                """

            except Exception as parse_error:
                logger.warning(f"Failed to format output: {str(parse_error)}")
                output = response_text

            return {"final_summary": output}

        except Exception as e:
            logger.error(f"Error in summary creation: {str(e)}")
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
    graph.add_node("combiner", combine_results)
    graph.add_node("summarizer", create_final_summary)

    # Add edges for parallel processing
    graph.add_edge(START, "code_processor")
    graph.add_edge(START, "doc_processor")
    graph.add_edge("code_processor", "combiner")
    graph.add_edge("doc_processor", "combiner")
    graph.add_edge("combiner", "summarizer")
    graph.add_edge("summarizer", END)

    # Create SQLite saver
    db_path = str(Path(__file__).parent.parent.parent / "chat_history.db")
    conn = connect(db_path, check_same_thread=False)  # Allow multi-threading
    checkpointer = SqliteSaver(conn)

    # Update graph compilation to use SQLite
    return graph.compile(checkpointer=checkpointer)

class SimpleAnalysisSystem:
    def __init__(self, tools: SearchTools):
        self.app = create_simple_agent(tools)
        self.db = ChatDatabase()
        self._lock = Lock()  # Add thread lock

    def analyze(self, query: str) -> Dict[str, Any]:
        """Process a query through the analysis system."""
        try:
            # Generate unique ID for the conversation
            conversation_id = str(uuid.uuid4())
            
            with self._lock:  # Use lock for thread safety
                result = self.app.invoke({
                    "messages": [HumanMessage(content=query)],
                    "code_context": {},
                    "doc_context": {},
                    "code_analysis": "",
                    "doc_analysis": "",
                    "combined_output": "",
                    "final_summary": ""
                },
                {"configurable": {"thread_id": conversation_id}}
                )
            
            # Prepare response data
            response_data = {
                "output": result.get("final_summary", "No response available"),
                "technical_details": result.get("combined_output", "No technical details available"),
                "code_context": result.get("code_context", {}),
                "query": query
            }
            
            # Save conversation to database
            with self._lock:  # Use lock for database operations
                self.db.save_conversation(conversation_id, response_data)
            
            return response_data
            
        except Exception as e:
            logger.error(f"Error in analysis: {str(e)}", exc_info=True)
            error_response = {
                "output": f"Error during analysis: {str(e)}",
                "technical_details": "",
                "code_context": {},
                "query": query
            }
            return error_response 