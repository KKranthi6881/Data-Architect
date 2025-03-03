from typing import Dict, List, Any, Annotated, Sequence, TypedDict, Optional
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from pathlib import Path
import uuid
import logging
from sqlite3 import connect
from threading import Lock
import asyncio
import json
import re
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

from src.tools import SearchTools
from src.db.database import ChatDatabase
from src.agents.data_architect.human_feedback import HumanFeedbackSystem

# Set up logger
logger = logging.getLogger(__name__)

# Define state type
class ParserState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "The messages in the conversation"]
    doc_context: Annotated[Dict, "Documentation search results"]
    parsed_question: Annotated[Dict, "Structured parsed question output"]
    feedback_status: Annotated[Optional[str], "Status of human feedback"]

# Define structured outputs
class TableInfo(BaseModel):
    name: str = Field(description="Name of the table")
    columns: List[str] = Field(description="List of relevant columns in the table")
    description: Optional[str] = Field(description="Description of the table's purpose", default=None)

class QueryIntent(BaseModel):
    primary_intent: str = Field(
        description="The primary intent of the user's question (e.g., 'data retrieval', 'analysis', 'comparison')"
    )
    time_period: Optional[str] = Field(
        description="Time period mentioned in the query if any", default=None
    )
    filters: Optional[Dict[str, Any]] = Field(
        description="Filters to apply to the data", default=None
    )
    metrics: List[str] = Field(
        description="Metrics or measurements requested in the query"
    )
    grouping: Optional[List[str]] = Field(
        description="Dimensions to group by if any", default=None
    )

class ParsedQuestion(BaseModel):
    original_question: str = Field(
        description="The original question asked by the user"
    )
    rephrased_question: str = Field(
        description="The question rephrased for clarity"
    )
    alternative_interpretations: List[str] = Field(
        description="Alternative ways to interpret the question",
        default_factory=list
    )
    business_context: str = Field(
        description="Business context relevant to the question"
    )
    relevant_tables: List[TableInfo] = Field(
        description="Tables relevant to answering the question"
    )
    query_intent: QueryIntent = Field(
        description="The intent and parameters of the query"
    )
    suggested_approach: str = Field(
        description="Suggested approach to answer the question"
    )
    confidence_score: float = Field(
        description="Confidence score of the interpretation (0-1)",
        default=0.0
    )

def create_question_parser(tools: SearchTools):
    # Initialize model
    parser_model = ChatOllama(
        model="llama3.2:latest",
        temperature=0.1,
        base_url="http://localhost:11434",
        timeout=120,
    )
    
    # Create a simpler prompt that directly asks for JSON
    parser_prompt = PromptTemplate(
        template="""
        You are an expert data analyst and SQL specialist. Parse the following user question 
        and provide a structured analysis based on the available documentation context.
        
        USER QUESTION:
        {query}
        
        DOCUMENTATION CONTEXT:
        {doc_context}
        
        Guidelines:
        - Identify the business context of the question
        - Determine which tables and columns are relevant
        - Understand the metrics and dimensions needed
        - Identify any filters or time periods mentioned
        - Rephrase the question for clarity if needed
        - Suggest an approach to answer the question
        
        Return your response as a valid JSON object with the following structure:
        
        ```json
        {{
            "original_question": "The original question asked by the user",
            "rephrased_question": "The question rephrased for clarity",
            "business_context": "Business context relevant to the question",
            "relevant_tables": [
                {{
                    "name": "table_name",
                    "columns": ["column1", "column2"],
                    "description": "Description of the table's purpose"
                }}
            ],
            "query_intent": {{
                "primary_intent": "data retrieval/analysis/comparison",
                "time_period": "Time period mentioned if any",
                "filters": {{"field": "value"}},
                "metrics": ["metric1", "metric2"],
                "grouping": ["dimension1", "dimension2"]
            }},
            "suggested_approach": "Suggested approach to answer the question"
        }}
        ```
        
        Make sure your response is a valid JSON object that follows this exact structure.
        """,
        input_variables=["query", "doc_context"]
    )

    def process_question(state: ParserState) -> Dict:
        """Process and parse the user's question."""
        try:
            messages = state['messages']
            if not messages:
                return state
            
            query = messages[-1].content if isinstance(messages[-1], BaseMessage) else str(messages[-1])
            
            # Search documentation for context
            search_results = tools.search_documentation(query)
            
            # Format doc context
            doc_snippets = []
            for result in search_results.get('results', []):
                doc_snippets.append(
                    f"Content:\n{result.get('content', '')}\n"
                    f"Metadata: {result.get('metadata', {})}\n"
                )
            
            doc_context = "\n".join(doc_snippets)
            
            # Format the prompt
            formatted_prompt = parser_prompt.format(
                query=query,
                doc_context=doc_context
            )
            
            # Get response from the model
            response = parser_model.invoke(formatted_prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Extract JSON from the response
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON without the markdown code block
                json_match = re.search(r'({[\s\S]*})', response_text)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    json_str = response_text
            
            try:
                # Parse the JSON
                parsed_dict = json.loads(json_str)
                
                # Ensure all required fields are present
                if "relevant_tables" not in parsed_dict or not parsed_dict["relevant_tables"]:
                    parsed_dict["relevant_tables"] = [{"name": "unknown", "columns": []}]
                
                if "query_intent" not in parsed_dict:
                    parsed_dict["query_intent"] = {
                        "primary_intent": "unknown",
                        "metrics": []
                    }
                elif "metrics" not in parsed_dict["query_intent"]:
                    parsed_dict["query_intent"]["metrics"] = []
                
                # Ensure all required fields are present
                for field in ["original_question", "rephrased_question", "business_context", "suggested_approach"]:
                    if field not in parsed_dict:
                        parsed_dict[field] = query if field == "original_question" else ""
                
            except Exception as parse_error:
                logger.error(f"Error parsing JSON output: {str(parse_error)}")
                # Fallback to a simpler structure if parsing fails
                parsed_dict = {
                    "original_question": query,
                    "rephrased_question": query,
                    "business_context": "Error occurred during parsing",
                    "relevant_tables": [{"name": "unknown", "columns": []}],
                    "query_intent": {
                        "primary_intent": "unknown",
                        "metrics": []
                    },
                    "suggested_approach": f"Error during question parsing: {str(parse_error)}"
                }
            
            return {
                "doc_context": {"query": query, "results": search_results.get('results', [])},
                "parsed_question": parsed_dict
            }
            
        except Exception as e:
            logger.error(f"Error in question parsing: {str(e)}", exc_info=True)
            return {
                "parsed_question": {
                    "original_question": query,
                    "rephrased_question": query,
                    "business_context": "Error occurred during parsing",
                    "relevant_tables": [{"name": "unknown", "columns": []}],
                    "query_intent": {
                        "primary_intent": "unknown",
                        "metrics": []
                    },
                    "suggested_approach": f"Error during question parsing: {str(e)}"
                },
                "doc_context": {}
            }

    # Build the graph
    graph = StateGraph(ParserState)
    
    # Add nodes
    graph.add_node("question_processor", process_question)

    # Add edges
    graph.add_edge(START, "question_processor")
    graph.add_edge("question_processor", END)

    # Create SQLite saver
    db_path = str(Path(__file__).parent.parent.parent.parent / "chat_history.db")
    conn = connect(db_path, check_same_thread=False)  # Allow multi-threading
    checkpointer = SqliteSaver(conn)

    # Compile graph with SQLite checkpointer
    return graph.compile(checkpointer=checkpointer)

class QuestionParserSystem:
    def __init__(self, tools: SearchTools, feedback_timeout: int = 300):
        self.app = create_question_parser(tools)
        self.db = ChatDatabase()
        self._lock = Lock()  # Add thread lock for thread safety
        self.feedback_system = HumanFeedbackSystem(timeout_seconds=feedback_timeout)

    async def parse_question_with_feedback(self, query: str) -> Dict[str, Any]:
        """Process a query through the question parser system with human feedback."""
        try:
            # Generate unique ID for the conversation
            conversation_id = str(uuid.uuid4())
            
            # First, parse the question
            with self._lock:  # Use lock for thread safety
                result = self.app.invoke({
                    "messages": [HumanMessage(content=query)],
                    "doc_context": {},
                    "parsed_question": {},
                    "feedback_status": None
                },
                {"configurable": {"thread_id": conversation_id}}
                )
            
            # Get the parsed question
            parsed_question = result.get("parsed_question", {})
            
            # Now, get human feedback on the parsed question
            feedback_result = await self.feedback_system.process_with_feedback(
                conversation_id,
                parsed_question,
                [HumanMessage(content=query)]
            )
            
            # If feedback was rejected, re-parse with the feedback
            if feedback_result.get("status") == "rejected":
                # Extract feedback comments
                feedback = feedback_result.get("feedback", {})
                comments = feedback.get("comments", "")
                
                # Create a new query with the feedback
                enhanced_query = f"{query}\n\nAdditional context: {comments}"
                
                # Re-parse with the enhanced query
                with self._lock:
                    result = self.app.invoke({
                        "messages": [HumanMessage(content=enhanced_query)],
                        "doc_context": {},
                        "parsed_question": {},
                        "feedback_status": "reparsing"
                    },
                    {"configurable": {"thread_id": conversation_id}}
                    )
                
                # Get the updated parsed question
                parsed_question = result.get("parsed_question", {})
                
                # Update the feedback result
                feedback_result["parsed_question"] = parsed_question
                feedback_result["status"] = "reparsed"
            
            # Prepare response data
            response_data = {
                "parsed_question": feedback_result.get("parsed_question", {}),
                "doc_context": result.get("doc_context", {}),
                "feedback": feedback_result.get("feedback", {}),
                "status": feedback_result.get("status", "unknown"),
                "query": query,
                "messages": feedback_result.get("messages", []),
                "conversation_id": conversation_id  # Add conversation_id to response
            }
            
            # Save conversation to database
            with self._lock:  # Use lock for database operations
                self.db.save_conversation(conversation_id, response_data)
            
            return response_data
            
        except Exception as e:
            logger.error(f"Error in question parsing with feedback: {str(e)}", exc_info=True)
            error_response = {
                "parsed_question": {
                    "original_question": query,
                    "rephrased_question": query,
                    "business_context": "Error occurred during parsing",
                    "relevant_tables": [{"name": "unknown", "columns": []}],
                    "query_intent": {
                        "primary_intent": "unknown",
                        "metrics": []
                    },
                    "suggested_approach": f"Error during question parsing: {str(e)}"
                },
                "doc_context": {},
                "feedback": None,
                "status": "error",
                "query": query,
                "messages": [HumanMessage(content=query)],
                "conversation_id": str(uuid.uuid4())
            }
            return error_response
    
    def parse_question(self, query: str) -> Dict[str, Any]:
        """
        Synchronous version of parse_question for backward compatibility.
        This will auto-approve without waiting for human feedback.
        """
        try:
            # Generate unique ID for the conversation
            conversation_id = str(uuid.uuid4())
            
            with self._lock:  # Use lock for thread safety
                result = self.app.invoke({
                    "messages": [HumanMessage(content=query)],
                    "doc_context": {},
                    "parsed_question": {},
                    "feedback_status": "auto_approved"
                },
                {"configurable": {"thread_id": conversation_id}}
                )
            
            # Prepare response data
            response_data = {
                "parsed_question": result.get("parsed_question", {}),
                "doc_context": result.get("doc_context", {}),
                "feedback": {"approved": True, "comments": "Auto-approved"},
                "status": "approved",
                "query": query,
                "conversation_id": conversation_id
            }
            
            # Save conversation to database
            with self._lock:  # Use lock for database operations
                self.db.save_conversation(conversation_id, response_data)
            
            return response_data
            
        except Exception as e:
            logger.error(f"Error in question parsing: {str(e)}", exc_info=True)
            error_response = {
                "parsed_question": {
                    "original_question": query,
                    "rephrased_question": query,
                    "business_context": "Error occurred during parsing",
                    "relevant_tables": [{"name": "unknown", "columns": []}],
                    "query_intent": {
                        "primary_intent": "unknown",
                        "metrics": []
                    },
                    "suggested_approach": f"Error during question parsing: {str(e)}"
                },
                "doc_context": {},
                "feedback": None,
                "status": "error",
                "query": query,
                "conversation_id": str(uuid.uuid4())
            }
            return error_response
