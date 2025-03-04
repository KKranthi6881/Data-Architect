from typing import Dict, List, Any, Annotated, Sequence, TypedDict, Optional, Union, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from pathlib import Path
import uuid
import logging
from sqlite3 import connect
from threading import Lock
import asyncio
from datetime import datetime, timedelta

from src.db.database import ChatDatabase

# Set up logger
logger = logging.getLogger(__name__)

# Define state type for human feedback system
class FeedbackState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "The messages in the conversation"]
    parsed_question: Annotated[Dict, "Structured parsed question output"]
    feedback: Annotated[Optional[Dict], "Human feedback on the parsed question"]
    status: Annotated[str, "Current status of the feedback process"]
    timeout: Annotated[Optional[datetime], "Timeout for waiting for feedback"]

# Define feedback response model
class FeedbackResponse(BaseModel):
    approved: bool = Field(description="Whether the parsed question is approved")
    comments: Optional[str] = Field(description="Additional comments or corrections", default=None)
    suggested_changes: Optional[Dict[str, Any]] = Field(description="Suggested changes to the parsed question", default=None)

class HumanFeedbackSystem:
    def __init__(self, timeout_seconds: int = 600):
        self.db = ChatDatabase()
        self._lock = Lock()
        self.pending_feedback = {}
        self.feedback_status = {}
        self.original_requests = {}  # Store original requests
        self.timeout_seconds = timeout_seconds
        self.feedback_callbacks = {}

    def submit_for_feedback(self, feedback_request: Dict) -> str:
        """Submit a request for human feedback."""
        conversation_id = str(uuid.uuid4())
        
        with self._lock:
            # Store the original request
            self.original_requests[conversation_id] = feedback_request
            
            self.pending_feedback[conversation_id] = {
                "parsed_question": feedback_request.get("parsed_question", {}),
                "doc_context": feedback_request.get("doc_context", {}),
                "timestamp": feedback_request.get("timestamp", datetime.now()),
                "status": "waiting",
                "message": feedback_request.get("message", ""),  # Store original message
                "response": feedback_request.get("response", "")  # Store response
            }
            self.feedback_status[conversation_id] = {
                "status": "pending",
                "feedback": None,
                "timestamp": datetime.now()
            }
        
        return conversation_id

    def provide_feedback(self, conversation_id: str, feedback: Dict) -> bool:
        """Provide feedback for a conversation."""
        with self._lock:
            if conversation_id not in self.pending_feedback:
                return False
            
            self.feedback_status[conversation_id] = {
                "status": "completed" if feedback.get("approved") else "rejected",
                "feedback": feedback,
                "timestamp": datetime.now()
            }
            
            # If there's a callback waiting, resolve it
            if conversation_id in self.feedback_callbacks:
                future = self.feedback_callbacks[conversation_id]
                if not future.done():
                    future.set_result(feedback)
            
            return True

    def get_feedback_status(self, conversation_id: str) -> Optional[Dict]:
        """Get the current status of feedback for a conversation."""
        return self.feedback_status.get(conversation_id)

    async def wait_for_feedback(self, conversation_id: str) -> Dict:
        """Wait for feedback to be provided."""
        if conversation_id not in self.pending_feedback:
            raise ValueError(f"No pending feedback request for conversation {conversation_id}")
        
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self.feedback_callbacks[conversation_id] = future
        
        try:
            return await asyncio.wait_for(future, timeout=self.timeout_seconds)
        except asyncio.TimeoutError:
            return {
                "approved": True,
                "comments": "Auto-approved due to timeout",
                "suggested_changes": None
            }
        finally:
            if conversation_id in self.feedback_callbacks:
                del self.feedback_callbacks[conversation_id]

    def get_pending_feedback_requests(self) -> Dict[str, Dict]:
        """Get all pending feedback requests."""
        with self._lock:
            return {
                conv_id: data for conv_id, data in self.pending_feedback.items()
                if self.feedback_status.get(conv_id, {}).get("status") == "pending"
            }

    def create_feedback_graph(self):
        """Create a graph for the human feedback process."""
        
        # Define the router function to determine next steps based on feedback
        def router(state: FeedbackState) -> Literal["approved", "rejected", "waiting"]:
            """Route based on feedback status."""
            if state.get("status") == "timeout":
                # If timeout occurred, treat as approved to continue the flow
                return "approved"
            
            feedback = state.get("feedback")
            if not feedback:
                return "waiting"
                
            if feedback.get("approved", False):
                return "approved"
            else:
                return "rejected"
        
        # Function to format parsed question for human review
        def format_for_review(state: FeedbackState) -> Dict:
            """Format the parsed question for human review."""
            parsed = state.get("parsed_question", {})
            
            # Create a user-friendly representation of the parsed question
            formatted_tables = []
            for table in parsed.get("relevant_tables", []):
                table_info = f"- {table.get('name')}: {', '.join(table.get('columns', []))}"
                if table.get('description'):
                    table_info += f" ({table.get('description')})"
                formatted_tables.append(table_info)
            
            query_intent = parsed.get("query_intent", {})
            metrics = ", ".join(query_intent.get("metrics", []))
            grouping = ", ".join(query_intent.get("grouping", [])) if query_intent.get("grouping") else "None"
            
            review_message = f"""
            ## Question Parsing Results
            
            **Original Question**: {parsed.get('original_question', 'N/A')}
            
            **Rephrased Question**: {parsed.get('rephrased_question', 'N/A')}
            
            **Business Context**: {parsed.get('business_context', 'N/A')}
            
            **Relevant Tables**:
            {chr(10).join(formatted_tables) if formatted_tables else "None identified"}
            
            **Query Intent**:
            - Primary Intent: {query_intent.get('primary_intent', 'N/A')}
            - Time Period: {query_intent.get('time_period', 'N/A')}
            - Metrics: {metrics if metrics else 'None'}
            - Grouping: {grouping}
            
            **Suggested Approach**: {parsed.get('suggested_approach', 'N/A')}
            
            Is this interpretation correct? Please approve or suggest changes.
            """
            
            # Add the review message to the state
            messages = list(state.get("messages", []))
            messages.append(AIMessage(content=review_message))
            
            # Set timeout for feedback
            timeout = datetime.now() + timedelta(seconds=self.timeout_seconds)
            
            return {
                "messages": messages,
                "status": "waiting",
                "timeout": timeout
            }
        
        # Function to handle approved feedback
        def handle_approved(state: FeedbackState) -> Dict:
            """Handle approved feedback."""
            messages = list(state.get("messages", []))
            messages.append(AIMessage(content="Thank you for confirming. Proceeding with the analysis."))
            
            return {
                "messages": messages,
                "status": "approved"
            }
        
        # Function to handle rejected feedback
        def handle_rejected(state: FeedbackState) -> Dict:
            """Handle rejected feedback and prepare for re-parsing."""
            feedback = state.get("feedback", {})
            comments = feedback.get("comments", "")
            
            messages = list(state.get("messages", []))
            messages.append(AIMessage(content=f"I'll adjust the interpretation based on your feedback: {comments}"))
            
            return {
                "messages": messages,
                "status": "rejected",
                "parsed_question": {}  # Clear the parsed question to trigger re-parsing
            }
        
        # Build the graph
        graph = StateGraph(FeedbackState)
        
        # Add nodes
        graph.add_node("format_review", format_for_review)
        graph.add_node("handle_approved", handle_approved)
        graph.add_node("handle_rejected", handle_rejected)
        
        # Add conditional edges
        graph.add_conditional_edges(
            "format_review",
            router,
            {
                "approved": "handle_approved",
                "rejected": "handle_rejected",
                "waiting": END  # End the graph to wait for human feedback
            }
        )
        
        # Add remaining edges
        graph.add_edge(START, "format_review")
        graph.add_edge("handle_approved", END)
        graph.add_edge("handle_rejected", END)
        
        # Create SQLite saver
        db_path = str(Path(__file__).parent.parent.parent.parent / "chat_history.db")
        conn = connect(db_path, check_same_thread=False)
        checkpointer = SqliteSaver(conn)
        
        # Compile graph with SQLite checkpointer
        return graph.compile(checkpointer=checkpointer)
    
    async def process_with_feedback(self, conversation_id: str, parsed_question: Dict, 
                                   original_messages: List[BaseMessage]) -> Dict[str, Any]:
        """Process the parsed question with human feedback."""
        try:
            # Create the feedback graph
            feedback_graph = self.create_feedback_graph()
            
            # Initialize the state
            initial_state = {
                "messages": original_messages,
                "parsed_question": parsed_question,
                "feedback": None,
                "status": "new",
                "timeout": None
            }
            
            # Start the feedback process
            with self._lock:
                result = feedback_graph.invoke(
                    initial_state,
                    {"configurable": {"thread_id": conversation_id}}
                )
            
            # Register this conversation as waiting for feedback
            self.pending_feedback[conversation_id] = {
                "timestamp": datetime.now(),
                "parsed_question": parsed_question,
                "status": "waiting"
            }
            
            # Create a future to wait for feedback
            feedback_future = asyncio.Future()
            self.feedback_callbacks[conversation_id] = feedback_future
            
            # Wait for feedback with timeout
            try:
                feedback_result = await asyncio.wait_for(
                    feedback_future, 
                    timeout=self.timeout_seconds
                )
                
                # Process the feedback
                if feedback_result.get("approved", False):
                    final_state = {
                        "messages": result.get("messages", []),
                        "parsed_question": parsed_question,
                        "feedback": feedback_result,
                        "status": "approved",
                        "timeout": None
                    }
                else:
                    final_state = {
                        "messages": result.get("messages", []),
                        "parsed_question": parsed_question,
                        "feedback": feedback_result,
                        "status": "rejected",
                        "timeout": None
                    }
                
                # Continue the graph with the feedback
                with self._lock:
                    final_result = feedback_graph.invoke(
                        final_state,
                        {"configurable": {"thread_id": conversation_id}}
                    )
                
                return {
                    "status": final_result.get("status", "unknown"),
                    "parsed_question": parsed_question,
                    "feedback": feedback_result,
                    "messages": final_result.get("messages", [])
                }
                
            except asyncio.TimeoutError:
                # Handle timeout - proceed with the original parsed question
                logger.warning(f"Feedback timeout for conversation {conversation_id}")
                
                timeout_state = {
                    "messages": result.get("messages", []),
                    "parsed_question": parsed_question,
                    "feedback": {"approved": True, "comments": "Auto-approved due to timeout"},
                    "status": "timeout",
                    "timeout": None
                }
                
                # Continue the graph with timeout
                with self._lock:
                    timeout_result = feedback_graph.invoke(
                        timeout_state,
                        {"configurable": {"thread_id": conversation_id}}
                    )
                
                return {
                    "status": "approved",  # Auto-approve on timeout
                    "parsed_question": parsed_question,
                    "feedback": {"approved": True, "comments": "Auto-approved due to timeout"},
                    "messages": timeout_result.get("messages", [])
                }
                
        except Exception as e:
            logger.error(f"Error in feedback process: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "parsed_question": parsed_question,
                "feedback": None,
                "error": str(e),
                "messages": original_messages
            }
        finally:
            # Clean up
            if conversation_id in self.feedback_callbacks:
                del self.feedback_callbacks[conversation_id]
            if conversation_id in self.pending_feedback:
                del self.pending_feedback[conversation_id]

    def get_original_request(self, conversation_id: str) -> Optional[Dict]:
        """Get the original request for a conversation."""
        return self.original_requests.get(conversation_id) 