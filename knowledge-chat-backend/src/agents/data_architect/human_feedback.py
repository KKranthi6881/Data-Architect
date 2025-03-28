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
import json  # Add this import at the top of the file

from src.db.database import ChatDatabase
from src.agents.data_architect.github_search import GitHubCodeSearchAgent
from src.agents.data_architect.schema_search import SchemaSearchAgent
from src.agents.data_architect.data_architect import DataArchitectAgent

# Set up logger
logger = logging.getLogger(__name__)

# Define state type for human feedback system
class FeedbackState(TypedDict):
    """State management for human feedback system"""
    messages: Annotated[Sequence[BaseMessage], "Conversation messages"]
    business_analysis: Annotated[Dict, "Business analysis results"]
    feedback: Annotated[Optional[Dict], "Human feedback"]
    status: Annotated[str, "Feedback status"]
    confidence_score: Annotated[float, "Confidence in analysis"]

# Define feedback response model
class FeedbackResponse(BaseModel):
    approved: bool = Field(description="Whether the parsed question is approved")
    comments: Optional[str] = Field(description="Additional comments or corrections", default=None)
    suggested_changes: Optional[Dict[str, Any]] = Field(description="Suggested changes to the parsed question", default=None)

class HumanFeedbackSystem:
    def __init__(self):
        self.pending_feedback = {}
        self.processed_feedback = {}
        self.logger = logging.getLogger(__name__)
        self.db = ChatDatabase()
        self._lock = Lock()
        self.feedback_callbacks = {}
        self.timeout_seconds = 300  # 5 minutes timeout

    async def process_feedback(self, 
                             feedback_id: str, 
                             approved: bool, 
                             comments: Optional[str] = None,
                             suggested_changes: Optional[Dict[str, Any]] = None,
                             message_id: Optional[str] = None):
        """Process feedback for a parsed question"""
        try:
            self.logger.info(f"Processing feedback: ID={feedback_id}, approved={approved}, comments={comments}")
            
            # Get the feedback request
            feedback_request = self.pending_feedback.get(feedback_id)
            if not feedback_request:
                self.logger.warning(f"No feedback request found for ID: {feedback_id}")
                return {"status": "error", "message": "Feedback request not found"}
            
            # Get conversation_id from feedback request
            conversation_id = feedback_request.get("conversation_id")
            if not conversation_id:
                self.logger.warning(f"No conversation_id found for feedback ID: {feedback_id}")
                return {"status": "error", "message": "Conversation ID not found"}
            
            # Process based on approval status
            if approved:
                self.logger.info(f"Processing approved feedback for conversation {conversation_id}")
                # Process approved feedback
                architect_result = await self.process_approved_feedback(
                    feedback_id=feedback_id,
                    conversation_id=conversation_id
                )
                
                if architect_result.get('status') == 'error':
                    self.logger.error(f"Error in architect processing: {architect_result.get('error')}")
                    return {
                        "status": "error",
                        "message": f"Architect processing failed: {architect_result.get('error')}",
                        "feedback_id": feedback_id
                    }
                
                return {
                    "status": "success",
                    "feedback_id": feedback_id,
                    "approved": approved,
                    "architect_response": architect_result.get('architect_response'),
                    "message": "Feedback processed successfully"
                }
            else:
                # Handle rejected feedback
                # Update feedback status in the database
                conversation = self.db.get_conversation(conversation_id)
                if conversation:
                    conversation_data = {
                        "query": conversation.get("query", ""),
                        "output": conversation.get("output", ""),
                        "technical_details": conversation.get("technical_details", ""),
                        "code_context": conversation.get("code_context", "{}"),
                        "feedback_status": "rejected",
                        "feedback_comments": comments,
                        "thread_id": conversation.get("thread_id")
                    }
                    self.db.save_conversation(conversation_id, conversation_data)
                
                return {
                    "status": "success",
                    "feedback_id": feedback_id,
                    "approved": approved,
                    "message": "Feedback recorded as rejected"
                }
        
        except Exception as e:
            self.logger.error(f"Error processing feedback: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

    def get_pending_feedback_requests(self) -> Dict:
        """Get all pending feedback requests"""
        return self.pending_feedback

    def get_feedback_status(self, feedback_id: str) -> Dict[str, Any]:
        """Get the status of a feedback request"""
        if feedback_id in self.pending_feedback:
            return self.pending_feedback[feedback_id]
        return None

    def add_feedback_request(self, feedback_id: str, business_analysis: Dict, conversation_id: str = None):
        """Add new business analysis feedback request"""
        self.pending_feedback[feedback_id] = {
            "business_analysis": business_analysis,
            "conversation_id": conversation_id,
            "timestamp": datetime.now().isoformat(),
            "status": "pending",
            "confidence_score": business_analysis.get("confidence_score", 0.0),
            "parsed_question": business_analysis  # Store the parsed question for later use
        }
        
        # Create a future for this feedback request
        if feedback_id not in self.feedback_callbacks:
            loop = asyncio.get_event_loop()
            self.feedback_callbacks[feedback_id] = loop.create_future()
        
        # Set timeout for auto-approval if timeout_seconds is positive
        if hasattr(self, 'timeout_seconds') and self.timeout_seconds > 0:
            self._set_timeout(feedback_id, self.timeout_seconds)
        
        self.logger.info(f"Added feedback request: {feedback_id} for conversation: {conversation_id}")

    def submit_for_feedback(self, request: Dict) -> str:
        """Submit a request for feedback."""
        with self._lock:
            feedback_id = str(uuid.uuid4())
            logger.info(f"Creating new feedback request with ID: {feedback_id}")
            
            self.pending_feedback[feedback_id] = {
                "message": request.get("message", ""),
                "parsed_question": request.get("parsed_question", {}),
                "doc_context": request.get("doc_context", {}),
                "timestamp": request.get("timestamp", datetime.now()),
                "response": request.get("response", {}),
                "status": "pending"
            }
            return feedback_id

    def provide_feedback(self, feedback_id: str, feedback: Dict) -> bool:
        """Provide feedback for a conversation."""
        with self._lock:
            logger.info(f"Processing feedback for ID: {feedback_id}")
            if feedback_id not in self.pending_feedback:
                logger.error(f"No pending feedback found for ID: {feedback_id}")
                return False
            
            self.pending_feedback[feedback_id].update({
                "feedback": feedback,
                "status": "completed" if feedback.get("approved") else "needs_improvement"
            })
            return True

    def get_original_request(self, feedback_id: str) -> Optional[Dict]:
        """Get the original request for a feedback ID."""
        with self._lock:
            return self.pending_feedback.get(feedback_id)

    async def wait_for_feedback(self, feedback_id: str) -> Dict:
        """Wait for feedback to be provided."""
        if feedback_id not in self.pending_feedback:
            raise ValueError(f"No pending feedback request for ID {feedback_id}")
        
        if feedback_id not in self.feedback_callbacks:
            loop = asyncio.get_event_loop()
            self.feedback_callbacks[feedback_id] = loop.create_future()
        
        try:
            return await asyncio.wait_for(self.feedback_callbacks[feedback_id], timeout=self.timeout_seconds)
        except asyncio.TimeoutError:
            return {
                "approved": True,
                "comments": "Auto-approved due to timeout",
                "suggested_changes": None
            }
        finally:
            if feedback_id in self.feedback_callbacks:
                del self.feedback_callbacks[feedback_id]

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
        return graph.compile()
    
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

    async def process_approved_feedback(self, feedback_id: str, conversation_id: str) -> Dict[str, Any]:
        """Process approved feedback by generating architect response"""
        try:
            self.logger.info(f"Starting architect processing for conversation {conversation_id}")
            
            # Get the conversation data
            conversation_data = self.db.get_conversation(conversation_id)
            if not conversation_data:
                raise ValueError(f"No conversation found for ID: {conversation_id}")
            
            # Create data architect agent
            architect_agent = DataArchitectAgent()
            
            # Get the original question and technical details
            original_question = conversation_data.get('query', '')
            technical_details = {}
            
            try:
                if isinstance(conversation_data.get('technical_details'), str):
                    technical_details = json.loads(conversation_data.get('technical_details', '{}'))
                else:
                    technical_details = conversation_data.get('technical_details', {})
            except json.JSONDecodeError as e:
                self.logger.error(f"Error parsing technical details: {e}")
                technical_details = {}

            self.logger.info(f"Generating architect response for question: {original_question}")
            
            # Generate architect response
            architect_response = architect_agent.generate_response(
                parsed_question=technical_details,
                schema_results=[],  # Add schema results if available
                code_results=[],    # Add code results if available
                original_question=original_question
            )
            
            if not architect_response:
                raise ValueError("Failed to generate architect response")

            self.logger.info(f"Generated architect response: {architect_response}")
            
            # Update conversation data
            updated_data = {
                'query': original_question,
                'technical_details': json.dumps(technical_details),
                'architect_response': json.dumps(architect_response),
                'feedback_status': 'approved',
                'last_updated': datetime.now().isoformat(),
                'thread_id': conversation_data.get('thread_id', conversation_id)
            }
            
            # Save to database
            self.db.save_conversation(conversation_id, updated_data)
            
            self.logger.info(f"Saved architect response for conversation {conversation_id}")
            
            return {
                'status': 'success',
                'architect_response': architect_response,
                'conversation_id': conversation_id
            }
            
        except Exception as e:
            self.logger.error(f"Error processing approved feedback: {e}", exc_info=True)
            return {
                'status': 'error',
                'error': str(e),
                'conversation_id': conversation_id
            }

    def _set_timeout(self, feedback_id: str, timeout_seconds: int):
        """Set a timeout for auto-approval of feedback"""
        async def timeout_callback():
            try:
                await asyncio.sleep(timeout_seconds)
                # Check if feedback is still pending
                if feedback_id in self.pending_feedback and self.pending_feedback[feedback_id].get("status") == "pending":
                    self.logger.info(f"Auto-approving feedback {feedback_id} due to timeout")
                    
                    # Get conversation_id
                    conversation_id = self.pending_feedback[feedback_id].get("conversation_id")
                    parsed_question = self.pending_feedback[feedback_id].get("parsed_question", {})
                    
                    if conversation_id:
                        # Auto-approve the feedback
                        await self.process_feedback(
                            feedback_id=feedback_id,
                            approved=True,
                            comments="Auto-approved due to timeout"
                        )
                        
                        # If there's a callback waiting, resolve it
                        if feedback_id in self.feedback_callbacks and not self.feedback_callbacks[feedback_id].done():
                            self.feedback_callbacks[feedback_id].set_result({
                                "approved": True,
                                "comments": "Auto-approved due to timeout"
                            })
            except Exception as e:
                self.logger.error(f"Error in timeout callback for {feedback_id}: {e}")
        
        # Create and start the timeout task
        try:
            loop = asyncio.get_event_loop()
            asyncio.create_task(timeout_callback())
        except Exception as e:
            self.logger.error(f"Error setting timeout for {feedback_id}: {e}")