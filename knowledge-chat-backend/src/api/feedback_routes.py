from fastapi import APIRouter, HTTPException, Body
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
import logging

from src.agents.data_architect.human_feedback import HumanFeedbackSystem

router = APIRouter()
logger = logging.getLogger(__name__)
feedback_system = HumanFeedbackSystem()

class FeedbackRequest(BaseModel):
    approved: bool
    comments: str = ""
    suggested_changes: Dict[str, Any] = {}

class PendingFeedbackResponse(BaseModel):
    conversation_id: str
    parsed_question: Dict[str, Any]
    timestamp: str

class FeedbackStatusResponse(BaseModel):
    status: str
    answer: Optional[str] = None
    parsed_question: Optional[Dict] = None
    sources: Optional[Dict] = None
    analysis: Optional[Dict] = None
    message: Optional[str] = None

@router.get("/pending", response_model=List[PendingFeedbackResponse])
async def get_pending_feedback():
    """Get all pending feedback requests."""
    try:
        pending = feedback_system.get_pending_feedback_requests()
        
        # Only return requests that are actually waiting for feedback
        valid_pending = {
            conv_id: data for conv_id, data in pending.items() 
            if data.get("status") == "waiting"
        }
        
        return [
            {
                "conversation_id": conv_id,
                "parsed_question": data["parsed_question"],
                "timestamp": data["timestamp"].isoformat()
            }
            for conv_id, data in valid_pending.items()
        ]
    except Exception as e:
        logger.error(f"Error getting pending feedback: {str(e)}")
        return []

@router.post("/{conversation_id}")
async def provide_feedback(conversation_id: str, feedback: FeedbackRequest = Body(...)):
    """Provide feedback for a specific conversation."""
    success = feedback_system.provide_feedback(
        conversation_id,
        {
            "approved": feedback.approved,
            "comments": feedback.comments,
            "suggested_changes": feedback.suggested_changes
        }
    )
    
    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"No pending feedback request found for conversation {conversation_id}"
        )
    
    return {"status": "success", "message": "Feedback received"}

@router.get("/feedback-status/{conversation_id}")
async def get_feedback_status(conversation_id: str):
    """Get the status of a feedback request."""
    try:
        status = feedback_system.get_feedback_status(conversation_id)
        
        if not status:
            return FeedbackStatusResponse(
                status="not_found",
                message=f"No feedback request found for conversation {conversation_id}"
            )
        
        return FeedbackStatusResponse(
            status=status.get("status", "unknown"),
            answer=status.get("answer"),
            parsed_question=status.get("parsed_question"),
            sources=status.get("sources"),
            analysis=status.get("analysis"),
            message=status.get("message")
        )
    except Exception as e:
        logger.error(f"Error getting feedback status: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) 