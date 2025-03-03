from fastapi import APIRouter, HTTPException, Body
from typing import Dict, Any, List
from pydantic import BaseModel

from src.agents.data_architect.human_feedback import HumanFeedbackSystem

router = APIRouter()
feedback_system = HumanFeedbackSystem()

class FeedbackRequest(BaseModel):
    approved: bool
    comments: str = ""
    suggested_changes: Dict[str, Any] = {}

class PendingFeedbackResponse(BaseModel):
    conversation_id: str
    parsed_question: Dict[str, Any]
    timestamp: str

@router.get("/pending", response_model=List[PendingFeedbackResponse])
async def get_pending_feedback():
    """Get all pending feedback requests."""
    pending = feedback_system.get_pending_feedback_requests()
    
    return [
        {
            "conversation_id": conv_id,
            "parsed_question": data["parsed_question"],
            "timestamp": data["timestamp"].isoformat()
        }
        for conv_id, data in pending.items()
    ]

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