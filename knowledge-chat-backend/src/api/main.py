from src.api.feedback_routes import router as feedback_router

# Add feedback routes
app.include_router(feedback_router, prefix="/api/feedback", tags=["feedback"]) 