from fastapi import FastAPI, UploadFile, File, Request, HTTPException, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
import tempfile
from pathlib import Path
from .utils import ChromaDBManager
from .tools import SearchTools
from .agents.code_research import SimpleAnalysisSystem
from .agents.data_architect.question_parser import QuestionParserSystem
import logging
from pydantic import BaseModel
from .db.database import ChatDatabase
import streamlit as st
from datetime import datetime
from langchain_community.utilities import SQLDatabase
from src.tools import SearchTools
from typing import Optional, List, Dict, Any

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Create required directories
for directory in ["static", "templates", "uploads"]:
    (BASE_DIR / directory).mkdir(exist_ok=True)

# Create a templates directory and mount it
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# Initialize ChromaDB manager
db_manager = ChromaDBManager(persist_directory=str(BASE_DIR / "chroma_db"))

# Initialize tools and analysis system
code_search_tools = SearchTools(db_manager)
analysis_system = SimpleAnalysisSystem(code_search_tools)

# Initialize question parser system
question_parser_system = QuestionParserSystem(code_search_tools, feedback_timeout=120)

# Initialize database
chat_db = ChatDatabase()

# Add new model for code analysis requests
class CodeAnalysisRequest(BaseModel):
    query: str

# Add new model for analysis requests
class AnalysisRequest(BaseModel):
    query: str

# Add new model for GitHub repository
class GitHubRepoRequest(BaseModel):
    repo_url: str
    username: str = ""
    token: str = ""

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None  # Add conversation ID support
    context: Optional[Dict[str, Any]] = None  # Add context support
    wait_for_feedback: bool = False  # New parameter to control feedback behavior

class FeedbackRequest(BaseModel):
    conversation_id: str
    approved: bool
    comments: str = ""
    suggested_changes: Dict[str, Any] = {}

class ChatResponse(BaseModel):
    answer: str
    conversation_id: str
    sources: Dict[str, Any]
    analysis: Dict[str, Any]
    technical_details: Optional[str] = None
    business_context: Optional[str] = None
    data_lineage: Optional[Dict[str, Any]] = None
    parsed_question: Optional[Dict[str, Any]] = None
    feedback_required: bool = False
    feedback_status: Optional[str] = None

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html", 
        {"request": request}
    )

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Log the upload attempt
        logger.info(f"Attempting to upload file: {file.filename}")
        
        # Create uploads directory if it doesn't exist
        upload_dir = BASE_DIR / "uploads"
        upload_dir.mkdir(exist_ok=True)
        
        # Save the uploaded file
        file_path = upload_dir / file.filename
        logger.info(f"Saving file to: {file_path}")
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process the file based on its extension
        file_extension = os.path.splitext(file.filename)[1].lower()
        
        logger.info(f"Processing file with extension: {file_extension}")
        
        if file_extension == '.pdf':
            logger.info("Processing PDF file...")
            docs = db_manager.process_document(str(file_path), 'pdf')
            collection_name = "pdf_documents"
        elif file_extension == '.sql':
            logger.info("Processing SQL file...")
            docs = db_manager.process_code(str(file_path), 'sql')
            collection_name = "sql_documents"
        elif file_extension == '.py':
            logger.info("Processing Python file...")
            docs = db_manager.process_code(str(file_path), 'python')
            collection_name = "sql_documents"  # Using the same collection for code
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")
        
        # Add documents to the appropriate collection
        result = db_manager.add_documents(
            collection_name,
            docs,
            metadata={
                "source": file.filename,
                "upload_time": datetime.now().isoformat()
            }
        )
        
        return JSONResponse({
            "status": "success",
            "filename": file.filename,
            "collection": collection_name,
            "document_count": len(docs),
            "message": f"File processed and added to {collection_name}"
        })
        
    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/github/")
async def add_github_repo(repo_request: GitHubRepoRequest):
    """Add a GitHub repository to the knowledge base"""
    try:
        logger.info(f"Processing GitHub repository: {repo_request.repo_url}")
        
        # Process the GitHub repository
        result = db_manager.add_github_repo(
            repo_request.repo_url,
            {
                "username": repo_request.username,
                "token": repo_request.token,
                "upload_time": datetime.now().isoformat()
            }
        )
        
        return JSONResponse({
            "status": "success",
            "repo_url": repo_request.repo_url,
            "message": "GitHub repository processed and added to knowledge base",
            "repo_info": result.get("repo_metadata", {})
        })
        
    except Exception as e:
        logger.error(f"Error processing GitHub repository: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/collections/")
async def get_collections():
    try:
        collections = db_manager.get_collection_names()
        return JSONResponse({
            "status": "success",
            "collections": collections
        })
    except Exception as e:
        logger.error(f"Error getting collections: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/files/")
async def list_files():
    """List all files in the knowledge base"""
    try:
        # Get all collection names - in v0.6.0 list_collections returns only names
        collection_names = db_manager.client.list_collections()
        
        # Initialize results
        files = []
        
        # Collection types we're interested in
        document_collections = ["sql_documents", "pdf_documents", "github_documents"]
        
        # Process each collection
        for collection_name in collection_names:
            # Skip collections that don't contain documents
            if collection_name not in document_collections:
                continue
            
            # Get the collection
            coll = db_manager.get_or_create_collection(collection_name)
            
            # Get all items
            items = coll.get(include=["metadatas"])
            
            if not items or not items["metadatas"]:
                continue
            
            # Process metadatas to extract file information
            for i, metadata in enumerate(items["metadatas"]):
                # Skip if already processed this file
                file_path = metadata.get("file_path", metadata.get("source", ""))
                
                # Determine file type
                file_type = "unknown"
                if collection_name == "sql_documents":
                    file_type = "sql"
                elif collection_name == "pdf_documents":
                    file_type = "pdf"
                elif collection_name == "github_documents":
                    file_type = "github"
                
                # Create file entry if not exists
                file_exists = any(f["name"] == file_path and f["type"] == file_type for f in files)
                
                if not file_exists and file_path:
                    files.append({
                        "name": file_path,
                        "type": file_type,
                        "size": metadata.get("size", 0),
                        "modified": metadata.get("upload_time", datetime.now().isoformat()),
                        "collection": collection_name
                    })
        
        return JSONResponse({
            "status": "success",
            "files": files
        })
        
    except Exception as e:
        logger.error(f"Error listing files: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# Add new endpoints for code analysis
@app.post("/analyze/")
async def analyze_code(request: CodeAnalysisRequest):
    """
    Endpoint for code analysis using the new agent system
    """
    try:
        logger.info(f"Analyzing query: {request.query}")
        
        # Use the new analysis system
        result = analysis_system.analyze(request.query)
        
        return JSONResponse({
            "status": "success",
            "output": result.get("output", "No output available"),
            "code_context": result.get("code_context", {})
        })
        
    except Exception as e:
        logger.error(f"Error in code analysis: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e)
            }
        )

@app.get("/schema")
async def get_schema():
    try:
        schema = code_search_tools.get_database_schema()
        return {"schema": schema}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/")
async def chat(request: ChatRequest):
    """Process a chat message."""
    try:
        logger.info(f"Processing chat request: {request.message}")
        
        if request.wait_for_feedback:
            logger.info("Using question parser with human feedback")
            result = await question_parser_system.parse_question_with_feedback(request.message)
            
            # Check if we're waiting for feedback
            if result.get("status") == "waiting":
                logger.info("Waiting for human feedback")
                parsed_question = result.get("parsed_question", {})
                
                # Format the interpretation message
                interpretation_msg = (
                    f"Based on your question: '{parsed_question.get('original_question')}'\n\n"
                    f"I understand you're asking about: {parsed_question.get('rephrased_question')}\n\n"
                    f"Business Context: {parsed_question.get('business_context')}\n\n"
                )
                
                # Add alternative interpretations if available
                alternatives = parsed_question.get('alternative_interpretations', [])
                if alternatives:
                    interpretation_msg += "\nAlternative interpretations:\n"
                    for i, alt in enumerate(alternatives, 1):
                        interpretation_msg += f"{i}. {alt}\n"
                
                return JSONResponse(
                    content={
                        "answer": interpretation_msg,
                        "conversation_id": result.get("conversation_id", ""),
                        "sources": {},
                        "analysis": {},
                        "parsed_question": parsed_question,
                        "feedback_required": True,
                        "feedback_status": "waiting",
                        "confidence_score": parsed_question.get("confidence_score", 0.0)
                    },
                    status_code=200
                )
            
            # Process approved/updated interpretation
            final_answer = f"Based on the confirmed interpretation, here's what I found:\n\n{result.get('parsed_question', {}).get('suggested_approach', '')}"
            
            return JSONResponse(
                content={
                    "answer": final_answer,
                    "conversation_id": result.get("conversation_id", ""),
                    "sources": result.get("doc_context", {}),
                    "analysis": {
                        "business_context": result.get("parsed_question", {}).get("business_context", ""),
                        "tables": result.get("parsed_question", {}).get("relevant_tables", []),
                        "intent": result.get("parsed_question", {}).get("query_intent", {})
                    },
                    "parsed_question": result.get("parsed_question", {}),
                    "feedback_required": False,
                    "feedback_status": "completed"
                },
                status_code=200
            )
        else:
            # Use the synchronous version without human feedback
            logger.info("Using question parser without human feedback")
            result = question_parser_system.parse_question(request.message)
            
            final_answer = f"Based on my understanding of your question, here's what I found:\n\n{result.get('parsed_question', {}).get('suggested_approach', '')}"
            
            return JSONResponse(
                content={
                    "answer": final_answer,
                    "conversation_id": result.get("conversation_id", ""),
                    "sources": result.get("doc_context", {}),
                    "analysis": {
                        "business_context": result.get("parsed_question", {}).get("business_context", ""),
                        "tables": result.get("parsed_question", {}).get("relevant_tables", []),
                        "intent": result.get("parsed_question", {}).get("query_intent", {})
                    },
                    "parsed_question": result.get("parsed_question", {}),
                    "feedback_required": False,
                    "feedback_status": "auto_approved"
                },
                status_code=200
            )
            
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
        return JSONResponse(
            content={
                "error": str(e),
                "feedback_required": False,
                "feedback_status": "error"
            },
            status_code=500
        )

@app.post("/feedback/")
async def provide_feedback(request: FeedbackRequest):
    """Provide feedback for a parsed question."""
    try:
        from src.agents.data_architect.human_feedback import HumanFeedbackSystem
        
        # Get the feedback system
        feedback_system = HumanFeedbackSystem()
        
        # Provide feedback
        success = feedback_system.provide_feedback(
            request.conversation_id,
            {
                "approved": request.approved,
                "comments": request.comments,
                "suggested_changes": request.suggested_changes
            }
        )
        
        if not success:
            return JSONResponse(
                status_code=404,
                content={
                    "status": "error",
                    "message": f"No pending feedback request found for conversation {request.conversation_id}"
                }
            )
        
        return JSONResponse(
            content={
                "status": "success",
                "message": "Feedback received"
            }
        )
        
    except Exception as e:
        logger.error(f"Error providing feedback: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e)
            }
        )

@app.get("/pending-feedback/")
async def get_pending_feedback():
    """Get all pending feedback requests."""
    try:
        from src.agents.data_architect.human_feedback import HumanFeedbackSystem
        
        # Get the feedback system
        feedback_system = HumanFeedbackSystem()
        
        # Get pending feedback requests
        pending = feedback_system.get_pending_feedback_requests()
        
        return JSONResponse(
            content={
                "status": "success",
                "pending_requests": [
                    {
                        "conversation_id": conv_id,
                        "parsed_question": data["parsed_question"],
                        "timestamp": data["timestamp"].isoformat()
                    }
                    for conv_id, data in pending.items()
                ]
            }
        )
        
    except Exception as e:
        logger.error(f"Error getting pending feedback: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e)
            }
        )

@app.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get conversation history."""
    try:
        history = chat_db.get_conversation(conversation_id)
        return {"history": history}
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Conversation not found: {str(e)}")

def render_history_sidebar():
    """Render the conversation history in the sidebar"""
    with st.sidebar:
        st.header("Conversation History")
        
        # Get history from database
        history = chat_db.get_conversation_history()
        
        if not history:
            st.info("No conversation history yet")
            return
            
        for item in history:
            # Create an expander for each conversation
            with st.expander(f"Query: {item['query'][:50]}..."):
                st.text(f"Time: {item['timestamp']}")
                
                # Show the query
                st.markdown("**Query:**")
                st.text(item['query'])
                
                # Show the response
                st.markdown("**Response:**")
                st.json(item['output'])
                
                # Show code context if available
                if item['code_context']:
                    st.markdown("**Code Context:**")
                    st.json(item['code_context'])
                
                # Add a divider between conversations
                st.divider()

def chat_interface():
    """Chat interface for the Streamlit app"""
    st.markdown("### Ask a Question")
    query = st.text_area("Enter your question:", height=100)
    
    if st.button("Submit"):
        if not query:
            st.warning("Please enter a question first.")
            return
            
        with st.spinner("Analyzing..."):
            result = analyze_code(query)
            
            if result:
                st.markdown("#### Response")
                st.markdown(result.get("output", "No output available"))
                
                if result.get("code_context"):
                    st.markdown("#### Code Context")
                    st.json(result["code_context"])

def file_upload_interface():
    """File upload interface for the Streamlit app"""
    st.markdown("### Upload Files")
    uploaded_file = st.file_uploader(
        "Choose a file to upload (.py, .sql, .pdf)", 
        type=["py", "sql", "pdf"]
    )
    
    if uploaded_file and st.button("Process File"):
        with st.spinner("Processing file..."):
            result = upload_file(uploaded_file)
            if result:
                st.success(f"File processed: {result.get('filename')}")
                st.json(result)

def main():
    st.set_page_config(page_title="Code Analysis Assistant", layout="wide")
    
    # Add custom CSS
    st.markdown("""
        <style>
        .stSidebar {
            background-color: #f5f5f5;
        }
        .stExpander {
            background-color: white;
            border-radius: 4px;
            margin-bottom: 10px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Render the history sidebar
    render_history_sidebar()
    
    # Main content
    st.title("Code Analysis Assistant")
    
    # Your existing tabs
    tab1, tab2 = st.tabs(["Chat", "File Upload"])
    
    with tab1:
        # Your existing chat interface
        chat_interface()
    
    with tab2:
        # Your existing file upload interface
        file_upload_interface()

if __name__ == "__main__":
    main() 