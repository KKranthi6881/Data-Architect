from fastapi import FastAPI, UploadFile, File, Request, HTTPException, Form, Depends
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
from typing import Optional, List, Dict, Any, Union
from src.agents.data_architect.human_feedback import HumanFeedbackSystem
from src.api.feedback_routes import router as feedback_router
import re
import uuid
import json

app = FastAPI(title="Knowledge Chat API")

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
question_parser_system = QuestionParserSystem(
    tools=code_search_tools,
    feedback_timeout=120
)

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

# Add FeedbackRequest model
class FeedbackRequest(BaseModel):
    feedback_id: str
    approved: bool
    comments: Optional[str] = None
    suggested_changes: Optional[Dict[str, Any]] = None
    conversation_id: Optional[str] = None
    message_id: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    conversation_id: str
    feedback_id: str
    sources: Dict[str, Any]
    analysis: Dict[str, Any]
    technical_details: Optional[str] = None
    business_context: Optional[str] = None
    data_lineage: Optional[Dict[str, Any]] = None
    parsed_question: Optional[Dict[str, Any]] = None
    feedback_required: bool = False
    feedback_status: Optional[str] = None

# Include feedback routes
app.include_router(feedback_router, prefix="/feedback-status", tags=["feedback"])

# Initialize feedback system
feedback_system = HumanFeedbackSystem()

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
    try:
        logger.info(f"Processing chat request: {request.message}")
        conversation_id = request.conversation_id or str(uuid.uuid4())
        feedback_id = str(uuid.uuid4())
        
        result = await question_parser_system.parse_question(request.message)
        
        # Safely get and format key points and questions
        key_points = result.get('key_points', [])
        similar_questions = result.get('similar_questions', [])
        
        # Convert any dictionary items to strings
        formatted_points = []
        for point in key_points:
            if isinstance(point, dict):
                formatted_points.append(str(point))
            elif isinstance(point, str):
                formatted_points.append(point.strip())
            else:
                formatted_points.append(str(point))

        formatted_questions = []
        for question in similar_questions:
            if isinstance(question, dict):
                formatted_questions.append(str(question))
            elif isinstance(question, str):
                formatted_questions.append(question.strip())
            else:
                formatted_questions.append(str(question))

        # Format response with better alignment and spacing
        response_text = f"""**I understand you want to:**
{result.get('rephrased_question', '')}

**Key Points:**
{chr(10).join([f"• {point}" for point in formatted_points])}

**Related Questions to Consider:**
{chr(10).join([f"• {q}" for q in formatted_questions])}

---
Would this analysis help you? Let me know if you'd like to adjust the focus."""

        return JSONResponse(
            content={
                "answer": response_text,
                "conversation_id": conversation_id,
                "feedback_id": feedback_id,
                "parsed_question": result,
                "requires_confirmation": True,
                "feedback_status": "pending"
            },
            status_code=200
        )
            
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}", exc_info=True)  # Added exc_info for better error tracking
        return JSONResponse(
            content={
                "answer": "I apologize, but I encountered an error processing your request. Could you please try rephrasing your question?",
                "error": str(e),
                "requires_confirmation": False
            },
            status_code=200  # Return 200 with error message instead of 500
        )

def format_technical_details(doc_context: Dict) -> str:
    """Format technical implementation details from the context"""
    if not doc_context:
        return "No technical details available"
    
    details = []
    
    # Add SQL query if available
    if 'suggested_query' in doc_context:
        details.append(f"Suggested SQL Query:\n```sql\n{doc_context['suggested_query']}\n```")
    
    # Add implementation steps if available
    if 'implementation_steps' in doc_context:
        details.append("Implementation Steps:")
        for step in doc_context['implementation_steps']:
            details.append(f"• {step}")
    
    # Add technical considerations
    if 'technical_considerations' in doc_context:
        details.append("\nTechnical Considerations:")
        for consideration in doc_context['technical_considerations']:
            details.append(f"• {consideration}")
    
    return "\n".join(details) if details else "Technical details will be provided after analysis"

def format_data_sources(doc_context: Dict) -> str:
    """Format information about available data sources"""
    if not doc_context or 'data_sources' not in doc_context:
        return "No specific data sources identified"
    
    sources = doc_context['data_sources']
    formatted_sources = []
    
    for source in sources:
        if isinstance(source, dict):
            source_info = [
                f"• Source: {source.get('name', 'Unnamed source')}",
                f"  Type: {source.get('type', 'Unknown type')}",
                f"  Description: {source.get('description', 'No description available')}"
            ]
            formatted_sources.append("\n".join(source_info))
    
    return "\n".join(formatted_sources) if formatted_sources else "Data sources will be analyzed"

@app.post("/feedback/")
async def submit_feedback(feedback: FeedbackRequest):
    """Submit feedback for a question analysis"""
    try:
        logger.info(f"Received feedback: {feedback}")
        
        if not feedback.feedback_id:
            raise HTTPException(status_code=422, detail="feedback_id is required")

        # Process the feedback
        try:
            feedback_result = await feedback_system.process_feedback(
                feedback_id=feedback.feedback_id,
                approved=feedback.approved,
                comments=feedback.comments
            )
            
            return JSONResponse(
                content={
                    "status": "success",
                    "message": "Feedback processed successfully",
                    "feedback_id": feedback.feedback_id,
                    "approved": feedback.approved
                }
            )
            
        except Exception as e:
            logger.error(f"Error processing feedback: {e}")
            raise HTTPException(status_code=500, detail=str(e))
            
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in feedback endpoint: {e}")
        raise HTTPException(status_code=422, detail=str(e))

def format_sql_solution(doc_context: Dict) -> str:
    """Format SQL solution with relevant tables and their key columns."""
    if not doc_context:
        return ""

    # Extract relevant tables and their key columns
    tables = []
    for result in doc_context.get('sql_results', []):
        content = result.get('content', '')
        table_match = re.search(r'CREATE TABLE (\w+)', content)
        if table_match:
            table_name = table_match.group(1)
            # Extract only relevant columns based on the table
            if table_name == 'LINEITEM':
                columns = [
                    'L_EXTENDEDPRICE - Base price',
                    'L_DISCOUNT - Discount rate',
                    'L_TAX - Tax rate',
                    'L_QUANTITY - Order quantity'
                ]
            elif table_name == 'ORDERS':
                columns = [
                    'O_ORDERKEY - Order identifier',
                    'O_TOTALPRICE - Total order price',
                    'O_ORDERDATE - Date of order'
                ]
            else:
                # For other tables, extract columns from schema
                columns = re.findall(r'(\w+)\s+(?:VARCHAR|CHAR|BIGINT|INTEGER|DECIMAL|DATE|SERIAL).*?,', content)
                columns = [f"{col} - Column" for col in columns[:3]] if columns else []
            
            if columns:
                tables.append(f"• {table_name}:\n  " + "\n  ".join(columns))

    return f"""**Available Tables and Columns:**
{chr(10).join(tables)}"""

@app.get("/pending-feedback/")
async def get_pending_feedback():
    """Get all pending feedback requests."""
    try:
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

@app.get("/api/chat-history")
async def get_chat_history():
    """Get chat history with checkpoints"""
    try:
        db = ChatDatabase()
        
        # Get chat history with checkpoints
        history = await db.get_conversation_history_with_checkpoints()
        
        # Format the response
        formatted_history = []
        for chat in history:
            # Extract messages from checkpoints
            messages = []
            for checkpoint in chat.get('checkpoints', []):
                if isinstance(checkpoint.get('checkpoint'), dict):
                    checkpoint_data = checkpoint['checkpoint']
                    
                    # Extract messages from checkpoint data
                    if 'messages' in checkpoint_data:
                        for msg in checkpoint_data['messages']:
                            messages.append({
                                'role': msg.get('role', 'unknown'),
                                'content': msg.get('content', ''),
                                'timestamp': checkpoint.get('timestamp'),
                                'analysis': msg.get('analysis', {})
                            })
            
            # Format chat session
            formatted_chat = {
                'session_id': chat['id'],
                'start_time': chat['timestamp'],
                'preview': chat.get('query', 'No preview available'),
                'message_count': len(messages),
                'messages': messages,
                'metadata': {
                    'technical_details': chat.get('technical_details', {}),
                    'code_context': chat.get('code_context', {})
                }
            }
            
            formatted_history.append(formatted_chat)
        
        return {
            "status": "success",
            "chat_history": formatted_history
        }
        
    except Exception as e:
        logger.error(f"Error fetching chat history: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

@app.post("/api/save-message")
async def save_message(message: dict):
    """Save a chat message"""
    try:
        db = ChatDatabase()
        await db.save_message(
            session_id=message['session_id'],
            role=message['role'],
            content=message['content'],
            analysis=message.get('analysis')
        )
        
        # Create a checkpoint for this message
        checkpoint_data = {
            'messages': [{
                'role': message['role'],
                'content': message['content'],
                'analysis': message.get('analysis', {})
            }],
            'timestamp': datetime.now().isoformat()
        }
        
        db.create_checkpoint(
            thread_id=message['session_id'],
            checkpoint_id=str(uuid.uuid4()),
            checkpoint_type='message',
            checkpoint_data=json.dumps(checkpoint_data)
        )
        
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Error saving message: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

@app.get("/api/chat/{session_id}")
async def get_chat_session(session_id: str):
    """Get a specific chat session with all its messages"""
    try:
        db = ChatDatabase()
        chat = await db.get_conversation(session_id)
        
        if not chat:
            raise HTTPException(status_code=404, detail="Chat session not found")
            
        # Get all checkpoints for this session
        checkpoints = db.get_conversation_history_with_checkpoints()
        messages = []
        
        # Extract messages from checkpoints
        for checkpoint in checkpoints:
            if checkpoint['thread_id'] == session_id:
                checkpoint_data = json.loads(checkpoint['checkpoint'])
                if 'messages' in checkpoint_data:
                    messages.extend(checkpoint_data['messages'])
        
        return {
            "status": "success",
            "session_id": session_id,
            "messages": messages,
            "metadata": {
                "technical_details": chat.get('technical_details', {}),
                "code_context": chat.get('code_context', {})
            }
        }
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error fetching chat session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    main() 