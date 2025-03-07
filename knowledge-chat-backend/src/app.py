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

# Initialize systems
code_search_tools = SearchTools(db_manager)
question_parser_system = QuestionParserSystem(tools=code_search_tools)
feedback_system = HumanFeedbackSystem()
chat_db = ChatDatabase()
chat_db.ensure_feedback_columns()
chat_db.ensure_cleared_column()

# Initialize tools and analysis system
analysis_system = SimpleAnalysisSystem(code_search_tools)

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
    conversation_id: Optional[str] = None
    thread_id: Optional[str] = None  # Add thread_id support
    context: Optional[Dict[str, Any]] = None
    wait_for_feedback: bool = False

# Add FeedbackRequest model
class FeedbackRequest(BaseModel):
    feedback_id: str
    conversation_id: str  # Make this required
    approved: bool
    comments: Optional[str] = None
    suggested_changes: Optional[Dict[str, Any]] = None
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
        thread_id = request.thread_id or conversation_id  # Default to conversation_id if thread_id not provided
        feedback_id = str(uuid.uuid4())
        
        # Get business analysis with thread_id support
        result = await question_parser_system.parse_question(
            question=request.message,
            thread_id=thread_id,
            conversation_id=conversation_id
        )
        
        # Extract thread_id from result
        thread_id = result.get("thread_id", thread_id)
        
        # Add to feedback system for review with conversation_id
        feedback_system.add_feedback_request(feedback_id, result, conversation_id)
        
        # Format response text - Include full analysis
        response_text = f"""I've analyzed your question from a business perspective. Please review my understanding below and let me know if any adjustments are needed.

Business Understanding:
{result['rephrased_question']}

Key Points:
{chr(10).join([f"• {point}" for point in result['key_points']])}"""

        # Save to conversation table
        conversation_data = {
            "query": request.message,
            "output": response_text,
            "technical_details": json.dumps(result),
            "code_context": json.dumps(request.context) if request.context else "{}",
            "thread_id": thread_id  # Always include thread_id
        }
        chat_db.save_conversation(conversation_id, conversation_data)
        
        # If wait_for_feedback is true, wait for feedback before responding
        feedback_status = "pending"
        github_search_results = []
        
        if request.wait_for_feedback:
            try:
                feedback = await feedback_system.wait_for_feedback(feedback_id)
                feedback_status = "approved" if feedback.get("approved", False) else "needs_improvement"
                
                # Get GitHub search results if available
                if feedback.get("approved", False) and "github_search_results" in feedback:
                    github_search_results = feedback["github_search_results"]
                
                # If feedback indicates changes needed, update the response
                if not feedback.get("approved", False):
                    response_text += f"\n\nNote: This analysis needs improvement based on feedback: {feedback.get('comments', 'No specific comments')}"
                    
                    # Update the conversation with the new response
                    conversation_data["output"] = response_text
                    conversation_data["feedback_status"] = feedback_status
                    conversation_data["feedback_comments"] = feedback.get("comments", "")
                    chat_db.save_conversation(conversation_id, conversation_data)
            except Exception as feedback_error:
                logger.error(f"Error waiting for feedback: {feedback_error}")
                feedback_status = "error"
        else:
            feedback_status = "pending"

        # Get technical details with GitHub search results if available
        technical_details = {}
        conversation = chat_db.get_conversation(conversation_id)
        if conversation and conversation.get("technical_details"):
            try:
                if isinstance(conversation["technical_details"], str):
                    technical_details = json.loads(conversation["technical_details"])
                else:
                    technical_details = conversation["technical_details"]
                    
                # Extract GitHub search results if available
                if "github_search_results" in technical_details:
                    github_search_results = technical_details["github_search_results"]
            except json.JSONDecodeError:
                pass

        return JSONResponse(
            content={
                "answer": response_text,
                "conversation_id": conversation_id,
                "thread_id": thread_id,
                "feedback_id": feedback_id,
                "parsed_question": result,
                "requires_confirmation": True,
                "feedback_status": feedback_status,
                "confidence_score": result.get("confidence_score", 0.0),
                "github_search_results": github_search_results,  # Include GitHub search results
                "details": result
            },
            status_code=200
        )
            
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}", exc_info=True)
        error_conversation_id = request.conversation_id or str(uuid.uuid4())
        error_thread_id = request.thread_id or error_conversation_id  # Default to conversation_id
        
        # Save error to conversation table
        error_message = f"I apologize, but I encountered an error analyzing your question: {str(e)}"
        error_data = {
            "query": request.message if hasattr(request, 'message') else "Unknown query",
            "output": error_message,
            "technical_details": json.dumps({"error": str(e)}),
            "code_context": "{}",
            "thread_id": error_thread_id  # Always include thread_id
        }
        chat_db.save_conversation(error_conversation_id, error_data)
        
        return JSONResponse(
            content={
                "answer": "I apologize, but I encountered an error analyzing your question. Could you please rephrase it?",
                "error": str(e),
                "conversation_id": error_conversation_id,
                "thread_id": error_thread_id,  # Include thread_id in response
                "requires_confirmation": False
            },
            status_code=200
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
async def submit_feedback(request: FeedbackRequest):
    """Submit feedback for a conversation"""
    try:
        logger.info(f"Processing feedback request: {request}")
        
        # Get the original conversation to retrieve thread_id
        conversation = chat_db.get_conversation(request.conversation_id)
        thread_id = conversation.get("thread_id") if conversation else None
        
        # Process the feedback
        feedback_result = await feedback_system.process_feedback(
            request.feedback_id,
            request.approved,
            request.comments
        )
        
        # Update the conversation with feedback status
        conversation_data = {
            "query": conversation.get("query", "") if conversation else "",
            "output": conversation.get("output", "") if conversation else "",
            "technical_details": conversation.get("technical_details", "") if conversation else "",
            "code_context": conversation.get("code_context", "") if conversation else "",
            "feedback_status": "approved" if request.approved else "needs_improvement",
            "feedback_comments": request.comments,
            "thread_id": thread_id  # Preserve thread_id
        }
        
        # Save updated conversation
        chat_db.save_conversation(request.conversation_id, conversation_data)
        
        # If feedback indicates changes needed, prepare for follow-up
        if not request.approved and request.comments:
            # Create a new conversation ID for the follow-up but keep the same thread_id
            follow_up_id = str(uuid.uuid4())
            
            # Save placeholder for follow-up
            follow_up_data = {
                "query": f"Follow-up based on feedback: {request.comments}",
                "output": "Processing follow-up...",
                "technical_details": json.dumps({"feedback_id": request.feedback_id}),
                "code_context": conversation.get("code_context", "") if conversation else "",
                "thread_id": thread_id,  # Use the same thread_id
                "feedback_status": "pending"
            }
            chat_db.save_conversation(follow_up_id, follow_up_data)
            
            # Include follow-up ID in response
            feedback_result["follow_up_id"] = follow_up_id
        
        return JSONResponse(
            content={
                "status": "success",
                "feedback_id": request.feedback_id,
                "conversation_id": request.conversation_id,
                "thread_id": thread_id,  # Include thread_id in response
                "approved": request.approved,
                "processed": True
            }
        )
        
    except Exception as e:
        logger.error(f"Error processing feedback: {e}", exc_info=True)
        return JSONResponse(
            content={
                "status": "error",
                "message": str(e)
            },
            status_code=500
        )

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

@app.get("/api/conversation/{conversation_id}")
async def get_conversation_details(conversation_id: str):
    """Get detailed information about a specific conversation"""
    try:
        # Get the conversation from the database
        logger.info(f"Fetching conversation with ID: {conversation_id}")
        conversation = chat_db.get_conversation(conversation_id)
        
        if not conversation:
            logger.warning(f"Conversation not found: {conversation_id}")
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        logger.info(f"Raw conversation data: {conversation}")
        
        # Format the response with safe access to fields
        formatted_conversation = {
            "id": conversation.get("id", conversation_id),
            "timestamp": conversation.get("created_at", ""),
            "query": conversation.get("query", ""),
            "response": "",  # Initialize with empty string
            "technical_details": {},
            "context": {},
            "feedback": {
                "status": conversation.get("feedback_status", "pending"),
                "comments": conversation.get("feedback_comments", "")
            }
        }
        
        # Handle output field - ensure it's a string
        try:
            output = conversation.get("output", "")
            logger.info(f"Output type: {type(output)}, value: {output}")
            
            if isinstance(output, dict):
                formatted_conversation["response"] = output.get("output", "")
            elif isinstance(output, str):
                # Try to parse as JSON first
                try:
                    output_json = json.loads(output)
                    if isinstance(output_json, dict):
                        formatted_conversation["response"] = output_json.get("output", output)
                    else:
                        formatted_conversation["response"] = output
                except json.JSONDecodeError:
                    # Not JSON, use as is
                    formatted_conversation["response"] = output
            else:
                formatted_conversation["response"] = str(output)
        except Exception as e:
            logger.error(f"Error processing output: {e}")
            formatted_conversation["response"] = "Error processing response"
        
        # Parse technical details if available
        try:
            if conversation.get("technical_details"):
                if isinstance(conversation["technical_details"], str):
                    formatted_conversation["technical_details"] = json.loads(conversation["technical_details"])
                else:
                    formatted_conversation["technical_details"] = conversation["technical_details"]
        except json.JSONDecodeError:
            formatted_conversation["technical_details"] = {"raw": conversation.get("technical_details", "")}
        
        # Parse context if available
        try:
            if conversation.get("code_context"):
                if isinstance(conversation["code_context"], str):
                    formatted_conversation["context"] = json.loads(conversation["code_context"])
                else:
                    formatted_conversation["context"] = conversation["code_context"]
        except json.JSONDecodeError:
            formatted_conversation["context"] = {"raw": conversation.get("code_context", "")}
        
        logger.info(f"Returning formatted conversation: {formatted_conversation}")
        
        return JSONResponse(
            content={
                "status": "success",
                "conversation": formatted_conversation
            }
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error getting conversation details: {e}", exc_info=True)
        return JSONResponse(
            content={
                "status": "error",
                "message": str(e)
            },
            status_code=500
        )

@app.get("/api/conversations")
async def get_conversations(limit: int = 10, offset: int = 0):
    """Get a list of recent conversations"""
    try:
        db = ChatDatabase()
        # Use get_recent_conversations instead of direct execute
        conversations = db.get_recent_conversations(limit=limit)
        
        # Format conversations
        formatted_conversations = []
        for conv in conversations:
            # Extract a preview of the conversation
            query = conv.get("query", "")
            query_preview = query[:100] + "..." if len(query) > 100 else query
            
            # Format the conversation with safe access to feedback_status
            formatted_conv = {
                "id": conv.get("id", ""),
                "timestamp": conv.get("created_at", ""),
                "preview": query_preview,
                "feedback_status": conv.get("feedback_status", "pending"),
                "has_response": bool(conv.get("output")),
                "cleared": conv.get("cleared", False)  # Add cleared status
            }
            
            # Only include non-cleared conversations
            if not formatted_conv["cleared"]:
                formatted_conversations.append(formatted_conv)
        
        return JSONResponse(
            content={
                "status": "success",
                "conversations": formatted_conversations,
                "total": len(formatted_conversations)
            }
        )
    except Exception as e:
        logger.error(f"Error getting conversations: {e}")
        return JSONResponse(
            content={"status": "error", "message": str(e)},
            status_code=500
        )

@app.post("/api/conversation/{conversation_id}/clear")
async def clear_conversation(conversation_id: str):
    """Mark a conversation as cleared"""
    try:
        db = ChatDatabase()
        # Use the connection properly
        with db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE conversations SET cleared = TRUE WHERE id = ?",
                (conversation_id,)
            )
            conn.commit()
        return JSONResponse({"status": "success"})
    except Exception as e:
        logger.error(f"Error clearing conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history", response_class=HTMLResponse)
async def conversation_history(request: Request):
    """Render the conversation history page"""
    return templates.TemplateResponse(
        "conversation_history.html", 
        {"request": request}
    )

@app.get("/api/thread/{thread_id}/conversations")
async def get_thread_conversations(thread_id: str):
    """Get all conversations associated with a thread ID"""
    try:
        # Ensure thread_id column exists
        chat_db.ensure_thread_id_column()
        
        # Get conversations by thread ID
        conversations = chat_db.get_conversations_by_thread(thread_id)
        
        if not conversations:
            return JSONResponse(
                content={
                    "status": "success",
                    "conversations": [],
                    "message": f"No conversations found for thread ID: {thread_id}"
                }
            )
        
        # Format conversations for response
        formatted_conversations = []
        for conv in conversations:
            # Extract a preview of the conversation
            query = conv.get("query", "")
            query_preview = query[:100] + "..." if len(query) > 100 else query
            
            # Format the conversation
            formatted_conv = {
                "id": conv.get("id", ""),
                "timestamp": conv.get("created_at", ""),
                "preview": query_preview,
                "query": query,
                "response": conv.get("output", ""),
                "feedback_status": conv.get("feedback_status", "pending"),
                "thread_id": thread_id
            }
            
            formatted_conversations.append(formatted_conv)
        
        return JSONResponse(
            content={
                "status": "success",
                "thread_id": thread_id,
                "conversations": formatted_conversations,
                "total": len(formatted_conversations)
            }
        )
    except Exception as e:
        logger.error(f"Error getting thread conversations: {e}")
        return JSONResponse(
            content={"status": "error", "message": str(e)},
            status_code=500
        )

if __name__ == "__main__":
    main() 