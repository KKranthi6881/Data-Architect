from fastapi import FastAPI, UploadFile, File, Request, HTTPException, Form, Depends, Query, Response, BackgroundTasks
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
from .agents.data_architect_agent import create_data_architect_agent
from .agents.data_architect.question_parser import QuestionParserSystem
import logging
from pydantic import BaseModel, Field
from .db.database import ChatDatabase
import streamlit as st
from datetime import datetime, timezone
from langchain_community.utilities import SQLDatabase
from src.tools import SearchTools
from typing import Optional, List, Dict, Any, Union, Callable
from src.agents.data_architect.human_feedback import HumanFeedbackSystem
from src.api.feedback_routes import router as feedback_router
import re
import uuid
import json
from src.agents.data_architect.schema_search import SchemaSearchAgent
import time

app = FastAPI(title="Knowledge Chat API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the base directory
BASE_DIR = Path(__file__).resolve().parent.parent
THREADS_DIR = os.path.join(BASE_DIR, "threads")

# Create threads directory if it doesn't exist
os.makedirs(THREADS_DIR, exist_ok=True)

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
chat_db.ensure_architect_response_column()
chat_db.ensure_parent_checkpoint_column()
chat_db.ensure_conversation_id_column()

# Initialize tools and analysis system
analysis_system = SimpleAnalysisSystem(code_search_tools)

# Initialize the Data Architect Agent
data_architect_agent = create_data_architect_agent(code_search_tools)

# Comment out or remove this line if DataArchitectSystem is no longer needed
# data_architect_system = DataArchitectSystem(code_search_tools)

# Add new model for code analysis requests
class CodeAnalysisRequest(BaseModel):
    query: str

# Add new model for analysis requests
class AnalysisRequest(BaseModel):
    query: str = Field(..., description="The query to analyze")
    conversation_id: Optional[str] = Field(None, description="Optional conversation ID")
    thread_id: Optional[str] = Field(None, description="Optional thread ID")

# Add new model for GitHub repository
class GitHubRepoRequest(BaseModel):
    repo_url: str
    username: str = ""
    token: str = ""
    is_dbt_repo: bool = False
    dbt_path: str = ""

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

# Add this with your other model definitions
class ArchitectRequest(BaseModel):
    query: str = Field(..., description="The question to analyze")
    conversation_id: Optional[str] = Field(None, description="Optional conversation ID")
    thread_id: Optional[str] = Field(None, description="Optional thread ID")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata")

# Add this model definition
class ArchitectResponse(BaseModel):
    response: str
    conversation_id: str
    processing_time: float
    question_type: str
    github_results: Optional[Dict[str, Any]] = None
    sql_results: Optional[Dict[str, Any]] = None
    doc_results: Optional[Dict[str, Any]] = None
    dbt_results: Optional[Dict[str, Any]] = None
    relationship_results: Optional[Dict[str, Any]] = None

# Include feedback routes
app.include_router(feedback_router, prefix="/feedback", tags=["feedback"])

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
async def add_github_repo(request: GitHubRepoRequest):
    """Add a GitHub repository to the vector store."""
    try:
        logger.info(f"Processing GitHub repository: {request.repo_url}")
        
        # Process the repository directly
        result = db_manager.process_github(
            repo_url=request.repo_url,
            username=request.username,
            token=request.token,
            is_dbt_repo=request.is_dbt_repo,
            dbt_path=request.dbt_path
        )
        
        if result.get("status") == "error":
            logger.error(f"Error processing GitHub repository: {result.get('error')}")
            return {
                "status": "error",
                "message": f"Error processing GitHub repository: {result.get('error')}",
                "repo_url": request.repo_url
            }
        
        # Return success response
        return {
            "status": "success",
            "message": f"Successfully processed GitHub repository: {request.repo_url}",
            "files_processed": result.get("files_processed", 0),
            "repo_url": request.repo_url,
            "is_dbt_repo": request.is_dbt_repo,
            "dbt_path": request.dbt_path
        }
        
    except Exception as e:
        logger.error(f"Error processing GitHub repository: {str(e)}")
        return {
            "status": "error",
            "message": f"Error processing GitHub repository: {str(e)}",
            "repo_url": request.repo_url
        }

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
    """Process a chat message"""
    try:
        # Generate a new conversation ID if not provided
        conversation_id = request.conversation_id or str(uuid.uuid4())
        
        # Use provided thread_id or create a new one
        thread_id = request.thread_id or str(uuid.uuid4())
        
        # Process the message
        agent_response = await process_message(
            message=request.message,
            conversation_id=conversation_id,
            thread_id=thread_id
        )
        
        # Ensure the response includes all required fields
        response = {
            "answer": agent_response.get("answer", ""),
            "conversation_id": conversation_id,
            "thread_id": agent_response.get("thread_id", thread_id),
            "feedback_id": agent_response.get("feedback_id", ""),
            "parsed_question": agent_response.get("parsed_question", {}),
            "feedback_required": True,  # Always require feedback for now
            "feedback_status": "pending",
            "confidence_score": agent_response.get("confidence_score", 0.0),
            "details": agent_response.get("details", {})
        }
        
        logger.info(f"Chat response: feedback_id={response['feedback_id']}, conversation_id={conversation_id}")
        
        return JSONResponse(content=response)
        
    except Exception as e:
        logger.error(f"Error processing chat message: {e}", exc_info=True)
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
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
async def process_feedback(request: dict):
    """Process feedback for a parsed question"""
    try:
        # Extract feedback data
        feedback_id = request.get("feedback_id")
        conversation_id = request.get("conversation_id")
        approved = request.get("approved", False)
        comments = request.get("comments")
        
        if not feedback_id:
            return {"status": "error", "message": "No feedback ID provided"}
        
        logger.info(f"Processing feedback request: feedback_id='{feedback_id}' conversation_id='{conversation_id}' approved={approved} comments={comments}")
        
        # Check if this feedback has already been processed
        feedback_status = feedback_system.get_feedback_status(feedback_id)
        if feedback_status and feedback_status.get("status") == "processed":
            logger.info(f"Feedback {feedback_id} has already been processed, skipping")
            return {"status": "success", "message": "Feedback already processed", "already_processed": True}
        
        # Process feedback
        result = await feedback_system.process_feedback(
            feedback_id=feedback_id,
            approved=approved,
            comments=comments
        )
        
        # Mark this feedback as processed to prevent loops
        if feedback_id in feedback_system.pending_feedback:
            feedback_system.pending_feedback[feedback_id]["status"] = "processed"
        
        # If feedback is approved, trigger the Data Architect Agent
        if approved and conversation_id:
            # Get the conversation to extract thread_id and parsed_question
            conversation = chat_db.get_conversation(conversation_id)
            
            if conversation:
                thread_id = conversation.get("thread_id")
                technical_details = conversation.get("technical_details", "{}")
                
                # Parse technical details
                if isinstance(technical_details, str):
                    try:
                        technical_details = json.loads(technical_details)
                    except json.JSONDecodeError:
                        technical_details = {}
                
                # Extract parsed_question and original_question
                parsed_question = technical_details
                original_question = conversation.get("query", "")
                
                # Get search results from the result
                code_search_results = result.get("search_results", {}).get("code_search_results", [])
                schema_search_results = result.get("search_results", {}).get("schema_search_results", [])
                
                # Generate architect response
                architect_response = data_architect_agent.generate_response(
                    parsed_question=parsed_question,
                    schema_results=schema_search_results,
                    code_results=code_search_results,
                    original_question=original_question
                )
                
                if architect_response:
                    # Include architect response in the result
                    result["architect_response"] = architect_response
                    
                    # Create a new conversation entry with the architect's response
                    new_conversation_id = str(uuid.uuid4())
                    new_conversation_data = {
                        "query": original_question,
                        "output": architect_response.get("response", ""),
                        "technical_details": json.dumps({
                            "parsed_question": parsed_question,
                            "schema_results": architect_response.get("schema_results", []),
                            "code_results": architect_response.get("code_results", []),
                            "sections": architect_response.get("sections", {})
                        }),
                        "code_context": json.dumps({
                            "schema_results": architect_response.get("schema_results", []),
                            "code_results": architect_response.get("code_results", [])
                        }),
                        "thread_id": thread_id,
                        "feedback_status": "completed"
                    }
                    
                    # Save the new conversation
                    chat_db.save_conversation(new_conversation_id, new_conversation_data)
                    
                    # Include the new conversation ID in the result
                    result["new_conversation_id"] = new_conversation_id
                    
                    # Log the architect response for debugging
                    logger.info(f"Generated architect response for {conversation_id}, new ID: {new_conversation_id}")
                    logger.debug(f"Architect response: {architect_response.get('response', '')[:100]}...")
                    
                    # Update the original conversation to mark it as processed
                    conversation_data = {
                        "query": conversation.get("query", ""),
                        "output": conversation.get("output", ""),
                        "technical_details": conversation.get("technical_details", ""),
                        "code_context": conversation.get("code_context", "{}"),
                        "feedback_status": "processed",  # Mark as processed
                        "thread_id": thread_id
                    }
                    chat_db.save_conversation(conversation_id, conversation_data)
                else:
                    logger.error("Failed to generate architect response")
                    result["error"] = "Failed to generate architect response"
        
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error processing feedback: {e}", exc_info=True)
        return JSONResponse(
            content={"status": "error", "message": str(e)},
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

@app.get("/api/conversation/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get a specific conversation by ID"""
    try:
        logger.info(f"Fetching conversation with ID: {conversation_id}")
        conversation = chat_db.get_conversation(conversation_id)
        
        if not conversation:
            logger.error(f"Conversation not found: {conversation_id}")
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        logger.info(f"Raw conversation data: {conversation}")
        
        # Format the response properly
        formatted_conversation = {
            "id": conversation.get("conversation_id"),
            "timestamp": conversation.get("timestamp", ""),
            "query": conversation.get("question", ""),
            "response": conversation.get("answer", ""),
            "technical_details": conversation.get("metadata", {}),
            "context": {},
            "feedback": {"status": "pending", "comments": ""}
        }
        
        logger.info(f"Returning formatted conversation: {formatted_conversation}")
        return JSONResponse(content=formatted_conversation)
    except Exception as e:
        logger.error(f"Error retrieving conversation: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

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

@app.get("/api/conversations")
async def get_conversations(limit: int = 10, offset: int = 0):
    """Get a list of recent conversations"""
    try:
        logger.info("Fetching conversations from database...")
        conversations = chat_db.get_recent_conversations(limit, offset)
        logger.info(f"Retrieved {len(conversations)} conversations from database")
        
        # Format conversations for response
        formatted_conversations = []
        for conv in conversations:
            try:
                # Extract preview from query
                query = conv.get("query", "")
                preview = query[:100] + "..." if len(query) > 100 else query
                
                # Format the conversation
                formatted_conv = {
                    "id": conv.get("id", ""),
                    "timestamp": conv.get("created_at", ""),  # Changed from timestamp to created_at
                    "preview": preview or "No preview available",
                    "feedback_status": conv.get("feedback_status", "pending"),
                    "has_response": bool(conv.get("output")),
                }
                logger.info(f"Formatted conversation: {formatted_conv}")
                formatted_conversations.append(formatted_conv)
                
            except Exception as e:
                logger.error(f"Error formatting conversation: {e}", exc_info=True)
                continue
        
        response_data = {
            "status": "success",
            "conversations": formatted_conversations,
            "total": len(formatted_conversations)
        }
        logger.info(f"Returning response with {len(formatted_conversations)} conversations")
        return JSONResponse(content=response_data)
        
    except Exception as e:
        logger.error(f"Error getting conversations: {e}", exc_info=True)
        return JSONResponse(
            content={
                "status": "success",
                "conversations": [],
                "message": str(e)
            }
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

@app.get("/api/thread-conversations/{thread_id}")
async def get_thread_conversations(thread_id: str):
    """Get all conversations in a thread"""
    try:
        logger.info(f"Fetching conversations for thread: {thread_id}")
        conversations = chat_db.get_conversations_by_thread(thread_id)
        return JSONResponse(content={"conversations": conversations})
    except Exception as e:
        logger.error(f"Error retrieving thread conversations: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/thread-conversations")
async def get_all_threads():
    """Get all thread IDs with their latest conversation"""
    try:
        logger.info("Fetching all threads")
        threads = chat_db.get_all_threads()
        return JSONResponse(content={"threads": threads})
    except Exception as e:
        logger.error(f"Error getting thread conversations: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to get threads: {str(e)}", "detail": "Check server logs for more information"}
        )

@app.post("/api/search-schemas")
async def search_schemas(request: dict):
    """Search for relevant database schemas"""
    try:
        # Extract query from request
        query = request.get("query", "")
        parsed_question = request.get("parsed_question", {})
        
        if not query and not parsed_question:
            return {"status": "error", "message": "No query or parsed question provided"}
        
        # Initialize schema search agent
        schema_search_agent = SchemaSearchAgent()
        
        # If we have a parsed question, use it directly
        if parsed_question:
            search_results = schema_search_agent.search_schemas(parsed_question)
        else:
            # Create a simple parsed question from the query
            simple_parsed_question = {
                "original_question": query,
                "rephrased_question": query,
                "business_context": {
                    "domain": "data",
                    "key_entities": []
                },
                "query_intent": {
                    "primary_intent": "search"
                }
            }
            search_results = schema_search_agent.search_schemas(simple_parsed_question)
        
        return {
            "status": "success",
            "results": search_results,
            "count": len(search_results)
        }
    except Exception as e:
        logger.error(f"Error searching schemas: {str(e)}", exc_info=True)
        return {"status": "error", "message": str(e)}

async def process_message(message: str, conversation_id: str, thread_id: str, feedback: dict = None):
    """Process a message with the data architect agent"""
    try:
        logger.info(f"Processing message: {message}")
        feedback_id = str(uuid.uuid4())
        
        # Get business analysis with thread_id support
        result = await question_parser_system.parse_question(
            question=message,
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
{result.get('rephrased_question', 'No rephrased question available')}

Key Points:
{chr(10).join([f"• {point}" for point in result.get('key_points', ['No key points available'])])}

Business Context:
{json.dumps(result.get('business_context', {}), indent=2)}

Debug Info:
feedback_id: {feedback_id}
conversation_id: {conversation_id}
"""
        
        # Save to conversation table
        conversation_data = {
            "query": message,
            "output": response_text,
            "technical_details": json.dumps(result),
            "code_context": "{}",
            "thread_id": thread_id,  # Always include thread_id
            "feedback_status": "pending"  # Mark as pending feedback
        }
        chat_db.save_conversation(conversation_id, conversation_data)
        
        # Process feedback if provided
        if feedback:
            await feedback_system.process_feedback(
                feedback_id=feedback.get("feedback_id", feedback_id),
                approved=feedback.get("approved", False),
                comments=feedback.get("comments")
            )
        
        return {
            "answer": response_text,
            "conversation_id": conversation_id,
            "thread_id": thread_id,
            "feedback_id": feedback_id,
            "parsed_question": result,
            "requires_confirmation": True,
            "feedback_required": True,  # Explicitly set to True
            "feedback_status": "pending",
            "confidence_score": result.get("confidence_score", 0.0),
            "details": result
        }
    except Exception as e:
        logger.error(f"Error processing message: {e}", exc_info=True)
        raise

@app.get("/chat/architect/{conversation_id}")
async def get_architect_response(conversation_id: str):
    try:
        # Get the conversation to extract necessary details
        conversation = chat_db.get_conversation(conversation_id)
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Get original question
        original_question = conversation.get("query", "")
        
        # Process the question with our new data architect agent
        result = data_architect_agent.process_question(original_question)
        
        # Extract the response
        response = result.get("response", "I couldn't generate a response. Please try again.")
        
        # Save the architect response to the database
        try:
            # Create a response object
            architect_response = {
                "response": response,
                "processing_time": result.get("processing_time", 0),
                "question_type": result.get("question_type", "UNKNOWN"),
                "github_results": result.get("github_results", {}),
                "sql_results": result.get("sql_results", {}),
                "doc_results": result.get("doc_results", {})
            }
            
            # Convert to JSON for storage
            architect_response_json = json.dumps(architect_response)
            
            # Update the conversation with the architect response
            with chat_db._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE conversations 
                    SET architect_response = ?,
                        feedback_status = 'completed'
                    WHERE id = ?
                """, (architect_response_json, conversation_id))
                conn.commit()
                
            logger.info(f"Saved architect response for conversation {conversation_id}")
        except Exception as e:
            logger.error(f"Error saving architect response: {e}")
            
        return JSONResponse(content={
            "status": "success",
            "response": response,
            "conversation_id": conversation_id,
            "processing_time": result.get("processing_time", 0),
            "question_type": result.get("question_type", "UNKNOWN")
        })
        
    except Exception as e:
        logger.error(f"Error generating architect response: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/git-zip/")
async def upload_git_zip(file: UploadFile = File(...)):
    """
    Upload and process a Git repository ZIP file
    """
    try:
        logger.info(f"Received Git ZIP file: {file.filename}")
        
        # Create a temporary directory for the ZIP extraction
        temp_dir = tempfile.mkdtemp(prefix="git_zip_")
        
        # Save the uploaded file to a temporary file
        temp_zip_path = os.path.join(temp_dir, file.filename)
        with open(temp_zip_path, "wb") as temp_file:
            content = await file.read()
            temp_file.write(content)
        
        # Process the ZIP file
        result = db_manager.process_git_zip(temp_zip_path, file.filename)
        
        return {
            "status": "success",
            "message": f"Successfully processed Git ZIP file: {file.filename}",
            "details": result
        }
    
    except Exception as e:
        logger.error(f"Error processing Git ZIP file: {str(e)}")
        return {
            "status": "error",
            "message": f"Error processing Git ZIP file: {str(e)}"
        }

@app.post("/architect/analyze/")
async def analyze_with_architect(request: ArchitectRequest):
    """Process a question with the data architect agent"""
    try:
        logger.info(f"Processing architect request: {request.query}... [conv_id: {request.conversation_id}]")
        
        # Create a new conversation ID if not provided
        conversation_id = request.conversation_id or str(uuid.uuid4())
        thread_id = request.thread_id or conversation_id
        
        # Process the question
        start_time = time.time()
        
        # Get the agent
        agent = get_data_architect_agent(
            db_manager=db_manager,
            chat_db=chat_db,
            save_chat_func=chat_db.save_conversation,
            conversation_id=conversation_id
        )
        
        # Process the question
        result = agent.process_question(
            question=request.query,
            conversation_id=conversation_id,
            thread_id=thread_id,
            metadata=request.metadata
        )
        
        # Calculate processing time
        processing_time = time.time() - start_time
        logger.info(f"Architect response generated in {processing_time:.2f} seconds")
        
        # Get the response
        response = result.get("response", "I couldn't generate a response. Please try again.")
        
        # Return the response
        return {
            "response": response,
            "conversation_id": result.get("conversation_id", conversation_id),
            "thread_id": result.get("thread_id", thread_id),
            "processing_time": result.get("processing_time", processing_time),
            "question_type": result.get("question_type", "UNKNOWN"),
            "github_results": result.get("github_results", {}),
            "sql_results": result.get("sql_results", {}),
            "doc_results": result.get("doc_results", {}),
            "dbt_results": result.get("dbt_results", {}),
            "relationship_results": result.get("relationship_results", {})
        }
        
    except Exception as e:
        logger.error(f"Error processing architect request: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )

@app.get("/conversations")
async def get_conversations(limit: int = 10, offset: int = 0):
    """Get a list of recent conversations"""
    try:
        conversations = chat_db.get_recent_conversations(limit, offset)
        return JSONResponse(content={"conversations": conversations})
    except Exception as e:
        logger.error(f"Error retrieving conversations: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

def get_data_architect_agent(db_manager, chat_db, save_chat_func, conversation_id=None):
    """Get or create a data architect agent with the latest GitHub configuration."""
    try:
        # Get the latest GitHub connector configuration
        with chat_db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT value FROM settings 
                WHERE key = 'github_connectors' 
                ORDER BY created_at DESC 
                LIMIT 1
            """)
            result = cursor.fetchone()
            
            if result:
                configs = json.loads(result[0])
                if configs:
                    # Use the most recent configuration
                    latest_config = configs[-1]
                    repo_url = latest_config.get('repoUrl', '')
                    username = latest_config.get('username', '')
                    token = latest_config.get('token', '')
                    
                    logger.info(f"Creating Data Architect Agent with repository: {repo_url}")
                    return create_data_architect_agent(repo_url, username, token)
            
            logger.warning("No GitHub connector configuration found, creating agent without repository")
            return create_data_architect_agent()
            
    except Exception as e:
        logger.error(f"Error getting data architect agent: {str(e)}")
        return create_data_architect_agent()

# Add these endpoints for thread-based conversation retrieval
@app.get("/thread-conversations/{thread_id}")
async def get_thread_conversations(thread_id: str):
    """Get all conversations in a thread"""
    try:
        conversations = chat_db.get_conversations_by_thread(thread_id)
        return JSONResponse(content={"conversations": conversations})
    except Exception as e:
        logger.error(f"Error retrieving thread conversations: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/thread-conversations")
async def get_all_threads():
    """Get all thread IDs with their latest conversation"""
    try:
        # Get all threads with their summaries
        threads = chat_db.get_all_threads()  # Make sure this is the correct method name
        return JSONResponse(content={"threads": threads})
    except Exception as e:
        # Add more detailed error logging
        logger.error(f"Error getting thread conversations: {str(e)}", exc_info=True)
        # Return a more helpful error message
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to get threads: {str(e)}", "detail": "Check server logs for more information"}
        )

@app.post("/api/settings/github_connectors")
async def save_github_connectors(configs: List[Dict[str, Any]]):
    """Save GitHub connector configurations to the database"""
    try:
        logger.info(f"Received GitHub connector configurations: {configs}")
        with chat_db._get_connection() as conn:
            cursor = conn.cursor()
            # First, ensure the settings table exists
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS settings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            logger.info("Settings table created/verified")
            
            # Save the configurations
            cursor.execute("""
                INSERT INTO settings (key, value)
                VALUES (?, ?)
            """, ('github_connectors', json.dumps(configs)))
            
            conn.commit()
            logger.info("Successfully saved GitHub configurations to database")
            
        return {"status": "success", "message": "GitHub configurations saved"}
    except Exception as e:
        logger.error(f"Error saving GitHub configurations: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/settings/github_connectors/{config_id}")
async def delete_github_connector(config_id: str):
    """Delete a specific GitHub connector configuration"""
    try:
        with chat_db._get_connection() as conn:
            cursor = conn.cursor()
            
            # Get current configurations
            cursor.execute("""
                SELECT value FROM settings 
                WHERE key = 'github_connectors' 
                ORDER BY created_at DESC 
                LIMIT 1
            """)
            result = cursor.fetchone()
            
            if not result:
                raise HTTPException(status_code=404, detail="No GitHub configurations found")
            
            configs = json.loads(result[0])
            
            # Convert config_id to string for comparison since frontend uses Date.now()
            config_id_str = str(config_id)
            
            # Filter out the configuration to delete
            updated_configs = [config for config in configs if str(config.get('id')) != config_id_str]
            
            if len(updated_configs) == len(configs):
                raise HTTPException(status_code=404, detail="Configuration not found")
            
            # Save updated configurations
            cursor.execute("""
                INSERT INTO settings (key, value)
                VALUES (?, ?)
            """, ('github_connectors', json.dumps(updated_configs)))
            
            conn.commit()
            logger.info(f"Successfully deleted GitHub configuration with ID: {config_id}")
            
        return {"status": "success", "message": "GitHub configuration deleted"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting GitHub configuration: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/settings/github_connectors")
async def get_github_connectors():
    """Get all GitHub connector configurations"""
    try:
        with chat_db._get_connection() as conn:
            cursor = conn.cursor()
            
            # Get latest configurations
            cursor.execute("""
                SELECT value FROM settings 
                WHERE key = 'github_connectors' 
                ORDER BY created_at DESC 
                LIMIT 1
            """)
            result = cursor.fetchone()
            
            if not result:
                return {"status": "success", "configs": []}
            
            configs = json.loads(result[0])
            return {"status": "success", "configs": configs}
            
    except Exception as e:
        logger.error(f"Error fetching GitHub configurations: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    main() 