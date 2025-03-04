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
    conversation_id: str
    approved: bool
    comments: Optional[str] = None
    suggested_changes: Optional[Dict] = None

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
        
        # Process the question
        result = question_parser_system.parse_question(request.message)
        
        # Submit for feedback with original question
        feedback_request = {
            "message": request.message,  # Store original message
            "parsed_question": result.get('parsed_question', {}),
            "doc_context": result.get('doc_context', {}),
            "timestamp": datetime.now(),
            "response": result  # Store full response
        }
        
        conversation_id = feedback_system.submit_for_feedback(feedback_request)
        
        # Format initial response while waiting for feedback
        parsed_question = result.get('parsed_question', {})
        business_context = parsed_question.get('business_context', {})
        
        response_text = f"""**Understanding Your Question:**
{parsed_question.get('rephrased_question')}

**Business Context:**
• Domain: {business_context.get('domain')}
• Focus: {business_context.get('primary_objective')}
• Key Entities: {', '.join(business_context.get('key_entities', []))}

{format_relevant_sources(result.get('doc_context', {}))}

_Waiting for human feedback to ensure accuracy..._"""
        
        return JSONResponse(
            content={
                "answer": response_text,
                "conversation_id": conversation_id,
                "sources": result.get("doc_context", {}),
                "analysis": {
                    "business_context": business_context,
                    "confidence_score": parsed_question.get("confidence_score", 0.0),
                    "alternative_interpretations": parsed_question.get("alternative_interpretations", [])
                },
                "parsed_question": parsed_question,
                "feedback_required": True,
                "feedback_status": "pending",
                "suggested_questions": generate_follow_up_questions(parsed_question, business_context, as_list=True)
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

def format_relevant_sources(doc_context: Dict) -> str:
    """Format relevant source information concisely."""
    sections = []
    
    # Format SQL results
    if sql_results := doc_context.get('sql_results', []):
        table_info = []
        for result in sql_results[:2]:
            content = result.get('content', '').strip()
            if content:
                table_match = re.search(r'CREATE TABLE (\w+)', content)
                if table_match:
                    table_name = table_match.group(1)
                    columns = re.findall(r'(\w+)\s+(?:VARCHAR|CHAR|BIGINT|INTEGER|DECIMAL|DATE|SERIAL).*?,', content)
                    if columns:
                        table_info.append(f"• {table_name} ({', '.join(columns[:3])}...)")
                    else:
                        table_info.append(f"• {table_name}")
        
        if table_info:
            sections.append("**Relevant Tables:**\n" + "\n".join(table_info))
    
    # Format relationships
    if doc_results := doc_context.get('doc_results', []):
        relationships = []
        for result in doc_results[:2]:
            content = result.get('content', '').strip()
            if content and ('relationship' in content.lower() or 'key' in content.lower()):
                # Extract meaningful relationship information
                clean_content = re.sub(r'\s+', ' ', content).strip()
                relationships.append(f"• {clean_content[:100]}...")
        
        if relationships:
            sections.append("**Key Relationships:**\n" + "\n".join(relationships))
    
    return "\n\n".join(sections) if sections else ""

def generate_follow_up_questions(parsed_question: Dict, business_context: Dict, as_list: bool = False) -> Union[str, List[str]]:
    """Generate relevant follow-up questions based on the context."""
    entities = business_context.get('key_entities', [])
    domain = business_context.get('domain', '')
    
    follow_up_questions = []
    
    # Add entity-specific questions
    for entity in entities:
        follow_up_questions.append(f"What are the key metrics for {entity}?")
        follow_up_questions.append(f"How is {entity} related to other business entities?")
    
    # Add domain-specific questions
    if domain:
        follow_up_questions.append(f"What are the common reporting needs for {domain}?")
        follow_up_questions.append(f"What are the best practices for {domain} analysis?")
    
    # Add general follow-up questions
    follow_up_questions.extend([
        "Would you like to see the detailed schema for these tables?",
        "Should I explain the relationships between these entities?",
        "Would you like to see example queries for this analysis?"
    ])
    
    # Return as list or formatted string
    if as_list:
        return follow_up_questions[:5]  # Limit to 5 questions
    else:
        return "\n".join(f"• {q}" for q in follow_up_questions[:5])

@app.post("/feedback/")
async def submit_feedback(request: FeedbackRequest):
    """Submit feedback for a conversation."""
    try:
        # Store the feedback
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
                content={"status": "error", "message": "No pending feedback request found"}
            )

        # If not approved, process with LLM to get improved response
        if not request.approved and request.comments:
            # Get original request and context
            original_request = feedback_system.get_original_request(request.conversation_id)
            if not original_request:
                raise ValueError("Original request not found")

            # Get the original question
            original_question = original_request.get("message", "")
            if not original_question:
                raise ValueError("Original question not found")

            # Create focused feedback context
            feedback_context = {
                "previous_summary": {
                    "interpretation": original_request.get("parsed_question", {}).get("rephrased_question", ""),
                    "key_entities": original_request.get("parsed_question", {}).get("business_context", {}).get("key_entities", []),
                    "tables": [table.get("name") for table in original_request.get("doc_context", {}).get("sql_results", [])]
                },
                "feedback": request.comments,
                "improvement_needed": True
            }

            # Process with LLM using focused context
            result = question_parser_system.parse_question(
                original_question,
                feedback_context=feedback_context
            )

            # Format the improved response
            business_context = result.get('parsed_question', {}).get('business_context', {})
            response_text = f"""**Understanding Your Question (Improved based on feedback):**
{result.get('parsed_question', {}).get('rephrased_question')}

**Business Context:**
• Domain: {business_context.get('domain')}
• Focus: {business_context.get('primary_objective')}
• Key Entities: {', '.join(business_context.get('key_entities', []))}

**Implementation Details:**
{format_sql_solution(result.get('doc_context', {}))}

_This response has been improved based on your feedback._
"""
            return JSONResponse(
                content={
                    "status": "success",
                    "message": "Feedback processed",
                    "answer": response_text,
                    "conversation_id": str(uuid.uuid4()),
                    "doc_context": result.get("doc_context", {}),
                    "analysis": {
                        "business_context": business_context,
                        "confidence_score": result.get('parsed_question', {}).get("confidence_score", 0.0),
                        "alternative_interpretations": result.get('parsed_question', {}).get("alternative_interpretations", [])
                    },
                    "parsed_question": result.get("parsed_question", {}),
                    "feedback_required": True
                }
            )

        return JSONResponse(content={"status": "success", "message": "Feedback received"})
        
    except Exception as e:
        logger.error(f"Error submitting feedback: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

def format_sql_solution(doc_context: Dict) -> str:
    """Format SQL solution with relevant tables and example query."""
    if not doc_context:
        return ""

    # Extract relevant tables and their key columns
    tables = []
    for result in doc_context.get('sql_results', []):
        content = result.get('content', '')
        table_match = re.search(r'CREATE TABLE (\w+)', content)
        if table_match:
            table_name = table_match.group(1)
            columns = re.findall(r'(\w+)\s+(?:VARCHAR|CHAR|BIGINT|INTEGER|DECIMAL|DATE|SERIAL).*?,', content)
            if columns:
                tables.append(f"• {table_name}: {', '.join(columns[:3])}...")

    # Format example query if available
    example_query = ""
    for result in doc_context.get('doc_results', []):
        if 'select' in result.get('content', '').lower():
            example_query = f"\nExample Query:\n```sql\n{result.get('content', '')}\n```"

    return f"""**Relevant Tables:**
{chr(10).join(tables)}
{example_query}"""

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

if __name__ == "__main__":
    main() 