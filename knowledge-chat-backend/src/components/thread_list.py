import streamlit as st
from src.db.database import ChatDatabase
import datetime
import json
from pathlib import Path

def format_timestamp(timestamp_str):
    """Format timestamp string to a readable format"""
    try:
        # Try parsing ISO format
        dt = datetime.datetime.fromisoformat(timestamp_str)
        return dt.strftime("%b %d, %Y %I:%M %p")
    except:
        # Return as is if parsing fails
        return timestamp_str

def render_thread_list():
    """Render a list of conversation threads"""
    st.sidebar.title("Conversation Threads")
    
    # Get threads from database
    db = ChatDatabase()
    threads = db.get_all_threads()
    
    if not threads:
        st.sidebar.info("No conversation threads found.")
        return None
    
    # Display threads in sidebar
    selected_thread = None
    for thread in threads:
        # Format timestamp
        timestamp = format_timestamp(thread.get("updated_at", thread.get("created_at", "")))
        
        # Create an expander for each thread
        with st.sidebar.expander(f"{thread.get('title', 'Thread')} ({thread.get('conversation_count', 0)})"):
            st.write(f"**Created:** {timestamp}")
            
            # Show thread details
            if thread.get("conversation_count", 0) > 0:
                st.write(f"Contains {thread.get('conversation_count')} messages")
            
            # Add button to select this thread
            if st.button("Open Thread", key=f"open_{thread.get('thread_id')}"):
                selected_thread = thread.get('thread_id')
    
    # Add button to create new thread
    if st.sidebar.button("New Thread"):
        selected_thread = "new"
    
    return selected_thread

def render_thread_detail(thread_id):
    """Render the details of a specific thread"""
    if not thread_id or thread_id == "new":
        st.info("Start a new conversation by typing a message below.")
        return
    
    # Get conversations in this thread
    db = ChatDatabase()
    conversations = db.get_conversations_by_thread(thread_id)
    
    if not conversations:
        st.info(f"No conversations found in this thread.")
        return
    
    # Display thread title
    thread_dir = Path(db.db_path).parent / "threads" / thread_id
    metadata_file = thread_dir / "metadata.json"
    thread_title = f"Thread {thread_id[:8]}..."
    
    if metadata_file.exists():
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                thread_title = metadata.get("title", thread_title)
        except:
            pass
    
    st.header(thread_title)
    
    # Display conversations
    for i, conv in enumerate(conversations):
        # Get message content
        query = conv.get("query", "")
        response = conv.get("output", "")
        
        # Format timestamp
        timestamp = format_timestamp(conv.get("timestamp", conv.get("created_at", "")))
        
        # Display in chat-like format
        st.markdown(f"**User ({timestamp}):**")
        st.markdown(f"> {query}")
        
        st.markdown("**Assistant:**")
        st.markdown(response)
        
        # Check for GitHub search results
        technical_details = {}
        if conv.get("technical_details"):
            try:
                if isinstance(conv["technical_details"], str):
                    technical_details = json.loads(conv["technical_details"])
                else:
                    technical_details = conv["technical_details"]
            except:
                pass
        
        # Display GitHub search results if available
        if "github_search_results" in technical_details and technical_details["github_search_results"]:
            with st.expander("📂 Relevant Code Examples"):
                search_results = technical_details["github_search_results"]
                
                for j, result in enumerate(search_results):
                    st.markdown(f"**{j+1}. {result.get('file_path', 'Unknown file')}**")
                    st.markdown(f"Relevance: {result.get('relevance_score', 0.0):.2f}")
                    st.markdown(f"*{result.get('explanation', '')}*")
                    
                    # Display code snippet
                    if "code_snippet" in result:
                        st.code(result["code_snippet"], language=result.get("repo_info", {}).get("language", ""))
                    
                    if j < len(search_results) - 1:
                        st.markdown("---")
        
        # Add separator between conversations
        if i < len(conversations) - 1:
            st.markdown("---")